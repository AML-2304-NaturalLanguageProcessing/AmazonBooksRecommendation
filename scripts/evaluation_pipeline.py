import numpy as np
import pandas as pd
import torch
import shap
from lime.lime_tabular import LimeTabularExplainer

import sys
import os

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (which is the project root)
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# Add the project root to the system path
sys.path.append(project_root)

from models.NCF_model import NCF, NCFDataset
from scripts.main import load_preprocess_data
from model_dataset import load_data, get_data_info, custom_collate_fn

def get_feature_names(review_embedding_dim):
    feature_names = [
        'user_id',
        'item_id',
        'emotion_id'
    ] + [f'review_emb_{i}' for i in range(review_embedding_dim)]

    return feature_names

def prepare_data_for_interpretation():
    load_preprocess_data(1000)
    user_item_interactions, emotion_labels, review_embeddings = load_data()
    num_users, num_items, num_emotions = get_data_info(user_item_interactions, emotion_labels)
    review_embedding_dim = review_embeddings.shape[1]
    
    # Combine all features into a single numpy array
    X = np.column_stack((
        user_item_interactions[:, 0],  # user_id
        user_item_interactions[:, 1],  # item_id
        emotion_labels,
        review_embeddings
    ))
    
    return X, num_users, num_items, num_emotions, review_embedding_dim

# Define the complete pipeline
def pipeline(X):
    num_users, num_items, num_emotions, review_embedding_dim, _ = prepare_data_for_interpretation()
    embedding_size = 32
    mlp_dims=[256, 128, 64]
    model = NCF(num_users, num_items, num_emotions, review_embedding_dim, embedding_size, mlp_dims)
    
    user = torch.tensor(X[:, 0], dtype=torch.long)
    item = torch.tensor(X[:, 1], dtype=torch.long)
    emotion = torch.tensor(X[:, 2], dtype=torch.long)
    review_emb = torch.tensor(X[:, 3:], dtype=torch.float)
    
    with torch.no_grad():
        predictions = model(user, item, emotion, review_emb)
    
    return predictions.numpy()

# LIME explanation
def lime_explain(instance, X):
    feature_names = get_feature_names(X.shape[1] - 3)  # Subtract user_id, item_id, and emotion_id
    explainer = LimeTabularExplainer(X, feature_names=feature_names, class_names=['rating'])
    exp = explainer.explain_instance(instance, pipeline, num_features=len(feature_names))
    return exp

# SHAP explanation
def shap_explain(X_background):
    explainer = shap.KernelExplainer(pipeline, X_background)
    shap_values = explainer.shap_values(X_background, nsamples=100)
    return shap_values

# Function for local interpretation
def interpret_prediction(instance, X):
    # Make prediction
    prediction = pipeline(instance.reshape(1, -1))[0]
    
    # Get LIME explanation
    lime_exp = lime_explain(instance, X)
    
    return {
        'prediction': prediction,
        'lime_explanation': lime_exp,
    }

# Function for global interpretation
def global_interpretation(X_background):
    # Get SHAP values
    shap_values = shap_explain(X_background)
    
    return {
        'shap_values': shap_values,
    }

# Example usage
X, num_users, num_items, num_emotions, review_embedding_dim = prepare_data_for_interpretation()

# Local interpretation
instance = X[0]  # Choose an instance to explain
local_results = interpret_prediction(instance, X)

print("Local Interpretation:")
print(f"Prediction: {local_results['prediction']}")
print("\nLIME Explanation:")
print(local_results['lime_explanation'].as_list())

# Global interpretation
X_background = X[:1000]  # Use a subset of samples for global interpretation to save time
global_results = global_interpretation(X_background)

print("\nGlobal Interpretation:")
print("SHAP Values shape:", global_results['shap_values'].shape)

# Uncomment the following lines if you're running this in a Jupyter notebook
# import shap
# shap.summary_plot(global_results['shap_values'], X_background, feature_names=get_feature_names(review_embedding_dim))