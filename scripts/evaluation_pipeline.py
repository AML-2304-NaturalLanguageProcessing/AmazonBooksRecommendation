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
    load_preprocess_data(10)
    user_item_interactions, emotion_labels, review_embeddings = load_data()
    num_users, num_items, num_emotions = get_data_info(user_item_interactions, emotion_labels)
    review_embedding_dim = review_embeddings.shape[1]
    X = NCFDataset(user_item_interactions, emotion_labels, review_embeddings)

    return X, num_users, num_items, num_emotions, review_embedding_dim

# Define the complete pipeline
def pipeline(X):
    _, num_users, num_items, num_emotions, review_embedding_dim = prepare_data_for_interpretation()
    embedding_size = 32
    mlp_dims=[256, 128, 64]
    model = NCF(num_users, num_items, num_emotions, review_embedding_dim, embedding_size, mlp_dims)
    # Load your trained model weights here
    model.load_state_dict(torch.load('../models/NCF_model.pth', map_location=torch.device('cpu')))
    user, item, emotion, review_embedding, _ = X
    print('Original shapes:')
    print('user shape', user.shape)
    print('item shape', item.shape)
    print('emotion shape', emotion.shape)
    print('review_emb shape', review_embedding.shape)
    print('---------------------------------------')
    model.eval() 
    # Convert inputs to tensors with appropriate dimensions
    user = torch.tensor(user, dtype=torch.long).view(1, 1)  # Shape: (1, 1)
    item = torch.tensor(item, dtype=torch.long).view(1, 1)  # Shape: (1, 1)
    emotion = torch.tensor(emotion, dtype=torch.long).view(1, -1)  # Shape: (1, num_emotions)
    review_emb = torch.tensor(review_embedding, dtype=torch.float).view(1, -1)  # Shape: (1, embedding_dim)
    print('Transformed shapes:')
    print('user shape', user.shape)
    print('item shape', item.shape)
    print('emotion shape', emotion.shape)
    print('review_emb shape', review_emb.shape)
    with torch.no_grad():
        predictions = model(user, item, emotion, review_emb)
        print('predictions', predictions)
    
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
    prediction = pipeline(instance)[0]
    
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
print(instance, 'instance')
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