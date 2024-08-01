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

from models.NCF_model import NCF, NCFDataset, create_ncf_dataset, check_id_ranges
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
    user_item_interactions, emotion_labels, review_embeddings = load_data()
    num_users, num_items, num_emotions = get_data_info(user_item_interactions, emotion_labels)
    review_embedding_dim = review_embeddings.shape[1]
    print('labels:')
    for label in emotion_labels:
        print(label)
    X = create_ncf_dataset(user_item_interactions, emotion_labels, review_embeddings)

    return X, num_users, num_items, num_emotions, review_embedding_dim

# Define the complete pipeline
def pipeline(instance):
    # Load the model state dict
    state_dict = torch.load('../models/NCF_model.pth', map_location=torch.device('cpu'))
    
    # Extract dimensions from the state dict
    num_users = state_dict['user_embedding_mf.weight'].shape[0]
    num_items = state_dict['item_embedding_mf.weight'].shape[0]
    num_emotions = state_dict['emotion_embedding.weight'].shape[0]
    embedding_size = state_dict['user_embedding_mf.weight'].shape[1]
    first_mlp_layer_weight = state_dict['mlp.0.weight']
    review_embedding_dim = first_mlp_layer_weight.shape[1] - (embedding_size * 3)
 
    # Extract MLP dimensions from the state dict
    mlp_dims = []
    i = 0
    while f'mlp.{i}.weight' in state_dict:
        mlp_dims.append(state_dict[f'mlp.{i}.weight'].shape[0])
        i += 4  # Skip 4 layers (Linear, ReLU, BatchNorm, Dropout)
    
    print('num_users, num_items, num_emotions, review_embedding_dim, embedding_size, mlp_dims')
    print(num_users, num_items, num_emotions, review_embedding_dim, embedding_size, mlp_dims)
    
    # Initialize the model
    model = NCF(num_users, num_items, num_emotions, review_embedding_dim, embedding_size, mlp_dims)
    
    # Load the trained model weights
    model.load_state_dict(state_dict)
    model.eval()
    
    # Unpack and prepare input data
    user, item, emotion, review_embedding, _ = instance
    print('user, item, emotion, review_embedding', user, item, emotion, review_embedding)
    
    # Convert inputs to tensors with appropriate dimensions
    user = torch.tensor([user], dtype=torch.long)
    item = torch.tensor([item], dtype=torch.long)
    emotion = torch.tensor([emotion], dtype=torch.long)
    review_emb = torch.tensor([review_embedding], dtype=torch.float)
    
    print("User ID:", user.item())
    print("Item ID:", item.item())
    print("Num users in model:", num_users)
    print("Num items in model:", num_items)
    print("Num emotions in model:", num_emotions)
    print("Emotion tensor shape:", emotion.shape)
    print("Review embedding shape:", review_emb.shape)
    
    # Check if item ID is out of range
    if item.item() >= num_items:
        print(f"Warning: Item ID {item.item()} is out of range. Using random embedding.")
        # Use a random embedding for out-of-range items
        with torch.no_grad():
            random_embedding = torch.randn(1, embedding_size)
            predictions = model.forward_with_item_embedding(user, random_embedding, emotion, review_emb)
    else:
        with torch.no_grad():
            try:
                predictions = model(user, item, emotion, review_emb)
            except Exception as e:
                print(f"Error in model forward pass: {str(e)}")
                print(f"User tensor: {user}")
                print(f"Item tensor: {item}")
                print(f"Emotion tensor: {emotion}")
                print(f"Review embedding tensor: {review_emb}")
                raise
    
    return predictions.numpy()

# LIME explanation
def lime_explain(instance, X):
    # Prepare the data for LIME
    feature_names = ['user_id', 'item_id'] + [f'emotion_{i}' for i in range(len(instance[2]))] \
        + [f'review_emb_{i}' for i in range(len(instance[3]))]
    
    # Flatten the instance
    flat_instance = np.array([instance[0], instance[1]] + list(instance[2]) + list(instance[3]))
    
    sample_data = []
    for i in range(len(X)):
        user, item, emotion, review_emb, _ = X[i]
        sample_data.append([user, item] + list(emotion) + list(review_emb))
    sample_data = np.array(sample_data)

    # Create a wrapper for the pipeline function
    def pipeline_wrapper(x):
        if x.ndim == 1:
            # Single instance
            user = int(x[0])
            item = int(x[1])
            emotion = x[2:2+num_emotions]
            review_emb = x[2+num_emotions:]
            return pipeline((user, item, emotion, review_emb, None))
        else:
            # Batch of instances
            results = []
            for instance in x:
                user = int(instance[0])
                item = int(instance[1])
                emotion = instance[2:2+num_emotions]
                review_emb = instance[2+num_emotions:]
                result = pipeline((user, item, emotion, review_emb, None))
                results.append(result)
            return np.array(results)

    explainer = LimeTabularExplainer(sample_data, feature_names=feature_names, class_names=['rating'], \
                                     discretize_continuous=False)
    exp = explainer.explain_instance(flat_instance, pipeline_wrapper, num_features=len(feature_names))
    return exp

# SHAP explanation
def shap_explain(X_background):
    explainer = shap.KernelExplainer(pipeline, X_background)
    shap_values = explainer.shap_values(X_background, nsamples=100)
    return shap_values

# Function for local interpretation
def interpret_prediction(instance, X):
    # Make prediction
    prediction = pipeline(instance)
    print('local prediction', prediction)
    # Get LIME explanation
    lime_exp = lime_explain(instance, X)
    print('lime_exp', lime_exp)
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