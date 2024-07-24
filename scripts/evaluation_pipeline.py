import numpy as np
import pandas as pd
import torch
import lime
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler

from models.NCF_model import NCF

# Prepare data
X, y = load_and_preprocess_data()  # Implement this function to load data
feature_names = ['user_id', 'item_id', 'user_feature1', 'user_feature2', 'item_feature1', 'item_feature2']

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NCF(num_users, num_items, num_emotions, 
            embedding_dim=32, 
            review_embedding_dim=100, 
            mlp_dims=[256, 128, 64], 
            dropout=0.2).to(device)
model.load_state_dict(torch.load('ncf_model.pth'))
model.eval()

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# LIME explanation
def lime_explain(instance):
    explainer = LimeTabularExplainer(X_scaled, feature_names=X_scaled.columns, class_names=['rating'])
    exp = explainer.explain_instance(instance, 
                                     lambda x: model(torch.tensor(x[:, 0], dtype=torch.long), 
                                                     torch.tensor(x[:, 1], dtype=torch.long)).detach().numpy())
    return exp

# SHAP explanation
def shap_explain(X_background, instance):
    explainer = shap.DeepExplainer(model, X_background)
    shap_values = explainer.shap_values(instance)
    return shap_values

# Pipeline function
def interpret_prediction(instance, X_background):
    # Make prediction
    user = torch.tensor(instance[0], dtype=torch.long)
    item = torch.tensor(instance[1], dtype=torch.long)
    prediction = model(user.unsqueeze(0), item.unsqueeze(0)).item()
    
    # Get LIME explanation
    lime_exp = lime_explain(instance)
    
    # Get SHAP explanation
    shap_values = shap_explain(X_background, instance.reshape(1, -1))
    
    return {
        'prediction': prediction,
        'lime_explanation': lime_exp,
        'shap_values': shap_values
    }

# Example usage
instance = X_scaled[0]  # Choose an instance to explain
X_background = X_scaled[:100]  # Choose a subset of your data as background

results = interpret_prediction(instance, X_background)

print(f"Prediction: {results['prediction']}")
print("\nLIME Explanation:")
results['lime_explanation'].show_in_notebook()
print("\nSHAP Values:")
shap.summary_plot(results['shap_values'], X_background, feature_names=feature_names)
