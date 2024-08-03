import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from lime import lime_tabular
import torch
import os, sys
# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (which is the project root)
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# Add the project root to the system path
sys.path.append(project_root)
from models.NCF_model import NCF, NCFDataset, create_ncf_dataset, check_id_ranges
from scripts.main import load_sample_data, preprocess_data, DataLoader
from model_dataset import load_data, get_data_info, custom_collate_fn


class NCFInterpretationPipeline:
    def __init__(self, model):
        self.model = model
        self.filename = None
        self.dataloader = DataLoader()

    def get_sample_data(self, num_samples=1000):
        sample_df, self.filename = load_sample_data(self.dataloader.file_reader, 
                                        self.dataloader.container_name, 
                                        self.dataloader.books_data_file, 
                                        self.dataloader.books_rating_file,
                                        num_samples)

        return sample_df

    def preprocess_sample(self, sample_data):
        if sample_data is not None:
            preprocess_data(sample_data, self.filename, self.dataloader.file_reader)

    def engineer_features(self):
        user_item_interactions, emotion_labels, review_embeddings = load_data()
        dataset = create_ncf_dataset(user_item_interactions, emotion_labels, review_embeddings)
        return dataset

    def pipeline(self, sample_data):
        self.preprocess_sample(sample_data)
        dataset = self.engineer_features()
        # Convert sample_data to tensor
        users, items, emotions, review_embeddings, _ = custom_collate_fn(dataset)

        # Make sure the model is in evaluation mode
        self.model.eval()

        # Get prediction
        with torch.no_grad():
            prediction = self.model(users, items, emotions, review_embeddings)

        return prediction.numpy()
    
    def preprocess_for_lime(self, sample_df):
        # Identify categorical columns
        categorical_columns = sample_df.select_dtypes(include=['object']).columns

        # Use LabelEncoder for categorical columns
        le = LabelEncoder()
        for column in categorical_columns:
            sample_df[column] = le.fit_transform(sample_df[column].astype(str))

        # Ensure all columns are numeric
        for column in sample_df.columns:
            sample_df[column] = pd.to_numeric(sample_df[column], errors='coerce')

        # Fill NaN values with column mean
        sample_df = sample_df.fillna(sample_df.mean())

        # Scale the features
        sample_data = self.scaler.fit_transform(sample_df)

        return sample_data, sample_df.columns.tolist()

    def interpret_with_lime(self, num_samples=1000):
        self.dataloader.load_connection()
        sample_df = self.get_sample_data(num_samples)
       ##TODO feature_names = dataset.get_features()

        sample_data, feature_names = self.preprocess_for_lime(sample_df)
    
        # Convert the sample dataframe to a numpy array
        sample_data = sample_df.to_numpy()
    
        # Create a LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            sample_data,
            feature_names=feature_names,
            class_names=['rating'],
            discretize_continuous=False
        )
        # Function that runs the entire pipeline
        def pipeline_predict(input_data):
            return self.pipeline(input_data)

        # Generate explanations for a subset of instances
        explanations = []
        for i in range(min(num_samples, len(sample_df))):
            exp = explainer.explain_instance(
                sample_data[i], 
                pipeline_predict, 
                num_features=len(feature_names)
            )
            explanations.append(exp)

        # Close the Azure connection
        self.data_loader.azure_connection.close()
        return explanations

def run_interpretation_pipeline(model, num_samples=1000):
    pipeline = NCFInterpretationPipeline(model)
    explanations = pipeline.interpret_with_lime(num_samples)
    
    # Analyze and present results
    for i, exp in enumerate(explanations):
        print(f"\nExplanation for sample {i+1}:")
        print(exp.as_list())
        # Optionally, you can visualize each explanation
        # exp.show_in_notebook(show_table=True)
    
    return explanations

# Usage example
if __name__ == "__main__":

    # Extract dimensions from the state dict
    state_dict = torch.load('../models/NCF_model.pth', map_location=torch.device('cpu'))
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
    
    # Instantiate the model with adjusted dimensions
    model = NCF(num_users, num_items, num_emotions, review_embedding_dim, embedding_size, mlp_dims)

    # Load the state dict into the model
    model.load_state_dict(state_dict)
    
    # Run the pipeline
    explanations = run_interpretation_pipeline(model, 10)