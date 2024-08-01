import torch
import torch.nn as nn
from torch.utils.data import Dataset


class NCFDataset(Dataset):
    def __init__(self, interactions, emotions, review_embeddings):
        self.users, self.items = interactions.nonzero()
        self.ratings = interactions[self.users, self.items].A1
        self.emotions = emotions[self.items].toarray()  # Convert to dense array
        self.review_embeddings = review_embeddings[self.items].toarray()  # Convert to dense array

        # Store the maximum user and item IDs
        self.max_user_id = self.users.max()
        self.max_item_id = self.items.max()

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return (self.users[idx], self.items[idx], self.emotions[idx], 
                self.review_embeddings[idx], self.ratings[idx])

    def get_max_ids(self):
        return self.max_user_id, self.max_item_id

class NCF(nn.Module):
    def __init__(self, num_users, num_items, num_emotions, review_embedding_dim, 
                 embedding_dim=64, mlp_dims=[256, 128, 64], dropout=0.2):
        super(NCF, self).__init__()
        print('NCF module',num_users, num_items, num_emotions, review_embedding_dim)
        # Embedding layers
        self.user_embedding_mf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mf = nn.Embedding(num_items, embedding_dim)
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)
        self.emotion_embedding = nn.Embedding(num_emotions, embedding_dim)
        
        # MF layer
        self.mf_output = embedding_dim
        
        # MLP layers
        self.mlp = nn.ModuleList()
        input_dim = embedding_dim * 3 + review_embedding_dim  # user + item + emotion + review
        mlp_dims = [input_dim] + mlp_dims
        for i in range(len(mlp_dims) - 1):
            self.mlp.append(nn.Linear(mlp_dims[i], mlp_dims[i+1]))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.BatchNorm1d(mlp_dims[i+1]))
            self.mlp.append(nn.Dropout(dropout))
        
        # Final prediction layer
        self.final = nn.Linear(self.mf_output + mlp_dims[-1], 1)
        
    def forward(self, user_indices, item_indices, emotion_indices, review_embeddings):
        # MF component
        user_embedding_mf = self.user_embedding_mf(user_indices)
        item_embedding_mf = self.item_embedding_mf(item_indices)
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)
        
        # MLP component
        user_embedding_mlp = self.user_embedding_mlp(user_indices)
        item_embedding_mlp = self.item_embedding_mlp(item_indices)
        emotion_embedding = self.emotion_embedding(emotion_indices).mean(dim=1)  # Adjusted for mean over emotions
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp, emotion_embedding, review_embeddings], dim=-1)
 
        for layer in self.mlp:
            mlp_vector = layer(mlp_vector)
        
        # Combine MF and MLP
        combined = torch.cat([mf_vector, mlp_vector], dim=-1)
        
        # Final prediction
        prediction = self.final(combined)
        
        return prediction.squeeze()

def create_ncf_dataset(user_item_interactions, emotion_labels, review_embeddings):
    dataset = NCFDataset(user_item_interactions, emotion_labels, review_embeddings)
    max_user_id, max_item_id = dataset.get_max_ids()
    print(f"Max user ID in dataset: {max_user_id}")
    print(f"Max item ID in dataset: {max_item_id}")
    return dataset

def check_id_ranges(dataset, model):
    max_user_id, max_item_id = dataset.get_max_ids()
    model_num_users = model.user_embedding_mf.num_embeddings
    model_num_items = model.item_embedding_mf.num_embeddings

    if max_user_id >= model_num_users:
        print(f"Warning: Max user ID in dataset ({max_user_id}) is >= number of user embeddings in model ({model_num_users})")
    if max_item_id >= model_num_items:
        print(f"Warning: Max item ID in dataset ({max_item_id}) is >= number of item embeddings in model ({model_num_items})")

    return max_user_id < model_num_users and max_item_id < model_num_items