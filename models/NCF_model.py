import torch
import torch.nn as nn
from torch.utils.data import Dataset


class NCFDataset(Dataset):
    def __init__(self, interactions, emotions, review_embeddings):
        self.users, self.items = interactions.nonzero()
        self.ratings = interactions[self.users, self.items].A1
        self.emotions = emotions[self.items].toarray()  # Convert to dense array
        self.review_embeddings = review_embeddings[self.items].toarray()  # Convert to dense array

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return (self.users[idx], self.items[idx], self.emotions[idx], 
                self.review_embeddings[idx], self.ratings[idx])

class NCF(nn.Module):
    def __init__(self, num_users, num_items, num_emotions, review_embedding_dim, 
                 embedding_dim=64, mlp_dims=[256, 128, 64], dropout=0.2):
        super(NCF, self).__init__()
        
        # Embedding layers
        self.user_embedding_mf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mf = nn.Embedding(num_items, embedding_dim)
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)
        self.emotion_embedding = nn.Embedding(num_emotions, embedding_dim)
        
        # Personalized Emotional Weighting
        self.emotion_weight = nn.Parameter(torch.ones(num_users, 1))
        
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
        emotion_embedding = self.emotion_embedding(emotion_indices)
        
        # Personalized Emotional Weighting
        emotion_weight = self.emotion_weight[user_indices]
        weighted_emotion_embedding = emotion_embedding * emotion_weight.unsqueeze(1)
        weighted_emotion_embedding = weighted_emotion_embedding.mean(dim=1)
        
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp, weighted_emotion_embedding, review_embeddings], dim=-1)
        
        for layer in self.mlp:
            mlp_vector = layer(mlp_vector)
        
        # Combine MF and MLP
        combined = torch.cat([mf_vector, mlp_vector], dim=-1)
        
        # Final prediction
        prediction = self.final(combined)
        
        return prediction.squeeze()

    def loss(self, prediction, target, gamma=2.0):
        loss = F.mse_loss(prediction, target, reduction='none')
        pt = torch.exp(-loss)  # Probability of the prediction
        focal_loss = ((1 - pt) ** gamma) * loss
        return focal_loss.mean()
