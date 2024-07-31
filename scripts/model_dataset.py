from scipy.sparse import load_npz
import torch

def load_data():
    user_book_matrix = load_npz('../working/user_book_matrix.npz')
    emotion_matrix = load_npz('../working/emotion_matrix.npz')
    book_embeddings = load_npz('../working/avg_embeddings_matrix.npz')
    
    return user_book_matrix, emotion_matrix, book_embeddings

def get_data_info(user_item_interactions, emotion_labels):
    # Get the number of unique users and items
    num_users = user_item_interactions.shape[0]
    num_items = user_item_interactions.shape[1]
    
    # Get the number of unique emotions
    num_emotions = emotion_labels.shape[1]
    
    return num_users, num_items, num_emotions

def custom_collate_fn(batch):
    users, items, emotions, review_embeddings, ratings = zip(*batch)
    
    users = torch.tensor(users, dtype=torch.long)
    items = torch.tensor(items, dtype=torch.long)
    emotions = torch.tensor(emotions, dtype=torch.long)
    review_embeddings = torch.tensor(review_embeddings, dtype=torch.float)
    ratings = torch.tensor(ratings, dtype=torch.float)
    
    # Move tensors to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    users = users.to(device)
    items = items.to(device)
    emotions = emotions.to(device)
    review_embeddings = review_embeddings.to(device)
    ratings = ratings.to(device)
    
    return users, items, emotions, review_embeddings, ratings