import pandas as pd
from pymongo import MongoClient
from azure.storage.blob import BlobServiceClient
import os
from io import StringIO

# Function to read a blob into a pandas DataFrame
def read_blob_to_dataframe(blob_service_client, container_name, blob_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_data = blob_client.download_blob().readall()
    blob_data_str = blob_data.decode('utf-8')
    return pd.read_csv(StringIO(blob_data_str))

# Azure Blob Storage connection string
azure_storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(azure_storage_connection_string)

# Read CSV files from Azure Blob Storage into pandas DataFrames
container_name = "nlpdata"
reviews_blob_name = "Books_rating.csv"
books_blob_name = "books_data.csv"

reviews_df = read_blob_to_dataframe(blob_service_client, container_name, reviews_blob_name)
books_df = read_blob_to_dataframe(blob_service_client, container_name, books_blob_name)

# MongoDB connection string (Azure Cosmos DB)
mongo_connection_string = os.getenv("MONGO_CONNECTION_STRING")
client = MongoClient(mongo_connection_string)
db = client['AmazonBooksReviews']

# Insert data into MongoDB collections
reviews_collection = db['raw_reviews']
books_collection = db['raw_books']

reviews_collection.insert_many(reviews_df.to_dict(orient='records'))
books_collection.insert_many(books_df.to_dict(orient='records'))

print("Data imported successfully into Azure Cosmos DB.")
