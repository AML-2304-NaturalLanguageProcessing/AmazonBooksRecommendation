import pandas as pd
from pymongo import MongoClient
from azure.storage.blob import BlobServiceClient
import os
from io import StringIO

# Function to read a blob into a pandas DataFrame
def read_blob_to_dataframe(blob_service_client, container_name, blob_name):
    try:
        print(f"Reading blob: {blob_name} from container: {container_name}")
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_data = blob_client.download_blob().readall()
        blob_data_str = blob_data.decode('utf-8')
        print(f"Successfully read blob: {blob_name}")
        return pd.read_csv(StringIO(blob_data_str))
    except Exception as e:
        print(f"Error reading blob {blob_name} from container {container_name}: {e}")
        return None

# Azure Blob Storage connection string
azure_storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# Validate Azure Blob Storage connection
try:
    print("Attempting to connect to Azure Blob Storage...")
    blob_service_client = BlobServiceClient.from_connection_string(azure_storage_connection_string)
    print("Connected to Azure Blob Storage successfully.")
except Exception as e:
    print(f"Failed to connect to Azure Blob Storage: {e}")
    exit(1)

# Read CSV files from Azure Blob Storage into pandas DataFrames
container_name = "nlpdata"
reviews_blob_name = "Books_rating.csv"
books_blob_name = "books_data.csv"

print("Reading reviews CSV from Azure Blob Storage...")
reviews_df = read_blob_to_dataframe(blob_service_client, container_name, reviews_blob_name)
print("Reading books CSV from Azure Blob Storage...")
books_df = read_blob_to_dataframe(blob_service_client, container_name, books_blob_name)

# Check if dataframes are loaded successfully
if reviews_df is None or books_df is None:
    print("Failed to load one or more dataframes from Azure Blob Storage.")
    exit(1)

# MongoDB connection string (Azure Cosmos DB)
mongo_connection_string = os.getenv("MONGO_CONNECTION_STRING")

# Validate MongoDB connection
try:
    print("Attempting to connect to Azure Cosmos DB...")
    client = MongoClient(mongo_connection_string)
    db = client['AmazonBooksReviews']
    print("Connected to Azure Cosmos DB successfully.")
except Exception as e:
    print(f"Failed to connect to Azure Cosmos DB: {e}")
    exit(1)

# Insert data into MongoDB collections
try:
    reviews_collection = db['raw_reviews']
    books_collection = db['raw_books']

    print("Inserting reviews data into Azure Cosmos DB...")
    reviews_collection.insert_many(reviews_df.to_dict(orient='records'))
    print("Inserting books data into Azure Cosmos DB...")
    books_collection.insert_many(books_df.to_dict(orient='records'))

    print("Data imported successfully into Azure Cosmos DB.")
except Exception as e:
    print(f"Error inserting data into Azure Cosmos DB: {e}")
    exit(1)
