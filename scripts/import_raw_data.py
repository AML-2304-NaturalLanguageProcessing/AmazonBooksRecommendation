import pandas as pd
import uuid
from pymongo import MongoClient
from azure.storage.blob import BlobServiceClient
import os
from io import StringIO
from dotenv import load_dotenv
import sys

# Load environment variables from .env file
load_dotenv()

# Function to read a blob into a pandas DataFrame
def read_blob_to_dataframe(blob_service_client, container_name, blob_name):
    try:
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_data = blob_client.download_blob().readall()
        blob_data_str = blob_data.decode('utf-8')
        return pd.read_csv(StringIO(blob_data_str))
    except Exception as e:
        print(f"Error reading blob {blob_name} from container {container_name}: {e}")
        raise

# Validate connection strings
def validate_connection_strings():
    azure_storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    mongo_connection_string = os.getenv("MONGO_CONNECTION_STRING")

    if not azure_storage_connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING is not set or empty.")
    if not mongo_connection_string:
        raise ValueError("MONGO_CONNECTION_STRING is not set or empty.")

    return azure_storage_connection_string, mongo_connection_string

# Main script
try:
    print("Starting data import...")

    # Validate connection strings
    azure_storage_connection_string, mongo_connection_string = validate_connection_strings()
    print("Successfully retrieved connection strings.")

    # Initialize BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(azure_storage_connection_string)
    print("Connected to Azure Blob Storage.")

    # Initialize MongoDB client
    client = MongoClient(mongo_connection_string)
    db = client['AmazonBooksReviews']
    print("Connected to Azure Cosmos DB.")

    # Read CSV files from Azure Blob Storage into pandas DataFrames
    container_name = "nlpdata"
    books_blob_name = "books_data.csv"
    reviews_blob_name = "Books_rating.csv"

    # Process books data
    books_df = read_blob_to_dataframe(blob_service_client, container_name, books_blob_name)
    print("Successfully read books data from blob storage.")

    # Generate UUID for each record in books data
    books_df['id'] = [str(uuid.uuid4()) for _ in books_df.index]

    # Ensure the books collection is created with the id field as the partition key
    if 'raw_books' not in db.list_collection_names():
        db.create_collection('raw_books', shard_key={'id': 'hashed'})
    books_collection = db['raw_books']

    # Insert books data into MongoDB
    books_collection.insert_many(books_df.to_dict(orient='records'))
    print("Successfully inserted books data into MongoDB.")

    # Process reviews data
    reviews_df = read_blob_to_dataframe(blob_service_client, container_name, reviews_blob_name)
    print("Successfully read reviews data from blob storage.")

    # Generate UUID for each record in reviews data
    reviews_df['id'] = [str(uuid.uuid4()) for _ in reviews_df.index]

    # Ensure the reviews collection is created with the id field as the partition key
    if 'raw_reviews' not in db.list_collection_names():
        db.create_collection('raw_reviews', shard_key={'id': 'hashed'})
    reviews_collection = db['raw_reviews']

    # Insert reviews data into MongoDB
    reviews_collection.insert_many(reviews_df.to_dict(orient='records'))
    print("Successfully inserted reviews data into MongoDB.")

    print("Data import completed successfully.")
except Exception as e:
    print(f"An error occurred: {e}")
    raise
