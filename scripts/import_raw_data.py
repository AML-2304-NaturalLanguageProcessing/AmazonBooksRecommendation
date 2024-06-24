import pandas as pd
from pymongo import MongoClient
from azure.storage.blob import BlobServiceClient
import os
from io import StringIO

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
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING is not set or it is empty.")
    if not mongo_connection_string:
        raise ValueError("MONGO_CONNECTION_STRING is not set or empty.")

    return azure_storage_connection_string, mongo_connection_string

# Main script
try:
    # Validate connection strings
    azure_storage_connection_string, mongo_connection_string = validate_connection_strings()
    print("Successfully retrieved connection strings.")

    # Initialize BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(azure_storage_connection_string)
    print("Connected to Azure Blob Storage.")

    # Read CSV files from Azure Blob Storage into pandas DataFrames
    container_name = "nlpdata"
    reviews_blob_name = "Books_rating.csv"
    books_blob_name = "books_data.csv"

    reviews_df = read_blob_to_dataframe(blob_service_client, container_name, reviews_blob_name)
    print("Successfully read reviews data from blob storage.")
    books_df = read_blob_to_dataframe(blob_service_client, container_name, books_blob_name)
    print("Successfully read books data from blob storage.")

    # Initialize MongoDB client
    client = MongoClient(mongo_connection_string)
    db = client['AmazonBooksReviews']
    print("Connected to Azure Cosmos DB.")

    # Insert data into MongoDB collections
    reviews_collection = db['raw_reviews']
    books_collection = db['raw_books']

    reviews_collection.insert_many(reviews_df.to_dict(orient='records'))
    print("Successfully inserted reviews data into MongoDB.")
    books_collection.insert_many(books_df.to_dict(orient='records'))
    print("Successfully inserted books data into MongoDB.")

    print("Data import completed successfully.")
except Exception as e:
    print(f"An error occurred: {e}")
    raise