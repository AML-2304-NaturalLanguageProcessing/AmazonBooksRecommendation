import pandas as pd
import uuid
from pymongo import MongoClient, errors
from pymongo.operations import UpdateOne
from azure.storage.blob import BlobServiceClient
import os
from io import StringIO
from dotenv import load_dotenv
import time

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

    if not azure_storage_connection_string or not mongo_connection_string:
        raise ValueError("One or more connection strings are not set or empty.")

    return azure_storage_connection_string, mongo_connection_string

# Function to bulk upsert documents with retry mechanism
def bulk_upsert_with_retry(collection, documents, max_retries=3, batch_size=500):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        operations = [UpdateOne({'id': doc['id']}, {'$set': doc}, upsert=True) for doc in batch]
        for attempt in range(max_retries):
            try:
                collection.bulk_write(operations, ordered=False)
                break  # Exit the retry loop if successful
            except errors.BulkWriteError as bwe:
                print(f"Bulk write error on attempt {attempt + 1}: {bwe.details}")
                time.sleep(5 * (2 ** attempt))  # Exponential back-off
            except Exception as e:
                print(f"An error occurred on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise  # Raise the error if the last retry fails

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

    # Process books data
    books_df = read_blob_to_dataframe(blob_service_client, "nlpdata", "books_data.csv")
    books_df['id'] = [str(uuid.uuid4()) for _ in books_df.index]
    if 'raw_books' not in db.list_collection_names():
        db.create_collection('raw_books', shard_key={'id': 'hashed'})
    bulk_upsert_with_retry(db['raw_books'], books_df.to_dict(orient='records'))

    # Process reviews data
    reviews_df = read_blob_to_dataframe(blob_service_client, "nlpdata", "Books_rating.csv")
    reviews_df['id'] = [str(uuid.uuid4()) for _ in reviews_df.index]
    if 'raw_reviews' not in db.list_collection_names():
        db.create_collection('raw_reviews', shard_key={'id': 'hashed'})
    bulk_upsert_with_retry(db['raw_reviews'], reviews_df.to_dict(orient='records'))

    print("Data import completed successfully.")
except Exception as e:
    print(f"An error occurred: {e}")
    raise
