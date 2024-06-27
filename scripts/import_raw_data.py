import pandas as pd
from pymongo import MongoClient, errors
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

    if not azure_storage_connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING is not set or empty.")
    if not mongo_connection_string:
        raise ValueError("MONGO_CONNECTION_STRING is not set or empty.")

    return azure_storage_connection_string, mongo_connection_string

# Function to insert or update documents with retry mechanism
def upsert_documents_with_retry(collection, documents, max_retries=5, initial_delay=1):
    for document in documents:
        for attempt in range(max_retries):
            try:
                collection.update_one({'id': document['id']}, {'$set': document}, upsert=True)
                break  # Exit the retry loop if successful
            except errors.WriteError as e:
                print(f"Write error on attempt {attempt + 1}: {e.details}")
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise

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
    print("Initializing MongoDB client...")
    client = MongoClient(mongo_connection_string)
    print("Successfully initialized MongoDB client.")
    db = client['AmazonBooksReviews']
    print("Connected to MongoDB.")

    # Read CSV files from Azure Blob Storage into pandas DataFrames
    container_name = "nlpdata"
    # books_blob_name = "books_data.csv"
    reviews_blob_name = "Books_rating.csv"

#Commenting out the books data import as we are already imported it
    # # Process books data
    # books_df = read_blob_to_dataframe(blob_service_client, container_name, books_blob_name)
    # print("Successfully read books data from blob storage.")

    # # Use DataFrame index as the id for each record in books data
    # books_df['id'] = books_df.index.astype(str)

    # # Ensure the books collection is created with the id field as the unique key
    # books_collection = db['raw_books']
    # books_collection.create_index([('id', 1)], unique=True)

    # # Upsert books data into MongoDB with retry mechanism
    # books_documents = books_df.to_dict(orient='records')
    # upsert_documents_with_retry(books_collection, books_documents)

    # Process reviews data
    reviews_df = read_blob_to_dataframe(blob_service_client, container_name, reviews_blob_name)
    print("Successfully read reviews data from blob storage.")

    # Use DataFrame index as the id for each record in reviews data
    reviews_df['id'] = reviews_df.index.astype(str)

    # Ensure the reviews collection is created with the id field as the unique key
    reviews_collection = db['raw_reviews']
    reviews_collection.create_index([('id', 1)], unique=True)

    # Upsert reviews data into MongoDB with retry mechanism
    reviews_documents = reviews_df.to_dict(orient='records')
    upsert_documents_with_retry(reviews_collection, reviews_documents)

    print("Book Reviews import completed successfully.")
except Exception as e:
    print(f"An error occurred: {e}")
    raise
