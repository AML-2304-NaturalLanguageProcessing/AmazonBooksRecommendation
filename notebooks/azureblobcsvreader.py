import pandas as pd
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
from io import StringIO
import pickle

class AzureBlobCSVReader:
    def __init__(self, credentials_file):
        self.credentials_file = credentials_file
        self.blob_service_client = None
        self.account_name = None
        self.account_key = None

    def load_credentials(self):
        try:
            with open(self.credentials_file, "rb") as f:
                credentials = pickle.load(f)
                self.account_name = credentials.get("account_name")
                self.account_key = credentials.get("account_key")
            print("Credentials loaded successfully")
        except Exception as e:
            print(f"Failed to load credentials: {e}")

    def connect(self):
        if self.account_name and self.account_key:
            try:
                connection_string = f"DefaultEndpointsProtocol=https;AccountName={self.account_name};AccountKey={self.account_key};EndpointSuffix=core.windows.net"
                self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                print("Connected to Azure Blob Storage successfully")
            except Exception as e:
                print(f"Failed to connect: {e}")
        else:
            print("Account name or key is missing")

    def list_blobs(self, container_name):
        if self.blob_service_client:
            try:
                container_client = self.blob_service_client.get_container_client(container_name)
                blobs = container_client.list_blobs()
                print(f"Blobs in container '{container_name}':")
                for blob in blobs:
                    print(blob.name)
            except Exception as e:
                print(f"Error listing blobs: {e}")

    def read_csv(self, container_name, file_name):
        if self.blob_service_client:
            try:
                container_client = self.blob_service_client.get_container_client(container_name)
                blob_client = container_client.get_blob_client(file_name)
                
                # Download the blob data
                blob_data = blob_client.download_blob().readall()
                
                # Read the data 
                csv_data = StringIO(blob_data.decode('utf-8'))
                df = pd.read_csv(csv_data)
                
                print("CSV file read into DataFrame successfully")
                return df
            except Exception as e:
                print(f"Error reading CSV file: {e}")
                return None
        else:
            print("Blob service client is not connected. Please call connect() first.")
            return None
        
    def read_csv_in_chunks(self, container_name, file_name, chunk_size=100000):
        if self.blob_service_client:
            try:
                container_client = self.blob_service_client.get_container_client(container_name)
                blob_client = container_client.get_blob_client(file_name)
                
                # Stream the blob data in chunks
                stream = blob_client.download_blob()
                buffer = StringIO()
                
                # Read and process the data in chunks
                for chunk in stream.chunks():
                    buffer.write(chunk.decode('utf-8'))
                    
                    # Use StringIO to read the data in chunks
                    buffer.seek(0)
                    try:
                        for chunk_df in pd.read_csv(buffer, chunksize=chunk_size, 
                                                    on_bad_lines='warn',  
                                                    encoding='utf-8'):
                            self.process_chunk(chunk_df)
                    except pd.errors.ParserError as e:
                        print(f"ParserError: {e}")
                    
                    # Clear buffer for next chunk
                    buffer.seek(0)
                    buffer.truncate(0)
                
                print("CSV file processed in chunks successfully")

            except Exception as e:
                print(f"Error reading CSV file: {e}")

    def process_chunk(self, chunk):
        print(chunk.head(1))

    def close(self):
        print("Connection to Azure Blob Storage closed")
