from azure.storage.blob import BlobServiceClient
import pickle


class AzureConnection:
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
            except Exception as e:
                print(f"Error listing blobs: {e}")

    def close(self):
        print("Connection to Azure Blob Storage closed...")
