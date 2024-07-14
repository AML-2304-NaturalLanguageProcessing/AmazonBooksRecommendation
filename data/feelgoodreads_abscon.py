# feelgoodread_abs.py

import pandas as pd
from azure.storage.blob import BlobServiceClient
import io
import os
from datetime import datetime
import time

class AzureBlobConnector:
    def __init__(self, connection_string, container_name, folder_name):
        self.connection_string = connection_string
        self.container_name = container_name
        self.folder_name = folder_name
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.container_client = self.blob_service_client.get_container_client(self.container_name)
        
        # Ensure the folder exists by creating a zero-byte blob
        self.create_folder_if_not_exists(folder_name)
    
    def create_folder_if_not_exists(self, folder_name):
        blob_client = self.container_client.get_blob_client(f"{folder_name}/")
        try:
            blob_client.upload_blob(b"", overwrite=False)
            print(f"Folder '{folder_name}' created or already exists.")
        except Exception as e:
            if "BlobAlreadyExists" not in str(e):
                print(f"Error creating folder '{folder_name}': {e}")

    def read_csv(self, blob_name):
        blob_client = self.container_client.get_blob_client(f"{self.folder_name}/{blob_name}")
        csv_data = blob_client.download_blob().readall()
        df = pd.read_csv(io.BytesIO(csv_data))
        return df

    def upload_file(self, local_file_name, blob_name):
        full_blob_name = f"{self.folder_name}/{blob_name}"
        blob_client = self.container_client.get_blob_client(full_blob_name)
        
        if blob_client.exists():
            overwrite = input(f"The file '{blob_name}' already exists. Do you want to overwrite it? (yes/no): ")
            if overwrite.lower() == 'yes':
                # Rename the existing file with a timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                renamed_blob_name = f"{self.folder_name}/{os.path.splitext(blob_name)[0]}_{timestamp}{os.path.splitext(blob_name)[1]}"
                renamed_blob_client = self.container_client.get_blob_client(renamed_blob_name)
                
                # Copy the existing blob to the new name
                renamed_blob_client.start_copy_from_url(blob_client.url)
                
                # Wait for the copy to complete
                while renamed_blob_client.get_blob_properties().copy.status != 'success':
                    time.sleep(1)
                
                # Delete the old blob
                blob_client.delete_blob()
                
                # Upload the new file with the original name
                self._upload_blob(blob_client, local_file_name)
                print(f"Existing file renamed to '{renamed_blob_name}' and new file uploaded as '{full_blob_name}'.")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_blob_name = f"{self.folder_name}/{os.path.splitext(blob_name)[0]}_{timestamp}{os.path.splitext(blob_name)[1]}"
                new_blob_client = self.container_client.get_blob_client(new_blob_name)
                self._upload_blob(new_blob_client, local_file_name)
                print(f"File saved as '{new_blob_name}'.")
        else:
            self._upload_blob(blob_client, local_file_name)

    def _upload_blob(self, blob_client, local_file_name):
        with open(local_file_name, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        print(f"Uploaded {local_file_name} to {blob_client.blob_name}")
