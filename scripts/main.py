import subprocess
import os

from sampledataset import SampleDataset
from azureconnection import AzureConnection
from filereader import FileReader
from datetime import datetime

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
import config

'''
Included in requirements.txt are the libraries that need to be installed:
If you are using Python 3 and the pip command defaults to Python 2, use pip3 instead:
    > pip install azure-storage-blob 
    > pip3 install azure-storage-blob
To verify:
    > pip show azure-storage-blob
If you encounter permission issues when installing the package:
    > pip install --user azure-storage-blob
Using jupyter to run the ipynb file:
    > pip install jupyter or pip install --upgrade jupyter nbconvert
To execute Jupyter notebooks without converting them to another format:
    > pip install nbclient

To get the 10 samples from dataset, in terminal run:"
> python main.py 10

Sample Output:
$ python main.py 10
Credentials loaded successfully
Connected to Azure Blob Storage successfully
books_data.csv already exists.
Books_rating.csv already exists.
Files merged successfully. Shape: (3000000, 19)
Sample of 10 rows obtained successfully.
Sample Dataset saved to sample_dataset_10_20240726_095929.csv
0.01s - Debugger warning: It seems that frozen modules are being used, which may
0.00s - make the debugger miss breakpoints. Please pass -Xfrozen_modules=off
....
Notebook amazon-books-data-preprocessing.ipynb executed successfully.
Preprocessing notebook executed successfully. Now running feature extraction notebook...
[NbConvertApp] Converting notebook feature-extraction.ipynb to notebook
...
Notebook feature-extraction.ipynb executed successfully.
Feature extraction notebook executed successfully.
Connection to Azure Blob Storage closed...
'''
# Establish connection to Azure Blob Storage
def setup_azure_connection():
    azure_connection = AzureConnection(config.credentials_file)
    azure_connection.load_credentials()
    azure_connection.connect()
    azure_connection.list_blobs(config.container_name)
    return azure_connection

# Download and read the csv files
def setup_file_reader(azure_connection):
    file_reader = FileReader(azure_connection)
    file_reader.datasets_directory()

    books_data_file_path = os.path.join(file_reader.base_path, config.books_data_file)
    books_rating_file_path = os.path.join(file_reader.base_path, config.books_rating_file)

    if not os.path.exists(books_data_file_path):
        print(f"Downloading {config.books_data_file} start...")
        file_reader.download_blob_to_file(config.container_name, config.books_data_file)
    else:
        print(f"{config.books_data_file} already exists.")

    if not os.path.exists(books_rating_file_path):
        print(f"Downloading {config.books_rating_file} start...")
        file_reader.download_blob_to_file(config.container_name, config.books_rating_file)
    else:
        print(f"{config.books_rating_file} already exists.")

    return file_reader, books_data_file_path, books_rating_file_path

# To execute Jupyter notebooks without converting them to another format
def run_notebook(notebook_path, filename=None):
    try:
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
        
        if filename:
            print(filename + " is set as the SAMPLE_DATA_FILENAME environment variable.")
            os.environ["SAMPLE_DATA_FILENAME"] = filename
        
        client = NotebookClient(nb)
        client.execute()

        print(f"Notebook {notebook_path} executed successfully.")
        return True
    except CellExecutionError as e:
        print(f"Error executing the notebook {notebook_path}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
    
# Passing the number of sample and execute all the notebook from cleaning to feature engineering 
def load_preprocess_data(num_sample):

    azure_connection = setup_azure_connection()
    file_reader, books_data_file_path, books_rating_file_path = setup_file_reader(azure_connection)

    # Check if files were downloaded successfully
    if os.path.exists(books_data_file_path) and os.path.exists(books_rating_file_path):

        # Create an instance of SampleDataset and merge files
        sample_dataset = SampleDataset(books_data_file_path, books_rating_file_path)
        sample_dataset.merge_files()

        # Generate filename with current date and time
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'sample_dataset_{num_sample}_{timestamp}.csv'
        
        # Get a sample of the merged data
        sample_df = sample_dataset.get_sample(num_samples=num_sample)

        if sample_df is not None:
            file_reader.save_dataframe(sample_df, filename)
                        
            # Run the preprocessing notebook
            notebooks_directory = file_reader.datasets_directory(folder='notebooks')

            if run_notebook(f'{notebooks_directory}/amazon-books-data-preprocessing.ipynb', filename=filename):
                print("Preprocessing notebook executed successfully. Now running feature extraction notebook...")

                # Run the feature extraction notebook
                if run_notebook(f'{notebooks_directory}/feature-extraction.ipynb'):
                    print("Feature extraction notebook executed successfully.")

                    # Run the emotion analysis notebook
                    if run_notebook(f'{notebooks_directory}/emotion-analysis.ipynb'):
                        print("Emotion Analysis notebook executed successfully.")

                        # Run the featur enengineering notebook
                        if run_notebook(f'{notebooks_directory}/feature_engineering.ipynb'):
                            print("Feature Engineering notebook executed successfully.")
                        else:
                            print("Failed to execute feature engineering notebook.")
                    else:
                        print("Failed to execute emotion analysis notebook.")
                else:
                    print("Failed to execute feature extraction notebook.")
            else:
                print("Failed to execute preprocessing notebook.")                    
        else:
            print("Failed to create sample. Exiting...")
    
    # Close the Azure connection
    azure_connection.close()

# Passing number of sample, execute amazon-books-data-preprocessing, process clean and return the result in dataframe
def clean_dataset(num_sample):
    azure_connection = setup_azure_connection()
    file_reader, books_data_file_path, books_rating_file_path = setup_file_reader(azure_connection)

    if os.path.exists(books_data_file_path) and os.path.exists(books_rating_file_path):
        sample_dataset = SampleDataset(books_data_file_path, books_rating_file_path)
        sample_dataset.merge_files()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'sample_dataset_{num_sample}_{timestamp}.csv'
        
        sample_df = sample_dataset.get_sample(num_samples=num_sample)

        if sample_df is not None:
            file_reader.save_dataframe(sample_df, filename)
            notebooks_directory = file_reader.datasets_directory(folder='notebooks')

            if run_notebook(f'{notebooks_directory}/amazon-books-data-preprocessing.ipynb', filename=filename):
                print("Preprocessing notebook executed successfully.")

                # After preprocessing, load data_cleaned.csv and return the dataframe
                file_reader.base_path = file_reader.datasets_directory()
                processed_file_path = os.path.join(file_reader.base_path, 'data_cleaned.csv')

                if os.path.exists(processed_file_path):
                    processed_df = file_reader.load_dataframe(processed_file_path)
                    azure_connection.close()
                    return processed_df
                else:
                    print(f"Processed file {filename} does not exist.")
            else:
                print("Failed to execute preprocessing notebook.")
        else:
            print("Failed to create sample. Exiting.")
    else:
        print("Books data files are not available.")

    azure_connection.close()


if __name__ == "__main__":
    import argparse
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process Azure Blob data and get a sample...')
    parser.add_argument('num_samples', type=int, help='Number of samples to retrieve from the merged data...')

    args = parser.parse_args()
    load_preprocess_data(args.num_samples)

    # Call clean_dataset function with the number of samples
    '''
    processed_df = clean_dataset(args.num_samples)
    
    if processed_df is not None:
        print(processed_df)
    else:
        print("Failed to load DataFrame.")
    '''