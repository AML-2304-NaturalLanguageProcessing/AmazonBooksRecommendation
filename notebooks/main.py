from azureblobcsvreader import AzureBlobCSVReader
from sample import sample_data

import pandas as pd

def main():
    credentials_file = "azure_credentials.pkl"
    
    # Replace with your container name and file name
    container_name = "nlpdata/raw" 
    books_data_file = "books_data.csv"  
    books_rating_file = "Books_rating.csv" 

    # Number of samples to extract
    # TODO: move this as parameter ex. python main.py 10
    no_of_sample = 10

    # Create an instance of the class, load credentials and read the CSV
    reader = AzureBlobCSVReader(credentials_file)
    reader.load_credentials()
    reader.connect()
    reader.list_blobs(container_name)
    
    # Read the books_data 
    books_data = reader.read_csv(container_name, books_data_file)
    print(books_data.head())

    if books_data is not None:  
        # Sample the data from books_data
        sampled_books_data = sample_data(books_data, no_of_sample)
        print(sampled_books_data.head())

        # Read the books_rating 
        # The book rating file is 2.66 GB in size, so we need to read it in chunks to manage the large volume
        books_rating = reader.read_csv_in_chunks(container_name, books_rating_file)

        if books_rating is not None:
            # Merge the csv on the "Title" column
            merged_df = pd.merge(
                sampled_books_data, 
                books_rating, 
                how="outer", 
                on="Title", 
                validate="one_to_many"
            )

            print("Merged:")
            print(merged_df.head()) 
                         
    
    reader.close()

if __name__ == "__main__":
    main()