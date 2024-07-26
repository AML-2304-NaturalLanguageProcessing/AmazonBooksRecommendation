import pandas as pd

class SampleDataset:
    def __init__(self, books_data_file, books_rating_file):
        self.books_data_file = books_data_file
        self.books_rating_file = books_rating_file
        self.merged_df = None

    def merge_files(self):
        try:
            books_data_df = pd.read_csv(self.books_data_file)
            books_rating_df = pd.read_csv(self.books_rating_file)
            
            # Perform an inner join on the 'Title' column
            self.merged_df = pd.merge(books_data_df, books_rating_df, on='Title', how='inner', validate="one_to_many")
            print(f"Files merged successfully. Shape: {self.merged_df.shape}")
        except Exception as e:
            print(f"Error merging files: {e}")

    def get_sample(self, num_samples):
        if self.merged_df is not None:
            try:
                sample_df = self.merged_df.sample(n=num_samples)
                print(f"Sample of {num_samples} rows obtained successfully.")
                return sample_df
            except Exception as e:
                print(f"Error getting sample: {e}")
                return None
        else:
            print("Merged DataFrame is not available...")
            return None
