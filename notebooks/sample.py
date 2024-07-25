# sample.py
import pandas as pd

def sample_data(df, no_of_sample):
    try:
        # Randomly sample the data
        sample_df = df.sample(n=no_of_sample)
        return sample_df
    except Exception as e:
        print(f"Error sampling data: {e}")
        return None
