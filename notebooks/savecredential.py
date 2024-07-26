'''
Steps:
Replace with your actual account name and key
Using terminal type: python save_credentials.py
'''

import pickle

# Replace with your actual account name and key
account_name = ""
account_key = ""

credentials = {
    "account_name": account_name,
    "account_key": account_key
}

# Save credentials to a pickle file
with open("azure_credentials.pkl", "wb") as f:
    pickle.dump(credentials, f)



