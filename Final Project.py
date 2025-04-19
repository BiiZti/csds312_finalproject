import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Set custom Kaggle config path if kaggle.json is NOT in default location
os.environ["KAGGLE_CONFIG_DIR"] = r"C:\Users\43887\Desktop\CSDS312 Project"

# Authenticate Kaggle API
api = KaggleApi()
api.authenticate()

# Define datasets
dataset1 = "bhadramohit/social-media-usage-datasetapplications"
dataset2 = "imyjoshua/average-time-spent-by-a-user-on-social-media"

# Create a folder to store datasets
os.makedirs("data", exist_ok=True)

# Download and unzip datasets
api.dataset_download_files(dataset1, path="data/social_media_usage", unzip=True)
api.dataset_download_files(dataset2, path="data/time_spent", unzip=True)

print("Datasets downloaded successfully!")
