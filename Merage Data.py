import pandas as pd
import os

# Define file paths
data_path = r"C:\Users\43887\Desktop\CSDS312 Project"
file1_path = os.path.join(data_path, "dummy_data.csv")  # Update with actual filename
file2_path = os.path.join(data_path, "social_media_usage.csv")  # Update with actual filename

# Load the datasets
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# Standardize column names for merging
df1.rename(columns={"platform": "App"}, inplace=True)

# Merge datasets on 'App' (social media platform)
merged_df = pd.merge(df1, df2, on="App", how="inner")  # Inner join keeps matching records only

# Save the merged dataset to the same folder
merged_file_path = os.path.join(data_path, "merged_social_media_data.csv")
merged_df.to_csv(merged_file_path, index=False)

# Display confirmation message
print(f"✅ Merged dataset saved successfully at: {merged_file_path}")

# Display first few rows of merged dataset
print(merged_df.head())
