import pandas as pd

# Load the merged dataset
merged_df = pd.read_csv("C:/Users/43887/Desktop/CSDS312 Project/merged_social_media_data.csv")  # Update with your actual path

# Step 1: Define the Target Variable (is_addicted)
# Users spending more than 300 minutes daily are considered addicted (1), else not addicted (0)
merged_df["is_addicted"] = (merged_df["Daily_Minutes_Spent"] > 300).astype(int)

# Save the updated dataset with the target variable
merged_df.to_csv("C:/Users/43887/Desktop/CSDS312 Project/merged_data_with_target.csv", index=False)

# Display first few rows with the new target variable
print(merged_df.head())
