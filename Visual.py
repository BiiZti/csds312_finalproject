import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the file path
file_path = r"C:\Users\43887\Desktop\CSDS312 Project\merged_data_with_target.csv"

# Load the dataset
df = pd.read_csv(file_path)

# Create a folder to save images
output_dir = r"C:\Users\43887\Desktop\CSDS312 Project\output_images"
os.makedirs(output_dir, exist_ok=True)  # Ensure the folder exists

# Display basic information about the dataset
print(df.info())
print(df.head())

# Set plot style
plt.style.use("ggplot")

# Plot distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(x=df["is_addicted"])
plt.title("Distribution of Addiction Status")
plt.xlabel("Is Addicted")
plt.ylabel("Count")
plt.savefig(os.path.join(output_dir, "addiction_distribution.png"))
plt.show()

# Plot age distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig(os.path.join(output_dir, "age_distribution.png"))
plt.show()

# Boxplot: Daily time spent vs addiction status
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["is_addicted"], y=df["Daily_Minutes_Spent"])
plt.title("Time Spent vs Addiction Status")
plt.xlabel("Is Addicted")
plt.ylabel("Daily Minutes Spent")
plt.savefig(os.path.join(output_dir, "time_spent_vs_addiction.png"))
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.show()
