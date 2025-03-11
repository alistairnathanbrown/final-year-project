import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Load the dataset
df = pd.read_csv("GPTDataset.csv")

# Ensure 'Cancelled' is treated as a categorical variable
df['Cancelled'] = df['Cancelled'].astype(int)

# Creating a stratified split object
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in sss.split(df, df['Cancelled']):
    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]

# Save the split datasets
train_df.to_csv("train_set.csv", index=False)
test_df.to_csv("test_set.csv", index=False)

# Print Entry IDs for reference
print("Train Set Entry IDs:")
print(train_df['Entry ID'].tolist())
print("\nTest Set Entry IDs:")
print(test_df['Entry ID'].tolist())