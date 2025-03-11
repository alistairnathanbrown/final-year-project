import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import seaborn as sns

# Load the data
file_path = "RDS4ML.csv"  # Update with your CSV file path
data = pd.read_csv(file_path)

# Drop unnecessary columns and handle missing values
data = data.drop(columns=["Owner"])

# Handle missing numerical and categorical columns (like before)
numerical_cols = [
    "Capacity (MW)", "Start year", "Latitude", "Longitude",
    "Estimated Project Cost (GBP)", "Annual Wind Speed (m/s)",
    "Distance to Shore For Offshore (km)", "Ocean Depth For Offshore (m)",
    "Inflation in Project Country (HCPI)", "Energy Inflation in Project Country (EPI)"
]
data[numerical_cols] = data[numerical_cols].replace({"?": None, "-": None, "": None}).apply(pd.to_numeric, errors="coerce")

# Impute missing values for numerical columns
num_imputer = SimpleImputer(strategy="median")
data[numerical_cols] = num_imputer.fit_transform(data[numerical_cols])

# Handle missing categorical values
data["Installation Type"] = data["Installation Type"].fillna("Unknown")
data["Country"] = data["Country"].fillna("Unknown")

# Preprocessing categorical 'Country' feature
label_encoder = LabelEncoder()
data['Country_encoded'] = label_encoder.fit_transform(data['Country'])

# Combine Longitude, Latitude, and encoded Country for clustering
X_clustering = data[['Longitude', 'Latitude', 'Country_encoded']]

# Perform KMeans Clustering with k=7
kmeans = KMeans(n_clusters=7, random_state=42)
data['Cluster_Label'] = kmeans.fit_predict(X_clustering)

# Visualize the Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Longitude', y='Latitude', hue='Cluster_Label', palette='Set1', data=data, s=100, alpha=0.7)
plt.title('Geographical Clustering of Wind Projects (k=7)', fontsize=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Cluster')
plt.show()

# Visualize cancellations and clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Longitude', y='Latitude', hue='Cancelled', palette='viridis', style='Cluster_Label', data=data, s=100, alpha=0.7)
plt.title('Wind Project Cancellations and Cluster Distribution', fontsize=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Cancellation Status')
plt.show()
