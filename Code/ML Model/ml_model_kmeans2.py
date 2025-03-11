import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load the data
file_path = "RDS5ML.csv"  # Update with your CSV file path
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

# Elbow Method to find the optimal number of clusters
wcss = []
for k in range(1, 11):  # Testing for 1 to 10 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_clustering)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Curve
plt.figure(figsize=(8, 6), dpi=300)  # High-quality figure
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.savefig("exports/elbow_diagram.png", dpi=300, bbox_inches='tight')  # Save as high-quality PNG
plt.show()