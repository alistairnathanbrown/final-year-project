import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import geopandas as gpd

# Load the data
file_path = "RDS5ML.csv"  # Update with your CSV file path
data = pd.read_csv(file_path)

# Drop unnecessary columns and handle missing values
data = data.drop(columns=["Owner"])

# Handle missing numerical and categorical columns
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

# Preprocessing the categorical 'Country' feature
label_encoder = LabelEncoder()
data['Country_encoded'] = label_encoder.fit_transform(data['Country'])

# Combine Longitude, Latitude, and encoded Country for clustering
X_clustering = data[['Longitude', 'Latitude', 'Country_encoded']]

# Perform KMeans clustering with k=7
kmeans = KMeans(n_clusters=7, random_state=42)
data['Cluster_Label'] = kmeans.fit_predict(X_clustering)

# Load custom country map using the provided shapefile
map_file = "shape_files/ne_10m_admin_0_countries.shp"
world = gpd.read_file(map_file)

# Define European bounds (approximate)
# Eastern bound set to 55 and lower latitude set to 25
europe_xlim = (-25, 55)
europe_ylim = (25, 72)

# ----- Cluster Visualization on European Map -----
plt.figure(figsize=(12, 8), dpi=300)
ax = world.plot(color='lightgrey', edgecolor='white', figsize=(12, 8))
plt.scatter(data['Longitude'], data['Latitude'],
            c=data['Cluster_Label'], cmap='Set2', s=30, alpha=0.6, edgecolor='k')
plt.title('Wind Project Clusters', fontsize=16)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
ax.set_xlim(europe_xlim)
ax.set_ylim(europe_ylim)
plt.savefig("exports/cluster_map.png", dpi=300, bbox_inches='tight')
# plt.show()

# ----- Cancelled Projects Visualization on European Map -----
plt.figure(figsize=(12, 8), dpi=300)
ax = world.plot(color='lightgrey', edgecolor='white', figsize=(12, 8))
sns.scatterplot(data=data, x="Longitude", y="Latitude", hue="Cancelled",
                palette="Set1", s=30, alpha=0.6, edgecolor='k', ax=ax)
plt.title("Wind Projects: Cancelled vs Not Cancelled", fontsize=16)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
ax.set_xlim(europe_xlim)
ax.set_ylim(europe_ylim)
plt.legend(title="Cancelled", fontsize=10)
plt.savefig("exports/cancelled_projects.png", dpi=300, bbox_inches='tight')
# plt.show()
