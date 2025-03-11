from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

# XGBOOST MODEL

# Reload data to ensure we have longitude & latitude
file_path = "RDS5ML.csv"  # Update with your CSV file path
data = pd.read_csv(file_path)

# Drop 'Owner' as before
data = data.drop(columns=["Owner"])

# Replace invalid or blank entries in numerical columns with NaN
numerical_cols = [
    "Capacity (MW)", "Start year", "Latitude", "Longitude",
    "Estimated Project Cost (GBP)", "Annual Wind Speed (m/s)",
    "Distance to Shore For Offshore (km)", "Ocean Depth For Offshore (m)",
    "Inflation in Project Country (HCPI)", "Energy Inflation in Project Country (EPI)",
    "Government Debt as Percentage of GDP", "Country Credit Rating", "Country GDP Growth Rate"
]
data[numerical_cols] = data[numerical_cols].replace({"?": None, "-": None, "": None}).apply(pd.to_numeric, errors="coerce")

# Impute missing values in numerical columns with the median
num_imputer = SimpleImputer(strategy="median")
data[numerical_cols] = num_imputer.fit_transform(data[numerical_cols])

# Handle missing categorical values
data["Installation Type"] = data["Installation Type"].fillna("Unknown")
data["Country"] = data["Country"].fillna("Unknown")

# Fill N/A for onshore projects
data["Distance to Shore For Offshore (km)"] = data["Distance to Shore For Offshore (km)"].fillna(0)
data["Ocean Depth For Offshore (m)"] = data["Ocean Depth For Offshore (m)"].fillna(0)

# Apply K-Means Clustering on Latitude & Longitude
scaler = StandardScaler()
coords = data[["Latitude", "Longitude"]]
coords_scaled = scaler.fit_transform(coords)

kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
data["Location Cluster"] = kmeans.fit_predict(coords_scaled)

# Drop 'Longitude', 'Latitude', and 'Country' columns as they are no longer needed
data = data.drop(columns=["Longitude", "Latitude", "Country"])

# Define features (X) and target (y)
X = data.drop(columns=["Cancelled"])
y = data["Cancelled"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Separate categorical and numerical columns
categorical_cols = ["Installation Type", "Location Cluster"]  # Cluster replaces 'Country'
numerical_cols = [
    "Capacity (MW)", "Start year", "Estimated Project Cost (GBP)",
    "Annual Wind Speed (m/s)", "Distance to Shore For Offshore (km)",
    "Ocean Depth For Offshore (m)", "Inflation in Project Country (HCPI)",
    "Energy Inflation in Project Country (EPI)", "Government Debt as Percentage of GDP",
    "Country Credit Rating", "Country GDP Growth Rate"
]

# Encode categorical features
encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
X_train_encoded = pd.DataFrame(
    encoder.fit_transform(X_train[categorical_cols]),
    columns=encoder.get_feature_names_out(categorical_cols),
    index=X_train.index
)
X_test_encoded = pd.DataFrame(
    encoder.transform(X_test[categorical_cols]),
    columns=encoder.get_feature_names_out(categorical_cols),
    index=X_test.index
)

# Combine numerical and encoded categorical features
X_train_preprocessed = pd.concat([X_train[numerical_cols], X_train_encoded], axis=1)
X_test_preprocessed = pd.concat([X_test[numerical_cols], X_test_encoded], axis=1)

# Scale numerical features before applying SMOTE
scaler = StandardScaler()
X_train_preprocessed[numerical_cols] = scaler.fit_transform(X_train_preprocessed[numerical_cols])
X_test_preprocessed[numerical_cols] = scaler.transform(X_test_preprocessed[numerical_cols])

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

# Train an XGBoost model
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8
)
xgb_model.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = xgb_model.predict(X_test_preprocessed)
y_pred_prob = xgb_model.predict_proba(X_test_preprocessed)[:, 1]

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Feature Importance Analysis
feature_importances = xgb_model.feature_importances_
feature_names = X_train_preprocessed.columns

# Sort features by importance
sorted_idx = np.argsort(feature_importances)[::-1]
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_importances = feature_importances[sorted_idx]

# Plot feature importances
plt.figure(figsize=(12, 8))
plt.barh(sorted_features[:10], sorted_importances[:10], color="skyblue")
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances (XGBoost)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()
