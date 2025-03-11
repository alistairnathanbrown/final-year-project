import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

# MODEL WITH LONG LAT REMOVED

file_path = "RDS4ML.csv"  # Update with your CSV file path
data = pd.read_csv(file_path)

data = data.drop(columns=["Owner"])

# Replace invalid or blank entries in numerical columns with NaN
numerical_cols = [
    "Capacity (MW)", "Start year", "Latitude", "Longitude",
    "Estimated Project Cost (GBP)", "Annual Wind Speed (m/s)",
    "Distance to Shore For Offshore (km)", "Ocean Depth For Offshore (m)",
    "Inflation in Project Country (HCPI)", "Energy Inflation in Project Country (EPI)"
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

# Drop 'Longitude' and 'Latitude' columns
data = data.drop(columns=["Longitude", "Latitude"])

# Define features (X) and target (y)
X = data.drop(columns=["Cancelled"])
y = data["Cancelled"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Identify categorical and numerical columns
categorical_cols = ["Installation Type", "Country"]  # Including country as categorical
numerical_cols = [
    "Capacity (MW)", "Start year", "Estimated Project Cost (GBP)",
    "Annual Wind Speed (m/s)", "Distance to Shore For Offshore (km)",
    "Ocean Depth For Offshore (m)", "Inflation in Project Country (HCPI)",
    "Energy Inflation in Project Country (EPI)"
]

# Preprocessing for categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# Define the model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, class_weight="balanced"))
    ]
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")







# Get the feature importances from the model
feature_importances = model.named_steps["classifier"].feature_importances_

# Get the feature names from the preprocessing pipeline
num_features = numerical_cols
cat_features = model.named_steps["preprocessor"].transformers_[1][1].get_feature_names_out(categorical_cols)

# Combine numerical and categorical feature names
all_features = np.concatenate([num_features, cat_features])

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({
    "Feature": all_features,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

# Display the top 10 most important features
print("Top 10 Important Features:")
print(importance_df.head(40))

# Plot the top 10 features
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"].head(40), importance_df["Importance"].head(40), color="skyblue")
plt.gca().invert_yaxis()  # Invert y-axis to show the highest importance at the top
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Feature Importances")
plt.show()