import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

# MODEL WITH HYPER PARAMETER TUNING

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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Separate categorical and numerical columns
categorical_cols = ["Installation Type", "Country"]
numerical_cols = [
    "Capacity (MW)", "Start year", "Estimated Project Cost (GBP)",
    "Annual Wind Speed (m/s)", "Distance to Shore For Offshore (km)",
    "Ocean Depth For Offshore (m)", "Inflation in Project Country (HCPI)",
    "Energy Inflation in Project Country (EPI)"
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

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

# Scale numerical features
scaler = StandardScaler()
X_train_resampled[numerical_cols] = scaler.fit_transform(X_train_resampled[numerical_cols])
X_test_preprocessed[numerical_cols] = scaler.transform(X_test_preprocessed[numerical_cols])

# Define the parameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

# Initialize the RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    verbose=2,
    n_jobs=-1
)

# Fit GridSearchCV to the resampled training data
grid_search.fit(X_train_resampled, y_train_resampled)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best ROC-AUC Score from GridSearchCV:", grid_search.best_score_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_preprocessed)
y_pred_prob = best_model.predict_proba(X_test_preprocessed)[:, 1]

# Print classification report and ROC-AUC score
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"Test Set ROC-AUC Score: {roc_auc:.4f}")
