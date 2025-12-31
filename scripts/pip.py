import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_csv("datasets/unsw_nb15/UNSW_NB15_training-set.csv")
test_df = pd.read_csv("datasets/unsw_nb15/UNSW_NB15_testing-set.csv")

# Separate features and labels
X_train = train_df.drop(["label", "attack_cat", "id"], axis=1, errors="ignore")
y_train = train_df["label"]
X_test = test_df.drop(["label", "attack_cat", "id"], axis=1, errors="ignore")
y_test = test_df["label"]

# Handle missing values
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Encode categorical features
categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    # Combine categories from both datasets to handle unseen values
    all_categories = np.union1d(X_train[col].astype(str).unique(), X_test[col].astype(str).unique())
    le.fit(all_categories)
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# Scale numerical features
numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Convert to NumPy arrays (this is what you requested!)
X_train_array = X_train.values
X_test_array = X_test.values
y_train_array = y_train.values
y_test_array = y_test.values

# Display the classified data as 2D NumPy arrays
print("=== FINAL CLASSIFIED DATA AS 2D NUMPY ARRAYS ===")
print(f"X_train shape: {X_train_array.shape}")  # (rows, features)
print(f"X_test shape: {X_test_array.shape}")    # (rows, features)
print(f"y_train shape: {y_train_array.shape}")  # (rows,)
print(f"y_test shape: {y_test_array.shape}")    # (rows,)

print("\nFirst 5 rows of X_train (features):")
print(X_train_array[:5])

print("\nFirst 5 labels of y_train:")
print(y_train_array[:5])

print(f"\nData types:")
print(f"X_train type: {type(X_train_array)}")
print(f"X_test type: {type(X_test_array)}")

print(f"\nArray confirmation:")
print(f"Features are columns: {X_train_array.shape[1]} columns")
print(f"Data instances are rows: {X_train_array.shape[0]} rows in training set")
print(f"Data instances are rows: {X_test_array.shape[0]} rows in test set")

# Train model and calculate accuracy
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_array, y_train_array)
pred = model.predict(X_test_array)
acc = accuracy_score(y_test_array, pred)

print(f"\nModel Accuracy: {acc:.4f}")