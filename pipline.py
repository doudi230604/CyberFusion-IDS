import pandas as pd
import numpy as np
import os

try:
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
except ImportError:
    print("Error: scikit-learn is not installed. Please run: pip install scikit-learn")
    exit(1)

def load_unsw_nb15():
    """Load UNSW-NB15 dataset and return train/test splits - EXACT PDF VERSION"""
    
    # Use your actual file paths but keep PDF structure
    train_path = 'datasets/unsw_nb15/UNSW_NB15_training-set.csv'
    test_path = 'datasets/unsw_nb15/UNSW_NB15_testing-set.csv'
    
    # Read the training and testing files
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Training set shape: {train_df.shape}")
    print(f"Testing set shape: {test_df.shape}")

    # Separate features and labels - EXACTLY as in PDF
    # The label column is called 'label' (0=normal, 1=attack)
    X_train = train_df.drop(['label', 'attack_cat', 'id'], axis=1, errors='ignore')
    y_train = train_df['label']

    X_test = test_df.drop(['label', 'attack_cat', 'id'], axis=1, errors='ignore')
    y_test = test_df['label']

    print(f"Features after dropping labels: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test

def handle_missing_values(X_train, X_test):
    """Check and handle missing values - EXACT PDF VERSION"""
    print("Missing values in training set:")
    print(X_train.isnull().sum().sum())
    print("Missing values in testing set:")
    print(X_test.isnull().sum().sum())

    # Fill missing values with 0 - EXACT PDF VERSION
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    return X_train, X_test

def identify_column_types(X_train):
    """Identify categorical and numerical columns - EXACT PDF VERSION"""
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {len(numerical_cols)}")

    return categorical_cols, numerical_cols

def encode_categorical_features(X_train, X_test, categorical_cols):
    """Encode categorical features using Label Encoding - FIXED VERSION"""
    label_encoders = {}

    for col in categorical_cols:
        if col in X_train.columns:
            le = LabelEncoder()
            
            # FIX: Combine train + test to handle all categories
            combined = pd.concat([X_train[col], X_test[col]], axis=0)
            le.fit(combined.astype(str))
            
            # Transform both train and test
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            label_encoders[col] = le

    print(f"Encoded {len(categorical_cols)} categorical columns")
    return X_train, X_test, label_encoders

def scale_numerical_features(X_train, X_test, numerical_cols):
    """Scale numerical features using StandardScaler - EXACT PDF VERSION"""
    scaler = StandardScaler()

    # Scale numerical columns
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    print("Numerical features scaled")
    return X_train, X_test, scaler

def convert_to_numpy(X_train, X_test, y_train, y_test):
    """Convert to 2D NumPy arrays as requested - EXACT PDF VERSION"""
    # Convert to NumPy arrays
    X_train_array = X_train.values.astype(np.float32)
    X_test_array = X_test.values.astype(np.float32)
    y_train_array = y_train.values.astype(np.int32)
    y_test_array = y_test.values.astype(np.int32)

    print(f"Final shapes:")
    print(f"X_train: {X_train_array.shape} (2D array: {X_train_array.ndim}D)")
    print(f"X_test: {X_test_array.shape} (2D array: {X_test_array.ndim}D)")
    print(f"y_train: {y_train_array.shape}")
    print(f"y_test: {y_test_array.shape}")

    return X_train_array, X_test_array, y_train_array, y_test_array

def full_preprocessing_pipeline():
    """Complete pipeline from raw data to ready-to-use arrays - EXACT PDF VERSION"""
    
    # Step 1: Load data
    X_train, X_test, y_train, y_test = load_unsw_nb15()
    
    # Step 2: Handle missing values
    X_train, X_test = handle_missing_values(X_train, X_test)
    
    # Step 3: Identify column types
    categorical_cols, numerical_cols = identify_column_types(X_train)
    
    # Step 4: Encode categorical features
    X_train, X_test, encoders = encode_categorical_features(X_train, X_test, categorical_cols)
    
    # Step 5: Scale numerical features
    X_train, X_test, scaler = scale_numerical_features(X_train, X_test, numerical_cols)
    
    # Step 6: Convert to NumPy arrays
    X_train_final, X_test_final, y_train_final, y_test_final = convert_to_numpy(X_train, X_test, y_train, y_test)
    
    return (X_train_final, X_test_final, y_train_final, y_test_final, encoders, scaler)

# Check if files exist first
print("Checking file existence...")
train_path = 'datasets/unsw_nb15/UNSW_NB15_training-set.csv'
test_path = 'datasets/unsw_nb15/UNSW_NB15_testing-set.csv'

if not os.path.exists(train_path):
    print(f"❌ Training file not found: {train_path}")
    print("Please make sure your files are in: datasets/unsw_nb15/")
    exit(1)
    
if not os.path.exists(test_path):
    print(f"❌ Testing file not found: {test_path}")
    print("Please make sure your files are in: datasets/unsw_nb15/")
    exit(1)

print("✅ Files found! Running EXACT PDF version pipeline...")

# Run the complete pipeline - EXACT PDF VERSION
(X_train, X_test, y_train, y_test, feature_encoders, feature_scaler) = full_preprocessing_pipeline()

print("\n" + "="*50)
print("Pipeline completed successfully!")
print("Final output: 2D NumPy arrays where features are columns and data instances are rows, ready for machine learning!")
print("="*50)
