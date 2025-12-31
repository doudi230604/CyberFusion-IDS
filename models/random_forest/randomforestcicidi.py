import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, auc
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore")

# GLOBAL variables
label_encoders = {}
scaler = StandardScaler()
imputer = SimpleImputer(strategy='mean')
model = None
feature_importance_df = None

def load_cic_ids():
    """Load CIC-IDS dataset."""
    print("Loading CIC-IDS dataset...")
    
    # Try different possible file locations (CIC-IDS has many versions)
    possible_files = [
        # Common CIC-IDS filenames
        "CIC-IDS.csv",
        "cic_ids.csv",
        "IDS.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
        
        # With paths
        "./CIC-IDS-2017.csv",
        "./datasets/CIC-IDS.csv",
        "../CIC-IDS.csv",
        "MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    ]
    
    df = None
    for filename in possible_files:
        path = Path(filename)
        if path.exists():
            print(f"Found dataset at: {path}")
            try:
                # CIC-IDS files can be large, so read in chunks or sample
                for encoding in ['utf-8', 'latin-1', 'ISO-8859-1']:
                    try:
                        # Try reading first 1000 rows to check format
                        test_df = pd.read_csv(path, nrows=1000, encoding=encoding)
                        print(f"Successfully loaded sample with {encoding} encoding")
                        
                        # Now read the entire file (or a sample if it's too large)
                        try:
                            # For large files, read 100,000 rows max
                            df = pd.read_csv(path, encoding=encoding, nrows=100000)
                        except:
                            # If still too large, use chunks
                            chunks = []
                            chunk_size = 50000
                            for chunk in pd.read_csv(path, encoding=encoding, chunksize=chunk_size):
                                chunks.append(chunk)
                                if len(chunks) * chunk_size >= 100000:
                                    break
                            df = pd.concat(chunks, ignore_index=True)
                        
                        print(f"Loaded {len(df)} rows")
                        break
                    except:
                        continue
                
                if df is not None:
                    break
            except Exception as e:
                print(f"Error reading {path}: {e}")
    
    if df is None:
        print("Local CIC-IDS files not found. Creating synthetic data for demo...")
        df = create_synthetic_cic_ids()
    
    return df

def create_synthetic_cic_ids(n_samples=60000):
    """Create synthetic CIC-IDS-like data."""
    np.random.seed(44)
    
    data = {}
    
    # Basic flow features
    data['Flow Duration'] = np.random.exponential(1000, n_samples)
    data['Total Fwd Packets'] = np.random.poisson(10, n_samples)
    data['Total Backward Packets'] = np.random.poisson(8, n_samples)
    data['Total Length of Fwd Packets'] = np.random.lognormal(10, 2, n_samples)
    data['Total Length of Bwd Packets'] = np.random.lognormal(9, 2, n_samples)
    
    # Packet timing statistics
    data['Fwd Packet Length Mean'] = np.random.exponential(500, n_samples)
    data['Bwd Packet Length Mean'] = np.random.exponential(400, n_samples)
    data['Flow Bytes/s'] = np.random.lognormal(12, 3, n_samples)
    data['Flow Packets/s'] = np.random.lognormal(8, 2, n_samples)
    
    # TCP flags statistics
    data['Fwd PSH Flags'] = np.random.binomial(1, 0.1, n_samples)
    data['Bwd PSH Flags'] = np.random.binomial(1, 0.05, n_samples)
    data['Fwd URG Flags'] = np.random.binomial(1, 0.01, n_samples)
    data['Bwd URG Flags'] = np.random.binomial(1, 0.005, n_samples)
    
    # Window size
    data['Fwd Init Win Bytes'] = np.random.randint(1000, 65535, n_samples)
    data['Bwd Init Win Bytes'] = np.random.randint(1000, 65535, n_samples)
    
    # Packet timing
    data['Fwd IAT Mean'] = np.random.exponential(100, n_samples)
    data['Bwd IAT Mean'] = np.random.exponential(120, n_samples)
    data['Fwd IAT Std'] = np.random.exponential(50, n_samples)
    data['Bwd IAT Std'] = np.random.exponential(60, n_samples)
    
    # Subflow statistics
    data['Subflow Fwd Packets'] = np.random.poisson(5, n_samples)
    data['Subflow Bwd Packets'] = np.random.poisson(4, n_samples)
    data['Subflow Fwd Bytes'] = np.random.lognormal(9, 1.5, n_samples)
    data['Subflow Bwd Bytes'] = np.random.lognormal(8, 1.5, n_samples)
    
    # Protocol information
    protocols = ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'DNS', 'FTP']
    data['Protocol'] = np.random.choice(protocols, n_samples, p=[0.5, 0.3, 0.05, 0.05, 0.05, 0.03, 0.02])
    
    # Direction
    data['Direction'] = np.random.choice(['Inbound', 'Outbound', 'Local'], n_samples, p=[0.4, 0.4, 0.2])
    
    # Generate various attack types (CIC-IDS has multiple attack categories)
    attack_categories = np.random.choice([
        'BENIGN', 'DDoS', 'PortScan', 'Botnet', 'Infiltration', 
        'WebAttack', 'BruteForce', 'DoS'
    ], n_samples, p=[0.7, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02])
    
    # Create binary label: 0 for BENIGN, 1 for attack
    data['Label'] = (attack_categories != 'BENIGN').astype(int)
    
    # Convert arrays for efficient operations
    total_fwd_packets = data['Total Fwd Packets'].copy()
    flow_packets_s = data['Flow Packets/s'].copy()
    flow_bytes_s = data['Flow Bytes/s'].copy()
    total_backward_packets = data['Total Backward Packets'].copy()
    fwd_iat_mean = data['Fwd IAT Mean'].copy()
    total_length_fwd = data['Total Length of Fwd Packets'].copy()
    total_length_bwd = data['Total Length of Bwd Packets'].copy()
    flow_duration = data['Flow Duration'].copy()
    
    # Modify features based on attack type
    for i in range(n_samples):
        if attack_categories[i] == 'DDoS':
            # DDoS: High packet rate, large flows
            total_fwd_packets[i] *= np.random.randint(10, 100)
            flow_packets_s[i] *= np.random.exponential(10)
            flow_bytes_s[i] *= np.random.lognormal(2, 0.5)
            
        elif attack_categories[i] == 'PortScan':
            # PortScan: Many small packets, different ports
            total_fwd_packets[i] *= np.random.randint(5, 20)
            total_backward_packets[i] = 0  # Often no response
            fwd_iat_mean[i] *= 0.1  # Faster
            
        elif attack_categories[i] == 'Botnet':
            # Botnet: Periodic, encrypted traffic
            total_length_fwd[i] *= np.random.lognormal(1.5, 0.3)
            fwd_iat_mean[i] = np.random.exponential(10000)  # Slower
            
        elif attack_categories[i] == 'Infiltration':
            # Infiltration: Large data transfer
            total_length_fwd[i] *= np.random.lognormal(3, 1)
            total_length_bwd[i] *= np.random.lognormal(2, 0.5)
            
        elif attack_categories[i] == 'WebAttack':
            # WebAttack: HTTP traffic patterns
            if data['Protocol'][i] not in ['HTTP', 'HTTPS']:
                protocols_list = list(protocols)
                data['Protocol'][i] = np.random.choice(['HTTP', 'HTTPS'])
            data['Fwd Packet Length Mean'][i] *= np.random.lognormal(1.2, 0.2)
            
        elif attack_categories[i] in ['BruteForce', 'DoS']:
            # BruteForce/DoS: Many failed connections
            total_fwd_packets[i] *= np.random.randint(3, 15)
            flow_duration[i] *= 0.5  # Shorter
    
    # Update the arrays with modified values
    data['Total Fwd Packets'] = total_fwd_packets
    data['Flow Packets/s'] = flow_packets_s
    data['Flow Bytes/s'] = flow_bytes_s
    data['Total Backward Packets'] = total_backward_packets
    data['Fwd IAT Mean'] = fwd_iat_mean
    data['Total Length of Fwd Packets'] = total_length_fwd
    data['Total Length of Bwd Packets'] = total_length_bwd
    data['Flow Duration'] = flow_duration
    
    # Add some NaN values (real datasets have missing values)
    # Convert integer columns to float first to allow NaN
    integer_cols = ['Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
                   'Fwd Init Win Bytes', 'Bwd Init Win Bytes',
                   'Subflow Fwd Packets', 'Subflow Bwd Packets']
    
    for col in integer_cols:
        if col in data:
            data[col] = data[col].astype(float)
    
    # Add NaN values to random columns
    cols_to_add_nan = np.random.choice(list(data.keys()), size=min(5, len(data)), replace=False)
    for col in cols_to_add_nan:
        if col not in ['Label', 'Protocol', 'Direction']:
            nan_indices = np.random.choice(n_samples, size=int(n_samples*0.01), replace=False)
            if isinstance(data[col], np.ndarray):
                # Create a copy and set values to NaN
                col_data = data[col].copy()
                col_data[nan_indices] = np.nan
                data[col] = col_data
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Add 1% label noise
    noise = np.random.rand(n_samples) < 0.01
    df['Label'] = np.where(noise, 1 - df['Label'], df['Label'])
    
    print(f"Created synthetic CIC-IDS dataset with {n_samples} samples")
    print(f"Class distribution: BENIGN: {(1-df['Label']).sum()} ({100*(1-df['Label']).mean():.1f}%), "
          f"Attack: {df['Label'].sum()} ({100*df['Label'].mean():.1f}%)")
    
    return df

def preprocess_data(df):
    """Preprocess the CIC-IDS dataset."""
    print("\nPreprocessing data...")
    df_processed = df.copy()
    
    # Clean column names (CIC-IDS often has spaces and special characters)
    df_processed.columns = df_processed.columns.str.strip()
    
    # Remove any completely empty columns
    df_processed = df_processed.dropna(axis=1, how='all')
    
    # Identify categorical columns
    categorical_cols = []
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            categorical_cols.append(col)
    
    print(f"Categorical columns: {categorical_cols}")
    
    # Encode categorical variables
    for col in categorical_cols:
        if col.lower() not in ['label', 'attack', 'class']:  # Don't encode the target
            try:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                label_encoders[col] = le
            except:
                # If encoding fails, drop the column for simplicity
                print(f"Dropping column {col} due to encoding issues")
                df_processed = df_processed.drop(col, axis=1)
    
    # Find target column (CIC-IDS uses various names)
    target_col = None
    target_candidates = ['Label', 'label', 'Attack', 'attack', 'Class', 'class', 
                        'Result', 'result', 'malicious']
    
    for candidate in target_candidates:
        if candidate in df_processed.columns:
            target_col = candidate
            break
    
    if target_col is None:
        # Try to find any column with binary values
        for col in df_processed.columns:
            unique_vals = df_processed[col].nunique()
            if unique_vals == 2 and df_processed[col].dtype != 'object':
                target_col = col
                print(f"Using {col} as target (binary column)")
                break
    
    if target_col is None:
        raise ValueError("Could not find target column in dataset")
    
    print(f"Target column: {target_col}")
    
    # Convert target to binary if needed
    if df_processed[target_col].dtype == 'object':
        # Map text labels to binary
        label_mapping = {
            'BENIGN': 0, 'Benign': 0, 'Normal': 0, 'normal': 0,
            'MALICIOUS': 1, 'Malicious': 1, 'Attack': 1, 'attack': 1
        }
        y = df_processed[target_col].map(label_mapping)
        # Fill any unmapped values
        y = y.fillna(df_processed[target_col].apply(
            lambda x: 0 if str(x).lower() in ['benign', 'normal'] else 1
        ))
    else:
        y = df_processed[target_col]
        # Ensure binary (0/1)
        if y.nunique() > 2:
            # If multi-class, convert to binary (majority class vs rest)
            most_common = y.mode()[0]
            y = (y != most_common).astype(int)
    
    # Features
    X = df_processed.drop(target_col, axis=1)
    
    # Handle missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(0)
    
    print(f"Dataset shape: {df_processed.shape}")
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    
    return X, y

def analyze_class_distribution(y):
    """Print class distribution."""
    unique, counts = np.unique(y, return_counts=True)
    print("\n=== CLASS DISTRIBUTION ===")
    for cls, count in zip(unique, counts):
        label = "Attack" if cls == 1 else "Normal/BENIGN"
        percentage = count / len(y) * 100
        print(f"{label}: {count:,} samples ({percentage:.2f}%)")

def create_random_forest():
    """Create optimized Random Forest model for CIC-IDS."""
    global model
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='log2',
        class_weight='balanced_subsample',
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    return model

def train_model(X_train, y_train):
    global model, scaler, imputer, feature_importance_df
    
    print("\n=== TRAINING RANDOM FOREST ===")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    
    # Handle missing values
    X_train_imputed = imputer.fit_transform(X_train)
    
    # Scale features
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    
    # Create and train model
    model = create_random_forest()
    model.fit(X_train_scaled, y_train)
    
    # Feature importance
    feature_importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print(f"Training complete!")
    print(f"OOB Score: {model.oob_score_:.4f}")
    print(f"Top 5 features: {feature_importance_df['feature'].head(5).tolist()}")

def evaluate_model(X_test, y_test):
    global model, scaler, imputer
    
    print("\n=== MODEL EVALUATION ===")
    
    # Preprocess test data
    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"OOB Score: {model.oob_score_:.4f}\n")
    
    # Classification report
    print("=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))
    
    # Return predictions for plotting
    return y_pred, y_prob, accuracy, auc_score

def plot_figure1_confusion_matrix(y_test, y_pred):
    """Figure 1: Confusion Matrix"""
    plt.figure(figsize=(8, 6))
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", 
                xticklabels=["Normal", "Attack"],
                yticklabels=["Normal", "Attack"])
    plt.title("Confusion Matrix", fontsize=16, fontweight='bold')
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_figure2_curves(y_test, y_prob):
    """Figure 2: ROC Curve and Precision-Recall Curve"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    
    ax1.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", linewidth=2, color='red')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel("False Positive Rate", fontsize=12)
    ax1.set_ylabel("True Positive Rate", fontsize=12)
    ax1.set_title("ROC Curve", fontsize=16, fontweight='bold')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    ax2.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}", linewidth=2, color='darkred')
    ax2.set_xlabel("Recall", fontsize=12)
    ax2.set_ylabel("Precision", fontsize=12)
    ax2.set_title("Precision-Recall Curve", fontsize=16, fontweight='bold')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_figure3_feature_importance(top_n=20):
    """Figure 3: Feature Importance"""
    global feature_importance_df
    
    if feature_importance_df is None:
        print("Model not trained yet!")
        return None
    
    print(f"\n=== TOP {top_n} FEATURE IMPORTANCES ===")
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_features)))
    bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel("Importance", fontsize=12)
    plt.title(f"Top {top_n} Feature Importances", fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{importance:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return top_features

def plot_figure4_prediction_probability(y_test, y_prob):
    """Figure 4: Prediction Probability Distribution"""
    plt.figure(figsize=(10, 6))
    
    # Separate probabilities by actual class
    normal_probs = y_prob[y_test == 0]
    attack_probs = y_prob[y_test == 1]
    
    # Create histogram
    bins = np.linspace(0, 1, 30)
    plt.hist(normal_probs, bins=bins, alpha=0.5, label='Normal (Actual)', color='blue', density=True)
    plt.hist(attack_probs, bins=bins, alpha=0.5, label='Attack (Actual)', color='red', density=True)
    
    # Add vertical line at decision threshold (0.5)
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Decision Boundary (0.5)')
    
    plt.xlabel("Predicted Probability of Attack", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Prediction Probability Distribution", fontsize=16, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Add some statistics
    stats_text = f"""
    Normal samples: {len(normal_probs):,}
    Attack samples: {len(attack_probs):,}
    Mean prob (Normal): {normal_probs.mean():.3f}
    Mean prob (Attack): {attack_probs.mean():.3f}
    """
    plt.text(0.65, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

def main():
    print("=" * 60)
    print("CIC-IDS DATASET ANALYSIS WITH RANDOM FOREST")
    print("=" * 60)
    
    # Load dataset
    df = load_cic_ids()
    print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    if len(df.columns) > 10:
        print(f"First 10 columns: {list(df.columns[:10])}...")
    else:
        print(f"Columns: {list(df.columns)}")
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Analyze class distribution
    analyze_class_distribution(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nData split:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    train_model(X_train, y_train)
    
    # Evaluate model and get predictions
    y_pred, y_prob, accuracy, auc_score = evaluate_model(X_test, y_test)
    
    # Plot the 4 figures
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    print("\nFigure 1: Confusion Matrix")
    plot_figure1_confusion_matrix(y_test, y_pred)
    
    print("\nFigure 2: ROC Curve and Precision-Recall Curve")
    plot_figure2_curves(y_test, y_prob)
    
    print("\nFigure 3: Feature Importance")
    plot_figure3_feature_importance(15)
    
    print("\nFigure 4: Prediction Probability Distribution")
    plot_figure4_prediction_probability(y_test, y_prob)
    
    print("=" * 60)
    print(f"🎯 FINAL RESULTS FOR CIC-IDS")
    print(f"📊 Accuracy: {accuracy*100:.2f}%")
    print(f"📈 AUC Score: {auc_score:.4f}")
    if model.oob_score_:
        print(f"🔄 OOB Score: {model.oob_score_:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
