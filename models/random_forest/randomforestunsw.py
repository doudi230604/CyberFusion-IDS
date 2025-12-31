import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
_plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))
os.makedirs(_plots_dir, exist_ok=True)
_plot_counters = {}
def _savefig(prefix):
    cnt = _plot_counters.get(prefix, 0) + 1
    _plot_counters[prefix] = cnt
    fname = f"{prefix}_fig{cnt}.png"
    path = os.path.join(_plots_dir, fname)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    print(f"Saved figure: {path}")
    plt.close()

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

def load_unsw_nb15():
    """Load UNSW-NB15 dataset from CSV files."""
    print("Loading UNSW-NB15 dataset...")
    
    # Try different possible file locations
    possible_paths = [
        Path("./UNSW-NB15.csv"),
        Path("./datasets/UNSW-NB15.csv"),
        Path("../UNSW-NB15.csv"),
        Path("UNSW_NB15_training-set.csv"),
        Path("UNSW_NB15_testing-set.csv")
    ]
    
    df = None
    for path in possible_paths:
        if path.exists():
            print(f"Found dataset at: {path}")
            try:
                # Try reading with different encodings
                for encoding in ['utf-8', 'latin-1', 'ISO-8859-1']:
                    try:
                        df = pd.read_csv(path, encoding=encoding)
                        print(f"Successfully loaded with {encoding} encoding")
                        break
                    except:
                        continue
                
                if df is not None:
                    break
            except Exception as e:
                print(f"Error reading {path}: {e}")
    
    if df is None:
        print("Local UNSW-NB15 files not found. Creating synthetic data for demo...")
        df = create_synthetic_unsw_nb15()
    
    return df

def create_synthetic_unsw_nb15(n_samples=50000):
    """Create synthetic UNSW-NB15-like data for demonstration."""
    np.random.seed(42)
    
    data = {}
    
    # Basic connection features (based on UNSW-NB15)
    data['dur'] = np.random.exponential(10, n_samples)
    data['proto'] = np.random.choice(['tcp', 'udp', 'icmp', 'arp', 'ospf', 'igmp'], 
                                     n_samples, p=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02])
    data['service'] = np.random.choice(['-', 'http', 'smtp', 'dns', 'ftp', 'ssh', 'pop3', 'smtp'], 
                                       n_samples, p=[0.4, 0.3, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02])
    data['state'] = np.random.choice(['FIN', 'CON', 'INT', 'REQ', 'RST', 'ACC', 'CLO'], 
                                     n_samples, p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05])
    
    # Packet statistics
    data['spkts'] = np.random.poisson(15, n_samples)
    data['dpkts'] = np.random.poisson(20, n_samples)
    data['sbytes'] = np.random.lognormal(7, 2, n_samples)
    data['dbytes'] = np.random.lognormal(8, 2, n_samples)
    data['rate'] = np.random.exponential(0.5, n_samples)
    
    # TCP flags and timing
    data['sttl'] = np.random.choice([32, 64, 128, 255], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    data['dttl'] = np.random.choice([32, 64, 128, 255], n_samples, p=[0.5, 0.3, 0.15, 0.05])
    data['sload'] = np.random.exponential(1000, n_samples)
    data['dload'] = np.random.exponential(800, n_samples)
    data['sloss'] = np.random.poisson(1, n_samples)
    data['dloss'] = np.random.poisson(1, n_samples)
    
    # Additional features
    data['sinpkt'] = np.random.exponential(0.1, n_samples)
    data['dinpkt'] = np.random.exponential(0.15, n_samples)
    data['sjit'] = np.random.exponential(10, n_samples)
    data['djit'] = np.random.exponential(10, n_samples)
    data['swin'] = np.random.randint(1000, 65535, n_samples)
    data['dwin'] = np.random.randint(1000, 65535, n_samples)
    data['tcprtt'] = np.random.exponential(0.05, n_samples)
    
    # Generate attack patterns (multiple attack types)
    attack_types = np.random.choice(['normal', 'dos', 'probe', 'r2l', 'u2r', 'backdoor'], 
                                    n_samples, p=[0.7, 0.15, 0.05, 0.04, 0.03, 0.03])
    
    # Create binary label: 0 for normal, 1 for attack
    data['label'] = (attack_types != 'normal').astype(int)
    
    # Modify features for specific attack types
    for i in range(n_samples):
        if attack_types[i] == 'dos':
            data['sbytes'][i] *= np.random.lognormal(2, 0.5)
            data['spkts'][i] *= np.random.randint(5, 50)
            data['rate'][i] *= np.random.exponential(10)
        elif attack_types[i] == 'probe':
            data['sttl'][i] = np.random.choice([64, 128])
            data['sinpkt'][i] = np.random.exponential(0.01)
        elif attack_types[i] in ['r2l', 'u2r']:
            data['dbytes'][i] *= np.random.lognormal(3, 1)
            data['dloss'][i] = np.random.poisson(3)
    
    # Add 2% label noise
    noise = np.random.rand(n_samples) < 0.02
    data['label'] = np.where(noise, 1 - data['label'], data['label'])
    
    df = pd.DataFrame(data)
    
    # Rename columns to match original UNSW-NB15 feature names
    column_mapping = {
        'dur': 'dur', 'proto': 'proto', 'service': 'service', 'state': 'state',
        'spkts': 'spkts', 'dpkts': 'dpkts', 'sbytes': 'sbytes', 'dbytes': 'dbytes',
        'rate': 'rate', 'sttl': 'sttl', 'dttl': 'dttl', 'sload': 'sload',
        'dload': 'dload', 'sloss': 'sloss', 'dloss': 'dloss', 'sinpkt': 'sinpkt',
        'dinpkt': 'dinpkt', 'sjit': 'sjit', 'djit': 'djit', 'swin': 'swin',
        'dwin': 'dwin', 'tcprtt': 'tcprtt', 'label': 'label'
    }
    
    df = df.rename(columns=column_mapping)
    
    print(f"Created synthetic UNSW-NB15 dataset with {n_samples} samples")
    print(f"Class distribution: Normal: {(1-data['label']).sum()} ({100*(1-data['label']).mean():.1f}%), "
          f"Attack: {data['label'].sum()} ({100*data['label'].mean():.1f}%)")
    
    return df

def preprocess_data(df):
    """Preprocess the UNSW-NB15 dataset."""
    print("\nPreprocessing data...")
    df_processed = df.copy()
    
    # Identify categorical columns
    categorical_cols = []
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            categorical_cols.append(col)
    
    print(f"Categorical columns: {categorical_cols}")
    
    # Encode categorical variables
    for col in categorical_cols:
        if col != 'label':  # Don't encode the target
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    
    # Handle missing values
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(0)
    
    # Separate features and target
    if 'label' in df_processed.columns:
        X = df_processed.drop("label", axis=1)
        y = df_processed["label"]
    elif 'Label' in df_processed.columns:
        X = df_processed.drop("Label", axis=1)
        y = df_processed["Label"]
    elif 'attack' in df_processed.columns:
        X = df_processed.drop("attack", axis=1)
        y = df_processed["attack"]
    else:
        # Try to find the target column
        target_candidates = ['label', 'Label', 'attack', 'Attack', 'class', 'Class']
        for candidate in target_candidates:
            if candidate in df_processed.columns:
                X = df_processed.drop(candidate, axis=1)
                y = df_processed[candidate]
                break
        else:
            raise ValueError("Could not find target column in dataset")
    
    print(f"Dataset shape: {df_processed.shape}")
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    
    return X, y

def analyze_class_distribution(y):
    """Print class distribution."""
    unique, counts = np.unique(y, return_counts=True)
    print("\n=== CLASS DISTRIBUTION ===")
    for cls, count in zip(unique, counts):
        label = "Attack" if cls == 1 else "Normal"
        percentage = count / len(y) * 100
        print(f"{label}: {count:,} samples ({percentage:.2f}%)")

def create_random_forest():
    """Create optimized Random Forest model."""
    global model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Normal", "Attack"],
                yticklabels=["Normal", "Attack"])
    plt.title("Confusion Matrix - UNSW-NB15 Dataset", fontsize=16, fontweight='bold')
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    _savefig('randomforestunsw')

def plot_figure2_curves(y_test, y_prob, auc_score):
    """Figure 2: ROC Curve and Precision-Recall Curve"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    ax1.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", linewidth=2, color='blue')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel("False Positive Rate", fontsize=12)
    ax1.set_ylabel("True Positive Rate", fontsize=12)
    ax1.set_title("ROC Curve - UNSW-NB15", fontsize=16, fontweight='bold')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    ax2.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}", linewidth=2, color='darkblue')
    ax2.set_xlabel("Recall", fontsize=12)
    ax2.set_ylabel("Precision", fontsize=12)
    ax2.set_title("Precision-Recall Curve - UNSW-NB15", fontsize=16, fontweight='bold')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    _savefig('randomforestunsw')

def plot_figure3_feature_importance(top_n=20):
    """Figure 3: Feature Importance"""
    global feature_importance_df
    
    if feature_importance_df is None:
        print("Model not trained yet!")
        return None
    
    print(f"\n=== TOP {top_n} FEATURE IMPORTANCES ===")
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'], fontsize=10)
    plt.xlabel("Importance", fontsize=12)
    plt.title(f"Top {top_n} Feature Importances - UNSW-NB15", fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{importance:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    _savefig('randomforestunsw')
    return top_features

def plot_figure4_prediction_probability(y_test, y_prob):
    """Figure 4: Prediction Probability Distribution"""
    plt.figure(figsize=(10, 6))
    
    # Separate probabilities by actual class
    normal_probs = y_prob[y_test == 0]
    attack_probs = y_prob[y_test == 1]
    
    # Create histogram
    bins = np.linspace(0, 1, 30)
    plt.hist(normal_probs, bins=bins, alpha=0.5, label='Normal (Actual)', color='green', density=True)
    plt.hist(attack_probs, bins=bins, alpha=0.5, label='Attack (Actual)', color='red', density=True)
    
    # Add vertical line at decision threshold (0.5)
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, linewidth=2, 
               label='Decision Boundary (0.5)')
    
    plt.xlabel("Predicted Probability of Attack", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Prediction Probability Distribution - UNSW-NB15", fontsize=16, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"""
    Normal samples: {len(normal_probs):,}
    Attack samples: {len(attack_probs):,}
    Mean prob (Normal): {normal_probs.mean():.3f}
    Mean prob (Attack): {attack_probs.mean():.3f}
    """
    plt.text(0.65, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    _savefig('randomforestunsw')

def main():
    print("=" * 60)
    print("UNSW-NB15 DATASET ANALYSIS WITH RANDOM FOREST")
    print("=" * 60)
    
    # Load dataset
    df = load_unsw_nb15()
    print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns[:5])}...")  # Show first 5 columns
    
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
    print("GENERATING 4 SEPARATE VISUALIZATIONS")
    print("=" * 60)
    
    print("\nFigure 1: Confusion Matrix")
    plot_figure1_confusion_matrix(y_test, y_pred)
    
    print("\nFigure 2: ROC Curve and Precision-Recall Curve")
    plot_figure2_curves(y_test, y_prob, auc_score)
    
    print("\nFigure 3: Feature Importance")
    plot_figure3_feature_importance(15)
    
    print("\nFigure 4: Prediction Probability Distribution")
    plot_figure4_prediction_probability(y_test, y_prob)
    
    print("=" * 60)
    print(f"🎯 FINAL RESULTS FOR UNSW-NB15")
    print(f"📊 Accuracy: {accuracy*100:.2f}%")
    print(f"📈 AUC Score: {auc_score:.4f}")
    if model.oob_score_:
        print(f"🔄 OOB Score: {model.oob_score_:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
