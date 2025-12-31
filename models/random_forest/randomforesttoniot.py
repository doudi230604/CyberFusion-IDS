import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# plots helper
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

from pathlib import Path
import joblib
import pickle
from datetime import datetime
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    f1_score, balanced_accuracy_score
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

def load_ton_iot(sample_size=10000):
    """Load ToN-IoT dataset with sampling for faster execution."""
    print("Loading ToN-IoT dataset...")
    
    # Try different possible file locations
    possible_paths = [
        Path("./TON_IoT.csv"),
        Path("./ToN-IoT.csv"),
        Path("./datasets/ToN-IoT.csv"),
        Path("./datasets/TON_IoT.csv"),
        Path("../ToN-IoT.csv"),
        Path("../TON_IoT.csv")
    ]
    
    df = None
    for path in possible_paths:
        if path.exists():
            print(f"Found dataset at: {path}")
            try:
                # Read only the first sample_size rows for speed
                df = pd.read_csv(path, nrows=sample_size)
                print(f"Successfully loaded {len(df)} samples for faster execution")
                break
            except Exception as e:
                print(f"Error reading {path}: {e}")
    
    if df is None:
        print("Local ToN-IoT files not found. Creating synthetic data for demo...")
        df = create_synthetic_ton_iot(min(sample_size, 20000))
    
    return df

def create_synthetic_ton_iot(n_samples=10000):
    """Create synthetic ToN-IoT-like data - optimized version."""
    np.random.seed(43)
    
    data = {}
    
    # IoT-specific features - simplified for speed
    data['src_port'] = np.random.randint(1024, 65535, n_samples)
    data['dst_port'] = np.random.choice([80, 443, 22, 53, 1883, 5683, 8883], n_samples)
    
    # Protocol distribution (IoT protocols)
    protocols = ['mqtt', 'coap', 'http', 'https', 'modbus', 'amqp', 'dns']
    data['protocol'] = np.random.choice(protocols, n_samples, p=[0.35, 0.25, 0.2, 0.1, 0.05, 0.03, 0.02])
    
    # Simplified feature generation
    data['pkt_len'] = np.random.exponential(200, n_samples)
    data['mqtt_len'] = np.where(data['protocol'] == 'mqtt', np.random.exponential(180, n_samples), 0)
    data['coap_len'] = np.where(data['protocol'] == 'coap', np.random.exponential(80, n_samples), 0)
    
    # IoT device types
    device_types = ['thermostat', 'camera', 'light_bulb', 'smart_lock', 'sensor', 'hub']
    data['device_type'] = np.random.choice(device_types, n_samples, p=[0.25, 0.2, 0.2, 0.15, 0.15, 0.05])
    
    # Network metrics
    data['duration'] = np.random.exponential(5, n_samples)
    data['bytes'] = np.random.lognormal(10, 2, n_samples)
    data['packets'] = np.random.poisson(10, n_samples)
    data['bytes_sent'] = np.random.lognormal(8, 1.5, n_samples)
    data['bytes_received'] = np.random.lognormal(9, 1.5, n_samples)
    
    # IoT-specific metrics
    data['response_time'] = np.random.exponential(0.1, n_samples)
    data['connection_rate'] = np.random.exponential(0.5, n_samples)
    
    # Security metrics
    data['encryption'] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    data['auth_attempts'] = np.random.poisson(1, n_samples)
    
    # Generate attack patterns
    attack_prob = 0.25  # 25% attacks for balanced dataset
    labels = np.random.choice([0, 1], n_samples, p=[1-attack_prob, attack_prob])
    
    # Modify attack samples
    attack_mask = labels == 1
    if attack_mask.any():
        # Scale up attack features
        data['bytes'][attack_mask] *= np.random.lognormal(2, 1, attack_mask.sum())
        data['packets'][attack_mask] *= np.random.randint(5, 20, attack_mask.sum())
        data['connection_rate'][attack_mask] *= np.random.exponential(5, attack_mask.sum())
        data['encryption'][attack_mask] = 0  # No encryption for attacks
    
    data['label'] = labels
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Add 1.5% label noise
    noise = np.random.rand(n_samples) < 0.015
    df['label'] = np.where(noise, 1 - df['label'], df['label'])
    
    print(f"Created synthetic ToN-IoT dataset with {n_samples} samples")
    print(f"Class distribution: Normal: {(1-df['label']).sum()} ({100*(1-df['label']).mean():.1f}%), "
          f"Attack: {df['label'].sum()} ({100*df['label'].mean():.1f}%)")
    
    return df

def preprocess_data(df, max_features=50):
    """Optimized preprocessing - limit features for speed."""
    print("\nPreprocessing data...")
    start_time = time.time()
    
    df_processed = df.copy()
    
    # Drop unnecessary columns for speed
    drop_cols = [col for col in df_processed.columns if 'ts' in col.lower() or 'ip' in col.lower()]
    if drop_cols:
        df_processed = df_processed.drop(columns=drop_cols)
    
    # Identify categorical columns
    categorical_cols = []
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            categorical_cols.append(col)
    
    print(f"Categorical columns: {categorical_cols}")
    
    # Simplified encoding - faster
    for col in categorical_cols:
        if col != 'label':
            try:
                # Use factorize for speed
                df_processed[col], _ = pd.factorize(df_processed[col])
            except:
                df_processed = df_processed.drop(columns=[col])
    
    # Separate features and target
    if 'label' in df_processed.columns:
        X = df_processed.drop("label", axis=1)
        y = df_processed["label"]
    else:
        raise ValueError("Could not find target column in dataset")
    
    # Limit number of features for speed
    if X.shape[1] > max_features:
        # Keep top features by variance
        variances = X.var().sort_values(ascending=False)
        keep_cols = variances.head(max_features).index.tolist()
        X = X[keep_cols]
        print(f"Limited to {max_features} features for faster execution")
    
    # Fill missing values
    X = X.fillna(0)
    
    preprocessing_time = time.time() - start_time
    print(f"Preprocessing completed in {preprocessing_time:.2f} seconds")
    print(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y

def analyze_class_distribution(y):
    """Print class distribution."""
    unique, counts = np.unique(y, return_counts=True)
    print("\n=== CLASS DISTRIBUTION ===")
    for cls, count in zip(unique, counts):
        label = "Attack" if cls == 1 else "Normal"
        percentage = count / len(y) * 100
        print(f"{label}: {count:,} samples ({percentage:.2f}%)")
    
    return dict(zip(unique, counts))

def create_lightweight_random_forest():
    """Create optimized but lightweight Random Forest."""
    global model
    
    model = RandomForestClassifier(
        n_estimators=50,  # Reduced for speed
        max_depth=15,
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

def train_model_fast(X_train, y_train):
    """Optimized training function."""
    global model, scaler, imputer, feature_importance_df
    
    print("\n=== TRAINING LIGHTWEIGHT RANDOM FOREST ===")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    
    start_time = time.time()
    
    # Simple scaling - faster
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Create and train model
    model = create_lightweight_random_forest()
    
    # Train with progress indicator
    print("Training model...", end="")
    model.fit(X_train_scaled, y_train)
    print(" Done!")
    
    # Feature importance
    feature_importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds!")
    print(f"OOB Score: {model.oob_score_:.4f}")
    print(f"Top 5 features: {feature_importance_df['feature'].head(5).tolist()}")
    
    return model

def evaluate_model_fast(X_test, y_test):
    """Optimized evaluation."""
    global model, scaler
    
    print("\n=== MODEL EVALUATION ===")
    start_time = time.time()
    
    # Preprocess test data
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    evaluation_time = time.time() - start_time
    
    print(f"Evaluation completed in {evaluation_time:.2f} seconds")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    if hasattr(model, 'oob_score_'):
        print(f"OOB Score: {model.oob_score_:.4f}\n")
    
    # Classification report
    print("=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))
    
    return y_pred, y_prob, accuracy, auc_score, f1, balanced_acc

def plot_figure1_confusion_matrix(y_test, y_pred):
    """Figure 1: Confusion Matrix"""
    plt.figure(figsize=(8, 6))
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", 
                xticklabels=["Normal", "Attack"],
                yticklabels=["Normal", "Attack"])
    plt.title("Confusion Matrix - ToN-IoT Dataset", fontsize=16, fontweight='bold')
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    _savefig('randomforesttoniot')

def plot_figure2_curves(y_test, y_prob, auc_score):
    """Figure 2: ROC Curve and Precision-Recall Curve"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    ax1.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", linewidth=2, color='green')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel("False Positive Rate", fontsize=12)
    ax1.set_ylabel("True Positive Rate", fontsize=12)
    ax1.set_title("ROC Curve - ToN-IoT", fontsize=16, fontweight='bold')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    ax2.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}", linewidth=2, color='darkgreen')
    ax2.set_xlabel("Recall", fontsize=12)
    ax2.set_ylabel("Precision", fontsize=12)
    ax2.set_title("Precision-Recall Curve - ToN-IoT", fontsize=16, fontweight='bold')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    _savefig('randomforesttoniot')

def plot_figure3_feature_importance(top_n=15):
    """Figure 3: Feature Importance - optimized"""
    global feature_importance_df
    
    if feature_importance_df is None:
        print("Model not trained yet!")
        return None
    
    print(f"\n=== TOP {top_n} FEATURE IMPORTANCES ===")
    
    # Limit to top_n features
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(top_features)))
    bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'], fontsize=10)
    plt.xlabel("Importance", fontsize=12)
    plt.title(f"Top {top_n} Feature Importances - ToN-IoT", fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    _savefig('randomforesttoniot')
    return top_features

def plot_figure4_prediction_probability(y_test, y_prob):
    """Figure 4: Prediction Probability Distribution - optimized"""
    plt.figure(figsize=(10, 6))
    
    # Separate probabilities by actual class
    normal_probs = y_prob[y_test == 0]
    attack_probs = y_prob[y_test == 1]
    
    # Create histogram
    bins = np.linspace(0, 1, 20)  # Fewer bins for speed
    
    plt.hist(normal_probs, bins=bins, alpha=0.5, label='Normal (Actual)', color='blue', density=True)
    plt.hist(attack_probs, bins=bins, alpha=0.5, label='Attack (Actual)', color='red', density=True)
    
    # Add vertical line at decision threshold
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, linewidth=2, 
               label='Decision Boundary (0.5)')
    
    plt.xlabel("Predicted Probability of Attack", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Prediction Probability Distribution - ToN-IoT", fontsize=16, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    _savefig('randomforesttoniot')

def main():
    """Main function with optimized execution."""
    print("=" * 60)
    print("ToN-IoT DATASET ANALYSIS - OPTIMIZED VERSION")
    print("Using subset for faster execution")
    print("=" * 60)
    
    total_start_time = time.time()
    
    # Configuration for fast execution
    SAMPLE_SIZE = 10000  # Limit dataset size
    MAX_FEATURES = 30    # Limit features
    
    # Load dataset with sampling
    print(f"\nLoading dataset (max {SAMPLE_SIZE} samples)...")
    df = load_ton_iot(sample_size=SAMPLE_SIZE)
    
    # Preprocess data
    X, y = preprocess_data(df, max_features=MAX_FEATURES)
    
    # Analyze class distribution
    analyze_class_distribution(y)
    
    # Split data (smaller test size for speed)
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # Smaller test size
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    train_model_fast(X_train, y_train)
    
    # Evaluate model
    y_pred, y_prob, accuracy, auc_score, f1, balanced_acc = evaluate_model_fast(X_test, y_test)
    
    # Plot the 4 figures
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    print("\nFigure 1: Confusion Matrix")
    plot_figure1_confusion_matrix(y_test, y_pred)
    
    print("\nFigure 2: ROC Curve and Precision-Recall Curve")
    plot_figure2_curves(y_test, y_prob, auc_score)
    
    print("\nFigure 3: Feature Importance")
    plot_figure3_feature_importance(12)  # Reduced for clarity
    
    print("\nFigure 4: Prediction Probability Distribution")
    plot_figure4_prediction_probability(y_test, y_prob)
    
    total_time = time.time() - total_start_time
    
    # Print final summary
    print("=" * 60)
    print(f"🎯 FINAL RESULTS FOR ToN-IoT")
    print(f"📊 Total execution time: {total_time:.2f} seconds")
    print(f"📈 Accuracy: {accuracy*100:.2f}%")
    print(f"📊 AUC Score: {auc_score:.4f}")
    print(f"🎯 F1 Score: {f1:.4f}")
    print(f"⚖️ Balanced Accuracy: {balanced_acc:.4f}")
    if model.oob_score_:
        print(f"🔄 OOB Score: {model.oob_score_:.4f}")
    print("=" * 60)

# Alternative: Ultra-fast version for quick testing
def ultra_fast_demo():
    """Even faster version for testing."""
    print("=" * 60)
    print("ULTRA-FAST ToN-IoT DEMO")
    print("=" * 60)
    
    # Create very small synthetic dataset
    df = create_synthetic_ton_iot(5000)
    
    # Very simple preprocessing
    X = df.select_dtypes(include=[np.number])
    if 'label' in df.columns:
        y = df['label']
    else:
        y = df.iloc[:, -1]  # Use last column as target
    
    # Keep only top 20 features by variance
    variances = X.var().sort_values(ascending=False)
    keep_cols = variances.head(20).index.tolist()
    X = X[keep_cols]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Very simple model
    model = RandomForestClassifier(
        n_estimators=30,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    print(f"Training on {X_train.shape[0]} samples, {X_train.shape[1]} features...")
    model.fit(X_train, y_train)
    
    # Quick predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    
    # Quick plot
    plt.figure(figsize=(10, 8))
    
    # Confusion Matrix
    plt.subplot(2, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.title(f"Confusion Matrix\nAccuracy: {accuracy:.3f}")
    
    # ROC Curve
    plt.subplot(2, 2, 2)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.legend()
    
    # Feature Importance (top 10)
    plt.subplot(2, 2, 3)
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    plt.barh(range(len(importance)), importance['importance'])
    plt.yticks(range(len(importance)), importance['feature'])
    plt.gca().invert_yaxis()
    plt.title("Top 10 Features")
    
    # Probability Distribution
    plt.subplot(2, 2, 4)
    normal_probs = y_prob[y_test == 0]
    attack_probs = y_prob[y_test == 1]
    plt.hist(normal_probs, alpha=0.5, label='Normal', density=True)
    plt.hist(attack_probs, alpha=0.5, label='Attack', density=True)
    plt.axvline(x=0.5, color='k', linestyle='--')
    plt.title("Probability Distribution")
    plt.legend()
    
    plt.tight_layout()
    _savefig('randomforesttoniot')
    
    print(f"\nResults: Accuracy = {accuracy:.3f}, AUC = {auc_score:.3f}")

if __name__ == "__main__":
    # Uncomment for ultra-fast demo
    # ultra_fast_demo()
    
    # Use optimized main function
    main()
