import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# GLOBAL variables
label_encoders = {}
scaler = RobustScaler()
imputer = SimpleImputer(strategy='median')
model = None
feature_importance_df = None

def find_unsw_nb15_files():
    """Find UNSW-NB15 files in the current directory and subdirectories."""
    print("🔍 Searching for UNSW-NB15 files...")
    
    # File patterns to look for
    file_patterns = [
        "UNSW_NB15_training-set.csv",
        "UNSW_NB15_testing-set.csv",
        "*training-set.csv",
        "*testing-set.csv",
        "UNSW-NB15_*.csv"
    ]
    
    found_files = {}
    
    # Search current directory and subdirectories
    for root, dirs, files in os.walk("."):
        for file in files:
            file_lower = file.lower()
            if ("unsw" in file_lower and "nb15" in file_lower) or \
               ("training-set" in file_lower) or ("testing-set" in file_lower):
                full_path = Path(root) / file
                
                # Categorize files
                if "training" in file_lower or "train" in file_lower:
                    found_files["training"] = full_path
                    print(f"  ✅ Found training file: {full_path}")
                elif "testing" in file_lower or "test" in file_lower:
                    found_files["testing"] = full_path
                    print(f"  ✅ Found testing file: {full_path}")
                else:
                    print(f"  Found: {full_path}")
    
    return found_files

def load_unsw_nb15():
    """Load UNSW-NB15 dataset with flexible file location handling."""
    print("=" * 70)
    print("LOADING UNSW-NB15 DATASET")
    print("=" * 70)
    
    # Find files
    found_files = find_unsw_nb15_files()
    
    # Check if we found the required files
    training_file = None
    testing_file = None
    
    # Look for training file
    if "training" in found_files:
        training_file = found_files["training"]
    else:
        # Try common variations
        possible_train_names = [
            "UNSW_NB15_training-set.csv",
            "UNSW-NB15_training-set.csv",
            "training-set.csv",
            "UNSW_NB15_train.csv"
        ]
        
        for name in possible_train_names:
            path = Path(name)
            if path.exists():
                training_file = path
                print(f"  ✅ Found training file: {training_file}")
                break
    
    # Look for testing file
    if "testing" in found_files:
        testing_file = found_files["testing"]
    else:
        # Try common variations
        possible_test_names = [
            "UNSW_NB15_testing-set.csv",
            "UNSW-NB15_testing-set.csv",
            "testing-set.csv",
            "UNSW_NB15_test.csv"
        ]
        
        for name in possible_test_names:
            path = Path(name)
            if path.exists():
                testing_file = path
                print(f"  ✅ Found testing file: {testing_file}")
                break
    
    # Check if files were found
    if training_file is None:
        print("\n❌ Could not find training file!")
        print("Looking in:")
        print(f"  Current directory: {Path('.').absolute()}")
        print("\nPlease ensure UNSW_NB15_training-set.csv is in one of these locations:")
        print("1. Current directory")
        print("2. A subdirectory")
        print("3. Named as one of: UNSW_NB15_training-set.csv, training-set.csv, etc.")
        return None
    
    if testing_file is None:
        print("\n❌ Could not find testing file!")
        print("Will use training file only...")
        testing_file = training_file  # Use training as fallback
    
    # Load the files
    data_frames = []
    
    try:
        print(f"\n📂 Loading {training_file.name}...")
        train_df = pd.read_csv(training_file, encoding='utf-8', low_memory=False)
        print(f"  ✅ Loaded: {train_df.shape[0]:,} rows, {train_df.shape[1]} columns")
        print(f"  Columns: {list(train_df.columns[:5])}...")
        data_frames.append(train_df)
    except Exception as e:
        print(f"  ❌ Failed to load {training_file}: {e}")
        return None
    
    if testing_file != training_file:
        try:
            print(f"\n📂 Loading {testing_file.name}...")
            test_df = pd.read_csv(testing_file, encoding='utf-8', low_memory=False)
            print(f"  ✅ Loaded: {test_df.shape[0]:,} rows, {test_df.shape[1]} columns")
            print(f"  Columns: {list(test_df.columns[:5])}...")
            data_frames.append(test_df)
        except Exception as e:
            print(f"  ⚠️  Failed to load {testing_file}: {e}")
            print("  Continuing with training data only...")
    
    # Combine datasets
    print(f"\n🔗 Combining datasets...")
    combined_df = pd.concat(data_frames, ignore_index=True)
    print(f"✅ Combined dataset: {combined_df.shape[0]:,} rows, {combined_df.shape[1]} columns")
    
    return combined_df

def preprocess_unsw_nb15(df):
    """Preprocess the UNSW-NB15 dataset."""
    print("\n" + "="*60)
    print("PREPROCESSING UNSW-NB15 DATASET")
    print("="*60)
    
    df_processed = df.copy()
    
    # Display initial information
    print(f"Original shape: {df_processed.shape}")
    print(f"Sample columns: {list(df_processed.columns[:10])}...")
    
    # Remove duplicate rows
    initial_rows = len(df_processed)
    df_processed = df_processed.drop_duplicates()
    duplicates_removed = initial_rows - len(df_processed)
    if duplicates_removed > 0:
        print(f"\nRemoved {duplicates_removed:,} duplicate rows ({duplicates_removed/initial_rows*100:.2f}%)")
    else:
        print(f"\nNo duplicate rows found")
    
    # Identify the target column
    target_col = None
    for col in ['label', 'Label', 'attack_cat', 'Attack_cat', 'attack', 'Attack']:
        if col in df_processed.columns:
            target_col = col
            print(f"✅ Target column identified: {target_col}")
            break
    
    if target_col is None:
        print("❌ Error: Could not find target column in dataset!")
        print(f"Available columns: {list(df_processed.columns)}")
        return None, None
    
    # Check for missing values
    missing_values = df_processed.isnull().sum().sum()
    if missing_values > 0:
        print(f"\nMissing values found: {missing_values:,}")
        print("Filling missing values...")
    
    # Fill missing values
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            df_processed[col] = df_processed[col].fillna('Unknown')
        else:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # Encode categorical variables (excluding target)
    categorical_cols = []
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object' and col != target_col:
            categorical_cols.append(col)
    
    print(f"\nEncoding {len(categorical_cols)} categorical columns...")
    for col in categorical_cols:
        try:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
        except Exception as e:
            print(f"  ⚠️  Could not encode {col}: {e}")
            df_processed = df_processed.drop(columns=[col])
    
    # Prepare features and target
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    
    # Handle multi-class target if needed
    if target_col in ['attack_cat', 'Attack_cat']:
        print(f"\nMulti-class target detected: {len(y.unique())} attack categories")
        print("Converting to binary: Normal vs All Attacks")
        
        # Create binary target (0 = normal, 1 = any attack)
        y_binary = (y != 'Normal').astype(int)
        y = y_binary
        print(f"Binary conversion complete: {sum(y==0):,} normal, {sum(y==1):,} attacks")
    elif len(y.unique()) > 2:
        print(f"\nMulti-class numeric target detected ({len(y.unique())} classes)")
        print("Converting to binary: 0 = Normal, 1 = Attack")
        y = (y > 0).astype(int)
    
    # Sample to exactly 25,000 rows
    print(f"\nOriginal dataset has {len(X):,} rows")
    if len(X) > 25000:
        print(f"Sampling to exactly 25,000 rows...")
        indices = np.random.RandomState(42).choice(len(X), 25000, replace=False)
        X = X.iloc[indices].reset_index(drop=True)
        y = y.iloc[indices].reset_index(drop=True)
    elif len(X) < 25000:
        print(f"⚠️  Warning: Only {len(X):,} rows available (less than target 25,000)")
        print(f"Using all available data...")
    
    print(f"✅ Final dataset: {len(X):,} rows, {X.shape[1]} features")
    
    return X, y

def analyze_class_distribution(y):
    """Print detailed class distribution."""
    unique, counts = np.unique(y, return_counts=True)
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    for cls, count in zip(unique, counts):
        label = "Normal" if cls == 0 else "Attack"
        percentage = count / len(y) * 100
        print(f"{label} (Class {cls}): {count:,} samples ({percentage:.2f}%)")
    
    # Check for class imbalance
    imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
    print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 3:
        print("⚠️  Class imbalance detected - using balanced class weights.")
    
    return imbalance_ratio

def create_optimized_random_forest():
    """Create optimized Random Forest model."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced_subsample',
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
        verbose=0,
        max_samples=0.8,
        min_impurity_decrease=0.0001,
        ccp_alpha=0.001
    )

def train_model(X_train, y_train):
    global model, scaler, imputer, feature_importance_df
    
    print("\n" + "="*60)
    print("TRAINING OPTIMIZED RANDOM FOREST")
    print("="*60)
    print(f"Training samples: {X_train.shape[0]:,}")
    print(f"Features: {X_train.shape[1]}")
    
    # Handle missing values
    X_train_imputed = imputer.fit_transform(X_train)
    
    # Scale features
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    
    # Create and train model
    model = create_optimized_random_forest()
    
    print("\nTraining model...")
    model.fit(X_train_scaled, y_train)
    
    # Feature importance
    feature_importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    
    print(f"\n✅ Training complete!")
    print(f"📊 OOB Score: {model.oob_score_:.4f}")
    print(f"📈 Cross-validation AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    return model

def evaluate_model(X_test, y_test):
    global model, scaler, imputer
    
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Preprocess test data
    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"Test samples: {X_test.shape[0]:,}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"OOB Score: {model.oob_score_:.4f}\n")
    
    # Classification report
    print("=== DETAILED CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"], digits=4))
    
    return accuracy, auc_score, y_pred, y_prob

def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix as Figure 1."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Normal", "Attack"],
                yticklabels=["Normal", "Attack"],
                ax=ax, annot_kws={"size": 14})
    
    ax.set_title("Figure 1: Confusion Matrix", fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    
    plt.tight_layout()
    plt.show()

def plot_roc_pr_curves(y_test, y_prob):
    """Plot ROC and Precision-Recall curves as Figure 2."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    
    axes[0].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', linewidth=2.5, color='darkblue')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5)
    axes[0].set_xlabel("False Positive Rate", fontsize=12)
    axes[0].set_ylabel("True Positive Rate", fontsize=12)
    axes[0].set_title("ROC Curve", fontsize=14, fontweight='bold', pad=15)
    axes[0].legend(loc="lower right", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].fill_between(fpr, tpr, alpha=0.1, color='blue')
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    axes[1].plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})', linewidth=2.5, color='darkred')
    axes[1].set_xlabel("Recall", fontsize=12)
    axes[1].set_ylabel("Precision", fontsize=12)
    axes[1].set_title("Precision-Recall Curve", fontsize=14, fontweight='bold', pad=15)
    axes[1].legend(loc="lower left", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(recall, precision, alpha=0.1, color='red')
    
    # Add baseline for PR curve
    pos_ratio = np.mean(y_test)
    axes[1].axhline(y=pos_ratio, color='k', linestyle='--', alpha=0.5, 
                    label=f'Baseline (pos ratio = {pos_ratio:.3f})')
    axes[1].legend(loc="lower left", fontsize=11)
    
    fig.suptitle("Figure 2: ROC Curve and Precision-Recall Curve", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(top_n=20):
    """Plot feature importance as Figure 3."""
    global feature_importance_df
    
    if feature_importance_df is None:
        print("❌ Model not trained yet!")
        return None
    
    print(f"\n" + "="*60)
    print(f"TOP {top_n} FEATURE IMPORTANCES")
    print("="*60)
    
    top_features = feature_importance_df.head(top_n)
    
    # Print top features
    for i, (idx, row) in enumerate(top_features.iterrows(), 1):
        print(f"{i:2d}. {row['feature']:30s}: {row['importance']:.6f}")
    
    # Plot
    plt.figure(figsize=(14, 10))
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(top_features)))
    bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors, edgecolor='black')
    
    plt.yticks(range(len(top_features)), top_features['feature'], fontsize=11)
    plt.xlabel("Feature Importance Score", fontsize=12)
    plt.title("Figure 3: Top 20 Feature Importances", fontsize=16, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{importance:.4f}', va='center', fontsize=10)
    
    # Add cumulative importance line
    cumulative_importance = np.cumsum(top_features['importance'])
    ax2 = plt.gca().twinx()
    ax2.plot(cumulative_importance, range(len(top_features)), 
             color='red', marker='o', linestyle='--', linewidth=2, markersize=4)
    ax2.set_ylabel('Cumulative Importance', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(plt.gca().get_ylim())
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    # Calculate and display cumulative importance
    total_importance = feature_importance_df['importance'].sum()
    top_n_importance = top_features['importance'].sum()
    print(f"\n📊 Cumulative importance of top {top_n} features: {top_n_importance:.3f} ({top_n_importance/total_importance*100:.1f}% of total)")
    
    return top_features

def plot_probability_distribution(y_test, y_prob):
    """Plot prediction probability distribution as Figure 4."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Separate probabilities by true class
    normal_probs = y_prob[y_test == 0]
    attack_probs = y_prob[y_test == 1]
    
    # Histogram
    axes[0].hist([normal_probs, attack_probs], 
                 bins=30, alpha=0.7, 
                 label=['Normal (Actual)', 'Attack (Actual)'],
                 color=['green', 'red'],
                 edgecolor='black',
                 stacked=True)
    axes[0].set_xlabel("Predicted Probability", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("Probability Distribution by True Class", fontsize=14, fontweight='bold', pad=15)
    axes[0].legend(loc='upper center', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Density plot
    from scipy import stats
    
    for probs, label, color in zip([normal_probs, attack_probs], 
                                   ['Normal', 'Attack'], 
                                   ['green', 'red']):
        if len(probs) > 1:
            kde = stats.gaussian_kde(probs)
            x_vals = np.linspace(0, 1, 100)
            axes[1].plot(x_vals, kde(x_vals), label=label, color=color, linewidth=2.5)
            axes[1].fill_between(x_vals, kde(x_vals), alpha=0.2, color=color)
    
    axes[1].axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Decision Boundary (0.5)')
    axes[1].set_xlabel("Predicted Probability", fontsize=12)
    axes[1].set_ylabel("Density", fontsize=12)
    axes[1].set_title("Probability Density by Class", fontsize=14, fontweight='bold', pad=15)
    axes[1].legend(loc='upper center', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = (f"Normal class:\n"
                  f"  Mean: {np.mean(normal_probs):.3f}\n"
                  f"  Std: {np.std(normal_probs):.3f}\n\n"
                  f"Attack class:\n"
                  f"  Mean: {np.mean(attack_probs):.3f}\n"
                  f"  Std: {np.std(attack_probs):.3f}")
    
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.suptitle("Figure 4: Prediction Probability Distribution", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def main():
    print("=" * 70)
    print("FLEXIBLE UNSW-NB15 ANALYSIS WITH RANDOM FOREST")
    print("=" * 70)
    print("Target: 25,000 samples | Top 20 Features | Optimized Model")
    print("=" * 70)
    
    # Show current directory
    current_dir = Path('.').absolute()
    print(f"Current directory: {current_dir}")
    print("Searching for UNSW-NB15 files...\n")
    
    # Load dataset
    df = load_unsw_nb15()
    
    if df is None:
        print("\n" + "="*70)
        print("ALTERNATIVE APPROACH")
        print("="*70)
        
        # Try to find the dataset in your specific directory structure
        possible_paths = [
            Path("./datasets/unsw_nb15/UNSW_NB15_training-set.csv"),
            Path("./datasets/UNSW_NB15_training-set.csv"),
            Path("./unsw_nb15/UNSW_NB15_training-set.csv"),
            Path("../datasets/unsw_nb15/UNSW_NB15_training-set.csv")
        ]
        
        for path in possible_paths:
            if path.exists():
                print(f"\nFound dataset at: {path}")
                try:
                    df = pd.read_csv(path, encoding='utf-8', low_memory=False)
                    print(f"Successfully loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
                    
                    # Try to load testing set from same location
                    test_path = path.parent / "UNSW_NB15_testing-set.csv"
                    if test_path.exists():
                        test_df = pd.read_csv(test_path, encoding='utf-8', low_memory=False)
                        print(f"Also found testing set: {test_df.shape[0]:,} rows")
                        df = pd.concat([df, test_df], ignore_index=True)
                        print(f"Combined dataset: {df.shape[0]:,} rows")
                    break
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
        
        if df is None:
            print("\n❌ Could not find UNSW-NB15 dataset!")
            print("\nPlease either:")
            print("1. Place the CSV files in the current directory")
            print("2. Update the file paths in the code")
            print("3. Or specify the full path to your dataset")
            return
    
    # Preprocess data
    X, y = preprocess_unsw_nb15(df)
    
    if X is None or y is None:
        print("\n❌ Preprocessing failed!")
        return
    
    # Analyze class distribution
    analyze_class_distribution(y)
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\n" + "="*60)
    print("DATA SPLIT")
    print("="*60)
    print(f"Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # Train model
    train_model(X_train, y_train)
    
    # Evaluate model
    accuracy, auc_score, y_pred, y_prob = evaluate_model(X_test, y_test)
    
    # Generate all required plots
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Figure 1: Confusion Matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # Figure 2: ROC and PR Curves
    plot_roc_pr_curves(y_test, y_prob)
    
    # Figure 3: Feature Importance
    top_features = plot_feature_importance(20)
    
    # Figure 4: Probability Distribution
    plot_probability_distribution(y_test, y_prob)
    
    # Final summary
    print("\n" + "="*70)
    print("🎯 FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"📊 Dataset Size: {len(X):,} samples")
    print(f"🎯 Accuracy: {accuracy*100:.2f}%")
    print(f"📈 AUC Score: {auc_score:.4f}")
    print(f"🎯 Precision: {precision_score(y_test, y_pred, zero_division=0)*100:.2f}%")
    print(f"🎯 Recall: {recall_score(y_test, y_pred, zero_division=0)*100:.2f}%")
    print(f"🎯 F1-Score: {f1_score(y_test, y_pred, zero_division=0)*100:.2f}%")
    print(f"📋 OOB Score: {model.oob_score_:.4f}")
    
    # Display top 5 features
    if top_features is not None:
        print(f"\n🏆 TOP 5 MOST IMPORTANT FEATURES:")
        for i in range(min(5, len(top_features))):
            feat = top_features.iloc[i]
            print(f"  {i+1}. {feat['feature']}: {feat['importance']:.6f}")
    
    # Save results to file
    try:
        if top_features is not None:
            top_features.to_csv('top_20_features.csv', index=False)
            print(f"\n💾 Top 20 features saved to 'top_20_features.csv'")
        
        results_df = pd.DataFrame({
            'Metric': ['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1-Score', 'OOB Score'],
            'Value': [accuracy, auc_score, precision_score(y_test, y_pred, zero_division=0), 
                     recall_score(y_test, y_pred, zero_division=0), f1_score(y_test, y_pred, zero_division=0), 
                     model.oob_score_]
        })
        results_df.to_csv('model_results.csv', index=False)
        print(f"💾 Model results saved to 'model_results.csv'")
        
    except Exception as e:
        print(f"\n⚠️ Could not save results: {e}")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()