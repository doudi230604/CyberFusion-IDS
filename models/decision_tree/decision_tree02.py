#!/usr/bin/env python3
# =============================================================================
# UNSW-NB15 DATASET ANALYSIS WITH DECISION TREES
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
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

import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("UNSW-NB15 CYBERSECURITY DATASET ANALYSIS")
print("=" * 70)

# Dataset directory
DATASET_DIR = "/home/darine/cybersecurity_assignment/datasets/unsw_nb15"

# Available dataset files
DATASET_FILES = {
    "1": "UNSW-NB15_1.csv",
    "2": "UNSW-NB15_2.csv", 
    "3": "UNSW-NB15_3.csv",
    "4": "UNSW-NB15_4.csv",
    "training": "UNSW_NB15_training-set.csv",
    "testing": "UNSW_NB15_testing-set.csv",
    "features": "NUSW-NB15_features.csv",
    "events": "UNSW-NB15_LIST_EVENTS.csv"
}

def create_synthetic_dataset(n_samples=20000):
    """Create a small synthetic UNSW-like dataset for demos."""
    print("Creating synthetic dataset for demo...")
    np.random.seed(42)
    data = {
        'dur': np.random.exponential(1.0, n_samples),
        'spkts': np.random.poisson(10, n_samples),
        'dpkts': np.random.poisson(10, n_samples),
        'sbytes': np.random.exponential(1000, n_samples),
        'dbytes': np.random.exponential(1000, n_samples),
        'sttl': np.random.randint(32, 255, n_samples),
        'dttl': np.random.randint(32, 255, n_samples),
        'label': np.random.choice([0,1], n_samples, p=[0.8,0.2])
    }
    df = pd.DataFrame(data)
    print(f"Synthetic dataset created with shape: {df.shape}")
    return df, 'synthetic'


def load_dataset(file_key):
    """Load a specific dataset file"""
    filename = DATASET_FILES.get(file_key)
    if not filename:
        print(f"❌ Invalid file key: {file_key}")
        return None
    
    filepath = os.path.join(DATASET_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        # Fallback to synthetic dataset for non-interactive runs
        return create_synthetic_dataset()
    
    print(f"\n📂 Loading {filename}...")
    
    try:
        # For large files, we'll load a sample first to check structure
        if filename.endswith("features.csv") or filename.endswith("events.csv"):
            # Small files, load completely
            df = pd.read_csv(filepath)
        else:
            # Large files, load sample first
            print("   This is a large file. Loading first 50,000 rows for analysis...")
            df = pd.read_csv(filepath, nrows=50000)
        
        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        return df, filename
        
    except Exception as e:
        print(f"❌ Error loading file: {str(e)}")
        return None, None

def analyze_dataset_structure(df, filename):
    """Analyze the structure of the dataset"""
    print(f"\n📊 DATASET STRUCTURE: {filename}")
    print("-" * 50)
    
    # Basic info
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage().sum() / 1024**2:.2f} MB")
    
    # Column information
    print(f"\n📝 COLUMNS ({len(df.columns)} total):")
    for i, col in enumerate(df.columns[:15], 1):  # Show first 15 columns
        dtype = df[col].dtype
        unique = df[col].nunique()
        print(f"  {i:2d}. {col:30s} {str(dtype):10s} {unique:6d} unique values")
    
    if len(df.columns) > 15:
        print(f"  ... and {len(df.columns) - 15} more columns")
    
    # Look for label/attack columns
    print(f"\n🎯 LOOKING FOR LABEL COLUMNS:")
    label_candidates = []
    attack_candidates = []
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Check for label indicators
        if any(keyword in col_lower for keyword in ['label', 'class', 'attack', 'result', 'category']):
            unique_vals = df[col].nunique()
            
            if 'label' in col_lower or 'class' in col_lower:
                label_candidates.append((col, unique_vals))
                print(f"  ✅ Potential label: '{col}' ({unique_vals} unique values)")
                
                if unique_vals < 10:
                    print(f"     Values: {df[col].unique()}")
            
            if 'attack' in col_lower:
                attack_candidates.append((col, unique_vals))
                print(f"  ⚠️  Attack-related: '{col}' ({unique_vals} unique values)")
    
    if not label_candidates and not attack_candidates:
        print("  ⚠️  No obvious label columns found")
        # Show all column names
        print(f"\n  All columns: {df.columns.tolist()}")
    
    # Sample data
    print(f"\n📋 SAMPLE DATA (first 3 rows):")
    print(df.head(3))
    
    # Data types summary
    print(f"\n🔧 DATA TYPES SUMMARY:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    return label_candidates, attack_candidates

def prepare_features_and_labels(df, label_column):
    """Prepare features and labels for machine learning"""
    print(f"\n🔧 PREPARING DATA with label column: '{label_column}'")
    
    # Create labels
    if df[label_column].dtype == 'object':
        # For string labels, check if it's binary
        unique_vals = df[label_column].unique()
        print(f"  Label values: {unique_vals}")
        
        # Try to convert to binary (Normal vs Attack)
        if 'Normal' in unique_vals or 'normal' in [str(v).lower() for v in unique_vals]:
            # Create binary labels: 0=Normal, 1=Attack
            y = (df[label_column] != 'Normal').astype(int)
            print(f"  ✅ Converted to binary: Normal=0, Attack=1")
        else:
            # Use LabelEncoder for other categorical labels
            le = LabelEncoder()
            y = le.fit_transform(df[label_column])
            print(f"  ✅ Encoded {len(unique_vals)} classes")
    else:
        # Already numeric
        y = df[label_column].values
        print(f"  ✅ Using numeric labels ({len(np.unique(y))} unique values)")
    
    # Select features (exclude label column)
    feature_cols = [col for col in df.columns if col != label_column]
    
    # Select only numeric features for decision tree
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\n  📊 Initial feature selection:")
    print(f"    Total features: {len(feature_cols)}")
    print(f"    Numeric features: {len(numeric_features)}")
    
    # If too many features, select top ones
    if len(numeric_features) > 30:
        print(f"  🔍 Selecting top 30 numeric features...")
        
        # Calculate correlation with target
        try:
            corrs = []
            for col in numeric_features:
                if len(df[col].unique()) > 1:
                    corr = abs(np.corrcoef(df[col], y, rowvar=False)[0, 1])
                    corrs.append((col, corr))
            
            corrs.sort(key=lambda x: x[1], reverse=True)
            selected_features = [col for col, _ in corrs[:30]]
            print(f"  ✅ Selected {len(selected_features)} features based on correlation")
        except:
            # Fallback: use variance
            variances = df[numeric_features].var().sort_values(ascending=False)
            selected_features = variances.head(30).index.tolist()
            print(f"  ✅ Selected {len(selected_features)} features based on variance")
    else:
        selected_features = numeric_features
    
    X = df[selected_features]
    
    # Handle missing values
    if X.isnull().any().any():
        missing_count = X.isnull().sum().sum()
        print(f"  🧹 Handling {missing_count} missing values...")
        X = X.fillna(X.median())
    
    print(f"\n  ✅ Final dataset:")
    print(f"    X shape: {X.shape}")
    print(f"    y shape: {y.shape}")
    print(f"    Classes: {np.unique(y)}")
    print(f"    Class distribution: {np.bincount(y) if len(y) > 0 else 'N/A'}")
    
    return X, y, selected_features

def train_decision_tree(X, y, features, filename, label_name):
    """Train and evaluate a decision tree"""
    print(f"\n" + "="*60)
    print(f"🤖 DECISION TREE TRAINING")
    print(f"Dataset: {filename}")
    print(f"Label: {label_name}")
    print("="*60)
    
    # Check if we have enough data and classes
    if len(np.unique(y)) < 2:
        print("❌ Need at least 2 classes for classification")
        return None, 0
    
    if len(X) < 100:
        print("❌ Need at least 100 samples for training")
        return None, 0
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data Split:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Class distribution in training: {np.bincount(y_train)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Decision Tree
    print(f"\n⚡ Training Decision Tree...")
    
    # Determine max depth based on data size
    max_depth = min(5, int(np.log2(len(X_train))) - 1)
    
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        class_weight='balanced'
    )
    
    dt.fit(X_train_scaled, y_train)
    
    # Predict and evaluate
    y_pred = dt.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ MODEL PERFORMANCE:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Training accuracy: {dt.score(X_train_scaled, y_train):.4f}")
    
    print(f"\n📋 CLASSIFICATION REPORT:")
    # Create class names
    if len(np.unique(y)) == 2:
        class_names = ['Normal', 'Attack']
    else:
        class_names = [f'Class_{i}' for i in range(len(np.unique(y)))]
    
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Feature importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': dt.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n🏆 TOP 10 MOST IMPORTANT FEATURES:")
    print(importance.head(10).to_string(index=False))
    
    return dt, accuracy, X_test_scaled, y_test, y_pred, importance

def visualize_results(dt, X_test, y_test, y_pred, importance, features, filename, accuracy):
    """Create visualizations of the results"""
    print(f"\n" + "="*60)
    print(f"📊 VISUALIZATIONS")
    print("="*60)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Feature Importance (top left)
    ax1 = plt.subplot(2, 2, 1)
    top_features = importance.head(15)
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(top_features)))
    
    bars = ax1.barh(range(len(top_features)), top_features['Importance'], color=colors)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['Feature'])
    ax1.set_xlabel('Importance Score')
    ax1.set_title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, imp) in enumerate(zip(bars, top_features['Importance'])):
        ax1.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2, 
                f'{imp:.3f}', va='center', fontsize=9)
    
    # 2. Confusion Matrix (top right)
    ax2 = plt.subplot(2, 2, 2)
    cm = confusion_matrix(y_test, y_pred)
    
    # For binary classification, use Normal/Attack labels
    if len(np.unique(y_test)) == 2:
        labels = ['Normal', 'Attack']
    else:
        labels = [f'C{i}' for i in range(len(np.unique(y_test)))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax2)
    ax2.set_title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}', 
                 fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    # 3. Decision Tree Visualization (bottom left)
    ax3 = plt.subplot(2, 2, 3)
    if len(np.unique(y_test)) == 2:
        class_names = ['Normal', 'Attack']
    else:
        class_names = [f'Class_{i}' for i in range(len(np.unique(y_test)))]
    
    plot_tree(dt, 
              feature_names=features,
              class_names=class_names,
              filled=True,
              rounded=True,
              max_depth=3,
              fontsize=8,
              ax=ax3,
              proportion=True)
    ax3.set_title('Decision Tree (First 3 Levels)', fontsize=14, fontweight='bold')
    
    # 4. Accuracy and Class Distribution (bottom right)
    ax4 = plt.subplot(2, 2, 4)
    
    # Create two subplots within this subplot
    ax4_1 = plt.axes([0.65, 0.55, 0.3, 0.35])
    ax4_2 = plt.axes([0.65, 0.15, 0.3, 0.35])
    
    # Accuracy bar
    ax4_1.bar(['Accuracy'], [accuracy], color='green', alpha=0.7, width=0.6)
    ax4_1.set_ylim(0, 1.0)
    ax4_1.set_ylabel('Score')
    ax4_1.set_title(f'Model Accuracy\n{accuracy:.4f}', fontweight='bold')
    ax4_1.grid(True, alpha=0.3)
    
    # Add value on bar
    ax4_1.text(0, accuracy + 0.03, f'{accuracy:.4f}', 
              ha='center', fontweight='bold', fontsize=12)
    
    # Class distribution
    unique, counts = np.unique(y_test, return_counts=True)
    colors_bar = ['blue', 'red', 'green', 'orange', 'purple'][:len(unique)]
    
    if len(unique) == 2:
        bar_labels = ['Normal', 'Attack']
    else:
        bar_labels = [f'Class {i}' for i in unique]
    
    bars = ax4_2.bar(bar_labels, counts, color=colors_bar, alpha=0.7)
    ax4_2.set_title('Class Distribution in Test Set', fontweight='bold')
    ax4_2.set_ylabel('Count')
    ax4_2.tick_params(axis='x', rotation=45)
    
    # Add counts on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax4_2.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                  f'{count}', ha='center', va='bottom')
    
    # Hide the main subplot 4 frame
    ax4.axis('off')
    
    # Main title
    plt.suptitle(f'UNSW-NB15 Analysis: {filename}\nDecision Tree Results', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    _savefig('decision_tree02')
    
    # Save the figure
    try:
        output_file = f'unsw_nb15_results_{filename.replace(".csv", "")}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"💾 Visualization saved as: {output_file}")
    except:
        print("⚠️  Could not save visualization file")

def main(choice=None, label_column=None):
    """Main function to run the analysis (non-interactive by default).

    Parameters:
    - choice: optional key for dataset (e.g., 'training' or '1')
    - label_column: optional column name to use as label
    """
    print("\n📁 AVAILABLE DATASET FILES:")
    for key, filename in DATASET_FILES.items():
        filepath = os.path.join(DATASET_DIR, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024*1024)  # MB
            print(f"  [{key}] {filename} ({size:.1f} MB)")
    
    print("\n💡 RECOMMENDATION: Start with 'training' or '1' (smaller files)")

    # Non-interactive dataset choice
    try:
        if choice is None:
            if 'training' in DATASET_FILES and os.path.exists(os.path.join(DATASET_DIR, DATASET_FILES['training'])):
                choice = 'training'
            else:
                choice = next((k for k,fn in DATASET_FILES.items() if os.path.exists(os.path.join(DATASET_DIR, fn))), 'training')
        else:
            choice = str(choice).strip().lower()
    except Exception:
        choice = 'training'

    # Load the dataset
    df, filename = load_dataset(choice)
    
    if df is None:
        print("❌ Failed to load dataset")
        return
    
    # Analyze structure
    label_candidates, attack_candidates = analyze_dataset_structure(df, filename)
    
    # Choose label column (automated)
    if label_column is None:
        if label_candidates:
            cols = [c for c, _ in label_candidates]
            if 'label' in cols:
                label_column = 'label'
            else:
                label_column = cols[0]
        else:
            label_column = 'label' if 'label' in df.columns else df.columns[-1]
    else:
        if label_column not in df.columns:
            print(f"⚠️ Provided label_column '{label_column}' not found. Using default column.")
            label_column = 'label' if 'label' in df.columns else df.columns[-1]
    
    print(f"\n✅ Selected label column: '{label_column}'")
    
    # Prepare features and labels
    X, y, features = prepare_features_and_labels(df, label_column)
    
    if X is None or len(X) == 0:
        print("❌ Failed to prepare features")
        return
    
    # Train decision tree
    dt, accuracy, X_test, y_test, y_pred, importance = train_decision_tree(
        X, y, features, filename, label_column
    )
    
    if dt is None:
        print("❌ Decision tree training failed")
        return
    
    # Visualize results
    visualize_results(dt, X_test, y_test, y_pred, importance, features, filename, accuracy)
    
    # Save results to CSV
    try:
        results_df = importance.copy()
        results_df['Dataset'] = filename
        results_df['Label_Column'] = label_column
        results_df['Accuracy'] = accuracy
        
        output_csv = f'unsw_nb15_analysis_{filename.replace(".csv", "")}.csv'
        results_df.to_csv(output_csv, index=False)
        print(f"\n💾 Detailed results saved to: {output_csv}")
    except Exception as e:
        print(f"⚠️  Could not save CSV results: {str(e)}")
    
    print(f"\n" + "="*70)
    print("🎉 ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📋 SUMMARY:")
    print(f"  Dataset: {filename}")
    print(f"  Samples analyzed: {len(df)}")
    print(f"  Label column: {label_column}")
    print(f"  Features used: {len(features)}")
    print(f"  Model accuracy: {accuracy:.4f}")
    print(f"  Decision tree depth: {dt.get_depth()}")
    print(f"\n✅ All decision tree models trained and tested successfully!")

# =============================================================================
# RUN THE ANALYSIS
# =============================================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()