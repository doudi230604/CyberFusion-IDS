# =============================================================================
# ENHANCED ISOLATION FOREST FOR UNSW-NB15 DATASET
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, auc, precision_recall_curve, 
                           average_precision_score)
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
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ENHANCED ISOLATION FOREST - Train on Normal Data Only")
print("="*70)

# ===== 1. LOAD REAL UNSW-NB15 DATA =====
print("\n📊 Loading Real UNSW-NB15 Dataset")
dataset_path = "/home/darine/cybersecurity_assignment/datasets/unsw_nb15"

def find_unsw_nb15_file():
    """Find and load the UNSW-NB15 dataset file"""
    
    print(f"Looking for UNSW-NB15 dataset in: {dataset_path}")
    
    # Check if directory exists
    if not os.path.exists(dataset_path):
        print(f"❌ Directory not found: {dataset_path}")
        return None
    
    print(f"✅ Found UNSW-NB15 directory")
    
    # List all files
    all_items = os.listdir(dataset_path)
    
    if not all_items:
        print("❌ Directory is empty!")
        return None
    
    # Look for CSV files
    csv_files = []
    
    for item in all_items:
        item_path = os.path.join(dataset_path, item)
        if os.path.isfile(item_path):
            if item.lower().endswith('.csv'):
                size = os.path.getsize(item_path) / 1024  # KB
                csv_files.append((item, size, item_path))
    
    # Show files
    if csv_files:
        print(f"\n📄 Found {len(csv_files)} CSV files:")
        for filename, size, path in csv_files:
            print(f"   📄 {filename} ({size/1024:.1f} MB)")
    
    # Common UNSW-NB15 file names (prefer training set)
    preferred_files = [
        'UNSW_NB15_training-set.csv',
        'UNSW_NB15_testing-set.csv', 
        'UNSW-NB15_1.csv',
        'UNSW-NB15_2.csv',
        'UNSW-NB15_3.csv',
        'UNSW-NB15_4.csv'
    ]
    
    # Try to find preferred files first
    selected_file = None
    for preferred in preferred_files:
        for filename, size, path in csv_files:
            if filename == preferred:
                print(f"\n✅ Found preferred file: {filename} ({size/1024:.1f} MB)")
                selected_file = (filename, size, path)
                break
        if selected_file:
            break
    
    # If no preferred file, use the largest CSV
    if not selected_file and csv_files:
        csv_files.sort(key=lambda x: x[1], reverse=True)  # Sort by size
        selected_file = csv_files[0]
        print(f"\n✅ Selected largest CSV file: {selected_file[0]} ({selected_file[1]/1024:.1f} MB)")
    
    return selected_file[2] if selected_file else None

# Find and load the file
file_path = find_unsw_nb15_file()
if file_path is None:
    print("\n❌ Could not find UNSW-NB15 CSV file!")
    print(f"💡 Please make sure there are CSV files in: {dataset_path}")
    exit()

print(f"\n📂 Loading: {os.path.basename(file_path)}")
print(f"   File size: {os.path.getsize(file_path) / (1024**3):.2f} GB")

# Load data - start with 100K rows (like TON-IoT)
print("   Loading 100,000 rows to ensure both normal and attack samples...")
df_large = pd.read_csv(file_path, nrows=100000, low_memory=False)

# Clean column names
df_large.columns = [col.strip().replace(' ', '_').replace('-', '_').replace('.', '_').lower() 
                   for col in df_large.columns]

# Check label column
if 'label' not in df_large.columns:
    # Try to find label column
    label_candidates = ['label', 'labels', 'attack_cat', 'category', 'class']
    for candidate in label_candidates:
        if candidate in df_large.columns:
            df_large.rename(columns={candidate: 'label'}, inplace=True)
            print(f"   Renamed '{candidate}' to 'label'")
            break

if 'label' not in df_large.columns:
    print("❌ No label column found!")
    exit()

# Check label distribution
label_counts = df_large['label'].value_counts()
print(f"   Label distribution in 100K sample:")
for label, count in label_counts.items():
    print(f"      Label {label}: {count:,} ({count/len(df_large):.2%})")

# Take balanced sample (EXACTLY like TON-IoT)
if len(label_counts) > 1:
    # Sample equal number from each class
    sample_per_class = min(20000, label_counts.min())  # Reduced for faster processing
    
    dfs = []
    for label in label_counts.index:
        class_df = df_large[df_large['label'] == label]
        if len(class_df) > sample_per_class:
            dfs.append(class_df.sample(sample_per_class, random_state=42))
        else:
            dfs.append(class_df)
    
    df = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42)
    print(f"   Created balanced sample: {len(df):,} rows")
else:
    df = df_large.sample(n=40000, random_state=42)

print(f"✅ Dataset loaded successfully!")
print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

# ===== 2. PREPARE LABELS (EXACTLY LIKE TON-IoT) =====
print("\n🔧 Preparing labels (0=normal, 1=attack)...")
# Since label is already 0/1, we can use it directly
# But check if 0 is normal or attack
label_counts = df['label'].value_counts()
print(f"   Original label distribution:")
for label, count in label_counts.items():
    print(f"      Label {label}: {count:,} rows")

# Based on UNSW-NB15 dataset documentation, usually:
# 0 = Normal/Benign, 1 = Attack
# But let's verify by checking 'attack_cat' column
if 'attack_cat' in df.columns:
    print(f"\n   Attack categories distribution:")
    for label in [0, 1]:
        types = df[df['label'] == label]['attack_cat'].unique()[:5]
        print(f"      Label {label} types: {list(types)}")
    
    # Check which label has 'normal' or 'benign' in attack_cat
    normal_labels = []
    for label in [0, 1]:
        type_values = df[df['label'] == label]['attack_cat'].astype(str).str.lower()
        if type_values.str.contains('normal|benign').any():
            normal_labels.append(label)
    
    if normal_labels:
        print(f"   Found normal labels: {normal_labels}")
        # If label 1 is normal and label 0 is attack, we need to swap
        if 1 in normal_labels and 0 not in normal_labels:
            print("   ⚠️ Label 1 appears to be normal, label 0 is attack")
            print("   Swapping labels: 0->attack, 1->normal")
            df['Label_encoded'] = df['label'].map({0: 1, 1: 0})  # Swap
        else:
            print("   Using original labels (0=normal, 1=attack)")
            df['Label_encoded'] = df['label']
    else:
        print("   Using original labels (assuming 0=normal, 1=attack)")
        df['Label_encoded'] = df['label']
else:
    print("   No 'attack_cat' column found. Using original labels.")
    df['Label_encoded'] = df['label']

# ===== 3. PREPARE FEATURES (SIMPLIFIED LIKE TON-IoT) =====
print("\n⚙️  Preparing features...")

# Drop label columns and non-numeric columns
columns_to_drop = ['label', 'Label_encoded']

# Also drop any non-numeric columns (IP addresses, protocols, etc.)
# But first convert what we can
print(f"   Converting object columns to numeric where possible...")
for col in df.columns:
    if df[col].dtype == 'object' and col not in columns_to_drop:
        try:
            # Try to convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                df[col].fillna(df[col].median(), inplace=True)
                print(f"      Converted {col} to numeric")
        except:
            # If can't convert, mark for dropping
            if col not in columns_to_drop:
                columns_to_drop.append(col)
                print(f"      Will drop {col} (cannot convert to numeric)")

# Now get numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_columns = [col for col in df.columns if col not in numeric_columns]

if non_numeric_columns:
    print(f"   Dropping non-numeric columns: {non_numeric_columns}")
    columns_to_drop.extend(non_numeric_columns)

# Remove duplicates from columns_to_drop
columns_to_drop = list(set(columns_to_drop))

X = df.drop(columns=columns_to_drop).values
y = df['Label_encoded'].values

print(f"   Features shape: {X.shape}")
print(f"   Labels shape: {y.shape}")

# Check class balance
normal_count = np.sum(y == 0)
attack_count = np.sum(y == 1)

print(f"\n📈 Final Class Distribution:")
print(f"   Normal samples (0): {normal_count:,} ({normal_count/len(y):.2%})")
print(f"   Attack samples (1): {attack_count:,} ({attack_count/len(y):.2%})")

# Create feature names
feature_names = df.drop(columns=columns_to_drop).columns.tolist()
print(f"\n🔡 Features used ({len(feature_names)}):")
for i, name in enumerate(feature_names[:10]):
    print(f"   {i:2d}. {name}")
if len(feature_names) > 10:
    print(f"   ... and {len(feature_names) - 10} more features")

# ===== 4. SPLIT DATA (EXACTLY LIKE TON-IoT) =====
print("\n✂️  Splitting Data (80% train, 20% test)")
split_idx = int(0.8 * len(X))
X_train = X[:split_idx]
X_test = X[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

print(f"   Training: {len(X_train):,} samples")
print(f"   Testing: {len(X_test):,} samples")

# ===== 5. GET NORMAL DATA FOR TRAINING (EXACTLY LIKE TON-IoT) =====
print("\n🎯 Getting Normal Data for Training (Y_train == 0)")
normal_mask = (y_train == 0)
X_train_normal = X_train[normal_mask]
print(f"   Using {len(X_train_normal):,} normal samples for training")

# ===== 6. PREPROCESS (EXACTLY LIKE TON-IoT) =====
print("\n⚙️  Preprocessing")
scaler = StandardScaler()
X_train_normal_scaled = scaler.fit_transform(X_train_normal)
X_test_scaled = scaler.transform(X_test)
print("   Scaling complete")

# ===== 7. TRAIN ISOLATION FOREST (EXACTLY LIKE TON-IoT) =====
print("\n🚀 Training Isolation Forest")
# Adjust contamination based on actual attack rate in training data
train_attack_rate = (len(y_train) - len(X_train_normal)) / len(y_train)
print(f"   Actual attack rate in training data: {train_attack_rate:.3%}")
print(f"   Setting contamination to: {max(0.01, min(0.1, train_attack_rate)):.3%}")

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=max(0.01, min(0.1, train_attack_rate)),
    random_state=42,
    n_jobs=-1,
    verbose=0,
    max_samples='auto'
)
print("   Training on normal data only...")
iso_forest.fit(X_train_normal_scaled)
print("   Training complete!")

# ===== 8. EVALUATE (EXACTLY LIKE TON-IoT) =====
print("\n🔍 Evaluating on Test Set")
test_scores = iso_forest.decision_function(X_test_scaled)
test_anomaly_scores = 0.5 - test_scores / 2

# Set threshold from normal training data
train_scores = iso_forest.decision_function(X_train_normal_scaled)
train_anomaly_scores = 0.5 - train_scores / 2
threshold = np.percentile(train_anomaly_scores, 95)

# Make predictions
test_predictions = (test_anomaly_scores > threshold).astype(int)

# ===== 9. VISUALIZATIONS & ANALYSIS (EXACTLY LIKE TON-IoT) =====
print("\n📊 Creating Visualizations...")

# Set plotting style (EXACTLY LIKE TON-IoT)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

fig = plt.figure(figsize=(20, 16))

# Plot 1: Anomaly Score Distribution (EXACTLY LIKE TON-IoT)
ax1 = plt.subplot(2, 3, 1)
normal_scores = test_anomaly_scores[y_test == 0]
attack_scores = test_anomaly_scores[y_test == 1]
sns.histplot(normal_scores, bins=50, kde=True, color='green', alpha=0.5, label='Normal', ax=ax1)
sns.histplot(attack_scores, bins=50, kde=True, color='red', alpha=0.5, label='Attack', ax=ax1)
ax1.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
ax1.set_xlabel('Anomaly Score')
ax1.set_ylabel('Count')
ax1.set_title('Anomaly Score Distribution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Confusion Matrix (EXACTLY LIKE TON-IoT)
ax2 = plt.subplot(2, 3, 2)
cm = confusion_matrix(y_test, test_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax2,
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
ax2.set_title('Confusion Matrix')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

# Plot 3: ROC Curve (EXACTLY LIKE TON-IoT)
ax3 = plt.subplot(2, 3, 3)
fpr, tpr, thresholds = roc_curve(y_test, test_anomaly_scores)
roc_auc = auc(fpr, tpr)
ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curve')
ax3.legend(loc="lower right")
ax3.grid(True, alpha=0.3)

# Plot 4: Precision-Recall Curve (EXACTLY LIKE TON-IoT)
ax4 = plt.subplot(2, 3, 4)
precision, recall, _ = precision_recall_curve(y_test, test_anomaly_scores)
avg_precision = average_precision_score(y_test, test_anomaly_scores)
ax4.plot(recall, precision, color='purple', lw=2, label=f'Avg Precision = {avg_precision:.3f}')
ax4.set_xlabel('Recall')
ax4.set_ylabel('Precision')
ax4.set_title('Precision-Recall Curve')
ax4.legend(loc="lower left")
ax4.grid(True, alpha=0.3)

# Plot 5: Feature Importance (EXACTLY LIKE TON-IoT)
ax5 = plt.subplot(2, 3, 5)
if hasattr(iso_forest.estimators_[0], 'feature_importances_'):
    feature_importance = np.std([tree.feature_importances_ for tree in iso_forest.estimators_], axis=0)
    sorted_idx = np.argsort(feature_importance)[-10:]
    bars = ax5.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
    ax5.set_yticks(range(len(sorted_idx)))
    ax5.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax5.set_xlabel('Feature Importance')
    ax5.set_title('Top 10 Important Features')
    plt.setp(ax5.get_yticklabels(), fontsize=9)

# Plot 6: Metrics Summary (EXACTLY LIKE TON-IoT)
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
TN, FP, FN, TP = cm.ravel()
metrics = {
    'Accuracy': (TP + TN) / (TP + TN + FP + FN),
    'Precision': TP / (TP + FP) if (TP + FP) > 0 else 0,
    'Recall': TP / (TP + FN) if (TP + FN) > 0 else 0,
    'F1-Score': 2 * (TP / (TP + FP)) * (TP / (TP + FN)) / (TP / (TP + FP) + TP / (TP + FN)) 
               if (TP + FP) > 0 and (TP + FN) > 0 else 0,
    'Specificity': TN / (TN + FP) if (TN + FP) > 0 else 0,
    'AUC-ROC': roc_auc
}
y_pos = 0.9
for metric, value in metrics.items():
    ax6.text(0.1, y_pos, f'{metric}: {value:.4f}', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    y_pos -= 0.12
ax6.set_title('Performance Metrics', fontsize=12)

plt.suptitle(f'UNSW-NB15 Isolation Forest - {len(X_train_normal):,} Normal Samples', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
_savefig('isolation_forest_unsw_nb15')

# ===== 10. PERFORMANCE METRICS (EXACTLY LIKE TON-IoT) =====
print("\n" + "="*60)
print("📊 PERFORMANCE METRICS")
print("="*60)

print("\n📋 Classification Report:")
print(classification_report(y_test, test_predictions, 
                          target_names=['Normal', 'Attack'],
                          digits=4))

print("\n🎯 Detailed Metrics:")
print(f"{'Metric':<15} {'Value':<10} {'Interpretation'}")
print("-" * 50)
print(f"{'Accuracy':<15} {metrics['Accuracy']:.4f}     {'Higher is better'}")
print(f"{'Precision':<15} {metrics['Precision']:.4f}     {'Attack predictions that are correct'}")
print(f"{'Recall':<15} {metrics['Recall']:.4f}     {'Attacks correctly detected'}")
print(f"{'F1-Score':<15} {metrics['F1-Score']:.4f}     {'Balance of Precision/Recall'}")
print(f"{'Specificity':<15} {metrics['Specificity']:.4f}     {'Normals correctly identified'}")
print(f"{'AUC-ROC':<15} {metrics['AUC-ROC']:.4f}     {'Overall model performance'}")

print("\n" + "="*70)
print("✅ ISOLATION FOREST ANALYSIS COMPLETE!")
print("="*70)