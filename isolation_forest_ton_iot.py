# ===== ENHANCED ISOLATION FOREST FOR TON_IoT =====
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, auc, precision_recall_curve, 
                           average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

print("="*70)
print("ENHANCED ISOLATION FOREST - Train on Normal Data Only")
print("="*70)

# ===== 1. LOAD REAL TON_IoT DATA =====
print("\n📊 Loading Real TON_IoT Dataset")
dataset_path = "/home/darine/cybersecurity_assignment/datasets/ton_iot/Extra-Column-removed-TonIoT.csv"

# Load the CSV file with balanced sampling
print(f"   Reading from: {dataset_path}")
print(f"   File size: {os.path.getsize(dataset_path) / (1024**3):.2f} GB")

# Load data - start with 500K rows
print("   Loading 500,000 rows to ensure both normal and attack samples...")
df_large = pd.read_csv(dataset_path, nrows=500000)

# Check label distribution
label_counts = df_large['label'].value_counts()
print(f"   Label distribution in 500K sample:")
for label, count in label_counts.items():
    print(f"      Label {label}: {count:,} ({count/len(df_large):.2%})")

# Take balanced sample
if len(label_counts) > 1:
    # Sample equal number from each class
    sample_per_class = min(50000, label_counts.min())
    
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
    df = df_large.sample(n=100000, random_state=42)

# Fix data types
print("\n🔧 Fixing data types...")
# Convert src_bytes from object to numeric
df['src_bytes'] = pd.to_numeric(df['src_bytes'], errors='coerce')
nan_count = df['src_bytes'].isna().sum()
if nan_count > 0:
    print(f"   Converted {nan_count} non-numeric src_bytes to numeric")
    df['src_bytes'].fillna(df['src_bytes'].median(), inplace=True)

print(f"✅ Dataset loaded successfully!")
print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
print(f"   Columns: {list(df.columns)}")

# Show data info
print("\n📋 Data Info:")
print(df.info())

# ===== 2. PREPARE LABELS =====
print("\n🔧 Preparing labels (0=normal, 1=attack)...")
# Since label is already 0/1, we can use it directly
# But check if 0 is normal or attack
label_counts = df['label'].value_counts()
print(f"   Original label distribution:")
for label, count in label_counts.items():
    print(f"      Label {label}: {count:,} rows")

# Based on TON_IoT dataset documentation, usually:
# 0 = Normal/Benign, 1 = Attack
# But let's verify by checking 'type' column
if 'type' in df.columns:
    print(f"\n   Attack types distribution:")
    for label in [0, 1]:
        types = df[df['label'] == label]['type'].unique()[:5]
        print(f"      Label {label} types: {list(types)}")
    
    # Check which label has 'normal' or 'benign' in type
    normal_labels = []
    for label in [0, 1]:
        type_values = df[df['label'] == label]['type'].astype(str).str.lower()
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
    print("   No 'type' column found. Using original labels.")
    df['Label_encoded'] = df['label']

# ===== 3. PREPARE FEATURES =====
print("\n⚙️  Preparing features...")

# Drop label columns and non-numeric columns
columns_to_drop = ['label', 'Label_encoded']

# Also drop any non-numeric columns (IP addresses, protocols, etc.)
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
for i, name in enumerate(feature_names):
    print(f"   {i:2d}. {name}")

# ===== 4. SPLIT DATA =====
print("\n✂️  Splitting Data (80% train, 20% test)")
split_idx = int(0.8 * len(X))
X_train = X[:split_idx]
X_test = X[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

print(f"   Training: {len(X_train):,} samples")
print(f"   Testing: {len(X_test):,} samples")

# ===== 5. GET NORMAL DATA FOR TRAINING =====
print("\n🎯 Getting Normal Data for Training (Y_train == 0)")
normal_mask = (y_train == 0)
X_train_normal = X_train[normal_mask]
print(f"   Using {len(X_train_normal):,} normal samples for training")



# ===== 6. PREPROCESS =====
print("\n⚙️  Preprocessing")
scaler = StandardScaler()
X_train_normal_scaled = scaler.fit_transform(X_train_normal)
X_test_scaled = scaler.transform(X_test)
print("   Scaling complete")




# ===== 7. TRAIN ISOLATION FOREST =====
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

# ===== 8. EVALUATE =====
print("\n🔍 Evaluating on Test Set")
test_scores = iso_forest.decision_function(X_test_scaled)
test_anomaly_scores = 0.5 - test_scores / 2

# Set threshold from normal training data
train_scores = iso_forest.decision_function(X_train_normal_scaled)
train_anomaly_scores = 0.5 - train_scores / 2
threshold = np.percentile(train_anomaly_scores, 95)

# Make predictions
test_predictions = (test_anomaly_scores > threshold).astype(int)

# ===== 9. VISUALIZATIONS & ANALYSIS =====
print("\n📊 Creating Visualizations...")

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

fig = plt.figure(figsize=(20, 16))

# Plot 1: Anomaly Score Distribution
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

# Plot 2: Confusion Matrix
ax2 = plt.subplot(2, 3, 2)
cm = confusion_matrix(y_test, test_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax2,
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
ax2.set_title('Confusion Matrix')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

# Plot 3: ROC Curve
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

# Plot 4: Precision-Recall Curve
ax4 = plt.subplot(2, 3, 4)
precision, recall, _ = precision_recall_curve(y_test, test_anomaly_scores)
avg_precision = average_precision_score(y_test, test_anomaly_scores)
ax4.plot(recall, precision, color='purple', lw=2, label=f'Avg Precision = {avg_precision:.3f}')
ax4.set_xlabel('Recall')
ax4.set_ylabel('Precision')
ax4.set_title('Precision-Recall Curve')
ax4.legend(loc="lower left")
ax4.grid(True, alpha=0.3)

# Plot 5: Feature Importance
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

# Plot 6: Metrics Summary
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

plt.suptitle(f'Ton_IoT Isolation Forest - {len(X_train_normal):,} Normal Samples', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ===== 10. PERFORMANCE METRICS =====
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