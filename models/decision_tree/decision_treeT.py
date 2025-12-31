# =============================================================================
# DECISION TREE ML MODEL FOR TON-IoT DATASET
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DECISION TREE ML MODEL - TON-IoT DATASET")
print("=" * 70)

# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("\n1. LOADING TON-IoT DATASET")
print("-" * 40)

# Path to TON-IoT dataset directory
TON_IOT_DIR = "/home/darine/cybersecurity_assignment/datasets/ton_iot"

def find_ton_iot_file():
    """Find and load a TON-IoT dataset file"""
    
    print(f"Looking for TON-IoT dataset in: {TON_IOT_DIR}")
    
    # Check if directory exists
    if not os.path.exists(TON_IOT_DIR):
        print(f"❌ Directory not found: {TON_IOT_DIR}")
        return None
    
    print(f"✅ Found TON-IoT directory")
    
    # List all files
    all_items = os.listdir(TON_IOT_DIR)
    
    if not all_items:
        print("❌ Directory is empty!")
        return None
    
    # Look for CSV files first (easier to load)
    csv_files = []
    other_files = []
    
    for item in all_items:
        item_path = os.path.join(TON_IOT_DIR, item)
        if os.path.isfile(item_path):
            if item.lower().endswith('.csv'):
                size = os.path.getsize(item_path) / 1024  # KB
                csv_files.append((item, size, item_path))
            else:
                other_files.append(item)
    
    # Show files
    if csv_files:
        print(f"\n📄 Found {len(csv_files)} CSV files:")
        for filename, size, path in csv_files:
            print(f"   📄 {filename} ({size:.1f} KB)")
    
    if other_files:
        print(f"\n📝 Other files ({len(other_files)}):")
        for file in other_files[:5]:
            print(f"   📝 {file}")
    
    # Use CSV file if available (easier to load)
    if csv_files:
        # Sort by size (largest first)
        csv_files.sort(key=lambda x: x[1], reverse=True)
        selected_file = csv_files[0]
        print(f"\n✅ Selected CSV file: {selected_file[0]} ({selected_file[1]:.1f} KB)")
        return selected_file[2]
    else:
        print("❌ No CSV files found!")
        return None

# Find and load the file
file_path = find_ton_iot_file()
if file_path is None:
    print("\n❌ Could not find TON-IoT CSV file!")
    print(f"💡 Please make sure there are CSV files in: {TON_IOT_DIR}")
    exit()

print(f"\n📂 Loading: {os.path.basename(file_path)}")

# Try different loading approaches
max_rows = 10000  # Reduced for faster processing

try:
    # First try: Standard CSV loading
    print(f"   Loading CSV file (first {max_rows:,} rows)...")
    df = pd.read_csv(file_path, nrows=max_rows, low_memory=False)
    print(f"✅ Successfully loaded {len(df)} rows, {len(df.columns)} columns")
    
except Exception as e:
    print(f"❌ Error with standard loading: {str(e)}")
    
    # Try different approaches
    encoding_tried = []
    
    # Try different encodings
    encodings = ['latin-1', 'ISO-8859-1', 'utf-8', 'cp1252']
    
    for encoding in encodings:
        try:
            print(f"   Trying with {encoding} encoding...")
            df = pd.read_csv(file_path, nrows=max_rows, encoding=encoding, low_memory=False)
            print(f"✅ Loaded with {encoding} encoding")
            print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
            break
        except Exception as e_enc:
            encoding_tried.append(f"{encoding}: {str(e_enc)[:50]}...")
    
    if 'df' not in locals():
        print("\n❌ All encoding attempts failed:")
        for attempt in encoding_tried:
            print(f"   {attempt}")
        
        # Try with error handling
        print("\n🔄 Trying with error handling...")
        try:
            df = pd.read_csv(file_path, nrows=max_rows, on_bad_lines='skip', low_memory=False)
            print(f"✅ Loaded with error skipping")
            print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
        except Exception as e_final:
            print(f"❌ Final attempt failed: {str(e_final)}")
            exit()

# =============================================================================
# 2. DATA EXPLORATION AND CLEANING
# =============================================================================
print("\n2. DATA EXPLORATION AND CLEANING")
print("-" * 40)

# Show basic info
print(f"\n📊 Dataset Information:")
print(f"   File: {os.path.basename(file_path)}")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Memory: {df.memory_usage().sum() / 1024**2:.2f} MB")

# Display column information
print(f"\n📝 First 20 columns:")
for i, col in enumerate(df.columns[:20], 1):
    dtype = df[col].dtype
    unique = df[col].nunique()
    print(f"   {i:2d}. {col:30s} {str(dtype):10s} {unique:6d} unique")

if len(df.columns) > 20:
    print(f"   ... and {len(df.columns) - 20} more columns")

# Clean column names (remove spaces, special characters)
print(f"\n🔧 Cleaning column names...")
original_columns = df.columns.tolist()
df.columns = [col.strip().replace(' ', '_').replace('-', '_').replace('.', '_').lower() for col in df.columns]
print(f"✅ Column names cleaned")

# Find label column
print(f"\n🔍 Searching for label/attack columns...")
label_col = None

# Common label column names in IoT datasets
label_keywords = ['label', 'attack', 'type', 'class', 'category', 
                  'malicious', 'anomaly', 'result', 'target']

for col in df.columns:
    col_lower = col.lower()
    if any(keyword in col_lower for keyword in label_keywords):
        unique_vals = df[col].nunique()
        if 2 <= unique_vals <= 20:  # Reasonable for classification
            label_col = col
            print(f"✅ Found label column: '{label_col}' ({unique_vals} unique values)")
            break

if label_col is None:
    print("⚠️  No obvious label column found. Checking all columns...")
    
    # Look for columns with few unique values (potential labels)
    potential_labels = []
    for col in df.columns:
        unique_vals = df[col].nunique()
        if 2 <= unique_vals <= 10:  # Good for classification
            potential_labels.append((col, unique_vals))
    
    if potential_labels:
        print("\nPotential label columns found:")
        for i, (col, unique_count) in enumerate(potential_labels[:5], 1):
            sample_vals = df[col].unique()[:3]
            print(f"   {i}. '{col}' ({unique_count} values): {sample_vals}")
        
        # Auto-select the first one
        label_col = potential_labels[0][0]
        print(f"\n✅ Auto-selected: '{label_col}'")
    else:
        # Last resort: check last column
        last_col = df.columns[-1]
        unique_vals = df[last_col].nunique()
        if 2 <= unique_vals <= 20:
            label_col = last_col
            print(f"⚠️  Using last column as label: '{label_col}'")
        else:
            print("❌ No suitable label column found!")
            print("\nColumn list:", df.columns.tolist())
            exit()

# Display label distribution
print(f"\n🎯 Label Analysis for '{label_col}':")
label_counts = df[label_col].value_counts()
print(f"Total unique labels: {len(label_counts)}")

print(f"\nLabel distribution (showing all):")
for i, (label, count) in enumerate(label_counts.items(), 1):
    percentage = (count / len(df)) * 100
    print(f"   {i:2d}. {str(label)[:25]:25s} {count:6d} ({percentage:.1f}%)")

# Identify normal vs attack
print(f"\n🔍 Identifying attack types...")
normal_labels = []
attack_labels = []
unknown_labels = []

for label in label_counts.index:
    label_str = str(label).lower()
    
    # Check for normal/benign
    if any(keyword in label_str for keyword in ['normal', 'benign', 'legitimate', '0', 'no', 'false']):
        normal_labels.append(label)
    # Check for attacks
    elif any(keyword in label_str for keyword in ['attack', 'malicious', 'anomaly', '1', 'yes', 'true', 
                                                  'ddos', 'dos', 'scan', 'inject', 'backdoor', 'xss', 
                                                  'malware', 'ransomware', 'botnet']):
        attack_labels.append(label)
    else:
        unknown_labels.append(label)

print(f"\n📊 Classification:")
if normal_labels:
    print(f"   ✅ Normal/Benign labels: {normal_labels}")
if attack_labels:
    print(f"   ⚠️  Attack labels: {attack_labels}")
if unknown_labels:
    print(f"   ❓ Unknown labels: {unknown_labels}")

# =============================================================================
# 3. DATA PREPROCESSING
# =============================================================================
print("\n3. DATA PREPROCESSING")
print("-" * 40)

# Create target variable
print(f"\n🎯 Creating target variable...")

# Convert to binary classification (Normal vs Attack)
binary_classification = True

if df[label_col].dtype == 'object':
    # String labels - convert to binary
    print(f"   Converting string labels to binary...")
    
    def label_to_binary(label):
        label_str = str(label).lower()
        
        # Check if it's normal
        if any(keyword in label_str for keyword in ['normal', 'benign', 'legitimate', '0', 'no', 'false']):
            return 0
        # Check if it's attack
        elif any(keyword in label_str for keyword in ['attack', 'malicious', 'anomaly', '1', 'yes', 'true']):
            return 1
        # Default: unknown -> treat as attack
        else:
            return 1
    
    y = df[label_col].apply(label_to_binary).astype(int)
    print(f"✅ Binary target created: 0=Normal, 1=Attack")
    
else:
    # Numeric labels
    print(f"   Processing numeric labels...")
    y = df[label_col].astype(int)
    
    # Check if binary
    unique_vals = np.unique(y)
    if set(unique_vals).issubset({0, 1}):
        print(f"   Already binary labels")
    else:
        print(f"   Converting to binary (0=lowest value, 1=others)")
        # Convert: minimum value = normal (0), others = attack (1)
        min_val = y.min()
        y = (y != min_val).astype(int)
        print(f"✅ Binary target created: {min_val}=Normal, others=Attack")

print(f"\n📊 Class Distribution:")
normal_count = sum(y == 0)
attack_count = sum(y == 1)
print(f"   Normal (0): {normal_count:,} samples ({normal_count/len(y)*100:.1f}%)")
print(f"   Attack (1): {attack_count:,} samples ({attack_count/len(y)*100:.1f}%)")

# Select features
print(f"\n🔧 Selecting features...")

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove label column if it's numeric
if label_col in numeric_cols:
    numeric_cols.remove(label_col)

print(f"   Found {len(numeric_cols)} numeric columns")

# Select features (choose 8-12 good features)
print(f"   Selecting 10 features for analysis...")

# Remove constant or near-constant features
selected_features = []
for col in numeric_cols:
    if df[col].nunique() > 1:  # Not constant
        selected_features.append(col)

# If we have many features, select top ones by variance
if len(selected_features) > 10:
    print(f"   Selecting top 10 features by variance...")
    variances = df[selected_features].var()
    selected_features = variances.nlargest(10).index.tolist()

print(f"✅ Selected {len(selected_features)} features:")
for i, feat in enumerate(selected_features, 1):
    print(f"   {i:2d}. {feat}")

# Create feature matrix
X = df[selected_features]

# Handle missing values
print(f"\n🧹 Cleaning data...")
missing_before = X.isnull().sum().sum()
if missing_before > 0:
    print(f"   Found {missing_before} missing values")
    X = X.fillna(X.median())
    print(f"✅ Filled missing values with median")

# Remove infinite values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# Feature scaling
print(f"\n⚖️  Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"✅ Features scaled using StandardScaler")

# =============================================================================
# 4. DATA SPLITTING
# =============================================================================
print("\n4. DATA SPLITTING")
print("-" * 40)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"✅ Data split complete:")
print(f"   Training set: {X_train.shape[0]:,} samples")
print(f"   Test set:     {X_test.shape[0]:,} samples")
print(f"   Features:     {X_train.shape[1]}")

print(f"\n📈 Class distribution:")
print(f"   Training - Normal: {sum(y_train == 0):,}, Attack: {sum(y_train == 1):,}")
print(f"   Testing  - Normal: {sum(y_test == 0):,}, Attack: {sum(y_test == 1):,}")

# =============================================================================
# 5. DECISION TREE MODEL TRAINING
# =============================================================================
print("\n5. DECISION TREE MODEL TRAINING")
print("-" * 40)

print("🤖 Training Decision Tree Classifier...")

# Create and train the model
dt_model = DecisionTreeClassifier(
    max_depth=5,           # Limit depth for interpretability
    min_samples_split=20,  # Minimum samples to split
    min_samples_leaf=10,   # Minimum samples in leaf
    random_state=42,       # Reproducibility
    class_weight='balanced' # Handle imbalance
)

dt_model.fit(X_train, y_train)
print(f"✅ Model trained successfully!")

print(f"\n📊 Model Parameters:")
print(f"   Tree depth: {dt_model.get_depth()}")
print(f"   Number of leaves: {dt_model.get_n_leaves()}")
print(f"   Features used: {dt_model.n_features_in_}")

# =============================================================================
# 6. MODEL EVALUATION
# =============================================================================
print("\n6. MODEL EVALUATION")
print("-" * 40)

# Make predictions
y_pred = dt_model.predict(X_test)
y_pred_proba = dt_model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"📈 Performance Metrics:")
print(f"   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")

# ROC-AUC
try:
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"   ROC-AUC:   {roc_auc:.4f}")
except:
    print(f"   ROC-AUC:   Could not calculate")

print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack'], zero_division=0))

# Cross-validation
print(f"\n🔍 Cross-Validation (5-fold)...")
cv_scores = cross_val_score(dt_model, X_scaled, y, cv=5, scoring='accuracy')
print(f"   CV Scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"   Mean CV Score: {cv_scores.mean():.4f}")
print(f"   Std CV Score:  {cv_scores.std():.4f}")

# =============================================================================
# 7. FEATURE IMPORTANCE
# =============================================================================
print("\n7. FEATURE IMPORTANCE ANALYSIS")
print("-" * 40)

feature_importance = dt_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print(f"🏆 Feature Importance Ranking:")
print(importance_df.to_string(index=False))

# =============================================================================
# 8. VISUALIZATIONS
# =============================================================================
print("\n8. VISUALIZATIONS")
print("-" * 40)
print("📊 Generating visualizations...")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# 1. Decision Tree
print("\n📈 Visualization 1: Decision Tree")
plt.figure(figsize=(18, 10))
plot_tree(dt_model,
          feature_names=selected_features,
          class_names=['Normal', 'Attack'],
          filled=True,
          rounded=True,
          max_depth=3,
          fontsize=9,
          proportion=True)
plt.title('TON-IoT Decision Tree (First 3 Levels)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# 2. Feature Importance
print("📈 Visualization 2: Feature Importance")
plt.figure(figsize=(12, 6))
top_features = importance_df.head(10)

bars = plt.barh(range(len(top_features)), top_features['Importance'], 
                color=plt.cm.Blues(np.linspace(0.4, 0.9, len(top_features))))
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.title('Top 10 Feature Importance - TON-IoT', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

for i, (bar, imp) in enumerate(zip(bars, top_features['Importance'])):
    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f'{imp:.3f}', va='center', fontsize=10, fontweight='bold')

plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# 3. Confusion Matrix
print("📈 Visualization 3: Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
plt.title('Confusion Matrix - TON-IoT', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()

# 4. Performance Metrics
print("📈 Visualization 4: Performance Metrics")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

metrics = [('Accuracy', accuracy, 'green'),
           ('Precision', precision, 'blue'),
           ('Recall', recall, 'orange'),
           ('F1-Score', f1, 'purple')]

for idx, (name, value, color) in enumerate(metrics):
    row, col = divmod(idx, 2)
    axes[row, col].bar([name], [value], color=color, alpha=0.7, width=0.6)
    axes[row, col].set_ylim(0, 1.1)
    axes[row, col].set_title(name, fontweight='bold')
    axes[row, col].set_ylabel('Score')
    axes[row, col].text(0, value + 0.02, f'{value:.3f}', 
                       ha='center', fontweight='bold')
    axes[row, col].grid(True, alpha=0.3)
    axes[row, col].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

plt.suptitle('TON-IoT Model Performance Metrics', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# 5. Dataset Summary
print("📈 Visualization 5: Dataset Summary")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Class distribution pie chart
class_counts = [normal_count, attack_count]
colors = ['lightblue', 'lightcoral']
ax1.pie(class_counts, labels=['Normal', 'Attack'], colors=colors,
        autopct='%1.1f%%', startangle=90, explode=(0.05, 0.05))
ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')

# Dataset info text
info_text = f"TON-IoT Dataset Summary\n{'='*25}\n\n"
info_text += f"File: {os.path.basename(file_path)}\n"
info_text += f"Samples: {len(df):,}\n"
info_text += f"Features: {len(selected_features)}\n"
info_text += f"Normal: {normal_count:,}\n"
info_text += f"Attack: {attack_count:,}\n\n"
info_text += f"Model Accuracy: {accuracy:.3f}\n"
info_text += f"Tree Depth: {dt_model.get_depth()}\n"
info_text += f"CV Score: {cv_scores.mean():.3f}"

ax2.text(0.1, 0.5, info_text, fontsize=12, 
         verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
ax2.axis('off')

plt.suptitle('TON-IoT Cybersecurity Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# =============================================================================
# 9. SAVE RESULTS
# =============================================================================
print("\n9. SAVING RESULTS")
print("-" * 40)

# Create directory
results_dir = "ton_iot_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"✅ Created directory: {results_dir}")

# Save results
results_files = []

# Feature importance
imp_file = os.path.join(results_dir, "feature_importance.csv")
importance_df.to_csv(imp_file, index=False)
results_files.append(imp_file)
print(f"✅ Saved: feature_importance.csv")

# Predictions
pred_file = os.path.join(results_dir, "predictions.csv")
pd.DataFrame({
    'y_true': y_test,
    'y_pred': y_pred,
    'y_prob': y_pred_proba
}).to_csv(pred_file, index=False)
results_files.append(pred_file)
print(f"✅ Saved: predictions.csv")

# Model metrics
metrics_file = os.path.join(results_dir, "model_metrics.csv")
pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'CV_Mean', 'CV_Std'],
    'Value': [accuracy, precision, recall, f1, roc_auc if 'roc_auc' in locals() else np.nan,
              cv_scores.mean(), cv_scores.std()]
}).to_csv(metrics_file, index=False)
results_files.append(metrics_file)
print(f"✅ Saved: model_metrics.csv")

print(f"\n📁 All results saved in: {results_dir}/")

# =============================================================================
# 10. FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("🎉 TON-IoT DECISION TREE ANALYSIS - COMPLETE!")
print("=" * 70)

print(f"\n📋 SUMMARY:")
print(f"   Dataset: {os.path.basename(file_path)}")
print(f"   Samples: {len(df):,}")
print(f"   Normal: {normal_count:,} ({normal_count/len(df)*100:.1f}%)")
print(f"   Attack: {attack_count:,} ({attack_count/len(df)*100:.1f}%)")
print(f"   Features used: {len(selected_features)}")
print(f"   Model accuracy: {accuracy:.3f}")
print(f"   Attack detection (Recall): {recall:.3f}")
print(f"   Cross-validation: {cv_scores.mean():.3f}")

print(f"\n✅ Successfully trained decision tree on TON-IoT dataset!")
print(f"📊 Generated 5 visualizations")
print(f"💾 Saved 3 CSV files with results")
print(f"🚀 Model ready for IoT intrusion detection")

print("\n" + "=" * 70)