# =============================================================================
# DECISION TREE ML MODEL FOR CICIDS2017 CYBERSECURITY DATASET
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DECISION TREE ML MODEL - CICIDS2017 DATASET")
print("=" * 70)

# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("\n1. LOADING CICIDS2017 DATASET")
print("-" * 40)

# Path to your CICIDS2017 directory
CICIDS2017_DIR = "/home/darine/cybersecurity_assignment/datasets/cicids2017"

def find_cicids2017_file():
    """Find and load a CICIDS2017 file from the directory"""
    
    # First check if the directory exists
    if not os.path.exists(CICIDS2017_DIR):
        print(f"❌ Directory not found: {CICIDS2017_DIR}")
        
        # Check parent directory
        parent_dir = "/home/darine/cybersecurity_assignment/datasets"
        if os.path.exists(parent_dir):
            print(f"\n📁 Files in parent directory {parent_dir}:")
            files = os.listdir(parent_dir)
            for item in files[:15]:  # Show first 15 items
                item_path = os.path.join(parent_dir, item)
                if os.path.isfile(item_path):
                    print(f"   📄 {item}")
                else:
                    print(f"   📂 {item}/")
        return None
    
    print(f"✅ Found CICIDS2017 directory: {CICIDS2017_DIR}")
    
    # List all files in the directory
    print(f"\n📁 Contents of CICIDS2017 directory:")
    all_files = os.listdir(CICIDS2017_DIR)
    
    if not all_files:
        print("❌ Directory is empty!")
        return None
    
    # Show files
    csv_files = []
    for item in all_files:
        item_path = os.path.join(CICIDS2017_DIR, item)
        if os.path.isfile(item_path):
            if item.lower().endswith('.csv'):
                size = os.path.getsize(item_path) / (1024*1024)  # MB
                print(f"   📄 {item} ({size:.1f} MB)")
                csv_files.append(item_path)
            else:
                print(f"   📄 {item} (not CSV)")
        else:
            print(f"   📂 {item}/")
    
    if not csv_files:
        print("❌ No CSV files found in the directory!")
        return None
    
    print(f"\n✅ Found {len(csv_files)} CSV file(s)")
    
    # Common CICIDS2017 file patterns
    cicids_patterns = [
        'friday', 'monday', 'tuesday', 'wednesday', 'thursday',
        'cicids', 'cic-ids', 'ids2017', 'ddos', 'portscan', 'webattack',
        'benign', 'attack', 'workinghours', 'iscx'
    ]
    
    # Try to identify CICIDS2017 files
    cicids_files = []
    for file_path in csv_files:
        filename = os.path.basename(file_path).lower()
        if any(pattern in filename for pattern in cicids_patterns):
            cicids_files.append(file_path)
            print(f"   🔍 Likely CICIDS2017 file: {os.path.basename(file_path)}")
    
    # If we found likely CICIDS2017 files, use the first one
    if cicids_files:
        selected_file = cicids_files[0]
        print(f"\n✅ Selected: {os.path.basename(selected_file)}")
        return selected_file
    
    # Otherwise, use the largest CSV file (CICIDS2017 files are typically large)
    print(f"\n📊 No obvious CICIDS2017 files found. Selecting largest CSV file...")
    file_sizes = []
    for file_path in csv_files:
        size = os.path.getsize(file_path) / (1024*1024)
        file_sizes.append((file_path, size))
    
    # Sort by size (largest first)
    file_sizes.sort(key=lambda x: x[1], reverse=True)
    
    selected_file = file_sizes[0][0]
    print(f"✅ Selected largest file: {os.path.basename(selected_file)} ({file_sizes[0][1]:.1f} MB)")
    return selected_file

# Find and load the file
file_path = find_cicids2017_file()
if file_path is None:
    print("\n❌ Could not find CICIDS2017 dataset!")
    print(f"💡 Please check that the directory exists: {CICIDS2017_DIR}")
    print(f"💡 And that it contains CSV files")
    exit()

print(f"\n📂 Loading: {os.path.basename(file_path)}")

# Load data - try different approaches
max_rows = 15000  # Reduced for faster processing
try:
    print(f"   Loading first {max_rows:,} rows...")
    df = pd.read_csv(file_path, nrows=max_rows, low_memory=False)
    print(f"✅ Successfully loaded {len(df)} rows, {len(df.columns)} columns")
except Exception as e:
    print(f"❌ Error with default loading: {str(e)}")
    print("   Trying with latin-1 encoding...")
    try:
        df = pd.read_csv(file_path, nrows=max_rows, encoding='latin-1', low_memory=False)
        print(f"✅ Loaded with latin-1 encoding")
    except Exception as e2:
        print(f"❌ Error with latin-1: {str(e2)}")
        print("   Trying with ISO-8859-1 encoding...")
        try:
            df = pd.read_csv(file_path, nrows=max_rows, encoding='ISO-8859-1', low_memory=False)
            print(f"✅ Loaded with ISO-8859-1 encoding")
        except Exception as e3:
            print(f"❌ Failed to load file: {str(e3)}")
            exit()

# =============================================================================
# 2. DATA EXPLORATION
# =============================================================================
print("\n2. DATA EXPLORATION")
print("-" * 40)

# Show basic info
print(f"\n📊 Dataset Information:")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")

# Display first few column names
print(f"\n📝 First 15 columns:")
for i, col in enumerate(df.columns[:15], 1):
    dtype = df[col].dtype
    unique = df[col].nunique()
    print(f"   {i:2d}. {col:40s} {str(dtype):10s} {unique:6d} unique")

if len(df.columns) > 15:
    print(f"   ... and {len(df.columns) - 15} more columns")

# Find label column - CICIDS2017 typically uses 'Label'
print(f"\n🔍 Searching for label column...")
label_col = None
label_candidates = ['Label', 'label', 'Class', 'class', 'Result', 'result', 'Attack', 'attack']

for col in label_candidates:
    if col in df.columns:
        label_col = col
        print(f"✅ Found label column: '{label_col}'")
        break

if label_col is None:
    print("⚠️  No standard label column found. Checking all columns...")
    # Look for columns that might be labels
    potential_labels = []
    for col in df.columns:
        unique_vals = df[col].nunique()
        if 2 <= unique_vals <= 20:  # Reasonable number for classification
            potential_labels.append((col, unique_vals))
    
    if potential_labels:
        print("Potential label columns found:")
        for col, unique_count in potential_labels[:5]:  # Show top 5
            print(f"   - '{col}' ({unique_count} unique values)")
        # Use the first one
        label_col = potential_labels[0][0]
        print(f"✅ Using '{label_col}' as label column")
    else:
        print("❌ No suitable label column found!")
        print("Available columns:", df.columns.tolist()[:20])
        exit()

# Display label distribution
print(f"\n🎯 Label Analysis:")
label_counts = df[label_col].value_counts()
print(f"Total unique labels: {len(label_counts)}")

print(f"\nLabel distribution (top 10):")
for i, (label, count) in enumerate(label_counts.head(10).items(), 1):
    percentage = (count / len(df)) * 100
    print(f"   {i:2d}. {str(label)[:40]:40s} {count:6d} samples ({percentage:.2f}%)")

# Check for BENIGN/NORMAL traffic
print(f"\n🔍 Identifying normal vs attack traffic...")
benign_keywords = ['BENIGN', 'Benign', 'benign', 'Normal', 'normal', 'legitimate', 'LEGITIMATE', '0']
attack_keywords = ['ATTACK', 'Attack', 'attack', 'Malicious', 'malicious', 'Malware', 'DDoS', 'PortScan', '1']

benign_labels = []
attack_labels = []

for label in label_counts.index:
    label_str = str(label)
    if any(keyword in label_str for keyword in benign_keywords):
        benign_labels.append(label)
    elif any(keyword in label_str for keyword in attack_keywords):
        attack_labels.append(label)

if benign_labels:
    print(f"✅ Benign/Normal labels found: {benign_labels[:5]}")  # Show first 5
if attack_labels:
    print(f"✅ Attack labels found: {attack_labels[:5]}")  # Show first 5

# =============================================================================
# 3. DATA PREPROCESSING
# =============================================================================
print("\n3. DATA PREPROCESSING")
print("-" * 40)

# Create target variable - binary classification (Benign vs Attack)
print(f"\n🎯 Creating target variable...")

if df[label_col].dtype == 'object' or len(label_counts) > 10:
    # String labels or many classes - convert to binary
    print(f"   Converting to binary classification: Benign (0) vs Attack (1)")
    
    def is_benign(label):
        label_str = str(label)
        # Check if it's a benign label
        if any(keyword in label_str for keyword in benign_keywords):
            return True
        # Also check if it's numeric 0
        try:
            if float(label_str) == 0:
                return True
        except:
            pass
        return False
    
    y = df[label_col].apply(lambda x: 0 if is_benign(x) else 1).astype(int)
    print(f"✅ Binary target created: 0=Benign, 1=Attack")
else:
    # Already numeric labels
    print(f"   Using existing numeric labels")
    y = df[label_col].astype(int)
    
    # If values are not 0/1, convert (assuming 0=benign, non-zero=attack)
    unique_vals = np.unique(y)
    if not (set(unique_vals).issubset({0, 1})):
        print(f"   Converting to binary: 0=Benign, non-zero=Attack")
        y = (y != 0).astype(int)

print(f"\n📊 Class Distribution:")
print(f"   Benign samples (0): {sum(y == 0):,} ({sum(y == 0)/len(y)*100:.2f}%)")
print(f"   Attack samples  (1): {sum(y == 1):,} ({sum(y == 1)/len(y)*100:.2f}%)")

# Select features
print(f"\n🔧 Selecting features...")
# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove label columns and other non-feature columns
exclude_cols = [label_col, 'Label', 'label', 'Class', 'class', 
                'Timestamp', 'timestamp', 'Time', 'time',
                'Flow ID', 'FlowID', 'flow_id',
                'Src IP', 'Source IP', 'SourceIP', 'src_ip',
                'Dst IP', 'Destination IP', 'DestinationIP', 'dst_ip',
                'Protocol', 'protocol']

feature_cols = [col for col in numeric_cols if col not in exclude_cols]

print(f"   Found {len(feature_cols)} numeric features")

# CICIDS2017 has many features - select important ones
print(f"   Selecting top 10 features for faster processing...")

# Strategy: Select features with good variance and low missing values
selected_features = []

# First, check for missing values
missing_counts = df[feature_cols].isnull().sum()
low_missing_features = missing_counts[missing_counts == 0].index.tolist()

if len(low_missing_features) >= 10:
    # Use features with no missing values
    candidate_features = low_missing_features
else:
    # Use all features
    candidate_features = feature_cols

# Select top features by variance
if len(candidate_features) >= 10:
    variances = df[candidate_features].var()
    selected_features = variances.nlargest(10).index.tolist()
else:
    selected_features = candidate_features[:min(10, len(candidate_features))]

print(f"✅ Selected {len(selected_features)} features:")
for i, feat in enumerate(selected_features, 1):
    print(f"   {i:2d}. {feat}")

# Create feature matrix
X = df[selected_features]

# Handle missing values
print(f"\n🧹 Cleaning data...")
# Replace infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# Check for missing values
missing_count = X.isnull().sum().sum()
if missing_count > 0:
    print(f"   Found {missing_count} missing/infinite values")
    # Fill with column median
    X = X.fillna(X.median())
    print(f"✅ Filled missing values with median")

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

# Class distribution
print(f"\n📈 Class distribution:")
print(f"   Training - Benign: {sum(y_train == 0):,}, Attack: {sum(y_train == 1):,}")
print(f"   Testing  - Benign: {sum(y_test == 0):,}, Attack: {sum(y_test == 1):,}")

# =============================================================================
# 5. DECISION TREE MODEL TRAINING
# =============================================================================
print("\n5. DECISION TREE MODEL TRAINING")
print("-" * 40)

print("🤖 Training Decision Tree Classifier...")

# Create and train the model
dt_model = DecisionTreeClassifier(
    max_depth=5,           # Reasonable depth for interpretability
    min_samples_split=20,  # Minimum samples to split a node
    min_samples_leaf=10,   # Minimum samples in a leaf node
    random_state=42,       # For reproducibility
    class_weight='balanced' # Handle class imbalance
)

# Train the model
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

# Detailed classification report
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack'], zero_division=0))

# Cross-validation score
print(f"\n🔍 Cross-Validation (5-fold)...")
cv_scores = cross_val_score(dt_model, X_scaled, y, cv=5, scoring='accuracy')
print(f"   CV Scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"   Mean CV Score: {cv_scores.mean():.4f}")
print(f"   Std CV Score:  {cv_scores.std():.4f}")

# =============================================================================
# 7. FEATURE IMPORTANCE ANALYSIS
# =============================================================================
print("\n7. FEATURE IMPORTANCE ANALYSIS")
print("-" * 40)

# Get feature importance
feature_importance = dt_model.feature_importances_

# Create importance dataframe
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

# =============================================================================
# MODIFIED: Save to plots/ directory
# =============================================================================
import os

# Ensure plots directory exists
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

_plot_counters = {}

def save_figure_to_plots(prefix):
    """Save figure to plots directory"""
    cnt = _plot_counters.get(prefix, 0) + 1
    _plot_counters[prefix] = cnt
    filename = f"{prefix}_fig{cnt}.png"
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    print(f"   Saved to: {filepath}")
    plt.close()

# Visualization 1: Decision Tree Structure
print("\n📈 Visualization 1: Decision Tree Diagram")
plt.figure(figsize=(20, 12))
plot_tree(dt_model,
          feature_names=selected_features,
          class_names=['Benign', 'Attack'],
          filled=True,
          rounded=True,
          max_depth=3,  # Show first 3 levels for clarity
          fontsize=9,
          proportion=True)
plt.title('CICIDS2017 Decision Tree Model (First 3 Levels)', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
save_figure_to_plots('cicids2017_decision_tree')

# Visualization 2: Feature Importance Bar Chart
print("📈 Visualization 2: Feature Importance")
plt.figure(figsize=(12, 6))
top_features = importance_df.head(10)

# Create gradient colors
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_features)))

bars = plt.barh(range(len(top_features)), top_features['Importance'], color=colors, height=0.7)
plt.yticks(range(len(top_features)), top_features['Feature'], fontsize=10)
plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.title('Top 10 Feature Importance - CICIDS2017 Dataset', 
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()  # Most important at top

# Add value labels
for i, (bar, imp) in enumerate(zip(bars, top_features['Importance'])):
    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f'{imp:.3f}', va='center', fontsize=10, fontweight='bold')

plt.grid(True, alpha=0.3, axis='x', linestyle='--')
plt.tight_layout()
save_figure_to_plots('cicids2017_feature_importance')

# Visualization 3: Confusion Matrix
print("📈 Visualization 3: Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=['Benign', 'Attack'],
            yticklabels=['Benign', 'Attack'],
            annot_kws={"size": 12})
plt.title('Confusion Matrix - CICIDS2017 Decision Tree', 
          fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=13, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
plt.tight_layout()
save_figure_to_plots('cicids2017_confusion_matrix')

# Visualization 4: Performance Metrics
print("📈 Visualization 4: Performance Metrics")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Accuracy
axes[0].bar(['Accuracy'], [accuracy], color='green', alpha=0.8, width=0.5)
axes[0].set_ylim(0, 1.0)
axes[0].set_title('Accuracy', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Score')
axes
