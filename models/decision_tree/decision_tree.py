# =============================================================================
# DECISION TREE ML MODEL FOR UNSW-NB15 CYBERSECURITY DATASET
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DECISION TREE ML MODEL - UNSW-NB15 DATASET")
print("=" * 70)

# =============================================================================
# SETUP FOR SAVING PLOTS
# =============================================================================
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

# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("\n1. LOADING UNSW-NB15 DATASET")
print("-" * 40)

# Path to your dataset
DATASET_PATH = "/home/darine/cybersecurity_assignment/datasets/unsw_nb15"

def find_unsw_file():
    """Find and load the UNSW-NB15 file"""
    possible_files = [
        "UNSW_NB15_training-set.csv",  # Training set
        "UNSW_NB15_testing-set.csv",   # Testing set
        "UNSW-NB15_1.csv",             # Part 1
        "UNSW-NB15_2.csv",             # Part 2
        "UNSW-NB15_3.csv",             # Part 3
        "UNSW-NB15_4.csv"              # Part 4
    ]
    
    for filename in possible_files:
        file_path = os.path.join(DATASET_PATH, filename)
        if os.path.exists(file_path):
            print(f"✅ Found: {filename}")
            return file_path
    
    print("❌ No UNSW-NB15 files found!")
    return None

# Find and load the file
file_path = find_unsw_file()
if file_path is None:
    exit()

print(f"\n📂 Loading: {os.path.basename(file_path)}")

# Load data (first 10000 rows for speed)
df = pd.read_csv(file_path, nrows=10000)
print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

# =============================================================================
# 2. DATA EXPLORATION
# =============================================================================
print("\n2. DATA EXPLORATION")
print("-" * 40)

# Show basic info
print(f"\n📊 Dataset Information:")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")

# Display column information
print(f"\n📝 First 10 columns:")
for i, col in enumerate(df.columns[:10], 1):
    dtype = df[col].dtype
    unique = df[col].nunique()
    print(f"   {i:2d}. {col:30s} {str(dtype):10s} {unique:6d} unique")

if len(df.columns) > 10:
    print(f"   ... and {len(df.columns) - 10} more columns")

# Find label column
print(f"\n🔍 Searching for label column...")
label_col = None
for col in ['label', 'Label', 'attack_cat', 'Class']:
    if col in df.columns:
        label_col = col
        print(f"✅ Found label column: '{label_col}'")
        break

if label_col is None:
    print("❌ No label column found!")
    print("Available columns:", df.columns.tolist())
    exit()

# Display label distribution
print(f"\n🎯 Label Analysis:")
if label_col == 'attack_cat':
    print("Attack categories found:")
    attack_counts = df[label_col].value_counts()
    print(attack_counts)
    
    # Show top 5 attack types
    print(f"\nTop 5 attack types:")
    for i, (attack_type, count) in enumerate(attack_counts.head().items(), 1):
        print(f"   {i}. {attack_type}: {count} samples")
else:
    print(f"Binary label distribution:")
    print(df[label_col].value_counts())

# =============================================================================
# 3. DATA PREPROCESSING
# =============================================================================
print("\n3. DATA PREPROCESSING")
print("-" * 40)

# Create target variable
print(f"\n🎯 Creating target variable...")
if label_col == 'attack_cat':
    # Binary classification: Normal vs Attack
    y = (df[label_col] != 'Normal').astype(int)
    print(f"✅ Binary target created: 0=Normal, 1=Attack")
else:
    # Already binary
    y = df[label_col].astype(int)
    print(f"✅ Using existing binary labels")

print(f"   Normal samples (0): {sum(y == 0)}")
print(f"   Attack samples (1): {sum(y == 1)}")
print(f"   Attack percentage: {sum(y == 1)/len(y)*100:.2f}%")

# Select features
print(f"\n🔧 Selecting features...")
# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove label columns
exclude_cols = [label_col, 'attack_cat', 'Label', 'Class', 'id', 'ID']
feature_cols = [col for col in numeric_cols if col not in exclude_cols]

print(f"   Found {len(feature_cols)} numeric features")

# Select top 10 features based on variance
if len(feature_cols) > 10:
    print(f"   Selecting top 10 features by variance...")
    variances = df[feature_cols].var().sort_values(ascending=False)
    selected_features = variances.head(10).index.tolist()
else:
    selected_features = feature_cols

print(f"✅ Selected {len(selected_features)} features:")
for i, feat in enumerate(selected_features, 1):
    print(f"   {i:2d}. {feat}")

# Create feature matrix
X = df[selected_features]

# Handle missing values
if X.isnull().any().any():
    print(f"\n🧹 Handling missing values...")
    missing_count = X.isnull().sum().sum()
    print(f"   Found {missing_count} missing values")
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
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set:     {X_test.shape[0]} samples")
print(f"   Features:     {X_train.shape[1]}")

# Class distribution
print(f"\n📈 Class distribution:")
print(f"   Training - Normal: {sum(y_train == 0)}, Attack: {sum(y_train == 1)}")
print(f"   Testing  - Normal: {sum(y_test == 0)}, Attack: {sum(y_test == 1)}")

# =============================================================================
# 5. DECISION TREE MODEL TRAINING
# =============================================================================
print("\n5. DECISION TREE MODEL TRAINING")
print("-" * 40)

print("🤖 Training Decision Tree Classifier...")

# Create and train the model
dt_model = DecisionTreeClassifier(
    max_depth=5,           # Limit tree depth to prevent overfitting
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
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"📈 Performance Metrics:")
print(f"   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")

# ROC-AUC if we have probability predictions
if len(np.unique(y_test)) > 1:
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"   ROC-AUC:   {roc_auc:.4f}")
    except:
        pass

# Detailed classification report
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

# Cross-validation score
print(f"\n🔍 Cross-Validation (5-fold)...")
cv_scores = cross_val_score(dt_model, X_scaled, y, cv=5)
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
# 8. VISUALIZATIONS (SAVED TO PLOTS/ DIRECTORY)
# =============================================================================
print("\n8. VISUALIZATIONS")
print("-" * 40)
print("📊 Generating visualizations...")

# Visualization 1: Decision Tree Structure
print("\n📈 Visualization 1: Decision Tree Diagram")
plt.figure(figsize=(20, 12))
plot_tree(dt_model,
          feature_names=selected_features,
          class_names=['Normal', 'Attack'],
          filled=True,
          rounded=True,
          max_depth=3,  # Show only first 3 levels for clarity
          fontsize=10,
          proportion=True)
plt.title('UNSW-NB15 Decision Tree Model (First 3 Levels)', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
save_figure_to_plots('unsw_decision_tree')

# Visualization 2: Feature Importance Bar Chart
print("📈 Visualization 2: Feature Importance")
plt.figure(figsize=(12, 6))
top_features = importance_df.head(10)

# Create gradient colors
colors = plt.cm.coolwarm(np.linspace(0.3, 0.8, len(top_features)))

bars = plt.barh(range(len(top_features)), top_features['Importance'], color=colors)
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance Score', fontsize=12)
plt.title('Top 10 Most Important Features - UNSW-NB15', 
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()  # Most important at top

# Add value labels
for i, (bar, imp) in enumerate(zip(bars, top_features['Importance'])):
    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f'{imp:.3f}', va='center', fontsize=10)

plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
save_figure_to_plots('unsw_feature_importance')

# Visualization 3: Confusion Matrix
print("📈 Visualization 3: Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
plt.title('Confusion Matrix - UNSW-NB15 Decision Tree', 
          fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
save_figure_to_plots('unsw_confusion_matrix')

# Visualization 4: Performance Metrics Comparison
print("📈 Visualization 4: Performance Metrics")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Subplot 1: Accuracy
axes[0, 0].bar(['Accuracy'], [accuracy], color='green', alpha=0.7, width=0.6)
axes[0, 0].set_ylim(0, 1.0)
axes[0, 0].set_title('Accuracy', fontweight='bold')
axes[0, 0].set_ylabel('Score')
axes[0, 0].text(0, accuracy + 0.02, f'{accuracy:.4f}', 
               ha='center', fontweight='bold', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)

# Subplot 2: Precision-Recall
metrics_names = ['Precision', 'Recall']
metrics_values = [precision, recall]
colors_pr = ['blue', 'orange']

axes[0, 1].bar(metrics_names, metrics_values, color=colors_pr, alpha=0.7, width=0.6)
axes[0, 1].set_ylim(0, 1.0)
axes[0, 1].set_title('Precision & Recall', fontweight='bold')
axes[0, 1].set_ylabel('Score')
for i, (name, value) in enumerate(zip(metrics_names, metrics_values)):
    axes[0, 1].text(i, value + 0.02, f'{value:.4f}', 
                   ha='center', fontweight='bold', fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# Subplot 3: F1-Score
axes[1, 0].bar(['F1-Score'], [f1], color='purple', alpha=0.7, width=0.6)
axes[1, 0].set_ylim(0, 1.0)
axes[1, 0].set_title('F1-Score', fontweight='bold')
axes[1, 0].set_ylabel('Score')
axes[1, 0].text(0, f1 + 0.02, f'{f1:.4f}', 
               ha='center', fontweight='bold', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

# Subplot 4: Class Distribution
axes[1, 1].pie([sum(y == 0), sum(y == 1)], 
               labels=['Normal', 'Attack'], 
               colors=['lightblue', 'lightcoral'],
               autopct='%1.1f%%',
               startangle=90)
axes[1, 1].set_title('Dataset Class Distribution', fontweight='bold')

plt.suptitle('UNSW-NB15 Decision Tree Model Performance', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
save_figure_to_plots('unsw_performance_metrics')

# Visualization 5: Model Summary
print("📈 Visualization 5: Model Summary")
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

summary_text = "UNSW-NB15 DECISION TREE MODEL SUMMARY\n"
summary_text += "=" * 40 + "\n\n"
summary_text += f"📊 Dataset: {os.path.basename(file_path)}\n"
summary_text += f"📈 Samples: {len(df):,}\n"
summary_text += f"🎯 Classes: Normal vs Attack\n"
summary_text += f"🔧 Features: {len(selected_features)}\n\n"
summary_text += "🏆 PERFORMANCE METRICS:\n"
summary_text += f"   • Accuracy:  {accuracy:.4f}\n"
summary_text += f"   • Precision: {precision:.4f}\n"
summary_text += f"   • Recall:    {recall:.4f}\n"
summary_text += f"   • F1-Score:  {f1:.4f}\n\n"
summary_text += "🌳 TREE CHARACTERISTICS:\n"
summary_text += f"   • Max Depth: {dt_model.get_depth()}\n"
summary_text += f"   • Leaves:    {dt_model.get_n_leaves()}\n"
summary_text += f"   • CV Score:  {cv_scores.mean():.4f}\n\n"
summary_text += "✅ Model successfully trained and evaluated!"

ax.text(0.1, 0.5, summary_text, 
        fontsize=12, 
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.title('Final Model Summary', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
save_figure_to_plots('unsw_model_summary')

# =============================================================================
# 9. FINAL SUMMARY (NO RESULTS SAVED)
# =============================================================================
print("\n" + "=" * 70)
print("🎉 UNSW-NB15 DECISION TREE MODEL - COMPLETE!")
print("=" * 70)

print(f"\n📋 FINAL SUMMARY:")
print(f"   Dataset: {os.path.basename(file_path)}")
print(f"   Total samples: {len(df):,}")
print(f"   Features used: {len(selected_features)}")
print(f"   Model accuracy: {accuracy:.4f}")
print(f"   Attack detection rate: {recall:.4f}")
print(f"   Model depth: {dt_model.get_depth()} levels")

print(f"\n✅ Decision Tree ML model successfully trained on UNSW-NB15 dataset!")
print(f"📊 5 visualizations saved to plots/ directory")

print("\n" + "=" * 70)