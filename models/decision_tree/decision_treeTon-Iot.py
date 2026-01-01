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
print("\n1. LOADING TON-IoT DATASET")
print("-" * 40)

# Use the SAME file as working code
dataset_path = "/home/darine/cybersecurity_assignment/datasets/ton_iot/Extra-Column-removed-TonIoT.csv"

print(f"📂 Loading: {os.path.basename(dataset_path)}")
print(f"   File size: {os.path.getsize(dataset_path) / (1024**3):.2f} GB")

# Load 500K rows to ensure both normal and attack samples
print("   Loading 500,000 rows to ensure both classes...")
df_large = pd.read_csv(dataset_path, nrows=500000, low_memory=False)

# Check label distribution
if 'label' in df_large.columns:
    label_counts = df_large['label'].value_counts()
    print(f"\n🔍 Label distribution in 500K sample:")
    for label, count in label_counts.items():
        percentage = (count / len(df_large)) * 100
        print(f"   Label {label}: {count:,} rows ({percentage:.1f}%)")
else:
    print("❌ ERROR: 'label' column not found!")
    print("Available columns:", df_large.columns.tolist())
    exit()

# Create balanced sample like the working code
if len(label_counts) > 1:
    # Sample equal number from each class (like isolation forest does)
    sample_per_class = min(50000, label_counts.min())
    
    print(f"\n⚖️  Creating balanced dataset...")
    dfs = []
    for label in label_counts.index:
        class_df = df_large[df_large['label'] == label]
        if len(class_df) > sample_per_class:
            dfs.append(class_df.sample(sample_per_class, random_state=42))
            print(f"   Label {label}: Sampled {sample_per_class:,} from {len(class_df):,}")
        else:
            dfs.append(class_df)
            print(f"   Label {label}: Using all {len(class_df):,}")
    
    df = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42)
    print(f"✅ Created balanced dataset: {len(df):,} rows total")
else:
    print("❌ Only one label found in 500K rows!")
    print("   Try loading more rows or check the file.")
    exit()

# =============================================================================
# 2. DATA EXPLORATION AND CLEANING
# =============================================================================
print("\n2. DATA EXPLORATION AND CLEANING")
print("-" * 40)

print(f"\n📊 Dataset Information:")
print(f"   File: {os.path.basename(dataset_path)}")
print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Clean column names (same as working code)
print(f"\n🔧 Cleaning column names...")
df.columns = [col.strip().replace(' ', '_').replace('-', '_').replace('.', '_').lower() 
             for col in df.columns]
print(f"✅ Column names cleaned")

# Show columns
print(f"\n📋 Dataset columns ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    dtype = df[col].dtype
    unique = df[col].nunique()
    print(f"   {i:2d}. {col:25s} {str(dtype):10s} {unique:6d} unique")

# =============================================================================
# 3. PREPARE LABELS
# =============================================================================
print("\n3. PREPARING LABELS")
print("-" * 40)

# Use 'label' column directly (0=normal, 1=attack)
label_col = 'label'

print(f"✅ Using label column: '{label_col}'")
print(f"   Data type: {df[label_col].dtype}")
print(f"   Unique values: {df[label_col].unique()}")

# Convert to binary (same logic as working code)
y = df[label_col].astype(int)

# Verify labels match TON-IoT convention
if 'type' in df.columns:
    print(f"\n🔍 Verifying label meanings with 'type' column:")
    for label_val in [0, 1]:
        types = df[df[label_col] == label_val]['type'].unique()[:3]
        print(f"   Label {label_val} -> Types: {list(types)}")
    
    # Check if we need to swap (like isolation forest does)
    normal_in_0 = df[df[label_col] == 0]['type'].astype(str).str.lower().str.contains('normal|benign').any()
    normal_in_1 = df[df[label_col] == 1]['type'].astype(str).str.lower().str.contains('normal|benign').any()
    
    if normal_in_1 and not normal_in_0:
        print(f"⚠️  Swapping labels (1=normal → 0, 0=attack → 1)")
        y = y.map({0: 1, 1: 0})  # Swap

# Show final distribution
normal_count = sum(y == 0)
attack_count = sum(y == 1)
print(f"\n📊 FINAL CLASS DISTRIBUTION:")
print(f"   Normal/Benign (0): {normal_count:,} samples ({normal_count/len(y)*100:.2f}%)")
print(f"   Attack (1):        {attack_count:,} samples ({attack_count/len(y)*100:.2f}%)")

if normal_count == 0 or attack_count == 0:
    print("\n❌ ERROR: Missing one class!")
    print("   Check your data loading or try different sampling.")
    exit()

# =============================================================================
# 4. PREPARE FEATURES
# =============================================================================
print("\n4. PREPARING FEATURES")
print("-" * 40)

# Drop non-numeric columns (same as isolation forest)
print(f"   Dropping non-numeric columns...")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove label column from features
if label_col in numeric_cols:
    numeric_cols.remove(label_col)

print(f"   Found {len(numeric_cols)} numeric features")

# Handle src_bytes conversion (same as isolation forest)
if 'src_bytes' in df.columns and df['src_bytes'].dtype == 'object':
    print(f"   Converting 'src_bytes' to numeric...")
    df['src_bytes'] = pd.to_numeric(df['src_bytes'], errors='coerce')
    nan_count = df['src_bytes'].isna().sum()
    if nan_count > 0:
        print(f"   Filled {nan_count} NaN values with median")
        df['src_bytes'].fillna(df['src_bytes'].median(), inplace=True)

# Select features (same as isolation forest)
feature_names = numeric_cols
X = df[feature_names].values

print(f"✅ Feature matrix shape: {X.shape}")
print(f"   Features used: {len(feature_names)}")

# Feature scaling
print(f"\n⚖️  Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"✅ Features scaled")

# =============================================================================
# 5. DATA SPLITTING
# =============================================================================
print("\n5. DATA SPLITTING")
print("-" * 40)

# Ensure we have both classes
unique_classes = np.unique(y)
if len(unique_classes) < 2:
    print(f"❌ ERROR: Only one class found: {unique_classes}")
    print("   Cannot split data for binary classification.")
    exit()

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"✅ Data split complete:")
print(f"   Training set: {X_train.shape[0]:,} samples")
print(f"   Test set:     {X_test.shape[0]:,} samples")
print(f"   Features:     {X_train.shape[1]}")

print(f"\n📈 Class distribution in split:")
print(f"   Training - Normal: {sum(y_train == 0):,} ({sum(y_train == 0)/len(y_train)*100:.1f}%)")
print(f"   Training - Attack: {sum(y_train == 1):,} ({sum(y_train == 1)/len(y_train)*100:.1f}%)")
print(f"   Testing  - Normal: {sum(y_test == 0):,} ({sum(y_test == 0)/len(y_test)*100:.1f}%)")
print(f"   Testing  - Attack: {sum(y_test == 1):,} ({sum(y_test == 1)/len(y_test)*100:.1f}%)")

# =============================================================================
# 6. DECISION TREE MODEL TRAINING
# =============================================================================
print("\n6. DECISION TREE MODEL TRAINING")
print("-" * 40)

print("🤖 Training Decision Tree Classifier...")

# Check if we need to adjust class weights
class_ratio = sum(y_train == 0) / max(sum(y_train == 1), 1)
if class_ratio > 10 or class_ratio < 0.1:
    print(f"   ⚠️  Class imbalance detected (ratio: {class_ratio:.2f}:1)")
    print(f"   Using balanced class weights")

# Create and train the model
dt_model = DecisionTreeClassifier(
    max_depth=7,           # Slightly deeper for complex patterns
    min_samples_split=10,  # More flexible splitting
    min_samples_leaf=5,    # Smaller leaves for rare attacks
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
# 7. MODEL EVALUATION
# =============================================================================
print("\n7. MODEL EVALUATION")
print("-" * 40)

# Make predictions
y_pred = dt_model.predict(X_test)

# Get probabilities if available
try:
    y_pred_proba = dt_model.predict_proba(X_test)
    if y_pred_proba.shape[1] > 1:
        y_pred_proba_1 = y_pred_proba[:, 1]
    else:
        y_pred_proba_1 = y_pred_proba[:, 0]
    proba_available = True
except:
    proba_available = False
    y_pred_proba_1 = None

# Calculate metrics
try:
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
    if proba_available and len(np.unique(y_test)) > 1:
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba_1)
            print(f"   ROC-AUC:   {roc_auc:.4f}")
        except:
            print(f"   ROC-AUC:   N/A")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack'], zero_division=0))
    
except Exception as e:
    print(f"❌ Error calculating metrics: {str(e)}")

# Cross-validation
print(f"\n🔍 Cross-Validation (5-fold)...")
try:
    cv_scores = cross_val_score(dt_model, X_scaled, y, cv=5, scoring='accuracy')
    print(f"   CV Scores: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"   Mean CV Score: {cv_scores.mean():.4f}")
    print(f"   Std CV Score:  {cv_scores.std():.4f}")
except Exception as e:
    print(f"   Cross-validation failed: {str(e)}")
    cv_scores = None

# =============================================================================
# 8. FEATURE IMPORTANCE
# =============================================================================
print("\n8. FEATURE IMPORTANCE ANALYSIS")
print("-" * 40)

try:
    feature_importance = dt_model.feature_importances_
    
    # Use feature_names instead of selected_features
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print(f"🏆 Feature Importance Ranking:")
    print(importance_df.to_string(index=False))
    
    # Check if features are actually useful
    if importance_df['Importance'].max() < 0.1:
        print(f"\n⚠️  WARNING: No strong feature importance detected")
        print(f"   Most important feature: {importance_df.iloc[0]['Feature']} ({importance_df.iloc[0]['Importance']:.4f})")
    
except Exception as e:
    print(f"⚠️  Could not calculate feature importance: {str(e)}")

# =============================================================================
# 9. VISUALIZATIONS (SAVED TO PLOTS/ DIRECTORY)
# =============================================================================
print("\n9. VISUALIZATIONS")
print("-" * 40)
print("📊 Generating visualizations...")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# 1. Decision Tree
if dt_model.get_depth() > 0:
    print("\n📈 Visualization 1: Decision Tree")
    plt.figure(figsize=(18, 10))
    plot_tree(dt_model,
              feature_names=feature_names,
              class_names=['Normal', 'Attack'],
              filled=True,
              rounded=True,
              max_depth=min(3, dt_model.get_depth()),
              fontsize=9,
              proportion=True)
    plt.title(f'TON-IoT Decision Tree (First {min(3, dt_model.get_depth())} levels)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    save_figure_to_plots('toniot_decision_tree')
else:
    print("⚠️  Skipping tree visualization (tree depth is 0)")

# 2. Feature Importance
try:
    if 'importance_df' in locals() and not importance_df.empty:
        print("📈 Visualization 2: Feature Importance")
        plt.figure(figsize=(12, 6))
        top_features = importance_df.head(10)
        
        bars = plt.barh(range(len(top_features)), top_features['Importance'], 
                       color=plt.cm.Blues(np.linspace(0.4, 0.9, len(top_features))))
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.title('Top Features Importance - TON-IoT', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        for i, (bar, imp) in enumerate(zip(bars, top_features['Importance'])):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{imp:.3f}', va='center', fontsize=10, fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        save_figure_to_plots('toniot_feature_importance')
except Exception as e:
    print(f"⚠️  Skipping feature importance visualization: {str(e)}")

# 3. Confusion Matrix
print("📈 Visualization 3: Confusion Matrix")
try:
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title('Confusion Matrix - TON-IoT', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    save_figure_to_plots('toniot_confusion_matrix')
except Exception as e:
    print(f"⚠️  Could not create confusion matrix: {str(e)}")

# 4. Performance Metrics
print("📈 Visualization 4: Performance Metrics")
try:
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
    save_figure_to_plots('toniot_performance_metrics')
except Exception as e:
    print(f"⚠️  Could not create metrics visualization: {str(e)}")

# 5. Dataset Summary
print("📈 Visualization 5: Dataset Summary")
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Class distribution pie chart
    class_counts = [normal_count, attack_count]
    colors = ['lightblue', 'lightcoral']
    ax1.pie(class_counts, labels=['Normal', 'Attack'], colors=colors,
            autopct='%1.1f%%', startangle=90, explode=(0.05, 0.05))
    ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
    
    # Dataset info text
    info_text = f"TON-IoT Dataset Summary\n{'='*25}\n\n"
    info_text += f"File: {os.path.basename(dataset_path)}\n"
    info_text += f"Samples: {len(df):,}\n"
    info_text += f"Features used: {len(feature_names)}\n"
    info_text += f"Normal: {normal_count:,}\n"
    info_text += f"Attack: {attack_count:,}\n"
    info_text += f"Model Depth: {dt_model.get_depth()}\n"
    
    if 'accuracy' in locals():
        info_text += f"Accuracy: {accuracy:.3f}\n"
    if 'recall' in locals():
        info_text += f"Attack Recall: {recall:.3f}\n"
    if cv_scores is not None and len(cv_scores) > 0:
        info_text += f"CV Score: {cv_scores.mean():.3f}"
    
    ax2.text(0.1, 0.5, info_text, fontsize=12, 
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax2.axis('off')
    
    plt.suptitle('TON-IoT Cybersecurity Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure_to_plots('toniot_dataset_summary')
except Exception as e:
    print(f"⚠️  Could not create summary visualization: {str(e)}")

# =============================================================================
# 10. FINAL SUMMARY (NO RESULTS SAVED)
# =============================================================================
print("\n" + "=" * 70)
print("🎉 TON-IoT DECISION TREE ANALYSIS - COMPLETE!")
print("=" * 70)

print(f"\n📋 SUMMARY:")
print(f"   Dataset: {os.path.basename(dataset_path)}")
print(f"   Samples: {len(df):,}")
print(f"   Normal: {normal_count:,} ({normal_count/len(df)*100:.1f}%)")
print(f"   Attack: {attack_count:,} ({attack_count/len(df)*100:.1f}%)")
print(f"   Label column used: '{label_col}'")
print(f"   Features used: {len(feature_names)}")

if 'accuracy' in locals():
    print(f"   Model accuracy: {accuracy:.4f}")
    print(f"   Attack detection (Recall): {recall:.4f}")

if cv_scores is not None and len(cv_scores) > 0:
    print(f"   Cross-validation: {cv_scores.mean():.4f}")

print(f"\n✅ Analysis completed successfully!")
print(f"📊 Generated 5 visualizations saved to plots/ directory")

print("\n" + "=" * 70)
