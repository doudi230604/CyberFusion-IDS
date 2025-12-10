import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ====================================================================
# STEP 1: LOAD AND PREPARE NORMAL DATA ONLY
# ====================================================================

def load_and_filter_normal_data():
    """
    Load UNSW-NB15 and filter only normal data for training
    """
    print("=" * 60)
    print("STEP 1: LOADING AND FILTERING NORMAL DATA")
    print("=" * 60)
    
    try:
        # Try to load from CSV
        df = pd.read_csv('UNSW-NB15.csv')
    except FileNotFoundError:
        # Try alternative filenames
        try:
            df = pd.read_csv('UNSW_NB15.csv')
        except FileNotFoundError:
            try:
                df = pd.read_csv('UNSW-NB15_1.csv')
            except FileNotFoundError:
                print("Dataset not found. Creating synthetic data...")
                df = create_synthetic_data()
    
    print(f"Dataset shape: {df.shape}")
    
    # Check if we have labels
    if 'label' not in df.columns:
        print("Warning: No 'label' column found. Assuming all data is normal.")
        df_normal = df.copy()
        df_attack = pd.DataFrame()
    else:
        # Separate normal and attack data
        df_normal = df[df['label'] == 0].copy()
        df_attack = df[df['label'] == 1].copy()
        
        print(f"\nNormal data: {df_normal.shape[0]} samples")
        print(f"Attack data: {df_attack.shape[0]} samples")
        print(f"Normal percentage: {df_normal.shape[0]/df.shape[0]*100:.1f}%")
        
        if 'attack_cat' in df.columns:
            print("\nAttack categories in data:")
            print(df_attack['attack_cat'].value_counts())
    
    return df_normal, df_attack

def create_synthetic_data():
    """Create synthetic UNSW-NB15-like data"""
    np.random.seed(42)
    n_samples = 10000
    
    # Create 90% normal, 10% attack
    n_normal = int(n_samples * 0.9)
    n_attack = n_samples - n_normal
    
    # Normal data (baseline behavior)
    normal_data = {
        'dur': np.random.exponential(10, n_normal),
        'proto': np.random.choice(['tcp', 'udp'], n_normal, p=[0.8, 0.2]),
        'service': np.random.choice(['http', 'dns', 'smtp'], n_normal, p=[0.6, 0.3, 0.1]),
        'state': np.random.choice(['FIN', 'CON'], n_normal, p=[0.3, 0.7]),
        'spkts': np.random.poisson(50, n_normal),
        'dpkts': np.random.poisson(30, n_normal),
        'sbytes': np.random.poisson(500, n_normal),
        'dbytes': np.random.poisson(300, n_normal),
        'sttl': np.random.randint(100, 255, n_normal),
        'dttl': np.random.randint(100, 255, n_normal),
        'label': 0
    }
    
    # Attack data (anomalous behavior)
    attack_data = {
        'dur': np.concatenate([
            np.random.exponential(0.1, int(n_attack * 0.4)),  # Short duration attacks
            np.random.exponential(100, int(n_attack * 0.6))   # Long duration attacks
        ]),
        'proto': np.random.choice(['tcp', 'udp', 'icmp'], n_attack),
        'service': np.random.choice(['http', 'dns', 'smtp', 'ftp', 'ssh'], n_attack),
        'state': np.random.choice(['FIN', 'CON', 'INT', 'REQ', 'RST'], n_attack),
        'spkts': np.concatenate([
            np.random.poisson(1000, int(n_attack * 0.5)),  # High packet count
            np.random.poisson(5, int(n_attack * 0.5))      # Low packet count
        ]),
        'dpkts': np.concatenate([
            np.random.poisson(500, int(n_attack * 0.5)),
            np.random.poisson(2, int(n_attack * 0.5))
        ]),
        'sbytes': np.concatenate([
            np.random.poisson(10000, int(n_attack * 0.5)),  # High byte count
            np.random.poisson(50, int(n_attack * 0.5))      # Low byte count
        ]),
        'dbytes': np.concatenate([
            np.random.poisson(5000, int(n_attack * 0.5)),
            np.random.poisson(20, int(n_attack * 0.5))
        ]),
        'sttl': np.random.randint(32, 100, n_attack),  # Lower TTL
        'dttl': np.random.randint(32, 100, n_attack),
        'label': 1
    }
    
    # Combine and shuffle
    df_normal = pd.DataFrame(normal_data)
    df_attack = pd.DataFrame(attack_data)
    df = pd.concat([df_normal, df_attack], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

# ====================================================================
# STEP 2: PREPROCESS NORMAL DATA (WITH CORRELATION HANDLING)
# ====================================================================

def preprocess_normal_data(df_normal, drop_correlated=True):
    """
    Preprocess normal data only with correlation handling
    Returns: processed features and columns_to_drop
    """
    print("\n" + "=" * 60)
    print("STEP 2: PREPROCESSING NORMAL DATA")
    print("=" * 60)
    
    # Select features from normal data
    numeric_features = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss']
    
    # Check which features exist
    available_features = [f for f in numeric_features if f in df_normal.columns]
    print(f"Available numeric features: {available_features}")
    
    # Start with numeric features
    X_normal = df_normal[available_features].copy()
    
    # Handle missing values
    if X_normal.isna().any().any():
        print(f"Missing values found. Filling with median...")
        X_normal = X_normal.fillna(X_normal.median())
    
    # Create additional features (as per document)
    print("\nCreating derived features...")
    
    # Rate features
    if 'spkts' in X_normal.columns and 'dur' in X_normal.columns:
        X_normal['packets_per_second'] = X_normal['spkts'] / (X_normal['dur'].replace(0, 0.001) + 0.001)
    
    if 'sbytes' in X_normal.columns and 'dur' in X_normal.columns:
        X_normal['bytes_per_second'] = X_normal['sbytes'] / (X_normal['dur'].replace(0, 0.001) + 0.001)
    
    # Ratio features
    if 'sbytes' in X_normal.columns and 'spkts' in X_normal.columns:
        X_normal['packet_size_ratio'] = X_normal['sbytes'] / (X_normal['spkts'].replace(0, 1) + 1)
    
    if 'sttl' in X_normal.columns and 'dttl' in X_normal.columns:
        X_normal['ttl_difference'] = abs(X_normal['sttl'] - X_normal['dttl'])
    
    # Connection pattern features
    if 'dur' in X_normal.columns:
        X_normal['is_background_flow'] = (X_normal['dur'] < 1).astype(int)
    
    print(f"Total features after engineering: {X_normal.shape[1]}")
    
    # Remove highly correlated features (>95% correlation)
    columns_to_drop = []
    if drop_correlated and X_normal.shape[1] > 1:
        corr_matrix = X_normal.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find columns to drop (correlation > 0.95)
        to_drop = []
        for column in upper.columns:
            if any(upper[column] > 0.95):
                to_drop.append(column)
        
        if to_drop:
            print(f"\nDropping highly correlated features: {to_drop}")
            columns_to_drop = to_drop
            X_normal = X_normal.drop(columns=to_drop)
    
    print(f"Final normal data shape: {X_normal.shape}")
    
    return X_normal, columns_to_drop

# ====================================================================
# STEP 3: TRAIN ISOLATION FOREST ON NORMAL DATA
# ====================================================================

def train_on_normal_data(X_normal):
    """
    Train Isolation Forest exclusively on normal data
    """
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING ISOLATION FOREST ON NORMAL DATA ONLY")
    print("=" * 60)
    
    # Scale the normal data
    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    
    # Document-recommended parameters for training on normal data
    params = {
        'n_estimators': 150,
        'max_samples': min(256, len(X_normal)),
        'contamination': 0.01,  # Very low since we're training on normal data
        'max_features': 0.75,
        'bootstrap': False,
        'random_state': 42,
        'n_jobs': -1
    }
    
    print("Training parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    print(f"\nTraining on {X_normal_scaled.shape[0]} normal samples...")
    
    # Train Isolation Forest
    iso_forest = IsolationForest(**params)
    iso_forest.fit(X_normal_scaled)
    
    # Calculate scores on training (normal) data
    normal_scores = iso_forest.decision_function(X_normal_scaled)
    
    print("\nTraining complete!")
    print(f"Mean anomaly score for normal data: {normal_scores.mean():.4f}")
    print(f"Std anomaly score for normal data: {normal_scores.std():.4f}")
    
    return iso_forest, scaler, normal_scores, X_normal.columns.tolist()

# ====================================================================
# STEP 4: EVALUATE ON ATTACK DATA (IF AVAILABLE)
# ====================================================================

def prepare_attack_data(df_attack, feature_names, columns_to_drop):
    """
    Prepare attack data with the same features as training data
    """
    if df_attack.empty:
        return pd.DataFrame()
    
    print("\nPreparing attack data with same features as training...")
    
    # Start with basic numeric features
    X_attack = pd.DataFrame()
    
    # First, add all features that should be in the final set
    for feature in feature_names:
        if feature in df_attack.columns:
            X_attack[feature] = df_attack[feature]
        else:
            # If feature is derived, we need to create it
            if feature == 'packets_per_second' and 'spkts' in df_attack.columns and 'dur' in df_attack.columns:
                X_attack[feature] = df_attack['spkts'] / (df_attack['dur'].replace(0, 0.001) + 0.001)
            elif feature == 'bytes_per_second' and 'sbytes' in df_attack.columns and 'dur' in df_attack.columns:
                X_attack[feature] = df_attack['sbytes'] / (df_attack['dur'].replace(0, 0.001) + 0.001)
            elif feature == 'packet_size_ratio' and 'sbytes' in df_attack.columns and 'spkts' in df_attack.columns:
                X_attack[feature] = df_attack['sbytes'] / (df_attack['spkts'].replace(0, 1) + 1)
            elif feature == 'ttl_difference' and 'sttl' in df_attack.columns and 'dttl' in df_attack.columns:
                X_attack[feature] = abs(df_attack['sttl'] - df_attack['dttl'])
            elif feature == 'is_background_flow' and 'dur' in df_attack.columns:
                X_attack[feature] = (df_attack['dur'] < 1).astype(int)
            else:
                # For other features, fill with 0 (or appropriate default)
                X_attack[feature] = 0
    
    # Remove any columns that should be dropped (highly correlated)
    if columns_to_drop:
        X_attack = X_attack.drop(columns=[col for col in columns_to_drop if col in X_attack.columns])
    
    # Ensure the columns are in the same order as training data
    X_attack = X_attack[feature_names]
    
    # Handle missing values
    if X_attack.isna().any().any():
        X_attack = X_attack.fillna(0)
    
    print(f"Attack data prepared. Shape: {X_attack.shape}")
    print(f"Features: {X_attack.columns.tolist()}")
    
    return X_attack

def evaluate_on_attack_data(iso_forest, scaler, X_attack, feature_names):
    """
    Evaluate the model on attack data
    """
    if X_attack.empty:
        print("\nNo attack data available for evaluation.")
        return None, None, None
    
    print("\n" + "=" * 60)
    print("STEP 4: EVALUATING ON ATTACK DATA")
    print("=" * 60)
    
    # Scale attack data using the same scaler
    X_attack_scaled = scaler.transform(X_attack)
    
    # Get anomaly scores for attack data
    attack_scores = iso_forest.decision_function(X_attack_scaled)
    
    # Get predictions (-1 = anomaly, 1 = normal)
    attack_predictions = iso_forest.predict(X_attack_scaled)
    
    print(f"\nAttack data statistics:")
    print(f"Mean anomaly score: {attack_scores.mean():.4f}")
    print(f"Std anomaly score: {attack_scores.std():.4f}")
    print(f"Min anomaly score: {attack_scores.min():.4f}")
    print(f"Max anomaly score: {attack_scores.max():.4f}")
    
    # Calculate detection rate
    anomalies_detected = (attack_predictions == -1).sum()
    detection_rate = anomalies_detected / len(attack_predictions) * 100
    print(f"\nAnomalies detected: {anomalies_detected}/{len(attack_predictions)}")
    print(f"Detection rate: {detection_rate:.1f}%")
    
    return X_attack_scaled, attack_scores, attack_predictions

# ====================================================================
# STEP 5: THRESHOLD DETERMINATION & VISUALIZATION
# ====================================================================

def determine_threshold_and_visualize(normal_scores, attack_scores=None):
    """
    Determine optimal threshold and create visualizations
    """
    print("\n" + "=" * 60)
    print("STEP 5: THRESHOLD DETERMINATION & VISUALIZATION")
    print("=" * 60)
    
    # Method 1: Percentile-based threshold (from document)
    threshold_95 = np.percentile(normal_scores, 5)  # 5th percentile = top 5% anomalies
    threshold_99 = np.percentile(normal_scores, 1)  # 1st percentile = top 1% anomalies
    
    print("\nThreshold options (based on normal data distribution):")
    print(f"95th percentile threshold: {threshold_95:.4f}")
    print(f"99th percentile threshold: {threshold_99:.4f}")
    
    # Method 2: Statistical threshold
    mean_score = normal_scores.mean()
    std_score = normal_scores.std()
    threshold_2std = mean_score - 2 * std_score
    threshold_3std = mean_score - 3 * std_score
    
    print(f"\nStatistical thresholds:")
    print(f"Mean - 2*Std: {threshold_2std:.4f}")
    print(f"Mean - 3*Std: {threshold_3std:.4f}")
    
    # Recommended threshold (combining methods)
    recommended_threshold = min(threshold_95, threshold_2std)
    print(f"\nRecommended threshold: {recommended_threshold:.4f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribution of normal scores
    axes[0, 0].hist(normal_scores, bins=50, alpha=0.7, color='blue', edgecolor='black', 
                   label='Normal Data', density=True)
    axes[0, 0].axvline(x=recommended_threshold, color='red', linestyle='--', 
                      linewidth=2, label=f'Threshold: {recommended_threshold:.3f}')
    axes[0, 0].axvline(x=threshold_95, color='orange', linestyle=':', 
                      linewidth=1.5, label=f'95th: {threshold_95:.3f}')
    axes[0, 0].axvline(x=threshold_2std, color='green', linestyle=':', 
                      linewidth=1.5, label=f'Mean-2σ: {threshold_2std:.3f}')
    axes[0, 0].set_xlabel('Anomaly Score')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Distribution of Anomaly Scores (Normal Data)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Cumulative distribution
    sorted_normal = np.sort(normal_scores)
    cdf_normal = np.arange(1, len(sorted_normal) + 1) / len(sorted_normal)
    axes[0, 1].plot(sorted_normal, cdf_normal, 'b-', linewidth=2, label='Normal Data CDF')
    axes[0, 1].axvline(x=recommended_threshold, color='red', linestyle='--', linewidth=2)
    axes[0, 1].axhline(y=0.05, color='orange', linestyle=':', linewidth=1.5)
    axes[0, 1].set_xlabel('Anomaly Score')
    axes[0, 1].set_ylabel('Cumulative Probability')
    axes[0, 1].set_title('Cumulative Distribution Function')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    if attack_scores is not None:
        # 3. Comparison of normal vs attack scores
        axes[1, 0].hist(normal_scores, bins=50, alpha=0.5, color='blue', 
                       label='Normal', density=True)
        axes[1, 0].hist(attack_scores, bins=50, alpha=0.5, color='red', 
                       label='Attack', density=True)
        axes[1, 0].axvline(x=recommended_threshold, color='red', linestyle='--', 
                          linewidth=2, label=f'Threshold')
        axes[1, 0].set_xlabel('Anomaly Score')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Normal vs Attack Score Distributions')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Box plot comparison
        box_data = [normal_scores, attack_scores]
        axes[1, 1].boxplot(box_data, labels=['Normal', 'Attack'], patch_artist=True,
                          boxprops=dict(facecolor='lightblue', color='blue'),
                          medianprops=dict(color='red'))
        axes[1, 1].axhline(y=recommended_threshold, color='red', linestyle='--', 
                          linewidth=2, label=f'Threshold')
        axes[1, 1].set_ylabel('Anomaly Score')
        axes[1, 1].set_title('Box Plot: Normal vs Attack Scores')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return recommended_threshold

# ====================================================================
# STEP 6: ANOMALY DETECTION & ANALYSIS
# ====================================================================

def analyze_anomalies(iso_forest, scaler, X_normal, threshold, X_attack=None, feature_names=None):
    """
    Analyze detected anomalies and provide insights
    """
    print("\n" + "=" * 60)
    print("STEP 6: ANOMALY ANALYSIS")
    print("=" * 60)
    
    # Get anomaly scores for normal data
    X_normal_scaled = scaler.transform(X_normal)
    normal_scores = iso_forest.decision_function(X_normal_scaled)
    
    # Find anomalies in normal data (false positives)
    normal_anomalies_mask = normal_scores < threshold
    normal_anomalies_count = normal_anomalies_mask.sum()
    
    print(f"\nAnalysis of normal data:")
    print(f"Total normal samples: {len(normal_scores)}")
    print(f"Anomalies detected: {normal_anomalies_count}")
    print(f"False positive rate: {normal_anomalies_count/len(normal_scores)*100:.2f}%")
    
    if normal_anomalies_count > 0 and feature_names:
        print(f"\nCharacteristics of false positives (top 5 features):")
        false_positives = X_normal[normal_anomalies_mask]
        
        for i, feature in enumerate(feature_names[:5]):  # Show top 5 features
            if feature in false_positives.columns:
                fp_mean = false_positives[feature].mean()
                all_mean = X_normal[feature].mean()
                diff_pct = (fp_mean - all_mean) / all_mean * 100 if all_mean != 0 else 0
                print(f"  {i+1}. {feature}:")
                print(f"     FP mean={fp_mean:.2f}, All mean={all_mean:.2f} ({diff_pct:+.1f}%)")
    
    if X_attack is not None and not X_attack.empty:
        print(f"\n{'='*40}")
        print("Attack Data Analysis")
        print(f"{'='*40}")
        
        # Get scores for attack data
        X_attack_scaled = scaler.transform(X_attack)
        attack_scores = iso_forest.decision_function(X_attack_scaled)
        
        # Find detected attacks
        attack_anomalies_mask = attack_scores < threshold
        attack_anomalies_count = attack_anomalies_mask.sum()
        
        print(f"Total attack samples: {len(attack_scores)}")
        print(f"Attacks detected: {attack_anomalies_count}")
        print(f"Detection rate: {attack_anomalies_count/len(attack_scores)*100:.2f}%")
        
        if attack_anomalies_count > 0 and feature_names:
            print(f"\nCharacteristics of detected attacks (top 5 features):")
            detected_attacks = X_attack[attack_anomalies_mask]
            
            for i, feature in enumerate(feature_names[:5]):
                if feature in detected_attacks.columns:
                    attack_mean = detected_attacks[feature].mean()
                    normal_mean = X_normal[feature].mean()
                    diff_pct = (attack_mean - normal_mean) / normal_mean * 100 if normal_mean != 0 else 0
                    print(f"  {i+1}. {feature}:")
                    print(f"     Attack mean={attack_mean:.2f}, Normal mean={normal_mean:.2f} ({diff_pct:+.1f}%)")
    
    # Create summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    
    summary_data = []
    
    # Normal data stats
    summary_data.append({
        'Data Type': 'Normal (Training)',
        'Samples': len(normal_scores),
        'Mean Score': f"{normal_scores.mean():.4f}",
        'Std Score': f"{normal_scores.std():.4f}",
        'Anomalies': normal_anomalies_count,
        'Rate': f"{normal_anomalies_count/len(normal_scores)*100:.2f}%"
    })
    
    # Attack data stats (if available)
    if X_attack is not None and not X_attack.empty:
        summary_data.append({
            'Data Type': 'Attack (Test)',
            'Samples': len(attack_scores),
            'Mean Score': f"{attack_scores.mean():.4f}",
            'Std Score': f"{attack_scores.std():.4f}",
            'Anomalies': attack_anomalies_count,
            'Rate': f"{attack_anomalies_count/len(attack_scores)*100:.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

# ====================================================================
# MAIN EXECUTION
# ====================================================================

def main():
    """
    Main pipeline: Train Isolation Forest only on normal data
    """
    print("ISOLATION FOREST - TRAINING ON NORMAL DATA ONLY")
    print("=" * 60)
    
    # Step 1: Load and filter normal data
    df_normal, df_attack = load_and_filter_normal_data()
    
    # Step 2: Preprocess normal data (with correlation handling)
    X_normal, columns_to_drop = preprocess_normal_data(df_normal, drop_correlated=True)
    
    # Step 3: Train Isolation Forest on normal data only
    iso_forest, scaler, normal_scores, feature_names = train_on_normal_data(X_normal)
    
    # Step 4: Prepare and evaluate on attack data (if available)
    if not df_attack.empty:
        X_attack = prepare_attack_data(df_attack, feature_names, columns_to_drop)
        if not X_attack.empty:
            X_attack_scaled, attack_scores, attack_predictions = evaluate_on_attack_data(
                iso_forest, scaler, X_attack, feature_names
            )
        else:
            attack_scores = None
    else:
        attack_scores = None
        X_attack = pd.DataFrame()
    
    # Step 5: Determine threshold and visualize
    threshold = determine_threshold_and_visualize(normal_scores, attack_scores)
    
    # Step 6: Analyze anomalies
    analyze_anomalies(iso_forest, scaler, X_normal, threshold, X_attack, feature_names)
    
    # Final model information
    print(f"\n{'='*60}")
    print("MODEL INFORMATION")
    print(f"{'='*60}")
    print(f"Model: Isolation Forest (trained on normal data only)")
    print(f"Training samples: {X_normal.shape[0]}")
    print(f"Features used: {len(feature_names)}")
    print(f"Feature names: {feature_names}")
    if columns_to_drop:
        print(f"Features dropped due to correlation: {columns_to_drop}")
    print(f"Optimal threshold: {threshold:.4f}")
    print(f"Contamination parameter: {iso_forest.contamination}")
    
    # Save feature names for future use
    feature_info = {
        'feature_names': feature_names,
        'columns_dropped': columns_to_drop
    }
    
    return {
        'model': iso_forest,
        'scaler': scaler,
        'X_normal': X_normal,
        'normal_scores': normal_scores,
        'attack_scores': attack_scores,
        'threshold': threshold,
        'feature_names': feature_names,
        'feature_info': feature_info
    }

# ====================================================================
# UTILITY FUNCTION FOR NEW DATA PREDICTION
# ====================================================================

def predict_anomalies(new_data, model_info):
    """
    Predict anomalies on new data using the trained model
    """
    iso_forest = model_info['model']
    scaler = model_info['scaler']
    feature_names = model_info['feature_names']
    threshold = model_info['threshold']
    
    # Prepare new data with same features
    new_data_prepared = pd.DataFrame()
    
    for feature in feature_names:
        if feature in new_data.columns:
            new_data_prepared[feature] = new_data[feature]
        else:
            # Handle missing features
            new_data_prepared[feature] = 0
    
    # Scale the data
    new_data_scaled = scaler.transform(new_data_prepared)
    
    # Get anomaly scores
    scores = iso_forest.decision_function(new_data_scaled)
    
    # Make predictions based on threshold
    predictions = (scores < threshold).astype(int)
    
    return scores, predictions

# ====================================================================
# RUN THE PIPELINE
# ====================================================================

if __name__ == "__main__":
    # Run the complete pipeline
    try:
        results = main()
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        print("\nThe Isolation Forest has been trained exclusively on normal data.")
        print("This approach follows the document's recommendation for Phase 1 training.")
        print("\nModel is ready for deployment.")
        
        # Example of how to use the model on new data
        print("\n" + "=" * 60)
        print("EXAMPLE: PREDICTING ON NEW DATA")
        print("=" * 60)
        
        # Create some test data
        test_data = pd.DataFrame({
            'dur': [5, 0.5, 100, 1, 0.1],
            'spkts': [40, 1000, 10, 60, 2000],
            'dpkts': [20, 500, 5, 30, 1000],
            'sbytes': [400, 10000, 100, 600, 20000],
            'dbytes': [200, 5000, 50, 300, 10000],
            'sttl': [200, 50, 150, 180, 30],
            'dttl': [200, 50, 150, 180, 30]
        })
        
        print("\nTest data:")
        print(test_data)
        
        scores, predictions = predict_anomalies(test_data, results)
        
        print("\nPredictions:")
        for i in range(len(test_data)):
            status = "ANOMALY" if predictions[i] == 1 else "NORMAL"
            print(f"Sample {i+1}: Score={scores[i]:.4f}, Prediction={status}")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
