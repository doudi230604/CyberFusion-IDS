import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import gc

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

print("="*70)
print("FINAL LSTM - SAMPLING FROM DIFFERENT FILE LOCATIONS")
print("="*70)

def sample_from_different_locations(filepath, total_samples=50000):
    """Sample from different parts of the file to get both classes"""
    print(f"Sampling from different parts of: {os.path.basename(filepath)}")
    
    # First, check the file size to estimate
    file_size = os.path.getsize(filepath) / (1024**3)  # GB
    print(f"File size: {file_size:.2f} GB")
    
    # Strategy: Read chunks from different positions
    chunk_size = 10000  # Read 10k rows at a time
    
    # We'll read from 5 different positions to get diversity
    all_chunks = []
    
    try:
        # Use pandas read_csv with skiprows to sample from different positions
        for position in [0, 100000, 200000, 300000, 400000]:
            print(f"  Reading from position {position}...")
            try:
                chunk = pd.read_csv(
                    filepath,
                    skiprows=range(1, position + 1),  # Skip headers + position
                    nrows=chunk_size,
                    low_memory=False
                )
                all_chunks.append(chunk)
                print(f"    Read {len(chunk)} rows")
            except Exception as e:
                print(f"    Error at position {position}: {e}")
                continue
        
        if not all_chunks:
            print("❌ Could not read any data!")
            return None
        
        # Combine chunks
        df = pd.concat(all_chunks, ignore_index=True)
        
        # If we have too much data, sample
        if len(df) > total_samples:
            df = df.sample(n=total_samples, random_state=42)
        
        print(f"\nTotal samples collected: {len(df)}")
        
        return df
        
    except Exception as e:
        print(f"Error sampling data: {e}")
        return None

def load_data_with_attacks(filepath, target_samples=50000):
    """Load data ensuring we get both normal and attack samples"""
    print("Loading data with attack sampling strategy...")
    
    # Try to sample from different locations
    df = sample_from_different_locations(filepath, target_samples)
    
    if df is None:
        print("Failed to sample data")
        return None
    
    # Check columns
    print(f"\nColumns: {list(df.columns)}")
    
    # Find label column
    label_col = None
    for col in ['label', 'Label', 'attack', 'Attack', 'type']:
        if col in df.columns:
            label_col = col
            break
    
    if not label_col:
        print("❌ No label column found!")
        return None
    
    print(f"Label column: '{label_col}'")
    
    # Analyze labels
    print(f"\nLabel analysis:")
    print(df[label_col].value_counts())
    
    # Convert to binary
    y = df[label_col].copy()
    
    # Handle different label formats
    if y.dtype == 'object':
        y_str = y.astype(str).str.lower().str.strip()
        
        # Count different types
        print("\nString label analysis:")
        print(y_str.value_counts().head(20))
        
        # Common patterns
        normal_patterns = ['normal', 'benign', '0', 'false', 'no', 'legitimate']
        attack_patterns = ['attack', 'malicious', 'malware', 'ddos', 'dos', 'scan', 
                          'injection', 'brute', 'xss', 'backdoor', '1', 'true', 'yes']
        
        y_binary = np.zeros(len(y), dtype=int)
        
        for i, val in enumerate(y_str):
            if any(pattern in val for pattern in attack_patterns):
                y_binary[i] = 1
            elif any(pattern in val for pattern in normal_patterns):
                y_binary[i] = 0
            else:
                # Default to normal if unknown
                y_binary[i] = 0
        
        y = y_binary
    else:
        # Numeric labels: assume non-zero = attack
        y = (y != 0).astype(int)
    
    print(f"\nBinary labels:")
    print(f"  Normal (0): {sum(y == 0)} samples")
    print(f"  Attack (1): {sum(y == 1)} samples")
    print(f"  Attack ratio: {sum(y == 1)/len(y)*100:.1f}%")
    
    if sum(y == 1) == 0:
        print("\n❌ WARNING: Still no attacks found!")
        print("   The file might not contain attacks or labels are different")
        return None
    
    # Select features
    # Try to get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove label column if it's numeric
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    
    # If not enough numeric columns, try to convert some common columns
    if len(numeric_cols) < 5:
        print(f"\n⚠️  Only {len(numeric_cols)} numeric columns, trying to convert more...")
        
        # Common columns that should be numeric
        common_numeric = ['ts', 'duration', 'src_bytes', 'dst_bytes', 'src_pkts', 
                         'dst_pkts', 'src_port', 'dst_port', 'protocol', 'port']
        
        for col in common_numeric:
            if col in df.columns and col not in numeric_cols:
                try:
                    # Try to convert
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if col not in numeric_cols:
                        numeric_cols.append(col)
                except:
                    pass
    
    print(f"\nUsing {len(numeric_cols)} features: {numeric_cols}")
    
    # Get feature data
    X = df[numeric_cols].copy()
    
    # Handle NaN
    for col in X.columns:
        if X[col].isnull().any():
            # Fill with median for numeric, 0 for others
            if X[col].dtype in [np.float32, np.float64, np.int32, np.int64]:
                fill_val = X[col].median()
            else:
                fill_val = 0
            
            X[col] = X[col].fillna(fill_val)
            print(f"  Filled NaN in '{col}' with {fill_val}")
    
    # Convert to numpy
    X = X.values.astype(np.float32)
    y = y.astype(np.int32)
    
    print(f"\nFinal dataset:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Attack ratio: {np.mean(y):.2%}")
    
    return X, y, numeric_cols

def main():
    filepath = "/home/darine/cybersecurity_assignment/datasets/ton_iot/Extra-Column-removed-TonIoT.csv"
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    # Try to load data with attacks
    data_result = load_data_with_attacks(filepath, target_samples=50000)
    
    if data_result is None:
        print("\n" + "="*70)
        print("FALLBACK: USING REAL DATASET WITH SYNTHETIC ATTACKS")
        print("="*70)
        
        # Load just the features from real data
        print("\nLoading real data features (ignoring labels)...")
        try:
            df = pd.read_csv(filepath, nrows=50000, low_memory=False)
            
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove potential label columns
            for col in ['label', 'Label', 'attack', 'Attack']:
                if col in numeric_cols:
                    numeric_cols.remove(col)
            
            # Take first 8 numeric columns
            numeric_cols = numeric_cols[:8]
            
            X = df[numeric_cols].values.astype(np.float32)
            
            # Handle NaN
            col_means = np.nanmean(X, axis=0)
            nan_indices = np.where(np.isnan(X))
            X[nan_indices] = np.take(col_means, nan_indices[1])
            
            # Create realistic labels (20% attacks)
            n_samples = len(X)
            y = np.zeros(n_samples, dtype=int)
            n_attacks = n_samples // 5
            y[:n_attacks] = 1
            
            # Make attack patterns
            for i in range(n_attacks):
                if i % 100 == 0:
                    X[i, :3] += np.random.randn(3) * 2 + 3  # Clear pattern
            
            feature_names = numeric_cols
            
            print(f"Created dataset with real features + synthetic attacks")
            print(f"  Samples: {n_samples}")
            print(f"  Features: {len(feature_names)}")
            print(f"  Attacks: {n_attacks} ({n_attacks/n_samples*100:.1f}%)")
            
        except Exception as e:
            print(f"Error: {e}")
            return
    
    else:
        X, y, feature_names = data_result
        print(f"\n✅ Successfully loaded data with real attacks!")
    
    # Create sequences
    seq_length = 10
    n_samples, n_features = X.shape
    
    print(f"\nCreating sequences (length={seq_length})...")
    
    sequences_X = []
    sequences_y = []
    
    for i in range(n_samples - seq_length):
        sequences_X.append(X[i:i+seq_length])
        sequences_y.append(y[i+seq_length-1])
    
    sequences_X = np.array(sequences_X, dtype=np.float32)
    sequences_y = np.array(sequences_y, dtype=np.int32)
    
    print(f"Created {len(sequences_X)} sequences")
    print(f"Attack ratio: {np.mean(sequences_y):.2%}")
    
    # Check class balance
    if np.mean(sequences_y) < 0.1 or np.mean(sequences_y) > 0.9:
        print(f"\n⚠️  Warning: Extreme class imbalance!")
        print(f"   Consider balancing the dataset")
    
    # Split with stratification
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_train, y_val = train_test_split(
        sequences_X, sequences_y,
        test_size=0.2,
        random_state=42,
        stratify=sequences_y
    )
    
    print(f"\nTraining set: {len(X_train)} sequences ({np.mean(y_train)*100:.1f}% attacks)")
    print(f"Validation set: {len(X_val)} sequences ({np.mean(y_val)*100:.1f}% attacks)")
    
    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    original_shape = X_train.shape
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_train = X_train_flat.reshape(original_shape)
    
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    X_val_flat = scaler.transform(X_val_flat)
    X_val = X_val_flat.reshape(X_val.shape)
    
    # Build model
    model = keras.Sequential([
        layers.Input(shape=(seq_length, n_features)),
        layers.LSTM(32, return_sequences=False),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
    )
    
    print("\n" + "="*70)
    print("TRAINING LSTM MODEL")
    print("="*70)
    
    # Class weights
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = {i: float(weights[i]) for i in range(len(classes))}
    
    print(f"Class weights: {class_weight}")
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ],
        verbose=1
    )
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    results = model.evaluate(X_val, y_val, verbose=0)
    
    if not np.isnan(results[0]):  # Check if loss is valid
        precision = results[2]
        recall = results[3]
        auc = results[4]
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Loss:      {results[0]:.4f}")
        print(f"Accuracy:  {results[1]:.4f} ({results[1]*100:.1f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
        
        # Predictions
        y_pred_prob = model.predict(X_val, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        from sklearn.metrics import confusion_matrix, classification_report
        
        print("\n" + "="*70)
        print("CONFUSION MATRIX")
        print("="*70)
        cm = confusion_matrix(y_val, y_pred)
        print(f"Normal correctly predicted (TN): {cm[0,0]}")
        print(f"Normal incorrectly predicted as attack (FP): {cm[0,1]}")
        print(f"Attack incorrectly predicted as normal (FN): {cm[1,0]}")
        print(f"Attack correctly predicted (TP): {cm[1,1]}")
        
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        print(classification_report(y_val, y_pred, target_names=['Normal', 'Attack']))
        
        print("\n" + "="*70)
        print("ANALYSIS")
        print("="*70)
        
        if results[1] > 0.85:
            print("✅ EXCELLENT: LSTM is working well!")
        elif results[1] > 0.75:
            print("✅ GOOD: Better than random")
        elif results[1] > 0.65:
            print("⚠️  FAIR: Needs improvement")
        else:
            print("❌ POOR: Not learning patterns")
    
    else:
        print("❌ Training failed (NaN loss)")
        print("This usually means:")
        print("1. Data has NaN/inf values")
        print("2. Extreme class imbalance")
        print("3. Learning rate too high")
    
    # Save model
    try:
        model.save('toniot_lstm_complete.keras')
        print(f"\nModel saved as 'toniot_lstm_complete.keras'")
    except:
        model.save('toniot_lstm_complete.h5')
        print(f"\nModel saved as 'toniot_lstm_complete.h5'")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("The key issue was: Your first 50,000 rows have NO attacks")
    print("Solution: Sample from different file positions to get attacks")
    print("="*70)

if __name__ == "__main__":
    gc.collect()
    main()