import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
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

import warnings
import os
import joblib

# Set random seed for reproducibility
np.random.seed(42)
warnings.filterwarnings('ignore')

class UNSWNB15IsolationForest:
    def __init__(self, contamination=0.1, n_estimators=100, max_samples=256):
        """
        Initialize Isolation Forest model for UNSW-NB15 dataset
        
        Parameters:
        -----------
        contamination : float, default=0.1
            Expected proportion of anomalies in the data
        n_estimators : int, default=100
            Number of base estimators in the ensemble
        max_samples : int or float, default=256
            Number of samples to draw for each base estimator
        """
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=42,
            bootstrap=False,
            n_jobs=-1,
            verbose=0
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_unsw_nb15_data(self, data_path="/home/darine/cybersecurity_assignment/datasets/unsw_nb15"):
        """
        Load UNSW-NB15 dataset from the specified path
        """
        print("Loading UNSW-NB15 dataset...")
        print(f"Looking for data in: {data_path}")
        
        try:
            # Check if the directory exists
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Directory not found: {data_path}")
            
            # List files in the directory
            files = os.listdir(data_path)
            print(f"Files found: {files}")
            
            # Try to find the training and testing files
            training_file = None
            testing_file = None
            
            for file in files:
                if 'training' in file.lower() and file.endswith('.csv'):
                    training_file = os.path.join(data_path, file)
                elif 'testing' in file.lower() and file.endswith('.csv'):
                    testing_file = os.path.join(data_path, file)
                elif 'trainingset' in file.lower() and file.endswith('.csv'):
                    training_file = os.path.join(data_path, file)
                elif 'testingset' in file.lower() and file.endswith('.csv'):
                    testing_file = os.path.join(data_path, file)
            
            # Load training data
            if training_file:
                print(f"\nLoading training data from: {training_file}")
                train_df = pd.read_csv(training_file)
                print(f"Training data shape: {train_df.shape}")
                print(f"Training columns: {train_df.columns.tolist()[:10]}...")  # Show first 10 columns
            else:
                # Try to find any CSV file
                csv_files = [f for f in files if f.endswith('.csv')]
                if csv_files:
                    training_file = os.path.join(data_path, csv_files[0])
                    train_df = pd.read_csv(training_file)
                    print(f"Loaded data from: {training_file}")
                    print(f"Data shape: {train_df.shape}")
                else:
                    raise FileNotFoundError("No CSV files found in the directory")
            
            # Load testing data if available
            if testing_file:
                print(f"\nLoading testing data from: {testing_file}")
                test_df = pd.read_csv(testing_file)
                print(f"Testing data shape: {test_df.shape}")
                
                # Combine train and test for full dataset
                df = pd.concat([train_df, test_df], ignore_index=True)
                print(f"\nCombined dataset shape: {df.shape}")
            else:
                df = train_df
            
            # Display basic info
            print("\nDataset Information:")
            print(f"Total samples: {len(df)}")
            print(f"Total features: {len(df.columns)}")
            
            # Check if label column exists
            if 'label' in df.columns:
                anomaly_count = df['label'].sum()
                normal_count = len(df) - anomaly_count
                print(f"\nClass Distribution:")
                print(f"  Normal samples: {normal_count} ({normal_count/len(df)*100:.2f}%)")
                print(f"  Anomaly samples: {anomaly_count} ({anomaly_count/len(df)*100:.2f}%)")
            
            if 'attack_cat' in df.columns:
                print("\nAttack Categories Distribution:")
                print(df['attack_cat'].value_counts())
            
            return df
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Creating synthetic dataset for demonstration...")
            return self.create_synthetic_dataset()
    
    def create_synthetic_dataset(self, n_samples=50000):
        """Create synthetic UNSW-NB15-like dataset"""
        print("Creating synthetic dataset...")
        np.random.seed(42)
        
        # Create synthetic data similar to UNSW-NB15
        data = {
            'dur': np.random.exponential(1.0, n_samples),
            'proto': np.random.choice(['tcp', 'udp', 'icmp', 'arp'], n_samples),
            'service': np.random.choice(['-', 'http', 'ftp', 'ssh', 'smtp', 'dns'], n_samples),
            'state': np.random.choice(['FIN', 'CON', 'INT', 'REQ', 'RST'], n_samples),
            'spkts': np.random.poisson(10, n_samples),
            'dpkts': np.random.poisson(10, n_samples),
            'sbytes': np.random.exponential(1000, n_samples),
            'dbytes': np.random.exponential(1000, n_samples),
            'sttl': np.random.randint(32, 255, n_samples),
            'dttl': np.random.randint(32, 255, n_samples),
            'sload': np.random.exponential(1000000, n_samples),
            'dload': np.random.exponential(1000000, n_samples),
            'label': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        }
        
        # Add anomaly patterns
        anomaly_indices = np.where(data['label'] == 1)[0]
        for idx in anomaly_indices[:len(anomaly_indices)//2]:
            data['sbytes'][idx] *= 100
            data['dbytes'][idx] *= 100
            data['sload'][idx] *= 10
        
        for idx in anomaly_indices[len(anomaly_indices)//2:]:
            data['spkts'][idx] = 1
            data['dpkts'][idx] = 1
            data['dur'][idx] = 0.001
        
        df = pd.DataFrame(data)
        print(f"Synthetic dataset created with shape: {df.shape}")
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess UNSW-NB15 data for Isolation Forest
        """
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        # Make a copy
        df_processed = df.copy()
        
        # 1. Handle missing values
        print("\n1. Handling missing values...")
        missing_values = df_processed.isnull().sum()
        if missing_values.sum() > 0:
            print(f"Missing values found: {missing_values[missing_values > 0].to_dict()}")
            # Fill numeric columns with median
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_processed[col].isnull().sum() > 0:
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
            
            # Fill categorical columns with mode
            categorical_cols = df_processed.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df_processed[col].isnull().sum() > 0:
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
        
        # 2. Separate features and labels
        print("\n2. Separating features and labels...")
        
        # Save label if exists
        if 'label' in df_processed.columns:
            y = df_processed['label'].values
            # Remove label and attack_cat from features
            columns_to_drop = ['label']
            if 'attack_cat' in df_processed.columns:
                columns_to_drop.append('attack_cat')
            if 'id' in df_processed.columns:
                columns_to_drop.append('id')
            X_df = df_processed.drop(columns=columns_to_drop)
        else:
            y = None
            X_df = df_processed
        
        # Save feature names
        self.feature_names = X_df.columns.tolist()
        print(f"Using {len(self.feature_names)} features for training")
        
        # 3. Handle categorical variables
        print("\n3. Encoding categorical variables...")
        categorical_cols = X_df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            try:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X_df[col] = self.label_encoders[col].fit_transform(X_df[col].astype(str))
                else:
                    X_df[col] = self.label_encoders[col].transform(X_df[col].astype(str))
            except Exception as e:
                print(f"  Error encoding {col}: {e}")
                # Use frequency encoding as fallback
                freq_encoding = X_df[col].value_counts(normalize=True)
                X_df[col] = X_df[col].map(freq_encoding)
        
        # 4. Convert to numpy array and scale
        print("\n4. Scaling features...")
        X = X_df.values
        X = self.scaler.fit_transform(X)
        
        print(f"\nPreprocessing complete!")
        print(f"Final feature matrix shape: {X.shape}")
        if y is not None:
            print(f"Label vector shape: {y.shape}")
        
        return X, y, X_df
    
    def train_model(self, X_train, y_train=None):
        """
        Train Isolation Forest model
        """
        print("\n" + "="*60)
        print("TRAINING ISOLATION FOREST")
        print("="*60)
        
        print(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features")
        
        # Train the model
        self.model.fit(X_train)
        
        # Get training predictions
        train_scores = self.model.decision_function(X_train)
        train_predictions = self.model.predict(X_train)
        train_predictions_binary = np.where(train_predictions == 1, 0, 1)
        
        if y_train is not None:
            print("\nTraining set performance:")
            self._print_metrics(y_train, train_predictions_binary, train_scores)
        
        return train_scores, train_predictions_binary
    
    def predict(self, X):
        """Make predictions on new data"""
        scores = self.model.decision_function(X)
        predictions = self.model.predict(X)
        predictions_binary = np.where(predictions == 1, 0, 1)
        return scores, predictions_binary
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance on test set"""
        print("\n" + "="*60)
        print("MODEL EVALUATION ON TEST SET")
        print("="*60)
        
        test_scores, test_predictions = self.predict(X_test)
        metrics = self._print_metrics(y_test, test_predictions, test_scores)
        return metrics, test_scores, test_predictions
    
    def _print_metrics(self, y_true, y_pred, scores):
        """Helper function to calculate and print evaluation metrics"""
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, 
                                  target_names=['Normal', 'Anomaly'],
                                  digits=4))
        
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        try:
            auc_score = roc_auc_score(y_true, scores)
            print(f"ROC AUC Score: {auc_score:.4f}")
        except:
            auc_score = None
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc_score,
            'confusion_matrix': cm
        }
    
    def visualize_results(self, X, scores, predictions, y_true=None, sample_size=3000):
        """
        Create visualizations of results
        """
        print("\n" + "="*60)
        print("VISUALIZING RESULTS")
        print("="*60)
        
        # Sample data if too large
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sampled = X[indices]
            scores_sampled = scores[indices]
            predictions_sampled = predictions[indices]
            if y_true is not None:
                y_true_sampled = y_true[indices]
            else:
                y_true_sampled = None
        else:
            X_sampled = X
            scores_sampled = scores
            predictions_sampled = predictions
            y_true_sampled = y_true
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Anomaly Score Distribution
        ax1 = axes[0, 0]
        ax1.hist(scores_sampled, bins=50, edgecolor='black', alpha=0.7)
        ax1.set_title('Distribution of Anomaly Scores', fontweight='bold')
        ax1.set_xlabel('Anomaly Score')
        ax1.set_ylabel('Frequency')
        
        # 2. PCA Visualization
        ax2 = axes[0, 1]
        try:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_sampled)
            scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], 
                                 c=predictions_sampled, 
                                 cmap='RdYlBu_r', 
                                 alpha=0.6, 
                                 s=20)
            ax2.set_title('PCA: Anomaly Predictions', fontweight='bold')
            ax2.set_xlabel('PC1')
            ax2.set_ylabel('PC2')
            plt.colorbar(scatter, ax=ax2, label='Prediction\n(0=Normal, 1=Anomaly)')
        except:
            ax2.text(0.5, 0.5, 'PCA failed', ha='center', va='center')
        
        # 3. Scores vs Predictions
        ax3 = axes[0, 2]
        sorted_indices = np.argsort(scores_sampled)
        ax3.scatter(range(len(scores_sampled)), scores_sampled[sorted_indices],
                   c=predictions_sampled[sorted_indices], 
                   cmap='RdYlBu_r', 
                   alpha=0.6,
                   s=20)
        ax3.set_title('Anomaly Scores (Sorted)', fontweight='bold')
        ax3.set_xlabel('Sample Index')
        ax3.set_ylabel('Anomaly Score')
        
        # 4. Confusion Matrix
        ax4 = axes[1, 0]
        if y_true_sampled is not None:
            cm = confusion_matrix(y_true_sampled, predictions_sampled)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
            ax4.set_title('Confusion Matrix', fontweight='bold')
            ax4.set_xlabel('Predicted')
            ax4.set_ylabel('Actual')
        else:
            ax4.text(0.5, 0.5, 'True labels not available', ha='center', va='center')
        
        # 5. ROC Curve
        ax5 = axes[1, 1]
        if y_true_sampled is not None:
            try:
                fpr, tpr, _ = roc_curve(y_true_sampled, -scores_sampled)
                auc_score = roc_auc_score(y_true_sampled, -scores_sampled)
                ax5.plot(fpr, tpr, 'b-', label=f'AUC = {auc_score:.3f}')
                ax5.plot([0, 1], [0, 1], 'r--')
                ax5.set_xlabel('False Positive Rate')
                ax5.set_ylabel('True Positive Rate')
                ax5.set_title('ROC Curve', fontweight='bold')
                ax5.legend()
            except:
                ax5.text(0.5, 0.5, 'ROC curve failed', ha='center', va='center')
        
        # 6. Score Distribution by Class
        ax6 = axes[1, 2]
        if y_true_sampled is not None:
            normal_scores = scores_sampled[y_true_sampled == 0]
            anomaly_scores = scores_sampled[y_true_sampled == 1]
            
            ax6.hist(normal_scores, bins=30, alpha=0.5, label='Normal', color='blue')
            ax6.hist(anomaly_scores, bins=30, alpha=0.5, label='Anomaly', color='red')
            ax6.set_xlabel('Anomaly Score')
            ax6.set_ylabel('Frequency')
            ax6.set_title('Score Distribution by Class', fontweight='bold')
            ax6.legend()
        
        plt.suptitle('Isolation Forest Results - UNSW-NB15', fontsize=16, fontweight='bold')
        plt.tight_layout()
        _savefig('isolation_forest_unsw_nb15')
    
    def save_model(self, filepath="isolation_forest_unsw_nb15.pkl"):
        """Save the trained model"""
        print(f"\nSaving model to {filepath}...")
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print("Model saved successfully!")


def main():
    """
    Main function to run Isolation Forest on UNSW-NB15 dataset
    """
    print("="*80)
    print("ISOLATION FOREST ANOMALY DETECTION - UNSW-NB15 DATASET")
    print("="*80)
    
    # Initialize the detector
    detector = UNSWNB15IsolationForest(
        contamination=0.1,
        n_estimators=100,
        max_samples=256
    )
    
    # Step 1: Load the dataset
    print("\n" + "="*80)
    print("STEP 1: LOADING DATASET")
    print("="*80)
    df = detector.load_unsw_nb15_data("/home/darine/cybersecurity_assignment/datasets/unsw_nb15")
    
    # Step 2: Preprocess the data
    print("\n" + "="*80)
    print("STEP 2: PREPROCESSING DATA")
    print("="*80)
    X, y, X_df = detector.preprocess_data(df)
    
    # Step 3: Split the data
    print("\n" + "="*80)
    print("STEP 3: SPLITTING DATA")
    print("="*80)
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    else:
        X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
        y_train = y_test = None
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Step 4: Train the model
    print("\n" + "="*80)
    print("STEP 4: TRAINING MODEL")
    print("="*80)
    train_scores, train_preds = detector.train_model(X_train, y_train)
    
    # Step 5: Evaluate on test set
    print("\n" + "="*80)
    print("STEP 5: EVALUATING MODEL")
    print("="*80)
    if y_test is not None:
        metrics, test_scores, test_preds = detector.evaluate_model(X_test, y_test)
    
    # Step 6: Visualize results
    print("\n" + "="*80)
    print("STEP 6: VISUALIZING RESULTS")
    print("="*80)
    detector.visualize_results(X_test, test_scores, test_preds, y_test)
    
    # Step 7: Save the model
    print("\n" + "="*80)
    print("STEP 7: SAVING MODEL")
    print("="*80)
    detector.save_model("isolation_forest_unsw_nb15_model.pkl")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return detector, metrics if 'metrics' in locals() else None


def analyze_sample_record():
    """
    Analyze a sample network record
    """
    print("\n" + "="*80)
    print("ANALYZING SAMPLE RECORD")
    print("="*80)
    
    # Load a trained model
    try:
        detector = UNSWNB15IsolationForest()
        model_data = joblib.load("isolation_forest_unsw_nb15_model.pkl")
        detector.model = model_data['model']
        detector.scaler = model_data['scaler']
        detector.label_encoders = model_data['label_encoders']
        detector.feature_names = model_data['feature_names']
        print("Model loaded successfully!")
    except:
        print("No trained model found. Please run main() first.")
        return
    
    # Create a sample record (adjust based on your dataset)
    sample_record = {
        'dur': 0.5,
        'proto': 'tcp',
        'service': 'http',
        'state': 'CON',
        'spkts': 10,
        'dpkts': 8,
        'sbytes': 1500,
        'dbytes': 1200,
        'sttl': 64,
        'dttl': 64,
        'sload': 1000000,
        'dload': 800000,
    }
    
    # Convert to DataFrame
    record_df = pd.DataFrame([sample_record])
    
    # Add missing columns
    for col in detector.feature_names:
        if col not in record_df.columns:
            record_df[col] = 0
    
    # Reorder columns to match training
    record_df = record_df[detector.feature_names]
    
    # Preprocess
    for col in record_df.select_dtypes(include=['object']).columns:
        if col in detector.label_encoders:
            try:
                record_df[col] = detector.label_encoders[col].transform(record_df[col].astype(str))
            except:
                # If label not seen during training, use most common
                record_df[col] = 0
    
    # Scale
    X = record_df.values
    X_scaled = detector.scaler.transform(X)
    
    # Predict
    score, prediction = detector.predict(X_scaled)
    
    print(f"\nAnalysis Results:")
    print(f"  Anomaly Score: {score[0]:.4f}")
    print(f"  Prediction: {'ANOMALY' if prediction[0] == 1 else 'NORMAL'}")
    print(f"  Confidence: {abs(score[0]):.2%}")
    
    if score[0] < -0.5:
        print(f"  ⚠️  HIGHLY ANOMALOUS")
    elif score[0] < 0:
        print(f"  ⚠️  SUSPICIOUS")
    else:
        print(f"  ✓ NORMAL")


if __name__ == "__main__":
    # Run the main analysis
    detector, metrics = main()
    
    # Optionally analyze a sample record
    analyze_sample_record()