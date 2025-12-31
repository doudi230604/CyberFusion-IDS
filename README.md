# LSTM-Based Intrusion Detection System (IDS) - Project Explanation

## Project Overview

This is a **Deep Learning project for network intrusion detection** using LSTM (Long Short-Term Memory) neural networks. It focuses on detecting cyber attacks in network traffic by analyzing temporal patterns in packet sequences rather than individual packets in isolation.

---

## Core Objectives

1. **Early Attack Detection** - Recognize multi-step attacks before they complete
2. **Sequential Pattern Recognition** - Capture temporal relationships in network flows
3. **Multi-Class Attack Classification** - Distinguish between attack types (DDoS, Port Scan, Brute Force, etc.)
4. **Operational Performance** - Balance precision vs recall based on security priorities

---

## Key Technical Components

### 1. **Why LSTM for IDS?**

| Problem | LSTM Solution |
|---------|--------------|
| Single packets look normal | Analyzes sequences of packets |
| Missing attack context | Remembers previous packets via memory cells |
| Can't detect multi-step attacks | Processes temporal dependencies |
| Traditional ML fails on sequences | Gate mechanisms (Forget/Input/Output) control information flow |

### 2. **LSTM Architecture for IDS**

```
Input: Sequence of 10 packets [P₁, P₂, P₃, ..., P₁₀]
                          ↓
                    [LSTM Layer 128 units] → Hidden state h₁
                          ↓
                    [LSTM Layer 64 units]  → Hidden state h₂
                          ↓
                    [Dense Layer 32]       → Feature extraction
                          ↓
                    [Output Layer]         → Attack/Normal classification
```

**Key Parameters:**
- **Optimizer:** Adam (adaptive learning) - adjusts based on pattern complexity
- **Loss Function:** Categorical Crossentropy (multi-class) or Binary Crossentropy
- **Metrics:** Precision, Recall, F1-Score, AUC-ROC
- **Dropout:** 0.2-0.3 to prevent overfitting

---

## LSTM Gates: The Control Mechanism

### Three Core Gates in LSTM

#### 1. **Forget Gate** - What to REMOVE from memory
```
f_t = sigmoid(W_f • [h_{t-1}, x_t] + b_f)
```
- **IDS Application:** Forgets old, irrelevant connections; remembers ongoing attacks
- **Analogy:** Connection Timeout Handler

#### 2. **Input Gate** - What NEW information to STORE
```
i_t = sigmoid(W_i • [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C • [h_{t-1}, x_t] + b_C)
```
- **IDS Application:** Learns new attack signatures from current packet
- **Analogy:** Feature Extractor

#### 3. **Output Gate** - What to OUTPUT
```
o_t = sigmoid(W_o • [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(C_t)
```
- **IDS Application:** Produces detection decision/feature for classification
- **Analogy:** Alert Generator

---

## Two Implementation Cases

### **Case 1: UNSW-NB15 Dataset** (Traditional Network IDS)

**Dataset Profile:**
- 2.5M records, 49 features
- 9 attack categories + normal
- Binary & multi-class classification

**Implementation Steps:**

#### Step 1: Data Preparation
- Load CSV files (UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv)
- Merge and select relevant features
- Handle categorical variables using Label Encoding (protocol, service, state)
- Normalize numerical features using StandardScaler

#### Step 2: Sequence Creation (CRITICAL Stage 2)
- Create sequences of 10 timesteps from network flows
- Group by source IP and destination IP for contextual sequences
- Create overlapping windows with stride=1

**Real Example: SSH Brute Force Attack**
```
Raw Packets (in time order):
1. TCP SYN to port 22
2. TCP SYN-ACK response
3. SSH Protocol negotiation
4. SSH Login attempt (wrong password) ← Attack start
5. SSH Login attempt (wrong password)
6. SSH Login attempt (wrong password)
7. SSH Login attempt (success!) ← Attack successful
8. Malicious command execution

AFTER SEQUENCE CREATION:
Sequence for this connection = [Packet1, Packet2, ..., Packet8]
LSTM sees the PROGRESSION of the attack
```

#### Step 3: Label Processing
- Binary classification: Normal (0) vs Attack (1)
- Multi-class: Map attack categories to 1-9
- Convert to categorical (one-hot encoding)

#### Step 4: Model Training
- Split data: 70% train, 15% validation, 15% test
- Train LSTM with EarlyStopping (patience=10) and ReduceLROnPlateau
- Use batch_size=64, epochs=50

#### Step 5: Evaluation
- Calculate accuracy, precision, recall, F1-score
- Generate confusion matrix
- Plot ROC curve for multi-class classification

**Attack Detection Patterns:**
```
Attack Type          Sequence Length
─────────────────────────────────────
Port Scan            5-10 packets
Brute Force          10-20 packets
DDoS                 50-100 packets
APT/C2               100-1000 packets
```

---

### **Case 2: ToN-IoT Dataset** (IoT Network IDS)

**Dataset Profile:**
- 44 IoT-specific features
- 7 attack types targeting IoT devices
- Time-series format with timestamps

**Key Differences from Case 1:**

#### Step 1: Dataset Loading
- Use parquet format for faster loading
- Extract IoT-specific features
- Focus on protocol-specific attributes (MQTT, CoAP)

#### Step 2: Temporal Feature Engineering
- Create time-based features: hour, day of week
- Generate statistical features over rolling windows
- Calculate packet rate, byte rate per time window

#### Step 3: Sequence Preparation
- Create sequences of 20 timesteps (IoT traffic has shorter patterns)
- Use sliding window with 50% overlap
- Balance classes using SMOTE for rare attack types

#### Step 4: Multi-Head LSTM Model
- Build parallel LSTMs for different protocol types
- Add attention mechanism for important timesteps
- Use Bidirectional LSTM for context from both directions

#### Step 5: IoT-Specific Evaluation
- Calculate detection latency (time to detect attack)
- Evaluate false positives for critical IoT services
- Test on individual IoT device types separately

---

## Critical Preprocessing Stage (Stage 2)

### **Sequential Processing - The Bridge**

**WITHOUT Sequence Creation:**
```
Packet 1 → Model → "Normal" ❌ (missed context)
Packet 2 → Model → "Normal" ❌
...
Packet 100 → Model → "Attack" ❌ (too late!)
```

**WITH Sequence Creation:**
```
[P₁, P₂, P₃, P₄, P₅] → LSTM → "Port Scan detected" ✓ (early!)
[P₆, P₇, P₈, P₉, P₁₀] → LSTM → "Exploit attempt" ✓
```

### **Why Sequential Processing Matters:**

| Attack Type | Why Sequential is Critical |
|-------------|---------------------------|
| Port Scan | Multiple probes over time; single packet is normal |
| DDoS | High volume pattern emerges over seconds/minutes |
| Brute Force | Multiple failed logins then success |
| Data Exfiltration | Slow, stealthy transfer over many packets |
| APT | Multi-stage attack over days/weeks |

---

## Libraries for Implementation

```python
# Core libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
```

---

## Model Architecture Code Example

```python
def build_lstm_model(input_shape, num_classes):
    """
    Build LSTM model for intrusion detection
    Parameters:
    - input_shape: (timesteps, features)
    - num_classes: Number of attack categories
    Returns:
    - Compiled Keras model
    """
    model = keras.Sequential([
        # LSTM layer with return_sequences=True for deep LSTM
        keras.layers.LSTM(128,
                         return_sequences=True,
                         input_shape=input_shape,
                         dropout=0.2,
                         recurrent_dropout=0.2),
        # Second LSTM layer
        keras.layers.LSTM(64,
                         dropout=0.2,
                         recurrent_dropout=0.2),
        # Dense layers for classification
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        # Output layer
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                keras.metrics.Precision(),
                keras.metrics.Recall(),
                keras.metrics.AUC()]
    )
    return model
```

---

## Optimization Parameters & Trade-offs

### **Step 1: Define Security Priorities**

| Priority | Optimizer | Loss Function | Primary Metric |
|----------|-----------|---------------|----------------|
| **Catch ALL attacks** (Healthcare/Finance) | Adam (low LR: 0.0001) | Weighted Binary CE | **Recall** ↑ |
| **Minimize false alarms** (Enterprise SOC) | RMSprop (LR: 0.001) | Standard CE | **Precision** ↑ |
| **Balanced approach** (Research/Evaluation) | Adam (default) | Categorical CE | **F1-Score** |

### **Core Parameters Breakdown**

#### A. OPTIMIZER: "The Learning Strategy"

| Optimizer | Learning Style | IDS Context | When to Use |
|-----------|----------------|-------------|------------|
| **Adam** | Adaptive learning - adjusts based on pattern complexity | Like an analyst who learns faster from obvious attacks | **DEFAULT CHOICE** - works for most IDS scenarios |
| **RMSprop** | Moderates learning for recurrent patterns | Good for temporal dependencies | When attack patterns have strong sequential correlations |
| **SGD** | Steady, consistent learning | Drilling basics repeatedly | Simple attacks, stable environments |

#### B. LOSS FUNCTION: "The Cost of Being Wrong"

**Binary Classification (Attack/Not Attack):**
- Binary Crossentropy: Measures divergence between predicted and actual labels
- Penalizes confident wrong predictions MORE than uncertain ones
- Example: 90% sure it's attack but normal → HIGH penalty

**Multi-Class Classification (Attack Types):**
- Categorical Crossentropy: For one-hot encoded labels
- Each attack type is separate class (DoS, Probe, R2L, U2R, Normal)
- Model must choose ONE primary classification

**Imbalanced Data Special Cases:**
- Focal Loss: Gives more weight to hard-to-classify examples
- Weighted Crossentropy: Manually assign higher penalty for missing critical attacks

#### C. METRICS: "The Report Card"

- **Accuracy:** What percentage of total predictions were correct?
  - ⚠️ PROBLEM: If 99% traffic is normal, 99% accuracy means nothing

- **Precision:** "When the IDS says 'ATTACK,' how often is it right?"
  - Formula: TP / (TP + FP)
  - **CRITICAL for IDS** - reduces alert fatigue
  - High precision = Few false alarms

- **Recall:** "Of all real attacks, what percentage did we catch?"
  - Formula: TP / (TP + FN)
  - Also called Sensitivity or True Positive Rate
  - High recall = Few missed attacks

- **F1-Score:** Harmonic mean of Precision and Recall
  - Good for imbalanced datasets
  - Penalizes extremes: Good precision AND good recall needed

- **AUC-ROC:** Overall discrimination ability
  - Measures model's ability to distinguish classes at ALL thresholds
  - Value: 0.5 (random) to 1.0 (perfect)
  - Good for comparing different models

---

## Handling Datasets - Recommendations

### **1. Missing Values**
- Numerical features: Median imputation
- Categorical features: 'Unknown' category

### **2. Address Class Imbalance**
- **SMOTE:** Generate synthetic minority samples
- **Class weights:** Penalize misclassifying rare attacks
- **Under-sampling:** Reduce majority if extreme imbalance (99% normal)

### **3. Feature Selection**
- Remove zero-variance features
- Use mutual information for relevance
- Remove highly correlated features (threshold > 0.95)

### **4. Variable-Length Sessions**
```
Different connections have different lengths:
- ssh_session = 1500 packets (Long)
- dns_query = 2 packets (Short)

Solutions:
1. Padding: Add zeros to short sequences
2. Truncation: Cut long sequences to max length
3. Hierarchical: Process chunks of fixed length
```

### **5. Sequence Quality Checks**

Essential validation:
1. Temporal order preserved (timestamps increasing)
2. Same connection/IP throughout sequence
3. No data leakage between train/test
4. Balanced attack types in sequences
5. Realistic time gaps (not combining unrelated packets)

---

## Training Phase Adaptation

### **Early Training:**
- Higher learning rate to find general patterns
- Focus on accuracy to establish baseline

### **Mid Training:**
- Reduce learning rate to refine detection
- Monitor precision/recall trade-off

### **Late Training:**
- Very low learning rate for fine-tuning
- Optimize for operational metrics (F1, AUC)

### **Healthy Training Signs:**
- Loss: Decreases steadily, then plateaus
- Validation metrics: Follow training metrics closely
- Precision/Recall: Both improve (though trade-off exists)
- AUC-ROC: Consistently above 0.9 for good IDS

### **Warning Signs:**
- Loss increasing: Learning rate too high
- Large gap between train/val metrics: Overfitting
- Low Precision, High Recall: Too sensitive, many false alarms
- High Precision, Low Recall: Too conservative, missing attacks

---

## SHAP Integration for LSTM IDS

### **Feasibility & Implementation Steps (Conceptual)**

**Question:** Should you use SHAP with LSTM?
**Answer:** YES, with considerations

### **Why Use SHAP?**
1. Interpret which packets/features triggered attack detection
2. Explain confidence level to security analysts
3. Audit false positives/negatives

### **Implementation Approach:**

1. **SHAP for sequence-level:** Attribute importance to each timestep in sequence
2. **SHAP for feature-level:** Show which features (bytes, protocol, flags) matter
3. **Limitation:** LSTM's internal state makes timestep attribution complex

### **Practical Steps:**
1. Convert LSTM to simplified explanation model (knowledge distillation)
2. Use SHAP on intermediate LSTM outputs
3. Create force plots showing packet sequence contribution
4. Focus on last few timesteps (closer to decision)

### **Trade-offs:**
- SHAP adds computational overhead
- Sequence dependencies harder to explain than static features
- Most useful for understanding misclassifications

---

## Summary: End-to-End Workflow

```
1. DATA PREPARATION
   ├─ Load UNSW-NB15 or ToN-IoT
   ├─ Encode categorical variables
   └─ Normalize numerical features

2. SEQUENCE CREATION (Stage 2 - Critical)
   ├─ Group packets by connection (src→dst IP)
   ├─ Create timestep windows (10 for UNSW, 20 for IoT)
   └─ Create overlapping windows (stride=1 or 50%)

3. MODEL ARCHITECTURE
   ├─ LSTM(128) + LSTM(64) + Dense layers
   ├─ Dropout (0.2-0.3) for regularization
   └─ Softmax output for classification

4. TRAINING
   ├─ Optimizer: Adam (for most cases)
   ├─ Loss: Categorical Crossentropy
   ├─ Metrics: Precision, Recall, F1, AUC
   └─ Callbacks: EarlyStopping, ReduceLROnPlateau

5. EVALUATION
   ├─ Confusion matrix
   ├─ ROC-AUC curves
   ├─ Per-attack-type precision/recall
   └─ Detection latency measurement

6. DEPLOYMENT CONSIDERATIONS
   ├─ Model compression for real-time IDS
   ├─ Sequence buffering strategy
   ├─ Handling concept drift (evolving attacks)
   └─ SHAP explanations for alert justification
```

---

## Key Takeaway

**LSTM outperforms traditional ML for IDS because it sees the "story" (sequence) rather than individual "words" (packets).**

The success depends on:
- **60% on good preprocessing** (Stage 2 sequence creation)
- **40% on optimal model parameters**

The art of model configuration for IDS is balancing mathematical optimization with security operational reality. The best configuration depends not just on your data, but on your security posture, team capacity, and risk tolerance.

---

## Dataset Characteristics Summary

| Aspect | UNSW-NB15 | ToN-IoT |
|--------|-----------|---------|
| **Records** | 2.5M | Variable |
| **Features** | 49 | 44 |
| **Attack Types** | 9 + Normal | 7 + Normal |
| **Format** | CSV | Parquet |
| **Sequence Length** | 10 timesteps | 20 timesteps |
| **Protocols** | TCP/UDP/HTTP | MQTT/CoAP/HTTP |
| **Focus** | General network | IoT devices |
| **Model Variant** | Standard LSTM | Multi-Head LSTM + Attention |
