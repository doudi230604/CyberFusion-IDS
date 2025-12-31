#!/usr/bin/env python3
"""Generate comparative ROC curve for RF, DT, IsolationForest on a synthetic UNSW-like dataset"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PLOTS_DIR = os.path.join(ROOT, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Create synthetic dataset
np.random.seed(42)
N = 20000
X = np.column_stack(
    [
        np.random.exponential(1.0, N),  # dur
        np.random.poisson(10, N),        # spkts
        np.random.poisson(10, N),        # dpkts
        np.random.exponential(1000, N),  # sbytes
        np.random.exponential(1000, N),  # dbytes
        np.random.randint(32, 255, N),   # sttl
        np.random.randint(32, 255, N),   # dttl
    ]
)
# labels: 0 normal, 1 attack
y = np.random.choice([0,1], size=N, p=[0.85,0.15])
# introduce stronger signals in attacks
attack_idx = np.where(y==1)[0]
X[attack_idx, 0] *= 3
X[attack_idx, 3] *= 10
X[attack_idx, 4] *= 10

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Models
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
dt = DecisionTreeClassifier(max_depth=15, random_state=42)
iforest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)

rf.fit(X_train_s, y_train)
dt.fit(X_train_s, y_train)
iforest.fit(X_train_s)

# Scores
rf_score = rf.predict_proba(X_test_s)[:,1]
dt_score = dt.predict_proba(X_test_s)[:,1]
if_score = -iforest.decision_function(X_test_s)  # higher means more anomalous

# ROC curves
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_score)
auc_rf = auc(fpr_rf, tpr_rf)

fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_score)
auc_dt = auc(fpr_dt, tpr_dt)

fpr_if, tpr_if, _ = roc_curve(y_test, if_score)
auc_if = auc(fpr_if, tpr_if)

plt.figure(figsize=(8,6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc_rf:.3f})', linewidth=2)
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC={auc_dt:.3f})', linewidth=2)
plt.plot(fpr_if, tpr_if, label=f'Isolation Forest (AUC={auc_if:.3f})', linewidth=2)
plt.plot([0,1],[0,1],'k--', alpha=0.6)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparative ROC - UNSW-like dataset')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(PLOTS_DIR, 'comparative_roc_unsw.png')
plt.savefig(out, dpi=150)
print('Saved', out)
plt.close()