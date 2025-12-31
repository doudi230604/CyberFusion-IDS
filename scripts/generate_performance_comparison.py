#!/usr/bin/env python3
"""Generate performance comparison bar chart (Recall) for models across datasets
Uses values from the report (hardcoded)"""
import os
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PLOTS_DIR = os.path.join(ROOT, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Recall values from report
models = ['Random Forest', 'Decision Tree', 'Isolation Forest', 'LSTM']
datasets = ['UNSW-NB15', 'CICIDS2017', 'TonIoT']
# Recall per model per dataset
recall = {
    'Random Forest': [0.9825, 0.9912, 0.9879],
    'Decision Tree': [0.9382, 0.9658, 0.9586],
    'Isolation Forest': [0.8943, 0.9287, 0.9124],
    'LSTM': [0.9685, 0.9846, 0.9812]
}

x = np.arange(len(datasets))
width = 0.18
plt.figure(figsize=(10,6))
for i, m in enumerate(models):
    vals = recall[m]
    plt.bar(x + (i-1.5)*width, vals, width, label=m)

plt.xticks(x, datasets)
plt.ylim(0,1.0)
plt.ylabel('Recall')
plt.title('Recall by Model across Datasets')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
out = os.path.join(PLOTS_DIR, 'performance_comparison_recall.png')
plt.savefig(out, dpi=150)
print('Saved', out)
plt.close()