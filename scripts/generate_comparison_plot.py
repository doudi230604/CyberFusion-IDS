#!/usr/bin/env python3
"""Generate a sample comparison plot: performance vs number of features
Saves to plots/feature_count_comparison.png
"""
import os
import matplotlib.pyplot as plt
import numpy as np

plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))
os.makedirs(plots_dir, exist_ok=True)

x = np.array([5,10,20,40,60,84])
# Example accuracies for Random Forest and Decision Tree (synthetic illustrative)
rf = np.array([0.93,0.95,0.985,0.987,0.988,0.993])
dt = np.array([0.88,0.9,0.954,0.956,0.957,0.978])

plt.figure(figsize=(8,5))
plt.plot(x, rf, marker='o', label='Random Forest', linewidth=2)
plt.plot(x, dt, marker='s', label='Decision Tree', linewidth=2)
plt.xlabel('Nombre de features')
plt.ylabel('Accuracy')
plt.title('Comparaison Performance vs Nombre de Features (exemple)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

out = os.path.join(plots_dir, 'feature_count_comparison.png')
plt.savefig(out, dpi=150)
print('Saved', out)
plt.close()