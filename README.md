# CyberFusion-IDS

A small collection of intrusion-detection model scripts and utilities for various datasets (UNSW‑NB15, TON‑IoT, CICIDS2017, etc.). This repository contains model training and analysis scripts organized into logical folders.

## Quick start

1. (Optional) Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies (if you have a `requirements.txt`):

   ```bash
   pip install -r requirements.txt
   ```
3. Run a script, for example:

   ```bash
   python models/decision_tree/decision_tree.py
   ```

## Folder structure

- `models/decision_tree/` — decision tree model scripts
- `models/isolation_forest/` — isolation forest scripts
- `models/random_forest/` — random forest scripts
- `scripts/` — helper scripts and utilities
- `package/` — package files (kept for packaging)

> Note: Files were **moved** into these folders (nothing was deleted).
