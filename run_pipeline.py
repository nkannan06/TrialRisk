"""
run_pipeline.py
End-to-end pipeline: pull data → engineer features → train model → evaluate.
Run from repo root: python run_pipeline.py
"""

import os
import time
import pandas as pd
from datetime import datetime

print("=" * 60)
print("  TrialRisk — Clinical Trial Termination Prediction")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ── Step 1: Pull data ────────────────────────────────────────────────────────
import fileinput

file_path = '/content/TrialRisk/run_pipeline.py'

with fileinput.FileInput(file_path, inplace=True) as file:
    for line in file:
        print(line.replace('from src.model import train', 'from model import train'), end='')

t0 = time.time()
df = pull_all(max_per_status=5000, out_dir="data")
print(f"  Done in {time.time()-t0:.1f}s  |  {len(df)} trials")

# ── Step 2: Feature check ────────────────────────────────────────────────────
print("\n[2/3] Engineering features...")
from src.features import build_features
X, y, cols = build_features(df)
print(f"  Feature matrix: {X.shape}")
print(f"  Termination rate: {y.mean():.2%}")

# ── Step 3: Train + evaluate ─────────────────────────────────────────────────
print("\n[3/3] Training model...")
from src.model import train
model, metrics = train(df)

print("\n" + "=" * 60)
print("  FINAL RESULTS")
print("=" * 60)
print(f"  Test AUC:          {metrics['test_auc']}")
print(f"  Test Avg Precision: {metrics['test_ap']}")
print(f"  CV AUC:            {metrics['cv_auc_mean']} ± {metrics['cv_auc_std']}")
print("\n  Outputs saved to /outputs/")
print("    model.pkl           — trained XGBoost model")
print("    metrics.json        — evaluation metrics")
print("    risk_scores.csv     — scored test set")
print("    roc_pr.png          — ROC + PR curves")
print("    feature_importance.png")
print("    shap_beeswarm.png")
print("    shap_waterfall_highrisk.png")
print("=" * 60)
print("\nTo score a new trial by NCT ID:")
print("  python src/score_trial.py NCT04280705")
