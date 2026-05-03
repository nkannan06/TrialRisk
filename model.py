"""
model.py
Trains XGBoost classifier to predict clinical trial termination at registration time.
Outputs: model artifact, evaluation metrics, SHAP plots, risk score on held-out set.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import pickle
import os
import json
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, brier_score_loss,
    RocCurveDisplay, PrecisionRecallDisplay,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb

from features import build_features


# ── Config ───────────────────────────────────────────────────────────────────
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
OUTPUT_DIR = "outputs"

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "eval_metric": "aucpr",
    "early_stopping_rounds": 40,
    "random_state": SEED,
    "n_jobs": -1,
    "use_label_encoder": False,
}

PALETTE = {
    "emerald":  "#059669",
    "bg":       "#0f172a",
    "slate":    "#1e293b",
    "text":     "#f8fafc",
    "muted":    "#94a3b8",
    "red":      "#ef4444",
    "amber":    "#f59e0b",
}


def set_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["slate"],
        "axes.edgecolor":    PALETTE["muted"],
        "axes.labelcolor":   PALETTE["text"],
        "xtick.color":       PALETTE["muted"],
        "ytick.color":       PALETTE["muted"],
        "text.color":        PALETTE["text"],
        "grid.color":        "#334155",
        "grid.linestyle":    "--",
        "grid.linewidth":    0.5,
        "font.family":       "monospace",
    })


def train(df: pd.DataFrame):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_style()

    X, y, feature_names = build_features(df)
    print(f"Dataset: {X.shape[0]} trials | Termination rate: {y.mean():.2%}")

    # ── Train / test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    # ── CV on train set ──────────────────────────────────────────────────────
    print("\nRunning 5-fold CV...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    cv_aucs, cv_aps = [], []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        m = xgb.XGBClassifier(**XGB_PARAMS)
        m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)

        preds = m.predict_proba(Xval)[:, 1]
        cv_aucs.append(roc_auc_score(yval, preds))
        cv_aps.append(average_precision_score(yval, preds))
        print(f"  Fold {fold}: AUC={cv_aucs[-1]:.4f}  AP={cv_aps[-1]:.4f}")

    print(f"\nCV AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
    print(f"CV AP:  {np.mean(cv_aps):.4f} ± {np.std(cv_aps):.4f}")

    # ── Final model ──────────────────────────────────────────────────────────
    print("\nTraining final model on full train set...")
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100,
    )

    # ── Test evaluation ──────────────────────────────────────────────────────
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "test_auc":    round(roc_auc_score(y_test, y_prob), 4),
        "test_ap":     round(average_precision_score(y_test, y_prob), 4),
        "test_brier":  round(brier_score_loss(y_test, y_prob), 4),
        "cv_auc_mean": round(np.mean(cv_aucs), 4),
        "cv_auc_std":  round(np.std(cv_aucs), 4),
        "cv_ap_mean":  round(np.mean(cv_aps), 4),
        "n_train":     len(X_train),
        "n_test":      len(X_test),
        "termination_rate": round(float(y.mean()), 4),
        "timestamp":   datetime.now().isoformat(),
    }
    print(f"\nTest AUC: {metrics['test_auc']}")
    print(f"Test AP:  {metrics['test_ap']}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Completed", "Terminated"]))

    with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Save model ───────────────────────────────────────────────────────────
    with open(f"{OUTPUT_DIR}/model.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_names": feature_names}, f)
    print(f"Model saved → {OUTPUT_DIR}/model.pkl")

    # ── ROC + PR curves ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("TrialRisk · Model Performance", color=PALETTE["text"], fontsize=14, fontweight="bold", y=1.01)

    RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[0], color=PALETTE["emerald"], name=f"XGBoost (AUC={metrics['test_auc']})")
    axes[0].plot([0,1],[0,1], "--", color=PALETTE["muted"], alpha=0.5)
    axes[0].set_title("ROC Curve", color=PALETTE["text"])
    axes[0].set_facecolor(PALETTE["slate"])

    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=axes[1], color=PALETTE["emerald"], name=f"XGBoost (AP={metrics['test_ap']})")
    axes[1].set_title("Precision-Recall Curve", color=PALETTE["text"])
    axes[1].set_facecolor(PALETTE["slate"])

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_pr.png", dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()

    # ── Feature importance ───────────────────────────────────────────────────
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    top_n = importances.tail(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["slate"])
    colors = [PALETTE["emerald"] if v >= top_n.quantile(0.7) else PALETTE["muted"] for v in top_n.values]
    ax.barh(top_n.index, top_n.values, color=colors, edgecolor="none")
    ax.set_title("Top 20 Feature Importances (Gain)", color=PALETTE["text"], fontsize=13)
    ax.set_xlabel("Importance", color=PALETTE["muted"])
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print("Feature importance plot saved.")

    # ── SHAP ─────────────────────────────────────────────────────────────────
    print("\nComputing SHAP values (test set)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # Summary beeswarm
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor(PALETTE["bg"])
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title("SHAP Feature Impact on Trial Termination Risk", color=PALETTE["text"], fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_beeswarm.png", dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()

    # Waterfall for highest-risk trial in test set
    highest_risk_idx = np.argmax(y_prob)
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(PALETTE["bg"])
    shap.plots.waterfall(shap_values[highest_risk_idx], max_display=15, show=False)
    plt.title(f"SHAP Explanation — Highest Risk Trial (score={y_prob[highest_risk_idx]:.2f})", color=PALETTE["text"], fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_waterfall_highrisk.png", dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print("SHAP plots saved.")

    # ── Risk score output ────────────────────────────────────────────────────
    test_results = X_test.copy()
    test_results["risk_score"] = (y_prob * 100).round(1)
    test_results["risk_tier"] = pd.cut(
        y_prob,
        bins=[0, 0.2, 0.4, 0.65, 1.0],
        labels=["Low", "Moderate", "High", "Critical"],
    )
    test_results["true_label"] = y_test.values
    test_results.to_csv(f"{OUTPUT_DIR}/risk_scores.csv", index=False)
    print(f"\nRisk scores saved → {OUTPUT_DIR}/risk_scores.csv")
    print(test_results["risk_tier"].value_counts())

    return model, metrics


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw_trials.csv"
    df = pd.read_csv(data_path)
    model, metrics = train(df)
    print("\nDone. All outputs in /outputs/")
