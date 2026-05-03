"""
score_trial.py
Score a single trial by NCT ID using the trained model.
Usage: python src/score_trial.py NCT04280705
"""

import sys
import json
import pickle
import requests
import pandas as pd
import numpy as np
import shap

from features import build_features


BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


def fetch_single(nct_id: str) -> dict:
    """Fetch a single study from ClinicalTrials.gov."""
    resp = requests.get(
        f"{BASE_URL}/{nct_id}",
        params={"format": "json"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def flatten_single(study: dict) -> dict:
    """Re-use flatten logic from data_pull."""
    import importlib.util, os
    spec = importlib.util.spec_from_file_location("data_pull", os.path.join(os.path.dirname(__file__), "data_pull.py"))
    dp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dp)
    return dp.flatten_study(study)


def score(nct_id: str, model_path: str = "outputs/model.pkl") -> dict:
    print(f"Fetching {nct_id}...")
    study = fetch_single(nct_id)
    row = flatten_single(study)
    df = pd.DataFrame([row])

    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    feature_names = artifact["feature_names"]

    X, _, _ = build_features(df)
    X = X[feature_names]

    prob = model.predict_proba(X)[0, 1]
    risk_score = round(prob * 100, 1)

    if prob < 0.20:
        tier = "🟢 Low"
    elif prob < 0.40:
        tier = "🟡 Moderate"
    elif prob < 0.65:
        tier = "🟠 High"
    else:
        tier = "🔴 Critical"

    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(X)
    sv = shap_vals[0].values
    top_idx = np.argsort(np.abs(sv))[::-1][:5]
    top_drivers = [
        {"feature": feature_names[i], "shap": round(float(sv[i]), 4)}
        for i in top_idx
    ]

    result = {
        "nct_id": nct_id,
        "title": row.get("title", ""),
        "status": row.get("overall_status", ""),
        "risk_score": risk_score,
        "risk_tier": tier,
        "top_risk_drivers": top_drivers,
    }

    print(f"\n{'='*50}")
    print(f"Trial:      {result['title'][:70]}")
    print(f"Status:     {result['status']}")
    print(f"Risk Score: {risk_score}/100  →  {tier}")
    print(f"\nTop 5 risk drivers:")
    for d in top_drivers:
        direction = "▲ increases risk" if d["shap"] > 0 else "▼ reduces risk"
        print(f"  {d['feature']:<30}  {direction}  (SHAP={d['shap']:+.3f})")
    print("="*50)

    return result


if __name__ == "__main__":
    nct_id = sys.argv[1] if len(sys.argv) > 1 else "NCT04280705"
    result = score(nct_id)
