"""
features.py
Transforms raw ClinicalTrials.gov data into ML-ready features.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime


# ── Phase encoding ──────────────────────────────────────────────────────────
PHASE_ORDER = {
    "NA": 0,
    "EARLY_PHASE1": 1,
    "PHASE1": 2,
    "PHASE1|PHASE2": 2.5,
    "PHASE2": 3,
    "PHASE2|PHASE3": 3.5,
    "PHASE3": 4,
    "PHASE4": 5,
}

SPONSOR_MAP = {
    "INDUSTRY": 0,
    "NIH": 1,
    "FED": 2,
    "OTHER_GOV": 3,
    "INDIV": 4,
    "NETWORK": 5,
    "OTHER": 6,
}

MASKING_ORDER = {
    "NONE": 0,           # open label
    "SINGLE": 1,
    "DOUBLE": 2,
    "TRIPLE": 3,
    "QUADRUPLE": 4,
}


def parse_age_to_years(age_str: str) -> float:
    """Convert age string like '18 Years' or '6 Months' to float years."""
    if not age_str or pd.isna(age_str):
        return np.nan
    age_str = str(age_str).strip()
    m = re.match(r"([\d.]+)\s*(year|month|week|day)", age_str, re.IGNORECASE)
    if not m:
        return np.nan
    val, unit = float(m.group(1)), m.group(2).lower()
    if "month" in unit:
        return val / 12
    elif "week" in unit:
        return val / 52
    elif "day" in unit:
        return val / 365
    return val


def parse_date(date_str: str) -> pd.Timestamp:
    if not date_str or pd.isna(date_str):
        return pd.NaT
    for fmt in ["%Y-%m-%d", "%B %d, %Y", "%Y-%m", "%B %Y"]:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except Exception:
            pass
    try:
        return pd.to_datetime(date_str)
    except Exception:
        return pd.NaT


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer all features. Returns feature matrix X and label series y."""
    df = df.copy()

    # ── Target ───────────────────────────────────────────────────────────────
    df["terminated"] = (df["overall_status"] == "TERMINATED").astype(int)

    # ── Phase ────────────────────────────────────────────────────────────────
    df["phase_num"] = df["phase"].map(PHASE_ORDER).fillna(0)
    df["is_early_phase"] = (df["phase_num"] <= 2).astype(int)

    # ── Sponsor ──────────────────────────────────────────────────────────────
    df["sponsor_code"] = df["sponsor_class"].map(SPONSOR_MAP).fillna(6)
    df["is_industry"] = (df["sponsor_class"] == "INDUSTRY").astype(int)
    df["is_nih"] = (df["sponsor_class"] == "NIH").astype(int)

    # ── Enrollment ───────────────────────────────────────────────────────────
    df["enrollment_count"] = pd.to_numeric(df["enrollment_count"], errors="coerce")
    df["log_enrollment"] = np.log1p(df["enrollment_count"])
    df["is_large_trial"] = (df["enrollment_count"] > 1000).astype(int)
    df["enrollment_is_estimated"] = (df["enrollment_type"] == "ESTIMATED").astype(int)

    # ── Duration ─────────────────────────────────────────────────────────────
    df["_start"] = df["start_date"].apply(parse_date)
    df["_end"] = df["primary_completion_date"].apply(parse_date)
    df["planned_duration_months"] = (
        (df["_end"] - df["_start"]).dt.days / 30.44
    ).clip(lower=0)
    df["is_long_trial"] = (df["planned_duration_months"] > 48).astype(int)

    # ── Design features ──────────────────────────────────────────────────────
    df["masking_level"] = (
        df["masking"]
        .str.upper()
        .map(MASKING_ORDER)
        .fillna(0)
    )
    df["is_randomized"] = (
        df["allocation"].str.upper().str.contains("RANDOMIZED", na=False)
    ).astype(int)
    df["is_treatment"] = (
        df["primary_purpose"].str.upper().str.contains("TREATMENT", na=False)
    ).astype(int)
    df["is_parallel"] = (
        df["intervention_model"].str.upper().str.contains("PARALLEL", na=False)
    ).astype(int)

    # ── Complexity features ──────────────────────────────────────────────────
    df["n_arms"] = pd.to_numeric(df["n_arms"], errors="coerce").fillna(1)
    df["n_conditions"] = pd.to_numeric(df["n_conditions"], errors="coerce").fillna(1)
    df["n_interventions"] = pd.to_numeric(df["n_interventions"], errors="coerce").fillna(1)
    df["n_primary_outcomes"] = pd.to_numeric(df["n_primary_outcomes"], errors="coerce").fillna(1)
    df["n_secondary_outcomes"] = pd.to_numeric(df["n_secondary_outcomes"], errors="coerce").fillna(0)
    df["n_countries"] = pd.to_numeric(df["n_countries"], errors="coerce").fillna(1)
    df["eligibility_criteria_len"] = pd.to_numeric(df["eligibility_criteria_len"], errors="coerce").fillna(0)
    df["is_multicondition"] = (df["n_conditions"] > 1).astype(int)
    df["is_multinational"] = (df["n_countries"] > 1).astype(int)
    df["outcome_burden"] = df["n_primary_outcomes"] + 0.5 * df["n_secondary_outcomes"]
    df["log_eligibility_len"] = np.log1p(df["eligibility_criteria_len"])

    # ── Drug type ────────────────────────────────────────────────────────────
    df["has_drug"] = df["intervention_types"].str.contains("DRUG", na=False).astype(int)
    df["has_biologic"] = df["intervention_types"].str.contains("BIOLOGICAL", na=False).astype(int)
    df["has_device"] = df["intervention_types"].str.contains("DEVICE", na=False).astype(int)
    df["has_procedure"] = df["intervention_types"].str.contains("PROCEDURE", na=False).astype(int)

    # ── Eligibility ──────────────────────────────────────────────────────────
    df["accepts_healthy"] = (
        df["healthy_volunteers"].str.upper().str.contains("YES", na=False)
    ).astype(int)
    df["min_age_years"] = df["min_age"].apply(parse_age_to_years)
    df["max_age_years"] = df["max_age"].apply(parse_age_to_years)
    df["age_range_years"] = (df["max_age_years"] - df["min_age_years"]).clip(lower=0)
    df["includes_children"] = df["std_ages"].str.contains("CHILD", na=False).astype(int)
    df["includes_elderly"] = df["std_ages"].str.contains("OLDER_ADULT", na=False).astype(int)
    df["sex_all"] = (df["sex"].str.upper() == "ALL").astype(int)

    # ── Has results ──────────────────────────────────────────────────────────
    df["has_results"] = df["has_results"].astype(int)

    # ── Final feature list ───────────────────────────────────────────────────
    FEATURE_COLS = [
        "phase_num", "is_early_phase",
        "sponsor_code", "is_industry", "is_nih",
        "log_enrollment", "is_large_trial", "enrollment_is_estimated",
        "planned_duration_months", "is_long_trial",
        "masking_level", "is_randomized", "is_treatment", "is_parallel",
        "n_arms", "n_conditions", "n_interventions",
        "n_primary_outcomes", "n_secondary_outcomes", "outcome_burden",
        "n_countries", "log_eligibility_len",
        "is_multicondition", "is_multinational",
        "has_drug", "has_biologic", "has_device", "has_procedure",
        "accepts_healthy", "min_age_years", "max_age_years",
        "age_range_years", "includes_children", "includes_elderly", "sex_all",
        "brief_summary_len",
    ]

    X = df[FEATURE_COLS].copy()
    y = df["terminated"]

    # fill remaining NaNs with median
    X = X.fillna(X.median(numeric_only=True))

    return X, y, FEATURE_COLS


if __name__ == "__main__":
    import sys
    df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/raw_trials.csv")
    X, y, cols = build_features(df)
    print(f"Feature matrix: {X.shape}")
    print(f"Termination rate: {y.mean():.2%}")
    print("\nFeature columns:")
    for c in cols:
        print(f"  {c}: {X[c].dtype}  nulls={X[c].isna().sum()}")
