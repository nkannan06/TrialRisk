"""
generate_synthetic.py
Generates realistic synthetic clinical trial data for local testing.
Mirrors the schema of ClinicalTrials.gov v2 API output.
Run: python generate_synthetic.py  →  data/raw_trials.csv
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

N_TERMINATED = 2500
N_COMPLETED = 7500

PHASES = ["EARLY_PHASE1", "PHASE1", "PHASE1|PHASE2", "PHASE2", "PHASE2|PHASE3", "PHASE3", "PHASE4", "NA"]
PHASE_WEIGHTS_TERM = [0.06, 0.20, 0.10, 0.25, 0.10, 0.18, 0.05, 0.06]
PHASE_WEIGHTS_COMP = [0.03, 0.12, 0.07, 0.22, 0.08, 0.28, 0.12, 0.08]

SPONSOR_CLASSES = ["INDUSTRY", "NIH", "FED", "OTHER_GOV", "INDIV", "NETWORK", "OTHER"]
SPONSOR_WEIGHTS_TERM = [0.35, 0.18, 0.05, 0.08, 0.15, 0.09, 0.10]
SPONSOR_WEIGHTS_COMP = [0.42, 0.22, 0.06, 0.07, 0.08, 0.08, 0.07]

ALLOCATIONS = ["RANDOMIZED", "NON_RANDOMIZED", "NA"]
MASKINGS = ["NONE", "SINGLE", "DOUBLE", "TRIPLE", "QUADRUPLE"]
PURPOSES = ["TREATMENT", "PREVENTION", "DIAGNOSTIC", "SUPPORTIVE_CARE", "SCREENING", "BASIC_SCIENCE", "DEVICE_FEASIBILITY"]
MODELS = ["PARALLEL_ASSIGNMENT", "CROSSOVER_ASSIGNMENT", "FACTORIAL_ASSIGNMENT", "SINGLE_GROUP_ASSIGNMENT"]
INTERVENTION_TYPES = ["DRUG", "BIOLOGICAL", "DEVICE", "PROCEDURE", "BEHAVIORAL", "DIETARY_SUPPLEMENT", "OTHER"]
CONDITIONS = ["Carcinoma", "Diabetes Mellitus", "Heart Failure", "COVID-19", "Hypertension",
              "Breast Cancer", "Non-small Cell Lung Carcinoma", "Major Depressive Disorder",
              "Alzheimer Disease", "Multiple Sclerosis", "Rheumatoid Arthritis", "Obesity",
              "Stroke", "Leukemia", "Chronic Kidney Disease"]
COUNTRIES_COMMON = ["United States", "Germany", "United Kingdom", "France", "Canada", "Australia",
                    "Japan", "China", "Italy", "Spain", "Netherlands", "Belgium", "Sweden", "Israel"]
STD_AGES = ["CHILD", "ADULT", "OLDER_ADULT"]


def rand_date(start_year=2000, end_year=2022) -> str:
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).strftime("%Y-%m-%d")


def make_trial(status: str) -> dict:
    is_term = status == "TERMINATED"

    phase = random.choices(PHASES, weights=PHASE_WEIGHTS_TERM if is_term else PHASE_WEIGHTS_COMP)[0]
    sponsor = random.choices(SPONSOR_CLASSES, weights=SPONSOR_WEIGHTS_TERM if is_term else SPONSOR_WEIGHTS_COMP)[0]

    # enrollment: terminated trials tend to have higher targets (overambitious) or very low (underpowered)
    if is_term:
        enrollment = int(np.abs(rng.normal(350, 600))) + 10
    else:
        enrollment = int(np.abs(rng.normal(280, 400))) + 10

    start_date = rand_date(2000, 2018)
    duration_months = max(6, rng.normal(36 if is_term else 42, 24))
    end_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=int(duration_months * 30.44))).strftime("%Y-%m-%d")

    n_arms = random.choices([1, 2, 3, 4, 5], weights=[0.25, 0.45, 0.15, 0.10, 0.05])[0]
    n_conditions = random.choices([1, 2, 3, 4], weights=[0.60, 0.25, 0.10, 0.05])[0]
    n_interventions = random.choices([1, 2, 3, 4], weights=[0.50, 0.30, 0.12, 0.08])[0]
    n_primary = random.choices([1, 2, 3, 4, 5], weights=[0.50, 0.25, 0.12, 0.08, 0.05])[0]
    n_secondary = random.choices([0, 1, 2, 3, 5, 8, 12], weights=[0.10, 0.20, 0.25, 0.20, 0.12, 0.08, 0.05])[0]

    # eligibility complexity: terminated trials often have more complex criteria
    elig_len = int(max(100, rng.normal(3500 if is_term else 2800, 1500)))

    n_countries = random.choices([1, 2, 3, 5, 10, 20], weights=[0.45, 0.15, 0.12, 0.12, 0.10, 0.06])[0]
    countries = random.sample(COUNTRIES_COMMON, min(n_countries, len(COUNTRIES_COMMON)))

    int_types = random.sample(INTERVENTION_TYPES, min(n_interventions, len(INTERVENTION_TYPES)))
    conditions = random.sample(CONDITIONS, min(n_conditions, len(CONDITIONS)))

    std_ages_sample = random.choices(
        [["ADULT"], ["ADULT", "OLDER_ADULT"], ["CHILD", "ADULT"], ["CHILD", "ADULT", "OLDER_ADULT"]],
        weights=[0.45, 0.30, 0.15, 0.10]
    )[0]

    min_age_years = random.choices([0, 18, 21, 65], weights=[0.10, 0.65, 0.15, 0.10])[0]
    max_age_years = random.choices([18, 65, 80, 120], weights=[0.05, 0.20, 0.35, 0.40])[0]
    max_age_years = max(min_age_years + 5, max_age_years)

    return {
        "nct_id": f"NCT{random.randint(10000000, 99999999)}",
        "title": f"A {phase} Study of Drug in {random.choice(CONDITIONS)}",
        "overall_status": status,
        "why_stopped": "Enrollment difficulties" if is_term else "",
        "start_date": start_date,
        "primary_completion_date": end_date,
        "study_first_submit_date": start_date,
        "phase": phase,
        "sponsor_class": sponsor,
        "enrollment_count": enrollment,
        "enrollment_type": random.choices(["ACTUAL", "ESTIMATED"], weights=[0.55, 0.45])[0],
        "allocation": random.choices(ALLOCATIONS, weights=[0.70, 0.20, 0.10])[0],
        "masking": random.choice(MASKINGS),
        "primary_purpose": random.choice(PURPOSES),
        "intervention_model": random.choice(MODELS),
        "n_arms": n_arms,
        "n_conditions": n_conditions,
        "conditions": "|".join(conditions),
        "n_intervention_types": len(int_types),
        "intervention_types": "|".join(int_types),
        "n_interventions": n_interventions,
        "n_primary_outcomes": n_primary,
        "n_secondary_outcomes": n_secondary,
        "eligibility_criteria_len": elig_len,
        "healthy_volunteers": random.choices(["Yes", "No"], weights=[0.25, 0.75])[0],
        "min_age": f"{min_age_years} Years",
        "max_age": f"{max_age_years} Years",
        "sex": random.choices(["ALL", "MALE", "FEMALE"], weights=[0.75, 0.12, 0.13])[0],
        "std_ages": "|".join(std_ages_sample),
        "n_countries": n_countries,
        "countries": "|".join(countries),
        "has_results": random.choices([True, False], weights=[0.35 if not is_term else 0.10, 0.65])[0],
        "brief_summary_len": int(max(50, rng.normal(800, 300))),
    }


def generate(n_terminated=N_TERMINATED, n_completed=N_COMPLETED, out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    records = []
    print(f"Generating {n_terminated} TERMINATED trials...")
    for _ in range(n_terminated):
        records.append(make_trial("TERMINATED"))
    print(f"Generating {n_completed} COMPLETED trials...")
    for _ in range(n_completed):
        records.append(make_trial("COMPLETED"))

    df = pd.DataFrame(records).sample(frac=1, random_state=SEED).reset_index(drop=True)
    out_path = os.path.join(out_dir, "raw_trials.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} synthetic trials → {out_path}")
    print(df["overall_status"].value_counts())
    return df


if __name__ == "__main__":
    df = generate()
