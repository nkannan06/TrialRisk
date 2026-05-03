# TrialRisk 🧪

> **Predicting clinical trial termination at the moment of registration — before a single patient is enrolled.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?style=flat)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable-059669?style=flat)](https://shap.readthedocs.io)
[![Data](https://img.shields.io/badge/Data-ClinicalTrials.gov-0EA5E9?style=flat)](https://clinicaltrials.gov)
[![License](https://img.shields.io/badge/License-MIT-gray?style=flat)](LICENSE)

---

## The Problem

Clinical trial termination is a $50B+ annual problem. ~20% of all registered interventional trials are terminated before completion — wasting years of research, billions in funding, and most critically, patient participation in trials that produce no usable evidence.

Existing tools focus on *managing* active trials. **No tool predicts termination risk before the trial begins**, at registration time, when course-correction is still possible.

TrialRisk changes that.

---

## What It Does

TrialRisk trains a gradient-boosted model on 10,000+ trials pulled from the public ClinicalTrials.gov API. Given a trial's registration metadata (design, phase, sponsor, enrollment target, eligibility criteria), it outputs:

- **Risk Score (0–100)**: probability × 100 that the trial will be terminated
- **Risk Tier**: Low / Moderate / High / Critical
- **SHAP explanations**: the top factors driving the specific trial's risk

The key insight: **all features are available at time of registration** — the model is a forward-looking screen, not a retrospective audit.

---

## Novel Contributions

| What's new | Why it matters |
|---|---|
| Registration-time prediction | Intervene before waste occurs, not after |
| SHAP-powered per-trial explanations | Actionable, not just a black box score |
| Eligibility criteria complexity as a feature | Proxy for trial overdesign |
| Sponsor class × phase interaction | NIH early-phase and industry late-phase have different failure modes |
| Live scoring via NCT ID | Real-world utility, not just a notebook |

---

## Architecture

```
ClinicalTrials.gov API v2
         │
         ▼
  src/data_pull.py          ← Pulls TERMINATED + COMPLETED interventional trials
         │
         ▼
  src/features.py           ← 36 engineered features (phase, design, complexity, eligibility)
         │
         ▼
  src/model.py              ← XGBoost + 5-fold CV + SHAP explainability
         │
         ▼
  outputs/
    model.pkl               ← Trained artifact
    risk_scores.csv         ← Scored test set with risk tiers
    shap_beeswarm.png       ← Global feature impact
    shap_waterfall_*.png    ← Per-trial explanation
    roc_pr.png              ← ROC + Precision-Recall curves
    metrics.json            ← Evaluation summary
```

---

## Quickstart

```bash
git clone https://github.com/yourusername/TrailRisk.git
cd TrialRisk
pip install -r requirements.txt

# Full pipeline: pull → feature engineer → train → evaluate
python run_pipeline.py

# Score any trial by NCT ID (live API call)
python src/score_trial.py NCT04280705
```

---

## Example Output

```
==================================================
Trial:      A Phase 3 Study of Drug X in Metastatic NSCLC
Status:     TERMINATED
Risk Score: 78.4/100  →  🔴 Critical

Top 5 risk drivers:
  phase_num                       ▲ increases risk  (SHAP=+0.412)
  log_enrollment                  ▲ increases risk  (SHAP=+0.287)
  is_industry                     ▼ reduces risk    (SHAP=-0.201)
  n_primary_outcomes              ▲ increases risk  (SHAP=+0.184)
  planned_duration_months         ▲ increases risk  (SHAP=+0.163)
==================================================
```

---

## Feature Engineering

36 features across 6 categories:

| Category | Features |
|---|---|
| **Phase** | `phase_num`, `is_early_phase` |
| **Sponsor** | `sponsor_code`, `is_industry`, `is_nih` |
| **Enrollment** | `log_enrollment`, `is_large_trial`, `enrollment_is_estimated` |
| **Design** | `masking_level`, `is_randomized`, `n_arms`, `intervention_model` |
| **Complexity** | `n_conditions`, `n_outcomes`, `n_countries`, `log_eligibility_len` |
| **Eligibility** | `min_age_years`, `accepts_healthy`, `includes_children`, `sex_all` |

---

## Model Performance

| Metric | Value |
|---|---|
| Test AUC | ~0.78 |
| Test Avg Precision | ~0.55 |
| CV AUC (5-fold) | ~0.77 ± 0.01 |
| Brier Score | ~0.14 |

> *Baseline termination rate: ~18% — meaningful lift over random.*

---

## Why This Is Hard

- Class imbalance (~18% termination rate) handled via stratified splits
- Right-censoring: some "completed" trials may eventually be terminated
- Feature sparsity: many trials have missing enrollment or date fields
- Causal ambiguity: some features (e.g. `why_stopped`) can only be observed post-facto and are excluded

---

## Limitations & Future Work

- [ ] NLP on `brief_summary` and `eligibility_criteria` text (current model uses only length as proxy)
- [ ] Temporal validation: train on pre-2018 trials, test on 2019+ to avoid leakage
- [ ] Disease-area stratification: oncology vs. CNS vs. infectious disease have different base rates
- [ ] Streamlit dashboard for non-technical users
- [ ] Integration with ClinicalTrials.gov RSS feed for real-time screening

---

## Data

All data is pulled from the public [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api) — no authentication required, freely available under NIH open data policy.

---

## License

MIT
