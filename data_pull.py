"""
data_pull.py
Pulls TERMINATED and COMPLETED interventional trials from ClinicalTrials.gov API v2.
Saves raw data to data/raw_trials.csv
"""

import requests
import pandas as pd
import time
import os
import json
from datetime import datetime

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

FIELDS = [
    "NCTId",
    "OverallStatus",
    "BriefTitle",
    "Phase",
    "LeadSponsorClass",
    "EnrollmentCount",
    "EnrollmentType",
    "StartDate",
    "PrimaryCompletionDate",
    "StudyFirstSubmitDate",
    "WhyStopped",
    "DesignAllocation",
    "DesignMasking",
    "DesignPrimaryPurpose",
    "DesignInterventionModel",
    "NumberOfArms",
    "EligibilityCriteria",
    "HealthyVolunteers",
    "MinimumAge",
    "MaximumAge",
    "Sex",
    "StdAge",
    "LocationCountry",
    "Condition",
    "InterventionType",
    "InterventionName",
    "PrimaryOutcomeMeasure",
    "SecondaryOutcomeMeasure",
    "OrgStudyId",
    "BriefSummary",
    "HasResults",
]

TARGET_STATUSES = ["TERMINATED", "COMPLETED"]


def fetch_trials(status: str, max_records: int = 5000) -> list[dict]:
    """Fetch trials of a given status from ClinicalTrials.gov API v2."""
    records = []
    next_page_token = None
    page_size = 1000

    print(f"  Fetching {status} trials...")

    while len(records) < max_records:
        params = {
            "query.term": f"AREA[OverallStatus]{status} AND AREA[StudyType]INTERVENTIONAL",
            "fields": ",".join(FIELDS),
            "pageSize": min(page_size, max_records - len(records)),
            "format": "json",
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Error fetching page: {e}")
            break

        studies = data.get("studies", [])
        if not studies:
            break

        for study in studies:
            records.append(flatten_study(study))

        next_page_token = data.get("nextPageToken")
        print(f"    Fetched {len(records)} {status} records so far...")

        if not next_page_token:
            break

        time.sleep(0.3)  # be nice to the API

    return records


def flatten_study(study: dict) -> dict:
    """Flatten nested ClinicalTrials.gov v2 JSON into a single row."""
    proto = study.get("protocolSection", {})
    id_mod = proto.get("identificationModule", {})
    status_mod = proto.get("statusModule", {})
    design_mod = proto.get("designModule", {})
    eligibility_mod = proto.get("eligibilityModule", {})
    contacts_mod = proto.get("contactsLocationsModule", {})
    outcomes_mod = proto.get("outcomesModule", {})
    arms_mod = proto.get("armsInterventionsModule", {})
    desc_mod = proto.get("descriptionModule", {})
    sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
    results_section = study.get("resultsSection", {})

    # locations
    locations = contacts_mod.get("locations", [])
    countries = list({loc.get("country", "") for loc in locations if loc.get("country")})

    # interventions
    interventions = arms_mod.get("interventions", [])
    intervention_types = list({i.get("type", "") for i in interventions})
    intervention_names = [i.get("name", "") for i in interventions]

    # conditions
    conditions = proto.get("conditionsModule", {}).get("conditions", [])

    # outcomes
    primary_outcomes = [o.get("measure", "") for o in outcomes_mod.get("primaryOutcomes", [])]
    secondary_outcomes = [o.get("measure", "") for o in outcomes_mod.get("secondaryOutcomes", [])]

    # phases
    phases = design_mod.get("phases", [])

    # arms
    arms = arms_mod.get("armGroups", [])

    # eligibility text
    elig_criteria = eligibility_mod.get("eligibilityCriteria", "")

    # std ages
    std_ages = eligibility_mod.get("stdAges", [])

    return {
        "nct_id": id_mod.get("nctId", ""),
        "title": id_mod.get("briefTitle", ""),
        "overall_status": status_mod.get("overallStatus", ""),
        "why_stopped": status_mod.get("whyStopped", ""),
        "start_date": status_mod.get("startDateStruct", {}).get("date", ""),
        "primary_completion_date": status_mod.get("primaryCompletionDateStruct", {}).get("date", ""),
        "study_first_submit_date": status_mod.get("studyFirstSubmitDate", ""),
        "phase": "|".join(phases) if phases else "NA",
        "sponsor_class": sponsor_mod.get("leadSponsor", {}).get("class", ""),
        "enrollment_count": design_mod.get("enrollmentInfo", {}).get("count", None),
        "enrollment_type": design_mod.get("enrollmentInfo", {}).get("type", ""),
        "allocation": design_mod.get("designInfo", {}).get("allocation", ""),
        "masking": design_mod.get("designInfo", {}).get("maskingInfo", {}).get("masking", ""),
        "primary_purpose": design_mod.get("designInfo", {}).get("primaryPurpose", ""),
        "intervention_model": design_mod.get("designInfo", {}).get("interventionModel", ""),
        "n_arms": len(arms),
        "n_conditions": len(conditions),
        "conditions": "|".join(conditions[:5]),
        "n_intervention_types": len(intervention_types),
        "intervention_types": "|".join(intervention_types),
        "n_interventions": len(interventions),
        "n_primary_outcomes": len(primary_outcomes),
        "n_secondary_outcomes": len(secondary_outcomes),
        "eligibility_criteria_len": len(elig_criteria),
        "healthy_volunteers": eligibility_mod.get("healthyVolunteers", ""),
        "min_age": eligibility_mod.get("minimumAge", ""),
        "max_age": eligibility_mod.get("maximumAge", ""),
        "sex": eligibility_mod.get("sex", ""),
        "std_ages": "|".join(std_ages),
        "n_countries": len(countries),
        "countries": "|".join(countries[:10]),
        "has_results": bool(results_section),
        "brief_summary_len": len(desc_mod.get("briefSummary", "")),
    }


def pull_all(max_per_status: int = 5000, out_dir: str = "data") -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    all_records = []

    for status in TARGET_STATUSES:
        records = fetch_trials(status, max_records=max_per_status)
        all_records.extend(records)
        print(f"  Done: {len(records)} {status} trials")

    df = pd.DataFrame(all_records)
    out_path = os.path.join(out_dir, "raw_trials.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} total records → {out_path}")
    return df


if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Pulling ClinicalTrials.gov data...")
    df = pull_all(max_per_status=5000)
    print(df["overall_status"].value_counts())
    print(df.head(3).to_string())
