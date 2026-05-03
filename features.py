import numpy as np
import pandas as pd

NUM_PHASES = {
    "EARLY PHASE 1": 0,
    "PHASE 1": 1,
    "PHASE 1|PHASE 2": 1.5,
    "PHASE 2": 2,
    "PHASE 2|PHASE 3": 2.5,
    "PHASE 3": 3,
    "PHASE 4": 4,
}

# Define all possible categorical values for one-hot encoding, derived from the training data.
# This list would ideally come from a configuration or saved artifacts of the training process.
# For now, I'll make educated guesses based on common values and the missing features list.
ALL_STUDY_TYPE_CATEGORIES = ['Interventional', 'Observational', 'Expanded Access']
ALL_OVERALL_STATUS_FIRST_CATEGORIES = ['Recruiting', 'Not yet recruiting', 'Active, not recruiting', 'Completed', 'Terminated', 'Suspended', 'Withdrawn', 'Enrolling by invitation']
ALL_ALLOCATION_CATEGORIES = ['Randomized', 'Non-Randomized']
ALL_INTERVENTION_MODEL_CATEGORIES = ['Parallel Assignment', 'Single Group Assignment', 'Crossover Assignment', 'Sequential Assignment', 'Factorial Assignment']
ALL_OBSERVATIONAL_MODEL_CATEGORIES = ['Cohort', 'Case-Control', 'Case-Only', 'Ecologic or Community', 'Other']
ALL_TIME_PERSPECTIVE_CATEGORIES = ['Prospective', 'Retrospective', 'Cross-Sectional', 'Other']
ALL_MASKING_CATEGORIES = ['None (Open Label)', 'Single', 'Double', 'Triple', 'Quadruple']
ALL_PRIMARY_PURPOSE_CATEGORIES = ['Treatment', 'Prevention', 'Diagnostic', 'Supportive Care', 'Screening', 'Health Services Research', 'Basic Science', 'Other']
# Assuming study_design_info_other_text_count is a numerical count, not categorical for get_dummies.
# If it were categorical, we'd need its categories. For now, treating as numeric/boolean.

def parse_age(age_str):
    if pd.isna(age_str) or age_str == "":
        return np.nan
    age_str = str(age_str).lower()
    value = float(age_str.split(" ")[0])
    if "year" in age_str:
        return value
    elif "month" in age_str:
        return value / 12
    elif "week" in age_str:
        return value / 52
    elif "day" in age_str:
        return value / 365
    return np.nan


def build_features(df: pd.DataFrame) -> (pd.DataFrame, pd.Series, pd.Series):
    # Create features
    X = pd.DataFrame(index=df.index)

    # Helper to safely get a column or a default Series if not present
    def get_col(col_name, default_val=0, dtype=None):
        if col_name in df.columns:
            return df[col_name].fillna(default_val) if default_val is not None else df[col_name]
        return pd.Series(default_val, index=df.index, dtype=dtype)

    # Target: Handle 'overall_status' potentially missing
    y = (get_col("overall_status", default_val="NOT_TERMINATED_FOR_BUILD_FEATURES") == "TERMINATED").astype(int)

    # Numeric features
    enrollment_val = get_col("enrollment", default_val=0)
    X["enrollment"] = enrollment_val
    X["num_outcomes"] = get_col("num_outcomes", default_val=0)

    # Convert date columns to datetime objects before calculating duration
    start_date = pd.to_datetime(get_col("start_date", default_val=pd.NaT, dtype='datetime64[ns]'), errors='coerce')
    primary_completion_date = pd.to_datetime(get_col("primary_completion_date", default_val=pd.NaT, dtype='datetime64[ns]'), errors='coerce')
    X["duration"] = (primary_completion_date - start_date).dt.days.fillna(0)

    X["is_fda_regulated_drug"] = get_col("is_fda_regulated_drug", default_val=False).astype(int)
    X["is_fda_regulated_device"] = get_col("is_fda_regulated_device", default_val=False).astype(int)
    X["has_us_facility"] = get_col("has_us_facility", default_val=False).astype(int)
    X["num_facilities"] = get_col("num_facilities", default_val=0)
    X["has_data_monitoring_committee"] = get_col("has_data_monitoring_committee", default_val=False).astype(int)
    X["has_dmc_in_protocol"] = get_col("has_dmc_in_protocol", default_val=False).astype(int)

    # Date features
    X["start_year"] = start_date.dt.year.fillna(0).astype(int)
    X["start_month"] = start_date.dt.month.fillna(0).astype(int)
    X["start_day"] = start_date.dt.day.fillna(0).astype(int)

    # Categorical features - Phase
    phase_series = get_col("phase", default_val=None, dtype=object)
    X["phase_num"] = phase_series.map(NUM_PHASES).fillna(0) # Renamed to phase_num as per error
    X["is_early_phase"] = (X["phase_num"] < 2).astype(int)

    # Sponsor related features
    sponsor_name = get_col("lead_sponsor_name", default_val="").astype(str).str.lower()
    X["is_industry"] = sponsor_name.apply(lambda x: 1 if "industry" in x else 0)
    X["is_nih"] = sponsor_name.apply(lambda x: 1 if "nih" in x else 0)
    X["sponsor_code"] = 0 # Placeholder, requires external mapping

    # Enrollment related features
    X["log_enrollment"] = np.log1p(enrollment_val)
    X["is_large_trial"] = (enrollment_val > 1000).astype(int) # Arbitrary threshold for large
    X["enrollment_is_estimated"] = get_col("enrollment_is_estimated", default_val=False).astype(int)

    # Duration related features
    X["planned_duration_months"] = (X["duration"] / 30.4).fillna(0)
    X["is_long_trial"] = (X["planned_duration_months"] > 12).astype(int) # Arbitrary threshold for long

    # Masking level
    masking_map = {"None (Open Label)": 0, "Single": 1, "Double": 2, "Triple": 3, "Quadruple": 4}
    masking_series = get_col("masking", default_val="None (Open Label)", dtype=object)
    X["masking_level"] = masking_series.map(masking_map).fillna(0)

    # Study design features
    study_type_val = get_col("study_type", default_val="").astype(str).str.lower()
    intervention_model_val = get_col("intervention_model", default_val="").astype(str).str.lower()
    X["is_randomized"] = study_type_val.apply(lambda x: 1 if "randomized" in x else 0)
    X["is_treatment"] = study_type_val.apply(lambda x: 1 if "intervention" in x else 0)
    X["is_parallel"] = intervention_model_val.apply(lambda x: 1 if "parallel" in x else 0)

    # Counts
    X["n_arms"] = get_col("number_of_arms", default_val=0)
    conditions_val = get_col("conditions", default_val="").astype(str)
    X["n_conditions"] = conditions_val.apply(lambda x: len(x.split('|')) if x else 0)
    interventions_val = get_col("interventions", default_val="").astype(str)
    X["n_interventions"] = interventions_val.apply(lambda x: len(x.split('|')) if x else 0)
    X["n_primary_outcomes"] = get_col("number_of_primary_outcomes", default_val=0)
    X["n_secondary_outcomes"] = get_col("number_of_secondary_outcomes", default_val=0)
    X["outcome_burden"] = X["n_primary_outcomes"] + X["n_secondary_outcomes"]

    # Location features
    locations_val = get_col("locations", default_val="").astype(str)
    X["n_countries"] = locations_val.apply(lambda x: len(x.split('|')) if x else 0)
    X["is_multinational"] = (X["n_countries"] > 1).astype(int)

    # Eligibility related features
    eligibility_criteria_val = get_col("eligibility_criteria", default_val="").astype(str)
    X["log_eligibility_len"] = np.log1p(eligibility_criteria_val.apply(len))

    min_age_str = get_col("minimum_age", default_val="", dtype=object)
    max_age_str = get_col("maximum_age", default_val="", dtype=object)
    
    # Apply age parsing
    min_age_years = min_age_str.apply(parse_age).fillna(0)
    max_age_years = max_age_str.apply(parse_age).fillna(0) # Use 0 for NaNs here to allow calculations

    X["min_age_years"] = min_age_years
    X["max_age_years"] = max_age_years
    X["age_range_years"] = (max_age_years - min_age_years).clip(lower=0).fillna(0) # Ensure no negative range
    X["includes_children"] = ((min_age_years < 18) & (min_age_years > 0)).astype(int)
    X["includes_elderly"] = (max_age_years >= 65).astype(int)

    gender_val = get_col("gender", default_val="").astype(str).str.lower()
    X["sex_all"] = gender_val.apply(lambda x: 1 if "all" in x else 0)

    # Intervention type features
    intervention_type_val = get_col("intervention_type", default_val="").astype(str).str.lower()
    X["has_drug"] = intervention_type_val.apply(lambda x: 1 if "drug" in x else 0)
    X["has_biologic"] = intervention_type_val.apply(lambda x: 1 if "biologic" in x else 0)
    X["has_device"] = intervention_type_val.apply(lambda x: 1 if "device" in x else 0)
    X["has_procedure"] = intervention_type_val.apply(lambda x: 1 if "procedure" in x else 0)

    X["is_multicondition"] = (X["n_conditions"] > 1).astype(int)

    # Binary indicator for 'healthy_volunteers' - Renamed to accepts_healthy
    healthy_volunteers_series = get_col("healthy_volunteers", default_val="").astype(str)
    X["accepts_healthy"] = healthy_volunteers_series.str.upper().str.contains("YES", na=False).astype(int)

    # Add a feature for the length of the official title
    X["title_len"] = get_col("official_title", default_val="").astype(str).apply(len)

    # Add brief_summary_len
    X["brief_summary_len"] = get_col("brief_summary", default_val="").astype(str).apply(len)


    # --- Handle one-hot encoded categorical features explicitly to ensure all expected columns exist --- (excluding 'masking')
    categorical_cols_to_process = [
        ("study_type", ALL_STUDY_TYPE_CATEGORIES),
        ("overall_status_first", ALL_OVERALL_STATUS_FIRST_CATEGORIES),
        ("allocation", ALL_ALLOCATION_CATEGORIES),
        ("intervention_model", ALL_INTERVENTION_MODEL_CATEGORIES),
        ("observational_model", ALL_OBSERVATIONAL_MODEL_CATEGORIES),
        ("time_perspective", ALL_TIME_PERSPECTIVE_CATEGORIES),
        ("primary_purpose", ALL_PRIMARY_PURPOSE_CATEGORIES)
        # "masking" is handled by masking_level
        # study_design_info_other_text_count is numeric/count, not for get_dummies
    ]

    for col_name, categories in categorical_cols_to_process:
        col_series = get_col(col_name, default_val=None, dtype=object)
        dummies = pd.get_dummies(col_series, prefix=col_name)

        # Ensure all expected categories are present, fill with 0 if not generated
        for cat in categories:
            dummy_col_name = f"{col_name}_{cat}"
            if dummy_col_name not in dummies.columns:
                dummies[dummy_col_name] = 0
        
        # Filter to only the expected categories, in case get_dummies produced unexpected ones
        dummies = dummies[[f"{col_name}_{cat}" for cat in categories]]
        X = pd.concat([X, dummies], axis=1)

    # Ensure all columns are numeric
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
        elif X[col].dtype == 'object': # Fallback for any remaining object columns
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            except:
                X[col] = 0 # If conversion fails, default to 0

    # Store nct_id for later use (e.g., merging with risk scores)
    nct_ids = get_col("nct_id", default_val="", dtype=object)

    return X, y, nct_ids
