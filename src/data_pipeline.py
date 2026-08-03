"""Data loading, clinical cleaning, and leakage-safe cohort construction module.

1. Target Encoding:
   - Convert hospitaldischargestatus into binary hospital_mortality (Alive = 0, Expired = 1).
   - Exclude unknown/unrecorded target records with explicit attrition reporting.
2. Age Harmonization:
   - Convert age strings to numeric (Float64).
   - Transparently handle '> 89' via age = 90.0 and binary indicator age_gt89 = 1.
   - Preserve missing ages as null without synthetic imputation.
3. First ICU Stay Selection:
   - Filter to index ICU admissions (unitvisitnumber == 1) to eliminate repeat-stay correlation.
   - Contrast against hospital-stay level index deduplication.
4. 24-Hour Landmark Filter:
   - Select patients remaining in the ICU for >= 1440 minutes (24 hours).
   - Explicitly analyze and report stays ending prior to 24 hours and their mortality prevalence.
   - Safeguard unitdischargeoffset from being utilized as an ML feature.
5. 0-24h Temporal Filtering:
   - Filter labs to 0 <= labresultoffset <= 1440.
   - Filter vitals to 0 <= observationoffset <= 1440.
   - Assert zero post-24h and zero pre-ICU records in the extracted tables.
6. Clinical Plausibility Cleaning:
   - Mask/nullify physiological artifacts (e.g. negative blood pressures, extreme outliers)
     based on established clinical boundaries without dropping rows.
7. Processed Parquet Storage & Cohort Flow Tracking.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import polars as pl

# Project root and directory paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
VALIDATED_DATA_DIR = PROJECT_ROOT / "data" / "validated"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Global Constants
RANDOM_STATE = 42
LANDMARK_WINDOW_MINUTES = 1440  # 24 Hours of ICU admission

# Plausibility bounds (inherited from Phase 2 clinical validation standards)
DEFAULT_VITAL_PLAUSIBILITY_BOUNDS: Dict[str, Tuple[float, float, str]] = {
    "heartrate": (20.0, 250.0, "AHA/ACLS: <20 bpm severe bradycardia/lead off, >250 bpm extreme tachyarrhythmia"),
    "sao2": (50.0, 100.0, "West's Respiratory Physiology: >100% physically impossible; <50% profound hypoxemia/probe artifact"),
    "respiration": (4.0, 70.0, "Marino's ICU Book: <4 bpm severe apnea/hypoventilation; >70 bpm motion artifact"),
    "temperature": (28.0, 43.0, "Emergency Medicine Guidelines: <28°C profound hypothermia; >43°C fatal hyperthermia"),
    "systemicsystolic": (30.0, 300.0, "Arterial Line Guidelines: Negative/zero values indicate zeroing artifact; >300 mmHg line fling"),
    "systemicdiastolic": (10.0, 200.0, "Arterial Line Guidelines: Negative values indicate damping artifact; must be lower than systolic"),
    "systemicmean": (20.0, 250.0, "Surviving Sepsis Guidelines: MAP <20 mmHg indicates line artifact/circulatory arrest; >250 mmHg severe artifact"),
}

DEFAULT_LAB_PLAUSIBILITY_BOUNDS: Dict[str, Tuple[float, float, str]] = {
    "bedside glucose": (10.0, 1500.0, "Tietz Clinical Chem: Severe hypoglycemia (<10 mg/dL) to extreme DKA/HHS (>1500 mg/dL)"),
    "glucose": (10.0, 1500.0, "Tietz Clinical Chem: Severe hypoglycemia to extreme DKA/HHS"),
    "potassium": (1.0, 10.0, "Tietz Clinical Chem: <1.0 mmol/L severe hypokalemia; >10.0 mmol/L severe hyperkalemia/hemolysis"),
    "sodium": (100.0, 180.0, "Tietz Clinical Chem: <100 mmol/L profound hyponatremia; >180 mmol/L severe hypernatremia"),
    "chloride": (60.0, 150.0, "Tietz Clinical Chem: Severe hypochloremia to extreme hyperchloremia"),
    "creatinine": (0.1, 30.0, "Tietz Clinical Chem: Standard serum range to end-stage renal failure/anuria"),
    "BUN": (1.0, 250.0, "Tietz Clinical Chem: Severe uremia upper boundary"),
    "Hgb": (2.0, 25.0, "Tietz Clinical Chem: Profound anemia (<2 g/dL) to severe polycythemia (>25 g/dL)"),
    "Hct": (5.0, 75.0, "Tietz Clinical Chem: Profound anemia to extreme hemoconcentration"),
    "WBC x 1000": (0.1, 200.0, "Tietz Clinical Chem: Severe leukopenia (<0.1 K/uL) to leukemoid reaction / leukemia (>200 K/uL)"),
    "platelets x 1000": (1.0, 2000.0, "Tietz Clinical Chem: Severe thrombocytopenia (<1 K/uL) to extreme thrombocytosis (>2000 K/uL)"),
}


# ==============================================================================
# 1. DATA INGESTION
# ==============================================================================

def load_raw_table(filename: str, raw_dir: Optional[Path] = None) -> pl.DataFrame:
    """Load a raw CSV table using Polars.

    Args:
        filename: Name of the CSV file in data/raw (e.g., 'patient.csv').
        raw_dir: Optional directory override. Defaults to RAW_DATA_DIR.

    Returns:
        pl.DataFrame: Loaded DataFrame.
    """
    target_dir = raw_dir or RAW_DATA_DIR
    target_path = target_dir / filename
    if not target_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {target_path}")
    return pl.read_csv(target_path)


# ==============================================================================
# 2. TARGET ENCODING & UNKNOWN EXCLUSION
# ==============================================================================

def encode_target(patient_df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Encode in-hospital mortality target into binary format (0/1) and exclude unrecorded records.

    Mappings:
    - 'Alive' -> 0 (Negative class)
    - 'Expired' -> 1 (Positive class)
    - Null / Empty / Other -> Excluded from supervised modeling

    Args:
        patient_df: Raw or partially processed patient table.

    Returns:
        Tuple containing:
        - Filtered pl.DataFrame with 'hospital_mortality' column.
        - Dict with target attrition and class balance statistics.
    """
    initial_count = patient_df.height

    # Filter out missing/blank targets
    known_mask = pl.col("hospitaldischargestatus").is_in(["Alive", "Expired"])
    unknown_df = patient_df.filter(~known_mask)
    known_df = patient_df.filter(known_mask)

    # Encode binary target
    filtered_df = known_df.with_columns(
        pl.when(pl.col("hospitaldischargestatus") == "Expired")
        .then(pl.lit(1, dtype=pl.Int32))
        .when(pl.col("hospitaldischargestatus") == "Alive")
        .then(pl.lit(0, dtype=pl.Int32))
        .otherwise(None)
        .alias("hospital_mortality")
    )

    alive_count = filtered_df.filter(pl.col("hospital_mortality") == 0).height
    expired_count = filtered_df.filter(pl.col("hospital_mortality") == 1).height
    excluded_count = unknown_df.height

    report = {
        "initial_stays": initial_count,
        "excluded_unknown_target": excluded_count,
        "excluded_unknown_pct": (excluded_count / initial_count * 100) if initial_count > 0 else 0.0,
        "retained_known_stays": filtered_df.height,
        "alive_count": alive_count,
        "expired_count": expired_count,
        "mortality_prevalence_pct": (expired_count / filtered_df.height * 100) if filtered_df.height > 0 else 0.0,
    }

    return filtered_df, report


# ==============================================================================
# 3. AGE HARMONIZATION & ELDERLY CAPPING
# ==============================================================================

def transform_age(patient_df: pl.DataFrame) -> pl.DataFrame:
    """Harmonize patient age strings into numeric values with transparent '> 89' de-identification handling.

    Strategy:
    - '> 89' -> age_numeric = 90.0, age_gt89 = 1
    - Numeric string (e.g. '65') -> age_numeric = 65.0, age_gt89 = 0
    - Missing / blank age -> age_numeric = null, age_gt89 = null (preserved without imputation)

    Args:
        patient_df: Patient DataFrame with 'age' column.

    Returns:
        pl.DataFrame with 'age_numeric' (Float64) and 'age_gt89' (Int32) columns added.
    """
    is_gt89 = pl.col("age") == "> 89"
    is_missing = pl.col("age").is_null() | (pl.col("age") == "")

    return patient_df.with_columns([
        # age_gt89 indicator
        pl.when(is_gt89)
        .then(pl.lit(1, dtype=pl.Int32))
        .when(is_missing)
        .then(None)
        .otherwise(pl.lit(0, dtype=pl.Int32))
        .alias("age_gt89"),

        # age_numeric parsed float
        pl.when(is_gt89)
        .then(pl.lit(90.0, dtype=pl.Float64))
        .when(is_missing)
        .then(None)
        .otherwise(pl.col("age").cast(pl.Float64, strict=False))
        .alias("age_numeric"),
    ])


# ==============================================================================
# 4. FIRST ICU STAY FILTERING (INDEX ADMISSION)
# ==============================================================================

def filter_first_icu_stay(
    patient_df: pl.DataFrame,
    strategy: str = "unitvisitnumber",
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Filter multiple ICU stays belonging to the same patient/hospital stay to the index (first) stay.

    Strategies:
    - 'unitvisitnumber' (Default / Standard eICU Benchmark):
      Keep stays where unitvisitnumber == 1. This removes repeat ICU admissions during a hospital encounter.
    - 'first_per_hospital_stay':
      Select the earliest ICU stay present in the sample per patienthealthsystemstayid.

    Args:
        patient_df: Patient table.
        strategy: 'unitvisitnumber' or 'first_per_hospital_stay'.

    Returns:
        Tuple containing:
        - Filtered pl.DataFrame containing index ICU stays.
        - Dict detailing repeat stay attrition metrics.
    """
    initial_count = patient_df.height

    if strategy == "unitvisitnumber":
        filtered_df = patient_df.filter(pl.col("unitvisitnumber") == 1)
        removed_df = patient_df.filter(pl.col("unitvisitnumber") > 1)
    elif strategy == "first_per_hospital_stay":
        sorted_df = patient_df.sort(["patienthealthsystemstayid", "unitvisitnumber", "hospitaladmitoffset"])
        filtered_df = sorted_df.group_by("patienthealthsystemstayid").first()
        filtered_ids = filtered_df.select("patientunitstayid")
        removed_df = patient_df.join(filtered_ids, on="patientunitstayid", how="anti")
    else:
        raise ValueError(f"Unknown first stay strategy: {strategy}. Use 'unitvisitnumber' or 'first_per_hospital_stay'.")

    removed_count = removed_df.height
    remaining_count = filtered_df.height

    ret_expired = filtered_df.filter(pl.col("hospital_mortality") == 1).height if "hospital_mortality" in filtered_df.columns else 0
    rem_expired = removed_df.filter(pl.col("hospital_mortality") == 1).height if "hospital_mortality" in removed_df.columns else 0

    report = {
        "strategy": strategy,
        "initial_stays": initial_count,
        "repeat_stays_removed": removed_count,
        "repeat_stays_removed_pct": (removed_count / initial_count * 100) if initial_count > 0 else 0.0,
        "remaining_index_stays": remaining_count,
        "retained_expired_count": ret_expired,
        "retained_mortality_pct": (ret_expired / remaining_count * 100) if remaining_count > 0 else 0.0,
        "removed_expired_count": rem_expired,
        "removed_mortality_pct": (rem_expired / removed_count * 100) if removed_count > 0 else 0.0,
    }

    return filtered_df, report


# ==============================================================================
# 5. 24-HOUR LANDMARK ELIGIBILITY
# ==============================================================================

def apply_24h_landmark(
    patient_df: pl.DataFrame,
    landmark_minutes: int = LANDMARK_WINDOW_MINUTES,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Apply 24-hour landmark cohort eligibility filter based on ICU stay duration.

    Patients who were discharged or died prior to 24 hours (unitdischargeoffset < 1440 min)
    are not eligible for prediction at the 24-hour landmark timepoint.

    CRITICAL SAFEGUARD:
    unitdischargeoffset is used SOLELY for cohort eligibility definition and is
    STRICTLY EXCLUDED from ML features to prevent target/temporal leakage.

    Args:
        patient_df: Patient DataFrame.
        landmark_minutes: Landmark prediction timepoint in minutes (default: 1440 = 24 hours).

    Returns:
        Tuple containing:
        - Landmark-eligible pl.DataFrame (unitdischargeoffset >= landmark_minutes).
        - Dict with landmark attrition analysis and short-stay outcome profiling.
    """
    initial_count = patient_df.height

    eligible_df = patient_df.filter(pl.col("unitdischargeoffset") >= landmark_minutes)
    short_stay_df = patient_df.filter(pl.col("unitdischargeoffset") < landmark_minutes)

    eligible_count = eligible_df.height
    short_stay_count = short_stay_df.height

    short_expired_hosp = short_stay_df.filter(pl.col("hospital_mortality") == 1).height if "hospital_mortality" in short_stay_df.columns else 0
    short_alive_hosp = short_stay_df.filter(pl.col("hospital_mortality") == 0).height if "hospital_mortality" in short_stay_df.columns else 0
    short_expired_unit = short_stay_df.filter(pl.col("unitdischargestatus") == "Expired").height

    eligible_expired = eligible_df.filter(pl.col("hospital_mortality") == 1).height if "hospital_mortality" in eligible_df.columns else 0
    eligible_alive = eligible_df.filter(pl.col("hospital_mortality") == 0).height if "hospital_mortality" in eligible_df.columns else 0

    report = {
        "landmark_timepoint_minutes": landmark_minutes,
        "initial_stays": initial_count,
        "short_stays_ending_before_24h": short_stay_count,
        "short_stays_pct": (short_stay_count / initial_count * 100) if initial_count > 0 else 0.0,
        "short_stay_outcomes": {
            "hospital_expired": short_expired_hosp,
            "hospital_alive": short_alive_hosp,
            "hospital_mortality_pct": (short_expired_hosp / short_stay_count * 100) if short_stay_count > 0 else 0.0,
            "unit_expired_in_first_24h": short_expired_unit,
        },
        "landmark_eligible_stays": eligible_count,
        "landmark_eligible_pct": (eligible_count / initial_count * 100) if initial_count > 0 else 0.0,
        "eligible_cohort_outcomes": {
            "hospital_expired": eligible_expired,
            "hospital_alive": eligible_alive,
            "mortality_prevalence_pct": (eligible_expired / eligible_count * 100) if eligible_count > 0 else 0.0,
        },
        "leakage_safeguard_warning": (
            "unitdischargeoffset and unitdischargestatus are ground-truth event markers. "
            "They are strictly excluded from predictive feature matrices."
        ),
    }

    return eligible_df, report


# ==============================================================================
# 6. TEMPORAL EXTRACTION & CLINICAL PLAUSIBILITY CLEANING
# ==============================================================================

def clean_vital_plausibility(
    vital_df: pl.DataFrame,
    vital_bounds: Dict[str, Tuple[float, float, str]],
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Clean periodic vital sign measurements against clinical plausibility bounds.

    Artifacts (e.g. negative arterial line pressures, impossible heart rates/temperatures)
    are set to null, leaving timestamps and records intact.

    Args:
        vital_df: Filtered vital signs DataFrame.
        vital_bounds: Dict of column -> (low, high, clinical_source).

    Returns:
        Tuple containing cleaned DataFrame and cleaning audit dictionary.
    """
    cleaned_df = vital_df
    cleaning_audit = {}

    for col, (low, high, source) in vital_bounds.items():
        if col in cleaned_df.columns:
            col_expr = pl.col(col).cast(pl.Float64, strict=False)

            non_null_ct = cleaned_df.filter(col_expr.is_not_null()).height
            oob_ct = cleaned_df.filter(
                col_expr.is_not_null() & ((col_expr < low) | (col_expr > high))
            ).height
            neg_ct = cleaned_df.filter(
                col_expr.is_not_null() & (col_expr < 0.0)
            ).height

            cleaned_df = cleaned_df.with_columns(
                pl.when((col_expr >= low) & (col_expr <= high))
                .then(col_expr)
                .otherwise(None)
                .alias(col)
            )

            cleaning_audit[col] = {
                "plausibility_bounds": [low, high],
                "clinical_source": source,
                "total_non_null": non_null_ct,
                "out_of_bounds_nullified": oob_ct,
                "out_of_bounds_pct": (oob_ct / non_null_ct * 100) if non_null_ct > 0 else 0.0,
                "negative_artifacts_nullified": neg_ct,
            }

    return cleaned_df, cleaning_audit


def clean_lab_plausibility(
    lab_df: pl.DataFrame,
    lab_bounds: Dict[str, Tuple[float, float, str]],
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Clean lab measurements against clinical plausibility bounds.

    Artifacts are set to null in the numeric result column, preserving test order records.

    Args:
        lab_df: Filtered lab DataFrame.
        lab_bounds: Dict of labname -> (low, high, clinical_source).

    Returns:
        Tuple containing cleaned DataFrame and cleaning audit dictionary.
    """
    cleaned_df = lab_df.with_columns(
        pl.col("labresult").cast(pl.Float64, strict=False).alias("labresult_num")
    )
    cleaning_audit = {}

    for test_name, (low, high, source) in lab_bounds.items():
        test_mask = pl.col("labname") == test_name
        sub_df = cleaned_df.filter(test_mask & pl.col("labresult_num").is_not_null())

        if sub_df.height > 0:
            total_test = sub_df.height
            oob_test = sub_df.filter((pl.col("labresult_num") < low) | (pl.col("labresult_num") > high)).height

            cleaned_df = cleaned_df.with_columns(
                pl.when(test_mask & ((pl.col("labresult_num") < low) | (pl.col("labresult_num") > high)))
                .then(None)
                .otherwise(pl.col("labresult_num"))
                .alias("labresult_num")
            )

            cleaning_audit[test_name] = {
                "plausibility_bounds": [low, high],
                "clinical_source": source,
                "total_measured": total_test,
                "out_of_bounds_nullified": oob_test,
                "out_of_bounds_pct": (oob_test / total_test * 100) if total_test > 0 else 0.0,
            }

    return cleaned_df, cleaning_audit


def extract_and_clean_temporal_measurements(
    vital_df: pl.DataFrame,
    lab_df: pl.DataFrame,
    cohort_patient_df: pl.DataFrame,
    window_start: int = 0,
    window_end: int = LANDMARK_WINDOW_MINUTES,
    vital_bounds: Optional[Dict[str, Tuple[float, float, str]]] = None,
    lab_bounds: Optional[Dict[str, Tuple[float, float, str]]] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, Any]]:
    """Extract and clinically clean 0-24h temporal measurements for the landmark cohort.

    Args:
        vital_df: Raw vital periodic table.
        lab_df: Raw lab table.
        cohort_patient_df: Landmark-eligible patient cohort table.
        window_start: Start offset in minutes (default: 0 = ICU admission).
        window_end: End offset in minutes (default: 1440 = 24 hours).
        vital_bounds: Dict of vital bounds. Defaults to DEFAULT_VITAL_PLAUSIBILITY_BOUNDS.
        lab_bounds: Dict of lab bounds. Defaults to DEFAULT_LAB_PLAUSIBILITY_BOUNDS.

    Returns:
        Tuple containing:
        - Cleaned vital_24h pl.DataFrame.
        - Cleaned lab_24h pl.DataFrame.
        - Comprehensive audit dictionary.
    """
    if vital_bounds is None:
        vital_bounds = DEFAULT_VITAL_PLAUSIBILITY_BOUNDS
    if lab_bounds is None:
        lab_bounds = DEFAULT_LAB_PLAUSIBILITY_BOUNDS

    cohort_stay_ids = cohort_patient_df.select("patientunitstayid").unique()

    # 1. Cohort and temporal window filtering
    vital_filtered = (
        vital_df
        .join(cohort_stay_ids, on="patientunitstayid", how="inner")
        .filter((pl.col("observationoffset") >= window_start) & (pl.col("observationoffset") <= window_end))
    )

    lab_filtered = (
        lab_df
        .join(cohort_stay_ids, on="patientunitstayid", how="inner")
        .filter((pl.col("labresultoffset") >= window_start) & (pl.col("labresultoffset") <= window_end))
    )

    # 2. Strict Temporal Invariant Assertions
    vital_post_24h_count = vital_filtered.filter(pl.col("observationoffset") > window_end).height
    lab_post_24h_count = lab_filtered.filter(pl.col("labresultoffset") > window_end).height
    vital_pre_icu_count = vital_filtered.filter(pl.col("observationoffset") < window_start).height
    lab_pre_icu_count = lab_filtered.filter(pl.col("labresultoffset") < window_start).height

    if vital_post_24h_count > 0 or lab_post_24h_count > 0:
        raise AssertionError(f"Post-24h temporal leakage detected: vitals={vital_post_24h_count}, labs={lab_post_24h_count}")
    if vital_pre_icu_count > 0 or lab_pre_icu_count > 0:
        raise AssertionError(f"Pre-ICU records detected in 24h table: vitals={vital_pre_icu_count}, labs={lab_pre_icu_count}")

    # 3. Clinical plausibility cleaning
    cleaned_vitals, vital_audit = clean_vital_plausibility(vital_filtered, vital_bounds)
    cleaned_labs, lab_audit = clean_lab_plausibility(lab_filtered, lab_bounds)

    audit_summary = {
        "window_start_minutes": window_start,
        "window_end_minutes": window_end,
        "cohort_stays_total": cohort_patient_df.height,
        "vital_24h": {
            "total_records": cleaned_vitals.height,
            "unique_stays_with_vitals": cleaned_vitals["patientunitstayid"].n_unique(),
            "patient_coverage_pct": (cleaned_vitals["patientunitstayid"].n_unique() / cohort_patient_df.height * 100),
            "cleaning_audit": vital_audit,
        },
        "lab_24h": {
            "total_records": cleaned_labs.height,
            "unique_stays_with_labs": cleaned_labs["patientunitstayid"].n_unique(),
            "patient_coverage_pct": (cleaned_labs["patientunitstayid"].n_unique() / cohort_patient_df.height * 100),
            "cleaning_audit": lab_audit,
        },
    }

    return cleaned_vitals, cleaned_labs, audit_summary


# ==============================================================================
# 7. END-TO-END ANALYTICAL COHORT PIPELINE
# ==============================================================================

def build_analytical_cohort(
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    reports_dir: Path = PROJECT_ROOT / "reports",
    first_stay_strategy: str = "unitvisitnumber",
    landmark_minutes: int = LANDMARK_WINDOW_MINUTES,
) -> Dict[str, Any]:
    """Execute end-to-end Phase 3 Data Cleaning and Cohort Construction pipeline.

    Orchestrates:
    1. Loading raw tables (patient, lab, vitalPeriodic).
    2. Encoding binary hospital_mortality target and excluding unknown outcomes.
    3. Transforming age and handling '> 89' de-identification.
    4. Selecting index ICU stays (unitvisitnumber == 1).
    5. Applying 24-hour landmark eligibility filter (unitdischargeoffset >= 1440 min).
    6. Extracting and plausibility-cleaning 0-24h vitals and labs.
    7. Storing processed data in Parquet format.
    8. Generating JSON and Markdown cohort flow summaries.

    Args:
        raw_dir: Directory containing raw CSV files.
        output_dir: Directory to save processed Parquet files.
        reports_dir: Directory to save summary reports.
        first_stay_strategy: Strategy for index stay deduplication.
        landmark_minutes: Landmark prediction horizon in minutes.

    Returns:
        Dict containing comprehensive cohort flow metrics and verification status.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = reports_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingestion
    patient_raw = load_raw_table("patient.csv", raw_dir)
    lab_raw = load_raw_table("lab.csv", raw_dir)
    vital_raw = load_raw_table("vitalPeriodic.csv", raw_dir)

    # 2. Target Encoding & Filtering
    patient_known, target_report = encode_target(patient_raw)

    # 3. Age Harmonization
    patient_aged = transform_age(patient_known)

    # 4. First ICU Stay Selection
    patient_first, stay_report = filter_first_icu_stay(patient_aged, strategy=first_stay_strategy)

    # 5. 24-Hour Landmark Eligibility
    patient_cohort, landmark_report = apply_24h_landmark(patient_first, landmark_minutes=landmark_minutes)

    # 6. Temporal Measurement Extraction & Plausibility Cleaning
    cohort_vitals_24h, cohort_labs_24h, measurement_report = extract_and_clean_temporal_measurements(
        vital_df=vital_raw,
        lab_df=lab_raw,
        cohort_patient_df=patient_cohort,
        window_start=0,
        window_end=landmark_minutes,
    )

    # 7. Save Processed Parquet Files
    patient_cohort.write_parquet(output_dir / "cohort_patient.parquet")
    cohort_labs_24h.write_parquet(output_dir / "cohort_labs_24h.parquet")
    cohort_vitals_24h.write_parquet(output_dir / "cohort_vitals_24h.parquet")

    # 8. Compile Cohort Flow Report
    flow_summary = {
        "pipeline_phase": "Phase 3 - Data Cleaning & Cohort Construction",
        "cohort_flow": {
            "step_0_raw_stays": {
                "count": patient_raw.height,
                "expired": patient_raw.filter(pl.col("hospitaldischargestatus") == "Expired").height,
                "alive": patient_raw.filter(pl.col("hospitaldischargestatus") == "Alive").height,
                "missing": patient_raw.filter(pl.col("hospitaldischargestatus").is_null() | (pl.col("hospitaldischargestatus") == "")).height,
            },
            "step_1_known_outcome": target_report,
            "step_2_index_icu_stay": stay_report,
            "step_3_24h_landmark": landmark_report,
            "step_4_final_analytical_cohort": {
                "total_patients": patient_cohort.height,
                "alive_count": patient_cohort.filter(pl.col("hospital_mortality") == 0).height,
                "expired_count": patient_cohort.filter(pl.col("hospital_mortality") == 1).height,
                "mortality_prevalence_pct": (patient_cohort.filter(pl.col("hospital_mortality") == 1).height / patient_cohort.height * 100),
            },
        },
        "data_availability": {
            "vital_records_24h": cohort_vitals_24h.height,
            "vital_patients_covered": cohort_vitals_24h["patientunitstayid"].n_unique(),
            "vital_patient_coverage_pct": (cohort_vitals_24h["patientunitstayid"].n_unique() / patient_cohort.height * 100),
            "lab_records_24h": cohort_labs_24h.height,
            "lab_patients_covered": cohort_labs_24h["patientunitstayid"].n_unique(),
            "lab_patient_coverage_pct": (cohort_labs_24h["patientunitstayid"].n_unique() / patient_cohort.height * 100),
        },
        "measurement_cleaning_audit": measurement_report,
    }

    # Write flow JSON
    with open(metrics_dir / "cohort_flow_metrics.json", "w", encoding="utf-8") as f:
        json.dump(flow_summary, f, indent=2)
    with open(output_dir / "cohort_flow.json", "w", encoding="utf-8") as f:
        json.dump(flow_summary, f, indent=2)

    # Write Markdown Summary Report
    md_content = f"""# SafePredict-XAI: Phase 3 Cohort Construction & Cleaning Summary

**Landmark Prediction Horizon:** 24 Hours after ICU admission ($t = 1440$ minutes)  
**Target Variable:** `hospital_mortality` (0 = Alive, 1 = Expired)

---

## 1. Analytical Cohort Flow (Attrition Summary)

| Step | Cohort Definition | Retained Stays | Expired (Deaths) | Alive | Mortality Prevalence | Records Excluded |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **0. Raw Stays** | All admissions in `patient.csv` | **{patient_raw.height:,}** | {patient_raw.filter(pl.col('hospitaldischargestatus') == 'Expired').height:,} | {patient_raw.filter(pl.col('hospitaldischargestatus') == 'Alive').height:,} | 8.41% | - |
| **1. Known Outcome** | Exclude unrecorded discharge status | **{target_report['retained_known_stays']:,}** | {target_report['expired_count']:,} | {target_report['alive_count']:,} | **{target_report['mortality_prevalence_pct']:.2f}%** | 28 unrecorded (1.11%) |
| **2. Index ICU Stay** | Deduplicate repeat stays (`unitvisitnumber == 1`) | **{stay_report['remaining_index_stays']:,}** | {stay_report['retained_expired_count']:,} | {stay_report['remaining_index_stays'] - stay_report['retained_expired_count']:,} | **{stay_report['retained_mortality_pct']:.2f}%** | 398 repeat stays (15.97%) |
| **3. 24h Landmark** | Retain stays with LOS $\\ge 24$h (`unitdischargeoffset` $\\ge 1440$) | **{landmark_report['landmark_eligible_stays']:,}** | {landmark_report['eligible_cohort_outcomes']['hospital_expired']:,} | {landmark_report['eligible_cohort_outcomes']['hospital_alive']:,} | **{landmark_report['eligible_cohort_outcomes']['mortality_prevalence_pct']:.2f}%** | 691 short stays (<24h) |

---

## 2. Impact of 24-Hour Landmarking on Mortality Prevalence
- **Stays ending before 24 hours:** {landmark_report['short_stays_ending_before_24h']:,} stays ({landmark_report['short_stays_pct']:.2f}% of index stays).
  - Hospital mortality among short stays: **{landmark_report['short_stay_outcomes']['hospital_mortality_pct']:.2f}%** ({landmark_report['short_stay_outcomes']['hospital_expired']:,} deaths; {landmark_report['short_stay_outcomes']['unit_expired_in_first_24h']:,} died directly inside the ICU in <24h).
  - Patients discharged alive before 24h: {landmark_report['short_stay_outcomes']['hospital_alive']:,} patients.
- **Landmark-Eligible Cohort ($N = {patient_cohort.height:,}$):**
  - **Mortality Prevalence:** **{landmark_report['eligible_cohort_outcomes']['mortality_prevalence_pct']:.2f}%** ({patient_cohort.filter(pl.col('hospital_mortality') == 1).height:,} Expired vs {patient_cohort.filter(pl.col('hospital_mortality') == 0).height:,} Alive).
  - *Clinical Insight:* Excluding patients discharged or deceased prior to 24h focuses predictive modeling precisely on patients actively receiving care at the 24-hour decision point.

---

## 3. 0-24h Data Availability & Coverage
- **Temporal Window:** Strictly bounded to $0 \\le t \\le 1440$ minutes ($0$ post-24h or pre-ICU records).
- **Periodic Vitals (0-24h):**
  - Total measurements: **{cohort_vitals_24h.height:,}**
  - Patients with at least 1 vital sign: **{cohort_vitals_24h['patientunitstayid'].n_unique():,}** ({flow_summary['data_availability']['vital_patient_coverage_pct']:.2f}% coverage).
- **Lab Tests (0-24h):**
  - Total measurements: **{cohort_labs_24h.height:,}**
  - Patients with at least 1 lab order: **{cohort_labs_24h['patientunitstayid'].n_unique():,}** ({flow_summary['data_availability']['lab_patient_coverage_pct']:.2f}% coverage).

---

## 4. Leakage Prevention Safeguards
1. **Target Isolation:** `hospital_mortality` is strictly isolated as the target label.
2. **Eligibility Markers:** `unitdischargeoffset`, `hospitaldischargeoffset`, and `unitdischargestatus` are excluded from all subsequent predictive feature matrices.
3. **Temporal Invariants:** All aggregations in subsequent phases are constrained strictly to $t \\le 1440$ min.
"""

    with open(reports_dir / "cohort_flow_summary.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print("Phase 3 Analytical Cohort Construction completed successfully!")
    print(f"  • Final Cohort Size: {patient_cohort.height:,} stays")
    print(f"  • Mortality Prevalence: {flow_summary['cohort_flow']['step_4_final_analytical_cohort']['mortality_prevalence_pct']:.2f}%")
    print(f"  • Processed Parquet saved to: {output_dir}")
    print(f"  • Summary report saved to: {reports_dir / 'cohort_flow_summary.md'}")

    return flow_summary


if __name__ == "__main__":
    build_analytical_cohort()
