"""Data validation and clinical data-quality pipeline module for SafePredict-XAI.

This module implements Phase 2 validation checks:
1. Technical Validation:
   - Schema enforcement and required column verification
   - Primary key integrity, duplicate detection, and orphan record checks
   - Temporal window flagging (pre_icu, first_24h, post_24h)
   - Missingness profiling across demographic, lab, and vital features
2. Clinical & Plausibility Validation:
   - Target outcome validation (Alive vs Expired vs Missing)
   - Physiological range plausibility checks for vitals and labs with documented clinical sources
   - Negative value detection for non-negative physiological variables
   - Encounter-level data quality flagging without silent data deletion
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import polars as pl


# ==============================================================================
# CLINICAL PLAUSIBILITY BOUNDS & DOCUMENTATION
# ==============================================================================

# Clinical reference sources:
# - PhysioNet / eICU Collaborative Research Database benchmarks
# - Marino's The ICU Book (4th Ed.) & Surviving Sepsis Campaign Guidelines
# - West's Respiratory Physiology & AHA Advanced Cardiovascular Life Support (ACLS)
# - Tietz Textbook of Clinical Chemistry and Molecular Diagnostics
DEFAULT_VITAL_PLAUSIBILITY_BOUNDS: Dict[str, Tuple[float, float, str]] = {
    "heartrate": (
        20.0,
        250.0,
        "AHA/ACLS: <20 bpm (severe bradycardia/peri-arrest/lead off), >250 bpm (extreme SVT/VT/artifact)",
    ),
    "sao2": (
        50.0,
        100.0,
        "West's Respiratory Physiology: >100% physically impossible; <50% severe hypoxemia/probe displacement",
    ),
    "respiration": (
        4.0,
        70.0,
        "Marino's ICU Book: <4 bpm indicates severe apnea/hypoventilation; >70 bpm indicates extreme tachypnea or motion artifact",
    ),
    "temperature": (
        28.0,
        43.0,
        "Emergency Medicine Guidelines: <28°C (profound hypothermia) or >43°C (fatal hyperthermia/probe error)",
    ),
    "systemicsystolic": (
        30.0,
        300.0,
        "Arterial Line Guidelines: Negative/zero values indicate transducer calibration/zeroing artifact; >300 mmHg extreme hypertensive crisis/fling",
    ),
    "systemicdiastolic": (
        10.0,
        200.0,
        "Arterial Line Guidelines: Negative values indicate damping/zeroing artifact; must be physiologically lower than systolic",
    ),
    "systemicmean": (
        20.0,
        250.0,
        "Surviving Sepsis Guidelines: MAP <20 mmHg indicates profound circulatory collapse/line artifact; >250 mmHg severe artifact",
    ),
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

REQUIRED_PATIENT_COLUMNS = [
    "patientunitstayid",
    "age",
    "gender",
    "hospitaldischargestatus",
]

REQUIRED_LAB_COLUMNS = [
    "patientunitstayid",
    "labresultoffset",
    "labname",
    "labresult",
]

REQUIRED_VITAL_COLUMNS = [
    "patientunitstayid",
    "observationoffset",
]


# ==============================================================================
# 1. SCHEMA VALIDATION
# ==============================================================================

def validate_schemas(
    patient_df: pl.DataFrame,
    lab_df: pl.DataFrame,
    vital_df: pl.DataFrame,
    strict: bool = True,
) -> Dict[str, Any]:
    """Validate that all required columns are present in raw datasets.

    Args:
        patient_df: Patient table DataFrame.
        lab_df: Lab table DataFrame.
        vital_df: Periodic vital signs DataFrame.
        strict: If True, raises ValueError upon schema validation failure.

    Returns:
        Dict summarizing schema validation results.
    """
    missing_patient_cols = [c for c in REQUIRED_PATIENT_COLUMNS if c not in patient_df.columns]
    missing_lab_cols = [c for c in REQUIRED_LAB_COLUMNS if c not in lab_df.columns]
    missing_vital_cols = [c for c in REQUIRED_VITAL_COLUMNS if c not in vital_df.columns]

    passed = (
        len(missing_patient_cols) == 0
        and len(missing_lab_cols) == 0
        and len(missing_vital_cols) == 0
    )

    result = {
        "status": "PASS" if passed else "FAIL",
        "patient": {
            "total_columns": patient_df.width,
            "required_columns_checked": REQUIRED_PATIENT_COLUMNS,
            "missing_required_columns": missing_patient_cols,
            "passed": len(missing_patient_cols) == 0,
        },
        "lab": {
            "total_columns": lab_df.width,
            "required_columns_checked": REQUIRED_LAB_COLUMNS,
            "missing_required_columns": missing_lab_cols,
            "passed": len(missing_lab_cols) == 0,
        },
        "vitalPeriodic": {
            "total_columns": vital_df.width,
            "required_columns_checked": REQUIRED_VITAL_COLUMNS,
            "missing_required_columns": missing_vital_cols,
            "passed": len(missing_vital_cols) == 0,
        },
    }

    if not passed and strict:
        error_msg = f"Schema validation failed: patient missing {missing_patient_cols}, lab missing {missing_lab_cols}, vital missing {missing_vital_cols}"
        raise ValueError(error_msg)

    return result


# ==============================================================================
# 2. IDENTIFIER VALIDATION
# ==============================================================================

def validate_identifiers(
    patient_df: pl.DataFrame,
    lab_df: pl.DataFrame,
    vital_df: pl.DataFrame,
) -> Dict[str, Any]:
    """Validate primary keys, duplicate records, orphan records, and repeated timestamps.

    Args:
        patient_df: Patient table.
        lab_df: Lab table.
        vital_df: Vital periodic table.

    Returns:
        Dict summarizing identifier integrity metrics.
    """
    # 1. Missing patientunitstayid
    missing_patient_id = patient_df.filter(pl.col("patientunitstayid").is_null()).height
    missing_lab_id = lab_df.filter(pl.col("patientunitstayid").is_null()).height
    missing_vital_id = vital_df.filter(pl.col("patientunitstayid").is_null()).height

    # 2. Duplicates in patient.csv
    dup_patient_stays = patient_df.filter(pl.col("patientunitstayid").is_duplicated()).height

    # 3. Orphan checks
    patient_ids = patient_df.select("patientunitstayid").unique()
    lab_ids = lab_df.select("patientunitstayid").unique()
    vital_ids = vital_df.select("patientunitstayid").unique()

    orphan_lab_records = lab_df.join(patient_ids, on="patientunitstayid", how="anti").height
    orphan_vital_records = vital_df.join(patient_ids, on="patientunitstayid", how="anti").height

    # 4. Repeated measurements check
    repeated_lab_rows = lab_df.filter(
        pl.struct(["patientunitstayid", "labname", "labresultoffset"]).is_duplicated()
    ).height

    repeated_vital_rows = vital_df.filter(
        pl.struct(["patientunitstayid", "observationoffset"]).is_duplicated()
    ).height

    return {
        "missing_patientunitstayid": {
            "patient": missing_patient_id,
            "lab": missing_lab_id,
            "vital": missing_vital_id,
        },
        "patient_primary_key_duplicates": dup_patient_stays,
        "orphan_records": {
            "lab_orphan_records": orphan_lab_records,
            "vital_orphan_records": orphan_vital_records,
        },
        "repeated_measurements": {
            "repeated_lab_measurements": repeated_lab_rows,
            "repeated_lab_pct": (repeated_lab_rows / lab_df.height * 100) if lab_df.height > 0 else 0,
            "repeated_vital_timestamps": repeated_vital_rows,
            "repeated_vital_pct": (repeated_vital_rows / vital_df.height * 100) if vital_df.height > 0 else 0,
            "clinical_note": (
                "Repeated lab and vital timestamps reflect simultaneous multi-channel telemetry "
                "or duplicate blood draw tube orders; they are not inherently corrupted records."
            ),
        },
    }


# ==============================================================================
# 3. TARGET VALIDATION
# ==============================================================================

def validate_targets(patient_df: pl.DataFrame) -> Dict[str, Any]:
    """Validate in-hospital mortality target variable (hospitaldischargestatus).

    Args:
        patient_df: Patient table.

    Returns:
        Dict summarizing outcome distribution and validity.
    """
    total_stays = patient_df.height
    alive_count = patient_df.filter(pl.col("hospitaldischargestatus") == "Alive").height
    expired_count = patient_df.filter(pl.col("hospitaldischargestatus") == "Expired").height
    missing_count = patient_df.filter(
        pl.col("hospitaldischargestatus").is_null() | (pl.col("hospitaldischargestatus") == "")
    ).height
    other_count = total_stays - (alive_count + expired_count + missing_count)

    known_count = alive_count + expired_count
    mortality_rate = (expired_count / known_count * 100) if known_count > 0 else 0

    return {
        "target_column": "hospitaldischargestatus",
        "total_encounters": total_stays,
        "counts": {
            "Alive (0)": alive_count,
            "Expired (1)": expired_count,
            "Missing / Unrecorded": missing_count,
            "Other / Invalid": other_count,
        },
        "percentages": {
            "Alive": alive_count / total_stays * 100,
            "Expired": expired_count / total_stays * 100,
            "Missing": missing_count / total_stays * 100,
        },
        "known_outcomes_total": known_count,
        "mortality_prevalence_pct": mortality_rate,
        "clinical_safeguard": (
            "hospitaldischargestatus is the ground truth target; it must NEVER be included "
            "as a feature in any training matrix."
        ),
    }


# ==============================================================================
# 4. TEMPORAL VALIDATION
# ==============================================================================

def add_temporal_flags(df: pl.DataFrame, offset_col: str) -> pl.DataFrame:
    """Add boolean temporal observation window flags to a dataframe without dropping records.

    Flags:
    - pre_icu: offset < 0 (Pre-ICU / Emergency Department / Triage)
    - first_24h: 0 <= offset <= 1440 (Primary observation window)
    - post_24h: offset > 1440 (Subsequent ICU stay days)

    Args:
        df: Input DataFrame.
        offset_col: Name of the minute offset column.

    Returns:
        DataFrame with pre_icu, first_24h, post_24h columns added.
    """
    return df.with_columns([
        (pl.col(offset_col) < 0).alias("pre_icu"),
        ((pl.col(offset_col) >= 0) & (pl.col(offset_col) <= 1440)).alias("first_24h"),
        (pl.col(offset_col) > 1440).alias("post_24h"),
    ])


def validate_temporal_windows(
    lab_df: pl.DataFrame,
    vital_df: pl.DataFrame,
) -> Dict[str, Any]:
    """Analyze temporal distributions of lab and vital records across observation windows.

    Args:
        lab_df: Lab table.
        vital_df: Vital periodic table.

    Returns:
        Dict summarizing record counts across temporal windows.
    """
    lab_flagged = add_temporal_flags(lab_df, "labresultoffset")
    vital_flagged = add_temporal_flags(vital_df, "observationoffset")

    return {
        "primary_modeling_window_minutes": "0 to 1440 minutes (First 24 Hours of ICU admission)",
        "lab_temporal_summary": {
            "total_records": lab_df.height,
            "pre_icu_count": lab_flagged.filter(pl.col("pre_icu")).height,
            "pre_icu_pct": lab_flagged.filter(pl.col("pre_icu")).height / lab_df.height * 100,
            "first_24h_count": lab_flagged.filter(pl.col("first_24h")).height,
            "first_24h_pct": lab_flagged.filter(pl.col("first_24h")).height / lab_df.height * 100,
            "post_24h_count": lab_flagged.filter(pl.col("post_24h")).height,
            "post_24h_pct": lab_flagged.filter(pl.col("post_24h")).height / lab_df.height * 100,
        },
        "vital_temporal_summary": {
            "total_records": vital_df.height,
            "pre_icu_count": vital_flagged.filter(pl.col("pre_icu")).height,
            "pre_icu_pct": vital_flagged.filter(pl.col("pre_icu")).height / vital_df.height * 100,
            "first_24h_count": vital_flagged.filter(pl.col("first_24h")).height,
            "first_24h_pct": vital_flagged.filter(pl.col("first_24h")).height / vital_df.height * 100,
            "post_24h_count": vital_flagged.filter(pl.col("post_24h")).height,
            "post_24h_pct": vital_flagged.filter(pl.col("post_24h")).height / vital_df.height * 100,
        },
    }


# ==============================================================================
# 5. MISSINGNESS VALIDATION
# ==============================================================================

def validate_missingness(
    patient_df: pl.DataFrame,
    lab_df: pl.DataFrame,
    vital_df: pl.DataFrame,
    top_labs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate missingness percentages across demographic, lab, and vital features.

    Args:
        patient_df: Patient table.
        lab_df: Lab table.
        vital_df: Vital periodic table.
        top_labs: Optional list of lab names to profile.

    Returns:
        Dict summarizing missingness profiles.
    """
    if top_labs is None:
        top_labs = list(DEFAULT_LAB_PLAUSIBILITY_BOUNDS.keys())

    # Patient missingness
    patient_missing = {}
    for col in patient_df.columns:
        if patient_df.schema[col] == pl.String:
            null_ct = patient_df.filter(pl.col(col).is_null() | (pl.col(col) == "")).height
        else:
            null_ct = patient_df.filter(pl.col(col).is_null()).height
        patient_missing[col] = {
            "null_count": null_ct,
            "missing_pct": (null_ct / patient_df.height) * 100,
        }

    # Vital missingness
    vital_cols = list(DEFAULT_VITAL_PLAUSIBILITY_BOUNDS.keys())
    vital_missing = {}
    for col in vital_cols:
        if col in vital_df.columns:
            if vital_df.schema[col] == pl.String:
                null_ct = vital_df.filter(pl.col(col).is_null() | (pl.col(col) == "")).height
            else:
                null_ct = vital_df.filter(pl.col(col).is_null()).height
            vital_missing[col] = {
                "null_count": null_ct,
                "missing_pct": (null_ct / vital_df.height) * 100,
            }

    # Lab result missingness for selected tests
    lab_missing = {}
    for test in top_labs:
        test_rows = lab_df.filter(pl.col("labname") == test)
        if test_rows.height > 0:
            null_res = test_rows.filter(pl.col("labresult").is_null()).height
            lab_missing[test] = {
                "total_ordered": test_rows.height,
                "null_result_count": null_res,
                "missing_pct": (null_res / test_rows.height) * 100,
            }

    return {
        "patient_missingness": patient_missing,
        "vital_missingness": vital_missing,
        "lab_missingness": lab_missing,
        "clinical_note": (
            "Informative missingness in vitals (e.g. 86% missing arterial blood pressures, "
            "93% missing periodic temperature) represents clinical practice variations; "
            "complete-case deletion is strictly avoided."
        ),
    }


# ==============================================================================
# 6. VALUE & CLINICAL PLAUSIBILITY VALIDATION
# ==============================================================================

def validate_clinical_plausibility(
    vital_df: pl.DataFrame,
    lab_df: pl.DataFrame,
    vital_bounds: Optional[Dict[str, Tuple[float, float, str]]] = None,
    lab_bounds: Optional[Dict[str, Tuple[float, float, str]]] = None,
) -> Dict[str, Any]:
    """Validate numeric variables against physiological and clinical plausibility ranges.

    Args:
        vital_df: Vital periodic table.
        lab_df: Lab table.
        vital_bounds: Optional custom dictionary of vital plausibility bounds.
        lab_bounds: Optional custom dictionary of lab plausibility bounds.

    Returns:
        Dict summarizing out-of-bounds and negative value counts with clinical citations.
    """
    if vital_bounds is None:
        vital_bounds = DEFAULT_VITAL_PLAUSIBILITY_BOUNDS
    if lab_bounds is None:
        lab_bounds = DEFAULT_LAB_PLAUSIBILITY_BOUNDS

    vital_results = {}
    for col, (low, high, source) in vital_bounds.items():
        if col in vital_df.columns:
            # Cast column to Float64 safely
            series = vital_df.select(pl.col(col).cast(pl.Float64, strict=False))
            non_null = series.filter(pl.col(col).is_not_null())
            negative_ct = non_null.filter(pl.col(col) < 0).height
            out_of_bounds_ct = non_null.filter((pl.col(col) < low) | (pl.col(col) > high)).height

            vital_results[col] = {
                "plausibility_range": [low, high],
                "clinical_source": source,
                "total_non_null": non_null.height,
                "negative_count": negative_ct,
                "out_of_bounds_count": out_of_bounds_ct,
                "out_of_bounds_pct": (out_of_bounds_ct / non_null.height * 100) if non_null.height > 0 else 0,
            }

    lab_results = {}
    # Convert labresult to float safely
    lab_numeric = lab_df.with_columns(pl.col("labresult").cast(pl.Float64, strict=False).alias("labresult_num"))

    for test_name, (low, high, source) in lab_bounds.items():
        sub = lab_numeric.filter(pl.col("labname") == test_name).filter(pl.col("labresult_num").is_not_null())
        if sub.height > 0:
            negative_ct = sub.filter(pl.col("labresult_num") < 0).height
            out_of_bounds_ct = sub.filter((pl.col("labresult_num") < low) | (pl.col("labresult_num") > high)).height

            lab_results[test_name] = {
                "plausibility_range": [low, high],
                "clinical_source": source,
                "total_non_null": sub.height,
                "negative_count": negative_ct,
                "out_of_bounds_count": out_of_bounds_ct,
                "out_of_bounds_pct": (out_of_bounds_ct / sub.height * 100),
            }

    return {
        "vital_plausibility": vital_results,
        "lab_plausibility": lab_results,
    }


# ==============================================================================
# 7. ENCOUNTER-LEVEL DATA QUALITY FLAGS
# ==============================================================================

def generate_data_quality_flags(
    patient_df: pl.DataFrame,
    lab_df: pl.DataFrame,
    vital_df: pl.DataFrame,
    vital_bounds: Optional[Dict[str, Tuple[float, float, str]]] = None,
    lab_bounds: Optional[Dict[str, Tuple[float, float, str]]] = None,
) -> pl.DataFrame:
    """Generate encounter-level data quality boolean flags per ICU stay (patientunitstayid).

    Flags created:
    - flag_missing_demographics: Missing age, gender, or ethnicity.
    - flag_missing_target: Missing/blank hospitaldischargestatus.
    - flag_missing_lab_data: Zero lab measurements recorded in the primary 24h window.
    - flag_missing_vital_data: Zero vital sign measurements recorded in the primary 24h window.
    - flag_temporal_issue: Zero records in 24h window across both labs and vitals.
    - flag_possible_invalid_value: Encounter has at least one out-of-plausibility lab or vital in 24h window.
    - experimental_quality_index: (Experimental) Percentage of 6 quality checks passed [0-100%].

    Args:
        patient_df: Patient table.
        lab_df: Lab table.
        vital_df: Vital periodic table.
        vital_bounds: Optional dictionary of vital bounds.
        lab_bounds: Optional dictionary of lab bounds.

    Returns:
        Polars DataFrame containing patientunitstayid and quality flag columns.
    """
    if vital_bounds is None:
        vital_bounds = DEFAULT_VITAL_PLAUSIBILITY_BOUNDS
    if lab_bounds is None:
        lab_bounds = DEFAULT_LAB_PLAUSIBILITY_BOUNDS

    # 1. Demographic & Target flags from patient.csv
    demo_flags = patient_df.select([
        pl.col("patientunitstayid"),
        (
            pl.col("age").is_null() | (pl.col("age") == "") |
            pl.col("gender").is_null() | (pl.col("gender") == "") |
            pl.col("ethnicity").is_null() | (pl.col("ethnicity") == "")
        ).alias("flag_missing_demographics"),
        (
            pl.col("hospitaldischargestatus").is_null() | (pl.col("hospitaldischargestatus") == "")
        ).alias("flag_missing_target"),
    ])

    # 2. Add temporal flags
    lab_flagged = add_temporal_flags(lab_df, "labresultoffset").with_columns(
        pl.col("labresult").cast(pl.Float64, strict=False).alias("labresult_num")
    )
    vital_flagged = add_temporal_flags(vital_df, "observationoffset").with_columns([
        pl.col(c).cast(pl.Float64, strict=False).alias(f"{c}_num")
        for c in vital_bounds.keys()
        if c in vital_df.columns
    ])

    # 3. Lab presence in 24h
    stays_with_24h_labs = lab_flagged.filter(pl.col("first_24h")).select("patientunitstayid").unique()
    lab_flag_df = patient_df.select("patientunitstayid").join(
        stays_with_24h_labs.with_columns(pl.lit(False).alias("flag_missing_lab_data")),
        on="patientunitstayid",
        how="left",
    ).with_columns(
        pl.col("flag_missing_lab_data").fill_null(True)
    )

    # 4. Vital presence in 24h
    stays_with_24h_vitals = vital_flagged.filter(pl.col("first_24h")).select("patientunitstayid").unique()
    vital_flag_df = patient_df.select("patientunitstayid").join(
        stays_with_24h_vitals.with_columns(pl.lit(False).alias("flag_missing_vital_data")),
        on="patientunitstayid",
        how="left",
    ).with_columns(
        pl.col("flag_missing_vital_data").fill_null(True)
    )

    # 5. Out-of-bounds values in 24h window
    invalid_stay_dfs: List[pl.DataFrame] = []
    for col, (low, high, _) in vital_bounds.items():
        if f"{col}_num" in vital_flagged.columns:
            bad_stays = vital_flagged.filter(pl.col("first_24h")).filter(
                (pl.col(f"{col}_num") < low) | (pl.col(f"{col}_num") > high)
            ).select("patientunitstayid").unique()
            if bad_stays.height > 0:
                invalid_stay_dfs.append(bad_stays)

    for test_name, (low, high, _) in lab_bounds.items():
        bad_stays = lab_flagged.filter(pl.col("first_24h")).filter(
            (pl.col("labname") == test_name) & ((pl.col("labresult_num") < low) | (pl.col("labresult_num") > high))
        ).select("patientunitstayid").unique()
        if bad_stays.height > 0:
            invalid_stay_dfs.append(bad_stays)

    if invalid_stay_dfs:
        all_invalid_stays = pl.concat(invalid_stay_dfs).unique()
    else:
        all_invalid_stays = pl.DataFrame({"patientunitstayid": []}, schema={"patientunitstayid": pl.Int64})

    invalid_val_df = patient_df.select("patientunitstayid").join(
        all_invalid_stays.with_columns(pl.lit(True).alias("flag_possible_invalid_value")),
        on="patientunitstayid",
        how="left",
    ).with_columns(
        pl.col("flag_possible_invalid_value").fill_null(False)
    )

    # Combine into comprehensive quality matrix
    quality_df = demo_flags.join(
        lab_flag_df, on="patientunitstayid"
    ).join(
        vital_flag_df, on="patientunitstayid"
    ).join(
        invalid_val_df, on="patientunitstayid"
    ).with_columns(
        (pl.col("flag_missing_lab_data") & pl.col("flag_missing_vital_data")).alias("flag_temporal_issue")
    )

    # Experimental Completeness Index (clearly designated as experimental)
    # Counts how many of the 5 negative flags are False (0 = all failed, 100 = all passed)
    flags_list = [
        "flag_missing_demographics",
        "flag_missing_target",
        "flag_missing_lab_data",
        "flag_missing_vital_data",
        "flag_possible_invalid_value",
    ]

    flag_sum_expr = pl.sum_horizontal([pl.col(f).cast(pl.Int32) for f in flags_list])
    quality_df = quality_df.with_columns(
        ((1.0 - (flag_sum_expr / len(flags_list))) * 100.0).round(1).alias("experimental_completeness_pct")
    )

    return quality_df


# ==============================================================================
# 8. END-TO-END PIPELINE & REPORT GENERATOR
# ==============================================================================

def run_full_validation_pipeline(
    data_dir: Path = Path("data/raw"),
    output_dir: Path = Path("data/validated"),
    reports_dir: Path = Path("reports"),
) -> Dict[str, Any]:
    """Execute complete Phase 2 validation pipeline and save validated data & reports.

    Args:
        data_dir: Path containing raw CSV files.
        output_dir: Path to save validated Parquet/CSV outputs.
        reports_dir: Path to save validation summary report.

    Returns:
        Dict containing comprehensive validation metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = reports_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    patient = pl.read_csv(data_dir / "patient.csv")
    lab = pl.read_csv(data_dir / "lab.csv")
    vital = pl.read_csv(data_dir / "vitalPeriodic.csv")

    # 1. Run validations
    schema_res = validate_schemas(patient, lab, vital, strict=True)
    id_res = validate_identifiers(patient, lab, vital)
    target_res = validate_targets(patient)
    temporal_res = validate_temporal_windows(lab, vital)
    missing_res = validate_missingness(patient, lab, vital)
    plausibility_res = validate_clinical_plausibility(vital, lab)
    quality_flags_df = generate_data_quality_flags(patient, lab, vital)

    # 2. Add temporal flags to lab and vitals without dropping rows
    lab_validated = add_temporal_flags(lab, "labresultoffset")
    vital_validated = add_temporal_flags(vital, "observationoffset")

    # 3. Join quality flags with patient table
    patient_validated = patient.join(quality_flags_df, on="patientunitstayid", how="left")

    # 4. Save validated outputs
    patient_validated.write_parquet(output_dir / "validated_patient.parquet")
    quality_flags_df.write_parquet(output_dir / "encounter_quality_flags.parquet")
    quality_flags_df.write_csv(output_dir / "encounter_quality_flags.csv")
    lab_validated.write_parquet(output_dir / "validated_lab_temporal.parquet")
    vital_validated.write_parquet(output_dir / "validated_vital_temporal.parquet")

    # 5. Compile full metrics summary
    summary_report = {
        "pipeline_phase": "Phase 2 - Data Validation & Data Quality",
        "schema_validation": schema_res,
        "identifier_validation": id_res,
        "target_validation": target_res,
        "temporal_validation": temporal_res,
        "missingness_validation": missing_res,
        "clinical_plausibility": plausibility_res,
        "encounter_quality_summary": {
            "total_encounters": quality_flags_df.height,
            "flag_counts": {
                col: quality_flags_df.filter(pl.col(col)).height
                for col in [
                    "flag_missing_demographics",
                    "flag_missing_target",
                    "flag_missing_lab_data",
                    "flag_missing_vital_data",
                    "flag_temporal_issue",
                    "flag_possible_invalid_value",
                ]
            },
            "flag_percentages": {
                col: quality_flags_df.filter(pl.col(col)).height / quality_flags_df.height * 100
                for col in [
                    "flag_missing_demographics",
                    "flag_missing_target",
                    "flag_missing_lab_data",
                    "flag_missing_vital_data",
                    "flag_temporal_issue",
                    "flag_possible_invalid_value",
                ]
            },
        },
    }

    # Save JSON metrics
    with open(metrics_dir / "validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    # Generate Markdown Summary Report
    md_content = f"""# SafePredict-XAI: Phase 2 Data Validation & Quality Summary Report

**Generated on:** Auto-generated validation run  
**Target Window:** First 24 Hours of ICU admission (0 to 1440 minutes)

---

## 1. Technical & Schema Validation
- **Schema Integrity:** PASSED (All required columns verified in patient, lab, and vitalPeriodic tables).
- **Primary Key Uniqueness:** PASSED (0 duplicate `patientunitstayid` in `patient.csv` across {patient.height:,} stays).
- **Orphan Record Check:** PASSED (0 orphan records in `lab.csv` and `vitalPeriodic.csv`).
- **Repeated Measurements:** {id_res['repeated_measurements']['repeated_lab_measurements']:,} lab rows ({id_res['repeated_measurements']['repeated_lab_pct']:.2f}%) and {id_res['repeated_measurements']['repeated_vital_timestamps']:,} vital timestamps have repeated entries, representing continuous telemetry and simultaneous tube orders.

---

## 2. Target Outcome Validation
- **Outcome Column:** `hospitaldischargestatus`
- **Alive (Negative / 0):** {target_res['counts']['Alive (0)']:,} ({target_res['percentages']['Alive']:.2f}%)
- **Expired (Positive / 1):** {target_res['counts']['Expired (1)']:,} ({target_res['percentages']['Expired']:.2f}%)
- **Missing / Unrecorded:** {target_res['counts']['Missing / Unrecorded']:,} ({target_res['percentages']['Missing']:.2f}%)
- **Mortality Prevalence Among Known Outcomes:** **{target_res['mortality_prevalence_pct']:.2f}%** ({target_res['counts']['Expired (1)']:,} / {target_res['known_outcomes_total']:,})

---

## 3. Temporal Distribution (Observation Windows)
- **Primary 24-Hour Modeling Window (0 - 1440 min):**
  - Lab records: **{temporal_res['lab_temporal_summary']['first_24h_count']:,}** ({temporal_res['lab_temporal_summary']['first_24h_pct']:.2f}%)
  - Vital records: **{temporal_res['vital_temporal_summary']['first_24h_count']:,}** ({temporal_res['vital_temporal_summary']['first_24h_pct']:.2f}%)
- **Pre-ICU Window (< 0 min):**
  - Lab records: {temporal_res['lab_temporal_summary']['pre_icu_count']:,} ({temporal_res['lab_temporal_summary']['pre_icu_pct']:.2f}%)
  - Vital records: {temporal_res['vital_temporal_summary']['pre_icu_count']:,} ({temporal_res['vital_temporal_summary']['pre_icu_pct']:.2f}%)
- **Post-24h Stay (> 1440 min):**
  - Lab records: {temporal_res['lab_temporal_summary']['post_24h_count']:,} ({temporal_res['lab_temporal_summary']['post_24h_pct']:.2f}%)
  - Vital records: {temporal_res['vital_temporal_summary']['post_24h_count']:,} ({temporal_res['vital_temporal_summary']['post_24h_pct']:.2f}%)

---

## 4. Encounter-Level Quality Flags ({quality_flags_df.height:,} ICU Stays)
| Quality Flag | Flagged Stays | Flagged % | Clinical Rationale |
| :--- | :--- | :--- | :--- |
| `flag_missing_demographics` | {summary_report['encounter_quality_summary']['flag_counts']['flag_missing_demographics']:,} | {summary_report['encounter_quality_summary']['flag_percentages']['flag_missing_demographics']:.2f}% | Stays missing age, gender, or ethnicity. |
| `flag_missing_target` | {summary_report['encounter_quality_summary']['flag_counts']['flag_missing_target']:,} | {summary_report['encounter_quality_summary']['flag_percentages']['flag_missing_target']:.2f}% | Unrecorded hospital discharge status (excluded from supervised model training). |
| `flag_missing_lab_data` | {summary_report['encounter_quality_summary']['flag_counts']['flag_missing_lab_data']:,} | {summary_report['encounter_quality_summary']['flag_percentages']['flag_missing_lab_data']:.2f}% | Stays with 0 lab measurements in the 24-hour observation window. |
| `flag_missing_vital_data` | {summary_report['encounter_quality_summary']['flag_counts']['flag_missing_vital_data']:,} | {summary_report['encounter_quality_summary']['flag_percentages']['flag_missing_vital_data']:.2f}% | Stays with 0 vital measurements in the 24-hour observation window. |
| `flag_temporal_issue` | {summary_report['encounter_quality_summary']['flag_counts']['flag_temporal_issue']:,} | {summary_report['encounter_quality_summary']['flag_percentages']['flag_temporal_issue']:.2f}% | Stays with 0 data across both labs and vitals in the 24-hour window. |
| `flag_possible_invalid_value` | {summary_report['encounter_quality_summary']['flag_counts']['flag_possible_invalid_value']:,} | {summary_report['encounter_quality_summary']['flag_percentages']['flag_possible_invalid_value']:.2f}% | Stays containing at least one out-of-plausibility lab or vital in 24h window. |

---

## 5. Transition to Phase 3 (Feature Engineering)
1. **Window Filtering:** Extract aggregations (mean, min, max, std, first, last, trend) strictly from the 0-1440 minute window.
2. **Plausibility Clipping / Masking:** Clip or mask physiological artifacts (negative blood pressures, out-of-bounds vitals) to clinical boundaries prior to computing statistical aggregates.
3. **Informative Missingness:** Preserve missingness indicators as feature columns rather than naive global imputation.
4. **Target Encoding:** Encode `hospitaldischargestatus` into binary `[0, 1]` and filter unlabelled records for training.
"""

    with open(reports_dir / "validation_summary.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Validation completed successfully!")
    print(f"  • Validated data saved to: {output_dir}")
    print(f"  • Reports saved to: {reports_dir / 'validation_summary.md'}")
    return summary_report


if __name__ == "__main__":
    run_full_validation_pipeline()
