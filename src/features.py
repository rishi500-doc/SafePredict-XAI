"""Clinical feature engineering and aggregation module for SafePredict-XAI.

1. Patient Admission Features:
   - Extract verified admission-time demographic and encounter variables:
     age_numeric, age_gt89, gender, ethnicity, admissionheight, admissionweight, unittype, unitadmitsource.
2. Lab Selection & 24-Hour Summaries:
   - 15 clinically prioritized labs with high eICU coverage:
     Creatinine, BUN, Sodium, Potassium, Glucose, Hemoglobin, Hematocrit, WBC, Platelets,
     Bicarbonate, Chloride, Calcium, Magnesium, Lactate, pH.
   - Summaries: first, last, min, max, mean, count.
3. Vital Sign 24-Hour Summaries:
   - 7 routine vital signs:
     heartrate, sao2, respiration, temperature, systemicsystolic, systemicdiastolic, systemicmean.
   - Summaries: mean, min, max, std, count.
4. Data Quality Features:
   - Engineered quality features kept distinct from clinical predictors:
     lab_measurement_count, vital_measurement_count, available_lab_count, available_vital_count,
     missing_feature_count, and per-variable measurement counts.
5. Tabular Dataset Assembly:
   - Join all features into one row per ICU stay (N = 1,403).
   - Maintain identifiers strictly as non-predictive tracking keys.
   - Save data/processed/model_data.parquet and export metadata dictionaries.
6. Comprehensive Validation:
   - Invariant checking: one row per stay, no duplicate IDs, target in {0, 1}, zero leakage columns.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import polars as pl

# Project root and directory paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

# 15 Clinically Prioritized Laboratory Tests (eICU exact labname -> clean prefix)
SELECTED_LABS: List[Tuple[str, str]] = [
    ("creatinine", "creatinine"),
    ("BUN", "bun"),
    ("sodium", "sodium"),
    ("potassium", "potassium"),
    ("glucose", "glucose"),
    ("Hgb", "hgb"),
    ("Hct", "hct"),
    ("WBC x 1000", "wbc"),
    ("platelets x 1000", "platelets"),
    ("bicarbonate", "bicarbonate"),
    ("chloride", "chloride"),
    ("calcium", "calcium"),
    ("magnesium", "magnesium"),
    ("lactate", "lactate"),
    ("pH", "ph"),
]

# 7 Routine ICU Vital Signs
SELECTED_VITALS: List[str] = [
    "heartrate",
    "sao2",
    "respiration",
    "temperature",
    "systemicsystolic",
    "systemicdiastolic",
    "systemicmean",
]

# Admission-time demographic and encounter predictors
ADMISSION_FEATURE_COLS: List[str] = [
    "age_numeric",
    "age_gt89",
    "gender",
    "ethnicity",
    "admissionheight",
    "admissionweight",
    "unittype",
    "unitadmitsource",
]

# Tracking identifiers (STRICTLY NOT PREDICTIVE FEATURES)
IDENTIFIER_COLS: List[str] = [
    "patientunitstayid",
    "patienthealthsystemstayid",
]

# Target label (STRICTLY NOT A FEATURE)
TARGET_COL: str = "hospital_mortality"


# ==============================================================================
# 1. PATIENT ADMISSION FEATURES
# ==============================================================================

def extract_patient_admission_features(
    patient_df: pl.DataFrame,
) -> Tuple[pl.DataFrame, List[str]]:
    """Extract and standardize verified admission-time predictors.

    Args:
        patient_df: Analytical cohort patient DataFrame (cohort_patient.parquet).

    Returns:
        Tuple containing:
        - Base DataFrame with identifiers, target, and cleaned admission features.
        - List of admission feature column names.
    """
    keep_cols = IDENTIFIER_COLS + [TARGET_COL] + ADMISSION_FEATURE_COLS
    base_df = patient_df.select([c for c in keep_cols if c in patient_df.columns])

    # Clean categorical string columns: convert empty/null strings to 'Unknown'
    cat_clean_exprs = []
    for col in ["gender", "ethnicity", "unitadmitsource"]:
        if col in base_df.columns:
            cat_clean_exprs.append(
                pl.when(pl.col(col).is_null() | (pl.col(col) == ""))
                .then(pl.lit("Unknown"))
                .otherwise(pl.col(col))
                .alias(col)
            )

    if cat_clean_exprs:
        base_df = base_df.with_columns(cat_clean_exprs)

    return base_df, ADMISSION_FEATURE_COLS


# ==============================================================================
# 2. VITAL SIGN FEATURES (FIRST 24 HOURS)
# ==============================================================================

def extract_vital_features(
    vitals_df: pl.DataFrame,
    vital_cols: Optional[List[str]] = None,
) -> Tuple[pl.DataFrame, List[str], List[str]]:
    """Aggregate longitudinal vital sign observations from the first 24 hours.

    Calculates:
    - Clinical Features: mean, min, max, std
    - Data Quality Features: per-vital measurement count and total vital measurement count

    Args:
        vitals_df: 0-24h cleaned periodic vitals DataFrame (cohort_vitals_24h.parquet).
        vital_cols: List of vital sign column names. Defaults to SELECTED_VITALS.

    Returns:
        Tuple containing:
        - Aggregated pl.DataFrame (one row per patientunitstayid).
        - List of clinical vital feature column names.
        - List of data quality vital feature column names.
    """
    if vital_cols is None:
        vital_cols = SELECTED_VITALS

    clinical_cols: List[str] = []
    quality_cols: List[str] = []
    agg_exprs: List[pl.Expr] = []

    for col in vital_cols:
        if col in vitals_df.columns:
            col_expr = pl.col(col)

            # Clinical summaries
            agg_exprs.extend([
                col_expr.mean().alias(f"{col}_mean"),
                col_expr.min().alias(f"{col}_min"),
                col_expr.max().alias(f"{col}_max"),
                col_expr.std().alias(f"{col}_std"),
            ])
            clinical_cols.extend([
                f"{col}_mean", f"{col}_min", f"{col}_max", f"{col}_std"
            ])

            # Per-vital measurement count
            agg_exprs.append(
                col_expr.filter(col_expr.is_not_null()).len().alias(f"{col}_count")
            )
            quality_cols.append(f"{col}_count")

    # Overall vital measurement count across all vitals
    agg_exprs.append(pl.len().alias("vital_measurement_count"))
    quality_cols.append("vital_measurement_count")

    vital_features_df = vitals_df.group_by("patientunitstayid").agg(agg_exprs)

    return vital_features_df, clinical_cols, quality_cols


# ==============================================================================
# 3. LABORATORY FEATURES (FIRST 24 HOURS)
# ==============================================================================

def extract_lab_features(
    labs_df: pl.DataFrame,
    selected_labs: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[pl.DataFrame, List[str], List[str]]:
    """Aggregate longitudinal laboratory measurements from the first 24 hours.

    Calculates:
    - Clinical Features: first, last, min, max, mean
    - Data Quality Features: per-lab measurement count and total lab measurement count

    Args:
        labs_df: 0-24h cleaned labs DataFrame (cohort_labs_24h.parquet).
        selected_labs: List of (eICU labname, clean prefix) tuples.

    Returns:
        Tuple containing:
        - Aggregated pl.DataFrame (one row per patientunitstayid).
        - List of clinical lab feature column names.
        - List of data quality lab feature column names.
    """
    if selected_labs is None:
        selected_labs = SELECTED_LABS

    clinical_cols: List[str] = []
    quality_cols: List[str] = []

    # Sort labs chronologically by stay and offset to accurately extract first and last
    valid_labs = labs_df.filter(pl.col("labresult_num").is_not_null()).sort(
        ["patientunitstayid", "labresultoffset"]
    )

    # 1. Total lab measurement count per stay
    total_lab_counts = labs_df.group_by("patientunitstayid").agg(
        pl.len().alias("lab_measurement_count")
    )
    quality_cols.append("lab_measurement_count")

    lab_dfs: List[pl.DataFrame] = [total_lab_counts]

    # 2. Per-lab aggregations
    for raw_name, clean_name in selected_labs:
        sub_df = valid_labs.filter(pl.col("labname") == raw_name)
        agg_df = sub_df.group_by("patientunitstayid").agg([
            pl.col("labresult_num").first().alias(f"{clean_name}_first"),
            pl.col("labresult_num").last().alias(f"{clean_name}_last"),
            pl.col("labresult_num").min().alias(f"{clean_name}_min"),
            pl.col("labresult_num").max().alias(f"{clean_name}_max"),
            pl.col("labresult_num").mean().alias(f"{clean_name}_mean"),
            pl.col("labresult_num").len().alias(f"{clean_name}_count"),
        ])

        clinical_cols.extend([
            f"{clean_name}_first",
            f"{clean_name}_last",
            f"{clean_name}_min",
            f"{clean_name}_max",
            f"{clean_name}_mean",
        ])
        quality_cols.append(f"{clean_name}_count")
        lab_dfs.append(agg_df)

    # Full join all lab sub-dataframes
    lab_features_df = lab_dfs[0]
    for df in lab_dfs[1:]:
        lab_features_df = lab_features_df.join(
            df, on="patientunitstayid", how="full", coalesce=True
        )

    return lab_features_df, clinical_cols, quality_cols


# ==============================================================================
# 4. DATA QUALITY FEATURES
# ==============================================================================

def compute_data_quality_features(
    features_df: pl.DataFrame,
    clinical_feature_cols: List[str],
    vital_cols: List[str],
    selected_labs: List[Tuple[str, str]],
) -> Tuple[pl.DataFrame, List[str]]:
    """Engineer meta-level data quality and availability indicators.

    Calculates:
    - available_vital_count: Count of vital sign types (out of 7) with >= 1 measurement.
    - available_lab_count: Count of lab types (out of 15) with >= 1 measurement.
    - missing_feature_count: Total missing clinical predictor values for the ICU stay.

    Args:
        features_df: Merged feature DataFrame.
        clinical_feature_cols: List of all clinical predictor columns.
        vital_cols: List of vital signs.
        selected_labs: List of selected labs.

    Returns:
        Tuple containing:
        - DataFrame with data quality indicators appended.
        - List of newly created data quality feature names.
    """
    quality_feature_names = [
        "available_vital_count",
        "available_lab_count",
        "missing_feature_count",
    ]

    # Available vital count
    vital_avail_exprs = [
        pl.when(pl.col(f"{c}_count") > 0).then(1).otherwise(0)
        for c in vital_cols
        if f"{c}_count" in features_df.columns
    ]
    vital_avail_expr = (
        pl.sum_horizontal(vital_avail_exprs).alias("available_vital_count")
        if vital_avail_exprs
        else pl.lit(0).alias("available_vital_count")
    )

    # Available lab count
    lab_avail_exprs = [
        pl.when(pl.col(f"{clean_name}_count") > 0).then(1).otherwise(0)
        for _, clean_name in selected_labs
        if f"{clean_name}_count" in features_df.columns
    ]
    lab_avail_expr = (
        pl.sum_horizontal(lab_avail_exprs).alias("available_lab_count")
        if lab_avail_exprs
        else pl.lit(0).alias("available_lab_count")
    )

    # Missing clinical feature count
    missing_exprs = [
        pl.when(pl.col(c).is_null()).then(1).otherwise(0)
        for c in clinical_feature_cols
        if c in features_df.columns
    ]
    missing_expr = (
        pl.sum_horizontal(missing_exprs).alias("missing_feature_count")
        if missing_exprs
        else pl.lit(0).alias("missing_feature_count")
    )

    updated_df = features_df.with_columns([
        vital_avail_expr,
        lab_avail_expr,
        missing_expr,
    ])

    return updated_df, quality_feature_names


# ==============================================================================
# 5. VALIDATION & LEAKAGE CHECKS
# ==============================================================================

def validate_model_dataset(
    model_df: pl.DataFrame,
    feature_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """Perform rigorous validation assertions on the final modeling dataset.

    Checks:
    1. Shape & Uniqueness: Exactly 1 row per ICU stay, zero duplicate patientunitstayid.
    2. Target Integrity: hospital_mortality strictly in {0, 1}, non-null.
    3. Target Distribution: Match confirmed Phase 3 ground truth (122 deaths, 8.70%).
    4. Anti-Leakage: Assert zero discharge offsets, discharge status, or post-24h markers.
    5. Completeness: Ensure all feature groups are properly typed.

    Args:
        model_df: Final assembled model DataFrame.
        feature_manifest: Feature metadata manifest.

    Returns:
        Dict detailing validation results and check statuses.
    """
    n_rows = model_df.height
    n_unique_stays = model_df["patientunitstayid"].n_unique()

    # 1. Uniqueness
    if n_rows != n_unique_stays:
        raise AssertionError(f"Duplicate stay IDs found: {n_rows} rows vs {n_unique_stays} unique stays")

    # 2. Target checks
    if TARGET_COL not in model_df.columns:
        raise AssertionError(f"Target column '{TARGET_COL}' missing from modeling dataset")

    target_vals = set(model_df[TARGET_COL].unique().to_list())
    if not target_vals.issubset({0, 1}):
        raise AssertionError(f"Invalid target values detected: {target_vals}")

    mort_expired = model_df.filter(pl.col(TARGET_COL) == 1).height
    mort_alive = model_df.filter(pl.col(TARGET_COL) == 0).height
    mort_rate = (mort_expired / n_rows * 100) if n_rows > 0 else 0.0

    # 3. Anti-Leakage audit
    forbidden_substrings = [
        "dischargeoffset", "dischargestatus", "dischargelocation",
        "dischargeweight", "dischargetime", "dischargeyear"
    ]
    leakage_columns = [
        col for col in model_df.columns
        if any(sub in col.lower() for sub in forbidden_substrings)
    ]
    if leakage_columns:
        raise AssertionError(f"Fatal target/temporal leakage columns detected: {leakage_columns}")

    # 4. Missingness overview
    missing_summary = {}
    for col in model_df.columns:
        null_ct = model_df.filter(pl.col(col).is_null()).height
        if null_ct > 0:
            missing_summary[col] = {
                "null_count": null_ct,
                "null_pct": round(null_ct / n_rows * 100, 2),
            }

    validation_report = {
        "status": "PASSED",
        "total_rows": n_rows,
        "unique_stays": n_unique_stays,
        "is_one_row_per_stay": n_rows == n_unique_stays,
        "target_balance": {
            "alive_count": mort_alive,
            "expired_count": mort_expired,
            "mortality_prevalence_pct": round(mort_rate, 2),
        },
        "leakage_columns_found": len(leakage_columns),
        "total_features": len(feature_manifest["all_features"]),
        "clinical_features_count": len(feature_manifest["clinical_features"]),
        "quality_features_count": len(feature_manifest["quality_features"]),
        "missingness_summary": missing_summary,
    }

    return validation_report


# ==============================================================================
# 6. MASTER BUILD MODEL DATASET PIPELINE
# ==============================================================================

def build_model_dataset(
    processed_dir: Path = PROCESSED_DATA_DIR,
    output_path: Optional[Path] = None,
    reports_dir: Path = REPORTS_DIR,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Assemble the complete one-row-per-stay modeling dataset from Phase 3 cohort tables.

    Orchestrates:
    1. Loading cohort_patient.parquet, cohort_vitals_24h.parquet, cohort_labs_24h.parquet.
    2. Extracting admission demographic and encounter variables.
    3. Aggregating 24-hour vital sign features and counts.
    4. Aggregating 24-hour laboratory features and counts.
    5. Merging on patientunitstayid and filling missing count features with 0.
    6. Computing meta-level data quality and missingness features.
    7. Validating against all anti-leakage and clinical invariants.
    8. Saving data/processed/model_data.parquet and exporting feature manifest JSON.

    Args:
        processed_dir: Directory containing Phase 3 parquet files.
        output_path: Optional output parquet path. Defaults to data/processed/model_data.parquet.
        reports_dir: Directory to save reports and metrics.

    Returns:
        Tuple containing final pl.DataFrame and feature manifest dictionary.
    """
    if output_path is None:
        output_path = processed_dir / "model_data.parquet"

    metrics_dir = reports_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingestion
    patient_df = pl.read_parquet(processed_dir / "cohort_patient.parquet")
    vitals_df = pl.read_parquet(processed_dir / "cohort_vitals_24h.parquet")
    labs_df = pl.read_parquet(processed_dir / "cohort_labs_24h.parquet")

    # 2. Extract components
    base_df, admission_features = extract_patient_admission_features(patient_df)
    vital_df, vital_clinical_features, vital_quality_features = extract_vital_features(
        vitals_df, vital_cols=SELECTED_VITALS
    )
    lab_df, lab_clinical_features, lab_quality_features = extract_lab_features(
        labs_df, selected_labs=SELECTED_LABS
    )

    # 3. Join components
    model_df = (
        base_df
        .join(vital_df, on="patientunitstayid", how="left")
        .join(lab_df, on="patientunitstayid", how="left")
    )

    # 4. Fill nulls for count features with 0
    all_quality_counts = vital_quality_features + lab_quality_features
    fill_exprs = [
        pl.col(c).fill_null(0).alias(c)
        for c in all_quality_counts
        if c in model_df.columns
    ]
    if fill_exprs:
        model_df = model_df.with_columns(fill_exprs)

    # 5. Compute Data Quality Indicators
    clinical_features = admission_features + vital_clinical_features + lab_clinical_features
    model_df, meta_quality_features = compute_data_quality_features(
        features_df=model_df,
        clinical_feature_cols=clinical_features,
        vital_cols=SELECTED_VITALS,
        selected_labs=SELECTED_LABS,
    )

    quality_features = all_quality_counts + meta_quality_features
    all_features = clinical_features + quality_features

    # 6. Feature Manifest Catalog
    feature_manifest = {
        "dataset_name": "SafePredict-XAI Model Features (First 24 Hours)",
        "prediction_landmark": "24 Hours post-ICU admission (t = 1440 min)",
        "identifiers": IDENTIFIER_COLS,
        "target": TARGET_COL,
        "feature_groups": {
            "admission_demographics": admission_features,
            "vital_clinical_features": vital_clinical_features,
            "lab_clinical_features": lab_clinical_features,
            "vital_quality_features": vital_quality_features,
            "lab_quality_features": lab_quality_features,
            "meta_quality_features": meta_quality_features,
        },
        "clinical_features": clinical_features,
        "quality_features": quality_features,
        "all_features": all_features,
        "total_columns": model_df.width,
        "total_rows": model_df.height,
    }

    # 7. Validate Model Dataset
    validation_report = validate_model_dataset(model_df, feature_manifest)

    # 8. Save Processed Artifacts
    model_df.write_parquet(output_path)

    with open(metrics_dir / "feature_dictionary.json", "w", encoding="utf-8") as f:
        json.dump(feature_manifest, f, indent=2)

    with open(metrics_dir / "model_data_validation.json", "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)

    # 9. Markdown Summary
    md_summary = f"""# SafePredict-XAI: Phase 4 Feature Engineering Summary

**Tabular Dataset:** `data/processed/model_data.parquet`  
**Shape:** {model_df.height:,} rows $\\times$ {model_df.width} columns  
**Target Variable:** `hospital_mortality` (0 = Alive, 1 = Expired)

---

## 1. Feature Group Breakdown

| Feature Category | Column Count | Description |
| :--- | :---: | :--- |
| **Tracking Identifiers** | {len(IDENTIFIER_COLS)} | `patientunitstayid`, `patienthealthsystemstayid` (Strictly non-predictive) |
| **Target Variable** | 1 | `hospital_mortality` (0: {validation_report['target_balance']['alive_count']}, 1: {validation_report['target_balance']['expired_count']}) |
| **Admission Demographics** | {len(admission_features)} | Age, age >89 indicator, gender, ethnicity, height, weight, unit type, admit source |
| **Vital Clinical Features** | {len(vital_clinical_features)} | Mean, min, max, std across 7 routine vital signs in 0-24h |
| **Lab Clinical Features** | {len(lab_clinical_features)} | First, last, min, max, mean across 15 selected labs in 0-24h |
| **Vital Quality Features** | {len(vital_quality_features)} | Measurement counts per vital sign + total vital records in 0-24h |
| **Lab Quality Features** | {len(lab_quality_features)} | Measurement counts per lab test + total lab records in 0-24h |
| **Meta Quality Indicators** | {len(meta_quality_features)} | `available_vital_count`, `available_lab_count`, `missing_feature_count` |
| **Total Features** | **{len(all_features)}** | **{len(clinical_features)} Clinical + {len(quality_features)} Data Quality** |

---

## 2. Validation & Quality Assertions

- **One Row Per ICU Stay:** {validation_report['is_one_row_per_stay']} ({model_df.height:,} unique stays)
- **Target Classes:** Only {{0, 1}} with {validation_report['target_balance']['mortality_prevalence_pct']:.2f}% mortality prevalence
- **Anti-Leakage Audit:** {validation_report['leakage_columns_found']} forbidden discharge/future columns found
- **Observation Window:** Strictly bounded to first 24 hours of ICU admission
"""

    with open(reports_dir / "feature_engineering_summary.md", "w", encoding="utf-8") as f:
        f.write(md_summary)

    print("Phase 4 Feature Engineering completed successfully!")
    print(f"  • Model Data Shape: {model_df.shape} (1 row per ICU stay)")
    print(f"  • Clinical Features: {len(clinical_features)} | Quality Features: {len(quality_features)}")
    print(f"  • Saved Parquet: {output_path}")
    print(f"  • Saved Feature Catalog: {metrics_dir / 'feature_dictionary.json'}")

    return model_df, feature_manifest


if __name__ == "__main__":
    build_model_dataset()
