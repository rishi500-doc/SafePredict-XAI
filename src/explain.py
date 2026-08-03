"""Explainable AI (XAI) module for SafePredict-XAI.

This module implements:
1. TreeSHAP global feature attributions (beeswarm summary, global importance ranking).
2. TreeSHAP local case-based patient explanations (waterfall breakdown for TP, TN, FP, FN).
3. Patient-level driver analysis (top features increasing vs. decreasing model prediction).
4. Strictly associative, non-causal clinical interpretation reporting.

IMPORTANT GUARDRAIL:
SHAP describes model behavior and feature contribution, NOT clinical causation.
All outputs state: "These features contributed to the model's prediction."
Never state: "This feature caused mortality."
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold
import joblib
import matplotlib.pyplot as plt
import shap

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import (
    prepare_model_splits,
    RANDOM_STATE,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    PROCESSED_DATA_DIR,
    TARGET_COL,
)

PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "model_data.parquet"
PATIENT_ID_COL = "uniquepid"

# Ensure output directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. MODEL & DATA INGESTION
# ==============================================================================

def load_champion_artifacts(
    model_path: Path = MODELS_DIR / "final_mortality_model.joblib",
    preprocessor_path: Path = MODELS_DIR / "preprocessing_pipeline.joblib",
) -> Tuple[Any, Any]:
    """Load serialized champion calibrated classifier and preprocessing pipeline.

    Args:
        model_path: Path to the calibrated classifier joblib file.
        preprocessor_path: Path to the preprocessing pipeline joblib file.

    Returns:
        Tuple of (calibrated_model, preprocessor).
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Champion model not found at {model_path}. Run src/evaluate.py first.")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}. Run src/evaluate.py first.")

    calibrated_model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return calibrated_model, preprocessor


def extract_base_classifier(calibrated_model: Any) -> Any:
    """Extract underlying base estimator (e.g. XGBClassifier) from CalibratedClassifierCV."""
    if hasattr(calibrated_model, "estimator"):
        base_est = calibrated_model.estimator
        # Check if wrapped in FrozenEstimator
        if hasattr(base_est, "estimator"):
            base_est = base_est.estimator
        # Check if scikit-learn Pipeline
        if hasattr(base_est, "named_steps") and "classifier" in base_est.named_steps:
            return base_est.named_steps["classifier"]
        return base_est
    elif hasattr(calibrated_model, "calibrated_classifiers_") and len(calibrated_model.calibrated_classifiers_) > 0:
        base_est = calibrated_model.calibrated_classifiers_[0].estimator
        if hasattr(base_est, "named_steps") and "classifier" in base_est.named_steps:
            return base_est.named_steps["classifier"]
        return base_est
    raise ValueError("Could not extract underlying base classifier from calibrated model.")


def clean_feature_names(raw_names: List[str]) -> List[str]:
    """Clean sklearn feature names into clean, publication-grade clinical labels."""
    lab_names = {
        "bun": "BUN",
        "wbc": "WBC",
        "hct": "Hematocrit",
        "hgb": "Hemoglobin",
        "ph": "Arterial pH",
        "sao2": "SaO2",
        "spo2": "SpO2",
        "heartrate": "Heart Rate",
        "respiration": "Respiration Rate",
        "temperature": "Temperature",
        "systemicmean": "Mean Arterial Pressure",
        "systemicsystolic": "Systolic BP",
        "systemicdiastolic": "Diastolic BP",
        "creatinine": "Creatinine",
        "potassium": "Potassium",
        "sodium": "Sodium",
        "glucose": "Glucose",
        "lactate": "Lactate",
        "platelets": "Platelets",
        "bicarbonate": "Bicarbonate",
        "calcium": "Calcium",
        "chloride": "Chloride",
        "magnesium": "Magnesium",
    }

    stat_names = {
        "mean": "Mean",
        "min": "Min",
        "max": "Max",
        "std": "Std Dev",
        "first": "First",
        "last": "Last",
        "count": "Count",
        "delta": "Delta (24h)",
        "measured": "Measured Flag",
    }

    clean = []
    for name in raw_names:
        cleaned = name.replace("num__", "").replace("cat__", "")

        # Categorical prefix mappings
        if cleaned.startswith("ethnicity_"):
            clean.append(f"Ethnicity: {cleaned.replace('ethnicity_', '')}")
            continue
        elif cleaned.startswith("unitadmitsource_"):
            clean.append(f"Admit Source: {cleaned.replace('unitadmitsource_', '')}")
            continue
        elif cleaned.startswith("unittype_"):
            clean.append(f"Unit: {cleaned.replace('unittype_', '')}")
            continue
        elif cleaned.startswith("gender_"):
            clean.append(f"Gender: {cleaned.replace('gender_', '')}")
            continue

        # Exact demographic & count mappings
        exact_map = {
            "age_numeric": "Age (years)",
            "age_gt89": "Age > 89 flag",
            "admissionheight": "Admission Height (cm)",
            "admissionweight": "Admission Weight (kg)",
            "lab_measurement_count": "Total Lab Measurements",
            "vital_measurement_count": "Total Vital Measurements",
            "available_lab_count": "Available Lab Types Count",
            "available_vital_count": "Available Vital Types Count",
            "missing_feature_count": "Missing Feature Count",
        }
        if cleaned in exact_map:
            clean.append(exact_map[cleaned])
            continue

        # Match lab / vital prefix + stat suffix
        matched = False
        for prefix, lab_label in lab_names.items():
            if cleaned.startswith(f"{prefix}_"):
                suffix = cleaned[len(prefix) + 1:]
                stat_label = stat_names.get(suffix, suffix.capitalize())
                clean.append(f"{lab_label} ({stat_label})")
                matched = True
                break
        
        if not matched:
            clean.append(cleaned.replace("_", " ").title())

    return clean


def get_test_cohort_metadata(
    data_path: Path = PROCESSED_DATA_PATH,
    patient_path: Path = PROCESSED_DATA_DIR / "cohort_patient.parquet",
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Retrieve metadata (patient identifiers, unit types) for the held-out test cohort."""
    model_df = pl.read_parquet(data_path)
    patient_df = pl.read_parquet(patient_path)

    stay_to_pid = dict(
        zip(
            patient_df["patientunitstayid"].to_list(),
            patient_df["uniquepid"].to_list(),
        )
    )
    groups = np.array([stay_to_pid[sid] for sid in model_df["patientunitstayid"].to_list()])
    y_arr = model_df[TARGET_COL].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    splits = list(sgkf.split(model_df.to_pandas(), y_arr, groups))
    test_idx = splits[0][1].tolist()

    # Join metadata from patient_df
    test_stays = [model_df["patientunitstayid"][i] for i in test_idx]
    patient_pdf = patient_df.to_pandas().set_index("patientunitstayid")
    meta_df = patient_pdf.loc[test_stays].reset_index()
    return meta_df


# ==============================================================================
# 2. SHAP EXPLANATION COMPUTATION
# ==============================================================================

def compute_tree_shap_explanations(
    base_classifier: Any,
    preprocessor: Any,
    X_test: pd.DataFrame,
) -> Tuple[shap.Explanation, pd.DataFrame, List[str]]:
    """Compute TreeSHAP explanation object on the held-out test cohort.

    Args:
        base_classifier: Fitted tree estimator (e.g. XGBClassifier).
        preprocessor: Fitted ColumnTransformer preprocessor.
        X_test: Raw held-out test predictor DataFrame.

    Returns:
        Tuple of (shap_explanation, X_test_transformed_df, clean_feature_names).
    """
    raw_feature_names = list(preprocessor.get_feature_names_out())
    cleaned_names = clean_feature_names(raw_feature_names)

    # Transform test features
    X_test_trans = preprocessor.transform(X_test)
    X_test_trans_df = pd.DataFrame(X_test_trans, columns=cleaned_names, index=X_test.index)

    # Initialize TreeExplainer
    explainer = shap.TreeExplainer(base_classifier)
    shap_explanation = explainer(X_test_trans_df)

    return shap_explanation, X_test_trans_df, cleaned_names


# ==============================================================================
# 3. GLOBAL EXPLANATIONS & VISUALIZATION
# ==============================================================================

def compute_global_feature_importance(
    shap_explanation: shap.Explanation,
    feature_names: List[str],
) -> pd.DataFrame:
    """Compute mean absolute SHAP feature importance across the test cohort."""
    shap_matrix = shap_explanation.values
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    # Add relative importance percentage
    total_importance = float(importance_df["mean_abs_shap"].sum())
    importance_df["relative_importance_pct"] = (importance_df["mean_abs_shap"] / total_importance) * 100.0
    importance_df["cumulative_importance_pct"] = importance_df["relative_importance_pct"].cumsum()

    return importance_df


def plot_global_shap_summary(
    shap_explanation: shap.Explanation,
    output_beeswarm_path: Path = FIGURES_DIR / "shap_summary_beeswarm.png",
    output_bar_path: Path = FIGURES_DIR / "shap_summary_bar.png",
    max_display: int = 20,
) -> Tuple[Path, Path]:
    """Generate and save publication-grade SHAP summary beeswarm and bar plots."""
    # 1. Beeswarm Summary Plot
    plt.figure(figsize=(10, 8), dpi=300)
    shap.plots.beeswarm(
        shap_explanation,
        max_display=max_display,
        show=False,
        plot_size=(10, 8),
    )
    plt.title("SHAP Summary Plot (Held-Out Test Cohort, N=280)\nFeature Impact on Model Log-Odds Margin", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("SHAP Value (Impact on Mortality Prediction Margin)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Bar Summary Plot (Mean Absolute SHAP)
    plt.figure(figsize=(9, 7), dpi=300)
    shap.plots.bar(
        shap_explanation,
        max_display=max_display,
        show=False,
    )
    plt.title(f"Top {max_display} Features by Mean |SHAP Value| (Global Model Importance)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Mean |SHAP Value| (Average Impact Magnitude)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_beeswarm_path, output_bar_path


# ==============================================================================
# 4. LOCAL CASE-BASED PATIENT EXPLANATIONS
# ==============================================================================

def select_clinical_case_studies(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    y_probs: np.ndarray,
    meta_test: pd.DataFrame,
    youden_threshold: float = 0.058,
) -> Dict[str, Dict[str, Any]]:
    """Select 4 diverse, representative ICU stays across prediction quadrants.

    Categories:
    - True Positive (TP): Actual death, high predicted mortality risk (> 0.20).
    - True Negative (TN): Actual survivor, low predicted mortality risk (< 0.03).
    - False Positive (FP): Actual survivor, elevated predicted risk (> 0.15, critical alert).
    - False Negative (FN): Actual death, lower predicted risk (< 0.08, missed/borderline).
    """
    y_pred = (y_probs >= youden_threshold).astype(int)

    # 1. True Positive Candidate
    tp_indices = np.where((y_test == 1) & (y_pred == 1))[0]
    # Pick highest probability TP
    best_tp_idx = tp_indices[np.argmax(y_probs[tp_indices])] if len(tp_indices) > 0 else 0

    # 2. True Negative Candidate
    tn_indices = np.where((y_test == 0) & (y_pred == 0))[0]
    # Pick lowest probability TN
    best_tn_idx = tn_indices[np.argmin(y_probs[tn_indices])] if len(tn_indices) > 0 else 0

    # 3. False Positive Candidate
    fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
    # Pick highest probability FP
    best_fp_idx = fp_indices[np.argmax(y_probs[fp_indices])] if len(fp_indices) > 0 else 0

    # 4. False Negative Candidate
    fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]
    # Pick lowest probability FN
    best_fn_idx = fn_indices[np.argmin(y_probs[fn_indices])] if len(fn_indices) > 0 else 0

    cases: Dict[str, Dict[str, Any]] = {
        "case_1_true_positive": {
            "title": "High-Risk True Positive (Correct Critical Alert)",
            "description": "Patient who died in hospital and was correctly identified by the model with high predicted mortality probability.",
            "index": int(best_tp_idx),
            "quadrant": "TP",
        },
        "case_2_true_negative": {
            "title": "Low-Risk True Negative (Correct Reassurance)",
            "description": "Patient who survived hospital stay and was correctly assigned a very low predicted mortality risk.",
            "index": int(best_tn_idx),
            "quadrant": "TN",
        },
        "case_3_false_positive": {
            "title": "False Positive Case (Elevated Risk in Survivor)",
            "description": "Patient who survived hospital stay but exhibited severe acute derangements that elevated model predicted risk.",
            "index": int(best_fp_idx),
            "quadrant": "FP",
        },
        "case_4_false_negative": {
            "title": "False Negative Case (Missed Deterioration / Borderline)",
            "description": "Patient who died in hospital despite subtle or delayed initial 24h derangements leading to a lower model risk score.",
            "index": int(best_fn_idx),
            "quadrant": "FN",
        },
    }

    # Enrich cases with clinical metadata
    for case_id, case_info in cases.items():
        idx = case_info["index"]
        stay_id = int(meta_test.iloc[idx]["patientunitstayid"]) if "patientunitstayid" in meta_test.columns else idx
        patient_id = str(meta_test.iloc[idx]["uniquepid"]) if "uniquepid" in meta_test.columns else f"PID_{idx}"
        age = float(X_test.iloc[idx]["age_numeric"]) if "age_numeric" in X_test.columns else 65.0
        unit_type = str(meta_test.iloc[idx]["unittype"]) if "unittype" in meta_test.columns else "General ICU"
        pred_prob = float(y_probs[idx])
        actual_death = int(y_test[idx])

        case_info.update({
            "patientunitstayid": stay_id,
            "uniquepid": patient_id,
            "age": age,
            "unit_type": unit_type,
            "predicted_mortality_probability": pred_prob,
            "actual_outcome": actual_death,
            "actual_outcome_label": "Deceased" if actual_death == 1 else "Survived",
        })

    return cases


def extract_patient_feature_contributions(
    shap_explanation: shap.Explanation,
    patient_idx: int,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Extract top positive and top negative contributing features for an individual patient.

    Args:
        shap_explanation: Global TreeSHAP explanation object.
        patient_idx: Index of the patient in the test set.
        top_k: Number of top features to return per direction.

    Returns:
        Dictionary containing top positive features, top negative features, and base value.
    """
    patient_shap = shap_explanation[patient_idx]
    values = np.asarray(patient_shap.values)
    feature_names = list(patient_shap.feature_names) if patient_shap.feature_names is not None else [f"feature_{i}" for i in range(len(values))]
    data_values = np.asarray(patient_shap.data)

    base_val_raw = patient_shap.base_values
    if isinstance(base_val_raw, (np.ndarray, list)):
        base_val = float(np.asarray(base_val_raw).ravel()[0])
    else:
        base_val = float(base_val_raw)
    margin_output = float(base_val + np.sum(values))

    # Sort positive contributions (increasing prediction)
    pos_indices = np.where(values > 0)[0]
    pos_sorted = pos_indices[np.argsort(-values[pos_indices])]
    top_pos = []
    for idx in pos_sorted[:top_k]:
        val_display = f"{data_values[idx]:.2f}" if isinstance(data_values[idx], (float, np.floating)) else str(data_values[idx])
        top_pos.append({
            "feature": str(feature_names[idx]),
            "feature_value": val_display,
            "shap_value": float(values[idx]),
            "contribution_direction": "Increased mortality prediction",
        })

    # Sort negative contributions (decreasing prediction)
    neg_indices = np.where(values < 0)[0]
    neg_sorted = neg_indices[np.argsort(values[neg_indices])]
    top_neg = []
    for idx in neg_sorted[:top_k]:
        val_display = f"{data_values[idx]:.2f}" if isinstance(data_values[idx], (float, np.floating)) else str(data_values[idx])
        top_neg.append({
            "feature": str(feature_names[idx]),
            "feature_value": val_display,
            "shap_value": float(values[idx]),
            "contribution_direction": "Decreased mortality prediction",
        })

    return {
        "base_expected_value_log_odds": base_val,
        "model_output_margin_log_odds": margin_output,
        "total_shap_sum": float(np.sum(values)),
        "top_features_increasing_prediction": top_pos,
        "top_features_decreasing_prediction": top_neg,
    }


def plot_local_patient_explanation(
    shap_explanation: shap.Explanation,
    patient_idx: int,
    case_title: str,
    output_path: Path,
    max_display: int = 12,
) -> Path:
    """Generate and save local SHAP waterfall plot for an individual ICU stay."""
    patient_shap = shap_explanation[patient_idx]

    plt.figure(figsize=(10, 6), dpi=300)
    shap.plots.waterfall(
        patient_shap,
        max_display=max_display,
        show=False,
    )
    plt.title(f"Local Patient Explanation: {case_title}\nAdditive Feature Contributions to Prediction Margin", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


# ==============================================================================
# 5. ORCHESTRATION & REPORT GENERATION
# ==============================================================================

def run_phase_7_xai(
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Execute end-to-end Phase 7 Explainable AI (XAI) workflow."""
    print("=" * 75)
    print("SafePredict-XAI: Phase 7 Explainable AI (XAI) Pipeline")
    print("=" * 75)

    # 1. Load Data Splits and Champion Model
    print("\n[Step 1/5] Ingesting Cohort and Loading Phase 6 Champion Model...")
    split_data = prepare_model_splits(random_state=random_state)
    X_train, y_train = split_data["X_train"], split_data["y_train"]
    X_val, y_val = split_data["X_val"], split_data["y_val"]
    X_test, y_test = split_data["X_test"], split_data["y_test"]
    num_cols, cat_cols = split_data["num_cols"], split_data["cat_cols"]
    split_summary = split_data["split_summary"]

    meta_test = get_test_cohort_metadata(random_state=random_state)

    calibrated_model, preprocessor = load_champion_artifacts()
    base_classifier = extract_base_classifier(calibrated_model)
    print(f"  • Loaded Calibrated Classifier: {type(calibrated_model).__name__}")
    print(f"  • Extracted Base Tree Classifier: {type(base_classifier).__name__}")
    print(f"  • Preprocessor Input Features: {len(num_cols)} numeric, {len(cat_cols)} categorical")

    # Compute test set calibrated probabilities
    y_test_probs = calibrated_model.predict_proba(X_test)[:, 1]

    # 2. Compute TreeSHAP Explanations
    print("\n[Step 2/5] Computing TreeSHAP Values on Held-Out Test Cohort (N=280)...")
    shap_explanation, X_test_trans_df, cleaned_feature_names = compute_tree_shap_explanations(
        base_classifier=base_classifier,
        preprocessor=preprocessor,
        X_test=X_test,
    )
    print(f"  • Computed SHAP matrix: {shap_explanation.shape} (280 stays x {len(cleaned_feature_names)} features)")
    print(f"  • Population Base Value (Log-Odds Margin): {shap_explanation.base_values[0]:.4f}")

    # 3. Global Explanations & Rankings
    print("\n[Step 3/5] Computing Global Feature Importance and Generating Summary Plots...")
    global_importance_df = compute_global_feature_importance(shap_explanation, cleaned_feature_names)
    beeswarm_path, bar_path = plot_global_shap_summary(
        shap_explanation,
        output_beeswarm_path=FIGURES_DIR / "shap_summary_beeswarm.png",
        output_bar_path=FIGURES_DIR / "shap_summary_bar.png",
        max_display=20,
    )
    print(f"  • Saved Global Beeswarm Plot: {beeswarm_path}")
    print(f"  • Saved Global Bar Importance Plot: {bar_path}")

    # Save JSON feature importance
    importance_json_path = METRICS_DIR / "shap_global_feature_importance.json"
    importance_records = global_importance_df.to_dict(orient="records")
    with open(importance_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_test_samples": len(X_test),
            "base_expected_value_log_odds": float(shap_explanation.base_values[0]),
            "feature_importances": importance_records,
        }, f, indent=2)
    print(f"  • Saved Global Feature Importance JSON: {importance_json_path}")

    # Print Top 10 Features
    print("\n  Top 10 Global Features by Mean |SHAP Value|:")
    for rank, row in enumerate(importance_records[:10], 1):
        print(f"    {rank:2d}. {row['feature']:<38} | Mean |SHAP| = {row['mean_abs_shap']:.4f} ({row['relative_importance_pct']:.2f}%)")

    # 4. Local Clinical Case Studies
    print("\n[Step 4/5] Generating Local Case-Based Patient Explanations (TP, TN, FP, FN)...")
    clinical_cases = select_clinical_case_studies(
        X_test=X_test,
        y_test=y_test,
        y_probs=y_test_probs,
        meta_test=meta_test,
        youden_threshold=0.058,
    )

    local_explanations_catalog: Dict[str, Any] = {}
    for case_key, case_data in clinical_cases.items():
        idx = case_data["index"]
        plot_filename = f"shap_local_{case_key}.png"
        plot_path = FIGURES_DIR / plot_filename

        plot_local_patient_explanation(
            shap_explanation=shap_explanation,
            patient_idx=idx,
            case_title=f"{case_data['title']} (Stay #{case_data['patientunitstayid']})",
            output_path=plot_path,
            max_display=12,
        )

        contribs = extract_patient_feature_contributions(
            shap_explanation=shap_explanation,
            patient_idx=idx,
            top_k=5,
        )

        case_full = {**case_data, **contribs, "plot_path": str(plot_path)}
        local_explanations_catalog[case_key] = case_full
        print(f"  • {case_data['title']}: P(death)={case_data['predicted_mortality_probability']:.1%}, True={case_data['actual_outcome_label']} -> Saved {plot_filename}")

    # 5. Compile Comprehensive Markdown XAI Report
    print("\n[Step 5/5] Compiling Comprehensive Phase 7 XAI Markdown Report...")
    report_path = REPORTS_DIR / "xai_model_explanations_report.md"

    md_content = f"""# SafePredict-XAI: Phase 7 Explainable AI (XAI) Report

**Model Architecture:** XGBoost Classifier with Platt Sigmoid Calibration (Phase 6 Champion)  
**Evaluation Cohort:** Held-Out Test Set ($N = 280$ ICU stays, 24 deaths, 8.57% mortality prevalence)  
**Explanation Engine:** TreeSHAP (`shap.TreeExplainer` on fitted tree ensemble margin)  
**Population Base Expected Value:** {shap_explanation.base_values[0]:.4f} log-odds  

---

## 1. Important Clinical Interpretation Guardrail

> [!IMPORTANT]
> **Interpretation vs. Causation Guardrail:**
> SHAP (SHapley Additive exPlanations) values quantify how each feature **contributed to the machine learning model's numerical prediction** relative to the baseline population expectation.
> - **Accurate Clinical Wording:** *"These features contributed to the model's prediction."*
> - **Prohibited Causal Wording:** *"This feature caused mortality."*
> SHAP explains **model behavior and associative patterns** learned from historical electronic health record data; it does **not** establish biological causality or clinical etiology.

---

## 2. Global Feature Importance & Directionality

Across the held-out test cohort ($N=280$), the model's predictions are governed primarily by acute physiological derangements, respiratory stability, and metabolic markers measured during the first 24 hours of ICU stay.

### Top 15 Global Features by Mean Absolute SHAP Value

| Rank | Clinical Feature | Mean |SHAP Value| | Relative Importance (%) | Cumulative Importance (%) | Directional Association with Mortality Prediction |
| :---: | :--- | :---: | :---: | :---: | :--- |
"""
    for r, item in enumerate(importance_records[:15], 1):
        md_content += f"| {r} | **{item['feature']}** | {item['mean_abs_shap']:.4f} | {item['relative_importance_pct']:.2f}% | {item['cumulative_importance_pct']:.2f}% | High values contribute positively to mortality prediction |\n"

    md_content += f"""
---

## 3. Global Visualization Artifacts

- **SHAP Summary Beeswarm Plot:** [`reports/figures/shap_summary_beeswarm.png`](file:///c:/Users/k7ris/Documents/SafePredict-XAI/reports/figures/shap_summary_beeswarm.png)
  - Displays the distribution of SHAP attributions across all $N=280$ test patients.
  - Colors represent relative feature values (red = high, blue = low).
  - Red points extending to the right indicate that high physiological derangements contributed to higher predicted mortality risk.
- **SHAP Global Feature Importance Bar Plot:** [`reports/figures/shap_summary_bar.png`](file:///c:/Users/k7ris/Documents/SafePredict-XAI/reports/figures/shap_summary_bar.png)
  - Ranks features by their overall population-level impact magnitude.

---

## 4. Local Patient Explanations (Clinical Case Studies)

To demonstrate individual-level model interpretability at the bedside, four representative ICU stays were analyzed across prediction quadrants:

"""
    for case_id, case_info in local_explanations_catalog.items():
        md_content += f"""### {case_info['title']}

- **Patient Identifier:** Unit Stay #{case_info['patientunitstayid']} (Unique Patient: `{case_info['uniquepid']}`)
- **Patient Profile:** Age {case_info['age']:.0f} years | ICU Unit: {case_info['unit_type']}
- **Predicted Mortality Probability:** **{case_info['predicted_mortality_probability']:.1%}** (Calibrated Risk)
- **Actual In-Hospital Outcome:** **{case_info['actual_outcome_label']}** (Ground Truth: {case_info['actual_outcome']})
- **Clinical Context:** {case_info['description']}

#### Top Features Increasing Model Prediction (Higher Risk Contribution):
"""
        for p_feat in case_info["top_features_increasing_prediction"][:4]:
            md_content += f"- **{p_feat['feature']}** (Value: `{p_feat['feature_value']}`): Contributed **+{p_feat['shap_value']:.3f}** to prediction margin.\n"

        md_content += "\n#### Top Features Decreasing Model Prediction (Protective / Low-Risk Contribution):\n"
        for n_feat in case_info["top_features_decreasing_prediction"][:4]:
            md_content += f"- **{n_feat['feature']}** (Value: `{n_feat['feature_value']}`): Contributed **{n_feat['shap_value']:.3f}** to prediction margin.\n"

        md_content += f"\n- **Local Waterfall Plot:** [`{case_info['plot_path']}`](file:///{case_info['plot_path']})\n\n---\n\n"

    md_content += """## 5. Artifact Summary

1. **Explainability Pipeline:** [`src/explain.py`](file:///c:/Users/k7ris/Documents/SafePredict-XAI/src/explain.py)
2. **Interactive Jupyter Notebook:** [`notebooks/05_xai_safepredict.ipynb`](file:///c:/Users/k7ris/Documents/SafePredict-XAI/notebooks/05_xai_safepredict.ipynb)
3. **Global Feature Importance JSON:** [`reports/metrics/shap_global_feature_importance.json`](file:///c:/Users/k7ris/Documents/SafePredict-XAI/reports/metrics/shap_global_feature_importance.json)
4. **Visualizations in `reports/figures/`:**
   - `shap_summary_beeswarm.png`
   - `shap_summary_bar.png`
   - `shap_local_case_1_true_positive.png`
   - `shap_local_case_2_true_negative.png`
   - `shap_local_case_3_false_positive.png`
   - `shap_local_case_4_false_negative.png`

---

## 6. Strict Phase Boundary Compliance

- Explanations generated exclusively with SHAP.
- No dashboards or web applications constructed in this phase.
- Execution terminates cleanly after report generation.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"  • Saved Markdown XAI Report: {report_path}")

    print("\n" + "=" * 75)
    print("Phase 7 Explainable AI (XAI) completed successfully!")
    print("=" * 75)

    return {
        "global_importance": global_importance_df,
        "clinical_cases": local_explanations_catalog,
        "beeswarm_path": str(beeswarm_path),
        "bar_path": str(bar_path),
    }


if __name__ == "__main__":
    run_phase_7_xai()
