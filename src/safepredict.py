"""SafePredict Reliability Layer — Phase 8 for SafePredict-XAI.

Implements an ACCEPT / ABSTAIN reliability framework that combines:

1. Bootstrap Uncertainty Estimation
   - 30 bootstrap resamples of Logistic Regression on the preprocessed feature matrix.
   - Measures prediction *variability*, NOT predict_proba directly.
   - Per-patient output: mean prediction and std deviation across bootstraps.

2. Data Quality Scoring
   - Composite score [0, 1] built from columns already in model_data.parquet:
     * Lab breadth (available_lab_count / 15)      — weight 40%
     * Vital breadth (available_vital_count / 7)    — weight 30%
     * Feature completeness (1 - missing / total)   — weight 20%
     * Encounter quality flag bonus                 — weight 10%

3. Multi-Strategy SafePredict Decision Engine
   - model_only:             always ACCEPT
   - uncertainty_abstain:    ABSTAIN when std >= thresh_u
   - dq_abstain:             ABSTAIN when dq_score < thresh_dq
   - safepredict_combined:   ABSTAIN when std >= thresh_u OR dq_score < thresh_dq

4. Threshold Selection (validation set ONLY — thresholds are then frozen)

5. Evaluation: coverage, abstention rate, error rate, AUROC, PR-AUC, Brier

6. Risk-vs-Coverage sweep for plotting

7. Individual patient reliability reports

IMPORTANT:
  ABSTAIN means: "The reliability criteria for accepting this model prediction were not met."
  It does NOT mean the patient is safe or high-risk.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

# ---------------------------------------------------------------------------
# Project-root path resolution
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import (
    prepare_model_splits,
    build_preprocessor,
    RANDOM_STATE,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    PROCESSED_DATA_DIR,
    IDENTIFIER_COLS,
    TARGET_COL,
)

VALIDATED_DATA_DIR = PROJECT_ROOT / "data" / "validated"
QUALITY_FLAGS_PATH = VALIDATED_DATA_DIR / "encounter_quality_flags.parquet"

# Number of bootstrap resamples
N_BOOTSTRAPS: int = 30
# Minimum coverage fraction required when selecting thresholds on val set
MIN_COVERAGE: float = 0.60


# ==============================================================================
# 1. BOOTSTRAP UNCERTAINTY ESTIMATION
# ==============================================================================

def compute_bootstrap_uncertainty(
    X_preprocessed: np.ndarray,
    y_train_preprocessed: np.ndarray,
    X_target_preprocessed: np.ndarray,
    n_bootstraps: int = N_BOOTSTRAPS,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate prediction uncertainty via bootstrap resampling of Logistic Regression.

    This method deliberately does NOT use predict_proba() of the champion model as
    uncertainty. Instead it quantifies *how much the predicted probability would
    change* if the model were trained on slightly different data.

    A lightweight Logistic Regression surrogate is trained on the already-preprocessed
    feature matrix (no data leakage). 30 bootstraps are practical at this dataset size.

    Args:
        X_preprocessed:         Preprocessed training feature matrix (n_train, n_features).
        y_train_preprocessed:   Training labels array (n_train,).
        X_target_preprocessed:  Preprocessed target set — typically val or test.
        n_bootstraps:           Number of bootstrap resamples.
        random_state:           Base random seed (each resample uses seed + i).

    Returns:
        Tuple of:
          - mean_preds  : np.ndarray (n_target,) — mean probability across bootstraps.
          - std_preds   : np.ndarray (n_target,) — std dev across bootstraps (variability).
    """
    n_train = X_preprocessed.shape[0]
    n_target = X_target_preprocessed.shape[0]
    all_preds = np.zeros((n_bootstraps, n_target), dtype=float)

    for i in range(n_bootstraps):
        rng = np.random.default_rng(random_state + i)
        boot_idx = rng.choice(n_train, size=n_train, replace=True)
        X_boot = X_preprocessed[boot_idx]
        y_boot = y_train_preprocessed[boot_idx]

        # Guard: skip degenerate bootstraps with only one class
        if len(np.unique(y_boot)) < 2:
            all_preds[i] = all_preds[max(0, i - 1)]
            continue

        clf = LogisticRegression(
            C=0.1,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=500,
            random_state=random_state + i,
        )
        clf.fit(X_boot, y_boot)
        all_preds[i] = clf.predict_proba(X_target_preprocessed)[:, 1]

    mean_preds = np.mean(all_preds, axis=0)
    std_preds = np.std(all_preds, axis=0)
    return mean_preds, std_preds


# ==============================================================================
# 2. DATA QUALITY SCORING
# ==============================================================================

def compute_data_quality_scores(
    model_df: pl.DataFrame,
    quality_flags_df: Optional[pl.DataFrame] = None,
    total_clinical_features: Optional[int] = None,
    max_labs: int = 15,
    max_vitals: int = 7,
) -> pd.Series:
    """Compute a composite data quality score [0, 1] per ICU stay.

    Weights:
      40% — Lab breadth   : available_lab_count  / max_labs
      30% — Vital breadth : available_vital_count / max_vitals
      20% — Completeness  : 1 - (missing_feature_count / total_clinical_features)
      10% — Quality flags : derived from encounter_quality_flags.parquet

    Args:
        model_df:                Polars DataFrame (model_data.parquet) with quality columns.
        quality_flags_df:        Optional Polars DataFrame (encounter_quality_flags.parquet).
        total_clinical_features: Denominator for missing-feature rate. Auto-detected if None.
        max_labs:                Maximum number of lab types (default 15).
        max_vitals:              Maximum number of vital types (default 7).

    Returns:
        pd.Series indexed by patientunitstayid containing scores in [0, 1].
    """
    pdf = model_df.to_pandas()
    stay_ids = pdf["patientunitstayid"].values

    # Component 1: Lab breadth
    if "available_lab_count" in pdf.columns:
        lab_score = np.clip(pdf["available_lab_count"].to_numpy(dtype=float) / max_labs, 0.0, 1.0)
    else:
        lab_score = np.zeros(len(pdf), dtype=float)

    # Component 2: Vital breadth
    if "available_vital_count" in pdf.columns:
        vital_score = np.clip(pdf["available_vital_count"].to_numpy(dtype=float) / max_vitals, 0.0, 1.0)
    else:
        vital_score = np.zeros(len(pdf), dtype=float)

    # Component 3: Completeness (1 - missing fraction)
    if "missing_feature_count" in pdf.columns:
        if total_clinical_features is None:
            exclude = {"patientunitstayid", "patienthealthsystemstayid", "hospital_mortality"}
            total_clinical_features = sum(
                1 for c in pdf.columns
                if c not in exclude
                and not c.endswith("_count")
                and c not in ("available_lab_count", "available_vital_count", "missing_feature_count")
            )
            total_clinical_features = max(total_clinical_features, 1)
        completeness_score = np.clip(
            1.0 - (pdf["missing_feature_count"].to_numpy(dtype=float) / total_clinical_features),
            0.0, 1.0
        )
    else:
        completeness_score = np.ones(len(pdf), dtype=float)

    # Component 4: Quality flags bonus
    flag_score = np.full(len(pdf), 0.5)  # default neutral
    if quality_flags_df is not None and len(quality_flags_df) > 0:
        flag_pdf = quality_flags_df.to_pandas()

        # Detect flag column
        flag_col = None
        for candidate in ["validation_flag", "quality_flag", "flag", "status"]:
            if candidate in flag_pdf.columns:
                flag_col = candidate
                break
        if flag_col is None:
            id_cols = {"patientunitstayid", "patienthealthsystemstayid"}
            non_id = [c for c in flag_pdf.columns if c not in id_cols]
            if non_id:
                flag_col = non_id[0]

        if flag_col is not None and "patientunitstayid" in flag_pdf.columns:
            clean_keywords = {"no_warnings", "pass", "ok", "clean", "valid", "no_warning"}
            flag_pdf["_is_clean"] = flag_pdf[flag_col].astype(str).str.lower().apply(
                lambda x: 1 if any(kw in x for kw in clean_keywords) else 0
            )
            stay_clean = (
                flag_pdf.groupby("patientunitstayid")["_is_clean"]
                .mean()
                .reset_index()
                .rename(columns={"_is_clean": "clean_frac"})
            )
            merged = pd.merge(
                pd.DataFrame({"patientunitstayid": stay_ids}),
                stay_clean,
                on="patientunitstayid",
                how="left",
            )
            flag_score = merged["clean_frac"].fillna(0.5).to_numpy(dtype=float)

    # Composite score
    dq_score = (
        0.40 * lab_score
        + 0.30 * vital_score
        + 0.20 * completeness_score
        + 0.10 * flag_score
    )
    dq_score = np.clip(dq_score, 0.0, 1.0)

    return pd.Series(dq_score, index=stay_ids, name="data_quality_score")


# ==============================================================================
# 3. SAFEPREDICT CONFIGURATION
# ==============================================================================

@dataclass
class SafePredictConfig:
    """Frozen thresholds selected on the validation set.

    Attributes:
        uncertainty_threshold:  std_dev above which the model ABSTAINS.
        dq_threshold:           data_quality_score below which the model ABSTAINS.
        min_coverage:           minimum fraction of val set that must be ACCEPTED.
        val_coverage:           actual coverage achieved on validation set.
        val_auroc_accepted:     AUROC among accepted val patients at these thresholds.
        selection_criterion:    description of the objective used.
        sweep_details:          list of all evaluated threshold combinations.
    """
    uncertainty_threshold: float = 0.10
    dq_threshold: float = 0.40
    min_coverage: float = MIN_COVERAGE
    val_coverage: float = 0.0
    val_auroc_accepted: float = 0.0
    selection_criterion: str = "max val_auroc_accepted subject to val_coverage >= 0.60"
    sweep_details: List[Dict[str, Any]] = field(default_factory=list)


# ==============================================================================
# 4. STRATEGY APPLICATION
# ==============================================================================

def apply_strategies(
    uncertainty_std: np.ndarray,
    dq_scores: np.ndarray,
    config: SafePredictConfig,
) -> Dict[str, np.ndarray]:
    """Generate ACCEPT (True) / ABSTAIN (False) masks for all four strategies.

    Args:
        uncertainty_std: Per-patient prediction standard deviation across bootstraps.
        dq_scores:       Per-patient data quality score in [0, 1].
        config:          Frozen SafePredictConfig with selected thresholds.

    Returns:
        Dictionary mapping strategy name to boolean mask (True = ACCEPT).
    """
    return {
        "model_only": np.ones(len(uncertainty_std), dtype=bool),
        "uncertainty_abstain": uncertainty_std < config.uncertainty_threshold,
        "dq_abstain": dq_scores >= config.dq_threshold,
        "safepredict_combined": (
            (uncertainty_std < config.uncertainty_threshold)
            & (dq_scores >= config.dq_threshold)
        ),
    }


# ==============================================================================
# 5. THRESHOLD SELECTION ON VALIDATION
# ==============================================================================

def select_thresholds_on_validation(
    val_uncertainty_std: np.ndarray,
    val_dq_scores: np.ndarray,
    val_y: np.ndarray,
    val_probs: np.ndarray,
    min_coverage: float = MIN_COVERAGE,
) -> SafePredictConfig:
    """Grid-search thresholds using validation set ONLY.

    Objective: maximise val_auroc_accepted, subject to val_coverage >= min_coverage.
    All thresholds are then FROZEN before the test set is ever examined.

    Args:
        val_uncertainty_std: Bootstrap std dev array for val patients.
        val_dq_scores:       DQ score array for val patients.
        val_y:               Validation ground truth labels.
        val_probs:           Calibrated model probabilities for val patients.
        min_coverage:        Minimum required fraction of val patients ACCEPTED.

    Returns:
        SafePredictConfig with optimal thresholds and sweep details.
    """
    u_candidates = np.unique(np.percentile(val_uncertainty_std, [50, 60, 70, 75, 80, 85, 90]))
    dq_candidates = np.unique(np.percentile(val_dq_scores, [10, 15, 20, 25, 30, 40, 50]))

    best_config = SafePredictConfig(min_coverage=min_coverage)
    best_auroc = -1.0
    sweep_details: List[Dict[str, Any]] = []

    for thresh_u in u_candidates:
        for thresh_dq in dq_candidates:
            mask = (val_uncertainty_std < thresh_u) & (val_dq_scores >= thresh_dq)
            coverage = float(np.mean(mask))
            n_accepted = int(np.sum(mask))

            if n_accepted < 10 or coverage < min_coverage:
                sweep_details.append({
                    "thresh_u": float(thresh_u),
                    "thresh_dq": float(thresh_dq),
                    "coverage": coverage,
                    "auroc_accepted": None,
                    "eligible": False,
                })
                continue

            y_acc = val_y[mask]
            p_acc = val_probs[mask]

            if len(np.unique(y_acc)) < 2:
                sweep_details.append({
                    "thresh_u": float(thresh_u),
                    "thresh_dq": float(thresh_dq),
                    "coverage": coverage,
                    "auroc_accepted": None,
                    "eligible": False,
                })
                continue

            auroc_acc = float(roc_auc_score(y_acc, p_acc))
            sweep_details.append({
                "thresh_u": float(thresh_u),
                "thresh_dq": float(thresh_dq),
                "coverage": coverage,
                "auroc_accepted": auroc_acc,
                "eligible": True,
            })

            if auroc_acc > best_auroc:
                best_auroc = auroc_acc
                best_config = SafePredictConfig(
                    uncertainty_threshold=float(thresh_u),
                    dq_threshold=float(thresh_dq),
                    min_coverage=min_coverage,
                    val_coverage=coverage,
                    val_auroc_accepted=auroc_acc,
                    selection_criterion=(
                        f"max val_auroc_accepted={auroc_acc:.4f} "
                        f"subject to val_coverage >= {min_coverage:.0%}"
                    ),
                    sweep_details=sweep_details,
                )

    # Fallback: widest thresholds if no eligible combination found
    if best_config.val_auroc_accepted == 0.0:
        best_config.uncertainty_threshold = float(np.max(u_candidates))
        best_config.dq_threshold = float(np.min(dq_candidates))
        mask_fb = (
            (val_uncertainty_std < best_config.uncertainty_threshold)
            & (val_dq_scores >= best_config.dq_threshold)
        )
        best_config.val_coverage = float(np.mean(mask_fb))
        best_config.selection_criterion = "fallback: widest thresholds (min_coverage not achievable)"
        best_config.sweep_details = sweep_details

    return best_config


# ==============================================================================
# 6. EVALUATION ENGINE
# ==============================================================================

def evaluate_strategy(
    accept_mask: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    default_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute reliability metrics for a single strategy.

    Returns:
        Dict with: coverage, abstention_rate, n_accepted, error_rate_accepted,
        auroc_accepted, pr_auc_accepted, brier_accepted.
    """
    n_total = len(y_true)
    n_accepted = int(np.sum(accept_mask))
    coverage = n_accepted / n_total

    result: Dict[str, Any] = {
        "n_total": n_total,
        "n_accepted": n_accepted,
        "coverage": coverage,
        "abstention_rate": 1.0 - coverage,
    }

    if n_accepted == 0:
        result.update({"error_rate_accepted": None, "auroc_accepted": None,
                        "pr_auc_accepted": None, "brier_accepted": None})
        return result

    y_acc = y_true[accept_mask]
    p_acc = y_prob[accept_mask]
    y_pred_acc = (p_acc >= default_threshold).astype(int)
    result["error_rate_accepted"] = float(np.mean(y_pred_acc != y_acc))
    result["brier_accepted"] = float(brier_score_loss(y_acc, p_acc))

    if len(np.unique(y_acc)) >= 2:
        result["auroc_accepted"] = float(roc_auc_score(y_acc, p_acc))
        result["pr_auc_accepted"] = float(average_precision_score(y_acc, p_acc))
    else:
        result["auroc_accepted"] = None
        result["pr_auc_accepted"] = None

    return result


def evaluate_all_strategies(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty_std: np.ndarray,
    dq_scores: np.ndarray,
    config: SafePredictConfig,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate all four SafePredict strategies."""
    masks = apply_strategies(uncertainty_std, dq_scores, config)
    return {name: evaluate_strategy(mask, y_true, y_prob) for name, mask in masks.items()}


# ==============================================================================
# 7. RISK vs COVERAGE SWEEP
# ==============================================================================

def compute_risk_vs_coverage(
    uncertainty_std: np.ndarray,
    dq_scores: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_points: int = 40,
) -> pd.DataFrame:
    """Sweep uncertainty threshold to generate Risk (AUROC) vs Coverage data.

    DQ threshold is held fixed at the median DQ score during the sweep.
    """
    thresh_u_range = np.linspace(
        float(np.min(uncertainty_std)) + 1e-6,
        float(np.max(uncertainty_std)) + 1e-6,
        n_points,
    )
    dq_fixed = float(np.median(dq_scores))
    records = []

    for t in thresh_u_range:
        mask = (uncertainty_std < t) & (dq_scores >= dq_fixed)
        n_acc = int(np.sum(mask))
        cov = float(np.mean(mask))

        if n_acc < 5:
            records.append({"thresh_u": t, "coverage": cov,
                             "auroc_accepted": np.nan, "pr_auc_accepted": np.nan,
                             "brier_accepted": np.nan, "n_accepted": n_acc})
            continue

        y_acc = y_true[mask]
        p_acc = y_prob[mask]
        brier = float(brier_score_loss(y_acc, p_acc))

        if len(np.unique(y_acc)) >= 2:
            auroc = float(roc_auc_score(y_acc, p_acc))
            pr_auc = float(average_precision_score(y_acc, p_acc))
        else:
            auroc, pr_auc = np.nan, np.nan

        records.append({"thresh_u": t, "coverage": cov, "auroc_accepted": auroc,
                         "pr_auc_accepted": pr_auc, "brier_accepted": brier, "n_accepted": n_acc})

    return pd.DataFrame(records)


# ==============================================================================
# 8. RISK vs COVERAGE PLOT
# ==============================================================================

def plot_risk_vs_coverage(
    sweep_df: pd.DataFrame,
    baseline_auroc: float,
    output_path: Path,
    strategy_coverages: Optional[Dict[str, float]] = None,
) -> Path:
    """Generate and save a two-panel Risk (AUROC) vs Coverage figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid = sweep_df.dropna(subset=["auroc_accepted"])
    valid2 = sweep_df.dropna(subset=["brier_accepted"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
    fig.patch.set_facecolor("#f9f9f9")

    # Left: AUROC vs Coverage
    ax = axes[0]
    ax.set_facecolor("#f9f9f9")
    ax.axhline(baseline_auroc, color="#888888", linestyle="--", lw=1.6,
               label=f"Baseline (all patients) AUROC = {baseline_auroc:.3f}")
    if len(valid) > 0:
        ax.plot(valid["coverage"], valid["auroc_accepted"],
                color="#2b5c8f", lw=2.5, marker="o", markersize=4,
                label="SafePredict combined sweep")
        ax.fill_between(valid["coverage"].values, baseline_auroc,
                        valid["auroc_accepted"].values,
                        where=valid["auroc_accepted"].to_numpy() >= baseline_auroc,
                        alpha=0.12, color="#2b5c8f", label="AUROC gain region")

    s_colors = {"uncertainty_abstain": "#e67e22", "dq_abstain": "#27ae60",
                 "safepredict_combined": "#8e44ad"}
    s_labels = {"uncertainty_abstain": "Uncertainty", "dq_abstain": "DQ",
                  "safepredict_combined": "Combined"}
    if strategy_coverages:
        for sname, scov in strategy_coverages.items():
            if sname in s_colors:
                ax.axvline(scov, color=s_colors[sname], linestyle=":", lw=1.6, alpha=0.85,
                           label=f"{s_labels[sname]} cov={scov:.0%}")

    ax.set_xlabel("Coverage (fraction ACCEPTED)", fontsize=11, fontweight="bold")
    ax.set_ylabel("AUROC among ACCEPTED patients", fontsize=11, fontweight="bold")
    ax.set_title("SafePredict — Risk vs Coverage\n(Uncertainty threshold sweep, DQ fixed at median)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlim(0.0, 1.0)
    lo = max(0.5, baseline_auroc - 0.15)
    hi = min(1.0, valid["auroc_accepted"].max() + 0.05) if len(valid) > 0 else baseline_auroc + 0.1
    ax.set_ylim(lo, hi)
    ax.legend(fontsize=8, frameon=True, facecolor="white", loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.5)

    # Right: Brier vs Coverage (inverted — lower is better, so up = better)
    ax2 = axes[1]
    ax2.set_facecolor("#f9f9f9")
    if len(valid2) > 0:
        ax2.plot(valid2["coverage"], valid2["brier_accepted"],
                 color="#c0392b", lw=2.5, marker="s", markersize=4,
                 label="Brier score (lower = better calibration)")
    ax2.set_xlabel("Coverage (fraction ACCEPTED)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Brier Score among ACCEPTED patients", fontsize=11, fontweight="bold")
    ax2.set_title("Calibration Quality vs Coverage\n(Brier score; y-axis inverted: up = better)",
                  fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlim(0.0, 1.0)
    ax2.legend(fontsize=8.5, frameon=True, facecolor="white")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.invert_yaxis()

    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#f9f9f9")
    plt.close()
    return output_path


# ==============================================================================
# 9. INDIVIDUAL PATIENT RELIABILITY REPORT
# ==============================================================================

def _band_label(value: float, lo: float, hi: float,
                lo_label: str = "LOW", hi_label: str = "HIGH") -> str:
    if value <= lo:
        return lo_label
    elif value >= hi:
        return hi_label
    return "MODERATE"


def generate_patient_report(
    patient_idx: int,
    y_prob_calibrated: float,
    uncertainty_mean: float,
    uncertainty_std: float,
    dq_score: float,
    dq_components: Dict[str, Any],
    accept_combined: bool,
    shap_contributions: Optional[Dict[str, Any]] = None,
    case_metadata: Optional[Dict[str, Any]] = None,
    uncertainty_thresh: float = 0.10,
    dq_thresh: float = 0.40,
) -> str:
    """Format a comprehensive SafePredict reliability report for one patient."""
    decision = "ACCEPT" if accept_combined else "ABSTAIN"
    decision_icon = "✅" if accept_combined else "⚠️"

    reasons = []
    if not accept_combined:
        if uncertainty_std >= uncertainty_thresh:
            reasons.append(
                f"High prediction variability (std={uncertainty_std:.3f} >= {uncertainty_thresh:.3f})"
            )
        if dq_score < dq_thresh:
            reasons.append(
                f"Low data quality score ({dq_score:.2f} < {dq_thresh:.2f})"
            )

    abstain_note = ""
    if not accept_combined:
        abstain_note = (
            "\n  NOTE: ABSTAIN means the reliability criteria were not met.\n"
            "        It does NOT mean the patient is safe or high-risk.\n"
        )

    u_band = _band_label(uncertainty_std, lo=uncertainty_thresh * 0.55, hi=uncertainty_thresh)
    dq_band = _band_label(dq_score, lo=dq_thresh, hi=0.75)

    avail_labs = dq_components.get("available_lab_count", "N/A")
    avail_vits = dq_components.get("available_vital_count", "N/A")
    miss_feats = dq_components.get("missing_feature_count", "N/A")
    max_labs = dq_components.get("max_labs", 15)
    max_vitals = dq_components.get("max_vitals", 7)

    meta_str = ""
    if case_metadata:
        age = case_metadata.get("age", "N/A")
        unit = case_metadata.get("unit_type", "N/A")
        uid = case_metadata.get("uniquepid", "N/A")
        stayid = case_metadata.get("patientunitstayid", "N/A")
        actual_label = case_metadata.get("actual_outcome_label", "N/A")
        age_str = f"{age:.0f} yr" if isinstance(age, (int, float)) else str(age)
        meta_str = (
            f"  Patient:        UniqueID {uid} | Stay #{stayid}\n"
            f"  Profile:        Age {age_str} | {unit}\n"
            f"  Actual outcome: {actual_label}"
            f"  (ground truth — NOT available at prediction time)\n"
            f"  {'─' * 62}\n"
        )

    shap_lines = []
    if shap_contributions:
        pos = shap_contributions.get("top_features_increasing_prediction", [])
        neg = shap_contributions.get("top_features_decreasing_prediction", [])
        shap_lines.append("  Features contributing to the model's prediction:")
        shap_lines.append("  (SHAP values show model behavior — not clinical causation)\n")
        if pos:
            shap_lines.append("  Increasing mortality prediction:")
            for f in pos[:3]:
                shap_lines.append(
                    f"    ↑  {f['feature']:<34} val={f['feature_value']:<8}  SHAP={f['shap_value']:+.3f}"
                )
        if neg:
            shap_lines.append("\n  Decreasing mortality prediction:")
            for f in neg[:3]:
                shap_lines.append(
                    f"    ↓  {f['feature']:<34} val={f['feature_value']:<8}  SHAP={f['shap_value']:+.3f}"
                )
    else:
        shap_lines.append("  SHAP contributions not available.")

    shap_str = "\n".join(shap_lines)

    reason_str = ""
    if reasons:
        reason_str = "  Reason(s):      " + "\n                  ".join(reasons) + "\n"

    report = (
        f"╔══════════════════════════════════════════════════════════════════╗\n"
        f"║  SafePredict — Patient Reliability Report #{patient_idx:04d}                ║\n"
        f"╠══════════════════════════════════════════════════════════════════╣\n"
        f"{meta_str}"
        f"  Mortality probability:  {y_prob_calibrated:>6.1%}  (calibrated champion model)\n"
        f"\n"
        f"  ── Uncertainty (Bootstrap, N=30 resamples) ──────────────────────\n"
        f"  Ensemble mean:          {uncertainty_mean:>6.1%}\n"
        f"  Prediction variability: {uncertainty_std:>6.3f}  [{u_band}]\n"
        f"  Threshold applied:      {uncertainty_thresh:.3f}\n"
        f"\n"
        f"  ── Data Quality ─────────────────────────────────────────────────\n"
        f"  Composite DQ score:     {dq_score:>6.2f}  [{dq_band}]  (threshold: {dq_thresh:.2f})\n"
        f"  Lab types available:    {str(avail_labs):>4} / {max_labs}\n"
        f"  Vital types available:  {str(avail_vits):>4} / {max_vitals}\n"
        f"  Missing feature count:  {str(miss_feats):>4}\n"
        f"\n"
        f"  ── SafePredict Decision ─────────────────────────────────────────\n"
        f"  {decision_icon}  {decision}\n"
        f"{reason_str}"
        f"{abstain_note}"
        f"\n"
        f"  ── SHAP Explanation ─────────────────────────────────────────────\n"
        f"{shap_str}\n"
        f"╚══════════════════════════════════════════════════════════════════╝"
    )
    return report


# ==============================================================================
# 10. STRATEGY COMPARISON TABLE
# ==============================================================================

def format_strategy_table(
    results: Dict[str, Dict[str, Any]],
    split_name: str = "Test",
) -> str:
    """Format a text comparison table of all SafePredict strategies."""
    strategy_labels = {
        "model_only": "Model Only (baseline)",
        "uncertainty_abstain": "Uncertainty Abstain",
        "dq_abstain": "DQ Score Abstain",
        "safepredict_combined": "SafePredict Combined",
    }
    header = (
        f"\nSafePredict Strategy Comparison - {split_name} Set\n"
        + "=" * 94 + "\n"
        + f"{'Strategy':<24} | {'Coverage':>8} | {'Abstained':>9} | "
          f"{'N Acc':>6} | {'Error Rate':>10} | "
          f"{'AUROC':>7} | {'PR-AUC':>7} | {'Brier':>7}\n"
        + "-" * 94
    )
    rows = []
    for strat, m in results.items():
        label = strategy_labels.get(strat, strat)
        cov = f"{m['coverage']:.1%}"
        abst = f"{m['abstention_rate']:.1%}"
        n_acc = str(m["n_accepted"])
        err = f"{m['error_rate_accepted']:.1%}" if m["error_rate_accepted"] is not None else "   N/A"
        auroc = f"{m['auroc_accepted']:.3f}" if m["auroc_accepted"] is not None else "  N/A"
        pr_auc = f"{m['pr_auc_accepted']:.3f}" if m["pr_auc_accepted"] is not None else "  N/A"
        brier = f"{m['brier_accepted']:.4f}" if m["brier_accepted"] is not None else "  N/A"
        rows.append(
            f"{label:<24} | {cov:>8} | {abst:>9} | "
            f"{n_acc:>6} | {err:>10} | "
            f"{auroc:>7} | {pr_auc:>7} | {brier:>7}"
        )
    return header + "\n".join(rows) + "\n" + "=" * 94


# ==============================================================================
# 11. ORCHESTRATION — run_phase_8_safepredict()
# ==============================================================================

def run_phase_8_safepredict(
    random_state: int = RANDOM_STATE,
    n_bootstraps: int = N_BOOTSTRAPS,
    min_coverage: float = MIN_COVERAGE,
) -> Dict[str, Any]:
    """Execute end-to-end Phase 8 SafePredict Reliability Layer.

    Steps:
      1. Load cohort splits and champion model.
      2. Bootstrap uncertainty estimation (train -> val, train -> test).
      3. Data quality score computation for val and test.
      4. Threshold selection on validation set only (thresholds then frozen).
      5. Evaluate all four strategies on held-out test set.
      6. Risk-vs-Coverage sweep and plot.
      7. Save results JSON.

    Returns:
        Dict with all intermediate and final outputs.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("SafePredict-XAI: Phase 8 - SafePredict Reliability Layer")
    print("=" * 75)

    # Step 1: Data & Model
    print("\n[Step 1/6] Loading cohort splits and champion model...")
    split_data = prepare_model_splits(random_state=random_state)
    X_train = split_data["X_train"]
    y_train = split_data["y_train"]
    X_val = split_data["X_val"]
    y_val = split_data["y_val"]
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]
    num_cols = split_data["num_cols"]
    cat_cols = split_data["cat_cols"]

    model_path = MODELS_DIR / "final_mortality_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Champion model not found at {model_path}. Run src/evaluate.py first.")
    champion_model = joblib.load(model_path)

    val_probs = champion_model.predict_proba(X_val)[:, 1]
    test_probs = champion_model.predict_proba(X_test)[:, 1]
    print(f"  • Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  • Val mortality: {y_val.mean():.2%} | Test mortality: {y_test.mean():.2%}")

    # Step 2: Bootstrap Uncertainty
    print(f"\n[Step 2/6] Fitting preprocessor & running {n_bootstraps} bootstrap resamples...")
    preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=True)
    preprocessor.fit(X_train, y_train)

    X_train_pre = preprocessor.transform(X_train)
    X_val_pre = preprocessor.transform(X_val)
    X_test_pre = preprocessor.transform(X_test)

    val_boot_mean, val_boot_std = compute_bootstrap_uncertainty(
        X_train_pre, y_train, X_val_pre, n_bootstraps=n_bootstraps, random_state=random_state,
    )
    test_boot_mean, test_boot_std = compute_bootstrap_uncertainty(
        X_train_pre, y_train, X_test_pre, n_bootstraps=n_bootstraps, random_state=random_state,
    )
    print(f"  • Val  uncertainty std: mean={val_boot_std.mean():.4f} ± {val_boot_std.std():.4f}")
    print(f"  • Test uncertainty std: mean={test_boot_std.mean():.4f} ± {test_boot_std.std():.4f}")

    # Step 3: Data Quality Scores
    print("\n[Step 3/6] Computing data quality scores...")
    model_df = pl.read_parquet(PROCESSED_DATA_DIR / "model_data.parquet")

    quality_flags_df: Optional[pl.DataFrame] = None
    if QUALITY_FLAGS_PATH.exists():
        quality_flags_df = pl.read_parquet(QUALITY_FLAGS_PATH)
        print(f"  • Quality flags loaded: {quality_flags_df.shape[0]} rows")
    else:
        print(f"  • Quality flags not found — using 3-component DQ score.")

    all_dq_scores = compute_data_quality_scores(model_df, quality_flags_df)

    # Align DQ scores to val/test ordering using the same split indices
    patient_df_path = PROCESSED_DATA_DIR / "cohort_patient.parquet"
    patient_df = pl.read_parquet(patient_df_path)
    stay_to_pid = dict(zip(
        patient_df["patientunitstayid"].to_list(),
        patient_df["uniquepid"].to_list(),
    ))
    groups = np.array([stay_to_pid[sid] for sid in model_df["patientunitstayid"].to_list()])
    feature_cols = [c for c in model_df.columns if c not in IDENTIFIER_COLS and c != TARGET_COL]
    X_df_full = model_df.select(feature_cols).to_pandas()
    y_arr_full = model_df[TARGET_COL].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    splits_full = list(sgkf.split(X_df_full, y_arr_full, groups))
    test_idx_full = splits_full[0][1]
    val_idx_full = splits_full[1][1]

    test_stay_ids = model_df["patientunitstayid"].to_numpy()[test_idx_full].tolist()
    val_stay_ids = model_df["patientunitstayid"].to_numpy()[val_idx_full].tolist()

    val_dq = all_dq_scores.loc[val_stay_ids].values
    test_dq = all_dq_scores.loc[test_stay_ids].values

    print(f"  • Val  DQ: mean={val_dq.mean():.3f} ± {val_dq.std():.3f}  "
          f"[{val_dq.min():.3f}, {val_dq.max():.3f}]")
    print(f"  • Test DQ: mean={test_dq.mean():.3f} ± {test_dq.std():.3f}  "
          f"[{test_dq.min():.3f}, {test_dq.max():.3f}]")

    # Step 4: Threshold selection on validation only
    print(f"\n[Step 4/6] Selecting thresholds on validation set (min_coverage={min_coverage:.0%})...")
    config = select_thresholds_on_validation(
        val_uncertainty_std=val_boot_std,
        val_dq_scores=val_dq,
        val_y=y_val,
        val_probs=val_probs,
        min_coverage=min_coverage,
    )
    print(f"  • Uncertainty threshold (frozen): {config.uncertainty_threshold:.4f}")
    print(f"  • DQ threshold (frozen):          {config.dq_threshold:.4f}")
    print(f"  • Val coverage achieved:          {config.val_coverage:.1%}")
    print(f"  • Val AUROC (accepted):           {config.val_auroc_accepted:.4f}")

    # Step 5: Test set evaluation
    print("\n[Step 5/6] Evaluating four strategies on held-out test set...")
    test_results = evaluate_all_strategies(
        y_true=y_test,
        y_prob=test_probs,
        uncertainty_std=test_boot_std,
        dq_scores=test_dq,
        config=config,
    )
    print(format_strategy_table(test_results, "Test"))

    # Step 6: Risk vs Coverage sweep
    print("\n[Step 6/6] Generating Risk-vs-Coverage sweep...")
    sweep_df = compute_risk_vs_coverage(
        uncertainty_std=test_boot_std,
        dq_scores=test_dq,
        y_true=y_test,
        y_prob=test_probs,
        n_points=40,
    )
    mo_auroc = test_results["model_only"].get("auroc_accepted")
    if mo_auroc is None and len(np.unique(y_test)) >= 2:
        mo_auroc = float(roc_auc_score(y_test, test_probs))
    baseline_auroc = mo_auroc or 0.0

    strategy_coverages = {
        "uncertainty_abstain": test_results["uncertainty_abstain"]["coverage"],
        "dq_abstain": test_results["dq_abstain"]["coverage"],
        "safepredict_combined": test_results["safepredict_combined"]["coverage"],
    }
    plot_path = FIGURES_DIR / "safepredict_risk_coverage.png"
    plot_risk_vs_coverage(sweep_df, baseline_auroc, plot_path, strategy_coverages)
    print(f"  • Saved: {plot_path}")

    # Save JSON
    def _safe_float(v: Any) -> Any:
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        return v

    payload: Dict[str, Any] = {
        "phase": 8,
        "description": "SafePredict Reliability Layer — ACCEPT / ABSTAIN decisions",
        "abstain_notice": (
            "ABSTAIN means the reliability criteria were not met. "
            "It does NOT mean the patient is safe or high-risk."
        ),
        "config": {
            "n_bootstraps": n_bootstraps,
            "min_coverage_constraint": min_coverage,
            "uncertainty_threshold": config.uncertainty_threshold,
            "dq_threshold": config.dq_threshold,
            "val_coverage": config.val_coverage,
            "val_auroc_accepted": config.val_auroc_accepted,
        },
        "test_results": {
            name: {k: _safe_float(v) for k, v in metrics.items()}
            for name, metrics in test_results.items()
        },
    }
    results_json_path = METRICS_DIR / "safepredict_results.json"
    with open(results_json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  • Saved results JSON: {results_json_path}")

    print("\n" + "=" * 75)
    print("Phase 8 SafePredict Reliability Layer - COMPLETE")
    print("=" * 75)

    return {
        "config": config,
        "test_results": test_results,
        "val_boot_mean": val_boot_mean,
        "val_boot_std": val_boot_std,
        "test_boot_mean": test_boot_mean,
        "test_boot_std": test_boot_std,
        "val_dq": val_dq,
        "test_dq": test_dq,
        "val_probs": val_probs,
        "test_probs": test_probs,
        "y_val": y_val,
        "y_test": y_test,
        "X_val": X_val,
        "X_test": X_test,
        "sweep_df": sweep_df,
        "plot_path": str(plot_path),
        "results_json_path": str(results_json_path),
        "val_stay_ids": val_stay_ids,
        "test_stay_ids": test_stay_ids,
        "model_df": model_df,
    }


if __name__ == "__main__":
    run_phase_8_safepredict()
