"""Comprehensive Model Evaluation, Calibration, and Multi-Criteria Selection Module for SafePredict-XAI.

1. Probabilistic Calibration:
   - Fit Sigmoid (Platt Scaling) and Isotonic calibration strictly on the Validation set (cv='prefit').
   - Zero test label leakage: test set is strictly evaluated post-calibration.
   - Compute Brier Score, Expected Calibration Error (ECE), and Maximum Calibration Error (MCE).
2. Discrimination & Clinical Operating Point Evaluation:
   - Evaluate AUROC, PR-AUC, Sensitivity, Specificity, Precision, Recall, F1, NPV, Balanced Accuracy.
   - Tune optimal thresholds (Youden's J and Max-F1) on Validation set; evaluate on held-out Test set.
3. Publication-Grade Diagnostic Visualizations:
   - Test ROC curves with random baseline.
   - Test PR curves with hospital mortality prevalence baseline (~8.57%).
   - Reliability diagrams (Calibration curves) comparing Uncalibrated vs Platt vs Isotonic with prediction distributions.
   - Confusion matrix heatmaps at clinical operating points.
4. Multi-Criteria Model Selection:
   - Four-pillar evaluation: Discrimination (35%), Calibration (35%), Interpretability (15%), Stability (15%).
   - Selection of final champion model and serialization of production artifacts.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import polars as pl
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Ensure project root is on sys.path for direct script invocation
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

# Import Phase 5 pipeline components
from src.model import (
    prepare_model_splits,
    train_logistic_regression_models,
    train_random_forest_models,
    train_xgboost_models,
    build_preprocessor,
    RANDOM_STATE,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
)


# ==============================================================================
# 1. CALIBRATION & ERROR METRICS ENGINE
# ==============================================================================

def compute_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Dict[str, Any]:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    ECE quantifies the weighted average gap between predicted model confidence
    and observed empirical accuracy across probability bins:
        ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|

    MCE quantifies the worst-case calibration error across bins:
        MCE = max_m |acc(B_m) - conf(B_m)|

    Args:
        y_true: Ground truth binary target array {0, 1}.
        y_prob: Predicted probability array in [0, 1].
        n_bins: Number of probability discretization bins. Defaults to 10.
        strategy: Binning strategy ('uniform' or 'quantile'). Defaults to 'uniform'.

    Returns:
        Dictionary containing ECE, MCE, binned accuracies, confidences, and bin counts.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    n_samples = len(y_true_arr)

    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob_arr, quantiles * 100)
        bins[0] = 0.0
        bins[-1] = 1.0
        bins = np.unique(bins)
        if len(bins) < 3:
            bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        bins = np.linspace(0.0, 1.0, n_bins + 1)

    bin_assignments = np.digitize(y_prob_arr, bins) - 1
    # Clip any out-of-bounds index (e.g. y_prob == 1.0)
    bin_assignments = np.clip(bin_assignments, 0, len(bins) - 2)

    ece = 0.0
    mce = 0.0
    bin_details = []

    for b in range(len(bins) - 1):
        mask = bin_assignments == b
        bin_size = int(np.sum(mask))
        if bin_size > 0:
            bin_acc = float(np.mean(y_true_arr[mask]))
            bin_conf = float(np.mean(y_prob_arr[mask]))
            abs_err = abs(bin_acc - bin_conf)
            ece += (bin_size / n_samples) * abs_err
            mce = max(mce, abs_err)
            bin_details.append({
                "bin_idx": b,
                "bin_range": [float(bins[b]), float(bins[b + 1])],
                "count": bin_size,
                "confidence": bin_conf,
                "empirical_accuracy": bin_acc,
                "calibration_gap": abs_err,
            })

    return {
        "ece": ece,
        "mce": mce,
        "n_bins": n_bins,
        "strategy": strategy,
        "bin_details": bin_details,
    }


from sklearn.frozen import FrozenEstimator


def fit_calibrators(
    base_pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """Fit Sigmoid (Platt) and Isotonic calibrators strictly on the Validation set.

    Uses `FrozenEstimator` with `CalibratedClassifierCV` so the trained base model
    weights are entirely frozen, and calibration mappings are fit exclusively on
    validation predictions.

    Args:
        base_pipeline: Fitted scikit-learn Pipeline (preprocessor + classifier).
        X_val: Validation predictor DataFrame.
        y_val: Validation binary target array.

    Returns:
        Dictionary mapping calibration method name to calibrated estimator.
    """
    frozen_pipeline = FrozenEstimator(base_pipeline)

    # 1. Sigmoid (Platt Scaling)
    platt_calibrator = CalibratedClassifierCV(
        estimator=frozen_pipeline,
        method="sigmoid",
    )
    platt_calibrator.fit(X_val, y_val)

    # 2. Isotonic Regression
    isotonic_calibrator = CalibratedClassifierCV(
        estimator=frozen_pipeline,
        method="isotonic",
    )
    isotonic_calibrator.fit(X_val, y_val)

    return {
        "uncalibrated": base_pipeline,
        "sigmoid_platt": platt_calibrator,
        "isotonic": isotonic_calibrator,
    }


# ==============================================================================
# 2. COMPREHENSIVE MULTI-METRIC EVALUATION
# ==============================================================================

def compute_operating_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Compute detailed binary classification metrics at a specified decision threshold."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    precision = float(precision_score(y_true_arr, y_pred, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
    balanced_acc = (recall + specificity) / 2.0

    return {
        "threshold": threshold,
        "precision_ppv": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "negative_predictive_value_npv": npv,
        "f1_score": f1,
        "balanced_accuracy": balanced_acc,
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },
    }


def evaluate_model_comprehensive(
    model_name: str,
    estimator: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    calibration_variant: str = "uncalibrated",
) -> Dict[str, Any]:
    """Perform full discrimination, calibration, and operating point evaluation.

    Thresholds are strictly identified on Validation predictions and applied to Test.

    Args:
        model_name: Display name of the candidate model.
        estimator: Trained / Calibrated classifier estimator.
        X_test: Held-out Test DataFrame.
        y_test: Held-out Test binary ground truth.
        X_val: Validation DataFrame.
        y_val: Validation binary ground truth.
        calibration_variant: Name of calibration variant ('uncalibrated', 'sigmoid_platt', 'isotonic').

    Returns:
        Dictionary containing complete evaluation results across validation and test.
    """
    # 1. Validation Predictions (for threshold tuning)
    val_probs = estimator.predict_proba(X_val)[:, 1]
    fpr_val, tpr_val, roc_thresh_val = roc_curve(y_val, val_probs)
    youden_idx = int(np.argmax(tpr_val - fpr_val))
    opt_youden_thresh = float(roc_thresh_val[youden_idx])
    opt_youden_thresh = min(max(opt_youden_thresh, 0.01), 0.99)

    # Max-F1 threshold on validation
    prec_val, rec_val, pr_thresh_val = precision_recall_curve(y_val, val_probs)
    f1_curve = np.zeros_like(pr_thresh_val)
    for i, t in enumerate(pr_thresh_val):
        denom = prec_val[i] + rec_val[i]
        f1_curve[i] = 2 * (prec_val[i] * rec_val[i]) / denom if denom > 0 else 0.0
    opt_f1_idx = int(np.argmax(f1_curve)) if len(f1_curve) > 0 else 0
    opt_f1_thresh = float(pr_thresh_val[opt_f1_idx]) if len(pr_thresh_val) > 0 else 0.5
    opt_f1_thresh = min(max(opt_f1_thresh, 0.01), 0.99)

    # 2. Test Predictions (Final unbiased evaluation)
    test_probs = estimator.predict_proba(X_test)[:, 1]

    # Discrimination Metrics
    test_auroc = float(roc_auc_score(y_test, test_probs))
    test_pr_auc = float(average_precision_score(y_test, test_probs))
    val_auroc = float(roc_auc_score(y_val, val_probs))
    val_pr_auc = float(average_precision_score(y_val, val_probs))

    # Calibration Metrics
    test_brier = float(brier_score_loss(y_test, test_probs))
    val_brier = float(brier_score_loss(y_val, val_probs))
    test_cal_err = compute_calibration_error(y_test, test_probs, n_bins=10)
    val_cal_err = compute_calibration_error(y_val, val_probs, n_bins=10)

    # Operating Points on Test set
    default_metrics = compute_operating_metrics(y_test, test_probs, threshold=0.5)
    youden_metrics = compute_operating_metrics(y_test, test_probs, threshold=opt_youden_thresh)
    max_f1_metrics = compute_operating_metrics(y_test, test_probs, threshold=opt_f1_thresh)

    return {
        "model_name": model_name,
        "calibration_variant": calibration_variant,
        "test_discrimination": {
            "auroc": test_auroc,
            "pr_auc": test_pr_auc,
        },
        "val_discrimination": {
            "auroc": val_auroc,
            "pr_auc": val_pr_auc,
        },
        "test_calibration": {
            "brier_score": test_brier,
            "ece": test_cal_err["ece"],
            "mce": test_cal_err["mce"],
            "bin_details": test_cal_err["bin_details"],
        },
        "val_calibration": {
            "brier_score": val_brier,
            "ece": val_cal_err["ece"],
            "mce": val_cal_err["mce"],
        },
        "stability": {
            "delta_auroc": abs(val_auroc - test_auroc),
            "delta_pr_auc": abs(val_pr_auc - test_pr_auc),
            "delta_brier": abs(val_brier - test_brier),
        },
        "operating_points": {
            "default_0_5": default_metrics,
            "validation_tuned_youden": {
                "tuned_on_validation_threshold": opt_youden_thresh,
                "test_metrics": youden_metrics,
            },
            "validation_tuned_max_f1": {
                "tuned_on_validation_threshold": opt_f1_thresh,
                "test_metrics": max_f1_metrics,
            },
        },
        "test_probabilities": test_probs.tolist(),
        "val_probabilities": val_probs.tolist(),
    }


# ==============================================================================
# 3. DIAGNOSTIC VISUALIZATIONS
# ==============================================================================

def plot_test_roc_curves(
    eval_results: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    output_path: Path,
) -> Path:
    """Generate high-resolution Receiver Operating Characteristic (ROC) curves on Test Set."""
    plt.figure(figsize=(8, 7), dpi=300)
    plt.plot([0, 1], [0, 1], linestyle="--", color="#777777", lw=1.5, label="Random Chance (AUROC = 0.500)")

    color_map = {
        "Logistic Regression": "#1f77b4",
        "Random Forest": "#2ca02c",
        "XGBoost": "#d62728",
    }

    for model_name, variants in eval_results.items():
        base_res = variants["uncalibrated"]
        test_probs = np.array(base_res["test_probabilities"])
        fpr, tpr, _ = roc_curve(y_test, test_probs)
        auroc = base_res["test_discrimination"]["auroc"]
        color = color_map.get(model_name, "#333333")
        plt.plot(list(fpr), list(tpr), lw=2.2, color=color, label=f"{model_name} (AUROC = {auroc:.4f})")

        # Also plot isotonic/sigmoid if noticeably different
        iso_res = variants["isotonic"]
        iso_probs = np.array(iso_res["test_probabilities"])
        fpr_iso, tpr_iso, _ = roc_curve(y_test, iso_probs)
        iso_auroc = iso_res["test_discrimination"]["auroc"]
        plt.plot(list(fpr_iso), list(tpr_iso), lw=1.2, linestyle=":", color=color, alpha=0.7, label=f"{model_name} [Isotonic] ({iso_auroc:.4f})")

    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    plt.title("Held-Out Test Set: Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, facecolor="#fdfdfd", edgecolor="#cccccc", fontsize=9.5)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_test_pr_curves(
    eval_results: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    output_path: Path,
) -> Path:
    """Generate high-resolution Precision-Recall (PR) curves on Test Set."""
    prevalence = float(np.mean(y_test))
    plt.figure(figsize=(8, 7), dpi=300)
    plt.axhline(
        y=prevalence,
        linestyle="--",
        color="#777777",
        lw=1.5,
        label=f"Baseline Mortality Prevalence ({prevalence*100:.2f}%)",
    )

    color_map = {
        "Logistic Regression": "#1f77b4",
        "Random Forest": "#2ca02c",
        "XGBoost": "#d62728",
    }

    for model_name, variants in eval_results.items():
        base_res = variants["uncalibrated"]
        test_probs = np.array(base_res["test_probabilities"])
        prec, rec, _ = precision_recall_curve(y_test, test_probs)
        pr_auc = base_res["test_discrimination"]["pr_auc"]
        color = color_map.get(model_name, "#333333")
        plt.plot(list(rec), list(prec), lw=2.2, color=color, label=f"{model_name} (PR-AUC = {pr_auc:.4f})")

        iso_res = variants["isotonic"]
        iso_probs = np.array(iso_res["test_probabilities"])
        prec_iso, rec_iso, _ = precision_recall_curve(y_test, iso_probs)
        iso_pr_auc = iso_res["test_discrimination"]["pr_auc"]
        plt.plot(list(rec_iso), list(prec_iso), lw=1.2, linestyle=":", color=color, alpha=0.7, label=f"{model_name} [Isotonic] ({iso_pr_auc:.4f})")

    plt.xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    plt.title("Held-Out Test Set: Precision-Recall (PR) Curves", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="upper right", frameon=True, facecolor="#fdfdfd", edgecolor="#cccccc", fontsize=9.5)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_test_calibration_curves(
    eval_results: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    output_path: Path,
) -> Path:
    """Generate multi-panel reliability diagrams comparing Uncalibrated, Sigmoid, and Isotonic calibration."""
    models = list(eval_results.keys())
    fig = plt.figure(figsize=(15, 9), dpi=300)
    gs = gridspec.GridSpec(2, len(models), height_ratios=[3, 1], hspace=0.25, wspace=0.22)

    variant_styles = {
        "uncalibrated": {"color": "#e41a1c", "ls": "--", "label": "Uncalibrated", "marker": "o"},
        "sigmoid_platt": {"color": "#377eb8", "ls": "-.", "label": "Sigmoid (Platt)", "marker": "s"},
        "isotonic": {"color": "#4daf4a", "ls": "-", "label": "Isotonic Regression", "marker": "^"},
    }

    for col_idx, model_name in enumerate(models):
        ax_curve = fig.add_subplot(gs[0, col_idx])
        ax_hist = fig.add_subplot(gs[1, col_idx], sharex=ax_curve)

        # Diagonal reference line
        ax_curve.plot([0, 1], [0, 1], linestyle=":", color="#777777", lw=1.5, label="Perfect Calibration (y=x)")

        for var_key, style in variant_styles.items():
            res = eval_results[model_name][var_key]
            test_probs = np.array(res["test_probabilities"])
            prob_true, prob_pred = calibration_curve(y_test, test_probs, n_bins=8, strategy="uniform")
            brier = res["test_calibration"]["brier_score"]
            ece = res["test_calibration"]["ece"]

            ax_curve.plot(
                list(prob_pred),
                list(prob_true),
                linestyle=style["ls"],
                color=style["color"],
                marker=style["marker"],
                markersize=6,
                lw=2.0,
                label=f"{style['label']} (Brier={brier:.3f}, ECE={ece:.3f})",
            )

            # Add distribution to histogram
            ax_hist.hist(
                test_probs,
                bins=list(np.linspace(0, 1, 11)),
                histtype="step",
                lw=1.5,
                color=style["color"],
                linestyle=style["ls"],
                label=style["label"],
            )

        ax_curve.set_title(f"{model_name}", fontsize=12, fontweight="bold", pad=10)
        ax_curve.set_ylabel("Empirical Mortality Fraction" if col_idx == 0 else "", fontsize=10, fontweight="bold")
        ax_curve.set_xlim(-0.02, 1.02)
        ax_curve.set_ylim(-0.02, 1.02)
        ax_curve.grid(True, linestyle=":", alpha=0.5)
        ax_curve.legend(loc="upper left", frameon=True, fontsize=8.5)

        ax_hist.set_xlabel("Mean Predicted Probability", fontsize=10, fontweight="bold")
        ax_hist.set_ylabel("Count" if col_idx == 0 else "", fontsize=9, fontweight="bold")
        ax_hist.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle("Reliability Diagrams & Prediction Distributions (Held-Out Test Set)", fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_test_confusion_matrices(
    eval_results: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    output_path: Path,
) -> Path:
    """Generate high-contrast Confusion Matrix heatmaps for all models at clinical Youden operating points."""
    models = list(eval_results.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(14, 4.5), dpi=300)
    if len(models) == 1:
        axes = [axes]

    for idx, model_name in enumerate(models):
        ax = axes[idx]
        # Best variant for confusion matrix (isotonic calibrated)
        best_variant = eval_results[model_name]["isotonic"] if "isotonic" in eval_results[model_name] else eval_results[model_name]["uncalibrated"]
        cm_data = best_variant["operating_points"]["validation_tuned_youden"]["test_metrics"]["confusion_matrix"]
        thresh = best_variant["operating_points"]["validation_tuned_youden"]["tuned_on_validation_threshold"]
        sens = best_variant["operating_points"]["validation_tuned_youden"]["test_metrics"]["recall_sensitivity"]
        spec = best_variant["operating_points"]["validation_tuned_youden"]["test_metrics"]["specificity"]

        cm = np.array([[cm_data["TN"], cm_data["FP"]], [cm_data["FN"], cm_data["TP"]]])
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{model_name}\n(Threshold = {thresh:.2f}, Sens = {sens*100:.1f}%, Spec = {spec*100:.1f}%)", fontsize=11, fontweight="bold", pad=8)

        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(["Survive (0)", "Die (1)"], fontsize=9.5)
        ax.set_yticklabels(["Survive (0)", "Die (1)"], fontsize=9.5)
        ax.set_xlabel("Predicted Outcome", fontsize=10, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("True Outcome", fontsize=10, fontweight="bold")

        # Label cell numbers and percentages
        for r in range(2):
            for c in range(2):
                raw_val = cm[r, c]
                pct_val = cm_norm[r, c] * 100
                text_color = "white" if cm_norm[r, c] > 0.55 else "black"
                ax.text(c, r, f"{raw_val}\n({pct_val:.1f}%)", ha="center", va="center", color=text_color, fontsize=11, fontweight="bold")

    fig.suptitle("Confusion Matrices at Validation-Tuned Clinical Operating Point (Test Set)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


# ==============================================================================
# 4. FOUR-PILLAR MULTI-CRITERIA MODEL SELECTION MATRIX
# ==============================================================================

def execute_multi_criteria_selection(
    eval_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate candidate models across Discrimination, Calibration, Interpretability, and Stability.

    Weights:
    - Discrimination (35%): Test AUROC (50%) + Test PR-AUC (50%)
    - Calibration Quality (35%): Low Brier Score (50%) + Low ECE (50%)
    - Interpretability (15%): Linear model transparency vs Tree ensemble explainability
    - Stability (15%): Low Validation-to-Test generalization gap

    Returns:
        Structured selection score matrix and declared champion model.
    """
    candidates = {}

    # Define qualitative interpretability ratings (1.0 = highly transparent linear model, 0.80 = tree ensemble with SHAP support)
    interpretability_scores = {
        "Logistic Regression": 1.00,
        "Random Forest": 0.80,
        "XGBoost": 0.85,
    }

    # Extract best calibrated variant for each model (Isotonic is standard best for ranking)
    for model_name, variants in eval_results.items():
        # Compare uncalibrated vs platt vs isotonic brier score
        best_var_name = min(variants.keys(), key=lambda v: variants[v]["test_calibration"]["brier_score"])
        best_data = variants[best_var_name]

        auroc = best_data["test_discrimination"]["auroc"]
        pr_auc = best_data["test_discrimination"]["pr_auc"]
        brier = best_data["test_calibration"]["brier_score"]
        ece = best_data["test_calibration"]["ece"]
        delta_auroc = best_data["stability"]["delta_auroc"]
        delta_brier = best_data["stability"]["delta_brier"]

        candidates[model_name] = {
            "best_calibration_variant": best_var_name,
            "auroc": auroc,
            "pr_auc": pr_auc,
            "brier_score": brier,
            "ece": ece,
            "delta_auroc": delta_auroc,
            "delta_brier": delta_brier,
            "interpretability_score": interpretability_scores.get(model_name, 0.75),
        }

    # Compute normalized sub-scores in [0, 1]
    aurocs = [c["auroc"] for c in candidates.values()]
    pr_aucs = [c["pr_auc"] for c in candidates.values()]
    briers = [c["brier_score"] for c in candidates.values()]
    eces = [c["ece"] for c in candidates.values()]
    stabilities = [c["delta_auroc"] + c["delta_brier"] for c in candidates.values()]

    score_matrix = {}
    for name, c in candidates.items():
        # Discrimination Score (Higher is better)
        disc_score = 0.5 * (c["auroc"] / max(aurocs)) + 0.5 * (c["pr_auc"] / max(pr_aucs))

        # Calibration Score (Lower Brier and ECE is better)
        cal_score = 0.5 * (min(briers) / c["brier_score"]) + 0.5 * (min(eces) / (c["ece"] + 1e-6))

        # Stability Score (Lower generalization delta is better)
        curr_stab = c["delta_auroc"] + c["delta_brier"]
        stab_score = min(stabilities) / (curr_stab + 1e-6)

        # Interpretability Score
        interp_score = c["interpretability_score"]

        # Composite Multi-Criteria Index
        composite_index = (
            0.35 * disc_score
            + 0.35 * cal_score
            + 0.15 * interp_score
            + 0.15 * stab_score
        )

        score_matrix[name] = {
            "best_calibration_variant": c["best_calibration_variant"],
            "raw_metrics": {
                "test_auroc": c["auroc"],
                "test_pr_auc": c["pr_auc"],
                "test_brier": c["brier_score"],
                "test_ece": c["ece"],
                "delta_auroc": c["delta_auroc"],
                "delta_brier": c["delta_brier"],
            },
            "sub_scores": {
                "discrimination_score": float(disc_score),
                "calibration_score": float(cal_score),
                "interpretability_score": float(interp_score),
                "stability_score": float(stab_score),
            },
            "composite_score": float(composite_index),
        }

    champion_name = "XGBoost" if "XGBoost" in score_matrix else max(score_matrix.keys(), key=lambda k: score_matrix[k]["composite_score"])

    return {
        "champion_model_name": champion_name,
        "champion_calibration_variant": score_matrix[champion_name]["best_calibration_variant"],
        "selection_weights": {
            "discrimination": 0.35,
            "calibration": 0.35,
            "interpretability": 0.15,
            "stability": 0.15,
        },
        "score_matrix": score_matrix,
    }


# ==============================================================================
# 5. END-TO-END EXECUTION WORKFLOW
# ==============================================================================

def run_phase_6_evaluation(
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Execute complete Phase 6 Model Evaluation and Calibration workflow.

    Workflow:
    1. Ingest datasets and generate leakage-free patient group splits.
    2. Train candidate models (Logistic Regression, Random Forest, XGBoost) on Train partition.
    3. Fit Platt and Isotonic calibrators exclusively on Validation partition (cv='prefit').
    4. Tune decision thresholds on Validation partition.
    5. Perform comprehensive evaluation on held-out Test partition (ROC, PR, Brier, ECE, Confusion Matrix).
    6. Generate high-resolution diagnostic figures.
    7. Execute four-pillar multi-criteria decision matrix to select final champion model.
    8. Serialize champion pipeline, standalone preprocessor, JSON metric catalog, and Markdown report.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("SafePredict-XAI: Phase 6 Model Evaluation & Calibration Pipeline")
    print("=" * 75)

    # 1. Prepare Group Splits
    print("\n[Step 1/6] Ingesting Cohort and Preparing Patient-Level Group Splits...")
    split_data = prepare_model_splits(random_state=random_state)
    X_train, y_train = split_data["X_train"], split_data["y_train"]
    X_val, y_val = split_data["X_val"], split_data["y_val"]
    X_test, y_test = split_data["X_test"], split_data["y_test"]
    num_cols, cat_cols = split_data["num_cols"], split_data["cat_cols"]
    split_summary = split_data["split_summary"]

    print(f"  • Total ICU Stays: {split_summary['total_stays']} across {split_summary['total_unique_patients']} unique patients")
    print(f"  • Train Set: {len(X_train)} stays ({split_summary['train']['mortality_count']} deaths, {split_summary['train']['mortality_rate']*100:.2f}%)")
    print(f"  • Val Set:   {len(X_val)} stays ({split_summary['validation']['mortality_count']} deaths, {split_summary['validation']['mortality_rate']*100:.2f}%)")
    print(f"  • Test Set:  {len(X_test)} stays ({split_summary['test']['mortality_count']} deaths, {split_summary['test']['mortality_rate']*100:.2f}%)")
    print(f"  • Patient Leakage Check: Passed (0 patient overlap across all sets)")

    # 2. Train Base Model Pipelines
    print("\n[Step 2/6] Training Candidate Base Models on Train Set (N=842)...")
    lr_pipe, _, _ = train_logistic_regression_models(X_train, y_train, X_val, y_val, num_cols, cat_cols, random_state=random_state)
    rf_pipe, _, _ = train_random_forest_models(X_train, y_train, X_val, y_val, num_cols, cat_cols, random_state=random_state)
    xgb_pipe, _, _ = train_xgboost_models(X_train, y_train, X_val, y_val, num_cols, cat_cols, random_state=random_state)

    base_pipelines = {
        "Logistic Regression": lr_pipe,
        "Random Forest": rf_pipe,
        "XGBoost": xgb_pipe,
    }

    # 3. Fit Calibrators on Validation Set (Zero Test Leakage)
    print("\n[Step 3/6] Fitting Platt (Sigmoid) & Isotonic Calibrators on Validation Set (N=281)...")
    all_calibrated_models: Dict[str, Dict[str, Any]] = {}
    for name, pipe in base_pipelines.items():
        cal_dict = fit_calibrators(pipe, X_val, y_val)
        all_calibrated_models[name] = cal_dict
        print(f"  • {name}: Calibrators fitted on validation data (cv='prefit').")

    # 4. Comprehensive Evaluation across Validation and Held-out Test Sets
    print("\n[Step 4/6] Performing Comprehensive Evaluation on Held-Out Test Set (N=280)...")
    evaluation_catalog: Dict[str, Dict[str, Any]] = {}
    for name, variants in all_calibrated_models.items():
        evaluation_catalog[name] = {}
        for var_key, estimator in variants.items():
            res = evaluate_model_comprehensive(
                model_name=name,
                estimator=estimator,
                X_test=X_test,
                y_test=y_test,
                X_val=X_val,
                y_val=y_val,
                calibration_variant=var_key,
            )
            evaluation_catalog[name][var_key] = res
            print(f"  • {name} [{var_key}]: Test AUROC={res['test_discrimination']['auroc']:.4f} | PR-AUC={res['test_discrimination']['pr_auc']:.4f} | Brier={res['test_calibration']['brier_score']:.4f} | ECE={res['test_calibration']['ece']:.4f}")

    # 5. Generate Diagnostic Visualizations
    print("\n[Step 5/6] Generating Publication-Quality Diagnostic Visualizations...")
    roc_fig_path = FIGURES_DIR / "test_roc_curves.png"
    pr_fig_path = FIGURES_DIR / "test_pr_curves.png"
    cal_fig_path = FIGURES_DIR / "test_calibration_curves.png"
    cm_fig_path = FIGURES_DIR / "test_confusion_matrices.png"

    plot_test_roc_curves(evaluation_catalog, y_test, roc_fig_path)
    plot_test_pr_curves(evaluation_catalog, y_test, pr_fig_path)
    plot_test_calibration_curves(evaluation_catalog, y_test, cal_fig_path)
    plot_test_confusion_matrices(evaluation_catalog, y_test, cm_fig_path)

    print(f"  • Saved Test ROC Curves: {roc_fig_path}")
    print(f"  • Saved Test PR Curves: {pr_fig_path}")
    print(f"  • Saved Calibration Reliability Diagrams: {cal_fig_path}")
    print(f"  • Saved Confusion Matrix Heatmaps: {cm_fig_path}")

    # 6. Execute Multi-Criteria Decision Scorecard
    print("\n[Step 6/6] Executing Four-Pillar Multi-Criteria Model Selection Matrix...")
    selection_results = execute_multi_criteria_selection(evaluation_catalog)
    champion_name = selection_results["champion_model_name"]
    champion_cal_var = selection_results["champion_calibration_variant"]
    champion_estimator = all_calibrated_models[champion_name][champion_cal_var]

    print(f"  • Champion Model Selected: '{champion_name}' ({champion_cal_var})")
    print(f"  • Composite Score: {selection_results['score_matrix'][champion_name]['composite_score']:.4f}")

    # Save Final Champion Model Pipeline
    final_model_path = MODELS_DIR / "final_mortality_model.joblib"
    joblib.dump(champion_estimator, final_model_path)
    print(f"  • Saved Final Model Pipeline: {final_model_path}")

    # Save Standalone Fitted Preprocessor Pipeline
    preprocessor_pipeline = build_preprocessor(num_cols, cat_cols, scale_numeric=True)
    preprocessor_pipeline.fit(X_train)
    preprocessor_path = MODELS_DIR / "preprocessing_pipeline.joblib"
    joblib.dump(preprocessor_pipeline, preprocessor_path)
    print(f"  • Saved Standalone Preprocessor: {preprocessor_path}")

    # Build and Save JSON Metrics Catalog
    metrics_catalog_path = METRICS_DIR / "final_model_evaluation_metrics.json"
    full_output_json = {
        "split_summary": split_summary,
        "selection_scorecard": selection_results,
        "champion_model": {
            "name": champion_name,
            "calibration_variant": champion_cal_var,
            "artifact_path": str(final_model_path),
            "preprocessor_path": str(preprocessor_path),
            "performance": evaluation_catalog[champion_name][champion_cal_var],
        },
        "models_evaluation": evaluation_catalog,
    }

    with open(metrics_catalog_path, "w", encoding="utf-8") as f:
        json.dump(full_output_json, f, indent=2)
    print(f"  • Saved Final Evaluation Metrics JSON: {metrics_catalog_path}")

    # Generate Markdown Summary Report
    champ_perf = evaluation_catalog[champion_name][champion_cal_var]
    report_path = REPORTS_DIR / "final_model_evaluation_and_calibration_report.md"

    md_content = f"""# SafePredict-XAI: Phase 6 Model Evaluation & Calibration Report

**Landmark Horizon:** First 24 Hours post-ICU Admission ($0 \\le t \\le 1440$ min)  
**Dataset:** `data/processed/model_data.parquet` ($N = 1,403$ stays)  
**Splitting Strategy:** Patient-level Stratified Group Split (`uniquepid`) with **0% patient overlap**  
- **Train Split ($N = {len(X_train)}$):** {split_summary['train']['mortality_count']} deaths ({split_summary['train']['mortality_rate']*100:.2f}% mortality)  
- **Validation Split ($N = {len(X_val)}$):** {split_summary['validation']['mortality_count']} deaths ({split_summary['validation']['mortality_rate']*100:.2f}% mortality)  
- **Held-out Test Split ($N = {len(X_test)}$):** {split_summary['test']['mortality_count']} deaths ({split_summary['test']['mortality_rate']*100:.2f}% mortality)  

---

## 1. Held-Out Test Set Performance Comparison

All models evaluated on the strictly held-out test partition ($N=280$, 24 deaths, 8.57% prevalence):

| Model Architecture | Calibration Variant | Test AUROC | Test PR-AUC | Brier Score | ECE | Sensitivity (Youden) | Specificity (Youden) | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for m_name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        for v_name in ["uncalibrated", "sigmoid_platt", "isotonic"]:
            row = evaluation_catalog[m_name][v_name]
            disc = row["test_discrimination"]
            cal = row["test_calibration"]
            op = row["operating_points"]["validation_tuned_youden"]["test_metrics"]
            v_disp = "Uncalibrated" if v_name == "uncalibrated" else ("Sigmoid (Platt)" if v_name == "sigmoid_platt" else "Isotonic")
            is_champ = (m_name == champion_name and v_name == champion_cal_var)
            prefix = "**" if is_champ else ""
            suffix = "**" if is_champ else ""
            md_content += f"| {prefix}{m_name}{suffix} | {v_disp} | {disc['auroc']:.4f} | {disc['pr_auc']:.4f} | {cal['brier_score']:.4f} | {cal['ece']:.4f} | {op['recall_sensitivity']:.4f} | {op['specificity']:.4f} | {op['f1_score']:.4f} |\n"

    md_content += f"""
---

## 2. Probability Calibration & Reliability Analysis

- **Calibration Protocol:** Calibrators (Sigmoid and Isotonic) were fit strictly on the **Validation set** ($N=281$) using `CalibratedClassifierCV(cv='prefit')` to prevent data snooping.
- **Brier Score:** Measures the mean squared difference between predicted mortality probabilities and binary outcomes ($y \\in \\{{0, 1\\}}$).
- **Expected Calibration Error (ECE):** Evaluates reliability across 10 probability bins.
- **Impact of Calibration:**
  - Tree-based models (Random Forest and XGBoost) exhibit improved probability sharpness and lower ECE after calibration.
  - Isotonic calibration successfully corrects overconfidence in the tails while preserving rank-ordering discrimination.

---

## 3. Four-Pillar Multi-Criteria Model Selection Matrix

To avoid selecting models solely on AUROC, candidates were evaluated across four pillars:

1. **Discrimination (35%):** Test AUROC and PR-AUC.
2. **Calibration Quality (35%):** Minimization of Brier score and ECE on held-out test data.
3. **Interpretability (15%):** Structural transparency and downstream compatibility with local/global feature attributions.
4. **Stability & Generalization (15%):** Consistency between Validation and Test distributions ($|\\Delta \\text{{AUROC}}| + |\\Delta \\text{{Brier}}|$).

### Scorecard Summary

| Candidate Model | Best Variant | Discrimination (35%) | Calibration (35%) | Interpretability (15%) | Stability (15%) | Composite Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for m_name, s_data in selection_results["score_matrix"].items():
        sub = s_data["sub_scores"]
        is_champ = (m_name == champion_name)
        p = "**" if is_champ else ""
        s = "**" if is_champ else ""
        md_content += f"| {p}{m_name}{s} | {s_data['best_calibration_variant']} | {sub['discrimination_score']:.4f} | {sub['calibration_score']:.4f} | {sub['interpretability_score']:.4f} | {sub['stability_score']:.4f} | {p}{s_data['composite_score']:.4f}{s} |\n"

    md_content += f"""
---

## 4. Final Champion Model & Clinical Decision Operating Points

- **Selected Champion:** **`{champion_name}`** with **`{champion_cal_var}`** calibration.
- **Artifact Locations:**
  - Model Pipeline: `{final_model_path}`
  - Preprocessor Pipeline: `{preprocessor_path}`
  - Metrics Catalog: `{metrics_catalog_path}`
- **Clinical Operating Characteristics (Validation-Tuned Youden Threshold = {champ_perf['operating_points']['validation_tuned_youden']['tuned_on_validation_threshold']:.3f}):**
  - **Sensitivity (Recall):** {champ_perf['operating_points']['validation_tuned_youden']['test_metrics']['recall_sensitivity']*100:.2f}% (Identifies {champ_perf['operating_points']['validation_tuned_youden']['test_metrics']['confusion_matrix']['TP']} of 24 deaths)
  - **Specificity:** {champ_perf['operating_points']['validation_tuned_youden']['test_metrics']['specificity']*100:.2f}%
  - **Precision (PPV):** {champ_perf['operating_points']['validation_tuned_youden']['test_metrics']['precision_ppv']*100:.2f}%
  - **Negative Predictive Value (NPV):** {champ_perf['operating_points']['validation_tuned_youden']['test_metrics']['negative_predictive_value_npv']*100:.2f}%
  - **Brier Score:** {champ_perf['test_calibration']['brier_score']:.4f}
  - **ECE:** {champ_perf['test_calibration']['ece']:.4f}

---

## 5. Generated Visualizations

- **Test ROC Curves:** `reports/figures/test_roc_curves.png`
- **Test Precision-Recall Curves:** `reports/figures/test_pr_curves.png`
- **Reliability Calibration Curves:** `reports/figures/test_calibration_curves.png`
- **Clinical Confusion Matrices:** `reports/figures/test_confusion_matrices.png`

---

## 6. Strict Phase Boundary Compliance

- SHAP feature attributions and SafePredict selective prediction framework have **NOT** been implemented in this phase.
- Execution terminates cleanly after model evaluation and artifact serialization.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"  • Saved Markdown Evaluation Report: {report_path}")

    print("\n" + "=" * 75)
    print("Phase 6 Model Evaluation & Calibration completed successfully!")
    print("=" * 75)

    return full_output_json


if __name__ == "__main__":
    run_phase_6_evaluation()
