"""Append Phase 8 SafePredict cells to notebooks/05_xai_safepredict.ipynb."""
import json
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
NB_PATH = PROJECT_ROOT / "notebooks" / "05_xai_safepredict.ipynb"


def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


CELLS = []

# ─── Section header ──────────────────────────────────────────────────────────
CELLS.append(md_cell(
    "---\n\n"
    "## Phase 8 \u2014 SafePredict Reliability Layer\n\n"
    "**Objective:** Build a simple, transparent reliability layer that determines whether a"
    " model prediction should be **ACCEPT**ed or **ABSTAIN**ed from.\n\n"
    "### Key components\n"
    "| # | Component | Method |\n"
    "|---|---|---|\n"
    "| 1 | Uncertainty | Bootstrap ensemble (30 \u00d7 Logistic Regression on preprocessed features) |\n"
    "| 2 | Data Quality | Weighted composite score: lab breadth + vital breadth + completeness + flags |\n"
    "| 3 | Decision | Four strategies compared; thresholds selected on validation only |\n"
    "| 4 | Evaluation | Coverage, abstention rate, error rate, AUROC, PR-AUC, Brier |\n"
    "| 5 | Report | Individual patient reliability card with SHAP top contributors |\n\n"
    "> **Important:** `ABSTAIN` means *\"the reliability criteria were not met.\"*  \n"
    "> It does **NOT** mean the patient is safe or high-risk."
))

# ─── Setup ───────────────────────────────────────────────────────────────────
CELLS.append(code_cell(
    "import sys\n"
    "from pathlib import Path\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import polars as pl\n"
    "import matplotlib.pyplot as plt\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n\n"
    "# Ensure project root is on path\n"
    "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
    "if str(PROJECT_ROOT) not in sys.path:\n"
    "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
    "from src.safepredict import (\n"
    "    compute_bootstrap_uncertainty,\n"
    "    compute_data_quality_scores,\n"
    "    SafePredictConfig,\n"
    "    apply_strategies,\n"
    "    select_thresholds_on_validation,\n"
    "    evaluate_all_strategies,\n"
    "    compute_risk_vs_coverage,\n"
    "    plot_risk_vs_coverage,\n"
    "    generate_patient_report,\n"
    "    format_strategy_table,\n"
    "    run_phase_8_safepredict,\n"
    "    QUALITY_FLAGS_PATH,\n"
    "    N_BOOTSTRAPS,\n"
    "    MIN_COVERAGE,\n"
    ")\n"
    "from src.model import (\n"
    "    prepare_model_splits,\n"
    "    build_preprocessor,\n"
    "    RANDOM_STATE,\n"
    "    MODELS_DIR,\n"
    "    FIGURES_DIR,\n"
    "    METRICS_DIR,\n"
    "    PROCESSED_DATA_DIR,\n"
    "    IDENTIFIER_COLS,\n"
    "    TARGET_COL,\n"
    ")\n"
    "from src.explain import (\n"
    "    load_champion_artifacts,\n"
    "    extract_base_classifier,\n"
    "    compute_tree_shap_explanations,\n"
    "    select_clinical_case_studies,\n"
    "    extract_patient_feature_contributions,\n"
    "    get_test_cohort_metadata,\n"
    ")\n"
    "from sklearn.model_selection import StratifiedGroupKFold\n"
    "from sklearn.metrics import roc_auc_score\n"
    "import joblib\n\n"
    "print('All Phase 8 imports successful.')"
))

# ─── Run full pipeline ────────────────────────────────────────────────────────
CELLS.append(md_cell(
    "### 8.1 Run Phase 8 End-to-End Pipeline\n\n"
    "This single call executes all steps deterministically:\n"
    "1. Load splits + champion model\n"
    "2. Bootstrap uncertainty (30 resamples of LR on preprocessed features)\n"
    "3. Data quality scores (lab breadth, vital breadth, completeness, flags)\n"
    "4. Threshold selection on **validation set only** \u2014 test never consulted\n"
    "5. Evaluate four strategies on held-out test set\n"
    "6. Risk-vs-Coverage sweep and plot\n"
    "7. Save results JSON"
))

CELLS.append(code_cell(
    "p8 = run_phase_8_safepredict(\n"
    "    random_state=RANDOM_STATE,\n"
    "    n_bootstraps=N_BOOTSTRAPS,   # 30 bootstraps\n"
    "    min_coverage=MIN_COVERAGE,   # >= 60% coverage constraint on val\n"
    ")"
))

# ─── Unpack & distributions ───────────────────────────────────────────────────
CELLS.append(md_cell(
    "### 8.2 Uncertainty & Data Quality Distributions\n\n"
    "Examine the bootstrap std dev and DQ score distributions across the val and test sets,"
    " and visualise their relationship to model predictions."
))

CELLS.append(code_cell(
    "config = p8['config']\n"
    "y_test   = p8['y_test']\n"
    "y_val    = p8['y_val']\n"
    "test_probs     = p8['test_probs']\n"
    "val_probs      = p8['val_probs']\n"
    "test_boot_std  = p8['test_boot_std']\n"
    "val_boot_std   = p8['val_boot_std']\n"
    "test_boot_mean = p8['test_boot_mean']\n"
    "val_boot_mean  = p8['val_boot_mean']\n"
    "test_dq = p8['test_dq']\n"
    "val_dq  = p8['val_dq']\n"
    "test_results  = p8['test_results']\n"
    "sweep_df      = p8['sweep_df']\n"
    "model_df      = p8['model_df']\n"
    "test_stay_ids = p8['test_stay_ids']\n"
    "val_stay_ids  = p8['val_stay_ids']\n\n"
    "fig, axes = plt.subplots(2, 2, figsize=(13, 8))\n"
    "fig.suptitle('Phase 8 \u2014 Uncertainty & Data Quality Distributions', fontsize=13, fontweight='bold')\n"
    "cv, ct = '#2b5c8f', '#c0392b'\n\n"
    "ax = axes[0, 0]\n"
    "ax.hist(val_boot_std, bins=30, alpha=0.65, color=cv, label=f'Val (n={len(val_boot_std)})', edgecolor='white')\n"
    "ax.hist(test_boot_std, bins=30, alpha=0.65, color=ct, label=f'Test (n={len(test_boot_std)})', edgecolor='white')\n"
    "ax.axvline(config.uncertainty_threshold, color='black', linestyle='--', lw=1.8, label=f'Threshold = {config.uncertainty_threshold:.4f}')\n"
    "ax.set_xlabel('Bootstrap Std Dev (Prediction Variability)'); ax.set_ylabel('Count')\n"
    "ax.set_title('Uncertainty Distribution', fontweight='bold'); ax.legend(fontsize=8); ax.grid(True, linestyle=':', alpha=0.5)\n\n"
    "ax = axes[0, 1]\n"
    "ax.hist(val_dq, bins=30, alpha=0.65, color=cv, label=f'Val (n={len(val_dq)})', edgecolor='white')\n"
    "ax.hist(test_dq, bins=30, alpha=0.65, color=ct, label=f'Test (n={len(test_dq)})', edgecolor='white')\n"
    "ax.axvline(config.dq_threshold, color='black', linestyle='--', lw=1.8, label=f'Threshold = {config.dq_threshold:.4f}')\n"
    "ax.set_xlabel('Data Quality Score'); ax.set_ylabel('Count')\n"
    "ax.set_title('DQ Score Distribution', fontweight='bold'); ax.legend(fontsize=8); ax.grid(True, linestyle=':', alpha=0.5)\n\n"
    "ax = axes[1, 0]\n"
    "sc = ax.scatter(test_boot_std, test_probs, c=y_test, cmap='RdYlGn_r', alpha=0.55, s=20)\n"
    "ax.axvline(config.uncertainty_threshold, color='black', linestyle='--', lw=1.5)\n"
    "ax.set_xlabel('Bootstrap Std Dev (Uncertainty)'); ax.set_ylabel('Calibrated Probability')\n"
    "ax.set_title('Uncertainty vs P(mortality) \\u2014 Test\\n(red=died, green=survived)', fontweight='bold')\n"
    "ax.grid(True, linestyle=':', alpha=0.5)\n\n"
    "ax = axes[1, 1]\n"
    "sc2 = ax.scatter(test_dq, test_probs, c=y_test, cmap='RdYlGn_r', alpha=0.55, s=20)\n"
    "ax.axvline(config.dq_threshold, color='black', linestyle='--', lw=1.5)\n"
    "ax.set_xlabel('Data Quality Score'); ax.set_ylabel('Calibrated Probability')\n"
    "ax.set_title('DQ Score vs P(mortality) \\u2014 Test\\n(red=died, green=survived)', fontweight='bold')\n"
    "ax.grid(True, linestyle=':', alpha=0.5)\n"
    "plt.colorbar(sc2, ax=ax, label='Outcome (1=died)')\n\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIGURES_DIR / 'safepredict_distributions.png', dpi=200, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Saved: reports/figures/safepredict_distributions.png')"
))

# ─── Threshold sweep detail ───────────────────────────────────────────────────
CELLS.append(md_cell(
    "### 8.3 Threshold Selection Details (Validation Set)\n\n"
    "All threshold combinations evaluated on the validation set. **Thresholds are frozen here \u2014"
    " the test set was never consulted.**"
))

CELLS.append(code_cell(
    "sweep_eligible = pd.DataFrame([\n"
    "    s for s in config.sweep_details if s.get('eligible', False)\n"
    "]).sort_values('auroc_accepted', ascending=False)\n\n"
    "print(f'Selected thresholds (frozen):')\n"
    "print(f'  Uncertainty threshold : {config.uncertainty_threshold:.4f}')\n"
    "print(f'  DQ threshold          : {config.dq_threshold:.4f}')\n"
    "print(f'  Val coverage achieved : {config.val_coverage:.1%}')\n"
    "print(f'  Val AUROC (accepted)  : {config.val_auroc_accepted:.4f}')\n"
    "print(f'  Criterion             : {config.selection_criterion}')\n"
    "print()\n"
    "if len(sweep_eligible) > 0:\n"
    "    print(f'Eligible combinations (coverage >= {MIN_COVERAGE:.0%}) on validation:')\n"
    "    print(sweep_eligible.to_string(index=False))\n"
    "else:\n"
    "    print('No eligible combinations found; fallback thresholds were applied.')"
))

# ─── Strategy comparison ──────────────────────────────────────────────────────
CELLS.append(md_cell(
    "### 8.4 Strategy Comparison \u2014 Held-Out Test Set\n\n"
    "| Strategy | ACCEPT criteria |\n"
    "|---|---|\n"
    "| Model Only | Always accept |\n"
    "| Uncertainty Abstain | `bootstrap_std < thresh_u` |\n"
    "| DQ Score Abstain | `dq_score >= thresh_dq` |\n"
    "| SafePredict Combined | Both criteria must be met |"
))

CELLS.append(code_cell(
    "print(format_strategy_table(test_results, 'Test'))\n\n"
    "strategy_labels = {\n"
    "    'model_only': 'Model Only',\n"
    "    'uncertainty_abstain': 'Uncertainty\\nAbstain',\n"
    "    'dq_abstain': 'DQ Score\\nAbstain',\n"
    "    'safepredict_combined': 'SafePredict\\nCombined',\n"
    "}\n"
    "strat_names = list(test_results.keys())\n"
    "auROCs    = [test_results[s]['auroc_accepted'] or 0.0 for s in strat_names]\n"
    "coverages = [test_results[s]['coverage'] for s in strat_names]\n"
    "briers    = [test_results[s]['brier_accepted'] or 0.0 for s in strat_names]\n"
    "x = np.arange(len(strat_names))\n"
    "colors_bar = ['#95a5a6', '#e67e22', '#27ae60', '#8e44ad']\n\n"
    "fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))\n"
    "fig.suptitle('SafePredict Strategy Comparison \u2014 Held-Out Test Set', fontsize=12, fontweight='bold')\n\n"
    "for ax, vals, title, ylabel, ylim_fn, fmt_fn in [\n"
    "    (axes[0], auROCs, 'AUROC among Accepted Patients', 'AUROC',\n"
    "     lambda v: (0.5, min(1.0, max(v) + 0.05)), lambda v: f'{v:.3f}'),\n"
    "    (axes[1], [c * 100 for c in coverages], 'Coverage (% Accepted)', 'Coverage (%)',\n"
    "     lambda v: (0, 110), lambda v: f'{v/100:.0%}'),\n"
    "    (axes[2], briers, 'Brier Score among Accepted', 'Brier (lower=better)',\n"
    "     lambda v: (0, max(v) + 0.01), lambda v: f'{v:.4f}'),\n"
    "]:\n"
    "    bars = ax.bar(x, vals, color=colors_bar, edgecolor='white', width=0.6)\n"
    "    ax.set_xticks(x)\n"
    "    ax.set_xticklabels([strategy_labels[s] for s in strat_names], fontsize=8)\n"
    "    ax.set_ylabel(ylabel, fontsize=10)\n"
    "    ax.set_title(title, fontsize=10, fontweight='bold')\n"
    "    ax.set_ylim(*ylim_fn(vals))\n"
    "    ax.grid(True, axis='y', linestyle=':', alpha=0.5)\n"
    "    for bar, v in zip(bars, vals):\n"
    "        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,\n"
    "                fmt_fn(v), ha='center', fontsize=8)\n\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIGURES_DIR / 'safepredict_strategy_comparison.png', dpi=200, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Saved: reports/figures/safepredict_strategy_comparison.png')"
))

# ─── Risk vs Coverage ─────────────────────────────────────────────────────────
CELLS.append(md_cell(
    "### 8.5 Risk vs Coverage Analysis\n\n"
    "As the uncertainty threshold is relaxed (more patients accepted), coverage increases.\n"
    "The DQ threshold is held at the **median DQ score** during this sweep."
))

CELLS.append(code_cell(
    "from IPython.display import Image, display\n"
    "risk_cov_path = FIGURES_DIR / 'safepredict_risk_coverage.png'\n"
    "print(f'Risk vs Coverage plot: {risk_cov_path}')\n"
    "display(Image(filename=str(risk_cov_path)))\n\n"
    "print('\\nKey points from Risk-vs-Coverage sweep (Test set):')\n"
    "print(f'{\"Coverage\":>10} | {\"N Accepted\":>11} | {\"AUROC\":>8} | {\"PR-AUC\":>8} | {\"Brier\":>8}')\n"
    "print('-' * 55)\n"
    "target_covs = [0.60, 0.75, 0.90, 1.00]\n"
    "shown = set()\n"
    "for tc in target_covs:\n"
    "    closest = sweep_df.iloc[(sweep_df['coverage'] - tc).abs().argsort()[:1]]\n"
    "    for _, row in closest.iterrows():\n"
    "        key = round(row['coverage'], 2)\n"
    "        if key in shown: continue\n"
    "        shown.add(key)\n"
    "        a = f\"{row['auroc_accepted']:.3f}\" if not pd.isna(row['auroc_accepted']) else '   N/A'\n"
    "        p = f\"{row['pr_auc_accepted']:.3f}\" if not pd.isna(row['pr_auc_accepted']) else '   N/A'\n"
    "        b = f\"{row['brier_accepted']:.4f}\" if not pd.isna(row['brier_accepted']) else '   N/A'\n"
    "        print(f\"{row['coverage']:>10.1%} | {int(row['n_accepted']):>11} | {a:>8} | {p:>8} | {b:>8}\")"
))

# ─── Patient reports ──────────────────────────────────────────────────────────
CELLS.append(md_cell(
    "### 8.6 Individual Patient Reliability Reports\n\n"
    "For each of the four SHAP case study patients (TP, TN, FP, FN), generate a complete"
    " SafePredict reliability report combining:\n"
    "- Mortality probability (calibrated champion model)\n"
    "- Prediction uncertainty (bootstrap ensemble std dev)\n"
    "- Data quality score and its components\n"
    "- ACCEPT / ABSTAIN decision with reason\n"
    "- Top SHAP contributors\n\n"
    "> SHAP values explain model behavior \u2014 not clinical causation."
))

CELLS.append(code_cell(
    "# Reload Phase 7 SHAP artifacts\n"
    "split_data = prepare_model_splits(random_state=RANDOM_STATE)\n"
    "X_test_r = split_data['X_test']\n"
    "y_test_r = split_data['y_test']\n\n"
    "calibrated_model, preprocessor_r = load_champion_artifacts()\n"
    "base_clf = extract_base_classifier(calibrated_model)\n"
    "test_probs_r = calibrated_model.predict_proba(X_test_r)[:, 1]\n\n"
    "print('Computing TreeSHAP explanations (test cohort)...')\n"
    "shap_expl, X_test_trans, feat_names = compute_tree_shap_explanations(base_clf, preprocessor_r, X_test_r)\n"
    "print(f'SHAP computed: {shap_expl.shape}')\n\n"
    "meta_test = get_test_cohort_metadata(random_state=RANDOM_STATE)\n"
    "cases = select_clinical_case_studies(\n"
    "    X_test=X_test_r, y_test=y_test_r, y_probs=test_probs_r,\n"
    "    meta_test=meta_test, youden_threshold=0.058,\n"
    ")\n\n"
    "combined_mask = apply_strategies(test_boot_std, test_dq, config)['safepredict_combined']\n"
    "model_pdf = model_df.to_pandas().set_index('patientunitstayid')\n\n"
    "for case_key, case_data in cases.items():\n"
    "    idx = case_data['index']\n"
    "    stay_id = test_stay_ids[idx]\n"
    "    contribs = extract_patient_feature_contributions(shap_expl, idx, top_k=4)\n"
    "    dq_row = model_pdf.loc[stay_id] if stay_id in model_pdf.index else None\n"
    "    dq_components = {\n"
    "        'available_lab_count': int(dq_row['available_lab_count']) if dq_row is not None and 'available_lab_count' in model_pdf.columns else 'N/A',\n"
    "        'available_vital_count': int(dq_row['available_vital_count']) if dq_row is not None and 'available_vital_count' in model_pdf.columns else 'N/A',\n"
    "        'missing_feature_count': int(dq_row['missing_feature_count']) if dq_row is not None and 'missing_feature_count' in model_pdf.columns else 'N/A',\n"
    "        'max_labs': 15, 'max_vitals': 7,\n"
    "    }\n"
    "    report = generate_patient_report(\n"
    "        patient_idx=idx,\n"
    "        y_prob_calibrated=float(test_probs_r[idx]),\n"
    "        uncertainty_mean=float(test_boot_mean[idx]),\n"
    "        uncertainty_std=float(test_boot_std[idx]),\n"
    "        dq_score=float(test_dq[idx]),\n"
    "        dq_components=dq_components,\n"
    "        accept_combined=bool(combined_mask[idx]),\n"
    "        shap_contributions=contribs,\n"
    "        case_metadata={\n"
    "            'age': case_data.get('age', 'N/A'),\n"
    "            'unit_type': case_data.get('unit_type', 'N/A'),\n"
    "            'uniquepid': case_data.get('uniquepid', 'N/A'),\n"
    "            'patientunitstayid': case_data.get('patientunitstayid', 'N/A'),\n"
    "            'actual_outcome_label': case_data.get('actual_outcome_label', 'N/A'),\n"
    "        },\n"
    "        uncertainty_thresh=config.uncertainty_threshold,\n"
    "        dq_thresh=config.dq_threshold,\n"
    "    )\n"
    "    print(f\"\\n{'='*68}\")\n"
    "    print(f\"CASE: {case_data['title']}\")\n"
    "    print(f\"{'='*68}\")\n"
    "    print(report)"
))

# ─── Validation assertions ────────────────────────────────────────────────────
CELLS.append(md_cell("### 8.7 Validation Assertions"))

CELLS.append(code_cell(
    "print('Running Phase 8 validation assertions...')\n"
    "failures = []\n\n"
    "# 1. Coverage sanity\n"
    "combined_cov = test_results['safepredict_combined']['coverage']\n"
    "assert test_results['model_only']['coverage'] == 1.0, 'model_only must have 100% coverage'\n"
    "if combined_cov <= 0.0:\n"
    "    failures.append('SafePredict combined: 0% coverage')\n"
    "if combined_cov >= 1.0:\n"
    "    failures.append('SafePredict combined: 100% coverage (no abstentions)')\n"
    "print(f'  [OK] Coverage: model_only=100%, combined={combined_cov:.1%}')\n\n"
    "# 2. ABSTAIN patients have higher uncertainty or lower DQ\n"
    "accept_mask = apply_strategies(test_boot_std, test_dq, config)['safepredict_combined']\n"
    "abstain_mask = ~accept_mask\n"
    "if abstain_mask.sum() > 0 and accept_mask.sum() > 0:\n"
    "    u_abs = test_boot_std[abstain_mask].mean()\n"
    "    u_acc = test_boot_std[accept_mask].mean()\n"
    "    dq_abs = test_dq[abstain_mask].mean()\n"
    "    dq_acc = test_dq[accept_mask].mean()\n"
    "    print(f'  [OK] Mean uncertainty std: ABSTAIN={u_abs:.4f}  ACCEPT={u_acc:.4f}')\n"
    "    print(f'  [OK] Mean DQ score:        ABSTAIN={dq_abs:.3f}   ACCEPT={dq_acc:.3f}')\n"
    "    assert u_abs >= u_acc or dq_abs <= dq_acc, 'ABSTAIN patients should have higher uncertainty or lower DQ'\n"
    "    print('  [OK] ABSTAIN patients have higher uncertainty or lower DQ than ACCEPT')\n\n"
    "# 3. Deterministic threshold selection (no leakage)\n"
    "config2 = select_thresholds_on_validation(\n"
    "    val_uncertainty_std=val_boot_std, val_dq_scores=val_dq,\n"
    "    val_y=y_val, val_probs=val_probs, min_coverage=MIN_COVERAGE,\n"
    ")\n"
    "assert abs(config2.uncertainty_threshold - config.uncertainty_threshold) < 1e-9\n"
    "assert abs(config2.dq_threshold - config.dq_threshold) < 1e-9\n"
    "print('  [OK] Threshold selection is deterministic')\n\n"
    "# 4. Output files\n"
    "import pathlib, json as _json\n"
    "rp = pathlib.Path(p8['results_json_path'])\n"
    "assert rp.exists()\n"
    "assert _json.load(open(rp))['phase'] == 8\n"
    "print(f'  [OK] Results JSON: {rp}')\n"
    "pp = pathlib.Path(p8['plot_path'])\n"
    "assert pp.exists()\n"
    "print(f'  [OK] Risk-vs-Coverage plot: {pp}')\n\n"
    "if failures:\n"
    "    print('\\n[WARNINGS]')\n"
    "    for f in failures: print(f'  WARNING: {f}')\n"
    "else:\n"
    "    print('\\nAll Phase 8 validation assertions passed. OK')\n"
    "    print('\\nPhase 8 outputs:')\n"
    "    print('  reports/metrics/safepredict_results.json')\n"
    "    print('  reports/figures/safepredict_risk_coverage.png')\n"
    "    print('  reports/figures/safepredict_distributions.png')\n"
    "    print('  reports/figures/safepredict_strategy_comparison.png')"
))

# ─── Summary markdown ─────────────────────────────────────────────────────────
CELLS.append(md_cell(
    "---\n\n"
    "### Phase 8 Complete\n\n"
    "| Artifact | Location |\n"
    "|---|---|\n"
    "| Results JSON | `reports/metrics/safepredict_results.json` |\n"
    "| Risk vs Coverage plot | `reports/figures/safepredict_risk_coverage.png` |\n"
    "| Distribution plot | `reports/figures/safepredict_distributions.png` |\n"
    "| Strategy comparison | `reports/figures/safepredict_strategy_comparison.png` |\n"
    "| Module | `src/safepredict.py` |\n\n"
    "> **Clinical guardrail:** SafePredict decisions are based on model reliability signals, "
    "not clinical judgment. `ABSTAIN` means the model's prediction should not be used as-is "
    "for this patient. Clinical decision-making always requires a qualified clinician."
))

# ---------------------------------------------------------------------------
# Append to notebook
# ---------------------------------------------------------------------------
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

original_count = len(nb["cells"])
nb["cells"].extend(CELLS)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Appended {len(CELLS)} cells to {NB_PATH}")
print(f"Total cells: {original_count} -> {len(nb['cells'])}")
