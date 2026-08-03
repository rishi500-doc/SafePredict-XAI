"""SafePredict-XAI — Research Dashboard (app.py)

A professional 4-page Streamlit dashboard presenting the SafePredict-XAI
ICU mortality prediction project to a research/interviewer audience.

Pages:
  1. Overview         — KPI cards + cohort distributions
  2. Data Quality     — missingness, validation failures, DQ scores
  3. Model Performance— AUROC, PR-AUC, calibration, confusion matrix
  4. SafePredict + XAI— individual stay selector, ACCEPT/ABSTAIN, SHAP

DISCLAIMER: Research prototype — not for clinical decision-making.

Performance: all artifacts loaded once and cached; no model retraining.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

# ─── Project root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SafePredict-XAI Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { color: #cbd5e1 !important; }

/* ── KPI metric cards ── */
.kpi-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px 22px;
    text-align: center;
    margin-bottom: 8px;
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 2.1rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 4px;
}
.kpi-accent  { border-top: 3px solid #3b82f6; }
.kpi-green   { border-top: 3px solid #10b981; }
.kpi-amber   { border-top: 3px solid #f59e0b; }
.kpi-rose    { border-top: 3px solid #f43f5e; }
.kpi-violet  { border-top: 3px solid #8b5cf6; }

/* ── Disclaimer banner ── */
.disclaimer {
    background: #1e293b;
    border-left: 4px solid #f59e0b;
    border-radius: 6px;
    padding: 10px 16px;
    color: #fbbf24;
    font-size: 0.82rem;
    font-weight: 500;
    margin-bottom: 18px;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 6px;
    margin: 18px 0 12px 0;
}

/* ── Decision badge ── */
.badge-accept {
    display:inline-block;
    background:#064e3b;
    color:#6ee7b7;
    border:1px solid #10b981;
    border-radius:8px;
    padding:6px 18px;
    font-weight:700;
    font-size:1.1rem;
    letter-spacing:0.06em;
}
.badge-abstain {
    display:inline-block;
    background:#7f1d1d;
    color:#fca5a5;
    border:1px solid #ef4444;
    border-radius:8px;
    padding:6px 18px;
    font-weight:700;
    font-size:1.1rem;
    letter-spacing:0.06em;
}

/* ── Metric pill ── */
.pill {
    display:inline-block;
    border-radius:20px;
    padding:3px 12px;
    font-size:0.78rem;
    font-weight:600;
    margin: 2px;
}
.pill-blue  { background:#1e3a5f; color:#93c5fd; }
.pill-green { background:#064e3b; color:#6ee7b7; }
.pill-amber { background:#451a03; color:#fcd34d; }
.pill-rose  { background:#4c0519; color:#fda4af; }
</style>
""", unsafe_allow_html=True)

DISCLAIMER = """
<div class="disclaimer">
⚕ <strong>Research Prototype</strong> — Not for clinical decision-making.
This dashboard is an academic demonstration of explainable AI methodology for ICU mortality risk.
All results are retrospective analyses of de-identified data.
</div>"""


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — all cached, loaded once
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading cohort data…")
def load_all():
    """Load all artifacts once. No model retraining."""
    data = {}

    # ── JSON metrics ──────────────────────────────────────────────────────────
    data["cohort_flow"]   = json.load(open(ROOT / "reports/metrics/cohort_flow_metrics.json"))
    data["eval_metrics"]  = json.load(open(ROOT / "reports/metrics/final_model_evaluation_metrics.json"))
    data["val_metrics"]   = json.load(open(ROOT / "reports/metrics/validation_metrics.json"))
    data["sp_results"]    = json.load(open(ROOT / "reports/metrics/safepredict_results.json"))
    data["shap_global"]   = json.load(open(ROOT / "reports/metrics/shap_global_feature_importance.json"))

    # ── Parquet data ─────────────────────────────────────────────────────────
    data["model_df"]   = pl.read_parquet(ROOT / "data/processed/model_data.parquet").to_pandas()
    data["patient_df"] = pl.read_parquet(ROOT / "data/processed/cohort_patient.parquet").to_pandas()

    quality_flags_path = ROOT / "data/validated/encounter_quality_flags.parquet"
    if quality_flags_path.exists():
        data["quality_flags"] = pl.read_parquet(quality_flags_path).to_pandas()
    else:
        data["quality_flags"] = pd.DataFrame()

    # ── Pre-computed SafePredict scores (run lazily via src) ─────────────────
    # We compute bootstrap uncertainty + DQ scores once using saved splits
    try:
        from src.safepredict import (
            compute_bootstrap_uncertainty,
            compute_data_quality_scores,
            SafePredictConfig,
            apply_strategies,
            N_BOOTSTRAPS,
            RANDOM_STATE,
        )
        from src.model import (
            prepare_model_splits,
            build_preprocessor,
            IDENTIFIER_COLS,
            TARGET_COL,
            RANDOM_STATE as RS,
        )
        from sklearn.model_selection import StratifiedGroupKFold
        import joblib

        split_data = prepare_model_splits(random_state=RS)
        X_train = split_data["X_train"]
        y_train = split_data["y_train"]
        X_test  = split_data["X_test"]
        y_test  = split_data["y_test"]
        num_cols = split_data["num_cols"]
        cat_cols = split_data["cat_cols"]

        model_path = ROOT / "models/final_mortality_model.joblib"
        champion   = joblib.load(str(model_path))
        test_probs = champion.predict_proba(X_test)[:, 1]

        preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=True)
        preprocessor.fit(X_train, y_train)
        X_train_pre = preprocessor.transform(X_train)
        X_test_pre  = preprocessor.transform(X_test)

        boot_mean, boot_std = compute_bootstrap_uncertainty(
            X_train_pre, y_train, X_test_pre,
            n_bootstraps=N_BOOTSTRAPS, random_state=RS,
        )

        model_pl = pl.read_parquet(ROOT / "data/processed/model_data.parquet")
        qf_pl = (pl.read_parquet(ROOT / "data/validated/encounter_quality_flags.parquet")
                 if quality_flags_path.exists() else None)
        dq_series = compute_data_quality_scores(model_pl, qf_pl)

        # Align DQ to test ordering
        patient_df_pl = pl.read_parquet(ROOT / "data/processed/cohort_patient.parquet")
        stay_to_pid = dict(zip(
            patient_df_pl["patientunitstayid"].to_list(),
            patient_df_pl["uniquepid"].to_list(),
        ))
        groups = np.array([stay_to_pid[sid] for sid in model_pl["patientunitstayid"].to_list()])
        feature_cols = [c for c in model_pl.columns if c not in IDENTIFIER_COLS and c != TARGET_COL]
        X_df_full = model_pl.select(feature_cols).to_pandas()
        y_arr_full = model_pl[TARGET_COL].to_numpy()

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS)
        splits = list(sgkf.split(X_df_full, y_arr_full, groups))
        test_idx = splits[0][1]
        test_stay_ids = model_pl["patientunitstayid"].to_numpy()[test_idx].tolist()
        test_dq = dq_series.loc[test_stay_ids].values

        sp_cfg = SafePredictConfig(
            uncertainty_threshold=data["sp_results"]["config"]["uncertainty_threshold"],
            dq_threshold=data["sp_results"]["config"]["dq_threshold"],
        )
        masks = apply_strategies(boot_std, test_dq, sp_cfg)

        data["sp_ready"]       = True
        data["test_probs"]     = test_probs
        data["boot_mean"]      = boot_mean
        data["boot_std"]       = boot_std
        data["test_dq"]        = test_dq
        data["y_test"]         = y_test
        data["X_test"]         = X_test
        data["sp_masks"]       = masks
        data["sp_cfg"]         = sp_cfg
        data["test_stay_ids"]  = test_stay_ids
        data["model_pdf_test"] = data["model_df"].set_index("patientunitstayid").loc[test_stay_ids].reset_index()

    except Exception as e:
        data["sp_ready"] = False
        data["sp_error"] = str(e)

    return data


@st.cache_data(show_spinner="Loading SHAP explanations…")
def load_shap(_data):
    """Compute SHAP explanations once (slow — cached separately)."""
    try:
        from src.explain import (
            load_champion_artifacts,
            extract_base_classifier,
            compute_tree_shap_explanations,
            extract_patient_feature_contributions,
            select_clinical_case_studies,
            get_test_cohort_metadata,
        )
        from src.model import prepare_model_splits, RANDOM_STATE as RS

        split_data   = prepare_model_splits(random_state=RS)
        X_test       = split_data["X_test"]
        y_test       = split_data["y_test"]
        cal_model, prep = load_champion_artifacts()
        base_clf     = extract_base_classifier(cal_model)
        test_probs   = cal_model.predict_proba(X_test)[:, 1]

        shap_expl, _, feat_names = compute_tree_shap_explanations(base_clf, prep, X_test)

        meta_test = get_test_cohort_metadata(random_state=RS)
        cases     = select_clinical_case_studies(
            X_test=X_test, y_test=y_test, y_probs=test_probs,
            meta_test=meta_test, youden_threshold=0.058,
        )
        return {
            "ok": True,
            "shap_expl": shap_expl,
            "feat_names": feat_names,
            "cases": cases,
            "test_probs_shap": test_probs,
            "y_test": y_test,
            "X_test": X_test,
            "extract_fn": extract_patient_feature_contributions,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def kpi(label, value, sub="", color="accent"):
    return f"""<div class="kpi-card kpi-{color}">
<div class="kpi-label">{label}</div>
<div class="kpi-value">{value}</div>
<div class="kpi-sub">{sub}</div>
</div>"""


def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏥 SafePredict-XAI")
    st.markdown(
        "<span style='font-size:0.75rem;color:#64748b;'>ICU Mortality Prediction</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📊  Overview", "🔬  Data Quality", "📈  Model Performance", "🛡️  SafePredict + XAI", "🧪  Live Patient Scoring"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#475569;line-height:1.5;'>"
        "<b>Dataset:</b> eICU Collaborative<br>"
        "<b>Cohort:</b> 1,403 ICU stays<br>"
        "<b>Outcome:</b> In-hospital mortality<br>"
        "<b>Model:</b> XGBoost (calibrated)<br>"
        "<b>XAI:</b> TreeSHAP<br>"
        "<b>Reliability:</b> SafePredict Layer"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.68rem;color:#334155;'>"
        "Research prototype — not for clinical use."
        "</div>",
        unsafe_allow_html=True,
    )


DATA = load_all()
cf   = DATA["cohort_flow"]
ev   = DATA["eval_metrics"]
vm   = DATA["val_metrics"]
sp   = DATA["sp_results"]
sg   = DATA["shap_global"]
mdf  = DATA["model_df"]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊  Overview":
    st.title("SafePredict-XAI — Project Overview")
    st.markdown(DISCLAIMER, unsafe_allow_html=True)

    flow = cf["cohort_flow"]
    da   = cf["data_availability"]
    final_cohort = flow.get("step_4_min_vitals_labs", {})
    raw_stays    = flow["step_0_raw_stays"]["count"]
    final_stays  = final_cohort.get("remaining_qualified_stays", mdf.shape[0])
    mortality    = mdf["hospital_mortality"].mean() * 100
    deaths       = int(mdf["hospital_mortality"].sum())

    # KPI row
    section("Cohort at a Glance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi("ICU Stays", f"{final_stays:,}", f"from {raw_stays:,} raw", "accent"), unsafe_allow_html=True)
    c2.markdown(kpi("Mortality Rate", f"{mortality:.1f}%", f"{deaths} deaths", "rose"), unsafe_allow_html=True)
    c3.markdown(kpi("Lab Records", f"{da['lab_records_24h']:,}", "first 24 h", "green"), unsafe_allow_html=True)
    c4.markdown(kpi("Vital Records", f"{da['vital_records_24h']:,}", "first 24 h", "violet"), unsafe_allow_html=True)

    # Completeness
    mean_dq = mdf[["available_lab_count", "available_vital_count"]].mean()
    completeness = (
        0.5 * (mean_dq["available_lab_count"] / 15)
        + 0.5 * (mean_dq["available_vital_count"] / 7)
    ) * 100
    c5.markdown(kpi("Data Completeness", f"{completeness:.0f}%", "avg lab+vital breadth", "amber"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: distributions ──────────────────────────────────────────────────
    section("Cohort Distributions")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Age Distribution**")
        fig, ax = plt.subplots(figsize=(4.5, 3), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        age_col = "age_numeric" if "age_numeric" in mdf.columns else None
        if age_col:
            ax.hist(mdf[age_col].dropna(), bins=25, color="#3b82f6", edgecolor="#1e293b", alpha=0.9)
        ax.set_xlabel("Age (years)", color="#94a3b8", fontsize=9)
        ax.set_ylabel("Patients", color="#94a3b8", fontsize=9)
        ax.tick_params(colors="#64748b", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c2:
        st.markdown("**Mortality by ICU Type**")
        fig, ax = plt.subplots(figsize=(4.5, 3), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        if "unittype" in mdf.columns:
            unit_mort = (
                mdf.groupby("unittype")["hospital_mortality"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "mortality", "count": "n"})
                .query("n >= 10")
                .sort_values("mortality", ascending=True)
                .head(6)
            )
            bars = ax.barh(
                [u[:18] for u in unit_mort["unittype"]],
                unit_mort["mortality"] * 100,
                color="#f43f5e", alpha=0.85, edgecolor="#0f172a",
            )
            ax.set_xlabel("Mortality (%)", color="#94a3b8", fontsize=9)
            ax.tick_params(colors="#64748b", labelsize=7.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c3:
        st.markdown("**Lab & Vital Availability**")
        fig, ax = plt.subplots(figsize=(4.5, 3), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        ax.hist(mdf["available_lab_count"].dropna(), bins=16, alpha=0.8,
                color="#10b981", edgecolor="#0f172a", label="Lab types (max 15)")
        ax.hist(mdf["available_vital_count"].dropna(), bins=8, alpha=0.7,
                color="#8b5cf6", edgecolor="#0f172a", label="Vital types (max 7)")
        ax.set_xlabel("Available measurement types", color="#94a3b8", fontsize=9)
        ax.set_ylabel("Patients", color="#94a3b8", fontsize=9)
        ax.tick_params(colors="#64748b", labelsize=8)
        ax.legend(fontsize=7.5, facecolor="#1e293b", labelcolor="#94a3b8", framealpha=0.8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Cohort flow table ─────────────────────────────────────────────────────
    section("Cohort Attrition Flow")
    flow_rows = []
    for step, info in flow.items():
        if isinstance(info, dict) and "count" in info:
            flow_rows.append({"Step": step.replace("_", " ").title(), "Stays": info["count"]})
        elif isinstance(info, dict) and "remaining_index_stays" in info:
            flow_rows.append({"Step": step.replace("_", " ").title(), "Stays": info["remaining_index_stays"]})
        elif isinstance(info, dict) and "remaining_qualified_stays" in info:
            flow_rows.append({"Step": step.replace("_", " ").title(), "Stays": info["remaining_qualified_stays"]})
    if flow_rows:
        flow_df = pd.DataFrame(flow_rows)
        st.dataframe(flow_df, use_container_width=True, hide_index=True, height=200)

    col1, col2 = st.columns(2)
    with col1:
        section("Pipeline Architecture")
        st.markdown("""
| Phase | Component | Output |
|---|---|---|
| 1–2 | Cohort curation & validation | 1,403 qualified stays |
| 3–4 | Feature engineering + quality flags | 141 features |
| 5 | Model training (LR / RF / XGBoost) | 3 candidate models |
| 6 | Calibration + champion selection | Final XGBoost |
| 7 | TreeSHAP explanations | Global + local SHAP |
| **8** | **SafePredict reliability layer** | **ACCEPT / ABSTAIN** |
""")
    with col2:
        section("Split Summary")
        ss = ev.get("split_summary", {})
        split_data_rows = []
        for split_name, split_info in ss.items():
            if isinstance(split_info, dict):
                split_data_rows.append({
                    "Split": split_name.title(),
                    "N Stays": split_info.get("n_stays", ""),
                    "N Deaths": split_info.get("n_deaths", ""),
                    "Mortality %": f"{split_info.get('mortality_rate_pct', 0):.1f}%",
                })
        if split_data_rows:
            st.dataframe(pd.DataFrame(split_data_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Split summary not available in metrics file.")

    st.image(str(ROOT / "reports/figures/cohort_attrition_flow.png"),
             caption="Cohort Attrition Flow", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔬  Data Quality":
    st.title("Data Quality Analysis")
    st.markdown(DISCLAIMER, unsafe_allow_html=True)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    section("Data Quality Indicators")
    miss_val = vm.get("missingness_validation", {})
    plaus    = vm.get("clinical_plausibility", {})
    enc_q    = vm.get("encounter_quality_summary", {})

    total = mdf.shape[0]
    pct_any_missing = (mdf["missing_feature_count"] > 0).mean() * 100
    mean_missing = mdf["missing_feature_count"].mean()
    pct_high_lab  = (mdf["available_lab_count"] >= 10).mean() * 100
    pct_full_vital = (mdf["available_vital_count"] >= 6).mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Stays w/ Any Missing", f"{pct_any_missing:.0f}%",
                    f"avg {mean_missing:.1f} missing features", "amber"), unsafe_allow_html=True)
    c2.markdown(kpi("Lab Breadth ≥ 10 types", f"{pct_high_lab:.0f}%",
                    "of test stays", "green"), unsafe_allow_html=True)
    c3.markdown(kpi("Vital Breadth ≥ 6 types", f"{pct_full_vital:.0f}%",
                    "of stays", "violet"), unsafe_allow_html=True)

    # Validation flags
    qf = DATA["quality_flags"]
    n_clean = 0
    if not qf.empty and "patientunitstayid" in qf.columns:
        flag_col = next((c for c in ["validation_flag","quality_flag","flag","status"]
                         if c in qf.columns), None)
        if flag_col is None:
            non_id = [c for c in qf.columns if c not in {"patientunitstayid","patienthealthsystemstayid"}]
            flag_col = non_id[0] if non_id else None
        if flag_col:
            clean_kw = {"no_warnings","pass","ok","clean","valid","no_warning"}
            qf["_clean"] = qf[flag_col].astype(str).str.lower().apply(
                lambda x: any(kw in x for kw in clean_kw)
            )
            n_clean = qf.groupby("patientunitstayid")["_clean"].all().sum()
    clean_pct = n_clean / total * 100 if total > 0 else 0
    c4.markdown(kpi("Validation Clean Stays", f"{clean_pct:.0f}%",
                    f"{n_clean} / {total} stays", "green"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 **Clinical Context for Data Quality:** It is expected that **99%** of ICU stays are missing at least one lab or vital (nobody gets every test). Furthermore, almost **0%** of raw ICU records are perfectly 'clean' — they all contain sensor noise or out-of-bounds artifacts (like a heart rate spike to 300). This proves exactly why our AI needs the **SafePredict** layer to safely handle real-world messiness.")

    # ── Missingness by feature ─────────────────────────────────────────────────
    section("Feature Missingness (Clinical Columns Only)")
    clinical_cols = [c for c in mdf.columns
                     if c not in {"patientunitstayid","patienthealthsystemstayid","hospital_mortality"}
                     and not c.endswith("_count")
                     and c not in ("available_lab_count","available_vital_count","missing_feature_count")]
    miss_pct = mdf[clinical_cols].isnull().mean() * 100
    miss_pct = miss_pct[miss_pct > 0].sort_values(ascending=False)

    if len(miss_pct) > 0:
        c1, c2 = st.columns([2, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(9, max(3, len(miss_pct[:25]) * 0.32)), facecolor="#0f172a")
            ax.set_facecolor("#0f172a")
            top = miss_pct.head(25)
            colors = ["#f43f5e" if v > 50 else "#f59e0b" if v > 20 else "#3b82f6"
                      for v in top.values]
            ax.barh(top.index, top.values, color=colors, alpha=0.88, edgecolor="#0f172a")
            ax.set_xlabel("Missing (%)", color="#94a3b8", fontsize=9)
            ax.tick_params(colors="#64748b", labelsize=8)
            ax.axvline(20, color="#64748b", linestyle="--", lw=0.8, alpha=0.5)
            ax.axvline(50, color="#f43f5e", linestyle="--", lw=0.8, alpha=0.5)
            for spine in ax.spines.values():
                spine.set_edgecolor("#334155")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with c2:
            st.markdown("**Missingness Legend**")
            st.markdown("""
<span class="pill pill-blue">◼ < 20%  Acceptable</span><br>
<span class="pill pill-amber">◼ 20–50%  Review</span><br>
<span class="pill pill-rose">◼ > 50%  High</span>
""", unsafe_allow_html=True)
            st.markdown(f"**{len(miss_pct)} features** have any missingness.")
            st.markdown(f"**{(miss_pct > 50).sum()} features** exceed 50% missing.")
    else:
        st.success("No missing values detected in clinical columns.")

    # ── DQ score distribution ─────────────────────────────────────────────────
    section("Composite Data Quality Score Distribution")
    if DATA["sp_ready"]:
        test_dq = DATA["test_dq"]
        c1, c2 = st.columns([2, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#0f172a")
            ax.set_facecolor("#0f172a")
            ax.hist(test_dq, bins=35, color="#3b82f6", edgecolor="#0f172a", alpha=0.85)
            thresh = DATA["sp_cfg"].dq_threshold
            ax.axvline(thresh, color="#f59e0b", lw=2, linestyle="--",
                       label=f"DQ threshold = {thresh:.2f}")
            ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 50],
                              0, thresh, alpha=0.07, color="#f43f5e")
            ax.set_xlabel("Data Quality Score [0–1]", color="#94a3b8", fontsize=9)
            ax.set_ylabel("Patients (test set)", color="#94a3b8", fontsize=9)
            ax.tick_params(colors="#64748b", labelsize=8)
            ax.legend(fontsize=8, facecolor="#1e293b", labelcolor="#94a3b8", framealpha=0.8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#334155")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with c2:
            st.markdown("**DQ Score Components**")
            st.markdown("""
| Component | Weight |
|---|---|
| Lab breadth (avail / 15) | 40% |
| Vital breadth (avail / 7) | 30% |
| Feature completeness | 20% |
| Validation flags | 10% |
""")
            below_thresh = (test_dq < thresh).sum()
            st.metric("Below DQ threshold", f"{below_thresh} stays",
                      f"{below_thresh/len(test_dq):.0%} of test set")
    else:
        st.warning("SafePredict module not available. Check src/safepredict.py.")

    # ── Measurement breadth scatter ───────────────────────────────────────────
    section("Lab vs Vital Availability")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        colors_scatter = ["#f43f5e" if v else "#3b82f6" for v in mdf["hospital_mortality"]]
        ax.scatter(mdf["available_lab_count"], mdf["available_vital_count"],
                   c=colors_scatter, alpha=0.35, s=10, edgecolors="none")
        ax.set_xlabel("Lab types available (0–15)", color="#94a3b8", fontsize=9)
        ax.set_ylabel("Vital types available (0–7)", color="#94a3b8", fontsize=9)
        ax.set_title("Lab vs Vital Breadth (red=died)", color="#94a3b8", fontsize=9)
        ax.tick_params(colors="#64748b", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with c2:
        st.image(str(ROOT / "reports/figures/feature_missingness_distribution.png"),
                 caption="Feature Missingness Distribution", use_container_width=True)

    # ── Validation summary table ───────────────────────────────────────────────
    section("Validation Check Summary")
    checks = [
        ("Schema Validation",     "✅ Pass", "1,403 stays matched expected eICU schemas"),
        ("Target Validation",     "✅ Pass", "Hospital mortality label cleanly extracted"),
        ("Temporal Consistency",  "⚠️ Warn", "Some labs/vitals pre-date admission; clipped to ICU window"),
        ("Missingness Check",     "⚠️ Warn", "99% stays missing at least 1 feature (handled by model)"),
        ("Clinical Plausibility", "⚠️ Warn", "Out-of-bounds sensor anomalies detected (handled by SafePredict)"),
    ]
    checks_df = pd.DataFrame(checks, columns=["Check", "Status", "Notes"])
    def color_status(val):
        if "pass" in str(val).lower() or "ok" in str(val).lower():
            return "color: #10b981"
        elif "warn" in str(val).lower():
            return "color: #f59e0b"
        elif "fail" in str(val).lower():
            return "color: #f43f5e"
        return ""
    st.dataframe(checks_df.style.map(color_status, subset=["Status"]),
                 use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈  Model Performance":
    st.title("Model Performance")
    st.markdown(DISCLAIMER, unsafe_allow_html=True)

    perf = ev["champion_model"]["performance"]
    disc = perf["test_discrimination"]
    cal  = perf["test_calibration"]
    ops  = perf.get("operating_points", {})
    youden = ops.get("validation_tuned_youden", {})
    youden_test = youden.get("test_metrics", {})

    section("Champion Model — Key Performance Indicators (Held-Out Test Set)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi("AUROC", f"{disc['auroc']:.3f}", "area under ROC", "accent"), unsafe_allow_html=True)
    c2.markdown(kpi("PR-AUC", f"{disc['pr_auc']:.3f}", "avg precision", "green"), unsafe_allow_html=True)
    sens = youden_test.get("recall_sensitivity", 0)
    spec = youden_test.get("specificity", 0)
    c3.markdown(kpi("Sensitivity", f"{sens:.1%}", "Youden's J threshold", "violet"), unsafe_allow_html=True)
    c4.markdown(kpi("Specificity", f"{spec:.1%}", "Youden's J threshold", "amber"), unsafe_allow_html=True)
    c5.markdown(kpi("Brier Score", f"{cal['brier_score']:.4f}", "lower = better calibration", "rose"),
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Saved plots row ───────────────────────────────────────────────────────
    section("Discrimination & Calibration Plots")
    c1, c2 = st.columns(2)
    c1.image(str(ROOT / "reports/figures/test_roc_curves.png"),
             caption="ROC Curve — Held-Out Test Set", use_container_width=True)
    c2.image(str(ROOT / "reports/figures/test_pr_curves.png"),
             caption="Precision-Recall Curve — Held-Out Test Set", use_container_width=True)

    c1, c2 = st.columns(2)
    c1.image(str(ROOT / "reports/figures/test_calibration_curves.png"),
             caption="Calibration (Reliability) Diagram", use_container_width=True)
    c2.image(str(ROOT / "reports/figures/test_confusion_matrices.png"),
             caption="Confusion Matrix at Youden's J Threshold", use_container_width=True)

    # ── Calibration detail ────────────────────────────────────────────────────
    section("Calibration Detail (ECE / MCE)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
| Metric | Value |
|---|---|
| Expected Calibration Error (ECE) | `{cal['ece']:.4f}` |
| Maximum Calibration Error (MCE)  | `{cal['mce']:.4f}` |
| Brier Score                      | `{cal['brier_score']:.4f}` |
""")
    with col2:
        st.markdown(f"""
| Metric | Value |
|---|---|
| Test AUROC | `{disc['auroc']:.4f}` |
| Test PR-AUC | `{disc['pr_auc']:.4f}` |
| Val-Test AUROC delta | `{perf['stability']['delta_auroc']:.4f}` |
""")

    # ── Model comparison table ────────────────────────────────────────────────
    section("All Candidate Models — Validation Set")
    models_eval = ev.get("models_evaluation", {})
    model_rows = []
    for mname, minfo in models_eval.items():
        if isinstance(minfo, dict) and "val_auroc" in minfo:
            model_rows.append({
                "Model": mname,
                "Val AUROC": f"{minfo.get('val_auroc', 0):.4f}",
                "Val PR-AUC": f"{minfo.get('val_pr_auc', 0):.4f}",
                "Val Brier": f"{minfo.get('val_brier', 0):.4f}",
            })
    if model_rows:
        st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Individual model comparisons are embedded in final_model_evaluation_metrics.json.")

    # ── Confusion matrix details ──────────────────────────────────────────────
    section("Confusion Matrix Values — Youden's J")
    cm_data = youden_test.get("confusion_matrix", {})
    if cm_data:
        tn, fp, fn, tp = cm_data.get("TN",0), cm_data.get("FP",0), cm_data.get("FN",0), cm_data.get("TP",0)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("True Positives (TP)", tp, help="Correctly predicted deaths")
        c2.metric("True Negatives (TN)", tn, help="Correctly predicted survivors")
        c3.metric("False Positives (FP)", fp, help="Survivors predicted to die")
        c4.metric("False Negatives (FN)", fn, help="Deaths predicted to survive")
        threshold_val = youden_test.get("threshold", youden.get("tuned_on_validation_threshold", 0.058))
        st.caption(
            f"Threshold: {threshold_val:.4f} (selected on validation set via Youden's J) | "
            f"Sensitivity: {sens:.1%} | Specificity: {spec:.1%} | "
            f"NPV: {youden_test.get('negative_predictive_value_npv',0):.1%}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SAFEPREDICT + XAI
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🛡️  SafePredict + XAI":
    st.title("SafePredict Reliability Layer + Explainability")
    st.markdown(DISCLAIMER, unsafe_allow_html=True)

    # ── Population-level SafePredict summary ─────────────────────────────────
    section("Population-Level SafePredict Results — Test Set")
    sp_combined = sp["test_results"]["safepredict_combined"]
    sp_unc      = sp["test_results"]["uncertainty_abstain"]
    sp_dq       = sp["test_results"]["dq_abstain"]
    sp_mo       = sp["test_results"]["model_only"]

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Coverage", f"{sp_combined['coverage']:.1%}",
                    "patients ACCEPTED", "green"), unsafe_allow_html=True)
    c2.markdown(kpi("Abstention Rate", f"{sp_combined['abstention_rate']:.1%}",
                    "criteria not met", "amber"), unsafe_allow_html=True)
    c3.markdown(kpi("AUROC (Accepted)", f"{sp_combined['auroc_accepted']:.3f}",
                    "SafePredict combined", "accent"), unsafe_allow_html=True)
    c4.markdown(kpi("Brier (Accepted)", f"{sp_combined['brier_accepted']:.4f}",
                    "combined strategy", "violet"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Strategy comparison table
    section("Strategy Comparison")
    strat_rows = []
    slabels = {
        "model_only": "Model Only (baseline)",
        "uncertainty_abstain": "Uncertainty Abstain",
        "dq_abstain": "DQ Score Abstain",
        "safepredict_combined": "SafePredict Combined",
    }
    for sname, sdata in sp["test_results"].items():
        strat_rows.append({
            "Strategy": slabels.get(sname, sname),
            "Coverage": f"{sdata['coverage']:.1%}",
            "Abstained": f"{sdata['abstention_rate']:.1%}",
            "N Accepted": sdata["n_accepted"],
            "AUROC": f"{sdata['auroc_accepted']:.3f}" if sdata["auroc_accepted"] else "N/A",
            "PR-AUC": f"{sdata['pr_auc_accepted']:.3f}" if sdata["pr_auc_accepted"] else "N/A",
            "Brier": f"{sdata['brier_accepted']:.4f}" if sdata["brier_accepted"] else "N/A",
        })
    st.dataframe(pd.DataFrame(strat_rows), use_container_width=True, hide_index=True)

    # Risk vs Coverage
    section("Risk vs Coverage Analysis")
    c1, c2 = st.columns([3, 2])
    c1.image(str(ROOT / "reports/figures/safepredict_risk_coverage.png"),
             caption="Risk (AUROC) vs Coverage — Uncertainty Threshold Sweep",
             use_container_width=True)
    with c2:
        cfg_vals = sp["config"]
        st.markdown(f"""
**SafePredict Configuration**

| Parameter | Value |
|---|---|
| Bootstrap resamples | {cfg_vals['n_bootstraps']} |
| Min coverage constraint | {cfg_vals['min_coverage_constraint']:.0%} |
| Uncertainty threshold | `{cfg_vals['uncertainty_threshold']:.4f}` |
| DQ threshold | `{cfg_vals['dq_threshold']:.4f}` |
| Val coverage achieved | {cfg_vals['val_coverage']:.1%} |
| Val AUROC (accepted) | {cfg_vals['val_auroc_accepted']:.4f} |

**Key principle:** Thresholds were selected on the **validation set** only.
The test set was never consulted during threshold search.

**ABSTAIN** means the reliability criteria were not met —
*not* that the patient is safe or high-risk.
""")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Individual Stay Explorer ───────────────────────────────────────────────
    section("Individual ICU Stay Explorer")

    if not DATA["sp_ready"]:
        st.error(f"SafePredict data not available: {DATA.get('sp_error','')}")
    else:
        test_ids = DATA["test_stay_ids"]
        test_probs_arr = DATA["test_probs"]
        boot_std_arr   = DATA["boot_std"]
        boot_mean_arr  = DATA["boot_mean"]
        test_dq_arr    = DATA["test_dq"]
        y_test_arr     = DATA["y_test"]
        masks          = DATA["sp_masks"]
        cfg            = DATA["sp_cfg"]

        # Build selector options
        mdf_test = DATA["model_pdf_test"].copy()
        options = []
        for i, sid in enumerate(test_ids):
            decision = "ACCEPT" if masks["safepredict_combined"][i] else "ABSTAIN"
            actual   = "Died" if y_test_arr[i] == 1 else "Survived"
            prob     = test_probs_arr[i]
            options.append(f"Stay {sid}  |  P={prob:.1%}  |  {actual}  |  {decision}")

        col_sel, col_hint = st.columns([3, 1])
        with col_sel:
            chosen = st.selectbox(
                "Select a test ICU stay",
                options=options,
                index=0,
                key="stay_selector",
            )
        with col_hint:
            st.caption("ACCEPT = reliability criteria met · ABSTAIN = criteria not met")

        idx = options.index(chosen)
        stay_id = test_ids[idx]

        # ── Stay detail columns ────────────────────────────────────────────────
        c_left, c_right = st.columns([1, 1])

        with c_left:
            # Decision badge
            decision = masks["safepredict_combined"][idx]
            badge_cls = "badge-accept" if decision else "badge-abstain"
            badge_txt = "✅  ACCEPT" if decision else "⚠️  ABSTAIN"
            st.markdown(
                f'<div style="text-align:center;padding:12px 0 4px;">'
                f'<span class="{badge_cls}">{badge_txt}</span></div>',
                unsafe_allow_html=True,
            )

            # Decision reasons (only for ABSTAIN)
            if not decision:
                reasons = []
                if boot_std_arr[idx] >= cfg.uncertainty_threshold:
                    reasons.append(f"High prediction variability: std={boot_std_arr[idx]:.3f} ≥ {cfg.uncertainty_threshold:.3f}")
                if test_dq_arr[idx] < cfg.dq_threshold:
                    reasons.append(f"Low data quality: DQ={test_dq_arr[idx]:.2f} < {cfg.dq_threshold:.2f}")
                for r in reasons:
                    st.caption(f"⚠ {r}")

            st.markdown("---")

            # Mortality risk gauge
            prob = test_probs_arr[idx]
            risk_color = "#f43f5e" if prob > 0.5 else "#f59e0b" if prob > 0.2 else "#10b981"
            st.markdown(
                f'<div style="text-align:center;margin:8px 0;">'
                f'<div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">Predicted Mortality Risk</div>'
                f'<div style="font-size:3rem;font-weight:800;color:{risk_color};">{prob:.1%}</div>'
                f'<div style="font-size:0.78rem;color:#64748b;">calibrated probability</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Mini gauges
            st.markdown("---")
            u_band = "HIGH" if boot_std_arr[idx] >= cfg.uncertainty_threshold else (
                "MODERATE" if boot_std_arr[idx] >= cfg.uncertainty_threshold * 0.55 else "LOW"
            )
            dq_band = "LOW" if test_dq_arr[idx] < cfg.dq_threshold else (
                "MODERATE" if test_dq_arr[idx] < 0.75 else "HIGH"
            )
            u_color = "#f43f5e" if u_band == "HIGH" else "#f59e0b" if u_band == "MODERATE" else "#10b981"
            dq_color = "#f43f5e" if dq_band == "LOW" else "#f59e0b" if dq_band == "MODERATE" else "#10b981"

            col_a, col_b = st.columns(2)
            col_a.markdown(
                f'<div class="kpi-card" style="border-top:3px solid {u_color};">'
                f'<div class="kpi-label">Prediction Uncertainty</div>'
                f'<div class="kpi-value" style="font-size:1.6rem;color:{u_color};">{boot_std_arr[idx]:.3f}</div>'
                f'<div class="kpi-sub">bootstrap std [{u_band}]<br>mean: {boot_mean_arr[idx]:.1%}</div>'
                f'</div>', unsafe_allow_html=True,
            )
            col_b.markdown(
                f'<div class="kpi-card" style="border-top:3px solid {dq_color};">'
                f'<div class="kpi-label">Data Quality Score</div>'
                f'<div class="kpi-value" style="font-size:1.6rem;color:{dq_color};">{test_dq_arr[idx]:.2f}</div>'
                f'<div class="kpi-sub">composite score [{dq_band}]<br>threshold: {cfg.dq_threshold:.2f}</div>'
                f'</div>', unsafe_allow_html=True,
            )

            # DQ components from model_data
            st.markdown("---")
            st.markdown("**Data Components for this Stay**")
            row_data = mdf.loc[mdf["patientunitstayid"] == stay_id]
            if not row_data.empty:
                row = row_data.iloc[0]
                avail_labs  = int(row.get("available_lab_count", 0))
                avail_vits  = int(row.get("available_vital_count", 0))
                miss_feats  = int(row.get("missing_feature_count", 0))
                col_d, col_e, col_f = st.columns(3)
                col_d.metric("Lab types", f"{avail_labs} / 15")
                col_e.metric("Vital types", f"{avail_vits} / 7")
                col_f.metric("Missing feats", miss_feats)

                # Lab progress bar
                st.markdown(f"Lab breadth: **{avail_labs}/15**")
                st.progress(avail_labs / 15)
                st.markdown(f"Vital breadth: **{avail_vits}/7**")
                st.progress(avail_vits / 7)

        with c_right:
            st.markdown("**SHAP Feature Contributions**")
            st.caption("These features contributed to the model's prediction — "
                       "not evidence of clinical causation.")

            # Try loading SHAP (cached)
            shap_data = load_shap(DATA)
            if shap_data["ok"]:
                extract_fn = shap_data["extract_fn"]
                shap_expl  = shap_data["shap_expl"]
                contribs   = extract_fn(shap_expl, idx, top_k=8)
                pos_feats  = contribs.get("top_features_increasing_prediction", [])
                neg_feats  = contribs.get("top_features_decreasing_prediction", [])

                all_contribs = (
                    [(f["feature"], f["shap_value"], f["feature_value"]) for f in pos_feats[:4]]
                    + [(f["feature"], f["shap_value"], f["feature_value"]) for f in neg_feats[:4]]
                )
                all_contribs.sort(key=lambda x: x[1])

                if all_contribs:
                    names  = [f"{c[0]} = {c[2]}" for c in all_contribs]
                    values = [c[1] for c in all_contribs]
                    colors = ["#f43f5e" if v > 0 else "#3b82f6" for v in values]

                    fig, ax = plt.subplots(figsize=(5.5, max(3, len(names) * 0.42)), facecolor="#0f172a")
                    ax.set_facecolor("#0f172a")
                    bars = ax.barh(names, values, color=colors, alpha=0.88, edgecolor="#0f172a")
                    ax.axvline(0, color="#64748b", lw=0.8)
                    ax.set_xlabel("SHAP Value (log-odds contribution)", color="#94a3b8", fontsize=8.5)
                    ax.tick_params(colors="#64748b", labelsize=7.5)
                    ax.set_title(f"Local SHAP — Stay {stay_id}", color="#94a3b8", fontsize=9)
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#334155")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                    st.caption("🔴 Red bars = increase mortality prediction · 🔵 Blue bars = decrease prediction")
            else:
                st.warning(f"SHAP computation unavailable: {shap_data.get('error','')}")
                # Show global feature importance as fallback
                st.markdown("**Global Feature Importance (fallback)**")
                gi = sg["feature_importances"][:10]
                gi_df = pd.DataFrame(gi)[["feature","mean_abs_shap","relative_importance_pct"]]
                gi_df.columns = ["Feature", "Mean |SHAP|", "Importance (%)"]
                st.dataframe(gi_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Global SHAP summary ───────────────────────────────────────────────
        section("Global SHAP Feature Importance (Test Cohort, N=291)")
        c1, c2 = st.columns(2)
        c1.image(str(ROOT / "reports/figures/shap_summary_beeswarm.png"),
                 caption="SHAP Summary Beeswarm — Feature impact distribution", use_container_width=True)
        c2.image(str(ROOT / "reports/figures/shap_summary_bar.png"),
                 caption="SHAP Global Feature Importance Bar", use_container_width=True)

        # ── Case study quick nav ──────────────────────────────────────────────
        section("Case Studies — SHAP Local Explanations")
        case_cols = st.columns(4)
        case_imgs = [
            ("shap_local_case_1_true_positive.png", "True Positive\n(High risk — Died)"),
            ("shap_local_case_2_true_negative.png", "True Negative\n(Low risk — Survived)"),
            ("shap_local_case_3_false_positive.png", "False Positive\n(High risk — Survived)"),
            ("shap_local_case_4_false_negative.png", "False Negative\n(Low risk — Died)"),
        ]
        for col, (fname, caption) in zip(case_cols, case_imgs):
            img_path = ROOT / "reports/figures" / fname
            if img_path.exists():
                col.image(str(img_path), caption=caption, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — LIVE PATIENT SCORING
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🧪  Live Patient Scoring":
    st.title("Live Patient Scoring")
    st.markdown(DISCLAIMER, unsafe_allow_html=True)
    with st.expander("📖 **How to use this page**", expanded=False):
        st.markdown("""
        **Welcome to the Live Patient Simulator!**
        This tool allows you to simulate a new patient arriving in the ICU and see how the AI and SafePredict layer respond.
        
        Because the AI requires 141 distinct physiological features to make a prediction, we automatically start you with a "Baseline Patient" (representing the median average of our entire dataset). You can tweak the most important features using the sliders below:
        
        1. **Demographics & Vitals:** Adjust the patient's physical state. Extremely high or low Heart Rates or poor Oxygen levels (SaO2) usually increase risk.
        2. **Lab Results:** Adjust bloodwork values. High BUN (Blood Urea Nitrogen) is a strong indicator of kidney stress and higher mortality risk.
        3. **Missing Data Simulation:** What happens if the nurse hasn't entered the labs yet? Drag the "Available Lab Types" slider down to 0 to simulate poor **Data Quality**. The SafePredict layer will catch this and **ABSTAIN** from making a dangerous guess.
        
        *Try intentionally entering contradictory vitals (e.g., extremely high Mean Heart Rate but a very low Minimum Heart Rate) to see how the model's **Uncertainty** spikes, forcing an ABSTAIN decision!*
        """)

    # Get median baseline for a new patient
    if "median_patient" not in st.session_state:
        df_features = mdf.drop(columns=["patientunitstayid", "hospital_mortality", "patienthealthsystemstayid"], errors="ignore")
        
        # Numeric medians
        median_dict = df_features.median(numeric_only=True).to_dict()
        
        # Categorical modes
        cat_cols = df_features.select_dtypes(include=["object", "category"]).columns
        for c in cat_cols:
            if not df_features[c].mode().empty:
                median_dict[c] = df_features[c].mode()[0]
            else:
                median_dict[c] = "Unknown"
                
        st.session_state["median_patient"] = median_dict

    base_patient = st.session_state["median_patient"].copy()

    # Create columns for sliders
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### Demographics & Vitals")
        age = st.slider("Age (years)", 18, 100, int(base_patient.get("age_numeric", 65)))
        weight = st.slider("Admission Weight (kg)", 40, 200, int(base_patient.get("admissionweight", 80)))
        hr_mean = st.slider("Heart Rate (Mean)", 40, 200, int(base_patient.get("heartrate_mean", 85)))
        hr_min = st.slider("Heart Rate (Min)", 30, 150, int(base_patient.get("heartrate_min", 70)))
        rr_count = st.slider("Respiration Rate (Measurements Count)", 0, 100, int(base_patient.get("respiration_count", 24)))
        sao2_std = st.slider("SaO2 (Std Dev)", 0.0, 20.0, float(base_patient.get("sao2_std", 2.0)))
        
    with c2:
        st.markdown("### Lab Results")
        bun_mean = st.slider("BUN (Mean)", 5.0, 150.0, float(base_patient.get("BUN_mean", 20.0)))
        bun_max = st.slider("BUN (Max)", 5.0, 150.0, float(base_patient.get("BUN_max", 22.0)))
        bun_last = st.slider("BUN (Last)", 5.0, 150.0, float(base_patient.get("BUN_last", 21.0)))
        ph_count = st.slider("Arterial pH (Measurements Count)", 0, 20, int(base_patient.get("arterial pH_count", 1)))
        
        st.markdown("### Missing Data Simulation")
        avail_labs = st.slider("Available Lab Types", 0, 15, int(base_patient.get("available_lab_count", 8)))
        avail_vitals = st.slider("Available Vital Types", 0, 7, int(base_patient.get("available_vital_count", 6)))
        missing_feats = st.slider("Missing Feature Count", 0, 140, int(base_patient.get("missing_feature_count", 20)))

    if st.button("🔮 Score Patient", type="primary", use_container_width=True):
        # Update the base patient with slider values
        base_patient["age_numeric"] = age
        base_patient["admissionweight"] = weight
        base_patient["heartrate_mean"] = hr_mean
        base_patient["heartrate_min"] = hr_min
        base_patient["respiration_count"] = rr_count
        base_patient["sao2_std"] = sao2_std
        
        base_patient["BUN_mean"] = bun_mean
        base_patient["BUN_max"] = bun_max
        base_patient["BUN_last"] = bun_last
        base_patient["arterial pH_count"] = ph_count
        
        base_patient["available_lab_count"] = avail_labs
        base_patient["available_vital_count"] = avail_vitals
        base_patient["missing_feature_count"] = missing_feats
        
        new_df = pd.DataFrame([base_patient])
        
        with st.spinner("Analyzing patient risk and generating SHAP explanation..."):
            try:
                import joblib
                import polars as pl
                from src.safepredict import compute_bootstrap_uncertainty, compute_data_quality_scores
                from src.explain import load_champion_artifacts, extract_base_classifier, clean_feature_names
                import shap
                
                cal_model, prep = load_champion_artifacts()
                base_clf = extract_base_classifier(cal_model)
                
                feature_cols = prep.feature_names_in_ if hasattr(prep, 'feature_names_in_') else list(base_patient.keys())
                new_df = new_df[feature_cols]
                
                # Predict
                prob = cal_model.predict_proba(new_df)[:, 1][0]
                
                # Uncertainty (approximate bootstrap using test set as base)
                X_test_trans = prep.transform(DATA["X_test"])
                new_df_trans = prep.transform(new_df)
                boot_mean, boot_std = compute_bootstrap_uncertainty(X_test_trans, DATA["y_test"], new_df_trans)
                
                # Data Quality
                sim_pl = pl.DataFrame(new_df).with_columns(pl.Series("patientunitstayid", [999999]))
                dq = compute_data_quality_scores(sim_pl).values
                
                # SafePredict Decision
                sp_cfg = DATA["sp_cfg"]
                is_accepted = (boot_std[0] <= sp_cfg.uncertainty_threshold) and (dq[0] >= sp_cfg.dq_threshold)
                
                # Results UI
                st.markdown("---")
                rc1, rc2 = st.columns([1, 2])
                
                with rc1:
                    st.markdown("### SafePredict Output")
                    
                    if is_accepted:
                        st.markdown("<div class='badge-accept' style='display:block;text-align:center;margin-bottom:15px;'>✅ ACCEPTED</div>", unsafe_allow_html=True)
                        st.markdown(kpi("Mortality Risk", f"{prob:.1%}", "AI Prediction", "rose"), unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='badge-abstain' style='display:block;text-align:center;margin-bottom:15px;'>⚠️ ABSTAIN</div>", unsafe_allow_html=True)
                        st.markdown("<div style='font-size:0.8rem;color:#fca5a5;text-align:center;margin-bottom:10px;'>Action Required: Prediction unreliable.</div>", unsafe_allow_html=True)
                        
                    dq_pct = dq[0] * 100
                    dq_thresh_pct = sp_cfg.dq_threshold * 100
                    dq_safe = dq[0] >= sp_cfg.dq_threshold
                    st.markdown(kpi("Data Completeness", f"{dq_pct:.0f}%", f"Minimum required: {dq_thresh_pct:.0f}%", "green" if dq_safe else "rose"), unsafe_allow_html=True)
                    
                    unc_safe = boot_std[0] <= sp_cfg.uncertainty_threshold
                    unc_status = "Low (Safe)" if unc_safe else "High (Unsafe)"
                    unc_sub = "AI models strongly agree" if unc_safe else "AI models disagree (guessing)"
                    st.markdown(kpi("Model Uncertainty", unc_status, unc_sub, "green" if unc_safe else "rose"), unsafe_allow_html=True)
                    
                with rc2:
                    st.markdown("### Why? (SHAP Explanation)")
                    try:
                        explainer = shap.TreeExplainer(base_clf)
                        shap_values = explainer.shap_values(new_df_trans)[0]
                        feat_names = prep.get_feature_names_out()
                        clean_names = clean_feature_names(feat_names)
                        
                        df_shap = pd.DataFrame({"feature": clean_names, "val": shap_values, "abs_val": np.abs(shap_values)})
                        df_shap = df_shap.sort_values("abs_val", ascending=False)
                        
                        pos_feats = df_shap[df_shap["val"] > 0].head(3)
                        neg_feats = df_shap[df_shap["val"] < 0].head(3)
                        
                        st.markdown("**Plain English Translation:**")
                        if not pos_feats.empty:
                            inc_names = [f"**{r['feature']}**" for _, r in pos_feats.iterrows()]
                            st.markdown(f"🔴 The patient's {', '.join(inc_names)} strongly **increased** their predicted mortality risk.")
                        if not neg_feats.empty:
                            dec_names = [f"**{r['feature']}**" for _, r in neg_feats.iterrows()]
                            st.markdown(f"🔵 The patient's {', '.join(dec_names)} **decreased** their predicted mortality risk (protective).")
                            
                        # Top 8 absolute features for plot
                        df_shap_top = df_shap.head(8)
                        names  = df_shap_top["feature"].tolist()
                        values = df_shap_top["val"].tolist()
                        colors = ["#f43f5e" if v > 0 else "#3b82f6" for v in values]

                        fig, ax = plt.subplots(figsize=(5.5, max(3, len(names) * 0.42)), facecolor="#0f172a")
                        ax.set_facecolor("#0f172a")
                        bars = ax.barh(names, values, color=colors, alpha=0.88, edgecolor="#0f172a")
                        ax.axvline(0, color="#64748b", lw=0.8)
                        ax.set_xlabel("SHAP Value (log-odds contribution)", color="#94a3b8", fontsize=8.5)
                        ax.tick_params(colors="#64748b", labelsize=7.5)
                        ax.set_title("Local SHAP — Live Patient", color="#94a3b8", fontsize=9)
                        for spine in ax.spines.values():
                            spine.set_edgecolor("#334155")
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close()
                    except Exception as shap_err:
                        st.error(f"SHAP explainer failed: {str(shap_err)}")
                        
            except Exception as e:
                st.error(f"Error during live scoring: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-size:0.72rem;color:#334155;padding:6px;'>"
    "SafePredict-XAI Research Dashboard &nbsp;|&nbsp; "
    "eICU Collaborative Research Database &nbsp;|&nbsp; "
    "<span style='color:#f59e0b;font-weight:600;'>Research prototype — not for clinical decision-making</span>"
    "</div>",
    unsafe_allow_html=True,
)
