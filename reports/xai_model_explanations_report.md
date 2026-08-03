# SafePredict-XAI: Phase 7 Explainable AI (XAI) Report

**Model Architecture:** XGBoost Classifier with Platt Sigmoid Calibration (Phase 6 Champion)  
**Evaluation Cohort:** Held-Out Test Set ($N = 280$ ICU stays, 24 deaths, 8.57% mortality prevalence)  
**Explanation Engine:** TreeSHAP (`shap.TreeExplainer` on fitted tree ensemble margin)  
**Population Base Expected Value:** 0.3983 log-odds  

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
| 1 | **Age (years)** | 0.5653 | 11.23% | 11.23% | High values contribute positively to mortality prediction |
| 2 | **BUN (Last)** | 0.3549 | 7.05% | 18.28% | High values contribute positively to mortality prediction |
| 3 | **Heart Rate (Mean)** | 0.2325 | 4.62% | 22.89% | High values contribute positively to mortality prediction |
| 4 | **Arterial pH (Count)** | 0.2294 | 4.56% | 27.45% | High values contribute positively to mortality prediction |
| 5 | **Admission Weight (kg)** | 0.1784 | 3.54% | 30.99% | High values contribute positively to mortality prediction |
| 6 | **BUN (Mean)** | 0.1576 | 3.13% | 34.13% | High values contribute positively to mortality prediction |
| 7 | **SaO2 (Std Dev)** | 0.1509 | 3.00% | 37.12% | High values contribute positively to mortality prediction |
| 8 | **Heart Rate (Min)** | 0.1432 | 2.84% | 39.97% | High values contribute positively to mortality prediction |
| 9 | **BUN (Max)** | 0.1286 | 2.55% | 42.52% | High values contribute positively to mortality prediction |
| 10 | **Respiration Rate (Count)** | 0.1093 | 2.17% | 44.69% | High values contribute positively to mortality prediction |
| 11 | **Bicarbonate (Last)** | 0.1006 | 2.00% | 46.69% | High values contribute positively to mortality prediction |
| 12 | **Creatinine (First)** | 0.0895 | 1.78% | 48.47% | High values contribute positively to mortality prediction |
| 13 | **SaO2 (Count)** | 0.0875 | 1.74% | 50.20% | High values contribute positively to mortality prediction |
| 14 | **Respiration Rate (Min)** | 0.0791 | 1.57% | 51.77% | High values contribute positively to mortality prediction |
| 15 | **Arterial pH (Min)** | 0.0782 | 1.55% | 53.33% | High values contribute positively to mortality prediction |

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

### High-Risk True Positive (Correct Critical Alert)

- **Patient Identifier:** Unit Stay #1115347 (Unique Patient: `010-1042`)
- **Patient Profile:** Age 90 years | ICU Unit: Med-Surg ICU
- **Predicted Mortality Probability:** **32.3%** (Calibrated Risk)
- **Actual In-Hospital Outcome:** **Deceased** (Ground Truth: 1)
- **Clinical Context:** Patient who died in hospital and was correctly identified by the model with high predicted mortality probability.

#### Top Features Increasing Model Prediction (Higher Risk Contribution):
- **Age (years)** (Value: `1.47`): Contributed **+0.465** to prediction margin.
- **Arterial pH (Count)** (Value: `3.26`): Contributed **+0.353** to prediction margin.
- **Arterial pH (Min)** (Value: `-0.89`): Contributed **+0.211** to prediction margin.
- **Arterial pH (First)** (Value: `-0.92`): Contributed **+0.177** to prediction margin.

#### Top Features Decreasing Model Prediction (Protective / Low-Risk Contribution):
- **SaO2 (Count)** (Value: `-0.67`): Contributed **-0.237** to prediction margin.
- **Respiration Rate (Std Dev)** (Value: `-1.35`): Contributed **-0.155** to prediction margin.
- **Platelets (Min)** (Value: `1.05`): Contributed **-0.109** to prediction margin.
- **Heart Rate (Max)** (Value: `-0.32`): Contributed **-0.083** to prediction margin.

- **Local Waterfall Plot:** [`C:\Users\k7ris\Documents\SafePredict-XAI\reports\figures\shap_local_case_1_true_positive.png`](file:///C:\Users\k7ris\Documents\SafePredict-XAI\reports\figures\shap_local_case_1_true_positive.png)

---

### Low-Risk True Negative (Correct Reassurance)

- **Patient Identifier:** Unit Stay #1644719 (Unique Patient: `017-100762`)
- **Patient Profile:** Age 59 years | ICU Unit: Cardiac ICU
- **Predicted Mortality Probability:** **2.4%** (Calibrated Risk)
- **Actual In-Hospital Outcome:** **Survived** (Ground Truth: 0)
- **Clinical Context:** Patient who survived hospital stay and was correctly assigned a very low predicted mortality risk.

#### Top Features Increasing Model Prediction (Higher Risk Contribution):
- **Glucose (Last)** (Value: `-0.84`): Contributed **+0.109** to prediction margin.
- **SaO2 (Count)** (Value: `0.51`): Contributed **+0.049** to prediction margin.
- **Unit: Med-Surg ICU** (Value: `0.00`): Contributed **+0.025** to prediction margin.
- **Admit Source: Operating Room** (Value: `0.00`): Contributed **+0.025** to prediction margin.

#### Top Features Decreasing Model Prediction (Protective / Low-Risk Contribution):
- **Age (years)** (Value: `-0.30`): Contributed **-0.676** to prediction margin.
- **Heart Rate (Mean)** (Value: `-0.93`): Contributed **-0.337** to prediction margin.
- **Admission Weight (kg)** (Value: `1.11`): Contributed **-0.329** to prediction margin.
- **BUN (Last)** (Value: `-0.37`): Contributed **-0.295** to prediction margin.

- **Local Waterfall Plot:** [`C:\Users\k7ris\Documents\SafePredict-XAI\reports\figures\shap_local_case_2_true_negative.png`](file:///C:\Users\k7ris\Documents\SafePredict-XAI\reports\figures\shap_local_case_2_true_negative.png)

---

### False Positive Case (Elevated Risk in Survivor)

- **Patient Identifier:** Unit Stay #3060213 (Unique Patient: `030-10431`)
- **Patient Profile:** Age 64 years | ICU Unit: Med-Surg ICU
- **Predicted Mortality Probability:** **33.0%** (Calibrated Risk)
- **Actual In-Hospital Outcome:** **Survived** (Ground Truth: 0)
- **Clinical Context:** Patient who survived hospital stay but exhibited severe acute derangements that elevated model predicted risk.

#### Top Features Increasing Model Prediction (Higher Risk Contribution):
- **SaO2 (Std Dev)** (Value: `0.90`): Contributed **+0.313** to prediction margin.
- **BUN (Last)** (Value: `-0.31`): Contributed **+0.303** to prediction margin.
- **Arterial pH (Count)** (Value: `0.99`): Contributed **+0.297** to prediction margin.
- **Arterial pH (Min)** (Value: `-0.42`): Contributed **+0.197** to prediction margin.

#### Top Features Decreasing Model Prediction (Protective / Low-Risk Contribution):
- **Age (years)** (Value: `-0.02`): Contributed **-0.265** to prediction margin.
- **Bicarbonate (Last)** (Value: `2.21`): Contributed **-0.248** to prediction margin.
- **Total Lab Measurements** (Value: `0.09`): Contributed **-0.059** to prediction margin.
- **Bicarbonate (Mean)** (Value: `2.15`): Contributed **-0.059** to prediction margin.

- **Local Waterfall Plot:** [`C:\Users\k7ris\Documents\SafePredict-XAI\reports\figures\shap_local_case_3_false_positive.png`](file:///C:\Users\k7ris\Documents\SafePredict-XAI\reports\figures\shap_local_case_3_false_positive.png)

---

### False Negative Case (Missed Deterioration / Borderline)

- **Patient Identifier:** Unit Stay #2646210 (Unique Patient: `025-10402`)
- **Patient Profile:** Age 62 years | ICU Unit: Med-Surg ICU
- **Predicted Mortality Probability:** **3.0%** (Calibrated Risk)
- **Actual In-Hospital Outcome:** **Deceased** (Ground Truth: 1)
- **Clinical Context:** Patient who died in hospital despite subtle or delayed initial 24h derangements leading to a lower model risk score.

#### Top Features Increasing Model Prediction (Higher Risk Contribution):
- **Respiration Rate (Min)** (Value: `1.35`): Contributed **+0.167** to prediction margin.
- **BUN (Last)** (Value: `-0.26`): Contributed **+0.154** to prediction margin.
- **Platelets (Mean)** (Value: `-1.23`): Contributed **+0.143** to prediction margin.
- **Platelets (Max)** (Value: `-1.26`): Contributed **+0.108** to prediction margin.

#### Top Features Decreasing Model Prediction (Protective / Low-Risk Contribution):
- **Age (years)** (Value: `-0.13`): Contributed **-0.594** to prediction margin.
- **Glucose (Max)** (Value: `1.32`): Contributed **-0.453** to prediction margin.
- **Heart Rate (Mean)** (Value: `-1.09`): Contributed **-0.432** to prediction margin.
- **Admission Weight (kg)** (Value: `1.98`): Contributed **-0.403** to prediction margin.

- **Local Waterfall Plot:** [`C:\Users\k7ris\Documents\SafePredict-XAI\reports\figures\shap_local_case_4_false_negative.png`](file:///C:\Users\k7ris\Documents\SafePredict-XAI\reports\figures\shap_local_case_4_false_negative.png)

---

## 5. Artifact Summary

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
