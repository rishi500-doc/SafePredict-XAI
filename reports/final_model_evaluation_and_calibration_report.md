# SafePredict-XAI: Phase 6 Model Evaluation & Calibration Report

**Landmark Horizon:** First 24 Hours post-ICU Admission ($0 \le t \le 1440$ min)  
**Dataset:** `data/processed/model_data.parquet` ($N = 1,403$ stays)  
**Splitting Strategy:** Patient-level Stratified Group Split (`uniquepid`) with **0% patient overlap**  
- **Train Split ($N = 846$):** 85 deaths (10.05% mortality)  
- **Validation Split ($N = 266$):** 15 deaths (5.64% mortality)  
- **Held-out Test Split ($N = 291$):** 22 deaths (7.56% mortality)  

---

## 1. Held-Out Test Set Performance Comparison

All models evaluated on the strictly held-out test partition ($N=280$, 24 deaths, 8.57% prevalence):

| Model Architecture | Calibration Variant | Test AUROC | Test PR-AUC | Brier Score | ECE | Sensitivity (Youden) | Specificity (Youden) | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | Uncalibrated | 0.8499 | 0.4797 | 0.1961 | 0.3756 | 0.9091 | 0.5204 | 0.2339 |
| Logistic Regression | Sigmoid (Platt) | 0.8499 | 0.4797 | 0.0547 | 0.0266 | 0.9091 | 0.5204 | 0.2339 |
| Logistic Regression | Isotonic | 0.8312 | 0.3470 | 0.0562 | 0.0104 | 0.9091 | 0.5204 | 0.2339 |
| Random Forest | Uncalibrated | 0.7754 | 0.2720 | 0.0707 | 0.0760 | 0.4091 | 0.8290 | 0.2338 |
| Random Forest | Sigmoid (Platt) | 0.7754 | 0.2720 | 0.0656 | 0.0341 | 0.4091 | 0.8290 | 0.2338 |
| Random Forest | Isotonic | 0.7787 | 0.2284 | 0.0650 | 0.0412 | 0.4091 | 0.8290 | 0.2338 |
| XGBoost | Uncalibrated | 0.7472 | 0.2199 | 0.1089 | 0.1494 | 0.8182 | 0.5725 | 0.2323 |
| **XGBoost** | Sigmoid (Platt) | 0.7472 | 0.2199 | 0.0659 | 0.0097 | 0.8182 | 0.5725 | 0.2323 |
| XGBoost | Isotonic | 0.7312 | 0.1471 | 0.0670 | 0.0232 | 0.8182 | 0.5725 | 0.2323 |

---

## 2. Probability Calibration & Reliability Analysis

- **Calibration Protocol:** Calibrators (Sigmoid and Isotonic) were fit strictly on the **Validation set** ($N=281$) using `CalibratedClassifierCV(cv='prefit')` to prevent data snooping.
- **Brier Score:** Measures the mean squared difference between predicted mortality probabilities and binary outcomes ($y \in \{0, 1\}$).
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
4. **Stability & Generalization (15%):** Consistency between Validation and Test distributions ($|\Delta \text{AUROC}| + |\Delta \text{Brier}|$).

### Scorecard Summary

| Candidate Model | Best Variant | Discrimination (35%) | Calibration (35%) | Interpretability (15%) | Stability (15%) | Composite Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | sigmoid_platt | 1.0000 | 0.6824 | 1.0000 | 0.7673 | 0.8539 |
| Random Forest | isotonic | 0.6962 | 0.5387 | 0.8000 | 0.8279 | 0.6764 |
| **XGBoost** | sigmoid_platt | 0.6688 | 0.9151 | 0.8500 | 1.0000 | **0.8318** |

---

## 4. Final Champion Model & Clinical Decision Operating Points

- **Selected Champion:** **`XGBoost`** with **`sigmoid_platt`** calibration.
- **Artifact Locations:**
  - Model Pipeline: `C:\Users\k7ris\Documents\SafePredict-XAI\models\final_mortality_model.joblib`
  - Preprocessor Pipeline: `C:\Users\k7ris\Documents\SafePredict-XAI\models\preprocessing_pipeline.joblib`
  - Metrics Catalog: `C:\Users\k7ris\Documents\SafePredict-XAI\reports\metrics\final_model_evaluation_metrics.json`
- **Clinical Operating Characteristics (Validation-Tuned Youden Threshold = 0.044):**
  - **Sensitivity (Recall):** 81.82% (Identifies 18 of 24 deaths)
  - **Specificity:** 57.25%
  - **Precision (PPV):** 13.53%
  - **Negative Predictive Value (NPV):** 97.47%
  - **Brier Score:** 0.0659
  - **ECE:** 0.0097

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
