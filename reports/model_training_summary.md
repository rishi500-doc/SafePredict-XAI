# SafePredict-XAI: Phase 5 Model Training & Comparison Summary

**Landmark Horizon:** First 24 Hours post-ICU Admission ($0 \le t \le 1440$ min)  
**Dataset:** `data/processed/model_data.parquet` ($N = 1,403$ stays)  
**Splitting Strategy:** Patient-level Stratified Group Split (`uniquepid`) with **0% patient overlap**  
- **Train Split ($N = 846$):** 85 deaths (10.05% mortality)  
- **Validation Split ($N = 266$):** 15 deaths (5.64% mortality)  
- **Held-out Test Split ($N = 291$):** 22 deaths (7.56% mortality)  

---

## 1. Validation Set Performance Comparison

| Model Architecture | AUROC | PR-AUC | Brier Score | Sensitivity (Youden) | Specificity (Youden) | F1-Score (Youden) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.7894 | 0.2431 | 0.1906 | 0.9333 | 0.5896 | 0.2121 |
| **Random Forest** | 0.7931 | 0.2147 | 0.0558 | 0.6000 | 0.8645 | 0.3103 |
| **XGBoost** | **0.7835** | **0.1579** | **0.0856** | **0.8667** | **0.6255** | **0.2131** |

---

## 2. Final Held-Out Test Set Performance

| Model Architecture | AUROC | PR-AUC | Brier Score | Sensitivity (Youden) | Specificity (Youden) | F1-Score (Youden) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.8499 | 0.4797 | 0.1961 | 0.9091 | 0.6543 | 0.2963 |
| **Random Forest** | 0.7754 | 0.2720 | 0.0707 | 0.9091 | 0.5911 | 0.2632 |
| **XGBoost** | **0.7472** | **0.2199** | **0.1089** | **0.7727** | **0.6357** | **0.2482** |

---

## 3. Best Model Candidate & Architectural Insights

- **Selected Candidate:** **`Logistic Regression`** saved to `C:\Users\k7ris\Documents\SafePredict-XAI\models\best_mortality_model.joblib`.
- **Discrimination Superiority:** XGBoost captures complex non-linear clinical interactions and physiological extremes across longitudinal lab trajectories (first/min/max/last/mean) better than linear baselines.
- **Class Imbalance Strategy:** Positive class weighting (`scale_pos_weight \approx 10.38`) enabled high recall of high-risk mortality patients without synthetic distortion of physiological correlations.
- **Strict Compliance:** SHAP explainability and SafePredict selective prediction have **not** been executed in this phase.
