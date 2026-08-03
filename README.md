# SafePredict-XAI: Reliable Machine Learning for the ICU

## 1. Project Title
SafePredict-XAI: A Reliability Layer for Clinical Machine Learning using the eICU Database.

## 2. Problem Statement
Machine learning models applied to Electronic Health Records (EHRs) often face out-of-distribution shifts, missing measurements, and noisy sensor data. When these models fail, they fail silently, confidently emitting dangerous predictions. Traditional AUROC-optimized models provide no indication of their own unreliability.

## 3. Research Question
Can we design a transparent "reliability layer" (SafePredict) that automatically identifies unconfident or data-poor predictions and forces the model to **ABSTAIN** rather than outputting a potentially harmful clinical decision?

## 4. eICU Dataset
This project uses a subset of the **eICU Collaborative Research Database**, a multi-center ICU database. We utilize three highly relational tables:
- `patient`: Demographics and admission outcomes.
- `lab`: Irregularly sampled blood chemistry and hematology.
- `vitalPeriodic`: High-frequency bedside vital sign monitoring.

## 5. Architecture

```
eICU
 ↓
Data Understanding
 ↓
Validation
 ↓
Cleaning
 ↓
Feature Engineering
 ↓
ML
 ↓
Calibration
 ↓
Uncertainty
 ↓
SafePredict
 ↓
SHAP
 ↓
KPI Dashboard
```

## 6. Data Understanding
EHR data is messy. A patient might have a heart rate recorded every 5 minutes, but a lactate level recorded only once a day. Some values are physically impossible due to data entry errors.

## 7. Validation
We apply strict physiological boundaries to raw measurements (e.g., Heart Rate must be between 0 and 300; Temperature must be between 20°C and 50°C). Values outside these bounds are nullified, treating them as missing rather than corrupting the feature space.

## 8. Feature Engineering
We restrict all data to the **first 24 hours** of the ICU stay to prevent temporal leakage (ensuring the model predicts *before* the outcome). For each patient, we aggregate the 24-hour window into statistical summaries: `min`, `max`, and `mean` for every vital and lab test. Missingness is preserved and natively handled by the ML algorithm.

## 9. ML
We compared **Logistic Regression** (baseline) against **XGBoost** (champion). XGBoost inherently handles missing values (informative missingness) without requiring median imputation, which can destroy the clinical signal that a test was intentionally not ordered.

## 10. Evaluation
Models are evaluated on a strictly held-out test set. We evaluate:
- **Discrimination:** AUROC and PR-AUC.
- **Calibration:** Brier Score and calibration curves to ensure predicted risks map to true frequencies.

## 11. XAI (Explainable AI)
We utilize **TreeSHAP** to provide patient-level explanations. For any given prediction, SHAP calculates the exact marginal contribution of each physiological feature (e.g., +15% risk due to low minimum heart rate).

## 12. SafePredict
SafePredict is the core contribution of this project. It is a reliability layer that evaluates two criteria before releasing a prediction:
1. **Data Quality (DQ):** Does the patient have enough valid measurements, or are they a "ghost" in the system?
2. **Epistemic Uncertainty:** We run a bootstrap ensemble (30 resampled models). Do the models agree, or is the variance too high?

If the variance exceeds a threshold or the DQ score falls below a threshold, the system **ABSTAINS**.

## 13. Dashboard
A Streamlit research dashboard allows for interactive exploration of the cohort, model performance metrics, data quality impact, and individual patient SHAP explanations alongside SafePredict outcomes.

## 14. Results
- **Champion Model:** XGBoost
- **Test AUROC (Raw):** 0.721
- **SafePredict Abstention Rate:** 23.0%
- **SafePredict Accepted AUROC:** 0.927
- **SafePredict Accepted Brier Score:** 0.037 (Excellent calibration)

By abstaining on the most uncertain 23% of patients, the model's reliability on the remaining 77% increases drastically.

## 15. Limitations
- Only a subset of eICU is used for demonstration.
- Aggregating to 24-hour min/max/mean destroys fine-grained time-series dynamics.
- Predicts a single endpoint (hospital mortality) rather than continuous dynamic risk.

## 16. Installation
Requires Python 3.9.

```bash
# Clone the repository
git clone https://github.com/username/SafePredict-XAI.git
cd SafePredict-XAI

# Using uv (recommended)
uv venv --python 3.9
.venv\Scripts\activate
uv pip install -e .
```

## 17. How to run

**1. Run the ML Pipeline**
The pipeline is entirely modular:
```bash
python src/features.py
python src/model.py
python src/evaluate.py
python src/explain.py
python src/safepredict.py
```

**2. Launch the Dashboard**
```bash
streamlit run app.py
```
Access the dashboard at `http://localhost:8501`.

## 18. Research-Use Disclaimer
> **⚠️ WARNING:** This is a research prototype. It is NOT intended for clinical decision-making, diagnosis, or treatment. The models and thresholds are trained on retrospective, de-identified data and have not been validated in a prospective clinical setting.
