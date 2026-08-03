# SafePredict-XAI: Interview Notes

This document contains answers to common technical and domain questions regarding the SafePredict-XAI architecture, methodology, and design choices.

### Why I chose eICU
The eICU Collaborative Research Database provides a realistic, multi-center, de-identified snapshot of critical care patient data. It contains highly complex, messy, real-world data (missing values, irregularly sampled vitals, lab errors) making it the perfect proving ground for a reliability layer like SafePredict. Standard benchmark datasets are often too clean to demonstrate the need for abstention mechanisms in ML.

### Why I used three relational tables
I used `patient` (demographics/outcomes), `lab` (bloodwork and chemistry), and `vitalPeriodic` (continuous bedside monitoring). This triad represents the core clinical state of an ICU patient. By joining them, I could demonstrate complex relational feature engineering (aggregating time-series data over a specific window) rather than just fitting a model to a pre-flattened CSV.

### Why Polars
Polars was used for the data processing pipeline (`src/features.py`) due to its exceptional speed and memory efficiency with large out-of-core datasets compared to Pandas. It utilizes multi-threading and lazy evaluation, which is critical when processing millions of rows of high-frequency vital signs and lab measurements in clinical datasets.

### How the tables are joined
The `patient` table is the backbone. I used the `patientunitstayid` as the primary key. For each stay, I joined `lab` and `vitalPeriodic` data by filtering measurements where the timestamp (`labresultoffset` or `observationoffset`) fell strictly within the first 24 hours of admission (0 to 1440 minutes). This was aggregated (min, max, mean) and then left-joined back to the patient cohort.

### What temporal leakage is
Temporal leakage (or data leakage) occurs when information from the future (relative to the time the model is supposed to make a prediction) is accidentally included in the training data. For example, if I used total length of stay or discharge status as a feature to predict mortality on day 1, the model would "cheat."

### Why first 24 hours
To make the model clinically actionable, it must predict risk *before* the outcome occurs. By restricting all features to data collected strictly between minute 0 and minute 1440 of the ICU stay, the model simulates making a prediction exactly at the 24-hour mark, ensuring no future data leaks into the feature set.

### How I validated data
I built a rule-based validation step that applies physiological bounds to raw data before aggregation. For example, Heart Rates outside [0, 300] or Temperatures outside [20, 50] Celsius are nullified. I also flagged patients missing critical vitals. This ensures the model learns from actual physiological states, not data entry errors (e.g., a weight of 999 kg).

### How I handled missing data
I handled missing data natively using XGBoost's sparsity-aware split finding. Instead of using median imputation (which destroys the information that a test was *not ordered*), XGBoost treats missingness as a feature itself. In the ICU, a missing lab test often implies the doctor didn't think the patient was sick enough to need it—a concept known as "informative missingness."

### How features were created
Features were engineered by aggregating time-series data over the 24-hour window. For continuous signals (vitals) and repeated measures (labs), I calculated summary statistics: `min`, `max`, and `mean`. I also extracted static demographics like `age` and `admissionweight`.

### Why Logistic Regression
Logistic Regression was used as a baseline model. It is intrinsically interpretable, fast to train, and provides a benchmark for evaluating whether the added complexity of a non-linear tree ensemble is actually necessary.

### Why XGBoost
XGBoost was chosen as the champion model because it natively handles missing values (crucial for EHR data), captures non-linear interactions between physiological variables, and typically achieves state-of-the-art performance on tabular clinical data. It outperformed Logistic Regression in this project.

### Why AUROC alone is insufficient
AUROC measures discrimination (the ability to rank a high-risk patient above a low-risk patient). However, it ignores class imbalance and *calibration*. In clinical settings, predicting a "rank" isn't enough; if a model says a patient has a 20% risk of mortality, the actual probability should be 20%. Furthermore, PR-AUC is more informative when the positive class (mortality) is rare.

### What calibration means
Calibration refers to the agreement between estimated probabilities and actual observed frequencies. A perfectly calibrated model that predicts 10% risk for 100 patients should see exactly 10 of those patients die. I evaluated this using the Brier Score and calibration curves.

### What SHAP means
SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain the output of machine learning models. It calculates the marginal contribution of each feature to the final prediction. In this project, it allows us to look at an individual patient's 25% mortality risk and explain exactly *why* the model predicted that (e.g., +15% due to high heart rate, -5% due to normal age).

### What uncertainty means
In this project, uncertainty is measured using the variance of predictions across an ensemble (Bootstrap resampling). If 30 variations of the model look at the same patient and predict vastly different risks, the model is uncertain about that patient. Note: `predict_proba()` is a probability, *not* a measure of confidence/uncertainty in that probability.

### What abstention means
Abstention is when the model refuses to make a prediction. Rather than outputting a potentially dangerous guess, the system flags the prediction as "ABSTAIN", indicating that human intervention is required because the reliability criteria were not met.

### What SafePredict adds
SafePredict is the reliability layer on top of the raw XGBoost model. It combines Data Quality checks (are there enough valid measurements?) with Epistemic Uncertainty (do the bootstrap models agree?). If a prediction is too uncertain or data quality is too poor, SafePredict forces an abstention. This increases the safety and trustworthiness of the deployment.

### Project limitations
1. **Data size:** I only used a small subset of the full eICU database (due to local compute constraints).
2. **Missing time-series dynamics:** Using 24-hour aggregates (min/max/mean) loses fine-grained temporal trends (e.g., a rapidly spiking heart rate vs a stable one).
3. **Single endpoint:** Predicting only in-hospital mortality is binary; clinical reality involves multiple competing risks.

### What I would improve with full eICU data
With the full 200,000+ patient database, I would:
1. Implement recurrent neural networks (LSTMs) or Transformers to model the raw, unaggregated time-series data.
2. Predict continuous patient trajectories (e.g., dynamic risk scoring every hour) rather than a single 24-hour snapshot.
3. Validate the model across different hospital systems within the dataset to prove geographic generalizability.
