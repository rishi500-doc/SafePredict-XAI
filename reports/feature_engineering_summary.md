# SafePredict-XAI: Phase 4 Feature Engineering Summary

**Tabular Dataset:** `data/processed/model_data.parquet`  
**Shape:** 1,403 rows $\times$ 141 columns  
**Target Variable:** `hospital_mortality` (0 = Alive, 1 = Expired)

---

## 1. Feature Group Breakdown

| Feature Category | Column Count | Description |
| :--- | :---: | :--- |
| **Tracking Identifiers** | 2 | `patientunitstayid`, `patienthealthsystemstayid` (Strictly non-predictive) |
| **Target Variable** | 1 | `hospital_mortality` (0: 1281, 1: 122) |
| **Admission Demographics** | 8 | Age, age >89 indicator, gender, ethnicity, height, weight, unit type, admit source |
| **Vital Clinical Features** | 28 | Mean, min, max, std across 7 routine vital signs in 0-24h |
| **Lab Clinical Features** | 75 | First, last, min, max, mean across 15 selected labs in 0-24h |
| **Vital Quality Features** | 8 | Measurement counts per vital sign + total vital records in 0-24h |
| **Lab Quality Features** | 16 | Measurement counts per lab test + total lab records in 0-24h |
| **Meta Quality Indicators** | 3 | `available_vital_count`, `available_lab_count`, `missing_feature_count` |
| **Total Features** | **138** | **111 Clinical + 27 Data Quality** |

---

## 2. Validation & Quality Assertions

- **One Row Per ICU Stay:** True (1,403 unique stays)
- **Target Classes:** Only {0, 1} with 8.70% mortality prevalence
- **Anti-Leakage Audit:** 0 forbidden discharge/future columns found
- **Observation Window:** Strictly bounded to first 24 hours of ICU admission
