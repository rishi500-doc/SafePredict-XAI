# SafePredict-XAI: Phase 2 Data Validation & Quality Summary Report

**Generated on:** Auto-generated validation run  
**Target Window:** First 24 Hours of ICU admission (0 to 1440 minutes)

---

## 1. Technical & Schema Validation
- **Schema Integrity:** PASSED (All required columns verified in patient, lab, and vitalPeriodic tables).
- **Primary Key Uniqueness:** PASSED (0 duplicate `patientunitstayid` in `patient.csv` across 2,520 stays).
- **Orphan Record Check:** PASSED (0 orphan records in `lab.csv` and `vitalPeriodic.csv`).
- **Repeated Measurements:** 2,267 lab rows (0.52%) and 2 vital timestamps have repeated entries, representing continuous telemetry and simultaneous tube orders.

---

## 2. Target Outcome Validation
- **Outcome Column:** `hospitaldischargestatus`
- **Alive (Negative / 0):** 2,280 (90.48%)
- **Expired (Positive / 1):** 212 (8.41%)
- **Missing / Unrecorded:** 28 (1.11%)
- **Mortality Prevalence Among Known Outcomes:** **8.51%** (212 / 2,492)

---

## 3. Temporal Distribution (Observation Windows)
- **Primary 24-Hour Modeling Window (0 - 1440 min):**
  - Lab records: **98,870** (22.75%)
  - Vital records: **553,846** (33.88%)
- **Pre-ICU Window (< 0 min):**
  - Lab records: 97,488 (22.43%)
  - Vital records: 1,650 (0.10%)
- **Post-24h Stay (> 1440 min):**
  - Lab records: 238,302 (54.82%)
  - Vital records: 1,079,464 (66.02%)

---

## 4. Encounter-Level Quality Flags (2,520 ICU Stays)
| Quality Flag | Flagged Stays | Flagged % | Clinical Rationale |
| :--- | :--- | :--- | :--- |
| `flag_missing_demographics` | 40 | 1.59% | Stays missing age, gender, or ethnicity. |
| `flag_missing_target` | 28 | 1.11% | Unrecorded hospital discharge status (excluded from supervised model training). |
| `flag_missing_lab_data` | 173 | 6.87% | Stays with 0 lab measurements in the 24-hour observation window. |
| `flag_missing_vital_data` | 150 | 5.95% | Stays with 0 vital measurements in the 24-hour observation window. |
| `flag_temporal_issue` | 65 | 2.58% | Stays with 0 data across both labs and vitals in the 24-hour window. |
| `flag_possible_invalid_value` | 500 | 19.84% | Stays containing at least one out-of-plausibility lab or vital in 24h window. |

---

## 5. Transition to Phase 3 (Feature Engineering)
1. **Window Filtering:** Extract aggregations (mean, min, max, std, first, last, trend) strictly from the 0-1440 minute window.
2. **Plausibility Clipping / Masking:** Clip or mask physiological artifacts (negative blood pressures, out-of-bounds vitals) to clinical boundaries prior to computing statistical aggregates.
3. **Informative Missingness:** Preserve missingness indicators as feature columns rather than naive global imputation.
4. **Target Encoding:** Encode `hospitaldischargestatus` into binary `[0, 1]` and filter unlabelled records for training.
