# SafePredict-XAI: Phase 3 Cohort Construction & Cleaning Summary

**Landmark Prediction Horizon:** 24 Hours after ICU admission ($t = 1440$ minutes)  
**Target Variable:** `hospital_mortality` (0 = Alive, 1 = Expired)

---

## 1. Analytical Cohort Flow (Attrition Summary)

| Step | Cohort Definition | Retained Stays | Expired (Deaths) | Alive | Mortality Prevalence | Records Excluded |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **0. Raw Stays** | All admissions in `patient.csv` | **2,520** | 212 | 2,280 | 8.41% | - |
| **1. Known Outcome** | Exclude unrecorded discharge status | **2,492** | 212 | 2,280 | **8.51%** | 28 unrecorded (1.11%) |
| **2. Index ICU Stay** | Deduplicate repeat stays (`unitvisitnumber == 1`) | **2,094** | 177 | 1,917 | **8.45%** | 398 repeat stays (15.97%) |
| **3. 24h Landmark** | Retain stays with LOS $\ge 24$h (`unitdischargeoffset` $\ge 1440$) | **1,403** | 122 | 1,281 | **8.70%** | 691 short stays (<24h) |

---

## 2. Impact of 24-Hour Landmarking on Mortality Prevalence
- **Stays ending before 24 hours:** 691 stays (33.00% of index stays).
  - Hospital mortality among short stays: **7.96%** (55 deaths; 39 died directly inside the ICU in <24h).
  - Patients discharged alive before 24h: 636 patients.
- **Landmark-Eligible Cohort ($N = 1,403$):**
  - **Mortality Prevalence:** **8.70%** (122 Expired vs 1,281 Alive).
  - *Clinical Insight:* Excluding patients discharged or deceased prior to 24h focuses predictive modeling precisely on patients actively receiving care at the 24-hour decision point.

---

## 3. 0-24h Data Availability & Coverage
- **Temporal Window:** Strictly bounded to $0 \le t \le 1440$ minutes ($0$ post-24h or pre-ICU records).
- **Periodic Vitals (0-24h):**
  - Total measurements: **375,584**
  - Patients with at least 1 vital sign: **1,364** (97.22% coverage).
- **Lab Tests (0-24h):**
  - Total measurements: **65,866**
  - Patients with at least 1 lab order: **1,382** (98.50% coverage).

---

## 4. Leakage Prevention Safeguards
1. **Target Isolation:** `hospital_mortality` is strictly isolated as the target label.
2. **Eligibility Markers:** `unitdischargeoffset`, `hospitaldischargeoffset`, and `unitdischargestatus` are excluded from all subsequent predictive feature matrices.
3. **Temporal Invariants:** All aggregations in subsequent phases are constrained strictly to $t \le 1440$ min.
