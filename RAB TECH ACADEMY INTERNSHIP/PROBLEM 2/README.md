# ML Problem Framing & Responsible Data Card
**Post-Discharge Transitional Care Management (TCM) 30-Day Readmission Risk**

This repository contains the complete production-grade machine learning problem framing, responsible data card, clinical risk register, and baseline evaluation notebook for high-stakes healthcare resource allocation.

---

## Deliverables & Proof Artifacts

| Document / Artifact | Description | Direct Link |
| :--- | :--- | :--- |
| **ML Problem-Framing Memo** | Defines the decision, prediction target, unit of observation, action window, justification scorecard (ML vs. heuristics vs. clinical scores), and asymmetric cost-utility framework. | [ML_PROBLEM_FRAMING_MEMO.md](file:///c:/Users/K%20Pranav%20Bhargav/OneDrive/Desktop/RAB%20TECH%20ACADEMY%20INTERNSHIP/ML_PROBLEM_FRAMING_MEMO.md) |
| **Responsible AI Data Card** | Documents provenance, legal basis (HIPAA Safe Harbor), sensitive attributes & dangerous proxies, missingness mechanisms (MCAR/MAR/MNAR), and data leakage audits. | [RESPONSIBLE_DATA_CARD.md](file:///c:/Users/K%20Pranav%20Bhargav/OneDrive/Desktop/RAB%20TECH%20ACADEMY%20INTERNSHIP/RESPONSIBLE_DATA_CARD.md) |
| **Clinical AI Risk Register** | Catalogs multi-dimensional risks (NIST AI RMF), allocative & representational harms, three-tier human triage architecture, programmatic abstention triggers, and automated circuit-breaker rollback protocols. | [RISK_REGISTER.md](file:///c:/Users/K%20Pranav%20Bhargav/OneDrive/Desktop/RAB%20TECH%20ACADEMY%20INTERNSHIP/RISK_REGISTER.md) |
| **Baseline Jupyter Notebook** | End-to-end executable notebook implementing patient-level group splits, the non-ML clinical LACE Index baseline, a regularized ML baseline, asymmetric cost threshold tuning, subgroup fairness audits, and abstention simulation. | [baseline_notebook.ipynb](file:///c:/Users/K%20Pranav%20Bhargav/OneDrive/Desktop/RAB%20TECH%20ACADEMY%20INTERNSHIP/baseline_notebook.ipynb) |
| **IDC-30 Curated Dataset** | Curated 5,000-encounter cohort ($N=3,800$ unique patients) reflecting clinical trajectories, labs with missingness, and 30-day outcomes. | [idc30_cohort_data.csv](file:///c:/Users/K%20Pranav%20Bhargav/OneDrive/Desktop/RAB%20TECH%20ACADEMY%20INTERNSHIP/idc30_cohort_data.csv) |

---

## Core Findings & Summary Highlights

1. **ML Justification:** Supervised ML delivers a **+0.081 ROC-AUC lift** (0.778 vs 0.697) and a **+0.139 PR-AUC lift** (0.493 vs 0.354) over the gold-standard clinical **LACE Index**.
2. **Asymmetric Cost Matrix:**
   - Cost of False Positive: **$450** (Nurse outreach + telemonitoring kit for a patient who wouldn't readmit).
   - Cost of False Negative: **$14,800** (Unprevented readmission + CMS penalization).
   - Operational Decision Threshold: **$\tau^* = 0.227$**.
   - Net Economic Value: **$193,160 saved per 1,000 discharges** over the non-ML baseline (~$2.9M annually for a 15,000-discharge health system).
3. **Equitable Resource Allocation:**
   - Disparate Impact Ratio across racial/ethnic minority groups exceeds **0.96** (well above the 0.80 Four-Fifths legal standard).
   - Equal Opportunity: True Positive Rates remain balanced between 72% and 80% across demographic cohorts.
4. **Safety & Fallback:**
   - Active abstention intercepting encounters with $>30\%$ physiological missingness.
   - Three-tier triage funnel directing high confidence to TCM, borderline to nurse review, and low risk to routine discharge.
   - Automated circuit-breaker rolling back 100% of inference to the LACE Index if $PSI \ge 0.25$ or $DIR < 0.80$.

---

## How to Run & Verify

```bash
# 1. Activate virtual environment
.venv\Scripts\activate

# 2. Re-execute notebook or inspect outputs
python -m nbconvert --to notebook --execute --inplace baseline_notebook.ipynb
```
