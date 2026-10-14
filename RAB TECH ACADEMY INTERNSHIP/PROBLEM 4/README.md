# Problem 4: Supervised Classification Modeling, Tuning & Champion Selection

This directory contains the machine learning modeling, hyperparameter optimization, model benchmarking, and serialized champion model deployment pipeline for Credit Default Risk prediction.

---

## Benchmark Results Summary

| Model Architecture | 5-Fold CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Test F1-Score | Brier Score | Latency (μs/sample) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (GBDT)** | **0.8682** | **0.8695** | **0.6720** | **0.6184** | **0.1124** | **21.4** | 🏆 **CHAMPION MODEL** |
| **Random Forest** | 0.8641 | 0.8642 | 0.6581 | 0.6042 | 0.1158 | 124.6 | Competitive Runner-Up |
| **Neural Network (MLP)** | 0.8412 | 0.8418 | 0.6015 | 0.5420 | 0.1248 | 32.1 | Non-linear Baseline |
| **Logistic Regression** | 0.8350 | 0.8341 | 0.5872 | 0.5218 | 0.1285 | 18.2 | Linear Baseline |

---

## Deliverables & Key Files

| File | Description |
| :--- | :--- |
| **`model_tuning_and_comparison.ipynb`** | Fully executed 23-cell Jupyter Notebook comparing 4 architectures, running 5-fold Stratified GridSearchCV, plotting ROC/PR curves and confusion matrices, and verifying champion inference. |
| **`champion_model.joblib`** | Serialized production pipeline bundle containing the leak-free `ColumnTransformer` preprocessor + tuned `LGBMClassifier` champion model ready for immediate scoring. |
| **`pipeline_module.py`** | Reusable Scikit-Learn `ColumnTransformer` builder used for automated data ingestion and transformation. |
| **`credit_default_data.csv`** | 10,000-applicant Credit Default Risk dataset ($N=10,000$, 21.2% base default rate). |
| **`generate_problem4_notebook.py`** | Script used to construct the clean, validated Jupyter Notebook. |

---

## Champion Model Justification

1. **Predictive Performance:** LightGBM achieved the highest Test ROC-AUC (0.8695) and PR-AUC (0.6720), capturing complex tabular interactions between sub-grades, debt-to-income, and interest rates.
2. **Probability Calibration:** Achieved the lowest Brier score (0.1124), providing accurate posterior probabilities for credit risk pricing.
3. **Pareto Operational Efficiency:** Microsecond inference latency (~21 $\mu\text{s}$ per record) enables real-time underwriting scoring at over $5\times$ the speed of Random Forest.

---

## Real-Time Inference Usage

```python
import joblib
import pandas as pd

# 1. Load serialized champion pipeline
pipeline = joblib.load("PROBLEM 4/champion_model.joblib")

# 2. Score raw, uncleaned applicant record
raw_applicant = pd.DataFrame([{
    "annual_income": 75000.0,
    "loan_amount": 15000.0,
    "interest_rate": 11.5,
    "installment": 494.0,
    "debt_to_income_ratio": 16.2,
    "revolving_balance": 8200.0,
    "revolving_utilization": 38.5,
    "total_accounts": 22,
    "open_accounts": 12,
    "delinq_2yrs": 0,
    "fico_score": 730,
    "months_since_last_delinq": None, # Missing values handled automatically
    "home_ownership": "MORTGAGE",
    "verification_status": "Verified",
    "loan_term_months": "36 months",
    "employment_industry": "Technology",
    "loan_purpose": "debt_consolidation",
    "sub_grade": "B2",
    "credit_grade": "B",
    "education_level": "Bachelor"
}])

# 3. Output prediction
default_probability = pipeline.predict_proba(raw_applicant)[0, 1]
print(f"Predicted Default Probability: {default_probability:.2%}")
```
