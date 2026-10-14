# Problem 3: Robust, Leak-Free ML Preprocessing Pipeline

This directory contains the production-grade Scikit-Learn preprocessing pipeline implementation for complex tabular data (Credit Default Risk).

---

## Architecture & Modalities

```mermaid
flowchart TD
    Raw[Raw Tabular Dataset: Credit Default Risk (N=10,000)] --> Split[Strict Stratified Split (80% Train, 20% Test)]
    Split --> Train[X_train, y_train]
    Split --> Test[X_test, y_test (Vaulted)]
    
    subgraph Preprocessor [Scikit-Learn ColumnTransformer]
        Num[Numerical Branch:<br/>SimpleImputer(median) + RobustScaler]
        Low[Low-Cardinality Categoricals:<br/>SimpleImputer(most_frequent) + OneHotEncoder]
        High[High-Cardinality Categoricals:<br/>SimpleImputer(most_frequent) + TargetEncoder(cv=5)]
        Ord[Ordinal Categoricals:<br/>SimpleImputer(most_frequent) + OrdinalEncoder]
    end
    
    Train --> Preprocessor
    Preprocessor -->|fit_transform(X_train, y_train)| TransTrain[Processed Training Matrix]
    Preprocessor -->|transform(X_test)| TransTest[Processed Test Matrix]
    
    TransTrain --> Corr[Correlation Analysis & Multicollinearity Heatmaps]
    TransTrain --> MI[Mutual Information Ranking]
    TransTrain --> RF[Random Forest Tree-based Feature Importance]
    TransTrain --> Model[Classifier Training & Evaluation on TransTest]
```

---

## Deliverables & Key Files

| File | Purpose |
| :--- | :--- |
| **`preprocessing_pipeline.ipynb`** | Executable Jupyter Notebook demonstrating dataset ingestion, leak-free splitting, ColumnTransformer execution, correlation analysis, and feature importance rankings. |
| **`pipeline_module.py`** | Reusable Python module exposing `build_preprocessing_pipeline()` and feature grouping definitions. |
| **`credit_default_data.csv`** | Curated 10,000-applicant credit default dataset with mixed feature modalities and realistic missingness. |

---

## Key Pipeline Features

1. **Leak-Free Guarantee:**
   - Train/test splitting occurs strictly **before** any fitting.
   - Transformers (`StandardScaler`, `RobustScaler`, `OneHotEncoder`, `TargetEncoder`) learn parameters **exclusively from `X_train`**.
   - Target encoding utilizes out-of-fold cross-fitting (`cv=5`) to eliminate in-sample target contamination.
2. **Missing Value Strategy:**
   - Median imputation with missingness indicators for continuous variables.
   - Most-frequent imputation for categorical variables.
3. **Multicollinearity & Importance Analysis:**
   - Pearson & Spearman correlation matrices.
   - Information-theoretic Mutual Information (`mutual_info_classif`).
   - Non-linear tree-based Gini importance via `RandomForestClassifier`.
