"""
Reusable Scikit-Learn Preprocessing Pipeline Module
Domain: Credit Default Risk Tabular Preprocessing
Adheres strictly to leak-free design principles: fit only on training data.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder
)

def get_feature_groups():
    """
    Returns dictionaries of feature column names mapped by their modalities.
    """
    numerical_features = [
        "annual_income",
        "loan_amount",
        "interest_rate",
        "installment",
        "debt_to_income_ratio",
        "revolving_balance",
        "revolving_utilization",
        "total_accounts",
        "open_accounts",
        "fico_score",
        "months_since_last_delinq"
    ]

    low_cardinality_nominal = [
        "home_ownership",
        "verification_status",
        "loan_term_months"
    ]

    high_cardinality_nominal = [
        "employment_industry",
        "loan_purpose",
        "sub_grade"
    ]

    ordinal_features = [
        "credit_grade",
        "education_level"
    ]

    ordinal_categories = [
        ["A", "B", "C", "D", "E", "F", "G"],
        ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
    ]

    return {
        "numerical": numerical_features,
        "low_cardinality": low_cardinality_nominal,
        "high_cardinality": high_cardinality_nominal,
        "ordinal": ordinal_features,
        "ordinal_categories": ordinal_categories
    }

def build_preprocessing_pipeline(use_robust_scaling=True, target_encoding_smooth="auto"):
    """
    Constructs an end-to-end, leak-free Scikit-Learn ColumnTransformer pipeline.

    Parameters:
    -----------
    use_robust_scaling : bool
        If True, uses RobustScaler (resilient to heavy-tailed income and debt outliers).
        If False, uses StandardScaler.
    target_encoding_smooth : float or "auto"
        Smoothing parameter for TargetEncoder to prevent overfitting on rare categories.

    Returns:
    --------
    sklearn.compose.ColumnTransformer
    """
    groups = get_feature_groups()

    # 1. Numerical Pipeline: Median Imputation + Outlier-resilient Scaling
    scaler = RobustScaler() if use_robust_scaling else StandardScaler()
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", scaler)
    ])

    # 2. Low-Cardinality Nominal Pipeline: Mode Imputation + One-Hot Encoding
    low_card_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])

    # 3. High-Cardinality Nominal Pipeline: Mode Imputation + Supervised Target Encoding
    # TargetEncoder uses cross-fitting internally during fit_transform to eliminate in-sample target leakage
    high_card_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("target_enc", TargetEncoder(smooth=target_encoding_smooth, cv=5, random_state=42))
    ])

    # 4. Ordinal Pipeline: Explicit Category Order Preservation
    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal_enc", OrdinalEncoder(
            categories=groups["ordinal_categories"],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    # Combine into Master ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, groups["numerical"]),
            ("cat_low", low_card_pipeline, groups["low_cardinality"]),
            ("cat_high", high_card_pipeline, groups["high_cardinality"]),
            ("cat_ord", ordinal_pipeline, groups["ordinal"])
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor

if __name__ == "__main__":
    preprocessor = build_preprocessing_pipeline()
    print("Successfully built ColumnTransformer pipeline with 4 modality branches:")
    for name, pipe, cols in preprocessor.transformers:
        print(f" - Branch '{name}': {len(cols)} columns -> {[s[0] for s in pipe.steps]}")
