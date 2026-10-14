"""
Pydantic Schemas for Real-Time ML Inference REST API
Defines input payloads, validation rules, and response contracts.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

class ApplicantFeatures(BaseModel):
    """
    Schema for a single loan applicant's credit risk features.
    """
    annual_income: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=5_000_000.0,
        description="Self-reported or verified gross annual income in USD."
    )
    loan_amount: float = Field(
        ...,
        ge=500.0,
        le=100_000.0,
        description="Principal amount of credit requested in USD."
    )
    interest_rate: float = Field(
        ...,
        ge=1.0,
        le=45.0,
        description="Assigned annual percentage interest rate (%)."
    )
    installment: float = Field(
        ...,
        ge=10.0,
        le=5_000.0,
        description="Expected monthly installment payment in USD."
    )
    debt_to_income_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Total monthly debt payments divided by monthly gross income (%)."
    )
    revolving_balance: float = Field(
        default=0.0,
        ge=0.0,
        description="Total active credit revolving balance in USD."
    )
    revolving_utilization: float = Field(
        default=0.0,
        ge=0.0,
        le=200.0,
        description="Revolving line utilization rate (%)."
    )
    total_accounts: int = Field(
        ...,
        ge=1,
        le=150,
        description="Total number of historical credit lines on file."
    )
    open_accounts: int = Field(
        ...,
        ge=0,
        le=100,
        description="Total number of currently open credit lines."
    )
    delinq_2yrs: int = Field(
        default=0,
        ge=0,
        description="Number of 30+ days past-due delinquency incidences in past 2 years."
    )
    fico_score: int = Field(
        ...,
        ge=300,
        le=850,
        description="FICO credit score at loan origination."
    )
    months_since_last_delinq: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Months elapsed since the applicant's last delinquency (null if never delinquent)."
    )
    home_ownership: str = Field(
        default="RENT",
        description="Residential status: RENT, MORTGAGE, OWN, or OTHER."
    )
    verification_status: str = Field(
        default="Not Verified",
        description="Income verification status: Verified, Source Verified, or Not Verified."
    )
    loan_term_months: str = Field(
        default="36 months",
        description="Loan duration: '36 months' or '60 months'."
    )
    employment_industry: str = Field(
        default="Finance",
        description="Primary industry or sector of employment."
    )
    loan_purpose: str = Field(
        default="debt_consolidation",
        description="Stated purpose for loan credit."
    )
    sub_grade: str = Field(
        default="B2",
        description="Assigned risk sub-grade (A1 through G5)."
    )
    credit_grade: str = Field(
        default="B",
        description="Broad credit grade category (A through G)."
    )
    education_level: str = Field(
        default="Bachelor",
        description="Highest achieved education level."
    )

    @field_validator("home_ownership")
    @classmethod
    def validate_home_ownership(cls, v: str) -> str:
        valid = {"RENT", "MORTGAGE", "OWN", "OTHER"}
        v_clean = v.strip().upper()
        if v_clean not in valid:
            raise ValueError(f"home_ownership must be one of {valid}, got '{v}'")
        return v_clean

    model_config = {
        "json_schema_extra": {
            "example": {
                "annual_income": 85000.0,
                "loan_amount": 18000.0,
                "interest_rate": 12.5,
                "installment": 602.0,
                "debt_to_income_ratio": 15.8,
                "revolving_balance": 9400.0,
                "revolving_utilization": 34.2,
                "total_accounts": 24,
                "open_accounts": 11,
                "delinq_2yrs": 0,
                "fico_score": 725,
                "months_since_last_delinq": None,
                "home_ownership": "MORTGAGE",
                "verification_status": "Verified",
                "loan_term_months": "36 months",
                "employment_industry": "Technology",
                "loan_purpose": "debt_consolidation",
                "sub_grade": "B3",
                "credit_grade": "B",
                "education_level": "Bachelor"
            }
        }
    }

class BatchApplicantFeatures(BaseModel):
    """
    Schema for batch prediction request.
    """
    applicants: List[ApplicantFeatures] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of applicant feature dictionaries for batch scoring."
    )

class PredictionResponse(BaseModel):
    """
    Contract for real-time single applicant inference.
    """
    prediction: int = Field(..., description="0 = Non-Default (Approved), 1 = Default (High Risk).")
    decision: str = Field(..., description="Actionable recommendation: 'APPROVED' or 'REVIEW_REQUIRED'.")
    default_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted posterior probability of loan default.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model decision confidence percentage.")
    risk_tier: str = Field(..., description="Categorical risk stratification: 'LOW', 'MEDIUM', or 'HIGH'.")
    model_version: str = Field(..., description="Serving model identifier and architecture version.")
    evaluated_at: datetime = Field(default_factory=datetime.utcnow, description="Evaluation timestamp (UTC).")

class BatchPredictionResponse(BaseModel):
    """
    Contract for batch prediction response.
    """
    total_records: int
    predictions: List[PredictionResponse]
    model_version: str
    processed_at: datetime = Field(default_factory=datetime.utcnow)

class HealthResponse(BaseModel):
    """
    Health check payload for container orchestration.
    """
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    environment: str
