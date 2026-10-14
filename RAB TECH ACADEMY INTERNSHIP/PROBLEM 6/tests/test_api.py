"""
Automated Unit & Integration Test Suite for ML Inference API
Uses FastAPI TestClient to validate endpoints, schemas, error handling, and predictions.
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient

# Ensure app package is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_root_endpoint(client):
    """Test root endpoint returns metadata and status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"
    assert "model_version" in data
    assert data["documentation"] == "/docs"

def test_health_check_endpoint(client):
    """Test health check returns healthy status and loaded model."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["model_loaded"] is True
    assert data["uptime_seconds"] >= 0.0

def test_predict_prime_applicant(client):
    """Test real-time prediction for a low-risk, creditworthy applicant."""
    payload = {
        "annual_income": 95000.0,
        "loan_amount": 12000.0,
        "interest_rate": 9.5,
        "installment": 384.0,
        "debt_to_income_ratio": 12.0,
        "revolving_balance": 6500.0,
        "revolving_utilization": 22.0,
        "total_accounts": 20,
        "open_accounts": 10,
        "delinq_2yrs": 0,
        "fico_score": 760,
        "months_since_last_delinq": None,
        "home_ownership": "MORTGAGE",
        "verification_status": "Verified",
        "loan_term_months": "36 months",
        "employment_industry": "Technology",
        "loan_purpose": "home_improvement",
        "sub_grade": "A3",
        "credit_grade": "A",
        "education_level": "Master"
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Assert response schema
    assert "prediction" in data
    assert "decision" in data
    assert "default_probability" in data
    assert "confidence" in data
    assert "risk_tier" in data
    
    # Prime applicant should have low default probability and be APPROVED
    assert data["prediction"] == 0
    assert data["decision"] == "APPROVED"
    assert data["risk_tier"] == "LOW"
    assert data["default_probability"] < 0.25

def test_predict_high_risk_applicant(client):
    """Test real-time prediction for a high-risk applicant."""
    payload = {
        "annual_income": 32000.0,
        "loan_amount": 28000.0,
        "interest_rate": 26.5,
        "installment": 1120.0,
        "debt_to_income_ratio": 42.0,
        "revolving_balance": 24000.0,
        "revolving_utilization": 92.5,
        "total_accounts": 15,
        "open_accounts": 7,
        "delinq_2yrs": 3,
        "fico_score": 595,
        "months_since_last_delinq": 4.0,
        "home_ownership": "RENT",
        "verification_status": "Not Verified",
        "loan_term_months": "60 months",
        "employment_industry": "Hospitality",
        "loan_purpose": "small_business",
        "sub_grade": "F4",
        "credit_grade": "F",
        "education_level": "High School"
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # High risk applicant should be flagged for review
    assert data["prediction"] == 1
    assert data["decision"] == "REVIEW_REQUIRED"
    assert data["risk_tier"] == "HIGH"
    assert data["default_probability"] > 0.50

def test_predict_missing_optional_fields(client):
    """Test that missing optional fields (imputer test) are handled seamlessly."""
    payload = {
        "annual_income": None, # Missing income
        "loan_amount": 15000.0,
        "interest_rate": 14.0,
        "installment": 512.0,
        "debt_to_income_ratio": None, # Missing DTI
        "revolving_balance": 8000.0,
        "revolving_utilization": 45.0,
        "total_accounts": 18,
        "open_accounts": 9,
        "delinq_2yrs": 0,
        "fico_score": 680,
        "months_since_last_delinq": None, # Clean delinquency history
        "home_ownership": "RENT",
        "verification_status": "Not Verified",
        "loan_term_months": "36 months",
        "employment_industry": "Retail",
        "loan_purpose": "debt_consolidation",
        "sub_grade": "C1",
        "credit_grade": "C",
        "education_level": "Bachelor"
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["default_probability"] <= 1.0

def test_predict_validation_error(client):
    """Test that invalid field values trigger HTTP 422 Unprocessable Entity."""
    # FICO score of 9999 violates ge=300, le=850 constraint
    invalid_payload = {
        "annual_income": 60000.0,
        "loan_amount": 10000.0,
        "interest_rate": 12.0,
        "installment": 332.0,
        "fico_score": 9999, # INVALID
        "total_accounts": 10,
        "open_accounts": 5
    }
    
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data

def test_batch_prediction(client):
    """Test batch prediction endpoint with multiple applicants."""
    applicant_a = {
        "annual_income": 110000.0, "loan_amount": 10000.0, "interest_rate": 8.5,
        "installment": 315.0, "debt_to_income_ratio": 10.5, "revolving_balance": 4000.0,
        "revolving_utilization": 18.0, "total_accounts": 25, "open_accounts": 12,
        "delinq_2yrs": 0, "fico_score": 780, "months_since_last_delinq": None,
        "home_ownership": "OWN", "verification_status": "Verified",
        "loan_term_months": "36 months", "employment_industry": "Finance",
        "loan_purpose": "credit_card", "sub_grade": "A1", "credit_grade": "A",
        "education_level": "Doctorate"
    }
    applicant_b = {
        "annual_income": 35000.0, "loan_amount": 25000.0, "interest_rate": 24.0,
        "installment": 985.0, "debt_to_income_ratio": 38.0, "revolving_balance": 19000.0,
        "revolving_utilization": 88.0, "total_accounts": 12, "open_accounts": 6,
        "delinq_2yrs": 1, "fico_score": 620, "months_since_last_delinq": 8.0,
        "home_ownership": "RENT", "verification_status": "Not Verified",
        "loan_term_months": "60 months", "employment_industry": "Construction",
        "loan_purpose": "debt_consolidation", "sub_grade": "E2", "credit_grade": "E",
        "education_level": "Associate"
    }
    
    payload = {"applicants": [applicant_a, applicant_b]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["decision"] == "APPROVED"
    assert data["predictions"][1]["decision"] == "REVIEW_REQUIRED"
