"""
FastAPI Microservice for Real-Time Machine Learning Inference
Production REST API serving the Credit Default Risk Champion Model.
"""

import time
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas import (
    ApplicantFeatures,
    BatchApplicantFeatures,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse
)
from app.model_loader import get_model_service, MODEL_VERSION

START_TIME = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Preload champion pipeline into memory
    print("Initiating ML Service Startup: Loading champion model bundle...")
    service = get_model_service()
    if not service.is_loaded:
        raise RuntimeError("Failed to load champion model during startup lifespan.")
    print(f"Service ready. Loaded {MODEL_VERSION}.")
    yield
    # Shutdown: Clean up resources
    print("Shutting down ML inference service.")

app = FastAPI(
    title="Credit Default Risk Prediction API",
    description=(
        "Production-grade Real-Time ML Inference REST API serving the "
        "Credit Default Champion Model (ColumnTransformer + LightGBM). "
        "Provides real-time underwriting risk stratification, probability estimation, "
        "and automated decisioning."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Metadata"])
def root():
    """
    Root endpoint returning service identity, documentation links, and health status.
    """
    return {
        "service": "Credit Default Risk Inference Microservice",
        "model_version": MODEL_VERSION,
        "status": "ONLINE",
        "documentation": "/docs",
        "health_check": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """
    Liveness and readiness probe for Docker / Kubernetes container orchestration.
    """
    service = get_model_service()
    uptime = time.time() - START_TIME
    
    return HealthResponse(
        status="HEALTHY" if service.is_loaded else "UNHEALTHY",
        model_loaded=service.is_loaded,
        model_version=MODEL_VERSION,
        uptime_seconds=round(uptime, 2),
        environment=os.getenv("ENVIRONMENT", "production")
    )

@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
def predict(applicant: ApplicantFeatures):
    """
    Real-Time Single Applicant Inference:
    Takes raw loan applicant attributes and returns predicted default probability,
    classification decision, confidence score, and risk tier.
    """
    try:
        service = get_model_service()
        response = service.predict_one(applicant)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failure: {str(e)}"
        )

@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
def predict_batch(batch: BatchApplicantFeatures):
    """
    High-Throughput Batch Inference:
    Scores up to 500 applicants in a single optimized matrix forward pass.
    """
    try:
        service = get_model_service()
        results = service.predict_batch(batch.applicants)
        return BatchPredictionResponse(
            total_records=len(results),
            predictions=results,
            model_version=MODEL_VERSION
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failure: {str(e)}"
        )
