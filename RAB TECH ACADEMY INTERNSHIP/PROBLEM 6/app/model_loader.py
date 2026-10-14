"""
Model Loader & Inference Service
Loads the serialized champion pipeline and encapsulates scoring logic.
"""

import os
import joblib
import pandas as pd
from typing import List, Dict, Any
from app.schemas import ApplicantFeatures, PredictionResponse

MODEL_VERSION = "1.0.0-lgbm-champion"
DEFAULT_THRESHOLD = 0.50

class ModelService:
    def __init__(self, model_path: str = None):
        if model_path is None:
            # Look in standard locations
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "model", "champion_model.joblib")
            
        self.model_path = model_path
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Champion model artifact not found at: {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        print(f"Loaded champion pipeline from {self.model_path}")

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def _features_to_df(self, applicants: List[ApplicantFeatures]) -> pd.DataFrame:
        data = [a.model_dump() for a in applicants]
        return pd.DataFrame(data)

    def predict_one(self, applicant: ApplicantFeatures) -> PredictionResponse:
        results = self.predict_batch([applicant])
        return results[0]

    def predict_batch(self, applicants: List[ApplicantFeatures]) -> List[PredictionResponse]:
        df = self._features_to_df(applicants)
        
        # Forward pass through pipeline (ColumnTransformer + LightGBM)
        probs = self.pipeline.predict_proba(df)[:, 1]
        
        responses = []
        for prob in probs:
            is_default = int(prob >= DEFAULT_THRESHOLD)
            
            # Risk tiering
            if prob < 0.20:
                tier = "LOW"
            elif prob < 0.50:
                tier = "MEDIUM"
            else:
                tier = "HIGH"
                
            decision = "REVIEW_REQUIRED" if is_default == 1 else "APPROVED"
            confidence = prob if is_default == 1 else 1.0 - prob
            
            responses.append(PredictionResponse(
                prediction=is_default,
                decision=decision,
                default_probability=round(float(prob), 4),
                confidence=round(float(confidence), 4),
                risk_tier=tier,
                model_version=MODEL_VERSION
            ))
            
        return responses

# Global singleton
_model_service = None

def get_model_service() -> ModelService:
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service
