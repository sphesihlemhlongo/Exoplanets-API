from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class PredictionInput(BaseModel):
    features: Dict[str, float] = Field(..., description="Dictionary of feature names and values")

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "koi_period": 3.52474859,
                    "koi_time0bk": 170.53875,
                    "koi_impact": 0.146,
                    "koi_duration": 2.95750,
                    "koi_depth": 615.8,
                    "koi_prad": 2.26,
                    "koi_teq": 793,
                    "koi_insol": 93.59,
                    "koi_steff": 5455,
                    "koi_slogg": 4.467,
                    "koi_srad": 0.927
                }
            }
        }


class BatchPredictionInput(BaseModel):
    batch_data: List[Dict[str, float]] = Field(..., description="List of feature dictionaries")


class PredictionOutput(BaseModel):
    prediction: int = Field(..., description="Binary classification: 1 for exoplanet, 0 for false positive")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    prediction_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]
    total_count: int
    exoplanet_count: int
    false_positive_count: int


class SHAPExplanation(BaseModel):
    prediction_id: str
    feature_contributions: Dict[str, float]
    base_value: float
    prediction_value: float


class OptimizationConfig(BaseModel):
    n_trials: int = Field(default=100, ge=10, le=1000)
    timeout: Optional[int] = Field(default=None, description="Timeout in seconds")
    optimization_metric: str = Field(default="accuracy", description="Metric to optimize")

    class Config:
        json_schema_extra = {
            "example": {
                "n_trials": 100,
                "timeout": 3600,
                "optimization_metric": "f1_score"
            }
        }


class OptimizationStatus(BaseModel):
    study_id: str
    status: str
    current_trial: int
    total_trials: int
    best_score: Optional[float]
    best_params: Optional[Dict[str, Any]]
    elapsed_time: float


class OptimizationResult(BaseModel):
    study_id: str
    best_score: float
    best_params: Dict[str, Any]
    n_trials: int
    optimization_history: List[Dict[str, Any]]
    timestamp: datetime


class HealthCheck(BaseModel):
    status: str
    version: str
    model_loaded: bool
    database_connected: bool


class DataQualityReport(BaseModel):
    total_rows: int
    total_features: int
    missing_values: Dict[str, int]
    feature_stats: Dict[str, Dict[str, float]]
    anomalies_detected: int
    quality_score: float
    recommendations: List[str]
