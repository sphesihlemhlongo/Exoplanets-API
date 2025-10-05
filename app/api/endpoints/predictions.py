from fastapi import APIRouter, HTTPException, Depends
from typing import List
from datetime import datetime
import uuid

from models.schemas import (
    PredictionInput,
    BatchPredictionInput,
    PredictionOutput,
    BatchPredictionOutput,
    SHAPExplanation
)
from services.ml_service import ml_service
from services.shap_service import create_shap_service
from core.database import get_supabase

router = APIRouter()
shap_service = create_shap_service(ml_service)


@router.post("/predict", response_model=PredictionOutput)
async def predict_single(input_data: PredictionInput):
    """Make a prediction for a single exoplanet candidate"""
    try:
        prediction, confidence, probabilities = ml_service.predict(input_data.features)

        prediction_id = str(uuid.uuid4())

        result = PredictionOutput(
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities,
            prediction_id=prediction_id,
            timestamp=datetime.utcnow()
        )

        try:
            supabase = get_supabase()
            supabase.table('predictions').insert({
                'id': prediction_id,
                'features': input_data.features,
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': probabilities,
                'created_at': result.timestamp.isoformat()
            }).execute()
        except Exception as db_error:
            print(f"Database insert warning: {db_error}")

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(input_data: BatchPredictionInput):
    """Make predictions for multiple exoplanet candidates"""
    try:
        results = ml_service.batch_predict(input_data.batch_data)

        predictions = []
        exoplanet_count = 0
        false_positive_count = 0

        for i, (prediction, confidence, probabilities) in enumerate(results):
            if 'error' in probabilities:
                continue

            prediction_id = str(uuid.uuid4())

            pred_output = PredictionOutput(
                prediction=prediction,
                confidence=confidence,
                probabilities=probabilities,
                prediction_id=prediction_id,
                timestamp=datetime.utcnow()
            )

            predictions.append(pred_output)

            if prediction == 1:
                exoplanet_count += 1
            else:
                false_positive_count += 1

        return BatchPredictionOutput(
            predictions=predictions,
            total_count=len(predictions),
            exoplanet_count=exoplanet_count,
            false_positive_count=false_positive_count
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/explain/{prediction_id}", response_model=SHAPExplanation)
async def explain_prediction(prediction_id: str, input_data: PredictionInput):
    """Get SHAP explanation for a prediction"""
    try:
        explanations = shap_service.get_explanation(input_data.features)

        prediction, confidence, probabilities = ml_service.predict(input_data.features)

        return SHAPExplanation(
            prediction_id=prediction_id,
            feature_contributions=explanations,
            base_value=0.5,
            prediction_value=probabilities.get('exoplanet', 0.5)
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/feature-importance")
async def get_feature_importance():
    """Get feature importance from the model"""
    try:
        importance = ml_service.get_feature_importance()
        return {"feature_importance": importance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    return {
        "model_type": "Stacked Ensemble",
        "base_models": ["Random Forest", "Gradient Boosting"],
        "meta_model": "Logistic Regression",
        "features": ml_service.feature_names,
        "is_trained": ml_service.is_trained
    }
