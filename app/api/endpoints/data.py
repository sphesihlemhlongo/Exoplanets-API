from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List
import pandas as pd

from models.schemas import DataQualityReport
from services.data_service import data_service

router = APIRouter()


@router.post("/upload/validate")
async def validate_upload(file: UploadFile = File(...)):
    """Validate an uploaded CSV file"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        content = await file.read()
        content_str = content.decode('utf-8')

        df = data_service.parse_csv_file(content_str)

        is_valid, errors = data_service.validate_dataframe(df)

        return {
            "valid": is_valid,
            "errors": errors,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/upload/quality-report", response_model=DataQualityReport)
async def generate_quality_report(file: UploadFile = File(...)):
    """Generate a data quality report for an uploaded file"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        content = await file.read()
        content_str = content.decode('utf-8')

        df = data_service.parse_csv_file(content_str)

        report = data_service.generate_quality_report(df)

        return DataQualityReport(**report)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/upload/process")
async def process_upload(file: UploadFile = File(...)):
    """Process an uploaded file and return preprocessed data"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        content = await file.read()
        content_str = content.decode('utf-8')

        df = data_service.parse_csv_file(content_str)

        is_valid, errors = data_service.validate_dataframe(df)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Validation failed: {'; '.join(errors)}")

        df_processed = data_service.preprocess_dataframe(df)

        batch_data = df_processed.to_dict('records')

        return {
            "processed_data": batch_data,
            "row_count": len(batch_data),
            "message": "Data processed successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/sample-data")
async def get_sample_data():
    """Get sample exoplanet data"""
    samples = data_service.get_sample_data()
    return {"samples": samples, "count": len(samples)}


@router.get("/feature-schema")
async def get_feature_schema():
    """Get the required feature schema"""
    return {
        "required_features": data_service.required_features,
        "feature_descriptions": {
            "koi_period": "Orbital period (days)",
            "koi_time0bk": "Transit epoch (BKJD)",
            "koi_impact": "Impact parameter",
            "koi_duration": "Transit duration (hours)",
            "koi_depth": "Transit depth (ppm)",
            "koi_prad": "Planetary radius (Earth radii)",
            "koi_teq": "Equilibrium temperature (K)",
            "koi_insol": "Insolation flux (Earth flux)",
            "koi_steff": "Stellar effective temperature (K)",
            "koi_slogg": "Stellar surface gravity (log10(cm/s²))",
            "koi_srad": "Stellar radius (Solar radii)"
        }
    }
