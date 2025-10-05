from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional
import uuid

from models.schemas import (
    OptimizationConfig,
    OptimizationStatus,
    OptimizationResult
)
from services.optimization_service import optimization_service
from core.database import get_supabase

router = APIRouter()


def run_optimization_task(study_id: str, config: OptimizationConfig):
    """Background task for running optimization"""
    optimization_service.start_optimization(
        study_id=study_id,
        n_trials=config.n_trials,
        timeout=config.timeout,
        metric=config.optimization_metric
    )

    try:
        results = optimization_service.get_study_results(study_id)
        if results:
            supabase = get_supabase()
            supabase.table('optimization_studies').insert({
                'id': study_id,
                'config': config.dict(),
                'results': results,
                'status': 'completed',
                'created_at': results['timestamp'].isoformat()
            }).execute()
    except Exception as e:
        print(f"Database insert warning for optimization: {e}")


@router.post("/optimize/start", response_model=dict)
async def start_optimization(
    config: OptimizationConfig,
    background_tasks: BackgroundTasks
):
    """Start a new hyperparameter optimization study"""
    try:
        study_id = str(uuid.uuid4())

        background_tasks.add_task(run_optimization_task, study_id, config)

        return {
            "study_id": study_id,
            "message": "Optimization started",
            "config": config.dict()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimize/status/{study_id}", response_model=OptimizationStatus)
async def get_optimization_status(study_id: str):
    """Get the status of an optimization study"""
    status = optimization_service.get_study_status(study_id)

    if not status:
        raise HTTPException(status_code=404, detail="Study not found")

    return OptimizationStatus(**status)


@router.get("/optimize/results/{study_id}", response_model=OptimizationResult)
async def get_optimization_results(study_id: str):
    """Get the results of a completed optimization study"""
    results = optimization_service.get_study_results(study_id)

    if not results:
        raise HTTPException(status_code=404, detail="Study not found")

    return OptimizationResult(**results)


@router.get("/optimize/importance/{study_id}")
async def get_parameter_importance(study_id: str):
    """Get parameter importance for an optimization study"""
    importance = optimization_service.get_parameter_importance(study_id)

    if importance is None:
        raise HTTPException(status_code=404, detail="Study not found or no importance data available")

    return {"parameter_importance": importance}


@router.get("/optimize/studies")
async def list_optimization_studies():
    """List all optimization studies"""
    try:
        supabase = get_supabase()
        response = supabase.table('optimization_studies').select('*').order('created_at', desc=True).limit(50).execute()

        return {"studies": response.data if response.data else []}
    except Exception as e:
        return {"studies": [], "error": str(e)}
