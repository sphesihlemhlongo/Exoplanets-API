from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from core.config import get_settings
from api.endpoints import predictions, optimization, data
from models.schemas import HealthCheck

settings = get_settings()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="NASA Exoplanet Discovery Platform - ML-powered exoplanet classification system"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions.router, prefix=f"{settings.API_V1_STR}/predictions", tags=["predictions"])
app.include_router(optimization.router, prefix=f"{settings.API_V1_STR}/optimization", tags=["optimization"])
app.include_router(data.router, prefix=f"{settings.API_V1_STR}/data", tags=["data"])


@app.get("/")
async def root():
    return {
        "message": "NASA Exoplanet Discovery API - A World Away",
        "version": settings.VERSION,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    from services.ml_service import ml_service
    from core.database import get_supabase

    db_connected = True
    try:
        get_supabase()
    except:
        db_connected = False

    return HealthCheck(
        status="healthy",
        version=settings.VERSION,
        model_loaded=ml_service.is_trained,
        database_connected=db_connected
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)