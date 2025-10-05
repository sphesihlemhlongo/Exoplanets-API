from pydantic_settings import BaseSettings
from functools import lru_cache
from dotenv import load_dotenv
import os
load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "NASA Exoplanet Discovery - A World Away"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    SUPABASE_URL: str
    SUPABASE_KEY: str

    CORS_ORIGINS: list = ["http://localhost:5173", "http://localhost:4173"]

    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024
    UPLOAD_DIR: str = "/tmp/uploads"

    MODEL_PATH: str = "models/exoplanet_ensemble.joblib"

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings():
    return Settings(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))  # Pass os.environ to ensure all env vars are considered
