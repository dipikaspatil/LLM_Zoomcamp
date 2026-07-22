# backend/app/config.py
import pathlib
from pydantic_settings import BaseSettings

# Resolve capstone/.env regardless of the current working directory:
# config.py is at backend/app/config.py, so parent.parent.parent = capstone/
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    FOOTBALL_DATA_API_KEY: str
    QDRANT_URL: str = "http://localhost:6333"

    class Config:
        env_file = PROJECT_ROOT / ".env"

settings = Settings()
