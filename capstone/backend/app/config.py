import pathlib
from pydantic_settings import BaseSettings

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    FOOTBALL_DATA_API_KEY: str
    NEWS_API_KEY: str  # newsapi.org
    QDRANT_URL: str = "http://localhost:6333"
    DATABASE_URL: str = "postgresql://soccermind:soccermind@localhost:5432/soccermind"

    class Config:
        env_file = PROJECT_ROOT / ".env"
        extra = "ignore"

settings = Settings()
