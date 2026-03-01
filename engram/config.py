"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://engram:engram@localhost:5432/engram"
    REDIS_URL: str = "redis://localhost:6379/0"
    EMBEDDING_DIM: int = 384
    ENGRAM_PLUGIN: str = "engram.plugins.default"

    model_config = {"env_prefix": "ENGRAM_", "env_file": ".env"}


settings = Settings()
