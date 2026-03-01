"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://engram:engram@localhost:5432/engram"
    REDIS_URL: str = "redis://localhost:6379/0"
    EMBEDDING_DIM: int = 384
    ENGRAM_PLUGIN: str = "engram.plugins.default"
    ENGRAM_PLUGINS: str = ""  # Comma-separated plugin modules (overrides ENGRAM_PLUGIN when set)

    model_config = {"env_prefix": "ENGRAM_", "env_file": ".env"}

    @property
    def plugin_paths(self) -> str:
        """Return the comma-separated plugin string to load."""
        return self.ENGRAM_PLUGINS if self.ENGRAM_PLUGINS else self.ENGRAM_PLUGIN


settings = Settings()
