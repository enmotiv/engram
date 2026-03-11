"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    engram_embedding_model: str = "openai/text-embedding-3-small"
    engram_embedding_dimensions: int = 1024
    engram_llm_model: str = "openai/gpt-4o-mini"
    database_url: str
    redis_url: str = "redis://localhost:6379/0"
    engram_api_port: int = 8000
    engram_log_level: str = "INFO"
    engram_extensions: str = ""
    engram_retrieval_exclude_tags: str = ""
    engram_admin_secret: str = ""

    @property
    def exclude_tags_set(self) -> set[str]:
        if not self.engram_retrieval_exclude_tags:
            return set()
        return {t.strip() for t in self.engram_retrieval_exclude_tags.split(",")}


settings = Settings()
