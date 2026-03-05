"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://engram:engram@localhost:5432/engram"
    REDIS_URL: str = "redis://localhost:6379/0"
    EMBEDDING_DIM: int = 1024
    ENGRAM_PLUGIN: str = "engram.plugins.brain_regions"
    ENGRAM_PLUGINS: str = ""  # Comma-separated plugin modules (overrides ENGRAM_PLUGIN when set)

    # Embedding: "local" (sentence-transformers) or "api" (OpenAI-compatible endpoint)
    EMBEDDING_PROVIDER: str = "api"  # "api" or "local"
    EMBEDDING_API_URL: str = ""  # e.g. https://openrouter.ai/api/v1/embeddings
    EMBEDDING_API_KEY: str = ""  # API key for embedding endpoint
    EMBEDDING_MODEL: str = "baai/bge-m3"  # model name for API calls

    # LLM provider: "none" (heuristic only), "ollama", or "openai" (OpenAI-compatible)
    LLM_PROVIDER: str = "none"
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""  # Auto-set per provider if empty
    LLM_MODEL: str = ""

    # Master integration flag — controls all Enmotiv integration intelligence
    INTEGRATION_ENABLED: bool = True

    # Amygdala asymmetry: high-amygdala memories get a retrieval score boost
    AMYGDALA_ASYMMETRY_ENABLED: bool = True
    AMYGDALA_ASYMMETRY_THRESHOLD: float = 0.75
    AMYGDALA_ASYMMETRY_MULTIPLIER: float = 1.3

    model_config = {"env_prefix": "ENGRAM_", "env_file": ".env"}

    @property
    def plugin_paths(self) -> str:
        """Return the comma-separated plugin string to load."""
        return self.ENGRAM_PLUGINS if self.ENGRAM_PLUGINS else self.ENGRAM_PLUGIN


settings = Settings()
