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

    # Rate limit defaults for newly-provisioned API keys.
    # Sliding window is 60 seconds; these are requests-per-minute.
    # Service-to-service workloads (sweeps, backfills) can easily
    # exceed the old defaults (50 writes / 100 reads) under burst load.
    engram_default_rate_limit_writes: int = 300
    engram_default_rate_limit_reads: int = 600

    # Feature flags — all off by default. Enable per-mechanism via env vars.
    engram_flag_context_retrieval: bool = False    # Phase 1: context-scaffolded retrieval
    engram_flag_forgetting: bool = False           # Phase 2: retrieval-induced forgetting
    engram_flag_attractor: bool = False            # Phase 3: attractor dynamics
    engram_flag_metaplasticity: bool = False       # Phase 4: metaplasticity
    engram_flag_stdp: bool = False                 # Phase 5: spike-timing-dependent plasticity

    @property
    def exclude_tags_set(self) -> set[str]:
        if not self.engram_retrieval_exclude_tags:
            return set()
        return {t.strip() for t in self.engram_retrieval_exclude_tags.split(",")}


settings = Settings()
