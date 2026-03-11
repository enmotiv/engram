"""OpenRouter embedding and LLM client with call tracking."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import structlog
from openai import AsyncOpenAI

from engram.config import settings
from engram.models import AXES, DIMENSION_PREFIXES
from engram.core.tracing import EMBED_LATENCY, OPENROUTER_ERRORS, Span, get_trace
from engram.utilities.vectors import cosine_similarity, normalize_l2

logger = structlog.get_logger()


# --- Embedding Provider Interface ---


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Interface for embedding providers. Swap implementations without LangChain."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenRouterEmbedding:
    """OpenRouter via OpenAI-compatible API."""

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
            )
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raw = await self._get_client().embeddings.with_raw_response.create(
            model=settings.engram_embedding_model,
            input=texts,
            dimensions=settings.engram_embedding_dimensions,
            encoding_format="float",
        )
        response = raw.parse()
        return [item.embedding for item in response.data]


_provider: OpenRouterEmbedding | None = None
_salience_anchor: list[float] | None = None


def _get_provider() -> OpenRouterEmbedding:
    global _provider  # noqa: PLW0603
    if _provider is None:
        _provider = OpenRouterEmbedding()
    return _provider


def get_client() -> AsyncOpenAI:
    """Return cached AsyncOpenAI client. Used by health check."""
    return _get_provider()._get_client()


def _record_openrouter_error(exc: Exception) -> None:
    """Increment the OpenRouter error counter."""
    status = getattr(exc, "status_code", "unknown")
    OPENROUTER_ERRORS.labels(status_code=str(status)).inc()


def _log_openrouter_call(
    operation: str,
    *,
    model: str,
    input_tokens: int,
    duration_ms: float,
    headers: object,
) -> None:
    """Log OpenRouter API call details. Never logs API keys."""
    logger.info(
        operation,
        component="embeddings",
        model=model,
        input_tokens=input_tokens,
        duration_ms=round(duration_ms, 2),
        openrouter_id=getattr(headers, "get", lambda *a: "unknown")(
            "x-request-id", "unknown"
        ),
        provider=getattr(headers, "get", lambda *a: "unknown")(
            "x-openrouter-provider", "unknown"
        ),
        dimensions=settings.engram_embedding_dimensions,
    )


async def embed_six_dimensions(
    content: str,
) -> dict[str, list[float]]:
    """Embed content into 6 dimensional vectors. One batched API call."""
    inputs = [prefix + content for prefix in DIMENSION_PREFIXES.values()]

    with Span(
        "embeddings.embed",
        component="embeddings",
        expected_ms=500,
        histogram=EMBED_LATENCY,
    ) as span:
        try:
            raw = await get_client().embeddings.with_raw_response.create(
                model=settings.engram_embedding_model,
                input=inputs,
                dimensions=settings.engram_embedding_dimensions,
                encoding_format="float",
            )
        except Exception as exc:
            _record_openrouter_error(exc)
            raise

    response = raw.parse()
    _log_openrouter_call(
        "openrouter.embedding",
        model=getattr(response, "model", settings.engram_embedding_model),
        input_tokens=response.usage.prompt_tokens if response.usage else 0,
        duration_ms=span.duration_ms,
        headers=raw.headers,
    )

    tc = get_trace()
    if tc:
        tc.embedding_ms = span.duration_ms

    result = {}
    for i, axis in enumerate(AXES):
        vec = response.data[i].embedding
        result[axis] = normalize_l2(vec)
    return result


async def get_salience_anchor() -> list[float]:
    """Get or compute the salience anchor vector. Cached after first call."""
    global _salience_anchor  # noqa: PLW0603
    if _salience_anchor is None:
        with Span(
            "embeddings.salience_anchor",
            component="embeddings",
            expected_ms=500,
            histogram=EMBED_LATENCY,
        ):
            try:
                raw = await get_client().embeddings.with_raw_response.create(
                    model=settings.engram_embedding_model,
                    input=[
                        "This is extremely urgent, critical,"
                        " and emotionally significant."
                    ],
                    dimensions=settings.engram_embedding_dimensions,
                    encoding_format="float",
                )
            except Exception as exc:
                _record_openrouter_error(exc)
                raise
        response = raw.parse()
        _salience_anchor = normalize_l2(response.data[0].embedding)
    return _salience_anchor


async def compute_salience(emotional_vector: list[float]) -> float:
    """Compute salience from emotional vector. Range [0.1, 1.0]."""
    anchor = await get_salience_anchor()
    similarity = cosine_similarity(emotional_vector, anchor)
    return min(1.0, max(0.1, similarity))


async def llm_classify(prompt: str) -> str:
    """Send a classification prompt to the LLM. Returns raw text."""
    with Span(
        "embeddings.llm_classify",
        component="embeddings",
        expected_ms=2000,
    ) as span:
        try:
            raw = await get_client().chat.completions.with_raw_response.create(
                model=settings.engram_llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
        except Exception as exc:
            _record_openrouter_error(exc)
            raise

    response = raw.parse()
    _log_openrouter_call(
        "openrouter.llm_classify",
        model=getattr(response, "model", settings.engram_llm_model),
        input_tokens=response.usage.prompt_tokens if response.usage else 0,
        duration_ms=span.duration_ms,
        headers=raw.headers,
    )

    return response.choices[0].message.content or ""
