"""Provider-agnostic LLM client. Configured via ENGRAM_LLM_* env vars."""

from __future__ import annotations

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Default base URLs per provider (used when LLM_BASE_URL is empty)
_DEFAULT_URLS = {
    "ollama": "http://localhost:11434",
    "openai": "https://api.openai.com/v1",
}


class LLMService:
    """Provider-agnostic LLM client.

    Supports two backends:
      - "ollama"  → {base_url}/api/generate  (Ollama wire protocol)
      - "openai"  → {base_url}/chat/completions  (OpenAI-compatible — covers
        OpenRouter, Anthropic, OpenAI, vLLM, llama.cpp, TGI, etc.)
      - "none"    → is_available() returns False, caller uses heuristic fallback
    """

    def __init__(self, provider: str, api_key: str, base_url: str, model: str):
        self._provider = provider.lower()
        self._api_key = api_key
        self._base_url = base_url.rstrip("/") if base_url else _DEFAULT_URLS.get(self._provider, "")
        self._model = model

        if self._provider not in ("none", "ollama", "openai"):
            logger.warning("Unknown LLM_PROVIDER %r, treating as 'none'", self._provider)
            self._provider = "none"

    def is_available(self) -> bool:
        return self._provider != "none"

    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 500,
        temperature: float = 0.3,
        json_mode: bool = False,
    ) -> str:
        """Send a completion request. Returns the raw text response."""
        if not self.is_available():
            raise RuntimeError("LLM provider is 'none'; check is_available() first")

        if self._provider == "ollama":
            return await self._complete_ollama(prompt, max_tokens, temperature, json_mode)
        return await self._complete_openai(prompt, max_tokens, temperature, json_mode)

    # ── Ollama backend ──────────────────────────────────────────────

    async def _complete_ollama(
        self, prompt: str, max_tokens: int, temperature: float, json_mode: bool,
    ) -> str:
        payload: dict = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{self._base_url}/api/generate", json=payload)
            resp.raise_for_status()

        return resp.json().get("response", "")

    # ── OpenAI-compatible backend ───────────────────────────────────

    async def _complete_openai(
        self, prompt: str, max_tokens: int, temperature: float, json_mode: bool,
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions", headers=headers, json=payload,
            )
            resp.raise_for_status()

        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ── Singleton ───────────────────────────────────────────────────────

_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Return the shared LLMService singleton (lazy-initialized from settings)."""
    global _llm_service
    if _llm_service is None:
        from engram.config import settings

        _llm_service = LLMService(
            provider=settings.LLM_PROVIDER,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            model=settings.LLM_MODEL,
        )
        if _llm_service.is_available():
            logger.info(
                "LLM service initialized: provider=%s model=%s",
                settings.LLM_PROVIDER,
                settings.LLM_MODEL,
            )
        else:
            logger.info("LLM service disabled (provider=none), using heuristic fallbacks")
    return _llm_service
