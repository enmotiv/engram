"""Engram client SDK — wraps the HTTP API for easy consumption."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class MemoryResponse:
    id: str
    namespace: str
    content: str
    memory_type: str = "episodic"
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    activation: float = 0.0
    salience: float = 0.5
    access_count: int = 0
    created_at: Optional[str] = None


@dataclass
class EdgeResponse:
    id: str
    source_memory_id: str
    target_memory_id: str
    edge_type: str
    weight: float
    namespace: str = "default"
    created_at: Optional[str] = None


@dataclass
class RetrievalResult:
    triggered: bool
    urgency: float = 0.0
    memories: List[Dict[str, Any]] = field(default_factory=list)
    suppressed: List[Dict[str, Any]] = field(default_factory=list)
    trace: Optional[str] = None
    retrieval_ms: float = 0.0


@dataclass
class NamespaceStats:
    namespace: str
    memory_count: int
    edge_count: int
    last_activity: Optional[str] = None


@dataclass
class HealthResponse:
    status: str = "ok"
    db: bool = False
    redis: bool = False
    plugin: str = ""


class EngramClient:
    """Async client for the Engram memory system API."""

    def __init__(self, base_url: str = "http://localhost:8100", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    # --- Memory operations ---

    async def store(
        self,
        namespace: str,
        content: str,
        memory_type: str = "episodic",
        metadata: Optional[dict] = None,
    ) -> MemoryResponse:
        resp = await self._client.post(
            "/v1/memories",
            json={
                "namespace": namespace,
                "content": content,
                "memory_type": memory_type,
                "metadata": metadata,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return MemoryResponse(**data)

    async def get_memory(self, memory_id: str) -> MemoryResponse:
        resp = await self._client.get(f"/v1/memories/{memory_id}")
        resp.raise_for_status()
        return MemoryResponse(**resp.json())

    async def delete_memory(self, memory_id: str) -> bool:
        resp = await self._client.delete(f"/v1/memories/{memory_id}")
        resp.raise_for_status()
        return resp.json().get("deleted", False)

    async def batch_store(
        self, memories: List[Dict[str, Any]]
    ) -> List[MemoryResponse]:
        payload = {
            "memories": [
                {
                    "namespace": m["namespace"],
                    "content": m["content"],
                    "memory_type": m.get("memory_type", "episodic"),
                    "metadata": m.get("metadata"),
                }
                for m in memories
            ]
        }
        resp = await self._client.post("/v1/memories/batch", json=payload)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("created", data) if isinstance(data, dict) else data
        return [MemoryResponse(**d) for d in items]

    async def retrieve(
        self,
        namespace: str,
        cue: str,
        context: Optional[dict] = None,
        max_results: int = 5,
        urgency_threshold: float = 0.3,
        hop_depth: int = 2,
        dimensional_cues: Optional[Dict[str, str]] = None,
    ) -> RetrievalResult:
        payload = {
            "namespace": namespace,
            "cue": cue,
            "context": context,
            "max_results": max_results,
            "urgency_threshold": urgency_threshold,
            "hop_depth": hop_depth,
        }
        if dimensional_cues:
            payload["dimensional_cues"] = dimensional_cues
        resp = await self._client.post(
            "/v1/memories/retrieve",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return RetrievalResult(
            triggered=data.get("triggered", False),
            urgency=data.get("urgency", 0.0),
            memories=data.get("memories") or [],
            suppressed=data.get("suppressed") or [],
            trace=data.get("trace"),
            retrieval_ms=data.get("retrieval_ms", 0.0),
        )

    # --- Edge operations ---

    async def create_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 0.5,
        context: Optional[dict] = None,
        namespace: str = "default",
    ) -> EdgeResponse:
        resp = await self._client.post(
            "/v1/edges",
            json={
                "source_id": source_id,
                "target_id": target_id,
                "edge_type": edge_type,
                "weight": weight,
                "context": context,
                "namespace": namespace,
            },
        )
        resp.raise_for_status()
        return EdgeResponse(**resp.json())

    async def get_edges(
        self, memory_id: str, direction: str = "both"
    ) -> List[EdgeResponse]:
        resp = await self._client.get(
            f"/v1/memories/{memory_id}/edges",
            params={"direction": direction},
        )
        resp.raise_for_status()
        return [EdgeResponse(**e) for e in resp.json()]

    async def delete_edge(self, edge_id: str) -> bool:
        resp = await self._client.delete(f"/v1/edges/{edge_id}")
        resp.raise_for_status()
        return resp.json().get("deleted", False)

    # --- Namespace operations ---

    async def namespace_stats(self, namespace: str) -> NamespaceStats:
        resp = await self._client.get(f"/v1/namespaces/{namespace}/stats")
        resp.raise_for_status()
        data = resp.json()
        return NamespaceStats(
            namespace=data["namespace"],
            memory_count=data["memory_count"],
            edge_count=data["edge_count"],
            last_activity=data.get("last_activity"),
        )

    async def delete_namespace(self, namespace: str) -> dict:
        resp = await self._client.delete(f"/v1/namespaces/{namespace}")
        resp.raise_for_status()
        return resp.json()

    # --- Health ---

    async def health(self) -> HealthResponse:
        resp = await self._client.get("/v1/health")
        resp.raise_for_status()
        return HealthResponse(**resp.json())


class EngramClientSync:
    """Synchronous wrapper around EngramClient for non-async callers."""

    def __init__(self, base_url: str = "http://localhost:8100", timeout: float = 30.0):
        self._async = EngramClient(base_url=base_url, timeout=timeout)

    def _run(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "Cannot use EngramClientSync from an async context. "
                "Use EngramClient (async) instead."
            )
        return asyncio.run(coro)

    def store(self, namespace: str, content: str, **kwargs) -> MemoryResponse:
        return self._run(self._async.store(namespace, content, **kwargs))

    def retrieve(self, namespace: str, cue: str, **kwargs) -> RetrievalResult:
        return self._run(self._async.retrieve(namespace, cue, **kwargs))

    def get_edges(self, memory_id: str, **kwargs) -> List[EdgeResponse]:
        return self._run(self._async.get_edges(memory_id, **kwargs))

    def namespace_stats(self, namespace: str) -> NamespaceStats:
        return self._run(self._async.namespace_stats(namespace))

    def health(self) -> HealthResponse:
        return self._run(self._async.health())

    def close(self):
        self._run(self._async.close())
