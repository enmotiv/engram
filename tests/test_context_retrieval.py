"""Tests for Phase 1: Context-scaffolded retrieval."""

import math
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from engram.models.common import SourceType


# --- Helpers ---


def _make_vector(seed: float, dims: int = 1024) -> list[float]:
    vec = [math.sin(seed * (i + 1) + i * 0.1) for i in range(dims)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


# --- Unit Tests ---


class TestWeightedConvergenceMath:
    """Verify axis_weights correctly scale convergence formula."""

    @pytest.mark.asyncio
    async def test_weighted_convergence_increases_emotional(self, enable_flag):
        """Emotional weight = 3.0 should boost memories with high emotional scores."""
        enable_flag("engram_flag_context_retrieval")
        from engram.services.read import _score_candidates

        node_a = uuid4()
        node_b = uuid4()
        candidates = {
            node_a: {
                "scores": {"emotional": 0.9, "semantic": 0.4},
                "matched_axes": ["emotional", "semantic"],
            },
            node_b: {
                "scores": {"semantic": 0.9, "action": 0.4},
                "matched_axes": ["semantic", "action"],
            },
        }

        # Mock DB call
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[
            {"id": node_a, "activation_level": 1.0, "salience": 0.5, "content": "a", "plasticity": 0.8},
            {"id": node_b, "activation_level": 1.0, "salience": 0.5, "content": "b", "plasticity": 0.8},
        ])

        # With emotional weight = 3.0, node_a should rank higher
        scored, _ = await _score_candidates(
            mock_conn, candidates,
            axis_weights={"emotional": 3.0, "semantic": 1.0, "action": 1.0},
        )

        assert len(scored) == 2
        assert scored[0]["id"] == node_a

    @pytest.mark.asyncio
    async def test_axis_weights_none_preserves_default(self, enable_flag):
        """When axis_weights is None, scoring is identical to current behavior."""
        enable_flag("engram_flag_context_retrieval")
        from engram.services.read import _score_candidates

        node_a = uuid4()
        candidates = {
            node_a: {
                "scores": {"emotional": 0.8, "semantic": 0.7, "action": 0.6},
                "matched_axes": ["emotional", "semantic", "action"],
            },
        }

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[
            {"id": node_a, "activation_level": 1.0, "salience": 0.5, "content": "test", "plasticity": 0.8},
        ])

        # Without weights
        scored_no_weights, _ = await _score_candidates(mock_conn, candidates)

        # With None weights (should be same)
        scored_none, _ = await _score_candidates(
            mock_conn, candidates, axis_weights=None
        )

        assert scored_no_weights[0]["convergence_score"] == scored_none[0]["convergence_score"]

    @pytest.mark.asyncio
    async def test_weighted_convergence_math_exact(self, enable_flag):
        """Hand-compute expected weighted convergence values."""
        enable_flag("engram_flag_context_retrieval")
        from engram.services.read import _score_candidates

        node = uuid4()
        candidates = {
            node: {
                "scores": {"emotional": 0.8, "semantic": 0.6},
                "matched_axes": ["emotional", "semantic"],
            },
        }

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[
            {"id": node, "activation_level": 1.0, "salience": 0.5, "content": "", "plasticity": 0.8},
        ])

        # Weighted: emotional=2.0, semantic=1.0
        # weighted_sum = 0.8*2.0 + 0.6*1.0 = 2.2
        # weight_total = 2.0 + 1.0 = 3.0
        # avg_score = 2.2 / 3.0 = 0.7333...
        # convergence = 2 * 0.7333 = 1.4666...
        scored, _ = await _score_candidates(
            mock_conn, candidates,
            axis_weights={"emotional": 2.0, "semantic": 1.0},
        )

        expected_conv = 2 * (0.8 * 2.0 + 0.6 * 1.0) / (2.0 + 1.0)
        assert abs(scored[0]["convergence_score"] - expected_conv) < 0.01


class TestFilteredSearchSQL:
    """Verify find_by_vector_similarity_filtered builds correct queries."""

    @pytest.mark.asyncio
    async def test_no_filters_matches_base(self):
        """With no filters, behaves like find_by_vector_similarity."""
        from engram.repositories.memory_repo import find_by_vector_similarity_filtered

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        owner_id = uuid4()
        vec = _make_vector(1.0)

        await find_by_vector_similarity_filtered(
            mock_conn, owner_id, vec, "semantic", limit=5
        )

        # Should have called fetch with basic WHERE clause
        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        assert "owner_id = $2" in sql
        assert "is_deleted = FALSE" in sql
        assert "session_id" not in sql
        assert "source_type" not in sql

    @pytest.mark.asyncio
    async def test_session_id_filter(self):
        from engram.repositories.memory_repo import find_by_vector_similarity_filtered

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        owner_id = uuid4()
        session = uuid4()
        vec = _make_vector(1.0)

        await find_by_vector_similarity_filtered(
            mock_conn, owner_id, vec, "semantic", limit=5,
            session_id=session,
        )

        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        assert "session_id" in sql

    @pytest.mark.asyncio
    async def test_source_types_filter(self):
        from engram.repositories.memory_repo import find_by_vector_similarity_filtered

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        owner_id = uuid4()
        vec = _make_vector(1.0)

        await find_by_vector_similarity_filtered(
            mock_conn, owner_id, vec, "semantic", limit=5,
            source_types=["correction", "observation"],
        )

        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        assert "source_type" in sql

    @pytest.mark.asyncio
    async def test_time_window_filter(self):
        from engram.repositories.memory_repo import find_by_vector_similarity_filtered

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        owner_id = uuid4()
        vec = _make_vector(1.0)

        await find_by_vector_similarity_filtered(
            mock_conn, owner_id, vec, "semantic", limit=5,
            time_window_hours=24,
        )

        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        assert "24 hours" in sql


class TestRecallNewFieldsIgnoredWhenFlagOff:
    """When context retrieval flag is OFF, new fields are accepted but ignored."""

    @pytest.mark.asyncio
    async def test_axis_weights_ignored_when_flag_off(self, disable_flag):
        """axis_weights should not affect scoring when flag is off."""
        disable_flag("engram_flag_context_retrieval")
        from engram.services.read import _score_candidates

        node = uuid4()
        candidates = {
            node: {
                "scores": {"emotional": 0.8, "semantic": 0.6},
                "matched_axes": ["emotional", "semantic"],
            },
        }

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[
            {"id": node, "activation_level": 1.0, "salience": 0.5, "content": ""},
        ])

        # Without weights
        scored_base, _ = await _score_candidates(mock_conn, candidates)

        # With weights (should be ignored because flag is off)
        scored_weighted, _ = await _score_candidates(
            mock_conn, candidates,
            axis_weights={"emotional": 10.0},
        )

        assert scored_base[0]["convergence_score"] == scored_weighted[0]["convergence_score"]


# --- Integration Tests (require TEST_DATABASE_URL) ---


@pytest.mark.asyncio
async def test_recall_with_no_new_fields_backward_compat(client, owner_a, seeded_memories):
    """POST /v1/recall with no new fields → identical to current behavior."""
    _, key = owner_a
    resp = await client.post(
        "/v1/recall",
        json={"cue": "dark mode preferences", "top_k": 5},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "memories" in data


@pytest.mark.asyncio
async def test_recall_with_axis_weights_accepted(client, owner_a, seeded_memories):
    """POST /v1/recall with axis_weights is accepted (even if flag is off)."""
    _, key = owner_a
    resp = await client.post(
        "/v1/recall",
        json={
            "cue": "dark mode preferences",
            "top_k": 5,
            "axis_weights": {"emotional": 3.0},
        },
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_recall_with_source_types_accepted(client, owner_a, seeded_memories):
    """POST /v1/recall with source_types is accepted."""
    _, key = owner_a
    resp = await client.post(
        "/v1/recall",
        json={
            "cue": "dark mode preferences",
            "top_k": 5,
            "source_types": ["conversation"],
        },
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_recall_with_time_window_accepted(client, owner_a, seeded_memories):
    """POST /v1/recall with time_window_hours is accepted."""
    _, key = owner_a
    resp = await client.post(
        "/v1/recall",
        json={
            "cue": "dark mode preferences",
            "top_k": 5,
            "time_window_hours": 24,
        },
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_recall_with_settle_accepted(client, owner_a, seeded_memories):
    """POST /v1/recall with settle flag is accepted."""
    _, key = owner_a
    resp = await client.post(
        "/v1/recall",
        json={
            "cue": "dark mode preferences",
            "top_k": 5,
            "settle": True,
        },
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp.status_code == 200
