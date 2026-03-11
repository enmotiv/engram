"""Unit tests for embeddings: salience computation, error recording."""

import pytest


class TestComputeSalience:
    @pytest.mark.asyncio
    async def test_returns_float(self):
        from unittest.mock import AsyncMock, patch

        from engram.services.embedding import compute_salience

        anchor = [1.0] + [0.0] * 1023
        with patch(
            "engram.services.embedding.get_salience_anchor",
            AsyncMock(return_value=anchor),
        ):
            result = await compute_salience(anchor)
            assert isinstance(result, float)

    @pytest.mark.asyncio
    async def test_floor_at_0_1(self):
        from unittest.mock import AsyncMock, patch

        from engram.services.embedding import compute_salience

        # Orthogonal vectors → similarity 0 → should floor at 0.1
        anchor = [1.0] + [0.0] * 1023
        orthogonal = [0.0, 1.0] + [0.0] * 1022
        with patch(
            "engram.services.embedding.get_salience_anchor",
            AsyncMock(return_value=anchor),
        ):
            result = await compute_salience(orthogonal)
            assert result == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_cap_at_1_0(self):
        from unittest.mock import AsyncMock, patch

        from engram.services.embedding import compute_salience

        anchor = [1.0] + [0.0] * 1023
        with patch(
            "engram.services.embedding.get_salience_anchor",
            AsyncMock(return_value=anchor),
        ):
            result = await compute_salience(anchor)
            assert result <= 1.0


class TestRecordOpenrouterError:
    def test_records_status_code(self):
        from unittest.mock import MagicMock

        from engram.services.embedding import _record_openrouter_error

        exc = MagicMock()
        exc.status_code = 429
        # Should not raise
        _record_openrouter_error(exc)

    def test_records_unknown_status(self):
        from engram.services.embedding import _record_openrouter_error

        exc = ValueError("no status_code attr")
        _record_openrouter_error(exc)


class TestGetClient:
    def test_returns_async_openai(self):
        from engram.services.embedding import get_client

        client = get_client()
        # Should be an AsyncOpenAI instance
        from openai import AsyncOpenAI
        assert isinstance(client, AsyncOpenAI)

    def test_cached(self):
        from engram.services.embedding import get_client

        c1 = get_client()
        c2 = get_client()
        assert c1 is c2


class TestEdgeFactors:
    def test_all_factors_defined(self):
        from engram.services.read import _EDGE_FACTORS

        assert "excitatory" in _EDGE_FACTORS
        assert "inhibitory" in _EDGE_FACTORS
        assert "associative" in _EDGE_FACTORS
        assert "temporal" in _EDGE_FACTORS
        assert "modulatory" in _EDGE_FACTORS

    def test_modulatory_is_none(self):
        from engram.services.read import _EDGE_FACTORS

        assert _EDGE_FACTORS["modulatory"] is None

    def test_excitatory_positive(self):
        from engram.services.read import _EDGE_FACTORS

        assert _EDGE_FACTORS["excitatory"] > 0

    def test_inhibitory_negative(self):
        from engram.services.read import _EDGE_FACTORS

        assert _EDGE_FACTORS["inhibitory"] < 0
