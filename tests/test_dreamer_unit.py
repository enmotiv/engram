"""Unit tests for Dreamer internals: edge storage validation, classify parsing."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
import pytest_asyncio


class TestStoreEdgesValidation:
    """Test _store_edges input validation without DB."""

    @pytest.mark.asyncio
    async def test_skips_invalid_axis(self):
        from engram.services.dreamer import _store_edges

        conn = AsyncMock()
        classification = {
            "invalid_axis": [{"type": "excitatory", "weight": 0.8}]
        }
        count = await _store_edges(
            conn, uuid4(), uuid4(), uuid4(), classification
        )
        assert count == 0
        conn.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_invalid_edge_type(self):
        from engram.services.dreamer import _store_edges

        conn = AsyncMock()
        classification = {
            "semantic": [{"type": "invalid_type", "weight": 0.8}]
        }
        count = await _store_edges(
            conn, uuid4(), uuid4(), uuid4(), classification
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_skips_low_weight(self):
        from engram.services.dreamer import _store_edges

        conn = AsyncMock()
        classification = {
            "semantic": [{"type": "excitatory", "weight": 0.2}]
        }
        count = await _store_edges(
            conn, uuid4(), uuid4(), uuid4(), classification
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_clamps_weight(self):
        from engram.services.dreamer import _store_edges

        conn = AsyncMock()
        classification = {
            "semantic": [{"type": "excitatory", "weight": 1.5}]
        }
        count = await _store_edges(
            conn, uuid4(), uuid4(), uuid4(), classification
        )
        assert count == 1
        # Check the weight arg is clamped to 1.0
        call_args = conn.execute.call_args[0]
        assert call_args[6] == 1.0  # weight param

    @pytest.mark.asyncio
    async def test_enforces_ordering(self):
        from engram.services.dreamer import _store_edges

        conn = AsyncMock()
        id_a = UUID("00000000-0000-0000-0000-000000000001")
        id_b = UUID("00000000-0000-0000-0000-000000000002")
        classification = {
            "semantic": [{"type": "excitatory", "weight": 0.8}]
        }

        # Pass source > target — should swap
        await _store_edges(conn, uuid4(), id_b, id_a, classification)
        call_args = conn.execute.call_args[0]
        assert call_args[2] == id_a  # source is min
        assert call_args[3] == id_b  # target is max

    @pytest.mark.asyncio
    async def test_skips_non_numeric_weight(self):
        from engram.services.dreamer import _store_edges

        conn = AsyncMock()
        classification = {
            "semantic": [{"type": "excitatory", "weight": "not_a_number"}]
        }
        count = await _store_edges(
            conn, uuid4(), uuid4(), uuid4(), classification
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_skips_non_list_edges(self):
        from engram.services.dreamer import _store_edges

        conn = AsyncMock()
        classification = {"semantic": "not a list"}
        count = await _store_edges(
            conn, uuid4(), uuid4(), uuid4(), classification
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_skips_non_dict_edge(self):
        from engram.services.dreamer import _store_edges

        conn = AsyncMock()
        classification = {"semantic": ["not a dict"]}
        count = await _store_edges(
            conn, uuid4(), uuid4(), uuid4(), classification
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_accepts_structural(self):
        from engram.services.dreamer import _store_edges

        conn = AsyncMock()
        classification = {
            "semantic": [{"type": "structural", "weight": 0.9}]
        }
        count = await _store_edges(
            conn, uuid4(), uuid4(), uuid4(), classification
        )
        assert count == 1


class TestClassifyPair:
    """Test _classify_pair JSON parsing."""

    @pytest.mark.asyncio
    async def test_parses_valid_json(self):
        from engram.services.dreamer import _classify_pair

        valid = json.dumps({
            "temporal": [],
            "emotional": [],
            "semantic": [{"type": "associative", "weight": 0.7}],
            "sensory": [],
            "action": [],
            "procedural": [],
        })

        with patch("engram.services.dreamer.llm_classify", AsyncMock(return_value=valid)):
            result = await _classify_pair("new", "old")
            assert "semantic" in result
            assert len(result["semantic"]) == 1

    @pytest.mark.asyncio
    async def test_handles_markdown_fences(self):
        from engram.services.dreamer import _classify_pair

        fenced = '```json\n{"temporal": [], "emotional": [], "semantic": [], "sensory": [], "action": [], "procedural": []}\n```'

        with patch("engram.services.dreamer.llm_classify", AsyncMock(return_value=fenced)):
            result = await _classify_pair("new", "old")
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_returns_empty_on_invalid_json(self):
        from engram.services.dreamer import _classify_pair

        with patch("engram.services.dreamer.llm_classify", AsyncMock(return_value="not json at all")):
            result = await _classify_pair("new", "old")
            assert result == {}


class TestValidEdgeTypes:
    def test_all_types_present(self):
        from engram.services.dreamer import _VALID_EDGE_TYPES

        expected = {
            "excitatory", "inhibitory", "associative",
            "temporal", "modulatory", "structural",
        }
        assert _VALID_EDGE_TYPES == expected


class TestValidAxes:
    def test_matches_axes(self):
        from engram.services.dreamer import _VALID_AXES
        from engram.models import AXES

        assert _VALID_AXES == set(AXES)
