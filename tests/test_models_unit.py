"""Unit tests for models: MemoryNode.to_response, EdgeType structural."""

from datetime import datetime
from uuid import uuid4

import pytest


class TestMemoryNodeToResponse:
    def test_converts_to_response(self):
        from engram.models import DimensionScores, MemoryNode, MemoryResponse

        node = MemoryNode(
            id=uuid4(),
            owner_id=uuid4(),
            content="test content",
            content_hash="a" * 64,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            activation_level=0.8,
            salience=0.5,
            source_type="conversation",
            embedding_model="test-model",
            embedding_dimensions=1024,
        )

        scores = DimensionScores(semantic=0.9, temporal=0.3)
        resp = node.to_response(
            convergence_score=2.5,
            dimension_scores=scores,
            matched_axes=["semantic", "temporal"],
        )

        assert isinstance(resp, MemoryResponse)
        assert resp.convergence_score == 2.5
        assert resp.dimension_scores.semantic == 0.9
        assert "semantic" in resp.matched_axes

    def test_session_id_none(self):
        from engram.models import DimensionScores, MemoryNode

        node = MemoryNode(
            id=uuid4(),
            owner_id=uuid4(),
            content="test",
            content_hash="a" * 64,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            embedding_model="test",
            embedding_dimensions=1024,
            source_type="conversation",
        )

        resp = node.to_response(0.0, DimensionScores(), [])
        assert resp.session_id is None

    def test_session_id_present(self):
        from engram.models import DimensionScores, MemoryNode

        sid = uuid4()
        node = MemoryNode(
            id=uuid4(),
            owner_id=uuid4(),
            content="test",
            content_hash="a" * 64,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            embedding_model="test",
            embedding_dimensions=1024,
            source_type="conversation",
            session_id=sid,
        )

        resp = node.to_response(0.0, DimensionScores(), [])
        assert resp.session_id == str(sid)


class TestEdgeTypeStructural:
    def test_structural_exists(self):
        from engram.models import EdgeType

        assert EdgeType.STRUCTURAL == "structural"

    def test_all_edge_types(self):
        from engram.models import EdgeType

        expected = {
            "excitatory", "inhibitory", "associative",
            "temporal", "modulatory", "structural",
        }
        assert {e.value for e in EdgeType} == expected


class TestEdgeResponse:
    def test_weight_bounds(self):
        from engram.models import EdgeResponse

        resp = EdgeResponse(
            source_id="abc",
            target_id="def",
            edge_type="excitatory",
            axis="semantic",
            weight=0.5,
        )
        assert resp.weight == 0.5

    def test_weight_rejects_negative(self):
        from engram.models import EdgeResponse

        with pytest.raises(ValueError):
            EdgeResponse(
                source_id="abc",
                target_id="def",
                edge_type="excitatory",
                axis="semantic",
                weight=-0.1,
            )

    def test_weight_rejects_over_one(self):
        from engram.models import EdgeResponse

        with pytest.raises(ValueError):
            EdgeResponse(
                source_id="abc",
                target_id="def",
                edge_type="excitatory",
                axis="semantic",
                weight=1.1,
            )


class TestRecallResponse:
    def test_defaults(self):
        from engram.models import RecallResponse

        resp = RecallResponse(memories=[], confidence="low")
        assert resp.edges == []
        assert resp.memories == []


class TestErrorResponse:
    def test_has_expected_annotations(self):
        from engram.models import ErrorResponse

        annotations = ErrorResponse.__annotations__
        assert "code" in annotations
        assert "message" in annotations
        assert "status" in annotations
        assert "correlation_id" in annotations
        assert "existing_id" in annotations
