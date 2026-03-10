"""Unit tests for pure math/scoring functions. No database required."""

import math

import pytest

# --- L2 normalization ---


class TestNormalizeL2:
    def test_unit_vector(self):
        from engram.embeddings import normalize_l2

        vec = [3.0, 4.0]
        result = normalize_l2(vec)
        norm = math.sqrt(sum(x * x for x in result))
        assert abs(norm - 1.0) < 1e-6

    def test_zero_vector(self):
        from engram.embeddings import normalize_l2

        vec = [0.0, 0.0, 0.0]
        result = normalize_l2(vec)
        assert result == [0.0, 0.0, 0.0]

    def test_already_normalized(self):
        from engram.embeddings import normalize_l2

        vec = [1.0, 0.0, 0.0]
        result = normalize_l2(vec)
        assert abs(result[0] - 1.0) < 1e-6
        assert abs(result[1]) < 1e-6

    def test_negative_values(self):
        from engram.embeddings import normalize_l2

        vec = [-3.0, 4.0]
        result = normalize_l2(vec)
        norm = math.sqrt(sum(x * x for x in result))
        assert abs(norm - 1.0) < 1e-6


# --- Cosine similarity ---


class TestCosineSimilarity:
    def test_identical_vectors(self):
        from engram.embeddings import cosine_similarity

        a = [1.0, 0.0]
        assert cosine_similarity(a, a) == 1.0

    def test_orthogonal_vectors(self):
        from engram.embeddings import cosine_similarity

        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        from engram.embeddings import cosine_similarity

        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == -1.0


# --- Content hash ---


class TestContentHash:
    def test_deterministic(self):
        from engram.models import compute_content_hash

        h1 = compute_content_hash("test")
        h2 = compute_content_hash("test")
        assert h1 == h2

    def test_length(self):
        from engram.models import compute_content_hash

        h = compute_content_hash("test")
        assert len(h) == 64

    def test_different_inputs(self):
        from engram.models import compute_content_hash

        h1 = compute_content_hash("test")
        h2 = compute_content_hash("other")
        assert h1 != h2

    def test_unicode(self):
        from engram.models import compute_content_hash

        h = compute_content_hash("日本語テスト")
        assert len(h) == 64


# --- Confidence computation ---


class TestConfidence:
    def test_high(self):
        from engram.models import ConfidenceLevel, compute_confidence

        result = compute_confidence(
            {"dims_matched": 5, "convergence_score": 4.0}
        )
        assert result == ConfidenceLevel.HIGH

    def test_medium_by_dims(self):
        from engram.models import ConfidenceLevel, compute_confidence

        result = compute_confidence(
            {"dims_matched": 2, "convergence_score": 0.5}
        )
        assert result == ConfidenceLevel.MEDIUM

    def test_medium_by_convergence(self):
        from engram.models import ConfidenceLevel, compute_confidence

        result = compute_confidence(
            {"dims_matched": 1, "convergence_score": 2.0}
        )
        assert result == ConfidenceLevel.MEDIUM

    def test_low(self):
        from engram.models import ConfidenceLevel, compute_confidence

        result = compute_confidence(
            {"dims_matched": 1, "convergence_score": 0.5}
        )
        assert result == ConfidenceLevel.LOW

    def test_boundary_high(self):
        from engram.models import ConfidenceLevel, compute_confidence

        result = compute_confidence(
            {"dims_matched": 4, "convergence_score": 3.0}
        )
        assert result == ConfidenceLevel.HIGH


# --- Correlation ID ---


class TestCorrelationId:
    def test_format(self):
        from engram.models import generate_correlation_id

        cid = generate_correlation_id()
        assert cid.startswith("req_")
        assert len(cid) == 16  # req_ + 12 hex chars

    def test_uniqueness(self):
        from engram.models import generate_correlation_id

        ids = {generate_correlation_id() for _ in range(100)}
        assert len(ids) == 100


# --- Node ID ---


class TestNodeId:
    def test_is_uuid(self):
        from uuid import UUID

        from engram.models import generate_node_id

        nid = generate_node_id()
        assert isinstance(nid, UUID)

    def test_uniqueness(self):
        from engram.models import generate_node_id

        ids = {generate_node_id() for _ in range(100)}
        assert len(ids) == 100


# --- API key hashing ---


class TestApiKeyHash:
    def test_deterministic(self):
        from engram.auth import hash_api_key

        h1 = hash_api_key("test_key")
        h2 = hash_api_key("test_key")
        assert h1 == h2

    def test_length(self):
        from engram.auth import hash_api_key

        h = hash_api_key("test_key")
        assert len(h) == 64

    def test_different_keys(self):
        from engram.auth import hash_api_key

        h1 = hash_api_key("key_a")
        h2 = hash_api_key("key_b")
        assert h1 != h2


# --- Rate limiter ---


class TestRateLimiter:
    def test_allows_within_limit(self):
        from engram.auth import RateLimiter

        rl = RateLimiter()
        allowed, remaining, _ = rl.check("test", limit=5)
        assert allowed is True
        assert remaining == 4

    def test_blocks_at_limit(self):
        from engram.auth import RateLimiter

        rl = RateLimiter()
        for _ in range(5):
            rl.check("test", limit=5)
        allowed, remaining, _ = rl.check("test", limit=5)
        assert allowed is False
        assert remaining == 0

    def test_separate_buckets(self):
        from engram.auth import RateLimiter

        rl = RateLimiter()
        for _ in range(5):
            rl.check("bucket_a", limit=5)
        allowed, _, _ = rl.check("bucket_b", limit=5)
        assert allowed is True


# --- Activation formulas ---


class TestActivationFormulas:
    def test_boost_caps_at_one(self):
        """Reconsolidation boost: LEAST(1.0, activation + 0.1)."""
        current = 0.95
        boosted = min(1.0, current + 0.1)
        assert boosted == 1.0

    def test_boost_from_zero(self):
        current = 0.0
        boosted = min(1.0, current + 0.1)
        assert boosted == pytest.approx(0.1)

    def test_decay_floor(self):
        """Decay: GREATEST(0.01, activation * 0.95)."""
        current = 0.02
        decayed = max(0.01, current * 0.95)
        assert decayed >= 0.01

    def test_decay_normal(self):
        current = 0.5
        decayed = max(0.01, current * 0.95)
        assert decayed == pytest.approx(0.475)

    def test_edge_weight_decay(self):
        """Edge decay: weight * 0.9."""
        current = 0.5
        decayed = current * 0.9
        assert decayed == pytest.approx(0.45)


# --- Vector validation ---


class TestVectorValidation:
    def test_valid_vectors(self):
        from engram.models import AXES
        from engram.write_path import _validate_vectors

        vectors = {axis: [0.1] * 1024 for axis in AXES}
        _validate_vectors(vectors)  # should not raise

    def test_wrong_dimensions(self):
        from engram.models import AXES
        from engram.write_path import _validate_vectors

        vectors = {axis: [0.1] * 512 for axis in AXES}
        with pytest.raises(ValueError, match="dims"):
            _validate_vectors(vectors)

    def test_non_finite_values(self):
        from engram.models import AXES
        from engram.write_path import _validate_vectors

        vectors = {axis: [0.1] * 1024 for axis in AXES}
        vectors["semantic"][0] = float("inf")
        with pytest.raises(ValueError, match="non-finite"):
            _validate_vectors(vectors)


# --- Pydantic models ---


class TestPydanticModels:
    def test_create_memory_request_valid(self):
        from engram.models import CreateMemoryRequest

        req = CreateMemoryRequest(
            content="test content", source_type="conversation"
        )
        assert req.content == "test content"

    def test_create_memory_request_empty_content(self):
        from engram.models import CreateMemoryRequest

        with pytest.raises(ValueError):
            CreateMemoryRequest(content="", source_type="conversation")

    def test_create_memory_request_content_too_long(self):
        from engram.models import CreateMemoryRequest

        with pytest.raises(ValueError):
            CreateMemoryRequest(
                content="x" * 4097, source_type="conversation"
            )

    def test_recall_request_defaults(self):
        from engram.models import RecallRequest

        req = RecallRequest(cue="test cue")
        assert req.top_k == 5
        assert req.min_convergence == 0.0
        assert req.include_edges is False

    def test_recall_request_top_k_bounds(self):
        from engram.models import RecallRequest

        with pytest.raises(ValueError):
            RecallRequest(cue="test", top_k=0)
        with pytest.raises(ValueError):
            RecallRequest(cue="test", top_k=21)

    def test_dimension_scores_defaults(self):
        from engram.models import DimensionScores

        ds = DimensionScores()
        assert ds.temporal == 0.0
        assert ds.semantic == 0.0


# --- Enum values ---


class TestEnums:
    def test_source_types(self):
        from engram.models import SourceType

        expected = {"conversation", "event", "observation", "correction", "system"}
        assert {s.value for s in SourceType} == expected

    def test_axis_names(self):
        from engram.models import AxisName

        expected = {
            "temporal", "emotional", "semantic",
            "sensory", "action", "procedural",
        }
        assert {a.value for a in AxisName} == expected

    def test_edge_types(self):
        from engram.models import EdgeType

        expected = {
            "excitatory", "inhibitory", "associative",
            "temporal", "modulatory",
        }
        assert {e.value for e in EdgeType} == expected

    def test_axes_list_matches_prefixes(self):
        from engram.models import AXES, DIMENSION_PREFIXES

        assert list(DIMENSION_PREFIXES.keys()) == AXES


# --- EngramError ---


class TestEngramError:
    def test_attributes(self):
        from engram.errors import EngramError

        err = EngramError("NOT_FOUND", "Memory not found", 404)
        assert err.code == "NOT_FOUND"
        assert err.message == "Memory not found"
        assert err.status == 404
        assert err.existing_id is None
        assert err.retry_after is None

    def test_optional_fields(self):
        from engram.errors import EngramError

        err = EngramError(
            "DUPLICATE_CONTENT", "Exists", 409,
            existing_id="abc", retry_after=60,
        )
        assert err.existing_id == "abc"
        assert err.retry_after == 60

    def test_is_exception(self):
        from engram.errors import EngramError

        err = EngramError("TEST", "test", 400)
        assert isinstance(err, Exception)
        assert str(err) == "test"


# --- Parse helpers ---


class TestParseHelpers:
    def test_parse_update_count(self):
        from engram.dreamer import _parse_update_count

        assert _parse_update_count("UPDATE 5") == 5
        assert _parse_update_count("UPDATE 0") == 0

    def test_parse_delete_count(self):
        from engram.dreamer import _parse_delete_count

        assert _parse_delete_count("DELETE 3") == 3
        assert _parse_delete_count("DELETE 0") == 0

    def test_parse_invalid(self):
        from engram.dreamer import _parse_update_count

        assert _parse_update_count("") == 0
        assert _parse_update_count("INVALID") == 0
