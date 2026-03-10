"""Integration tests for read path: recall, trace, filtering."""


from tests.conftest import auth_headers


async def test_recall_returns_results(client, owner_a, seeded_memories):
    _, key = owner_a
    response = await client.post(
        "/v1/recall",
        json={"cue": "dark mode preference"},
        headers=auth_headers(key),
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data["memories"]) > 0
    assert "confidence" in data

    mem = data["memories"][0]
    assert "id" in mem
    assert "content" in mem
    assert "convergence_score" in mem
    assert "dimension_scores" in mem
    assert "matched_axes" in mem
    assert "activation_level" in mem
    assert "salience" in mem


async def test_recall_with_trace(client, owner_a, seeded_memories):
    _, key = owner_a
    response = await client.post(
        "/v1/recall?trace=true",
        json={"cue": "programming language"},
        headers=auth_headers(key),
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert "trace" in data

    trace = data["trace"]
    assert "correlation_id" in trace
    assert "total_ms" in trace
    assert "embedding_ms" in trace
    assert "per_dimension_results" in trace
    assert "unique_candidates" in trace
    assert "convergence_scores" in trace
    assert "spreading_activation" in trace
    assert "reconsolidation" in trace
    assert "post_filter" in trace


async def test_recall_top_k(client, owner_a, seeded_memories):
    _, key = owner_a
    response = await client.post(
        "/v1/recall",
        json={"cue": "any topic", "top_k": 2},
        headers=auth_headers(key),
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data["memories"]) <= 2


async def test_recall_min_convergence_filters(client, owner_a, seeded_memories):
    _, key = owner_a
    response = await client.post(
        "/v1/recall",
        json={"cue": "anything", "min_convergence": 999.0},
        headers=auth_headers(key),
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data["memories"]) == 0
    assert data["confidence"] == "low"


async def test_recall_include_edges(client, owner_a, seeded_memories):
    _, key = owner_a
    response = await client.post(
        "/v1/recall",
        json={"cue": "test", "include_edges": True},
        headers=auth_headers(key),
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert "edges" in data


async def test_recall_empty_cue_rejected(client, owner_a):
    _, key = owner_a
    response = await client.post(
        "/v1/recall",
        json={"cue": ""},
        headers=auth_headers(key),
    )
    assert response.status_code == 400


async def test_recall_cue_too_long(client, owner_a):
    _, key = owner_a
    response = await client.post(
        "/v1/recall",
        json={"cue": "x" * 4097},
        headers=auth_headers(key),
    )
    assert response.status_code == 400


async def test_recall_empty_db(client, owner_c):
    """Fresh owner with no memories. Recall should return empty with low confidence."""
    _, key = owner_c
    response = await client.post(
        "/v1/recall",
        json={"cue": "anything"},
        headers=auth_headers(key),
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data["memories"]) == 0
    assert data["confidence"] == "low"


async def test_recall_dimension_scores_structure(client, owner_a, seeded_memories):
    _, key = owner_a
    response = await client.post(
        "/v1/recall",
        json={"cue": "test query"},
        headers=auth_headers(key),
    )
    data = response.json()["data"]
    if data["memories"]:
        scores = data["memories"][0]["dimension_scores"]
        expected_axes = {
            "temporal", "emotional", "semantic",
            "sensory", "action", "procedural",
        }
        assert set(scores.keys()) == expected_axes


async def test_recall_response_model(client, owner_a, seeded_memories):
    """Verify all MemoryResponse fields are present."""
    _, key = owner_a
    response = await client.post(
        "/v1/recall",
        json={"cue": "test"},
        headers=auth_headers(key),
    )
    data = response.json()["data"]
    if data["memories"]:
        mem = data["memories"][0]
        required = {
            "id", "content", "content_hash", "convergence_score",
            "activation_level", "dimension_scores", "matched_axes",
            "metadata", "salience", "created_at", "last_accessed",
            "access_count", "source_type",
        }
        assert required.issubset(set(mem.keys()))
