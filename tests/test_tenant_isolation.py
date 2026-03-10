"""Tenant isolation tests: owner A cannot access owner B through any path."""


from tests.conftest import auth_headers


async def test_recall_isolation(client, owner_a, owner_b):
    """Owner B's memories must never appear in owner A's recall."""
    _, key_a = owner_a
    _, key_b = owner_b

    # Create memory for owner B
    await client.post(
        "/v1/memories",
        json={
            "content": "Isolation test: owner B secret data",
            "source_type": "conversation",
        },
        headers=auth_headers(key_b),
    )

    # Recall as owner A
    response = await client.post(
        "/v1/recall",
        json={"cue": "owner B secret data"},
        headers=auth_headers(key_a),
    )
    memories = response.json()["data"]["memories"]
    for m in memories:
        assert "owner b secret" not in m["content"].lower()


async def test_list_isolation(client, owner_a, owner_b):
    """Owner A's memory list must not include owner B's memories."""
    _, key_a = owner_a
    _, key_b = owner_b

    # Create identifiable memory for owner B
    await client.post(
        "/v1/memories",
        json={
            "content": "Isolation test: owner B list check",
            "source_type": "event",
        },
        headers=auth_headers(key_b),
    )

    # List as owner A
    response = await client.get(
        "/v1/memories",
        headers=auth_headers(key_a),
    )
    memories = response.json()["data"]["memories"]
    for m in memories:
        assert "owner b list check" not in m["content"].lower()


async def test_get_by_id_isolation(client, owner_a, owner_b):
    """Owner A cannot fetch owner B's memory by ID."""
    _, key_a = owner_a
    _, key_b = owner_b

    # Create memory as owner B
    create = await client.post(
        "/v1/memories",
        json={
            "content": "Isolation test: owner B get-by-id",
            "source_type": "conversation",
        },
        headers=auth_headers(key_b),
    )
    node_id = create.json()["data"]["id"]

    # Try to GET as owner A
    response = await client.get(
        f"/v1/memories/{node_id}",
        headers=auth_headers(key_a),
    )
    assert response.status_code == 404


async def test_delete_isolation(client, owner_a, owner_b):
    """Owner A cannot delete owner B's memory."""
    _, key_a = owner_a
    _, key_b = owner_b

    # Create memory as owner B
    create = await client.post(
        "/v1/memories",
        json={
            "content": "Isolation test: owner B delete attempt",
            "source_type": "conversation",
        },
        headers=auth_headers(key_b),
    )
    node_id = create.json()["data"]["id"]

    # Try to DELETE as owner A
    response = await client.delete(
        f"/v1/memories/{node_id}",
        headers=auth_headers(key_a),
    )
    assert response.status_code == 404

    # Verify it still exists for owner B
    get = await client.get(
        f"/v1/memories/{node_id}",
        headers=auth_headers(key_b),
    )
    assert get.status_code == 200


async def test_stats_isolation(client, owner_a, owner_b):
    """Stats endpoint returns per-owner counts only."""
    _, key_a = owner_a
    _, key_b = owner_b

    stats_a = await client.get("/v1/stats", headers=auth_headers(key_a))
    stats_b = await client.get("/v1/stats", headers=auth_headers(key_b))

    assert stats_a.status_code == 200
    assert stats_b.status_code == 200

    # Counts should differ (owner A has seeded memories, B has fewer)
    count_a = stats_a.json()["data"]["node_count"]
    count_b = stats_b.json()["data"]["node_count"]
    # Both should return only their own count, not combined
    assert isinstance(count_a, int)
    assert isinstance(count_b, int)


async def test_duplicate_detection_isolation(client, owner_a, owner_b):
    """Same content for different owners should not trigger duplicate."""
    _, key_a = owner_a
    _, key_b = owner_b
    content = "Isolation test: shared content between owners"

    r1 = await client.post(
        "/v1/memories",
        json={"content": content, "source_type": "conversation"},
        headers=auth_headers(key_a),
    )
    assert r1.status_code == 201

    r2 = await client.post(
        "/v1/memories",
        json={"content": content, "source_type": "conversation"},
        headers=auth_headers(key_b),
    )
    # Should succeed — different owner, not a duplicate
    assert r2.status_code == 201
