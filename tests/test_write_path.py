"""Integration tests for write path: create, duplicate, validation."""

from tests.conftest import auth_headers


async def test_create_memory(client, owner_a):
    _, key = owner_a
    response = await client.post(
        "/v1/memories",
        json={
            "content": "Write path test: user prefers dark mode",
            "source_type": "conversation",
        },
        headers=auth_headers(key),
    )
    assert response.status_code == 201
    data = response.json()["data"]
    assert "id" in data
    assert "salience" in data


async def test_create_with_metadata(client, owner_a):
    _, key = owner_a
    response = await client.post(
        "/v1/memories",
        json={
            "content": "Write path test: metadata support",
            "source_type": "observation",
            "metadata": {"source": "test", "priority": "high"},
        },
        headers=auth_headers(key),
    )
    assert response.status_code == 201


async def test_create_with_session_id(client, owner_a):
    _, key = owner_a
    response = await client.post(
        "/v1/memories",
        json={
            "content": "Write path test: session tracking",
            "source_type": "conversation",
            "session_id": "00000000-0000-0000-0000-000000000001",
        },
        headers=auth_headers(key),
    )
    assert response.status_code == 201


async def test_duplicate_content(client, owner_a):
    _, key = owner_a
    content = "Write path test: exact duplicate detection"

    r1 = await client.post(
        "/v1/memories",
        json={"content": content, "source_type": "conversation"},
        headers=auth_headers(key),
    )
    assert r1.status_code == 201

    r2 = await client.post(
        "/v1/memories",
        json={"content": content, "source_type": "conversation"},
        headers=auth_headers(key),
    )
    assert r2.status_code == 409
    error = r2.json()["error"]
    assert error["code"] == "DUPLICATE_CONTENT"
    assert "existing_id" in error


async def test_content_too_long(client, owner_a):
    _, key = owner_a
    response = await client.post(
        "/v1/memories",
        json={
            "content": "x" * 4097,
            "source_type": "conversation",
        },
        headers=auth_headers(key),
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_INPUT"


async def test_missing_content(client, owner_a):
    _, key = owner_a
    response = await client.post(
        "/v1/memories",
        json={"source_type": "conversation"},
        headers=auth_headers(key),
    )
    assert response.status_code == 400


async def test_invalid_source_type(client, owner_a):
    _, key = owner_a
    response = await client.post(
        "/v1/memories",
        json={"content": "test", "source_type": "invalid_type"},
        headers=auth_headers(key),
    )
    assert response.status_code == 400


async def test_empty_content(client, owner_a):
    _, key = owner_a
    response = await client.post(
        "/v1/memories",
        json={"content": "", "source_type": "conversation"},
        headers=auth_headers(key),
    )
    assert response.status_code == 400


async def test_get_memory(client, owner_a):
    _, key = owner_a
    create = await client.post(
        "/v1/memories",
        json={
            "content": "Write path test: get by ID",
            "source_type": "event",
        },
        headers=auth_headers(key),
    )
    node_id = create.json()["data"]["id"]

    response = await client.get(
        f"/v1/memories/{node_id}",
        headers=auth_headers(key),
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["id"] == node_id
    assert data["content"] == "Write path test: get by ID"
    assert data["source_type"] == "event"


async def test_get_memory_not_found(client, owner_a):
    _, key = owner_a
    response = await client.get(
        "/v1/memories/00000000-0000-0000-0000-000000000099",
        headers=auth_headers(key),
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


async def test_list_memories(client, owner_a, seeded_memories):
    _, key = owner_a
    response = await client.get(
        "/v1/memories",
        headers=auth_headers(key),
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert "memories" in data
    assert len(data["memories"]) > 0
    assert "next_cursor" in data


async def test_list_memories_with_limit(client, owner_a, seeded_memories):
    _, key = owner_a
    response = await client.get(
        "/v1/memories?limit=2",
        headers=auth_headers(key),
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data["memories"]) <= 2


async def test_list_memories_pagination(client, owner_a, seeded_memories):
    _, key = owner_a

    # Page 1
    r1 = await client.get(
        "/v1/memories?limit=2",
        headers=auth_headers(key),
    )
    page1 = r1.json()["data"]
    cursor = page1["next_cursor"]
    assert cursor is not None

    # Page 2
    r2 = await client.get(
        f"/v1/memories?limit=2&cursor={cursor}",
        headers=auth_headers(key),
    )
    page2 = r2.json()["data"]

    # No overlap
    ids_1 = {m["id"] for m in page1["memories"]}
    ids_2 = {m["id"] for m in page2["memories"]}
    assert ids_1.isdisjoint(ids_2)


async def test_list_memories_sort(client, owner_a, seeded_memories):
    _, key = owner_a
    response = await client.get(
        "/v1/memories?sort=created_at:asc",
        headers=auth_headers(key),
    )
    assert response.status_code == 200


async def test_list_memories_invalid_sort(client, owner_a):
    _, key = owner_a
    response = await client.get(
        "/v1/memories?sort=invalid",
        headers=auth_headers(key),
    )
    assert response.status_code == 400


async def test_list_memories_filter_source_type(client, owner_a, seeded_memories):
    _, key = owner_a
    response = await client.get(
        "/v1/memories?source_type=conversation",
        headers=auth_headers(key),
    )
    assert response.status_code == 200
    for m in response.json()["data"]["memories"]:
        assert m["source_type"] == "conversation"


async def test_delete_memory(client, owner_a):
    _, key = owner_a
    create = await client.post(
        "/v1/memories",
        json={
            "content": "Write path test: delete me",
            "source_type": "conversation",
        },
        headers=auth_headers(key),
    )
    node_id = create.json()["data"]["id"]

    response = await client.delete(
        f"/v1/memories/{node_id}",
        headers=auth_headers(key),
    )
    assert response.status_code == 200
    assert response.json()["data"]["deleted"] is True

    # Verify it's gone
    get = await client.get(
        f"/v1/memories/{node_id}",
        headers=auth_headers(key),
    )
    assert get.status_code == 404


async def test_delete_memory_not_found(client, owner_a):
    _, key = owner_a
    response = await client.delete(
        "/v1/memories/00000000-0000-0000-0000-000000000099",
        headers=auth_headers(key),
    )
    assert response.status_code == 404


async def test_delete_idempotent(client, owner_a):
    _, key = owner_a
    create = await client.post(
        "/v1/memories",
        json={
            "content": "Write path test: double delete",
            "source_type": "conversation",
        },
        headers=auth_headers(key),
    )
    node_id = create.json()["data"]["id"]

    await client.delete(f"/v1/memories/{node_id}", headers=auth_headers(key))
    r2 = await client.delete(
        f"/v1/memories/{node_id}", headers=auth_headers(key)
    )
    assert r2.status_code == 404
