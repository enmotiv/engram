"""Security tests: auth, injection, payloads, rate limiting, error safety."""


from tests.conftest import auth_headers

# --- Authentication ---


async def test_missing_auth_header(client):
    response = await client.post(
        "/v1/memories",
        json={"content": "test", "source_type": "conversation"},
    )
    assert response.status_code == 401
    error = response.json()["error"]
    assert error["code"] == "UNAUTHORIZED"
    assert "correlation_id" in error


async def test_invalid_auth_format(client):
    response = await client.post(
        "/v1/memories",
        json={"content": "test", "source_type": "conversation"},
        headers={"Authorization": "Basic abc123"},
    )
    assert response.status_code == 401


async def test_empty_bearer_token(client):
    response = await client.post(
        "/v1/memories",
        json={"content": "test", "source_type": "conversation"},
        headers={"Authorization": "Bearer "},
    )
    assert response.status_code == 401


async def test_invalid_api_key(client):
    response = await client.post(
        "/v1/memories",
        json={"content": "test", "source_type": "conversation"},
        headers={"Authorization": "Bearer totally_fake_key_12345"},
    )
    assert response.status_code == 401


async def test_revoked_key(client, db_pool, owner_a):
    """A revoked key should be rejected."""
    import hashlib
    from uuid import uuid4

    owner_id, _ = owner_a
    raw_key = f"revoked_{uuid4().hex[:16]}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO api_keys (owner_id, key_hash, label, revoked_at) "
            "VALUES ($1, $2, $3, NOW())",
            owner_id,
            key_hash,
            "revoked-test",
        )

    response = await client.post(
        "/v1/memories",
        json={"content": "test", "source_type": "conversation"},
        headers=auth_headers(raw_key),
    )
    assert response.status_code == 401


# --- SQL injection ---


async def test_sql_injection_in_content(client, owner_a):
    """SQL injection in content should be stored as a string, not executed."""
    _, key = owner_a
    payloads = [
        "'; DROP TABLE memory_nodes; --",
        "' OR '1'='1",
        "Robert'); DROP TABLE memory_nodes;--",
        "1; SELECT * FROM api_keys",
    ]
    for payload in payloads:
        response = await client.post(
            "/v1/memories",
            json={"content": payload, "source_type": "conversation"},
            headers=auth_headers(key),
        )
        # Should succeed (stored as content) or 409 (if duplicate)
        assert response.status_code in (201, 409)


async def test_sql_injection_in_source_type(client, owner_a):
    """SQL injection in source_type should be rejected by validation."""
    _, key = owner_a
    response = await client.post(
        "/v1/memories",
        json={
            "content": "test",
            "source_type": "conversation'; DROP TABLE memory_nodes;--",
        },
        headers=auth_headers(key),
    )
    assert response.status_code == 400


async def test_sql_injection_in_sort(client, owner_a):
    """SQL injection in sort parameter should be rejected."""
    _, key = owner_a
    response = await client.get(
        "/v1/memories?sort=created_at:desc;DROP TABLE memory_nodes",
        headers=auth_headers(key),
    )
    assert response.status_code == 400


async def test_sql_injection_in_cursor(client, owner_a):
    """SQL injection in cursor should be rejected."""
    _, key = owner_a
    response = await client.get(
        "/v1/memories?cursor='; DROP TABLE memory_nodes;--",
        headers=auth_headers(key),
    )
    assert response.status_code == 400


# --- Payload validation ---


async def test_oversized_content(client, owner_a):
    _, key = owner_a
    response = await client.post(
        "/v1/memories",
        json={"content": "x" * 4097, "source_type": "conversation"},
        headers=auth_headers(key),
    )
    assert response.status_code == 400


async def test_oversized_cue(client, owner_a):
    _, key = owner_a
    response = await client.post(
        "/v1/recall",
        json={"cue": "x" * 4097},
        headers=auth_headers(key),
    )
    assert response.status_code == 400


async def test_invalid_json(client, owner_a):
    _, key = owner_a
    response = await client.post(
        "/v1/memories",
        content=b"not json",
        headers={**auth_headers(key), "Content-Type": "application/json"},
    )
    assert response.status_code == 400


async def test_missing_required_fields(client, owner_a):
    _, key = owner_a
    response = await client.post(
        "/v1/memories",
        json={},
        headers=auth_headers(key),
    )
    assert response.status_code == 400


async def test_invalid_uuid_path(client, owner_a):
    _, key = owner_a
    response = await client.get(
        "/v1/memories/not-a-uuid",
        headers=auth_headers(key),
    )
    assert response.status_code == 422 or response.status_code == 400


async def test_top_k_out_of_range(client, owner_a):
    _, key = owner_a
    response = await client.post(
        "/v1/recall",
        json={"cue": "test", "top_k": 100},
        headers=auth_headers(key),
    )
    assert response.status_code == 400


async def test_negative_min_convergence(client, owner_a):
    _, key = owner_a
    response = await client.post(
        "/v1/recall",
        json={"cue": "test", "min_convergence": -1.0},
        headers=auth_headers(key),
    )
    assert response.status_code == 400


# --- Rate limiting ---


async def test_rate_limit_enforced(client, db_pool, owner_a):
    """Exceeding write rate limit should return 429."""
    import hashlib
    from uuid import uuid4

    from engram.auth import _rate_limiter

    # Create a key with very low limit
    owner_id, _ = owner_a
    raw_key = f"ratelimit_{uuid4().hex[:16]}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO api_keys "
            "(owner_id, key_hash, label, rate_limit_writes) "
            "VALUES ($1, $2, $3, $4)",
            owner_id,
            key_hash,
            "rate-limit-test",
            3,  # very low limit
        )

    responses = []
    for _i in range(5):
        r = await client.post(
            "/v1/memories",
            json={
                "content": f"Rate limit test message {uuid4().hex}",
                "source_type": "conversation",
            },
            headers=auth_headers(raw_key),
        )
        responses.append(r.status_code)

    assert 429 in responses

    # Clean up rate limiter state
    bucket = f"write:{key_hash}"
    _rate_limiter._windows.pop(bucket, None)


async def test_rate_limit_headers(client, owner_a):
    """Rate limit info should be in response headers."""
    _, key = owner_a
    response = await client.get(
        "/v1/memories",
        headers=auth_headers(key),
    )
    assert "x-ratelimit-remaining" in response.headers
    assert "x-ratelimit-reset" in response.headers


async def test_rate_limit_429_has_retry_after(client, db_pool, owner_a):
    """429 response should include Retry-After header."""
    import hashlib
    from uuid import uuid4

    from engram.auth import _rate_limiter

    owner_id, _ = owner_a
    raw_key = f"retry_{uuid4().hex[:16]}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO api_keys "
            "(owner_id, key_hash, label, rate_limit_reads) "
            "VALUES ($1, $2, $3, $4)",
            owner_id,
            key_hash,
            "retry-test",
            1,  # 1 read per minute
        )

    # First request succeeds
    await client.get("/v1/memories", headers=auth_headers(raw_key))
    # Second should be rate limited
    r2 = await client.get("/v1/memories", headers=auth_headers(raw_key))
    assert r2.status_code == 429
    assert "retry-after" in r2.headers

    bucket = f"read:{key_hash}"
    _rate_limiter._windows.pop(bucket, None)


# --- Error response safety ---


async def test_error_never_leaks_db_info(client):
    """Error responses should not contain database details."""
    response = await client.post(
        "/v1/memories",
        json={"content": "test", "source_type": "conversation"},
    )
    body = response.text.lower()
    leak_terms = [
        "postgresql", "asyncpg", "traceback",
        "stack trace", "select", "insert",
    ]
    for term in leak_terms:
        assert term not in body, f"Error leaked DB info: {term}"


async def test_correlation_id_on_all_errors(client, owner_a):
    """Every error response should have a correlation_id."""
    _, key = owner_a
    error_requests = [
        # 400: invalid input
        client.post(
            "/v1/memories",
            json={"content": "", "source_type": "conversation"},
            headers=auth_headers(key),
        ),
        # 401: no auth
        client.post(
            "/v1/memories",
            json={"content": "test", "source_type": "conversation"},
        ),
        # 404: not found
        client.get(
            "/v1/memories/00000000-0000-0000-0000-000000000099",
            headers=auth_headers(key),
        ),
    ]
    for coro in error_requests:
        r = await coro
        if r.status_code >= 400:
            error = r.json().get("error", {})
            assert "correlation_id" in error, (
                f"Missing correlation_id on {r.status_code}"
            )


async def test_correlation_id_in_response_header(client, owner_a):
    """X-Correlation-ID header should be present on all responses."""
    _, key = owner_a
    response = await client.get(
        "/v1/memories",
        headers=auth_headers(key),
    )
    assert "x-correlation-id" in response.headers
    assert response.headers["x-correlation-id"].startswith("req_")


# --- Health endpoint (no auth) ---


async def test_health_no_auth(client):
    """Health endpoint should not require authentication."""
    response = await client.get("/v1/health")
    assert response.status_code in (200, 503)
    data = response.json()
    assert "status" in data
    assert "db" in data
