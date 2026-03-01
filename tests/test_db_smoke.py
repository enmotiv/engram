"""Smoke tests: verify memories and edges tables exist with correct columns."""

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from engram.db.models import Edge, Memory


@pytest.mark.asyncio
async def test_insert_and_query_memory(db):
    mem = Memory(
        namespace="test",
        content="hello world",
        memory_type="episodic",
        salience=0.7,
    )
    db.add(mem)
    await db.flush()

    result = await db.get(Memory, mem.id)
    assert result is not None
    assert result.content == "hello world"
    assert result.namespace == "test"
    assert result.memory_type == "episodic"
    assert result.salience == 0.7
    assert result.access_count == 0
    assert result.activation == 0.0


@pytest.mark.asyncio
async def test_insert_and_query_edge(db):
    m1 = Memory(namespace="test", content="memory one")
    m2 = Memory(namespace="test", content="memory two")
    db.add_all([m1, m2])
    await db.flush()

    edge = Edge(
        source_memory_id=m1.id,
        target_memory_id=m2.id,
        edge_type="excitatory",
        weight=0.8,
        namespace="test",
    )
    db.add(edge)
    await db.flush()

    result = await db.get(Edge, edge.id)
    assert result is not None
    assert result.edge_type == "excitatory"
    assert result.weight == 0.8
    assert result.source_memory_id == m1.id
    assert result.target_memory_id == m2.id


@pytest.mark.asyncio
async def test_edge_unique_constraint(db):
    m1 = Memory(namespace="test", content="node A")
    m2 = Memory(namespace="test", content="node B")
    db.add_all([m1, m2])
    await db.flush()

    e1 = Edge(source_memory_id=m1.id, target_memory_id=m2.id, edge_type="associative", weight=0.5)
    db.add(e1)
    await db.flush()

    e2 = Edge(source_memory_id=m1.id, target_memory_id=m2.id, edge_type="associative", weight=0.9)
    db.add(e2)
    with pytest.raises(IntegrityError):
        await db.flush()


@pytest.mark.asyncio
async def test_cascade_delete(db):
    m1 = Memory(namespace="test", content="parent")
    m2 = Memory(namespace="test", content="child")
    db.add_all([m1, m2])
    await db.flush()

    edge = Edge(source_memory_id=m1.id, target_memory_id=m2.id, edge_type="temporal", weight=0.6)
    db.add(edge)
    await db.flush()
    edge_id = edge.id

    await db.delete(m1)
    await db.flush()

    result = await db.get(Edge, edge_id)
    assert result is None


@pytest.mark.asyncio
async def test_memory_to_dict(db):
    mem = Memory(namespace="test", content="dict test", memory_type="semantic")
    db.add(mem)
    await db.flush()

    d = mem.to_dict()
    assert d["namespace"] == "test"
    assert d["content"] == "dict test"
    assert d["memory_type"] == "semantic"
    assert isinstance(d["id"], str)
    assert d["embedding"] is None
    assert d["dimension_scores"] == {}
