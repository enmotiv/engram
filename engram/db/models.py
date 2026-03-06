"""SQLAlchemy ORM models for the memories and edges tables."""

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Memory(Base):
    __tablename__ = "memories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    namespace = Column(Text, nullable=False, index=True)
    content = Column(Text, nullable=False)
    memory_type = Column(Text, default="episodic")
    embedding = Column(Vector(1024))
    hippo_embedding = Column(Vector(1024))
    amyg_embedding = Column(Vector(1024))
    pfc_embedding = Column(Vector(1024))
    sensory_embedding = Column(Vector(1024))
    striatum_embedding = Column(Vector(1024))
    cerebellum_embedding = Column(Vector(1024))
    dimension_vectors = Column(JSONB, default=dict)
    dimension_scores = Column(JSONB, default=dict)
    features = Column(JSONB, default=dict)
    feature_vector = Column(Vector(32))
    # Node vitals — updated by different systems:
    # activation: recomputed by dreamer EdgeDecayJob each cycle (edge_strength × recency)
    # salience: set at write time from mean axis scores; updated by AxisRescoringJob
    # access_count, last_accessed: updated by reconsolidation on every retrieval
    activation = Column(Float, default=0.0)
    salience = Column(Float, default=0.5)
    access_count = Column(Integer, default=0)
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    last_accessed = Column(DateTime(timezone=True))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    outgoing_edges = relationship(
        "Edge", foreign_keys="Edge.source_memory_id", back_populates="source", cascade="all, delete-orphan"
    )
    incoming_edges = relationship(
        "Edge", foreign_keys="Edge.target_memory_id", back_populates="target", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "namespace": self.namespace,
            "content": self.content,
            "memory_type": self.memory_type,
            "embedding": list(self.embedding) if self.embedding is not None else None,
            "dimension_vectors": self.dimension_vectors or {},
            "dimension_scores": self.dimension_scores or {},
            "features": self.features or {},
            "feature_vector": list(self.feature_vector) if self.feature_vector is not None else None,
            "activation": self.activation,
            "salience": self.salience,
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


VALID_EDGE_TYPES = frozenset({"excitatory", "inhibitory", "associative", "temporal", "modulatory"})


class Graph(Base):
    """Graph snapshot — one row per snapshot per namespace."""

    __tablename__ = "graphs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    namespace = Column(Text, nullable=False)
    content = Column(Text, nullable=False)  # mermaid syntax string
    node_count = Column(Integer, nullable=False)
    cluster_count = Column(Integer, default=0)
    memory_count = Column(Integer, nullable=False)  # total memories represented
    version = Column(Integer, default=1)
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    metadata_ = Column("metadata", JSONB, default=dict)

    __table_args__ = (
        Index("idx_graphs_namespace", "namespace"),
        Index("idx_graphs_namespace_latest", "namespace", "created_at"),
    )


class Edge(Base):
    __tablename__ = "edges"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_memory_id = Column(
        UUID(as_uuid=True), ForeignKey("memories.id", ondelete="CASCADE"), nullable=False
    )
    target_memory_id = Column(
        UUID(as_uuid=True), ForeignKey("memories.id", ondelete="CASCADE"), nullable=False
    )
    edge_type = Column(Text, nullable=False)
    weight = Column(Float, default=0.5)
    context = Column(JSONB, default=dict)
    context_features = Column(JSONB)
    context_embedding = Column(Vector(1024))
    namespace = Column(Text, default="default")
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    last_activated = Column(DateTime(timezone=True))

    source = relationship("Memory", foreign_keys=[source_memory_id], back_populates="outgoing_edges")
    target = relationship("Memory", foreign_keys=[target_memory_id], back_populates="incoming_edges")

    __table_args__ = (
        UniqueConstraint("source_memory_id", "target_memory_id", "edge_type", name="uq_edge_source_target_type"),
        Index("idx_edges_source", "source_memory_id"),
        Index("idx_edges_target", "target_memory_id"),
        Index("idx_edges_source_type", "source_memory_id", "edge_type"),
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "source_memory_id": str(self.source_memory_id),
            "target_memory_id": str(self.target_memory_id),
            "edge_type": self.edge_type,
            "weight": self.weight,
            "context": self.context or {},
            "context_features": self.context_features,
            "namespace": self.namespace,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_activated": self.last_activated.isoformat() if self.last_activated else None,
        }
