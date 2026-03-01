"""Default plugin — 3 dimensions (semantic, temporal, importance) + sentence-transformers encoder."""

from engram.plugins.default.dimensions import DefaultDimensionSet
from engram.plugins.default.encoder import DefaultEncoder


def register(registry):
    registry.register_dimensions(DefaultDimensionSet())
    registry.register_encoder(DefaultEncoder())
