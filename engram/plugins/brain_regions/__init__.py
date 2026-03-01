"""Brain-region plugin — 5 neural subsystem dimensions + LLM/heuristic encoder."""

from engram.plugins.brain_regions.dimensions import BrainRegionDimensionSet
from engram.plugins.brain_regions.encoder import BrainRegionEncoder


def register(registry):
    registry.register_dimensions(BrainRegionDimensionSet())
    registry.register_encoder(BrainRegionEncoder())
