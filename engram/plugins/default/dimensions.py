"""Default 3-dimension set: semantic, temporal, importance."""

from typing import Dict, List

from engram.engine.interfaces import DimensionSet


class DefaultDimensionSet(DimensionSet):

    def names(self) -> List[str]:
        return ["semantic", "temporal", "importance"]

    def default_weights(self) -> Dict[str, float]:
        return {"semantic": 1.0, "temporal": 0.7, "importance": 0.8}
