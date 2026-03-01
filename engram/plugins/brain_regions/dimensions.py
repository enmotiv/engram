"""Brain-region dimension set — 5 dimensions mapping to neural subsystems."""

from typing import Dict, List

from engram.engine.interfaces import DimensionSet


class BrainRegionDimensionSet(DimensionSet):
    """Maps text to 5 brain-region dimensions.

    hippocampus: episodic details, spatial context, temporal sequences, novelty
    amygdala: observed intensity, valence, sensitivity
    prefrontal_cortex: abstract concepts, goals, semantic meaning
    sensory_cortices: visual, auditory, specific sensory details
    striatum: action sequences, habits, reward patterns
    """

    def names(self) -> List[str]:
        return [
            "hippocampus",
            "amygdala",
            "prefrontal_cortex",
            "sensory_cortices",
            "striatum",
        ]

    def default_weights(self) -> Dict[str, float]:
        return {
            "hippocampus": 0.9,
            "amygdala": 0.8,
            "prefrontal_cortex": 1.0,
            "sensory_cortices": 0.6,
            "striatum": 0.7,
        }
