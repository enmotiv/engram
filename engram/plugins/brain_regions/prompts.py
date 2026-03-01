"""LLM prompt template for brain-region scoring."""

BRAIN_REGION_PROMPT = (
    "Score the following text on 5 brain-region dimensions from 0.0 to 1.0. "
    "Return ONLY a JSON object with these keys:\n"
    "- hippocampus (episodic/temporal/spatial/novelty content)\n"
    "- amygdala (emotional intensity/valence/sensitivity)\n"
    "- prefrontal_cortex (abstract concepts/goals/semantic meaning)\n"
    "- sensory_cortices (visual/auditory/specific sensory details)\n"
    "- striatum (action sequences/habits/reward patterns)\n\n"
    "Text: {text}"
)
