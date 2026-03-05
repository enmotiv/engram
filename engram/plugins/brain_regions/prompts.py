"""LLM prompt templates for brain-region scoring and decomposition."""

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

BRAIN_REGION_DECOMPOSE_PROMPT = (
    "Decompose the following text into 6 brain-region dimensions. "
    "For each region, extract a short text string capturing that dimension's content "
    "from the input and assign a relevance score from 0.0 to 1.0.\n\n"
    "Regions:\n"
    "- hippo: temporal context — when did this happen? recency, sequence, time references\n"
    "- amyg: emotional tone — urgency, importance, emotional valence, intensity\n"
    "- pfc: semantic meaning — concepts, goals, abstract reasoning, plans\n"
    "- sensory: specific details — names, numbers, concrete facts, identifiers\n"
    "- striatum: action context — what is being done or attempted? behavioral goals\n"
    "- cerebellum: procedural memory — is this a repeated pattern or routine? familiar sequences\n\n"
    "Return ONLY a JSON object. Each key is a region name, each value is an object "
    'with "text" (short phrase from input) and "score" (float 0.0-1.0).\n'
    "The text should capture that region's content from the input. "
    "If a dimension has no relevant content, use a brief description of absence "
    "and score near 0.\n\n"
    "Text: {text}"
)
