"""Feature extraction job — extract structural features from memory content."""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from engram.db.models import Memory
from engram.engine.interfaces import WorkerJob
from engram.engine.llm import get_llm_service

logger = logging.getLogger(__name__)

# Keyword sets for heuristic classification
_DOMAIN_KEYWORDS = {
    "infrastructure": {"server", "cloud", "aws", "ec2", "gcp", "azure", "deploy", "docker",
                       "kubernetes", "database", "rds", "s3", "cdn", "redis", "postgres"},
    "product": {"feature", "user", "launch", "release", "roadmap", "sprint", "mvp",
                "customer", "ux", "ui", "onboarding", "retention"},
    "content": {"blog", "article", "video", "documentation", "tutorial", "guide",
                "write", "publish", "edit", "content"},
    "process": {"workflow", "pipeline", "ci", "cd", "automation", "monitor",
                "alert", "process", "procedure", "sop"},
    "people": {"team", "hire", "meeting", "standup", "manager", "lead",
               "interview", "culture", "feedback", "review"},
    "finance": {"cost", "budget", "revenue", "pricing", "expense", "roi",
                "billing", "subscription", "payment", "invoice"},
}

_ACTION_KEYWORDS = {
    "migration": {"migrate", "migrated", "migration", "move", "moved", "transfer",
                  "transition", "switch", "switched"},
    "optimization": {"optimize", "optimized", "optimization", "improve", "improved",
                     "reduce", "reduced", "faster", "performance", "efficient"},
    "debugging": {"debug", "fix", "fixed", "bug", "error", "issue", "crash",
                  "broken", "fail", "failed"},
    "building": {"build", "built", "create", "created", "implement", "implemented",
                 "develop", "developed", "setup", "configure", "configured"},
    "deciding": {"decide", "decided", "decision", "choose", "chose", "evaluate",
                 "compare", "option", "alternative"},
    "learning": {"learn", "learned", "discover", "discovered", "understand",
                 "understood", "study", "research"},
}

_DYNAMIC_KEYWORDS = {
    "growth": {"grow", "growth", "increase", "expand", "scale", "improve"},
    "decline": {"decline", "decrease", "drop", "reduce", "shrink", "worse"},
    "transition": {"transition", "change", "shift", "move", "migrate", "transform"},
    "stability": {"stable", "consistent", "maintain", "steady", "reliable"},
    "conflict": {"conflict", "disagree", "tension", "challenge", "problem", "issue"},
    "resolution": {"resolve", "resolved", "solution", "fix", "fixed", "settle"},
}

_SCOPE_KEYWORDS = {
    "individual": {"i", "my", "me", "personal"},
    "team": {"team", "we", "our", "group", "squad"},
    "system": {"system", "service", "api", "infrastructure", "platform"},
    "organization": {"company", "org", "organization", "enterprise", "business"},
}

_VALENCE_KEYWORDS = {
    "positive": {"success", "great", "excellent", "improved", "achieved", "completed", "better"},
    "negative": {"failed", "broken", "crash", "error", "problem", "worse", "bad"},
    "mixed": {"however", "but", "although", "tradeoff", "despite"},
}

_ABSTRACTION_KEYWORDS = {
    "concrete_event": {"yesterday", "today", "last", "monday", "tuesday", "january",
                       "february", "march", "just", "recently"},
    "recurring_pattern": {"always", "usually", "often", "every", "pattern", "trend",
                          "typically", "regularly"},
    "general_principle": {"principle", "rule", "best practice", "guideline", "should",
                          "must", "never", "always"},
}

# Feature encoding for feature_vector (32 dims)
# domain: 6, action_type: 6, dynamic: 6, scope: 4, causality: 4, valence: 3, abstraction: 3 = 32
_FEATURE_ENCODING = {
    "domain": ["infrastructure", "product", "content", "process", "people", "finance"],
    "action_type": ["migration", "optimization", "debugging", "building", "deciding", "learning"],
    "dynamic": ["growth", "decline", "transition", "stability", "conflict", "resolution"],
    "scope": ["individual", "team", "system", "organization"],
    "causality": ["cause", "effect", "correlation", "independent"],
    "valence": ["positive", "negative", "mixed"],
    "abstraction": ["concrete_event", "recurring_pattern", "general_principle"],
}


def _classify_field(text_lower: str, keywords: Dict[str, set]) -> str:
    """Match content against keyword sets, return best-matching category."""
    words = set(re.findall(r"\w+", text_lower))
    best = None
    best_count = 0
    for category, kws in keywords.items():
        hits = len(words & kws)
        if hits > best_count:
            best = category
            best_count = hits
    return best or list(keywords.keys())[0]


def extract_features_heuristic(content: str) -> Dict[str, str]:
    """Extract structural features using keyword matching."""
    lower = content.lower()
    features = {
        "domain": _classify_field(lower, _DOMAIN_KEYWORDS),
        "action_type": _classify_field(lower, _ACTION_KEYWORDS),
        "dynamic": _classify_field(lower, _DYNAMIC_KEYWORDS),
        "scope": _classify_field(lower, _SCOPE_KEYWORDS),
        "valence": _classify_field(lower, _VALENCE_KEYWORDS),
        "abstraction": _classify_field(lower, _ABSTRACTION_KEYWORDS),
    }

    # Causality heuristic
    if any(w in lower for w in ["because", "caused", "due to", "reason", "led to"]):
        features["causality"] = "cause"
    elif any(w in lower for w in ["resulted", "consequence", "therefore", "effect"]):
        features["causality"] = "effect"
    elif any(w in lower for w in ["correlat", "associated", "related"]):
        features["causality"] = "correlation"
    else:
        features["causality"] = "independent"

    return features


_FEATURE_EXTRACTION_PROMPT = (
    "Classify this memory on the following dimensions. Return JSON only, "
    "with exactly these keys and one value per key.\n\n"
    "domain: infrastructure | product | content | process | people | finance\n"
    "action_type: migration | optimization | debugging | building | deciding | learning\n"
    "dynamic: growth | decline | transition | stability | conflict | resolution\n"
    "scope: individual | team | system | organization\n"
    "causality: cause | effect | correlation | independent\n"
    "valence: positive | negative | mixed\n"
    "abstraction: concrete_event | recurring_pattern | general_principle\n\n"
    "Memory: {content}"
)

_VALID_VALUES = {
    "domain": {"infrastructure", "product", "content", "process", "people", "finance"},
    "action_type": {"migration", "optimization", "debugging", "building", "deciding", "learning"},
    "dynamic": {"growth", "decline", "transition", "stability", "conflict", "resolution"},
    "scope": {"individual", "team", "system", "organization"},
    "causality": {"cause", "effect", "correlation", "independent"},
    "valence": {"positive", "negative", "mixed"},
    "abstraction": {"concrete_event", "recurring_pattern", "general_principle"},
}


async def extract_features_llm(content: str) -> Optional[Dict[str, str]]:
    """Extract features via LLM. Returns None on failure."""
    llm = get_llm_service()
    if not llm.is_available():
        return None

    prompt = _FEATURE_EXTRACTION_PROMPT.format(content=content)
    try:
        raw = await llm.complete(prompt, max_tokens=150, temperature=0.1, json_mode=True)
        features = json.loads(raw)

        # Validate all keys present with valid values
        for key, valid in _VALID_VALUES.items():
            if key not in features or features[key] not in valid:
                return None
        return {k: features[k] for k in _VALID_VALUES}
    except Exception:
        logger.debug("LLM feature extraction failed, will use heuristic")
        return None


def features_to_vector(features: Dict[str, str]) -> List[float]:
    """One-hot encode features into a 32-dim vector."""
    vec = []
    for field, categories in _FEATURE_ENCODING.items():
        value = features.get(field, categories[0])
        for cat in categories:
            vec.append(1.0 if cat == value else 0.0)
    return vec


class FeatureExtractionJob(WorkerJob):
    def name(self) -> str:
        return "feature_extraction"

    async def should_run(self, namespace: str, **kwargs) -> bool:
        return True

    async def _extract(self, content: str) -> Dict[str, str]:
        """Try LLM extraction, fall back to heuristic."""
        features = await extract_features_llm(content)
        if features is not None:
            return features
        return extract_features_heuristic(content)

    async def execute(self, namespace: str, **kwargs) -> dict:
        db: AsyncSession = kwargs["db"]
        memory_id = kwargs.get("memory_id")

        if memory_id:
            mem = await db.get(Memory, memory_id)
            if mem is None:
                return {"processed": 0}
            features = await self._extract(mem.content)
            mem.features = features
            mem.feature_vector = features_to_vector(features)
            await db.flush()
            return {"processed": 1, "features": features}

        # Batch mode: process all memories without features
        stmt = select(Memory).where(
            Memory.namespace == namespace,
            or_(
                Memory.features == None,  # noqa: E711
                Memory.features == {},
            ),
        )
        result = await db.execute(stmt)
        memories = list(result.scalars().all())

        for mem in memories:
            features = await self._extract(mem.content)
            mem.features = features
            mem.feature_vector = features_to_vector(features)

        await db.flush()
        return {"processed": len(memories)}
