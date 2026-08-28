"""
router.py -- decides which handler answers a question.

Cheap keyword pass first (free, instant, deterministic, easy to unit-test without an
API key), LLM classification only as a fallback for genuinely ambiguous questions.
This mirrors a normal engineering instinct: don't spend an API call on a decision a
few keyword checks can make correctly most of the time.
"""

import re
from llm import call_llm

CAUSAL_PATTERNS = [
    r"\beffect of\b", r"\bimpact of\b", r"\bcaused?\b", r"\bcausal\b",
    r"\blead(s|ing)? to\b", r"\bdue to\b", r"\bbecause of\b", r"\bdid .* (increase|decrease|improve|reduce)\b",
    r"\bwhat happens if\b", r"\bcounterfactual\b",
]
VIZ_PATTERNS = [
    r"\bplot\b", r"\bchart\b", r"\bgraph\b", r"\bvisuali[sz]e?\b", r"\bhistogram\b",
    r"\bscatter\b", r"\bbar chart\b", r"\bshow me .* (trend|distribution)\b",
]
PREDICTIVE_PATTERNS = [
    r"\bpredict\b", r"\bforecast\b", r"\bmodel\b", r"\bclassif(y|ication)\b",
    r"\bregression\b", r"\bestimate future\b", r"\bwhich features (drive|matter|predict)\b",
    r"\bbuild a .*model\b",
]
EDA_PATTERNS = [
    r"\beda\b", r"\bsummary\b", r"\bdescribe\b", r"\bmissing\b", r"\bclean\b", r"\bexplore\b",
    r"\bdistribution\b", r"\bcorrelation\b", r"\bstats?\b", r"\boutliers?\b", r"\bnull\b",
]

INTENTS = ["causal", "visualization", "predictive", "eda", "general"]


def _keyword_match(question: str):
    q = question.lower()
    if any(re.search(p, q) for p in CAUSAL_PATTERNS):
        return "causal"
    if any(re.search(p, q) for p in VIZ_PATTERNS):
        return "visualization"
    if any(re.search(p, q) for p in PREDICTIVE_PATTERNS):
        return "predictive"
    if any(re.search(p, q) for p in EDA_PATTERNS):
        return "eda"
    return None


def classify_intent(question: str, columns) -> str:
    """Returns one of INTENTS. Tries a fast keyword match first; falls back to a
    one-word LLM classification call only if no keyword pattern matched."""
    hit = _keyword_match(question)
    if hit:
        return hit

    prompt = (
        f"Dataset columns: {list(columns)}\n"
        f"User question: {question}\n\n"
        "Classify this question into exactly one word from this list: "
        "causal, visualization, predictive, eda, general.\n"
        "- causal: asks about the EFFECT/IMPACT/CAUSE of something, not just a pattern.\n"
        "- visualization: asks for a chart/plot/graph.\n"
        "- predictive: asks to predict/forecast/classify a future or unknown value.\n"
        "- eda: asks for summary stats, cleaning, distributions, correlations.\n"
        "- general: anything else (greetings, unrelated questions, unclear requests).\n"
        "Reply with ONLY the single word, nothing else."
    )
    try:
        reply = call_llm(prompt, max_tokens=10, temperature=0.0).strip().lower()
        for intent in INTENTS:
            if intent in reply:
                return intent
    except Exception:
        pass
    return "general"
