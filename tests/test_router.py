import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""Tests the keyword fast-path (no API key needed) and the LLM fallback (mocked)."""
from unittest.mock import patch
import router


def test_keyword_paths():
    cases = {
        "What was the effect of the campaign on sales?": "causal",
        "Did the marketing spend cause higher sales?": "causal",
        "Show me a bar chart of sales by region": "visualization",
        "Plot the distribution of ad spend": "visualization",
        "Predict next month's sales using a regression model": "predictive",
        "Build a model to classify high vs low performing stores": "predictive",
        "Give me summary statistics and check for missing values": "eda",
        "What's the correlation between foot traffic and sales?": "eda",
        # Regression test: found by hand -- typing the literal word "eda" (the
        # analysis type's own name, e.g. "do eda") fell through to the LLM
        # fallback and could land on "general" with a confusing reply, because
        # the old EDA_PATTERNS list had no pattern for "eda" itself.
        "do eda": "eda",
        "run eda please": "eda",
    }
    for q, expected in cases.items():
        got = router.classify_intent(q, columns=["sales", "region", "ad_spend"])
        assert got == expected, f"{q!r} -> {got}, expected {expected}"
    print(f"[PASS] {len(cases)} keyword-routed questions all classified correctly, no API call")


def test_llm_fallback_used_for_ambiguous_question():
    with patch("router.call_llm", return_value="predictive") as m:
        got = router.classify_intent("what should I expect next quarter", columns=["sales"])
        assert got == "predictive", got
        assert m.called, "LLM fallback should have been called for an ambiguous question"
    print("[PASS] ambiguous question (no keyword match) falls back to LLM classification")


def test_llm_fallback_failure_defaults_to_general():
    with patch("router.call_llm", side_effect=Exception("network down")):
        got = router.classify_intent("totally unrelated ambiguous question", columns=["sales"])
        assert got == "general", got
    print("[PASS] LLM fallback failure degrades gracefully to 'general' instead of crashing")


if __name__ == "__main__":
    test_keyword_paths()
    test_llm_fallback_used_for_ambiguous_question()
    test_llm_fallback_failure_defaults_to_general()
    print("\nAll router.py tests passed.")
