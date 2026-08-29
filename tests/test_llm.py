import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""Tests llm.py against mocked SDK responses -- no real API key or network call needed."""
import os
from unittest.mock import MagicMock, patch

import llm


def test_missing_key_raises_clear_error():
    os.environ["MODEL_PROVIDER"] = "openai"
    os.environ.pop("OPENAI_API_KEY", None)
    try:
        llm.call_llm("hi")
        raise AssertionError("should have raised LLMError")
    except llm.LLMError:
        print("[PASS] missing OPENAI_API_KEY raises LLMError, not a stack trace")


def test_unknown_provider_raises():
    os.environ["MODEL_PROVIDER"] = "bogus"
    try:
        llm.call_llm("hi")
        raise AssertionError("should have raised LLMError")
    except llm.LLMError:
        print("[PASS] unknown provider raises LLMError")


def test_openai_call_shape():
    os.environ["MODEL_PROVIDER"] = "openai"
    os.environ["OPENAI_API_KEY"] = "sk-test"
    fake_choice = MagicMock()
    fake_choice.message.content = "hello from mocked openai"
    fake_resp = MagicMock(choices=[fake_choice])
    with patch("openai.resources.chat.completions.Completions.create", return_value=fake_resp) as m:
        out = llm.call_llm("say hi", system="be terse")
        assert out == "hello from mocked openai", out
        kwargs = m.call_args.kwargs
        assert kwargs["messages"][0] == {"role": "system", "content": "be terse"}
        assert kwargs["messages"][1] == {"role": "user", "content": "say hi"}
    print("[PASS] openai call sends correct message shape and parses response")


def test_anthropic_call_shape():
    os.environ["MODEL_PROVIDER"] = "anthropic"
    os.environ["ANTHROPIC_API_KEY"] = "ant-test"
    fake_block = MagicMock(type="text", text="hello from mocked claude")
    fake_resp = MagicMock(content=[fake_block])
    with patch("anthropic.resources.messages.Messages.create", return_value=fake_resp) as m:
        out = llm.call_llm("say hi", system="be terse")
        assert out == "hello from mocked claude", out
        kwargs = m.call_args.kwargs
        assert kwargs["system"] == "be terse"
        assert kwargs["messages"] == [{"role": "user", "content": "say hi"}]
    print("[PASS] anthropic call sends correct message shape and parses response")


def test_demo_mode_bypasses_provider_entirely():
    os.environ["DEMO_MODE"] = "1"
    os.environ.pop("OPENAI_API_KEY", None)  # would raise LLMError if DEMO_MODE didn't short-circuit
    try:
        out = llm.call_llm("what was the effect?", system="You are a causal-inference assistant.")
        assert "COMMENTARY" in out and "CODE" in out, out
    finally:
        os.environ.pop("DEMO_MODE", None)
    print("[PASS] DEMO_MODE=1 returns a fixture response without touching any provider or requiring a key")


def test_demo_mode_grounding_call_echoes_real_facts_not_generic_fallback():
    # Regression test: found via a live DEMO_MODE screenshot check, not by reading
    # code. generate_grounded_commentary()'s system prompt doesn't match any of the
    # pre-execution branches (causal-inference/predictive-modeling/etc.), so it used
    # to silently fall through to the generic "I can help with EDA..." non-answer.
    os.environ["DEMO_MODE"] = "1"
    os.environ.pop("OPENAI_API_KEY", None)
    try:
        system = ("You explain the ACTUAL results of a data analysis that has already run. "
                   "You will be given the real computed values below -- use ONLY those.")
        user = (
            "Original question: Build a model to predict sales\nAnalysis type: predictive\n\n"
            "Dataset info:\n(irrelevant for this test)\n\n"
            "Actual computed results (this is everything the code produced -- nothing "
            "else is available):\nR2: 0.676\nMAE: 2102.24\n\n"
            "Write the explanation now, grounded only in the results above."
        )
        out = llm.call_llm(user, system=system)
        assert "R2: 0.676" in out, out
        assert "I can help with EDA" not in out, out
    finally:
        os.environ.pop("DEMO_MODE", None)
    print("[PASS] DEMO_MODE grounding call echoes the real computed facts instead of a generic fallback")


def test_extract_code():
    fenced = "Here you go:\n```python\nprint('hi')\n```\nDone."
    assert llm.extract_code(fenced) == "print('hi')"
    unfenced = "print('hi')"
    assert llm.extract_code(unfenced) == "print('hi')"
    print("[PASS] extract_code handles fenced and unfenced responses")


if __name__ == "__main__":
    test_missing_key_raises_clear_error()
    test_unknown_provider_raises()
    test_openai_call_shape()
    test_anthropic_call_shape()
    test_demo_mode_bypasses_provider_entirely()
    test_demo_mode_grounding_call_echoes_real_facts_not_generic_fallback()
    test_extract_code()
    print("\nAll llm.py tests passed.")
