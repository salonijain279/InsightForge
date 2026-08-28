"""
pipeline.py -- the actual "answer a question about this dataframe" logic.

Kept separate from app.py (the Streamlit UI) so it can be integration-tested without
needing a Streamlit runtime -- see test_pipeline.py, which runs this against the real
sample dataset with a MOCKED LLM response shaped like what a real model would return,
checking the whole path (route -> prompt -> parse -> execute against real data) works,
not just each piece in isolation.
"""

import sys
import signal
import traceback
import contextlib
from io import StringIO

import pandas as pd

import router
from handlers import HANDLERS
from llm import call_llm, parse_commentary_and_code, extract_code, LLMError
from dataset_info import summarize_dataset
from exec_namespace import build_exec_namespace
from safe_exec import safe_exec, UnsafeCodeError

CODE_TIMEOUT_SECONDS = 20


@contextlib.contextmanager
def capture_stdout():
    old = sys.stdout
    buf = StringIO()
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = old


@contextlib.contextmanager
def _wall_clock_timeout(seconds):
    """Best-effort wall-clock timeout for generated code (e.g. an accidental
    infinite loop). Uses signal.alarm, which only works in the main thread of the
    main interpreter -- and Streamlit runs the app script in its OWN script-runner
    thread, not the main thread, so under a real `streamlit run` this always
    degrades to a no-op (confirmed via live testing, not assumed). It still works
    when pipeline.run_code is called directly from a plain script (e.g. the test
    suite), which is where the actual timeout behavior gets exercised.

    IMPORTANT: `yield` must never sit inside an `except` block here -- doing so
    made Python chain any later exception from the `with` body onto this one
    (the confusing "during handling of the above exception..." traceback), even
    though the ValueError itself was already handled and harmless."""
    has_alarm = hasattr(signal, "SIGALRM")
    old_handler = None
    alarm_armed = False

    if has_alarm:
        def _handler(signum, frame):
            raise TimeoutError(f"Generated code exceeded the {seconds}s execution limit.")
        try:
            old_handler = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(seconds)
            alarm_armed = True
        except ValueError:
            # Not the main thread (e.g. Streamlit's script-runner thread) --
            # alarm() isn't usable here. Degrade gracefully to a no-op instead
            # of crashing the app over a missing safety net.
            alarm_armed = False

    try:
        yield
    finally:
        if alarm_armed:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def run_code(code: str, local_vars: dict):
    """Returns (success, output_text, error_text_or_None). Runs the safety scan
    first (see safe_exec.py) -- unsafe code never reaches exec() at all and is
    reported back as a clean error, same as any other execution failure."""
    with capture_stdout() as out:
        try:
            with _wall_clock_timeout(CODE_TIMEOUT_SECONDS):
                safe_exec(code, local_vars)
            return True, out.getvalue(), None
        except UnsafeCodeError as e:
            return False, out.getvalue(), f"Blocked before execution (safety scan): {e}"
        except Exception:
            return False, out.getvalue(), traceback.format_exc()


def _format_items_for_grounding(items: list, output: str, max_chars: int = 3000) -> str:
    """Turns the RecordingUI's captured display items (see recording_ui.py) -- the
    ACTUAL values the executed code produced -- into plain text a second LLM call
    can read. This is the ground truth for generate_grounded_commentary(): every
    number in it genuinely came out of a real computation, not a guess."""
    lines = []
    for kind, payload in items:
        if kind in ("write", "text", "success", "warning", "error"):
            lines.append(str(payload))
        elif kind == "metric":
            label, value = payload
            lines.append(f"{label}: {value}")
        elif kind in ("dataframe", "table"):
            try:
                lines.append(payload.head(15).to_string())
            except Exception:
                lines.append(str(payload)[:1000])
        elif kind == "plotly_chart":
            title = None
            try:
                title = payload.layout.title.text
            except Exception:
                pass
            lines.append(f"[a chart was generated{f': ' + title if title else ''}]")
        elif kind == "pyplot":
            lines.append("[a matplotlib figure was generated]")
        elif kind == "json":
            lines.append(str(payload)[:1000])
        # "code" items are the code itself, not a result -- not useful as grounding text.
    if output.strip():
        lines.append(f"(stdout output): {output.strip()}")
    text = "\n".join(lines).strip()
    return text[:max_chars] if text else "(the code ran successfully but displayed no output)"


def generate_grounded_commentary(question: str, kind: str, items: list, output: str, dataset_summary: str, fallback: str) -> str:
    """Generates the explanation shown to the user from the ACTUAL computed results,
    not before them. Fixes a real gap: the original commentary is written by the
    same LLM call that writes the code, BEFORE that code has run -- so it could only
    describe what the code was ABOUT to compute, sometimes literally as a template
    ("replace X, SE, CI-low...") rather than the real numbers. This second call runs
    AFTER execution and is given nothing but the real results to describe.
    Falls back to the original (pre-execution) commentary if this call fails, so a
    network hiccup here degrades gracefully instead of losing the answer entirely."""
    grounded_facts = _format_items_for_grounding(items, output)
    system = (
        "You explain the ACTUAL results of a data analysis that has already run. "
        "You will be given the real computed values below -- use ONLY those. Never "
        "invent, guess, or restate a placeholder for a number that isn't given to "
        "you. If something needed to answer the question well isn't in the results "
        "(e.g. no confidence interval was printed), say so explicitly instead of "
        "making one up. Be concise (3-6 sentences), plain language, no code. If the "
        "analysis is causal, still state the key identifying assumption and a "
        "limitation even though you're describing results, not writing new code."
    )
    user = (
        f"Original question: {question}\nAnalysis type: {kind}\n\n"
        f"Dataset info:\n{dataset_summary}\n\n"
        f"Actual computed results (this is everything the code produced -- nothing "
        f"else is available):\n{grounded_facts}\n\n"
        "Write the explanation now, grounded only in the results above."
    )
    try:
        return call_llm(user, system=system, max_tokens=500, temperature=0.0).strip()
    except Exception:
        return fallback


def attempt_fix(code: str, error: str, dataset_summary: str, system: str = None):
    """Asks the LLM to fix code that raised an error. IMPORTANT: `system` must be the
    SAME handler system prompt used to generate the original code (predictive.py's or
    causal.py's methodology + display rules), not omitted. Found live: without it, a
    fix pass has zero knowledge of the rules the original prompt spent a lot of care
    on -- baseline comparison, correct SE clustering, "don't dump the full
    model.summary() table" -- so a fix that happens to also change the model's formula
    can silently reintroduce exactly the mistakes those rules existed to prevent,
    while still "successfully" running. Passing system through keeps the fix pass
    honoring the same rules the original generation had to follow."""
    prompt = (
        f"This Python code raised an error when executed against a DataFrame `df`.\n\n"
        f"Dataset info:\n{dataset_summary}\n\n"
        f"Code:\n```python\n{code}\n```\n\n"
        f"Error:\n{error[-2000:]}\n\n"
        "Fix ONLY what's broken. Keep following every rule from your system instructions "
        "-- do not drop the baseline comparison, the display format, the SE/clustering "
        "choice, or any other requirement just because you're only fixing an error. "
        "Return ONLY the corrected code in a single ```python code block, no explanation."
    )
    try:
        reply = call_llm(prompt, system=system, max_tokens=2000, temperature=0.0)
        return extract_code(reply)
    except Exception:
        return None


def process_question(question: str, df: pd.DataFrame) -> dict:
    """Routes the question, calls the right handler's prompt, calls the LLM, executes
    the generated code (with one auto-fix retry on failure), and returns a result dict:
    {question, kind, commentary?, code?, output?, error?, auto_fixed?}
    """
    dataset_summary = summarize_dataset(df)
    kind = router.classify_intent(question, df.columns)
    result = {"question": question, "kind": kind}

    if kind == "general":
        system, user = HANDLERS["general"].build_prompt(question, dataset_summary)
        try:
            result["commentary"] = call_llm(user, system=system)
        except LLMError as e:
            result["error"] = str(e)
        except Exception as e:
            # Catches provider-SDK-level failures LLMError doesn't cover: an invalid/
            # expired key, rate limiting, a model name that doesn't exist, a network
            # timeout. Without this, any of those would crash the whole Streamlit
            # script run instead of showing a clean, in-chat error message.
            result["error"] = f"LLM call failed ({type(e).__name__}): {e}"
        return result

    system, user = HANDLERS[kind].build_prompt(question, dataset_summary)
    try:
        response = call_llm(user, system=system)
    except LLMError as e:
        result["error"] = str(e)
        return result
    except Exception as e:
        result["error"] = f"LLM call failed ({type(e).__name__}): {e}"
        return result

    commentary, code = parse_commentary_and_code(response)
    result["commentary"] = commentary  # pre-execution draft -- overwritten below on
                                        # success, kept only as a fallback if grounding fails
    result["code"] = code

    ns = build_exec_namespace(df)
    success, output, error = run_code(code, ns)

    if not success:
        fixed_code = attempt_fix(code, error, dataset_summary, system=system)
        if fixed_code:
            ns2 = build_exec_namespace(df)
            success2, output2, error2 = run_code(fixed_code, ns2)
            if success2:
                result["code"] = fixed_code
                result["auto_fixed"] = True
                result["output"] = output2
                result["display_items"] = ns2["st"].items
                result["commentary"] = generate_grounded_commentary(
                    question, kind, ns2["st"].items, output2, dataset_summary, fallback=commentary
                )
                return result
        result["error"] = error
        return result

    result["output"] = output
    result["display_items"] = ns["st"].items
    # Replace the pre-execution draft with an explanation grounded in what the code
    # ACTUALLY produced -- see generate_grounded_commentary()'s docstring for why
    # the original commentary alone isn't trustworthy (it's written before the
    # numbers exist).
    result["commentary"] = generate_grounded_commentary(
        question, kind, ns["st"].items, output, dataset_summary, fallback=commentary
    )
    return result
