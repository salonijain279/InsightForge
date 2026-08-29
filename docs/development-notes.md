# InsightForge — Development Notes

This document records the system design, the failures found during testing, and the
tradeoffs that remain. It is intentionally more detailed than the project README so the
reasoning behind the implementation is auditable.

---

## 1. What the system does

Upload a dataset (or pick a sample) and ask a question in plain English. The app figures
out what kind of question it is — exploratory, predictive, causal, or a chart request —
and answers it: it writes and runs real Python code against the actual data, then explains
the real result. Two things run with zero LLM calls, for a fast $0 baseline: a full EDA
profile and a baseline model.

## 2. Architecture

| File | What it does |
|---|---|
| `app.py` | The Streamlit interface — file upload, dataset preview, chat window, buttons. Calls into every file below; contains no analysis logic itself. |
| `router.py` | Reads the question and decides which kind of analysis it needs (EDA / predictive / causal / chart / general chit-chat). Keyword match first (free, instant); only asks the LLM to classify if the question is genuinely ambiguous. |
| `handlers/` | One file per analysis type. Each just holds the instructions (prompt) telling the LLM how to write correct code for that type of question — e.g. the predictive handler's prompt requires a train/test split and a baseline comparison; the causal handler's prompt requires stating the assumption behind the method used. |
| `llm.py` | The one place that actually talks to an LLM provider. Supports OpenAI, Anthropic, and Groq behind a single switch (`MODEL_PROVIDER` in `.env`) — this project runs on Groq (free), the other two are wired in and tested so switching providers later needs no code changes. |
| `safe_exec.py` | Checks the LLM's generated code before running it, and blocks anything dangerous (file access, internet access, imports outside a small safe list) so the app can't accidentally do something harmful. |
| `exec_namespace.py` | The fixed set of tools (pandas, numpy, plotly, sklearn, statsmodels, the dataset) the generated code is allowed to use — nothing outside this list is reachable. |
| `recording_ui.py` | A technical workaround (explained below) so chart/table output isn't lost when the chat screen refreshes. |
| `pipeline.py` | Ties it together: route the question → get code from the LLM → run it safely → write up what actually happened. |
| `trust_card.py` | Scans the code that just ran and reports, in plain facts, what method it used and how it handled the data — built to never guess, only report what it can see in the code. |
| `dataset_info.py` | Summarizes whatever dataset is loaded (columns, types, missing values) so every LLM prompt is grounded in the real data, and builds the "Try asking" suggestion chips. |
| `deterministic.py` / `report.py` | The zero-LLM EDA profile and baseline model, plus the downloadable session report. |

## 3. How a question actually gets answered

1. You type a question. `router.py` decides its type.
2. The matching handler in `handlers/` sends the LLM a prompt containing the real dataset's
   columns and the rules for that analysis type.
3. The LLM returns Python code plus a first-draft explanation.
4. `safe_exec.py` checks the code is safe, then runs it for real against your data.
5. A **second** LLM call writes the explanation you actually see — this one is only shown
   the real output from step 4, so it can't describe a number that didn't really happen.
6. `trust_card.py` scans the code that ran and reports what it actually did.
7. If the code fails, one automatic retry sends the error back to the LLM to fix.

**Why step 5 exists:** the first draft explanation (step 3) is written *before* the code
runs, so it's really a guess at what the output will look like. Writing the real
explanation only after the code has actually run means every number shown to you came from
a real computation, not a prediction.

**What `recording_ui.py` is for:** Streamlit normally draws a chart or table the instant
your code calls it. But this app refreshes the whole chat screen after every answer (so old
messages stay visible), and that refresh throws away anything drawn during the previous run.
The fix: the generated code doesn't get the real Streamlit — it gets a stand-in that just
*records* what it was asked to show (a chart, a table, a number), and the app replays that
recording with the real Streamlit every time the screen redraws. Nothing gets lost.

## 4. Bugs found and fixed

Every one of these was found by actually running the app and clicking through it — not by
reading the code and guessing. Each has a test that checks it can't silently come back.

| # | What broke | Fix |
|---|---|---|
| 1 | Generated code ran with full Python access (could technically read files, hit the internet) | Added `safe_exec.py`: scans code before running it, blocks anything dangerous |
| 2 | Predictive/causal answers could describe numbers that never actually ran | Added a second LLM call that only writes the explanation after seeing the real output |
| 3 | A text-labeled prediction target (e.g. "malignant"/"benign") crashed the baseline model | Fixed the type check that only worked for old-style pandas string columns |
| 4 | Same text-label case: the LLM tried to use `LabelEncoder` and got blocked by the safety scan | Added `LabelEncoder` to the allowed toolset |
| 5 | Groq's model (`llama-3.3-70b-versatile`) was shut down by Groq on Aug 16 | Switched default model to `openai/gpt-oss-120b` |
| 6 | Every answer showed a confusing double error message, even successful ones | Fixed a Python exception-chaining bug in the code timeout logic |
| 7 | Typing "do eda" gave a confusing generic reply instead of running EDA | Added "eda" as its own keyword to the router |
| 8 | The suggested-question chips could misread a `sales`-style numeric column as an ID column | Fixed the ID-detection rule to require whole numbers, not just "every value is unique" |
| 9 | Predictive questions could leak the target column or skip a baseline comparison | Rewrote the predictive prompt: no target imputation, exclude ID columns, always show a baseline vs. the real model |
| 10 | Causal (DiD) questions could produce a statistically broken formula | Rewrote the causal prompt to use the correct formula and cluster standard errors correctly |
| 11 | Switching datasets left the old dataset's chat answers on screen | Added a proper reset that clears results when the dataset changes |
| 12 | The new "likely identifier column" data-quality check reflagged `sales` as an ID column — the same bug as #8, reintroduced by a second, less careful implementation | Fixed by excluding continuous (float) columns from that check, same fix as #8 |
| 13 | The DEMO_MODE screenshot fixture showed a broken-looking generic message instead of a real answer | Updated the fixture to echo real computed values |
| 14 | **Live bug on the real app:** predictive questions crashed with `NameError: name 'DummyRegressor' is not defined`, even though it was provided | `safe_exec.py` was running code with two separate variable scopes (a Python gotcha that breaks any function or list/dict comprehension referencing the provided tools); fixed to use one shared scope, matching how real Python code actually runs |
| 15 | The reset button only cleared the chat, not the loaded dataset or its sidebar description | Reset now clears the dataset too and returns to the empty starting screen |

## 5. Known limitations, stated plainly

- **Safety is a blocklist, not a sandbox.** `safe_exec.py` blocks known-dangerous patterns
  before running code; it does not run code in an isolated process or container. Reasonable
  for a single-user local tool, not something to expose to the public internet as-is.
- **The signal-based timeout is not enforceable inside Streamlit's worker thread.** It works
  in direct script execution; a production version needs process or container isolation for
  a hard execution limit.
- **Only Groq has been live-tested.** OpenAI and Anthropic are implemented identically and
  covered by mocked tests, but haven't been run against a real key in this project.
- **The zero-LLM deterministic path covers EDA and one baseline-model shape only** — it's a
  fast, free complement to the chat path, not a full replacement for it.
- **`requirements.txt` uses minimum-version pins, not an exact lockfile.**

## 6. Test coverage

The automated suite is collected with `pytest` and covers routing, provider wrappers,
pipeline execution, deterministic modelling, safety checks, suggested questions, trust
cards, report generation, cross-domain generalization, and causal-method recovery.

```bash
pip install -r requirements-dev.txt
pytest -q
python eval_suite.py
```

The separate batch evaluation runs ten representative questions end to end through the
pipeline. The generalization tests exercise three datasets from different domains to make
sure the logic is not tuned only to the bundled campaign sample.

The causal-inference code patterns (DiD, IV, RDD, PSM) were separately validated against a
synthetic dataset with a known, built-in true effect before being trusted in any prompt —
each method recovers the true effect within a reasonable margin.

## 7. What automated well, and what didn't

Mechanical execution — running the right kind of analysis on well-specified questions, recovering from a broken column
name automatically, refusing to run unsafe code — automated cleanly and reliably. What
didn't automate away: judging *whether* a causal claim is defensible for a given dataset,
translating a statistical result into a business decision, and — found directly through this
project — noticing when the system itself was subtly wrong (the ID-column bug reappeared
because a second implementation didn't know about the first fix; a human had to catch it
by actually looking at the result, not by trusting that "tests passed" meant it was correct).
