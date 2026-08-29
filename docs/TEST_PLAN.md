# Plan: fully test Lens locally, before submission

**Who this is for:** a Claude Code session running locally on Saloni's laptop, inside
the `automated-data-scientist/` (a.k.a. Lens) folder. Follow the phases in order.
Each phase has a goal, exact commands, and a checkpoint -- confirm the checkpoint
before moving to the next phase. If something fails, stop and report the exact error
instead of guessing a fix and continuing.

**Why this plan exists, separate from `plan.md`:** `plan.md` turns this project into
a GitHub repo (git init, commits, optional push). This plan does something different
and more urgent -- it proves the app actually works end to end, with a real LLM
provider, on Saloni's own machine. Everything tested so far (30+ tests, the batch
eval, the 3-dataset generalization test) used mocked LLM responses because there was
no API key available in the build environment. That is the single biggest untested
risk before submission, and this plan closes it. Run this plan BEFORE `plan.md`, or
in either order -- they don't conflict, just don't skip this one.

---

## Phase 0 -- Pre-flight checks

Goal: confirm the environment is ready.

```bash
pwd                      # should end in .../automated-data-scientist (or lens/ if renamed)
python3 --version        # expect 3.13.5 (confirmed compatible) or 3.11.x
ls                        # app.py, pipeline.py, requirements.txt, .env-template should be here
```

**Checkpoint:** all three commands run without error and the folder looks right.

---

## Phase 1 -- Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Checkpoint:** `pip install` finishes with no red error lines. If a package fails to
build, report the exact package name and error -- do not silently skip it.

---

## Phase 2 -- Get a free API key and fill in `.env`

This step needs Saloni, not Claude Code -- an API key can't be generated
automatically. Claude Code should pause here and ask her to do this manually, then
confirm before continuing:

1. Go to https://console.groq.com , sign up (free, no credit card).
2. Create an API key (Groq dashboard -> API Keys -> Create).
3. In the project folder:
   ```bash
   cp .env-template .env
   ```
4. Open `.env` and set:
   ```
   MODEL_PROVIDER=groq
   GROQ_API_KEY=<paste the real key here>
   ```
   Leave `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` blank -- only one provider's key is
   needed. Groq's free tier is enough for this test (30 requests/minute, 1000/day).

**Checkpoint:** `.env` exists, has `MODEL_PROVIDER=groq` and a non-empty
`GROQ_API_KEY`, and `.env` is NOT staged in git (`git status` should not list it, or
git isn't initialized yet -- either is fine here).

---

## Phase 3 -- Run the full automated test suite (mocked, fast, free)

This confirms nothing is broken before spending real API calls on Phase 4-6.

```bash
python3 test_llm.py
python3 test_router.py
python3 test_pipeline.py
python3 test_safe_exec.py
python3 test_deterministic.py
python3 test_generalization.py
```

**Checkpoint:** all six print "All ... tests passed" / "ALL PASSED". If any fail,
stop -- do not proceed to live testing with a broken codebase.

---

## Phase 4 -- Run the app for real

```bash
streamlit run app.py
```

This opens the app in a browser tab (usually http://localhost:8501). Leave this
terminal running for the rest of this plan.

**Checkpoint:** the app loads, sidebar shows "LLM provider: groq" (or whatever was
set in `.env`), and no error banner is shown on load.

---

## Phase 5 -- Live test on the original sample dataset

In the running app:

1. In the sidebar, use the "Choose a dataset" dropdown and pick "Store sales
   campaign (has a known, verifiable causal effect)".
2. Click "Generate full EDA profile" -- should render instantly (zero-LLM, so this
   always works even before Phase 2's key is real -- if this fails, it's a code bug,
   not an API problem).
3. Expand "Quick baseline model," pick `sales` as target, click Train -- should show
   baseline vs. LinearRegression metrics instantly (also zero-LLM).
4. Ask in chat: `What was the effect of the campaign on sales?` -- this is the FIRST
   real LLM call. Should route to the causal handler and return an estimate with a
   stated identification assumption and limitation.
5. Ask: `Show me a bar chart of average sales by region` -- should route to
   visualization and render a real chart.
6. Ask: `What's the correlation between ad spend and foot traffic?` -- should route
   to EDA.
7. Ask a question with a deliberately wrong column name, e.g.
   `Show me sales by store_regoin` (typo) -- should either auto-fix and recover, or
   show a clean error message, NOT a raw Python traceback.

**Checkpoint:** all 6 interactions complete without a raw traceback on screen. Note
which provider/model actually answered (visible in the generated-code expander or
sidebar) and roughly how long each response took -- write this down to report back.

---

## Phase 6 -- Live test on a dataset the app has never seen (the real generalization proof)

Everything above uses the bundled sample dataset. This step is the one thing no
amount of mocked testing can substitute for: upload a genuinely different dataset and
ask real questions, with the real LLM writing real code against columns it has never
seen before.

1. In the sidebar, upload `sample_data/generalization_test/hr_attrition.csv`
   (or `supply_chain_late_delivery.csv` -- either is fine, pick one).
2. Click "Generate full EDA profile" -- zero-LLM, should work regardless.
3. Ask: `Build a model to predict attrition` (or `late_delivery_risk` for the supply
   chain file) -- this is the real test: the LLM has to look at THIS dataset's
   actual columns and write working code against them, live.
4. Ask: `Show me a bar chart of average monthly income by department` (or the
   supply-chain equivalent) -- confirms visualization also generalizes live.

**Checkpoint:** both real-LLM questions return a working result (a trained model with
metrics, a rendered chart) referencing the correct column names for THIS dataset, not
leftover column names from the sales_campaign sample. This is the strongest evidence
for the "any CSV, any domain" claim in the report -- if it works here, it's proven,
not just argued.

---

## Phase 7 -- Final sign-off

Report back with:

- Which provider/model was used (should be Groq / openai/gpt-oss-120b unless a
  different key was used).
- Phase 3 result: all 6 test files passed, yes/no.
- Phase 5 result: all 6 interactions worked cleanly, yes/no -- note anything that
  needed the auto-fix/error-recovery path.
- Phase 6 result: which new-domain dataset was used, and whether both live questions
  worked against its real columns.
- Any error message seen anywhere, verbatim, even if the app recovered from it.

If everything above passes, the "no live-provider test" limitation currently listed
in the report (Section 10) is closed -- this is worth telling Saloni explicitly, since
it changes the report from "here's what should work" to "here's what was verified to
work."
