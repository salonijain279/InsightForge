# Plan: turn this project into a proper GitHub repo

**Project name: Lens** (an automated data scientist). Use this name in README.md's
title, the GitHub repo name (Phase 5), and anywhere else a project name is needed --
not "Automated Data Scientist" or "automated-data-scientist" (that's just the local
folder name from development; the actual project is called Lens).

**Who this is for:** a Claude Code session running locally on Saloni's laptop,
inside the `automated-data-scientist/` folder (feel free to rename this folder to
`lens/` in Phase 1 if you want the local folder name to match -- not required, just
tidier). Follow the phases below in order.
Each phase has a goal, exact commands, and a checkpoint to confirm before moving on.
Do not skip the checkpoints -- if one fails, stop and report what happened instead
of continuing to the next phase.

**Ground rule:** do not fabricate history. Every commit made here happens "now," in
one sitting -- do not backdate commit timestamps or write commit messages that imply
they happened over multiple days when they didn't. Grouping related files into
separate, well-labeled commits is honest and normal practice for publishing a
finished project; pretending to a multi-day timeline is not, and if a professor
checks `git log`, it should hold up.

---

## Phase 0 -- Pre-flight checks

Goal: confirm the environment is actually ready before touching git.

```bash
pwd                          # should end in .../automated-data-scientist
git --version                # confirm git is installed
python3 --version            # note the version
ls                            # sanity check: app.py, pipeline.py, requirements.txt etc. should be here
```

Run the full test suite and confirm all pass before committing anything -- a repo's
first commit should be a working state, not a broken one:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 test_llm.py && python3 test_router.py && python3 test_pipeline.py && python3 test_safe_exec.py && python3 test_deterministic.py
```

**Checkpoint:** all 5 test files print "All ... tests passed." If anything fails,
stop and report the failure -- do not proceed to committing broken code.

---

## Phase 1 -- Initialize the repo

```bash
git init
git branch -M main
```

Create `.gitignore` (if it doesn't already exist) with at least:

```
venv/
__pycache__/
*.pyc
.env
.DS_Store
*.egg-info/
```

`.env` must be gitignored -- it holds a real API key once Saloni fills it in.
`.env-template` (the blank version) is what gets committed.

**Checkpoint:** `git status` shows `.env` is NOT listed (either it doesn't exist yet,
or it's correctly ignored) and `venv/` is not listed as trackable.

---

## Phase 2 -- Write a GitHub-facing README.md

`PROJECT_LOG.md` already documents the build in detail (architecture, ideas borrowed,
testing, security fix, known limitations) -- that file stays as-is, it's the full
writeup. `README.md` is the shorter, GitHub-front-page version: what this is, how to
run it, and a pointer to `PROJECT_LOG.md` for the full story. Title it `# Lens`
with a one-line subtitle ("an automated data scientist"). Include:

- One-paragraph project description (Path 4, MSBA 6461).
- Quick-start: clone, venv, install, copy `.env-template` to `.env` and fill in a
  key, `streamlit run app.py`.
- A short architecture list (mirror PROJECT_LOG.md Section 3's module list, condensed
  to one line each).
- "Run the tests" section with the exact 5 commands from Phase 0.
- A "Known limitations" section -- pull the bullet list from PROJECT_LOG.md Section 8
  (no live-provider test run yet, deterministic coverage is partial, safety gate is a
  blocklist not a sandbox, etc.). Do not omit this section -- it's what makes the repo
  credible rather than oversold.
- Credit table for borrowed ideas (Auto-Analyst, ydata-profiling, LIDA, PandasAI) --
  copy from PROJECT_LOG.md Section 1.

**Checkpoint:** README.md exists, is under ~150 lines, and someone who has never seen
this project could clone the repo and get it running from README.md alone.

---

## Phase 3 -- Structured commits (not one giant commit)

Goal: the commit history should read as a system, not a data dump. Stage and commit
in this grouped order -- each command is a separate commit:

```bash
git add .gitignore README.md
git commit -m "Initial project setup: gitignore, README"

git add llm.py .env-template
git commit -m "Add provider-agnostic LLM wrapper (OpenAI/Anthropic/Groq)"

git add router.py dataset_info.py
git commit -m "Add keyword-first intent router and dataset summarizer"

git add handlers/
git commit -m "Add per-analysis-type prompt handlers (EDA/predictive/causal/viz/general)"

git add exec_namespace.py recording_ui.py pipeline.py
git commit -m "Add execution pipeline: namespace, Streamlit-display recording, routing/exec/auto-fix"

git add safe_exec.py
git commit -m "Add execution safety gate: AST blocklist + restricted builtins + timeout"

git add deterministic.py
git commit -m "Add deterministic (zero-LLM) baseline model and EDA profile"

git add report.py app.py
git commit -m "Add Streamlit UI and downloadable report generation"

git add sample_data/
git commit -m "Add synthetic sample dataset with known causal effect (for validation)"

git add test_llm.py test_router.py test_pipeline.py test_safe_exec.py test_deterministic.py eval_suite.py eval_results.txt
git commit -m "Add test suite (30 tests) and 10-question batch evaluation"

git add requirements.txt PROJECT_LOG.md plan.md
git commit -m "Add dependencies, project log, and this plan"
```

Adjust file names to whatever actually exists in the folder at commit time --
run `git status` before each `git add` to confirm you're staging what you expect.

**Checkpoint:** `git log --oneline` shows ~11 commits, each with a clear scope. No
single commit should be "everything."

---

## Phase 4 -- Optional: basic CI so the tests run on every push

Create `.github/workflows/tests.yml`:

```yaml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: python3 test_llm.py
      - run: python3 test_router.py
      - run: python3 test_pipeline.py
      - run: python3 test_safe_exec.py
      - run: python3 test_deterministic.py
```

```bash
git add .github/workflows/tests.yml
git commit -m "Add CI: run full test suite on push"
```

This directly answers the "reproducibility" criticism from the review -- a green
checkmark on GitHub is stronger evidence than a claim in a report. Skip this phase
if there's no time left; it's a nice-to-have, not a requirement.

**Checkpoint:** workflow file is valid YAML (no tabs, correct indentation).

---

## Phase 5 -- Push to GitHub (only if Saloni wants a remote copy)

This requires a GitHub repo to already exist (create it empty, no README/license/
gitignore, at github.com/new) or the `gh` CLI:

```bash
# Option A: gh CLI (if installed and authenticated)
gh repo create lens --private --source=. --remote=origin --push

# Option B: manual, after creating an empty repo on github.com
git remote add origin <the-repo-URL-from-github>
git push -u origin main
```

Ask Saloni whether the repo should be public or private before running either
command -- default to **private** unless she says otherwise, since this is a course
submission and public-by-default risks another student finding it before the
deadline.

**Checkpoint:** repo is visible at the GitHub URL, all files present, README renders.

---

## Phase 6 -- Final verification

Simulate a stranger cloning the repo fresh, to catch anything that only worked
because of local state:

```bash
cd /tmp
git clone <repo-path-or-URL> verify-clone
cd verify-clone
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env-template .env   # still needs a real key filled in manually to fully run
python3 test_llm.py && python3 test_router.py && python3 test_pipeline.py && python3 test_safe_exec.py && python3 test_deterministic.py
```

**Checkpoint:** all 5 test files pass from the fresh clone, with nothing missing.
Report back to Saloni: commit count, whether CI was set up, whether it was pushed to
GitHub (and the URL if so), and confirm the fresh-clone test run passed.
