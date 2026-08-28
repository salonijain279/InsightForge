# InsightForge

InsightForge is a locally hosted, evidence-grounded automated data scientist for
tabular data. A user uploads a CSV or Excel file, asks a question in natural
language, and receives exploratory, predictive, visualization, or guarded causal
analysis produced from real Python execution.

## Assignment fit

- Proper Streamlit front end with file upload, sample datasets, chat history,
  suggested questions, interactive charts, and report download.
- Functional analytics back end with intent routing, dataset-aware prompts,
  deterministic EDA and baseline modelling, guarded generated-code execution,
  error recovery, and evidence-grounded explanations.
- Exploratory and predictive analysis are core capabilities; causal analysis is
  an explicitly guarded extension.
- The reflection in the project report distinguishes repeatable tasks automated by
  the app from judgment that remains the responsibility of a human data scientist.

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env-template .env
streamlit run app.py
```

Configure one provider in `.env`:

```text
MODEL_PROVIDER=groq
GROQ_API_KEY=your_key_here
```

Open `http://localhost:8501`. Deterministic EDA and the baseline model work
without an API call. Chat-driven custom analysis requires a configured provider.

## Recommended demonstration

1. Select the store-sales campaign sample or upload a CSV/Excel file.
2. Run **Full EDA profile**.
3. Select `sales` and train the baseline plus simple model.
4. Ask for a chart or a predictive analysis in chat.
5. Ask about the campaign effect and inspect the generated code and trust card.
6. Download the Markdown session report.

## Verification

Run the deterministic tests individually:

```bash
python test_router.py
python test_deterministic.py
python test_generalization.py
python test_pipeline.py
python test_safe_exec.py
python test_causal_methods.py
python eval_suite.py
```

The evaluation suite uses mocked LLM responses so routing, execution, rendering,
retry, and safe-failure behavior are reproducible. A live-provider smoke test is
still recommended immediately before presenting.

## Important limitations

- Generated code can be safe to execute yet use an inappropriate analytical method.
- The AST scan and restricted builtins are a local mitigation, not a security sandbox.
- The signal-based 20-second timeout works in direct script execution but cannot be
  enforced inside Streamlit's worker thread; production use needs process isolation.
- Causal estimates are valid only under defensible identification assumptions.
- The downloadable report captures textual outputs; interactive chart state remains
  in the running app.

## Development transparency

Claude Code was used as an implementation assistant, as permitted by the assignment.
The system-architecture decisions, evaluation design, validation criteria, and final
claims remain the author's responsibility. The source comments identify external
design ideas considered during development.
