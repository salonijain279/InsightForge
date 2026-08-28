"""
llm.py -- a single, plain function to call whichever LLM provider is configured.

No agent framework here on purpose: one function, one job (send a prompt, get text
back), so anyone reading this project can see exactly what's happening on every call
instead of tracing through a framework's internals.

MODEL_PROVIDER in .env picks the provider: openai (default) | anthropic | groq
"""

import os


class LLMError(RuntimeError):
    """Raised for missing/misconfigured provider setup -- caught in app.py and shown
    to the user as a clear message instead of a raw stack trace."""


def call_llm(prompt: str, system: str = None, max_tokens: int = 2000, temperature: float = 0.0) -> str:
    if os.environ.get("DEMO_MODE") == "1":
        # Opt-in only, off by default, and never touched by a real run: lets the app be
        # clicked through end to end (for a screenshot or a demo) without spending an API
        # call or requiring a key. See _demo_fake_response() and PROJECT_LOG.md.
        return _demo_fake_response(system or "", prompt or "")

    provider = os.environ.get("MODEL_PROVIDER", "openai").strip().lower()

    if provider == "openai":
        return _call_openai(prompt, system, max_tokens, temperature)
    if provider == "anthropic":
        return _call_anthropic(prompt, system, max_tokens, temperature)
    if provider == "groq":
        return _call_groq(prompt, system, max_tokens, temperature)
    raise LLMError(f"Unknown MODEL_PROVIDER '{provider}'. Use openai, anthropic, or groq.")


def _demo_fake_response(system: str, user: str = "") -> str:
    """Canned, realistic-shaped responses used ONLY when DEMO_MODE=1 is set (never
    the case in a normal run -- see call_llm above). Same fixtures used by
    eval_suite.py, kept here so a screenshot/demo run and the eval suite agree.

    generate_grounded_commentary() (pipeline.py) makes a SECOND call per question,
    after the code has already run, with a distinct system prompt that doesn't match
    any branch below -- without this check it fell through to the generic "I can
    help with..." message, which looked like a broken/non-answer in a DEMO_MODE
    screenshot even though the real (non-demo) path works correctly (verified against
    live Groq calls). Rather than inventing numbers, this echoes back the real
    computed facts the grounding prompt was given (see _format_items_for_grounding
    in pipeline.py) -- genuinely grounded, same as the real call, just without an
    LLM doing the sentence-writing."""
    if "actual results of a data analysis" in system.lower():
        marker = "Actual computed results"
        if marker in user:
            facts = user.split(marker, 1)[1].split("Write the explanation now", 1)[0]
            facts = facts.lstrip(":\n ").strip()
            return f"Based on the actual results from this run: {facts[:400]}"
        return "Based on the actual results from this run (see the output above)."
    if "causal-inference" in system:
        return ("COMMENTARY:\n1. Identification: Difference-in-Differences -- treated marks "
                "campaign stores, post marks the post-campaign period.\n"
                "2. Assumption: parallel trends -- absent the campaign, treated and control "
                "stores' sales would have moved together.\n"
                "3. Effect size: see the coefficient, SE, and p-value printed below.\n"
                "4. Limitation: 40 stores over 12 months; unobserved store-level shocks "
                "coinciding with the campaign could bias this. An estimate under the stated "
                "assumption, not a proven fact.\n\n"
                "CODE:\n```python\nmodel = smf.ols('sales ~ treated * post', data=df).fit(cov_type='HC1')\n"
                "st.write(f\"DiD estimate: {model.params['treated:post']:.2f}\")\n"
                "st.write(f\"p-value: {model.pvalues['treated:post']:.5f}\")\n```")
    if "predictive-modeling" in system:
        return ("COMMENTARY:\nTrains a linear regression on ad spend, foot traffic, and store "
                "size to predict sales, with an 80/20 train/test split.\n\n"
                "CODE:\n```python\nfeatures=['ad_spend','foot_traffic','size_sqft']\n"
                "data=df.dropna(subset=features+['sales'])\nX=data[features]\ny=data['sales']\n"
                "X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)\n"
                "m=LinearRegression().fit(X_train,y_train)\npreds=m.predict(X_test)\n"
                "st.write(f\"R2: {metrics.r2_score(y_test, preds):.3f}\")\n"
                "st.write(f\"MAE: {metrics.mean_absolute_error(y_test, preds):.2f}\")\n```")
    if "data-visualization" in system:
        return ("COMMENTARY:\nBar chart comparing average sales across regions.\n\n"
                "CODE:\n```python\nfig = px.bar(df.groupby('region')['sales'].mean().reset_index(), "
                "x='region', y='sales', title='Average sales by region', template='plotly_white')\n"
                "st.plotly_chart(fig, use_container_width=True)\n```")
    if "exploratory-data-analysis" in system:
        return ("COMMENTARY:\nMissing-value counts and summary statistics for every numeric column.\n\n"
                "CODE:\n```python\nst.write(df.isna().sum())\nst.write(df.describe())\n```")
    return "I can help with EDA, predictive modeling, causal analysis, and charts -- try asking about one of those."


def _call_openai(prompt, system, max_tokens, temperature):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise LLMError("MODEL_PROVIDER=openai but OPENAI_API_KEY is not set in your .env")
    try:
        from openai import OpenAI
    except ImportError as e:
        raise LLMError("Install the 'openai' package: pip install openai") from e

    client = OpenAI(api_key=api_key)
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
    )
    return resp.choices[0].message.content


def _call_anthropic(prompt, system, max_tokens, temperature):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMError("MODEL_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set in your .env")
    try:
        import anthropic
    except ImportError as e:
        raise LLMError("Install the 'anthropic' package: pip install anthropic") from e

    client = anthropic.Anthropic(api_key=api_key)
    model = os.environ.get("MODEL_NAME", "claude-haiku-4-5-20251001")
    kwargs = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
              "messages": [{"role": "user", "content": prompt}]}
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")


def _call_groq(prompt, system, max_tokens, temperature):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMError("MODEL_PROVIDER=groq but GROQ_API_KEY is not set in your .env")
    try:
        from groq import Groq
    except ImportError as e:
        raise LLMError("Install the 'groq' package: pip install groq") from e

    client = Groq(api_key=api_key)
    # llama-3.3-70b-versatile was shut down by Groq on 2026-08-16; openai/gpt-oss-120b
    # is Groq's current recommended production replacement (verified 2026-08-19).
    model = os.environ.get("MODEL_NAME", "openai/gpt-oss-120b")
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
    )
    return resp.choices[0].message.content


RESPONSE_FORMAT_INSTRUCTIONS = """
Do not write any import statements. df, pd, np, px, go, sm, smf, train_test_split,
LinearRegression, LogisticRegression, RandomForestRegressor, RandomForestClassifier,
DummyRegressor, DummyClassifier, metrics, StandardScaler, OneHotEncoder, LabelEncoder,
and (when available) IV2SLS are already in scope -- note LabelEncoder is provided for
a text/string classification target (sklearn classifiers also accept string labels
directly, but encode first if you prefer), and DummyRegressor/DummyClassifier are
provided specifically for a naive baseline comparison in predictive-modeling
questions. Code is statically scanned before execution and import statements for
anything outside a small stdlib allowlist (re, math, itertools, collections,
statistics, datetime, json) will be rejected.

Respond in EXACTLY this format, nothing else:

COMMENTARY:
<your plain-language explanation, no code>

CODE:
```python
<your python code here>
```
"""


def parse_commentary_and_code(text: str):
    """Splits a COMMENTARY:/CODE: formatted response into (commentary, code).
    Falls back gracefully if the model didn't follow the format exactly, since LLMs
    occasionally drop a label even when instructed not to."""
    commentary, code = "", ""
    if "CODE:" in text:
        before, after = text.split("CODE:", 1)
        commentary = before.replace("COMMENTARY:", "").strip()
        code = extract_code(after)
    else:
        code = extract_code(text)
        commentary = text.split("```")[0].replace("COMMENTARY:", "").strip()
    return commentary, code


def extract_code(text: str) -> str:
    """Pulls a python code block out of an LLM response. Falls back to the raw text
    if there's no fenced code block, since some models omit the fence for short
    snippets."""
    if "```" in text:
        parts = text.split("```")
        # parts alternate: text, code, text, code, ... -- code blocks are odd-indexed
        for i in range(1, len(parts), 2):
            block = parts[i]
            if block.strip().startswith("python"):
                block = block.strip()[len("python"):]
            if block.strip():
                return block.strip()
    return text.strip()
