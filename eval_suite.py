"""
eval_suite.py -- a small batch evaluation, not just isolated unit tests.

Honesty note, since this is exactly the kind of thing worth being precise about:
this suite runs with the LLM call MOCKED (there is no live API key in this
environment), using responses written to be representative of what each handler's
system prompt actually asks for. It does NOT evaluate whether a real, live LLM call
would generate equally good code for these exact questions -- that can only be
checked with a live key, which happens on your machine. What this DOES validate,
against real data with real execution, is everything downstream of the LLM call:
routing correctness, code execution, the safety gate, and auto-fix recovery -- the
parts of the system that are deterministic and don't depend on which provider or
model you configure.

Run: python3 eval_suite.py
"""
import time
from unittest.mock import patch

import pandas as pd
import pipeline

DF = pd.read_csv("sample_data/sales_campaign.csv")


def _fake_llm(user, system=None, **kwargs):
    s = system or ""
    if "causal-inference" in s:
        return ("COMMENTARY:\n1. Identification: DiD. 2. Assumption: parallel trends. "
                "3. Effect + SE below. 4. Limitation: 40 stores, unobserved shocks possible.\n\n"
                "CODE:\n```python\nmodel = smf.ols('sales ~ treated * post', data=df).fit(cov_type='HC1')\n"
                "st.write(f\"DiD estimate: {model.params['treated:post']:.2f}\")\n```")
    if "predictive-modeling" in s:
        return ("COMMENTARY:\nTrains a linear model on ad_spend, foot_traffic, size_sqft.\n\n"
                "CODE:\n```python\nfeatures=['ad_spend','foot_traffic','size_sqft']\n"
                "data=df.dropna(subset=features+['sales'])\nX=data[features]\ny=data['sales']\n"
                "X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)\n"
                "m=LinearRegression().fit(X_train,y_train)\n"
                "st.write(f\"R2: {metrics.r2_score(y_test, m.predict(X_test)):.3f}\")\n```")
    if "data-visualization" in s:
        return ("COMMENTARY:\nBar chart of average sales by region.\n\n"
                "CODE:\n```python\nfig = px.bar(df.groupby('region')['sales'].mean().reset_index(), "
                "x='region', y='sales', title='Avg sales by region', template='plotly_white')\n"
                "st.plotly_chart(fig, use_container_width=True)\n```")
    if "exploratory-data-analysis" in s:
        return ("COMMENTARY:\nMissing values and summary stats.\n\n"
                "CODE:\n```python\nst.write(df.isna().sum())\nst.write(df.describe())\n```")
    return "I can help with EDA, predictive modeling, causal analysis, and charts -- try asking about one of those."


def _fake_llm_broken_then_fixed(user, system=None, **kwargs):
    """Simulates a typo'd column name on the first call; the fix prompt (identifiable
    by containing 'raised an error') gets a corrected second response."""
    if "raised an error" in user:
        return "```python\nst.write(df['sales'].mean())\n```"
    return "COMMENTARY:\nAverage sales.\n\nCODE:\n```python\nst.write(df['saless'].mean())\n```"


def _fake_llm_unsafe():
    return ("COMMENTARY:\nChecking the environment.\n\n"
            "CODE:\n```python\nimport os\nst.write(os.listdir('/'))\n```")


CASES = [
    ("Give me summary statistics and check for missing values", "eda", _fake_llm),
    ("What's the correlation between ad spend and foot traffic?", "eda", _fake_llm),
    ("Build a model to predict sales", "predictive", _fake_llm),
    ("Which features drive foot traffic?", "predictive", _fake_llm),
    ("What was the effect of the campaign on sales?", "causal", _fake_llm),
    ("Show me a bar chart of average sales by region", "visualization", _fake_llm),
    ("Plot the distribution of ad spend", "visualization", _fake_llm),
    ("Hello, what can you do?", "general", _fake_llm),
]


def run():
    rows = []

    for question, expected_kind, fake in CASES:
        t0 = time.perf_counter()
        with patch("pipeline.call_llm", side_effect=fake):
            result = pipeline.process_question(question, DF)
        elapsed = round((time.perf_counter() - t0) * 1000)
        routed_ok = result["kind"] == expected_kind
        executed_ok = "error" not in result
        rows.append((question, expected_kind, result["kind"], routed_ok, executed_ok, elapsed, ""))

    # Case: broken code should be auto-fixed and recover
    t0 = time.perf_counter()
    with patch("pipeline.call_llm", side_effect=_fake_llm_broken_then_fixed):
        result = pipeline.process_question("Give me summary stats on average sales", DF)
    elapsed = round((time.perf_counter() - t0) * 1000)
    rows.append((
        "Give me summary stats on average sales (typo'd column, first attempt)",
        "eda", result["kind"], result["kind"] == "eda",
        "error" not in result and result.get("auto_fixed") is True,
        elapsed, "auto-fix path" if result.get("auto_fixed") else "DID NOT AUTO-FIX",
    ))

    # Case: unsafe code should be blocked by the safety gate, not executed
    t0 = time.perf_counter()
    with patch("pipeline.call_llm", return_value=_fake_llm_unsafe()):
        result = pipeline.process_question("What was the effect of the campaign on sales?", DF)
    elapsed = round((time.perf_counter() - t0) * 1000)
    blocked_correctly = "error" in result and "safety scan" in result["error"]
    rows.append((
        "(adversarial) causal question, LLM response contains 'import os'",
        "causal", result["kind"], result["kind"] == "causal",
        blocked_correctly, elapsed,
        "blocked before execution, as intended" if blocked_correctly else "SAFETY GATE FAILED TO BLOCK",
    ))

    print(f"{'Question':<62} {'Expect':<8} {'Got':<8} {'Route':<6} {'Exec/Handled':<12} {'ms':<6} Notes")
    print("-" * 130)
    n_route_ok = n_exec_ok = 0
    for question, expected, got, routed_ok, executed_ok, elapsed, note in rows:
        n_route_ok += routed_ok
        n_exec_ok += bool(executed_ok)
        q_disp = (question[:59] + "...") if len(question) > 62 else question
        print(f"{q_disp:<62} {expected:<8} {got:<8} {'OK' if routed_ok else 'FAIL':<6} "
              f"{'OK' if executed_ok else 'FAIL':<12} {elapsed:<6} {note}")
    print("-" * 130)
    print(f"Routing correct: {n_route_ok}/{len(rows)}   Execution/handling correct: {n_exec_ok}/{len(rows)}")
    return rows


if __name__ == "__main__":
    run()
