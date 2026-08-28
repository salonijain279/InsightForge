"""
Integration test: runs the REAL pipeline (router -> handler prompt -> parse -> exec)
against the REAL sample dataset, with only the LLM call itself mocked out (since that
needs a live key). The mocked response is written the way a real model would actually
respond to this handler's system prompt, so this checks the full path end to end,
not just each piece in isolation.

sample_data/sales_campaign.csv was generated with a KNOWN true causal effect of the
campaign on sales: 4500 (see the generation script referenced in PROJECT_LOG.md).
A correct DiD analysis on this data should land close to that number.
"""
import os
from unittest.mock import patch

import pandas as pd
import pipeline

DF = pd.read_csv(os.path.join(os.path.dirname(__file__), "sample_data", "sales_campaign.csv"))


def test_causal_question_end_to_end():
    fake_llm_response = """COMMENTARY:
1. Identification strategy: Difference-in-Differences. `treated` marks stores that received the campaign, `post` marks the post-campaign period.
2. Key assumption: parallel trends -- absent the campaign, treated and control stores' sales would have moved in parallel.
3. Effect size: see the printed regression coefficient on treated:post below, with its standard error and p-value.
4. Limitation: only 40 stores and 12 months; unobserved store-level shocks coinciding with the campaign could bias this estimate. This is an estimate under the parallel-trends assumption, not a proven fact.

CODE:
```python
model = smf.ols('sales ~ treated * post', data=df).fit(cov_type='HC1')
st.write(f"DiD estimate: {model.params['treated:post']:.2f}")
st.write(f"p-value: {model.pvalues['treated:post']:.5f}")
```
"""
    with patch("pipeline.call_llm", return_value=fake_llm_response) as m:
        result = pipeline.process_question(
            "What was the effect of the campaign on sales?", DF
        )

    assert result["kind"] == "causal", result["kind"]
    assert "error" not in result, result.get("error")
    write_items = [payload for kind, payload in result["display_items"] if kind == "write"]
    assert any("DiD estimate" in str(x) for x in write_items), result["display_items"]
    # Parse the printed estimate back out and check it's close to the known true effect (4500)
    estimate_text = [str(x) for x in write_items if "DiD estimate" in str(x)][0]
    estimate = float(estimate_text.split(":")[1].strip())
    assert abs(estimate - 4500) < 800, f"DiD estimate {estimate} too far from true 4500"
    print(f"[PASS] causal question routed correctly, executed against real data, "
          f"estimate={estimate:.0f} (true=4500)")


def test_eda_question_end_to_end():
    fake_llm_response = """COMMENTARY:
This reports missing values per column and summary statistics for the numeric columns.

CODE:
```python
st.write(df.isna().sum())
st.write(df.describe())
```
"""
    with patch("pipeline.call_llm", return_value=fake_llm_response):
        result = pipeline.process_question("Give me summary statistics and check for missing values", DF)
    assert result["kind"] == "eda", result["kind"]
    assert "error" not in result, result.get("error")
    print("[PASS] eda question routed correctly and executed cleanly against real data")


def test_predictive_question_end_to_end():
    """The assignment requires predictive analysis as a mandatory capability (causal
    is the bonus), so this gets the same real-data, real-execution treatment as the
    causal and EDA tests above -- not just a claim that it works."""
    fake_llm_response = """COMMENTARY:
Trains a linear regression to predict sales from ad spend, foot traffic and store
size, using an 80/20 train/test split, and reports R-squared and MAE on the held-out
test set.

CODE:
```python
features = ['ad_spend', 'foot_traffic', 'size_sqft']
data = df.dropna(subset=features + ['sales'])
X = data[features]
y = data['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
preds = model.predict(X_test)
r2 = metrics.r2_score(y_test, preds)
mae = metrics.mean_absolute_error(y_test, preds)
st.write(f"R2: {r2:.3f}")
st.write(f"MAE: {mae:.2f}")
```
"""
    with patch("pipeline.call_llm", return_value=fake_llm_response):
        result = pipeline.process_question("Build a model to predict sales", DF)

    assert result["kind"] == "predictive", result["kind"]
    assert "error" not in result, result.get("error")
    write_items = [str(payload) for kind, payload in result["display_items"] if kind == "write"]
    assert any(x.startswith("R2:") for x in write_items), result["display_items"]
    r2 = float([x for x in write_items if x.startswith("R2:")][0].split(":")[1].strip())
    assert -1.0 <= r2 <= 1.0, f"R2 {r2} outside a sane range -- something is structurally wrong"
    print(f"[PASS] predictive question routed correctly, trained and evaluated against "
          f"real held-out data, R2={r2:.3f}")


def test_broken_code_gets_auto_fixed():
    """First LLM call returns code with a typo'd column name; the auto-fix call
    (also mocked, second call) returns the corrected version; the third call
    produces the post-execution grounded explanation. Confirms the retry path
    actually re-executes, grounds, and recovers instead of just giving up."""
    broken = """COMMENTARY:
Average sales.

CODE:
```python
st.write(df['saless'].mean())
```
"""
    fixed = "```python\nst.write(df['sales'].mean())\n```"

    grounded = "The corrected analysis ran successfully and reported the computed mean sales."
    with patch("pipeline.call_llm", side_effect=[broken, fixed, grounded]) as m:
        # phrased to hit the EDA keyword fast-path deterministically (contains
        # "summary"), so this test isn't relying on a second, unmocked LLM call
        # inside the router's fallback classifier
        result = pipeline.process_question("Give me summary stats on average sales", DF)

    assert result.get("auto_fixed") is True, result
    assert "error" not in result, result.get("error")
    assert m.call_count == 3, (
        "should call the LLM once for the answer, once for the fix, and once for "
        "the post-execution grounded explanation"
    )
    print("[PASS] broken generated code (bad column name) is auto-fixed and re-executed successfully")


def test_missing_api_key_reported_cleanly():
    os.environ["MODEL_PROVIDER"] = "openai"
    os.environ.pop("OPENAI_API_KEY", None)
    result = pipeline.process_question("Show me a chart of sales", DF)
    assert "error" in result and "OPENAI_API_KEY" in result["error"]
    print("[PASS] missing API key surfaces as a clear message in the result, not a crash")


if __name__ == "__main__":
    test_causal_question_end_to_end()
    test_eda_question_end_to_end()
    test_predictive_question_end_to_end()
    test_broken_code_gets_auto_fixed()
    test_missing_api_key_reported_cleanly()
    print("\nAll pipeline integration tests passed.")
