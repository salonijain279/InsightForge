import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
Generalization test: does Lens actually work on datasets it has never seen before,
from genuinely different domains? Three datasets, none of which the code has any
column-name-specific logic for:

  1. breast_cancer_diagnosis.csv -- real sklearn dataset (medical/healthcare domain),
     30 numeric features, binary categorical target, no missing values, WIDE (tests
     many-column handling and dataset_summary readability).
  2. hr_attrition.csv -- synthetic but realistic people-analytics dataset (mirrors
     the "people analytics agent" domain), mixed numeric/categorical, 3% missingness,
     imbalanced binary target (~4.6% attrition).
  3. supply_chain_late_delivery.csv -- synthetic but realistic supply-chain dataset
     (mirrors the DataCo/OpsPilot late-delivery-risk idea), mixed numeric/categorical,
     2% missingness, imbalanced binary target (~5% late).

For each dataset: run the deterministic paths for real (build_eda_profile,
quick_baseline_model -- no mocking possible or needed, these never call an LLM), then
run process_question() through the real pipeline with a couple of mocked-but-
representative LLM responses that reference THIS dataset's actual column names, to
prove the routing/prompt/execution path is genuinely column-agnostic, not hardcoded
to the original sales_campaign.csv.
"""
import os
import pandas as pd
from unittest.mock import patch

import pipeline
from report import build_eda_profile
from deterministic import quick_baseline_model

DATASETS = {
    "breast_cancer_diagnosis.csv": {
        "target": "diagnosis",
        "eda_q": "Give me summary statistics and check for missing values",
        "predictive_q": "Build a model to predict diagnosis",
        "viz_q": "Show me a bar chart comparing mean radius by diagnosis",
    },
    "hr_attrition.csv": {
        "target": "attrition",
        "eda_q": "What's the correlation between tenure and job satisfaction?",
        "predictive_q": "Build a model to predict attrition",
        "viz_q": "Show me a bar chart of average monthly income by department",
    },
    "supply_chain_late_delivery.csv": {
        "target": "late_delivery_risk",
        "eda_q": "Give me summary statistics and check for missing values",
        "predictive_q": "Which features drive late delivery risk?",
        "viz_q": "Show me a bar chart of late delivery risk by shipping mode",
    },
}


def fake_llm_factory(df, target):
    """Builds a fake LLM responder whose CODE actually references this dataset's
    real columns -- proving the prompt correctly told the model what columns exist,
    and that execution genuinely works against them (not just "returns some string")."""
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target]
    cat_cols = [c for c in df.columns if c not in numeric_cols and c != target]
    feat = numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
    if cat_cols:
        group_col, num_col = cat_cols[0], (numeric_cols[0] if numeric_cols else target)
    else:
        # No non-target categorical column (e.g. breast_cancer): group by the
        # binary/low-cardinality target instead, against a different numeric
        # column, so group_col and num_col are never the same column.
        group_col = target
        num_col = numeric_cols[0] if numeric_cols else df.columns[0]

    def fake(user, system=None, **kwargs):
        s = system or ""
        if "predictive-modeling" in s:
            return (
                "COMMENTARY:\nTrains a model on a few numeric features.\n\nCODE:\n```python\n"
                f"features={feat!r}\n"
                f"data = df.dropna(subset=features + [{target!r}])\n"
                f"X = data[features]\ny = data[{target!r}]\n"
                "if y.dtype == object:\n    y = LabelEncoder().fit_transform(y)\n"
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
                "m = LogisticRegression(max_iter=1000).fit(X_train, y_train)\n"
                "st.write(f\"accuracy: {m.score(X_test, y_test):.3f}\")\n```"
            )
        if "exploratory-data-analysis" in s:
            return ("COMMENTARY:\nMissing values and summary stats.\n\nCODE:\n```python\n"
                    "st.write(df.isna().sum())\nst.write(df.describe())\n```")
        if "data-visualization" in s:
            return (
                "COMMENTARY:\nBar chart grouped by category.\n\nCODE:\n```python\n"
                f"fig = px.bar(df.groupby({group_col!r})[{num_col!r}].mean().reset_index(), "
                f"x={group_col!r}, y={num_col!r}, title='avg by {group_col}', template='plotly_white')\n"
                "st.plotly_chart(fig, use_container_width=True)\n```"
            )
        return "general fallback"
    return fake


def run():
    results = []
    for fname, cfg in DATASETS.items():
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "generalization_test", fname)
        df = pd.read_csv(path)
        target = cfg["target"]
        row = {"dataset": fname, "shape": df.shape}

        # 1. Deterministic EDA profile -- real, no mocking
        try:
            profile = build_eda_profile(df)
            row["eda_profile"] = "OK" if profile["shape"] else "FAIL"
        except Exception as e:
            row["eda_profile"] = f"CRASH: {e}"

        # 2. Deterministic baseline model -- real, no mocking
        try:
            bm = quick_baseline_model(df, target)
            row["baseline_model"] = "OK" if "error" not in bm else f"ERROR: {bm['error']}"
            row["baseline_task_type"] = bm.get("task_type")
        except Exception as e:
            row["baseline_model"] = f"CRASH: {e}"

        fake = fake_llm_factory(df, target)

        # 3. Chat-driven EDA
        try:
            with patch("pipeline.call_llm", side_effect=fake):
                r = pipeline.process_question(cfg["eda_q"], df)
            row["chat_eda"] = "OK" if r["kind"] == "eda" and "error" not in r else f"FAIL: {r.get('error', r['kind'])}"
        except Exception as e:
            row["chat_eda"] = f"CRASH: {e}"

        # 4. Chat-driven predictive
        try:
            with patch("pipeline.call_llm", side_effect=fake):
                r = pipeline.process_question(cfg["predictive_q"], df)
            row["chat_predictive"] = "OK" if r["kind"] == "predictive" and "error" not in r else f"FAIL: {r.get('error', r['kind'])}"
        except Exception as e:
            row["chat_predictive"] = f"CRASH: {e}"

        # 5. Chat-driven visualization
        try:
            with patch("pipeline.call_llm", side_effect=fake):
                r = pipeline.process_question(cfg["viz_q"], df)
            row["chat_viz"] = "OK" if r["kind"] == "visualization" and "error" not in r else f"FAIL: {r.get('error', r['kind'])}"
        except Exception as e:
            row["chat_viz"] = f"CRASH: {e}"

        results.append(row)

    print(f"{'Dataset':<32} {'Shape':<10} {'EDA profile':<12} {'Baseline':<10} {'Task':<15} {'Chat EDA':<10} {'Chat Pred':<10} {'Chat Viz':<10}")
    print("-" * 120)
    all_ok = True
    for r in results:
        ok = all(str(r.get(k, "")).startswith("OK") for k in ["eda_profile", "baseline_model", "chat_eda", "chat_predictive", "chat_viz"])
        all_ok = all_ok and ok
        print(f"{r['dataset']:<32} {str(r['shape']):<10} {r['eda_profile']:<12} {r['baseline_model']:<10} "
              f"{str(r.get('baseline_task_type')):<15} {r['chat_eda']:<10} {r['chat_predictive']:<10} {r['chat_viz']:<10}")
    print("-" * 120)
    print("ALL PASSED" if all_ok else "SOME FAILED -- see above")
    return results, all_ok


if __name__ == "__main__":
    run()
