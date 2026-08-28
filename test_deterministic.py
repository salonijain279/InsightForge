"""
Tests deterministic.py's quick_baseline_model() -- the zero-LLM predictive path --
against the real shipped sample dataset. No mocking needed: there's no LLM call to
mock, which is exactly the point of this module.
"""
import os
import pandas as pd

from deterministic import quick_baseline_model

DF = pd.read_csv(os.path.join(os.path.dirname(__file__), "sample_data", "sales_campaign.csv"))


def test_regression_target_trains_and_beats_baseline():
    result = quick_baseline_model(DF, "sales")
    assert "error" not in result, result.get("error")
    assert result["task_type"] == "regression"
    r2_baseline = result["metrics"]["baseline (predict the mean)"]["R2"]
    r2_model = result["metrics"]["LinearRegression"]["R2"]
    assert r2_baseline < 0.05, f"baseline R2 {r2_baseline} should be ~0 (predicting the mean)"
    assert r2_model > r2_baseline, "the real model should beat the do-nothing baseline"
    print(f"[PASS] regression target 'sales': baseline R2={r2_baseline}, LinearRegression R2={r2_model}")


def test_classification_target_trains_and_beats_baseline():
    result = quick_baseline_model(DF, "treated")
    assert "error" not in result, result.get("error")
    assert result["task_type"] == "classification"
    acc_baseline = result["metrics"]["baseline (predict the majority class)"]["accuracy"]
    acc_model = result["metrics"]["LogisticRegression"]["accuracy"]
    assert 0 <= acc_baseline <= 1 and 0 <= acc_model <= 1
    print(f"[PASS] classification target 'treated': baseline acc={acc_baseline}, LogisticRegression acc={acc_model}")


def test_string_labeled_classification_target():
    """Regression test for a real bug: _infer_task_type() used to check
    `y.dtype == object`, which is False for pandas's native StringDtype (the
    default for CSV string columns on pandas >= 3.0) -- silently falling through
    to a `% 1` numeric check that crashes on strings. Caught by testing against a
    genuinely new dataset (a string-labeled classification target) rather than only
    the numeric 0/1 target this suite already had. Fixed by checking
    pd.api.types.is_numeric_dtype instead."""
    labeled = DF.copy()
    labeled["treated_label"] = labeled["treated"].map({1: "treatment", 0: "control"})
    result = quick_baseline_model(labeled, "treated_label")
    assert "error" not in result, result.get("error")
    assert result["task_type"] == "classification", result["task_type"]
    acc_model = result["metrics"]["LogisticRegression"]["accuracy"]
    assert 0 <= acc_model <= 1
    print(f"[PASS] string-labeled classification target ('treatment'/'control' text, not 0/1) "
          f"trains without crashing, accuracy={acc_model}")


def test_categorical_features_handled():
    # 'region' is a string column -- must be one-hot encoded, not crash the fit
    result = quick_baseline_model(DF, "sales")
    assert "region" in result["categorical_features"]
    print("[PASS] categorical feature 'region' handled without crashing")


def test_too_few_rows_reports_clean_error():
    tiny = DF.head(5)
    result = quick_baseline_model(tiny, "sales")
    assert "error" in result and "too few" in result["error"]
    print("[PASS] too-few-rows case returns a clean error instead of crashing")


def test_leakage_heuristic_flags_a_near_duplicate_column():
    leaky = DF.copy()
    leaky["sales_copy"] = leaky["sales"] * 1.0001  # near-perfect correlation with target
    result = quick_baseline_model(leaky, "sales")
    assert any("sales_copy" in w for w in result["leakage_warnings"]), result["leakage_warnings"]
    print("[PASS] leakage heuristic flags a near-duplicate-of-target column")


if __name__ == "__main__":
    test_regression_target_trains_and_beats_baseline()
    test_classification_target_trains_and_beats_baseline()
    test_string_labeled_classification_target()
    test_categorical_features_handled()
    test_too_few_rows_reports_clean_error()
    test_leakage_heuristic_flags_a_near_duplicate_column()
    print("\nAll deterministic.py tests passed.")
