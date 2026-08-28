"""Deterministic checks for the four causal estimation patterns used by the app.

Each test creates synthetic data with a known effect and verifies that the
estimator recovers it within a reasonable tolerance. This validates the
statistical patterns, not the reliability of live LLM code generation.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
from sklearn.linear_model import LogisticRegression

np.random.seed(0)


def test_difference_in_differences():
    n = 200
    df = pd.DataFrame({
        "treated": np.repeat([0, 1], n // 2),
        "post": np.tile([0, 1], n // 2),
    })
    true_effect = 3
    df["outcome"] = (
        5 + 2 * df["treated"] + df["post"]
        + true_effect * df["treated"] * df["post"]
        + np.random.normal(0, 1, n)
    )
    model = smf.ols("outcome ~ treated * post", data=df).fit(cov_type="HC1")
    estimate = model.params["treated:post"]
    assert abs(estimate - true_effect) < 0.5
    print(f"[PASS] DiD: estimate={estimate:.2f}, true={true_effect}")


def test_instrumental_variables():
    n = 1000
    true_effect = 2
    instrument = np.random.normal(0, 1, n)
    confounder = np.random.normal(0, 1, n)
    treatment = 0.8 * instrument + confounder + np.random.normal(0, 1, n)
    outcome = true_effect * treatment + confounder + np.random.normal(0, 1, n)
    df = pd.DataFrame({"outcome": outcome, "treatment": treatment, "instrument": instrument})
    result = IV2SLS.from_formula(
        "outcome ~ 1 + [treatment ~ instrument]", data=df
    ).fit()
    estimate = result.params["treatment"]
    assert abs(estimate - true_effect) < 0.3
    print(f"[PASS] IV/2SLS: estimate={estimate:.2f}, true={true_effect}")


def test_regression_discontinuity():
    n = 2000
    true_effect = 4
    running = np.random.uniform(-10, 10, n)
    treated = (running >= 0).astype(int)
    outcome = 1 + 0.5 * running + true_effect * treated + np.random.normal(0, 1, n)
    df = pd.DataFrame({"running_c": running, "treated": treated, "outcome": outcome})
    local = df[df["running_c"].abs() <= 3]
    model = smf.ols("outcome ~ running_c * treated", data=local).fit()
    estimate = model.params["treated"]
    assert abs(estimate - true_effect) < 0.5
    print(f"[PASS] RDD: estimate={estimate:.2f}, true={true_effect}")


def test_propensity_score_ipw():
    n = 1000
    true_effect = 3
    x1 = np.random.normal(0, 1, n)
    propensity = 1 / (1 + np.exp(-0.5 * x1))
    treated = np.random.binomial(1, propensity)
    outcome = true_effect * treated + 2 * x1 + np.random.normal(0, 1, n)
    df = pd.DataFrame({"x1": x1, "treated": treated, "outcome": outcome})
    model = LogisticRegression().fit(df[["x1"]], df["treated"])
    df["ps"] = model.predict_proba(df[["x1"]])[:, 1]
    treated_mean = (df["treated"] * df["outcome"] / df["ps"]).sum() / (
        df["treated"] / df["ps"]
    ).sum()
    control_mean = (
        ((1 - df["treated"]) * df["outcome"] / (1 - df["ps"])).sum()
        / ((1 - df["treated"]) / (1 - df["ps"])).sum()
    )
    estimate = treated_mean - control_mean
    assert abs(estimate - true_effect) < 0.5
    print(f"[PASS] PSM/IPW: estimate={estimate:.2f}, true={true_effect}")


if __name__ == "__main__":
    test_difference_in_differences()
    test_instrumental_variables()
    test_regression_discontinuity()
    test_propensity_score_ipw()
    print("All four causal-method checks passed.")
