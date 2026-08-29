"""Tests trust_card.py's fact-extraction: confirms it detects real patterns in
actual generated code and does NOT claim a method was used when it wasn't --
the whole point of a "trust card" is that it can't lie, so a false positive here
would be worse than a missed detection."""

import pandas as pd

from trust_card import build_trust_card

DF = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 5, 6, 7]})


def test_detects_predictive_method_and_baseline():
    code = (
        "X_train, X_test, y_train, y_test = train_test_split(df[['a']], df['b'], test_size=0.2)\n"
        "m = LinearRegression().fit(X_train, y_train)\n"
        "base = DummyRegressor(strategy='mean').fit(X_train, y_train)\n"
        "st.write(m.score(X_test, y_test))\n"
    )
    turn = {"kind": "predictive", "code": code}
    card = build_trust_card(turn, DF)
    assert "Linear regression" in card["methods_detected"]
    assert "Naive baseline (mean) compared" in card["methods_detected"]
    assert "Train/test split performed before fitting" in card["data_handling_detected"]


def test_does_not_claim_baseline_when_absent():
    code = "m = LinearRegression().fit(df[['a']], df['b'])\nst.write(m.score(df[['a']], df['b']))\n"
    turn = {"kind": "predictive", "code": code}
    card = build_trust_card(turn, DF)
    assert "Linear regression" in card["methods_detected"]
    assert not any("baseline" in m.lower() for m in card["methods_detected"])
    assert "Train/test split performed before fitting" not in card["data_handling_detected"]


def test_detects_causal_clustering_and_fixed_effects():
    code = (
        "model = smf.ols('sales ~ treated:post + C(store) + C(month)', data=df)"
        ".fit(cov_type='cluster', cov_kwds={'groups': df['store']})\n"
        "st.write(model.params['treated:post'])\n"
    )
    turn = {"kind": "causal", "code": code}
    card = build_trust_card(turn, DF)
    assert "OLS regression (statsmodels)" in card["methods_detected"]
    assert "Standard errors clustered by entity" in card["data_handling_detected"]
    assert "Entity/time fixed effects included" in card["data_handling_detected"]


def test_reports_dataset_shape_and_auto_fixed_flag():
    turn = {"kind": "eda", "code": "st.write(df.describe())", "auto_fixed": True}
    card = build_trust_card(turn, DF)
    assert card["dataset_rows"] == 4
    assert card["dataset_cols"] == 2
    assert card["auto_fixed"] is True


def test_empty_code_detects_nothing_rather_than_guessing():
    turn = {"kind": "eda", "code": ""}
    card = build_trust_card(turn, DF)
    assert card["methods_detected"] == []
    assert card["data_handling_detected"] == []


if __name__ == "__main__":
    test_detects_predictive_method_and_baseline()
    test_does_not_claim_baseline_when_absent()
    test_detects_causal_clustering_and_fixed_effects()
    test_reports_dataset_shape_and_auto_fixed_flag()
    test_empty_code_detects_nothing_rather_than_guessing()
    print("All trust_card tests passed (detects real patterns, never invents a method that wasn't in the code).")
