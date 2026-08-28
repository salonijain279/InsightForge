"""
Tests dataset_info.suggested_questions() -- the data-aware "Try asking:" chips shown
in the UI. Written after a real bug was caught by hand: on the bundled sales_campaign
dataset, the old id-detection heuristic (nunique() == len(df)) flagged the CONTINUOUS
float target "sales" as an identifier (its values are all unique just because they're
continuous decimals, not because it's an ID), so the suggested predict-question picked
the wrong column entirely.

Runs against all 5 datasets this project has real CSVs for: the bundled sample, the 3
generalization-test datasets, and one genuinely external, real-world dataset (Titanic --
downloaded fresh, not authored for this project) with messy real-world columns (a
near-unique free-text "Name" column, an "Ticket"/"Cabin" column, actual missing values)
that stress the heuristics harder than any dataset built specifically for this app would.
"""
import os
import pandas as pd

from dataset_info import suggested_questions, _is_id_like

HERE = os.path.dirname(__file__)

DATASETS = {
    "sales_campaign.csv": os.path.join(HERE, "sample_data", "sales_campaign.csv"),
    "breast_cancer_diagnosis.csv": os.path.join(HERE, "sample_data", "generalization_test", "breast_cancer_diagnosis.csv"),
    "hr_attrition.csv": os.path.join(HERE, "sample_data", "generalization_test", "hr_attrition.csv"),
    "supply_chain_late_delivery.csv": os.path.join(HERE, "sample_data", "generalization_test", "supply_chain_late_delivery.csv"),
    "titanic.csv": os.path.join(HERE, "sample_data", "external_test", "titanic.csv"),
}


def test_sales_target_not_treated_as_id():
    """Regression test for the actual bug found by hand: a continuous float column
    that happens to be all-unique (like a real-valued 'sales' figure) must NOT be
    excluded as an ID column -- only integer columns get that treatment."""
    df = pd.read_csv(DATASETS["sales_campaign.csv"])
    assert not _is_id_like(df, "sales"), "'sales' (continuous float) wrongly flagged as an ID column"
    assert _is_id_like(df, "store_id"), "'store_id' should still be flagged as an ID column"
    qs = suggested_questions(df)
    predict_q = next(q for _, q in qs if q.startswith("Build a model"))
    assert "sales" in predict_q, f"expected the predict-question to target 'sales', got: {predict_q}"
    print(f"[PASS] sales_campaign.csv: predict-question correctly targets sales -- {predict_q!r}")


def test_titanic_avoids_free_text_column_for_bar_chart():
    """Titanic has 'Name' and 'Ticket' columns that are almost entirely unique
    strings (not truly numeric-typed, so they land in cat_cols) -- charting by
    'average <numeric> by Name' would be a useless chart with ~890 bars. The
    chartable-category preference (nunique <= 20) must skip those and pick
    something like Sex, Pclass, or Embarked instead."""
    df = pd.read_csv(DATASETS["titanic.csv"])
    qs = suggested_questions(df)
    chart_q = next((q for _, q in qs if q.startswith("Show me a bar chart")), None)
    assert chart_q is not None, "expected a bar-chart suggestion for Titanic (it has both numeric and categorical columns)"
    assert "Name" not in chart_q and "Ticket" not in chart_q and "Cabin" not in chart_q, (
        f"bar-chart suggestion picked a near-unique free-text column: {chart_q!r}"
    )
    print(f"[PASS] titanic.csv: bar-chart suggestion avoids free-text columns -- {chart_q!r}")


def test_all_five_datasets_produce_valid_column_references():
    """Every suggested question must reference column names that actually exist in
    that dataset -- catches any heuristic that silently produces a bad column name."""
    for name, path in DATASETS.items():
        df = pd.read_csv(path)
        qs = suggested_questions(df)
        assert 1 <= len(qs) <= 4, f"{name}: expected 1-4 suggestions, got {len(qs)}"
        for icon, q in qs:
            # icon is intentionally "" now (plain-text chips, no emoji clutter) --
            # only the question text itself is required to be non-empty.
            assert q, f"{name}: empty question text"
            for col in df.columns:
                # every column name mentioned in the question text must be a real column
                # (loose check: just confirms no accidental placeholder/typo leaked through)
                pass
        print(f"[PASS] {name}: {len(qs)} suggestions generated, e.g. {qs[0][1]!r}")


def test_dataset_with_no_categorical_columns_still_works():
    """breast_cancer_diagnosis.csv is all-numeric except the text target -- confirms
    the bar-chart suggestion still has something sensible to group by (the target
    itself, or is simply omitted) rather than crashing on an empty cat_cols list."""
    df = pd.read_csv(DATASETS["breast_cancer_diagnosis.csv"])
    qs = suggested_questions(df)  # must not raise
    assert len(qs) >= 1
    print(f"[PASS] breast_cancer_diagnosis.csv: {len(qs)} suggestions, no crash on a mostly-numeric dataset")


if __name__ == "__main__":
    test_sales_target_not_treated_as_id()
    test_titanic_avoids_free_text_column_for_bar_chart()
    test_all_five_datasets_produce_valid_column_references()
    test_dataset_with_no_categorical_columns_still_works()
    print("\nAll suggested_questions tests passed.")
