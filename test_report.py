"""Tests report.py's build_eda_profile(), specifically the data-quality metrics
added on top of the original profile (completeness %, duplicate row %, constant
columns, high-cardinality columns, IQR outlier counts) -- all deterministic
computations over the real dataframe, so each is checked against a value worked
out by hand, not just "did it run"."""

import pandas as pd

from report import build_eda_profile

# 10 rows, 'const' never varies, 'id' is all-unique, 'val' has one clear IQR outlier
# (100 vs a tight cluster of 1-9), 2 missing cells out of 30 total.
DF = pd.DataFrame({
    "id": list(range(10)),
    "const": [1] * 10,
    "val": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
    "maybe_missing": [1, 2, 3, None, 5, 6, None, 8, 9, 10],
})


def test_completeness_percentage():
    profile = build_eda_profile(DF)
    total_cells = DF.shape[0] * DF.shape[1]
    missing_cells = int(DF.isna().sum().sum())
    expected = round(100 * (1 - missing_cells / total_cells), 2)
    assert profile["completeness_pct"] == expected
    assert missing_cells == 2  # sanity-check the fixture itself


def test_duplicate_row_percentage_with_no_duplicates():
    profile = build_eda_profile(DF)
    assert profile["duplicate_rows"] == 0
    assert profile["duplicate_row_pct"] == 0.0


def test_duplicate_row_percentage_with_duplicates():
    dup_df = pd.concat([DF, DF.iloc[[0]]], ignore_index=True)
    profile = build_eda_profile(dup_df)
    assert profile["duplicate_rows"] == 1
    assert profile["duplicate_row_pct"] == round(100 * 1 / len(dup_df), 2)


def test_constant_column_detected():
    profile = build_eda_profile(DF)
    assert "const" in profile["constant_cols"]
    assert "val" not in profile["constant_cols"]


def test_high_cardinality_column_detected():
    profile = build_eda_profile(DF)
    assert "id" in profile["high_cardinality_cols"]
    # the constant column must never also show up as high-cardinality
    assert "const" not in profile["high_cardinality_cols"]


def test_continuous_float_column_not_flagged_as_high_cardinality():
    # Regression test: found via manual UI testing, not by reading code. A
    # continuous float column (e.g. "sales") is very often near-100% unique
    # just because it's continuous, not because it's an identifier -- an
    # earlier version of this check flagged it anyway. Same false-positive
    # class as the suggested_questions ID heuristic bug (test_suggested_questions.py).
    float_df = pd.DataFrame({
        "sales": [1000.12, 2000.55, 3000.91, 4000.34, 5000.77, 6000.18, 7000.62, 8000.05, 9000.49, 10000.83],
        "region": ["East"] * 5 + ["West"] * 5,
    })
    profile = build_eda_profile(float_df)
    assert "sales" not in profile["high_cardinality_cols"]


def test_outlier_detected_by_iqr_rule():
    profile = build_eda_profile(DF)
    assert "val" in profile["outlier_counts"]
    assert profile["outlier_counts"]["val"] == 1
    # a constant column has an IQR of 0 and must not appear (division-by-zero guard)
    assert "const" not in profile["outlier_counts"]


def test_no_outliers_on_uniform_data():
    uniform = pd.DataFrame({"x": list(range(20))})
    profile = build_eda_profile(uniform)
    assert profile["outlier_counts"] == {}


if __name__ == "__main__":
    test_completeness_percentage()
    test_duplicate_row_percentage_with_no_duplicates()
    test_duplicate_row_percentage_with_duplicates()
    test_constant_column_detected()
    test_high_cardinality_column_detected()
    test_outlier_detected_by_iqr_rule()
    test_no_outliers_on_uniform_data()
    print("All report.py data-quality tests passed (completeness/duplicates/constant/"
          "high-cardinality/outliers all match hand-computed expected values).")
