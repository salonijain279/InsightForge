"""Builds the plain-text dataset summary every handler sends to the LLM as context,
plus the data-aware suggested-question chips shown in the app UI. Kept out of app.py
(same pattern as deterministic.py) so this logic is importable and testable without a
Streamlit runtime -- see test_suggested_questions.py."""

import re

import pandas as pd


def summarize_dataset(df, max_rows_preview=3) -> str:
    parts = [
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        f"Columns and dtypes:\n{df.dtypes.to_string()}",
        f"Missing values per column:\n{df.isna().sum().to_string()}",
        f"First {max_rows_preview} rows:\n{df.head(max_rows_preview).to_string()}",
    ]
    return "\n\n".join(parts)


def _is_id_like(df: pd.DataFrame, col: str) -> bool:
    """An identifier column makes a meaningless "average of" / prediction target.
    Match "id" as its own name segment (id, store_id, id_number -- not "sales" or
    "avoid" or "width"), and only treat every-value-unique as an ID signal for
    INTEGER columns -- a continuous float target (like "sales") is very often
    all-unique too, just because it's continuous, not because it's an identifier."""
    if re.search(r"(^|_)id($|_)", col.lower()):
        return True
    return bool(pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() == len(df))


def suggested_questions(df: pd.DataFrame) -> list:
    """Builds 3-4 example questions from the ACTUAL columns of whatever dataset is
    loaded -- generic by construction, not hardcoded to one dataset, so this works
    the same way for the bundled sample or any uploaded CSV (see generalization
    testing in PROJECT_LOG.md Section 9.2 for why that distinction matters).
    Returns (icon, question_text) pairs so the UI can render each as a labeled chip."""
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    numeric_for_agg = [c for c in numeric_cols if not _is_id_like(df, c)] or numeric_cols
    # No emoji prefix -- a run of mismatched icons (target/bar-chart/link) on
    # plain-text suggestion chips read as clutter, not as helpful labeling.
    qs = [("", "Give me summary statistics and check for missing values")]
    if numeric_for_agg:
        target = numeric_for_agg[-1]
        qs.append(("", f"Build a model to predict {target}"))
    if cat_cols and numeric_for_agg:
        # Prefer a categorical column with a manageable number of groups for the
        # bar-chart suggestion -- a free-text/near-unique column (e.g. a "Name" or
        # "Ticket" field) makes a useless, unreadable bar chart with hundreds of bars.
        chartable_cat = next((c for c in cat_cols if df[c].nunique() <= 20), cat_cols[0])
        qs.append(("", f"Show me a bar chart of average {numeric_for_agg[0]} by {chartable_cat}"))
    # The bundled sample dataset specifically has a known, validatable causal effect
    # baked in (see PROJECT_LOG.md Section 1) -- surface that question only for it.
    if {"treated", "post", "sales"}.issubset(df.columns):
        qs.append(("", "What was the effect of the campaign on sales?"))
    return qs[:4]
