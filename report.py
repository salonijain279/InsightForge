"""Assembles the session's Q&A history into one downloadable Markdown report --
the "report generation" the assignment asks for, not just chat scrollback."""

import datetime as dt

import pandas as pd


def build_report(dataset_name: str, df, history: list) -> str:
    lines = [
        "# Automated Data Scientist -- Analysis Report",
        "",
        f"**Dataset:** {dataset_name}  ",
        f"**Shape:** {df.shape[0]} rows x {df.shape[1]} columns  ",
        f"**Generated:** {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Questions asked and answers",
        "",
    ]
    for i, turn in enumerate(history, 1):
        if turn.get("kind") in (None, "pending"):
            continue
        lines.append(f"### {i}. {turn['question']}")
        lines.append("")
        lines.append(f"*Routed to: `{turn['kind']}`*" + (" *(auto-fixed after an error)*" if turn.get("auto_fixed") else ""))
        lines.append("")
        if turn.get("commentary"):
            lines.append(turn["commentary"])
            lines.append("")
        if turn.get("code"):
            lines.append("```python")
            lines.append(turn["code"])
            lines.append("```")
            lines.append("")
        if turn.get("display_items"):
            text_items = [str(payload) for kind, payload in turn["display_items"] if kind in ("write", "text")]
            if text_items:
                lines.append("**Output:**")
                lines.append("")
                for t in text_items:
                    lines.append(f"> {t}")
                lines.append("")
        if turn.get("error"):
            lines.append(f"**Error (not resolved):** {turn['error'][:500]}")
            lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("*Generated automatically by the Automated Data Scientist system -- "
                  "MSBA 6461 course project.*")
    return "\n".join(lines)


def _outlier_counts(df, numeric_cols) -> dict:
    """IQR-rule outlier count per numeric column (values outside Q1-1.5*IQR,
    Q3+1.5*IQR) -- the standard, deterministic definition, not a judgment call.
    A count, not a removal: this is a data-quality signal to look at, not an
    automatic action taken on the data."""
    counts = {}
    for c in numeric_cols:
        col = df[c].dropna()
        if len(col) < 4:
            continue
        q1, q3 = col.quantile(0.25), col.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((col < lo) | (col > hi)).sum())
        if n_out > 0:
            counts[c] = n_out
    return counts


def build_eda_profile(df) -> dict:
    """Deterministic, $0-cost, no-LLM full EDA profile -- inspired by ydata-profiling's
    one-click report idea, but hand-rolled here to keep the dependency footprint small
    (ydata-profiling itself pulls in a fairly heavy dependency chain).

    Returns a dict of {section_name: pandas object or string} for the caller to render.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    missing_cells = int(df.isna().sum().sum())
    completeness_pct = round(100 * (1 - missing_cells / total_cells), 2) if total_cells else 100.0

    # Constant / near-constant columns: zero information for modeling or grouping,
    # worth flagging so a user doesn't waste a question asking about one.
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]

    # High-cardinality columns: every/almost-every value unique -- likely an
    # identifier or free-text column, not something to aggregate or group by.
    # Deliberately excludes float columns: a continuous measurement (e.g. "sales",
    # "ad_spend") is very often near-100% unique just because it's continuous, not
    # because it's an identifier -- flagging it here would be the exact same
    # false-positive already found and fixed in dataset_info._is_id_like() for the
    # suggested-question chips (see test_suggested_questions.py); this check needs
    # the same guard, not a fresh, looser one. Only integer and text/object columns
    # are candidates.
    high_cardinality_cols = [
        c for c in df.columns
        if c not in constant_cols
        and n_rows > 0
        and not pd.api.types.is_float_dtype(df[c])
        and df[c].nunique(dropna=True) / n_rows > 0.95
    ]

    profile = {
        "shape": f"{n_rows} rows x {n_cols} columns",
        "dtypes": df.dtypes.astype(str),
        "missing": df.isna().sum()[df.isna().sum() > 0],
        "numeric_summary": df[numeric_cols].describe().T if numeric_cols else None,
        "categorical_summary": (
            {c: df[c].value_counts().head(10) for c in cat_cols} if cat_cols else None
        ),
        "correlation": df[numeric_cols].corr() if len(numeric_cols) > 1 else None,
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_row_pct": round(100 * df.duplicated().sum() / n_rows, 2) if n_rows else 0.0,
        "completeness_pct": completeness_pct,
        "constant_cols": constant_cols,
        "high_cardinality_cols": high_cardinality_cols,
        "outlier_counts": _outlier_counts(df, numeric_cols),
    }
    return profile
