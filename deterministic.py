"""
deterministic.py -- analysis that runs with ZERO LLM calls: instant, free, and not
dependent on a model correctly writing code.

report.py's build_eda_profile() was already this kind of deterministic function.
This adds the predictive-analysis equivalent, so "predictive" isn't only reachable
through LLM-generated code (handlers/predictive.py + pipeline.py) -- there's a plain,
auditable Python path too. The chat-driven predictive handler stays for custom,
open-ended modeling questions ("what if I only use urban stores", "try a random
forest instead"); this deterministic path is for the common case (train something
against a target column) where a fixed function is genuinely more dependable than
asking an LLM to write it correctly every time.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import metrics


def _infer_task_type(y: pd.Series) -> str:
    # Use pandas' own numeric-dtype check rather than comparing dtype == object --
    # newer pandas (>=3.0) can read CSV string columns into a native StringDtype
    # instead of legacy object dtype, and `dtype == object` silently misses that,
    # falling through to `% 1` on string data and crashing. is_numeric_dtype covers
    # object, native string, and category dtypes correctly by construction (it's
    # false for all of them), whatever pandas version is installed.
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    if y.nunique() <= 10 and (y.dropna() % 1 == 0).all():
        return "classification"
    return "regression"


def _leakage_warnings(df: pd.DataFrame, target: str, features: list) -> list:
    """Cheap, honest heuristics -- not a proof of no leakage, just the two most
    common accidental-leakage patterns: a feature that's an exact duplicate of the
    target, or a numeric feature suspiciously perfectly correlated with it."""
    warnings = []
    for col in features:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            if df[[col, target]].dropna().shape[0] > 5:
                corr = df[col].corr(df[target])
                if pd.notna(corr) and abs(corr) > 0.98:
                    warnings.append(
                        f"'{col}' correlates {corr:.3f} with the target -- check it isn't "
                        "derived from the target itself (a leak) before trusting this model."
                    )
        except Exception:
            pass
    return warnings


def quick_baseline_model(df: pd.DataFrame, target: str) -> dict:
    """Trains a DummyRegressor/DummyClassifier (the floor -- what you'd get by
    guessing) alongside a real, simple model (Linear/LogisticRegression), on an
    80/20 split, with median imputation for numeric columns and one-hot encoding
    for categorical ones. Returns a result dict for the caller to render. Zero LLM
    calls -- every number here comes from actually fitting a model, not from an
    LLM describing what a model would probably report."""
    data = df.dropna(subset=[target]).copy()
    if len(data) < 20:
        return {"error": f"Only {len(data)} non-missing rows for target '{target}' -- too few to split and train reliably."}

    features = [c for c in data.columns if c != target]
    if not features:
        return {"error": "No feature columns left besides the target."}

    task_type = _infer_task_type(data[target])
    numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(data[c])]
    categorical_features = [c for c in features if c not in numeric_features]

    X = data[features]
    y = data[target]

    stratify = y if (task_type == "classification" and y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric_features),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_features),
        ],
        remainder="drop",
    )

    if task_type == "regression":
        dummy = Pipeline([("prep", preprocessor), ("model", DummyRegressor(strategy="mean"))])
        real = Pipeline([("prep", preprocessor), ("model", LinearRegression())])
        dummy.fit(X_train, y_train)
        real.fit(X_train, y_train)
        dummy_preds = dummy.predict(X_test)
        real_preds = real.predict(X_test)
        result_metrics = {
            "baseline (predict the mean)": {
                "R2": round(metrics.r2_score(y_test, dummy_preds), 3),
                "MAE": round(metrics.mean_absolute_error(y_test, dummy_preds), 3),
            },
            "LinearRegression": {
                "R2": round(metrics.r2_score(y_test, real_preds), 3),
                "MAE": round(metrics.mean_absolute_error(y_test, real_preds), 3),
            },
        }
    else:
        dummy = Pipeline([("prep", preprocessor), ("model", DummyClassifier(strategy="most_frequent"))])
        real = Pipeline([("prep", preprocessor), ("model", LogisticRegression(max_iter=1000))])
        dummy.fit(X_train, y_train)
        real.fit(X_train, y_train)
        dummy_preds = dummy.predict(X_test)
        real_preds = real.predict(X_test)
        result_metrics = {
            "baseline (predict the majority class)": {
                "accuracy": round(metrics.accuracy_score(y_test, dummy_preds), 3),
            },
            "LogisticRegression": {
                "accuracy": round(metrics.accuracy_score(y_test, real_preds), 3),
                "F1 (weighted)": round(metrics.f1_score(y_test, real_preds, average="weighted"), 3),
            },
        }

    return {
        "task_type": task_type,
        "target": target,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "metrics": result_metrics,
        "leakage_warnings": _leakage_warnings(data, target, features),
    }
