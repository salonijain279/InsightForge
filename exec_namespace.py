"""Builds the fixed namespace LLM-generated code executes against.

Explicit and closed on purpose: generated code can only use what's listed here (plus
Python builtins), so behavior is predictable and easy to reason about -- no import
statements need to appear in generated code, and no surprise access to the wider
filesystem/network beyond what these specific libraries expose.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from recording_ui import RecordingUI


def build_exec_namespace(df: pd.DataFrame) -> dict:
    """`st` here is a RecordingUI, not the real streamlit module -- see
    recording_ui.py for why. The caller (pipeline.run_code) reads ns['st'].items
    after exec() to get what was "displayed"."""
    ns = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "st": RecordingUI(),
        "px": px,
        "go": go,
        "sm": sm,
        "smf": smf,
        "train_test_split": train_test_split,
        "LinearRegression": LinearRegression,
        "LogisticRegression": LogisticRegression,
        "RandomForestRegressor": RandomForestRegressor,
        "RandomForestClassifier": RandomForestClassifier,
        "DummyRegressor": DummyRegressor,
        "DummyClassifier": DummyClassifier,
        "metrics": metrics,
        "StandardScaler": StandardScaler,
        "OneHotEncoder": OneHotEncoder,
        "LabelEncoder": LabelEncoder,
    }
    try:
        from linearmodels.iv import IV2SLS
        ns["IV2SLS"] = IV2SLS
    except ImportError:
        pass  # IV questions will fail with a clear ImportError inside run_code, not at app startup
    return ns
