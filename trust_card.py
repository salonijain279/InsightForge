"""trust_card.py -- the "Analysis Trust Card" feature from the external review.

Deliberately built as a DETERMINISTIC fact-extractor, not an LLM call. The whole
point of a trust card is to tell the user what they can rely on -- handing that job
to an LLM (e.g. "rate your own trustworthiness 1-10") would just reintroduce the
exact ungrounded-commentary problem generate_grounded_commentary() in pipeline.py
was built to fix, one level up. Instead, every fact on the card is either read
directly off the dataset (rows/columns actually in scope) or detected by scanning
the ACTUAL generated code that ran, via plain regex over known library calls --
nothing here is guessed or inferred by a model.

This means the card can be wrong in only one direction: it can fail to notice a
pattern (a false negative, shown as simply absent from the card), never invent one
that isn't in the code. That asymmetry is intentional.
"""

import re

# label -> regex; matched against the exact code string that was executed (after
# any auto-fix), so this reflects what actually ran, not what the LLM said it would do.
_METHOD_PATTERNS = [
    ("Linear regression", r"\bLinearRegression\b"),
    ("Logistic regression", r"\bLogisticRegression\b"),
    ("Random forest (regression)", r"\bRandomForestRegressor\b"),
    ("Random forest (classification)", r"\bRandomForestClassifier\b"),
    ("Naive baseline (mean) compared", r"\bDummyRegressor\b"),
    ("Naive baseline (majority class) compared", r"\bDummyClassifier\b"),
    ("OLS regression (statsmodels)", r"\bsmf\.ols\b"),
    ("Instrumental variables (2SLS)", r"\bIV2SLS\b"),
    ("Chart generated (plotly)", r"\bpx\.\w+\(|\bgo\.Figure\b"),
]

_DATA_HANDLING_PATTERNS = [
    ("Rows with missing values dropped (dropna)", r"\bdropna\b"),
    ("Missing values filled (fillna)", r"\bfillna\b"),
    ("Train/test split performed before fitting", r"\btrain_test_split\b"),
    ("Features scaled (StandardScaler)", r"\bStandardScaler\b"),
    ("Categorical columns encoded", r"\bOneHotEncoder\b|\bget_dummies\b|\bLabelEncoder\b"),
    ("Standard errors clustered by entity", r"cov_type\s*=\s*['\"]cluster['\"]"),
    ("Entity/time fixed effects included", r"C\(\s*\w+\s*\)"),
]


def _scan(code: str, patterns) -> list:
    return [label for label, pattern in patterns if re.search(pattern, code)]


def build_trust_card(turn: dict, df) -> dict:
    """turn is one entry from st.session_state.history (see pipeline.process_question
    for its shape). Returns a plain dict of facts to render -- no free text, no
    generated language, so there's nothing here for a reader to mistake for an
    LLM's opinion of itself."""
    code = turn.get("code") or ""
    return {
        "kind": turn.get("kind"),
        "dataset_rows": int(len(df)),
        "dataset_cols": int(len(df.columns)),
        "auto_fixed": bool(turn.get("auto_fixed")),
        "had_error": bool(turn.get("error")),
        "methods_detected": _scan(code, _METHOD_PATTERNS),
        "data_handling_detected": _scan(code, _DATA_HANDLING_PATTERNS),
    }


def render_trust_card(st_module, card: dict):
    """Renders the card built by build_trust_card via a compact expander. Kept
    separate from build_trust_card so the fact-extraction logic (the part that
    matters for correctness) stays independently testable without a Streamlit
    runtime -- see test_trust_card.py."""
    with st_module.expander("\U0001F50E Trust card -- what this answer is actually based on"):
        st_module.caption(
            "Every line below is detected directly from the dataset and the code that "
            "ran, not described by the model -- if something isn't listed, it wasn't "
            "detected, not necessarily absent."
        )
        st_module.write(f"**Ran against:** {card['dataset_rows']:,} rows x {card['dataset_cols']} columns")
        if card["auto_fixed"]:
            st_module.write("**Note:** the first generated code failed and was auto-corrected before this ran.")
        if card["methods_detected"]:
            st_module.write("**Method(s) detected in the executed code:** " + ", ".join(card["methods_detected"]))
        else:
            st_module.write("**Method(s) detected in the executed code:** none of the known patterns matched.")
        if card["data_handling_detected"]:
            st_module.write("**Data handling detected:** " + ", ".join(card["data_handling_detected"]))
        else:
            st_module.write("**Data handling detected:** none of the known patterns matched (e.g. no dropna/split found).")
