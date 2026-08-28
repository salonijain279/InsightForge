"""
InsightForge -- an automated data scientist. MSBA 6461 course project (Path 4).
================================================================
Author: Saloni Jain

A substantially reworked course implementation, developed with Claude Code and
informed by ideas from open-source projects researched for this assignment:

  - Auto-Analyst (FireBird Technologies) -- the idea of routing a question to a
    specialist handler (EDA / predictive / causal / viz) instead of one giant
    prompt trying to do everything.
  - ydata-profiling -- the idea of a one-click full EDA profile report, as a
    complement to chat-driven ad-hoc questions.
  - LIDA (Microsoft) -- keeping visualization generation chat-driven and code-based
    rather than templated.
  - This project's own validated causal-inference work (test_causal_methods.py,
    built earlier tonight) -- DiD / IV / RDD / PSM identification strategies.

Architecture (built step by step, see PROJECT_LOG.md for the build order):
    app.py          <- this file: Streamlit UI, file upload, chat loop
    llm.py          <- plain-function LLM wrapper (OpenAI + Anthropic, no framework)
    router.py       <- classifies each question into eda/predictive/causal/viz/general
    handlers/       <- one module per analysis type, each returns (code, commentary)
    report.py       <- assembles the session's Q&A into a downloadable report

Deliberately NOT using a heavyweight agent framework (DSPy, LangChain, etc.) --
plain functions and explicit control flow are easier to read end to end and to
explain to a grader than framework internals.
"""

import os
import datetime as dt

import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

from pipeline import process_question
from recording_ui import render_items
from report import build_report, build_eda_profile
from deterministic import quick_baseline_model
from dataset_info import suggested_questions
from trust_card import build_trust_card, render_trust_card

load_dotenv()

st.set_page_config(page_title="InsightForge -- Automated Data Scientist", page_icon="\U0001F4A1", layout="wide")

# ---------------------------------------------------------------------------
# Cosmetic polish only -- every rule below is presentation (font, spacing,
# color, rounded corners); no rule changes what the app does. Base theme
# colors (background/accent) live in .streamlit/config.toml, this layers
# finer detail on top.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* One typeface for the whole app, not just headings -- Plus Jakarta Sans has
   clean regular/medium weights that read fine as body text too, so the UI
   feels like one designed product instead of headings in one font and
   everything else in the browser default. */
html, body, [class*="css"] {font-family: 'Plus Jakarta Sans', sans-serif;}
/* Bumped up from the browser default (16px) -- since every other size in this
   file is in rem, raising the root size scales body text, captions, and
   buttons together instead of needing every single rule touched by hand. */
html {font-size: 17.5px;}
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1120px;}

/* A soft top-down gradient instead of one flat hex -- gives the page a little
   depth rather than a single uniform block of color behind everything. */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #1A1510 0%, #12100C 420px, #12100C 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #221B13 0%, #1C1712 100%);
}

/* Hero header with a soft warm glow behind it */
.forge-hero-wrap {position: relative; margin-bottom: 0.3rem;}
.forge-hero-wrap::before {
    content: ""; position: absolute; top: -60px; left: -40px;
    width: 320px; height: 220px; z-index: -1;
    background: radial-gradient(circle, rgba(201,164,107,0.28), rgba(184,131,77,0.10) 55%, transparent 75%);
    filter: blur(10px);
}
.forge-hero {display: flex; align-items: center; gap: 0.7rem;}
.forge-icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 46px; height: 46px; border-radius: 13px; font-size: 1.5rem;
    background: linear-gradient(135deg, rgba(201,164,107,0.22), rgba(184,131,77,0.18));
    border: 1px solid rgba(201,164,107,0.35);
}
.forge-hero h1 {
    font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800;
    font-size: 2.3rem; margin: 0; letter-spacing: -0.01em;
    background: linear-gradient(90deg, #E8D5B5 0%, #C9A46B 55%, #A87C4F 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.forge-subtitle {
    /* Wide and flowing, not a narrow squeezed column -- a tight max-width made
       this wrap into many short lines against a lot of empty space to the
       right, which read as cramped rather than clean. This spans most of the
       block-container's width instead. */
    opacity: 0.80; font-size: 1.05rem; line-height: 1.6;
    margin: 0.5rem 0 1.3rem 0; max-width: 900px;
}

/* Section titles get the same display font for consistency */
h2, h3 {font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700;}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(160deg, rgba(201,164,107,0.10), rgba(184,131,77,0.05));
    border: 1px solid rgba(201,164,107,0.28);
    border-radius: 14px;
    padding: 0.95rem 1.1rem 0.65rem 1.1rem;
    transition: border-color 0.15s ease, transform 0.15s ease;
}
[data-testid="stMetric"]:hover {border-color: rgba(201,164,107,0.55); transform: translateY(-1px);}
[data-testid="stMetricLabel"] {opacity: 0.75;}

/* Buttons -- pill shaped, glow on hover */
.stButton > button {
    border-radius: 999px;
    border: 1px solid rgba(201,164,107,0.40);
    transition: all 0.15s ease;
}
.stButton > button:hover {
    border-color: #C9A46B;
    color: #E8D5B5;
    box-shadow: 0 0 0 3px rgba(201,164,107,0.15);
}

/* Chat + expanders */
[data-testid="stChatMessage"] {border-radius: 14px; padding: 0.4rem 0.2rem;}
div[data-testid="stExpander"] {
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* "routed to" pill badge -- one distinct color per analysis type, so the kind
   of answer is visible at a glance in a scrolled chat history, not just from
   reading the label text. */
.route-badge {
    display: inline-block; font-size: 0.78rem; font-weight: 600;
    padding: 0.15rem 0.65rem; border-radius: 999px; letter-spacing: 0.02em;
    margin-bottom: 0.35rem;
}
.route-badge-eda {
    background: rgba(56,189,248,0.15); border: 1px solid rgba(56,189,248,0.40); color: #7DD3FC;
}
.route-badge-predictive {
    background: rgba(251,191,36,0.15); border: 1px solid rgba(251,191,36,0.40); color: #FCD34D;
}
.route-badge-causal {
    background: rgba(139,92,246,0.15); border: 1px solid rgba(139,92,246,0.40); color: #C4B5FD;
}
.route-badge-visualization {
    background: rgba(52,211,153,0.15); border: 1px solid rgba(52,211,153,0.40); color: #6EE7B7;
}
.route-badge-general {
    background: rgba(148,163,184,0.15); border: 1px solid rgba(148,163,184,0.40); color: #CBD5E1;
}

/* Sidebar dataset caption gets a left accent bar instead of plain gray text,
   so the sidebar isn't visually flat next to the colorful hero. */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    border-left: 2px solid rgba(201,164,107,0.35); padding-left: 0.6rem;
}

/* Section dividers get a faint gradient instead of a flat line */
hr {
    border: none !important; height: 1px !important;
    background: linear-gradient(90deg, rgba(201,164,107,0.35), rgba(184,131,77,0.20), transparent) !important;
}

.forge-footer {opacity: 0.45; font-size: 0.76rem; margin-top: 1.5rem;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar / header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="forge-hero-wrap"><div class="forge-hero">'
    '<span class="forge-icon">\U0001F4A1</span><h1>InsightForge</h1>'
    '</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="forge-subtitle">An automated data scientist for plain-English questions. '
    'Upload a dataset, ask what you want to know, and InsightForge runs the right '
    'analysis and shows you exactly how it got there.</div>',
    unsafe_allow_html=True,
)

# Every bundled sample dataset this app has actually been tested against (the
# original demo dataset + the 3 genuinely-different-domain generalization-test
# datasets, see PROJECT_LOG.md Section 9.2) -- surfaced as one dropdown instead of
# a single "use sample data" button, so switching between them for a demo takes
# one click instead of re-uploading a file each time.
SAMPLE_DATASETS = {
    "Store sales campaign (has a known, verifiable causal effect)": (
        "sample_data/sales_campaign.csv",
        "480 rows, synthetic -- built with a known causal effect baked in so the "
        "causal-inference handler's estimate can be checked against ground truth.",
    ),
    "Breast cancer diagnosis (real medical dataset)": (
        "sample_data/generalization_test/breast_cancer_diagnosis.csv",
        "569 rows, the real sklearn/UCI breast cancer dataset -- 30 numeric "
        "features, text target (malignant/benign).",
    ),
    "HR attrition (people-analytics domain)": (
        "sample_data/generalization_test/hr_attrition.csv",
        "1200 rows, synthetic but realistic -- mixed types, missing values, "
        "imbalanced target.",
    ),
    "Supply chain late delivery (logistics domain)": (
        "sample_data/generalization_test/supply_chain_late_delivery.csv",
        "3000 rows, synthetic but realistic -- mixed types, missing values, "
        "imbalanced target.",
    ),
}

if "ui_epoch" not in st.session_state:
    # Bumped on a full reset so the dataset selectbox and file uploader below get
    # fresh widget keys -- Streamlit has no API to clear a widget's value in place,
    # giving it a new key is the standard way to force it back to its default.
    st.session_state.ui_epoch = 0

with st.sidebar:
    st.header("Data")
    _epoch = st.session_state.ui_epoch
    data_source = st.selectbox(
        "Choose a dataset",
        options=["Upload your own file"] + list(SAMPLE_DATASETS.keys()),
        key=f"data_source_{_epoch}",
    )
    uploaded_file = None
    selected_sample = None
    if data_source == "Upload your own file":
        uploaded_file = st.file_uploader(
            "Upload a CSV or Excel file", type=["csv", "xlsx", "xls"], key=f"uploader_{_epoch}"
        )
        st.caption(
            "Any tabular CSV or Excel file works -- it isn't limited to a "
            "pre-loaded schema. Don't have one handy? Switch to one of the "
            "sample datasets in the dropdown above instead."
        )
    else:
        selected_sample = data_source
        st.caption(SAMPLE_DATASETS[data_source][1])

    reset_clicked = st.button("\U0001F504 New analysis / reset session", width="stretch")

    # A persistent export action, not part of the main reading flow -- it used to
    # sit between the baseline-model section and the chat, which put it in the
    # middle of the page every time you scrolled past it. The sidebar is where a
    # one-off "export this session" action belongs, next to Reset.
    if st.session_state.get("history"):
        report_md = build_report(st.session_state.dataset_name, st.session_state.df, st.session_state.history)
        st.download_button(
            "⬇️ Download full analysis report (.md)",
            data=report_md,
            file_name=f"analysis_report_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            width="stretch",
        )

    # Quick-start questions moved out of the sidebar -- they now live inline,
    # right above the chat input at the bottom of the conversation (see below),
    # so there's exactly one copy instead of two.
    clicked_question = None

    # The provider/key status line and setup instructions were always-on
    # sidebar clutter -- internal config detail a demo viewer doesn't need to
    # see, not something that helps anyone using the app. Now this only ever
    # shows up if something's actually broken (no key found), since that's
    # the one case where staying silent would just look like the app is
    # stuck with no explanation.
    _provider = os.environ.get("MODEL_PROVIDER", "").strip().lower()
    _key_present = bool(os.environ.get({"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
                                         "groq": "GROQ_API_KEY"}.get(_provider, ""), ""))
    if not _provider or not _key_present:
        st.divider()
        st.warning(
            "No LLM provider configured -- set MODEL_PROVIDER and the matching "
            "API key in your `.env` file (see `.env-template`) before asking a "
            "question."
        )

    st.markdown(
        '<div class="forge-footer">Built for MSBA 6461 &middot; Advanced AI for NLP</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {question, kind, code, commentary, output, error}
if "loaded_source" not in st.session_state:
    st.session_state.loaded_source = None


def _reset_analysis_state():
    """Clears everything tied to the PREVIOUS dataset -- chat history, the baseline
    model result, the EDA profile toggle, and the target-column selection. Without
    this, switching datasets (e.g. sales_campaign -> supply_chain_late_delivery)
    left the old dataset's chat answers and baseline-model numbers on screen,
    which reads as the new dataset's results if you don't look closely -- a real
    bug found by manually switching datasets, not a hypothetical one."""
    st.session_state.history = []
    st.session_state.baseline_result = None
    st.session_state.show_profile = False
    # target_col is a widget key (the selectbox below owns it) -- delete it so the
    # widget re-initializes to its default (the new dataset's first column) instead
    # of keeping a column name that may not even exist in the new dataset.
    st.session_state.pop("target_col", None)


def _full_reset():
    """The 'New analysis / reset session' button: unlike a dataset SWITCH (which
    only needs _reset_analysis_state), this clears the loaded dataset itself, so
    the app goes all the way back to the empty "choose a dataset" state -- found
    live: without this, clicking reset cleared the chat but left the old dataset,
    its row/column metrics, and its sidebar description still on screen, which
    reads as a partial, confusing reset. Bumping ui_epoch gives the dataset
    selectbox and file uploader fresh widget keys on the next render, so they
    visually clear too instead of still showing the old selection/file."""
    _reset_analysis_state()
    st.session_state.df = None
    st.session_state.dataset_name = None
    st.session_state.loaded_source = None
    st.session_state.ui_epoch += 1


if reset_clicked:
    _full_reset()
    st.rerun()

if selected_sample is not None:
    # Only reload from disk when the dropdown selection actually changed --
    # re-running this every rerender would wipe st.session_state.history whenever
    # the user asks a question (a rerun), since every question triggers a rerun.
    if st.session_state.loaded_source != f"sample::{selected_sample}":
        rel_path, _ = SAMPLE_DATASETS[selected_sample]
        full_path = os.path.join(os.path.dirname(__file__), *rel_path.split("/"))
        st.session_state.df = pd.read_csv(full_path)
        st.session_state.dataset_name = f"{os.path.basename(rel_path)} (sample -- {selected_sample.lower()})"
        st.session_state.loaded_source = f"sample::{selected_sample}"
        _reset_analysis_state()
        # The sidebar (rendered earlier in this same script run) already read
        # st.session_state.df before this assignment happened, so the "Try
        # asking" chips there would show stale/empty for one full pass without
        # this -- same off-by-one-rerun issue as the EDA profile button above.
        st.rerun()
elif uploaded_file is not None:
    if st.session_state.loaded_source != f"upload::{uploaded_file.name}::{uploaded_file.size}":
        if uploaded_file.name.endswith(("xlsx", "xls")):
            st.session_state.df = pd.read_excel(uploaded_file)
        else:
            st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.dataset_name = uploaded_file.name
        st.session_state.loaded_source = f"upload::{uploaded_file.name}::{uploaded_file.size}"
        _reset_analysis_state()
        st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
if st.session_state.df is None:
    st.info("Pick a dataset from the sidebar to get started -- upload your own file or try one of the samples.")
    st.stop()

df = st.session_state.df
st.subheader(f"Dataset: {st.session_state.dataset_name}")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", len(df.columns))
c3.metric("Missing values", int(df.isna().sum().sum()))
st.dataframe(df.head(10), width="stretch")

if st.session_state.get("show_profile"):
    profile = build_eda_profile(df)
    with st.expander("Full EDA profile", expanded=True):
        st.write(f"**Shape:** {profile['shape']}  |  **Duplicate rows:** {profile['duplicate_rows']} "
                 f"({profile['duplicate_row_pct']}%)  |  **Completeness:** {profile['completeness_pct']}% "
                 f"of all cells non-missing")
        st.write("**Column types**")
        st.dataframe(profile["dtypes"], width="stretch")
        if len(profile["missing"]):
            st.write("**Missing values (columns with at least one)**")
            st.dataframe(profile["missing"], width="stretch")
        else:
            st.write("**Missing values:** none")
        if profile["constant_cols"]:
            st.warning(f"**Constant/near-constant columns (no variation, no analytical value):** "
                       f"{', '.join(profile['constant_cols'])}")
        if profile["high_cardinality_cols"]:
            st.warning(f"**Likely identifier / free-text columns (>95% unique values):** "
                       f"{', '.join(profile['high_cardinality_cols'])} -- probably not useful "
                       f"to group by or predict on directly.")
        if profile["outlier_counts"]:
            st.write("**Possible outliers (IQR rule, count of values outside 1.5x IQR)**")
            st.dataframe(
                pd.DataFrame(list(profile["outlier_counts"].items()), columns=["column", "outlier_count"]),
                width="stretch",
            )
        if profile["numeric_summary"] is not None:
            st.write("**Numeric column summary**")
            st.dataframe(profile["numeric_summary"], width="stretch")
        if profile["categorical_summary"]:
            st.write("**Categorical column top values**")
            for col, counts in profile["categorical_summary"].items():
                st.write(f"*{col}*")
                st.dataframe(counts, width="stretch")
        if profile["correlation"] is not None:
            st.write("**Correlation matrix (numeric columns)**")
            fig = px.imshow(profile["correlation"], text_auto=".2f", color_continuous_scale="RdBu_r",
                             title="Correlation matrix", template="plotly_white")
            st.plotly_chart(fig, width="stretch")

with st.expander("\U0001F3AF Quick baseline model (no LLM, deterministic, $0)"):
    st.caption(
        "Trains a real baseline (predict the mean/majority class) alongside a real simple "
        "model (Linear/LogisticRegression) on an 80/20 split -- every number here comes from "
        "actually fitting scikit-learn, not from an LLM describing what a model would report. "
        "For custom modeling questions, ask in chat instead -- that path can use any approach "
        "you describe, this one is fixed and predictable."
    )
    target_col = st.selectbox("Target column to predict", options=list(df.columns), key="target_col")
    if st.button("Train baseline + simple model"):
        st.session_state.baseline_result = quick_baseline_model(df, target_col)
    if st.session_state.get("baseline_result"):
        res = st.session_state.baseline_result
        if "error" in res:
            st.error(res["error"])
        else:
            st.write(f"**Task type (inferred):** {res['task_type']}  |  "
                     f"**Train/test rows:** {res['n_train']}/{res['n_test']}")
            st.write(f"**Numeric features:** {res['numeric_features'] or 'none'}")
            st.write(f"**Categorical features:** {res['categorical_features'] or 'none'}")
            for model_name, m in res["metrics"].items():
                st.write(f"*{model_name}:* " + ", ".join(f"{k}={v}" for k, v in m.items()))
            if res["leakage_warnings"]:
                for w in res["leakage_warnings"]:
                    st.warning(w)

st.divider()

# A quirky one-line intro, shown only before the first message -- gives the empty
# chat window some personality instead of a blank box, the way a real product's
# first-run state would.
if not st.session_state.history:
    with st.chat_message("assistant"):
        st.write(
            "Hi, I'm InsightForge \U0001F4A1 -- think of me as a data scientist who "
            "never sleeps, never needs a coffee break, and shows every line of code "
            "it runs. Pick a question below, or just ask me anything about this data."
        )

# Replay chat history
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        # Worded as a capability, not an incident report -- this fires when the first
        # generated attempt hit an error and the pipeline silently retried and
        # succeeded before ever showing you anything broken (see pipeline.py
        # attempt_fix()). "auto-fixed after an error" reads like something went
        # wrong on screen; "self-corrected" describes the same event as the designed
        # recovery behavior it actually is.
        fixed_note = " &middot; self-corrected" if turn.get("auto_fixed") else ""
        badge_kind = turn.get("kind") or "general"
        st.markdown(
            f'<span class="route-badge route-badge-{badge_kind}">{badge_kind}{fixed_note}</span>',
            unsafe_allow_html=True,
        )
        if turn.get("commentary"):
            st.write(turn["commentary"])
        if turn.get("code"):
            with st.expander("Show generated code"):
                st.code(turn["code"], language="python")
        if turn.get("display_items"):
            render_items(st, turn["display_items"])
        elif turn.get("output"):
            # rare fallback: code used print() instead of st.write()
            st.text(turn["output"])
        if turn.get("error"):
            st.error(turn["error"][:1200])
        if turn.get("kind") not in (None, "pending", "general") and not turn.get("error"):
            render_trust_card(st, build_trust_card(turn, df))

if st.session_state.pop("_just_answered", False):
    # components.html runs in a sandboxed iframe, but it's same-origin with the
    # main Streamlit page, so window.parent.document is reachable -- this is the
    # standard workaround for the fact that Streamlit has no built-in
    # "scroll to bottom" API. Fires once per new answer (popped from session
    # state immediately), not on every rerun, so it doesn't fight you if you've
    # scrolled up to reread an earlier answer and then toggle an unrelated
    # widget (which also triggers a rerun).
    st.iframe(
        """<script>
        const el = window.parent.document.querySelector('[data-testid="stAppScrollToBottomContainer"]');
        if (el) { el.scrollTop = el.scrollHeight; }
        </script>""",
        height=1,
    )

# One persistent "Try asking" row, always visible regardless of how many
# questions you've already asked -- previously this only showed before the
# first message (so it "went away" the moment you asked anything) and then,
# briefly, lived only in the sidebar (so it scrolled out of the visible
# viewport once the conversation grew). Placed here, right above the chat
# input, it scrolls along with the conversation and lands right next to the
# input box every time the page auto-scrolls to a fresh answer.
st.caption("Try asking:")
_suggestions = suggested_questions(df)
_cols = st.columns(len(_suggestions) + 1)
with _cols[0]:
    if st.button("Full EDA profile (instant, $0)", key="suggest_eda_profile", width="stretch"):
        st.session_state.show_profile = True
        st.rerun()
for _col, (_, _q) in zip(_cols[1:], _suggestions):
    with _col:
        if st.button(_q, key=f"suggest_{_q}", width="stretch"):
            clicked_question = _q

question = st.chat_input("Ask a question about your data...")

if question or clicked_question:
    st.session_state.history.append({"question": question or clicked_question, "kind": "pending"})
    st.rerun()

if st.session_state.history and st.session_state.history[-1]["kind"] == "pending":
    with st.spinner("Routing and analyzing..."):
        result = process_question(st.session_state.history[-1]["question"], df)
    st.session_state.history[-1] = result
    # Consumed once, right after the history replay loop below, to scroll to the
    # answer that just landed -- without this the page stayed wherever it was
    # scrolled, so a new answer appeared off-screen and had to be found by hand.
    st.session_state._just_answered = True
    st.rerun()
