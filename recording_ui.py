"""
recording_ui.py -- why this file exists (a real bug found while testing, not
a hypothetical one):

Generated code is told to use st.write/st.plotly_chart (since it runs "inside
Streamlit"). But this app calls process_question() from a plain code path in
app.py, appends the result to st.session_state.history, then calls st.rerun() to
re-render the chat from history. Streamlit's st.write etc. only actually draw
something when called during a live script run that's still on screen -- a call
made right before st.rerun() throws its own render away when the page immediately
re-executes from the top. Anything the generated code "displayed" via real st.*
calls during that first pass would be silently lost after the rerun, and this
would NOT show up as an error anywhere -- it would just look like the answer was
blank. Caught this with test_pipeline.py, not by reading the code.

Fix: don't hand generated code the real `st` module. Hand it a RecordingUI that
looks like st (same method names) but just records what was displayed, in order,
without drawing anything. app.py then replays that recorded list with the REAL
st.* calls every time the chat history renders -- including after a rerun -- so
output is never lost.
"""

import pandas as pd


def _arrow_safe(payload):
    """Normalize mixed-type object columns before Streamlit serializes them.

    LLM-generated summaries such as ``df.describe(include="all").T`` can place
    strings and numbers in the same object column. Pandas accepts that shape,
    but Arrow may log a conversion traceback before Streamlit applies its own
    fallback. Converting only genuinely mixed object columns keeps numeric data
    numeric while making replay quiet and deterministic.
    """
    if not isinstance(payload, pd.DataFrame):
        return payload

    safe = payload.copy()
    for column in safe.columns:
        series = safe[column]
        if series.dtype != "object":
            continue
        observed_types = series.dropna().map(type).nunique()
        if observed_types > 1:
            safe[column] = series.map(lambda value: None if pd.isna(value) else str(value))
    return safe


class RecordingUI:
    def __init__(self):
        self.items = []  # list of (kind, payload) tuples, in display order

    def write(self, *args, **kwargs):
        self.items.append(("write", args[0] if len(args) == 1 else args))

    def dataframe(self, data, **kwargs):
        self.items.append(("dataframe", data))

    def table(self, data, **kwargs):
        self.items.append(("table", data))

    def plotly_chart(self, fig, **kwargs):
        self.items.append(("plotly_chart", fig))

    def pyplot(self, fig=None, **kwargs):
        self.items.append(("pyplot", fig))

    def metric(self, label, value, **kwargs):
        self.items.append(("metric", (label, value)))

    def code(self, code, **kwargs):
        self.items.append(("code", code))

    def error(self, *args, **kwargs):
        self.items.append(("error", args[0] if args else ""))

    def warning(self, *args, **kwargs):
        self.items.append(("warning", args[0] if args else ""))

    def success(self, *args, **kwargs):
        self.items.append(("success", args[0] if args else ""))

    def text(self, *args, **kwargs):
        self.items.append(("text", args[0] if args else ""))

    def json(self, data, **kwargs):
        self.items.append(("json", data))


def render_items(st_module, items):
    """Replays recorded items with the REAL streamlit module -- called from app.py
    on every render of chat history, so display survives st.rerun()."""
    for kind, payload in items:
        try:
            if kind == "write":
                st_module.write(_arrow_safe(payload))
            elif kind == "dataframe":
                st_module.dataframe(_arrow_safe(payload), width="stretch")
            elif kind == "table":
                st_module.table(_arrow_safe(payload))
            elif kind == "plotly_chart":
                st_module.plotly_chart(payload, width="stretch")
            elif kind == "pyplot":
                st_module.pyplot(payload)
            elif kind == "metric":
                label, value = payload
                st_module.metric(label, value)
            elif kind == "code":
                st_module.code(payload, language="python")
            elif kind == "error":
                st_module.error(payload)
            elif kind == "warning":
                st_module.warning(payload)
            elif kind == "success":
                st_module.success(payload)
            elif kind == "text":
                st_module.text(payload)
            elif kind == "json":
                st_module.json(payload)
        except Exception as e:
            st_module.warning(f"(couldn't re-render a {kind} item: {e})")
