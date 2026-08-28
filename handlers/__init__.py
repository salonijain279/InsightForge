from . import eda, predictive, causal, visualization, general

HANDLERS = {
    "eda": eda,
    "predictive": predictive,
    "causal": causal,
    "visualization": visualization,
    "general": general,
}
