"""Fallback for questions that aren't EDA/predictive/causal/visualization -- e.g.
greetings, clarifying questions, or requests the router genuinely can't place. No
code is generated here; the LLM just replies in plain language."""


def build_prompt(question: str, dataset_summary: str):
    system = (
        "You are a helpful data-analysis assistant. Answer briefly and, if the "
        "user's question sounds like it should be an EDA, prediction, causal, or "
        "visualization request, suggest how to rephrase it so the system can route "
        "it correctly. Do not write code."
    )
    user = f"Dataset info:\n{dataset_summary}\n\nQuestion: {question}"
    return system, user
