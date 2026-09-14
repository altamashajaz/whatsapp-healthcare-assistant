"""Small deterministic tools used by the LangGraph workflow.

These are deliberately conservative. They do not diagnose, prescribe doses,
or replace a clinician.
"""
from __future__ import annotations

from langchain_core.tools import tool


@tool
def explain_next_step(symptom_or_question: str) -> str:
    """Return a conservative instruction to seek appropriate medical advice.

    This tool exists as a controlled fallback rather than a medical knowledge
    database. The LLM must not turn it into a diagnosis or prescription.
    """
    return (
        "Provide general educational information, ask for missing context when needed, "
        "and recommend a doctor/pharmacist when diagnosis or treatment decisions are required."
    )
