from __future__ import annotations

from typing import Any

from langgraph.graph import MessagesState


class HealthcareState(MessagesState):
    """State for one conversational healthcare-assistant thread."""

    user_text: str
    report_text: str | None
    image_data_url: str | None
    ocr_confidence: float

    is_prescription: bool
    prescription_safe_to_interpret: bool
    prescription_reason: str | None

    detected_language: str
    language_name: str

    # Hybrid triage state. `triage_tier` is the final merged result.
    rule_triage_tier: str
    llm_triage_tier: str | None
    llm_triage_reason: str | None
    llm_triage_confidence: float
    triage_tier: str
    triage_rule: str | None

    health_related: bool
    intent: str
    response: str

    safety_passed: bool
    safety_reason: str | None

    product_discovery_allowed: bool
    product_category_key: str | None
    product_discovery_reason: str | None
    product_discovery_confidence: float
    product_discovery_source: str | None
    product_query: str | None
    product_search_url: str | None
    product_label: str | None

    consult_url: str | None
    offer_emergency_help: bool

    metadata: dict[str, Any]
