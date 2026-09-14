from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    answer_with_llm,
    detect_language,
    deterministic_triage,
    emergency_response,
    finalize,
    llm_triage,
    non_health_response,
    product_discovery_classifier,
    prescription_gate,
    prescription_handoff,
    route_after_llm_triage,
    route_after_prescription_gate,
    route_after_triage,
    safety_validate,
)
from app.graph.state import HealthcareState


def build_graph():
    graph = StateGraph(HealthcareState)

    # Nodes
    graph.add_node("detect_language", detect_language)
    graph.add_node("triage", deterministic_triage)
    graph.add_node("llm_triage", llm_triage)
    graph.add_node("non_health", non_health_response)
    graph.add_node("prescription_gate", prescription_gate)
    graph.add_node("product_discovery", product_discovery_classifier)
    graph.add_node("emergency", emergency_response)
    graph.add_node("prescription_handoff", prescription_handoff)
    graph.add_node("answer", answer_with_llm)
    graph.add_node("safety_validate", safety_validate)
    graph.add_node("finalize", finalize)

    # Start
    graph.add_edge(START, "detect_language")
    graph.add_edge("detect_language", "triage")

    # Known deterministic emergencies short-circuit immediately.
    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "emergency": "emergency",
            "llm_triage": "llm_triage",
        },
    )

    # Structured LLM triage can escalate unseen phrasing but cannot downgrade
    # the deterministic result.
    graph.add_conditional_edges(
        "llm_triage",
        route_after_llm_triage,
        {
            "emergency": "emergency",
            "non_health": "non_health",
            "prescription_gate": "prescription_gate",
        },
    )

    # Prescription safety routing
    graph.add_conditional_edges(
        "prescription_gate",
        route_after_prescription_gate,
        {
            "emergency": "emergency",
            "prescription_handoff": "prescription_handoff",
            "product_discovery": "product_discovery",
        },
    )

    # Product discovery is a separate structured classification step.
    graph.add_edge("product_discovery", "answer")

    # Normal answer path
    graph.add_edge("answer", "safety_validate")
    graph.add_edge("safety_validate", "finalize")

    # Emergency, non-health and prescription handoff paths
    graph.add_edge("emergency", "finalize")
    graph.add_edge("non_health", "finalize")
    graph.add_edge("prescription_handoff", "finalize")

    # End
    graph.add_edge("finalize", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


healthcare_graph = build_graph()
