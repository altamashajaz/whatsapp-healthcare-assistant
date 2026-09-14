"""Small local smoke test for the LangGraph pipeline.

Run from the project root:
    python -m tests.test_pipeline
"""

from app.graph.graph import healthcare_graph


def run_case(text: str, thread_id: str):
    result = healthcare_graph.invoke(
        {"user_text": text},
        config={"configurable": {"thread_id": thread_id}}
    )

    print("\nUSER:", text)
    print("TIER:", result.get("triage_tier"))
    print("LANG:", result.get("detected_language"))
    print("RESPONSE:\n", result.get("response"))

    print("HEALTH RELATED:", result.get("health_related"))

    if result.get("consult_url"):
        print("DOCTOR CONSULT:", result["consult_url"])

    if result.get("product_search_url"):
        print("PHARMEASY SEARCH:", result["product_search_url"])
        print("PRODUCT LABEL:", result.get("product_label"))


if __name__ == "__main__":
    run_case(
        "What can I do for mild acidity?",
        "test-acidity"
    )

    run_case(
        "I have severe chest pain and difficulty breathing",
        "test-emergency"
    )