"""Fast business/safety checks that do not intentionally call OpenAI or WhatsApp.

Run locally inside the project venv:
    python -m tests.test_business_rules
"""
from app.safety_rules import Tier, pre_check, validate_output
from app.tools.pharmeasy_tools import (
    get_product_category_metadata,
    match_reviewed_product,
    match_reviewed_report_product,
    validate_product_category_for_context,
)


def main():
    # Emergency rules
    assert pre_check("I can't breathe").tier == Tier.URGENT
    assert pre_check("Mujhe Saas nai a raha hain").tier == Tier.URGENT
    assert pre_check("Mujhe saans nahi aa rahi").tier == Tier.URGENT
    assert pre_check("Mera dum ghut raha hai").tier == Tier.URGENT
    assert pre_check("I have mild fever").tier == Tier.LOW

    # High-precision deterministic product fallbacks
    assert match_reviewed_product("I have mild fever") is not None
    assert match_reviewed_product("I have a headache") is not None
    assert match_reviewed_product("report says low vitamin d") is not None
    assert match_reviewed_product("Vitamin D is low (17.7; normal 30-100)") is not None
    assert match_reviewed_product("Tell me about Coldplay songs") is None
    assert match_reviewed_product("I have stomach pain") is None

    # The new scalable category registry includes supportive eye care. Eye swelling
    # does not need a hard-coded phrase mapping: the structured product classifier
    # can choose this allowlisted category, and the app validates it here.
    eye = get_product_category_metadata("eye_care")
    assert eye is not None
    assert "eye" in eye["label"].lower()

    allowed, reason, meta = validate_product_category_for_context(
        "eye_care",
        "I have swelling in my eyes",
        "low",
        False,
    )
    assert allowed and reason is None and meta is not None

    # Eye red flags suppress the product CTA even if an LLM selected eye_care.
    allowed, reason, _ = validate_product_category_for_context(
        "eye_care",
        "My eye is swollen and I suddenly have blurred vision",
        "moderate",
        False,
    )
    assert not allowed and reason

    # Invalid/invented LLM categories cannot escape the app allowlist.
    allowed, _, _ = validate_product_category_for_context(
        "antibiotic_eye_drops",
        "My eyes are red",
        "low",
        False,
    )
    assert not allowed

    # High-risk contexts and emergencies never get automatic product promotion.
    allowed, _, _ = validate_product_category_for_context(
        "pain_fever",
        "I am pregnant and have a headache",
        "low",
        True,
    )
    assert not allowed
    allowed, _, _ = validate_product_category_for_context(
        "eye_care",
        "My eyes are red",
        "urgent",
        False,
    )
    assert not allowed

    # Output policy: neutral discovery is okay; direct treatment language is not.
    unsafe, _ = validate_output("You should take this medicine.")
    safe, _ = validate_output("You can explore OTC options on PharmEasy.")
    assert not unsafe
    assert safe

    # Hybrid triage merge: LLM may escalate, never downgrade deterministic rules.
    from app.graph.nodes import _merge_triage, route_after_triage, route_after_llm_triage
    assert _merge_triage("urgent", "low") == "urgent"
    assert _merge_triage("moderate", "low") == "moderate"
    assert _merge_triage("low", "urgent") == "urgent"
    assert _merge_triage("low", "self_harm") == "self_harm"
    assert route_after_triage({"triage_tier": "urgent"}) == "emergency"
    assert route_after_triage({"triage_tier": "low"}) == "llm_triage"
    assert route_after_llm_triage({"triage_tier": "low", "health_related": False}) == "non_health"
    assert route_after_llm_triage({"triage_tier": "low", "health_related": True}) == "prescription_gate"

    # Generated report prose may only match the narrow report categories.
    assert match_reviewed_report_product(
        "Eye swelling can have different causes. Seek care if pain or fever develops."
    ) is None
    vitamin_d = match_reviewed_report_product("Your Vitamin D is low at 17.7 ng/mL.")
    assert vitamin_d is not None
    assert vitamin_d["category_key"] == "vitamin_d"

    print("Business-rule checks passed.")


if __name__ == "__main__":
    main()


def test_simple_greeting_scope():
    from app.graph.nodes import _is_simple_greeting_or_intro

    assert _is_simple_greeting_or_intro("Hello")
    assert _is_simple_greeting_or_intro("Hello, My name is Altamash")
    assert _is_simple_greeting_or_intro("Thanks")
    assert not _is_simple_greeting_or_intro("Hi, I have fever")
    assert not _is_simple_greeting_or_intro("Mere bukhar h")


def test_roman_hindi_breathing_emergency_variants():
    assert pre_check("Mujhe Saas nai a raha hain").tier == Tier.URGENT
    assert pre_check("mujhe sans ni a rha").tier == Tier.URGENT
    assert pre_check("saans lene me takleef hai").tier == Tier.URGENT
    assert pre_check("mera dum ghut raha hai").tier == Tier.URGENT
