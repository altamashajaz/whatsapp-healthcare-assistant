from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage

from app import language, safety_rules
from app.config import settings
from app.graph.state import HealthcareState
from app.services.llm import SYSTEM_PROMPT, get_llm
from app.safety_rules import Tier

from app.tools.pharmeasy_tools import (
    match_reviewed_product,
    match_reviewed_report_product,
    build_search_url,
    get_product_category_metadata,
    product_category_prompt_lines,
    validate_product_category_for_context,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LANGUAGE
# ---------------------------------------------------------------------------

def detect_language(state: HealthcareState) -> HealthcareState:
    """
    Detect the language of the user's message or uploaded report.

    Safety rules do not depend on this detection. They run independently
    across supported languages.
    """
    text = (
        state.get("user_text", "")
        or state.get("report_text", "")
        or ""
    )

    lang = language.detect_language(text) if text else "en"

    return {
        "detected_language": lang,
        "language_name": language.language_name(lang),
    }


# ---------------------------------------------------------------------------
# HYBRID TRIAGE: DETERMINISTIC RULES + STRUCTURED LLM ESCALATION
# ---------------------------------------------------------------------------


class LLMTriageDecision(BaseModel):
    """Structured output returned by the LLM triage/scope classifier."""

    tier: Literal["low", "moderate", "urgent", "self_harm"]
    health_related: bool = Field(
        description="True only when the current request is genuinely health/medical/wellness related."
    )
    reason: str = Field(
        description="Short reason for the triage classification. Do not diagnose."
    )
    confidence: float = Field(ge=0.0, le=1.0)


class LLMProductDecision(BaseModel):
    """Structured product-discovery classification; never a medicine prescription."""

    eligible: bool
    category_key: str | None = Field(
        default=None,
        description="One application-approved product category key, or null when no category is suitable.",
    )
    reason: str = Field(
        description="Short reason for allowing or declining optional product discovery."
    )
    confidence: float = Field(ge=0.0, le=1.0)


_TRIAGE_SEVERITY = {
    Tier.LOW.value: 0,
    Tier.MODERATE.value: 1,
    Tier.URGENT.value: 2,
    # Self-harm is an emergency route with a dedicated crisis response.
    Tier.SELF_HARM.value: 3,
}


def _merge_triage(rule_tier: str, llm_tier: str) -> str:
    """Return the more severe triage result; the LLM can never downgrade rules."""
    if _TRIAGE_SEVERITY.get(llm_tier, -1) > _TRIAGE_SEVERITY.get(rule_tier, -1):
        return llm_tier
    return rule_tier


def deterministic_triage(state: HealthcareState) -> HealthcareState:
    """Run high-recall deterministic red-flag rules before any LLM triage.

    Known emergencies are short-circuited immediately. Non-emergency messages
    continue to the structured LLM triage classifier so unseen phrasing and
    multilingual variation can still be escalated.
    """
    text = (
        f"{state.get('user_text', '')} "
        f"{state.get('report_text') or ''}"
    ).strip()

    result = safety_rules.pre_check(text)

    return {
        "rule_triage_tier": result.tier.value,
        "triage_tier": result.tier.value,
        "triage_rule": result.matched_rule,
        "offer_emergency_help": result.tier in (Tier.URGENT, Tier.SELF_HARM),
    }


def route_after_triage(state: HealthcareState) -> str:
    """Known rule-based emergencies bypass the LLM triage call entirely."""
    if state.get("triage_tier") in {Tier.URGENT.value, Tier.SELF_HARM.value}:
        return "emergency"
    return "llm_triage"


def llm_triage(state: HealthcareState) -> HealthcareState:
    """Conservative structured triage for phrasing not caught by hard rules.

    This classifier is allowed to ESCALATE a message but never downgrade the
    deterministic result. It does not diagnose and does not generate user-facing
    medical advice. If the classifier fails, the deterministic result is kept.
    """
    rule_tier = state.get("rule_triage_tier") or state.get("triage_tier") or Tier.LOW.value

    # Defense in depth: rule emergencies should already have been routed away.
    if rule_tier in {Tier.URGENT.value, Tier.SELF_HARM.value}:
        return {
            "triage_tier": rule_tier,
            "offer_emergency_help": True,
        }

    user_text = (state.get("user_text") or "").strip()
    report_text = (state.get("report_text") or "").strip()

    # If the user uploaded a report without text, give the classifier a concise
    # report excerpt. For ordinary chats, the CURRENT user wording is primary.
    classifier_input = user_text
    if report_text:
        classifier_input += "\n\nOptional report context:\n" + report_text[:5000]

    if not classifier_input.strip():
        return {
            "llm_triage_tier": rule_tier,
            "llm_triage_reason": "No text available for LLM triage.",
            "llm_triage_confidence": 0.0,
            "triage_tier": rule_tier,
        }

    triage_prompt = f"""
You are a conservative medical SAFETY TRIAGE CLASSIFIER for a WhatsApp health assistant.
Your only job is to classify urgency. Do not diagnose. Do not give treatment advice.

Classify the CURRENT user message into exactly one tier:
- low: mild/self-care type concern with no obvious red flags
- moderate: should receive clinician review but is not an obvious emergency
- urgent: possible immediate medical emergency requiring urgent in-person/emergency care
- self_harm: suicidal intent, self-harm intent, or an acute mental-health safety crisis

URGENT includes, but is not limited to: severe breathing difficulty or inability to get air,
concerning chest pain, stroke-like symptoms, unconsciousness/fainting with danger signs,
severe bleeding, seizure, severe allergic reaction/anaphylaxis, poisoning/overdose, blue lips,
or other symptoms that could represent an immediate threat to life.

Important rules:
1. Understand spelling mistakes, slang, Hinglish/Roman Hindi, and other supported Indian-language phrasing.
2. When the wording plausibly describes inability to breathe or another major red flag, choose urgent.
3. If uncertain between moderate and urgent where delay could be dangerous, choose urgent.
4. Set health_related=false for greetings, introductions, thanks, weather, sports, politics, entertainment, coding, finance, or other clearly non-health requests.
5. Set health_related=true for symptoms, medicines, reports/prescriptions, diet/nutrition, wellness, healthcare services, or medical questions.
6. Return only the requested structured fields.

Deterministic rule tier already assigned: {rule_tier}
Current message and context:
{classifier_input}
"""

    try:
        classifier = get_llm().with_structured_output(LLMTriageDecision)
        decision = classifier.invoke(triage_prompt)
        llm_tier = decision.tier
        final_tier = _merge_triage(rule_tier, llm_tier)

        logger.info(
            "Hybrid triage: rule=%s llm=%s final=%s health_related=%s confidence=%.2f reason=%s",
            rule_tier,
            llm_tier,
            final_tier,
            decision.health_related,
            decision.confidence,
            decision.reason,
        )

        return {
            "llm_triage_tier": llm_tier,
            "llm_triage_reason": decision.reason,
            "llm_triage_confidence": decision.confidence,
            "triage_tier": final_tier,
            "health_related": bool(decision.health_related),
            "offer_emergency_help": final_tier in {Tier.URGENT.value, Tier.SELF_HARM.value},
        }
    except Exception as exc:  # deterministic safety remains available if the LLM fails
        logger.exception("LLM triage classifier failed; keeping rule tier=%s: %s", rule_tier, exc)
        return {
            "llm_triage_tier": None,
            "llm_triage_reason": "LLM triage unavailable; deterministic result retained.",
            "llm_triage_confidence": 0.0,
            "triage_tier": rule_tier,
            "health_related": True,
            "offer_emergency_help": rule_tier in {Tier.URGENT.value, Tier.SELF_HARM.value},
        }


def route_after_llm_triage(state: HealthcareState) -> str:
    """Route after hybrid merge; emergency and non-health requests short-circuit."""
    if state.get("triage_tier") in {Tier.URGENT.value, Tier.SELF_HARM.value}:
        return "emergency"
    if state.get("health_related") is False:
        return "non_health"
    return "prescription_gate"


# ---------------------------------------------------------------------------
# PRESCRIPTION SAFETY GATE
# ---------------------------------------------------------------------------

def prescription_gate(state: HealthcareState) -> HealthcareState:
    """
    Conservative gate for uploaded prescription images.

    A prescription is considered safe to interpret only when:
    - an image was uploaded,
    - OCR found prescription-like content,
    - OCR confidence is reasonably high.

    This is an engineering safety heuristic, not a clinical validation threshold.
    """

    image_data_url = state.get("image_data_url")
    report_text = state.get("report_text") or ""
    ocr_confidence = state.get("ocr_confidence", 0.0)

    # No uploaded image.
    if not image_data_url:
        return {
            "is_prescription": False,
            "prescription_safe_to_interpret": True,
            "prescription_reason": None,
        }

    # First-pass prescription detection from OCR text.
    prescription_keywords = [
        "prescription",
        "rx",
        "tablet",
        "tab",
        "capsule",
        "cap",
        "syrup",
        "mg",
        "ml",
        "dose",
        "dosage",
        "medicine",
        "medication",
    ]

    text_lower = report_text.lower()

    is_prescription = any(
        keyword in text_lower
        for keyword in prescription_keywords
    )

    if not is_prescription:
        return {
            "is_prescription": False,
            "prescription_safe_to_interpret": True,
            "prescription_reason": None,
        }

    # Conservative OCR readability threshold.
    if ocr_confidence < 70:
        return {
            "is_prescription": True,
            "prescription_safe_to_interpret": False,
            "prescription_reason": (
                "Prescription detected but OCR confidence is too low "
                f"({ocr_confidence:.1f}%)."
            ),
        }

    return {
        "is_prescription": True,
        "prescription_safe_to_interpret": True,
        "prescription_reason": (
            "Prescription detected with OCR confidence "
            f"{ocr_confidence:.1f}%."
        ),
    }


def route_after_prescription_gate(state: HealthcareState) -> str:
    """
    Route prescription images after deterministic safety checks.
    """

    # Emergency always wins.
    if state.get("triage_tier") in {
        Tier.URGENT.value,
        Tier.SELF_HARM.value,
    }:
        return "emergency"

    # Unreadable prescription → safe handoff.
    if (
        state.get("is_prescription")
        and not state.get("prescription_safe_to_interpret")
    ):
        return "prescription_handoff"

    return "product_discovery"



# ---------------------------------------------------------------------------
# LLM-ASSISTED PRODUCT CATEGORY DISCOVERY
# ---------------------------------------------------------------------------

_PRODUCT_MIN_CONFIDENCE = 0.55


def _clear_product_discovery(reason: str, source: str = "none") -> HealthcareState:
    return {
        "product_discovery_allowed": False,
        "product_category_key": None,
        "product_discovery_reason": reason,
        "product_discovery_confidence": 0.0,
        "product_discovery_source": source,
        "product_query": None,
        "product_search_url": None,
        "product_label": None,
    }


def product_discovery_classifier(state: HealthcareState) -> HealthcareState:
    """Choose an optional PharmEasy product category without prescribing.

    Strategy:
    1. Hard safety gates always win.
    2. Known high-precision symptom/report phrases use deterministic mapping.
    3. Unseen wording is mapped by a structured LLM into an app-owned allowlist.
    4. The app validates the chosen category and builds the URL itself.

    The LLM never returns a medicine name, dose, frequency or SKU.
    """
    triage_tier = state.get("triage_tier", Tier.LOW.value)
    user_text = (state.get("user_text") or "").strip()
    report_text = (state.get("report_text") or "").strip()

    if state.get("health_related") is False:
        return _clear_product_discovery("Current request is not health-related.", "scope")

    if triage_tier in {Tier.URGENT.value, Tier.SELF_HARM.value}:
        return _clear_product_discovery("Emergency/self-harm flow: commerce disabled.", "safety")

    # Prescriptions have a separate, safer flow that searches only medicines
    # visibly present in the prescription. Do not add a symptom-category CTA.
    if state.get("is_prescription"):
        return _clear_product_discovery("Prescription uses prescription-specific medicine search.", "prescription")

    high_risk_context = safety_rules.has_high_risk_medicine_context(user_text)
    if high_risk_context:
        return _clear_product_discovery("High-risk medicine context: automatic OTC discovery disabled.", "safety")

    # Fast path for reviewed phrases we already know with high precision. Broad
    # symptom matching is restricted to the user's actual message; raw report text
    # may contain incidental symptom words in test names/notes. Reports use the
    # narrow report matcher instead.
    deterministic = match_reviewed_product(user_text) if user_text else None
    if not deterministic and report_text:
        deterministic = match_reviewed_report_product(report_text)
    if deterministic:
        allowed, block_reason, metadata = validate_product_category_for_context(
            deterministic.get("category_key"),
            user_text,
            triage_tier,
            high_risk_context,
        )
        if allowed and metadata:
            logger.info(
                "Product category selected deterministically: key=%s query=%s",
                metadata["category_key"],
                metadata["search_term"],
            )
            return {
                "product_discovery_allowed": True,
                "product_category_key": metadata["category_key"],
                "product_discovery_reason": "Reviewed deterministic symptom/report mapping.",
                "product_discovery_confidence": 1.0,
                "product_discovery_source": "deterministic",
                "product_query": metadata["search_term"],
                "product_search_url": metadata["pharmeasy_url"],
                "product_label": metadata["label"],
            }
        if block_reason:
            logger.info("Deterministic product candidate blocked: %s", block_reason)
            return _clear_product_discovery(block_reason, "safety")

    classifier_input = user_text
    if report_text:
        classifier_input += "\n\nReport/extracted text:\n" + report_text[:7000]

    if not classifier_input.strip():
        return _clear_product_discovery("No symptom/report text available for product classification.")

    prompt = f"""
You are a conservative PRODUCT-DISCOVERY CLASSIFIER for a PharmEasy healthcare assistant.
You are NOT prescribing treatment. Your task is only to decide whether an optional,
non-prescription/supportive PharmEasy product category is relevant enough to browse.

Allowed category keys:
{product_category_prompt_lines()}

Rules:
1. Return eligible=true only when one allowed category clearly fits the CURRENT symptom or a clear report finding.
2. Return exactly one category_key from the allowlist. Never invent a medicine, brand, SKU, strength, dose, frequency, or duration.
3. Product discovery is optional browsing, not a statement that the product is right for the individual.
4. If the symptom is vague, severe, potentially dangerous, or needs examination before self-care, return eligible=false.
5. For eye redness/dryness/itching/mild puffiness or swelling WITHOUT severe eye pain, vision change, injury, chemical exposure, or contact-lens red flags, eye_care is appropriate for basic eye-care/lubricating options.
6. Never map eye symptoms to pain_fever merely because pain/fever might be mentioned as a possible red flag.
7. For reports, choose vitamin_d/vitamin_b12/iron_nutrition only when the report clearly indicates that finding is low/deficient or below its stated reference range. Do not infer deficiency from a normal value. If the user supplied only a report and no symptom/question, do not choose symptom categories from incidental report wording.
8. If the user's context includes pregnancy, a child, breastfeeding, major kidney/liver disease, medicine allergy, or anticoagulants, return eligible=false; clinician/pharmacist review comes first.
9. This message has already passed emergency triage as: {triage_tier}. Still be conservative about category suitability.

Current health context:
{classifier_input}
"""

    try:
        classifier = get_llm().with_structured_output(LLMProductDecision)
        decision = classifier.invoke(prompt)

        if not decision.eligible or not decision.category_key:
            logger.info(
                "Product classifier declined discovery: confidence=%.2f reason=%s",
                decision.confidence,
                decision.reason,
            )
            return _clear_product_discovery(decision.reason, "llm")

        if decision.confidence < _PRODUCT_MIN_CONFIDENCE:
            return _clear_product_discovery(
                f"Product classifier confidence too low ({decision.confidence:.2f}).",
                "llm",
            )

        allowed, block_reason, metadata = validate_product_category_for_context(
            decision.category_key,
            user_text,
            triage_tier,
            high_risk_context,
        )
        if not allowed or not metadata:
            logger.info(
                "LLM product candidate rejected by app validation: key=%s reason=%s",
                decision.category_key,
                block_reason,
            )
            return _clear_product_discovery(block_reason or "Category rejected by app validation.", "safety")

        logger.info(
            "LLM product category approved: key=%s confidence=%.2f reason=%s",
            metadata["category_key"],
            decision.confidence,
            decision.reason,
        )
        return {
            "product_discovery_allowed": True,
            "product_category_key": metadata["category_key"],
            "product_discovery_reason": decision.reason,
            "product_discovery_confidence": decision.confidence,
            "product_discovery_source": "llm",
            "product_query": metadata["search_term"],
            "product_search_url": metadata["pharmeasy_url"],
            "product_label": metadata["label"],
        }
    except Exception as exc:
        logger.exception("Product category classifier failed; no speculative product CTA: %s", exc)
        return _clear_product_discovery("Product classifier unavailable; no speculative CTA shown.", "error")


# ---------------------------------------------------------------------------
# PRESCRIPTION HANDOFF
# ---------------------------------------------------------------------------

def prescription_handoff(state: HealthcareState) -> HealthcareState:
    """
    Safe response when a prescription cannot be read reliably.
    """

    lang = state.get("detected_language", "en")

    messages = {
        "en": (
            "I couldn't read this prescription clearly enough to interpret it safely. "
            "Please upload a clearer, well-lit photo of the full prescription, "
            "or ask a pharmacist or doctor to confirm the medicines and dosage."
        ),

        "hi": (
            "मैं इस प्रिस्क्रिप्शन को सुरक्षित रूप से समझने के लिए इसे पर्याप्त "
            "स्पष्टता से पढ़ नहीं पा रहा हूँ। कृपया पूरी प्रिस्क्रिप्शन की एक साफ़ "
            "और अच्छी रोशनी वाली फोटो भेजें, या दवा और उसकी खुराक के लिए डॉक्टर "
            "या फार्मासिस्ट से पुष्टि करें।"
        ),

        "te": (
            "ఈ ప్రిస్క్రిప్షన్‌ను సురక్షితంగా అర్థం చేసుకునేంత స్పష్టంగా "
            "చదవలేకపోతున్నాను. దయచేసి పూర్తి ప్రిస్క్రిప్షన్‌కు స్పష్టమైన "
            "మరియు మంచి వెలుతురు ఉన్న ఫోటోను పంపండి లేదా మందులు మరియు మోతాదును "
            "డాక్టర్ లేదా ఫార్మసిస్ట్‌తో నిర్ధారించుకోండి."
        ),

        "kn": (
            "ಈ ಪ್ರಿಸ್ಕ್ರಿಪ್ಷನ್ ಅನ್ನು ಸುರಕ್ಷಿತವಾಗಿ ಅರ್ಥಮಾಡಿಕೊಳ್ಳಲು ಸಾಕಷ್ಟು "
            "ಸ್ಪಷ್ಟವಾಗಿ ಓದಲು ಸಾಧ್ಯವಾಗುತ್ತಿಲ್ಲ. ದಯವಿಟ್ಟು ಸಂಪೂರ್ಣ ಪ್ರಿಸ್ಕ್ರಿಪ್ಷನ್‌ನ "
            "ಸ್ಪಷ್ಟವಾದ, ಚೆನ್ನಾಗಿ ಬೆಳಗಿದ ಫೋಟೋವನ್ನು ಕಳುಹಿಸಿ ಅಥವಾ ಔಷಧಿ ಮತ್ತು "
            "ಡೋಸ್ ಅನ್ನು ವೈದ್ಯರು ಅಥವಾ ಫಾರ್ಮಸಿಸ್ಟ್ ಬಳಿ ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ."
        ),

        "ta": (
            "இந்த மருந்துச் சீட்டை பாதுகாப்பாக புரிந்துகொள்ளும் அளவுக்கு "
            "தெளிவாகப் படிக்க முடியவில்லை. தயவுசெய்து முழு மருந்துச் சீட்டின் "
            "தெளிவான, நல்ல வெளிச்சமுள்ள புகைப்படத்தை அனுப்புங்கள் அல்லது "
            "மருந்துகள் மற்றும் அளவை மருத்துவர் அல்லது மருந்தாளரிடம் உறுதிப்படுத்துங்கள்."
        ),

        "ml": (
            "ഈ പ്രിസ്ക്രിപ്ഷൻ സുരക്ഷിതമായി മനസ്സിലാക്കാൻ കഴിയുന്നത്ര വ്യക്തമായി "
            "വായിക്കാൻ കഴിയുന്നില്ല. ദയവായി മുഴുവൻ പ്രിസ്ക്രിപ്ഷന്റെയും വ്യക്തമായ, "
            "നല്ല വെളിച്ചമുള്ള ഫോട്ടോ അയയ്ക്കുക, അല്ലെങ്കിൽ മരുന്നുകളും ഡോസും "
            "ഡോക്ടറുമായോ ഫാർമസിസ്റ്റുമായോ സ്ഥിരീകരിക്കുക."
        ),
    }

    reply = messages.get(lang, messages["en"])

    return {
        "response": reply,
        "messages": [AIMessage(content=reply)],
        "safety_passed": True,
        "health_related": True,
        "consult_url": settings.PHARMEASY_DOCTOR_URL,
    }


# ---------------------------------------------------------------------------
# EMERGENCY RESPONSE
# ---------------------------------------------------------------------------

def emergency_response(state: HealthcareState) -> HealthcareState:
    """
    Deterministic emergency/self-harm response.

    GPT is completely bypassed for these cases.
    """

    tier = state.get("triage_tier")
    lang = state.get("detected_language", "en")

    if tier == Tier.SELF_HARM.value:
        reply = safety_rules.CRISIS_MESSAGE.get(
            lang,
            safety_rules.CRISIS_MESSAGE["en"],
        )
    else:
        reply = safety_rules.EMERGENCY_MESSAGE.get(
            lang,
            safety_rules.EMERGENCY_MESSAGE["en"],
        )

    return {
        "response": reply,
        "messages": [AIMessage(content=reply)],
        "safety_passed": True,
        "health_related": True,
        "product_search_url": None,
        "product_label": None,
        "consult_url": None,
    }


# ---------------------------------------------------------------------------
# MAIN LLM ANSWER
# ---------------------------------------------------------------------------

def _extract_medicine_trailer(text: str) -> tuple[str, list[str]]:
    """
    Parse and strip the "MEDICINES: name1 | name2" trailer line the model
    is instructed to add when explaining a readable prescription. Returns
    (cleaned_response_text, [medicine_names]). Deliberately tolerant of the
    model getting the exact format slightly wrong - if no trailer line is
    found, just returns the text unchanged with an empty medicine list
    rather than erroring.
    """
    lines = text.strip().splitlines()
    names: list[str] = []
    kept_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("MEDICINES:"):
            payload = stripped.split(":", 1)[1].strip()
            if payload and payload.lower() != "none":
                names = [n.strip() for n in payload.split("|") if n.strip()]
            continue  # never show this raw line to the user
        kept_lines.append(line)

    return "\n".join(kept_lines).strip(), names


def _extract_scope_trailer(text: str) -> tuple[str, bool | None]:
    """Strip the model's internal SCOPE marker wherever it appears."""
    raw = text or ""
    health_related: bool | None = None

    if re.search(r"SCOPE\s*:\s*NON[_ -]?HEALTH", raw, flags=re.IGNORECASE):
        health_related = False
    elif re.search(r"SCOPE\s*:\s*HEALTH", raw, flags=re.IGNORECASE):
        health_related = True

    cleaned = re.sub(
        r"(?:^|\s*)SCOPE\s*:\s*(?:NON[_ -]?HEALTH|HEALTH)\s*",
        " ",
        raw,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip(), health_related


def _normalize_whatsapp_formatting(text: str) -> str:
    """Convert common Markdown into WhatsApp-friendly formatting."""
    text = text or ""
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text, flags=re.DOTALL)
    text = re.sub(r"__(.+?)__", r"*\1*", text, flags=re.DOTALL)
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)
    return text.strip()


def _is_simple_greeting_or_intro(text: str) -> bool:
    """Return True only for standalone greetings/intros/thanks.

    Keep this intentionally narrow so messages such as "Hi, I have fever"
    still enter the healthcare flow.
    """
    text = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not text:
        return False

    patterns = (
        r"(?:hi|hello|hey|namaste|good morning|good afternoon|good evening)[!., ]*",
        r"(?:thanks|thank you|thankyou)[!., ]*",
        r"(?:hi|hello|hey)[!., ]+(?:my name is|i am|i'm)\s+[a-z][a-z .'-]{0,60}[!., ]*",
        r"(?:namaste)[!., ]+(?:mera naam|main)\s+.+",
    )
    return any(re.fullmatch(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _generic_greeting_response(language_name: str) -> str:
    """Generic greeting; never assumes the current user's name."""
    lang = (language_name or "English").lower()

    if "hindi" in lang or "hinglish" in lang:
        return (
            "Namaste! Main aapka PharmEasy health assistant hoon.\n\n"
            "Main symptoms, medicines, lab reports, diet, nutrition aur doctor consultation "
            "se jude sawaalon mein madad kar sakta hoon.\n\n"
            "Aaj main aapki health se judi kis baat mein madad kar sakta hoon?"
        )

    return (
        "Hi! I'm your PharmEasy health assistant.\n\n"
        "I can help with symptoms, medicines, lab reports, diet, nutrition, "
        "general health questions, and doctor consultation.\n\n"
        "How can I help with your health today?"
    )


def _fallback_health_scope(user_text: str, report_text: str | None, image_data_url: str | None) -> bool:
    """Conservative fallback only when the model omitted the SCOPE trailer."""
    if report_text or image_data_url:
        return True

    import re

    text = (user_text or "").lower()
    patterns = (
        r"\bhealth\b", r"\bpain\b", r"\bache\b", r"\bfever\b", r"\bcough\b",
        r"\bcold\b", r"\bheadache\b", r"\bstomach\b", r"\bmedicine\b",
        r"\btablet\b", r"\bdrug\b", r"\bdoctor\b", r"\bhospital\b",
        r"\breport\b", r"\bblood\b", r"\bvitamin\b", r"\bdiet\b",
        r"\bnutrition\b", r"\bweight\b", r"\bsleep\b", r"\bacidity\b",
        r"\bdiarrh?oe?a\b", r"\bconstipat\w*\b", r"\ballergy\b", r"\brash\b",
        r"\bvomit\w*\b", r"\bnausea\b", r"\bdizziness\b", r"\bweakness\b",
        r"\btired\w*\b", r"\binfection\b", r"\bprescription\b", r"\bthyroid\b",
        r"\bcholesterol\b", r"\bkidney\b", r"\bliver\b", r"\bheart\b",
        r"\blung\b", r"\bbreath\w*\b", r"\bpregnan\w*\b", r"\bperiod\b",
        r"\bhair fall\b", r"\bhairfall\b", r"\bskin\b", r"\beye\b",
        r"\beyes\b", r"\bswelling\b", r"\bswollen\b", r"\bdard\b",
        r"\bbukhar\b", r"\bkhansi\b", r"\bpet\b", r"\bpait\b",
        r"\bdawai\b", r"\bsehat\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def non_health_response(state: HealthcareState) -> HealthcareState:
    """Deterministic out-of-scope response with zero commercial CTAs."""
    language_name = state.get("language_name", "English")
    reply = _generic_greeting_response(language_name)
    return {
        "response": reply,
        "messages": [AIMessage(content=reply)],
        "health_related": False,
        "product_discovery_allowed": False,
        "product_category_key": None,
        "product_query": None,
        "product_search_url": None,
        "product_label": None,
        "consult_url": None,
    }


def answer_with_llm(state: HealthcareState) -> HealthcareState:
    """Generate the user-facing health response after safety/product classification.

    Triage and product-category decisions have already happened before this node.
    The answer model explains the health issue naturally but cannot invent a
    commercial category. The app-owned product URL/label in state is authoritative.
    """
    llm = get_llm()

    user_text = state.get("user_text", "") or "Please explain the uploaded report in simple terms."
    report_text = state.get("report_text")
    image_data_url = state.get("image_data_url")
    language_name = state.get("language_name", "English")
    triage_tier = state.get("triage_tier", Tier.LOW.value)

    # Defense in depth if the scope classifier was unavailable.
    if (
        _is_simple_greeting_or_intro(state.get("user_text", "") or "")
        and not report_text
        and not image_data_url
    ):
        logger.info("Greeting/introduction detected; skipping health-commerce CTAs.")
        return non_health_response(state)

    context = ""
    if report_text:
        context = "\nExtracted document text:\n" + report_text[:12000] + "\n"

    is_prescription = state.get("is_prescription", False)
    prescription_readable = state.get("prescription_safe_to_interpret", True)
    explain_prescription_medicines = is_prescription and prescription_readable

    prescription_instructions = ""
    if explain_prescription_medicines:
        prescription_instructions = (
            "\n\nThis is a readable prescription. For each medicine you can clearly "
            "identify, briefly explain what that type of medicine is generally used for. "
            "If you mention dosage/frequency, reflect only what is visibly written; never "
            "invent or change it. If a medicine name is unclear, say so rather than guessing.\n"
            "At the end, output a separate hidden line exactly as:\n"
            "MEDICINES: name1 | name2 | name3\n"
            "If no medicine name is clear, output: MEDICINES: none"
        )

    approved_product_label = state.get("product_label")
    approved_product_key = state.get("product_category_key")
    product_context = ""
    if state.get("product_discovery_allowed") and approved_product_label:
        product_context = (
            "\n\nThe application has independently approved an OPTIONAL PharmEasy browsing "
            f"category for this request: {approved_product_label} (internal key: {approved_product_key}). "
            "You may mention this softly as an optional product-discovery choice. Do not say it is "
            "the correct treatment, do not name an unapproved medicine/brand, and do not provide "
            "dose/frequency/duration. The application sends the actual CTA separately."
        )
    else:
        product_context = (
            "\n\nThe application has NOT approved a product-discovery category for this request. "
            "Do not invent or recommend a PharmEasy product/medicine in the response."
        )

    prompt = (
        f"Detected response language: {language_name}.\n"
        "Use the CURRENT user message to decide response language; previous turns may provide "
        "context but must not override the current language.\n"
        f"Triage tier: {triage_tier}.\n"
        f"User message: {user_text}\n"
        f"{context}"
        f"{prescription_instructions}"
        f"{product_context}\n\n"
        "Give useful health information rather than only saying 'consult a doctor'. Start with the "
        "user's actual concern. Explain likely context cautiously without diagnosing. Give practical "
        "self-care when appropriate, simple diet/food/hydration ideas when relevant, and specific "
        "red flags/next steps. For uploaded reports, summarize important findings and explain clear "
        "high/low values using the report's own ranges when available. Do not fabricate ranges.\n\n"
        "PharmEasy Doctor Consult is available 24/7 starting at ₹199. The application adds a separate "
        "consultation CTA to normal health conversations, so do not print its URL. You may naturally "
        "mention it as an option for personalised guidance, but never as a substitute for emergency care.\n\n"
        "Never say the user should take/use/start a medicine or supplement based only on this chat. "
        "Never invent dose, strength, frequency or duration. For eye symptoms, never suggest antibiotic "
        "or steroid drops unless they are clearly present in an uploaded prescription; basic eye-care "
        "product discovery means supportive/lubricating options only.\n\n"
        "For every genuine health response, end with the hidden line SCOPE: HEALTH. Ask at most one "
        "short follow-up question when it materially improves safety. For WhatsApp, use plain text or "
        "single-asterisk emphasis like *Vitamin D*; never use Markdown double asterisks."
    )

    history = state.get("messages", [])
    current_content = [{"type": "text", "text": prompt}]
    if image_data_url:
        current_content.append({"type": "image_url", "image_url": {"url": image_data_url}})

    base_messages = [
        ("system", SYSTEM_PROMPT),
        *history[:-1],
        ("human", current_content),
    ]

    final = llm.invoke(base_messages)
    response = final.content if isinstance(final.content, str) else str(final.content)

    # Report-only fallback: if the pre-answer product classifier could not infer a
    # numeric lab finding, the generated explanation may make a reviewed low-value
    # finding explicit. Only the narrow report matcher is allowed over generated prose.
    product_url = state.get("product_search_url")
    product_label = state.get("product_label")
    product_query = state.get("product_query")
    product_category_key = state.get("product_category_key")

    if report_text and not product_url:
        interpreted_product = match_reviewed_report_product(response)
        if interpreted_product:
            high_risk_context = safety_rules.has_high_risk_medicine_context(
                state.get("user_text", "") or ""
            )
            allowed, block_reason, metadata = validate_product_category_for_context(
                interpreted_product.get("category_key"),
                state.get("user_text", "") or "",
                triage_tier,
                high_risk_context,
            )
            if allowed and metadata:
                product_category_key = metadata["category_key"]
                product_url = metadata["pharmeasy_url"]
                product_label = metadata["label"]
                product_query = metadata["search_term"]
                logger.info(
                    "Reviewed report-product fallback approved: key=%s query=%s",
                    product_category_key,
                    product_query,
                )
            elif block_reason:
                logger.info("Report-product fallback blocked: %s", block_reason)

    # Strip internal scope signal before user-visible output.
    response, model_health_related = _extract_scope_trailer(response)
    health_related = state.get("health_related", True)
    if model_health_related is False:
        health_related = False
    elif model_health_related is None and state.get("health_related") is None:
        health_related = _fallback_health_scope(user_text, report_text, image_data_url)

    if not health_related:
        logger.info("Answer model/fallback identified non-health request; clearing all commerce CTAs.")
        return non_health_response(state)

    # Prescription-commerce behavior: search only names visibly extracted from
    # the prescription, never names invented as general advice.
    medicine_links: list[dict] = []
    if explain_prescription_medicines:
        response, medicine_names = _extract_medicine_trailer(response)
        for name in medicine_names[:8]:
            medicine_links.append({"name": name, "url": build_search_url(name)})

        if medicine_links:
            links_block = "\n".join(f"- {m['name']}: {m['url']}" for m in medicine_links)
            response = f"{response}\n\nYou can search these prescribed medicines on PharmEasy:\n{links_block}"

    if product_url:
        logger.info(
            "Approved PharmEasy product-discovery CTA: key=%s query=%s source=%s confidence=%.2f",
            product_category_key,
            product_query,
            state.get("product_discovery_source", "report_fallback"),
            float(state.get("product_discovery_confidence", 0.0) or 0.0),
        )

    consult_url = None
    if triage_tier not in {Tier.URGENT.value, Tier.SELF_HARM.value}:
        consult_url = settings.PHARMEASY_DOCTOR_URL
        logger.info("Attached PharmEasy Doctor Consult CTA (24/7, starting ₹199).")

    return {
        "messages": [final],
        "response": response,
        "health_related": True,
        "product_category_key": product_category_key,
        "product_query": product_query,
        "product_search_url": product_url,
        "product_label": product_label,
        "consult_url": consult_url,
    }


# ---------------------------------------------------------------------------
# OUTPUT SAFETY VALIDATION
# ---------------------------------------------------------------------------

def safety_validate(state: HealthcareState) -> HealthcareState:
    """
    Validate the generated response before it reaches the user.
    """

    response = state.get(
        "response",
        "",
    )

    safe, reason = safety_rules.validate_output(
        response
    )

    if not safe:

        logger.warning(
            "LLM response blocked: %s",
            reason,
        )

        response = (
            "I want to avoid giving you unsafe medical advice. "
            "Please speak with a doctor or pharmacist for guidance "
            "on this question."
        )

    return {
        "response": response,
        "safety_passed": safe,
        "safety_reason": reason,
    }


# ---------------------------------------------------------------------------
# FINALIZE
# ---------------------------------------------------------------------------

def finalize(state: HealthcareState) -> HealthcareState:
    """
    Add the appropriate healthcare disclaimer to non-emergency responses.
    """

    lang = state.get(
        "detected_language",
        "en",
    )

    response = state.get(
        "response",
        "",
    )

    tier = state.get(
        "triage_tier",
        Tier.LOW.value,
    )

    # Emergency/crisis messages already contain their own safety language.
    # Non-health requests should not receive a medical disclaimer.
    if (
        state.get("health_related", True)
        and tier not in {Tier.URGENT.value, Tier.SELF_HARM.value}
    ):
        response = safety_rules.apply_disclaimer(response, lang)

    response = _normalize_whatsapp_formatting(response)

    return {
        "response": response,
    }