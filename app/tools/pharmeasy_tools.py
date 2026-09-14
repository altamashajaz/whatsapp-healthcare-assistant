"""PharmEasy product-discovery helpers with a strict medical-safety boundary.

The application promotes *category-level discovery*, not personalised treatment.
The LLM may classify a user's need into one approved category, but the app owns
and validates the allowlist, safety blockers, labels and search URLs.
"""
from __future__ import annotations

import json
import re
from urllib.parse import quote_plus

from langchain_core.tools import tool

from app.config import settings


PHARMEASY_SEARCH_URL = settings.PHARMEASY_SEARCH_URL


# ---------------------------------------------------------------------------
# APPROVED PRODUCT CATEGORY REGISTRY
# ---------------------------------------------------------------------------
# The LLM is never allowed to invent a medicine/SKU. It can only choose one of
# these category keys. Search terms intentionally stay broad and non-prescriptive.
PRODUCT_CATEGORY_REGISTRY: dict[str, dict[str, str]] = {
    "pain_fever": {
        "search_term": "fever and pain relief",
        "label": "fever and pain-relief options",
        "description": "mild fever, headache, or general body aches without red flags",
    },
    "cold_cough": {
        "search_term": "cold and cough care",
        "label": "cold and cough-care options",
        "description": "mild cold, cough, runny nose, or congestion without breathing red flags",
    },
    "sore_throat": {
        "search_term": "throat lozenges",
        "label": "sore-throat and lozenge options",
        "description": "mild sore throat or throat irritation",
    },
    "nasal_care": {
        "search_term": "saline nasal spray",
        "label": "saline and nasal-care options",
        "description": "mild nasal dryness or congestion where supportive nasal care is reasonable",
    },
    "eye_care": {
        "search_term": "eye care lubricating eye drops",
        "label": "eye-care and lubricating-eye-drop options",
        "description": "mild red, dry, irritated, itchy, or puffy/swollen eyes without eye red flags",
    },
    "acidity_digestive": {
        "search_term": "acidity and digestive care",
        "label": "acidity and digestive-care options",
        "description": "mild acidity, reflux, heartburn, or indigestion without abdominal red flags",
    },
    "hydration": {
        "search_term": "ors and electrolytes",
        "label": "ORS and electrolyte options",
        "description": "mild diarrhea, loose motions, dehydration, or fluid replacement needs",
    },
    "constipation": {
        "search_term": "constipation and digestive care",
        "label": "constipation and digestive-care options",
        "description": "mild uncomplicated constipation",
    },
    "allergy_care": {
        "search_term": "allergy care",
        "label": "allergy-care options",
        "description": "mild seasonal/allergic symptoms without facial swelling or breathing problems",
    },
    "first_aid": {
        "search_term": "first aid",
        "label": "first-aid products",
        "description": "small superficial cuts, scrapes, or minor wounds",
    },
    "skin_care": {
        "search_term": "skin care soothing lotion",
        "label": "basic skin-care and soothing options",
        "description": "mild dry, itchy, or irritated skin without severe rash red flags",
    },
    "oral_care": {
        "search_term": "oral care mouth ulcer",
        "label": "oral-care options",
        "description": "minor mouth ulcers or basic oral-care needs without severe swelling/infection signs",
    },
    "muscle_joint_care": {
        "search_term": "muscle joint pain relief",
        "label": "muscle and joint-care options",
        "description": "mild muscle/joint soreness without injury or severe pain red flags",
    },
    "vitamin_d": {
        "search_term": "vitamin d supplements",
        "label": "Vitamin D products",
        "description": "a clearly low Vitamin D report finding; browsing only, no dose advice",
    },
    "vitamin_b12": {
        "search_term": "vitamin b12 supplements",
        "label": "Vitamin B12 products",
        "description": "a clearly low Vitamin B12 report finding; browsing only, no dose advice",
    },
    "iron_nutrition": {
        "search_term": "iron supplements",
        "label": "iron and nutrition products",
        "description": "a clearly low iron/ferritin report finding; browsing only, no dose advice",
    },
}

ALLOWED_PRODUCT_CATEGORY_KEYS = frozenset(PRODUCT_CATEGORY_REGISTRY)


# Known deterministic mappings remain as a fast/high-precision fallback. They
# are NOT the complete universe anymore; the structured LLM classifier can map
# unseen wording into PRODUCT_CATEGORY_REGISTRY.
PRODUCT_DISCOVERY_OPTIONS: dict[str, dict] = {
    "headache": {
        "category_key": "pain_fever",
        "keywords": [
            "headache", "head ache", "sar dard", "sir dard", "sar mein dard",
            "sir mein dard", "सिर दर्द", "सर दर्द", "తలనొప్పి", "ತಲೆನೋವು",
            "தலைவலி", "തലവേദന",
        ],
    },
    "fever": {
        "category_key": "pain_fever",
        "keywords": ["fever", "bukhar", "bukhaar", "बुखार", "జ్వరం", "ಜ್ವರ", "காய்ச்சல்", "പനി"],
    },
    "body_ache": {
        "category_key": "pain_fever",
        "keywords": [
            "body ache", "body pain", "body pains", "body mein dard", "body me dard",
            "shareer mein dard", "sharir mein dard", "शरीर दर्द", "శరీర నొప్పి",
            "ದೇಹ ನೋವು", "உடல் வலி", "ശരീര വേദന",
        ],
    },
    "acidity": {
        "category_key": "acidity_digestive",
        "keywords": [
            "acidity", "acid reflux", "heartburn", "indigestion", "pet mein jalan",
            "pet me jalan", "seene mein jalan", "seene me jalan", "एसिडिटी", "सीने में जलन",
            "అసిడిటీ", "ಆಸಿಡಿಟಿ", "அசிடிட்டி", "അസിഡിറ്റി",
        ],
    },
    "diarrhea": {
        "category_key": "hydration",
        "keywords": [
            "diarrhea", "diarrhoea", "loose motion", "loose motions", "dast", "दस्त",
            "पेट खराब", "విరేచనాలు", "ಅತಿಸಾರ", "வயிற்றுப்போக்கு", "വയറിളക്കം",
        ],
    },
    "dehydration": {
        "category_key": "hydration",
        "keywords": ["dehydration", "dehydrated", "electrolyte", "electrolytes", "oral rehydration"],
    },
    "cold_cough": {
        "category_key": "cold_cough",
        "keywords": [
            "cold", "cough", "runny nose", "blocked nose", "nasal congestion",
            "khansi", "zukam", "sardi", "खांसी", "जुकाम", "सर्दी",
        ],
    },
    "sore_throat": {
        "category_key": "sore_throat",
        "keywords": ["sore throat", "gala kharab", "throat irritation"],
    },
    "constipation": {
        "category_key": "constipation",
        "keywords": ["constipation", "kabz", "कब्ज", "மலச்சிக்கல்", "മലബന്ധം"],
    },
    "minor_allergy": {
        "category_key": "allergy_care",
        "keywords": ["mild allergy", "seasonal allergy", "hay fever", "allergic rhinitis"],
    },
    "first_aid": {
        "category_key": "first_aid",
        "keywords": ["minor cut", "small cut", "minor wound", "small wound", "scrape", "abrasion"],
    },
    "vitamin_d": {
        "category_key": "vitamin_d",
        "keywords": [
            "low vitamin d", "vitamin d low", "vitamin d is low", "vitamin d level is low",
            "vitamin d is lower", "vitamin d deficiency", "vit d deficiency",
        ],
    },
    "vitamin_b12": {
        "category_key": "vitamin_b12",
        "keywords": [
            "low vitamin b12", "vitamin b12 low", "vitamin b12 is low",
            "vitamin b12 level is low", "b12 deficiency", "low b12",
        ],
    },
    "iron": {
        "category_key": "iron_nutrition",
        "keywords": ["low iron", "iron is low", "iron deficiency", "low ferritin", "ferritin low", "ferritin is low"],
    },
}

# Backward-compatible alias used by older code/tests.
OTC_OPTIONS = PRODUCT_DISCOVERY_OPTIONS
REPORT_PRODUCT_KEYS = {"vitamin_d", "vitamin_b12", "iron"}


# Category-specific blockers are defense-in-depth. The hybrid triage layer is
# still the primary emergency gate, but these prevent a product CTA when the
# same message contains a contraindicating/red-flag context.
CATEGORY_BLOCKER_PATTERNS: dict[str, tuple[str, ...]] = {
    "eye_care": (
        r"\b(severe|intense)\s+eye\s+pain\b",
        r"\b(sudden\s+)?(vision\s+loss|loss\s+of\s+vision|blurred\s+vision|blurry\s+vision)\b",
        r"\bchemical\b.*\beye\b|\beye\b.*\bchemical\b",
        r"\beye\s+(injury|trauma)\b",
        r"\bforeign\s+body\b.*\beye\b",
        r"\bcontact\s+lens(?:es)?\b.*\b(red|pain|swelling|swollen)\b",
    ),
    "allergy_care": (
        r"\b(lip|tongue|throat|face|facial)\s+swelling\b",
        r"\bwheez\w*\b",
        r"\b(can'?t|cannot)\s+breathe\b",
    ),
    "acidity_digestive": (
        r"\bsevere\s+(abdominal|stomach)\s+pain\b",
        r"\b(vomit|vomiting)\s+blood\b",
        r"\bblack\s+(stool|stools)\b",
        r"\bblood\s+in\s+(stool|stools)\b",
    ),
    "constipation": (
        r"\bsevere\s+(abdominal|stomach)\s+pain\b",
        r"\b(vomit|vomiting)\b.*\bconstipat\w*\b",
        r"\bblood\s+in\s+(stool|stools)\b",
    ),
    "cold_cough": (
        r"\bcoughing\s+blood\b",
        r"\b(can'?t|cannot)\s+breathe\b",
        r"\bsevere\s+shortness\s+of\s+breath\b",
    ),
    "sore_throat": (
        r"\bunable\s+to\s+swallow\b",
        r"\bdrooling\b",
        r"\bthroat\s+swelling\b",
        r"\b(can'?t|cannot)\s+breathe\b",
    ),
    "skin_care": (
        r"\b(blistering|peeling)\s+(rash|skin)\b",
        r"\brash\b.*\b(eyes|mouth|genitals)\b",
        r"\bfever\b.*\bwidespread\s+rash\b",
    ),
    "first_aid": (
        r"\b(deep|gaping)\s+(cut|wound)\b",
        r"\b(uncontrolled|heavy|severe)\s+bleeding\b",
        r"\b(animal|dog|cat|human)\s+bite\b",
    ),
    "pain_fever": (
        r"\bchest\s+pain\b",
        r"\bsevere\s+pain\b",
        r"\bnewborn\b.*\bfever\b",
        r"\binfant\b.*\bhigh\s+fever\b",
    ),
    "muscle_joint_care": (
        r"\b(deformity|cannot\s+bear\s+weight|unable\s+to\s+walk)\b",
        r"\bmajor\s+(injury|trauma)\b",
    ),
}


def _keyword_matches(text_l: str, keyword: str) -> bool:
    keyword_l = keyword.lower().strip()
    if not keyword_l:
        return False
    if keyword_l.isascii():
        return re.search(rf"(?<!\w){re.escape(keyword_l)}(?!\w)", text_l) is not None
    return keyword_l in text_l


def build_search_url(query: str) -> str:
    return PHARMEASY_SEARCH_URL.format(query=quote_plus(query))


def get_product_category_metadata(category_key: str | None) -> dict | None:
    """Return app-owned metadata for one allowlisted category key."""
    if not category_key or category_key not in PRODUCT_CATEGORY_REGISTRY:
        return None
    item = PRODUCT_CATEGORY_REGISTRY[category_key]
    return {
        "category_key": category_key,
        "category": item["search_term"],
        "search_term": item["search_term"],
        "label": item["label"],
        "description": item["description"],
        "pharmeasy_url": build_search_url(item["search_term"]),
    }


def product_category_block_reason(category_key: str, user_text: str) -> str | None:
    """Return why a category should be suppressed for this context, if any."""
    text = user_text or ""
    for pattern in CATEGORY_BLOCKER_PATTERNS.get(category_key, ()):
        if re.search(pattern, text, flags=re.IGNORECASE):
            return f"Category '{category_key}' blocked by safety context: {pattern}"
    return None


def validate_product_category_for_context(
    category_key: str | None,
    user_text: str,
    triage_tier: str,
    high_risk_context: bool,
) -> tuple[bool, str | None, dict | None]:
    """App-level validation for an LLM/deterministic product-category candidate."""
    metadata = get_product_category_metadata(category_key)
    if not metadata:
        return False, "Category is not in the application allowlist.", None
    if triage_tier in {"urgent", "self_harm"}:
        return False, "Emergency/self-harm flows never receive commercial product CTAs.", None
    if high_risk_context:
        return False, "High-risk medicine context requires clinician/pharmacist guidance first.", None
    blocker = product_category_block_reason(category_key or "", user_text)
    if blocker:
        return False, blocker, None
    return True, None, metadata


def product_category_prompt_lines() -> str:
    """Compact allowlist description for the structured LLM classifier prompt."""
    return "\n".join(
        f"- {key}: {item['description']}"
        for key, item in PRODUCT_CATEGORY_REGISTRY.items()
    )


def _find_option(text: str) -> dict | None:
    text_l = (text or "").lower().strip()
    if not text_l:
        return None
    for option in PRODUCT_DISCOVERY_OPTIONS.values():
        if any(_keyword_matches(text_l, keyword) for keyword in option.get("keywords", [])):
            return get_product_category_metadata(option["category_key"])
    return None


def _find_report_option(text: str) -> dict | None:
    text_l = (text or "").lower().strip()
    if not text_l:
        return None
    for key in REPORT_PRODUCT_KEYS:
        option = PRODUCT_DISCOVERY_OPTIONS[key]
        if any(_keyword_matches(text_l, keyword) for keyword in option.get("keywords", [])):
            return get_product_category_metadata(option["category_key"])
    return None


def has_reviewed_otc_option(text: str) -> bool:
    return _find_option(text) is not None


def has_reviewed_product_option(text: str) -> bool:
    return _find_option(text) is not None


def match_reviewed_report_product(text: str) -> dict | None:
    """Match only reviewed lab/report findings in generated report explanations."""
    return _find_report_option(text)


def match_reviewed_product(text: str) -> dict | None:
    """High-precision deterministic category fallback for known symptom wording."""
    return _find_option(text)


@tool
def get_otc_product_discovery(symptom_or_finding: str) -> str:
    """Backward-compatible reviewed product discovery tool.

    New graph code primarily uses the structured product-category classifier.
    This tool remains available for compatibility and never returns dosing advice.
    """
    option = _find_option(symptom_or_finding)
    if not option:
        return json.dumps({"eligible": False, "reason": "No reviewed category matched."}, ensure_ascii=False)
    return json.dumps(
        {
            "eligible": True,
            **option,
            "positioning": "optional_product_discovery_not_personalised_treatment",
        },
        ensure_ascii=False,
    )


@tool
def suggest_otc_medicine(symptom: str) -> str:
    """Deprecated compatibility wrapper for category-level product discovery."""
    option = _find_option(symptom)
    if not option:
        return json.dumps({"eligible": False}, ensure_ascii=False)
    return json.dumps(
        {"eligible": True, **option, "positioning": "optional_product_discovery_not_personalised_treatment"},
        ensure_ascii=False,
    )


@tool
def doctor_consultation(reason: str = "Health question") -> str:
    """Return PharmEasy's official online doctor consultation destination."""
    return json.dumps(
        {
            "consultation": True,
            "reason": reason,
            "availability": "24/7",
            "starting_price_inr": 199,
            "pharmeasy_doctor_url": settings.PHARMEASY_DOCTOR_URL,
        },
        ensure_ascii=False,
    )


# Backward-compatible broad mapping helpers used by older callers.
SYMPTOM_TO_SEARCH_TERM = {
    "cold": "cold and cough care",
    "cough": "cold and cough care",
    "sore throat": "throat lozenges",
    "fever": "fever and pain relief",
    "headache": "fever and pain relief",
    "body ache": "fever and pain relief",
    "acidity": "acidity and digestive care",
    "indigestion": "acidity and digestive care",
    "constipation": "constipation and digestive care",
    "diarrhea": "ors and electrolytes",
    "loose motion": "ors and electrolytes",
    "dehydration": "ors and electrolytes",
    "vitamin d": "vitamin d supplements",
    "vitamin b12": "vitamin b12 supplements",
    "low ferritin": "iron supplements",
    "iron deficiency": "iron supplements",
    "red eyes": "eye care lubricating eye drops",
    "dry eyes": "eye care lubricating eye drops",
}


def match_search_term(text: str) -> str | None:
    text_l = (text or "").lower()
    for keyword, term in SYMPTOM_TO_SEARCH_TERM.items():
        if keyword in text_l:
            return term
    return None


def match_pharmeasy_url(text: str) -> str | None:
    term = match_search_term(text)
    return build_search_url(term) if term else None
