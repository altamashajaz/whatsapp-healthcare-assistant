"""Compatibility entry point around the LangGraph workflow."""
from __future__ import annotations

from dataclasses import dataclass
import base64
import logging

from app.graph.graph import healthcare_graph
from app.safety_rules import Tier
from app import language
from app.document_extraction import extract_document, ExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    reply_text: str
    tier: Tier
    lang: str
    product_search_url: str | None = None
    product_label: str | None = None
    consult_url: str | None = None
    health_related: bool = True
    offer_emergency_help: bool = False

    # Compatibility with older call-sites that used result.consult.
    @property
    def consult(self):
        return self.consult_url


def handle_message(
    text: str | None = None,
    file_bytes: bytes | None = None,
    mime_type: str | None = None,
    audio_bytes: bytes | None = None,
    lang_hint: str | None = None,
    thread_id: str | None = None,
) -> PipelineResult:
    detected_lang = None
    if audio_bytes:
        try:
            text, detected_lang = language.transcribe_voice_note(audio_bytes)
        except Exception as exc:
            logger.exception("Voice-note transcription failed before graph invocation: %s", exc)
            return PipelineResult(
                "Sorry, I couldn't understand that voice note. Please try recording it again, or type your health question instead.",
                Tier.LOW,
                "en",
                health_related=False,
            )

    extraction: ExtractionResult | None = None
    if file_bytes:
        extraction = extract_document(file_bytes, mime_type or "image/jpeg")

    user_text = text or ""
    lang = detected_lang or (
        language.detect_language(user_text)
        if user_text
        else (lang_hint or "en")
    )

    image_data_url = None
    if file_bytes and mime_type and mime_type.startswith("image/"):
        encoded = base64.b64encode(file_bytes).decode("utf-8")
        image_data_url = f"data:{mime_type};base64,{encoded}"

    state = {
        "user_text": user_text,
        "report_text": extraction.raw_text if extraction else None,
        "image_data_url": image_data_url,
        "ocr_confidence": extraction.ocr_confidence if extraction else 0.0,
        "detected_language": lang,
        "messages": [
            {
                "role": "user",
                "content": user_text or "Please inspect the uploaded image/report.",
            }
        ],
    }

    conversation_thread_id = thread_id or "local_dev"
    result = healthcare_graph.invoke(
        state,
        config={"configurable": {"thread_id": conversation_thread_id}},
    )

    return PipelineResult(
        reply_text=result.get("response", ""),
        tier=Tier(result.get("triage_tier", Tier.LOW.value)),
        lang=result.get("detected_language", lang),
        product_search_url=result.get("product_search_url"),
        product_label=result.get("product_label"),
        consult_url=result.get("consult_url"),
        health_related=result.get("health_related", True),
        offer_emergency_help=result.get("offer_emergency_help", False),
    )
