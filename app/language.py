"""
Language detection + speech-to-text.

Detection: script-based for text (fast, no model needed). For audio, the
ASR model itself detects language as part of transcription (Whisper-family
models do this natively), so there's no separate detection step for voice.

TRANSLATION APPROACH: we deliberately do NOT run a separate translate-to-
English-then-translate-back pipeline. Claude handles Hindi/Tamil/Telugu/
Kannada/Malayalam generation directly with good quality, so the LLM is
instructed (see app/llm.py) to answer directly in the user's language. This
avoids an extra model hop and the quality loss of double translation. If a
dedicated translation model is wanted later for the *retrieval query* (e.g.
to match an English-only knowledge base), IndicTrans2 (AI4Bharat, open
source) is the right choice for Indian languages - not implemented here
for use with the live PharmEasy web-search tool,
which searches with the source terms directly.
"""
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

# crude script-based detection - good enough for routing without a model
SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F),  # Devanagari (Hindi/Marathi)
    "ta": (0x0B80, 0x0BFF),  # Tamil
    "te": (0x0C00, 0x0C7F),  # Telugu
    "kn": (0x0C80, 0x0CFF),  # Kannada
    "ml": (0x0D00, 0x0D7F),  # Malayalam
}

LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "ta": "Tamil",
    "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
    "hinglish": "Hindi written in Roman/English script (Hinglish)",
}

try:
    from faster_whisper import WhisperModel
    _whisper_model = None  # lazy-loaded, see _get_whisper_model()
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


def detect_language(text: str) -> str:
    for ch in text:
        cp = ord(ch)
        for lang, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                return lang
    if _looks_like_hinglish(text):
        return "hinglish"
    return "en"


# Common Hindi/Hindustani words written in Roman script. This is a heuristic,
# not a language model - it exists so "language_name" can actually
# distinguish true English from Hinglish (previously both returned "en",
# since script-range detection can't tell Latin-script Hindi from English).
# Deliberately short/common-word-based to keep false positives low; extend
# with more words if real traffic shows gaps.
HINGLISH_MARKERS = {
    "hai", "hain", "nahi", "nahin", "kya", "kaise", "kyun", "kyu", "mujhe",
    "mera", "meri", "mere", "aap", "aapko", "aapka", "tum", "tumhe",
    "dard", "bukhar", "bukhaar", "pait", "pet", "sar", "sir", "thoda",
    "zyada", "raha", "rahi", "rahe", "hoon", "hu", "ho", "karo", "kijiye",
    "dawa", "dawai", "tabiyat", "accha", "theek", "bimari", "khansi",
    "ulti", "chakkar", "chot", "dikkat", "samasya",
}


def _looks_like_hinglish(text: str) -> bool:
    words = set(w.strip(".,!?;:").lower() for w in text.split())
    return len(words & HINGLISH_MARKERS) >= 1


def language_name(lang_code: str) -> str:
    return LANGUAGE_NAMES.get(lang_code, "English")


def _get_whisper_model():
    """Lazy-load so the model isn't loaded until actually needed (it's a
    few hundred MB - loading it at import time would slow every cold start).
    Model size tradeoff: 'small' is fast but weaker on Indic languages;
    'medium' or 'large-v3' are noticeably better for Hindi/Tamil/etc but
    need more RAM/CPU or a GPU. Set WHISPER_MODEL_SIZE in .env to tune this
    per your hosting budget."""
    global _whisper_model
    if _whisper_model is None:
        from app.config import settings
        # compute_type="int8" keeps CPU inference usable without a GPU;
        # switch to "float16" if running on GPU for better accuracy/speed.
        _whisper_model = WhisperModel(
            settings.WHISPER_MODEL_SIZE, device="cpu", compute_type="int8"
        )
    return _whisper_model


def transcribe_voice_note(audio_bytes: bytes) -> tuple[str, str]:
    """Transcribe a WhatsApp voice note with faster-whisper.

    WhatsApp voice notes are commonly OGG/Opus.  Writing the payload to a
    temporary file is more reliable across PyAV/faster-whisper versions than
    passing a BytesIO object directly, especially on Windows.

    Returns: (transcribed_text, detected_language_code)
    """
    if not WHISPER_AVAILABLE:
        raise RuntimeError(
            "faster-whisper is not installed. Run `pip install faster-whisper`."
        )

    if not audio_bytes:
        raise RuntimeError("The downloaded WhatsApp voice note was empty.")

    logger.info("Voice transcription started (%d bytes).", len(audio_bytes))

    temp_path = None
    try:
        # WhatsApp voice notes are normally OGG/Opus. PyAV (used by
        # faster-whisper) handles the decoding; a separate ffmpeg executable
        # is normally not required for faster-whisper.
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name

        model = _get_whisper_model()
        logger.info("Whisper model loaded: %s", model.__class__.__name__)

        segments, info = model.transcribe(
            temp_path,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()

        if not text:
            raise RuntimeError(
                "Voice note was decoded but no speech could be transcribed."
            )

        whisper_lang = getattr(info, "language", None) or "en"
        probability = getattr(info, "language_probability", 0.0) or 0.0
        detected_lang = whisper_lang if whisper_lang in LANGUAGE_NAMES else "en"

        logger.info(
            "Voice transcription complete: language=%s confidence=%.2f text=%r",
            whisper_lang,
            probability,
            text[:160],
        )
        return text, detected_lang

    except Exception as exc:
        logger.exception("Voice transcription failed: %s", exc)
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"Voice transcription failed: {exc}") from exc
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning("Could not remove temporary voice file: %s", temp_path)
