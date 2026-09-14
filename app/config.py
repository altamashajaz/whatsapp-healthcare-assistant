"""
Central configuration. All secrets come from environment variables (.env).
Never hardcode keys here.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- OpenAI ---
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4.1")

    # --- WhatsApp Cloud API ---
    WHATSAPP_ACCESS_TOKEN: str = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
    WHATSAPP_PHONE_NUMBER_ID: str = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
    WHATSAPP_VERIFY_TOKEN: str = os.getenv(
        "WHATSAPP_VERIFY_TOKEN",
        "dev-verify-token"
    )
    WHATSAPP_API_VERSION: str = os.getenv(
        "WHATSAPP_API_VERSION",
        "v20.0"
    )

    # --- ASR/TTS ---
    BHASHINI_API_KEY: str = os.getenv("BHASHINI_API_KEY", "")
    BHASHINI_USER_ID: str = os.getenv("BHASHINI_USER_ID", "")

    SUPPORTED_LANGUAGES = ["en", "hinglish", "hi", "ta", "te", "kn", "ml"]

    # --- PharmEasy website integration ---
    PHARMEASY_HOME_URL: str = os.getenv(
        "PHARMEASY_HOME_URL",
        "https://pharmeasy.in/"
    )
    PHARMEASY_SEARCH_URL: str = os.getenv(
        "PHARMEASY_SEARCH_URL",
        "https://pharmeasy.in/search/all?name={query}"
    )
    PHARMEASY_DOCTOR_URL: str = os.getenv(
        "PHARMEASY_DOCTOR_URL",
        "https://pharmeasy.in/online-doctor-consultation/?src=homecard"
    )

    # --- Emergency numbers (India) ---
    NATIONAL_EMERGENCY_NUMBER: str = os.getenv(
        "NATIONAL_EMERGENCY_NUMBER",
        "112"
    )
    AMBULANCE_NUMBER: str = os.getenv(
        "AMBULANCE_NUMBER",
        "108"
    )

    # --- ASR ---
    WHISPER_MODEL_SIZE: str = os.getenv(
        "WHISPER_MODEL_SIZE",
        "small"
    )


settings = Settings()
