"""
Thin wrapper around Meta's WhatsApp Cloud API (Graph API).
Requires WHATSAPP_ACCESS_TOKEN + WHATSAPP_PHONE_NUMBER_ID in .env.
Docs: https://developers.facebook.com/docs/whatsapp/cloud-api
"""
import httpx
from app.config import settings

BASE_URL = f"https://graph.facebook.com/{settings.WHATSAPP_API_VERSION}"


def _headers():
    return {
        "Authorization": f"Bearer {settings.WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }


async def send_text_message(to: str, body: str) -> dict:
    """Send a plain text reply. `to` is the user's WhatsApp number (E.164, no '+')."""
    url = f"{BASE_URL}/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body},
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=_headers(), json=payload, timeout=20)
        if resp.status_code >= 400:
            print("WhatsApp API error:", resp.status_code, resp.text)

        resp.raise_for_status()
        return resp.json()


async def send_language_picker(to: str) -> dict:
    """Interactive list message for language selection - use on first contact."""
    url = f"{BASE_URL}/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "list",
            "body": {"text": "Please choose your preferred language / भाषा चुनें"},
            "action": {
                "button": "Select / चुनें",
                "sections": [{
                    "title": "Languages",
                    "rows": [
                        {"id": "lang_en", "title": "English"},
                        {"id": "lang_hi", "title": "हिंदी (Hindi)"},
                        {"id": "lang_ta", "title": "தமிழ் (Tamil)"},
                        {"id": "lang_te", "title": "తెలుగు (Telugu)"},
                        {"id": "lang_kn", "title": "ಕನ್ನಡ (Kannada)"},
                        {"id": "lang_ml", "title": "മലയാളം (Malayalam)"},
                    ],
                }],
            },
        },
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=_headers(), json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()


async def send_doctor_consult_button(to: str, body: str) -> dict:
    """Attach a CTA button for moderate/urgent-adjacent replies."""
    url = f"{BASE_URL}/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": body},
            "action": {
                "buttons": [
                    {"type": "reply", "reply": {"id": "book_consult", "title": "Book doctor consult"}},
                    {"type": "reply", "reply": {"id": "talk_pharmacist", "title": "Talk to pharmacist"}},
                ]
            },
        },
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=_headers(), json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()


async def send_cta_url_button(to: str, body: str, button_text: str, url_link: str) -> dict:
    """Link-out button that opens a URL - use this for 'Book consult on
    PharmEasy' / 'View products on PharmEasy' where you want the user to
    land directly on the real PharmEasy page/flow rather than a reply."""
    url = f"{BASE_URL}/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "cta_url",
            "body": {"text": body},
            "action": {
                "name": "cta_url",
                "parameters": {"display_text": button_text, "url": url_link},
            },
        },
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=_headers(), json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()


async def download_media(media_id: str) -> bytes:
    """WhatsApp media download is two-step: get a temp URL, then fetch it."""
    async with httpx.AsyncClient() as client:
        meta_resp = await client.get(f"{BASE_URL}/{media_id}", headers=_headers(), timeout=20)
        meta_resp.raise_for_status()
        media_url = meta_resp.json()["url"]

        file_resp = await client.get(media_url, headers=_headers(), timeout=30)
        file_resp.raise_for_status()
        return file_resp.content


async def send_contact_card(to: str, display_name: str, phone_number: str) -> dict:
    """Sends a tappable contact card - the WhatsApp-native way to offer a
    one-tap call action. (interactive cta_url buttons only accept http/https
    URLs - the Cloud API rejects tel: links on those - so this 'contacts'
    message type is the right building block for a call action. Needs no
    template pre-approval and works immediately in a live session.)"""
    url = f"{BASE_URL}/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "contacts",
        "contacts": [{
            "name": {"formatted_name": display_name, "first_name": display_name},
            "phones": [{"phone": phone_number, "type": "CELL"}],
        }],
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=_headers(), json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()


async def send_location_request(to: str, body: str) -> dict:
    """Asks the user to share their location - send this after an emergency
    message so we can point them to the nearest hospital. Native WhatsApp
    interactive type; renders a 'Send Location' button in the chat."""
    url = f"{BASE_URL}/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "location_request_message",
            "body": {"text": body},
            "action": {"name": "send_location"},
        },
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=_headers(), json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()


async def send_nearby_hospitals_link(to: str, latitude: float, longitude: float) -> dict:
    """Once we have the user's shared location, send a Maps link centered
    on nearby hospitals/emergency rooms."""
    maps_url = f"https://www.google.com/maps/search/hospital+emergency+room/@{latitude},{longitude},14z"
    return await send_cta_url_button(
        to, body="Here are hospitals near your location:",
        button_text="View nearby hospitals", url_link=maps_url,
    )
