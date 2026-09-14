import logging
import time

from fastapi import FastAPI, Request, Response, Query, HTTPException, BackgroundTasks

from app.config import settings
from app import whatsapp_client
from app.orchestrator import handle_message
from app.safety_rules import Tier


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=getattr(settings, "LOG_LEVEL", "INFO")
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="PharmEasy WhatsApp Health Assistant - LangGraph"
)


# ---------------------------------------------------------
# Duplicate message protection
# ---------------------------------------------------------
# Meta can retry the same webhook event.
# We keep track of message IDs that have already been processed.
#
# NOTE:
# This is in-memory protection. It resets when the application
# restarts. For production, use Redis/database-based deduplication.
# ---------------------------------------------------------

processed_message_ids = set()


# ---------------------------------------------------------
# GET /webhook
# Meta webhook verification
# ---------------------------------------------------------

@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
):
    logger.info(
        "Webhook verification: mode=%r, challenge=%r, token_received=%r",
        hub_mode,
        hub_challenge,
        hub_verify_token,
    )

    if (
        hub_mode == "subscribe"
        and hub_verify_token == settings.WHATSAPP_VERIFY_TOKEN
    ):
        return Response(
            content=hub_challenge,
            media_type="text/plain",
        )

    logger.error(
        "Webhook verification failed: mode_match=%s, token_match=%s",
        hub_mode == "subscribe",
        hub_verify_token == settings.WHATSAPP_VERIFY_TOKEN,
    )

    raise HTTPException(
        status_code=403,
        detail="Verification failed",
    )


# ---------------------------------------------------------
# POST /webhook
# Receive WhatsApp messages
# ---------------------------------------------------------

@app.post("/webhook")
async def receive_message(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Receive the WhatsApp webhook event.

    IMPORTANT:
    We acknowledge Meta immediately instead of waiting for
    OpenAI/LangGraph to finish.

    This prevents Meta from waiting several minutes and
    potentially retrying the same webhook event.
    """

    try:
        payload = await request.json()

        entry = (payload.get("entry") or [{}])[0]

        changes = (
            (entry.get("changes") or [{}])[0]
            .get("value", {})
        )

        messages = changes.get("messages")

        # Ignore webhook events that don't contain messages.
        # For example, status updates.
        if not messages:
            logger.info("Webhook event received without messages.")
            return {"status": "ignored"}

        # Schedule slow AI processing in the background.
        background_tasks.add_task(
            _process_message,
            payload,
        )

        # Return immediately to Meta.
        return {"status": "received"}

    except Exception as e:
        logger.exception(
            "Error receiving webhook payload: %s",
            e,
        )

        # Still return a valid response where appropriate.
        return {"status": "error"}


# ---------------------------------------------------------
# Background message processing
# ---------------------------------------------------------

async def _process_message(payload: dict):
    """
    Process the WhatsApp message after Meta has already
    received the webhook acknowledgement.
    """

    total_start = time.perf_counter()

    try:
        # -------------------------------------------------
        # Extract message
        # -------------------------------------------------

        entry = payload["entry"][0]

        changes = entry["changes"][0]["value"]

        messages = changes.get("messages")

        if not messages:
            return

        msg = messages[0]

        # -------------------------------------------------
        # Message ID
        # -------------------------------------------------

        message_id = msg.get("id")

        if not message_id:
            logger.warning(
                "WhatsApp message does not contain a message ID."
            )
        else:
            # ---------------------------------------------
            # Duplicate protection
            # ---------------------------------------------

            if message_id in processed_message_ids:
                logger.info(
                    "Duplicate WhatsApp message ignored: %s",
                    message_id,
                )
                return

            processed_message_ids.add(message_id)

            logger.info(
                "Processing WhatsApp message: %s",
                message_id,
            )

        # -------------------------------------------------
        # Basic message information
        # -------------------------------------------------

        from_number = msg["from"]

        msg_type = msg["type"]

        logger.info(
            "WhatsApp message type=%s from=%s",
            msg_type,
            from_number,
        )

        # -------------------------------------------------
        # TEXT
        # -------------------------------------------------

        if msg_type == "text":

            text = msg["text"]["body"]

            logger.info(
                "Processing text message: %r",
                text,
            )

            start_time = time.perf_counter()

            result = handle_message(
                text=text,
                thread_id=f"whatsapp_{from_number}",
            )

            elapsed = time.perf_counter() - start_time

            logger.info(
                "handle_message(text) took %.2f seconds",
                elapsed,
            )

        # -------------------------------------------------
        # IMAGE
        # -------------------------------------------------

        elif msg_type == "image":

            media_id = msg["image"]["id"]

            logger.info(
                "Downloading image media: %s",
                media_id,
            )

            start_time = time.perf_counter()

            data = await whatsapp_client.download_media(
                media_id
            )

            download_time = time.perf_counter() - start_time

            logger.info(
                "Image download took %.2f seconds",
                download_time,
            )

            start_time = time.perf_counter()

            result = handle_message(
                text=msg["image"].get("caption", ""),
                file_bytes=data,
                mime_type=msg["image"].get(
                    "mime_type",
                    "image/jpeg",
                ),
                thread_id=f"whatsapp_{from_number}",
            )

            elapsed = time.perf_counter() - start_time

            logger.info(
                "handle_message(image) took %.2f seconds",
                elapsed,
            )

        # -------------------------------------------------
        # DOCUMENT
        # -------------------------------------------------

        elif msg_type == "document":

            media_id = msg["document"]["id"]

            logger.info(
                "Downloading document media: %s",
                media_id,
            )

            start_time = time.perf_counter()

            data = await whatsapp_client.download_media(
                media_id
            )

            download_time = time.perf_counter() - start_time

            logger.info(
                "Document download took %.2f seconds",
                download_time,
            )

            mime = msg["document"].get(
                "mime_type",
                "application/pdf",
            )

            # Only allow PDFs and images.
            if (
                mime != "application/pdf"
                and not mime.startswith("image/")
            ):
                await whatsapp_client.send_text_message(
                    from_number,
                    "I can currently read PDF and image files.",
                )
                return

            start_time = time.perf_counter()

            result = handle_message(
                text=msg["document"].get("caption", ""),
                file_bytes=data,
                mime_type=mime,
                thread_id=f"whatsapp_{from_number}",
            )

            elapsed = time.perf_counter() - start_time

            logger.info(
                "handle_message(document) took %.2f seconds",
                elapsed,
            )

        # -------------------------------------------------
        # AUDIO
        # -------------------------------------------------

        elif msg_type == "audio":

            media_id = msg["audio"]["id"]

            logger.info(
                "Downloading audio media: %s",
                media_id,
            )

            start_time = time.perf_counter()

            data = await whatsapp_client.download_media(
                media_id
            )

            download_time = time.perf_counter() - start_time
            audio_mime = msg.get("audio", {}).get("mime_type", "unknown")

            logger.info(
                "Audio download took %.2f seconds (mime=%s, bytes=%d)",
                download_time,
                audio_mime,
                len(data),
            )

            start_time = time.perf_counter()

            result = handle_message(
                audio_bytes=data,
                thread_id=f"whatsapp_{from_number}",
            )

            elapsed = time.perf_counter() - start_time

            logger.info(
                "handle_message(audio) took %.2f seconds",
                elapsed,
            )

        # -------------------------------------------------
        # LOCATION
        # -------------------------------------------------

        elif msg_type == "location":

            loc = msg["location"]

            logger.info(
                "Location received from %s",
                from_number,
            )

            await whatsapp_client.send_nearby_hospitals_link(
                from_number,
                loc["latitude"],
                loc["longitude"],
            )

            return

        # -------------------------------------------------
        # OTHER MESSAGE TYPES
        # -------------------------------------------------

        else:

            logger.info(
                "Unsupported WhatsApp message type: %s",
                msg_type,
            )

            start_time = time.perf_counter()

            result = handle_message(
                text="",
                thread_id=f"whatsapp_{from_number}",
            )

            elapsed = time.perf_counter() - start_time

            logger.info(
                "handle_message(other) took %.2f seconds",
                elapsed,
            )

        # -------------------------------------------------
        # Send AI response
        # -------------------------------------------------

        logger.info(
            "Sending WhatsApp response to %s",
            from_number,
        )

        start_time = time.perf_counter()

        await whatsapp_client.send_text_message(
            from_number,
            result.reply_text,
        )

        send_time = time.perf_counter() - start_time

        logger.info(
            "WhatsApp text response took %.2f seconds",
            send_time,
        )

        # -------------------------------------------------
        # Emergency support
        # -------------------------------------------------

        if result.offer_emergency_help:

            await whatsapp_client.send_contact_card(
                from_number,
                "Emergency (112)",
                settings.NATIONAL_EMERGENCY_NUMBER,
            )

            if result.tier == Tier.URGENT:

                await whatsapp_client.send_contact_card(
                    from_number,
                    "Ambulance (108)",
                    settings.AMBULANCE_NUMBER,
                )

                await whatsapp_client.send_location_request(
                    from_number,
                    "Would you like me to find the nearest hospital? "
                    "Share your location:",
                )

        # -------------------------------------------------
        # PharmEasy doctor consultation CTA
        # -------------------------------------------------
        # Normal health conversations should make professional help easy to
        # access. Urgent/self-harm cases intentionally receive NO paid CTA;
        # emergency routing above takes priority.

        if (
            result.health_related
            and result.consult_url
            and result.tier not in {Tier.URGENT, Tier.SELF_HARM}
        ):
            await whatsapp_client.send_cta_url_button(
                from_number,
                body=(
                    "Need personalised medical guidance? "
                    "PharmEasy Doctor Consult is available 24/7, "
                    "starting at ₹199."
                ),
                button_text="Consult doctor",
                url_link=result.consult_url,
            )

        # -------------------------------------------------
        # PharmEasy OTC / health-product discovery CTA
        # -------------------------------------------------
        # This is optional product discovery, not a personalised medicine
        # recommendation. The graph only sets the URL for reviewed categories
        # that pass the triage/high-risk medicine gates. Report findings can be
        # MODERATE and still safely offer optional category-level discovery.

        if (
            result.health_related
            and result.tier not in {Tier.URGENT, Tier.SELF_HARM}
            and result.product_search_url
        ):
            label = result.product_label or "relevant OTC options"

            await whatsapp_client.send_cta_url_button(
                from_number,
                body=(
                    f"If you'd like, you can explore {label} on PharmEasy. "
                    "Choose based on the product label and pharmacist/doctor advice."
                ),
                button_text="Explore PharmEasy",
                url_link=result.product_search_url,
            )

        # -------------------------------------------------
        # Total processing time
        # -------------------------------------------------

        total_elapsed = time.perf_counter() - total_start

        logger.info(
            "TOTAL WhatsApp message processing time: %.2f seconds",
            total_elapsed,
        )

    except Exception as e:

        total_elapsed = time.perf_counter() - total_start

        logger.exception(
            "Failed to process webhook message after %.2f seconds: %s",
            total_elapsed,
            e,
        )


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "architecture": "langchain-langgraph",
        "rag": False,
    }
