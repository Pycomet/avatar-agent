import json
import logging
import os

import httpx
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    room_io,
)
from livekit.plugins import anam, google, liveavatar, noise_cancellation, silero

from menu_data import find_item_by_name, get_all_items, get_item_by_id

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Map ISO 639-1 language codes to full language names for the LLM
LANGUAGE_CODES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "tr": "Turkish",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "id": "Indonesian",
    "pl": "Polish",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "el": "Greek",
    "he": "Hebrew",
    "th": "Thai",
    "vi": "Vietnamese",
    "uk": "Ukrainian",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
}


def get_language_name(code: str) -> str:
    """Convert language code to full name, or return as-is if already a name."""
    # If it's a 2-letter code, look it up
    if len(code) == 2:
        return LANGUAGE_CODES.get(code.lower(), code)
    # Otherwise assume it's already a full name
    return code


class Assistant(Agent):
    def __init__(self, user_language: str = "English") -> None:
        self.user_language = user_language
        self.current_order = []  # Track current order items
        super().__init__(
            instructions=f"""You are a helpful restaurant booking assistant. The user is interacting with you via voice.
            You help users make reservations, check availability, and answer questions about the restaurant.
            Your responses are concise, friendly, and conversational.
            Keep your responses natural and to the point, without complex formatting, emojis, or asterisks.

            You can help users:
            - Answer questions about the menu (what's available, ingredients, prices, categories)
            - Show images of menu items when they ask to see a specific dish
            - Place orders by collecting items and any special requests

            When discussing the menu:
            - Don't read the entire menu unless asked - answer specific questions naturally
            - For "what do you have" or "what's on the menu", give a brief overview of categories or highlight a few items
            - When asked about specific items (ingredients, price), provide details for just that item
            - Offer to show images when discussing specific dishes

            IMPORTANT: The user's preferred language is {user_language}. Start by speaking in {user_language}.
            If the user switches to a different language, follow their lead and respond in that language.
            Always match the language the user is currently speaking.""",
        )

    @function_tool
    async def get_menu(self, context: RunContext, category: str = ""):
        """Get restaurant menu items, optionally filtered by category.

        Use this tool when the user asks about the menu, available items, or specific types of food.
        You should interpret the results and answer the user's question naturally - don't read
        the entire list unless specifically asked.

        Args:
            category: Optional category filter (e.g., "Appetizers", "Mains", "Desserts", "Drinks").
                     Leave empty to get all items.

        Returns structured data about menu items that you can use to answer questions.
        """
        logger.info(f"Fetching menu items (category: {category or 'all'})")

        menu_items = get_all_items()

        # Filter by category if specified
        if category:
            menu_items = [
                item
                for item in menu_items
                if item["category"].lower() == category.lower()
            ]

        # Return structured data for the LLM to interpret naturally
        return json.dumps(menu_items, indent=2)

    @function_tool
    async def show_item(self, context: RunContext, item_name: str):
        """Show an image of a specific menu item to the user.

        Use this when the user asks to see what a dish looks like or wants a visual.

        Args:
            item_name: The name of the menu item (e.g., "Bruschetta", "pasta", "pizza")
                      Partial names work - will match the first item found.
        """
        logger.info(f"Showing image for item: {item_name}")

        # Try to find by name (more natural for users)
        item = find_item_by_name(item_name)

        # Fallback to ID search if name doesn't work
        if not item:
            item = get_item_by_id(item_name)

        if not item:
            return f"Sorry, I couldn't find '{item_name}' on the menu."

        # Send data message to frontend to display the image
        data_payload = json.dumps(
            {
                "type": "show_image",
                "url": item["image_url"],
                "title": item["name"],
            }
        )

        await context.room.local_participant.publish_data(
            data_payload.encode("utf-8"),
            reliable=True,
        )

        return f"Showing you an image of {item['name']}"

    @function_tool
    async def place_order(self, context: RunContext, items: str, notes: str = ""):
        """Place an order with the restaurant.

        Use this when the user is ready to finalize their order.

        Args:
            items: A JSON string of items and quantities, e.g., '[{"id": "1", "quantity": 2}]'
            notes: Any special requests or notes for the order (optional)
        """
        logger.info(f"Placing order: {items} with notes: {notes}")

        try:
            order_items = json.loads(items)
        except json.JSONDecodeError:
            return "Sorry, I couldn't understand the order format. Please try again."

        # Get ORDER_API_URL from environment
        api_url = os.getenv("ORDER_API_URL")
        if not api_url:
            logger.warning("ORDER_API_URL not set, logging order locally")
            return f"Order received: {len(order_items)} items. Notes: {notes or 'None'}"

        # Send order to NextJS API
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    api_url,
                    json={
                        "items": order_items,
                        "notes": notes,
                        "room_id": context.room.name,
                    },
                    timeout=10.0,
                )
                response.raise_for_status()

            return f"Order successfully placed! {len(order_items)} items ordered."
        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            return "Sorry, there was an error submitting your order. Please try again."

    async def on_enter(self):
        # Note: allow_interruptions not supported with RealtimeModel's server-side turn detection
        await self.session.generate_reply(
            instructions=f"""Greet the user warmly in {self.user_language} and say this exactly - Hello, how can I help you today?""",
        )


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="ai-waiter")
async def my_agent(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Get user's preferred language and avatar provider from job metadata (dispatch)
    # Frontend sends metadata via createDispatch: {"language": "tr", "avatar_provider": "anam"}
    # avatar_provider can be: "anam", "liveavatar", or "none"
    user_language = "English"  # Default fallback
    avatar_provider = "none"  # Default: no avatar (voice-only)

    # Valid avatar providers - validates frontend requests
    valid_avatar_providers = {"anam", "liveavatar", "none"}

    if ctx.job.metadata:
        try:
            metadata = json.loads(ctx.job.metadata)
            logger.info(f"Job metadata received: {metadata}")
            
            # Parse language
            lang_code = metadata.get("language", "en")
            user_language = get_language_name(lang_code)
            logger.info(f"User language: {user_language} (from code: {lang_code})")

            # Parse avatar provider with validation
            raw_avatar_provider = metadata.get("avatar_provider", "none").lower()
            if raw_avatar_provider in valid_avatar_providers:
                avatar_provider = raw_avatar_provider
                logger.info(f"Avatar provider requested: {avatar_provider}")
            else:
                logger.warning(
                    f"Invalid avatar_provider '{raw_avatar_provider}' - "
                    f"must be one of {valid_avatar_providers}. Defaulting to 'none'"
                )
                avatar_provider = "none"
        except json.JSONDecodeError:
            # If metadata is plain text, try to convert it as language
            user_language = get_language_name(ctx.job.metadata)
            logger.info(f"User language from plain metadata: {user_language}")
    else:
        logger.info("No job metadata received - using defaults")

    # Set up voice AI pipeline using Gemini Live API (realtime model)
    # This handles STT + LLM + TTS all-in-one using GOOGLE_API_KEY
    # See https://docs.livekit.io/agents/models/realtime/plugins/gemini
    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            model="gemini-2.0-flash-exp",
            voice="Kore",  # Options: Puck, Charon, Kore, Fenrir, Aoede
        ),
        # VAD is used to determine when the user is speaking
        vad=ctx.proc.userdata["vad"],
    )

    # Initialize avatar based on frontend request (via room metadata)
    # Set ANAM_API_KEY + ANAM_AVATAR_ID or LIVEAVATAR_AVATAR_ID in your .env file
    avatar = None  # Track avatar instance for cleanup
    logger.info(f"Avatar provider from metadata: '{avatar_provider}'")

    if avatar_provider == "anam":
        avatar_id = os.getenv("ANAM_AVATAR_ID")
        if avatar_id:
            logger.info(f"Initializing Anam avatar with id: {avatar_id}")
            try:
                avatar = anam.AvatarSession(
                    persona_config=anam.PersonaConfig(name="Sofia", avatarId=avatar_id),
                )
                await avatar.start(session, room=ctx.room)
                logger.info("Anam avatar started successfully")
            except Exception as e:
                logger.warning(f"Anam avatar failed (continuing voice-only): {e}")
                avatar = None
        else:
            logger.warning("avatar_provider=anam but ANAM_AVATAR_ID env var not set")

    elif avatar_provider == "liveavatar":
        liveavatar_id = os.getenv("LIVEAVATAR_AVATAR_ID")
        if liveavatar_id:
            logger.info(f"Initializing LiveAvatar with id: {liveavatar_id}")
            try:
                avatar = liveavatar.AvatarSession(avatar_id=liveavatar_id)
                await avatar.start(session, room=ctx.room)
                logger.info("LiveAvatar started successfully")
            except Exception as e:
                logger.warning(f"LiveAvatar failed (continuing voice-only): {e}")
                avatar = None
        else:
            logger.warning(
                "avatar_provider=liveavatar but LIVEAVATAR_AVATAR_ID env var not set"
            )

    else:
        logger.info("No avatar requested (voice-only mode)")

    # Register shutdown callback for graceful cleanup
    async def shutdown_callback():
        """Clean up resources when the session ends."""
        logger.info("Shutdown callback triggered - cleaning up resources")
        # The avatar session listens to the AgentSession's close event,
        # which triggers cleanup of its internal tasks and WebSocket connection.
        # We explicitly close the session here to ensure clean termination.
        try:
            await session.aclose()
            logger.info("Agent session closed successfully")
        except Exception as e:
            logger.warning(f"Error during session cleanup: {e}")

    ctx.add_shutdown_callback(shutdown_callback)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(user_language=user_language),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            delete_room_on_close=True,
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
