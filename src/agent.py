import json
import logging
import os
from typing import Optional

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
    get_job_context,
    room_io,
)
from livekit.plugins import anam, google, liveavatar, noise_cancellation, silero

from menu_data import MenuData, fetch_menu_data, set_menu_data

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
    def __init__(self, menu_data: MenuData, user_language: str = "English") -> None:
        self.user_language = user_language
        self.menu_data = menu_data
        self.selected_restaurant_id: Optional[str] = None
        self.current_order: list = []

        # Build restaurant info for instructions
        restaurant_info = menu_data.get_restaurant_summary()

        super().__init__(
            instructions=f"""You are a helpful restaurant ordering assistant. The user is interacting with you via voice.
You help users browse menus and place orders from our partner restaurants.
Your responses are concise, friendly, and conversational.
Keep your responses natural and to the point, without complex formatting, emojis, or asterisks.

AVAILABLE RESTAURANTS:
{restaurant_info}

CONVERSATION FLOW:
1. First, help the user select which restaurant they want to order from
2. Once they select a restaurant, you can help them browse the menu
3. Help them explore categories, specific items, prices, and descriptions
4. When they're ready, help them place their order

You can help users:
- See available restaurants and their cuisines
- Select a restaurant to order from
- Browse the menu by category or see all items
- Get details about specific dishes (ingredients, price, description)
- Show images of menu items when they ask to see a dish
- Place orders with any special requests

When discussing the menu:
- Don't read the entire menu unless asked - answer specific questions naturally
- For "what do you have" or "what's on the menu", give a brief overview of categories or highlight a few popular items
- When asked about specific items, provide details for just that item
- Offer to show images when discussing specific dishes

IMPORTANT: The user's preferred language is {user_language}. Start by speaking in {user_language}.
If the user switches to a different language, follow their lead and respond in that language.
Always match the language the user is currently speaking.""",
        )

    @function_tool
    async def get_restaurants(self, context: RunContext):
        """Get the list of available restaurants.

        Use this when the user asks what restaurants are available or wants to choose where to order from.
        """
        logger.info("Fetching restaurant list")

        restaurants = self.menu_data.get_all_restaurants()
        if not restaurants:
            return "Sorry, no restaurants are currently available."

        # Return structured data for the LLM to interpret
        result = []
        for r in restaurants:
            result.append(
                {
                    "id": r.get("id"),
                    "name": r.get("name"),
                    "cuisine": r.get("cuisine", ""),
                    "description": r.get("description", ""),
                }
            )
        return json.dumps(result, indent=2)

    @function_tool
    async def select_restaurant(self, context: RunContext, restaurant_name: str):
        """Select a restaurant to order from.

        Use this when the user indicates which restaurant they want to order from.

        Args:
            restaurant_name: The name of the restaurant (partial match works)
        """
        logger.info(f"Selecting restaurant: {restaurant_name}")

        restaurant = self.menu_data.find_restaurant_by_name(restaurant_name)
        if not restaurant:
            return f"Sorry, I couldn't find a restaurant matching '{restaurant_name}'. Let me show you the available options."

        self.selected_restaurant_id = str(restaurant.get("id"))
        name = restaurant.get("name")
        cuisine = restaurant.get("cuisine", "")

        # Get categories for this restaurant
        categories = self.menu_data.get_categories_for_restaurant(
            self.selected_restaurant_id
        )
        category_names = [c.get("name") for c in categories]

        return f"Selected {name} ({cuisine}). They have these menu categories: {', '.join(category_names)}. What would you like to know about?"

    @function_tool
    async def get_menu(self, context: RunContext, category: str = ""):
        """Get menu items from the selected restaurant, optionally filtered by category.

        Use this tool when the user asks about the menu, available items, or specific types of food.
        You should interpret the results and answer the user's question naturally - don't read
        the entire list unless specifically asked.

        Args:
            category: Optional category filter (e.g., "Appetizers", "Mains", "Desserts", "Drinks").
                     Leave empty to get all items.

        Returns structured data about menu items that you can use to answer questions.
        """
        if not self.selected_restaurant_id:
            return "Please select a restaurant first. Which restaurant would you like to order from?"

        logger.info(
            f"Fetching menu items for restaurant {self.selected_restaurant_id} (category: {category or 'all'})"
        )

        if category:
            # Filter by category name
            menu_items = self.menu_data.get_items_by_category_name(
                self.selected_restaurant_id, category
            )
        else:
            # Get all items for the restaurant
            menu_items = self.menu_data.get_items_for_restaurant(
                self.selected_restaurant_id
            )

        if not menu_items:
            if category:
                return f"No items found in the '{category}' category."
            return "No menu items found for this restaurant."

        # Return structured data for the LLM to interpret naturally
        result = []
        for item in menu_items:
            result.append(
                {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "price": item.get("price"),
                    "description": item.get("description", ""),
                    "category": item.get("categoryName", ""),
                }
            )
        return json.dumps(result, indent=2)

    @function_tool
    async def show_item(self, context: RunContext, item_name: str):
        """Show an image of a specific menu item to the user.

        Use this when the user asks to see what a dish looks like or wants a visual.

        Args:
            item_name: The name of the menu item (e.g., "Bruschetta", "pasta", "pizza")
                      Partial names work - will match the first item found.
        """
        logger.info(f"Showing image for item: {item_name}")

        # Find item, preferring the selected restaurant if one is chosen
        item = self.menu_data.find_item_by_name(item_name, self.selected_restaurant_id)

        if not item:
            return f"Sorry, I couldn't find '{item_name}' on the menu."

        # API uses 'image' field for image URLs
        image_url = item.get("image") or item.get("image_url") or item.get("imageUrl")
        if not image_url:
            return f"Sorry, I don't have an image for {item.get('name')}."

        # Send data message to frontend to display the image
        data_payload = json.dumps(
            {
                "type": "show_image",
                "url": image_url,
                "title": item.get("name"),
            }
        )

        # Use get_job_context().room to access the room from within function tools
        room = get_job_context().room
        await room.local_participant.publish_data(
            data_payload.encode("utf-8"),
            reliable=True,
        )

        return f"Showing you an image of {item.get('name')}"

    @function_tool
    async def place_order(
        self, context: RunContext, item_name: str, quantity: int = 1, notes: str = ""
    ):
        """Place an order for a menu item.

        Use this when the user wants to order a specific item.

        Args:
            item_name: The name of the menu item to order (e.g., "Mushroom Swiss Burger")
            quantity: How many of this item to order (default 1)
            notes: Any special requests or notes for the order (optional)
        """
        if not self.selected_restaurant_id:
            return "Please select a restaurant first before placing an order."

        logger.info(f"Placing order: {quantity}x {item_name} with notes: {notes}")

        # Find the item by name to get its ID
        item = self.menu_data.find_item_by_name(item_name, self.selected_restaurant_id)
        if not item:
            return f"Sorry, I couldn't find '{item_name}' on the menu. Please check the item name."

        # Build the order in the expected format
        order_items = [{"id": str(item.get("id")), "quantity": quantity}]

        # Get ORDER_API_URL from environment
        api_url = os.getenv("ORDER_API_URL")
        if not api_url:
            logger.warning("ORDER_API_URL not set, logging order locally")
            return f"Order received: {len(order_items)} items. Notes: {notes or 'None'}"

        # Send order to NextJS API
        room = get_job_context().room
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    api_url,
                    json={
                        "items": order_items,
                        "notes": notes,
                        "restaurant_id": self.selected_restaurant_id,
                        "room_id": room.name,
                    },
                    timeout=10.0,
                )
                response.raise_for_status()

            # Send order notification to frontend
            order_payload = json.dumps(
                {
                    "type": "order_notification",
                    "items": order_items,
                    "notes": notes,
                }
            )
            await room.local_participant.publish_data(
                order_payload.encode("utf-8"),
                reliable=True,
            )

            return f"Order successfully placed! {len(order_items)} items ordered."
        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            return "Sorry, there was an error submitting your order. Please try again."

    async def on_enter(self):
        # Greet and ask which restaurant they'd like to order from
        await self.session.generate_reply(
            instructions=f"""Greet the user warmly in {self.user_language}, say - Hello, how can I help you today?""",
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

    # Fetch menu data from API
    logger.info("Fetching menu data from API...")
    menu_data = await fetch_menu_data()
    set_menu_data(menu_data)  # Set global for any legacy code

    # Count total items across all restaurants and categories
    total_items = sum(
        len(cat.get("items", []))
        for r in menu_data.restaurants
        for cat in r.get("categories", [])
    )
    logger.info(
        f"Menu loaded: {len(menu_data.restaurants)} restaurants, "
        f"{total_items} total items"
    )

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
        agent=Assistant(menu_data=menu_data, user_language=user_language),
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
