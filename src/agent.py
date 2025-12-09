import json
import logging
import os

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
)
from livekit.plugins import google, liveavatar, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

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
        super().__init__(
            instructions=f"""You are a helpful restaurant booking assistant. The user is interacting with you via voice.
            You help users make reservations, check availability, and answer questions about the restaurant.
            Your responses are concise, friendly, and conversational.
            Keep your responses natural and to the point, without complex formatting, emojis, or asterisks.
            
            IMPORTANT: The user's preferred language is {user_language}. Start by speaking in {user_language}.
            If the user switches to a different language, follow their lead and respond in that language.
            Always match the language the user is currently speaking.""",
        )

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=f"""Greet the user warmly in {self.user_language} and ask how you can help them 
            with their restaurant reservation or inquiry. Keep it brief and natural.""",
            allow_interruptions=True,
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

    # Get user's preferred language from room metadata
    # Frontend sends metadata like: {"language": "tr"} or {"language": "en"}
    user_language = "English"  # Default fallback
    if ctx.room.metadata:
        try:
            metadata = json.loads(ctx.room.metadata)
            lang_code = metadata.get("language", "en")
            user_language = get_language_name(lang_code)
            logger.info(f"User language: {user_language} (from code: {lang_code})")
        except json.JSONDecodeError:
            # If metadata is plain text, try to convert it
            user_language = get_language_name(ctx.room.metadata)
            logger.info(f"User language from plain metadata: {user_language}")

    # Set up voice AI pipeline
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # Use the multilingual model for auto-detection of user's language
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=inference.STT(model="assemblyai/universal-streaming-multilingual"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(model="gemini-2.5-flash"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # cartesia/sonic-3 supports 40+ languages and auto-detects from text
        # The LiveAvatar plugin will automatically receive this audio for avatar generation
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",  # Jacqueline - works with multilingual
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # Initialize LiveAvatar if configured
    # Set LIVEAVATAR_API_KEY and LIVEAVATAR_AVATAR_ID in your .env file
    avatar_id = os.getenv("LIVEAVATAR_AVATAR_ID")

    if avatar_id:
        logger.info(f"Initializing LiveAvatar with avatar_id: {avatar_id}")
        avatar = liveavatar.AvatarSession(avatar_id=avatar_id)

        # Start the avatar and wait for it to join the room
        await avatar.start(session, room=ctx.room)
        logger.info("LiveAvatar started and joined the room")
    else:
        logger.info("LiveAvatar disabled (LIVEAVATAR_AVATAR_ID not set)")

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
