"""Text-to-Speech service for Movie Mode character voices.

Supports multiple TTS providers:
- ElevenLabs (best quality, paid)
- OpenAI TTS (good quality, paid)
- Edge TTS (free, Microsoft Azure voices)
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Callable, Optional

from pydub import AudioSegment

from src.config import config
from src.models.schemas import Character, DialogueLine, Emotion, VoiceSettings

logger = logging.getLogger(__name__)


# Default voices for each provider
DEFAULT_VOICES = {
    "elevenlabs": {
        "male": "ErXwobaYiN019PkySvjV",  # Antoni
        "female": "EXAVITQu4vr4xnSDxMaL",  # Bella
        "narrator": "VR6AewLTigWG4xSOukaG",  # Arnold (deep, narrative)
    },
    "openai": {
        "male": "onyx",
        "female": "nova",
        "narrator": "echo",
    },
    "edge": {
        "male": "en-US-GuyNeural",
        "female": "en-US-AriaNeural",
        "narrator": "en-US-DavisNeural",
    },
}


class TTSService:
    """Text-to-Speech service supporting multiple providers."""

    def __init__(self, default_provider: str = "edge"):
        """Initialize TTS service.

        Args:
            default_provider: Default TTS provider (elevenlabs, openai, edge)
        """
        self.default_provider = default_provider
        self._elevenlabs_client = None
        self._openai_client = None

    def _get_elevenlabs_client(self):
        """Lazy-load ElevenLabs client."""
        if self._elevenlabs_client is None:
            try:
                from elevenlabs.client import ElevenLabs
            except ImportError as exc:
                raise ImportError(
                    "elevenlabs package not installed. Run: pip install elevenlabs"
                ) from exc

            api_key = config.elevenlabs_api_key
            if not api_key:
                raise ValueError(
                    "ELEVENLABS_API_KEY not set. "
                    "Add it to your .env file or use Edge TTS (free)."
                )
            self._elevenlabs_client = ElevenLabs(api_key=api_key)
        return self._elevenlabs_client

    def _get_openai_client(self):
        """Lazy-load OpenAI client."""
        if self._openai_client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "openai package not installed. Run: pip install openai"
                ) from exc

            api_key = config.openai_api_key
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not set. "
                    "Add it to your .env file or use Edge TTS (free)."
                )
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    def generate_speech(
        self,
        text: str,
        voice_settings: VoiceSettings,
        emotion: Emotion = Emotion.NEUTRAL,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Generate speech audio from text.

        Args:
            text: Text to convert to speech
            voice_settings: Voice configuration
            emotion: Emotion/delivery direction
            output_path: Output path for audio file (auto-generated if None)

        Returns:
            Path to generated audio file
        """
        provider = voice_settings.provider

        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix=".mp3"))

        if provider == "elevenlabs":
            return self._generate_elevenlabs(text, voice_settings, output_path)
        if provider == "openai":
            return self._generate_openai(text, voice_settings, output_path)
        if provider == "edge":
            return self._generate_edge(text, voice_settings, output_path)

        raise ValueError(f"Unknown TTS provider: {provider}")

    def _generate_elevenlabs(
        self,
        text: str,
        voice_settings: VoiceSettings,
        output_path: Path,
    ) -> Path:
        """Generate speech using ElevenLabs."""
        client = self._get_elevenlabs_client()

        # Get voice ID (use default if not specified)
        voice_id = voice_settings.voice_id
        if not voice_id:
            voice_id = DEFAULT_VOICES["elevenlabs"]["narrator"]

        try:
            # Generate audio using the new API
            audio = client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id="eleven_turbo_v2_5",
                voice_settings={
                    "stability": voice_settings.stability,
                    "similarity_boost": voice_settings.similarity_boost,
                },
            )

            # Save to file (audio is a generator of bytes)
            with open(output_path, "wb") as f:
                for chunk in audio:
                    f.write(chunk)

            logger.info("Generated ElevenLabs audio: %s", output_path)
            return output_path

        except Exception as e:
            logger.error("ElevenLabs TTS failed: %s", e)
            raise

    def _generate_openai(
        self,
        text: str,
        voice_settings: VoiceSettings,
        output_path: Path,
    ) -> Path:
        """Generate speech using OpenAI TTS."""
        client = self._get_openai_client()

        # Get voice (use default if not specified)
        voice = voice_settings.voice_id or DEFAULT_VOICES["openai"]["narrator"]

        try:
            response = client.audio.speech.create(
                model="tts-1-hd",
                voice=voice,
                input=text,
                speed=voice_settings.speed,
            )

            # Save to file
            response.stream_to_file(str(output_path))

            logger.info("Generated OpenAI audio: %s", output_path)
            return output_path

        except Exception as e:
            logger.error("OpenAI TTS failed: %s", e)
            raise

    def _generate_edge(
        self,
        text: str,
        voice_settings: VoiceSettings,
        output_path: Path,
    ) -> Path:
        """Generate speech using Edge TTS (free)."""
        try:
            import edge_tts
        except ImportError as exc:
            raise ImportError(
                "edge-tts package not installed. Run: pip install edge-tts"
            ) from exc

        # Get voice (use default if not specified)
        voice = voice_settings.voice_id or DEFAULT_VOICES["edge"]["narrator"]

        # Adjust rate (Edge TTS accepts percentage like "+10%" or "-10%")
        rate_pct = int((voice_settings.speed - 1) * 100)
        rate = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"

        async def _generate():
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=rate,
            )
            await communicate.save(str(output_path))

        # Run async
        asyncio.run(_generate())

        logger.info("Generated Edge TTS audio: %s", output_path)
        return output_path

    def generate_dialogue_audio(
        self,
        dialogue: DialogueLine,
        character: Character,
        output_dir: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Path:
        """Generate audio for a single dialogue line.

        Args:
            dialogue: The dialogue line to generate
            character: The character speaking
            output_dir: Directory to save audio
            progress_callback: Optional progress callback

        Returns:
            Path to generated audio file
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        text_hash = hash(dialogue.text) % 10000
        filename = f"dialogue_{character.id}_{text_hash:04d}.mp3"
        output_path = output_dir / filename

        if progress_callback:
            progress_callback(f"Generating voice for {character.name}...", 0.0)

        # Generate speech
        audio_path = self.generate_speech(
            text=dialogue.text,
            voice_settings=character.voice,
            emotion=dialogue.emotion,
            output_path=output_path,
        )

        return audio_path

    def get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of an audio file in seconds."""
        audio = AudioSegment.from_file(str(audio_path))
        return len(audio) / 1000.0

    def concatenate_audio(
        self,
        audio_paths: list[Path],
        output_path: Path,
        silence_between_ms: int = 500,
    ) -> Path:
        """Concatenate multiple audio files with silence between.

        Args:
            audio_paths: List of audio file paths
            output_path: Output path for combined audio
            silence_between_ms: Milliseconds of silence between clips

        Returns:
            Path to combined audio file
        """
        if not audio_paths:
            raise ValueError("No audio paths provided")

        # Create silence segment
        silence = AudioSegment.silent(duration=silence_between_ms)

        # Load and concatenate
        combined = AudioSegment.empty()
        for i, path in enumerate(audio_paths):
            audio = AudioSegment.from_file(str(path))
            combined += audio
            if i < len(audio_paths) - 1:
                combined += silence

        # Export
        combined.export(str(output_path), format="mp3")
        logger.info("Concatenated %d audio files to %s", len(audio_paths), output_path)

        return output_path


def check_elevenlabs_available() -> bool:
    """Check if ElevenLabs is available."""
    try:
        from elevenlabs.client import ElevenLabs  # noqa: F401
        return bool(config.elevenlabs_api_key)
    except ImportError:
        return False


def check_openai_tts_available() -> bool:
    """Check if OpenAI TTS is available."""
    try:
        from openai import OpenAI  # noqa: F401
        return bool(config.openai_api_key)
    except ImportError:
        return False


def check_edge_tts_available() -> bool:
    """Check if Edge TTS is available."""
    try:
        import edge_tts  # noqa: F401
        return True
    except ImportError:
        return False


def get_available_providers() -> list[str]:
    """Get list of available TTS providers."""
    providers = []
    # Edge TTS first since it's free
    if check_edge_tts_available():
        providers.append("edge")
    if check_elevenlabs_available():
        providers.append("elevenlabs")
    if check_openai_tts_available():
        providers.append("openai")
    return providers
