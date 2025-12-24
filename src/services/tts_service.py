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


# Default voices for each provider (used for auto-assignment)
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

# All available voices per provider (for voice picker)
# Updated December 2025 with latest voices
AVAILABLE_VOICES = {
    "openai": {
        # Original voices
        "alloy": "Alloy (Neutral)",
        "echo": "Echo (Male)",
        "fable": "Fable (British Male)",
        "onyx": "Onyx (Deep Male)",
        "nova": "Nova (Female)",
        "shimmer": "Shimmer (Female)",
        # New expressive voices (Oct 2024+)
        "ash": "Ash (Male, Expressive)",
        "ballad": "Ballad (Male, Expressive)",
        "coral": "Coral (Female, Expressive)",
        "sage": "Sage (Female, Expressive)",
        "verse": "Verse (Male, Expressive)",
    },
    "edge": {
        # Male voices
        "en-US-AndrewNeural": "Andrew (Male, Warm & Confident)",
        "en-US-BrianNeural": "Brian (Male, Casual & Sincere)",
        "en-US-ChristopherNeural": "Christopher (Male, Authoritative)",
        "en-US-EricNeural": "Eric (Male, Rational)",
        "en-US-GuyNeural": "Guy (Male, Passionate)",
        "en-US-RogerNeural": "Roger (Male, Lively)",
        "en-US-SteffanNeural": "Steffan (Male, Rational)",
        # Female voices
        "en-US-AnaNeural": "Ana (Female, Cute & Cartoon)",
        "en-US-AriaNeural": "Aria (Female, Positive & Confident)",
        "en-US-AvaNeural": "Ava (Female, Expressive & Caring)",
        "en-US-EmmaNeural": "Emma (Female, Cheerful & Clear)",
        "en-US-JennyNeural": "Jenny (Female, Friendly)",
        "en-US-MichelleNeural": "Michelle (Female, Pleasant)",
        # British voices
        "en-GB-RyanNeural": "Ryan (Male, British)",
        "en-GB-SoniaNeural": "Sonia (Female, British)",
        # Australian voices
        "en-AU-WilliamNeural": "William (Male, Australian)",
        "en-AU-NatashaNeural": "Natasha (Female, Australian)",
    },
    "elevenlabs": {
        "ErXwobaYiN019PkySvjV": "Antoni (Male)",
        "VR6AewLTigWG4xSOukaG": "Arnold (Male, Deep)",
        "pNInz6obpgDQGcFmaJgB": "Adam (Male)",
        "EXAVITQu4vr4xnSDxMaL": "Bella (Female)",
        "21m00Tcm4TlvDq8ikWAM": "Rachel (Female)",
        "AZnzlk1XvdvUeBnXmlld": "Domi (Female)",
    },
}

# Voice rotation for auto-assignment (to give different characters different voices)
# Alternates male/female for variety
VOICE_ROTATION = {
    "openai": ["onyx", "nova", "ash", "coral", "echo", "sage", "ballad", "shimmer", "verse", "alloy", "fable"],
    "edge": [
        "en-US-AndrewNeural", "en-US-AriaNeural", "en-US-BrianNeural",
        "en-US-AvaNeural", "en-US-GuyNeural", "en-US-EmmaNeural",
        "en-US-ChristopherNeural", "en-US-JennyNeural",
    ],
    "elevenlabs": [
        "ErXwobaYiN019PkySvjV", "EXAVITQu4vr4xnSDxMaL",
        "VR6AewLTigWG4xSOukaG", "21m00Tcm4TlvDq8ikWAM",
    ],
}


def get_available_voices_for_provider(provider: str) -> dict[str, str]:
    """Get available voices for a provider.

    Returns:
        Dict mapping voice_id to display name
    """
    return AVAILABLE_VOICES.get(provider, {})


def get_rotated_voice(provider: str, index: int) -> str:
    """Get a voice from the rotation based on character index.

    This ensures different characters get different voices automatically.
    """
    voices = VOICE_ROTATION.get(provider, [])
    if not voices:
        return DEFAULT_VOICES.get(provider, {}).get("narrator", "")
    return voices[index % len(voices)]


def infer_voice_id_from_name(voice_name: str, provider: str) -> str:
    """Infer a voice ID from a voice name description.

    Parses voice descriptions like "female, 30s, British accent" or
    "male narrator" to select an appropriate voice ID.

    Args:
        voice_name: Human-readable voice description
        provider: TTS provider name

    Returns:
        Voice ID for the specified provider
    """
    if not voice_name:
        return DEFAULT_VOICES.get(provider, {}).get("narrator", "")

    voice_lower = voice_name.lower()

    # Check for gender keywords
    if any(kw in voice_lower for kw in ["female", "woman", "girl", "she", "her"]):
        return DEFAULT_VOICES.get(provider, {}).get("female", "")
    elif any(kw in voice_lower for kw in ["male", "man", "boy", "he", "him"]):
        return DEFAULT_VOICES.get(provider, {}).get("male", "")
    elif any(kw in voice_lower for kw in ["narrator", "narration", "announcer"]):
        return DEFAULT_VOICES.get(provider, {}).get("narrator", "")

    # Default to narrator if no gender detected
    return DEFAULT_VOICES.get(provider, {}).get("narrator", "")


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

        # Get voice ID - infer from voice_name if not explicitly set
        voice_id = voice_settings.voice_id
        if not voice_id:
            voice_id = infer_voice_id_from_name(voice_settings.voice_name, "elevenlabs")
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

        # Get voice - infer from voice_name if not explicitly set
        voice = voice_settings.voice_id
        if not voice:
            voice = infer_voice_id_from_name(voice_settings.voice_name, "openai")
        if not voice:
            voice = DEFAULT_VOICES["openai"]["narrator"]

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

        # Get voice - infer from voice_name if not explicitly set
        voice = voice_settings.voice_id
        if not voice:
            voice = infer_voice_id_from_name(voice_settings.voice_name, "edge")
        if not voice:
            voice = DEFAULT_VOICES["edge"]["narrator"]

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

        # Generate unique filename using UUID to avoid collisions
        import uuid
        unique_id = uuid.uuid4().hex[:8]
        filename = f"dialogue_{character.id}_{unique_id}.mp3"
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

    def extract_word_timing(self, audio_path: Path, expected_text: str = None) -> list:
        """Extract word-level timing from an audio file using WhisperX.

        Args:
            audio_path: Path to audio file
            expected_text: Optional expected text (for alignment accuracy)

        Returns:
            List of Word objects with timing information
        """
        from src.models.schemas import Word

        try:
            from src.services.audio_processor import AudioProcessor
            processor = AudioProcessor()

            # Transcribe with word-level alignment
            transcript = processor.transcribe(audio_path, language="en")

            if transcript and transcript.all_words:
                return transcript.all_words
            else:
                logger.warning("No words extracted from %s", audio_path)
                return []

        except Exception as e:
            logger.warning("Failed to extract word timing from %s: %s", audio_path, e)
            return []

    def generate_dialogue_with_timing(
        self,
        dialogue: DialogueLine,
        character: Character,
        output_dir: Path,
        extract_words: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> tuple[Path, list]:
        """Generate audio for dialogue and extract word-level timing.

        Args:
            dialogue: The dialogue line to generate
            character: The character speaking
            output_dir: Directory to save audio
            extract_words: Whether to extract word timing
            progress_callback: Optional progress callback

        Returns:
            Tuple of (audio_path, list of Word objects)
        """
        # Generate the audio
        audio_path = self.generate_dialogue_audio(
            dialogue=dialogue,
            character=character,
            output_dir=output_dir,
            progress_callback=progress_callback,
        )

        # Extract word timing if requested
        words = []
        if extract_words:
            if progress_callback:
                progress_callback(f"Extracting word timing for {character.name}...", 0.5)
            words = self.extract_word_timing(audio_path, dialogue.text)

        return audio_path, words

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

    def mix_dialogue_with_timing(
        self,
        dialogue_items: list[dict],
        output_path: Path,
    ) -> Path:
        """Mix multiple dialogue audio files with precise timing for overlaps.

        Each dialogue_item should contain:
            - audio_path: Path to the audio file
            - start_time: When this audio should start (in seconds)

        This allows for overlapping dialogue (characters talking over each other).

        Args:
            dialogue_items: List of dicts with audio_path and start_time
            output_path: Output path for mixed audio

        Returns:
            Path to mixed audio file
        """
        if not dialogue_items:
            raise ValueError("No dialogue items provided")

        # Sort by start time
        sorted_items = sorted(dialogue_items, key=lambda x: x["start_time"])

        # Find total duration needed
        max_end_time = 0.0
        for item in sorted_items:
            audio = AudioSegment.from_file(str(item["audio_path"]))
            duration = len(audio) / 1000.0
            end_time = item["start_time"] + duration
            max_end_time = max(max_end_time, end_time)

        # Create a base track of silence
        total_duration_ms = int(max_end_time * 1000) + 100  # Add 100ms padding
        mixed = AudioSegment.silent(duration=total_duration_ms)

        # Overlay each audio at its start time
        for item in sorted_items:
            audio = AudioSegment.from_file(str(item["audio_path"]))
            start_ms = int(item["start_time"] * 1000)
            mixed = mixed.overlay(audio, position=start_ms)

        # Export
        mixed.export(str(output_path), format="mp3")
        logger.info(
            "Mixed %d dialogue tracks with overlaps to %s (%.1f seconds)",
            len(dialogue_items),
            output_path,
            max_end_time,
        )

        return output_path

    def calculate_overlap_timing(
        self,
        dialogues: list[DialogueLine],
        default_pause_ms: float = 300,
    ) -> list[float]:
        """Calculate start times for dialogues considering overlaps.

        Uses start_offset and interrupt_at_word to determine timing.
        Negative start_offset means overlap with previous dialogue.

        Args:
            dialogues: List of DialogueLine objects with timing info
            default_pause_ms: Default pause between lines in milliseconds

        Returns:
            List of start times in seconds for each dialogue
        """
        start_times = []
        current_time = 0.0

        for i, dialogue in enumerate(dialogues):
            if i == 0:
                # First dialogue starts at 0
                start_times.append(0.0)
            else:
                prev_dialogue = dialogues[i - 1]
                prev_duration = prev_dialogue.duration or 0.0
                prev_end = start_times[i - 1] + prev_duration

                # Check for interrupt_at_word (precise word-level interruption)
                if dialogue.interrupt_at_word is not None and prev_dialogue.has_word_timing:
                    word_idx = dialogue.interrupt_at_word
                    words = prev_dialogue.words

                    # Handle negative indices
                    if word_idx < 0:
                        word_idx = len(words) + word_idx

                    # Clamp to valid range
                    word_idx = max(0, min(word_idx, len(words) - 1))

                    if words and 0 <= word_idx < len(words):
                        # Start when that word starts in the previous dialogue
                        word_start = words[word_idx].start
                        start_time = start_times[i - 1] + word_start
                        start_times.append(start_time)
                        continue

                # Use start_offset (negative = overlap, positive = extra pause)
                offset_seconds = dialogue.start_offset
                natural_start = prev_end + (default_pause_ms / 1000)
                start_time = max(0, natural_start + offset_seconds)
                start_times.append(start_time)

        return start_times

    def apply_timing_to_scenes(
        self,
        scenes: list,
        default_pause_ms: float = 300,
        scene_pause_ms: float = 1000,
    ) -> float:
        """Apply timing to all dialogue and scenes, handling overlaps.

        This method:
        1. Calculates dialogue start times considering overlaps
        2. Sets start_time and end_time on each dialogue
        3. Sets start_time and end_time on each scene based on its dialogue

        Args:
            scenes: List of MovieScene objects with dialogue
            default_pause_ms: Default pause between dialogue lines in milliseconds
            scene_pause_ms: Pause between scenes in milliseconds

        Returns:
            Total duration in seconds
        """
        from src.models.schemas import MovieScene

        running_time = 0.0

        for scene_idx, scene in enumerate(scenes):
            if not scene.dialogue:
                # Empty scene - give it a default duration
                scene.start_time = running_time
                scene.end_time = running_time + 3.0  # 3 second default
                running_time = scene.end_time + (scene_pause_ms / 1000)
                continue

            # Mark scene start
            scene.start_time = running_time

            # Calculate start times for all dialogue in this scene
            # considering overlaps
            start_times = self.calculate_overlap_timing(
                scene.dialogue,
                default_pause_ms=default_pause_ms,
            )

            # Offset start times by scene start
            for i, dialogue in enumerate(scene.dialogue):
                if dialogue.duration is None:
                    # Need duration - try to get it from audio
                    if dialogue.audio_path:
                        dialogue_duration = self.get_audio_duration(
                            Path(dialogue.audio_path)
                        )
                    else:
                        # Estimate based on text length (~150 WPM)
                        word_count = len(dialogue.text.split())
                        dialogue_duration = word_count / 2.5  # ~2.5 words per second
                else:
                    dialogue_duration = dialogue.duration

                dialogue.start_time = running_time + start_times[i]
                dialogue.end_time = dialogue.start_time + dialogue_duration

            # Scene ends when last dialogue ends (accounting for overlaps)
            max_end = max(d.end_time for d in scene.dialogue if d.end_time)
            scene.end_time = max_end

            # Add scene pause before next scene
            running_time = scene.end_time + (scene_pause_ms / 1000)

        return running_time


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
