"""Audio processing service using AssemblyAI for word-level transcription."""

from pathlib import Path
from typing import Callable, Optional

from pydub import AudioSegment

from src.config import Config, config as default_config
from src.models.schemas import Word, Segment, Transcript


class AssemblyAIProcessor:
    """Process audio files and extract word-level timestamps using AssemblyAI."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config
        self._validate_api_key()

    def _validate_api_key(self) -> None:
        """Validate that AssemblyAI API key is set."""
        if not self.config.assemblyai_api_key:
            raise ValueError(
                "ASSEMBLYAI_API_KEY is not set. "
                "Please set it in your .env file or environment variables."
            )

    def get_audio_duration(self, audio_path: Path) -> float:
        """Get the duration of an audio file in seconds."""
        audio = AudioSegment.from_file(str(audio_path))
        return len(audio) / 1000.0

    def transcribe(
        self,
        audio_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Transcript:
        """
        Transcribe audio file with word-level timestamps using AssemblyAI.

        Args:
            audio_path: Path to the audio file (MP3, WAV, etc.)
            progress_callback: Optional callback for progress updates (message, progress 0-1)

        Returns:
            Transcript with word-level timestamps
        """
        import assemblyai as aai

        if progress_callback:
            progress_callback("Initializing AssemblyAI...", 0.1)

        # Set API key
        aai.settings.api_key = self.config.assemblyai_api_key

        if progress_callback:
            progress_callback("Getting audio duration...", 0.15)

        duration = self.get_audio_duration(audio_path)

        if progress_callback:
            progress_callback("Uploading and transcribing audio...", 0.2)

        # Create transcriber and transcribe
        transcriber = aai.Transcriber()
        transcript_result = transcriber.transcribe(str(audio_path))

        if progress_callback:
            progress_callback("Processing transcription results...", 0.8)

        # Check for errors
        if transcript_result.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"AssemblyAI transcription failed: {transcript_result.error}")

        # Convert to our models
        # AssemblyAI returns timestamps in milliseconds, we need seconds
        words = []
        if transcript_result.words:
            for word_data in transcript_result.words:
                words.append(
                    Word(
                        word=word_data.text,
                        start=word_data.start / 1000.0,  # Convert ms to seconds
                        end=word_data.end / 1000.0,
                    )
                )

        # Create segments from sentences if available
        segments = []
        try:
            sentences = transcript_result.get_sentences()
            if sentences:
                for sentence in sentences:
                    # Find words that belong to this sentence
                    sentence_words = [
                        w for w in words
                        if w.start >= sentence.start / 1000.0 and w.end <= sentence.end / 1000.0 + 0.1
                    ]
                    segments.append(
                        Segment(
                            text=sentence.text.strip(),
                            start=sentence.start / 1000.0,
                            end=sentence.end / 1000.0,
                            words=sentence_words,
                        )
                    )
        except Exception:
            # If sentences not available, create single segment with all words
            if words:
                segments.append(
                    Segment(
                        text=transcript_result.text or "",
                        start=words[0].start if words else 0.0,
                        end=words[-1].end if words else duration,
                        words=words,
                    )
                )

        if progress_callback:
            progress_callback("Transcription complete!", 1.0)

        # Detect language (AssemblyAI defaults to English)
        language = "en"
        if hasattr(transcript_result, 'language_code') and transcript_result.language_code:
            language = transcript_result.language_code

        return Transcript(
            segments=segments,
            language=language,
            duration=duration,
        )

    def cleanup(self) -> None:
        """No cleanup needed for cloud API."""
        pass


def transcribe_audio_assemblyai(
    audio_path: Path,
    config: Optional[Config] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Transcript:
    """
    Convenience function to transcribe audio using AssemblyAI.

    Args:
        audio_path: Path to the audio file
        config: Optional configuration
        progress_callback: Optional callback for progress updates

    Returns:
        Transcript with word-level timestamps
    """
    processor = AssemblyAIProcessor(config)
    try:
        return processor.transcribe(audio_path, progress_callback)
    finally:
        processor.cleanup()
