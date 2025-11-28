"""Audio processing service with support for multiple transcription backends."""

import gc
from pathlib import Path
from typing import Callable, Literal, Optional

from pydub import AudioSegment

from src.config import Config, config as default_config
from src.models.schemas import Word, Segment, Transcript


TranscriptionBackend = Literal["whisperx", "assemblyai"]


class AudioProcessor:
    """Process audio files and extract word-level timestamps.

    Supports multiple transcription backends:
    - whisperx: Local processing using WhisperX (requires GPU for best performance)
    - assemblyai: Cloud-based processing using AssemblyAI API
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        backend: Optional[TranscriptionBackend] = None,
    ):
        """
        Initialize the audio processor.

        Args:
            config: Optional configuration object
            backend: Transcription backend to use. If None, uses config.transcription_backend
        """
        self.config = config or default_config
        self.backend = backend or self.config.transcription_backend

        # WhisperX model state (only used if backend is whisperx)
        self._model = None
        self._align_model = None
        self._align_metadata = None

    def _load_whisperx_models(self) -> None:
        """Lazy load WhisperX models."""
        if self._model is not None:
            return

        import whisperx

        whisper_config = self.config.whisper

        # Load transcription model
        self._model = whisperx.load_model(
            whisper_config.model,
            whisper_config.device,
            compute_type=whisper_config.compute_type,
        )

    def _load_align_model(self, language_code: str) -> None:
        """Load alignment model for the specified language."""
        if self._align_model is not None:
            return

        import whisperx

        self._align_model, self._align_metadata = whisperx.load_align_model(
            language_code=language_code,
            device=self.config.whisper.device,
        )

    def get_audio_duration(self, audio_path: Path) -> float:
        """Get the duration of an audio file in seconds."""
        audio = AudioSegment.from_file(str(audio_path))
        return len(audio) / 1000.0

    def transcribe(
        self,
        audio_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        backend: Optional[TranscriptionBackend] = None,
    ) -> Transcript:
        """
        Transcribe audio file with word-level timestamps.

        Args:
            audio_path: Path to the audio file (MP3, WAV, etc.)
            progress_callback: Optional callback for progress updates (message, progress 0-1)
            backend: Override the default backend for this transcription

        Returns:
            Transcript with word-level timestamps
        """
        use_backend = backend or self.backend

        if use_backend == "assemblyai":
            return self._transcribe_assemblyai(audio_path, progress_callback)
        else:
            return self._transcribe_whisperx(audio_path, progress_callback)

    def _transcribe_whisperx(
        self,
        audio_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Transcript:
        """Transcribe using WhisperX (local processing)."""
        import whisperx

        if progress_callback:
            progress_callback("Loading WhisperX models...", 0.1)

        self._load_whisperx_models()

        if progress_callback:
            progress_callback("Loading audio file...", 0.2)

        # Load audio
        audio = whisperx.load_audio(str(audio_path))
        duration = self.get_audio_duration(audio_path)

        if progress_callback:
            progress_callback("Transcribing audio...", 0.3)

        # Transcribe
        result = self._model.transcribe(
            audio,
            batch_size=self.config.whisper.batch_size,
        )

        language = result.get("language", "en")

        if progress_callback:
            progress_callback("Loading alignment model...", 0.5)

        # Load alignment model
        self._load_align_model(language)

        if progress_callback:
            progress_callback("Aligning words to audio...", 0.7)

        # Align for word-level timestamps
        result = whisperx.align(
            result["segments"],
            self._align_model,
            self._align_metadata,
            audio,
            self.config.whisper.device,
            return_char_alignments=False,
        )

        if progress_callback:
            progress_callback("Processing results...", 0.9)

        # Convert to our models
        segments = []
        for seg in result["segments"]:
            words = []
            for word_data in seg.get("words", []):
                # Skip words without timing (can happen with alignment issues)
                if "start" not in word_data or "end" not in word_data:
                    continue
                words.append(
                    Word(
                        word=word_data["word"],
                        start=word_data["start"],
                        end=word_data["end"],
                    )
                )

            segments.append(
                Segment(
                    text=seg["text"].strip(),
                    start=seg["start"],
                    end=seg["end"],
                    words=words,
                )
            )

        if progress_callback:
            progress_callback("Transcription complete!", 1.0)

        return Transcript(
            segments=segments,
            language=language,
            duration=duration,
        )

    def _transcribe_assemblyai(
        self,
        audio_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Transcript:
        """Transcribe using AssemblyAI (cloud processing)."""
        from src.services.assemblyai_processor import AssemblyAIProcessor

        processor = AssemblyAIProcessor(self.config)
        return processor.transcribe(audio_path, progress_callback)

    def cleanup(self) -> None:
        """Release model memory (only relevant for WhisperX)."""
        if self.backend != "whisperx":
            return

        try:
            import torch

            self._model = None
            self._align_model = None
            self._align_metadata = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            # torch not installed, nothing to clean up
            pass


def transcribe_audio(
    audio_path: Path,
    config: Optional[Config] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    backend: Optional[TranscriptionBackend] = None,
) -> Transcript:
    """
    Convenience function to transcribe audio.

    Args:
        audio_path: Path to the audio file
        config: Optional configuration
        progress_callback: Optional callback for progress updates
        backend: Transcription backend to use (whisperx or assemblyai)

    Returns:
        Transcript with word-level timestamps
    """
    processor = AudioProcessor(config, backend=backend)
    try:
        return processor.transcribe(audio_path, progress_callback)
    finally:
        processor.cleanup()


def get_available_backends(config: Optional[Config] = None) -> list[tuple[str, str]]:
    """
    Get list of available transcription backends.

    Returns:
        List of (value, label) tuples for use in dropdown menus
    """
    cfg = config or default_config
    backends = [("whisperx", "WhisperX (Local)")]

    if cfg.assemblyai_api_key:
        backends.append(("assemblyai", "AssemblyAI (Cloud)"))

    return backends
