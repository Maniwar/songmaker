"""Audio processing service with support for multiple transcription backends."""

import gc
import logging
import tempfile
from pathlib import Path
from typing import Callable, Literal, Optional

from pydub import AudioSegment

from src.config import Config, config as default_config
from src.models.schemas import Word, Segment, Transcript

logger = logging.getLogger(__name__)

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
        self._current_asr_options = None  # Track current model options

    def _load_whisperx_models(
        self,
        initial_prompt: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        """Lazy load WhisperX models.

        Args:
            initial_prompt: Optional prompt to improve transcription (e.g., known lyrics)
            language: Optional language code to use for transcription
        """
        import whisperx

        whisper_config = self.config.whisper

        # Build asr_options
        asr_options = {}
        if initial_prompt:
            asr_options["initial_prompt"] = initial_prompt

        # Check if we need to reload the model (different options)
        if self._model is not None:
            if asr_options == self._current_asr_options:
                return  # Model already loaded with same options
            else:
                # Need to reload with new options
                self._model = None

        # Load transcription model with asr_options
        load_kwargs = {
            "whisper_arch": whisper_config.model,
            "device": whisper_config.device,
            "compute_type": whisper_config.compute_type,
        }

        if asr_options:
            load_kwargs["asr_options"] = asr_options
            print(f"[WhisperX] Loading model with asr_options: {list(asr_options.keys())}")

        if language:
            load_kwargs["language"] = language
            print(f"[WhisperX] Language set to: {language}")

        print(f"[WhisperX] Load kwargs: {list(load_kwargs.keys())}")
        self._model = whisperx.load_model(**load_kwargs)
        self._current_asr_options = asr_options
        print("[WhisperX] Model loaded successfully")

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
        lyrics_hint: Optional[str] = None,
        language: Optional[str] = None,
        use_demucs: Optional[bool] = None,
    ) -> Transcript:
        """
        Transcribe audio file with word-level timestamps.

        Args:
            audio_path: Path to the audio file (MP3, WAV, etc.)
            progress_callback: Optional callback for progress updates (message, progress 0-1)
            backend: Override the default backend for this transcription
            lyrics_hint: Optional known lyrics to improve transcription accuracy.
                         For music, this dramatically improves recognition.
            language: Optional language code (e.g., 'en' for English).
                      If not provided, will be auto-detected.
            use_demucs: Whether to use Demucs vocal separation before transcription.
                       If None, uses config.use_demucs. Only applies to WhisperX backend.
                       NOTE: The separated vocals are only for transcription; final video
                       uses the original audio.

        Returns:
            Transcript with word-level timestamps
        """
        use_backend = backend or self.backend

        if use_backend == "assemblyai":
            return self._transcribe_assemblyai(audio_path, progress_callback)
        else:
            return self._transcribe_whisperx(
                audio_path, progress_callback, lyrics_hint, language, use_demucs
            )

    def _transcribe_whisperx(
        self,
        audio_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        lyrics_hint: Optional[str] = None,
        language: Optional[str] = None,
        use_demucs: Optional[bool] = None,
    ) -> Transcript:
        """Transcribe using WhisperX (local processing).

        Args:
            audio_path: Path to audio file
            progress_callback: Optional progress callback
            lyrics_hint: Known lyrics to improve transcription accuracy.
                        Passed to WhisperX via asr_options when loading the model.
            language: Language code (e.g., 'en'). If None, auto-detected.
            use_demucs: Whether to use Demucs vocal separation. If None, uses config.
        """
        import os
        import re
        import shutil
        import whisperx

        # Determine whether to use Demucs preprocessing
        should_use_demucs = use_demucs if use_demucs is not None else self.config.use_demucs
        vocals_path: Optional[Path] = None
        temp_dir: Optional[Path] = None

        # If Demucs is enabled, separate vocals first
        if should_use_demucs:
            if progress_callback:
                progress_callback("Separating vocals with Demucs (for better transcription)...", 0.05)

            from src.services.demucs_separator import (
                check_demucs_available,
                separate_vocals_for_transcription,
            )

            if check_demucs_available():
                temp_dir = Path(tempfile.mkdtemp(prefix="demucs_transcription_"))

                def demucs_progress(msg: str, prog: float):
                    # Map Demucs progress (0-1) to our progress range (0.05-0.25)
                    if progress_callback:
                        mapped_prog = 0.05 + (prog * 0.20)
                        progress_callback(f"Demucs: {msg}", mapped_prog)

                vocals_path = separate_vocals_for_transcription(
                    audio_path=audio_path,
                    output_dir=temp_dir,
                    model_name=self.config.demucs_model,
                    progress_callback=demucs_progress,
                )

                if vocals_path and vocals_path.exists():
                    logger.info(f"Using separated vocals for transcription: {vocals_path}")
                else:
                    logger.warning("Demucs vocal separation failed, using original audio")
                    vocals_path = None
            else:
                logger.warning(
                    "Demucs not installed but USE_DEMUCS=true. "
                    "Install with: pip install demucs"
                )

        # Use separated vocals for transcription if available, otherwise original
        transcription_audio_path = vocals_path if vocals_path else audio_path

        try:
            if progress_callback:
                base_progress = 0.25 if should_use_demucs else 0.1
                if lyrics_hint:
                    progress_callback("Loading WhisperX with lyrics hint...", base_progress)
                else:
                    progress_callback("Loading WhisperX models...", base_progress)

            # Clean up lyrics for use as initial prompt
            clean_lyrics = None
            if lyrics_hint:
                # Remove section markers like [Verse], [Chorus], etc.
                clean_lyrics = re.sub(r'\[.*?\]', '', lyrics_hint)
                clean_lyrics = ' '.join(clean_lyrics.split())  # Normalize whitespace
                # Log that we're using lyrics hint
                print(f"[WhisperX] Using lyrics hint ({len(clean_lyrics)} chars): {clean_lyrics[:100]}...")

            # Load model with lyrics hint (passed via asr_options)
            self._load_whisperx_models(initial_prompt=clean_lyrics, language=language)

            if progress_callback:
                progress_callback("Loading audio file...", 0.3 if should_use_demucs else 0.2)

            # Load audio for transcription (use separated vocals if available)
            # But ALWAYS use original audio_path for duration (final video uses original audio)
            audio = whisperx.load_audio(str(transcription_audio_path))
            duration = self.get_audio_duration(audio_path)  # Use original for duration

            if progress_callback:
                progress_msg = "Transcribing "
                if vocals_path:
                    progress_msg += "separated vocals"
                else:
                    progress_msg += "audio"
                if lyrics_hint:
                    progress_msg += " with lyrics hint..."
                else:
                    progress_msg += "..."
                progress_callback(progress_msg, 0.4 if should_use_demucs else 0.3)

            # Build transcribe options (language passed at model load time)
            transcribe_options = {
                "batch_size": self.config.whisper.batch_size,
            }

            # Transcribe
            result = self._model.transcribe(audio, **transcribe_options)

            detected_language = language or result.get("language", "en")

            if progress_callback:
                progress_callback("Loading alignment model...", 0.6 if should_use_demucs else 0.5)

            # Load alignment model
            self._load_align_model(detected_language)

            if progress_callback:
                progress_callback("Aligning words to audio...", 0.75 if should_use_demucs else 0.7)

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
                language=detected_language,
                duration=duration,
            )

        finally:
            # Clean up Demucs temp files
            if temp_dir and temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up Demucs temp directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")

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
    lyrics_hint: Optional[str] = None,
    language: Optional[str] = None,
    use_demucs: Optional[bool] = None,
) -> Transcript:
    """
    Convenience function to transcribe audio.

    Args:
        audio_path: Path to the audio file
        config: Optional configuration
        progress_callback: Optional callback for progress updates
        backend: Transcription backend to use (whisperx or assemblyai)
        lyrics_hint: Known lyrics to improve transcription accuracy.
                    Dramatically helps with music where vocals are mixed with instruments.
        language: Language code (e.g., 'en'). If None, auto-detected.
        use_demucs: Whether to use Demucs vocal separation before transcription.
                   If None, uses config.use_demucs. Only applies to WhisperX backend.
                   NOTE: Separated vocals are only for transcription; final video
                   uses the original audio.

    Returns:
        Transcript with word-level timestamps
    """
    processor = AudioProcessor(config, backend=backend)
    try:
        return processor.transcribe(
            audio_path,
            progress_callback,
            lyrics_hint=lyrics_hint,
            language=language,
            use_demucs=use_demucs,
        )
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
