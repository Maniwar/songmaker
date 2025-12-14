"""Demucs vocal separation service for improved music transcription.

This service uses Meta's Demucs (Hybrid Demucs) to separate vocals from music,
which dramatically improves WhisperX transcription accuracy for songs with
heavy instrumentation.

NOTE: The separated vocals are ONLY used for transcription. The final video
always uses the original audio track to preserve full music quality.
"""

import logging
import tempfile
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def check_demucs_available() -> bool:
    """Check if Demucs is installed and available."""
    try:
        import demucs
        return True
    except ImportError:
        return False


class DemucsVocalSeparator:
    """Separate vocals from music using Hybrid Demucs.

    Uses the 'htdemucs' model which provides 4-stem separation:
    - drums
    - bass
    - other
    - vocals

    We extract just the vocals for transcription purposes.
    """

    def __init__(self, model_name: str = "htdemucs"):
        """Initialize the separator.

        Args:
            model_name: Demucs model to use. Options:
                - "htdemucs": Hybrid Transformer Demucs (recommended)
                - "htdemucs_ft": Fine-tuned version (higher quality, slower)
                - "mdx": MDX model (good for vocals)
                - "mdx_extra": MDX extra (highest quality, slowest)
        """
        self.model_name = model_name
        self._model = None
        self._device = None

    def _get_device(self) -> str:
        """Auto-detect best device for Demucs.

        Note: MPS support in Demucs/torchaudio is unreliable,
        so we only use CUDA or CPU.
        """
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            # MPS is not reliably supported by Demucs/torchaudio
            # elif torch.backends.mps.is_available():
            #     return "mps"
        except (ImportError, AttributeError):
            pass
        return "cpu"

    def _load_model(self):
        """Load the Demucs model (lazy loading)."""
        if self._model is not None:
            return

        try:
            from demucs.pretrained import get_model
            from demucs.apply import BagOfModels
            import torch

            self._device = self._get_device()
            logger.info(f"Loading Demucs model '{self.model_name}' on {self._device}...")

            self._model = get_model(self.model_name)

            # Handle bag of models (ensemble)
            if isinstance(self._model, BagOfModels):
                for sub_model in self._model.models:
                    sub_model.to(self._device)
            else:
                self._model.to(self._device)

            self._model.eval()
            logger.info("Demucs model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Demucs model: {e}")
            raise

    def separate_vocals(
        self,
        audio_path: Path,
        output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Optional[Path]:
        """Separate vocals from an audio file.

        Args:
            audio_path: Path to the input audio file (MP3, WAV, etc.)
            output_dir: Directory to save output. If None, uses temp directory.
            progress_callback: Optional callback for progress updates

        Returns:
            Path to the separated vocals audio file, or None if separation failed
        """
        if not check_demucs_available():
            logger.warning("Demucs not installed. Install with: pip install demucs")
            return None

        try:
            import torch
            import torchaudio
            from demucs.apply import apply_model
            from demucs.audio import convert_audio

            if progress_callback:
                progress_callback("Loading audio...", 0.1)

            # Load audio
            waveform, sample_rate = torchaudio.load(str(audio_path))

            if progress_callback:
                progress_callback("Loading Demucs model...", 0.2)

            # Load model
            self._load_model()

            # Convert audio to model's expected format
            # Demucs expects stereo audio at its sample rate
            model_sample_rate = self._model.samplerate
            model_channels = self._model.audio_channels

            if progress_callback:
                progress_callback("Preparing audio...", 0.3)

            # Convert to model format
            audio = convert_audio(
                waveform,
                sample_rate,
                model_sample_rate,
                model_channels
            )

            # Move to device
            audio = audio.to(self._device)

            # Add batch dimension
            audio = audio.unsqueeze(0)

            if progress_callback:
                progress_callback("Separating vocals (this may take a while)...", 0.4)

            # Apply separation
            with torch.no_grad():
                sources = apply_model(
                    self._model,
                    audio,
                    device=self._device,
                    progress=False,  # We handle progress ourselves
                    num_workers=0,
                )

            if progress_callback:
                progress_callback("Extracting vocals...", 0.8)

            # Get vocals (sources shape: [batch, stems, channels, samples])
            # Stem order for htdemucs: drums, bass, other, vocals
            stem_names = self._model.sources
            try:
                vocals_idx = stem_names.index("vocals")
            except ValueError:
                logger.error(f"Model doesn't have 'vocals' stem. Available: {stem_names}")
                return None

            vocals = sources[0, vocals_idx]  # [channels, samples]

            # Create output directory
            if output_dir is None:
                output_dir = Path(tempfile.mkdtemp(prefix="demucs_"))
            else:
                output_dir.mkdir(parents=True, exist_ok=True)

            # Save vocals
            output_path = output_dir / f"{audio_path.stem}_vocals.wav"

            if progress_callback:
                progress_callback("Saving vocals...", 0.9)

            torchaudio.save(
                str(output_path),
                vocals.cpu(),
                model_sample_rate
            )

            logger.info(f"Vocals separated and saved to: {output_path}")

            if progress_callback:
                progress_callback("Vocal separation complete!", 1.0)

            return output_path

        except Exception as e:
            logger.error(f"Vocal separation failed: {e}", exc_info=True)
            if progress_callback:
                progress_callback(f"Error: {str(e)[:100]}", 0.0)
            return None

    def cleanup(self):
        """Release model from memory."""
        if self._model is not None:
            del self._model
            self._model = None

            # Clear CUDA cache if applicable
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


def separate_vocals_for_transcription(
    audio_path: Path,
    output_dir: Optional[Path] = None,
    model_name: str = "htdemucs",
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Optional[Path]:
    """Convenience function to separate vocals for transcription.

    Args:
        audio_path: Path to the input audio file
        output_dir: Directory to save output. If None, uses temp directory.
        model_name: Demucs model to use
        progress_callback: Optional callback for progress updates

    Returns:
        Path to the separated vocals audio file, or None if separation failed
    """
    if not check_demucs_available():
        logger.warning(
            "Demucs not available. For better music transcription, install it with:\n"
            "  pip install demucs\n"
            "Proceeding with original audio..."
        )
        return None

    separator = DemucsVocalSeparator(model_name=model_name)
    try:
        return separator.separate_vocals(
            audio_path,
            output_dir=output_dir,
            progress_callback=progress_callback,
        )
    finally:
        separator.cleanup()
