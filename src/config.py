"""Configuration management for Songmaker."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load environment variables (override=True to ensure .env takes precedence)
load_dotenv(override=True)


def _get_default_whisper_device() -> str:
    """Auto-detect the best device for WhisperX.

    Priority:
    1. WHISPER_DEVICE env var if set (mps is translated to cpu)
    2. CUDA if available
    3. CPU as fallback

    Note: WhisperX uses CTranslate2/faster-whisper which only accepts "cpu" or "cuda".
    On Apple Silicon, "cpu" automatically uses the Accelerate framework for speed.
    If WHISPER_DEVICE=mps is set, we translate it to "cpu" since CTranslate2
    doesn't accept "mps" as a device string.
    """
    env_device = os.getenv("WHISPER_DEVICE")
    if env_device:
        # CTranslate2 doesn't accept "mps" - translate to "cpu"
        # (Apple Silicon acceleration via Accelerate framework is automatic with "cpu")
        if env_device.lower() == "mps":
            return "cpu"
        return env_device

    # Try CUDA
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    return "cpu"


@dataclass
class WhisperConfig:
    """WhisperX configuration."""

    model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "large-v3"))
    device: str = field(default_factory=_get_default_whisper_device)
    compute_type: str = field(
        default_factory=lambda: os.getenv("WHISPER_COMPUTE_TYPE", "float32")
    )
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("WHISPER_BATCH_SIZE", "16"))
    )


@dataclass
class VideoConfig:
    """Video generation configuration."""

    resolution: str = field(
        default_factory=lambda: os.getenv("VIDEO_RESOLUTION", "1920x1080")
    )
    fps: int = field(default_factory=lambda: int(os.getenv("VIDEO_FPS", "30")))

    @property
    def width(self) -> int:
        return int(self.resolution.split("x")[0])

    @property
    def height(self) -> int:
        return int(self.resolution.split("x")[1])


@dataclass
class ImageConfig:
    """Image generation configuration."""

    default_style: str = field(
        default_factory=lambda: os.getenv(
            "DEFAULT_ART_STYLE", "cinematic digital art, dramatic lighting, 8k quality"
        )
    )
    # Nano Banana Pro (Gemini 3 Pro Image) - highest quality
    # Alternatives: imagen-4.0-ultra-generate-001, gemini-2.5-flash-image-preview
    model: str = field(
        default_factory=lambda: os.getenv("IMAGE_MODEL", "gemini-3-pro-image-preview")
    )
    aspect_ratio: str = "16:9"
    image_size: str = field(
        default_factory=lambda: os.getenv("IMAGE_SIZE", "2K")
    )


@dataclass
class Config:
    """Main application configuration."""

    # API Keys
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    google_api_key: str = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY", "")
    )
    assemblyai_api_key: str = field(
        default_factory=lambda: os.getenv("ASSEMBLYAI_API_KEY", "")
    )
    fal_api_key: str = field(
        default_factory=lambda: os.getenv("FAL_KEY", "")
    )
    atlascloud_api_key: str = field(
        default_factory=lambda: os.getenv("ATLASCLOUD_API_KEY", "")
    )
    hf_token: Optional[str] = field(
        default_factory=lambda: os.getenv("HF_TOKEN")
    )

    # Lip sync backend: "wan2s2v" (free, slow) or "kling" (paid, fast)
    lip_sync_backend: str = field(
        default_factory=lambda: os.getenv("LIP_SYNC_BACKEND", "wan2s2v")
    )

    # Transcription backend: "whisperx" or "assemblyai"
    transcription_backend: str = field(
        default_factory=lambda: os.getenv("TRANSCRIPTION_BACKEND", "whisperx")
    )

    # Enable Demucs vocal separation for better music transcription
    # Set to "true" to preprocess audio through Demucs before WhisperX
    # Note: Only affects transcription, final video uses original audio
    use_demucs: bool = field(
        default_factory=lambda: os.getenv("USE_DEMUCS", "false").lower() == "true"
    )

    # Demucs model: "htdemucs" (default), "htdemucs_ft", "mdx", "mdx_extra"
    demucs_model: str = field(
        default_factory=lambda: os.getenv("DEMUCS_MODEL", "htdemucs")
    )

    # Claude model selection:
    # - claude-sonnet-4-5-20250929 (default, best quality)
    # - claude-3-5-haiku-20241022 (fastest, cheapest)
    # - claude-opus-4-5-20251101 (highest quality, most expensive)
    claude_model: str = field(
        default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
    )

    # Sub-configurations
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    image: ImageConfig = field(default_factory=ImageConfig)

    # Paths
    project_root: Path = field(
        default_factory=lambda: Path(__file__).parent.parent
    )

    @property
    def output_dir(self) -> Path:
        return self.project_root / os.getenv("OUTPUT_DIR", "output")

    @property
    def songs_dir(self) -> Path:
        return self.output_dir / "songs"

    @property
    def images_dir(self) -> Path:
        return self.output_dir / "images"

    @property
    def videos_dir(self) -> Path:
        return self.output_dir / "videos"

    @property
    def subtitles_dir(self) -> Path:
        return self.output_dir / "subtitles"

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        if not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is not set")
        if not self.google_api_key:
            errors.append("GOOGLE_API_KEY is not set")
        return errors

    def ensure_directories(self) -> None:
        """Create output directories if they don't exist."""
        for path in [
            self.songs_dir,
            self.images_dir,
            self.videos_dir,
            self.subtitles_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()
