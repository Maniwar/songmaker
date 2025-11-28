"""Configuration management for Songmaker."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class WhisperConfig:
    """WhisperX configuration."""

    model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "large-v3"))
    device: str = field(default_factory=lambda: os.getenv("WHISPER_DEVICE", "cpu"))
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
    hf_token: Optional[str] = field(
        default_factory=lambda: os.getenv("HF_TOKEN")
    )

    # Transcription backend: "whisperx" or "assemblyai"
    transcription_backend: str = field(
        default_factory=lambda: os.getenv("TRANSCRIPTION_BACKEND", "whisperx")
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
