"""AtlasCloud Lipsync animation service.

This service applies lip sync to existing videos using AtlasCloud's
lip sync models. Unlike Wan2.2-S2V which takes image+audio,
these models take VIDEO+audio and sync the lips in the video.

Supported models:
- bytedance/lipsync/audio-to-video (ByteDance)
- kwaivgi/kling-lipsync/audio-to-video (Kling)

Workflow:
1. Generate a base video (e.g., with Seedance motion animation)
2. Apply lip sync with this model

Pricing: ~$0.0224/second ($0.112 base) for both models

Note: Uses base64 data URLs for video and audio inputs.
"""

import base64
import logging
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import requests
from pydub import AudioSegment

from src.config import Config, config as default_config

logger = logging.getLogger(__name__)


# Available lip sync models on AtlasCloud
LIPSYNC_MODELS = {
    "bytedance": "bytedance/lipsync/audio-to-video",
    "kling": "kwaivgi/kling-lipsync/audio-to-video",
}

# Model constraints
# Kling: video 2-10s, 720p/1080p, dimensions 720-1920px, audio max 5MB
# ByteDance: similar constraints


class AtlasCloudLipsyncAnimator:
    """Apply lip sync to videos using AtlasCloud's lip sync APIs.

    This service takes an existing video and audio, then generates a new
    video with synchronized lip movements.

    IMPORTANT: These models take VIDEO + AUDIO, not IMAGE + AUDIO.
    For image-to-video lip sync, use Wan2.2-S2V or Kling via fal.ai instead.

    Supported models:
    - bytedance: ByteDance Lipsync
    - kling: Kling Lipsync (KWAIVGI)

    Constraints (Kling):
    - Video: 2-10 seconds, 720p or 1080p
    - Video dimensions: 720-1920px
    - Audio: mp3/wav/m4a/aac, max 5MB

    Features:
    - Audio-driven lip sync on existing videos
    - Fast processing (no queue waits)
    - Paid service (~$0.11 for 5 seconds)

    Use cases:
    - Adding lip sync to motion-animated videos
    - Syncing lips in existing character videos
    - Post-processing step after initial video generation
    """

    BASE_URL = "https://api.atlascloud.ai/api/v1/model"

    def __init__(self, config: Optional[Config] = None, model: str = "kling"):
        self.config = config or default_config
        self._api_key = self.config.atlascloud_api_key
        # Use kling by default (more documented constraints)
        self.model = LIPSYNC_MODELS.get(model, LIPSYNC_MODELS["kling"])

    def _get_headers(self) -> dict:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _encode_video(self, video_path: Path) -> str:
        """Encode video to base64 data URL."""
        with open(video_path, "rb") as f:
            video_data = f.read()

        # Determine mime type
        suffix = video_path.suffix.lower()
        mime_map = {
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".mov": "video/quicktime",
        }
        mime_type = mime_map.get(suffix, "video/mp4")

        base64_data = base64.b64encode(video_data).decode("utf-8")
        return f"data:{mime_type};base64,{base64_data}"

    def _encode_audio(self, audio_path: Path) -> str:
        """Encode audio to base64 data URL."""
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        # Determine mime type
        suffix = audio_path.suffix.lower()
        mime_map = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".aac": "audio/aac",
        }
        mime_type = mime_map.get(suffix, "audio/wav")

        base64_data = base64.b64encode(audio_data).decode("utf-8")
        return f"data:{mime_type};base64,{base64_data}"

    def apply_lipsync(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        poll_interval: float = 2.0,
        max_wait_time: float = 600.0,
    ) -> Optional[Path]:
        """
        Apply lip sync to a video using ByteDance Lipsync.

        Args:
            video_path: Path to the input video file
            audio_path: Path to the audio file for lip sync
            output_path: Path to save the output video
            progress_callback: Optional callback for progress updates
            poll_interval: Seconds between status checks (default 2)
            max_wait_time: Maximum seconds to wait for completion (default 600)

        Returns:
            Path to the lip-synced video, or None if failed
        """
        if not self._api_key:
            logger.error("AtlasCloud API key not configured")
            if progress_callback:
                progress_callback("Error: AtlasCloud API key not set", 0.0)
            return None

        if progress_callback:
            progress_callback("Encoding video...", 0.1)

        try:
            # Encode video to base64 data URL
            video_data_url = self._encode_video(video_path)
            logger.info(f"Encoded video: {len(video_data_url)} chars")

            if progress_callback:
                progress_callback("Encoding audio...", 0.2)

            # Encode audio to base64 data URL
            audio_data_url = self._encode_audio(audio_path)
            logger.info(f"Encoded audio: {len(audio_data_url)} chars")

            if progress_callback:
                model_name = "Kling" if "kling" in self.model else "ByteDance"
                progress_callback(f"Submitting to {model_name} Lipsync...", 0.3)

            # Build request payload with base64 data URLs
            payload = {
                "model": self.model,
                "video": video_data_url,
                "audio": audio_data_url,
            }

            logger.info(f"AtlasCloud Lipsync request: model={self.model}, video={video_path.name}")

            # Submit generation request
            response = requests.post(
                f"{self.BASE_URL}/generateVideo",
                headers=self._get_headers(),
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()

            # Extract prediction ID
            data = result.get("data", {})
            prediction_id = data.get("id")
            if not prediction_id:
                logger.error(f"No prediction ID in response: {result}")
                if progress_callback:
                    progress_callback("Error: No request ID returned", 0.0)
                return None

            logger.info(f"ByteDance Lipsync request submitted: {prediction_id}")

            if progress_callback:
                progress_callback("Processing lip sync...", 0.4)

            # Poll for completion using /result endpoint
            poll_url = f"{self.BASE_URL}/result/{prediction_id}"
            start_poll_time = time.time()

            while True:
                elapsed = time.time() - start_poll_time
                if elapsed > max_wait_time:
                    logger.error(f"Timeout waiting for result: {prediction_id}")
                    if progress_callback:
                        progress_callback("Error: Request timed out", 0.0)
                    return None

                # Check status
                status_response = requests.get(
                    poll_url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    timeout=30,
                )
                status_response.raise_for_status()
                status_result = status_response.json()

                poll_data = status_result.get("data", {})
                status = poll_data.get("status", "unknown")
                logger.debug(f"Lipsync status: {status}")

                if status in ["completed", "succeeded"]:
                    # Get video URL from outputs
                    outputs = poll_data.get("outputs", [])
                    result_video_url = outputs[0] if outputs else None

                    if not result_video_url:
                        logger.error(f"No video URL in result: {status_result}")
                        if progress_callback:
                            progress_callback("Error: No video URL returned", 0.0)
                        return None

                    if progress_callback:
                        progress_callback("Downloading video...", 0.9)

                    # Download video
                    video_response = requests.get(result_video_url, timeout=120)
                    video_response.raise_for_status()

                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "wb") as f:
                        f.write(video_response.content)

                    logger.info(f"Lip-synced video saved to: {output_path}")

                    if progress_callback:
                        progress_callback("Lip sync complete!", 1.0)

                    return output_path

                elif status == "failed":
                    error_msg = poll_data.get("error") or "Unknown error"
                    logger.error(f"Lipsync generation failed: {error_msg}")
                    if progress_callback:
                        progress_callback(f"Error: {error_msg[:100]}", 0.0)
                    return None

                else:
                    # Still processing
                    progress = min(0.4 + (elapsed / max_wait_time) * 0.45, 0.85)
                    if progress_callback:
                        mins = int(elapsed // 60)
                        secs = int(elapsed % 60)
                        progress_callback(f"Processing ({mins}:{secs:02d})...", progress)
                    time.sleep(poll_interval)

        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            if e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg = error_detail.get("message", error_msg)
                except Exception:
                    pass
            logger.error(f"ByteDance Lipsync API error: {error_msg}")
            if progress_callback:
                progress_callback(f"API Error: {error_msg[:100]}", 0.0)
            return None

        except Exception as e:
            logger.error(f"ByteDance Lipsync failed: {e}", exc_info=True)
            if progress_callback:
                progress_callback(f"Error: {str(e)[:100]}", 0.0)
            return None

    def apply_lipsync_to_scene(
        self,
        video_path: Path,
        audio_path: Path,
        start_time: float,
        duration: float,
        output_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Optional[Path]:
        """
        Apply lip sync to a video using a segment of audio.

        Args:
            video_path: Path to the input video file
            audio_path: Path to the full audio file
            start_time: Start time in the audio (seconds)
            duration: Duration of the audio segment (seconds)
            output_path: Path to save the output video
            progress_callback: Optional callback for progress updates

        Returns:
            Path to the lip-synced video, or None if failed
        """
        if progress_callback:
            progress_callback("Preparing audio clip...", 0.05)

        # Extract the audio segment
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            audio_clip_path = temp_dir / "audio_clip.wav"

            try:
                logger.info(f"Extracting audio: {start_time:.2f}s to {start_time + duration:.2f}s")
                audio = AudioSegment.from_file(str(audio_path))
                start_ms = int(start_time * 1000)
                end_ms = int((start_time + duration) * 1000)
                clip = audio[start_ms:end_ms]
                clip.export(str(audio_clip_path), format="wav")

                return self.apply_lipsync(
                    video_path=video_path,
                    audio_path=audio_clip_path,
                    output_path=output_path,
                    progress_callback=progress_callback,
                )

            except Exception as e:
                logger.error(f"Failed to extract audio clip: {e}")
                if progress_callback:
                    progress_callback(f"Error: {str(e)[:100]}", 0.0)
                return None


def check_atlascloud_lipsync_available() -> bool:
    """Check if AtlasCloud Lipsync is available (API key configured)."""
    return bool(default_config.atlascloud_api_key)


# Backward compatibility alias
check_bytedance_lipsync_available = check_atlascloud_lipsync_available
ByteDanceLipsyncAnimator = AtlasCloudLipsyncAnimator
