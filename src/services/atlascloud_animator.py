"""AtlasCloud animation service using WAN 2.6 and Seedance 1.5 Pro.

This service generates motion videos from images/videos using the AtlasCloud API,
which hosts the WAN 2.6 and Seedance 1.5 Pro models.

WAN 2.6 Features:
- Text-to-video, Image-to-video, Video-to-video
- Up to 15 seconds duration (5, 10, or 15)
- 480p, 720p, or 1080p resolution
- Native audio generation with lip sync
- Multi-language support
- First frame / Last frame control

Seedance 1.5 Pro Features:
- Precision lip-syncing with audio
- Multi-language support
- Cinematic camera control
- Native audio generation
"""

import base64
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import requests

from src.config import Config, config as default_config

logger = logging.getLogger(__name__)


class WanModel(str, Enum):
    """Available WAN 2.6 models on AtlasCloud."""
    TEXT_TO_VIDEO = "alibaba/wan-2.6/text-to-video"
    IMAGE_TO_VIDEO = "alibaba/wan-2.6/image-to-video"
    VIDEO_TO_VIDEO = "alibaba/wan-2.6/video-to-video"
    # Legacy models for backwards compatibility
    WAN_25_FAST = "alibaba/wan-2.5/image-to-video-fast"


class SeedanceModel(str, Enum):
    """Available Seedance 1.5 Pro models on AtlasCloud."""
    TEXT_TO_VIDEO = "bytedance/seedance-v1.5-pro/text-to-video"
    IMAGE_TO_VIDEO = "bytedance/seedance-v1.5-pro/image-to-video"


class AtlasCloudAnimator:
    """Generate animations using AtlasCloud's WAN 2.6 or Seedance 1.5 Pro APIs.

    This service uses WAN 2.6 or Seedance 1.5 Pro via AtlasCloud's API to animate
    images/videos based on text prompts describing the motion.

    WAN 2.6 Features:
    - Text-to-video, Image-to-video, Video-to-video
    - Up to 15 seconds duration (5, 10, or 15)
    - 480p, 720p, or 1080p resolution
    - Native audio with multi-person dialogue
    - First frame / Last frame control for smooth transitions

    Seedance 1.5 Pro Features:
    - Precision lip-syncing with millisecond accuracy
    - Multi-language support with natural speech
    - Cinematic camera control
    - Native audio generation

    Pricing (as of Dec 2025):
    - WAN 2.6: $0.075/second
    - Seedance 1.5 Pro: $0.0147/second
    """

    BASE_URL = "https://api.atlascloud.ai/api/v1/model"
    # Default to WAN 2.6 image-to-video
    MODEL = WanModel.IMAGE_TO_VIDEO

    def __init__(
        self,
        config: Optional[Config] = None,
        model: Optional[str] = None,
    ):
        """Initialize AtlasCloud animator.

        Args:
            config: Configuration object
            model: Model to use. Options:
                - WanModel.IMAGE_TO_VIDEO (default)
                - WanModel.TEXT_TO_VIDEO
                - WanModel.VIDEO_TO_VIDEO
                - SeedanceModel.IMAGE_TO_VIDEO
                - SeedanceModel.TEXT_TO_VIDEO
        """
        self.config = config or default_config
        self._api_key = self.config.atlascloud_api_key
        self._model = model or WanModel.IMAGE_TO_VIDEO

    def _get_headers(self) -> dict:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _encode_file(self, file_path: Path) -> str:
        """Encode image or video to base64 data URL."""
        with open(file_path, "rb") as f:
            file_data = f.read()

        # Determine mime type
        suffix = file_path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".mov": "video/quicktime",
        }
        mime_type = mime_map.get(suffix, "application/octet-stream")

        base64_data = base64.b64encode(file_data).decode("utf-8")
        return f"data:{mime_type};base64,{base64_data}"

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64 data URL (legacy compatibility)."""
        return self._encode_file(image_path)

    def animate_scene(
        self,
        image_path: Path,
        prompt: str,
        output_path: Path,
        duration_seconds: float = 5.0,
        resolution: str = "720p",
        seed: int = -1,
        model: Optional[str] = None,
        first_frame: Optional[Path] = None,
        last_frame: Optional[Path] = None,
        source_video: Optional[Path] = None,
        source_video_urls: Optional[list[str]] = None,
        audio_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        poll_interval: float = 5.0,
        max_wait_time: float = 300.0,
    ) -> Optional[Path]:
        """
        Animate a scene with WAN 2.6 or Seedance 1.5 Pro.

        Args:
            image_path: Path to the scene image (for image-to-video)
            prompt: Text description of the motion
            output_path: Path to save the output video
            duration_seconds: Video duration (5, 10, or 15 seconds for WAN 2.6)
            resolution: Output resolution ("480p", "720p", or "1080p")
            seed: Random seed (-1 for random)
            model: Override model (WanModel or SeedanceModel)
            first_frame: Optional first frame image for smooth transitions
            last_frame: Optional last frame image for smooth transitions
            source_video: Source video for video-to-video mode (local path)
            source_video_urls: List of video URLs for V2V mode (WAN 2.6 V2V requires URLs)
            audio_path: Optional audio file for lip sync (Seedance 1.5 Pro)
            progress_callback: Optional callback for progress updates
            poll_interval: Seconds between status checks (default 5)
            max_wait_time: Maximum seconds to wait for completion (default 300)

        Returns:
            Path to the generated video, or None if generation failed
        """
        if not self._api_key:
            logger.error("AtlasCloud API key not configured")
            if progress_callback:
                progress_callback("Error: AtlasCloud API key not set", 0.0)
            return None

        # Select model
        active_model = model or self._model
        is_seedance = "seedance" in str(active_model).lower()
        is_video_to_video = source_video_urls is not None or source_video is not None or "video-to-video" in str(active_model)

        # Validate duration
        # WAN 2.6 I2V/T2V supports 5, 10, 15 seconds; V2V supports 5, 10 seconds
        # Seedance supports 3-12 seconds
        if is_seedance:
            valid_durations = list(range(3, 13))  # 3-12 seconds per API spec
        elif is_video_to_video:
            valid_durations = [5, 10]  # WAN 2.6 V2V only supports 5 or 10
        else:
            valid_durations = [5, 10, 15]  # WAN 2.6 I2V/T2V

        duration = int(duration_seconds)
        if duration not in valid_durations:
            if is_seedance:
                duration = max(3, min(15, duration))
            else:
                # Round to nearest valid WAN 2.6 duration
                if duration < 7:
                    duration = 5
                elif duration < 12:
                    duration = 10
                else:
                    duration = 15
            logger.warning(
                f"Duration {duration_seconds}s adjusted to {duration}s"
            )

        # Validate resolution - Seedance supports 480p/720p, WAN 2.6 supports 720p/1080p
        if is_seedance:
            valid_resolutions = ["480p", "720p"]
            if resolution not in valid_resolutions:
                resolution = "720p"
                logger.warning(f"Seedance only supports 480p/720p, using {resolution}")
        else:
            # WAN 2.6 only supports 720p and 1080p (no 480p per API spec)
            valid_resolutions = ["720p", "1080p"]
            if resolution not in valid_resolutions:
                resolution = "720p"
                logger.warning(f"WAN 2.6 only supports 720p/1080p, using {resolution}")

        if progress_callback:
            progress_callback("Encoding media...", 0.1)

        try:
            # Determine which model to actually use based on inputs
            if is_video_to_video and source_video:
                actual_model = WanModel.VIDEO_TO_VIDEO
            elif is_seedance:
                actual_model = (
                    SeedanceModel.IMAGE_TO_VIDEO
                    if image_path
                    else SeedanceModel.TEXT_TO_VIDEO
                )
            else:
                actual_model = (
                    WanModel.IMAGE_TO_VIDEO
                    if image_path
                    else WanModel.TEXT_TO_VIDEO
                )

            if progress_callback:
                progress_callback("Submitting to AtlasCloud...", 0.2)

            # Build request payload - different format for V2V vs I2V/T2V
            if is_video_to_video and not is_seedance:
                # WAN 2.6 Video-to-Video uses different API format
                # Size format: "1280*720" instead of "720p"
                size_map = {
                    "720p": "1280*720",
                    "1080p": "1920*1080",
                    "480p": "854*480",
                }
                v2v_size = size_map.get(resolution, "1280*720")

                payload = {
                    "model": WanModel.VIDEO_TO_VIDEO.value,
                    "prompt": prompt,
                    "size": v2v_size,
                    "duration": duration,
                    "enable_prompt_expansion": False,  # Disable for precise control
                    "shot_type": "single",  # Single camera angle for consistency
                }

                # V2V requires video URLs (not base64)
                if source_video_urls:
                    payload["videos"] = source_video_urls[:3]  # Max 3 videos
                    logger.info(f"Using {len(payload['videos'])} video URLs for V2V")
                else:
                    logger.warning("V2V mode selected but no video URLs provided - V2V requires URLs, not local files")
                    # Fall back to I2V if we have an image
                    if image_path and image_path.exists():
                        logger.info("Falling back to I2V mode")
                        is_video_to_video = False
                        payload = {
                            "model": WanModel.IMAGE_TO_VIDEO.value,
                            "prompt": prompt,
                            "resolution": resolution,
                            "duration": duration,
                        }
                        payload["image"] = self._encode_file(image_path)

                if seed >= 0:
                    payload["seed"] = seed
                else:
                    payload["seed"] = -1

            else:
                # Standard I2V/T2V payload format
                payload = {
                    "model": str(actual_model.value if hasattr(actual_model, 'value') else actual_model),
                    "prompt": prompt,
                    "resolution": resolution,
                    "duration": duration,
                }

                # Add image for image-to-video
                if image_path and image_path.exists():
                    payload["image"] = self._encode_file(image_path)

                # Add last frame for Seedance transitions (WAN 2.6 doesn't support first/last frame per API spec)
                if last_frame and last_frame.exists() and is_seedance:
                    payload["last_image"] = self._encode_file(last_frame)

                # Seedance-specific parameters
                if is_seedance:
                    payload["aspect_ratio"] = "16:9"  # Default to 16:9 for video
                    payload["camera_fixed"] = False   # Allow camera movement
                    payload["generate_audio"] = True  # Generate audio with video
                else:
                    # WAN 2.6 specific parameters
                    payload["enable_prompt_expansion"] = False  # Disable prompt expansion for precise control
                    payload["shot_type"] = "single"  # Single camera angle for consistency
                    payload["generate_audio"] = True  # Auto-generate audio

                # Add audio for Seedance lip sync (if provided separately)
                if audio_path and audio_path.exists() and is_seedance:
                    payload["audio"] = self._encode_file(audio_path)
                    logger.info("Added audio for lip sync")

                if seed >= 0:
                    payload["seed"] = seed
                elif not is_seedance:
                    # WAN 2.6 defaults to seed=0, use -1 for random
                    payload["seed"] = -1

            logger.info(f"Using model: {payload['model']}, duration: {duration}s")

            # Submit generation request
            response = requests.post(
                f"{self.BASE_URL}/generateVideo",
                headers=self._get_headers(),
                json=payload,
                timeout=180,  # 3 minutes for initial request
            )
            response.raise_for_status()
            result = response.json()

            # Extract prediction ID from response: result["data"]["id"]
            data = result.get("data", {})
            prediction_id = data.get("id")
            if not prediction_id:
                logger.error(f"No prediction ID in response: {result}")
                if progress_callback:
                    progress_callback("Error: No request ID returned", 0.0)
                return None

            logger.info(f"AtlasCloud request submitted: {prediction_id}")

            if progress_callback:
                progress_callback("Processing animation...", 0.3)

            # Poll for completion using /result/{id} endpoint (per API spec)
            poll_url = f"{self.BASE_URL}/result/{prediction_id}"
            start_time = time.time()
            consecutive_timeouts = 0
            max_consecutive_timeouts = 5  # Allow up to 5 timeout retries

            while True:
                elapsed = time.time() - start_time
                if elapsed > max_wait_time:
                    logger.error(f"Timeout waiting for AtlasCloud result: {prediction_id}")
                    if progress_callback:
                        progress_callback("Error: Request timed out", 0.0)
                    return None

                # Check status with retry on timeout
                try:
                    status_response = requests.get(
                        poll_url,
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        timeout=120,  # 2 minutes for poll requests (increased from 60s)
                    )
                    status_response.raise_for_status()
                    status_result = status_response.json()
                    consecutive_timeouts = 0  # Reset on success
                except requests.exceptions.ReadTimeout:
                    consecutive_timeouts += 1
                    logger.warning(f"Poll request timed out ({consecutive_timeouts}/{max_consecutive_timeouts}), retrying...")
                    if consecutive_timeouts >= max_consecutive_timeouts:
                        logger.error(f"Too many consecutive timeouts for: {prediction_id}")
                        if progress_callback:
                            progress_callback("Error: Poll requests keep timing out", 0.0)
                        return None
                    # Wait a bit before retry
                    time.sleep(poll_interval)
                    continue
                except requests.exceptions.ConnectionError as e:
                    consecutive_timeouts += 1
                    logger.warning(f"Connection error during poll ({consecutive_timeouts}/{max_consecutive_timeouts}): {e}")
                    if consecutive_timeouts >= max_consecutive_timeouts:
                        logger.error(f"Too many connection errors for: {prediction_id}")
                        if progress_callback:
                            progress_callback("Error: Connection errors during polling", 0.0)
                        return None
                    time.sleep(poll_interval)
                    continue

                # Status is in result["data"]["status"]
                poll_data = status_result.get("data", {})
                status = poll_data.get("status", "unknown")
                logger.debug(f"AtlasCloud status: {status}")

                if status in ["completed", "succeeded"]:
                    # Get video URL from result["data"]["outputs"][0]
                    outputs = poll_data.get("outputs", [])
                    video_url = outputs[0] if outputs else None

                    if not video_url:
                        logger.error(f"No video URL in completed result: {status_result}")
                        if progress_callback:
                            progress_callback("Error: No video URL returned", 0.0)
                        return None

                    if progress_callback:
                        progress_callback("Downloading video...", 0.9)

                    # Download video
                    video_response = requests.get(video_url, timeout=300)  # 5 minutes for download
                    video_response.raise_for_status()

                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "wb") as f:
                        f.write(video_response.content)

                    logger.info(f"Animation saved to: {output_path}")

                    if progress_callback:
                        progress_callback("Animation complete!", 1.0)

                    return output_path

                elif status == "failed":
                    error_msg = poll_data.get("error") or "Unknown error"
                    logger.error(f"AtlasCloud generation failed: {error_msg}")
                    if progress_callback:
                        progress_callback(f"Error: {error_msg[:100]}", 0.0)
                    return None

                else:
                    # Still processing (starting, processing, queued, etc.)
                    progress = min(0.3 + (elapsed / max_wait_time) * 0.5, 0.85)
                    if progress_callback:
                        progress_callback(f"Processing ({status})...", progress)
                    time.sleep(poll_interval)

        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            if e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg = error_detail.get("message", error_msg)
                except Exception:
                    pass
            logger.error(f"AtlasCloud API error: {error_msg}")
            if progress_callback:
                progress_callback(f"API Error: {error_msg[:100]}", 0.0)
            return None

        except Exception as e:
            logger.error(f"AtlasCloud animation failed: {e}", exc_info=True)
            if progress_callback:
                progress_callback(f"Error: {str(e)[:100]}", 0.0)
            return None

    def animate_scenes(
        self,
        scenes: list,
        output_dir: Path,
        duration_seconds: float = 5.0,
        resolution: str = "720p",
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> dict[int, Optional[Path]]:
        """
        Animate multiple scenes marked for AtlasCloud animation.

        Args:
            scenes: List of Scene objects (only those with animation_type=ATLASCLOUD
                    will be processed)
            output_dir: Directory to save output videos
            duration_seconds: Video duration per scene (5 or 10 seconds)
            resolution: Output resolution ("720p" or "1080p")
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping scene index to output video path (or None if failed)
        """
        from src.models.schemas import AnimationType

        # Filter to only AtlasCloud-animated scenes with images
        atlascloud_scenes = [
            s for s in scenes
            if getattr(s, 'animation_type', None) == AnimationType.ATLASCLOUD
            and s.image_path and Path(s.image_path).exists()
        ]

        if not atlascloud_scenes:
            return {}

        results = {}
        total = len(atlascloud_scenes)

        for i, scene in enumerate(atlascloud_scenes):
            if progress_callback:
                progress_callback(
                    f"Animating scene {scene.index + 1} ({i + 1}/{total})...",
                    i / total
                )

            output_path = output_dir / f"animated_scene_{scene.index:03d}.mp4"

            # Use motion_prompt if available, otherwise fall back to visual_prompt
            prompt = getattr(scene, 'motion_prompt', None) or scene.visual_prompt

            def scene_progress(msg: str, prog: float):
                if progress_callback:
                    overall = (i + prog) / total
                    progress_callback(f"Scene {scene.index + 1}: {msg}", overall)

            result = self.animate_scene(
                image_path=Path(scene.image_path),
                prompt=prompt,
                output_path=output_path,
                duration_seconds=duration_seconds,
                resolution=resolution,
                progress_callback=scene_progress,
            )

            results[scene.index] = result

        if progress_callback:
            success_count = sum(1 for v in results.values() if v is not None)
            progress_callback(
                f"Animated {success_count}/{total} scenes",
                1.0
            )

        return results


    def video_to_video(
        self,
        source_video: Path,
        prompt: str,
        output_path: Path,
        duration_seconds: float = 5.0,
        resolution: str = "720p",
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Optional[Path]:
        """Transform a video using WAN 2.6 video-to-video.

        Applies style or motion transformations to an existing video.

        Args:
            source_video: Path to the source video
            prompt: Description of the transformation
            output_path: Path to save the output video
            duration_seconds: Output duration (5, 10, or 15 seconds)
            resolution: Output resolution ("480p", "720p", or "1080p")
            progress_callback: Optional progress callback

        Returns:
            Path to the transformed video, or None if failed
        """
        return self.animate_scene(
            image_path=None,  # No image for video-to-video
            prompt=prompt,
            output_path=output_path,
            duration_seconds=duration_seconds,
            resolution=resolution,
            model=WanModel.VIDEO_TO_VIDEO,
            source_video=source_video,
            progress_callback=progress_callback,
        )

    def text_to_video(
        self,
        prompt: str,
        output_path: Path,
        duration_seconds: float = 5.0,
        resolution: str = "720p",
        first_frame: Optional[Path] = None,
        last_frame: Optional[Path] = None,
        use_seedance: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Optional[Path]:
        """Generate video purely from text prompt.

        Args:
            prompt: Description of the video content
            output_path: Path to save the output video
            duration_seconds: Output duration
            resolution: Output resolution
            first_frame: Optional first frame for smooth start
            last_frame: Optional last frame for smooth end
            use_seedance: Use Seedance 1.5 Pro instead of WAN 2.6
            progress_callback: Optional progress callback

        Returns:
            Path to the generated video, or None if failed
        """
        model = SeedanceModel.TEXT_TO_VIDEO if use_seedance else WanModel.TEXT_TO_VIDEO

        return self.animate_scene(
            image_path=None,
            prompt=prompt,
            output_path=output_path,
            duration_seconds=duration_seconds,
            resolution=resolution,
            model=model,
            first_frame=first_frame,
            last_frame=last_frame,
            progress_callback=progress_callback,
        )

    def lip_sync_video(
        self,
        image_path: Path,
        audio_path: Path,
        prompt: str,
        output_path: Path,
        resolution: str = "720p",
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Optional[Path]:
        """Generate video with lip-synced audio using Seedance 1.5 Pro.

        Creates a video from an image with lip movements synchronized
        to the provided audio. Great for talking head videos.

        Args:
            image_path: Path to the character/face image
            audio_path: Path to the audio file (speech)
            prompt: Description of the character's expression and movement
            output_path: Path to save the output video
            resolution: Output resolution ("480p", "720p", or "1080p")
            progress_callback: Optional progress callback

        Returns:
            Path to the lip-synced video, or None if failed
        """
        # Calculate duration from audio
        try:
            import subprocess
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(audio_path)],
                capture_output=True, text=True
            )
            duration = float(result.stdout.strip())
            # Clamp to Seedance limits (3-15 seconds)
            duration = max(3, min(15, int(duration)))
        except Exception:
            duration = 8  # Default if we can't determine

        return self.animate_scene(
            image_path=image_path,
            prompt=prompt,
            output_path=output_path,
            duration_seconds=duration,
            resolution=resolution,
            model=SeedanceModel.IMAGE_TO_VIDEO,
            audio_path=audio_path,
            progress_callback=progress_callback,
        )


def check_atlascloud_available() -> bool:
    """Check if AtlasCloud animation is available (API key configured)."""
    return bool(default_config.atlascloud_api_key)


def get_wan26_pricing(duration_seconds: float, num_scenes: int = 1) -> dict:
    """Estimate WAN 2.6 costs.

    Args:
        duration_seconds: Duration per scene
        num_scenes: Number of scenes

    Returns:
        Dict with pricing breakdown
    """
    cost_per_second = 0.075  # $0.075/second for WAN 2.6
    total_seconds = duration_seconds * num_scenes
    total_cost = total_seconds * cost_per_second

    return {
        "model": "WAN 2.6",
        "cost_per_second": cost_per_second,
        "duration_per_scene": duration_seconds,
        "num_scenes": num_scenes,
        "total_seconds": total_seconds,
        "estimated_cost": round(total_cost, 2),
    }


def get_seedance_pricing(duration_seconds: float, num_scenes: int = 1) -> dict:
    """Estimate Seedance 1.5 Pro costs.

    Args:
        duration_seconds: Duration per scene
        num_scenes: Number of scenes

    Returns:
        Dict with pricing breakdown
    """
    cost_per_second = 0.0147  # $0.0147/second for Seedance 1.5 Pro
    total_seconds = duration_seconds * num_scenes
    total_cost = total_seconds * cost_per_second

    return {
        "model": "Seedance 1.5 Pro",
        "cost_per_second": cost_per_second,
        "duration_per_scene": duration_seconds,
        "num_scenes": num_scenes,
        "total_seconds": total_seconds,
        "estimated_cost": round(total_cost, 2),
    }
