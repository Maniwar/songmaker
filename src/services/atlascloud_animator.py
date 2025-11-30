"""AtlasCloud animation service using Wan 2.5 image-to-video.

This service generates motion videos from images using the AtlasCloud API,
which hosts the Wan 2.5 image-to-video model. Unlike HuggingFace free tier,
this is a paid service with no GPU quota limits.
"""

import base64
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import requests

from src.config import Config, config as default_config

logger = logging.getLogger(__name__)


class AtlasCloudAnimator:
    """Generate animations using AtlasCloud's Wan 2.5 image-to-video API.

    This service uses the Wan 2.5 model via AtlasCloud's API to animate
    images based on text prompts describing the motion.

    Features:
    - Text prompt-driven animation
    - No GPU quota limits (paid service)
    - Supports 5 or 10 second durations
    - Supports 720p or 1080p resolutions

    Use cases:
    - Playing instruments (guitar, drums, piano)
    - Dancing or moving to music
    - Ambient motion (wind, water, etc.)
    - Any motion that can be described with text
    """

    BASE_URL = "https://api.atlascloud.ai/api/v1/model"
    MODEL = "alibaba/wan-2.5/image-to-video-fast"

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config
        self._api_key = self.config.atlascloud_api_key

    def _get_headers(self) -> dict:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64 data URL."""
        with open(image_path, "rb") as f:
            image_data = f.read()

        # Determine mime type
        suffix = image_path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(suffix, "image/png")

        base64_data = base64.b64encode(image_data).decode("utf-8")
        return f"data:{mime_type};base64,{base64_data}"

    def animate_scene(
        self,
        image_path: Path,
        prompt: str,
        output_path: Path,
        duration_seconds: float = 5.0,
        resolution: str = "720p",
        seed: int = -1,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        poll_interval: float = 5.0,
        max_wait_time: float = 300.0,
    ) -> Optional[Path]:
        """
        Animate a scene image with prompt-driven motion using AtlasCloud Wan 2.5.

        Args:
            image_path: Path to the scene image
            prompt: Text description of the motion (e.g., "playing guitar passionately")
            output_path: Path to save the output video
            duration_seconds: Video duration (5 or 10 seconds)
            resolution: Output resolution ("720p" or "1080p")
            seed: Random seed (-1 for random)
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

        # Validate duration (API only supports 5 or 10 seconds)
        valid_durations = [5, 10]
        duration = int(duration_seconds)
        if duration not in valid_durations:
            # Round to nearest valid duration
            duration = 5 if duration < 7.5 else 10
            logger.warning(
                f"Duration {duration_seconds}s not supported, using {duration}s"
            )

        # Validate resolution
        valid_resolutions = ["720p", "1080p"]
        if resolution not in valid_resolutions:
            resolution = "720p"
            logger.warning(f"Resolution not supported, using {resolution}")

        if progress_callback:
            progress_callback("Encoding image...", 0.1)

        try:
            # Encode image to base64 data URL
            image_data_url = self._encode_image(image_path)

            if progress_callback:
                progress_callback("Submitting to AtlasCloud...", 0.2)

            # Build request payload - fields at root level per AtlasCloud API schema
            payload = {
                "model": self.MODEL,
                "image": image_data_url,
                "prompt": prompt,
                "resolution": resolution,
                "duration": duration,
            }

            if seed >= 0:
                payload["seed"] = seed

            # Submit generation request
            response = requests.post(
                f"{self.BASE_URL}/generateVideo",
                headers=self._get_headers(),
                json=payload,
                timeout=60,
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

            # Poll for completion using /prediction/{id} endpoint
            poll_url = f"{self.BASE_URL}/prediction/{prediction_id}"
            start_time = time.time()
            while True:
                elapsed = time.time() - start_time
                if elapsed > max_wait_time:
                    logger.error(f"Timeout waiting for AtlasCloud result: {prediction_id}")
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
                    video_response = requests.get(video_url, timeout=120)
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


def check_atlascloud_available() -> bool:
    """Check if AtlasCloud animation is available (API key configured)."""
    return bool(default_config.atlascloud_api_key)
