"""Seedance Pro animation service via AtlasCloud.

This service generates motion videos from images using the AtlasCloud API
with the ByteDance Seedance Pro model. Supports longer durations (2-12 seconds)
and additional camera controls compared to Wan 2.5.
"""

import base64
import logging
import time
from pathlib import Path
from typing import Callable, Literal, Optional

import requests

from src.config import Config, config as default_config

logger = logging.getLogger(__name__)

# Valid parameter values from API schema
VALID_DURATIONS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
VALID_RESOLUTIONS = ["480p", "720p", "1080p"]
VALID_ASPECT_RATIOS = ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]


class SeedanceAnimator:
    """Generate animations using AtlasCloud's Seedance Pro image-to-video API.

    This service uses the ByteDance Seedance Pro model via AtlasCloud's API
    to animate images based on text prompts describing the motion.

    Features:
    - Text prompt-driven animation (up to 2000 characters)
    - Supports 2-12 second durations
    - Supports 480p, 720p, or 1080p resolutions
    - Supports various aspect ratios (21:9 to 9:16)
    - Camera fixed option for stable shots
    - No GPU quota limits (paid service)

    Use cases:
    - Cinematic motion with longer durations
    - Fixed camera shots for dialogue/singing
    - Any motion that can be described with text
    """

    BASE_URL = "https://api.atlascloud.ai/api/v1/model"
    MODEL = "bytedance/seedance-v1-pro-fast/image-to-video"

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
        duration_seconds: int = 5,
        resolution: str = "720p",
        aspect_ratio: Optional[str] = None,
        camera_fixed: bool = False,
        seed: int = -1,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        poll_interval: float = 5.0,
        max_wait_time: float = 600.0,
    ) -> Optional[Path]:
        """
        Animate a scene image with prompt-driven motion using Seedance Pro.

        Args:
            image_path: Path to the scene image (jpg/jpeg/png, min 300x300px, max 10MB)
            prompt: Text description of the motion (max 2000 chars)
            output_path: Path to save the output video
            duration_seconds: Video duration (2-12 seconds, default 5)
            resolution: Output resolution ("480p", "720p", or "1080p")
            aspect_ratio: Aspect ratio ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
                         If None, uses image's native aspect ratio
            camera_fixed: Whether to fix the camera position (default False)
            seed: Random seed (-1 for random)
            progress_callback: Optional callback for progress updates
            poll_interval: Seconds between status checks (default 5)
            max_wait_time: Maximum seconds to wait for completion (default 600)

        Returns:
            Path to the generated video, or None if generation failed
        """
        if not self._api_key:
            logger.error("AtlasCloud API key not configured")
            if progress_callback:
                progress_callback("Error: AtlasCloud API key not set", 0.0)
            return None

        # Validate duration (API supports 2-12 seconds)
        duration = int(duration_seconds)
        if duration not in VALID_DURATIONS:
            # Clamp to valid range
            duration = max(2, min(12, duration))
            logger.warning(
                f"Duration {duration_seconds}s adjusted to {duration}s"
            )

        # Validate resolution
        if resolution not in VALID_RESOLUTIONS:
            resolution = "720p"
            logger.warning(f"Resolution not supported, using {resolution}")

        # Validate aspect ratio if provided
        if aspect_ratio and aspect_ratio not in VALID_ASPECT_RATIOS:
            logger.warning(f"Aspect ratio {aspect_ratio} not supported, omitting")
            aspect_ratio = None

        # Truncate prompt if too long
        if len(prompt) > 2000:
            logger.warning(f"Prompt truncated from {len(prompt)} to 2000 chars")
            prompt = prompt[:2000]

        if progress_callback:
            progress_callback("Encoding image...", 0.1)

        try:
            # Encode image to base64 data URL
            image_data_url = self._encode_image(image_path)

            if progress_callback:
                progress_callback("Submitting to Seedance Pro...", 0.2)

            # Build request payload per Seedance API schema
            payload = {
                "model": self.MODEL,
                "image": image_data_url,
                "prompt": prompt,
                "resolution": resolution,
                "duration": duration,
                "camera_fixed": camera_fixed,
            }

            if aspect_ratio:
                payload["aspect_ratio"] = aspect_ratio

            if seed >= 0:
                payload["seed"] = seed

            logger.info(f"Seedance request: duration={duration}s, resolution={resolution}, camera_fixed={camera_fixed}")

            # Submit generation request
            response = requests.post(
                f"{self.BASE_URL}/generateVideo",
                headers=self._get_headers(),
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()

            # Extract prediction ID from response
            data = result.get("data", {})
            prediction_id = data.get("id")
            if not prediction_id:
                logger.error(f"No prediction ID in response: {result}")
                if progress_callback:
                    progress_callback("Error: No request ID returned", 0.0)
                return None

            logger.info(f"Seedance request submitted: {prediction_id}")

            if progress_callback:
                progress_callback("Processing animation...", 0.3)

            # Poll for completion
            poll_url = f"{self.BASE_URL}/prediction/{prediction_id}"
            start_time = time.time()
            while True:
                elapsed = time.time() - start_time
                if elapsed > max_wait_time:
                    logger.error(f"Timeout waiting for Seedance result: {prediction_id}")
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
                logger.debug(f"Seedance status: {status}")

                if status in ["completed", "succeeded"]:
                    # Get video URL from outputs
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

                    logger.info(f"Seedance animation saved to: {output_path}")

                    if progress_callback:
                        progress_callback("Animation complete!", 1.0)

                    return output_path

                elif status == "failed":
                    error_msg = poll_data.get("error") or "Unknown error"
                    logger.error(f"Seedance generation failed: {error_msg}")
                    if progress_callback:
                        progress_callback(f"Error: {error_msg[:100]}", 0.0)
                    return None

                else:
                    # Still processing
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
            logger.error(f"Seedance API error: {error_msg}")
            if progress_callback:
                progress_callback(f"API Error: {error_msg[:100]}", 0.0)
            return None

        except Exception as e:
            logger.error(f"Seedance animation failed: {e}", exc_info=True)
            if progress_callback:
                progress_callback(f"Error: {str(e)[:100]}", 0.0)
            return None

    def animate_scenes(
        self,
        scenes: list,
        output_dir: Path,
        duration_seconds: int = 5,
        resolution: str = "720p",
        camera_fixed: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> dict[int, Optional[Path]]:
        """
        Animate multiple scenes marked for Seedance animation.

        Args:
            scenes: List of Scene objects (only those with animation_type=SEEDANCE
                    will be processed)
            output_dir: Directory to save output videos
            duration_seconds: Video duration per scene (2-12 seconds)
            resolution: Output resolution ("480p", "720p", or "1080p")
            camera_fixed: Whether to fix camera position
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping scene index to output video path (or None if failed)
        """
        from src.models.schemas import AnimationType

        # Filter to only Seedance-animated scenes with images
        seedance_scenes = [
            s for s in scenes
            if getattr(s, 'animation_type', None) == AnimationType.SEEDANCE
            and s.image_path and Path(s.image_path).exists()
        ]

        if not seedance_scenes:
            return {}

        results = {}
        total = len(seedance_scenes)

        for i, scene in enumerate(seedance_scenes):
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
                camera_fixed=camera_fixed,
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


def check_seedance_available() -> bool:
    """Check if Seedance animation is available (API key configured)."""
    return bool(default_config.atlascloud_api_key)
