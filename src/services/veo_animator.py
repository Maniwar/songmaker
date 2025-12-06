"""Veo 3.1 animation service using Google's Gemini API.

This service generates high-quality videos from images using Google's Veo 3.1 model.
Unlike the free HuggingFace options, this is a PAID service but offers superior quality.
"""

import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from src.config import Config, config as default_config

logger = logging.getLogger(__name__)


class VeoAnimator:
    """Generate animations from images using Google Veo 3.1.

    This service uses the Veo 3.1 model via the Gemini API to animate
    images based on text prompts describing the motion.

    Features:
    - High-quality video generation (720p/1080p)
    - 4, 6, or 8 second durations
    - Standard or Fast model selection
    - Audio generation toggle (disable to save ~33% cost)
    - Uses existing Google API key (same as image generation)

    Note: This is a PAID API. Check Google AI pricing for costs.
    """

    # Model options
    MODELS = {
        "veo-3.1": "veo-3.1-generate-preview",
        "veo-3.1-fast": "veo-3.1-fast-generate-preview",
        "veo-3.0": "veo-3.0-generate-001",
        "veo-3.0-fast": "veo-3.0-fast-generate-001",
    }

    # Default to fast model (cheaper, faster)
    DEFAULT_MODEL = "veo-3.1-fast"

    # Duration options in seconds
    DURATION_OPTIONS = [4, 6, 8]

    # Resolution options
    RESOLUTION_OPTIONS = ["720p", "1080p"]

    def __init__(self, config: Optional[Config] = None, model: str = None):
        self.config = config or default_config
        self._client = None
        # Use specified model, or config, or default to fast
        self.model = model or getattr(self.config, 'veo_model', self.DEFAULT_MODEL)
        if self.model not in self.MODELS:
            self.model = self.DEFAULT_MODEL

    def _get_client(self):
        """Lazy load Google GenAI client."""
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.config.google_api_key)
        return self._client

    def _get_best_duration(self, target_duration: float) -> int:
        """Get the best Veo duration for the target scene duration.

        Veo only supports 4, 6, or 8 second videos.
        We pick the closest option that's >= target duration when possible.
        """
        # Round up to nearest valid duration
        for duration in self.DURATION_OPTIONS:
            if duration >= target_duration:
                return duration
        # If scene is longer than 8s, use 8s (will need to loop/extend)
        return 8

    def animate_scene(
        self,
        image_path: Path,
        prompt: str,
        output_path: Path,
        duration_seconds: float = 4.0,
        resolution: str = "720p",
        aspect_ratio: str = "16:9",
        generate_audio: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Optional[Path]:
        """
        Animate a scene image using Veo 3.1.

        Args:
            image_path: Path to the scene image
            prompt: Text description of the motion
            output_path: Path to save the output video
            duration_seconds: Target duration (will use closest Veo option: 4, 6, or 8)
            resolution: Video resolution ("720p" or "1080p")
            aspect_ratio: Aspect ratio ("16:9" or "9:16")
            generate_audio: Whether to generate audio (False saves ~33% cost)
            progress_callback: Optional callback for progress updates

        Returns:
            Path to the generated video, or None if generation failed
        """
        from google.genai import types

        if progress_callback:
            progress_callback("Preparing image for Veo 3.1...", 0.1)

        try:
            # Load and prepare the image
            pil_image = Image.open(image_path)

            # Convert to bytes for the API (JPEG for smaller size)
            img_buffer = BytesIO()
            # Convert RGBA to RGB if needed (JPEG doesn't support alpha)
            if pil_image.mode == 'RGBA':
                pil_image = pil_image.convert('RGB')
            pil_image.save(img_buffer, format="JPEG", quality=95)
            img_bytes = img_buffer.getvalue()

            # Get the appropriate duration
            veo_duration = self._get_best_duration(duration_seconds)

            if progress_callback:
                progress_callback(f"Connecting to Veo 3.1 ({veo_duration}s video)...", 0.2)

            client = self._get_client()

            # Create the Image object for Veo API (not Part)
            # Veo expects types.Image, not types.Part
            veo_image = types.Image(image_bytes=img_bytes, mime_type="image/jpeg")

            if progress_callback:
                progress_callback(f"Generating video: {prompt[:50]}...", 0.3)

            # Generate video using selected Veo model
            # Note: Veo is an async operation - we need to poll for completion
            # generate_audio=False saves ~33% cost since we have our own audio
            model_id = self.MODELS[self.model]
            logger.info(f"Using Veo model: {model_id} (audio={'on' if generate_audio else 'off'})")

            operation = client.models.generate_videos(
                model=model_id,
                prompt=prompt,
                image=veo_image,
                config=types.GenerateVideosConfig(
                    aspect_ratio=aspect_ratio,
                    duration_seconds=veo_duration,
                    resolution=resolution,
                    generateAudio=generate_audio,
                ),
            )

            if progress_callback:
                progress_callback("Waiting for Veo to generate video...", 0.4)

            # Poll until the video is ready (can take 11s to 6 minutes)
            poll_count = 0
            max_polls = 60  # 10 minutes max wait time
            while not operation.done:
                poll_count += 1
                if poll_count > max_polls:
                    raise TimeoutError("Video generation timed out after 10 minutes")

                time.sleep(10)  # Poll every 10 seconds
                operation = client.operations.get(operation)

                # Update progress (stay between 0.4 and 0.9)
                progress = 0.4 + (0.5 * min(poll_count / 30, 1.0))
                if progress_callback:
                    progress_callback(
                        f"Generating video... ({poll_count * 10}s elapsed)",
                        progress
                    )

            if progress_callback:
                progress_callback("Downloading generated video...", 0.9)

            # Get the generated video
            if not operation.response or not operation.response.generated_videos:
                logger.error("Veo returned no video in response")
                if progress_callback:
                    progress_callback("Veo returned no video", 0.0)
                return None

            video = operation.response.generated_videos[0]

            # Download the video
            client.files.download(file=video.video)

            # Save to output path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            video.video.save(str(output_path))

            logger.info(f"Veo animation saved to: {output_path}")

            if progress_callback:
                progress_callback("Veo animation complete!", 1.0)

            return output_path

        except Exception as e:
            error_str = str(e)
            logger.error(f"Veo animation failed: {e}", exc_info=True)

            # Provide user-friendly error messages
            if "quota" in error_str.lower() or "limit" in error_str.lower():
                user_msg = "Google API quota exceeded. Check your billing."
            elif "invalid" in error_str.lower() and "api" in error_str.lower():
                user_msg = "Invalid Google API key. Check your GOOGLE_API_KEY."
            elif "timeout" in error_str.lower():
                user_msg = "Video generation timed out. Please try again."
            elif "billing" in error_str.lower():
                user_msg = "Veo requires billing to be enabled on your Google Cloud account."
            else:
                user_msg = f"Veo animation failed: {error_str[:100]}"

            if progress_callback:
                progress_callback(user_msg, 0.0)
            return None

    def animate_scenes(
        self,
        scenes: list,
        output_dir: Path,
        resolution: str = "720p",
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> dict[int, Optional[Path]]:
        """
        Animate multiple scenes marked for Veo animation.

        Args:
            scenes: List of Scene objects (only those with animation_type=VEO will be processed)
            output_dir: Directory to save output videos
            resolution: Video resolution ("720p" or "1080p")
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping scene index to output video path (or None if failed)
        """
        from src.models.schemas import AnimationType

        # Filter to only VEO-animated scenes with images
        veo_scenes = [
            s for s in scenes
            if getattr(s, 'animation_type', None) == AnimationType.VEO
            and s.image_path and Path(s.image_path).exists()
        ]

        if not veo_scenes:
            return {}

        results = {}
        total = len(veo_scenes)

        for i, scene in enumerate(veo_scenes):
            if progress_callback:
                progress_callback(
                    f"Animating scene {scene.index + 1} with Veo ({i + 1}/{total})...",
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
                duration_seconds=scene.duration,
                resolution=resolution,
                progress_callback=scene_progress,
            )

            results[scene.index] = result

        if progress_callback:
            success_count = sum(1 for v in results.values() if v is not None)
            progress_callback(
                f"Veo animated {success_count}/{total} scenes",
                1.0
            )

        return results


def check_veo_available() -> bool:
    """Check if Veo animation is available (requires google-genai package)."""
    try:
        from google import genai  # noqa: F401
        from google.genai import types  # noqa: F401
        return True
    except ImportError:
        return False
