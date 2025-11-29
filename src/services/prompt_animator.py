"""Prompt-based animation service using Wan2.2-TI2V-5B.

This service generates motion videos from images using text prompts,
perfect for non-singing animations like playing instruments, dancing, etc.
"""

import logging
import shutil
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class PromptAnimator:
    """Generate prompt-driven animations from images using Wan2.2-TI2V-5B.

    This service uses the Wan2.2-TI2V-5B model hosted on Hugging Face Spaces
    to animate images based on text prompts describing the motion.

    Features:
    - Text prompt-driven animation
    - Works with any image
    - Supports custom motion descriptions
    - Free and cross-platform (cloud-based)

    Use cases:
    - Playing instruments (guitar, drums, piano)
    - Dancing or moving to music
    - Ambient motion (wind, water, etc.)
    - Any motion that can be described with text
    """

    # Hugging Face Space URL for Wan2.2-TI2V-5B
    SPACE_URL = "Wan-AI/Wan-2.2-5B"

    def __init__(self):
        self._client = None

    def _get_client(self):
        """Lazy load Gradio client."""
        if self._client is None:
            from gradio_client import Client

            self._client = Client(self.SPACE_URL)
        return self._client

    # Quality presets balancing quality vs GPU time limits
    # NOTE: Duration is NOT included - it must match scene timing for audio sync
    QUALITY_PRESETS = {
        "fast": {"height": 320, "width": 576, "sampling_steps": 8},
        "standard": {"height": 480, "width": 848, "sampling_steps": 15},
        "quality": {"height": 704, "width": 1280, "sampling_steps": 25},
    }

    def animate_scene(
        self,
        image_path: Path,
        prompt: str,
        output_path: Path,
        duration_seconds: float = 2.0,
        height: int = 480,  # Reduced from 704 to stay within GPU limits
        width: int = 848,   # Reduced from 1280 to stay within GPU limits
        guide_scale: float = 5.0,
        sampling_steps: int = 15,  # Reduced from 38 to stay within GPU limits
        seed: int = -1,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        quality_preset: Optional[str] = None,  # "fast", "standard", "quality"
    ) -> Optional[Path]:
        """
        Animate a scene image with prompt-driven motion using Wan2.2-TI2V-5B.

        Args:
            image_path: Path to the scene image
            prompt: Text description of the motion (e.g., "playing guitar passionately")
            output_path: Path to save the output video
            duration_seconds: Video duration (1-2 seconds recommended for free tier)
            height: Video height (default 480 for GPU limits)
            width: Video width (default 848 for GPU limits)
            guide_scale: How closely to follow the prompt (default 5.0)
            sampling_steps: Quality vs speed tradeoff (default 15 for GPU limits)
            seed: Random seed (-1 for random)
            progress_callback: Optional callback for progress updates
            quality_preset: Quality preset - "fast", "standard", or "quality"
                - fast: 320x576, 8 steps, ~10s GPU (works reliably)
                - standard: 480x848, 15 steps, ~40s GPU (default)
                - quality: 704x1280, 25 steps, ~90s GPU (may hit limits)

        Returns:
            Path to the generated video, or None if generation failed

        Note:
            The free HuggingFace tier has GPU time limits (~90-120s max).
            Higher resolutions/steps may fail with "GPU duration exceeded" error.
            Use "fast" or "standard" presets for reliable results.
        """
        from gradio_client import handle_file

        # Apply quality preset if specified (only resolution and steps, NOT duration)
        if quality_preset and quality_preset in self.QUALITY_PRESETS:
            preset = self.QUALITY_PRESETS[quality_preset]
            height = preset["height"]
            width = preset["width"]
            sampling_steps = preset["sampling_steps"]
            logger.info(f"Using quality preset '{quality_preset}': {width}x{height}, {sampling_steps} steps")

        if progress_callback:
            progress_callback("Preparing image...", 0.1)

        try:
            if progress_callback:
                progress_callback("Connecting to Wan2.2-TI2V-5B...", 0.2)

            # Get the Gradio client
            client = self._get_client()

            if progress_callback:
                progress_callback(f"Generating animation: {prompt[:50]}...", 0.3)

            # Call the Wan2.2-TI2V-5B API
            # API endpoint: /generate_video
            # Parameters: image, prompt, height, width, duration_seconds,
            #             sampling_steps, guide_scale, shift, seed
            result = client.predict(
                image=handle_file(str(image_path)),
                prompt=prompt,
                height=height,
                width=width,
                duration_seconds=duration_seconds,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                shift=5.0,  # Default shift value
                seed=seed,
                api_name="/generate_video"
            )

            if progress_callback:
                progress_callback("Processing result...", 0.9)

            logger.info(f"API result type: {type(result)}, value: {result}")

            # Parse result - can be dict, tuple, or string depending on Gradio version
            video_path = None

            if isinstance(result, dict):
                # Dict with 'video' key: {'video': filepath, 'subtitles': None}
                video_path = result.get('video')
                logger.info(f"Result is dict, video key: {video_path}")
            elif isinstance(result, tuple):
                # Tuple: (video_filepath, subtitles_filepath)
                video_path = result[0] if result else None
                logger.info(f"Result is tuple, first element: {video_path}")
            elif isinstance(result, str):
                # Direct filepath string
                video_path = result
                logger.info(f"Result is string: {video_path}")
            elif hasattr(result, 'video'):
                # Object with video attribute
                video_path = result.video
                logger.info(f"Result has video attribute: {video_path}")

            # Handle nested dict (some Gradio versions return {'video': {'path': ...}})
            if isinstance(video_path, dict):
                video_path = video_path.get('path') or video_path.get('video')
                logger.info(f"Extracted path from nested dict: {video_path}")

            if video_path and Path(video_path).exists():
                # Copy to output path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(video_path, output_path)
                logger.info(f"Animation saved to: {output_path}")

                if progress_callback:
                    progress_callback("Animation complete!", 1.0)

                return output_path
            else:
                logger.warning(f"No valid video path found. Result: {result}")
                if progress_callback:
                    progress_callback("No video generated", 0.0)

        except Exception as e:
            logger.error(f"Prompt animation failed: {e}", exc_info=True)
            if progress_callback:
                progress_callback(f"Animation failed: {e}", 0.0)
            return None

        return None

    def animate_scenes(
        self,
        scenes: list,
        output_dir: Path,
        duration_seconds: float = 2.0,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> dict[int, Optional[Path]]:
        """
        Animate multiple scenes marked for prompt animation.

        Args:
            scenes: List of Scene objects (only those with animation_type=PROMPT will be processed)
            output_dir: Directory to save output videos
            duration_seconds: Video duration per scene
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping scene index to output video path (or None if failed)
        """
        from src.models.schemas import AnimationType

        # Filter to only prompt-animated scenes with images
        prompt_scenes = [
            s for s in scenes
            if getattr(s, 'animation_type', None) == AnimationType.PROMPT
            and s.image_path and Path(s.image_path).exists()
        ]

        if not prompt_scenes:
            return {}

        results = {}
        total = len(prompt_scenes)

        for i, scene in enumerate(prompt_scenes):
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


def check_prompt_animator_available() -> bool:
    """Check if prompt animation is available (requires gradio_client)."""
    try:
        from gradio_client import Client  # noqa: F401
        return True
    except ImportError:
        return False
