"""Lip sync animation service with multiple backend support.

Backends:
- wan2s2v: Free Wan2.2-S2V via Hugging Face Spaces (slow, ~10 min/scene)
- kling: Paid Kling AI via fal.ai (fast, ~1-2 min/scene, ~$0.10/5s)
"""

import shutil
import tempfile
from pathlib import Path
from typing import Callable, Literal, Optional

from pydub import AudioSegment

from src.config import Config, config as default_config

# Type for lip sync backend
LipSyncBackend = Literal["wan2s2v", "kling"]


class Wan2S2VAnimator:
    """Generate lip-synced singing animations from images using Wan2.2-S2V.

    This service uses the free Wan2.2-S2V model hosted on Hugging Face Spaces
    to animate characters in images with full body lip-sync animation.

    Features:
    - Full body, half-body, and portrait animation
    - Supports singing (not just speaking)
    - Works with custom images
    - Free and cross-platform (cloud-based)

    Limitations:
    - Slow (~10 minutes per scene)
    - Queue times on busy days
    """

    # Hugging Face Space URL for Wan2.2-S2V
    SPACE_URL = "Wan-AI/Wan2.2-S2V"

    def __init__(self):
        self._client = None

    def _get_client(self):
        """Lazy load Gradio client."""
        if self._client is None:
            from gradio_client import Client

            self._client = Client(self.SPACE_URL)
        return self._client

    def animate_scene(
        self,
        image_path: Path,
        audio_path: Path,
        start_time: float,
        duration: float,
        output_path: Path,
        resolution: str = "720P",
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Optional[Path]:
        """
        Animate a scene image with lip-synced singing using Wan2.2-S2V.

        Args:
            image_path: Path to the scene image
            audio_path: Path to the full audio file
            start_time: Start time in the audio (seconds)
            duration: Duration of the scene (seconds)
            output_path: Path to save the output video
            resolution: Video resolution ("480P" or "720P")
            progress_callback: Optional callback for progress updates

        Returns:
            Path to the generated video, or None if generation failed

        Note:
            Wan2.2-S2V is optimized for singing/music content. The API only
            accepts ref_img, audio, and resolution parameters.
        """
        from gradio_client import handle_file

        if progress_callback:
            progress_callback("Preparing audio clip...", 0.1)

        # Extract the audio segment for this scene
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            audio_clip_path = temp_dir / "audio_clip.wav"

            try:
                # Extract audio segment
                audio = AudioSegment.from_file(str(audio_path))
                start_ms = int(start_time * 1000)
                end_ms = int((start_time + duration) * 1000)
                clip = audio[start_ms:end_ms]
                clip.export(str(audio_clip_path), format="wav")

                if progress_callback:
                    progress_callback("Connecting to Wan2.2-S2V...", 0.2)

                # Get the Gradio client
                client = self._get_client()

                if progress_callback:
                    progress_callback("Generating lip-sync animation...", 0.3)

                # Call the Wan2.2-S2V API
                # API expects: ref_img, audio, resolution
                # The model is optimized for singing/music content
                result = client.predict(
                    ref_img=handle_file(str(image_path)),
                    audio=handle_file(str(audio_clip_path)),
                    resolution=resolution,
                    api_name="/predict"
                )

                if progress_callback:
                    progress_callback("Processing result...", 0.9)

                # Result is a dict with 'video' and 'subtitles' keys
                # The video value is a filepath string
                video_path = None
                if isinstance(result, dict):
                    video_path = result.get('video')
                elif isinstance(result, str):
                    video_path = result

                if video_path and Path(video_path).exists():
                    # Copy to output path
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(video_path, output_path)

                    if progress_callback:
                        progress_callback("Animation complete!", 1.0)

                    return output_path
                else:
                    print(f"Lip sync result: {result}")
                    if progress_callback:
                        progress_callback("No video generated", 0.0)

            except Exception as e:
                print(f"Lip sync animation failed: {e}")
                if progress_callback:
                    progress_callback(f"Animation failed: {e}", 0.0)
                return None

        return None

    def animate_scenes(
        self,
        scenes: list,
        audio_path: Path,
        output_dir: Path,
        resolution: str = "480P",
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> dict[int, Optional[Path]]:
        """
        Animate multiple scenes marked for animation.

        Args:
            scenes: List of Scene objects (only those with animated=True will be processed)
            audio_path: Path to the full audio file
            output_dir: Directory to save output videos
            resolution: Video resolution ("480P" or "720P")
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping scene index to output video path (or None if failed)
        """
        # Filter to only animated scenes with images
        animated_scenes = [
            s for s in scenes
            if s.animated and s.image_path and Path(s.image_path).exists()
        ]

        if not animated_scenes:
            return {}

        results = {}
        total = len(animated_scenes)

        for i, scene in enumerate(animated_scenes):
            if progress_callback:
                progress_callback(
                    f"Animating scene {scene.index + 1} ({i + 1}/{total})...",
                    i / total
                )

            output_path = output_dir / f"animated_scene_{scene.index:03d}.mp4"

            def scene_progress(msg: str, prog: float):
                if progress_callback:
                    overall = (i + prog) / total
                    progress_callback(f"Scene {scene.index + 1}: {msg}", overall)

            result = self.animate_scene(
                image_path=Path(scene.image_path),
                audio_path=audio_path,
                start_time=scene.start_time,
                duration=scene.duration,
                output_path=output_path,
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


# Keep LipSyncAnimator as an alias for backward compatibility
LipSyncAnimator = Wan2S2VAnimator


def get_lip_sync_animator(
    backend: Optional[LipSyncBackend] = None,
    config: Optional[Config] = None,
):
    """
    Factory function to get the appropriate lip sync animator.

    Args:
        backend: Lip sync backend to use ("wan2s2v" or "kling").
                 If None, uses config default.
        config: Optional configuration object

    Returns:
        Animator instance (Wan2S2VAnimator or KlingAnimator)
    """
    cfg = config or default_config
    use_backend = backend or cfg.lip_sync_backend

    if use_backend == "kling":
        from src.services.kling_animator import KlingAnimator
        return KlingAnimator(config=cfg)
    else:
        return Wan2S2VAnimator()


def check_lip_sync_available(backend: Optional[LipSyncBackend] = None) -> bool:
    """Check if lip sync animation is available for the specified backend."""
    if backend == "kling":
        from src.services.kling_animator import check_kling_available
        return check_kling_available()
    else:
        # Check for Wan2S2V (requires gradio_client)
        try:
            from gradio_client import Client  # noqa: F401
            return True
        except ImportError:
            return False


def get_available_lip_sync_backends(
    config: Optional[Config] = None
) -> list[tuple[str, str]]:
    """
    Get list of available lip sync backends.

    Args:
        config: Optional configuration

    Returns:
        List of (value, label) tuples for use in dropdown menus
    """
    backends = []

    # Wan2S2V is always available if gradio_client is installed
    if check_lip_sync_available("wan2s2v"):
        backends.append(("wan2s2v", "Wan2.2-S2V (Free, Slow)"))

    # Kling requires fal_client and API key
    if check_lip_sync_available("kling"):
        backends.append(("kling", "Kling AI (Paid, Fast)"))

    return backends
