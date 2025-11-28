"""Lip sync animation service using Wan2.2-S2V via Hugging Face Spaces."""

import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional

from pydub import AudioSegment


class LipSyncAnimator:
    """Generate lip-synced singing animations from images using Wan2.2-S2V.

    This service uses the free Wan2.2-S2V model hosted on Hugging Face Spaces
    to animate characters in images with full body lip-sync animation.

    Features:
    - Full body, half-body, and portrait animation
    - Supports singing (not just speaking)
    - Works with custom images
    - Free and cross-platform (cloud-based)
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
        resolution: str = "480P",
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
                # The API typically expects: image, audio, resolution
                result = client.predict(
                    image=handle_file(str(image_path)),
                    audio=handle_file(str(audio_clip_path)),
                    resolution=resolution,
                    api_name="/generate"
                )

                if progress_callback:
                    progress_callback("Processing result...", 0.9)

                # Result is typically a file path to the generated video
                if result and Path(result).exists():
                    # Copy to output path
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(result, output_path)

                    if progress_callback:
                        progress_callback("Animation complete!", 1.0)

                    return output_path

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


def check_lip_sync_available() -> bool:
    """Check if lip sync animation is available (requires gradio_client)."""
    try:
        from gradio_client import Client  # noqa: F401
        return True
    except ImportError:
        return False
