"""Wan 2.2 Speech-to-Video animation service via fal.ai API.

This provides full motion + lip sync in a single step:
- Takes image + audio as input
- Outputs video with realistic facial expressions AND body movements
- Much faster than free HuggingFace (paid service)
"""

import os
import tempfile
from pathlib import Path
from typing import Callable, Optional

import requests
from pydub import AudioSegment

from src.config import Config, config as default_config


class WanS2VAnimator:
    """Generate speech-to-video animations using Wan 2.2 via fal.ai.

    This is the ideal one-step solution:
    - Takes image + audio directly
    - Produces full video with motion + lip sync
    - Same Wan2.2-S2V model as HuggingFace but faster (paid)

    Pricing: ~$0.10-0.20 per video second
    """

    # fal.ai endpoint for Wan 2.2 S2V
    S2V_ENDPOINT = "fal-ai/wan/v2.2-14b/speech-to-video"

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config
        self._fal_client = None

    def _get_fal_client(self):
        """Lazy load fal.ai client."""
        if self._fal_client is None:
            import fal_client

            # Set API key from config or environment
            api_key = self.config.fal_api_key or os.getenv("FAL_KEY", "")
            if not api_key:
                raise ValueError(
                    "FAL_KEY not set. Get your API key from https://fal.ai/dashboard/keys"
                )
            os.environ["FAL_KEY"] = api_key
            self._fal_client = fal_client
        return self._fal_client

    def _upload_to_fal(self, file_path: Path) -> str:
        """Upload a file to fal.ai storage and return the URL."""
        fal = self._get_fal_client()
        url = fal.upload_file(str(file_path))
        return url

    def _download_video(self, url: str, output_path: Path) -> Path:
        """Download video from URL to local path."""
        response = requests.get(url, stream=True)
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return output_path

    def animate_scene(
        self,
        image_path: Path,
        audio_path: Path,
        start_time: float,
        duration: float,
        output_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Optional[Path]:
        """
        Generate a video with motion and lip sync from image + audio.

        Args:
            image_path: Path to the scene image
            audio_path: Path to the full audio file
            start_time: Start time in the audio (seconds)
            duration: Duration of the scene (seconds)
            output_path: Path to save the output video
            progress_callback: Optional callback for progress updates

        Returns:
            Path to the generated video, or None if generation failed
        """
        if progress_callback:
            progress_callback("Preparing audio clip...", 0.1)

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
                    progress_callback("Uploading files to fal.ai...", 0.2)

                # Upload files to fal.ai storage
                image_url = self._upload_to_fal(image_path)
                audio_url = self._upload_to_fal(audio_clip_path)

                if progress_callback:
                    progress_callback("Generating video with Wan 2.2 S2V...", 0.4)

                # Call Wan 2.2 S2V endpoint
                fal = self._get_fal_client()

                result = fal.subscribe(
                    self.S2V_ENDPOINT,
                    arguments={
                        "image_url": image_url,
                        "audio_url": audio_url,
                    },
                    with_logs=True,
                    on_queue_update=lambda update: self._handle_queue_update(
                        update, progress_callback
                    ),
                )

                if progress_callback:
                    progress_callback("Downloading result...", 0.9)

                # Download result video
                if result and "video" in result:
                    result_url = result["video"]["url"]
                    self._download_video(result_url, output_path)

                    if progress_callback:
                        progress_callback("Animation complete!", 1.0)

                    return output_path
                else:
                    print(f"Wan 2.2 S2V result: {result}")
                    if progress_callback:
                        progress_callback("No video generated", 0.0)
                    return None

            except Exception as e:
                print(f"Wan 2.2 S2V animation failed: {e}")
                if progress_callback:
                    progress_callback(f"Animation failed: {e}", 0.0)
                return None

    def _handle_queue_update(self, update, progress_callback):
        """Handle queue status updates from fal.ai."""
        if progress_callback and hasattr(update, "logs"):
            for log in update.logs:
                print(f"[Wan S2V] {log.message}")

    def animate_scenes(
        self,
        scenes: list,
        audio_path: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> dict[int, Optional[Path]]:
        """
        Animate multiple scenes marked for animation.

        Args:
            scenes: List of Scene objects (only those with animated=True will be processed)
            audio_path: Path to the full audio file
            output_dir: Directory to save output videos
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


def check_wan_s2v_available(config: Optional[Config] = None) -> bool:
    """Check if Wan S2V animation is available (requires fal_client and API key)."""
    cfg = config or default_config
    try:
        import fal_client  # noqa: F401
        api_key = cfg.fal_api_key or os.getenv("FAL_KEY", "")
        return bool(api_key)
    except ImportError:
        return False
