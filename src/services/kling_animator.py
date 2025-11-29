"""Kling AI lip sync animation service via fal.ai API."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

import requests
from pydub import AudioSegment

from src.config import Config, config as default_config


class KlingAnimator:
    """Generate lip-synced animations using Kling AI via fal.ai.

    Kling LipSync requires video input, so we use a 2-step process:
    1. Convert image to video (static or using Kling I2V)
    2. Apply Kling LipSync to the video

    Pricing (~$0.10 per 5 seconds of output):
    - Much faster than free Wan2.2-S2V (1-2 min vs 10 min per scene)
    - Supports singing (not just speech)
    - 720p/1080p output
    """

    # fal.ai endpoints
    I2V_ENDPOINT = "fal-ai/kling-video/v2/master/image-to-video"
    LIPSYNC_ENDPOINT = "fal-ai/kling-video/lipsync/audio-to-video"

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

    def _create_static_video(
        self,
        image_path: Path,
        duration: float,
        output_path: Path,
        fps: int = 30,
    ) -> Path:
        """Create a static video from an image using FFmpeg.

        This is a cheaper alternative to Kling I2V when you just need
        a base video for lip sync.
        """
        cmd = [
            "ffmpeg",
            "-y",
            "-loop", "1",
            "-i", str(image_path),
            "-t", str(min(duration, 10)),  # Kling max is 10s for lip sync
            "-vf", f"scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path

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
        resolution: str = "720p",
        use_i2v: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Optional[Path]:
        """
        Animate a scene image with lip-synced singing using Kling AI.

        Args:
            image_path: Path to the scene image
            audio_path: Path to the full audio file
            start_time: Start time in the audio (seconds)
            duration: Duration of the scene (seconds)
            output_path: Path to save the output video
            resolution: Video resolution ("720p" or "1080p")
            use_i2v: If True, use Kling I2V for initial animation (more expensive)
            progress_callback: Optional callback for progress updates

        Returns:
            Path to the generated video, or None if generation failed
        """
        if progress_callback:
            progress_callback("Preparing audio clip...", 0.1)

        # Kling lip sync max duration is 60s, video max is 10s
        # We'll process in chunks if needed
        effective_duration = min(duration, 10.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            audio_clip_path = temp_dir / "audio_clip.wav"
            video_path = temp_dir / "base_video.mp4"

            try:
                # Extract audio segment
                audio = AudioSegment.from_file(str(audio_path))
                start_ms = int(start_time * 1000)
                end_ms = int((start_time + effective_duration) * 1000)
                clip = audio[start_ms:end_ms]
                clip.export(str(audio_clip_path), format="wav")

                if progress_callback:
                    progress_callback("Creating base video...", 0.2)

                # Step 1: Create base video from image
                if use_i2v:
                    # Use Kling I2V for animated base (more expensive but better)
                    video_path = self._create_i2v_video(
                        image_path, effective_duration, video_path, progress_callback
                    )
                else:
                    # Use FFmpeg for static base video (cheaper)
                    video_path = self._create_static_video(
                        image_path, effective_duration, video_path
                    )

                if progress_callback:
                    progress_callback("Uploading files to Kling...", 0.4)

                # Upload files to fal.ai storage
                video_url = self._upload_to_fal(video_path)
                audio_url = self._upload_to_fal(audio_clip_path)

                if progress_callback:
                    progress_callback("Generating lip-sync animation...", 0.5)

                # Step 2: Apply Kling LipSync
                fal = self._get_fal_client()

                result = fal.subscribe(
                    self.LIPSYNC_ENDPOINT,
                    arguments={
                        "video_url": video_url,
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
                    print(f"Kling lip sync result: {result}")
                    if progress_callback:
                        progress_callback("No video generated", 0.0)
                    return None

            except Exception as e:
                print(f"Kling lip sync animation failed: {e}")
                if progress_callback:
                    progress_callback(f"Animation failed: {e}", 0.0)
                return None

    def _create_i2v_video(
        self,
        image_path: Path,
        duration: float,
        output_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Path:
        """Create animated video from image using Kling I2V."""
        fal = self._get_fal_client()

        if progress_callback:
            progress_callback("Uploading image for I2V...", 0.25)

        image_url = self._upload_to_fal(image_path)

        if progress_callback:
            progress_callback("Generating animation with Kling I2V...", 0.3)

        # Call Kling I2V
        result = fal.subscribe(
            self.I2V_ENDPOINT,
            arguments={
                "image_url": image_url,
                "duration": "5",  # Kling I2V supports 5s or 10s
                "aspect_ratio": "16:9",
            },
            with_logs=True,
        )

        if result and "video" in result:
            video_url = result["video"]["url"]
            return self._download_video(video_url, output_path)
        else:
            raise ValueError(f"Kling I2V failed: {result}")

    def _handle_queue_update(self, update, progress_callback):
        """Handle queue status updates from fal.ai."""
        if progress_callback and hasattr(update, "logs"):
            for log in update.logs:
                print(f"[Kling] {log.message}")

    def animate_scenes(
        self,
        scenes: list,
        audio_path: Path,
        output_dir: Path,
        resolution: str = "720p",
        use_i2v: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> dict[int, Optional[Path]]:
        """
        Animate multiple scenes marked for animation.

        Args:
            scenes: List of Scene objects (only those with animated=True will be processed)
            audio_path: Path to the full audio file
            output_dir: Directory to save output videos
            resolution: Video resolution ("720p" or "1080p")
            use_i2v: If True, use Kling I2V for initial animation
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
                use_i2v=use_i2v,
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


def check_kling_available(config: Optional[Config] = None) -> bool:
    """Check if Kling animation is available (requires fal_client and API key)."""
    cfg = config or default_config
    try:
        import fal_client  # noqa: F401
        api_key = cfg.fal_api_key or os.getenv("FAL_KEY", "")
        return bool(api_key)
    except ImportError:
        return False
