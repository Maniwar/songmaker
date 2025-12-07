"""Lip sync animation service with multiple backend support.

Backends:
- wan2s2v: Free Wan2.2-S2V via Hugging Face Spaces (slow, ~10 min/scene)
- kling: Paid Kling AI via fal.ai (fast, ~1-2 min/scene, ~$0.10/5s)
"""

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Literal, Optional

from pydub import AudioSegment

from src.config import Config, config as default_config

logger = logging.getLogger(__name__)


def _format_elapsed(seconds: float) -> str:
    """Format elapsed time as mm:ss or hh:mm:ss."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


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
    - Slow (~10-20 minutes per scene due to queue)
    - Queue times vary significantly based on server load
    - May timeout on very busy days
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
                logger.info(f"Extracting audio segment: {start_time:.2f}s to {start_time + duration:.2f}s")
                audio = AudioSegment.from_file(str(audio_path))
                start_ms = int(start_time * 1000)
                end_ms = int((start_time + duration) * 1000)
                clip = audio[start_ms:end_ms]
                clip.export(str(audio_clip_path), format="wav")
                logger.info(f"Audio clip exported to: {audio_clip_path}")

                if progress_callback:
                    progress_callback("Connecting to Wan2.2-S2V...", 0.2)

                # Get the Gradio client
                logger.info("Getting Gradio client for Wan2.2-S2V...")
                client = self._get_client()
                logger.info("Gradio client connected")

                if progress_callback:
                    progress_callback("Submitting to Wan2.2-S2V queue...", 0.3)

                # Call the Wan2.2-S2V API using submit() for queue status
                # API expects: ref_img, audio, resolution
                logger.info(f"Calling Wan2.2-S2V API with image={image_path}, resolution={resolution}")
                api_start = time.time()

                # Use submit() to get queue status updates
                job = client.submit(
                    ref_img=handle_file(str(image_path)),
                    audio=handle_file(str(audio_clip_path)),
                    resolution=resolution,
                    api_name="/predict"
                )

                # Monitor job with queue status updates
                max_wait_seconds = 3600  # 1 hour max wait
                max_queue_eta = 7200  # Fail if queue ETA > 2 hours
                last_status_update = 0.0

                while not job.done():
                    elapsed = time.time() - api_start

                    # Check timeout
                    if elapsed > max_wait_seconds:
                        logger.error(f"Wan2.2-S2V timed out after {_format_elapsed(elapsed)}")
                        if progress_callback:
                            progress_callback(f"Timed out after {_format_elapsed(elapsed)}", 0.0)
                        job.cancel()
                        return None

                    # Get queue status
                    try:
                        status = job.status()
                        # Debug: log full status object on first check
                        if elapsed < 10:
                            logger.info(f"[Queue Debug] Status object: {status}")
                            logger.info(f"[Queue Debug] Status attrs: {dir(status)}")

                        if hasattr(status, 'code'):
                            status_name = status.code.name if hasattr(status.code, 'name') else str(status.code)

                            if status_name == 'IN_QUEUE':
                                queue_pos = getattr(status, 'rank', 0)
                                queue_size = getattr(status, 'queue_size', 0)
                                eta_seconds = getattr(status, 'eta', 0) or 0

                                # Debug: log queue info
                                logger.info(f"[Queue] pos={queue_pos}, size={queue_size}, eta={eta_seconds}s ({eta_seconds/60:.1f}min)")

                                # Fail early if queue is too long
                                if eta_seconds > max_queue_eta:
                                    eta_hours = eta_seconds / 3600
                                    logger.error(f"Queue too long: {queue_pos}/{queue_size}, ETA {eta_hours:.1f}h")
                                    if progress_callback:
                                        progress_callback(
                                            f"Queue too long ({queue_pos}/{queue_size}, ~{eta_hours:.1f}h wait). "
                                            f"Use Seedance/AtlasCloud instead.", 0.0
                                        )
                                    job.cancel()
                                    return None

                                # Update progress with queue info (every 15 seconds)
                                if time.time() - last_status_update > 15:
                                    if eta_seconds > 0:
                                        eta_min = eta_seconds / 60
                                        msg = f"In queue: {queue_pos}/{queue_size} (~{eta_min:.0f}min wait)"
                                    else:
                                        msg = f"In queue: position {queue_pos} of {queue_size}"
                                    logger.info(msg)
                                    if progress_callback:
                                        progress_callback(msg, 0.35)
                                    last_status_update = time.time()

                            elif status_name == 'PROCESSING':
                                if time.time() - last_status_update > 10:
                                    msg = f"Processing... ({_format_elapsed(elapsed)})"
                                    logger.info(msg)
                                    if progress_callback:
                                        progress_callback(msg, 0.7)
                                    last_status_update = time.time()

                    except Exception as status_error:
                        logger.debug(f"Status check error (non-fatal): {status_error}")

                    time.sleep(5)

                # Get result
                result = job.result()
                elapsed = time.time() - api_start
                logger.info(f"API call completed in {_format_elapsed(elapsed)}")
                logger.info(f"API result type: {type(result)}, value: {result}")

                if progress_callback:
                    progress_callback(f"Processing result... (took {_format_elapsed(elapsed)})", 0.9)

                # Result is a dict with 'video' and 'subtitles' keys
                # The video value is a filepath string
                video_path = None
                if isinstance(result, dict):
                    video_path = result.get('video')
                    logger.info(f"Extracted video path from dict: {video_path}")
                elif isinstance(result, str):
                    video_path = result
                    logger.info(f"Result is string path: {video_path}")

                if video_path:
                    video_exists = Path(video_path).exists()
                    logger.info(f"Video path exists: {video_exists}")
                    if video_exists:
                        # Copy to output path
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(video_path, output_path)
                        logger.info(f"Video copied to: {output_path}")

                        if progress_callback:
                            progress_callback("Animation complete!", 1.0)

                        return output_path
                    else:
                        logger.error(f"Video file does not exist: {video_path}")
                        if progress_callback:
                            progress_callback("Video file not found", 0.0)
                else:
                    logger.error(f"No video path in result: {result}")
                    if progress_callback:
                        progress_callback("No video generated", 0.0)

            except Exception as e:
                logger.error(f"Lip sync animation failed: {e}", exc_info=True)
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
