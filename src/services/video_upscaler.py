"""Video upscaling service for 4K output.

This service upscales videos to 4K (3840x2160) resolution using:
- Core ML Real-ESRGAN (Apple Silicon Neural Engine - FASTEST, recommended)
- MPS Real-ESRGAN (Apple Silicon GPU - slower fallback)
- fx-upscale with MetalFX (Apple Silicon, fast but no detail enhancement)
- FFmpeg with lanczos scaling (fast, always available)
- Video2X with Real-ESRGAN/Real-CUGAN (best quality, requires video2x)
- Real-ESRGAN ncnn-vulkan (good quality, requires realesrgan-ncnn-vulkan)

Quality/Speed ranking (best first):
1. coreml_realesrgan - FASTEST on Apple Silicon (Neural Engine, ~10x faster than MPS)
2. mps_realesrgan - Good AI upscaling on Apple Silicon GPU (slower)
3. video2x with Real-CUGAN or Real-ESRGAN - Best AI upscaling on NVIDIA
4. realesrgan-ncnn-vulkan - Good AI upscaling (may crash on Apple Silicon)
5. fx-upscale - Fast MetalFX upscaling (no detail enhancement)
6. ffmpeg lanczos - Fast, acceptable quality

Installation:
- Core ML Real-ESRGAN: Built-in (requires coremltools, spandrel)
- MPS Real-ESRGAN: Built-in (requires PyTorch with MPS support)
- fx-upscale: brew install finnvoor/tools/fx-upscale (macOS only)
- Video2X: pip install video2x (requires ffmpeg, ncnn models)
- Real-ESRGAN: brew install realesrgan-ncnn-vulkan (macOS) or download from GitHub
- FFmpeg: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)
"""

import logging
import subprocess
import shutil
from pathlib import Path
from typing import Callable, Literal, Optional

from src.config import config

logger = logging.getLogger(__name__)

# Target resolutions
RESOLUTIONS = {
    "1080p": (1920, 1080),
    "2K": (2560, 1440),
    "4K": (3840, 2160),
}

# Upscaling methods in order of preference (best first)
# realesrgan (ncnn-vulkan) is most stable and uses GPU via Vulkan
UPSCALE_METHODS = ["realesrgan", "coreml_realesrgan", "mps_realesrgan", "video2x", "fxupscale", "ffmpeg"]


def check_coreml_available() -> bool:
    """Check if Core ML is available (macOS with Apple Silicon)."""
    try:
        import platform
        if platform.system() != "Darwin":
            return False
        import coremltools
        return True
    except ImportError:
        return False


def check_mps_available() -> bool:
    """Check if MPS (Metal Performance Shaders) is available for AI upscaling."""
    try:
        import torch
        return torch.backends.mps.is_available()
    except ImportError:
        return False


def check_fxupscale_available() -> bool:
    """Check if fx-upscale is available (macOS with Apple Silicon)."""
    return shutil.which("fx-upscale") is not None


def check_video2x_available() -> bool:
    """Check if Video2X is available for AI upscaling."""
    try:
        result = subprocess.run(
            ["python3", "-c", "import video2x"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        # Also check for video2x CLI
        return shutil.which("video2x") is not None


def get_realesrgan_path() -> Optional[str]:
    """Find Real-ESRGAN binary in common locations."""
    # Check standard PATH first
    path = shutil.which("realesrgan-ncnn-vulkan")
    if path:
        return path

    # Check common install locations
    home = Path.home()
    common_paths = [
        home / ".local" / "bin" / "realesrgan-ncnn-vulkan",
        Path("/usr/local/bin/realesrgan-ncnn-vulkan"),
        Path("/opt/homebrew/bin/realesrgan-ncnn-vulkan"),
    ]

    for p in common_paths:
        if p.exists() and p.is_file():
            return str(p)

    return None


def check_realesrgan_available() -> bool:
    """Check if Real-ESRGAN is available for AI upscaling."""
    return get_realesrgan_path() is not None


def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available."""
    return shutil.which("ffmpeg") is not None


def get_best_available_method() -> str:
    """Get the best available upscaling method."""
    if check_coreml_available():
        return "coreml_realesrgan"  # FASTEST on Apple Silicon
    elif check_mps_available():
        return "mps_realesrgan"
    elif check_video2x_available():
        return "video2x"
    elif check_realesrgan_available():
        return "realesrgan"
    elif check_fxupscale_available():
        return "fxupscale"
    elif check_ffmpeg_available():
        return "ffmpeg"
    else:
        raise RuntimeError("No upscaling method available. Install ffmpeg at minimum.")


class VideoUpscaler:
    """Upscale videos to higher resolutions.

    Supports five methods (in order of quality):
    - 'mps_realesrgan': Best AI upscaling on Apple Silicon (adds detail)
    - 'video2x': Best AI upscaling on NVIDIA GPUs
    - 'realesrgan': Good AI upscaling via ncnn-vulkan (may crash on Apple Silicon)
    - 'fxupscale': Fast MetalFX upscaling (no detail enhancement)
    - 'ffmpeg': Fast lanczos scaling (always available)

    Usage:
        upscaler = VideoUpscaler()
        upscaler.upscale(
            input_path=Path("video_1080p.mp4"),
            output_path=Path("video_4k.mp4"),
            target_resolution="4K",
            method="auto",  # Uses best available
        )
    """

    def __init__(self):
        self._ffmpeg_available = check_ffmpeg_available()
        self._realesrgan_available = check_realesrgan_available()
        self._video2x_available = check_video2x_available()
        self._fxupscale_available = check_fxupscale_available()
        self._mps_available = check_mps_available()
        self._coreml_available = check_coreml_available()

        if not self._ffmpeg_available:
            raise RuntimeError("FFmpeg is required for video upscaling")

    def get_available_methods(self) -> list[str]:
        """Get list of available upscaling methods."""
        methods = []
        if self._coreml_available:
            methods.append("coreml_realesrgan")  # FASTEST on Apple Silicon
        if self._mps_available:
            methods.append("mps_realesrgan")
        if self._video2x_available:
            methods.append("video2x")
        if self._realesrgan_available:
            methods.append("realesrgan")
        if self._fxupscale_available:
            methods.append("fxupscale")
        if self._ffmpeg_available:
            methods.append("ffmpeg")
        return methods

    def upscale(
        self,
        input_path: Path,
        output_path: Path,
        target_resolution: Literal["1080p", "2K", "4K"] = "4K",
        method: Literal["auto", "coreml_realesrgan", "mps_realesrgan", "fxupscale", "video2x", "realesrgan", "ffmpeg"] = "auto",
        preserve_audio: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        model: Optional[str] = None,
        batch_size: Optional[int] = None,
        tile_size: Optional[int] = None,
        blocking: bool = True,
    ) -> bool:
        """
        Upscale a video to the target resolution.

        Args:
            input_path: Path to input video
            output_path: Path for upscaled output
            target_resolution: Target resolution ("1080p", "2K", or "4K")
            method: Upscaling method ("auto", "coreml_realesrgan", "mps_realesrgan", etc.)
                    "auto" uses the best available method (Core ML preferred on Apple Silicon)
            preserve_audio: Whether to preserve audio track
            progress_callback: Optional callback for progress updates
            model: AI model to use (for realesrgan):
                   - "RealESRGAN_General_x4_v3" (default, 10x faster, for cinematic/general content)
                   - "4xLSDIRCompactC3" (16x faster, slightly lower quality)
                   - "realesrgan-x4plus" (best quality but slow)
                   - "realesr-animevideov3" (for anime content)
            batch_size: Number of tiles to process at once for MPS. If None, auto-calculated
                       based on available GPU memory.
            tile_size: Size of tiles for MPS upscaling. If None, auto-calculated based on
                       available GPU memory.
            blocking: If False, returns immediately when workers are running (for realesrgan),
                     allowing UI to poll for progress. Returns True to indicate "in progress".

        Returns:
            True if successful, False otherwise
        """
        self._batch_size = batch_size
        self._tile_size = tile_size
        # Store model for use in realesrgan method
        # RealESRGAN_General_x4_v3 is 10x faster than x4plus and designed for general/cinematic content
        self._realesrgan_model = model or "RealESRGAN_General_x4_v3"
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return False

        # Auto-select best available method (Core ML is fastest on Apple Silicon)
        if method == "auto":
            if self._coreml_available:
                method = "coreml_realesrgan"  # FASTEST - uses Neural Engine
            elif self._mps_available:
                method = "mps_realesrgan"
            elif self._video2x_available:
                method = "video2x"
            elif self._realesrgan_available:
                method = "realesrgan"
            elif self._fxupscale_available:
                method = "fxupscale"
            else:
                method = "ffmpeg"
            logger.info(f"Auto-selected upscaling method: {method}")

        # Validate method availability
        if method == "coreml_realesrgan" and not self._coreml_available:
            logger.warning("Core ML not available, trying MPS")
            method = "mps_realesrgan" if self._mps_available else (
                "video2x" if self._video2x_available else (
                    "realesrgan" if self._realesrgan_available else (
                        "fxupscale" if self._fxupscale_available else "ffmpeg"
                    )
                )
            )

        if method == "mps_realesrgan" and not self._mps_available:
            logger.warning("MPS not available, trying Video2X")
            method = "video2x" if self._video2x_available else (
                "realesrgan" if self._realesrgan_available else (
                    "fxupscale" if self._fxupscale_available else "ffmpeg"
                )
            )

        if method == "fxupscale" and not self._fxupscale_available:
            logger.warning("fx-upscale not available, trying Video2X")
            method = "video2x" if self._video2x_available else (
                "realesrgan" if self._realesrgan_available else "ffmpeg"
            )

        if method == "video2x" and not self._video2x_available:
            logger.warning("Video2X not available, trying Real-ESRGAN")
            method = "realesrgan" if self._realesrgan_available else "ffmpeg"

        if method == "realesrgan" and not self._realesrgan_available:
            logger.warning("Real-ESRGAN not available, falling back to FFmpeg")
            method = "ffmpeg"

        # Handle "native" resolution (keep 4x upscaled size)
        if target_resolution == "native":
            target_width, target_height = None, None  # Will be set during assembly
            res_label = "native (4x upscaled)"
        else:
            target_width, target_height = RESOLUTIONS.get(target_resolution, (3840, 2160))
            res_label = target_resolution

        logger.info(f"Final method after validation: {method}")
        if progress_callback:
            progress_callback(f"Upscaling to {res_label} using {method}...", 0.1)

        try:
            if method == "coreml_realesrgan":
                logger.info("Calling _upscale_coreml (Neural Engine)...")
                success = self._upscale_coreml(
                    input_path, output_path, target_width, target_height,
                    preserve_audio, progress_callback
                )
            elif method == "mps_realesrgan":
                logger.info("Calling _upscale_mps_realesrgan...")
                success = self._upscale_mps_realesrgan(
                    input_path, output_path, target_width, target_height,
                    preserve_audio, progress_callback
                )
            elif method == "fxupscale":
                success = self._upscale_fxupscale(
                    input_path, output_path, target_width, target_height,
                    preserve_audio, progress_callback
                )
            elif method == "video2x":
                success = self._upscale_video2x(
                    input_path, output_path, target_width, target_height,
                    preserve_audio, progress_callback
                )
            elif method == "realesrgan":
                success = self._upscale_realesrgan(
                    input_path, output_path, target_width, target_height,
                    preserve_audio, progress_callback, blocking=blocking
                )
            else:
                success = self._upscale_ffmpeg(
                    input_path, output_path, target_width, target_height,
                    preserve_audio, progress_callback
                )

            # Only show "complete" if output file actually exists
            # In non-blocking mode, success=True but output may not exist yet
            if success and progress_callback and output_path.exists():
                progress_callback("Upscaling complete!", 1.0)

            return success

        except Exception as e:
            logger.error(f"Upscaling failed: {e}")
            return False

    def _upscale_coreml(
        self,
        input_path: Path,
        output_path: Path,
        target_width: int,
        target_height: int,
        preserve_audio: bool,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> bool:
        """Upscale using Core ML with Apple Neural Engine (FASTEST on Apple Silicon).

        This uses the dedicated Neural Engine hardware which is ~10x faster than
        MPS GPU and ~78x faster than CPU.
        """
        logger.info(f"_upscale_coreml called with input={input_path}")

        if progress_callback:
            progress_callback("Starting Core ML upscaler (Neural Engine)...", 0.12)

        try:
            from src.services.coreml_upscaler import CoreMLUpscaler
        except Exception as e:
            logger.error(f"Failed to import coreml_upscaler: {e}")
            if progress_callback:
                progress_callback(f"Core ML import failed: {e}", 0.4)
            raise

        if progress_callback:
            progress_callback("Initializing AI upscaler (Neural Engine)...", 0.15)

        try:
            # Use realesr-general-x4v3 for speed (recommended for Core ML)
            model_name = "realesr-general-x4v3"

            upscaler = CoreMLUpscaler(model_name=model_name)
            success = upscaler.upscale_video(input_path, output_path, progress_callback)

            # If output resolution doesn't match target, resize with ffmpeg
            if success and output_path.exists():
                actual_w, actual_h = self.get_video_resolution(output_path)
                if actual_w != target_width or actual_h != target_height:
                    if progress_callback:
                        progress_callback("Resizing to exact target resolution...", 0.95)
                    temp_path = output_path.with_suffix(".temp.mp4")
                    output_path.rename(temp_path)
                    self._resize_video(temp_path, output_path, target_width, target_height)
                    temp_path.unlink()

            return success

        except Exception as e:
            import traceback
            error_msg = f"Core ML upscaling failed: {type(e).__name__}: {e}"
            tb = traceback.format_exc()
            logger.error(error_msg)
            logger.error(f"Traceback: {tb}")
            if progress_callback:
                progress_callback(f"Core ML error: {str(e)[:150]}", 0.0)
            raise

    def _upscale_mps_realesrgan(
        self,
        input_path: Path,
        output_path: Path,
        target_width: int,
        target_height: int,
        preserve_audio: bool,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> bool:
        """Upscale using Real-ESRGAN with MPS (Apple Silicon GPU acceleration).

        This uses our custom MPS implementation that doesn't depend on
        the problematic basicsr/realesrgan packages.
        """
        logger.info(f"_upscale_mps_realesrgan called with input={input_path}")

        if progress_callback:
            progress_callback("Starting MPS Real-ESRGAN upscaler...", 0.12)

        try:
            from src.services.mps_upscaler import MPSUpscaler
        except Exception as e:
            logger.error(f"Failed to import mps_upscaler: {e}")
            if progress_callback:
                progress_callback(f"MPS import failed: {e}", 0.4)
            raise

        if progress_callback:
            progress_callback("Initializing AI upscaler (MPS)...", 0.15)

        try:
            # Use x4plus for 4K targets, x2plus for smaller
            model_name = "realesrgan-x4plus" if target_width >= 3840 else "realesrgan-x4plus"

            # Get settings from instance (set in upscale() method)
            # If None, MPSUpscaler will auto-calculate safe values based on GPU memory
            batch_size = getattr(self, '_batch_size', None)
            tile_size = getattr(self, '_tile_size', None)
            upscaler = MPSUpscaler(
                model_name=model_name,
                tile_size=tile_size,
                batch_size=batch_size,
                auto_memory=True,  # Enable automatic memory management
            )
            success = upscaler.upscale_video(input_path, output_path, progress_callback)

            # If output resolution doesn't match target, resize with ffmpeg
            if success and output_path.exists():
                actual_w, actual_h = self.get_video_resolution(output_path)
                if actual_w != target_width or actual_h != target_height:
                    if progress_callback:
                        progress_callback("Resizing to exact target resolution...", 0.95)
                    temp_path = output_path.with_suffix(".temp.mp4")
                    output_path.rename(temp_path)
                    self._resize_video(temp_path, output_path, target_width, target_height)
                    temp_path.unlink()

            return success

        except Exception as e:
            import traceback
            error_msg = f"MPS Real-ESRGAN failed: {type(e).__name__}: {e}"
            tb = traceback.format_exc()
            logger.error(error_msg)
            logger.error(f"Traceback: {tb}")
            # Show error in UI - DO NOT fall back to FFmpeg
            if progress_callback:
                progress_callback(f"MPS error: {str(e)[:150]}", 0.0)
            # Re-raise so user sees the actual error
            raise

    def _upscale_video2x(
        self,
        input_path: Path,
        output_path: Path,
        target_width: int,
        target_height: int,
        preserve_audio: bool,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> bool:
        """Upscale using Video2X with Real-ESRGAN or Real-CUGAN models."""

        if progress_callback:
            progress_callback("Upscaling with Video2X (AI)...", 0.2)

        # Calculate upscale ratio needed
        current_width, current_height = self.get_video_resolution(input_path)
        if current_width == 0:
            current_width, current_height = 1920, 1080  # Assume 1080p

        # Video2X uses integer scale factors (2x, 3x, 4x)
        width_ratio = target_width / current_width
        height_ratio = target_height / current_height
        scale_ratio = max(2, min(4, int(max(width_ratio, height_ratio) + 0.5)))

        # Try using video2x CLI first
        if shutil.which("video2x"):
            cmd = [
                "video2x",
                "-i", str(input_path),
                "-o", str(output_path),
                "-p", "realesrgan",  # Use Real-ESRGAN processor
                "-r", str(scale_ratio),
            ]

            logger.info(f"Running Video2X: {' '.join(cmd)}")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=7200,  # 2 hour timeout
                )

                if result.returncode != 0:
                    logger.error(f"Video2X error: {result.stderr}")
                    # Fall back to Real-ESRGAN or FFmpeg
                    if self._realesrgan_available:
                        return self._upscale_realesrgan(
                            input_path, output_path, target_width, target_height,
                            preserve_audio, progress_callback
                        )
                    return self._upscale_ffmpeg(
                        input_path, output_path, target_width, target_height,
                        preserve_audio, progress_callback
                    )

                # Video2X output may not be exact resolution, resize if needed
                if output_path.exists():
                    actual_w, actual_h = self.get_video_resolution(output_path)
                    if actual_w != target_width or actual_h != target_height:
                        # Resize to exact target
                        temp_path = output_path.with_suffix(".temp.mp4")
                        output_path.rename(temp_path)
                        self._resize_video(temp_path, output_path, target_width, target_height)
                        temp_path.unlink()

                return output_path.exists()

            except subprocess.TimeoutExpired:
                logger.error("Video2X timed out")
                return False
            except Exception as e:
                logger.error(f"Video2X failed: {e}")
                return False

        # Fallback: try Python video2x module
        try:
            # This would require video2x Python package
            logger.warning("Video2X CLI not found, falling back to Real-ESRGAN")
            if self._realesrgan_available:
                return self._upscale_realesrgan(
                    input_path, output_path, target_width, target_height,
                    preserve_audio, progress_callback
                )
            return self._upscale_ffmpeg(
                input_path, output_path, target_width, target_height,
                preserve_audio, progress_callback
            )
        except Exception as e:
            logger.error(f"Video2X Python module failed: {e}")
            return False

    def _upscale_fxupscale(
        self,
        input_path: Path,
        output_path: Path,
        target_width: int,
        target_height: int,
        preserve_audio: bool,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> bool:
        """Upscale using fx-upscale with Apple MetalFX (macOS only).

        fx-upscale uses Apple's MetalFX framework for fast GPU-accelerated upscaling.
        This is the fastest method on Apple Silicon Macs.
        """
        import tempfile

        if progress_callback:
            progress_callback("Upscaling with MetalFX (Apple GPU)...", 0.2)

        # fx-upscale outputs HEVC by default which has good quality
        # It only accepts --width or --height, not both
        # Use h264 codec for better compatibility

        # Determine if we need audio handling
        # fx-upscale preserves audio by default

        # Build the command - fx-upscale handles the output path automatically
        # Output goes to same directory with "_upscaled" suffix, so we use temp dir
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            temp_input = tmpdir_path / f"input{input_path.suffix}"

            # Copy input to temp dir to control output location
            import shutil as sh
            sh.copy2(input_path, temp_input)

            cmd = [
                "fx-upscale",
                str(temp_input),
                "--width", str(target_width),
                "--codec", "h264",  # h264 for compatibility
            ]

            logger.info(f"Running fx-upscale: {' '.join(cmd)}")

            if progress_callback:
                progress_callback("MetalFX upscaling in progress...", 0.4)

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=7200,  # 2 hour timeout
                )

                if result.returncode != 0:
                    logger.error(f"fx-upscale error: {result.stderr}")
                    return False

                if progress_callback:
                    progress_callback("MetalFX upscaling complete, finalizing...", 0.8)

                # Find the output file - fx-upscale adds " Upscaled" suffix (with space)
                stem = temp_input.stem
                suffix = temp_input.suffix

                # Try different possible output names (fx-upscale uses " Upscaled" with space)
                possible_outputs = [
                    tmpdir_path / f"{stem} Upscaled.mp4",
                    tmpdir_path / f"{stem} Upscaled.mov",
                    tmpdir_path / f"{stem} Upscaled{suffix}",
                    tmpdir_path / f"{stem}_upscaled.mp4",
                    tmpdir_path / f"{stem}_upscaled.mov",
                ]

                upscaled_file = None
                for p in possible_outputs:
                    if p.exists():
                        upscaled_file = p
                        break

                if not upscaled_file:
                    # List temp dir contents for debugging
                    contents = list(tmpdir_path.iterdir())
                    logger.error(f"fx-upscale output not found. Temp dir contains: {contents}")
                    return False

                # Move to final output path
                sh.move(str(upscaled_file), str(output_path))

                return output_path.exists()

            except subprocess.TimeoutExpired:
                logger.error("fx-upscale timed out")
                return False
            except Exception as e:
                logger.error(f"fx-upscale failed: {e}")
                return False

    def _resize_video(
        self, input_path: Path, output_path: Path, width: int, height: int
    ) -> bool:
        """Resize video to exact dimensions using FFmpeg."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", f"scale={width}:{height}:flags=lanczos",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "copy",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

    def _kill_stalled_realesrgan_processes(
        self,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> int:
        """Detect and kill any stalled realesrgan processes.

        Uses pgrep to find ALL running realesrgan processes (handles multiple workers).
        Returns the number of healthy workers still running.
        """
        import time

        work_dir_base = Path.home() / ".cache" / "songmaker" / "upscale_work"

        # Find all running realesrgan processes via pgrep
        try:
            result = subprocess.run(
                ["pgrep", "-f", "realesrgan-ncnn-vulkan"],
                capture_output=True, text=True, timeout=2
            )
            running_pids = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
        except Exception:
            running_pids = []

        if not running_pids:
            # No processes running, clean up any stale lock files
            for lock_file in work_dir_base.glob("*/upscale.lock"):
                logger.info(f"Cleaning up stale lock file (no running processes)")
                try:
                    lock_file.unlink()
                except OSError:
                    pass
            return 0

        # Processes are running - check if they're making progress
        # Find the active work directory
        active_work_dir = None
        for lock_file in work_dir_base.glob("*/upscale.lock"):
            active_work_dir = lock_file.parent
            break

        if not active_work_dir:
            # Workers running but no lock file - find by most recent upscaled dir
            for work_dir in sorted(work_dir_base.glob("realesrgan_*"), key=lambda p: p.stat().st_mtime, reverse=True):
                if (work_dir / "upscaled").exists():
                    active_work_dir = work_dir
                    break

        if not active_work_dir:
            return len(running_pids)  # Can't check progress without work dir

        upscaled_dir = active_work_dir / "upscaled"
        initial_count = len(list(upscaled_dir.glob("*.jpg"))) if upscaled_dir.exists() else 0

        if progress_callback:
            progress_callback(f"Checking {len(running_pids)} workers...", 0.05)

        # Wait 5 seconds and check again
        time.sleep(5)

        # Re-check running processes
        try:
            result = subprocess.run(
                ["pgrep", "-f", "realesrgan-ncnn-vulkan"],
                capture_output=True, text=True, timeout=2
            )
            current_pids = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
        except Exception:
            current_pids = []

        current_count = len(list(upscaled_dir.glob("*.jpg"))) if upscaled_dir.exists() else 0

        if current_count > initial_count:
            # Workers are making progress
            logger.info(f"{len(current_pids)} workers healthy: {current_count - initial_count} new frames in 5s")
            return len(current_pids)
        elif current_pids:
            # Workers running but no progress - stalled
            logger.warning(f"{len(current_pids)} workers stalled (0 frames in 5s), killing all...")
            if progress_callback:
                progress_callback(f"Killing {len(current_pids)} stalled workers...", 0.05)
            try:
                subprocess.run(["pkill", "-9", "-f", "realesrgan-ncnn-vulkan"],
                              capture_output=True, timeout=5)
                time.sleep(1)
            except Exception:
                pass
            # Clean up lock files
            for lock_file in work_dir_base.glob("*/upscale.lock"):
                try:
                    lock_file.unlink()
                except OSError:
                    pass
            return 0

        return 0  # No workers running

    def _get_running_worker_count(self) -> int:
        """Get count of running realesrgan workers."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "realesrgan-ncnn-vulkan"],
                capture_output=True, text=True, timeout=2
            )
            pids = [p for p in result.stdout.strip().split('\n') if p.strip()]
            return len(pids)
        except Exception:
            return 0

    def get_upscaling_status(self) -> dict:
        """Get current upscaling status for UI polling.

        Returns dict with:
        - status: "idle", "running", "complete", or "error"
        - workers: number of active workers
        - upscaled: number of upscaled frames
        - total: total number of frames
        - work_dir: path to work directory
        - progress: float 0-1
        """
        work_dir_base = Path.home() / ".cache" / "songmaker" / "upscale_work"

        # Find active work directory
        active_work_dir = None
        for lock_file in work_dir_base.glob("*/upscale.lock"):
            active_work_dir = lock_file.parent
            break

        if not active_work_dir:
            # No lock file, find most recent work dir
            for work_dir in sorted(work_dir_base.glob("realesrgan_*"), key=lambda p: p.stat().st_mtime, reverse=True):
                if (work_dir / "upscaled").exists():
                    active_work_dir = work_dir
                    break

        if not active_work_dir:
            return {"status": "idle", "workers": 0, "upscaled": 0, "total": 0, "work_dir": None, "progress": 0}

        frames_dir = active_work_dir / "frames"
        upscaled_dir = active_work_dir / "upscaled"

        total = len(list(frames_dir.glob("*.png"))) if frames_dir.exists() else 0
        upscaled = len(list(upscaled_dir.glob("*.jpg"))) if upscaled_dir.exists() else 0
        workers = self._get_running_worker_count()

        if total == 0:
            return {"status": "idle", "workers": workers, "upscaled": upscaled, "total": total, "work_dir": str(active_work_dir), "progress": 0}

        progress = upscaled / total

        if upscaled >= total:
            status = "complete"
        elif workers > 0:
            status = "running"
        else:
            status = "paused"  # Frames remaining but no workers

        return {
            "status": status,
            "workers": workers,
            "upscaled": upscaled,
            "total": total,
            "work_dir": str(active_work_dir),
            "progress": progress,
        }

    def _upscale_ffmpeg(
        self,
        input_path: Path,
        output_path: Path,
        target_width: int,
        target_height: int,
        preserve_audio: bool,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> bool:
        """Upscale using FFmpeg with lanczos scaling."""

        if progress_callback:
            progress_callback("Upscaling with FFmpeg (lanczos)...", 0.2)

        # Build FFmpeg command
        # Use lanczos scaling for best quality
        # -sws_flags lanczos+accurate_rnd for high quality scaling
        scale_filter = (
            f"scale={target_width}:{target_height}:"
            f"flags=lanczos+accurate_rnd,"
            f"unsharp=5:5:0.5:5:5:0.5"  # Light sharpening to reduce softness
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", scale_filter,
            "-c:v", "libx264",
            "-preset", "slow",  # Better quality
            "-crf", "18",  # High quality (lower = better, 18-23 is good)
            "-pix_fmt", "yuv420p",
        ]

        if preserve_audio:
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        else:
            cmd.extend(["-an"])

        cmd.append(str(output_path))

        logger.info(f"Running FFmpeg upscale: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout for long videos
            )

            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False

            if progress_callback:
                progress_callback("FFmpeg upscaling complete", 0.9)

            return output_path.exists()

        except subprocess.TimeoutExpired:
            logger.error("FFmpeg upscaling timed out")
            return False

    def _upscale_realesrgan(
        self,
        input_path: Path,
        output_path: Path,
        target_width: int,
        target_height: int,
        preserve_audio: bool,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        blocking: bool = True,
    ) -> bool:
        """Upscale using Real-ESRGAN AI model (ncnn-vulkan).

        Real-ESRGAN upscales by a fixed factor (usually 4x), so we may need
        to do a two-step process: AI upscale then resize to exact target.

        Uses persistent work directory for resume capability.

        Args:
            blocking: If False, returns immediately when workers are running,
                     allowing UI to poll for progress. Returns True to indicate
                     "in progress" (call again to check status).
        """
        import hashlib
        import os
        import time

        if progress_callback:
            progress_callback("Starting Real-ESRGAN Vulkan upscaler...", 0.1)

        # HEALTH CHECK: Detect and kill any stalled realesrgan processes
        # This runs every time the function is called (on button click or page reload)
        self._kill_stalled_realesrgan_processes(progress_callback)

        # Real-ESRGAN works on images, so we need to:
        # 1. Extract frames (or reuse existing)
        # 2. Upscale each frame (resume from where we left off)
        # 3. Reassemble video
        # 4. Add audio back

        # Use persistent work directory for resume capability
        work_dir_base = Path.home() / ".cache" / "songmaker" / "upscale_work"

        # Create unique work dir based on input file content (not mtime or filename which change)
        # Use file size + first 1MB content hash for stability across re-uploads
        file_stat = input_path.stat()
        hasher = hashlib.md5()
        hasher.update(str(file_stat.st_size).encode())
        # Read first 1MB for content hash (enough to identify unique files)
        with open(input_path, "rb") as f:
            hasher.update(f.read(1024 * 1024))
        work_hash = hasher.hexdigest()[:12]
        work_dir = work_dir_base / f"realesrgan_{work_hash}"

        logger.info(f"Input: {input_path.name}, size: {file_stat.st_size}, hash: {work_hash}")

        frames_dir = work_dir / "frames"
        upscaled_dir = work_dir / "upscaled"
        frames_dir.mkdir(parents=True, exist_ok=True)
        upscaled_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Work directory: {work_dir}")

        # Get expected frame count first
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_packets", "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0",
            str(input_path),
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        expected_frames = int(probe_result.stdout.strip()) if probe_result.returncode == 0 else 0

        # Check if frames already extracted
        existing_frames = len(list(frames_dir.glob("*.png")))

        if existing_frames >= expected_frames and expected_frames > 0:
            # Frames already extracted, skip extraction
            if progress_callback:
                progress_callback(f"Found {existing_frames} existing frames, skipping extraction", 0.20)
            logger.info(f"Reusing {existing_frames} existing extracted frames")
            total_frames = existing_frames
        else:
            # Step 1: Extract frames with progress monitoring
            if progress_callback:
                progress_callback(f"Extracting frames (expecting ~{expected_frames})...", 0.12)

            extract_cmd = [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-qscale:v", "2",
                str(frames_dir / "frame_%06d.png"),
            ]

            # Run extraction in background and monitor progress
            process = subprocess.Popen(
                extract_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Monitor extraction progress
            if expected_frames > 0 and progress_callback:
                while process.poll() is None:
                    current_frames = len(list(frames_dir.glob("*.png")))
                    if current_frames > 0:
                        extract_progress = 0.12 + (0.08 * current_frames / expected_frames)
                        progress_callback(
                            f"Extracting frames: {current_frames}/{expected_frames}",
                            min(0.20, extract_progress)
                        )
                    time.sleep(0.5)

            stdout, stderr = process.communicate()
            if process.returncode != 0:
                logger.error(f"Frame extraction failed: {stderr}")
                return False

            total_frames = len(list(frames_dir.glob("*.png")))
            logger.info(f"Extracted {total_frames} frames")

        # Step 2: Upscale frames with Real-ESRGAN (with resume support)
        # Get Real-ESRGAN binary path
        realesrgan_bin = get_realesrgan_path()
        if not realesrgan_bin:
            logger.error("Real-ESRGAN binary not found")
            return False

        # Use model from instance or default to RealESRGAN_General_x4_v3
        model_name = getattr(self, '_realesrgan_model', 'RealESRGAN_General_x4_v3')

        # Save metadata for reliable completion detection and resume
        # This prevents false "complete" detection if frames dir is corrupted
        # Also stores model name so corrupted frame fixes use the same model
        metadata_file = work_dir / "metadata.json"
        import json
        from datetime import datetime
        metadata = {
            "total_frames": total_frames,
            "input_file": str(input_path),
            "model_name": model_name,  # Store model for corrupted frame fixes
            "created_at": datetime.now().isoformat(),
        }
        metadata_file.write_text(json.dumps(metadata, indent=2))
        logger.info(f"Saved metadata: {total_frames} frames, model={model_name}")

        # Check which frames still need to be upscaled (for resume capability)
        all_frames = sorted(frames_dir.glob("*.png"))
        upscaled_files = {f.stem for f in upscaled_dir.glob("*.jpg")}
        remaining_frames = [f for f in all_frames if f.stem not in upscaled_files]

        already_done = len(all_frames) - len(remaining_frames)

        if progress_callback:
            progress_callback(f"📁 Work: {work_dir}", 0.21)
            if already_done > 0:
                progress_callback(f"⏩ Resuming: {already_done}/{total_frames} already done, {len(remaining_frames)} remaining", 0.22)
            progress_callback(f"🤖 Model: {model_name} | Frames: {total_frames}", 0.22)

        # If all frames are already upscaled, skip to reassembly
        if len(remaining_frames) == 0:
            if progress_callback:
                progress_callback(f"✅ All {total_frames} frames already upscaled!", 0.80)
            logger.info(f"All {total_frames} frames already upscaled, skipping to reassembly")
        else:
            # Check for existing realesrgan processes (not just lock file PID)
            # This properly handles multiple parallel workers
            import os
            lock_file = work_dir / "upscale.lock"

            # Check if ANY realesrgan process is running (handles multiple workers)
            try:
                result = subprocess.run(
                    ["pgrep", "-f", "realesrgan-ncnn-vulkan"],
                    capture_output=True, text=True, timeout=2
                )
                running_pids = [int(p) for p in result.stdout.strip().split('\n') if p.strip()]
            except Exception:
                running_pids = []

            if running_pids:
                logger.info(f"Upscaling already in progress ({len(running_pids)} workers)")
                upscaled_count = len(list(upscaled_dir.glob("*.jpg")))

                # NON-BLOCKING MODE: Return immediately, let UI poll for updates
                if not blocking:
                    upscale_progress = 0.22 + (0.58 * upscaled_count / total_frames)
                    if progress_callback:
                        progress_callback(
                            f"🚀 {len(running_pids)} workers | {upscaled_count}/{total_frames} | Work: {work_dir}",
                            min(0.80, upscale_progress)
                        )
                    logger.info(f"Non-blocking mode: returning (workers running, {upscaled_count}/{total_frames} done)")
                    return True  # Return True = in progress, call again to check

                # BLOCKING MODE: Monitor workers with watchdog
                if progress_callback:
                    progress_callback(f"⏳ {len(running_pids)} workers running, monitoring...", 0.22)

                start_time = time.time()
                initial_upscaled = already_done
                last_progress_count = initial_upscaled
                last_progress_time = time.time()
                STALL_TIMEOUT = 8  # 8 seconds for parallel workers
                process_stalled = False

                while True:
                    # Check if ANY realesrgan process is still running
                    try:
                        result = subprocess.run(
                            ["pgrep", "-f", "realesrgan-ncnn-vulkan"],
                            capture_output=True, text=True, timeout=2
                        )
                        current_pids = [p for p in result.stdout.strip().split('\n') if p.strip()]
                    except Exception:
                        current_pids = []

                    if not current_pids:
                        break  # All workers finished

                    upscaled_count = len(list(upscaled_dir.glob("*.jpg")))

                    # Watchdog: check for stall
                    if upscaled_count > last_progress_count:
                        last_progress_count = upscaled_count
                        last_progress_time = time.time()
                    else:
                        stall_duration = time.time() - last_progress_time
                        if stall_duration > STALL_TIMEOUT:
                            logger.warning(f"Workers stalled for {stall_duration:.0f}s, killing all...")
                            if progress_callback:
                                progress_callback(
                                    f"⚠️ Workers stalled, restarting... ({upscaled_count}/{total_frames} done)",
                                    0.22 + (0.58 * upscaled_count / total_frames)
                                )
                            # Kill all stalled workers
                            subprocess.run(["pkill", "-9", "-f", "realesrgan-ncnn-vulkan"],
                                          capture_output=True, timeout=5)
                            process_stalled = True
                            break

                    new_this_run = upscaled_count - initial_upscaled
                    if new_this_run > 0 and total_frames > 0:
                        elapsed = time.time() - start_time
                        time_per_frame = elapsed / new_this_run
                        remaining_count = total_frames - upscaled_count
                        remaining_time = time_per_frame * remaining_count
                        if remaining_time < 60:
                            eta = f"{int(remaining_time)}s"
                        elif remaining_time < 3600:
                            eta = f"{int(remaining_time/60)}m {int(remaining_time%60)}s"
                        else:
                            eta = f"{int(remaining_time/3600)}h {int((remaining_time%3600)/60)}m"
                        upscale_progress = 0.22 + (0.58 * upscaled_count / total_frames)
                        if progress_callback:
                            progress_callback(
                                f"🚀 {len(current_pids)} workers | {upscaled_count}/{total_frames} (ETA: {eta})",
                                min(0.80, upscale_progress)
                            )
                    time.sleep(1.0)

                # Clean up lock file
                if lock_file.exists():
                    lock_file.unlink()

                # After workers finish, check if we need to continue
                if process_stalled:
                    logger.info("Stalled workers killed, continuing with batch processing...")
                    time.sleep(2)  # Brief delay for GPU recovery
                    goto_reassembly = False  # Continue to batch processing
                else:
                    # Workers finished normally, check if all done
                    upscaled_count = len(list(upscaled_dir.glob("*.jpg")))
                    logger.info(f"Workers finished: {upscaled_count} frames total")
                    if upscaled_count < total_frames:
                        # Not all done - continue to batch processing
                        logger.info(f"Workers done but {total_frames - upscaled_count} frames remaining, continuing...")
                        goto_reassembly = False  # Continue to batch processing
                    else:
                        # All frames done - skip to reassembly
                        goto_reassembly = True
            elif lock_file.exists():
                # Lock file exists but no process - stale lock
                logger.info("Cleaning up stale lock file (no running workers)")
                lock_file.unlink()
                goto_reassembly = False
            else:
                goto_reassembly = False

            if goto_reassembly:
                pass  # Skip to reassembly section below
            else:
                # Recalculate remaining frames (in case we killed a stalled attached process)
                all_frames = sorted(frames_dir.glob("*.png"))
                upscaled_files = {f.stem for f in upscaled_dir.glob("*.jpg")}
                remaining_frames = [f for f in all_frames if f.stem not in upscaled_files]
                already_done = len(all_frames) - len(remaining_frames)

                if progress_callback and remaining_frames:
                    progress_callback(f"📷 {len(remaining_frames)} frames remaining to upscale...", 0.23)

                # PARALLEL PROCESSING: Run multiple realesrgan workers simultaneously
                # This significantly speeds up processing by utilizing more GPU/CPU resources
                NUM_WORKERS = 8  # Run 8 parallel processes
                # Each worker gets ALL its share of remaining frames (not small batches)
                # This allows the process to complete unattended

                # Find models directory once
                home = Path.home()
                models_dir = home / ".local" / "share" / "realesrgan-ncnn-vulkan" / "models"
                if not models_dir.exists():
                    bin_path = Path(realesrgan_bin).parent
                    models_dir = bin_path / "models"

                # Process in parallel batches
                start_time = time.time()
                round_num = 0
                consecutive_stalls = 0
                max_consecutive_stalls = 20  # Higher threshold for parallel processing

                while remaining_frames:
                    # Check for too many consecutive stalls
                    if consecutive_stalls >= max_consecutive_stalls:
                        logger.error(f"Too many consecutive stalls ({consecutive_stalls}), aborting")
                        if progress_callback:
                            current_done = len(list(upscaled_dir.glob("*.jpg")))
                            progress_callback(
                                f"❌ Process keeps stalling. {current_done}/{total_frames} done.",
                                0.22 + (0.58 * current_done / total_frames)
                            )
                        break

                    round_num += 1

                    # Determine how many workers to use
                    num_workers_this_round = min(NUM_WORKERS, len(remaining_frames))
                    num_workers_this_round = max(1, num_workers_this_round)

                    # Give ALL remaining frames to workers (divide evenly)
                    # This allows unattended completion - no need for multiple rounds
                    frames_this_round = remaining_frames
                    remaining_frames = []  # All frames assigned

                    # Split frames evenly among workers
                    worker_frames = []
                    frames_per = len(frames_this_round) // num_workers_this_round
                    for i in range(num_workers_this_round):
                        start_idx = i * frames_per
                        end_idx = start_idx + frames_per if i < num_workers_this_round - 1 else len(frames_this_round)
                        worker_frames.append(frames_this_round[start_idx:end_idx])

                    current_done = len(list(upscaled_dir.glob("*.jpg")))
                    if progress_callback:
                        progress_callback(
                            f"🚀 Round {round_num}: {num_workers_this_round} parallel workers, {len(frames_this_round)} frames ({current_done}/{total_frames} done)",
                            0.22 + (0.58 * current_done / total_frames)
                        )

                    # Create pending directories and start all workers
                    processes = []
                    pending_dirs = []

                    for worker_id, frames in enumerate(worker_frames):
                        if not frames:
                            continue

                        # Create unique pending directory for this worker
                        pending_dir = work_dir / f"pending_{worker_id}"
                        if pending_dir.exists():
                            shutil.rmtree(pending_dir)
                        pending_dir.mkdir(parents=True, exist_ok=True)
                        pending_dirs.append(pending_dir)

                        # Hard link frames to worker's pending dir
                        for frame in frames:
                            os.link(str(frame.absolute()), str(pending_dir / frame.name))

                        # Start worker process (detached so it survives Streamlit restarts)
                        upscale_cmd = [
                            realesrgan_bin,
                            "-i", str(pending_dir),
                            "-o", str(upscaled_dir),
                            "-n", model_name,
                            "-f", "jpg",
                            "-j", "1:2:1",  # Conservative per-worker (total: 8x this)
                        ]
                        if models_dir.exists():
                            upscale_cmd.extend(["-m", str(models_dir)])

                        # Use start_new_session=True so workers survive parent termination
                        # Redirect output to /dev/null since we can't read from detached processes
                        process = subprocess.Popen(
                            upscale_cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            start_new_session=True,  # Survives Streamlit restarts
                        )
                        processes.append((worker_id, process, len(frames)))
                        logger.info(f"Started detached worker {worker_id} (PID {process.pid}) with {len(frames)} frames")

                    # Write lock file with first worker PID
                    if processes:
                        lock_file.write_text(f"{processes[0][1].pid}\n{work_dir}")

                    # NON-BLOCKING MODE: Return immediately after starting workers
                    if not blocking:
                        current_done = len(list(upscaled_dir.glob("*.jpg")))
                        upscale_progress = 0.22 + (0.58 * current_done / total_frames)
                        if progress_callback:
                            progress_callback(
                                f"🚀 Started {len(processes)} workers | {current_done}/{total_frames} | Work: {work_dir}",
                                min(0.80, upscale_progress)
                            )
                        logger.info(f"Non-blocking mode: started {len(processes)} workers, returning immediately")
                        return True  # Return True = in progress, call again to check

                    # BLOCKING MODE: Monitor all workers with watchdog
                    last_progress_count = len(list(upscaled_dir.glob("*.jpg")))
                    last_progress_time = time.time()
                    STALL_TIMEOUT_FIRST = 15  # First frame needs model load time (per worker)
                    STALL_TIMEOUT = 8  # Subsequent: 8 seconds (more lenient for parallel)
                    first_frame_done = False
                    any_stalled = False

                    while any(p.poll() is None for _, p, _ in processes):
                        upscaled_count = len(list(upscaled_dir.glob("*.jpg")))
                        total_done_now = upscaled_count

                        # Watchdog: check for stall (global progress)
                        if upscaled_count > last_progress_count:
                            last_progress_count = upscaled_count
                            last_progress_time = time.time()
                            first_frame_done = True
                        else:
                            stall_duration = time.time() - last_progress_time
                            current_timeout = STALL_TIMEOUT_FIRST if not first_frame_done else STALL_TIMEOUT
                            if stall_duration > current_timeout:
                                logger.warning(f"Workers stalled for {stall_duration:.0f}s, killing all...")
                                if progress_callback:
                                    progress_callback(
                                        f"⚠️ Workers stalled, restarting... ({total_done_now}/{total_frames} done)",
                                        0.22 + (0.58 * total_done_now / total_frames)
                                    )
                                # Kill all workers
                                for _, p, _ in processes:
                                    try:
                                        p.kill()
                                    except Exception:
                                        pass
                                any_stalled = True
                                break

                        # Progress update
                        if total_done_now > already_done and total_frames > 0:
                            elapsed = time.time() - start_time
                            frames_this_session = total_done_now - already_done
                            time_per_frame = elapsed / frames_this_session
                            remaining_count = total_frames - total_done_now
                            remaining_time = time_per_frame * remaining_count

                            if remaining_time < 60:
                                eta = f"{int(remaining_time)}s"
                            elif remaining_time < 3600:
                                eta = f"{int(remaining_time/60)}m {int(remaining_time%60)}s"
                            else:
                                eta = f"{int(remaining_time/3600)}h {int((remaining_time%3600)/60)}m"

                            upscale_progress = 0.22 + (0.58 * total_done_now / total_frames)
                            if progress_callback:
                                progress_callback(
                                    f"🚀 {num_workers_this_round} workers | {total_done_now}/{total_frames} (ETA: {eta}) | Work: {work_dir}",
                                    min(0.80, upscale_progress)
                                )
                        time.sleep(1.0)

                    # Wait for all workers to finish and collect results
                    for worker_id, process, _ in processes:
                        try:
                            process.communicate(timeout=5)
                        except Exception:
                            pass

                    # Clean up lock file
                    if lock_file.exists():
                        lock_file.unlink()

                    # Handle stalls
                    if any_stalled:
                        consecutive_stalls += 1
                        logger.warning(f"Round {round_num} stalled (stall #{consecutive_stalls})")

                        # Re-check which frames still need processing
                        upscaled_files = {f.stem for f in upscaled_dir.glob("*.jpg")}
                        unprocessed = [f for f in frames_this_round if f.stem not in upscaled_files]
                        if unprocessed:
                            remaining_frames = unprocessed + remaining_frames
                            logger.info(f"Re-queued {len(unprocessed)} frames for retry")

                        time.sleep(2)  # GPU recovery
                    else:
                        consecutive_stalls = 0  # Reset on success

                    # Clean up all pending directories
                    for pending_dir in pending_dirs:
                        if pending_dir.exists():
                            shutil.rmtree(pending_dir)

                    current_done = len(list(upscaled_dir.glob("*.jpg")))
                    logger.info(f"Round {round_num} complete: {current_done}/{total_frames} frames done")

        upscaled_count = len(list(upscaled_dir.glob("*.jpg")))
        logger.info(f"Upscaled {upscaled_count} frames total")

        # Validate all frames were upscaled before proceeding to assembly
        if upscaled_count < total_frames:
            logger.error(f"Not all frames upscaled: {upscaled_count}/{total_frames}")
            if progress_callback:
                progress_callback(f"⚠️ Only {upscaled_count}/{total_frames} frames upscaled. Restart to continue.", 0.25)
            return False

        # Step 3: Get original video info (fps)
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        fps = result.stdout.strip() if result.returncode == 0 else "30"

        # Step 4: Reassemble video at target resolution
        if progress_callback:
            progress_callback("Reassembling video...", 0.8)

        # Build FFmpeg command with correct option order:
        # inputs first, then filters/options, then output
        reassemble_cmd = [
            "ffmpeg", "-y",
            "-framerate", fps,
            "-i", str(upscaled_dir / "frame_%06d.jpg"),
        ]

        if preserve_audio:
            # Add audio source as second input BEFORE any output options
            reassemble_cmd.extend(["-i", str(input_path)])

        # Now add output options (filters, codecs, etc.)
        # Only add scale filter if target dimensions are specified (not native)
        if target_width and target_height:
            reassemble_cmd.extend(["-vf", f"scale={target_width}:{target_height}:flags=lanczos"])

        reassemble_cmd.extend([
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
        ])

        if preserve_audio:
            # NOTE: Don't use -shortest as it can truncate video/audio if durations differ
            # Use all video frames AND full audio track
            reassemble_cmd.extend([
                "-map", "0:v", "-map", "1:a",
                "-c:a", "aac", "-b:a", "192k",
            ])
        else:
            reassemble_cmd.extend(["-an"])

        reassemble_cmd.append(str(output_path))

        result = subprocess.run(reassemble_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Video reassembly failed: {result.stderr}")
            return False

        return output_path.exists()

    def assemble_video(
        self,
        work_dir: Path,
        output_path: Path,
        original_input: Optional[Path] = None,
        target_resolution: str = "4K",
        preserve_audio: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        encoder: str = "hevc_videotoolbox",
        quality: str = "medium",
        threads: int = 0,
        buffer_size: int = 128,
    ) -> bool:
        """
        Assemble upscaled frames from a work directory into final video.

        This is a standalone assembly operation for when frames are upscaled
        but the video hasn't been reassembled yet.

        Args:
            work_dir: Path to work directory containing frames/ and upscaled/
            output_path: Path for final output video
            original_input: Path to original input video (for fps and audio).
                          If None, will try to find it from the work dir hash.
            target_resolution: Target resolution ("native", "1080p", "2K", "4K")
            preserve_audio: Whether to copy audio from original
            progress_callback: Optional callback for progress updates
            encoder: Video encoder ("hevc_videotoolbox", "h264_videotoolbox", "libx264")
            quality: Quality level ("high", "medium", "low")
            threads: Number of threads (0=auto)
            buffer_size: I/O buffer size in frames for thread_queue_size

        Returns:
            True if successful, False otherwise
        """
        work_dir = Path(work_dir)
        upscaled_dir = work_dir / "upscaled"
        frames_dir = work_dir / "frames"

        if not upscaled_dir.exists():
            logger.error(f"Upscaled directory not found: {upscaled_dir}")
            return False

        # Count frames
        upscaled_count = len(list(upscaled_dir.glob("*.jpg")))
        total_frames = len(list(frames_dir.glob("*.png")))

        if upscaled_count == 0:
            logger.error("No upscaled frames found")
            return False

        if upscaled_count < total_frames:
            logger.warning(f"Only {upscaled_count}/{total_frames} frames upscaled")
            if progress_callback:
                progress_callback(f"⚠️ Only {upscaled_count}/{total_frames} frames upscaled", 0.1)

        # Check for corrupted (0-byte) frames and re-upscale them
        corrupted_frames = [f for f in upscaled_dir.glob("*.jpg") if f.stat().st_size == 0]
        if corrupted_frames:
            logger.warning(f"Found {len(corrupted_frames)} corrupted (0-byte) frames, re-upscaling...")
            if progress_callback:
                progress_callback(f"🔧 Fixing {len(corrupted_frames)} corrupted frame(s)...", 0.12)

            realesrgan_bin = get_realesrgan_path()
            if realesrgan_bin:
                # Read model from metadata (if available) to use the same model as original upscaling
                import json
                metadata_file = work_dir / "metadata.json"
                model_name = "RealESRGAN_General_x4_v3"  # Default fallback
                if metadata_file.exists():
                    try:
                        metadata = json.loads(metadata_file.read_text())
                        model_name = metadata.get("model_name", model_name)
                        logger.info(f"Using model from metadata: {model_name}")
                    except Exception as e:
                        logger.warning(f"Could not read metadata: {e}, using default model")

                if progress_callback:
                    progress_callback(f"🔧 Fixing {len(corrupted_frames)} frame(s) with {model_name}...", 0.12)

                # Find models directory
                home = Path.home()
                models_dir = home / ".local" / "share" / "realesrgan-ncnn-vulkan" / "models"
                if not models_dir.exists():
                    bin_path = Path(realesrgan_bin).parent
                    models_dir = bin_path / "models"

                for corrupted in corrupted_frames:
                    frame_name = corrupted.stem  # e.g., "frame_006767"
                    source_png = frames_dir / f"{frame_name}.png"

                    if source_png.exists():
                        # Remove corrupted file
                        corrupted.unlink()
                        logger.info(f"Re-upscaling {frame_name} with {model_name}...")

                        # Re-upscale this frame (same params as main upscaling)
                        upscale_cmd = [
                            realesrgan_bin,
                            "-i", str(source_png),
                            "-o", str(upscaled_dir / f"{frame_name}.jpg"),
                            "-n", model_name,
                            "-f", "jpg",
                        ]
                        if models_dir.exists():
                            upscale_cmd.extend(["-m", str(models_dir)])

                        result = subprocess.run(upscale_cmd, capture_output=True, text=True, timeout=120)
                        if result.returncode == 0:
                            new_size = (upscaled_dir / f"{frame_name}.jpg").stat().st_size
                            logger.info(f"Fixed {frame_name}: {new_size} bytes")
                        else:
                            logger.error(f"Failed to re-upscale {frame_name}: {result.stderr}")
                    else:
                        logger.error(f"Source PNG not found for {frame_name}")

                if progress_callback:
                    progress_callback(f"✅ Fixed {len(corrupted_frames)} corrupted frame(s)", 0.15)
            else:
                logger.error("Cannot fix corrupted frames: realesrgan binary not found")
                if progress_callback:
                    progress_callback("❌ Cannot fix corrupted frames - realesrgan not found", 0.1)
                return False

        if progress_callback:
            progress_callback(f"🎬 Assembling {upscaled_count} upscaled frames...", 0.1)

        # Find original input if not provided
        if original_input is None:
            # Extract hash from work dir name (e.g., "realesrgan_abc123" -> "abc123")
            work_hash = work_dir.name.replace("realesrgan_", "")
            upload_dir = Path(config.output_dir) / "uploads"

            # Search for any video format
            for ext in ["mp4", "mov", "webm", "mkv", "avi"]:
                for f in upload_dir.glob(f"upscale_input_*{work_hash}*.{ext}"):
                    original_input = f
                    logger.info(f"Found original input: {original_input}")
                    break
                if original_input:
                    break

        if original_input and original_input.exists():
            if progress_callback:
                progress_callback(f"🔊 Audio source: {original_input.name}", 0.15)
        else:
            logger.warning("No original video found - output will have no audio")
            if progress_callback:
                progress_callback("⚠️ No audio source found", 0.15)

        # Get FPS from original or use default
        fps = "30"
        if original_input and original_input.exists():
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(original_input),
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                fps = result.stdout.strip()
                logger.info(f"Using FPS from original: {fps}")

        if progress_callback:
            progress_callback(f"📹 Using framerate: {fps}", 0.2)

        # Kill any stale FFmpeg processes using this work directory
        try:
            result = subprocess.run(
                ["pgrep", "-f", f"ffmpeg.*{work_dir.name}"],
                capture_output=True, text=True, timeout=2
            )
            if result.stdout.strip():
                stale_pids = result.stdout.strip().split('\n')
                logger.warning(f"Found {len(stale_pids)} stale FFmpeg processes for this session, killing...")
                if progress_callback:
                    progress_callback(f"🧹 Cleaning up {len(stale_pids)} stale processes...", 0.25)
                for pid in stale_pids:
                    try:
                        subprocess.run(["kill", "-9", pid.strip()], timeout=2)
                    except Exception:
                        pass
                import time
                time.sleep(1)  # Brief pause for cleanup
        except Exception as e:
            logger.debug(f"Process cleanup check failed: {e}")

        # Get target dimensions (None for native = no scaling)
        if target_resolution == "native":
            target_width, target_height = None, None
            res_label = "Native"
        else:
            target_width, target_height = RESOLUTIONS.get(target_resolution, (3840, 2160))
            res_label = f"{target_width}x{target_height}"

        if progress_callback:
            progress_callback(f"📐 Output: {res_label}", 0.22)

        # Build FFmpeg reassembly command
        reassemble_cmd = ["ffmpeg", "-y"]

        # Only add threads if user specified (0 = auto/let FFmpeg decide)
        if threads > 0:
            reassemble_cmd.extend(["-threads", str(threads)])

        # Add thread_queue_size for better I/O buffering
        # Use -start_number 1 since frames start at frame_000001.jpg
        reassemble_cmd.extend([
            "-thread_queue_size", str(buffer_size),
            "-framerate", fps,
            "-start_number", "1",
            "-i", str(upscaled_dir / "frame_%06d.jpg"),
        ])

        if preserve_audio and original_input and original_input.exists():
            reassemble_cmd.extend(["-i", str(original_input)])

        # Quality mapping for different encoders
        quality_map = {
            "hevc_videotoolbox": {"high": 80, "medium": 65, "low": 50},
            "h264_videotoolbox": {"high": 80, "medium": 65, "low": 50},
            "libx264": {"high": 18, "medium": 23, "low": 28},  # CRF (lower = better)
        }

        # Add video filter for scaling (skip if native)
        if target_width and target_height:
            reassemble_cmd.extend(["-vf", f"scale={target_width}:{target_height}:flags=lanczos"])

        # Check if hardware encoder can handle the resolution
        # H.264 VideoToolbox doesn't support 8K - fall back to software
        if encoder == "h264_videotoolbox" and (target_width is None or target_width > 4096):
            logger.warning("H.264 hardware encoder doesn't support 8K, falling back to libx264")
            if progress_callback:
                progress_callback("⚠️ H.264 hardware doesn't support 8K - using software encoder", 0.32)
            encoder = "libx264"

        # Apply encoder-specific settings
        if encoder in ["hevc_videotoolbox", "h264_videotoolbox"]:
            # Hardware encoding (VideoToolbox)
            q_value = quality_map.get(encoder, {}).get(quality, 65)
            reassemble_cmd.extend([
                "-c:v", encoder,
                "-q:v", str(q_value),
            ])
            if encoder == "hevc_videotoolbox":
                reassemble_cmd.extend(["-tag:v", "hvc1"])  # Compatibility tag for HEVC
            encoder_label = "Hardware"
        else:
            # Software encoding (libx264)
            crf = quality_map.get("libx264", {}).get(quality, 23)
            preset = "medium" if quality == "high" else "fast" if quality == "medium" else "veryfast"
            reassemble_cmd.extend([
                "-c:v", "libx264",
                "-preset", preset,
                "-crf", str(crf),
                "-pix_fmt", "yuv420p",
            ])
            encoder_label = "Software"

        if preserve_audio and original_input and original_input.exists():
            # NOTE: Don't use -shortest as it can truncate video if audio duration differs
            # Let FFmpeg use all video frames - audio will end naturally if shorter
            reassemble_cmd.extend([
                "-map", "0:v", "-map", "1:a",
                "-c:a", "aac", "-b:a", "192k",
            ])
        else:
            reassemble_cmd.extend(["-an"])

        reassemble_cmd.append(str(output_path))

        if progress_callback:
            progress_callback(f"🔄 Encoding {upscaled_count:,} frames ({encoder_label} - {quality})...", 0.3)

        logger.info(f"Running assembly: {' '.join(reassemble_cmd)}")

        import re
        import time

        # Use a log file for FFmpeg progress - allows monitoring from detached process
        ffmpeg_log = work_dir / "ffmpeg_assembly.log"
        lock_file = work_dir / "assembly.lock"

        # Check if assembly is ALREADY in progress - if so, just monitor it (don't restart!)
        if lock_file.exists():
            try:
                lock_content = lock_file.read_text().strip()
                pid = int(lock_content.split('\n')[0])
                # Check if process is still running
                result = subprocess.run(["ps", "-p", str(pid)], capture_output=True)
                if result.returncode == 0:
                    # Process still running - DON'T kill it, just return True
                    # UI will use check_assembly_status() to show progress
                    if progress_callback:
                        progress_callback(f"🔄 Assembly already in progress (PID {pid}) - do not restart", 0.35)
                    logger.info(f"Found existing assembly process: PID {pid}, NOT restarting")
                    return True  # Assembly is running, don't interfere
                else:
                    # Stale lock - clean up
                    logger.info(f"Stale lock file (PID {pid} not running), cleaning up")
                    lock_file.unlink()
            except Exception as e:
                logger.warning(f"Error checking lock file: {e}")
                lock_file.unlink()

        # No active assembly - kill any orphaned FFmpeg processes for this work dir
        try:
            result = subprocess.run(
                ["pgrep", "-f", f"ffmpeg.*{work_dir.name}"],
                capture_output=True, text=True, timeout=2
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                if progress_callback:
                    progress_callback(f"🧹 Cleaning up {len(pids)} orphaned FFmpeg process(es)...", 0.25)
                logger.warning(f"Killing {len(pids)} orphaned FFmpeg processes")
                for pid in pids:
                    try:
                        subprocess.run(["kill", "-9", pid.strip()], timeout=2)
                    except Exception:
                        pass
                time.sleep(1)
        except Exception as e:
            logger.debug(f"Process cleanup check failed: {e}")

        # Check if output already complete
        if output_path.exists():
            # Verify it has the expected frames
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(output_path),
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            is_complete = False
            if result.returncode == 0:
                try:
                    existing_frames = int(result.stdout.strip())
                    if existing_frames >= upscaled_count * 0.99:  # Allow 1% tolerance
                        file_size_mb = output_path.stat().st_size / 1024 / 1024
                        if progress_callback:
                            progress_callback(f"✅ Video already assembled: {file_size_mb:.1f} MB ({existing_frames:,} frames)", 1.0)
                        return True
                    else:
                        # Incomplete - remove and restart
                        logger.warning(f"Incomplete output: {existing_frames}/{upscaled_count} frames, removing")
                except ValueError:
                    logger.warning("Could not parse frame count from existing output, removing")

            if not is_complete:
                # Remove incomplete/corrupted output before restarting
                if progress_callback:
                    progress_callback("🗑️ Removing incomplete output file...", 0.28)
                output_path.unlink()
                time.sleep(0.5)

        # Use a TEMP file during encoding to prevent half-finished videos being shown
        # Only rename to final path when complete
        temp_output = output_path.with_suffix(".encoding.mp4")
        if temp_output.exists():
            temp_output.unlink()  # Remove any previous temp file

        # Replace output path in command with temp path
        reassemble_cmd[-1] = str(temp_output)

        # Start FFmpeg in detached session so it survives Streamlit restarts
        with open(ffmpeg_log, 'w') as log_file:
            # Add -progress pipe:1 to get structured progress output
            cmd_with_progress = reassemble_cmd.copy()
            # Insert -progress before output path
            output_idx = len(cmd_with_progress) - 1  # Last element is the output
            cmd_with_progress.insert(output_idx, "pipe:1")
            cmd_with_progress.insert(output_idx, "-progress")

            process = subprocess.Popen(
                cmd_with_progress,
                stdout=log_file,  # Progress goes here
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # SURVIVES STREAMLIT RESTARTS
            )

        # Write lock file with temp path info so we know to rename when done
        lock_file.write_text(f"{process.pid}\n{temp_output}\n{output_path}")
        logger.info(f"Started detached FFmpeg assembly: PID {process.pid}")

        if progress_callback:
            progress_callback(f"🚀 Assembly started (PID {process.pid}) - survives page refresh", 0.35)

        # Monitor progress for a while (but don't block forever)
        start_time = time.time()
        last_update = time.time()
        max_monitor_time = 600  # Monitor for up to 10 minutes, then return anyway

        # Track frames for our own fps calculation (more accurate than FFmpeg's average)
        last_frame_count = 0
        last_fps_check_time = time.time()
        calculated_fps = 0.0

        while time.time() - start_time < max_monitor_time:
            # Check if process still running
            result = subprocess.run(["ps", "-p", str(process.pid)], capture_output=True)
            if result.returncode != 0:
                # Process finished - check result
                break

            # Parse progress from log file
            if ffmpeg_log.exists():
                try:
                    log_content = ffmpeg_log.read_text()
                    # FFmpeg progress format: frame=1234\n
                    frame_matches = re.findall(r'frame=(\d+)', log_content)

                    if frame_matches:
                        current_frame = int(frame_matches[-1])  # Get latest
                        pct = min(0.95, 0.35 + 0.60 * (current_frame / upscaled_count))

                        # Calculate our own fps (more accurate than FFmpeg's slow-updating average)
                        now = time.time()
                        fps_interval = now - last_fps_check_time
                        if fps_interval >= 2.0:  # Update fps every 2 seconds
                            frames_done = current_frame - last_frame_count
                            if frames_done > 0:
                                calculated_fps = frames_done / fps_interval
                            last_frame_count = current_frame
                            last_fps_check_time = now

                        current_fps = calculated_fps if calculated_fps > 0 else 0

                        # Calculate ETA
                        remaining_frames = upscaled_count - current_frame
                        if current_fps > 0:
                            eta_seconds = remaining_frames / current_fps
                            if eta_seconds > 3600:
                                eta_str = f"{int(eta_seconds/3600)}h {int((eta_seconds%3600)/60)}m"
                            elif eta_seconds > 60:
                                eta_str = f"{int(eta_seconds/60)}m {int(eta_seconds%60)}s"
                            else:
                                eta_str = f"{int(eta_seconds)}s"
                        else:
                            eta_str = "calculating..."

                        # Update every 1 second
                        if time.time() - last_update > 1.0:
                            if progress_callback:
                                progress_callback(
                                    f"🎬 Encoding: {current_frame:,}/{upscaled_count:,} | {current_fps:.1f} fps | ETA: {eta_str}",
                                    pct
                                )
                            last_update = time.time()

                            # Check if done
                            if current_frame >= upscaled_count * 0.99:
                                break
                except Exception:
                    pass

            time.sleep(1.0)

        # Check final result
        # Give it a moment to finalize
        time.sleep(2)

        # Check if process is still running
        result = subprocess.run(["ps", "-p", str(process.pid)], capture_output=True)
        process_running = result.returncode == 0

        if process_running:
            # Still encoding - don't rename yet, just report progress
            if progress_callback:
                progress_callback(f"🔄 Assembly still running (PID {process.pid}) - safe to leave page", 0.5)
            return True  # Process is running, will complete in background

        # Process finished - check if temp file is complete, then rename
        if temp_output.exists():
            # Verify frame count in temp file
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(temp_output),
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            actual_frames = 0
            try:
                actual_frames = int(result.stdout.strip()) if result.returncode == 0 else 0
            except ValueError:
                pass

            if actual_frames >= upscaled_count * 0.99:
                # Complete! Rename temp to final
                import shutil
                shutil.move(str(temp_output), str(output_path))
                file_size_mb = output_path.stat().st_size / 1024 / 1024

                # Clean up lock file
                if lock_file.exists():
                    lock_file.unlink()

                if progress_callback:
                    progress_callback(f"✅ Video assembled: {output_path.name} ({file_size_mb:.1f} MB, {actual_frames:,} frames)", 1.0)
                logger.info(f"Successfully assembled video: {output_path}")
                return True
            else:
                # Temp file incomplete - encoding failed
                logger.error(f"Encoding incomplete: {actual_frames}/{upscaled_count} frames in temp file")
                if progress_callback:
                    progress_callback(f"❌ Encoding failed: only {actual_frames:,}/{upscaled_count:,} frames", 0.5)
                # Clean up
                if lock_file.exists():
                    lock_file.unlink()
                return False
        else:
            # No temp file - encoding failed to start or was interrupted
            logger.error("Assembly failed - no output file created")
            if progress_callback:
                progress_callback("❌ Assembly failed - check logs", 0.5)
            if lock_file.exists():
                lock_file.unlink()
            return False

    def check_assembly_status(self, work_dir: Path) -> dict:
        """
        Check if there's an assembly in progress for this work directory.

        Returns:
            dict with keys:
                - status: "idle" | "running" | "complete" | "failed"
                - pid: process ID if running
                - progress: 0-1 float
                - current_frame: int
                - total_frames: int
                - output_path: Path if complete
                - temp_path: Path if running
        """
        work_dir = Path(work_dir)
        lock_file = work_dir / "assembly.lock"
        ffmpeg_log = work_dir / "ffmpeg_assembly.log"
        upscaled_dir = work_dir / "upscaled"
        frames_dir = work_dir / "frames"

        total_frames = len(list(upscaled_dir.glob("*.jpg"))) if upscaled_dir.exists() else 0

        result = {
            "status": "idle",
            "pid": None,
            "progress": 0,
            "current_frame": 0,
            "total_frames": total_frames,
            "output_path": None,
            "temp_path": None,
        }

        if not lock_file.exists():
            return result

        try:
            lock_content = lock_file.read_text().strip().split('\n')
            pid = int(lock_content[0])
            temp_path = Path(lock_content[1]) if len(lock_content) > 1 else None
            final_path = Path(lock_content[2]) if len(lock_content) > 2 else None

            # Check if process is still running
            ps_result = subprocess.run(["ps", "-p", str(pid)], capture_output=True)
            if ps_result.returncode == 0:
                # Process is running
                result["status"] = "running"
                result["pid"] = pid
                result["temp_path"] = temp_path

                # Parse progress from log file
                if ffmpeg_log.exists():
                    import re
                    log_content = ffmpeg_log.read_text()
                    frame_matches = re.findall(r'frame=(\d+)', log_content)
                    if frame_matches:
                        current_frame = int(frame_matches[-1])
                        result["current_frame"] = current_frame
                        if total_frames > 0:
                            result["progress"] = current_frame / total_frames
            else:
                # Process finished - check if complete
                if temp_path and temp_path.exists():
                    # Check frame count
                    probe_cmd = [
                        "ffprobe", "-v", "error",
                        "-select_streams", "v:0",
                        "-show_entries", "stream=nb_frames",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        str(temp_path),
                    ]
                    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                    try:
                        actual_frames = int(probe_result.stdout.strip())
                        if actual_frames >= total_frames * 0.99:
                            # Complete - rename and return
                            import shutil
                            if final_path:
                                shutil.move(str(temp_path), str(final_path))
                                result["status"] = "complete"
                                result["output_path"] = final_path
                                result["progress"] = 1.0
                                result["current_frame"] = actual_frames
                                lock_file.unlink()
                        else:
                            result["status"] = "failed"
                            result["current_frame"] = actual_frames
                    except ValueError:
                        result["status"] = "failed"
                elif final_path and final_path.exists():
                    result["status"] = "complete"
                    result["output_path"] = final_path
                    result["progress"] = 1.0
                    lock_file.unlink()
                else:
                    result["status"] = "failed"
                    lock_file.unlink()

        except Exception as e:
            logger.warning(f"Error checking assembly status: {e}")

        return result

    def get_video_resolution(self, video_path: Path) -> tuple[int, int]:
        """Get the current resolution of a video."""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            str(video_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])

        return 0, 0

    def needs_upscaling(
        self, video_path: Path, target_resolution: str = "4K"
    ) -> bool:
        """Check if a video needs upscaling to reach target resolution."""
        current_width, current_height = self.get_video_resolution(video_path)
        target_width, target_height = RESOLUTIONS.get(target_resolution, (3840, 2160))

        # Needs upscaling if either dimension is smaller than target
        return current_width < target_width or current_height < target_height


def upscale_video(
    input_path: Path,
    output_path: Path,
    target_resolution: str = "4K",
    method: str = "ffmpeg",
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> bool:
    """Convenience function to upscale a video.

    Args:
        input_path: Path to input video
        output_path: Path for upscaled output
        target_resolution: "1080p", "2K", or "4K"
        method: "ffmpeg" (fast) or "realesrgan" (AI, better quality)
        progress_callback: Optional progress callback

    Returns:
        True if successful
    """
    try:
        upscaler = VideoUpscaler()
        return upscaler.upscale(
            input_path=input_path,
            output_path=output_path,
            target_resolution=target_resolution,
            method=method,
            progress_callback=progress_callback,
        )
    except Exception as e:
        logger.error(f"Upscaling failed: {e}")
        return False
