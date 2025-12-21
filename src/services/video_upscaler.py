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

        target_width, target_height = RESOLUTIONS[target_resolution]

        logger.info(f"Final method after validation: {method}")
        if progress_callback:
            progress_callback(f"Upscaling to {target_resolution} using {method}...", 0.1)

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
                    preserve_audio, progress_callback
                )
            else:
                success = self._upscale_ffmpeg(
                    input_path, output_path, target_width, target_height,
                    preserve_audio, progress_callback
                )

            if success and progress_callback:
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
    ) -> bool:
        """Upscale using Real-ESRGAN AI model (ncnn-vulkan).

        Real-ESRGAN upscales by a fixed factor (usually 4x), so we may need
        to do a two-step process: AI upscale then resize to exact target.

        Uses persistent work directory for resume capability.
        """
        import hashlib
        import time

        if progress_callback:
            progress_callback("Starting Real-ESRGAN Vulkan upscaler...", 0.1)

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
            # Check for existing process via lock file to prevent duplicates
            import os
            lock_file = work_dir / "upscale.lock"
            if lock_file.exists():
                try:
                    lock_data = lock_file.read_text().strip().split("\n")
                    existing_pid = int(lock_data[0])
                    # Check if process is still running
                    try:
                        os.kill(existing_pid, 0)  # Signal 0 = check if process exists
                        # Process exists - check if it's actually realesrgan
                        result = subprocess.run(
                            ["ps", "-p", str(existing_pid), "-o", "comm="],
                            capture_output=True, text=True
                        )
                        if "realesrgan" in result.stdout.lower():
                            logger.info(f"Upscaling already in progress (PID {existing_pid}), attaching to existing process")
                            if progress_callback:
                                progress_callback(f"⏳ Upscaling already running (PID {existing_pid}), monitoring...", 0.22)
                            # Monitor the existing process with watchdog
                            start_time = time.time()
                            initial_upscaled = already_done
                            last_progress_count = initial_upscaled
                            last_progress_time = time.time()
                            STALL_TIMEOUT = 5  # Kill if no progress for 5 seconds
                            process_stalled = False

                            while True:
                                # Check if process still running
                                try:
                                    os.kill(existing_pid, 0)
                                except OSError:
                                    break  # Process finished

                                upscaled_count = len(list(upscaled_dir.glob("*.jpg")))

                                # Watchdog: check for stall
                                if upscaled_count > last_progress_count:
                                    last_progress_count = upscaled_count
                                    last_progress_time = time.time()
                                else:
                                    stall_duration = time.time() - last_progress_time
                                    if stall_duration > STALL_TIMEOUT:
                                        logger.warning(f"Attached process stalled for {stall_duration:.0f}s, killing...")
                                        if progress_callback:
                                            progress_callback(
                                                f"⚠️ Process stalled, killing and restarting... ({upscaled_count}/{total_frames} done)",
                                                0.22 + (0.58 * upscaled_count / total_frames)
                                            )
                                        # Kill the stalled process
                                        try:
                                            os.kill(existing_pid, 9)  # SIGKILL
                                        except OSError:
                                            pass
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
                                            f"Upscaling: {upscaled_count}/{total_frames} (ETA: {eta})",
                                            min(0.80, upscale_progress)
                                        )
                                time.sleep(1.0)

                            # Clean up lock file
                            if lock_file.exists():
                                lock_file.unlink()

                            # If process stalled, DON'T return - fall through to batch processing
                            if process_stalled:
                                logger.info("Stalled process killed, continuing with batch processing...")
                                time.sleep(2)  # Brief delay for GPU recovery
                                goto_reassembly = False  # Continue to batch processing
                            else:
                                # Process finished normally, check validation
                                upscaled_count = len(list(upscaled_dir.glob("*.jpg")))
                                logger.info(f"Attached process finished: {upscaled_count} frames total")
                                if upscaled_count < total_frames:
                                    logger.error(f"Not all frames upscaled: {upscaled_count}/{total_frames}")
                                    if progress_callback:
                                        progress_callback(f"⚠️ Only {upscaled_count}/{total_frames} frames upscaled. Restart to continue.", 0.25)
                                    return False
                                # Skip to reassembly
                                goto_reassembly = True
                        else:
                            # PID exists but not realesrgan - stale lock
                            lock_file.unlink()
                            goto_reassembly = False
                    except OSError:
                        # Process doesn't exist - stale lock file
                        lock_file.unlink()
                        goto_reassembly = False
                except (ValueError, IndexError):
                    # Invalid lock file
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

                # Process frames in batches to avoid GPU hangs
                # Using smaller batches due to ncnn-vulkan stability issues on Apple Silicon
                BATCH_SIZE = 50  # Small batches for stability
                SMALL_BATCH_SIZE = 25  # Even smaller after stalls
                pending_dir = work_dir / "pending"

                # Find models directory once
                home = Path.home()
                models_dir = home / ".local" / "share" / "realesrgan-ncnn-vulkan" / "models"
                if not models_dir.exists():
                    bin_path = Path(realesrgan_bin).parent
                    models_dir = bin_path / "models"

                # Process in batches
                start_time = time.time()
                total_to_process = len(remaining_frames)
                batch_num = 0
                consecutive_stalls = 0
                max_consecutive_stalls = 10  # Give up after 10 consecutive stalls

                while remaining_frames:
                    # Check for too many consecutive stalls
                    if consecutive_stalls >= max_consecutive_stalls:
                        logger.error(f"Too many consecutive stalls ({consecutive_stalls}), aborting")
                        if progress_callback:
                            current_done = len(list(upscaled_dir.glob("*.jpg")))
                            progress_callback(
                                f"❌ Process keeps stalling. {current_done}/{total_frames} done. Try smaller batches or different model.",
                                0.22 + (0.58 * current_done / total_frames)
                            )
                        break

                    batch_num += 1
                    # Use smaller batch after stalls for better stability
                    current_batch_size = SMALL_BATCH_SIZE if consecutive_stalls > 0 else BATCH_SIZE
                    batch = remaining_frames[:current_batch_size]
                    remaining_frames = remaining_frames[current_batch_size:]

                    # Create pending directory with hard links for this batch
                    if pending_dir.exists():
                        shutil.rmtree(pending_dir)
                    pending_dir.mkdir(parents=True, exist_ok=True)

                    for frame in batch:
                        os.link(str(frame.absolute()), str(pending_dir / frame.name))

                    current_done = len(list(upscaled_dir.glob("*.jpg")))
                    if progress_callback:
                        progress_callback(
                            f"📷 Batch {batch_num}: Processing {len(batch)} frames ({current_done}/{total_frames} total done)...",
                            0.22 + (0.58 * current_done / total_frames)
                        )

                    upscale_cmd = [
                        realesrgan_bin,
                        "-i", str(pending_dir),
                        "-o", str(upscaled_dir),
                        "-n", model_name,
                        "-f", "jpg",
                        "-j", "1:2:2",  # Conservative parallelism
                    ]
                    if models_dir.exists():
                        upscale_cmd.extend(["-m", str(models_dir)])

                    logger.info(f"Batch {batch_num}: Processing {len(batch)} frames")

                    # Run this batch
                    process = subprocess.Popen(
                        upscale_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )

                    # Write lock file
                    lock_file.write_text(f"{process.pid}\n{work_dir}")

                    # Monitor this batch with watchdog for stall detection
                    batch_start = len(list(upscaled_dir.glob("*.jpg")))
                    last_progress_count = batch_start
                    last_progress_time = time.time()
                    STALL_TIMEOUT = 5  # Kill process if no progress for 5 seconds
                    process_stalled = False

                    if progress_callback:
                        while process.poll() is None:
                            upscaled_count = len(list(upscaled_dir.glob("*.jpg")))
                            batch_done = upscaled_count - batch_start
                            total_done_now = upscaled_count

                            # Watchdog: check for stall
                            if upscaled_count > last_progress_count:
                                last_progress_count = upscaled_count
                                last_progress_time = time.time()
                            else:
                                stall_duration = time.time() - last_progress_time
                                if stall_duration > STALL_TIMEOUT:
                                    logger.warning(f"Process stalled for {stall_duration:.0f}s, killing and restarting...")
                                    if progress_callback:
                                        progress_callback(
                                            f"⚠️ Process stalled, auto-restarting... ({total_done_now}/{total_frames} done)",
                                            0.22 + (0.58 * total_done_now / total_frames)
                                        )
                                    process.kill()
                                    process_stalled = True
                                    break

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
                                progress_callback(
                                    f"Batch {batch_num}: {batch_done}/{len(batch)} | Total: {total_done_now}/{total_frames} (ETA: {eta})",
                                    min(0.80, upscale_progress)
                                )
                            time.sleep(1.0)

                    stdout, stderr = process.communicate(timeout=5) if not process_stalled else ("", "Process killed due to stall")

                    # Clean up lock file
                    if lock_file.exists():
                        lock_file.unlink()

                    if process.returncode != 0 or process_stalled:
                        if process_stalled:
                            consecutive_stalls += 1
                            logger.warning(f"Batch {batch_num} stalled (stall #{consecutive_stalls}) - will retry remaining frames")
                        else:
                            logger.error(f"Batch {batch_num} failed: {stderr}")
                            consecutive_stalls += 1  # Also count failures

                        # Re-check which frames still need processing and add to remaining
                        upscaled_files = {f.stem for f in upscaled_dir.glob("*.jpg")}
                        frames_from_this_batch = [f for f in batch if f.stem not in upscaled_files]
                        if frames_from_this_batch:
                            # Add un-processed frames back to beginning of queue
                            remaining_frames = frames_from_this_batch + remaining_frames
                            logger.info(f"Re-queued {len(frames_from_this_batch)} frames for retry")

                        # Brief delay before restart to let GPU recover
                        time.sleep(2)
                    else:
                        # Successful batch - reset stall counter
                        consecutive_stalls = 0

                    # Clean up pending directory after each batch
                    if pending_dir.exists():
                        shutil.rmtree(pending_dir)

                    current_done = len(list(upscaled_dir.glob("*.jpg")))
                    logger.info(f"Batch {batch_num} complete: {current_done}/{total_frames} frames done")

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
        reassemble_cmd.extend([
            "-vf", f"scale={target_width}:{target_height}:flags=lanczos",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
        ])

        if preserve_audio:
            reassemble_cmd.extend([
                "-map", "0:v", "-map", "1:a",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
            ])
        else:
            reassemble_cmd.extend(["-an"])

        reassemble_cmd.append(str(output_path))

        result = subprocess.run(reassemble_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Video reassembly failed: {result.stderr}")
            return False

        return output_path.exists()

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
