"""Video upscaling service for 4K output.

This service upscales videos to 4K (3840x2160) resolution using:
- MPS Real-ESRGAN (Apple Silicon, best quality AI with GPU acceleration)
- fx-upscale with MetalFX (Apple Silicon, fast but no detail enhancement)
- FFmpeg with lanczos scaling (fast, always available)
- Video2X with Real-ESRGAN/Real-CUGAN (best quality, requires video2x)
- Real-ESRGAN ncnn-vulkan (good quality, requires realesrgan-ncnn-vulkan)

Quality ranking (best to worst):
1. mps_realesrgan - Best AI upscaling on Apple Silicon (adds detail)
2. video2x with Real-CUGAN or Real-ESRGAN - Best AI upscaling on NVIDIA
3. realesrgan-ncnn-vulkan - Good AI upscaling (may crash on Apple Silicon)
4. fx-upscale - Fast MetalFX upscaling (no detail enhancement)
5. ffmpeg lanczos - Fast, acceptable quality

Installation:
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

# Upscaling methods in order of quality (best first)
UPSCALE_METHODS = ["mps_realesrgan", "video2x", "realesrgan", "fxupscale", "ffmpeg"]


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
    if check_mps_available():
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

        if not self._ffmpeg_available:
            raise RuntimeError("FFmpeg is required for video upscaling")

    def get_available_methods(self) -> list[str]:
        """Get list of available upscaling methods."""
        methods = []
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
        method: Literal["auto", "mps_realesrgan", "fxupscale", "video2x", "realesrgan", "ffmpeg"] = "auto",
        preserve_audio: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        model: Optional[str] = None,
        batch_size: int = 8,
        tile_size: int = 768,
    ) -> bool:
        """
        Upscale a video to the target resolution.

        Args:
            input_path: Path to input video
            output_path: Path for upscaled output
            target_resolution: Target resolution ("1080p", "2K", or "4K")
            method: Upscaling method ("auto", "mps_realesrgan", "video2x", "realesrgan", "fxupscale", or "ffmpeg")
                    "auto" uses the best available method
            preserve_audio: Whether to preserve audio track
            progress_callback: Optional callback for progress updates
            model: AI model to use (for realesrgan: "realesrgan-x4plus", "realesr-animevideov3",
                   or custom models like "4xNomos2_hq_dat2" if installed)
            batch_size: Number of tiles to process at once for MPS (higher = faster, more memory)
            tile_size: Size of tiles for MPS upscaling (larger = fewer tiles = faster)
                       - M1 (8-16GB): 384-512
                       - M1 Pro/Max (16-32GB): 768-896
                       - M2/M3 Ultra (64-128GB): 1024

        Returns:
            True if successful, False otherwise
        """
        self._batch_size = batch_size
        self._tile_size = tile_size
        # Store model for use in realesrgan method
        self._realesrgan_model = model or "realesrgan-x4plus"
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return False

        # Auto-select best available method
        if method == "auto":
            if self._mps_available:
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

        if progress_callback:
            progress_callback(f"Upscaling to {target_resolution} using {method}...", 0.1)

        try:
            if method == "mps_realesrgan":
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
        from src.services.mps_upscaler import MPSUpscaler

        if progress_callback:
            progress_callback("Initializing AI upscaler (MPS)...", 0.15)

        try:
            # Use x4plus for 4K targets, x2plus for smaller
            model_name = "realesrgan-x4plus" if target_width >= 3840 else "realesrgan-x4plus"

            # Get settings from instance (set in upscale() method)
            batch_size = getattr(self, '_batch_size', 8)
            tile_size = getattr(self, '_tile_size', 768)
            upscaler = MPSUpscaler(
                model_name=model_name,
                tile_size=tile_size,
                batch_size=batch_size,
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
            logger.error(f"MPS Real-ESRGAN failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Fall back to ffmpeg
            if progress_callback:
                progress_callback(f"MPS failed ({type(e).__name__}), falling back to FFmpeg...", 0.5)
            return self._upscale_ffmpeg(
                input_path, output_path, target_width, target_height,
                preserve_audio, progress_callback
            )

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
        """Upscale using Real-ESRGAN AI model.

        Real-ESRGAN upscales by a fixed factor (usually 4x), so we may need
        to do a two-step process: AI upscale then resize to exact target.
        """
        import tempfile

        if progress_callback:
            progress_callback("Upscaling with Real-ESRGAN AI...", 0.2)

        # Real-ESRGAN works on images, so we need to:
        # 1. Extract frames
        # 2. Upscale each frame
        # 3. Reassemble video
        # 4. Add audio back

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            frames_dir = tmpdir / "frames"
            upscaled_dir = tmpdir / "upscaled"
            frames_dir.mkdir()
            upscaled_dir.mkdir()

            # Step 1: Extract frames
            if progress_callback:
                progress_callback("Extracting frames...", 0.3)

            extract_cmd = [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-qscale:v", "2",
                str(frames_dir / "frame_%06d.png"),
            ]

            result = subprocess.run(extract_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Frame extraction failed: {result.stderr}")
                return False

            # Step 2: Upscale frames with Real-ESRGAN
            if progress_callback:
                progress_callback("AI upscaling frames...", 0.5)

            # Get Real-ESRGAN binary path
            realesrgan_bin = get_realesrgan_path()
            if not realesrgan_bin:
                logger.error("Real-ESRGAN binary not found")
                return False

            # Use model from instance or default to realesrgan-x4plus
            model_name = getattr(self, '_realesrgan_model', 'realesrgan-x4plus')

            # Find models directory
            home = Path.home()
            models_dir = home / ".local" / "share" / "realesrgan-ncnn-vulkan" / "models"
            if not models_dir.exists():
                # Check next to binary
                bin_path = Path(realesrgan_bin).parent
                models_dir = bin_path / "models"

            upscale_cmd = [
                realesrgan_bin,
                "-i", str(frames_dir),
                "-o", str(upscaled_dir),
                "-n", model_name,
                "-f", "png",
            ]

            # Add models path if it exists
            if models_dir.exists():
                upscale_cmd.extend(["-m", str(models_dir)])

            logger.info(f"Running Real-ESRGAN: {' '.join(upscale_cmd)}")
            result = subprocess.run(upscale_cmd, capture_output=True, text=True, timeout=7200)
            if result.returncode != 0:
                logger.error(f"Real-ESRGAN failed: {result.stderr}")
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

            reassemble_cmd = [
                "ffmpeg", "-y",
                "-framerate", fps,
                "-i", str(upscaled_dir / "frame_%06d.png"),
                "-vf", f"scale={target_width}:{target_height}:flags=lanczos",
                "-c:v", "libx264",
                "-preset", "slow",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
            ]

            if preserve_audio:
                # Add audio from original
                reassemble_cmd.extend([
                    "-i", str(input_path),
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
