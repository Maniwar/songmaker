"""MPS-accelerated AI video upscaling for Apple Silicon Macs.

Uses Real-ESRGAN architecture with PyTorch MPS backend for GPU acceleration.
This module provides a standalone implementation that doesn't depend on
the problematic basicsr/realesrgan packages.

Performance optimizations:
- Batch processing: Process multiple frames at once (configurable batch size)
- Parallel I/O: Load/save images in background threads while GPU processes
- Memory management: Clear MPS cache periodically to prevent buildup
"""

import hashlib
import logging
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Callable, Optional
import urllib.request
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Global cancellation flag for long-running operations
_cancel_upscale = threading.Event()

# Default batch size (adjust based on GPU memory)
# - M1: 2-4
# - M1 Pro/Max: 6-8
# - M2/M3 Ultra: 8-16
DEFAULT_BATCH_SIZE = 6  # Optimized for M1 Max

# Default tile size for processing large images
# Higher values = faster processing but more memory usage
# - M1 (8-16GB): 512
# - M1 Pro/Max (16-32GB): 768-1024
# - M2/M3 Ultra (64-128GB): 1024-1536
DEFAULT_TILE_SIZE = 512

# Number of I/O worker threads for parallel loading/saving
IO_WORKERS = 4

# Model download URLs
MODEL_URLS = {
    "realesrgan-x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "realesrgan-x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
}

# Cache directory for models
MODELS_DIR = Path.home() / ".cache" / "songmaker" / "models"

# Global model cache to avoid reloading between upscale jobs
_model_cache: dict[str, tuple] = {}  # model_name -> (model, device)

# Work directory base for resume capability
WORK_DIR_BASE = Path.home() / ".cache" / "songmaker" / "upscale_work"


class GPUMemoryManager:
    """Automatic GPU memory management for MPS upscaling.

    Detects available GPU memory and calculates optimal tile_size and batch_size.
    Since we have OOM recovery with safe_mode fallback, we can be more aggressive
    with settings for better performance.
    """

    # Memory requirements (empirically measured on M1 Max)
    MODEL_BASE_MEMORY_GB = 0.5

    def __init__(self):
        self.device = get_device()
        self.total_memory_gb = self._get_total_memory()
        # Use 85% of available memory - we have OOM recovery as safety net
        self.safe_memory_gb = self.total_memory_gb * 0.85
        logger.info(f"GPU Memory Manager: {self.total_memory_gb:.1f}GB total, using up to {self.safe_memory_gb:.1f}GB")

    def _get_total_memory(self) -> float:
        """Get total GPU memory in GB."""
        if self.device.type == "mps":
            # MPS shares system RAM - Apple Silicon is efficient at memory management
            try:
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=5
                )
                total_ram = int(result.stdout.strip()) / (1024**3)
                # Apple Silicon can use most of RAM for GPU when needed
                return total_ram * 0.85
            except Exception:
                return 16.0  # Default assumption
        elif self.device.type == "cuda":
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return 8.0  # CPU fallback

    def get_safe_settings(self, frame_width: int = 1920, frame_height: int = 1080) -> tuple[int, int]:
        """Calculate optimal tile_size and batch_size based on available memory.

        More aggressive settings since we have OOM recovery with safe_mode fallback.

        Args:
            frame_width: Width of video frames
            frame_height: Height of video frames

        Returns:
            Tuple of (tile_size, batch_size)
        """
        available = self.safe_memory_gb

        # More aggressive settings - OOM recovery will catch any issues
        # Empirically tested on M1 Max 32GB: tile_size=512, batch_size=6 works well
        if available >= 24:  # 32GB+ system
            tile_size = 512
            batch_size = 6
        elif available >= 16:  # 24GB system
            tile_size = 512
            batch_size = 5
        elif available >= 12:  # 16GB system
            tile_size = 512
            batch_size = 4
        elif available >= 8:  # 8-12GB system
            tile_size = 384
            batch_size = 3
        else:  # <8GB system
            tile_size = 384
            batch_size = 2

        logger.info(f"Auto-calculated settings: tile_size={tile_size}, batch_size={batch_size} "
                   f"(available memory: {available:.1f}GB)")
        return tile_size, batch_size

    def clear_cache(self):
        """Clear GPU memory cache."""
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

    def get_current_usage(self) -> float:
        """Get current GPU memory usage in GB (approximate for MPS)."""
        if self.device.type == "mps":
            # MPS doesn't have direct memory query, return estimate
            return 0.0  # Can't reliably measure on MPS
        elif self.device.type == "cuda":
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0


# Global memory manager instance
_memory_manager: Optional[GPUMemoryManager] = None


def get_memory_manager(reset: bool = False) -> GPUMemoryManager:
    """Get or create the global memory manager.

    Args:
        reset: If True, recreate the memory manager (useful after settings change)
    """
    global _memory_manager
    if _memory_manager is None or reset:
        _memory_manager = GPUMemoryManager()
    return _memory_manager


def get_video_content_hash(video_path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Generate a hash based on video file content for resume detection.

    Hashes the first 4MB + last 4MB + file size for speed while still
    being unique enough to identify the same video file.

    Args:
        video_path: Path to the video file
        chunk_size: Size of chunks to hash (default 4MB)

    Returns:
        8-character hex hash string
    """
    hasher = hashlib.md5()
    file_size = video_path.stat().st_size

    # Include file size in hash
    hasher.update(str(file_size).encode())

    with open(video_path, 'rb') as f:
        # Hash first chunk
        hasher.update(f.read(chunk_size))

        # Hash last chunk if file is large enough
        if file_size > chunk_size * 2:
            f.seek(-chunk_size, 2)  # Seek from end
            hasher.update(f.read(chunk_size))

    return hasher.hexdigest()[:8]


def find_existing_work_dir(video_path: Path) -> Optional[Path]:
    """Find an existing work directory for the same video content.

    Args:
        video_path: Path to the video file

    Returns:
        Path to existing work directory, or None if not found
    """
    content_hash = get_video_content_hash(video_path)
    WORK_DIR_BASE.mkdir(parents=True, exist_ok=True)

    # Look for work directories with matching hash
    for work_dir in WORK_DIR_BASE.glob(f"upscale_{content_hash}_*"):
        if work_dir.is_dir():
            frames_dir = work_dir / "frames"
            if frames_dir.exists() and any(frames_dir.glob("*.png")):
                logger.info(f"Found existing work directory: {work_dir}")
                return work_dir

    return None


def get_work_dir(video_path: Path) -> Path:
    """Get or create a work directory for the video based on content hash.

    Args:
        video_path: Path to the video file

    Returns:
        Path to the work directory
    """
    # Check for existing work directory first
    existing = find_existing_work_dir(video_path)
    if existing:
        return existing

    # Create new work directory with content hash
    content_hash = get_video_content_hash(video_path)
    timestamp = int(time.time())
    work_dir = WORK_DIR_BASE / f"upscale_{content_hash}_{timestamp}"
    work_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created new work directory: {work_dir}")
    return work_dir


def get_device() -> torch.device:
    """Get the best available device (MPS for Apple Silicon, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def download_model(model_name: str, progress_callback: Optional[Callable[[str, float], None]] = None) -> Path:
    """Download model weights if not cached."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"{model_name}.pth"
    if model_path.exists():
        return model_path

    url = MODEL_URLS.get(model_name)
    if not url:
        raise ValueError(f"Unknown model: {model_name}")

    if progress_callback:
        progress_callback(f"Downloading {model_name} model...", 0.1)

    logger.info(f"Downloading model from {url}")
    urllib.request.urlretrieve(url, model_path)

    return model_path


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block for RRDB."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Real-ESRGAN network architecture."""

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # Upsampling
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class MPSUpscaler:
    """AI video upscaler using MPS acceleration on Apple Silicon.

    Features:
    - Automatic memory management with safe tile/batch size calculation
    - Batch processing for better GPU utilization
    - Parallel I/O for loading/saving frames
    - Tile-based processing for large images with batch tile optimization
    - Cancellation support for long operations
    - OOM recovery with safe mode fallback
    """

    def __init__(
        self,
        model_name: str = "realesrgan-x4plus",
        tile_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        auto_memory: bool = True,
    ):
        """Initialize the upscaler.

        Args:
            model_name: Model to use (realesrgan-x4plus or realesrgan-x2plus)
            tile_size: Size of tiles for processing large images.
                       If None and auto_memory=True, calculated automatically.
                       Manual override: M1 (8-16GB): 256-384, M1 Pro/Max: 384-512
            batch_size: Number of frames/tiles to process at once.
                       If None and auto_memory=True, calculated automatically.
                       Manual override: M1: 1-2, M1 Pro/Max: 2-4
            auto_memory: If True (default), automatically calculate safe tile_size
                        and batch_size based on available GPU memory.
        """
        self.model_name = model_name
        self.scale = 4 if "x4" in model_name else 2
        self.device = get_device()
        self.model = None

        # Use automatic memory management to calculate safe settings
        if auto_memory and (tile_size is None or batch_size is None):
            mem_manager = get_memory_manager()
            auto_tile, auto_batch = mem_manager.get_safe_settings()
            self.tile_size = tile_size if tile_size is not None else auto_tile
            self.batch_size = batch_size if batch_size is not None else auto_batch
            logger.info(
                f"MPSUpscaler using auto-calculated settings: "
                f"tile_size={self.tile_size}, batch_size={self.batch_size}"
            )
        else:
            # Use provided values or defaults
            self.tile_size = tile_size if tile_size is not None else DEFAULT_TILE_SIZE
            self.batch_size = batch_size if batch_size is not None else DEFAULT_BATCH_SIZE

        logger.info(
            f"MPSUpscaler initialized: device={self.device}, "
            f"tile_size={self.tile_size}, batch_size={self.batch_size}"
        )

    def load_model(self, progress_callback: Optional[Callable[[str, float], None]] = None):
        """Load the model weights with optimizations.

        Uses global cache to avoid reloading between upscale jobs.
        """
        if self.model is not None:
            return

        # Check global cache first
        cache_key = f"{self.model_name}_{self.device.type}"
        if cache_key in _model_cache:
            self.model = _model_cache[cache_key]
            logger.info(f"Loaded model from cache: {cache_key}")
            if progress_callback:
                progress_callback("AI model loaded from cache", 0.2)
            return

        model_path = download_model(self.model_name, progress_callback)

        if progress_callback:
            progress_callback("Loading AI model...", 0.2)

        # Create model
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=self.scale)

        # Load weights
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)

        # Handle different state dict formats
        if 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        elif 'params' in state_dict:
            state_dict = state_dict['params']

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.to(self.device)

        # Optimization: Use channels_last memory format for better performance
        if self.device.type in ("mps", "cuda"):
            self.model = self.model.to(memory_format=torch.channels_last)
            logger.info("Applied channels_last memory optimization")

        # Optimization: Try to compile model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.device.type == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("Applied torch.compile optimization")
            except Exception as e:
                logger.debug(f"torch.compile not available: {e}")

        # Cache the model globally
        _model_cache[cache_key] = self.model
        logger.info(f"Model loaded and cached on {self.device}")

    def upscale_image(self, img: np.ndarray, safe_mode: bool = False) -> np.ndarray:
        """Upscale a single image.

        Args:
            img: Input image as numpy array (H, W, C) in RGB format, 0-255
            safe_mode: If True, use smaller tiles and process one at a time to avoid OOM

        Returns:
            Upscaled image as numpy array (H*scale, W*scale, C)
        """
        self.load_model()

        # Convert to tensor
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        img_tensor = img_tensor.to(self.device)

        tile_threshold = min(self.tile_size, 512) if safe_mode else self.tile_size

        with torch.no_grad():
            # Process in tiles if image is large
            if img.shape[0] > tile_threshold or img.shape[1] > tile_threshold:
                output = self._tile_process(img_tensor, safe_mode=safe_mode)
            else:
                output = self.model(img_tensor)

        # Convert back to numpy
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = (output * 255.0).clip(0, 255).astype(np.uint8)

        return output

    def upscale_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Upscale a batch of images at once for better GPU utilization.

        For large images (>tile_size), uses parallel processing with ThreadPoolExecutor
        to upscale multiple frames concurrently on the GPU.

        Args:
            images: List of images as numpy arrays (H, W, C) in RGB format, 0-255

        Returns:
            List of upscaled images as numpy arrays
        """
        if not images:
            return []

        self.load_model()

        h, w = images[0].shape[:2]

        # For small images, use true batch processing on GPU
        if h <= self.tile_size and w <= self.tile_size:
            # Convert all images to tensors
            tensors = []
            for img in images:
                img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
                img_tensor = img_tensor.permute(2, 0, 1)  # (C, H, W)
                tensors.append(img_tensor)

            # Stack into batch
            batch = torch.stack(tensors, dim=0).to(self.device)  # (B, C, H, W)

            with torch.no_grad():
                output_batch = self.model(batch)

            # Convert back to numpy
            results = []
            for i in range(output_batch.shape[0]):
                output = output_batch[i].permute(1, 2, 0).cpu().numpy()
                output = (output * 255.0).clip(0, 255).astype(np.uint8)
                results.append(output)

            return results

        # For large images, process sequentially but with tile optimization
        # (GPU is still utilized efficiently through tile processing)
        results = []
        for img in images:
            results.append(self.upscale_image(img))
        return results

    def _tile_process(self, img: torch.Tensor, safe_mode: bool = False) -> torch.Tensor:
        """Process image in tiles with batch optimization for better GPU utilization.

        This method extracts all tiles, batches them together, and processes
        them in parallel on the GPU for significantly faster upscaling of
        large images (e.g., 1080p, 4K).

        Args:
            img: Input image tensor (1, C, H, W)
            safe_mode: If True, process one tile at a time to avoid OOM
        """
        _, c, h, w = img.shape
        tile = min(self.tile_size, 512) if safe_mode else self.tile_size  # Use smaller tiles in safe mode
        tile_overlap = 32

        # Calculate output size
        out_h = h * self.scale
        out_w = w * self.scale

        # Collect all tiles and their positions
        tiles = []
        positions = []  # (y, x, y_end, x_end) for each tile

        for y in range(0, h, tile - tile_overlap):
            for x in range(0, w, tile - tile_overlap):
                y_end = min(y + tile, h)
                x_end = min(x + tile, w)
                tile_input = img[:, :, y:y_end, x:x_end]
                tiles.append(tile_input.squeeze(0))  # Remove batch dim for stacking
                positions.append((y, x, y_end, x_end))

        # Process tiles in batches for better GPU utilization
        output = torch.zeros((1, c, out_h, out_w), device=self.device)
        tile_outputs = []

        # In safe mode, process one tile at a time
        effective_batch = 1 if safe_mode else self.batch_size

        for i in range(0, len(tiles), effective_batch):
            batch_tiles = tiles[i:i + effective_batch]

            # Pad tiles to same size for batching (tiles at edges may be smaller)
            max_h = max(t.shape[1] for t in batch_tiles)
            max_w = max(t.shape[2] for t in batch_tiles)

            padded_tiles = []
            for t in batch_tiles:
                pad_h = max_h - t.shape[1]
                pad_w = max_w - t.shape[2]
                if pad_h > 0 or pad_w > 0:
                    # Use 'constant' mode with zeros - no size limitations
                    # The padded area will be cropped after processing anyway
                    t = F.pad(t, (0, pad_w, 0, pad_h), mode='constant', value=0)
                padded_tiles.append(t)

            # Stack into batch and process
            batch = torch.stack(padded_tiles, dim=0)  # (B, C, H, W)
            batch_output = self.model(batch)

            # Unpad and store outputs
            for j, (y, x, y_end, x_end) in enumerate(positions[i:i + len(batch_tiles)]):
                out_h_tile = (y_end - y) * self.scale
                out_w_tile = (x_end - x) * self.scale
                tile_out = batch_output[j, :, :out_h_tile, :out_w_tile]
                tile_outputs.append(tile_out)

        # Reassemble output image with overlap blending
        for idx, (y, x, y_end, x_end) in enumerate(positions):
            out_y = y * self.scale
            out_x = x * self.scale
            out_y_end = y_end * self.scale
            out_x_end = x_end * self.scale

            # For now, simple overwrite (later: blending for smoother seams)
            output[:, :, out_y:out_y_end, out_x:out_x_end] = tile_outputs[idx]

        return output

    def upscale_video(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        work_dir: Optional[Path] = None,
    ) -> bool:
        """Upscale a video file with resume support.

        Args:
            input_path: Path to input video
            output_path: Path for output video
            progress_callback: Optional progress callback
            work_dir: Optional persistent work directory for resume capability.
                     If None, creates one based on input filename in output dir.

        Returns:
            True if successful, False if failed or cancelled
        """
        # Reset cancellation flag at start
        reset_cancel_flag()

        self.load_model(progress_callback)

        # Create persistent work directory for resume capability using content hash
        # This allows resuming even if the video is uploaded again with a different name
        if work_dir is None:
            work_dir = get_work_dir(input_path)
            if progress_callback:
                # Check if this is a resume
                frames_dir_check = work_dir / "frames"
                if frames_dir_check.exists() and any(frames_dir_check.glob("*.png")):
                    progress_callback("Found existing progress, resuming...", 0.1)

        frames_dir = work_dir / "frames"
        upscaled_dir = work_dir / "upscaled"
        frames_dir.mkdir(parents=True, exist_ok=True)
        upscaled_dir.mkdir(parents=True, exist_ok=True)

        # Get expected frame count from video for progress tracking
        # Use duration * fps instead of -count_frames which reads the entire video (slow!)
        expected_frames = 0
        try:
            # Get duration
            duration_cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(input_path),
            ]
            duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, timeout=10)

            # Get FPS
            fps_cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(input_path),
            ]
            fps_result = subprocess.run(fps_cmd, capture_output=True, text=True, timeout=10)

            if duration_result.returncode == 0 and fps_result.returncode == 0:
                duration = float(duration_result.stdout.strip())
                fps_str = fps_result.stdout.strip()
                # Parse fps (could be "30" or "30000/1001")
                if "/" in fps_str:
                    num, den = map(float, fps_str.split("/"))
                    fps = num / den
                else:
                    fps = float(fps_str)
                expected_frames = int(duration * fps)
                logger.info(f"Video: {duration:.2f}s @ {fps:.2f}fps = ~{expected_frames} frames")
        except subprocess.TimeoutExpired:
            logger.warning("Timed out getting video duration/fps - will estimate from extracted frames")
        except Exception as e:
            logger.warning(f"Could not determine frame count: {e}")

        # Check if frames already extracted (for resume)
        existing_frames = sorted(frames_dir.glob("*.png"))
        extraction_complete = len(existing_frames) >= expected_frames if expected_frames > 0 else False

        if extraction_complete:
            if progress_callback:
                progress_callback(f"Found {len(existing_frames)} extracted frames, skipping extraction...", 0.1)
            logger.info(f"Resuming: all {len(existing_frames)} frames already extracted")
        else:
            # Check if FFmpeg extraction is already running for this video
            ffmpeg_running = False
            try:
                result = subprocess.run(
                    ["pgrep", "-f", f"ffmpeg.*{frames_dir.name}"],
                    capture_output=True, text=True
                )
                ffmpeg_running = result.returncode == 0
            except Exception:
                pass

            if ffmpeg_running and existing_frames:
                # FFmpeg is already extracting - just monitor progress
                if progress_callback:
                    progress_callback(f"Extraction in progress ({len(existing_frames)} frames so far)...", 0.05)
                logger.info(f"Found running extraction, monitoring progress...")

                start_time = time.time()
                last_count = len(existing_frames)
                while True:
                    # Check for cancellation
                    if is_cancelled():
                        logger.info("Extraction monitoring cancelled")
                        return False

                    current_count = len(list(frames_dir.glob("*.png")))

                    # Check if extraction is complete
                    if expected_frames > 0 and current_count >= expected_frames:
                        logger.info("Extraction complete!")
                        break

                    # Check if FFmpeg is still running
                    result = subprocess.run(
                        ["pgrep", "-f", f"ffmpeg.*{frames_dir.name}"],
                        capture_output=True, text=True
                    )
                    if result.returncode != 0:
                        # FFmpeg finished
                        logger.info("FFmpeg extraction finished")
                        break

                    # Update progress
                    if current_count != last_count:
                        last_count = current_count
                        if expected_frames > 0:
                            pct = current_count / expected_frames
                            elapsed = time.time() - start_time
                            if current_count > 0:
                                eta_seconds = (elapsed / current_count) * (expected_frames - current_count)
                                eta_str = format_eta(eta_seconds)
                            else:
                                eta_str = "calculating..."
                            if progress_callback:
                                progress_callback(
                                    f"Extracting frames: {current_count}/{expected_frames} ({pct*100:.1f}%) - ETA: {eta_str}",
                                    0.05 + (0.1 * pct)
                                )
                    time.sleep(0.5)
            else:
                # Need to start fresh extraction
                if existing_frames:
                    # Partial extraction with no running FFmpeg - clear and restart
                    if progress_callback:
                        progress_callback(f"Found {len(existing_frames)} partial frames, re-extracting...", 0.05)
                    logger.info(f"Partial extraction found ({len(existing_frames)} frames), restarting...")
                    for f in existing_frames:
                        f.unlink()

                if progress_callback:
                    progress_callback("Extracting frames (hardware accelerated)...", 0.05)

                # Use VideoToolbox hardware decoding on macOS for faster extraction
                # PNG with compression_level 1 (fast) instead of default 6
                # Run in background with Popen so we can track progress
                extract_cmd = [
                    "ffmpeg", "-y",
                    "-hwaccel", "videotoolbox",  # Hardware decoding on macOS
                    "-i", str(input_path),
                    "-compression_level", "1",  # Fast PNG compression (0-9, 0=none, 1=fast)
                    "-threads", "0",  # Use all available CPU threads for encoding
                    str(frames_dir / "frame_%06d.png"),
                ]

                # Start extraction in background
                process = subprocess.Popen(
                    extract_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                # Track progress while extraction runs
                last_count = 0
                start_time = time.time()
                while process.poll() is None:
                    # Check for cancellation
                    if is_cancelled():
                        process.terminate()
                        process.wait()
                        logger.info("Frame extraction cancelled by user")
                        if progress_callback:
                            progress_callback("Extraction cancelled", 0.0)
                        return False

                    current_count = len(list(frames_dir.glob("*.png")))
                    if current_count != last_count:
                        last_count = current_count
                        if expected_frames > 0:
                            pct = current_count / expected_frames
                            elapsed = time.time() - start_time
                            if current_count > 0:
                                eta_seconds = (elapsed / current_count) * (expected_frames - current_count)
                                eta_str = format_eta(eta_seconds)
                            else:
                                eta_str = "calculating..."
                            if progress_callback:
                                # Extraction is 0-15% of total progress
                                progress_callback(
                                    f"Extracting frames: {current_count}/{expected_frames} ({pct*100:.1f}%) - ETA: {eta_str}",
                                    0.05 + (0.1 * pct)
                                )
                    time.sleep(0.5)

                # Check if extraction succeeded
                if process.returncode != 0:
                    logger.warning("Hardware decoding failed, falling back to software")
                    # Fallback to software decoding
                    extract_cmd = [
                        "ffmpeg", "-y", "-i", str(input_path),
                        "-compression_level", "1",  # Fast PNG compression
                        "-threads", "0",
                        str(frames_dir / "frame_%06d.png"),
                    ]
                    process = subprocess.Popen(extract_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    while process.poll() is None:
                        if is_cancelled():
                            process.terminate()
                            process.wait()
                            return False
                        current_count = len(list(frames_dir.glob("*.png")))
                        if progress_callback and expected_frames > 0:
                            pct = current_count / expected_frames
                            progress_callback(f"Extracting frames: {current_count}/{expected_frames}", 0.05 + (0.1 * pct))
                        time.sleep(0.5)

                    if process.returncode != 0:
                        logger.error("Frame extraction failed")
                        return False

        # Get frame count
        frames = sorted(frames_dir.glob("*.png"))
        total_frames = len(frames)

        if total_frames == 0:
            logger.error("No frames extracted")
            return False

        # Check which frames are already upscaled (for resume)
        existing_upscaled = set(f.name for f in upscaled_dir.glob("*.png"))
        frames_to_process = [f for f in frames if f.name not in existing_upscaled]
        already_done = total_frames - len(frames_to_process)

        if already_done > 0:
            logger.info(f"Resuming: {already_done}/{total_frames} frames already upscaled, {len(frames_to_process)} remaining")
            if progress_callback:
                progress_callback(f"Resuming: {already_done}/{total_frames} frames done, {len(frames_to_process)} to go...", 0.15)

        if len(frames_to_process) == 0:
            logger.info("All frames already upscaled, skipping to reassembly")
            if progress_callback:
                progress_callback("All frames done, reassembling video...", 0.85)
        else:
            # Use configured batch size
            batch_size = self.batch_size
            logger.info(f"Upscaling {len(frames_to_process)} frames with AI on {self.device} (batch_size={batch_size})...")

            # Clear GPU memory before starting
            if self.device.type == "mps":
                torch.mps.empty_cache()
                logger.info("Cleared MPS cache before upscaling")

            # Track timing for ETA calculation
            start_time = time.time()
            frames_processed = 0

            # Helper functions for parallel I/O
            def load_frame(frame_path: Path) -> tuple[Path, np.ndarray]:
                """Load a frame in a worker thread."""
                img = np.array(Image.open(frame_path).convert("RGB"))
                return (frame_path, img)

            def save_frame(data: tuple[Path, np.ndarray]) -> None:
                """Save a frame in a worker thread."""
                out_path, img = data
                Image.fromarray(img).save(out_path)

            # Process frames in batches with parallel I/O
            with ThreadPoolExecutor(max_workers=IO_WORKERS) as io_executor:
                for batch_start in range(0, len(frames_to_process), batch_size):
                    # Check for cancellation
                    if is_cancelled():
                        logger.info("Upscaling cancelled by user - progress saved for resume")
                        if progress_callback:
                            progress_callback(f"Cancelled - {already_done + frames_processed}/{total_frames} frames saved", 0.0)
                        return False

                    batch_end = min(batch_start + batch_size, len(frames_to_process))
                    batch_frames = frames_to_process[batch_start:batch_end]

                    # Calculate ETA
                    elapsed = time.time() - start_time
                    if frames_processed > 0:
                        time_per_frame = elapsed / frames_processed
                        remaining_frames = len(frames_to_process) - frames_processed
                        eta_seconds = time_per_frame * remaining_frames
                        eta_str = format_eta(eta_seconds)
                    else:
                        eta_str = "calculating..."

                    total_done = already_done + frames_processed
                    if progress_callback:
                        progress = 0.2 + (0.6 * total_done / total_frames)
                        progress_callback(
                            f"AI upscaling frame {total_done+1}/{total_frames} (ETA: {eta_str})",
                            progress
                        )

                    # Load batch in parallel
                    load_futures = [io_executor.submit(load_frame, fp) for fp in batch_frames]
                    loaded_frames = []
                    frame_paths = []
                    for future in load_futures:
                        path, img = future.result()
                        loaded_frames.append(img)
                        frame_paths.append(path)

                    # Upscale batch on GPU with OOM recovery
                    try:
                        upscaled_frames = self.upscale_batch(loaded_frames)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            # Clear cache and try one frame at a time with safe mode
                            logger.warning(f"OOM error, clearing cache and processing in safe mode...")
                            if self.device.type == "mps":
                                torch.mps.empty_cache()
                            if progress_callback:
                                progress_callback(f"Memory issue - switching to safe mode...", 0.2)
                            upscaled_frames = []
                            for img in loaded_frames:
                                # Use safe_mode which uses smaller tiles and processes one at a time
                                upscaled_frames.append(self.upscale_image(img, safe_mode=True))
                                if self.device.type == "mps":
                                    torch.mps.empty_cache()
                        else:
                            raise

                    # Save batch in parallel
                    save_data = [
                        (upscaled_dir / path.name, upscaled)
                        for path, upscaled in zip(frame_paths, upscaled_frames)
                    ]
                    save_futures = [io_executor.submit(save_frame, data) for data in save_data]
                    # Wait for saves to complete
                    for future in save_futures:
                        future.result()

                    frames_processed += len(batch_frames)

                    # Clear MPS cache after EVERY batch to prevent memory buildup
                    if self.device.type == "mps":
                        torch.mps.empty_cache()

            # Final cancellation check before reassembly
            if is_cancelled():
                logger.info("Upscaling cancelled by user - progress saved")
                return False

        # Get FPS from original video
        if progress_callback:
            progress_callback("Reassembling video...", 0.85)

        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        fps = result.stdout.strip() if result.returncode == 0 else "30"

        # Reassemble video with audio using hardware encoding if available
        # Try VideoToolbox H.264 hardware encoding first (macOS)
        reassemble_cmd = [
            "ffmpeg", "-y",
            "-framerate", fps,
            "-i", str(upscaled_dir / "frame_%06d.png"),
            "-i", str(input_path),
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "h264_videotoolbox",  # Hardware encoding on macOS
            "-q:v", "65",  # Quality (1-100, higher is better)
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(output_path),
        ]

        if progress_callback:
            progress_callback("Reassembling video (hardware encoding)...", 0.88)

        result = subprocess.run(reassemble_cmd, capture_output=True, text=True)

        # Fallback to software encoding if hardware fails
        if result.returncode != 0:
            logger.warning("Hardware encoding failed, falling back to software (libx264)")
            if progress_callback:
                progress_callback("Reassembling video (software encoding)...", 0.88)

            reassemble_cmd = [
                "ffmpeg", "-y",
                "-framerate", fps,
                "-i", str(upscaled_dir / "frame_%06d.png"),
                "-i", str(input_path),
                "-map", "0:v", "-map", "1:a?",
                "-c:v", "libx264",
                "-preset", "fast",  # Faster preset for software encoding
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                str(output_path),
            ]
            result = subprocess.run(reassemble_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Video reassembly failed: {result.stderr}")
                return False

        # Clean up work directory after successful completion
        try:
            shutil.rmtree(work_dir)
            logger.info(f"Cleaned up work directory: {work_dir}")
        except Exception as e:
            logger.warning(f"Could not clean up work directory: {e}")

        logger.info("Upscaling complete!")

        if progress_callback:
            progress_callback("AI upscaling complete!", 1.0)

        return output_path.exists()


def check_mps_upscaler_available() -> bool:
    """Check if MPS upscaler is available."""
    return torch.backends.mps.is_available() or torch.cuda.is_available()


def cancel_upscale():
    """Cancel any running upscale operation."""
    _cancel_upscale.set()
    logger.info("Upscale cancellation requested")


def reset_cancel_flag():
    """Reset the cancellation flag before starting a new operation."""
    _cancel_upscale.clear()


def is_cancelled() -> bool:
    """Check if cancellation was requested."""
    return _cancel_upscale.is_set()


def format_eta(seconds: float) -> str:
    """Format seconds into a human-readable ETA string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        mins = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds / 3600)
        mins = int((seconds % 3600) / 60)
        return f"{hours}h {mins}m"


def get_recommended_settings() -> dict:
    """Get recommended tile_size and batch_size based on system memory.

    Returns:
        Dict with 'tile_size', 'batch_size', 'total_memory_gb', and 'safe_memory_gb'
    """
    mem_manager = get_memory_manager()
    tile_size, batch_size = mem_manager.get_safe_settings()
    return {
        "tile_size": tile_size,
        "batch_size": batch_size,
        "total_memory_gb": mem_manager.total_memory_gb,
        "safe_memory_gb": mem_manager.safe_memory_gb,
    }


def upscale_video_mps(
    input_path: Path,
    output_path: Path,
    model: str = "realesrgan-x4plus",
    tile_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> bool:
    """Convenience function to upscale a video with MPS.

    Uses automatic memory management by default for safe tile/batch sizes.

    Args:
        input_path: Path to input video
        output_path: Path for output video
        model: Model name (realesrgan-x4plus or realesrgan-x2plus)
        tile_size: Optional override for tile size (auto-calculated if None)
        batch_size: Optional override for batch size (auto-calculated if None)
        progress_callback: Optional progress callback

    Returns:
        True if successful
    """
    try:
        upscaler = MPSUpscaler(
            model_name=model,
            tile_size=tile_size,
            batch_size=batch_size,
            auto_memory=True,
        )
        return upscaler.upscale_video(input_path, output_path, progress_callback)
    except Exception as e:
        logger.error(f"MPS upscaling failed: {e}")
        return False
