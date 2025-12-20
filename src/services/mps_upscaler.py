"""MPS-accelerated AI video upscaling for Apple Silicon Macs.

Uses Real-ESRGAN architecture with PyTorch MPS backend for GPU acceleration.
This module provides a standalone implementation that doesn't depend on
the problematic basicsr/realesrgan packages.

Performance optimizations:
- Batch processing: Process multiple frames at once (configurable batch size)
- Parallel I/O: Load/save images in background threads while GPU processes
- Memory management: Clear MPS cache periodically to prevent buildup
"""

import logging
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

# Default batch size (adjust based on GPU memory - M1 Max can handle 4-8)
DEFAULT_BATCH_SIZE = 4
# Number of I/O worker threads
IO_WORKERS = 4

# Model download URLs
MODEL_URLS = {
    "realesrgan-x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "realesrgan-x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
}

# Cache directory for models
MODELS_DIR = Path.home() / ".cache" / "songmaker" / "models"


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
    - Batch processing for better GPU utilization
    - Parallel I/O for loading/saving frames
    - Tile-based processing for large images
    - Cancellation support for long operations
    """

    def __init__(
        self,
        model_name: str = "realesrgan-x4plus",
        tile_size: int = 512,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """Initialize the upscaler.

        Args:
            model_name: Model to use (realesrgan-x4plus or realesrgan-x2plus)
            tile_size: Size of tiles for processing large images (reduces memory)
            batch_size: Number of frames to process at once (adjust for GPU memory)
                       - M1: 2-4
                       - M1 Pro/Max: 4-8
                       - M2/M3 Ultra: 8-16
        """
        self.model_name = model_name
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.scale = 4 if "x4" in model_name else 2
        self.device = get_device()
        self.model = None

        logger.info(f"MPSUpscaler initialized with device: {self.device}, batch_size: {self.batch_size}")

    def load_model(self, progress_callback: Optional[Callable[[str, float], None]] = None):
        """Load the model weights."""
        if self.model is not None:
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

        logger.info(f"Model loaded on {self.device}")

    def upscale_image(self, img: np.ndarray) -> np.ndarray:
        """Upscale a single image.

        Args:
            img: Input image as numpy array (H, W, C) in RGB format, 0-255

        Returns:
            Upscaled image as numpy array (H*scale, W*scale, C)
        """
        self.load_model()

        # Convert to tensor
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            # Process in tiles if image is large
            if img.shape[0] > self.tile_size or img.shape[1] > self.tile_size:
                output = self._tile_process(img_tensor)
            else:
                output = self.model(img_tensor)

        # Convert back to numpy
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = (output * 255.0).clip(0, 255).astype(np.uint8)

        return output

    def upscale_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Upscale a batch of images at once for better GPU utilization.

        Args:
            images: List of images as numpy arrays (H, W, C) in RGB format, 0-255

        Returns:
            List of upscaled images as numpy arrays
        """
        if not images:
            return []

        self.load_model()

        # Check if images are too large for batching (use tile processing instead)
        h, w = images[0].shape[:2]
        if h > self.tile_size or w > self.tile_size:
            # Fall back to sequential processing for large images
            return [self.upscale_image(img) for img in images]

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

    def _tile_process(self, img: torch.Tensor) -> torch.Tensor:
        """Process image in tiles to handle large images."""
        _, c, h, w = img.shape
        tile = self.tile_size
        tile_overlap = 32

        # Calculate output size
        out_h = h * self.scale
        out_w = w * self.scale
        out_tile = tile * self.scale
        out_overlap = tile_overlap * self.scale

        output = torch.zeros((1, c, out_h, out_w), device=self.device)

        # Process tiles
        for y in range(0, h, tile - tile_overlap):
            for x in range(0, w, tile - tile_overlap):
                # Extract tile
                y_end = min(y + tile, h)
                x_end = min(x + tile, w)
                tile_input = img[:, :, y:y_end, x:x_end]

                # Upscale tile
                tile_output = self.model(tile_input)

                # Place in output
                out_y = y * self.scale
                out_x = x * self.scale
                out_y_end = y_end * self.scale
                out_x_end = x_end * self.scale

                # Blend overlapping regions
                output[:, :, out_y:out_y_end, out_x:out_x_end] = tile_output

        return output

    def upscale_video(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> bool:
        """Upscale a video file.

        Args:
            input_path: Path to input video
            output_path: Path for output video
            progress_callback: Optional progress callback

        Returns:
            True if successful, False if failed or cancelled
        """
        # Reset cancellation flag at start
        reset_cancel_flag()

        self.load_model(progress_callback)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            frames_dir = tmpdir_path / "frames"
            upscaled_dir = tmpdir_path / "upscaled"
            frames_dir.mkdir()
            upscaled_dir.mkdir()

            # Extract frames
            if progress_callback:
                progress_callback("Extracting frames...", 0.1)

            extract_cmd = [
                "ffmpeg", "-y", "-i", str(input_path),
                "-qscale:v", "2",
                str(frames_dir / "frame_%06d.png"),
            ]
            result = subprocess.run(extract_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Frame extraction failed: {result.stderr}")
                return False

            # Get frame count
            frames = sorted(frames_dir.glob("*.png"))
            total_frames = len(frames)

            if total_frames == 0:
                logger.error("No frames extracted")
                return False

            # Use configured batch size
            batch_size = self.batch_size
            logger.info(f"Upscaling {total_frames} frames with AI on {self.device} (batch_size={batch_size})...")

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
                output_path, img = data
                Image.fromarray(img).save(output_path)

            # Process frames in batches with parallel I/O
            with ThreadPoolExecutor(max_workers=IO_WORKERS) as io_executor:
                for batch_start in range(0, total_frames, batch_size):
                    # Check for cancellation
                    if is_cancelled():
                        logger.info("Upscaling cancelled by user")
                        if progress_callback:
                            progress_callback("Upscaling cancelled", 0.0)
                        return False

                    batch_end = min(batch_start + batch_size, total_frames)
                    batch_frames = frames[batch_start:batch_end]

                    # Calculate ETA
                    elapsed = time.time() - start_time
                    if frames_processed > 0:
                        time_per_frame = elapsed / frames_processed
                        remaining_frames = total_frames - frames_processed
                        eta_seconds = time_per_frame * remaining_frames
                        eta_str = format_eta(eta_seconds)
                    else:
                        eta_str = "calculating..."

                    if progress_callback:
                        progress = 0.2 + (0.6 * frames_processed / total_frames)
                        progress_callback(
                            f"AI upscaling frames {batch_start+1}-{batch_end}/{total_frames} (ETA: {eta_str})",
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

                    # Upscale batch on GPU
                    upscaled_frames = self.upscale_batch(loaded_frames)

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

                    # Clear MPS cache periodically to prevent memory buildup
                    if self.device.type == "mps" and frames_processed % 50 == 0:
                        torch.mps.empty_cache()

            # Final cancellation check before reassembly
            if is_cancelled():
                logger.info("Upscaling cancelled by user")
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

            # Reassemble video with audio
            reassemble_cmd = [
                "ffmpeg", "-y",
                "-framerate", fps,
                "-i", str(upscaled_dir / "frame_%06d.png"),
                "-i", str(input_path),
                "-map", "0:v", "-map", "1:a?",
                "-c:v", "libx264",
                "-preset", "medium",
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

            # Log total time
            total_time = time.time() - start_time
            logger.info(f"Upscaling complete in {format_eta(total_time)}")

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


def upscale_video_mps(
    input_path: Path,
    output_path: Path,
    model: str = "realesrgan-x4plus",
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> bool:
    """Convenience function to upscale a video with MPS.

    Args:
        input_path: Path to input video
        output_path: Path for output video
        model: Model name (realesrgan-x4plus or realesrgan-x2plus)
        progress_callback: Optional progress callback

    Returns:
        True if successful
    """
    try:
        upscaler = MPSUpscaler(model_name=model)
        return upscaler.upscale_video(input_path, output_path, progress_callback)
    except Exception as e:
        logger.error(f"MPS upscaling failed: {e}")
        return False
