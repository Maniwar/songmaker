"""Core ML-accelerated video upscaling using Apple's Neural Engine.

This module provides fast AI upscaling on Apple Silicon Macs by:
1. Converting Real-ESRGAN PyTorch models to Core ML format
2. Using the Neural Engine for inference (much faster than MPS GPU)

Performance comparison:
- CPU (PyTorch): ~500 hours for 7 min video
- MPS (GPU via PyTorch): ~50 hours
- Neural Engine (Core ML): ~30 minutes (target)

Based on: https://medium.com/@ronregev/optimized-ai-image-video-upscaling-on-macs-with-apple-silicon
"""

import hashlib
import logging
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Cache directory for Core ML models
COREML_MODELS_DIR = Path.home() / ".cache" / "songmaker" / "coreml_models"

# Work directory for resume capability
WORK_DIR_BASE = Path.home() / ".cache" / "songmaker" / "upscale_work"


def get_video_content_hash(video_path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Generate a hash based on video file content for resume detection."""
    hasher = hashlib.md5()
    file_size = video_path.stat().st_size
    hasher.update(str(file_size).encode())

    with open(video_path, 'rb') as f:
        # Hash first chunk
        first_chunk = f.read(chunk_size)
        hasher.update(first_chunk)

        # Hash last chunk if file is large enough
        if file_size > chunk_size * 2:
            f.seek(-chunk_size, 2)
            last_chunk = f.read(chunk_size)
            hasher.update(last_chunk)

    return hasher.hexdigest()[:8]


def find_existing_work_dir(video_path: Path) -> Optional[Path]:
    """Find an existing work directory for the video if one exists."""
    content_hash = get_video_content_hash(video_path)
    WORK_DIR_BASE.mkdir(parents=True, exist_ok=True)

    # Look for work directories with matching hash (prefer coreml_ dirs)
    for prefix in ["coreml_", "upscale_"]:
        for work_dir in sorted(WORK_DIR_BASE.glob(f"{prefix}{content_hash}_*"), reverse=True):
            if work_dir.is_dir():
                frames_dir = work_dir / "frames"
                if frames_dir.exists() and any(frames_dir.glob("*.png")):
                    logger.info(f"Found existing work directory: {work_dir}")
                    return work_dir

    return None


def get_work_dir(video_path: Path) -> Path:
    """Get or create a work directory for the video based on content hash."""
    content_hash = get_video_content_hash(video_path)
    timestamp = int(time.time())
    work_dir = WORK_DIR_BASE / f"coreml_{content_hash}_{timestamp}"
    work_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created new work directory: {work_dir}")
    return work_dir

# Global cancellation flag
_cancel_upscale = threading.Event()


def cancel_upscale():
    """Cancel any running upscale operation."""
    _cancel_upscale.set()


def reset_cancel_flag():
    """Reset the cancellation flag."""
    _cancel_upscale.clear()


def is_cancelled() -> bool:
    """Check if cancellation was requested."""
    return _cancel_upscale.is_set()


def check_coreml_available() -> bool:
    """Check if Core ML is available (macOS only)."""
    try:
        import coremltools
        import platform
        return platform.system() == "Darwin"
    except ImportError:
        return False


def get_coreml_model_path(model_name: str = "realesr-general-x4v3") -> Path:
    """Get the path to the Core ML model, downloading/converting if needed."""
    COREML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlmodel_path = COREML_MODELS_DIR / f"{model_name}.mlpackage"

    if mlmodel_path.exists():
        logger.info(f"Found cached Core ML model: {mlmodel_path}")
        return mlmodel_path

    # Need to convert from PyTorch
    logger.info(f"Core ML model not found, will convert from PyTorch...")
    return mlmodel_path


def download_realesrgan_model(model_name: str = "realesr-general-x4v3") -> Path:
    """Download Real-ESRGAN PyTorch model if not cached."""
    import urllib.request

    models_dir = Path.home() / ".cache" / "songmaker" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_urls = {
        "realesr-general-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "realesr-general-wdn-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
        "realesrgan-x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    }

    pth_path = models_dir / f"{model_name}.pth"

    if pth_path.exists():
        logger.info(f"Found cached PyTorch model: {pth_path}")
        return pth_path

    url = model_urls.get(model_name)
    if not url:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_urls.keys())}")

    logger.info(f"Downloading {model_name} from {url}...")
    urllib.request.urlretrieve(url, pth_path)
    logger.info(f"Downloaded to {pth_path}")

    return pth_path


def convert_to_coreml(
    pth_path: Path,
    output_path: Path,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Path:
    """Convert a Real-ESRGAN PyTorch model to Core ML format.

    Uses MPS GPU for conversion (much faster than CPU).
    """
    import torch
    import coremltools as ct
    from spandrel import MAIN_REGISTRY, ModelLoader, ImageModelDescriptor
    from spandrel_extra_arches import EXTRA_REGISTRY

    if progress_callback:
        progress_callback("Loading PyTorch model for conversion...", 0.1)

    # Add extra architectures
    MAIN_REGISTRY.add(*EXTRA_REGISTRY)

    # Load the model
    model = ModelLoader().load_from_file(str(pth_path))
    assert isinstance(model, ImageModelDescriptor), "Expected image model"

    # Use MPS for conversion if available
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS GPU for model conversion")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU for model conversion (slower)")

    if progress_callback:
        progress_callback("Converting to Core ML format...", 0.3)

    # Wrapper that clamps output to valid range
    class CoreMLModel(torch.nn.Module):
        def __init__(self, m):
            super(CoreMLModel, self).__init__()
            self.m = m

        def forward(self, image):
            pred = self.m(image)
            output = torch.clamp(pred * 255, min=0, max=255)
            return output

    with torch.no_grad():
        coreml_model = CoreMLModel(model.model.to(device)).eval()

        # Trace with sample input (512x512 is good balance)
        example_input = torch.randn(1, 3, 512, 512).to(device)

        if progress_callback:
            progress_callback("Tracing model (this may take a minute)...", 0.5)

        traced = torch.jit.trace(coreml_model, example_input)

        if progress_callback:
            progress_callback("Converting to Core ML (Neural Engine optimized)...", 0.7)

        # Convert to Core ML with Neural Engine optimization
        mlmodel = ct.convert(
            traced,
            inputs=[ct.ImageType(shape=example_input.shape, scale=1/255)],
            outputs=[ct.ImageType(name="output")],
            convert_to="mlprogram",  # Use ML Program for Neural Engine
            compute_precision=ct.precision.FLOAT16,  # FP16 for speed
            minimum_deployment_target=ct.target.macOS13,
        )

        if progress_callback:
            progress_callback("Saving Core ML model...", 0.9)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        mlmodel.save(str(output_path))

        logger.info(f"Saved Core ML model to {output_path}")

    if progress_callback:
        progress_callback("Model conversion complete!", 1.0)

    return output_path


class CoreMLUpscaler:
    """Video upscaler using Core ML and Apple's Neural Engine.

    This is MUCH faster than MPS/PyTorch because it uses the dedicated
    Neural Engine hardware on Apple Silicon.
    """

    def __init__(self, model_name: str = "realesr-general-x4v3"):
        """Initialize the Core ML upscaler.

        Args:
            model_name: Model to use. Options:
                - realesr-general-x4v3: Fast, general-purpose (recommended)
                - realesr-general-wdn-x4v3: Better denoising
                - realesrgan-x4plus: Higher quality, slower
        """
        self.model_name = model_name
        self.scale = 4
        self.model = None
        self.mlmodel_path = get_coreml_model_path(model_name)

    def ensure_model(
        self,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        """Ensure the Core ML model is available, converting if needed."""
        if self.model is not None:
            return

        if not self.mlmodel_path.exists():
            # Download PyTorch model and convert
            if progress_callback:
                progress_callback("Downloading Real-ESRGAN model...", 0.05)

            pth_path = download_realesrgan_model(self.model_name)

            if progress_callback:
                progress_callback("Converting to Core ML (one-time setup)...", 0.1)

            convert_to_coreml(pth_path, self.mlmodel_path, progress_callback)

        # Load the Core ML model
        if progress_callback:
            progress_callback("Loading Core ML model...", 0.15)

        import coremltools as ct
        self.model = ct.models.MLModel(str(self.mlmodel_path))
        logger.info(f"Loaded Core ML model from {self.mlmodel_path}")

    def upscale_tile(self, tile: np.ndarray) -> np.ndarray:
        """Upscale a single 512x512 tile using Neural Engine.

        Args:
            tile: Input tile as numpy array (512, 512, 3) in RGB format, 0-255

        Returns:
            Upscaled tile as numpy array (2048, 2048, 3)
        """
        # Convert to PIL Image (Core ML expects this)
        pil_img = Image.fromarray(tile.astype(np.uint8))

        # Run inference
        result = self.model.predict({"image": pil_img})

        # Get output image
        output_img = result["output"]

        # Convert back to numpy, ensuring RGB (strip alpha if present)
        if isinstance(output_img, Image.Image):
            # Convert to RGB to strip any alpha channel
            if output_img.mode == 'RGBA':
                output_img = output_img.convert('RGB')
            return np.array(output_img)
        else:
            # Handle numpy array output - strip alpha if present
            arr = np.array(output_img)
            if arr.ndim == 3 and arr.shape[2] == 4:
                return arr[:, :, :3]  # Take only RGB channels
            return arr

    def upscale_image(self, img: np.ndarray) -> np.ndarray:
        """Upscale a full image using tile-based processing.

        The Core ML model only accepts 512x512 input, so we split the image
        into tiles, upscale each, and stitch them back together.

        Args:
            img: Input image as numpy array (H, W, C) in RGB format, 0-255

        Returns:
            Upscaled image as numpy array (H*4, W*4, C)
        """
        self.ensure_model()

        h, w, c = img.shape
        tile_size = 512
        scale = self.scale

        # Pad image to be divisible by tile_size
        pad_h = (tile_size - h % tile_size) % tile_size
        pad_w = (tile_size - w % tile_size) % tile_size

        if pad_h > 0 or pad_w > 0:
            padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            padded = img

        padded_h, padded_w = padded.shape[:2]

        # Calculate output size
        out_h = padded_h * scale
        out_w = padded_w * scale

        # Create output array
        output = np.zeros((out_h, out_w, c), dtype=np.uint8)

        # Process tiles
        tiles_y = padded_h // tile_size
        tiles_x = padded_w // tile_size

        for ty in range(tiles_y):
            for tx in range(tiles_x):
                # Extract tile
                y1 = ty * tile_size
                x1 = tx * tile_size
                tile = padded[y1:y1+tile_size, x1:x1+tile_size]

                # Upscale tile
                upscaled_tile = self.upscale_tile(tile)

                # Place in output
                out_y1 = ty * tile_size * scale
                out_x1 = tx * tile_size * scale
                output[out_y1:out_y1+tile_size*scale, out_x1:out_x1+tile_size*scale] = upscaled_tile

        # Crop to original output size
        final_h = h * scale
        final_w = w * scale
        output = output[:final_h, :final_w]

        return output

    def upscale_video(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> bool:
        """Upscale a video using Core ML Neural Engine.

        Args:
            input_path: Path to input video
            output_path: Path for output video
            progress_callback: Optional progress callback

        Returns:
            True if successful
        """
        reset_cancel_flag()

        self.ensure_model(progress_callback)

        # Check for existing work directory (resume capability)
        existing_work_dir = find_existing_work_dir(input_path)

        if existing_work_dir:
            work_dir = existing_work_dir
            frames_dir = work_dir / "frames"
            upscaled_dir = work_dir / "upscaled"
            upscaled_dir.mkdir(parents=True, exist_ok=True)

            frames = sorted(frames_dir.glob("*.png"))
            total_frames = len(frames)

            if progress_callback:
                progress_callback(f"Resuming: Found {total_frames} existing frames", 0.12)
            logger.info(f"Resuming from existing work dir with {total_frames} frames")
        else:
            # Create new work directory
            work_dir = get_work_dir(input_path)
            frames_dir = work_dir / "frames"
            upscaled_dir = work_dir / "upscaled"
            frames_dir.mkdir(parents=True, exist_ok=True)
            upscaled_dir.mkdir(parents=True, exist_ok=True)

            # Extract frames
            if progress_callback:
                progress_callback("Extracting frames (this may take a minute)...", 0.1)

            # Get frame count first for progress
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-count_packets", "-show_entries", "stream=nb_read_packets",
                "-of", "csv=p=0",
                str(input_path),
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            expected_frames = int(probe_result.stdout.strip()) if probe_result.returncode == 0 else 0

            # Extract frames with progress
            extract_cmd = [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-q:v", "1",
                str(frames_dir / "frame_%06d.png"),
            ]

            logger.info(f"Running: {' '.join(extract_cmd)}")

            # Run ffmpeg and monitor progress
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
                        extract_progress = 0.1 + (0.05 * current_frames / expected_frames)
                        progress_callback(
                            f"Extracting frames: {current_frames}/{expected_frames}",
                            min(0.15, extract_progress)
                        )
                    time.sleep(0.5)

            stdout, stderr = process.communicate()

            if process.returncode != 0:
                logger.error(f"FFmpeg frame extraction failed: {stderr}")
                if progress_callback:
                    progress_callback(f"Frame extraction failed: {stderr[:100]}", 0.0)
                return False

            # Get frame list
            frames = sorted(frames_dir.glob("*.png"))
            total_frames = len(frames)

            if total_frames == 0:
                logger.error(f"No frames extracted. FFmpeg stderr: {stderr}")
                if progress_callback:
                    progress_callback("No frames found after extraction", 0.0)
                return False

            logger.info(f"Extracted {total_frames} frames to {frames_dir}")

        # Count already upscaled frames for resume
        existing_upscaled = set(f.name for f in upscaled_dir.glob("*.png"))
        frames_to_process = [(i, f) for i, f in enumerate(frames) if f.name not in existing_upscaled]
        already_done = len(existing_upscaled)

        if already_done > 0:
            logger.info(f"Resuming: {already_done} frames already upscaled, {len(frames_to_process)} remaining")
            if progress_callback:
                progress_callback(f"Resuming: {already_done}/{total_frames} already done", 0.15)

        # Upscale remaining frames
        start_time = time.time()
        processed_count = 0

        for i, frame_path in frames_to_process:
            if is_cancelled():
                logger.info("Upscaling cancelled")
                return False

            # Load frame
            img = np.array(Image.open(frame_path).convert("RGB"))

            # Upscale
            upscaled = self.upscale_image(img)

            # Save
            output_frame = upscaled_dir / frame_path.name
            Image.fromarray(upscaled).save(output_frame)

            processed_count += 1

            # Progress
            if progress_callback:
                elapsed = time.time() - start_time
                if processed_count > 0:
                    time_per_frame = elapsed / processed_count
                    remaining = time_per_frame * (len(frames_to_process) - processed_count)
                    eta = format_eta(remaining)
                else:
                    eta = "calculating..."

                total_done = already_done + processed_count
                progress = 0.15 + (0.7 * total_done / total_frames)
                progress_callback(
                    f"Upscaling frame {total_done}/{total_frames} (ETA: {eta})",
                    progress
                )

        # Get FPS from original
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        fps = result.stdout.strip() if result.returncode == 0 else "30"

        # Reassemble video
        if progress_callback:
            progress_callback("Reassembling video...", 0.9)

        reassemble_cmd = [
            "ffmpeg", "-y",
            "-framerate", fps,
            "-i", str(upscaled_dir / "frame_%06d.png"),
            "-i", str(input_path),
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "h264_videotoolbox",
            "-q:v", "65",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(output_path),
        ]

        result = subprocess.run(reassemble_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            # Fallback to software encoding
            reassemble_cmd[10] = "libx264"
            reassemble_cmd[11:13] = ["-preset", "fast", "-crf", "18"]
            subprocess.run(reassemble_cmd, capture_output=True, text=True)

        # Cleanup work directory on success
        if output_path.exists():
            shutil.rmtree(work_dir)

        if progress_callback:
            progress_callback("Upscaling complete!", 1.0)

        return output_path.exists()


def format_eta(seconds: float) -> str:
    """Format seconds into human-readable ETA."""
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


def upscale_video_coreml(
    input_path: Path,
    output_path: Path,
    model: str = "realesr-general-x4v3",
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> bool:
    """Convenience function to upscale a video with Core ML.

    Args:
        input_path: Path to input video
        output_path: Path for output video
        model: Model name
        progress_callback: Optional progress callback

    Returns:
        True if successful
    """
    try:
        upscaler = CoreMLUpscaler(model_name=model)
        return upscaler.upscale_video(input_path, output_path, progress_callback)
    except Exception as e:
        logger.error(f"Core ML upscaling failed: {e}")
        return False
