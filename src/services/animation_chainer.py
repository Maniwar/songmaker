"""Animation chaining service for long scenes.

This module handles scenes that exceed the maximum duration limits of animation APIs
by generating multiple segments and chaining them together using the last frame
of each segment as the starting image for the next.

Supported backends:
- Prompt animation (Wan2.2-TI2V): max 5 seconds per segment
- Veo (Google): max 8 seconds per segment
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Literal, Optional

logger = logging.getLogger(__name__)

# Maximum durations per animation type (seconds)
MAX_DURATION = {
    "prompt": 5.0,  # Wan2.2-TI2V-5B
    "veo": 8.0,     # Google Veo 3.1
}


def extract_last_frame(video_path: Path, output_path: Path) -> Optional[Path]:
    """
    Extract the last frame from a video file.

    Args:
        video_path: Path to the input video
        output_path: Path to save the extracted frame (as PNG)

    Returns:
        Path to the extracted frame, or None if extraction failed
    """
    try:
        # Use sseof to seek from end (more reliable for last frame)
        cmd = [
            "ffmpeg",
            "-y",
            "-sseof", "-0.1",  # Seek 0.1s from end
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",  # High quality
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        if output_path.exists():
            return output_path
        else:
            logger.error(f"Frame extraction produced no output: {output_path}")
            return None

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract last frame: {e.stderr.decode()}")
        return None


def calculate_segments(
    total_duration: float,
    max_segment_duration: float,
    overlap: float = 0.0,
) -> list[tuple[float, float]]:
    """
    Calculate segment boundaries for a long scene.

    Args:
        total_duration: Total scene duration in seconds
        max_segment_duration: Maximum duration per segment
        overlap: Overlap between segments for smoother transitions (default 0)

    Returns:
        List of (start_time, duration) tuples for each segment
    """
    segments = []
    current_time = 0.0
    effective_segment = max_segment_duration - overlap

    while current_time < total_duration:
        remaining = total_duration - current_time
        segment_duration = min(max_segment_duration, remaining + overlap)

        # Ensure we don't create tiny segments at the end
        if remaining < max_segment_duration * 0.5 and segments:
            # Extend the last segment instead
            break

        segments.append((current_time, segment_duration))
        current_time += effective_segment

    # Adjust last segment to cover exact duration
    if segments:
        last_start, _ = segments[-1]
        last_duration = total_duration - last_start
        if last_duration > 0:
            segments[-1] = (last_start, min(last_duration, max_segment_duration))

    return segments


def generate_continuation_prompt(
    base_prompt: str,
    segment_index: int,
    total_segments: int,
) -> str:
    """
    Generate a continuation prompt for chained segments.

    Adds continuity language to motion prompts to ensure smooth
    visual transitions between segments.

    Args:
        base_prompt: Original motion prompt
        segment_index: Current segment index (0-based)
        total_segments: Total number of segments

    Returns:
        Modified prompt with continuity language
    """
    if segment_index == 0:
        # First segment: use original prompt
        return base_prompt

    # Subsequent segments: add continuation language
    continuation_prefixes = [
        "Continuing the motion,",
        "Seamlessly continuing,",
        "Smoothly transitioning,",
        "Maintaining momentum,",
    ]

    prefix = continuation_prefixes[segment_index % len(continuation_prefixes)]

    # Modify verbs to continuous form if not already
    modified_prompt = base_prompt

    # Common verb transformations for continuity
    verb_transforms = {
        "plays": "continues playing",
        "strums": "continues strumming",
        "sings": "continues singing",
        "dances": "continues dancing",
        "moves": "continues moving",
        "walks": "continues walking",
        "runs": "continues running",
        "jumps": "continues jumping",
    }

    for verb, continuous in verb_transforms.items():
        if verb in modified_prompt.lower():
            modified_prompt = modified_prompt.lower().replace(verb, continuous)
            break

    return f"{prefix} {modified_prompt}"


def concatenate_video_segments(
    segment_paths: list[Path],
    output_path: Path,
    crossfade_duration: float = 0.3,
) -> Optional[Path]:
    """
    Concatenate video segments with optional crossfade.

    Args:
        segment_paths: List of video segment paths in order
        output_path: Path for the final concatenated video
        crossfade_duration: Duration of crossfade between segments

    Returns:
        Path to the concatenated video, or None if failed
    """
    if not segment_paths:
        return None

    if len(segment_paths) == 1:
        # Just copy single segment
        import shutil
        shutil.copy(segment_paths[0], output_path)
        return output_path

    try:
        # Build FFmpeg filter for crossfade
        n_clips = len(segment_paths)

        # Get durations
        durations = []
        for path in segment_paths:
            duration = _get_video_duration(path)
            if duration <= 0:
                logger.error(f"Could not get duration for {path}")
                return None
            durations.append(duration)

        # Build inputs
        inputs = []
        for path in segment_paths:
            inputs.extend(["-i", str(path)])

        # Build xfade filter chain
        filter_parts = []
        current_stream = "[0:v]"
        cumulative_offset = durations[0] - crossfade_duration

        for i in range(1, n_clips):
            next_stream = f"[{i}:v]"
            output_stream = f"[v{i}]" if i < n_clips - 1 else "[outv]"

            filter_parts.append(
                f"{current_stream}{next_stream}xfade=transition=fade:"
                f"duration={crossfade_duration}:offset={cumulative_offset:.3f}{output_stream}"
            )
            current_stream = output_stream

            if i < n_clips - 1:
                cumulative_offset += durations[i] - crossfade_duration

        filter_complex = ";".join(filter_parts)

        cmd = [
            "ffmpeg",
            "-y",
        ]
        cmd.extend(inputs)
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            str(output_path),
        ])

        subprocess.run(cmd, check=True, capture_output=True)
        return output_path

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to concatenate segments: {e.stderr.decode()}")
        # Fall back to simple concat
        return _simple_concat(segment_paths, output_path)


def _simple_concat(segment_paths: list[Path], output_path: Path) -> Optional[Path]:
    """Simple concatenation without transitions as fallback."""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for path in segment_paths:
                f.write(f"file '{path}'\n")
            concat_file = f.name

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            str(output_path),
        ]

        subprocess.run(cmd, check=True, capture_output=True)
        Path(concat_file).unlink()
        return output_path

    except subprocess.CalledProcessError as e:
        logger.error(f"Simple concat also failed: {e}")
        return None


def trim_video_to_duration(
    video_path: Path,
    output_path: Path,
    target_duration: float,
) -> Optional[Path]:
    """
    Trim a video to exact target duration.

    Args:
        video_path: Input video path
        output_path: Output video path
        target_duration: Target duration in seconds

    Returns:
        Path to trimmed video, or None if failed
    """
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-t", f"{target_duration:.3f}",
            "-c", "copy",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to trim video: {e.stderr.decode()}")
        return None


def _get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


class PromptAnimationChainer:
    """
    Chain prompt animations for scenes longer than 5 seconds.

    Uses the Wan2.2-TI2V-5B model via PromptAnimator, generating
    multiple 5-second segments and chaining them using the last
    frame of each segment as the starting image for the next.
    """

    def __init__(self):
        self._prompt_animator = None

    def _get_animator(self):
        """Lazy load the prompt animator."""
        if self._prompt_animator is None:
            from src.services.prompt_animator import PromptAnimator
            self._prompt_animator = PromptAnimator()
        return self._prompt_animator

    def animate_scene(
        self,
        image_path: Path,
        prompt: str,
        output_path: Path,
        duration_seconds: float,
        quality_preset: str = "fast",
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Optional[Path]:
        """
        Animate a scene of any duration using chained segments.

        For scenes <= 5s: generates single segment
        For scenes > 5s: generates multiple segments and chains them

        Args:
            image_path: Path to the starting scene image
            prompt: Motion description prompt
            output_path: Path to save the final video
            duration_seconds: Target total duration
            quality_preset: Quality preset for PromptAnimator
            progress_callback: Optional progress callback

        Returns:
            Path to the generated video, or None if failed
        """
        animator = self._get_animator()
        max_duration = MAX_DURATION["prompt"]

        # If scene is short enough, generate directly
        if duration_seconds <= max_duration:
            return animator.animate_scene(
                image_path=image_path,
                prompt=prompt,
                output_path=output_path,
                duration_seconds=duration_seconds,
                quality_preset=quality_preset,
                progress_callback=progress_callback,
            )

        # Calculate segments for long scene
        segments = calculate_segments(duration_seconds, max_duration)
        n_segments = len(segments)

        if progress_callback:
            progress_callback(
                f"Scene requires {n_segments} segments ({duration_seconds:.1f}s total)",
                0.05
            )

        # Generate segments
        segment_paths = []
        current_image = image_path

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            for i, (start_time, segment_duration) in enumerate(segments):
                segment_prompt = generate_continuation_prompt(prompt, i, n_segments)
                segment_output = temp_dir / f"segment_{i:03d}.mp4"
                frame_output = temp_dir / f"frame_{i:03d}.png"

                if progress_callback:
                    base_progress = 0.1 + (0.7 * i / n_segments)
                    progress_callback(
                        f"Generating segment {i + 1}/{n_segments}...",
                        base_progress
                    )

                # Generate this segment
                def segment_progress(msg: str, prog: float):
                    if progress_callback:
                        overall = 0.1 + (0.7 * (i + prog) / n_segments)
                        progress_callback(f"Segment {i + 1}: {msg}", overall)

                result = animator.animate_scene(
                    image_path=current_image,
                    prompt=segment_prompt,
                    output_path=segment_output,
                    duration_seconds=segment_duration,
                    quality_preset=quality_preset,
                    progress_callback=segment_progress,
                )

                if result is None:
                    logger.error(f"Failed to generate segment {i + 1}")
                    # Try to salvage with what we have
                    if segment_paths:
                        break
                    return None

                segment_paths.append(result)

                # Extract last frame for next segment (unless this is the last one)
                if i < n_segments - 1:
                    frame_result = extract_last_frame(result, frame_output)
                    if frame_result is None:
                        logger.warning(f"Could not extract frame from segment {i + 1}, using original")
                    else:
                        current_image = frame_result

            if not segment_paths:
                return None

            if progress_callback:
                progress_callback("Concatenating segments...", 0.85)

            # Concatenate all segments
            concat_output = temp_dir / "concatenated.mp4"
            concat_result = concatenate_video_segments(
                segment_paths,
                concat_output,
                crossfade_duration=0.2,  # Short crossfade for seamless transitions
            )

            if concat_result is None:
                logger.error("Failed to concatenate segments")
                return None

            if progress_callback:
                progress_callback("Trimming to exact duration...", 0.95)

            # Trim to exact target duration
            trim_result = trim_video_to_duration(
                concat_result,
                output_path,
                duration_seconds,
            )

            if trim_result is None:
                # Fall back to untrimmed
                import shutil
                shutil.copy(concat_result, output_path)
                trim_result = output_path

            if progress_callback:
                progress_callback("Animation complete!", 1.0)

            return trim_result


class VeoAnimationChainer:
    """
    Chain Veo animations for scenes longer than 8 seconds.

    Uses Google Veo 3.1 via VeoAnimator, generating multiple
    8-second segments and chaining them using the last frame
    of each segment as the starting image for the next.
    """

    def __init__(self, config=None):
        self._config = config
        self._veo_animator = None

    def _get_animator(self):
        """Lazy load the Veo animator."""
        if self._veo_animator is None:
            from src.services.veo_animator import VeoAnimator
            self._veo_animator = VeoAnimator(config=self._config)
        return self._veo_animator

    def animate_scene(
        self,
        image_path: Path,
        prompt: str,
        output_path: Path,
        duration_seconds: float,
        resolution: str = "720p",
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Optional[Path]:
        """
        Animate a scene of any duration using chained Veo segments.

        For scenes <= 8s: generates single segment
        For scenes > 8s: generates multiple segments and chains them

        Args:
            image_path: Path to the starting scene image
            prompt: Motion description prompt
            output_path: Path to save the final video
            duration_seconds: Target total duration
            resolution: Video resolution (720p or 1080p)
            progress_callback: Optional progress callback

        Returns:
            Path to the generated video, or None if failed
        """
        animator = self._get_animator()
        max_duration = MAX_DURATION["veo"]

        # If scene is short enough, generate directly
        if duration_seconds <= max_duration:
            return animator.animate_scene(
                image_path=image_path,
                prompt=prompt,
                output_path=output_path,
                duration_seconds=duration_seconds,
                resolution=resolution,
                progress_callback=progress_callback,
            )

        # Calculate segments for long scene
        segments = calculate_segments(duration_seconds, max_duration)
        n_segments = len(segments)

        if progress_callback:
            progress_callback(
                f"Scene requires {n_segments} Veo segments ({duration_seconds:.1f}s total)",
                0.05
            )

        # Generate segments
        segment_paths = []
        current_image = image_path

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            for i, (start_time, segment_duration) in enumerate(segments):
                segment_prompt = generate_continuation_prompt(prompt, i, n_segments)
                segment_output = temp_dir / f"veo_segment_{i:03d}.mp4"
                frame_output = temp_dir / f"veo_frame_{i:03d}.png"

                if progress_callback:
                    base_progress = 0.1 + (0.7 * i / n_segments)
                    progress_callback(
                        f"Generating Veo segment {i + 1}/{n_segments}...",
                        base_progress
                    )

                # Generate this segment
                def segment_progress(msg: str, prog: float):
                    if progress_callback:
                        overall = 0.1 + (0.7 * (i + prog) / n_segments)
                        progress_callback(f"Veo segment {i + 1}: {msg}", overall)

                result = animator.animate_scene(
                    image_path=current_image,
                    prompt=segment_prompt,
                    output_path=segment_output,
                    duration_seconds=segment_duration,
                    resolution=resolution,
                    progress_callback=segment_progress,
                )

                if result is None:
                    logger.error(f"Failed to generate Veo segment {i + 1}")
                    if segment_paths:
                        break
                    return None

                segment_paths.append(result)

                # Extract last frame for next segment
                if i < n_segments - 1:
                    frame_result = extract_last_frame(result, frame_output)
                    if frame_result is None:
                        logger.warning(f"Could not extract frame from Veo segment {i + 1}")
                    else:
                        current_image = frame_result

            if not segment_paths:
                return None

            if progress_callback:
                progress_callback("Concatenating Veo segments...", 0.85)

            # Concatenate all segments
            concat_output = temp_dir / "veo_concatenated.mp4"
            concat_result = concatenate_video_segments(
                segment_paths,
                concat_output,
                crossfade_duration=0.3,  # Slightly longer crossfade for Veo
            )

            if concat_result is None:
                logger.error("Failed to concatenate Veo segments")
                return None

            if progress_callback:
                progress_callback("Trimming to exact duration...", 0.95)

            # Trim to exact target duration
            trim_result = trim_video_to_duration(
                concat_result,
                output_path,
                duration_seconds,
            )

            if trim_result is None:
                import shutil
                shutil.copy(concat_result, output_path)
                trim_result = output_path

            if progress_callback:
                progress_callback("Veo animation complete!", 1.0)

            return trim_result
