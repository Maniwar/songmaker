"""Video generation service using FFmpeg with Ken Burns effects."""

import logging
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Literal, Optional

# Extension mode for short animated clips
# - "all": Use Ken Burns on all scenes that need extension (current behavior)
# - "end_only": Pack animations back-to-back, Ken Burns at the end to fill remaining time
# - "none": No Ken Burns on any animated scene (trim to animation duration)
ExtensionMode = Literal["all", "end_only", "none"]

logger = logging.getLogger(__name__)

from src.config import Config, config as default_config
from src.models.schemas import Scene, KenBurnsEffect, AnimationType


def _escape_ffmpeg_path(path: Path) -> str:
    r"""
    Escape a path for use in FFmpeg filter arguments.

    FFmpeg filter syntax requires escaping of special characters:
    - Backslashes must be escaped as \\
    - Colons must be escaped as \:
    - Single quotes must be escaped as \'
    """
    path_str = str(path)
    # Escape backslashes first
    path_str = path_str.replace("\\", "\\\\")
    # Escape colons
    path_str = path_str.replace(":", "\\:")
    # Escape single quotes
    path_str = path_str.replace("'", "\\'")
    return path_str


class VideoGenerator:
    """Generate videos with Ken Burns effects using FFmpeg."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config
        self._hw_encoder = self._detect_hw_encoder()

    def _detect_hw_encoder(self) -> Optional[str]:
        """Detect available hardware encoder."""
        system = platform.system()

        # macOS - VideoToolbox
        if system == "Darwin":
            try:
                result = subprocess.run(
                    ["ffmpeg", "-encoders"],
                    capture_output=True,
                    text=True,
                )
                if "h264_videotoolbox" in result.stdout:
                    return "h264_videotoolbox"
            except Exception:
                pass

        # Linux/Windows - NVIDIA NVENC
        try:
            result = subprocess.run(
                ["ffmpeg", "-encoders"],
                capture_output=True,
                text=True,
            )
            if "h264_nvenc" in result.stdout:
                return "h264_nvenc"
        except Exception:
            pass

        return None

    def _get_encoder_args(self, quality: str = "medium") -> list[str]:
        """
        Get encoder arguments based on available hardware and desired quality.

        Args:
            quality: "fast" for intermediate files, "medium" for final output
        """
        if self._hw_encoder == "h264_videotoolbox":
            # macOS VideoToolbox - much faster than software encoding
            if quality == "fast":
                return ["-c:v", "h264_videotoolbox", "-q:v", "65"]
            else:
                return ["-c:v", "h264_videotoolbox", "-q:v", "50"]

        elif self._hw_encoder == "h264_nvenc":
            # NVIDIA NVENC
            if quality == "fast":
                return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "28"]
            else:
                return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "23"]

        else:
            # Software encoding with libx264
            if quality == "fast":
                # Use ultrafast for intermediate files
                return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23"]
            else:
                return ["-c:v", "libx264", "-preset", "medium", "-crf", "23"]

    def _get_ken_burns_filter(
        self,
        effect: KenBurnsEffect,
        duration: float,
        resolution: str,
        fps: int,
        start_zoom: float = 1.0,
    ) -> str:
        """Get FFmpeg zoompan filter for the specified Ken Burns effect.

        Uses smooth progress-based calculations instead of per-frame increments
        to eliminate judder. The zoom/pan is calculated as a function of
        normalized progress (on/d) for smooth, consistent motion.

        Args:
            effect: The Ken Burns effect type
            duration: Duration in seconds
            resolution: Output resolution as "WIDTHxHEIGHT"
            fps: Frames per second
            start_zoom: Starting zoom level (default 1.0, use >1 to continue from zoomed)
        """
        frames = int(duration * fps)
        width, height = resolution.split("x")

        # Zoom range: 15% zoom over the duration for smooth, subtle motion
        zoom_range = 0.15
        end_zoom = start_zoom + zoom_range

        effects = {
            # Smooth zoom in: linear interpolation from start_zoom to end_zoom
            KenBurnsEffect.ZOOM_IN: (
                f"zoompan=z='{start_zoom}+{zoom_range}*on/{frames}':"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={frames}:s={resolution}:fps={fps}"
            ),
            # Smooth zoom out: linear interpolation from end_zoom to start_zoom
            KenBurnsEffect.ZOOM_OUT: (
                f"zoompan=z='{end_zoom}-{zoom_range}*on/{frames}':"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={frames}:s={resolution}:fps={fps}"
            ),
            # Pan left with fixed zoom
            KenBurnsEffect.PAN_LEFT: (
                f"zoompan=z='1.15':"
                f"x='(iw-iw/zoom)*(1-on/{frames})':y='(ih-ih/zoom)/2':"
                f"d={frames}:s={resolution}:fps={fps}"
            ),
            # Pan right with fixed zoom
            KenBurnsEffect.PAN_RIGHT: (
                f"zoompan=z='1.15':"
                f"x='(iw-iw/zoom)*on/{frames}':y='(ih-ih/zoom)/2':"
                f"d={frames}:s={resolution}:fps={fps}"
            ),
            # Pan up with fixed zoom
            KenBurnsEffect.PAN_UP: (
                f"zoompan=z='1.15':"
                f"x='(iw-iw/zoom)/2':y='(ih-ih/zoom)*(1-on/{frames})':"
                f"d={frames}:s={resolution}:fps={fps}"
            ),
            # Pan down with fixed zoom
            KenBurnsEffect.PAN_DOWN: (
                f"zoompan=z='1.15':"
                f"x='(iw-iw/zoom)/2':y='(ih-ih/zoom)*on/{frames}':"
                f"d={frames}:s={resolution}:fps={fps}"
            ),
        }

        return effects[effect]

    def create_scene_clip(
        self,
        image_path: Path,
        duration: float,
        effect: KenBurnsEffect,
        output_path: Path,
        resolution: Optional[tuple[int, int]] = None,
        fps: Optional[int] = None,
    ) -> Path:
        """
        Create a video clip from an image with Ken Burns effect.

        Args:
            image_path: Path to the source image
            duration: Duration of the clip in seconds
            effect: Ken Burns effect to apply
            output_path: Path for the output video clip
            resolution: Optional (width, height) tuple to override config
            fps: Optional frames per second to override config

        Returns:
            Path to the created video clip
        """
        if resolution:
            res_str = f"{resolution[0]}x{resolution[1]}"
        else:
            res_str = self.config.video.resolution
        fps = fps or self.config.video.fps

        filter_str = self._get_ken_burns_filter(effect, duration, res_str, fps)

        # Use fast encoding for intermediate scene clips
        encoder_args = self._get_encoder_args(quality="fast")

        cmd = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(image_path),
            "-vf",
            filter_str,
            "-t",
            str(duration),
            "-pix_fmt",
            "yuv420p",
        ]
        cmd.extend(encoder_args)
        cmd.append(str(output_path))

        subprocess.run(cmd, check=True, capture_output=True)
        return output_path

    def concatenate_clips(
        self,
        clip_paths: list[Path],
        output_path: Path,
        crossfade_duration: float = 0.5,
    ) -> Path:
        """
        Concatenate video clips with optional crossfade transitions.

        Args:
            clip_paths: List of video clip paths
            output_path: Path for the output video
            crossfade_duration: Duration of crossfade between clips (0 for none)

        Returns:
            Path to the concatenated video
        """
        if not clip_paths:
            raise ValueError("No clips to concatenate")

        if len(clip_paths) == 1:
            # Just copy the single clip
            import shutil

            shutil.copy(clip_paths[0], output_path)
            return output_path

        if crossfade_duration > 0:
            # Use xfade filter for crossfade transitions
            return self._concatenate_with_crossfade(
                clip_paths, output_path, crossfade_duration
            )
        else:
            # Simple concatenation using concat demuxer
            return self._concatenate_simple(clip_paths, output_path)

    def _concatenate_simple(
        self,
        clip_paths: list[Path],
        output_path: Path,
    ) -> Path:
        """Simple concatenation without transitions."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            for clip_path in clip_paths:
                f.write(f"file '{clip_path}'\n")
            concat_file = f.name

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file,
            "-c",
            "copy",
            str(output_path),
        ]

        subprocess.run(cmd, check=True, capture_output=True)
        Path(concat_file).unlink()

        return output_path

    def _concatenate_with_crossfade(
        self,
        clip_paths: list[Path],
        output_path: Path,
        crossfade_duration: float,
    ) -> Path:
        """Concatenate with crossfade transitions using xfade filter."""
        n_clips = len(clip_paths)

        # Get durations of all clips
        clip_durations = []
        for clip_path in clip_paths:
            duration = self._get_media_duration(clip_path)
            if duration <= 0:
                # If we can't get duration, fall back to simple concat
                return self._concatenate_simple(clip_paths, output_path)
            clip_durations.append(duration)

        # Build input arguments
        inputs = []
        for clip_path in clip_paths:
            inputs.extend(["-i", str(clip_path)])

        # Build xfade filter chain with proper offsets
        filter_parts = []
        current_stream = "[0:v]"
        cumulative_offset = clip_durations[0] - crossfade_duration

        for i in range(1, n_clips):
            next_stream = f"[{i}:v]"
            output_stream = f"[v{i}]" if i < n_clips - 1 else "[outv]"

            # The offset is when the crossfade starts (relative to output timeline)
            filter_parts.append(
                f"{current_stream}{next_stream}xfade=transition=fade:"
                f"duration={crossfade_duration}:offset={cumulative_offset:.3f}{output_stream}"
            )
            current_stream = output_stream

            # Add next clip's duration minus crossfade for next offset
            if i < n_clips - 1:
                cumulative_offset += clip_durations[i] - crossfade_duration

        # Combine all filters
        filter_complex = ";".join(filter_parts)

        # Use fast encoding for intermediate concatenation
        encoder_args = self._get_encoder_args(quality="fast")

        cmd = [
            "ffmpeg",
            "-y",
        ]
        cmd.extend(inputs)
        cmd.extend(
            [
                "-filter_complex",
                filter_complex,
                "-map",
                "[outv]",
            ]
        )
        cmd.extend(encoder_args)
        cmd.append(str(output_path))

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            # Fall back to simple concatenation if xfade fails
            return self._concatenate_simple(clip_paths, output_path)

        return output_path

    def _get_media_duration(self, path: Path) -> float:
        """Get duration of a media file using ffprobe."""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 0.0

    def _extract_last_frame(self, video_path: Path, output_path: Path) -> bool:
        """Extract the last frame of a video as an image.

        Args:
            video_path: Path to the video file
            output_path: Path to save the extracted frame (PNG format)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get video duration
            duration = self._get_media_duration(video_path)
            if duration <= 0:
                return False

            # Seek to near the end (0.1s before end) and extract one frame
            seek_time = max(0, duration - 0.1)

            cmd = [
                "ffmpeg",
                "-y",
                "-ss", str(seek_time),
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "2",  # High quality
                str(output_path),
            ]

            subprocess.run(cmd, check=True, capture_output=True)
            return output_path.exists()
        except subprocess.CalledProcessError:
            return False

    def add_audio(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
    ) -> Path:
        """
        Add audio track to a video, matching lengths properly.

        Args:
            video_path: Path to the video file
            audio_path: Path to the audio file
            output_path: Path for the output video with audio

        Returns:
            Path to the video with audio
        """
        video_duration = self._get_media_duration(video_path)
        audio_duration = self._get_media_duration(audio_path)

        # If video is shorter than audio, extend by holding the last frame
        if video_duration < audio_duration - 0.5:
            # Use tpad filter to extend the last frame instead of looping
            # This prevents the jarring "reset" effect when the video restarts
            pad_duration = audio_duration - video_duration

            # Use fast encoding for this intermediate step
            encoder_args = self._get_encoder_args(quality="fast")

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(audio_path),
                "-vf",
                f"tpad=stop_mode=clone:stop_duration={pad_duration}",
            ]
            cmd.extend(encoder_args)
            cmd.extend([
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-t",
                str(audio_duration),
                "-map",
                "0:v",
                "-map",
                "1:a",
                str(output_path),
            ])
        else:
            # Normal merge - video is long enough
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(audio_path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-t",
                str(audio_duration),
                "-map",
                "0:v",
                "-map",
                "1:a",
                str(output_path),
            ]

        subprocess.run(cmd, check=True, capture_output=True)
        return output_path

    def add_subtitles(
        self,
        video_path: Path,
        subtitle_path: Path,
        output_path: Path,
        burn_in: bool = True,
    ) -> Path:
        """
        Add subtitles to a video.

        Args:
            video_path: Path to the video file
            subtitle_path: Path to the ASS subtitle file
            output_path: Path for the output video
            burn_in: If True, burn subtitles into video; if False, add as track

        Returns:
            Path to the video with subtitles
        """
        if burn_in:
            # Burn subtitles into video using ASS filter
            # Use medium quality since this is often the final output
            encoder_args = self._get_encoder_args(quality="medium")

            # Escape the subtitle path for FFmpeg filter syntax
            escaped_path = _escape_ffmpeg_path(subtitle_path)

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"ass={escaped_path}",
            ]
            cmd.extend(encoder_args)
            cmd.extend([
                "-c:a",
                "copy",
                str(output_path),
            ])
        else:
            # Add as subtitle track
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(subtitle_path),
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-c:s",
                "ass",
                str(output_path),
            ]

        subprocess.run(cmd, check=True, capture_output=True)
        return output_path

    def create_animation_preview(
        self,
        video_path: Path,
        audio_path: Path,
        subtitle_path: Optional[Path],
        start_time: float,
        duration: float,
        output_path: Path,
        resolution: Optional[tuple[int, int]] = None,
    ) -> Path:
        """
        Create a preview clip for an animated scene with audio and optional subtitles.

        Args:
            video_path: Path to the animated video clip
            audio_path: Path to the full audio file
            subtitle_path: Optional path to ASS subtitle file (for scene's words only)
            start_time: Start time in the audio (seconds)
            duration: Duration of the scene (seconds)
            output_path: Path for the output preview
            resolution: Optional (width, height) tuple

        Returns:
            Path to the preview video
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            # Scale/prepare the animation video
            if resolution:
                res_str = f"{resolution[0]}x{resolution[1]}"
            else:
                res_str = self.config.video.resolution

            width, height = res_str.split("x")

            scaled_video = temp_dir / "scaled.mp4"
            encoder_args = self._get_encoder_args(quality="fast")

            # Scale to target resolution
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
                "-t",
                str(duration),
            ]
            cmd.extend(encoder_args)
            cmd.extend(["-an", str(scaled_video)])
            subprocess.run(cmd, check=True, capture_output=True)

            # Extract audio segment for this scene
            audio_segment = temp_dir / "audio_segment.mp3"
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-ss",
                str(start_time),
                "-t",
                str(duration),
                "-c:a",
                "libmp3lame",
                "-q:a",
                "2",
                str(audio_segment),
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            # Add audio to video
            with_audio = temp_dir / "with_audio.mp4"
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(scaled_video),
                "-i",
                str(audio_segment),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                str(with_audio),
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            if subtitle_path and subtitle_path.exists():
                # For preview, we need to shift subtitle timing to start from 0
                # Create a temporary shifted subtitle file
                shifted_sub = temp_dir / "shifted.ass"
                self._shift_subtitles(subtitle_path, shifted_sub, -start_time)

                # Add subtitles
                self.add_subtitles(with_audio, shifted_sub, output_path, burn_in=True)
            else:
                import shutil
                shutil.copy(with_audio, output_path)

        return output_path

    def create_scene_preview(
        self,
        image_path: Path,
        audio_path: Path,
        subtitle_path: Optional[Path],
        start_time: float,
        duration: float,
        effect: KenBurnsEffect,
        output_path: Path,
        resolution: Optional[tuple[int, int]] = None,
    ) -> Path:
        """
        Create a preview clip for a single scene with audio and optional subtitles.

        Args:
            image_path: Path to the scene image
            audio_path: Path to the full audio file
            subtitle_path: Optional path to ASS subtitle file
            start_time: Start time in the audio (seconds)
            duration: Duration of the scene (seconds)
            effect: Ken Burns effect to apply
            output_path: Path for the output preview
            resolution: Optional (width, height) tuple

        Returns:
            Path to the preview video
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            # Create the scene clip with Ken Burns effect
            scene_clip = temp_dir / "scene.mp4"
            self.create_scene_clip(
                image_path=image_path,
                duration=duration,
                effect=effect,
                output_path=scene_clip,
                resolution=resolution,
            )

            # Extract audio segment for this scene
            audio_segment = temp_dir / "audio_segment.mp3"
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-ss",
                str(start_time),
                "-t",
                str(duration),
                "-c:a",
                "libmp3lame",
                "-q:a",
                "2",
                str(audio_segment),
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            # Add audio to scene clip
            with_audio = temp_dir / "with_audio.mp4"
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(scene_clip),
                "-i",
                str(audio_segment),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                str(with_audio),
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            if subtitle_path and subtitle_path.exists():
                # For preview, we need to shift subtitle timing to start from 0
                # Create a temporary shifted subtitle file
                shifted_sub = temp_dir / "shifted.ass"
                self._shift_subtitles(subtitle_path, shifted_sub, -start_time)

                # Add subtitles
                self.add_subtitles(with_audio, shifted_sub, output_path, burn_in=True)
            else:
                import shutil
                shutil.copy(with_audio, output_path)

        return output_path

    def _shift_subtitles(
        self,
        input_path: Path,
        output_path: Path,
        offset_seconds: float,
    ) -> None:
        """Shift all subtitle timings by offset_seconds."""
        with open(input_path, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        shifted_lines = []

        for line in lines:
            if line.startswith("Dialogue:"):
                # Parse and shift timing
                parts = line.split(",", 9)
                if len(parts) >= 10:
                    start_ts = parts[1]
                    end_ts = parts[2]

                    new_start = self._shift_ass_time(start_ts, offset_seconds)
                    new_end = self._shift_ass_time(end_ts, offset_seconds)

                    # Only include if timing is positive
                    if new_start >= 0 and new_end > 0:
                        parts[1] = self._format_ass_time(new_start)
                        parts[2] = self._format_ass_time(new_end)
                        shifted_lines.append(",".join(parts))
            else:
                shifted_lines.append(line)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(shifted_lines))

    def _shift_ass_time(self, timestamp: str, offset: float) -> float:
        """Parse ASS timestamp and apply offset, return seconds."""
        # ASS format: H:MM:SS.CC
        parts = timestamp.strip().split(":")
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            total = hours * 3600 + minutes * 60 + seconds + offset
            return max(0, total)
        return 0

    def _format_ass_time(self, seconds: float) -> str:
        """Format seconds as ASS timestamp H:MM:SS.CC."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}:{minutes:02d}:{secs:05.2f}"

    def prepare_animated_clip(
        self,
        video_path: Path,
        target_duration: float,
        output_path: Path,
        resolution: Optional[tuple[int, int]] = None,
        fps: Optional[int] = None,
        crossfade_pad: float = 0.0,
        strict_duration: bool = True,
        effect: Optional[KenBurnsEffect] = None,
    ) -> Path:
        """
        Prepare an animated clip to match target resolution, fps, and duration.

        IMPORTANT: For lip-synced animations, we NEVER adjust playback speed
        because the lip movements are baked in to match the original audio timing.
        We only scale resolution and trim/pad to fit the target duration.

        If the video is shorter than the target duration and an effect is provided,
        we extend with Ken Burns animation on the last frame instead of freezing.

        Args:
            video_path: Path to the animated video clip
            target_duration: Target duration in seconds. If <= 0, use natural duration.
            output_path: Path for the output video
            resolution: Optional (width, height) tuple
            fps: Optional frames per second
            crossfade_pad: Duration to pad at START with first frame (for crossfade sync)
            strict_duration: If True (lip sync), pad even small gaps to maintain audio sync.
                           If False (prompt/veo), only pad large gaps (>0.5s).
            effect: Optional Ken Burns effect to use when extending short videos.
                   If None, falls back to freeze frame extension.

        Returns:
            Path to the prepared clip
        """
        if resolution:
            res_str = f"{resolution[0]}x{resolution[1]}"
        else:
            res_str = self.config.video.resolution
        fps_val = fps or self.config.video.fps

        width, height = res_str.split("x")

        # Get current video duration
        current_duration = self._get_media_duration(video_path)

        encoder_args = self._get_encoder_args(quality="fast")

        # If target_duration <= 0, use natural duration (no extension)
        use_natural_duration = target_duration <= 0
        effective_target_duration = current_duration if use_natural_duration else target_duration

        # Check if we need to extend the video
        needs_extension = (
            not use_natural_duration
            and current_duration > 0
            and current_duration < effective_target_duration - 0.05
        )
        pad_duration = effective_target_duration - current_duration if needs_extension else 0

        # If we need to extend and have a Ken Burns effect, use that instead of freeze frame
        if needs_extension and effect is not None and pad_duration > 0.1:
            logger.info(f"Video {pad_duration:.1f}s short, extending with Ken Burns effect")
            return self._extend_with_ken_burns(
                video_path=video_path,
                target_duration=target_duration,
                output_path=output_path,
                resolution=(int(width), int(height)),
                fps=fps_val,
                effect=effect,
                crossfade_pad=crossfade_pad,
            )

        # Standard processing (no Ken Burns extension needed)
        # Build filter string with proper ordering for lip sync:
        # 1. First apply start padding (BEFORE any other processing) so timing is correct
        # 2. Then scale and letterbox
        # 3. Finally normalize fps
        # IMPORTANT: Do NOT adjust playback speed - this would break lip sync!
        filter_parts = []

        # For crossfade synchronization: pad the START with frozen first frame FIRST
        # This ensures the lip sync animation starts AFTER the crossfade completes,
        # matching when the audio for this scene actually begins playing.
        # Apply tpad BEFORE other filters to ensure frame timing is correct.
        if crossfade_pad > 0:
            filter_parts.append(f"tpad=start_duration={crossfade_pad}:start_mode=clone")

        # Scale and letterbox
        filter_parts.append(f"scale={width}:{height}:force_original_aspect_ratio=decrease")
        filter_parts.append(f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2")

        # Normalize fps (AFTER tpad to preserve timing)
        filter_parts.append(f"fps={fps_val}")

        # If video is shorter than target and no Ken Burns effect, pad with last frame
        # If video is longer, it will be trimmed by -t parameter
        # Account for crossfade pad in the total duration
        effective_target = effective_target_duration + crossfade_pad
        if needs_extension:
            logger.warning(f"Video {pad_duration:.1f}s short, padding with freeze frame")
            filter_parts.append(f"tpad=stop_duration={pad_duration}:stop_mode=clone")

        filter_str = ",".join(filter_parts)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            filter_str,
        ]

        cmd.extend(encoder_args)
        cmd.extend([
            "-an",  # Remove audio (we'll add the original audio track later)
            "-t",
            str(effective_target),
            str(output_path),
        ])

        subprocess.run(cmd, check=True, capture_output=True)
        return output_path

    def _extend_with_ken_burns(
        self,
        video_path: Path,
        target_duration: float,
        output_path: Path,
        resolution: tuple[int, int],
        fps: int,
        effect: KenBurnsEffect,
        crossfade_pad: float = 0.0,
    ) -> Path:
        """
        Extend a short video by adding Ken Burns animation on the last frame.

        This creates a more visually interesting extension than freezing the last frame.

        Args:
            video_path: Path to the short video
            target_duration: Target total duration
            output_path: Path for the output video
            resolution: (width, height) tuple
            fps: Frames per second
            effect: Ken Burns effect to apply to the extension
            crossfade_pad: Duration to pad at START (for crossfade sync)

        Returns:
            Path to the extended video
        """
        current_duration = self._get_media_duration(video_path)
        extension_duration = target_duration - current_duration

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            # Step 1: Extract the last frame
            last_frame_path = temp_dir / "last_frame.png"
            if not self._extract_last_frame(video_path, last_frame_path):
                # Fall back to freeze frame if extraction fails
                logger.warning("Failed to extract last frame, falling back to freeze")
                return self._prepare_with_freeze(
                    video_path, target_duration, output_path, resolution, fps, crossfade_pad
                )

            # Step 2: Create Ken Burns extension clip from the last frame
            extension_clip_path = temp_dir / "extension.mp4"
            res_str = f"{resolution[0]}x{resolution[1]}"
            self.create_scene_clip(
                image_path=last_frame_path,
                duration=extension_duration,
                effect=effect,
                output_path=extension_clip_path,
                resolution=resolution,
                fps=fps,
            )

            # Step 3: Prepare the original video (scale, fps, trim to actual duration)
            prepared_video_path = temp_dir / "prepared.mp4"
            filter_parts = []
            if crossfade_pad > 0:
                filter_parts.append(f"tpad=start_duration={crossfade_pad}:start_mode=clone")
            filter_parts.append(f"scale={resolution[0]}:{resolution[1]}:force_original_aspect_ratio=decrease")
            filter_parts.append(f"pad={resolution[0]}:{resolution[1]}:(ow-iw)/2:(oh-ih)/2")
            filter_parts.append(f"fps={fps}")
            filter_str = ",".join(filter_parts)

            encoder_args = self._get_encoder_args(quality="fast")
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", filter_str,
                "-t", str(current_duration + crossfade_pad),
                "-an",
            ]
            cmd.extend(encoder_args)
            cmd.append(str(prepared_video_path))
            subprocess.run(cmd, check=True, capture_output=True)

            # Step 4: Concatenate the prepared video with the Ken Burns extension
            self._concatenate_simple([prepared_video_path, extension_clip_path], output_path)

            logger.info(f"Extended video with {extension_duration:.1f}s Ken Burns animation")
            return output_path

    def _prepare_with_freeze(
        self,
        video_path: Path,
        target_duration: float,
        output_path: Path,
        resolution: tuple[int, int],
        fps: int,
        crossfade_pad: float = 0.0,
    ) -> Path:
        """Fallback method: prepare video with freeze frame extension."""
        current_duration = self._get_media_duration(video_path)
        pad_duration = target_duration - current_duration

        filter_parts = []
        if crossfade_pad > 0:
            filter_parts.append(f"tpad=start_duration={crossfade_pad}:start_mode=clone")
        filter_parts.append(f"scale={resolution[0]}:{resolution[1]}:force_original_aspect_ratio=decrease")
        filter_parts.append(f"pad={resolution[0]}:{resolution[1]}:(ow-iw)/2:(oh-ih)/2")
        filter_parts.append(f"fps={fps}")
        if pad_duration > 0:
            filter_parts.append(f"tpad=stop_duration={pad_duration}:stop_mode=clone")
        filter_str = ",".join(filter_parts)

        encoder_args = self._get_encoder_args(quality="fast")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", filter_str,
            "-t", str(target_duration + crossfade_pad),
            "-an",
        ]
        cmd.extend(encoder_args)
        cmd.append(str(output_path))

        subprocess.run(cmd, check=True, capture_output=True)
        return output_path

    def generate_music_video(
        self,
        scenes: list[Scene],
        audio_path: Path,
        subtitle_path: Optional[Path],
        output_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        resolution: Optional[tuple[int, int]] = None,
        fps: Optional[int] = None,
        extension_mode: ExtensionMode = "all",
    ) -> Path:
        """
        Generate a complete music video from scenes.

        Args:
            scenes: List of Scene objects with image paths and effects
            audio_path: Path to the audio file
            subtitle_path: Optional path to ASS subtitle file
            output_path: Path for the final video
            progress_callback: Optional callback for progress updates
            resolution: Optional (width, height) tuple to override config
            fps: Optional frames per second to override config
            extension_mode: How to extend short animated clips:
                - "all": Ken Burns on all scenes that need extension (default)
                - "end_only": Pack animations back-to-back at natural duration,
                    then add Ken Burns at the very end to fill remaining time
                - "none": No Ken Burns, animations play at natural duration (may be short)

        Returns:
            Path to the final video
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            clip_paths = []

            # Count animated vs static scenes for progress reporting
            animated_count = sum(
                1 for s in scenes
                if getattr(s, 'animated', False)
                and getattr(s, 'video_path', None)
                and Path(s.video_path).exists()
            )

            # IMPORTANT: If there are ANY animated clips, disable crossfade entirely.
            # Crossfade creates timing overlaps that break lip sync. Hard cuts
            # guarantee that animated clips stay perfectly synced with their audio.
            # Only use crossfade for videos with all static scenes.
            crossfade_duration = 0.0 if animated_count > 0 else 0.3

            # Generate individual scene clips
            total_scenes = len(scenes)
            for i, scene in enumerate(scenes):
                if progress_callback:
                    scene_type = "animated" if (
                        getattr(scene, 'animated', False)
                        and getattr(scene, 'video_path', None)
                        and Path(scene.video_path).exists()
                    ) else "static"
                    progress_callback(
                        f"Creating scene {i + 1}/{total_scenes} ({scene_type})...",
                        (i / total_scenes) * 0.5,
                    )

                clip_path = temp_dir / f"clip_{i:03d}.mp4"

                # Check if this scene has an animation video
                has_animation = (
                    getattr(scene, 'animated', False)
                    and getattr(scene, 'video_path', None)
                    and Path(scene.video_path).exists()
                )

                if has_animation:
                    # Use the pre-generated animated video clip
                    # No crossfade padding needed since we disable crossfade for animated videos
                    is_lip_sync = getattr(scene, 'animation_type', None) == AnimationType.LIP_SYNC

                    # Determine target duration and effect based on extension_mode:
                    # - "all": Ken Burns on all scenes (default behavior)
                    # - "end_only": Use natural duration, Ken Burns at the very end
                    # - "none": Use natural duration, no Ken Burns
                    # Lip-sync always uses freeze frame regardless of mode.
                    if extension_mode in ("end_only", "none"):
                        # Pack mode: use natural animation duration, no per-clip extension
                        clip_target_duration = 0  # 0 = use natural duration
                        extend_effect = None
                    elif is_lip_sync:
                        # Lip-sync must freeze frame to maintain audio timing
                        clip_target_duration = scene.duration
                        extend_effect = None
                    else:
                        # Default "all" mode: Ken Burns on each scene
                        clip_target_duration = scene.duration
                        extend_effect = scene.effect

                    self.prepare_animated_clip(
                        video_path=Path(scene.video_path),
                        target_duration=clip_target_duration,
                        output_path=clip_path,
                        resolution=resolution,
                        fps=fps,
                        strict_duration=is_lip_sync,
                        effect=extend_effect,
                    )
                elif scene.image_path is not None:
                    # Create Ken Burns clip from static image
                    self.create_scene_clip(
                        image_path=scene.image_path,
                        duration=scene.duration,
                        effect=scene.effect,
                        output_path=clip_path,
                        resolution=resolution,
                        fps=fps,
                    )
                else:
                    # No image or animation, skip this scene
                    continue

                clip_paths.append(clip_path)

            if progress_callback:
                progress_callback("Concatenating scenes...", 0.6)

            # Concatenate clips with crossfade transitions
            video_no_audio = temp_dir / "video_no_audio.mp4"
            self.concatenate_clips(clip_paths, video_no_audio, crossfade_duration=crossfade_duration)

            # For "end_only" mode, check if we need Ken Burns padding at the end
            if extension_mode == "end_only":
                audio_duration = self._get_media_duration(audio_path)
                video_duration = self._get_media_duration(video_no_audio)
                gap = audio_duration - video_duration

                if gap > 0.2:  # Significant gap to fill
                    if progress_callback:
                        progress_callback(f"Extending video by {gap:.1f}s with Ken Burns...", 0.65)
                    logger.info(f"end_only mode: Video {video_duration:.1f}s, audio {audio_duration:.1f}s, extending by {gap:.1f}s")

                    # Extract last frame and extend with Ken Burns
                    last_frame_path = temp_dir / "last_frame.png"
                    extract_cmd = [
                        "ffmpeg", "-y", "-sseof", "-0.1", "-i", str(video_no_audio),
                        "-vframes", "1", "-q:v", "2", str(last_frame_path)
                    ]
                    subprocess.run(extract_cmd, check=True, capture_output=True)

                    # Create Ken Burns extension clip
                    extension_clip = temp_dir / "extension.mp4"
                    # Pick a reasonable Ken Burns effect for the end
                    end_effect = scenes[-1].effect if scenes and scenes[-1].effect else KenBurnsEffect.ZOOM_OUT
                    self.create_scene_clip(
                        image_path=last_frame_path,
                        duration=gap + 0.1,  # Slight overlap for smooth concat
                        effect=end_effect,
                        output_path=extension_clip,
                        resolution=resolution,
                        fps=fps,
                    )

                    # Concatenate original + extension
                    video_extended = temp_dir / "video_extended.mp4"
                    self.concatenate_clips([video_no_audio, extension_clip], video_extended, crossfade_duration=0)

                    # Trim to exact audio duration
                    video_trimmed = temp_dir / "video_trimmed.mp4"
                    trim_cmd = [
                        "ffmpeg", "-y", "-i", str(video_extended),
                        "-t", str(audio_duration), "-c", "copy", str(video_trimmed)
                    ]
                    subprocess.run(trim_cmd, check=True, capture_output=True)
                    video_no_audio = video_trimmed

            if progress_callback:
                progress_callback("Adding audio...", 0.7)

            # Add audio
            video_with_audio = temp_dir / "video_with_audio.mp4"
            self.add_audio(video_no_audio, audio_path, video_with_audio)

            if subtitle_path:
                if progress_callback:
                    progress_callback("Adding lyrics overlay...", 0.9)

                # Add subtitles
                self.add_subtitles(
                    video_with_audio, subtitle_path, output_path, burn_in=True
                )
            else:
                # Just copy the final video
                import shutil

                shutil.copy(video_with_audio, output_path)

            if progress_callback:
                progress_callback("Video complete!", 1.0)

            return output_path

    def generate_slideshow(
        self,
        scenes: list[Scene],
        subtitle_path: Optional[Path],
        output_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        resolution: Optional[tuple[int, int]] = None,
        fps: Optional[int] = None,
    ) -> Path:
        """
        Generate a slideshow video without audio (demo mode).

        Args:
            scenes: List of Scene objects with image paths and effects
            subtitle_path: Optional path to ASS subtitle file
            output_path: Path for the final video
            progress_callback: Optional callback for progress updates
            resolution: Optional (width, height) tuple to override config
            fps: Optional frames per second to override config

        Returns:
            Path to the final video
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            clip_paths = []

            # Count animated scenes
            animated_count = sum(
                1 for s in scenes
                if getattr(s, 'animated', False)
                and getattr(s, 'video_path', None)
                and Path(s.video_path).exists()
            )

            # IMPORTANT: If there are ANY animated clips, disable crossfade entirely.
            # Crossfade creates timing overlaps that break lip sync. Hard cuts
            # guarantee that animated clips stay perfectly synced with their audio.
            crossfade_duration = 0.0 if animated_count > 0 else 0.3

            # Generate individual scene clips
            total_scenes = len(scenes)
            for i, scene in enumerate(scenes):
                if progress_callback:
                    scene_type = "animated" if (
                        getattr(scene, 'animated', False)
                        and getattr(scene, 'video_path', None)
                        and Path(scene.video_path).exists()
                    ) else "static"
                    progress_callback(
                        f"Creating scene {i + 1}/{total_scenes} ({scene_type})...",
                        (i / total_scenes) * 0.7,
                    )

                clip_path = temp_dir / f"clip_{i:03d}.mp4"

                # Check if this scene has an animation video
                has_animation = (
                    getattr(scene, 'animated', False)
                    and getattr(scene, 'video_path', None)
                    and Path(scene.video_path).exists()
                )

                if has_animation:
                    # Use the pre-generated animated video clip
                    # No crossfade padding needed since we disable crossfade for animated videos
                    # Lip sync needs strict duration (exact timing with audio)
                    # Prompt/VEO can use Ken Burns to extend short videos
                    is_lip_sync = getattr(scene, 'animation_type', None) == AnimationType.LIP_SYNC
                    # Only use Ken Burns extension for non-lip-sync animations
                    # Lip-sync must freeze frame to maintain audio timing
                    extend_effect = None if is_lip_sync else scene.effect
                    self.prepare_animated_clip(
                        video_path=Path(scene.video_path),
                        target_duration=scene.duration,
                        output_path=clip_path,
                        resolution=resolution,
                        fps=fps,
                        strict_duration=is_lip_sync,
                        effect=extend_effect,
                    )
                elif scene.image_path is not None:
                    # Create Ken Burns clip from static image
                    self.create_scene_clip(
                        image_path=scene.image_path,
                        duration=scene.duration,
                        effect=scene.effect,
                        output_path=clip_path,
                        resolution=resolution,
                        fps=fps,
                    )
                else:
                    # No image or animation, skip this scene
                    continue

                clip_paths.append(clip_path)

            if progress_callback:
                progress_callback("Concatenating scenes...", 0.8)

            # Concatenate clips with crossfade transitions
            video_no_subs = temp_dir / "video_no_subs.mp4"
            self.concatenate_clips(clip_paths, video_no_subs, crossfade_duration=crossfade_duration)

            if subtitle_path:
                if progress_callback:
                    progress_callback("Adding lyrics overlay...", 0.9)

                # Add subtitles
                self.add_subtitles(
                    video_no_subs, subtitle_path, output_path, burn_in=True
                )
            else:
                # Just copy the final video
                import shutil

                shutil.copy(video_no_subs, output_path)

            if progress_callback:
                progress_callback("Slideshow complete!", 1.0)

            return output_path
