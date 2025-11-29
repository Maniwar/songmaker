"""Video generation service using FFmpeg with Ken Burns effects."""

import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

from src.config import Config, config as default_config
from src.models.schemas import Scene, KenBurnsEffect


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
    ) -> str:
        """Get FFmpeg zoompan filter for the specified Ken Burns effect."""
        frames = int(duration * fps)
        width, height = resolution.split("x")

        effects = {
            KenBurnsEffect.ZOOM_IN: (           
                f"zoompan=z='min(zoom+0.001,1.3)':"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={frames}:s={resolution}:fps={fps}"
            ),
            KenBurnsEffect.ZOOM_OUT: (
                f"zoompan=z='if(lte(zoom,1.0),1.3,max(1.001,zoom-0.001))':"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                                                                 f"d={frames}:s={resolution}:fps={fps}"
            ),
            KenBurnsEffect.PAN_LEFT: (
                f"zoompan=z='1.15':"
                f"x='(iw-iw/zoom)*(1-on/{frames})':y='(ih-ih/zoom)/2':"
                f"d={frames}:s={resolution}:fps={fps}"
            ),
            KenBurnsEffect.PAN_RIGHT: (
                f"zoompan=z='1.15':"
                f"x='(iw-iw/zoom)*on/{frames}':y='(ih-ih/zoom)/2':"
                f"d={frames}:s={resolution}:fps={fps}"
            ),
            KenBurnsEffect.PAN_UP: (
                f"zoompan=z='1.15':"
                f"x='(iw-iw/zoom)/2':y='(ih-ih/zoom)*(1-on/{frames})':"
                f"d={frames}:s={resolution}:fps={fps}"
            ),
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
                # Add subtitles (already time-shifted for scene start = 0)
                self.add_subtitles(with_audio, subtitle_path, output_path, burn_in=True)
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
    ) -> Path:
        """
        Prepare an animated clip to match target resolution, fps, and duration.

        IMPORTANT: For lip-synced animations, we NEVER adjust playback speed
        because the lip movements are baked in to match the original audio timing.
        We only scale resolution and trim/pad to fit the target duration.

        Args:
            video_path: Path to the animated video clip
            target_duration: Target duration in seconds
            output_path: Path for the output video
            resolution: Optional (width, height) tuple
            fps: Optional frames per second
            crossfade_pad: Duration to pad at START with first frame (for crossfade sync)

        Returns:
            Path to the prepared clip
        """
        if resolution:
            res_str = f"{resolution[0]}x{resolution[1]}"
        else:
            res_str = self.config.video.resolution
        fps = fps or self.config.video.fps

        width, height = res_str.split("x")

        # Get current video duration
        current_duration = self._get_media_duration(video_path)

        encoder_args = self._get_encoder_args(quality="fast")

        # Build filter string: scale to target resolution
        # IMPORTANT: Do NOT adjust playback speed - this would break lip sync!
        # The animated video's timing is synced to the audio it was generated with.
        filter_str = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps}"

        # For crossfade synchronization: pad the START with frozen first frame
        # This ensures the lip sync animation starts AFTER the crossfade completes,
        # matching when the audio for this scene actually begins playing.
        if crossfade_pad > 0:
            filter_str = f"{filter_str},tpad=start_mode=clone:start_duration={crossfade_pad}"

        # If video is shorter than target, pad with last frame (tpad)
        # If video is longer, it will be trimmed by -t parameter
        # Account for crossfade pad in the total duration
        effective_target = target_duration + crossfade_pad
        if current_duration > 0 and current_duration < target_duration - 0.1:
            pad_duration = target_duration - current_duration
            filter_str = f"{filter_str},tpad=stop_mode=clone:stop_duration={pad_duration}"

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

    def generate_music_video(
        self,
        scenes: list[Scene],
        audio_path: Path,
        subtitle_path: Optional[Path],
        output_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        resolution: Optional[tuple[int, int]] = None,
        fps: Optional[int] = None,
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

            # Crossfade duration - animated clips after the first need padding
            crossfade_duration = 0.3

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
                    # For animated clips after the first clip, add crossfade padding
                    # This ensures lip sync starts AFTER the crossfade completes
                    # Use len(clip_paths) > 0 instead of i > 0 to handle skipped scenes
                    crossfade_pad = crossfade_duration if len(clip_paths) > 0 else 0.0

                    # Use the pre-generated animated video clip
                    self.prepare_animated_clip(
                        video_path=Path(scene.video_path),
                        target_duration=scene.duration,
                        output_path=clip_path,
                        resolution=resolution,
                        fps=fps,
                        crossfade_pad=crossfade_pad,
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

            # Crossfade duration - animated clips after the first need padding
            crossfade_duration = 0.3

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
                    # For animated clips after the first clip, add crossfade padding
                    # This ensures lip sync starts AFTER the crossfade completes
                    # Use len(clip_paths) > 0 instead of i > 0 to handle skipped scenes
                    crossfade_pad = crossfade_duration if len(clip_paths) > 0 else 0.0

                    # Use the pre-generated animated video clip
                    self.prepare_animated_clip(
                        video_path=Path(scene.video_path),
                        target_duration=scene.duration,
                        output_path=clip_path,
                        resolution=resolution,
                        fps=fps,
                        crossfade_pad=crossfade_pad,
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
