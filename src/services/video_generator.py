"""Video generation service using FFmpeg with Ken Burns effects."""

import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

from src.config import Config, config as default_config
from src.models.schemas import Scene, KenBurnsEffect


class VideoGenerator:
    """Generate videos with Ken Burns effects using FFmpeg."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config

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
    ) -> Path:
        """
        Create a video clip from an image with Ken Burns effect.

        Args:
            image_path: Path to the source image
            duration: Duration of the clip in seconds
            effect: Ken Burns effect to apply
            output_path: Path for the output video clip
            resolution: Optional (width, height) tuple to override config

        Returns:
            Path to the created video clip
        """
        if resolution:
            res_str = f"{resolution[0]}x{resolution[1]}"
        else:
            res_str = self.config.video.resolution
        fps = self.config.video.fps

        filter_str = self._get_ken_burns_filter(effect, duration, res_str, fps)

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
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            str(output_path),
        ]

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
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
                str(output_path),
            ]
        )

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
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(audio_path),
                "-vf",
                f"tpad=stop_mode=clone:stop_duration={pad_duration}",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
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
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"ass='{subtitle_path}'",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-c:a",
                "copy",
                str(output_path),
            ]
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

    def generate_music_video(
        self,
        scenes: list[Scene],
        audio_path: Path,
        subtitle_path: Optional[Path],
        output_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        resolution: Optional[tuple[int, int]] = None,
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

        Returns:
            Path to the final video
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            clip_paths = []

            # Generate individual scene clips
            total_scenes = len(scenes)
            for i, scene in enumerate(scenes):
                if progress_callback:
                    progress_callback(
                        f"Creating scene {i + 1}/{total_scenes}...",
                        (i / total_scenes) * 0.5,
                    )

                if scene.image_path is None:
                    continue

                clip_path = temp_dir / f"clip_{i:03d}.mp4"
                self.create_scene_clip(
                    image_path=scene.image_path,
                    duration=scene.duration,
                    effect=scene.effect,
                    output_path=clip_path,
                    resolution=resolution,
                )
                clip_paths.append(clip_path)

            if progress_callback:
                progress_callback("Concatenating scenes...", 0.6)

            # Concatenate clips
            video_no_audio = temp_dir / "video_no_audio.mp4"
            self.concatenate_clips(clip_paths, video_no_audio, crossfade_duration=0.3)

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
    ) -> Path:
        """
        Generate a slideshow video without audio (demo mode).

        Args:
            scenes: List of Scene objects with image paths and effects
            subtitle_path: Optional path to ASS subtitle file
            output_path: Path for the final video
            progress_callback: Optional callback for progress updates
            resolution: Optional (width, height) tuple to override config

        Returns:
            Path to the final video
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            clip_paths = []

            # Generate individual scene clips
            total_scenes = len(scenes)
            for i, scene in enumerate(scenes):
                if progress_callback:
                    progress_callback(
                        f"Creating scene {i + 1}/{total_scenes}...",
                        (i / total_scenes) * 0.7,
                    )

                if scene.image_path is None:
                    continue

                clip_path = temp_dir / f"clip_{i:03d}.mp4"
                self.create_scene_clip(
                    image_path=scene.image_path,
                    duration=scene.duration,
                    effect=scene.effect,
                    output_path=clip_path,
                    resolution=resolution,
                )
                clip_paths.append(clip_path)

            if progress_callback:
                progress_callback("Concatenating scenes...", 0.8)

            # Concatenate clips
            video_no_subs = temp_dir / "video_no_subs.mp4"
            self.concatenate_clips(clip_paths, video_no_subs, crossfade_duration=0.3)

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
