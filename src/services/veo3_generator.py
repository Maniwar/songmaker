"""Veo 3 Video Generator for Movie Mode.

Generates video clips with synchronized dialogue audio directly from prompts.
This replaces the need for separate TTS + image generation + lip sync.
"""

import logging
import time
from pathlib import Path
from typing import Callable, Optional

from src.config import config
from src.models.schemas import Character, MovieScene, Script

logger = logging.getLogger(__name__)


class Veo3Generator:
    """Generate video clips with dialogue using Google Veo 3."""

    def __init__(
        self,
        model: str = "veo-3.1-generate-preview",
        resolution: str = "1080p",
        duration: int = 8,  # 4, 6, or 8 seconds
    ):
        """Initialize Veo 3 generator.

        Args:
            model: Veo model to use (veo-3.1-generate-preview, veo-3.0-generate-preview)
            resolution: Output resolution (720p or 1080p)
            duration: Clip duration in seconds (4, 6, or 8)
        """
        self.model = model
        self.resolution = resolution
        self.duration = duration
        self._client = None
        self._uploaded_files = {}  # Cache for uploaded video files

    def _get_client(self):
        """Lazy-load the Gemini client."""
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise ImportError(
                    "google-genai package not installed. Run: pip install google-genai"
                ) from exc

            api_key = config.google_api_key
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set in environment")

            self._client = genai.Client(api_key=api_key)

        return self._client

    def upload_video_reference(self, video_path: Path) -> Optional[object]:
        """Upload a video file using Files API for use as reference.

        Per Gemini docs: Use Files API when total request size >20MB
        or when reusing the file across multiple requests.

        Args:
            video_path: Path to the video file to upload

        Returns:
            Uploaded file object, or None if upload failed
        """
        if not video_path.exists():
            logger.warning(f"Video reference path does not exist: {video_path}")
            return None

        # Check cache first
        cache_key = str(video_path)
        if cache_key in self._uploaded_files:
            logger.info(f"Using cached video reference: {video_path.name}")
            return self._uploaded_files[cache_key]

        client = self._get_client()

        try:
            logger.info(f"Uploading video reference via Files API: {video_path.name}")
            uploaded_file = client.files.upload(file=str(video_path))
            logger.info(f"Video uploaded: {uploaded_file.name}, URI: {uploaded_file.uri}")

            # Cache the uploaded file
            self._uploaded_files[cache_key] = uploaded_file
            return uploaded_file

        except Exception as e:
            logger.error(f"Failed to upload video reference: {e}")
            return None

    def analyze_video_for_continuity(
        self,
        video_path: Path,
        prompt: str = "Describe the ending of this video clip in detail, including character positions, actions, and visual style."
    ) -> Optional[str]:
        """Analyze a video to extract continuity information for the next scene.

        Uses Gemini's video understanding to describe the ending state,
        which can be used to maintain visual continuity.

        Args:
            video_path: Path to the video to analyze
            prompt: Analysis prompt

        Returns:
            Analysis text, or None if failed
        """
        client = self._get_client()

        # Upload the video
        uploaded_video = self.upload_video_reference(video_path)
        if not uploaded_video:
            return None

        try:
            # Use Gemini to analyze the video
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[uploaded_video, prompt],
            )
            analysis = response.text
            logger.info(f"Video analysis for continuity: {analysis[:100]}...")
            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze video: {e}")
            return None

    def build_scene_prompt(
        self,
        scene: MovieScene,
        script: Script,
        style: Optional[str] = None,
    ) -> str:
        """Build a Veo 3 prompt from a movie scene.

        Constructs a prompt that includes:
        - Visual setting and camera direction
        - Character descriptions
        - Dialogue in quotes (Veo 3 will generate speech)
        - Sound effects and ambient audio

        Args:
            scene: The movie scene to generate
            script: The full script (for character info)
            style: Optional visual style override

        Returns:
            Formatted prompt for Veo 3
        """
        parts = []

        # Visual style
        visual_style = style or script.visual_style or "cinematic film"
        parts.append(f"Style: {visual_style}")

        # Scene setting
        parts.append(f"Setting: {scene.direction.setting}")

        # Camera and lighting
        parts.append(f"Camera: {scene.direction.camera}")
        if scene.direction.lighting:
            parts.append(f"Lighting: {scene.direction.lighting}")

        # Mood
        parts.append(f"Mood: {scene.direction.mood}")

        # Character descriptions for visual and voice consistency
        for char_id in scene.direction.visible_characters:
            char = script.get_character(char_id)
            if char:
                parts.append(f"Character {char.name}: {char.description}")
                # Add voice description for voice continuity across scenes
                veo_voice = getattr(char, 'veo_voice_description', None)
                if veo_voice:
                    parts.append(f"{char.name}'s voice: {veo_voice}")

        # Build dialogue section with quotes
        if scene.dialogue:
            parts.append("\nAction and dialogue:")
            for dialogue in scene.dialogue:
                char = script.get_character(dialogue.character_id)
                char_name = char.name if char else dialogue.character_id

                # Build dialogue line with emotion/action cues
                line_parts = []

                # Character name
                line_parts.append(char_name)

                # Emotion/delivery cue
                if dialogue.emotion.value != "neutral":
                    line_parts.append(f"({dialogue.emotion.value})")

                # Action if present
                if dialogue.action:
                    line_parts.append(f"[{dialogue.action}]")

                # The actual dialogue in quotes (Veo 3 will speak this)
                line_parts.append(f'says, "{dialogue.text}"')

                parts.append(" ".join(line_parts))

        # Ambient audio cues
        if scene.direction.mood:
            ambient_map = {
                "tense": "Ambient: low ominous drone, subtle tension",
                "happy": "Ambient: light cheerful atmosphere",
                "sad": "Ambient: quiet, melancholic tone",
                "mysterious": "Ambient: eerie atmosphere, subtle mystery",
                "romantic": "Ambient: soft, warm atmosphere",
                "action": "Ambient: intense, energetic background",
                "comedic": "Ambient: light playful tone",
            }
            if scene.direction.mood in ambient_map:
                parts.append(ambient_map[scene.direction.mood])

        return "\n".join(parts)

    def generate_scene(
        self,
        scene: MovieScene,
        script: Script,
        output_path: Path,
        style: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        reference_images: Optional[list[Path]] = None,
        first_frame: Optional[Path] = None,
        previous_video: Optional[Path] = None,
        use_video_continuity: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Optional[Path]:
        """Generate a video clip for a movie scene.

        Args:
            scene: The scene to generate
            script: The full script
            output_path: Where to save the video
            style: Optional visual style override
            custom_prompt: Optional custom video prompt (overrides auto-generated prompt)
            reference_images: Up to 3 reference images for character/asset consistency
            first_frame: Optional image to use as first frame
            previous_video: Optional path to previous scene's video for continuity
            use_video_continuity: If True, analyze previous video for continuity cues
            progress_callback: Optional progress callback

        Returns:
            Path to generated video, or None if failed
        """
        client = self._get_client()

        # Use custom prompt if provided, otherwise build from scene
        if custom_prompt and custom_prompt.strip():
            # Use custom prompt but prepend style for consistency
            visual_style = style or script.visual_style or "cinematic film"
            prompt = f"VISUAL STYLE (CRITICAL - maintain throughout): {visual_style}\n\n{custom_prompt}"
        else:
            prompt = self.build_scene_prompt(scene, script, style)

        # Add style enforcement at the end for consistency
        visual_style = style or script.visual_style or "cinematic film"
        if "photorealistic" in visual_style.lower() or "realistic" in visual_style.lower():
            prompt += f"\n\nIMPORTANT: Maintain {visual_style} style throughout. Characters must look photorealistic and consistent with their reference images."

        # Add continuity information from previous video if available
        if previous_video and use_video_continuity and previous_video.exists():
            if progress_callback:
                progress_callback(f"Analyzing previous scene for continuity...", 0.05)

            continuity_info = self.analyze_video_for_continuity(previous_video)
            if continuity_info:
                prompt = f"""CONTINUITY FROM PREVIOUS SCENE:
{continuity_info}

CURRENT SCENE (continue seamlessly from above):
{prompt}

IMPORTANT: Maintain visual continuity with the previous scene - match character positions, lighting, and style."""

        logger.info("Generating scene %d with Veo 3.1", scene.index)
        logger.debug("Prompt: %s", prompt)

        if progress_callback:
            progress_callback(f"Generating scene {scene.index}...", 0.1)

        try:
            from google.genai import types

            # Build config - don't specify person_generation as allow_adult is not supported in some regions
            config = types.GenerateVideosConfig(
                aspect_ratio="16:9",
                number_of_videos=1,
                duration_seconds=self.duration,
                resolution=self.resolution,
            )

            # Add reference images for character consistency (Veo 3.1 only)
            if reference_images:
                ref_images = []
                for ref_path in reference_images[:3]:  # Max 3 reference images
                    if ref_path.exists():
                        # Load image and create reference
                        with open(ref_path, "rb") as f:
                            image_data = f.read()
                        ref_image = types.VideoGenerationReferenceImage(
                            image=types.Image(data=image_data),
                            reference_type="asset",
                        )
                        ref_images.append(ref_image)
                if ref_images:
                    config.reference_images = ref_images

            # Prepare generation kwargs
            gen_kwargs = {
                "model": self.model,
                "prompt": prompt,
                "config": config,
            }

            # Add first frame image if provided
            if first_frame and first_frame.exists():
                with open(first_frame, "rb") as f:
                    image_data = f.read()
                gen_kwargs["image"] = types.Image(data=image_data)

            # Start video generation
            operation = client.models.generate_videos(**gen_kwargs)

            # Poll for completion
            poll_count = 0
            max_polls = 120  # 20 minutes max (10s intervals)

            while not operation.done:
                poll_count += 1
                if poll_count > max_polls:
                    logger.error("Video generation timed out")
                    return None

                if progress_callback:
                    progress = min(0.1 + (poll_count / max_polls) * 0.8, 0.9)
                    progress_callback(f"Generating scene {scene.index}...", progress)

                time.sleep(10)
                operation = client.operations.get(operation)

            # Download the video
            if operation.response and operation.response.generated_videos:
                generated_video = operation.response.generated_videos[0]

                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Download and save
                client.files.download(file=generated_video.video)
                generated_video.video.save(str(output_path))

                if progress_callback:
                    progress_callback(f"Scene {scene.index} complete", 1.0)

                logger.info("Generated scene %d: %s", scene.index, output_path)
                return output_path
            else:
                logger.error("No video generated for scene %d", scene.index)
                return None

        except Exception as e:
            logger.error("Veo 3.1 generation failed for scene %d: %s", scene.index, e)
            if progress_callback:
                progress_callback(f"Scene {scene.index} failed: {e}", 0.0)
            return None

    def generate_all_scenes(
        self,
        script: Script,
        output_dir: Path,
        style: Optional[str] = None,
        use_character_references: bool = True,
        use_scene_continuity: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> list[Path]:
        """Generate video clips for all scenes in a script.

        Args:
            script: The script with scenes
            output_dir: Directory for output videos
            style: Optional visual style
            use_character_references: Use character portrait images as references
            use_scene_continuity: Use scene images for visual continuity between clips
            progress_callback: Optional progress callback

        Returns:
            List of paths to generated videos
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        generated = []
        total_scenes = len(script.scenes)

        # Collect character reference images if available
        reference_images = []
        if use_character_references:
            for char in script.characters:
                if char.reference_image_path and Path(char.reference_image_path).exists():
                    reference_images.append(Path(char.reference_image_path))
                    if len(reference_images) >= 3:  # Veo 3.1 max is 3
                        break

        previous_scene_image = None

        for i, scene in enumerate(script.scenes):
            output_path = output_dir / f"scene_{scene.index:03d}.mp4"

            def scene_progress(msg, prog):
                if progress_callback:
                    overall = (i + prog) / total_scenes
                    progress_callback(msg, overall)

            # Determine first frame for scene continuity
            first_frame = None
            if use_scene_continuity and i > 0 and previous_scene_image:
                first_frame = previous_scene_image
                logger.info("Using previous scene image for continuity in scene %d", scene.index)

            result = self.generate_scene(
                scene=scene,
                script=script,
                output_path=output_path,
                style=style,
                reference_images=reference_images if reference_images else None,
                first_frame=first_frame,
                progress_callback=scene_progress,
            )

            if result:
                generated.append(result)
                # Store path in scene for later use
                scene.video_path = result

                # Update previous_scene_image if scene has an image
                if scene.image_path and Path(scene.image_path).exists():
                    previous_scene_image = Path(scene.image_path)
            else:
                logger.warning("Failed to generate scene %d", scene.index)

        return generated

    def extend_video(
        self,
        video_path: Path,
        prompt: str,
        output_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Optional[Path]:
        """Extend a previously generated Veo video by ~7 seconds.

        Note: Only works with Veo-generated videos up to 141 seconds.

        Args:
            video_path: Path to the Veo-generated video to extend
            prompt: Prompt describing what happens next
            output_path: Where to save the extended video
            progress_callback: Optional progress callback

        Returns:
            Path to extended video, or None if failed
        """
        client = self._get_client()

        if progress_callback:
            progress_callback("Extending video...", 0.1)

        try:
            from google.genai import types

            # Load the video
            with open(video_path, "rb") as f:
                video_data = f.read()

            # Start extension
            operation = client.models.generate_videos(
                model=self.model,
                video=types.Video(data=video_data),
                prompt=prompt,
                config=types.GenerateVideosConfig(
                    number_of_videos=1,
                    resolution="720p",  # Extension only supports 720p
                ),
            )

            # Poll for completion
            poll_count = 0
            max_polls = 120

            while not operation.done:
                poll_count += 1
                if poll_count > max_polls:
                    logger.error("Video extension timed out")
                    return None

                if progress_callback:
                    progress = min(0.1 + (poll_count / max_polls) * 0.8, 0.9)
                    progress_callback("Extending video...", progress)

                time.sleep(10)
                operation = client.operations.get(operation)

            # Download the extended video
            if operation.response and operation.response.generated_videos:
                extended_video = operation.response.generated_videos[0]

                output_path.parent.mkdir(parents=True, exist_ok=True)
                client.files.download(file=extended_video.video)
                extended_video.video.save(str(output_path))

                if progress_callback:
                    progress_callback("Extension complete", 1.0)

                logger.info("Extended video saved to: %s", output_path)
                return output_path
            else:
                logger.error("No extended video generated")
                return None

        except Exception as e:
            logger.error("Video extension failed: %s", e)
            if progress_callback:
                progress_callback(f"Extension failed: {e}", 0.0)
            return None


def check_veo3_available() -> bool:
    """Check if Veo 3.1 is available."""
    try:
        from google import genai  # noqa: F401
        return bool(config.google_api_key)
    except ImportError:
        return False


def get_veo3_pricing_estimate(num_scenes: int, duration: int = 8) -> dict:
    """Estimate Veo 3 costs for a movie.

    Args:
        num_scenes: Number of scenes to generate
        duration: Duration per scene in seconds

    Returns:
        Dict with pricing breakdown
    """
    # Veo 3 pricing: $0.75 per second with audio
    cost_per_second = 0.75
    total_seconds = num_scenes * duration
    total_cost = total_seconds * cost_per_second

    return {
        "num_scenes": num_scenes,
        "duration_per_scene": duration,
        "total_seconds": total_seconds,
        "cost_per_second": cost_per_second,
        "estimated_total": total_cost,
        "note": "Prices as of Dec 2025. Actual costs may vary.",
    }
