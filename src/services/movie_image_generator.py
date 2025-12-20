"""Image generation service for Movie Mode with character consistency.

Ensures consistent character appearance across all scenes by including
detailed character descriptions in every image prompt.
"""

import logging
from pathlib import Path
from typing import Callable, Optional

from src.models.schemas import Character, MovieScene, Script, SceneDirection
from src.services.image_generator import ImageGenerator

logger = logging.getLogger(__name__)


class MovieImageGenerator:
    """Generates images for movie scenes with character consistency."""

    def __init__(self, style: str = "cinematic digital art, 4K quality, photorealistic"):
        """Initialize the movie image generator.

        Args:
            style: Base visual style for all images
        """
        self.style = style
        self.image_generator = ImageGenerator()

    def build_scene_prompt(
        self,
        scene: MovieScene,
        script: Script,
        include_characters: bool = True,
    ) -> str:
        """Build a comprehensive prompt for a scene with character descriptions.

        Args:
            scene: The scene to generate an image for
            script: The full script (for character lookup)
            include_characters: Whether to include character descriptions

        Returns:
            Complete prompt string for image generation
        """
        parts = []

        # Start with scene direction
        direction = scene.direction

        # Setting and camera
        parts.append(f"{direction.camera} of {direction.setting}")

        # Add characters if present
        if include_characters and direction.visible_characters:
            char_descriptions = []
            for char_id in direction.visible_characters:
                character = script.get_character(char_id)
                if character:
                    char_descriptions.append(f"{character.name}: {character.description}")

            if char_descriptions:
                parts.append("Characters present: " + "; ".join(char_descriptions))

        # Add lighting if specified
        if direction.lighting:
            parts.append(f"Lighting: {direction.lighting}")

        # Add mood
        parts.append(f"Mood: {direction.mood}")

        # Add world description if available
        if script.world_description:
            parts.append(f"Setting style: {script.world_description}")

        # Add visual style
        parts.append(f"Style: {script.visual_style or self.style}")

        # Quality modifiers
        parts.append("highly detailed, professional cinematography")

        return ". ".join(parts)

    def generate_scene_image(
        self,
        scene: MovieScene,
        script: Script,
        output_dir: Path,
        reference_image: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Optional[Path]:
        """Generate an image for a movie scene.

        Args:
            scene: The scene to generate
            script: Full script for character lookup
            output_dir: Directory to save images
            reference_image: Optional reference image for style consistency
            progress_callback: Optional progress callback

        Returns:
            Path to generated image, or None if failed
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build the prompt
        prompt = self.build_scene_prompt(scene, script)

        if progress_callback:
            progress_callback(f"Generating image for scene {scene.index + 1}...", 0.0)

        logger.info(f"Generating scene {scene.index}: {prompt[:100]}...")

        # Generate the image
        output_path = output_dir / f"scene_{scene.index:03d}.png"

        try:
            result = self.image_generator.generate(
                prompt=prompt,
                output_path=output_path,
                reference_image=reference_image,
            )

            if result and result.exists():
                logger.info(f"Generated scene image: {result}")
                return result
            else:
                logger.error(f"Failed to generate scene {scene.index}")
                return None

        except Exception as e:
            logger.error(f"Error generating scene {scene.index}: {e}")
            return None

    def generate_all_scenes(
        self,
        script: Script,
        output_dir: Path,
        use_sequential_mode: bool = False,
        hero_image: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> list[Path]:
        """Generate images for all scenes in a script.

        Args:
            script: The complete script
            output_dir: Directory to save images
            use_sequential_mode: Use previous image as reference for consistency
            hero_image: Optional hero image for style reference
            progress_callback: Optional progress callback

        Returns:
            List of generated image paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_images = []
        total_scenes = len(script.scenes)

        reference_image = hero_image

        for i, scene in enumerate(script.scenes):
            if progress_callback:
                progress = i / total_scenes
                progress_callback(
                    f"Generating scene {i + 1}/{total_scenes}...",
                    progress
                )

            image_path = self.generate_scene_image(
                scene=scene,
                script=script,
                output_dir=output_dir,
                reference_image=reference_image,
                progress_callback=None,  # Don't nest callbacks
            )

            if image_path:
                generated_images.append(image_path)
                # Use this image as reference for next in sequential mode
                if use_sequential_mode:
                    reference_image = image_path

        if progress_callback:
            progress_callback("Image generation complete!", 1.0)

        return generated_images

    def generate_character_reference(
        self,
        character: Character,
        output_dir: Path,
        style: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Optional[Path]:
        """Generate a reference image for a character.

        Args:
            character: The character to generate
            output_dir: Directory to save the image
            style: Optional style override
            progress_callback: Optional progress callback

        Returns:
            Path to generated reference image
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        prompt = (
            f"Portrait of {character.description}. "
            f"Character portrait, neutral expression, centered composition. "
            f"Style: {style or self.style}. "
            f"Highly detailed, professional quality."
        )

        if progress_callback:
            progress_callback(f"Generating reference for {character.name}...", 0.0)

        output_path = output_dir / f"character_{character.id}_reference.png"

        try:
            result = self.image_generator.generate(
                prompt=prompt,
                output_path=output_path,
            )

            if result and result.exists():
                logger.info(f"Generated character reference: {result}")
                return result
            else:
                logger.error(f"Failed to generate reference for {character.name}")
                return None

        except Exception as e:
            logger.error(f"Error generating character reference: {e}")
            return None


def build_character_consistency_prompt(
    characters: list[Character],
    scene_setting: str,
    visible_char_ids: list[str],
    camera_angle: str = "medium shot",
    mood: str = "neutral",
    style: str = "cinematic digital art",
) -> str:
    """Build a prompt that ensures character consistency.

    This helper function builds a prompt that includes full character
    descriptions to maintain visual consistency across scenes.

    Args:
        characters: List of all characters
        scene_setting: The scene's location/setting
        visible_char_ids: IDs of characters visible in this scene
        camera_angle: Camera angle/shot type
        mood: Scene mood
        style: Visual style

    Returns:
        Complete prompt string
    """
    # Get visible characters
    visible_chars = [c for c in characters if c.id in visible_char_ids]

    parts = [f"{camera_angle} of {scene_setting}"]

    # Add each visible character with full description
    for char in visible_chars:
        parts.append(f"{char.name} ({char.description})")

    parts.extend([
        f"Mood: {mood}",
        f"Style: {style}",
        "highly detailed, consistent character appearance, professional cinematography",
    ])

    return ". ".join(parts)
