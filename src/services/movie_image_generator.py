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

        # Use custom visual prompt if provided, otherwise build from scene data
        if scene.visual_prompt and scene.visual_prompt.strip():
            prompt = scene.visual_prompt
            logger.info(f"Using custom visual prompt for scene {scene.index}")
        else:
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
        use_character_references: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> list[Path]:
        """Generate images for all scenes in a script.

        Args:
            script: The complete script
            output_dir: Directory to save images
            use_sequential_mode: Use previous image as reference for consistency
            hero_image: Optional hero image for style reference
            use_character_references: Use character portraits as references
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

            # Determine reference image for this scene
            scene_reference = reference_image

            # Use character portrait as reference if available
            # Note: Image generators typically only support one reference image,
            # so we use the first character's portrait. All character descriptions
            # are still included in the prompt via build_scene_prompt().
            if use_character_references and scene.direction.visible_characters:
                chars_with_portraits = []
                for char_id in scene.direction.visible_characters:
                    character = script.get_character(char_id)
                    if character and character.reference_image_path:
                        char_ref_path = Path(character.reference_image_path)
                        if char_ref_path.exists():
                            chars_with_portraits.append((character, char_ref_path))

                if chars_with_portraits:
                    # Use first character's portrait as reference
                    first_char, scene_reference = chars_with_portraits[0]
                    logger.info(f"Using {first_char.name}'s portrait as reference for scene {i+1}")

                    if len(chars_with_portraits) > 1:
                        other_chars = ", ".join(c.name for c, _ in chars_with_portraits[1:])
                        logger.info(f"Note: {other_chars} also in scene - descriptions included in prompt")

            image_path = self.generate_scene_image(
                scene=scene,
                script=script,
                output_dir=output_dir,
                reference_image=scene_reference,
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
        image_size: str = "2K",
        aspect_ratio: str = "1:1",
        model: Optional[str] = None,
        source_image: Optional["Image.Image"] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Optional[Path]:
        """Generate a reference image for a character.

        Args:
            character: The character to generate
            output_dir: Directory to save the image
            style: Optional style override
            image_size: Image size ("2K" or "4K")
            aspect_ratio: Aspect ratio (e.g., "1:1", "3:4", "16:9")
            model: Optional model override (e.g., "gemini-2.0-flash-exp")
            source_image: Optional source photo to transform into character
            progress_callback: Optional progress callback

        Returns:
            Path to generated reference image
        """
        from io import BytesIO
        from PIL import Image as PILImage

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"character_{character.id}_reference.png"
        effective_style = style or self.style

        if progress_callback:
            progress_callback(f"Generating reference for {character.name}...", 0.0)

        if source_image:
            # Use dedicated photo transformation
            return self._transform_photo_to_character(
                source_image=source_image,
                character=character,
                style=effective_style,
                output_path=output_path,
                image_size=image_size,
                aspect_ratio=aspect_ratio,
                model=model,
            )
        else:
            # Generate from scratch
            prompt = (
                f"{effective_style} style portrait. "
                f"A {effective_style} character portrait of {character.description}. "
                f"Centered composition, neutral expression, professional lighting. "
                f"Highly detailed, {effective_style}."
            )

            try:
                result_image = self.image_generator.generate_scene_image(
                    prompt=prompt,
                    style_prefix=effective_style,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    output_path=output_path,
                    model=model,
                )

                if result_image and output_path.exists():
                    logger.info(f"Generated character reference: {output_path}")
                    return output_path
                else:
                    logger.error(f"Failed to generate reference for {character.name}")
                    return None

            except Exception as e:
                logger.error(f"Error generating character reference: {e}")
                return None

    def _transform_photo_to_character(
        self,
        source_image,
        character: Character,
        style: str,
        output_path: Path,
        image_size: str = "2K",
        aspect_ratio: str = "1:1",
        model: Optional[str] = None,
    ) -> Optional[Path]:
        """Transform a source photo into a character portrait using Gemini.

        Uses gemini-2.5-flash-image or gemini-3-pro-image-preview for image editing.
        Imagen models don't support image input.
        """
        from io import BytesIO
        from PIL import Image as PILImage

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            logger.error("google-genai not installed")
            return None

        from src.config import config
        client = genai.Client(api_key=config.google_api_key)

        # Use Gemini models for image editing - Imagen doesn't support image input
        # gemini-2.5-flash-image is optimized for image editing
        # gemini-3-pro-image-preview is best for complex edits up to 4K
        if model and "imagen" in model.lower():
            logger.warning("Imagen doesn't support image editing, using gemini-2.5-flash-image")
            model_name = "gemini-2.5-flash-image"
        elif model and "gemini" in model.lower():
            model_name = model
        else:
            # Default to gemini-2.5-flash-image for fast image editing
            model_name = "gemini-2.5-flash-image"

        # Build transformation prompt following Google's best practices
        transform_prompt = f"""Transform this photograph into a {style} style character portrait.

KEEP the person's face shape, eye shape, and basic facial features recognizable.
APPLY this character description: {character.description}
CHANGE the art style to: {style}
Make it a professional quality portrait with good composition and lighting."""

        logger.info(f"Transforming photo with {model_name}, size={image_size}")

        try:
            # Use the simpler format: contents=[prompt, image]
            response = client.models.generate_content(
                model=model_name,
                contents=[transform_prompt, source_image],
                config=types.GenerateContentConfig(
                    response_modalities=['IMAGE', 'TEXT'],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=image_size,
                    ),
                ),
            )

            # Extract image from response
            result_image = None
            error_text = None

            for part in response.parts:
                if part.inline_data and part.inline_data.data:
                    result_image = PILImage.open(BytesIO(part.inline_data.data))
                    break
                elif hasattr(part, 'as_image') and callable(part.as_image):
                    result_image = part.as_image()
                    break
                elif part.text:
                    error_text = part.text

            if result_image:
                result_image.save(output_path, format="PNG")
                logger.info(f"Transformed photo saved: {output_path}")
                return output_path
            else:
                if error_text:
                    logger.error(f"Model returned text instead of image: {error_text[:200]}")
                else:
                    logger.error("No image returned from transformation")
                return None

        except Exception as e:
            logger.error(f"API error during transformation: {e}")
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
