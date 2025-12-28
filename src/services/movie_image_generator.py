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

    def __init__(self, style: str = ""):
        """Initialize the movie image generator.

        Args:
            style: Base visual style for all images (from project config)
        """
        self.style = style
        self.image_generator = ImageGenerator()

    def build_scene_prompt(
        self,
        scene: MovieScene,
        script: Script,
        include_characters: bool = True,
        has_reference_image: bool = False,
    ) -> str:
        """Build a comprehensive prompt for a scene with character descriptions.

        Args:
            scene: The scene to generate an image for
            script: The full script (for character lookup)
            include_characters: Whether to include character descriptions
            has_reference_image: If True, minimize setting descriptions (preserve from reference)

        Returns:
            Complete prompt string for image generation
        """
        parts = []
        direction = scene.direction
        visual_style = self.style  # Use config style passed at init, not script's AI-generated style

        # Detect if photorealistic style
        style_lower = visual_style.lower()
        is_photorealistic = any(kw in style_lower for kw in ["photorealistic", "realistic", "photo", "cinematic"])

        # Photorealistic quality keywords for Gemini Native Image Generation
        if is_photorealistic:
            photo_quality = "RAW photograph, natural skin texture with visible pores, subsurface scattering, shot on Sony A7IV with 85mm f/1.4 lens, shallow depth of field, 8K resolution, film grain, hyperrealistic, photojournalistic"
        else:
            photo_quality = "highly detailed, professional quality"

        if has_reference_image:
            # REFERENCE MODE - preserve environment from reference image but include setting details
            if is_photorealistic:
                parts.append(f"RAW photograph, {visual_style}")
            else:
                parts.append(f"VISUAL STYLE: {visual_style}")

            # Include setting description to ensure environment consistency
            parts.append(f"{direction.camera} of {direction.setting}")

            # Include character descriptions for consistency
            if include_characters and direction.visible_characters:
                char_descriptions = []
                for char_id in direction.visible_characters:
                    character = script.get_character(char_id)
                    if character:
                        char_descriptions.append(f"{character.name}: {character.description}")
                if char_descriptions:
                    parts.append("Characters: " + "; ".join(char_descriptions))

            parts.append(f"Mood: {direction.mood}")
            if is_photorealistic:
                parts.append("natural skin texture, subsurface scattering, soft diffused lighting, 8K resolution, film grain")

            # Explicit environment preservation instructions
            parts.append("CRITICAL ENVIRONMENT PRESERVATION: The reference image shows the EXACT room/location. You MUST keep ALL furniture, props, decorations, and environmental details EXACTLY as shown in the reference. Do NOT add, remove, or change any furniture or objects in the room.")
            parts.append("REFERENCE USAGE: Match character FACES from portraits. Preserve ENVIRONMENT/FURNITURE exactly from scene reference. Character POSITIONS and CAMERA ANGLE come from this prompt.")
        else:
            # FULL PROMPT - no reference, describe everything
            if is_photorealistic:
                parts.append(f"RAW photograph, unedited photo, {visual_style}")
            else:
                parts.append(f"VISUAL STYLE: {visual_style}")
            parts.append(f"{direction.camera} of {direction.setting}")

            if include_characters and direction.visible_characters:
                char_descriptions = []
                for char_id in direction.visible_characters:
                    character = script.get_character(char_id)
                    if character:
                        char_descriptions.append(f"{character.name}: {character.description}")
                if char_descriptions:
                    parts.append("Characters present: " + "; ".join(char_descriptions))

            if direction.lighting:
                if is_photorealistic:
                    parts.append(f"Lighting: {direction.lighting}, soft diffused natural light, volumetric lighting")
                else:
                    parts.append(f"Lighting: {direction.lighting}")

            parts.append(f"Mood: {direction.mood}")

            if script.world_description:
                parts.append(f"Setting style: {script.world_description}")

            # Add quality keywords based on style
            if is_photorealistic:
                parts.append("natural skin texture with visible pores, subsurface scattering on skin, shot on Sony A7IV with 85mm f/1.4 lens, shallow depth of field, creamy bokeh, 8K resolution, film grain, hyperrealistic, photojournalistic, candid moment aesthetic")
            else:
                parts.append("highly detailed, professional cinematography")

        # Add negative prompt / avoid keywords (Gemini embeds these in prompt)
        image_negative_prompt = getattr(scene, 'image_negative_prompt', None)
        if image_negative_prompt:
            parts.append(f"AVOID: {image_negative_prompt}")
        elif is_photorealistic:
            # Default anti-CGI keywords for photorealistic
            parts.append("AVOID: CGI, cartoon, anime, 3D render, digital art, stylized, artificial, plastic skin, video game, smooth skin, airbrushed")

        return ". ".join(parts)

    def generate_scene_image(
        self,
        scene: MovieScene,
        script: Script,
        output_dir: Path,
        reference_image: Optional[Path] = None,
        character_portraits: Optional[list[Path]] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        model: Optional[str] = None,
        image_size: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
    ) -> Optional[Path]:
        """Generate an image for a movie scene.

        Args:
            scene: The scene to generate
            script: Full script for character lookup
            output_dir: Directory to save images
            reference_image: Optional reference image for style consistency (e.g., previous scene)
            character_portraits: Optional list of character portrait paths for all visible characters
            progress_callback: Optional progress callback
            model: Optional image model override (e.g., "gemini-3-pro-image-preview")
            image_size: Optional image size override (e.g., "2K", "4K")
            aspect_ratio: Optional aspect ratio override (e.g., "16:9", "1:1")

        Returns:
            Path to generated image, or None if failed
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check for location reference image in scene direction
        location_ref_path = getattr(scene.direction, 'location_reference_path', None)
        if location_ref_path and Path(location_ref_path).exists():
            # Use location reference as the primary reference image (environment/location)
            location_ref = Path(location_ref_path)
            logger.info(f"Using scene location reference: {location_ref.name}")
            # If we also have a previous scene reference, we'll prioritize location ref
            reference_image = location_ref

        # Determine if we have reference images (affects prompt building)
        has_refs = bool(reference_image) or bool(character_portraits)
        has_location_ref = location_ref_path and Path(location_ref_path).exists()

        # Use custom visual prompt if provided, otherwise build from scene data
        if scene.visual_prompt and scene.visual_prompt.strip():
            prompt = scene.visual_prompt
            # For custom prompts, respect the user's camera angle and composition
            # Only use reference images for face/appearance matching, NOT for composition
            if has_refs:
                if has_location_ref:
                    prompt = prompt + ". CRITICAL: Use the location reference image for the ENVIRONMENT/ROOM/SETTING. Match character FACES from portraits. Character POSITIONS and CAMERA ANGLE come from this prompt."
                else:
                    prompt = prompt + ". Match character FACES and APPEARANCE from reference portraits, but use THIS PROMPT's camera angle and character positions."
            logger.info(f"Using custom visual prompt for scene {scene.index}")
        else:
            prompt = self.build_scene_prompt(scene, script, has_reference_image=has_refs)

        if progress_callback:
            progress_callback(f"Generating image for scene {scene.index + 1}...", 0.0)

        logger.info("=" * 60)
        logger.info(f"IMAGE PROMPT (Scene {scene.index}, refs={has_refs}):")
        logger.info("-" * 60)
        for line in prompt.split('\n'):
            logger.info(line)
        logger.info("=" * 60)

        # Generate the image in per-scene images subdirectory for variant management
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scene_images_dir = output_dir / f"scene_{scene.index:03d}" / "images"
        scene_images_dir.mkdir(parents=True, exist_ok=True)
        output_path = scene_images_dir / f"take_{timestamp}.png"

        try:
            from PIL import Image as PILImage

            # Load all character portraits as reference images
            ref_images = []
            if character_portraits:
                for portrait_path in character_portraits:
                    if portrait_path and portrait_path.exists():
                        try:
                            ref_images.append(PILImage.open(portrait_path))
                            logger.info(f"Added character portrait: {portrait_path.name}")
                        except Exception as e:
                            logger.warning(f"Could not load portrait {portrait_path}: {e}")

            # Add style reference image (e.g., previous scene) if provided
            ref_img = None
            if reference_image and reference_image.exists():
                ref_img = PILImage.open(reference_image)

            result = self.image_generator.generate_scene_image(
                prompt=prompt,
                style_prefix=self.style,  # Use config style passed at init, not script's AI-generated style
                visual_world=script.world_description,
                reference_image=ref_img,
                reference_images=ref_images if ref_images else None,
                output_path=output_path,
                model=model,
                image_size=image_size,
                aspect_ratio=aspect_ratio or "16:9",
            )

            if result and output_path.exists():
                logger.info(f"Generated scene image: {output_path}")
                return output_path
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
        transform_prompt: Optional[str] = None,
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
            transform_prompt: Optional custom prompt for photo transformation

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
                custom_prompt=transform_prompt,
            )
        else:
            # Generate from scratch - use custom prompt if provided
            if transform_prompt:
                prompt = transform_prompt
            else:
                # Build detailed prompt based on style
                style_lower = effective_style.lower()
                is_photorealistic = any(kw in style_lower for kw in ["photorealistic", "realistic", "photo", "cinematic"])
                is_anime = any(kw in style_lower for kw in ["anime", "manga", "cartoon"])

                if is_photorealistic:
                    # Photorealistic portrait prompt
                    prompt = f"""RAW photograph portrait of {character.description}.

Shot on Sony A7IV with 85mm f/1.4 lens, shallow depth of field, creamy bokeh.
Natural skin texture with visible pores, subsurface scattering.
Professional studio lighting, soft diffused light, natural shadows.
8K resolution, film grain, hyperrealistic, photojournalistic.
Centered composition, neutral background, professional headshot.

AVOID: CGI, cartoon, anime, 3D render, digital art, stylized, artificial, plastic skin, airbrushed, smooth skin."""
                elif is_anime:
                    # Anime/manga portrait prompt
                    prompt = f"""High-quality anime illustration of {character.description}.

{effective_style} style, clean linework, vibrant colors.
Professional anime character portrait, centered composition.
Detailed eyes, expressive face, polished illustration.
Neutral background, studio lighting style.

AVOID: photorealistic, real photo, live action, western cartoon style."""
                else:
                    # General artistic style
                    prompt = f"""{effective_style} style character portrait of {character.description}.

Professional {effective_style} illustration, highly detailed.
Centered composition, neutral expression, professional lighting.
Consistent art style throughout, clean execution.

Art style: {effective_style}."""

            logger.info(f"Generating character portrait with prompt: {prompt[:200]}...")

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
        custom_prompt: Optional[str] = None,
    ) -> Optional[Path]:
        """Transform a source photo into a character portrait using Gemini.

        Uses gemini-2.5-flash-image or gemini-3-pro-image-preview for image editing.
        Imagen models don't support image input.

        Args:
            custom_prompt: Optional custom transformation prompt. If not provided,
                          a default prompt will be generated.
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
        # gemini-2.5-flash-image is optimized for image editing (max 2K)
        # gemini-3-pro-image-preview is best for complex edits up to 4K
        if model and "imagen" in model.lower():
            logger.warning("Imagen doesn't support image editing, using gemini-2.5-flash-image")
            model_name = "gemini-2.5-flash-image"
        elif model and "gemini" in model.lower():
            model_name = model
        else:
            # Default to gemini-2.5-flash-image for fast image editing
            model_name = "gemini-2.5-flash-image"

        # gemini-2.5-flash-image doesn't support image_size parameter - omit it
        # gemini-3-pro-image-preview supports 2K, 4K
        is_flash_model = "flash" in model_name.lower() or "2.5" in model_name
        if is_flash_model:
            effective_size = None  # Let the API use its default
            logger.info(f"{model_name} doesn't support image_size parameter, using default")
        else:
            effective_size = image_size

        # Use custom prompt if provided, otherwise build default
        if custom_prompt:
            transform_prompt = custom_prompt
        else:
            # Build transformation prompt following Google's best practices
            transform_prompt = f"""Transform this photograph into a {style} style character portrait.

KEEP the person's face shape, eye shape, and basic facial features recognizable.
APPLY this character description: {character.description}
CHANGE the art style to: {style}
Make it a professional quality portrait with good composition and lighting."""

        logger.info(f"Transforming photo with {model_name}, size={effective_size}")
        logger.info(f"Character: {character.name}")
        logger.info(f"Style: {style}")
        logger.info(f"Transform prompt: {transform_prompt}")

        try:
            # Use the simpler format: contents=[prompt, image]
            # Build image_config - only include image_size if supported
            if effective_size:
                image_config = types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=effective_size,
                )
            else:
                image_config = types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                )

            response = client.models.generate_content(
                model=model_name,
                contents=[transform_prompt, source_image],
                config=types.GenerateContentConfig(
                    response_modalities=['IMAGE', 'TEXT'],
                    image_config=image_config,
                ),
            )

            # Extract image from response
            result_image = None
            error_text = None

            # Log response structure for debugging
            try:
                parts = response.parts if response.parts else []
                logger.info(f"Response has {len(parts)} parts")
            except Exception as e:
                logger.error(f"Error accessing response.parts: {e}")
                # Try to access response in different ways
                logger.info(f"Response type: {type(response)}")
                logger.info(f"Response dir: {[a for a in dir(response) if not a.startswith('_')]}")
                if hasattr(response, 'candidates') and response.candidates:
                    logger.info(f"Candidates: {len(response.candidates)}")
                    if response.candidates[0].content and response.candidates[0].content.parts:
                        parts = response.candidates[0].content.parts
                        logger.info(f"Got {len(parts)} parts from candidates")
                    else:
                        parts = []
                else:
                    parts = []

            for i, part in enumerate(parts):
                logger.debug(f"Part {i}: inline_data={part.inline_data is not None}, text={part.text is not None if hasattr(part, 'text') else 'N/A'}")
                if part.inline_data is not None:
                    # Use inline_data.data to get PIL Image (more reliable)
                    if part.inline_data.data:
                        result_image = PILImage.open(BytesIO(part.inline_data.data))
                        logger.info(f"Got image via inline_data.data from part {i}")
                        break
                    # Fallback to as_image() - returns google genai Image, not PIL
                    elif hasattr(part, 'as_image') and callable(part.as_image):
                        try:
                            genai_image = part.as_image()
                            # Google's Image.save() doesn't take format param, just filename
                            genai_image.save(str(output_path))
                            logger.info(f"Saved image via as_image() from part {i}")
                            return output_path  # Already saved, return directly
                        except Exception as e:
                            logger.warning(f"as_image() failed: {e}")
                elif part.text:
                    error_text = part.text
                    logger.info(f"Part {i} has text: {part.text[:100]}...")

            if result_image:
                result_image.save(output_path, format="PNG")
                logger.info(f"Transformed photo saved: {output_path}")
                return output_path
            else:
                if error_text:
                    logger.error(f"Model returned text instead of image: {error_text[:500]}")
                else:
                    # Log more details about why no image was found
                    logger.error(f"No image returned from transformation. Response candidates: {len(response.candidates) if hasattr(response, 'candidates') and response.candidates else 0}")
                    if hasattr(response, 'prompt_feedback'):
                        logger.error(f"Prompt feedback: {response.prompt_feedback}")
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
    style: str = "",
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
        style: Visual style (from project config)

    Returns:
        Complete prompt string
    """
    # Get visible characters
    visible_chars = [c for c in characters if c.id in visible_char_ids]

    parts = [f"{camera_angle} of {scene_setting}"]

    # Add each visible character with full description
    for char in visible_chars:
        parts.append(f"{char.name} ({char.description})")

    parts.append(f"Mood: {mood}")
    if style:
        parts.append(f"Style: {style}")
    parts.append("highly detailed, consistent character appearance, professional cinematography")

    return ". ".join(parts)
