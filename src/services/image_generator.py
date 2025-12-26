"""Image generation service using Google Gemini/Imagen.

Based on Google Gemini API documentation best practices:
- Images placed FIRST in prompt for better results
- Files API used for images >20MB
- Specific, detailed instructions
- Proper error handling with file status checking
"""

import base64
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from src.config import Config, config as default_config

logger = logging.getLogger(__name__)

# Threshold for using Files API (20MB as per documentation)
FILES_API_THRESHOLD = 20 * 1024 * 1024  # 20MB in bytes


class ImageGenerator:
    """Generate images using Google Imagen API."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config
        self._client = None
        self._uploaded_files = {}  # Cache for uploaded files

    def _get_client(self):
        """Lazy load Gemini client."""
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.config.google_api_key)
        return self._client

    def _get_image_size_bytes(self, img: Image.Image) -> int:
        """Get approximate size of PIL image in bytes."""
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return buffer.tell()

    def _upload_large_image(self, img: Image.Image, display_name: str = "reference") -> Optional[str]:
        """Upload large image using Files API, returns file URI."""
        from google.genai import types
        import tempfile
        import os

        client = self._get_client()

        try:
            # Save to temp file for upload
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img.save(tmp.name, format="PNG")
                tmp_path = tmp.name

            # Upload using Files API
            uploaded_file = client.files.upload(file=tmp_path)
            logger.info(f"Uploaded large image via Files API: {uploaded_file.name}")

            # Clean up temp file
            os.unlink(tmp_path)

            return uploaded_file
        except Exception as e:
            logger.error(f"Failed to upload image via Files API: {e}")
            return None

    def generate_scene_image(
        self,
        prompt: str,
        style_prefix: Optional[str] = None,
        character_description: Optional[str] = None,
        visual_world: Optional[str] = None,
        reference_image: Optional[Image.Image] = None,
        reference_images: Optional[list[Image.Image]] = None,
        aspect_ratio: str = "16:9",
        output_path: Optional[Path] = None,
        image_size: Optional[str] = None,
        show_character: bool = True,
        model: Optional[str] = None,
    ) -> Optional[Image.Image]:
        """
        Generate a scene image using Gemini or Imagen models.

        Default model is gemini-2.5-flash-image (Nano Banana) which supports
        both text-to-image and image editing/transformation.

        Args:
            prompt: The scene description
            style_prefix: Optional style prefix for consistent visual style
            character_description: Optional character description for consistency
            visual_world: Optional visual world/setting for consistency across scenes
            reference_image: Optional single reference image for composition hints (legacy)
            reference_images: Optional list of reference images (character portraits, etc.)
            aspect_ratio: Output aspect ratio (default 16:9 for video)
            output_path: Optional path to save the image
            image_size: Image size for Gemini models ("2K" or "4K")
            show_character: Whether the character appears in this scene (default True)
            model: Optional model override. Gemini models (e.g., "gemini-2.5-flash-image",
                "gemini-3-pro-image-preview") support image editing. Imagen models
                (e.g., "imagen-4.0-generate-001") are text-to-image only.

        Returns:
            Generated PIL Image or None if generation failed
        """
        from google.genai import types

        client = self._get_client()

        # Build the full prompt with strong style enforcement
        full_prompt_parts = []

        # STYLE FIRST - Critical for consistency
        if style_prefix:
            full_prompt_parts.append(f"CRITICAL VISUAL STYLE (maintain exactly): {style_prefix}")

        # Visual world - anchors to the same universe
        if visual_world:
            full_prompt_parts.append(f"Setting/World: {visual_world}")

        # Character consistency - only include if character appears in this scene
        if character_description and show_character:
            full_prompt_parts.append(f"Character (match reference exactly): {character_description}")

        # The actual scene content
        full_prompt_parts.append(f"Scene: {prompt}")

        # Technical specs - just aspect ratio, let the style handle the rest
        full_prompt_parts.append(f"{aspect_ratio} aspect ratio, high detail")

        # Add reference matching instructions if we have reference images
        if reference_images or reference_image:
            # Distinguish between scene reference (environment) and character portraits (faces)
            if reference_image and reference_images:
                full_prompt_parts.append("REFERENCE USAGE: Match character FACES and APPEARANCES exactly from the portraits. Use the scene reference for STYLE and LIGHTING consistency. IMPORTANT: Character POSITIONS and CAMERA ANGLE come from the prompt above, NOT the reference images.")
            elif reference_image:
                full_prompt_parts.append("REFERENCE USAGE: Use scene reference for STYLE and LIGHTING consistency. IMPORTANT: Character POSITIONS and CAMERA ANGLE come from the prompt above, NOT the reference image.")
            else:
                full_prompt_parts.append("REFERENCE USAGE: Match character FACES and APPEARANCES exactly from the portraits. Character positions come from the prompt above.")

        full_prompt = ". ".join(full_prompt_parts)

        try:
            # Use provided model or fall back to config default
            model_name = model or self.config.image.model
            is_gemini_model = "gemini" in model_name.lower()

            if is_gemini_model:
                # Following Gemini API documentation best practices:
                # - Images FIRST for better results
                # - Use Files API for images >20MB
                # - Specific, detailed instructions

                # Gemini 3 Pro supports up to 14 reference images (6 objects + 5 humans + 3 style)
                # Gemini 2.5 Flash supports up to 4 reference images
                is_pro_model = "gemini-3" in model_name.lower() or "pro" in model_name.lower()
                max_ref_images = 14 if is_pro_model else 4

                contents = []

                # Collect all reference images (support both single and list)
                all_ref_images = []
                if reference_images:
                    all_ref_images.extend(reference_images)
                if reference_image is not None and reference_image not in all_ref_images:
                    all_ref_images.append(reference_image)

                # IMAGES FIRST - Per documentation: "placing a single image before
                # the text prompt might lead to better results"
                if all_ref_images:
                    for idx, ref_img in enumerate(all_ref_images[:max_ref_images]):
                        # Check if image is large enough to need Files API
                        img_size = self._get_image_size_bytes(ref_img)
                        if img_size > FILES_API_THRESHOLD:
                            # Use Files API for large images
                            uploaded = self._upload_large_image(ref_img, f"reference_{idx}")
                            if uploaded:
                                contents.append(uploaded)
                            else:
                                # Fallback to inline if upload fails
                                contents.append(ref_img)
                        else:
                            # Small enough for inline data
                            contents.append(ref_img)

                    # Add specific instruction for reference image usage AFTER images
                    if len(all_ref_images) == 1:
                        contents.append(
                            "REFERENCE IMAGE ABOVE: Use this image as a visual reference. "
                            "Maintain the exact character appearance (face, hair, clothing, colors). "
                            "Create a new scene that is visually consistent with this reference."
                        )
                    else:
                        contents.append(
                            f"REFERENCE IMAGES ABOVE ({len(all_ref_images[:max_ref_images])} images): "
                            "These are character portraits. CRITICAL: Maintain the EXACT appearance "
                            "of each character (face shape, hair style/color, clothing, features). "
                            "Include ALL these characters in the new scene with perfect visual consistency."
                        )

                # Add the main scene prompt AFTER images and reference instructions
                contents.append(f"GENERATE THIS SCENE: {full_prompt}")

                # Add specific output requirements
                contents.append(
                    "OUTPUT REQUIREMENTS: Generate a high-quality image matching the scene description. "
                    "Ensure character consistency with any reference images provided. "
                    "Use cinematic composition and professional lighting."
                )

                # Use provided image_size or fall back to config default
                effective_image_size = image_size or self.config.image.image_size

                # Configure generation with image output
                generate_content_config = types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=effective_image_size,
                    ),
                )

                # Log the image generation prompt
                logger.info("=" * 60)
                logger.info(f"GEMINI IMAGE PROMPT (model={model_name}):")
                logger.info("-" * 60)
                for line in full_prompt.split('\n'):
                    logger.info(line)
                logger.info(f"Refs: {len(all_ref_images)}, Size: {effective_image_size}, Ratio: {aspect_ratio}")
                logger.info("=" * 60)

                # Use generate_content (not streaming) for image generation
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=generate_content_config,
                )

                # Extract image from response
                image = None
                for part in response.parts:
                    if part.text is not None:
                        # Model may return text description along with image
                        logger.debug(f"Model text response: {part.text[:100]}...")
                    elif part.inline_data is not None:
                        # Get the generated image
                        image = Image.open(BytesIO(part.inline_data.data))
                        logger.info(f"Generated image: {image.size}")
                        break

            else:
                # Fallback to Imagen API for non-Gemini models
                logger.info("=" * 60)
                logger.info(f"IMAGEN PROMPT (model={model_name}):")
                logger.info("-" * 60)
                for line in full_prompt.split('\n'):
                    logger.info(line)
                logger.info("=" * 60)

                response = client.models.generate_images(
                    model=model_name,
                    prompt=full_prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio=aspect_ratio,
                        safety_filter_level="BLOCK_LOW_AND_ABOVE",
                    ),
                )

                image = None
                if response.generated_images:
                    generated = response.generated_images[0]
                    image_data = base64.b64decode(generated.image.image_bytes)
                    image = Image.open(BytesIO(image_data))

            if image and output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(str(output_path))

            return image

        except Exception as e:
            print(f"Image generation failed: {e}")
            # Try fallback to placeholder image
            return self._create_placeholder_image(prompt, output_path)

    def generate_motion_prompt_from_image(
        self,
        image_path: Path,
    ) -> Optional[str]:
        """
        Analyze an image and generate a motion prompt for animation.

        Uses Gemini's vision capabilities to describe what motion/action
        would be appropriate for animating the scene.

        Args:
            image_path: Path to the scene image

        Returns:
            Motion prompt string, or None if analysis failed
        """
        from google.genai import types

        try:
            client = self._get_client()

            # Load the image
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            # Build the prompt for motion analysis
            prompt_text = """Analyze this image and suggest a short motion prompt for video animation.

Focus on what natural movement would bring this scene to life:
- If there's a person: describe their action (playing guitar, singing, dancing, etc.)
- If it's a landscape: describe ambient motion (wind through trees, waves, clouds moving)
- If there are objects: describe subtle movements (flickering lights, swaying items)

Requirements:
- Keep it SHORT (under 15 words)
- Be SPECIFIC to what's in the image
- Focus on ONE primary motion
- Use present participle verbs (playing, singing, swaying, flowing)

Examples:
- "playing guitar passionately with subtle head movement"
- "wind gently moving through the forest leaves"
- "performer dancing energetically on stage"

Respond with ONLY the motion prompt, nothing else."""

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                        types.Part.from_text(text=prompt_text),
                    ],
                ),
            ]

            # Use the text model for analysis
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
            )

            if response.text:
                # Clean up the response
                motion_prompt = response.text.strip()
                # Remove quotes if present
                motion_prompt = motion_prompt.strip('"\'')
                return motion_prompt

        except Exception as e:
            print(f"Motion prompt generation failed: {e}")

        return None

    def generate_scene_variations(
        self,
        prompt: str,
        num_variations: int = 3,
        style_prefix: Optional[str] = None,
        character_description: Optional[str] = None,
        visual_world: Optional[str] = None,
        output_dir: Optional[Path] = None,
        image_size: Optional[str] = None,
    ) -> list[tuple[Image.Image, Optional[Path]]]:
        """
        Generate multiple variations of a scene image for user selection.

        Args:
            prompt: The scene description
            num_variations: Number of variations to generate (default 3)
            style_prefix: Optional style prefix for consistent visual style
            character_description: Optional character description for consistency
            visual_world: Optional visual world/setting for consistency
            output_dir: Optional directory to save variations
            image_size: Image size for Gemini models ("2K" or "4K")

        Returns:
            List of tuples (PIL Image, optional path) for each variation
        """
        variations = []

        for i in range(num_variations):
            output_path = None
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"variation_{i:02d}.png"

            image = self.generate_scene_image(
                prompt=prompt,
                style_prefix=style_prefix,
                character_description=character_description,
                visual_world=visual_world,
                output_path=output_path,
                image_size=image_size,
            )

            if image:
                variations.append((image, output_path))

        return variations

    def _create_placeholder_image(
        self, prompt: str, output_path: Optional[Path] = None
    ) -> Optional[Image.Image]:
        """Create a placeholder image when generation fails."""
        from PIL import ImageDraw, ImageFont

        # Create a dark gradient background
        width, height = 1920, 1080
        image = Image.new("RGB", (width, height), color=(30, 30, 40))
        draw = ImageDraw.Draw(image)

        # Add gradient effect
        for y in range(height):
            alpha = y / height
            color = (
                int(30 + 20 * alpha),
                int(30 + 10 * alpha),
                int(40 + 30 * alpha),
            )
            draw.line([(0, y), (width, y)], fill=color)

        # Add text
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        except (IOError, OSError):
            font = ImageFont.load_default()

        # Wrap and center the prompt text
        text = f"Scene: {prompt[:100]}..." if len(prompt) > 100 else f"Scene: {prompt}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (width - text_width) // 2
        text_y = height // 2 - 20

        draw.text((text_x, text_y), text, fill=(200, 200, 200), font=font)

        # Add "placeholder" label
        label = "[Placeholder - Image generation unavailable]"
        label_bbox = draw.textbbox((0, 0), label, font=font)
        label_width = label_bbox[2] - label_bbox[0]
        draw.text(
            ((width - label_width) // 2, text_y + 60),
            label,
            fill=(150, 150, 150),
            font=font,
        )

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(str(output_path))

        return image

    def generate_storyboard(
        self,
        scene_prompts: list[str],
        style_prefix: str,
        character_description: str,
        output_dir: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        max_retries: int = 2,
        image_size: Optional[str] = None,
        sequential_mode: bool = False,
        visual_world: Optional[str] = None,
        max_workers: int = 4,
        hero_image: Optional[Image.Image] = None,
        show_character_flags: Optional[list[bool]] = None,
    ) -> list[Path]:
        """
        Generate a series of images for a storyboard.

        Args:
            scene_prompts: List of scene descriptions
            style_prefix: Style prefix for all images
            character_description: Character description for consistency
            output_dir: Directory to save images
            progress_callback: Optional callback for progress updates
            max_retries: Number of retry attempts for failed generations
            image_size: Image size for Gemini models
            sequential_mode: If True, use previous image as reference for consistency
            visual_world: Visual world/setting for consistency across all scenes
            max_workers: Maximum number of parallel image generations (ignored if sequential_mode=True)
            hero_image: Optional reference image for visual consistency
                - In parallel mode: used as reference for ALL scene generations
                - In sequential mode: seeds the FIRST scene (scene 0)
            show_character_flags: Optional list of bools indicating if character appears in each scene

        Returns:
            List of paths to generated images (always same length as scene_prompts)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_paths: list[Optional[Path]] = [None] * len(scene_prompts)
        generated_images: list[Optional[Image.Image]] = [None] * len(scene_prompts)
        failed_indices: list[int] = []

        total_scenes = len(scene_prompts)

        if sequential_mode or max_workers <= 1:
            # Sequential mode: generate one at a time, using previous image as reference
            for i, prompt in enumerate(scene_prompts):
                if progress_callback:
                    mode_str = " (sequential)" if sequential_mode else ""
                    progress_callback(
                        f"Generating scene {i + 1}/{total_scenes}{mode_str}...",
                        (i / total_scenes) * 0.8,
                    )

                output_path = output_dir / f"scene_{i:03d}.png"

                # Get reference image for this scene
                # In sequential mode: use hero_image for scene 0, then previous scene's image
                reference_image = None
                if sequential_mode:
                    if i == 0 and hero_image is not None:
                        # First scene uses hero_image as seed
                        reference_image = hero_image
                    elif i > 0 and generated_images[i - 1] is not None:
                        # Subsequent scenes use previous scene's image
                        reference_image = generated_images[i - 1]

                # Determine if character should appear in this scene
                show_char = True
                if show_character_flags is not None and i < len(show_character_flags):
                    show_char = show_character_flags[i]

                image = self.generate_scene_image(
                    prompt=prompt,
                    style_prefix=style_prefix,
                    character_description=character_description,
                    visual_world=visual_world,
                    reference_image=reference_image,
                    output_path=output_path,
                    image_size=image_size,
                    show_character=show_char,
                )

                if image and output_path.exists():
                    generated_paths[i] = output_path
                    generated_images[i] = image
                else:
                    failed_indices.append(i)
        else:
            # Parallel mode: generate multiple images concurrently
            if progress_callback:
                progress_callback(
                    f"Generating {total_scenes} images in parallel ({max_workers} workers)...",
                    0.05,
                )

            def generate_single_image(index: int, prompt: str, show_char: bool) -> tuple[int, Optional[Image.Image], Optional[Path]]:
                """Generate a single image and return (index, image, path)."""
                output_path = output_dir / f"scene_{index:03d}.png"

                image = self.generate_scene_image(
                    prompt=prompt,
                    style_prefix=style_prefix,
                    character_description=character_description,
                    visual_world=visual_world,
                    reference_image=hero_image,  # Use hero_image for ALL scenes in parallel mode
                    output_path=output_path,
                    image_size=image_size,
                    show_character=show_char,
                )

                if image and output_path.exists():
                    return (index, image, output_path)
                return (index, None, None)

            # Submit all tasks to thread pool
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for i, prompt in enumerate(scene_prompts):
                    # Determine if character should appear in this scene
                    show_char = True
                    if show_character_flags is not None and i < len(show_character_flags):
                        show_char = show_character_flags[i]
                    futures[executor.submit(generate_single_image, i, prompt, show_char)] = i

                # Collect results as they complete and update progress from main thread
                completed_count = 0
                for future in as_completed(futures):
                    idx, image, path = future.result()
                    completed_count += 1

                    # Update progress from main thread (safe for Streamlit)
                    if progress_callback:
                        progress_callback(
                            f"Generated {completed_count}/{total_scenes} images...",
                            (completed_count / total_scenes) * 0.8,
                        )

                    if path:
                        generated_paths[idx] = path
                        generated_images[idx] = image
                    else:
                        failed_indices.append(idx)

        # Retry failed images (always sequential for retries)
        for retry in range(max_retries):
            if not failed_indices:
                break

            if progress_callback:
                progress_callback(
                    f"Retrying {len(failed_indices)} failed images (attempt {retry + 1})...",
                    0.8 + (0.15 * (retry + 1) / max_retries),
                )

            still_failed = []
            for i in failed_indices:
                output_path = output_dir / f"scene_{i:03d}.png"
                image = self.generate_scene_image(
                    prompt=scene_prompts[i],
                    style_prefix=style_prefix,
                    character_description=character_description,
                    visual_world=visual_world,
                    output_path=output_path,
                    image_size=image_size,
                )

                if image and output_path.exists():
                    generated_paths[i] = output_path
                else:
                    still_failed.append(i)

            failed_indices = still_failed

        # For any still-failed images, create placeholders
        for i in failed_indices:
            output_path = output_dir / f"scene_{i:03d}.png"
            placeholder = self._create_placeholder_image(
                scene_prompts[i], output_path
            )
            if placeholder:
                generated_paths[i] = output_path

        if progress_callback:
            success_count = sum(1 for p in generated_paths if p is not None)
            progress_callback(
                f"Generated {success_count}/{total_scenes} images", 1.0
            )

        # Return list with same length as input (None for any failures)
        return generated_paths
