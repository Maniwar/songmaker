"""Image generation service using Google Imagen."""

import base64
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from src.config import Config, config as default_config


class ImageGenerator:
    """Generate images using Google Imagen API."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config
        self._client = None

    def _get_client(self):
        """Lazy load Gemini client."""
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.config.google_api_key)
        return self._client

    def generate_scene_image(
        self,
        prompt: str,
        style_prefix: Optional[str] = None,
        character_description: Optional[str] = None,
        visual_world: Optional[str] = None,
        reference_image: Optional[Image.Image] = None,
        aspect_ratio: str = "16:9",
        output_path: Optional[Path] = None,
        image_size: Optional[str] = None,
    ) -> Optional[Image.Image]:
        """
        Generate a scene image using Nano Banana Pro (Gemini 3 Pro Image).

        Args:
            prompt: The scene description
            style_prefix: Optional style prefix for consistent visual style
            character_description: Optional character description for consistency
            visual_world: Optional visual world/setting for consistency across scenes
            reference_image: Optional reference image for composition hints
            aspect_ratio: Output aspect ratio (default 16:9 for video)
            output_path: Optional path to save the image
            image_size: Image size for Gemini models ("2K" or "4K")

        Returns:
            Generated PIL Image or None if generation failed
        """
        from google.genai import types

        client = self._get_client()

        # Build the full prompt with cinematography style guidance
        full_prompt_parts = []

        # Visual world comes FIRST - anchors everything to the same universe
        if visual_world:
            full_prompt_parts.append(f"Setting/World: {visual_world}")

        # Cinematography style - sets the visual language
        if style_prefix:
            full_prompt_parts.append(f"Cinematography: {style_prefix}")

        # Character consistency
        if character_description:
            full_prompt_parts.append(f"Character: {character_description}")

        # The actual scene content
        full_prompt_parts.append(f"Scene: {prompt}")

        # Technical specs - just aspect ratio, let the style handle the rest
        full_prompt_parts.append(f"{aspect_ratio} aspect ratio, high detail")

        full_prompt = ". ".join(full_prompt_parts)

        try:
            model_name = self.config.image.model
            is_gemini_model = "gemini" in model_name.lower()

            if is_gemini_model:
                # Build contents - can include reference images for consistency
                contents_parts = []

                # Add reference image first if provided (for sequential/consistency mode)
                if reference_image is not None:
                    # Convert PIL Image to bytes
                    img_buffer = BytesIO()
                    reference_image.save(img_buffer, format="PNG")
                    img_bytes = img_buffer.getvalue()
                    contents_parts.append(
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                    )
                    # Add instruction to use as reference
                    contents_parts.append(
                        types.Part.from_text(
                            text="Use the above image as a visual reference for style, "
                            "character appearance, and color palette. Create a new scene "
                            "that maintains visual consistency with the reference. "
                        )
                    )

                # Add the main prompt
                contents_parts.append(types.Part.from_text(text=full_prompt))

                contents = [
                    types.Content(role="user", parts=contents_parts),
                ]

                # Use provided image_size or fall back to config default
                effective_image_size = image_size or self.config.image.image_size

                generate_content_config = types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                    image_config=types.ImageConfig(
                        image_size=effective_image_size,
                    ),
                )

                image = None
                for chunk in client.models.generate_content_stream(
                    model=model_name,
                    contents=contents,
                    config=generate_content_config,
                ):
                    if (
                        chunk.candidates is None
                        or chunk.candidates[0].content is None
                        or chunk.candidates[0].content.parts is None
                    ):
                        continue

                    for part in chunk.candidates[0].content.parts:
                        if part.inline_data and part.inline_data.data:
                            # inline_data.data is already bytes, not base64
                            image = Image.open(BytesIO(part.inline_data.data))
                            break

                    if image:
                        break

            else:
                # Fallback to Imagen API for non-Gemini models
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

        Returns:
            List of paths to generated images (always same length as scene_prompts)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_paths: list[Optional[Path]] = [None] * len(scene_prompts)
        generated_images: list[Optional[Image.Image]] = [None] * len(scene_prompts)
        failed_indices: list[int] = []

        total_scenes = len(scene_prompts)

        # First pass: try to generate all images
        for i, prompt in enumerate(scene_prompts):
            if progress_callback:
                mode_str = " (sequential)" if sequential_mode else ""
                progress_callback(
                    f"Generating scene {i + 1}/{total_scenes}{mode_str}...",
                    (i / total_scenes) * 0.8,
                )

            output_path = output_dir / f"scene_{i:03d}.png"

            # Get reference image from previous scene if in sequential mode
            reference_image = None
            if sequential_mode and i > 0 and generated_images[i - 1] is not None:
                reference_image = generated_images[i - 1]

            image = self.generate_scene_image(
                prompt=prompt,
                style_prefix=style_prefix,
                character_description=character_description,
                visual_world=visual_world,
                reference_image=reference_image,
                output_path=output_path,
                image_size=image_size,
            )

            if image and output_path.exists():
                generated_paths[i] = output_path
                generated_images[i] = image  # Store for sequential mode
            else:
                failed_indices.append(i)

        # Retry failed images
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
