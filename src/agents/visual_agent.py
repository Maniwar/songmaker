"""Visual Agent for scene planning and image prompt generation."""

import json
import logging
import math
from typing import Optional

import anthropic

from src.config import Config, config as default_config
from src.models.schemas import (
    SongConcept,
    Transcript,
    Scene,
    KenBurnsEffect,
    Word,
)

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a creative director for music videos. Your role is to create compelling visual scenes that match the song's lyrics, mood, and narrative arc.

When planning scenes:
1. Analyze the lyrics for key imagery, emotions, and story beats
2. Create visual prompts that translate lyrics into cinematic scenes
3. Maintain visual consistency through the video
4. Consider the emotional journey from start to finish
5. Match scene mood to the musical energy at that point

For each scene, provide:
- A detailed visual prompt (for image generation)
- The mood/emotion of the scene
- A Ken Burns effect - IMPORTANT: Use VARIED effects across scenes for visual interest:
  - zoom_in: For intimate moments, drawing attention inward
  - zoom_out: For reveals, establishing shots, or resolution
  - pan_left/pan_right: For movement, transitions, or following action
  - pan_up: For uplifting moments, hope, looking to the sky
  - pan_down: For grounding, introspection, or somber moments

Consider the narrative arc:
- Introduction: Establish setting (use zoom_out or pan effects)
- Build-up: Increase visual intensity (mix of zoom_in and pans)
- Climax: Most dramatic imagery (zoom_in for intensity)
- Resolution: Bring visual closure (zoom_out or pan effects)

Always maintain character consistency by referencing the same character description throughout.
IMPORTANT: Vary the Ken Burns effects across scenes - don't use the same effect for all scenes!"""


class VisualAgent:
    """Agent for planning visual scenes and generating image prompts."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config
        self._client = None

    def _get_client(self) -> anthropic.Anthropic:
        """Lazy load Anthropic client."""
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
        return self._client

    def calculate_scene_boundaries(
        self,
        duration: float,
        transcript: Transcript,
        min_scene_duration: float = 4.0,
        max_scene_duration: float = 12.0,
        target_scenes_per_minute: float = 4.0,
    ) -> list[tuple[float, float, list[Word]]]:
        """
        Calculate optimal scene boundaries based on song duration.

        Args:
            duration: Total song duration in seconds
            transcript: Transcript with word timestamps
            min_scene_duration: Minimum scene length
            max_scene_duration: Maximum scene length
            target_scenes_per_minute: Target number of scenes per minute

        Returns:
            List of (start_time, end_time, words) tuples
        """
        # Calculate target scene count
        target_count = max(
            math.ceil(duration / max_scene_duration),
            min(
                math.ceil(duration * target_scenes_per_minute / 60),
                math.floor(duration / min_scene_duration),
            ),
        )

        # Ensure at least 1 scene
        target_count = max(1, target_count)

        # Find natural break points (gaps between words)
        words = transcript.all_words
        break_points = self._find_break_points(words)

        # Distribute scenes
        scene_duration = duration / target_count
        scenes = []
        current_time = 0.0

        for i in range(target_count):
            ideal_end = current_time + scene_duration

            # Find nearest break point
            actual_end = self._find_nearest_break(break_points, ideal_end, tolerance=2.0)
            if actual_end is None or actual_end > duration:
                actual_end = min(ideal_end, duration)

            # Get words in this time range
            scene_words = [
                w for w in words if w.start >= current_time and w.end <= actual_end
            ]

            scenes.append((current_time, actual_end, scene_words))
            current_time = actual_end

        # Ensure last scene extends to song end
        if scenes and scenes[-1][1] < duration:
            start, _, words = scenes[-1]
            # Get any remaining words
            remaining_words = [w for w in transcript.all_words if w.start >= start]
            scenes[-1] = (start, duration, remaining_words)

        return scenes

    def _find_break_points(self, words: list[Word]) -> list[float]:
        """Find natural break points (gaps) between words."""
        break_points = []
        for i in range(len(words) - 1):
            gap = words[i + 1].start - words[i].end
            if gap > 0.3:  # Significant pause
                break_points.append(words[i].end + gap / 2)
        return break_points

    def _find_nearest_break(
        self,
        break_points: list[float],
        target: float,
        tolerance: float,
    ) -> Optional[float]:
        """Find the nearest break point within tolerance."""
        if not break_points:
            return None

        nearest = min(break_points, key=lambda x: abs(x - target))
        if abs(nearest - target) <= tolerance:
            return nearest
        return None

    def _suggest_effect_distribution(self, num_scenes: int) -> list[str]:
        """Suggest varied Ken Burns effects for visual interest."""
        effects = ["zoom_out", "zoom_in", "pan_right", "pan_left", "pan_up", "pan_down"]

        if num_scenes <= 1:
            return ["zoom_out"]

        # Create a varied pattern
        suggestions = []
        # Start with establishing shot (zoom out)
        suggestions.append("zoom_out")

        for i in range(1, num_scenes - 1):
            # Cycle through effects but prefer zoom_in for middle scenes
            if i % 4 == 0:
                suggestions.append("pan_right")
            elif i % 4 == 1:
                suggestions.append("zoom_in")
            elif i % 4 == 2:
                suggestions.append("pan_left")
            else:
                suggestions.append("zoom_in")

        # End with resolution (zoom out)
        if num_scenes > 1:
            suggestions.append("zoom_out")

        return suggestions[:num_scenes]

    def generate_scene_prompts(
        self,
        concept: SongConcept,
        scene_boundaries: list[tuple[float, float, list[Word]]],
        full_lyrics: str,
    ) -> list[Scene]:
        """
        Generate visual prompts for each scene.

        Args:
            concept: The song concept with visual style info
            scene_boundaries: List of (start, end, words) from calculate_scene_boundaries
            full_lyrics: Complete lyrics text

        Returns:
            List of Scene objects with visual prompts
        """
        client = self._get_client()

        # Build scene descriptions from words
        scene_descriptions = []
        all_scene_lyrics = []
        for i, (start, end, words) in enumerate(scene_boundaries):
            lyrics_segment = " ".join(w.word for w in words) if words else "[Instrumental]"
            all_scene_lyrics.append(lyrics_segment)
            scene_descriptions.append(
                f"Scene {i + 1} ({start:.1f}s - {end:.1f}s): {lyrics_segment}"
            )

        # Use transcript lyrics if full_lyrics is placeholder text
        effective_lyrics = full_lyrics
        if not full_lyrics or "[Lyrics" in full_lyrics or len(full_lyrics) < 50:
            effective_lyrics = "\n".join(all_scene_lyrics)

        num_scenes = len(scene_boundaries)
        # Suggest effect distribution for variety
        effect_hints = self._suggest_effect_distribution(num_scenes)

        prompt = f"""Create visual prompts for a music video with these scenes:

Song Concept:
- Genre: {concept.genre}
- Mood: {concept.mood}
- Themes: {', '.join(concept.themes)}
- Visual Style: {concept.visual_style or 'cinematic, dramatic lighting'}
- Character: {concept.character_description or 'No specific character'}

Lyrics from each scene:
{effective_lyrics}

Scenes to create (with suggested camera effects):
{chr(10).join(f"{desc} [suggested: {effect_hints[i]}]" for i, desc in enumerate(scene_descriptions))}

For each scene, provide:
1. A detailed visual prompt (2-3 sentences describing what's shown in the image)
   - The prompt should MATCH the specific lyrics for that scene's timestamp
   - Include visual elements that represent the words being sung
2. The mood of this specific scene
3. A Ken Burns effect: zoom_in, zoom_out, pan_left, pan_right, pan_up, pan_down
   - IMPORTANT: Use VARIED effects - don't use the same effect for all scenes!

Respond in this exact JSON format:
{{
    "scenes": [
        {{
            "visual_prompt": "Detailed description for image generation",
            "mood": "emotional tone of this scene",
            "effect": "zoom_in"
        }}
    ]
}}

Create exactly {num_scenes} scenes. Only respond with the JSON."""

        try:
            logger.info(f"Calling Claude API to generate {num_scenes} scene prompts...")
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.content[0].text
            logger.info(f"Claude API response received: {len(response_text)} chars")

            # Try to extract JSON from the response
            # Sometimes the response has extra text before/after the JSON
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text

            data = json.loads(json_text)
            logger.info(f"JSON parsed successfully, got {len(data.get('scenes', []))} scenes")

            scenes = []
            for i, (start, end, words) in enumerate(scene_boundaries):
                scene_data = data["scenes"][i] if i < len(data["scenes"]) else {}

                effect_str = scene_data.get("effect", effect_hints[i] if i < len(effect_hints) else "zoom_in").lower()
                try:
                    effect = KenBurnsEffect(effect_str)
                except ValueError:
                    # Use suggested effect from effect_hints as fallback
                    effect = KenBurnsEffect(effect_hints[i]) if i < len(effect_hints) else KenBurnsEffect.ZOOM_IN

                scenes.append(
                    Scene(
                        index=i,
                        start_time=start,
                        end_time=end,
                        visual_prompt=scene_data.get(
                            "visual_prompt", f"Scene {i + 1}"
                        ),
                        mood=scene_data.get("mood", concept.mood),
                        effect=effect,
                        words=words,
                    )
                )

            return scenes

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Response text was: {response_text[:500] if 'response_text' in dir() else 'N/A'}...")
        except (KeyError, IndexError) as e:
            logger.error(f"Data extraction failed: {e}")
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error generating scene prompts: {e}")

        # Fallback: create scenes using the actual transcript lyrics
        logger.warning("Falling back to transcript-based scene generation")
        return self._create_fallback_scenes(
            concept=concept,
            scene_boundaries=scene_boundaries,
            all_scene_lyrics=all_scene_lyrics,
            effect_hints=effect_hints,
        )

    def _create_fallback_scenes(
        self,
        concept: SongConcept,
        scene_boundaries: list[tuple[float, float, list[Word]]],
        all_scene_lyrics: list[str],
        effect_hints: list[str],
    ) -> list[Scene]:
        """
        Create fallback scenes when Claude API fails.
        Uses transcript lyrics and varied effects instead of generic prompts.
        """
        style = concept.visual_style or "cinematic, dramatic lighting"
        character = concept.character_description or ""

        scenes = []
        for i, (start, end, words) in enumerate(scene_boundaries):
            # Get the lyrics for this scene
            lyrics = all_scene_lyrics[i] if i < len(all_scene_lyrics) else ""

            # Create a prompt based on the actual lyrics
            if lyrics and lyrics != "[Instrumental]":
                # Build a visual prompt from the lyrics
                prompt = f"{style}. Scene depicting: {lyrics}"
                if character:
                    prompt = f"{character} - {prompt}"
            else:
                # Instrumental section
                prompt = f"{style}. {concept.mood} instrumental moment, abstract visual"

            # Get the varied effect
            effect_str = effect_hints[i] if i < len(effect_hints) else "zoom_in"
            try:
                effect = KenBurnsEffect(effect_str)
            except ValueError:
                effect = KenBurnsEffect.ZOOM_IN

            scenes.append(
                Scene(
                    index=i,
                    start_time=start,
                    end_time=end,
                    visual_prompt=prompt,
                    mood=concept.mood,
                    effect=effect,
                    words=words,
                )
            )

        return scenes

    def plan_video(
        self,
        concept: SongConcept,
        transcript: Transcript,
        full_lyrics: str,
    ) -> list[Scene]:
        """
        Plan complete video with scenes.

        Args:
            concept: Song concept
            transcript: Audio transcript with word timestamps
            full_lyrics: Complete lyrics text

        Returns:
            List of Scene objects ready for image generation
        """
        # Calculate scene boundaries
        boundaries = self.calculate_scene_boundaries(
            duration=transcript.duration,
            transcript=transcript,
        )

        # Generate prompts for each scene
        scenes = self.generate_scene_prompts(
            concept=concept,
            scene_boundaries=boundaries,
            full_lyrics=full_lyrics,
        )

        return scenes
