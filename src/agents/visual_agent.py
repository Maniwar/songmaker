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

CRITICAL: VISUAL WORLD CONSISTENCY
- Every scene MUST exist in the SAME visual world/setting
- If the world is "medieval fantasy", ALL scenes must be medieval fantasy
- Never mix different time periods, universes, or settings
- Reference the visual world description in EVERY scene prompt

When planning scenes:
1. Analyze the lyrics for key imagery, emotions, and story beats
2. Create visual prompts that translate lyrics into cinematic scenes
3. MAINTAIN VISUAL CONSISTENCY - every scene must feel like the same universe
4. Consider the emotional journey from start to finish
5. Match scene mood to the musical energy at that point

For each scene, provide:
- A detailed visual prompt (for image generation) - MUST include visual world context
- A motion prompt (for animation) - describes the CHARACTER'S PHYSICAL ACTION/MOVEMENT
- The mood/emotion of the scene
- A Ken Burns effect - IMPORTANT: Use VARIED effects across scenes for visual interest:
  - zoom_in: For intimate moments, drawing attention inward
  - zoom_out: For reveals, establishing shots, or resolution
  - pan_left/pan_right: For movement, transitions, or following action
  - pan_up: For uplifting moments, hope, looking to the sky
  - pan_down: For grounding, introspection, or somber moments

MOTION PROMPTS (for animation):
Motion prompts describe what the character DOES - their physical action and movement.
Keep motion prompts SHORT (5-15 words) and focused on ACTION, not appearance.
Examples:
- "strumming guitar passionately while swaying to the beat"
- "walking slowly through rain, head bowed in sorrow"
- "dancing energetically with arms raised"
- "sitting at piano, fingers moving gracefully across keys"
- "standing still, wind gently blowing through hair"

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
        min_scene_duration: float = 3.0,
        max_scene_duration: float = 15.0,
        target_scenes_per_minute: float = 4.0,
    ) -> list[tuple[float, float, list[Word]]]:
        """
        Calculate optimal scene boundaries using intelligent multi-pass analysis.

        This algorithm prioritizes:
        1. Lyrical coherence - keeping phrases/sentences together
        2. Natural pauses - using silence as scene transitions
        3. Variable duration - allowing scenes to breathe based on content
        4. Instrumental awareness - proper handling of non-vocal sections

        Args:
            duration: Total song duration in seconds
            transcript: Transcript with word timestamps
            min_scene_duration: Minimum scene length (default 3s)
            max_scene_duration: Maximum scene length (default 15s)
            target_scenes_per_minute: Target scenes per minute (default 4)

        Returns:
            List of (start_time, end_time, words) tuples
        """
        words = transcript.all_words
        if not words:
            # No words - single instrumental scene
            return [(0.0, duration, [])]

        # Pass 1: Analyze gaps and classify them by significance
        gap_analysis = self._analyze_gaps(words, duration)

        # Pass 2: Identify instrumental sections (long gaps with no words)
        instrumental_sections = self._find_instrumental_sections(words, duration)

        # Pass 3: Group words into logical phrases
        phrases = self._group_into_phrases(words, gap_analysis)

        # Pass 4: Build scenes from phrases, respecting constraints
        scenes = self._build_scenes_from_phrases(
            phrases=phrases,
            instrumental_sections=instrumental_sections,
            gap_analysis=gap_analysis,
            duration=duration,
            min_duration=min_scene_duration,
            max_duration=max_scene_duration,
            target_per_minute=target_scenes_per_minute,
        )

        # Pass 5: Final validation and adjustment
        scenes = self._validate_and_adjust_scenes(scenes, duration, min_scene_duration)

        logger.info(f"Created {len(scenes)} scenes with intelligent boundaries")
        for i, (start, end, scene_words) in enumerate(scenes):
            word_preview = " ".join(w.word for w in scene_words[:5])
            if len(scene_words) > 5:
                word_preview += "..."
            logger.debug(f"  Scene {i+1}: {start:.1f}s-{end:.1f}s ({end-start:.1f}s) - {len(scene_words)} words: {word_preview}")

        return scenes

    def _analyze_gaps(self, words: list[Word], duration: float) -> dict:
        """
        Analyze gaps between words and classify them by significance.

        Gap classifications:
        - micro (<0.3s): Normal speech pause, not a boundary
        - phrase (0.3-0.8s): Phrase boundary, weak scene break candidate
        - sentence (0.8-1.5s): Sentence/line boundary, good scene break
        - section (1.5-3.0s): Section boundary, strong scene break
        - major (>3.0s): Major section change, definite scene break
        """
        gaps = []

        # Check for intro silence (before first word)
        if words and words[0].start > 0.5:
            gaps.append({
                "time": words[0].start / 2,
                "duration": words[0].start,
                "type": self._classify_gap(words[0].start),
                "position": "intro",
                "before_word_idx": 0,
            })

        # Analyze gaps between words
        for i in range(len(words) - 1):
            gap_duration = words[i + 1].start - words[i].end
            if gap_duration > 0.2:  # Ignore very tiny gaps
                gap_time = words[i].end + gap_duration / 2
                gaps.append({
                    "time": gap_time,
                    "duration": gap_duration,
                    "type": self._classify_gap(gap_duration),
                    "position": "between",
                    "after_word_idx": i,
                    "before_word_idx": i + 1,
                })

        # Check for outro silence (after last word)
        if words and (duration - words[-1].end) > 0.5:
            outro_gap = duration - words[-1].end
            gaps.append({
                "time": words[-1].end + outro_gap / 2,
                "duration": outro_gap,
                "type": self._classify_gap(outro_gap),
                "position": "outro",
                "after_word_idx": len(words) - 1,
            })

        return {
            "gaps": gaps,
            "section_breaks": [g for g in gaps if g["type"] in ("section", "major")],
            "sentence_breaks": [g for g in gaps if g["type"] == "sentence"],
            "phrase_breaks": [g for g in gaps if g["type"] == "phrase"],
        }

    def _classify_gap(self, gap_duration: float) -> str:
        """Classify a gap by its duration."""
        if gap_duration < 0.3:
            return "micro"
        elif gap_duration < 0.8:
            return "phrase"
        elif gap_duration < 1.5:
            return "sentence"
        elif gap_duration < 3.0:
            return "section"
        else:
            return "major"

    def _find_instrumental_sections(
        self, words: list[Word], duration: float
    ) -> list[tuple[float, float]]:
        """
        Find instrumental sections (periods with no vocals).

        An instrumental section is a gap > 3 seconds with no words.
        """
        instrumentals = []

        # Check intro
        if words and words[0].start > 3.0:
            instrumentals.append((0.0, words[0].start))

        # Check between words
        for i in range(len(words) - 1):
            gap_start = words[i].end
            gap_end = words[i + 1].start
            if (gap_end - gap_start) > 3.0:
                instrumentals.append((gap_start, gap_end))

        # Check outro
        if words and (duration - words[-1].end) > 3.0:
            instrumentals.append((words[-1].end, duration))

        return instrumentals

    def _group_into_phrases(
        self, words: list[Word], gap_analysis: dict
    ) -> list[list[Word]]:
        """
        Group words into logical phrases based on natural pauses.

        A phrase is a group of words that should stay together in a scene.
        We break on sentence-level gaps or stronger, keeping phrase-level
        gaps together when possible.
        """
        if not words:
            return []

        # Get indices where we should break (sentence or stronger)
        break_indices = set()
        for gap in gap_analysis["gaps"]:
            if gap["type"] in ("sentence", "section", "major"):
                if "after_word_idx" in gap:
                    break_indices.add(gap["after_word_idx"])

        # Group words into phrases
        phrases = []
        current_phrase = []

        for i, word in enumerate(words):
            current_phrase.append(word)
            if i in break_indices:
                if current_phrase:
                    phrases.append(current_phrase)
                    current_phrase = []

        # Don't forget the last phrase
        if current_phrase:
            phrases.append(current_phrase)

        return phrases

    def _build_scenes_from_phrases(
        self,
        phrases: list[list[Word]],
        instrumental_sections: list[tuple[float, float]],
        gap_analysis: dict,
        duration: float,
        min_duration: float,
        max_duration: float,
        target_per_minute: float,
    ) -> list[tuple[float, float, list[Word]]]:
        """
        Build scenes by combining phrases while respecting constraints.

        Strategy:
        1. Section breaks (>1.5s gaps) are mandatory scene boundaries
        2. Combine phrases until we approach target duration
        3. Split if a combined phrase would exceed max duration
        4. Handle instrumental sections as their own scenes
        """
        scenes = []
        target_duration = 60.0 / target_per_minute  # e.g., 15s for 4 scenes/min

        # Get all mandatory break points (section and major gaps)
        mandatory_breaks = set()
        for gap in gap_analysis["section_breaks"]:
            mandatory_breaks.add(gap["time"])

        # Track what time ranges are instrumental
        def is_instrumental(start: float, end: float) -> bool:
            for inst_start, inst_end in instrumental_sections:
                # Check if this range overlaps significantly with instrumental
                overlap_start = max(start, inst_start)
                overlap_end = min(end, inst_end)
                if overlap_end > overlap_start:
                    overlap = overlap_end - overlap_start
                    range_duration = end - start
                    if overlap / range_duration > 0.5:
                        return True
            return False

        # Build scenes by accumulating phrases
        current_words = []
        current_start = 0.0

        for phrase in phrases:
            phrase_start = phrase[0].start
            phrase_end = phrase[-1].end
            phrase_duration = phrase_end - phrase_start

            # Check if there's a mandatory break before this phrase
            has_mandatory_break = any(
                current_words and current_words[-1].end < brk < phrase_start
                for brk in mandatory_breaks
            )

            # Calculate what duration would be if we add this phrase
            potential_end = phrase_end
            potential_duration = potential_end - current_start

            # Decide whether to start a new scene
            start_new_scene = False

            if not current_words:
                # First phrase - check if we need an intro instrumental scene
                if phrase_start > 2.0:
                    # Add instrumental intro scene
                    scenes.append((0.0, phrase_start, []))
                    current_start = phrase_start
            elif has_mandatory_break:
                # Mandatory break - must start new scene
                start_new_scene = True
            elif potential_duration > max_duration:
                # Would exceed max - start new scene
                start_new_scene = True
            elif potential_duration > target_duration * 1.3:
                # Significantly over target and we have enough content
                current_duration = (current_words[-1].end if current_words else current_start) - current_start
                if current_duration >= min_duration:
                    start_new_scene = True

            if start_new_scene and current_words:
                # Finalize current scene
                scene_end = current_words[-1].end
                # Extend slightly into the gap for visual continuity
                gap_to_next = phrase_start - scene_end
                if gap_to_next > 0:
                    scene_end += min(gap_to_next * 0.3, 0.5)
                scenes.append((current_start, scene_end, current_words))

                # Check for instrumental gap between scenes
                if phrase_start - scene_end > 2.0:
                    scenes.append((scene_end, phrase_start, []))
                    current_start = phrase_start
                else:
                    current_start = scene_end

                current_words = []

            # Add phrase to current scene
            current_words.extend(phrase)

        # Finalize last scene
        if current_words:
            scene_end = current_words[-1].end
            # Extend to end if close, or add outro scene
            if duration - scene_end < 2.0:
                scene_end = duration
                scenes.append((current_start, scene_end, current_words))
            else:
                # Extend slightly, then add outro
                scene_end += min((duration - scene_end) * 0.2, 0.5)
                scenes.append((current_start, scene_end, current_words))
                if duration - scene_end > 1.0:
                    scenes.append((scene_end, duration, []))
        elif current_start < duration:
            # Only instrumental remaining
            scenes.append((current_start, duration, []))

        return scenes

    def _validate_and_adjust_scenes(
        self,
        scenes: list[tuple[float, float, list[Word]]],
        duration: float,
        min_duration: float,
    ) -> list[tuple[float, float, list[Word]]]:
        """
        Final validation pass to ensure scene quality.

        - Merge scenes that are too short
        - Ensure no gaps between scenes
        - Ensure last scene ends at song duration
        """
        if not scenes:
            return [(0.0, duration, [])]

        validated = []

        for i, (start, end, words) in enumerate(scenes):
            scene_duration = end - start

            # If scene is too short, try to merge with previous
            if scene_duration < min_duration and validated:
                prev_start, prev_end, prev_words = validated[-1]
                combined_duration = end - prev_start

                # Merge if combined duration is reasonable
                if combined_duration <= 18.0:  # Allow slightly over max for merging
                    merged_words = prev_words + words
                    validated[-1] = (prev_start, end, merged_words)
                    continue

            # Ensure no gaps - adjust start to previous end
            if validated:
                prev_start, prev_end, prev_words = validated[-1]
                if start > prev_end + 0.1:
                    # There's a gap - extend previous scene or adjust current start
                    gap = start - prev_end
                    if gap < 1.0:
                        # Small gap - just adjust current start
                        start = prev_end
                    else:
                        # Larger gap - extend previous scene halfway
                        midpoint = prev_end + gap / 2
                        validated[-1] = (prev_start, midpoint, prev_words)
                        start = midpoint

            validated.append((start, end, words))

        # Ensure first scene starts at 0
        if validated and validated[0][0] > 0.1:
            start, end, words = validated[0]
            validated[0] = (0.0, end, words)

        # Ensure last scene ends at duration
        if validated and validated[-1][1] < duration - 0.1:
            start, end, words = validated[-1]
            validated[-1] = (start, duration, words)

        return validated

    def _find_break_points(self, words: list[Word]) -> list[float]:
        """Find natural break points (gaps) between words. (Legacy method for compatibility)"""
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
        """Find the nearest break point within tolerance. (Legacy method for compatibility)"""
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

    def extract_visual_world(
        self,
        lyrics: str,
        concept: SongConcept,
    ) -> str:
        """
        Extract a consistent visual world/setting from lyrics and concept.

        This defines the persistent universe for ALL scenes - ensuring
        we don't mix medieval with WW2, cyberpunk with western, etc.

        Args:
            lyrics: The song lyrics
            concept: Song concept with themes and mood

        Returns:
            A description of the visual world/setting
        """
        # If concept already has visual_world, use it
        if concept.visual_world:
            return concept.visual_world

        client = self._get_client()

        prompt = f"""Analyze these song lyrics and concept to define ONE consistent visual world/setting.

Song Concept:
- Genre: {concept.genre}
- Mood: {concept.mood}
- Themes: {', '.join(concept.themes)}

Lyrics:
{lyrics[:2000]}

Based on the lyrics and themes, define a SINGLE consistent visual world that ALL scenes must exist in.
This should be a specific time period, place, and aesthetic - NOT generic.

Examples of good visual worlds:
- "Medieval fantasy kingdom with stone castles, misty forests, knights in plate armor"
- "1980s neon-lit Tokyo with rain-slicked streets, Japanese signage, retro technology"
- "Post-apocalyptic desert wasteland with rusted vehicles, dust storms, survivalist camps"
- "Victorian steampunk London with brass machinery, fog, gas lamps, industrial architecture"

Respond with ONLY the visual world description (1-2 sentences). Be specific about:
- Time period/era
- Location/environment type
- Key visual elements that should appear consistently"""

        try:
            response = client.messages.create(
                model=default_config.claude_model,  # Use global config for dynamic model selection
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            visual_world = response.content[0].text.strip()
            logger.info(f"Extracted visual world: {visual_world}")
            return visual_world
        except Exception as e:
            logger.error(f"Failed to extract visual world: {e}")
            # Fallback based on genre/themes
            return f"{concept.mood} {concept.genre} setting with {', '.join(concept.themes[:2])} elements"

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

        # Extract visual world for consistency across all scenes
        visual_world = self.extract_visual_world(effective_lyrics, concept)
        logger.info(f"Using visual world: {visual_world}")

        # Store visual_world in concept for later use by image generator
        concept.visual_world = visual_world

        num_scenes = len(scene_boundaries)
        # Suggest effect distribution for variety
        effect_hints = self._suggest_effect_distribution(num_scenes)

        prompt = f"""Create visual prompts for a music video with these scenes.

*** CRITICAL - VISUAL WORLD (MUST BE REFERENCED IN EVERY SCENE) ***
{visual_world}

ALL scenes MUST exist in this visual world. Do NOT mix time periods, settings, or aesthetics.
Every visual prompt you write MUST include elements from this visual world.

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
   - MUST reference the visual world setting (e.g., if medieval fantasy, show castles/knights/forests)
   - The prompt should MATCH the specific lyrics for that scene's timestamp
   - Include visual elements that represent the words being sung
2. A motion prompt (5-15 words describing CHARACTER ACTION for animation)
   - Describes physical movement/action (NOT appearance)
   - Examples: "playing guitar passionately", "dancing with arms raised", "walking through rain"
3. The mood of this specific scene
4. A Ken Burns effect: zoom_in, zoom_out, pan_left, pan_right, pan_up, pan_down
   - IMPORTANT: Use VARIED effects - don't use the same effect for all scenes!

Respond in this exact JSON format:
{{
    "scenes": [
        {{
            "visual_prompt": "Detailed description INCLUDING the visual world setting",
            "motion_prompt": "Short action description for animation",
            "mood": "emotional tone of this scene",
            "effect": "zoom_in"
        }}
    ]
}}

Create exactly {num_scenes} scenes. Only respond with the JSON."""

        try:
            # Log the visual agent prompt
            logger.info("=" * 60)
            logger.info(f"VISUAL AGENT PROMPT (generating {num_scenes} scenes):")
            logger.info("-" * 60)
            logger.info(f"Visual World: {visual_world}")
            logger.info(f"Genre: {concept.genre}, Mood: {concept.mood}")
            logger.info(f"Visual Style: {concept.visual_style or 'cinematic'}")
            logger.info("=" * 60)

            response = client.messages.create(
                model=default_config.claude_model,  # Use global config for dynamic model selection
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
                        motion_prompt=scene_data.get("motion_prompt"),
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
                # Generic motion prompt for fallback
                motion_prompt = "performing with emotion, moving to the music"
            else:
                # Instrumental section
                prompt = f"{style}. {concept.mood} instrumental moment, abstract visual"
                motion_prompt = "swaying gently to the music"

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
                    motion_prompt=motion_prompt,
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
