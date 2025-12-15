"""Visual Style Agent for collaborative visual style and scene prompt development."""

import logging
import time
from typing import Optional

import anthropic

from src.config import Config, config as default_config
from src.models.schemas import (
    SongConcept,
    GeneratedLyrics,
    Transcript,
    VisualPlan,
    ScenePrompt,
    KenBurnsEffect,
)

logger = logging.getLogger(__name__)


def _build_system_prompt(num_scenes: int, scene_info: str, full_lyrics: str = "") -> str:
    """Build system prompt with scene count context and full lyrics.

    Args:
        num_scenes: Number of scenes required for the song
        scene_info: Pre-calculated scene breakdown with timing and lyrics segments
        full_lyrics: The complete song lyrics for narrative context
    """
    lyrics_section = ""
    if full_lyrics:
        lyrics_section = f"""## Full Song Lyrics (for narrative context)
{full_lyrics}

"""

    return f"""You are a visual creative director for music videos. You're working with a user to develop the visual style for their song's music video.

## CRITICAL: Scene Count
This song requires EXACTLY {num_scenes} scenes. You MUST create prompts for all {num_scenes} scenes.

{lyrics_section}{scene_info}

## Your Goal
Collaboratively develop:
1. A consistent VISUAL WORLD - the unified setting/universe for all {num_scenes} scenes
2. CHARACTER DESCRIPTION - how the main character(s) should look
3. CINEMATOGRAPHY STYLE - lighting, camera work, color palette
4. SCENE-BY-SCENE PROMPTS - specific visuals for EACH of the {num_scenes} scenes

## Conversation Flow

Start by:
1. Acknowledging you'll be creating {num_scenes} scenes for this song
2. Analyzing the lyrics and themes
3. Proposing 2-3 distinct visual world options (e.g., "cyberpunk city", "pastoral countryside", "abstract dreamscape")
4. Asking which direction resonates with the user

Then:
1. Develop character descriptions based on their choice
2. Discuss cinematography and color palette
3. Create scene-by-scene visual prompts for ALL {num_scenes} scenes

## Important Guidelines

- Keep responses conversational but efficient
- After 3-4 exchanges, proactively offer to finalize the visual plan
- Each scene prompt should:
  - Match the specific lyrics/words being sung in that time segment
  - Maintain visual world consistency
  - Include mood and suggested camera movement
  - Be detailed enough for AI image generation

## When Ready to Finalize

When the user says they're ready, or after sufficient discussion, provide a COMPLETE VISUAL PLAN with these clearly labeled sections:

**VISUAL WORLD**
[Description of the consistent visual universe for all scenes]

**CHARACTER DESCRIPTION**
[Detailed description of the main character's appearance, clothing, features]

**CINEMATOGRAPHY STYLE**
[Camera work, lighting, color palette description]

**COLOR PALETTE**
[Specific colors and tones]

**SCENE PROMPTS** (MUST include all {num_scenes} scenes!)
Scene 1 (0:00-X:XX): "[Detailed visual prompt for scene 1]"
Scene 2 (X:XX-Y:YY): "[Detailed visual prompt for scene 2]"
...
Scene {num_scenes} (Z:ZZ-end): "[Detailed visual prompt for scene {num_scenes}]"

Always end by asking if they want to refine anything or proceed to generate images."""


class VisualStyleAgent:
    """Agent for collaborative visual style and scene prompt development."""

    # Required sections in a complete visual plan
    REQUIRED_SECTIONS = [
        "VISUAL WORLD",
        "CHARACTER DESCRIPTION",
        "CINEMATOGRAPHY STYLE",
        "SCENE PROMPTS",
    ]

    def __init__(
        self,
        concept: Optional[SongConcept] = None,
        lyrics: Optional[GeneratedLyrics] = None,
        transcript: Optional[Transcript] = None,
        config: Optional[Config] = None,
    ):
        self.config = config or default_config
        self.concept = concept
        self.lyrics = lyrics
        self.transcript = transcript
        self._client = None
        self.conversation_history: list[dict] = []

        # Pre-calculate scene info
        self._num_scenes = 4  # Default
        self._scene_segments: list[dict] = []
        self._scene_info_text = ""
        if transcript and transcript.segments:
            self._calculate_scene_info()

    def _calculate_scene_info(self) -> None:
        """Calculate scene boundaries and store for use throughout conversation."""
        if not self.transcript:
            return

        total_duration = self.transcript.duration
        self._num_scenes = max(4, int(total_duration / 15))  # ~4 scenes per minute
        scene_duration = total_duration / self._num_scenes

        self._scene_segments = []
        scene_lines = []

        for i in range(self._num_scenes):
            start_time = i * scene_duration
            end_time = (i + 1) * scene_duration

            # Get words in this time range
            words = [
                w for w in self.transcript.all_words
                if w.start >= start_time and w.end <= end_time
            ]
            lyrics_segment = " ".join(w.word for w in words) if words else ""

            self._scene_segments.append({
                "index": i,
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "lyrics_segment": lyrics_segment,
            })

            # Format for display
            scene_lines.append(
                f"Scene {i+1} ({start_time:.1f}s - {end_time:.1f}s): \"{lyrics_segment[:60]}{'...' if len(lyrics_segment) > 60 else ''}\""
            )

        # For large scene counts, show only summary + first/last few scenes to save tokens
        if self._num_scenes > 30:
            preview_lines = scene_lines[:5] + [f"... (scenes 6-{self._num_scenes - 5} omitted for brevity) ..."] + scene_lines[-5:]
            self._scene_info_text = f"""## Scene Breakdown ({self._num_scenes} scenes total)
Song duration: {total_duration:.1f} seconds
Each scene: ~{scene_duration:.1f} seconds

NOTE: With {self._num_scenes} scenes, showing preview only. Full timing will be provided during extraction.

""" + "\n".join(preview_lines)
        else:
            self._scene_info_text = f"""## Scene Breakdown ({self._num_scenes} scenes total)
Song duration: {total_duration:.1f} seconds
Each scene: ~{scene_duration:.1f} seconds

""" + "\n".join(scene_lines)

    @property
    def num_scenes(self) -> int:
        """Return the number of scenes required for this song."""
        return self._num_scenes

    @property
    def scene_segments(self) -> list[dict]:
        """Return the pre-calculated scene segments."""
        return self._scene_segments

    def _get_client(self) -> anthropic.Anthropic:
        """Lazy load Anthropic client."""
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
        return self._client

    def set_num_scenes(self, num_scenes: int) -> None:
        """
        Update the number of scenes (e.g., from user slider).
        Recalculates scene segments accordingly.
        """
        if not self.transcript:
            self._num_scenes = num_scenes
            return

        total_duration = self.transcript.duration
        self._num_scenes = num_scenes
        scene_duration = total_duration / num_scenes

        self._scene_segments = []
        scene_lines = []

        for i in range(num_scenes):
            start_time = i * scene_duration
            end_time = (i + 1) * scene_duration

            words = [
                w for w in self.transcript.all_words
                if w.start >= start_time and w.end <= end_time
            ]
            lyrics_segment = " ".join(w.word for w in words) if words else ""

            self._scene_segments.append({
                "index": i,
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "lyrics_segment": lyrics_segment,
            })

            scene_lines.append(
                f"Scene {i+1} ({start_time:.1f}s - {end_time:.1f}s): \"{lyrics_segment[:60]}{'...' if len(lyrics_segment) > 60 else ''}\""
            )

        # For large scene counts, show only summary + first/last few scenes to save tokens
        if self._num_scenes > 30:
            preview_lines = scene_lines[:5] + [f"... (scenes 6-{self._num_scenes - 5} omitted for brevity) ..."] + scene_lines[-5:]
            self._scene_info_text = f"""## Scene Breakdown ({self._num_scenes} scenes total)
Song duration: {total_duration:.1f} seconds
Each scene: ~{scene_duration:.1f} seconds

NOTE: With {self._num_scenes} scenes, showing preview only. Full timing will be provided during extraction.

""" + "\n".join(preview_lines)
        else:
            self._scene_info_text = f"""## Scene Breakdown ({self._num_scenes} scenes total)
Song duration: {total_duration:.1f} seconds
Each scene: ~{scene_duration:.1f} seconds

""" + "\n".join(scene_lines)

    def _get_system_prompt(self) -> str:
        """Get the system prompt with current scene info and full lyrics."""
        full_lyrics = ""
        if self.lyrics and self.lyrics.lyrics:
            full_lyrics = self.lyrics.lyrics
        return _build_system_prompt(self._num_scenes, self._scene_info_text, full_lyrics)

    def _build_context(self) -> str:
        """Build context string from concept, lyrics, and transcript."""
        context_parts = []

        # Always include scene count prominently
        context_parts.append(f"## IMPORTANT: This song requires {self._num_scenes} scenes\n")

        if self.concept:
            context_parts.append(f"""## Song Concept
- Genre: {self.concept.genre}
- Mood: {self.concept.mood}
- Themes: {', '.join(self.concept.themes) if self.concept.themes else 'Not specified'}
- Influences: {', '.join(self.concept.influences) if self.concept.influences else 'Not specified'}
""")
            if self.concept.character_description:
                context_parts.append(f"- Character (from concept): {self.concept.character_description}")
            if self.concept.visual_world:
                context_parts.append(f"- Visual World (from concept): {self.concept.visual_world}")

        if self.lyrics:
            context_parts.append(f"""## Lyrics
Title: {self.lyrics.title}

{self.lyrics.lyrics}
""")

        # Include pre-calculated scene info
        if self._scene_info_text:
            context_parts.append(self._scene_info_text)

        return "\n".join(context_parts)

    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response in the visual style development conversation.

        Args:
            user_message: The user's message

        Returns:
            The assistant's response
        """
        client = self._get_client()

        # For the first message, inject context
        if not self.conversation_history:
            context = self._build_context()
            full_user_message = f"""Here's the song context:

{context}

User's message: {user_message}"""
        else:
            full_user_message = user_message

        # Add user message to history
        self.conversation_history.append({"role": "user", "content": full_user_message})

        # Send to Claude with dynamic system prompt (with retry for overload)
        max_retries = 5
        base_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=8192,  # Increased to handle 20+ scene descriptions
                    system=self._get_system_prompt(),
                    messages=self.conversation_history,
                )
                break  # Success, exit retry loop
            except anthropic.OverloadedError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Anthropic API overloaded, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"Anthropic API overloaded after {max_retries} attempts")
                    raise

        assistant_message = response.content[0].text

        # Add assistant response to history
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_message}
        )

        return assistant_message

    def is_ready_to_finalize(self) -> bool:
        """
        Check if the conversation contains a complete visual plan ready for finalization.

        Returns:
            True if all required sections are present in the last assistant message
        """
        if not self.conversation_history:
            return False

        # Get the last assistant message
        last_assistant_msg = None
        for msg in reversed(self.conversation_history):
            if msg["role"] == "assistant":
                last_assistant_msg = msg["content"]
                break

        if not last_assistant_msg:
            return False

        # Check for required sections (case-insensitive)
        content_upper = last_assistant_msg.upper()
        for section in self.REQUIRED_SECTIONS:
            if section.upper() not in content_upper:
                return False

        return True

    def get_missing_sections(self) -> list[str]:
        """
        Get list of sections that are still missing from the conversation.

        Returns:
            List of missing section names
        """
        if not self.conversation_history:
            return self.REQUIRED_SECTIONS.copy()

        # Get the last assistant message
        last_assistant_msg = None
        for msg in reversed(self.conversation_history):
            if msg["role"] == "assistant":
                last_assistant_msg = msg["content"]
                break

        if not last_assistant_msg:
            return self.REQUIRED_SECTIONS.copy()

        content_upper = last_assistant_msg.upper()
        missing = []
        for section in self.REQUIRED_SECTIONS:
            if section.upper() not in content_upper:
                missing.append(section)

        return missing

    def get_readiness_status(self) -> dict:
        """
        Get detailed readiness status for UI display.

        Returns:
            Dict with 'ready', 'missing', 'message_count', 'has_content' fields
        """
        missing = self.get_missing_sections()
        return {
            "ready": len(missing) == 0,
            "missing": missing,
            "message_count": len([m for m in self.conversation_history if m["role"] == "user"]),
            "has_content": len(self.conversation_history) > 0,
        }

    # Batch size for scene extraction (to avoid token limits)
    SCENE_BATCH_SIZE = 25

    def extract_visual_plan(self) -> Optional[VisualPlan]:
        """
        Extract a structured VisualPlan from the conversation.

        For large scene counts (>25), uses batched extraction to avoid token limits.
        Each batch maintains consistency by referencing previous scene prompts.

        Returns:
            VisualPlan if extraction successful, None otherwise
        """
        import json
        import re

        # For small scene counts, use single extraction
        if self._num_scenes <= self.SCENE_BATCH_SIZE:
            return self._extract_single_batch()

        # For large scene counts, use batched extraction
        return self._extract_batched()

    def _extract_single_batch(self) -> Optional[VisualPlan]:
        """Extract visual plan in a single API call (for ≤25 scenes)."""
        import json
        import re

        client = self._get_client()

        # Use pre-calculated scene segments
        scene_info = json.dumps(self._scene_segments, indent=2) if self._scene_segments else "[]"

        extraction_prompt = f"""Based on our conversation, extract the visual plan as JSON.

CRITICAL: Create EXACTLY {self._num_scenes} scene prompts using these timings:
{scene_info}

Return ONLY valid JSON:
{{
  "visual_world": "setting description",
  "character_description": "character appearance",
  "cinematography_style": "style description",
  "color_palette": "colors or null",
  "scene_prompts": [
    {{"index": 0, "start_time": X, "end_time": Y, "lyrics_segment": "...", "visual_prompt": "scene-specific action and composition only", "motion_prompt": "short action description for animation (5-10 words)", "mood": "mood", "effect": "zoom_in", "show_character": true}}
  ]
}}

RULES:
1. EXACTLY {self._num_scenes} scene_prompts - no more, no less
2. Use EXACT start_time/end_time from timings above
3. visual_prompt: ONLY describe what's UNIQUE to this specific scene (action, composition, key elements)
   - DO NOT include character description (it's added separately when show_character=true)
   - DO NOT include visual world/setting (it's added separately)
   - DO NOT include cinematography style (it's added separately)
   - JUST describe: what is happening, camera framing, key visual elements in THIS scene
   - Example: "Close-up of hands strumming guitar strings, warm spotlight from above"
4. motion_prompt: SHORT action for animation (e.g. "strumming guitar passionately", "walking through rain")
5. effect: zoom_in, zoom_out, pan_left, pan_right, pan_up, or pan_down
6. show_character: true if the main character appears in this scene, false for landscapes/objects/establishing shots
7. NO markdown, NO explanation - ONLY the JSON object"""

        messages = self.conversation_history.copy()
        messages.append({"role": "user", "content": extraction_prompt})

        logger.info(f"Extracting visual plan for {self._num_scenes} scenes (single batch)...")

        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=16384,
            system="You are a JSON extraction assistant. Return only valid JSON, no markdown. visual_prompt should ONLY describe scene-specific action/composition (1-2 sentences) - DO NOT include character descriptions, visual world, or cinematography style as those are added separately.",
            messages=messages,
        )

        response_text = response.content[0].text.strip()
        logger.debug(f"Extraction response (first 500 chars): {response_text[:500]}")

        return self._parse_full_plan(response_text)

    def _extract_batched(self) -> Optional[VisualPlan]:
        """Extract visual plan in batches (for >25 scenes)."""
        import json
        import time

        client = self._get_client()
        num_batches = (self._num_scenes + self.SCENE_BATCH_SIZE - 1) // self.SCENE_BATCH_SIZE

        logger.info(f"Extracting visual plan for {self._num_scenes} scenes in {num_batches} batches...")

        # First batch: Extract style info + first batch of scenes
        first_batch_segments = self._scene_segments[:self.SCENE_BATCH_SIZE]
        first_batch_info = json.dumps(first_batch_segments, indent=2)

        first_prompt = f"""Based on our conversation, extract the visual plan as JSON.

This song has {self._num_scenes} scenes total. We'll extract them in batches.
This is BATCH 1 of {num_batches} - extract the visual style AND scenes 1-{len(first_batch_segments)}.

Scene timings for this batch:
{first_batch_info}

Return ONLY valid JSON:
{{
  "visual_world": "setting description",
  "character_description": "character appearance",
  "cinematography_style": "style description",
  "color_palette": "colors or null",
  "scene_prompts": [
    {{"index": 0, "start_time": X, "end_time": Y, "lyrics_segment": "...", "visual_prompt": "scene-specific action and composition only", "motion_prompt": "short action description for animation (5-10 words)", "mood": "mood", "effect": "zoom_in", "show_character": true}}
  ]
}}

RULES:
1. Create {len(first_batch_segments)} scene_prompts for this batch (scenes 1-{len(first_batch_segments)})
2. Use EXACT start_time/end_time from timings above
3. visual_prompt: ONLY describe what's UNIQUE to this specific scene
   - DO NOT include character description (it's added separately when show_character=true)
   - DO NOT include visual world/setting (it's added separately)
   - DO NOT include cinematography style (it's added separately)
   - JUST describe: action, camera framing, key visual elements specific to THIS scene
   - Example: "Close-up of hands on piano keys, sheet music in soft focus"
4. motion_prompt: SHORT action for animation (e.g. "fingers dancing on piano keys")
5. effect: zoom_in, zoom_out, pan_left, pan_right, pan_up, or pan_down
6. show_character: true if the main character appears in this scene, false for landscapes/objects/establishing shots
7. NO markdown, NO explanation - ONLY the JSON object"""

        messages = self.conversation_history.copy()
        messages.append({"role": "user", "content": first_prompt})

        # First batch with retry logic
        first_response = self._call_with_retry(
            client, messages,
            "You are a JSON extraction assistant. Return only valid JSON, no markdown. visual_prompt should ONLY describe scene-specific action/composition (1-2 sentences) - DO NOT include character descriptions, visual world, or cinematography style as those are added separately."
        )

        if not first_response:
            logger.error("First batch extraction failed")
            return None

        # Parse first batch to get style info and initial scenes
        first_data = self._parse_json_response(first_response)
        if not first_data:
            logger.error("Failed to parse first batch JSON")
            return None

        # Extract style info
        visual_world = first_data.get("visual_world", "Cinematic visual style")
        character_description = first_data.get("character_description", "Main character")
        cinematography_style = first_data.get("cinematography_style", "Cinematic film style")
        color_palette = first_data.get("color_palette")

        # Collect all scene prompts
        all_scene_prompts = first_data.get("scene_prompts", [])
        logger.info(f"Batch 1: extracted {len(all_scene_prompts)} scenes")

        # Subsequent batches: Extract remaining scenes
        for batch_idx in range(1, num_batches):
            start_idx = batch_idx * self.SCENE_BATCH_SIZE
            end_idx = min(start_idx + self.SCENE_BATCH_SIZE, self._num_scenes)
            batch_segments = self._scene_segments[start_idx:end_idx]
            batch_info = json.dumps(batch_segments, indent=2)

            # Get last few scenes from previous batch for context
            prev_scenes_context = all_scene_prompts[-3:] if len(all_scene_prompts) >= 3 else all_scene_prompts
            prev_context_str = json.dumps(prev_scenes_context, indent=2)

            batch_prompt = f"""Continue extracting scene prompts for the music video.

BATCH {batch_idx + 1} of {num_batches} - scenes {start_idx + 1}-{end_idx} of {self._num_scenes}.

Visual style (for context, but DO NOT repeat in visual_prompt):
- Visual World: {visual_world}
- Character: {character_description}
- Cinematography: {cinematography_style}

Previous scenes for reference (maintain visual continuity):
{prev_context_str}

Scene timings for THIS BATCH:
{batch_info}

Return ONLY a JSON array of scene_prompts for scenes {start_idx + 1}-{end_idx}:
[
  {{"index": {start_idx}, "start_time": X, "end_time": Y, "lyrics_segment": "...", "visual_prompt": "scene-specific action and composition only", "motion_prompt": "short action for animation", "mood": "...", "effect": "zoom_in", "show_character": true}}
]

RULES:
1. Create EXACTLY {len(batch_segments)} scene prompts (indices {start_idx}-{end_idx - 1})
2. Use EXACT start_time/end_time from timings above
3. visual_prompt: ONLY describe what's UNIQUE to this specific scene
   - DO NOT include character description (it's added separately when show_character=true)
   - DO NOT include visual world/setting (it's added separately)
   - DO NOT include cinematography style (it's added separately)
   - JUST describe: action, camera framing, key visual elements specific to THIS scene
4. motion_prompt: SHORT action for animation (e.g. "strumming guitar passionately")
5. effect: zoom_in, zoom_out, pan_left, pan_right, pan_up, or pan_down
6. show_character: true if the main character appears in this scene, false for landscapes/objects/establishing shots
7. NO markdown, NO explanation - ONLY the JSON array"""

            batch_messages = [{"role": "user", "content": batch_prompt}]

            batch_response = self._call_with_retry(
                client, batch_messages,
                f"You are extracting scene prompts for a music video. visual_prompt should ONLY describe scene-specific action/composition - DO NOT repeat character descriptions, visual world, or cinematography style. Return only a JSON array."
            )

            if batch_response:
                batch_scenes = self._parse_json_response(batch_response)
                if isinstance(batch_scenes, list):
                    all_scene_prompts.extend(batch_scenes)
                    logger.info(f"Batch {batch_idx + 1}: extracted {len(batch_scenes)} scenes (total: {len(all_scene_prompts)})")
                elif isinstance(batch_scenes, dict) and "scene_prompts" in batch_scenes:
                    all_scene_prompts.extend(batch_scenes["scene_prompts"])
                    logger.info(f"Batch {batch_idx + 1}: extracted {len(batch_scenes['scene_prompts'])} scenes (total: {len(all_scene_prompts)})")
                else:
                    logger.warning(f"Batch {batch_idx + 1}: unexpected response format")
            else:
                logger.warning(f"Batch {batch_idx + 1} failed, continuing with partial results")

            # Small delay between batches to avoid rate limiting
            time.sleep(0.5)

        logger.info(f"Batched extraction complete: {len(all_scene_prompts)} scenes extracted")

        # Build final plan data
        plan_data = {
            "visual_world": visual_world,
            "character_description": character_description,
            "cinematography_style": cinematography_style,
            "color_palette": color_palette,
            "scene_prompts": all_scene_prompts,
        }

        return self._build_visual_plan(plan_data)

    def _call_with_retry(self, client, messages: list, system: str, max_retries: int = 3) -> Optional[str]:
        """Call Claude API with retry logic for overload errors."""
        import time

        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=8192,
                    system=system,
                    messages=messages,
                )
                return response.content[0].text.strip()
            except anthropic.OverloadedError:
                if attempt < max_retries - 1:
                    delay = 2 * (2 ** attempt)
                    logger.warning(f"API overloaded, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"API overloaded after {max_retries} attempts")
                    return None
            except Exception as e:
                logger.error(f"API call failed: {e}")
                return None
        return None

    def _parse_json_response(self, response_text: str) -> Optional[dict | list]:
        """Parse JSON from API response, handling various formats."""
        import json
        import re

        # Try direct parse
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try markdown code block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object or array in text
        for pattern in [r'\[[\s\S]*\]', r'\{[\s\S]*\}']:
            json_match = re.search(pattern, response_text)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass

        logger.debug(f"Failed to parse JSON from: {response_text[:200]}...")
        return None

    def _parse_full_plan(self, response_text: str) -> Optional[VisualPlan]:
        """Parse a complete visual plan from response text."""
        plan_data = self._parse_json_response(response_text)
        if not plan_data or not isinstance(plan_data, dict):
            logger.error("Failed to parse visual plan JSON")
            return None
        return self._build_visual_plan(plan_data)

    def _build_visual_plan(self, data: dict) -> Optional[VisualPlan]:
        """Build a VisualPlan from parsed JSON data."""
        # Ensure required fields have values
        if "visual_world" not in data or not data["visual_world"]:
            logger.warning("Missing visual_world, using default")
            data["visual_world"] = "Cinematic visual style"
        if "character_description" not in data or not data["character_description"]:
            logger.warning("Missing character_description, using default")
            data["character_description"] = "Main character"
        if "cinematography_style" not in data or not data["cinematography_style"]:
            logger.warning("Missing cinematography_style, using default")
            data["cinematography_style"] = "Cinematic film style"

        # Parse scene prompts
        scene_prompts = []
        for sp_data in data.get("scene_prompts", []):
            try:
                # Convert effect string to enum
                effect_str = sp_data.get("effect", "zoom_in").lower()
                effect_map = {
                    "zoom_in": KenBurnsEffect.ZOOM_IN,
                    "zoom_out": KenBurnsEffect.ZOOM_OUT,
                    "pan_left": KenBurnsEffect.PAN_LEFT,
                    "pan_right": KenBurnsEffect.PAN_RIGHT,
                    "pan_up": KenBurnsEffect.PAN_UP,
                    "pan_down": KenBurnsEffect.PAN_DOWN,
                }
                effect = effect_map.get(effect_str, KenBurnsEffect.ZOOM_IN)

                scene_prompts.append(ScenePrompt(
                    index=sp_data.get("index", len(scene_prompts)),
                    start_time=sp_data.get("start_time", 0.0),
                    end_time=sp_data.get("end_time", 8.0),
                    lyrics_segment=sp_data.get("lyrics_segment", ""),
                    visual_prompt=sp_data.get("visual_prompt", ""),
                    mood=sp_data.get("mood", "neutral"),
                    effect=effect,
                    user_notes=sp_data.get("user_notes"),
                    motion_prompt=sp_data.get("motion_prompt"),
                    show_character=sp_data.get("show_character", True),
                ))
            except Exception as e:
                logger.error(f"Error parsing scene prompt {sp_data}: {e}")
                continue

        logger.info(f"Parsed {len(scene_prompts)} scene prompts (expected {self._num_scenes})")

        try:
            return VisualPlan(
                visual_world=data["visual_world"],
                character_description=data["character_description"],
                cinematography_style=data["cinematography_style"],
                color_palette=data.get("color_palette"),
                scene_prompts=scene_prompts,
            )
        except Exception as e:
            logger.error(f"VisualPlan validation error: {e}")
            return None

    def reset(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []

    def start_conversation(self) -> str:
        """
        Start the visual style conversation with an initial analysis.

        Returns:
            The assistant's opening message
        """
        return self.chat(
            "Let's develop the visual style for my music video. "
            "Please analyze the lyrics and suggest some visual world options."
        )
