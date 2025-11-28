"""Visual Style Agent for collaborative visual style and scene prompt development."""

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


SYSTEM_PROMPT = """You are a visual creative director for music videos. You're working with a user to develop the visual style for their song's music video.

Context you have:
- Song concept: genre, mood, themes
- Complete lyrics
- Transcript showing word-level timing

Your goal is to collaboratively develop:
1. A consistent VISUAL WORLD - the unified setting/universe for all scenes
2. CHARACTER DESCRIPTION - how the main character(s) should look
3. CINEMATOGRAPHY STYLE - lighting, camera work, color palette
4. SCENE-BY-SCENE PROMPTS - specific visuals for each scene segment

## Conversation Flow

Start by:
1. Analyzing the lyrics and themes
2. Proposing 2-3 distinct visual world options (e.g., "cyberpunk city", "pastoral countryside", "abstract dreamscape")
3. Asking which direction resonates with the user

Then:
1. Develop character descriptions based on their choice
2. Discuss cinematography and color palette
3. Create scene-by-scene visual prompts

## Important Guidelines

- Keep responses conversational but efficient
- After 3-4 exchanges, proactively offer to finalize the visual plan
- Each scene prompt should:
  - Match the specific lyrics/words being sung
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

**SCENE PROMPTS**
Scene 1 (0:00-0:08): "[Detailed visual prompt for scene 1]"
Scene 2 (0:08-0:16): "[Detailed visual prompt for scene 2]"
...

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

    def _get_client(self) -> anthropic.Anthropic:
        """Lazy load Anthropic client."""
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
        return self._client

    def _build_context(self) -> str:
        """Build context string from concept, lyrics, and transcript."""
        context_parts = []

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

        if self.transcript and self.transcript.segments:
            # Calculate scene boundaries for context
            total_duration = self.transcript.duration
            num_scenes = max(4, int(total_duration / 15))  # Roughly 4 scenes per minute
            scene_duration = total_duration / num_scenes

            segments_text = []
            for i in range(num_scenes):
                start_time = i * scene_duration
                end_time = (i + 1) * scene_duration

                # Get words in this time range
                words = [
                    w for w in self.transcript.all_words
                    if w.start >= start_time and w.end <= end_time
                ]
                if words:
                    lyrics_segment = " ".join(w.word for w in words)
                    segments_text.append(
                        f"Scene {i+1} ({start_time:.1f}s - {end_time:.1f}s): \"{lyrics_segment}\""
                    )

            if segments_text:
                context_parts.append(f"""## Scene Segments
{chr(10).join(segments_text)}
""")

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

        # Send to Claude
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=self.conversation_history,
        )

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

    def extract_visual_plan(self) -> Optional[VisualPlan]:
        """
        Extract a structured VisualPlan from the conversation.

        Returns:
            VisualPlan if extraction successful, None otherwise
        """
        import json
        import re

        client = self._get_client()

        # Build scene info for extraction
        scene_info = ""
        if self.transcript and self.transcript.segments:
            total_duration = self.transcript.duration
            num_scenes = max(4, int(total_duration / 15))
            scene_duration = total_duration / num_scenes

            scene_segments = []
            for i in range(num_scenes):
                start_time = i * scene_duration
                end_time = (i + 1) * scene_duration
                words = [
                    w for w in self.transcript.all_words
                    if w.start >= start_time and w.end <= end_time
                ]
                lyrics_segment = " ".join(w.word for w in words) if words else ""
                scene_segments.append({
                    "index": i,
                    "start_time": round(start_time, 2),
                    "end_time": round(end_time, 2),
                    "lyrics_segment": lyrics_segment,
                })

            scene_info = f"\nScene timing for {num_scenes} scenes:\n" + json.dumps(scene_segments, indent=2)

        extraction_prompt = f"""Based on our entire conversation, extract the finalized visual plan as JSON.
{scene_info}

Return ONLY valid JSON with these REQUIRED fields:
{{
  "visual_world": "description of the consistent visual universe",
  "character_description": "detailed character appearance description",
  "cinematography_style": "camera, lighting, and style description",
  "color_palette": "color direction (can be null)",
  "scene_prompts": [
    {{
      "index": 0,
      "start_time": 0.0,
      "end_time": 8.0,
      "lyrics_segment": "the lyrics for this scene",
      "visual_prompt": "detailed visual prompt for AI image generation",
      "mood": "emotional tone of the scene",
      "effect": "zoom_in"  // one of: zoom_in, zoom_out, pan_left, pan_right, pan_up, pan_down
    }},
    // ... more scenes
  ]
}}

IMPORTANT:
- visual_world, character_description, and cinematography_style are REQUIRED
- Include ALL scene prompts discussed
- Each scene_prompt should have detailed visual_prompt text
- effect should be one of: zoom_in, zoom_out, pan_left, pan_right, pan_up, pan_down"""

        messages = self.conversation_history.copy()
        messages.append({"role": "user", "content": extraction_prompt})

        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            system="You are a JSON extraction assistant. Return only valid JSON, no markdown formatting, no explanation text.",
            messages=messages,
        )

        response_text = response.content[0].text.strip()

        def try_parse_plan(data: dict) -> Optional[VisualPlan]:
            """Try to create VisualPlan with fallbacks for missing fields."""
            # Ensure required fields have values
            if "visual_world" not in data or not data["visual_world"]:
                data["visual_world"] = "Cinematic visual style"
            if "character_description" not in data or not data["character_description"]:
                data["character_description"] = "Main character"
            if "cinematography_style" not in data or not data["cinematography_style"]:
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
                    ))
                except Exception as e:
                    print(f"Error parsing scene prompt: {e}")
                    continue

            try:
                return VisualPlan(
                    visual_world=data["visual_world"],
                    character_description=data["character_description"],
                    cinematography_style=data["cinematography_style"],
                    color_palette=data.get("color_palette"),
                    scene_prompts=scene_prompts,
                )
            except Exception as e:
                print(f"VisualPlan validation error: {e}")
                return None

        # Try to extract JSON from various formats
        try:
            # First, try direct parse
            plan_data = json.loads(response_text)
            result = try_parse_plan(plan_data)
            if result:
                return result
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            try:
                plan_data = json.loads(json_match.group(1))
                result = try_parse_plan(plan_data)
                if result:
                    return result
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in text
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                plan_data = json.loads(json_match.group(0))
                result = try_parse_plan(plan_data)
                if result:
                    return result
            except json.JSONDecodeError:
                pass

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
