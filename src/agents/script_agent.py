"""Script Agent for iterative movie/podcast script development.

Helps users create scripts for animated videos, podcasts, and educational content
with consistent characters, dialogue, and scene descriptions.
"""

import json
import re
from typing import Optional

import anthropic

from src.config import Config, config as default_config
from src.models.schemas import (
    Character,
    DialogueLine,
    Emotion,
    MovieScene,
    SceneDirection,
    Script,
    VoiceSettings,
)


SYSTEM_PROMPT = """You are a professional screenwriter and creative director specializing in animated content, podcasts, and educational videos. Your role is to help users develop compelling scripts through iterative collaboration.

When a user describes their project, you should:
1. Understand their vision and target audience
2. Help define memorable characters with distinct personalities and visual descriptions
3. Structure the narrative into clear scenes with dialogue
4. Ensure visual descriptions are detailed enough for AI image generation

Be conversational and creative. After 3-4 exchanges, proactively offer to finalize the script.

When the user is ready, or after sufficient discussion, provide a COMPLETE SCRIPT with:

1. **TITLE AND DESCRIPTION**
   - Title
   - Brief description/logline
   - Target duration (e.g., "5 minutes")

2. **CHARACTERS** (for each character):
   - Name and ID (lowercase, no spaces)
   - Detailed visual description (for consistent AI image generation)
   - Personality traits
   - Voice type (age, gender, accent, tone)

3. **SCENES** (for each scene):
   - Scene heading (e.g., "INT. OFFICE - DAY")
   - Setting description
   - Characters present
   - Dialogue with emotions/actions in parentheses
   - Camera direction (wide shot, close-up, etc.)

4. **VISUAL STYLE**
   - Art style (e.g., "3D animated Pixar-style", "anime", "photorealistic")
   - Color palette
   - Mood/atmosphere

FORMAT EXAMPLE:
```
TITLE: The Discovery
DESCRIPTION: A scientist discovers something extraordinary in her lab.
DURATION: 3 minutes

CHARACTERS:
- DR_MAYA (maya): A 35-year-old South Asian woman with long black hair in a ponytail, wearing a white lab coat over a blue blouse. Sharp intelligent eyes, warm smile. Confident and curious.
  Voice: Female, 30s, slight British accent, warm but professional

- LAB_AI (labai): Appears as a holographic blue orb with pulsing light patterns. Represents advanced AI assistant.
  Voice: Neutral, calm, slightly robotic but friendly

SCENE 1: INT. RESEARCH LAB - NIGHT
Setting: High-tech laboratory with glowing monitors, test tubes, and advanced equipment. Blue ambient lighting.
Camera: Wide shot establishing the lab, then medium shot on Maya
Characters: maya

MAYA: (excited, typing rapidly) This can't be right... the energy readings are off the charts!

LAB_AI: (calm) Dr. Chen, I've verified the data three times. The anomaly is real.

MAYA: (thoughtful, standing up) If this is what I think it is... (whispers) ...this changes everything.

[End Scene 1]
```

Always ask if they want to refine any characters, add scenes, or adjust the tone."""


class ScriptAgent:
    """Agent for iterative script development for movies and podcasts."""

    REQUIRED_SECTIONS = [
        "TITLE",
        "CHARACTERS",
        "SCENE",
    ]

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config
        self._client = None
        self.conversation_history: list[dict] = []

    def _get_client(self) -> anthropic.Anthropic:
        """Lazy load Anthropic client."""
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
        return self._client

    def chat(self, user_message: str) -> str:
        """Send a message and get a response in the script development conversation.

        Args:
            user_message: The user's message

        Returns:
            The assistant's response
        """
        client = self._get_client()

        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})

        # Send to Claude
        response = client.messages.create(
            model=self.config.claude_model,
            max_tokens=4096,  # Scripts can be long
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
        """Check if the conversation contains a complete script.

        Returns:
            True if all required sections are present
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

        # Check for required sections
        content_upper = last_assistant_msg.upper()
        for section in self.REQUIRED_SECTIONS:
            if section.upper() not in content_upper:
                return False

        return True

    def get_missing_sections(self) -> list[str]:
        """Get list of sections still missing from the script."""
        if not self.conversation_history:
            return self.REQUIRED_SECTIONS.copy()

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
        """Get detailed readiness status for UI display."""
        missing = self.get_missing_sections()
        return {
            "ready": len(missing) == 0,
            "missing": missing,
            "message_count": len([m for m in self.conversation_history if m["role"] == "user"]),
            "has_content": len(self.conversation_history) > 0,
        }

    def extract_script(self) -> Optional[Script]:
        """Extract a structured Script from the conversation.

        Returns:
            Script object if extraction successful, None otherwise
        """
        client = self._get_client()

        extraction_prompt = """Based on our conversation, extract the complete script as JSON.

Return ONLY valid JSON with this structure:
{
  "title": "Script Title",
  "description": "Brief description",
  "target_duration": 300,
  "visual_style": "art style description",
  "world_description": "consistent setting/world description",
  "characters": [
    {
      "id": "character_id",
      "name": "Character Name",
      "description": "Detailed visual description for AI image generation",
      "personality": "personality traits",
      "voice_type": "male/female, age, accent, tone"
    }
  ],
  "scenes": [
    {
      "index": 0,
      "title": "INT. LOCATION - TIME",
      "setting": "detailed setting description",
      "camera": "wide shot/medium shot/close-up",
      "lighting": "lighting description",
      "mood": "scene mood",
      "visible_characters": ["character_id1", "character_id2"],
      "dialogue": [
        {
          "character_id": "character_id",
          "text": "dialogue text",
          "emotion": "neutral/happy/sad/angry/excited/thoughtful/surprised/scared/sarcastic/whisper",
          "action": "optional action/stage direction"
        }
      ]
    }
  ]
}

IMPORTANT:
- target_duration is in seconds (e.g., 300 for 5 minutes)
- character IDs must be lowercase with underscores
- emotion must be one of: neutral, happy, sad, angry, excited, thoughtful, surprised, scared, sarcastic, whisper
- Include ALL scenes and ALL dialogue from the script"""

        messages = self.conversation_history.copy()
        messages.append({"role": "user", "content": extraction_prompt})

        response = client.messages.create(
            model=self.config.claude_model,
            max_tokens=8192,  # Scripts can be very long
            system="You are a JSON extraction assistant. Return only valid JSON, no markdown formatting, no explanation text.",
            messages=messages,
        )

        response_text = response.content[0].text.strip()

        def parse_script_data(data: dict) -> Optional[Script]:
            """Parse JSON data into Script object."""
            try:
                # Parse characters
                characters = []
                for char_data in data.get("characters", []):
                    voice_type = char_data.get("voice_type", "neutral")
                    characters.append(Character(
                        id=char_data.get("id", char_data.get("name", "unknown").lower().replace(" ", "_")),
                        name=char_data.get("name", "Unknown"),
                        description=char_data.get("description", "A person"),
                        personality=char_data.get("personality"),
                        voice=VoiceSettings(
                            voice_name=voice_type,
                        ),
                    ))

                # Parse scenes
                scenes = []
                for scene_data in data.get("scenes", []):
                    # Parse dialogue
                    dialogue = []
                    for dial_data in scene_data.get("dialogue", []):
                        emotion_str = dial_data.get("emotion", "neutral").lower()
                        try:
                            emotion = Emotion(emotion_str)
                        except ValueError:
                            emotion = Emotion.NEUTRAL

                        dialogue.append(DialogueLine(
                            character_id=dial_data.get("character_id", "unknown"),
                            text=dial_data.get("text", ""),
                            emotion=emotion,
                            action=dial_data.get("action"),
                        ))

                    # Parse scene direction
                    direction = SceneDirection(
                        setting=scene_data.get("setting", "Unknown location"),
                        camera=scene_data.get("camera", "medium shot"),
                        lighting=scene_data.get("lighting"),
                        mood=scene_data.get("mood", "neutral"),
                        visible_characters=scene_data.get("visible_characters", []),
                    )

                    scenes.append(MovieScene(
                        index=scene_data.get("index", len(scenes)),
                        title=scene_data.get("title"),
                        direction=direction,
                        dialogue=dialogue,
                    ))

                return Script(
                    title=data.get("title", "Untitled"),
                    description=data.get("description"),
                    target_duration=float(data.get("target_duration", 300)),
                    characters=characters,
                    scenes=scenes,
                    visual_style=data.get("visual_style", "cinematic digital art"),
                    world_description=data.get("world_description"),
                )

            except Exception as e:
                print(f"Script parsing error: {e}")
                return None

        # Try to extract JSON from various formats
        try:
            script_data = json.loads(response_text)
            result = parse_script_data(script_data)
            if result:
                return result
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            try:
                script_data = json.loads(json_match.group(1))
                result = parse_script_data(script_data)
                if result:
                    return result
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in text
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                script_data = json.loads(json_match.group(0))
                result = parse_script_data(script_data)
                if result:
                    return result
            except json.JSONDecodeError:
                pass

        return None

    def reset(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []

    def suggest_characters(self, premise: str, num_characters: int = 3) -> list[dict]:
        """Suggest characters for a given premise.

        Args:
            premise: The story premise
            num_characters: Number of characters to suggest

        Returns:
            List of character suggestions with name, description, and personality
        """
        client = self._get_client()

        prompt = f"""Given this story premise: "{premise}"

Suggest {num_characters} distinct characters that would work well. For each character provide:
- Name
- Detailed visual description (age, appearance, clothing) for AI image generation
- Personality traits
- Voice type (age, gender, accent, tone)

Respond with only a JSON array:
[
  {{
    "name": "Character Name",
    "description": "detailed visual description",
    "personality": "personality traits",
    "voice_type": "voice description"
  }}
]"""

        response = client.messages.create(
            model=self.config.claude_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            # Try to parse JSON from response
            text = response.content[0].text
            json_match = re.search(r'\[[\s\S]*\]', text)
            if json_match:
                return json.loads(json_match.group(0))
        except (json.JSONDecodeError, AttributeError):
            pass

        return []

    def expand_scene(self, scene_description: str, characters: list[str]) -> str:
        """Expand a brief scene description into full dialogue and direction.

        Args:
            scene_description: Brief description of what happens
            characters: List of character names in the scene

        Returns:
            Expanded scene with dialogue and direction
        """
        client = self._get_client()

        prompt = f"""Expand this scene into full dialogue and direction:

Scene: {scene_description}
Characters present: {', '.join(characters)}

Write:
1. Setting description with visual details
2. Camera direction
3. Full dialogue with emotions in parentheses
4. Any actions or movements

Keep it concise but vivid."""

        response = client.messages.create(
            model=self.config.claude_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text

    def get_visual_style_suggestions(self, premise: str) -> list[str]:
        """Suggest visual styles for the project.

        Args:
            premise: The story premise

        Returns:
            List of visual style suggestions
        """
        client = self._get_client()

        prompt = f"""Given this story premise: "{premise}"

Suggest 5 visual art styles that would work well for an animated video. Be specific.

Respond with only a JSON array of strings:
["style1", "style2", "style3", "style4", "style5"]"""

        response = client.messages.create(
            model=self.config.claude_model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            return json.loads(response.content[0].text)
        except json.JSONDecodeError:
            return []
