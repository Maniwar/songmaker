"""Script Agent for iterative movie/podcast script development.

Helps users create scripts for animated videos, podcasts, and educational content
with consistent characters, dialogue, and scene descriptions.
"""

import json
import logging
import re
from typing import Optional

import anthropic

from src.config import Config, config as default_config

logger = logging.getLogger(__name__)
from src.models.schemas import (
    Character,
    DialogueLine,
    Emotion,
    MovieConfig,
    MovieFormat,
    MovieScene,
    MovieTone,
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
        self._config = config  # Store optional override
        self._client = None
        self.conversation_history: list[dict] = []
        self._project_config: Optional[MovieConfig] = None

    def set_project_config(self, project_config: MovieConfig) -> None:
        """Set the project configuration to guide script development.

        This information will be used to tailor the script to the user's
        specified format, duration, tone, and character count.

        Args:
            project_config: The MovieConfig from the setup step
        """
        self._project_config = project_config

    def _get_config_context(self) -> str:
        """Generate context string from project config for the system prompt."""
        if not self._project_config:
            return ""

        cfg = self._project_config

        # Format mapping
        format_desc = {
            MovieFormat.PODCAST: "a podcast-style discussion or interview",
            MovieFormat.EDUCATIONAL: "an educational tutorial or lesson",
            MovieFormat.SHORT_FILM: "an animated short film or story",
            MovieFormat.EXPLAINER: "an explainer video or product breakdown",
            MovieFormat.INTERVIEW: "an interview or Q&A format video",
        }

        # Tone mapping
        tone_desc = {
            MovieTone.CASUAL: "casual and conversational",
            MovieTone.PROFESSIONAL: "professional and business-like",
            MovieTone.EDUCATIONAL: "clear and instructive",
            MovieTone.HUMOROUS: "funny and lighthearted",
            MovieTone.DRAMATIC: "intense and emotional",
        }

        # Duration in readable format
        minutes = cfg.target_duration // 60
        seconds = cfg.target_duration % 60
        if seconds:
            duration_str = f"{minutes} minutes and {seconds} seconds"
        else:
            duration_str = f"{minutes} minutes"

        # Character description
        if cfg.num_characters == 1:
            char_desc = "a solo narrator"
        elif cfg.num_characters == 2:
            char_desc = "two characters in dialogue"
        else:
            char_desc = f"{cfg.num_characters} characters (ensemble)"

        # Handle custom format (with backward compatibility)
        custom_format = getattr(cfg, 'custom_format', None)
        if custom_format:
            format_text = custom_format
        else:
            format_text = format_desc.get(cfg.format, str(cfg.format))

        # Handle character names (with backward compatibility)
        character_names = getattr(cfg, 'character_names', [])
        if character_names:
            char_names = ", ".join(character_names)
            char_desc = f"{cfg.num_characters} characters named: {char_names}"
        else:
            char_desc = char_desc  # Use the solo/duo/ensemble description

        # Get recommended scenes and overlap settings (with backward compatibility)
        recommended_scenes = getattr(cfg, 'recommended_scenes', 5)
        allow_overlap = getattr(cfg, 'allow_overlap', True)

        # Handle custom tone (with backward compatibility)
        custom_tone = getattr(cfg, 'custom_tone', None)
        if custom_tone:
            tone_text = custom_tone
        else:
            tone_text = tone_desc.get(cfg.tone, str(cfg.tone))

        # Technical settings (with backward compatibility)
        generation_method = getattr(cfg, 'generation_method', 'tts_images')
        voice_provider = getattr(cfg, 'voice_provider', 'edge')
        veo_model = getattr(cfg, 'veo_model', 'veo-3.1-generate-preview')
        veo_duration = getattr(cfg, 'veo_duration', 8)
        veo_resolution = getattr(cfg, 'veo_resolution', '720p')

        # Build technical context based on generation method
        if generation_method == "veo3":
            # Calculate correct word limit: ~2.5 words/second speaking rate
            max_words_per_scene = int(veo_duration * 2.5)
            tech_context = f"""
TECHNICAL SETTINGS (Video Generation):
- Method: Veo 3.1 AI Video (generates video with spoken dialogue directly)
- Model: {veo_model}
- Clip Duration: {veo_duration} seconds per scene
- Resolution: {veo_resolution}

⚠️ CRITICAL VEO CONSTRAINT: Each scene = ONE {veo_duration}-second video clip!

STRICT SCENE STRUCTURE FOR VEO:
1. Each scene MUST fit in {veo_duration} seconds of video
2. Maximum dialogue per scene: ~{max_words_per_scene} words TOTAL (all characters combined)
3. That's roughly {max_words_per_scene // 2} words per character if 2 characters speak
4. Short, punchy exchanges work best - NOT long monologues

WORD COUNT GUIDE (at 2.5 words/second speaking rate):
- 4-second clip: max ~10 words total
- 6-second clip: max ~15 words total
- 8-second clip: max ~20 words total

SCENE DESIGN FOR VEO:
- Write MANY short scenes instead of FEW long scenes
- Each scene = one camera angle, one moment, one beat
- Break conversations into multiple scenes/clips
- Think of it like movie cuts - quick exchanges across scenes

EXAMPLE (8-second clip):
SCENE 5: INT. OFFICE - DAY
MAYA: (excited) Did you see the results?
ALEX: (surprised) They're incredible!
[End Scene - ~8 words = fits in 8 seconds]

NOT THIS (too long for one clip):
BAD: MAYA gives a 50-word explanation in one scene
INSTEAD: Break it into 3-4 separate scenes with shorter lines"""
        else:
            voice_labels = {
                "edge": "Edge TTS (Free)",
                "openai": "OpenAI TTS",
                "elevenlabs": "ElevenLabs (Premium)",
            }
            tech_context = f"""
TECHNICAL SETTINGS (TTS + Images):
- Method: Text-to-Speech with generated images
- Voice Provider: {voice_labels.get(voice_provider, voice_provider)}

IMPORTANT FOR TTS: Dialogue will be converted to speech. Write dialogue that:
- Sounds natural when read aloud by TTS
- Has appropriate pacing and punctuation for speech synthesis
- Avoids complex words that TTS might mispronounce"""

        # For Veo mode, calculate recommended scenes based on clip duration
        if generation_method == "veo3":
            # Each scene = one clip, so scenes = total_duration / clip_duration
            veo_recommended_scenes = cfg.target_duration // veo_duration
            scene_guidance = f"""- Target: {veo_recommended_scenes} scenes (= {veo_recommended_scenes} clips × {veo_duration}s each)
- This gives {veo_recommended_scenes * veo_duration} seconds of video to match {duration_str} target"""
        else:
            scene_guidance = f"- Recommended Scenes: {recommended_scenes} scenes (aim for this number to achieve the target duration)"

        context = f"""
PROJECT REQUIREMENTS (from user's setup):
- Format: {format_text}
- Target Duration: {duration_str}
{scene_guidance}
- Characters: {char_desc}
- Tone: {tone_text}
- Visual Style: {cfg.visual_style}
- Overlapping Dialogue: {"allowed (characters can interrupt each other)" if allow_overlap else "not allowed (sequential dialogue only)"}
{tech_context}
"""
        return context

    @property
    def config(self) -> Config:
        """Get config, always reading current global config for model selection."""
        return self._config or default_config

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

        # Send to Claude - always use current model from config
        model_to_use = default_config.claude_model
        logger.info(f"ScriptAgent using model: {model_to_use}")

        # Build system prompt with optional config context
        system_prompt = SYSTEM_PROMPT
        config_context = self._get_config_context()
        if config_context:
            system_prompt = system_prompt + "\n" + config_context

        # Log the prompt being sent
        logger.info("=" * 60)
        logger.info("SCRIPT AGENT PROMPT:")
        logger.info("-" * 60)
        logger.info(f"User message: {user_message[:200]}..." if len(user_message) > 200 else f"User message: {user_message}")
        logger.info(f"Conversation history: {len(self.conversation_history)} messages")
        logger.info("=" * 60)

        response = client.messages.create(
            model=model_to_use,
            max_tokens=default_config.claude_max_tokens,  # Use model's max output tokens
            system=system_prompt,
            messages=self.conversation_history,
        )

        # Log which model actually responded
        actual_model = response.model
        logger.info(f"Response from model: {actual_model}")

        assistant_message = response.content[0].text

        # Add assistant response to history
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_message}
        )

        # Store last used model for verification
        self._last_model_used = actual_model

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

        # Build Veo-specific extraction constraints if in Veo mode
        veo_constraints = ""
        if self._project_config:
            generation_method = getattr(self._project_config, 'generation_method', 'tts_images')
            if generation_method == "veo3":
                veo_duration = getattr(self._project_config, 'veo_duration', 8)
                max_words = int(veo_duration * 2.5)
                veo_constraints = f"""

⚠️ CRITICAL VEO MODE CONSTRAINTS:
This is for Veo 3.1 video generation where each scene = ONE {veo_duration}-second clip!

SCENE VALIDATION RULES:
1. Each scene's TOTAL dialogue must be ≤{max_words} words (all characters combined)
2. If a scene has too much dialogue, SPLIT it into multiple scenes
3. Speaking rate: ~2.5 words/second
4. Word limits by clip duration: 4s=10 words, 6s=15 words, 8s=20 words

BEFORE OUTPUTTING JSON:
- Count words in each scene's dialogue
- If any scene exceeds {max_words} words, split it into multiple scenes
- Renumber scene indices sequentially after splitting"""

        extraction_prompt = f"""Based on our conversation, extract the complete script as JSON.

Return ONLY valid JSON with this structure:
{{
  "title": "Script Title",
  "description": "Brief description",
  "target_duration": 300,
  "visual_style": "art style description",
  "world_description": "consistent setting/world description",
  "characters": [
    {{
      "id": "character_id",
      "name": "Character Name",
      "description": "Detailed visual description for AI image generation",
      "personality": "personality traits",
      "voice_type": "male/female, age, accent, tone"
    }}
  ],
  "scenes": [
    {{
      "index": 0,
      "title": "INT. LOCATION - TIME",
      "setting": "detailed setting description",
      "camera": "wide shot/medium shot/close-up",
      "lighting": "lighting description",
      "mood": "scene mood",
      "visible_characters": ["character_id1", "character_id2"],
      "dialogue": [
        {{
          "character_id": "character_id",
          "text": "dialogue text",
          "emotion": "neutral/happy/sad/angry/excited/thoughtful/surprised/scared/sarcastic/whisper",
          "action": "optional action/stage direction"
        }}
      ]
    }}
  ]
}}

IMPORTANT:
- target_duration is in seconds (e.g., 300 for 5 minutes)
- character IDs must be lowercase with underscores
- emotion must be one of: neutral, happy, sad, angry, excited, thoughtful, surprised, scared, sarcastic, whisper
- Include ALL scenes and ALL dialogue from the script{veo_constraints}"""

        messages = self.conversation_history.copy()
        messages.append({"role": "user", "content": extraction_prompt})

        # Log the extraction prompt
        logger.info("=" * 60)
        logger.info("SCRIPT EXTRACTION PROMPT:")
        logger.info("-" * 60)
        logger.info(f"Extracting script from {len(self.conversation_history)} conversation messages")
        if veo_constraints:
            logger.info("VEO MODE: Scene dialogue constraints enabled")
        logger.info("=" * 60)

        response = client.messages.create(
            model=default_config.claude_model,  # Use global config for model
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
                        index=scene_data.get("index", len(scenes) + 1),  # 1-based indexing
                        title=scene_data.get("title"),
                        direction=direction,
                        dialogue=dialogue,
                    ))

                # Re-index scenes to ensure proper 1-based ordering
                for i, scene in enumerate(scenes):
                    scene.index = i + 1

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
            model=default_config.claude_model,  # Use global config for model
            max_tokens=4096,  # Character lists can be detailed
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
            model=default_config.claude_model,  # Use global config for model
            max_tokens=4096,  # Scenes with dialogue can be long
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
            model=default_config.claude_model,  # Use global config for model
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            return json.loads(response.content[0].text)
        except json.JSONDecodeError:
            return []
