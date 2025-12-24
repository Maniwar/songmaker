"""Concept Agent for iterative song concept development."""

import logging
from typing import Optional

import anthropic

from src.config import Config, config as default_config
from src.models.schemas import SongConcept

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a creative music producer and songwriter assistant specializing in creating songs for Suno AI. Your role is to help users develop compelling song concepts through iterative collaboration.

When a user describes their song idea, you should:
1. Ask 1-2 clarifying questions max to understand their vision
2. Suggest genre options that fit their idea
3. Propose mood and emotional tones
4. Suggest themes and lyrical directions
5. Recommend musical influences

Be conversational but efficient. After 3-4 exchanges, proactively offer to finalize and generate the complete package.

When the user says they're ready, or after sufficient discussion, provide a COMPLETE FINAL PACKAGE with:

1. **SONG CONCEPT SUMMARY**
   - Genre (specific, e.g., "epic symphonic power metal")
   - Mood (e.g., "triumphant, awe-inspiring")
   - Themes (3-5 themes)
   - Tempo (slow/medium/fast)
   - Influences (artists)

2. **SUNO STYLE TAGS** (comma-separated tags for Suno)
   Example: "epic power metal, soaring vocals, orchestral, male vocalist, fast tempo, choir"

3. **COMPLETE LYRICS** with structure markers:
   [Intro], [Verse 1], [Chorus], [Verse 2], [Bridge], [Outro], etc.

4. **VISUAL DESCRIPTION** for music video (character, setting, style)

Always end by asking if they want to refine anything or proceed to generate."""


class ConceptAgent:
    """Agent for iterative song concept development."""

    # Required sections in a complete package
    REQUIRED_SECTIONS = [
        "SONG CONCEPT SUMMARY",
        "SUNO STYLE TAGS",
        "LYRICS",  # Can be "COMPLETE LYRICS" or just "LYRICS"
        "VISUAL DESCRIPTION",
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
        """
        Send a message and get a response in the concept development conversation.

        Args:
            user_message: The user's message

        Returns:
            The assistant's response
        """
        client = self._get_client()

        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})

        # Send to Claude - use current model from config
        model_to_use = default_config.claude_model
        max_tokens = default_config.claude_max_tokens
        logger.info(f"ConceptAgent using model: {model_to_use} (max_tokens: {max_tokens})")

        response = client.messages.create(
            model=model_to_use,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=self.conversation_history,
        )

        # Log which model actually responded
        actual_model = response.model
        logger.info(f"Response from model: {actual_model}")

        assistant_message = response.content[0].text

        # Store last used model for verification
        self._last_model_used = actual_model

        # Add assistant response to history
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_message}
        )

        return assistant_message

    def is_ready_to_finalize(self) -> bool:
        """
        Check if the conversation contains a complete package ready for finalization.

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

    def extract_concept(self) -> Optional[SongConcept]:
        """
        Extract a structured SongConcept from the conversation.

        Returns:
            SongConcept if extraction successful, None otherwise
        """
        import json
        import re

        client = self._get_client()

        extraction_prompt = """Based on our entire conversation, extract the finalized song concept as JSON.

Return ONLY valid JSON with these REQUIRED fields:
{
  "idea": "brief description of the song concept",
  "genre": "specific genre (e.g., 'epic symphonic power metal')",
  "mood": "emotional tone (e.g., 'triumphant, uplifting')",
  "themes": ["theme1", "theme2", "theme3"],
  "tempo": "slow/medium/fast",
  "influences": ["artist1", "artist2"],
  "character_description": "visual character description for music video",
  "visual_style": "art style for visuals",
  "suno_tags": "comma-separated style tags for Suno AI",
  "draft_lyrics": "complete lyrics with [Verse], [Chorus] markers"
}

IMPORTANT:
- idea, genre, and mood are REQUIRED - provide best guesses from our discussion
- Include the complete draft_lyrics if any lyrics were created
- Include suno_tags for Suno style prompt"""

        messages = self.conversation_history.copy()
        messages.append({"role": "user", "content": extraction_prompt})

        response = client.messages.create(
            model=default_config.claude_model,  # Use global config for dynamic model selection
            max_tokens=4096,  # Increased for longer lyrics
            system="You are a JSON extraction assistant. Return only valid JSON, no markdown formatting, no explanation text.",
            messages=messages,
        )

        response_text = response.content[0].text.strip()

        def try_parse_concept(data: dict) -> Optional[SongConcept]:
            """Try to create SongConcept with fallbacks for missing fields."""
            # Ensure required fields have values
            if "idea" not in data or not data["idea"]:
                # Try to extract from conversation
                for msg in self.conversation_history:
                    if msg["role"] == "user":
                        data["idea"] = msg["content"][:200]
                        break
                else:
                    data["idea"] = "Song concept"

            if "genre" not in data or not data["genre"]:
                data["genre"] = "pop"

            if "mood" not in data or not data["mood"]:
                data["mood"] = "uplifting"

            # Ensure list fields are lists
            if "themes" not in data or not isinstance(data.get("themes"), list):
                data["themes"] = []
            if "influences" not in data or not isinstance(data.get("influences"), list):
                data["influences"] = []

            try:
                return SongConcept(**data)
            except Exception as e:
                print(f"SongConcept validation error: {e}")
                return None

        # Try to extract JSON from various formats
        try:
            # First, try direct parse
            concept_data = json.loads(response_text)
            result = try_parse_concept(concept_data)
            if result:
                return result
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            try:
                concept_data = json.loads(json_match.group(1))
                result = try_parse_concept(concept_data)
                if result:
                    return result
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in text
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                concept_data = json.loads(json_match.group(0))
                result = try_parse_concept(concept_data)
                if result:
                    return result
            except json.JSONDecodeError:
                pass

        # Last resort: create minimal concept from conversation
        if self.conversation_history:
            first_user_msg = ""
            for msg in self.conversation_history:
                if msg["role"] == "user":
                    first_user_msg = msg["content"]
                    break

            return SongConcept(
                idea=first_user_msg[:200] if first_user_msg else "Song concept",
                genre="pop",
                mood="uplifting",
                themes=[],
                influences=[],
            )

        return None

    def reset(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []

    def get_genre_suggestions(self, idea: str) -> list[str]:
        """
        Get genre suggestions for a song idea.

        Args:
            idea: The song idea description

        Returns:
            List of suggested genres
        """
        client = self._get_client()

        prompt = f"""Given this song idea: "{idea}"

Suggest 5 specific genres that would work well. Be specific (e.g., "dream pop with shoegaze influences" not just "pop").

Respond with only a JSON array of strings, no other text:
["genre1", "genre2", "genre3", "genre4", "genre5"]"""

        response = client.messages.create(
            model=default_config.claude_model,  # Use global config for dynamic model selection
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            import json

            return json.loads(response.content[0].text)
        except json.JSONDecodeError:
            return []

    def get_mood_suggestions(self, idea: str, genre: str) -> list[str]:
        """
        Get mood suggestions for a song idea and genre.

        Args:
            idea: The song idea description
            genre: The chosen genre

        Returns:
            List of suggested moods
        """
        client = self._get_client()

        prompt = f"""Given this song:
Idea: "{idea}"
Genre: "{genre}"

Suggest 5 specific moods/emotional tones that would work well.

Respond with only a JSON array of strings, no other text:
["mood1", "mood2", "mood3", "mood4", "mood5"]"""

        response = client.messages.create(
            model=default_config.claude_model,  # Use global config for dynamic model selection
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            import json

            return json.loads(response.content[0].text)
        except json.JSONDecodeError:
            return []
