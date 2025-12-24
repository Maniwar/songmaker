"""Lyrics Agent for generating song lyrics with Suno-compatible tags."""

from typing import Optional

import anthropic

from src.config import Config, config as default_config
from src.models.schemas import SongConcept, GeneratedLyrics


SYSTEM_PROMPT = """You are an expert songwriter who creates compelling, emotionally resonant lyrics. You also understand how to format lyrics for AI music generation platforms like Suno.

When generating lyrics:
1. Create authentic, meaningful lyrics that match the concept
2. Use natural language that sounds good when sung
3. Include appropriate song structure markers
4. Keep verses cohesive and choruses memorable
5. Match the mood and themes specified

Song structure markers to use:
[Intro], [Verse 1], [Verse 2], [Chorus], [Pre-Chorus], [Bridge], [Outro]
[Instrumental], [Guitar Solo], [Break]

Suno style tags format (for the metatags field):
- Genre tags: indie rock, dream pop, electronic, etc.
- Mood tags: melancholic, uplifting, energetic, etc.
- Instrument tags: acoustic guitar, synth, piano, etc.
- Vocal tags: male vocals, female vocals, ethereal vocals, etc.
- Other: fast tempo, slow tempo, 120 bpm, etc.

Example style tags: "indie folk, acoustic guitar, female vocals, melancholic, slow tempo"

Always provide:
1. A title for the song
2. Complete lyrics with structure markers
3. Suno-compatible style tags
4. The structure breakdown"""


class LyricsAgent:
    """Agent for generating song lyrics with Suno tags."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config
        self._client = None

    def _get_client(self) -> anthropic.Anthropic:
        """Lazy load Anthropic client."""
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
        return self._client

    def generate_lyrics(self, concept: SongConcept) -> GeneratedLyrics:
        """
        Generate lyrics based on a song concept.

        Args:
            concept: The finalized song concept

        Returns:
            GeneratedLyrics with title, lyrics, and Suno tags
        """
        client = self._get_client()

        prompt = f"""Generate complete song lyrics based on this concept:

Song Idea: {concept.idea}
Genre: {concept.genre}
Mood: {concept.mood}
Themes: {', '.join(concept.themes)}
Tempo: {concept.tempo or 'medium'}
Influences: {', '.join(concept.influences) if concept.influences else 'None specified'}

Please provide:
1. A compelling title
2. Complete lyrics with structure markers ([Verse 1], [Chorus], etc.)
3. Suno-compatible style tags (comma-separated)
4. Structure breakdown (list of sections)

Respond in this exact JSON format:
{{
    "title": "Song Title",
    "lyrics": "[Verse 1]\\nFirst verse lyrics...\\n\\n[Chorus]\\nChorus lyrics...",
    "suno_tags": "genre, mood, instrument, vocal style, tempo",
    "structure": ["Verse 1", "Chorus", "Verse 2", "Chorus", "Bridge", "Chorus", "Outro"]
}}

Only respond with the JSON, no other text."""

        response = client.messages.create(
            model=default_config.claude_model,  # Use global config for dynamic model selection
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            import json

            lyrics_data = json.loads(response.content[0].text)
            return GeneratedLyrics(**lyrics_data)
        except (json.JSONDecodeError, ValueError) as e:
            # Return a default structure if parsing fails
            return GeneratedLyrics(
                title="Untitled",
                lyrics=response.content[0].text,
                suno_tags=f"{concept.genre}, {concept.mood}",
                structure=["Verse", "Chorus"],
            )

    def refine_lyrics(
        self,
        current_lyrics: GeneratedLyrics,
        feedback: str,
    ) -> GeneratedLyrics:
        """
        Refine lyrics based on user feedback.

        Args:
            current_lyrics: The current lyrics to refine
            feedback: User's feedback on what to change

        Returns:
            Refined GeneratedLyrics
        """
        client = self._get_client()

        prompt = f"""Here are the current lyrics:

Title: {current_lyrics.title}

{current_lyrics.lyrics}

Style Tags: {current_lyrics.suno_tags}

User feedback: {feedback}

Please refine the lyrics based on this feedback. Keep the same JSON format:
{{
    "title": "Song Title",
    "lyrics": "Complete lyrics with structure markers",
    "suno_tags": "updated style tags if needed",
    "structure": ["list", "of", "sections"]
}}

Only respond with the JSON, no other text."""

        response = client.messages.create(
            model=default_config.claude_model,  # Use global config for dynamic model selection
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            import json

            lyrics_data = json.loads(response.content[0].text)
            return GeneratedLyrics(**lyrics_data)
        except (json.JSONDecodeError, ValueError):
            return current_lyrics

    def generate_suno_prompt(self, lyrics: GeneratedLyrics) -> str:
        """
        Generate a ready-to-paste prompt for Suno.

        Args:
            lyrics: The generated lyrics

        Returns:
            Formatted string for Suno
        """
        return f"""Style: {lyrics.suno_tags}

{lyrics.lyrics}"""
