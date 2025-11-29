"""Subtitle generation service for karaoke-style lyrics."""

from pathlib import Path
from typing import Optional

from src.config import Config, config as default_config
from src.models.schemas import Word, Transcript


def format_ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp format (H:MM:SS.CC)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    # ASS uses centiseconds
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


class SubtitleGenerator:
    """Generate ASS subtitle files with karaoke highlighting."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config

    def generate_karaoke_ass(
        self,
        words: list[Word],
        output_path: Path,
        font_name: str = "Arial",
        font_size: int = 48,
        primary_color: str = "&H00FFFFFF",
        highlight_color: str = "&H0000FFFF",
        outline_color: str = "&H00000000",
        back_color: str = "&H80000000",
        max_words_per_line: int = 8,
    ) -> Path:
        """
        Generate ASS subtitle file with word-by-word karaoke highlighting.

        Args:
            words: List of Word objects with timing information
            output_path: Path to save the ASS file
            font_name: Font to use for subtitles
            font_size: Font size in points
            primary_color: Primary text color (AABBGGRR format)
            highlight_color: Color when word is being sung
            outline_color: Outline color
            back_color: Background/shadow color
            max_words_per_line: Maximum words per line for readability

        Returns:
            Path to the generated ASS file
        """
        video_width = self.config.video.width
        video_height = self.config.video.height

        # For karaoke: PrimaryColour = highlighted (sung) color, SecondaryColour = base (not yet sung)
        # \kf effect transitions from SecondaryColour to PrimaryColour
        header = f"""[Script Info]
Title: Karaoke Lyrics
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{highlight_color},{primary_color},{outline_color},{back_color},-1,0,0,0,100,100,0,0,1,3,2,2,20,20,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        # Group words into lines (returns tuples of (words, followed_by_gap))
        lines = self._group_words_into_lines(words, max_words_per_line)

        # Configuration for better readability
        lead_time = 0.3  # Show line before first word starts
        min_line_duration = 0.8  # Minimum time a line should be visible
        line_gap = 0.05  # Gap between consecutive lines
        gap_buffer = 0.5  # How long to show line after last word when followed by gap

        events = []
        for i, (line_words, followed_by_gap) in enumerate(lines):
            if not line_words:
                continue

            first_word_start = line_words[0].start
            last_word_end = line_words[-1].end

            # Line appears BEFORE first word starts (lead time)
            line_start = max(0, first_word_start - lead_time)

            # Calculate end time based on whether there's a gap after this line
            if followed_by_gap:
                # Line is followed by instrumental/silence - disappear shortly after last word
                line_end = last_word_end + gap_buffer
            else:
                # Normal case - calculate based on next line timing
                desired_end = last_word_end + 0.3

                # Find when next line's first word starts (if any)
                next_line_first_word = None
                if i < len(lines) - 1:
                    next_line_words, _ = lines[i + 1]
                    if next_line_words:
                        next_line_first_word = next_line_words[0].start

                if next_line_first_word is not None:
                    # End before next line needs to appear (accounting for its lead time)
                    max_end = next_line_first_word - lead_time - line_gap
                    line_end = min(desired_end, max_end)
                    # But ensure we at least show until the last word ends
                    line_end = max(line_end, last_word_end + 0.05)
                else:
                    # Last line gets full buffer
                    line_end = desired_end

            # Ensure minimum duration for readability (but not if followed by gap)
            if not followed_by_gap and line_end - line_start < min_line_duration:
                line_end = line_start + min_line_duration
                # Re-check overlap constraint
                if i < len(lines) - 1:
                    next_line_words, _ = lines[i + 1]
                    if next_line_words:
                        next_line_first_word = next_line_words[0].start
                        max_allowed = next_line_first_word - lead_time - line_gap
                        if line_end > max_allowed:
                            line_end = max_allowed

            # Build karaoke text with \kf (karaoke fill) tags
            # \kf<duration> fills the word over the duration (in centiseconds)
            text_parts = []

            # Add lead time as unhighlighted pause at start
            # This gives viewers time to see the line before singing starts
            lead_cs = int((first_word_start - line_start) * 100)
            if lead_cs > 0:
                text_parts.append(f"{{\\k{lead_cs}}}")

            prev_end = first_word_start

            for j, word in enumerate(line_words):
                # Calculate delay before this word starts (in centiseconds)
                delay_cs = int((word.start - prev_end) * 100)
                if delay_cs < 0:
                    delay_cs = 0

                # Duration of the word itself (in centiseconds)
                word_duration_cs = int((word.end - word.start) * 100)
                if word_duration_cs < 10:
                    word_duration_cs = 10  # Minimum 0.1s

                # Add delay (unhighlighted time before word)
                if delay_cs > 0:
                    text_parts.append(f"{{\\k{delay_cs}}}")

                # Add the word with karaoke fill effect
                text_parts.append(f"{{\\kf{word_duration_cs}}}{word.word} ")

                prev_end = word.end

            line_text = "".join(text_parts).strip()

            # Format timestamps
            start_ts = format_ass_time(line_start)
            end_ts = format_ass_time(line_end)

            events.append(
                f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{line_text}"
            )

        # Write file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header + "\n".join(events))

        return output_path

    def _group_words_into_lines(
        self,
        words: list[Word],
        max_words: int,
        gap_threshold: float = 1.5,
    ) -> list[tuple[list[Word], bool]]:
        """
        Group words into lines for subtitle display.

        Tries to break at natural points (punctuation), max_words limit,
        or when there's a significant time gap between words.

        Args:
            words: List of Word objects with timing
            max_words: Maximum words per line
            gap_threshold: Time gap (seconds) that triggers a line break.
                          Lines followed by gaps will disappear instead of
                          staying on screen until the next line.

        Returns:
            List of (line_words, followed_by_gap) tuples.
            followed_by_gap=True means there's an instrumental/silence after this line.
        """
        if not words:
            return []

        lines = []
        current_line = []

        for i, word in enumerate(words):
            current_line.append(word)

            # Check for natural break points
            is_natural_break = word.word.rstrip().endswith((".", "!", "?", ",", ";"))

            # Check if there's a time gap before the next word
            has_gap_after = False
            if i < len(words) - 1:
                next_word = words[i + 1]
                gap = next_word.start - word.end
                has_gap_after = gap >= gap_threshold

            # Break line if max words reached, natural break, or gap detected
            if len(current_line) >= max_words or is_natural_break or has_gap_after:
                lines.append((current_line, has_gap_after))
                current_line = []

        # Add remaining words (no gap after the last line)
        if current_line:
            lines.append((current_line, False))

        return lines

    def generate_from_transcript(
        self,
        transcript: Transcript,
        output_path: Path,
        **kwargs,
    ) -> Path:
        """
        Generate ASS file from a Transcript object.

        Args:
            transcript: Transcript with word-level timestamps
            output_path: Path to save the ASS file
            **kwargs: Additional arguments passed to generate_karaoke_ass

        Returns:
            Path to the generated ASS file
        """
        all_words = transcript.all_words
        return self.generate_karaoke_ass(all_words, output_path, **kwargs)

    def generate_simple_srt(
        self,
        words: list[Word],
        output_path: Path,
        max_words_per_line: int = 8,
    ) -> Path:
        """
        Generate a simple SRT subtitle file (without karaoke effects).

        Args:
            words: List of Word objects with timing information
            output_path: Path to save the SRT file
            max_words_per_line: Maximum words per line

        Returns:
            Path to the generated SRT file
        """
        lines = self._group_words_into_lines(words, max_words_per_line)

        srt_content = []
        for i, (line_words, followed_by_gap) in enumerate(lines, 1):
            if not line_words:
                continue

            start = line_words[0].start
            # If followed by gap, end sooner; otherwise add buffer
            end = line_words[-1].end + (0.3 if followed_by_gap else 0.5)

            # SRT timestamp format: HH:MM:SS,mmm
            start_ts = self._format_srt_time(start)
            end_ts = self._format_srt_time(end)

            text = " ".join(word.word for word in line_words)

            srt_content.append(f"{i}")
            srt_content.append(f"{start_ts} --> {end_ts}")
            srt_content.append(text)
            srt_content.append("")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_content))

        return output_path

    def _format_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
