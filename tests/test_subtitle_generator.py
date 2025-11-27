"""Tests for subtitle generator."""

import tempfile
from pathlib import Path

import pytest

from src.models.schemas import Word, Transcript, Segment
from src.services.subtitle_generator import SubtitleGenerator, format_ass_time


class TestFormatAssTime:
    def test_simple_time(self):
        assert format_ass_time(0.0) == "0:00:00.00"

    def test_seconds(self):
        assert format_ass_time(5.5) == "0:00:05.50"

    def test_minutes(self):
        assert format_ass_time(65.25) == "0:01:05.25"

    def test_hours(self):
        assert format_ass_time(3665.5) == "1:01:05.50"


class TestSubtitleGenerator:
    @pytest.fixture
    def generator(self):
        return SubtitleGenerator()

    @pytest.fixture
    def sample_words(self):
        return [
            Word(word="Hello", start=0.0, end=0.5),
            Word(word="world", start=0.6, end=1.0),
            Word(word="this", start=1.2, end=1.4),
            Word(word="is", start=1.5, end=1.6),
            Word(word="a", start=1.7, end=1.8),
            Word(word="test.", start=1.9, end=2.5),
        ]

    @pytest.fixture
    def sample_transcript(self, sample_words):
        return Transcript(
            segments=[
                Segment(
                    text="Hello world this is a test.",
                    start=0.0,
                    end=2.5,
                    words=sample_words,
                )
            ],
            duration=3.0,
        )

    def test_group_words_into_lines(self, generator, sample_words):
        lines = generator._group_words_into_lines(sample_words, max_words=3)

        # Should create multiple lines
        assert len(lines) >= 2
        # First line should have <= 3 words
        assert len(lines[0]) <= 3

    def test_group_words_breaks_at_punctuation(self, generator):
        words = [
            Word(word="Hello.", start=0.0, end=0.5),
            Word(word="World", start=0.6, end=1.0),
        ]

        lines = generator._group_words_into_lines(words, max_words=8)

        # Should break after punctuation
        assert len(lines) == 2
        assert lines[0][0].word == "Hello."

    def test_generate_karaoke_ass(self, generator, sample_words):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.ass"

            result = generator.generate_karaoke_ass(
                words=sample_words,
                output_path=output_path,
            )

            assert result.exists()

            content = result.read_text()

            # Check header
            assert "[Script Info]" in content
            assert "[V4+ Styles]" in content
            assert "[Events]" in content

            # Check dialogue lines exist
            assert "Dialogue:" in content

    def test_generate_from_transcript(self, generator, sample_transcript):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.ass"

            result = generator.generate_from_transcript(
                transcript=sample_transcript,
                output_path=output_path,
            )

            assert result.exists()

    def test_generate_simple_srt(self, generator, sample_words):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.srt"

            result = generator.generate_simple_srt(
                words=sample_words,
                output_path=output_path,
            )

            assert result.exists()

            content = result.read_text()

            # Check SRT format
            assert "1\n" in content
            assert "-->" in content
            assert "Hello" in content

    def test_srt_time_format(self, generator):
        time_str = generator._format_srt_time(65.5)
        assert time_str == "00:01:05,500"
