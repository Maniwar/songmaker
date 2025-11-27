"""Tests for visual agent scene planning."""

import pytest

from src.models.schemas import Word, Segment, Transcript, SongConcept

try:
    from src.agents.visual_agent import VisualAgent
    SKIP_AGENT_TESTS = False
except ImportError:
    SKIP_AGENT_TESTS = True
    VisualAgent = None


pytestmark = pytest.mark.skipif(
    SKIP_AGENT_TESTS, reason="anthropic library not installed"
)


class TestVisualAgent:
    @pytest.fixture
    def agent(self):
        return VisualAgent()

    @pytest.fixture
    def sample_transcript(self):
        """Create a sample 60-second transcript."""
        words = []
        segments = []

        # Generate words spread over 60 seconds
        for i in range(100):
            start = i * 0.5
            end = start + 0.4
            word = Word(word=f"word{i}", start=start, end=end)
            words.append(word)

        # Group into segments of 10 words each
        for i in range(0, len(words), 10):
            segment_words = words[i : i + 10]
            if segment_words:
                segments.append(
                    Segment(
                        text=" ".join(w.word for w in segment_words),
                        start=segment_words[0].start,
                        end=segment_words[-1].end,
                        words=segment_words,
                    )
                )

        return Transcript(
            segments=segments,
            language="en",
            duration=60.0,
        )

    @pytest.fixture
    def sample_concept(self):
        return SongConcept(
            idea="A song about testing",
            genre="indie rock",
            mood="energetic",
            themes=["code", "testing"],
            visual_style="cinematic, dramatic lighting",
        )

    def test_calculate_scene_boundaries(self, agent, sample_transcript):
        boundaries = agent.calculate_scene_boundaries(
            duration=60.0,
            transcript=sample_transcript,
            min_scene_duration=4.0,
            max_scene_duration=12.0,
            target_scenes_per_minute=4.0,
        )

        # Should have approximately 4 scenes per minute for 60s = ~4 scenes
        assert len(boundaries) >= 3
        assert len(boundaries) <= 15

        # First scene should start at 0
        assert boundaries[0][0] == 0.0

        # Last scene should end at or near song duration
        assert boundaries[-1][1] >= 59.0

        # No gaps between scenes
        for i in range(len(boundaries) - 1):
            assert boundaries[i][1] == boundaries[i + 1][0]

    def test_calculate_scene_boundaries_short_song(self, agent):
        """Test with a very short song."""
        words = [Word(word=f"w{i}", start=i * 0.5, end=i * 0.5 + 0.3) for i in range(10)]
        transcript = Transcript(
            segments=[
                Segment(
                    text=" ".join(w.word for w in words),
                    start=0.0,
                    end=5.0,
                    words=words,
                )
            ],
            duration=5.0,
        )

        boundaries = agent.calculate_scene_boundaries(
            duration=5.0,
            transcript=transcript,
        )

        # Should have at least 1 scene
        assert len(boundaries) >= 1

        # Scene should cover the whole song
        assert boundaries[0][0] == 0.0
        assert boundaries[-1][1] == 5.0

    def test_find_break_points(self, agent):
        """Test finding natural breaks between words."""
        words = [
            Word(word="hello", start=0.0, end=0.3),
            Word(word="there", start=0.35, end=0.6),  # Small gap
            Word(word="world", start=1.2, end=1.5),  # Large gap
            Word(word="test", start=1.55, end=1.8),  # Small gap
        ]

        breaks = agent._find_break_points(words)

        # Should find the large gap around 0.9
        assert len(breaks) >= 1
        assert any(0.7 < b < 1.1 for b in breaks)

    def test_find_nearest_break(self, agent):
        break_points = [5.0, 10.0, 15.0, 20.0]

        # Should find nearest within tolerance
        assert agent._find_nearest_break(break_points, 9.5, tolerance=2.0) == 10.0

        # Should return None if no break within tolerance
        assert agent._find_nearest_break(break_points, 12.5, tolerance=1.0) is None

        # Empty list should return None
        assert agent._find_nearest_break([], 10.0, tolerance=2.0) is None

    def test_scene_coverage_no_gaps(self, agent, sample_transcript):
        """Ensure scenes cover entire song with no gaps."""
        boundaries = agent.calculate_scene_boundaries(
            duration=60.0,
            transcript=sample_transcript,
        )

        # Check coverage
        total_covered = sum(end - start for start, end, _ in boundaries)
        assert total_covered >= 59.5  # Allow small rounding errors

        # Check no gaps
        for i in range(len(boundaries) - 1):
            _, end, _ = boundaries[i]
            next_start, _, _ = boundaries[i + 1]
            assert abs(end - next_start) < 0.01  # Allow tiny floating point errors
