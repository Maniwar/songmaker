"""Tests for video generator (Ken Burns effects)."""

import pytest

from src.models.schemas import KenBurnsEffect
from src.services.video_generator import VideoGenerator


class TestVideoGenerator:
    @pytest.fixture
    def generator(self):
        return VideoGenerator()

    def test_ken_burns_filter_zoom_in(self, generator):
        filter_str = generator._get_ken_burns_filter(
            effect=KenBurnsEffect.ZOOM_IN,
            duration=5.0,
            resolution="1920x1080",
            fps=30,
        )

        # Should contain zoompan filter
        assert "zoompan" in filter_str
        # Should have 150 frames (5s * 30fps)
        assert "d=150" in filter_str
        # Should have correct resolution
        assert "s=1920x1080" in filter_str
        # Zoom in should use smooth progress-based calculation (positive increment)
        assert "+0.15*on/" in filter_str

    def test_ken_burns_filter_zoom_out(self, generator):
        filter_str = generator._get_ken_burns_filter(
            effect=KenBurnsEffect.ZOOM_OUT,
            duration=5.0,
            resolution="1920x1080",
            fps=30,
        )

        assert "zoompan" in filter_str
        # Zoom out should use smooth progress-based calculation (negative decrement)
        assert "-0.15*on/" in filter_str

    def test_ken_burns_filter_pan_left(self, generator):
        filter_str = generator._get_ken_burns_filter(
            effect=KenBurnsEffect.PAN_LEFT,
            duration=5.0,
            resolution="1920x1080",
            fps=30,
        )

        assert "zoompan" in filter_str
        # Pan left should have x position calculation
        assert "1-on/" in filter_str

    def test_ken_burns_filter_pan_right(self, generator):
        filter_str = generator._get_ken_burns_filter(
            effect=KenBurnsEffect.PAN_RIGHT,
            duration=5.0,
            resolution="1920x1080",
            fps=30,
        )

        assert "zoompan" in filter_str
        # Pan right should have x position calculation
        assert "*on/" in filter_str

    def test_ken_burns_filter_pan_up(self, generator):
        filter_str = generator._get_ken_burns_filter(
            effect=KenBurnsEffect.PAN_UP,
            duration=5.0,
            resolution="1920x1080",
            fps=30,
        )

        assert "zoompan" in filter_str
        # Pan up should modify y position
        assert "y=" in filter_str
        assert "1-on/" in filter_str

    def test_ken_burns_filter_pan_down(self, generator):
        filter_str = generator._get_ken_burns_filter(
            effect=KenBurnsEffect.PAN_DOWN,
            duration=5.0,
            resolution="1920x1080",
            fps=30,
        )

        assert "zoompan" in filter_str
        # Pan down should modify y position
        assert "y=" in filter_str
        assert "*on/" in filter_str

    def test_all_effects_generate_valid_filters(self, generator):
        """Ensure all Ken Burns effects generate valid FFmpeg filters."""
        for effect in KenBurnsEffect:
            filter_str = generator._get_ken_burns_filter(
                effect=effect,
                duration=10.0,
                resolution="1280x720",
                fps=24,
            )

            # All should have basic zoompan components
            assert "zoompan" in filter_str
            assert "z=" in filter_str
            assert "x=" in filter_str
            assert "y=" in filter_str
            assert "d=" in filter_str
            assert "s=" in filter_str
            assert "fps=" in filter_str

    def test_frame_calculation(self, generator):
        """Test that frame count is calculated correctly."""
        filter_str = generator._get_ken_burns_filter(
            effect=KenBurnsEffect.ZOOM_IN,
            duration=10.0,
            resolution="1920x1080",
            fps=24,
        )

        # 10 seconds * 24 fps = 240 frames
        assert "d=240" in filter_str

    def test_different_resolutions(self, generator):
        """Test with different video resolutions."""
        resolutions = ["1920x1080", "1280x720", "3840x2160"]

        for resolution in resolutions:
            filter_str = generator._get_ken_burns_filter(
                effect=KenBurnsEffect.ZOOM_IN,
                duration=5.0,
                resolution=resolution,
                fps=30,
            )

            assert f"s={resolution}" in filter_str
