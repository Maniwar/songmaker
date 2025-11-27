"""Tests for data models."""

import pytest

from src.models.schemas import (
    Word,
    Segment,
    Transcript,
    SongConcept,
    GeneratedLyrics,
    Scene,
    KenBurnsEffect,
    WorkflowStep,
    AppState,
)


class TestWord:
    def test_word_creation(self):
        word = Word(word="hello", start=1.0, end=1.5)
        assert word.word == "hello"
        assert word.start == 1.0
        assert word.end == 1.5

    def test_word_duration(self):
        word = Word(word="test", start=0.0, end=0.5)
        assert word.duration == 0.5


class TestSegment:
    def test_segment_creation(self):
        words = [
            Word(word="hello", start=0.0, end=0.3),
            Word(word="world", start=0.4, end=0.8),
        ]
        segment = Segment(
            text="hello world",
            start=0.0,
            end=0.8,
            words=words,
        )
        assert segment.text == "hello world"
        assert len(segment.words) == 2

    def test_segment_duration(self):
        segment = Segment(text="test", start=1.0, end=2.5, words=[])
        assert segment.duration == 1.5


class TestTranscript:
    def test_transcript_all_words(self):
        words1 = [Word(word="hello", start=0.0, end=0.3)]
        words2 = [Word(word="world", start=0.5, end=0.8)]

        transcript = Transcript(
            segments=[
                Segment(text="hello", start=0.0, end=0.3, words=words1),
                Segment(text="world", start=0.5, end=0.8, words=words2),
            ],
            duration=1.0,
        )

        all_words = transcript.all_words
        assert len(all_words) == 2
        assert all_words[0].word == "hello"
        assert all_words[1].word == "world"

    def test_transcript_full_text(self):
        transcript = Transcript(
            segments=[
                Segment(text="hello", start=0.0, end=0.3, words=[]),
                Segment(text="world", start=0.5, end=0.8, words=[]),
            ],
            duration=1.0,
        )

        assert transcript.full_text == "hello world"


class TestSongConcept:
    def test_concept_creation(self):
        concept = SongConcept(
            idea="A song about coding",
            genre="indie electronic",
            mood="uplifting",
            themes=["technology", "creativity"],
        )

        assert concept.idea == "A song about coding"
        assert concept.genre == "indie electronic"
        assert len(concept.themes) == 2


class TestGeneratedLyrics:
    def test_lyrics_creation(self):
        lyrics = GeneratedLyrics(
            title="Code Dreams",
            lyrics="[Verse 1]\nWriting code all night...",
            suno_tags="indie electronic, uplifting, synth",
            structure=["Verse 1", "Chorus"],
        )

        assert lyrics.title == "Code Dreams"
        assert "[Verse 1]" in lyrics.lyrics


class TestScene:
    def test_scene_creation(self):
        scene = Scene(
            index=0,
            start_time=0.0,
            end_time=5.0,
            visual_prompt="A sunset over the ocean",
            mood="peaceful",
            effect=KenBurnsEffect.ZOOM_IN,
        )

        assert scene.duration == 5.0
        assert scene.effect == KenBurnsEffect.ZOOM_IN

    def test_scene_with_words(self):
        words = [
            Word(word="sunset", start=1.0, end=1.5),
            Word(word="dreams", start=2.0, end=2.5),
        ]

        scene = Scene(
            index=0,
            start_time=0.0,
            end_time=5.0,
            visual_prompt="test",
            mood="test",
            words=words,
        )

        assert len(scene.words) == 2


class TestWorkflowStep:
    def test_workflow_steps_exist(self):
        """Test that all workflow steps are defined."""
        steps = list(WorkflowStep)
        assert len(steps) == 5
        assert WorkflowStep.CONCEPT in steps
        assert WorkflowStep.LYRICS in steps
        assert WorkflowStep.UPLOAD in steps
        assert WorkflowStep.GENERATE in steps
        assert WorkflowStep.COMPLETE in steps


class TestAppState:
    def test_initial_state(self):
        state = AppState()
        assert state.current_step == WorkflowStep.CONCEPT
        assert state.song_idea == ""
        assert state.concept is None
        assert state.lyrics is None
        assert state.audio_path is None
        assert len(state.scenes) == 0

    def test_state_update(self):
        state = AppState()
        state.song_idea = "A song about rain"
        state.current_step = WorkflowStep.LYRICS

        assert state.song_idea == "A song about rain"
        assert state.current_step == WorkflowStep.LYRICS
