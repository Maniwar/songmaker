"""Video Generation page - Step 4 of the workflow."""

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Optional
import threading

import streamlit as st
from pydub import AudioSegment

from src.config import config
from src.agents.visual_agent import VisualAgent
from src.services.audio_processor import AudioProcessor
from src.services.image_generator import ImageGenerator
from src.services.video_generator import VideoGenerator
from src.services.subtitle_generator import SubtitleGenerator
from src.services.lip_sync_animator import LipSyncAnimator, check_lip_sync_available
from src.services.prompt_animator import PromptAnimator, check_prompt_animator_available
from src.services.veo_animator import VeoAnimator, check_veo_available
from src.services.atlascloud_animator import AtlasCloudAnimator, check_atlascloud_available
from src.services.seedance_animator import SeedanceAnimator, check_seedance_available
from src.services.kling_animator import KlingAnimator, check_kling_available
from src.services.wan_s2v_animator import WanS2VAnimator, check_wan_s2v_available
from src.services.animation_chainer import PromptAnimationChainer, VeoAnimationChainer
from src.ui.components.state import get_state, update_state, advance_step, go_to_step, save_scene_metadata
from src.models.schemas import WorkflowStep, Scene, KenBurnsEffect, Word, AnimationType


def _sanitize_filename(name: str, max_length: int = 20) -> str:
    """
    Sanitize a string for use in file/directory names.

    Removes or replaces characters that are problematic in file paths.
    """
    import re
    # Replace spaces with underscores
    sanitized = name.replace(" ", "_")
    # Remove or replace problematic characters: / \ : * ? " < > |
    sanitized = re.sub(r'[/\\:*?"<>|]', '_', sanitized)
    # Remove any other non-alphanumeric characters except underscore and hyphen
    sanitized = re.sub(r'[^\w\-]', '', sanitized)
    # Convert to lowercase and truncate
    return sanitized.lower()[:max_length]


def _get_motion_prompt(scene: Scene) -> str:
    """
    Get a proper short motion prompt for animation.

    Motion prompts should be SHORT action descriptions (under 15 words),
    NOT the full visual prompt which is verbose and meant for image generation.

    Args:
        scene: The scene to get motion prompt for

    Returns:
        A short motion/action prompt for animation
    """
    # Use the scene's motion_prompt if it exists and is not empty
    if scene.motion_prompt and scene.motion_prompt.strip():
        return scene.motion_prompt.strip()

    # Generate a short default based on mood
    mood = getattr(scene, 'mood', 'emotional')
    mood_lower = mood.lower() if mood else 'emotional'

    # Map moods to short action prompts
    mood_prompts = {
        'happy': 'moving joyfully to the music',
        'joyful': 'dancing and moving with energy',
        'sad': 'moving slowly with emotion',
        'melancholy': 'swaying gently with sadness',
        'energetic': 'moving energetically with rhythm',
        'intense': 'moving with intensity and power',
        'romantic': 'moving softly and tenderly',
        'nostalgic': 'swaying gently with reflection',
        'hopeful': 'moving with growing energy',
        'dramatic': 'moving with dramatic expression',
        'peaceful': 'swaying calmly and peacefully',
        'angry': 'moving with forceful intensity',
    }

    # Find matching mood or use default
    for key, prompt in mood_prompts.items():
        if key in mood_lower:
            return prompt

    # Default short motion prompt
    return 'moving naturally to the music'


def _update_scene_lyrics(scene: Scene, new_lyrics: str) -> None:
    """
    Update a scene's lyrics, creating Word objects with proper timing.

    Always creates properly distributed Word objects for all words.
    This ensures each word has its own timing slot for karaoke display.
    """
    new_words_text = new_lyrics.split()
    if not new_words_text:
        scene.words = []
        return

    scene_duration = scene.end_time - scene.start_time

    # Calculate timing for new words
    # Leave small buffer at start and end for readability
    buffer = min(0.1, scene_duration * 0.05)
    usable_duration = scene_duration - (buffer * 2)
    word_duration = usable_duration / len(new_words_text)

    # Ensure minimum word duration for readability
    min_word_duration = 0.15
    if word_duration < min_word_duration:
        word_duration = min_word_duration
        # Adjust buffer if we need more time
        total_word_time = word_duration * len(new_words_text)
        if total_word_time > scene_duration:
            # Too many words - just distribute evenly
            word_duration = scene_duration / len(new_words_text)
            buffer = 0

    # Create new Word objects with proper timing for each word
    new_word_objects = []
    for i, word_text in enumerate(new_words_text):
        start = scene.start_time + buffer + (i * word_duration)
        # Leave small gap between words (5% of word duration)
        gap = word_duration * 0.05
        end = start + word_duration - gap
        # Ensure end doesn't exceed scene end
        end = min(end, scene.end_time - 0.01)
        new_word_objects.append(Word(word=word_text, start=start, end=end))

    scene.words = new_word_objects


def _resync_single_scene_lyrics(state, scene_index: int) -> None:
    """
    Re-sync lyrics for a single scene from the original transcript.

    Args:
        state: App state with transcript and scenes
        scene_index: Index of the scene to re-sync
    """
    if not state.transcript or not state.scenes:
        return

    if scene_index >= len(state.scenes):
        return

    all_words = state.transcript.all_words
    scene = state.scenes[scene_index]

    # Find all words that fall within this scene's time range
    scene_words = [
        Word(word=w.word, start=w.start, end=w.end)
        for w in all_words
        if w.start >= scene.start_time and w.end <= scene.end_time
    ]

    # Also include words that overlap significantly with the scene
    for w in all_words:
        # Word starts before scene but ends within it
        if w.start < scene.start_time and w.end > scene.start_time:
            overlap = min(w.end, scene.end_time) - scene.start_time
            word_duration = w.end - w.start
            if overlap > word_duration * 0.5:  # >50% overlap
                if not any(sw.word == w.word and abs(sw.start - w.start) < 0.1 for sw in scene_words):
                    scene_words.insert(0, Word(word=w.word, start=max(w.start, scene.start_time), end=w.end))

        # Word starts within scene but ends after it
        if w.start >= scene.start_time and w.start < scene.end_time and w.end > scene.end_time:
            overlap = scene.end_time - w.start
            word_duration = w.end - w.start
            if overlap > word_duration * 0.5:  # >50% overlap
                if not any(sw.word == w.word and abs(sw.start - w.start) < 0.1 for sw in scene_words):
                    scene_words.append(Word(word=w.word, start=w.start, end=min(w.end, scene.end_time)))

    # Sort by start time
    scene_words.sort(key=lambda w: w.start)
    state.scenes[scene_index].words = scene_words
    update_state(scenes=state.scenes)

    # Clear cached lyrics text area values so they refresh with new words
    # Two key conventions are used in the UI:
    # - Prompt review uses loop index: key=f"lyrics_{i}" where i is list position
    # - Storyboard cards use scene.index: key=f"card_lyrics_{scene.index}"
    # We clear both to ensure consistency
    scene = state.scenes[scene_index]
    lyrics_keys_to_clear = [
        f"lyrics_{scene_index}",           # prompt review (uses list position as loop index)
        f"card_lyrics_{scene.index}",      # storyboard card (uses scene's internal index)
    ]
    # Also clear using scene.index for prompt review if it differs from list position
    if scene.index != scene_index:
        lyrics_keys_to_clear.append(f"lyrics_{scene.index}")

    for key in lyrics_keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def _resync_lyrics_to_scenes(state) -> None:
    """
    Re-sync lyrics from the original transcript to scene boundaries.

    This reassigns words from the transcript to scenes based on their
    timing, which is useful after scene boundaries have been adjusted
    or if the initial assignment was incorrect.
    """
    if not state.transcript or not state.scenes:
        return

    # Re-sync each scene individually
    for i in range(len(state.scenes)):
        _resync_single_scene_lyrics(state, i)


def _redo_transcription(
    state,
    lyrics_hint: Optional[str] = None,
    use_demucs: Optional[bool] = None,
) -> bool:
    """
    Re-run WhisperX transcription on the audio file.

    Args:
        state: App state with audio path
        lyrics_hint: Optional lyrics text to improve transcription accuracy
        use_demucs: Whether to use Demucs vocal separation. If None, uses config default.

    Returns:
        True if transcription succeeded, False otherwise
    """
    if not state.audio_path or state.audio_path == "demo_mode":
        st.error("Cannot redo transcription in demo mode. Please upload a real audio file.")
        return False

    audio_path = Path(state.audio_path)
    if not audio_path.exists():
        st.error(f"Audio file not found: {audio_path}")
        return False

    processor = AudioProcessor()

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def progress_callback(message: str, progress: float):
        status_text.text(message)
        progress_bar.progress(progress)

    try:
        transcript = processor.transcribe(
            audio_path,
            progress_callback=progress_callback,
            lyrics_hint=lyrics_hint,
            use_demucs=use_demucs,
        )

        # Update state with new transcript
        update_state(
            transcript=transcript,
            audio_duration=transcript.duration,
        )

        # Re-sync lyrics to scenes if scenes exist
        if state.scenes:
            # Get fresh state after update
            new_state = get_state()
            _resync_lyrics_to_scenes(new_state)

        return True

    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return False

    finally:
        processor.cleanup()


def _regenerate_all_motion_prompts(state, progress_bar=None, status_text=None) -> int:
    """
    Regenerate AI motion prompts for all scenes from their visual prompts.

    Uses Claude to batch-convert visual prompts to short motion descriptions.
    This is much faster than image-based analysis since it's text-to-text.

    Args:
        state: App state with scenes
        progress_bar: Optional st.progress() bar to update
        status_text: Optional st.empty() placeholder for status text

    Returns:
        Number of prompts successfully generated
    """
    import anthropic
    from src.config import config

    if not state.scenes:
        return 0

    scenes = list(state.scenes)

    # Filter to scenes with visual prompts
    scenes_with_prompts = [s for s in scenes if s.visual_prompt]
    total_to_process = len(scenes_with_prompts)

    if total_to_process == 0:
        return 0

    if progress_bar:
        progress_bar.progress(0.1)
    if status_text:
        status_text.text(f"Generating motion prompts for {total_to_process} scenes...")

    # Build batch request - send all visual prompts at once
    prompt_list = "\n".join([
        f"Scene {s.index + 1}: {s.visual_prompt}"
        for s in scenes_with_prompts
    ])

    system_prompt = """You are a motion prompt generator for video animation.
For each scene's visual description, create a SHORT motion prompt (5-10 words max) describing natural movement.

Rules:
- Use present participle verbs (playing, singing, dancing, swaying, flowing)
- Focus on ONE primary motion
- Be specific to what's in the scene
- Keep it SHORT (under 10 words)

Examples:
- Visual: "A bearded man playing guitar in a tavern" → Motion: "strumming guitar while nodding to rhythm"
- Visual: "A forest with sunlight filtering through trees" → Motion: "leaves swaying gently in breeze"
- Visual: "A singer on stage under spotlights" → Motion: "singing passionately with subtle gestures"

Respond with ONLY a numbered list of motion prompts, one per line, matching the scene numbers."""

    user_prompt = f"""Generate short motion prompts for these {total_to_process} scenes:

{prompt_list}

Return ONLY the motion prompts as a numbered list (1. prompt, 2. prompt, etc.)"""

    try:
        client = anthropic.Anthropic(api_key=config.anthropic_api_key)

        if progress_bar:
            progress_bar.progress(0.3)

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        if progress_bar:
            progress_bar.progress(0.7)
        if status_text:
            status_text.text("Parsing motion prompts...")

        # Parse response - expect numbered list
        response_text = response.content[0].text.strip()
        lines = response_text.split("\n")

        success_count = 0
        for i, scene in enumerate(scenes_with_prompts):
            if i < len(lines):
                # Extract motion prompt (remove numbering like "1. " or "1) ")
                line = lines[i].strip()
                # Remove common numbering patterns
                import re
                motion = re.sub(r'^[\d]+[\.\)]\s*', '', line).strip()

                if motion:
                    scenes[scene.index].motion_prompt = motion
                    # Use temp key pattern - will be transferred to widget on next render
                    ai_result_key = f"_ai_motion_result_{scene.index}"
                    st.session_state[ai_result_key] = motion
                    success_count += 1

        # Final progress update
        if progress_bar:
            progress_bar.progress(1.0)
        if status_text:
            status_text.text(f"Done! Generated {success_count}/{total_to_process} motion prompts")

        update_state(scenes=scenes)
        return success_count

    except Exception as e:
        if status_text:
            status_text.error(f"Error: {e}")
        return 0


def _get_selected_scene_indices(state) -> list[int]:
    """
    Get indices of scenes selected for bulk re-animation.

    Returns:
        List of scene indices that have their selection checkbox checked
    """
    if not state.scenes:
        return []

    selected = []
    for scene in state.scenes:
        checkbox_key = f"reanimate_select_{scene.index}"
        if st.session_state.get(checkbox_key, False):
            selected.append(scene.index)
    return selected


def _clear_selected_scene_checkboxes(state) -> None:
    """Clear all scene selection checkboxes.

    Note: In Streamlit, you cannot modify session state for a widget key after
    the widget has been rendered. Instead, we delete the key so when the widget
    re-renders it uses its default value (False).

    Important: Caller should call st.rerun() after this if they want the UI to update.
    """
    if not state.scenes:
        return

    for scene in state.scenes:
        checkbox_key = f"reanimate_select_{scene.index}"
        if checkbox_key in st.session_state:
            del st.session_state[checkbox_key]


def _redo_scene_transcription(
    state,
    scene_index: int,
    lyrics_hint: Optional[str] = None,
    use_demucs: Optional[bool] = None,
) -> bool:
    """
    Re-run WhisperX transcription on just a single scene's audio clip.

    This extracts the audio for the scene's time range, transcribes it,
    and updates the scene's words with properly offset timestamps.

    Args:
        state: App state with audio path and scenes
        scene_index: Index of the scene to re-transcribe
        lyrics_hint: Optional known lyrics to improve transcription accuracy
        use_demucs: Whether to use Demucs vocal separation (None = use config default)

    Returns:
        True if transcription succeeded, False otherwise
    """
    import tempfile

    if not state.audio_path or state.audio_path == "demo_mode":
        st.error("Cannot redo transcription in demo mode.")
        return False

    if not state.scenes or scene_index >= len(state.scenes):
        st.error("Invalid scene index.")
        return False

    audio_path = Path(state.audio_path)
    if not audio_path.exists():
        st.error(f"Audio file not found: {audio_path}")
        return False

    scene = state.scenes[scene_index]

    # Extract the audio clip for this scene
    try:
        audio = AudioSegment.from_file(str(audio_path))
        start_ms = int(scene.start_time * 1000)
        end_ms = int(scene.end_time * 1000)
        clip = audio[start_ms:end_ms]

        # Save to temp file for transcription
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            clip.export(tmp.name, format="wav")
            tmp_path = Path(tmp.name)

        processor = AudioProcessor()

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def progress_callback(message: str, progress: float):
            status_text.text(message)
            progress_bar.progress(progress)

        try:
            # Transcribe the clip
            transcript = processor.transcribe(
                tmp_path,
                progress_callback=progress_callback,
                lyrics_hint=lyrics_hint,
                use_demucs=use_demucs,
            )

            # Offset all word timestamps by the scene's start time
            scene_words = []
            for word in transcript.all_words:
                scene_words.append(
                    Word(
                        word=word.word,
                        start=word.start + scene.start_time,
                        end=word.end + scene.start_time,
                    )
                )

            # Update the scene's words
            state.scenes[scene_index].words = scene_words
            update_state(scenes=state.scenes)

            # Clear cached lyrics text area values so they refresh with new words
            # Multiple key conventions are used in the UI:
            # - Prompt review uses loop index: key=f"lyrics_{i}" where i is list position
            # - Storyboard cards use scene.index: key=f"card_lyrics_{scene.index}"
            # - Scene transcription options use: key=f"scene_lyrics_hint_{scene.index}"
            # We clear all to ensure consistency
            lyrics_keys_to_clear = [
                f"lyrics_{scene_index}",               # prompt review (uses list position)
                f"card_lyrics_{scene.index}",          # storyboard card (uses scene's internal index)
                f"scene_lyrics_hint_{scene.index}",    # scene transcription options hint
            ]
            # Also clear using scene.index for prompt review if it differs from list position
            if scene.index != scene_index:
                lyrics_keys_to_clear.append(f"lyrics_{scene.index}")

            for key in lyrics_keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]

            return True

        except Exception as e:
            st.error(f"Scene transcription failed: {e}")
            return False

        finally:
            processor.cleanup()
            # Clean up temp file
            if tmp_path.exists():
                tmp_path.unlink()

    except Exception as e:
        st.error(f"Failed to extract audio clip: {e}")
        return False


@st.cache_data(show_spinner=False)
def _get_audio_clip(audio_path: str, start_time: float, end_time: float) -> bytes:
    """Extract an audio clip from the full audio file. Cached for performance."""
    try:
        audio = AudioSegment.from_file(audio_path)
        # Convert to milliseconds
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        clip = audio[start_ms:end_ms]
        # Export to bytes
        buffer = BytesIO()
        clip.export(buffer, format="mp3")
        buffer.seek(0)
        return buffer.read()
    except Exception:
        return None


def _generate_scene_preview_with_lyrics(state, scene: Scene) -> bytes:
    """Generate a video preview for a scene with karaoke subtitles.

    For animated scenes, uses the existing animation video with lyrics overlay.
    For static scenes, creates Ken Burns effect preview.
    """
    from tempfile import NamedTemporaryFile
    import os

    # Generate subtitles for just this scene's words
    subtitle_gen = SubtitleGenerator()
    video_gen = VideoGenerator()

    with NamedTemporaryFile(suffix=".ass", delete=False) as sub_file:
        sub_path = Path(sub_file.name)

    with NamedTemporaryFile(suffix=".mp4", delete=False) as vid_file:
        vid_path = Path(vid_file.name)

    try:
        # Generate subtitle file for this scene's words
        # Note: create_scene_preview will handle time shifting via _shift_subtitles
        if scene.words:
            subtitle_gen.generate_karaoke_ass(
                words=scene.words,
                output_path=sub_path,
            )
        else:
            sub_path = None

        # Check if scene has animation
        has_animation = scene.has_animation

        if has_animation:
            # Use animation video with lyrics overlay
            video_gen.create_animation_preview(
                video_path=Path(scene.video_path),
                audio_path=Path(state.audio_path),
                subtitle_path=sub_path,
                start_time=scene.start_time,
                duration=scene.duration,
                output_path=vid_path,
                resolution=(1280, 720),  # Lower res for preview
            )
        else:
            # Use Ken Burns effect on static image
            video_gen.create_scene_preview(
                image_path=Path(scene.image_path),
                audio_path=Path(state.audio_path),
                subtitle_path=sub_path,
                start_time=scene.start_time,
                duration=scene.duration,
                effect=scene.effect,
                output_path=vid_path,
                resolution=(1280, 720),  # Lower res for preview
            )

        # Read video bytes
        with open(vid_path, "rb") as f:
            video_bytes = f.read()

        return video_bytes
    finally:
        # Cleanup
        if sub_path and sub_path.exists():
            os.unlink(sub_path)
        if vid_path.exists():
            os.unlink(vid_path)


def _adjust_scene_timing(state, scene_index: int, new_start: float, new_end: float) -> None:
    """Adjust scene timing and redistribute words accordingly."""
    scenes = state.scenes
    scene = scenes[scene_index]
    old_start = scene.start_time
    old_end = scene.end_time

    # Update scene timing
    scene.start_time = new_start
    scene.end_time = new_end

    # Redistribute words if they exist
    if scene.words:
        # Scale word timings proportionally
        old_duration = old_end - old_start
        new_duration = new_end - new_start
        if old_duration > 0:
            scale = new_duration / old_duration
            for word in scene.words:
                # Shift and scale
                relative_start = word.start - old_start
                relative_end = word.end - old_start
                word.start = new_start + (relative_start * scale)
                word.end = new_start + (relative_end * scale)

    update_state(scenes=scenes)


# Resolution configurations
RESOLUTION_OPTIONS = {
    "1080p": {"width": 1920, "height": 1080, "image_size": "2K"},
    "2K": {"width": 2560, "height": 1440, "image_size": "2K"},
    "4K": {"width": 3840, "height": 2160, "image_size": "4K"},
}

# FPS configurations
FPS_OPTIONS = {
    "24 fps (cinematic)": 24,
    "30 fps (standard)": 30,
    "60 fps (smooth)": 60,
}

# Cinematography style presets with detailed camera/lens/grading descriptions
CINEMATOGRAPHY_STYLES = {
    "Cinematic Film": {
        "description": "Hollywood blockbuster look",
        "prompt": (
            "Shot on ARRI Alexa with Cooke S4 anamorphic lenses, 2.39:1 aspect ratio feel, "
            "teal and orange color grading, shallow depth of field, subtle film grain, "
            "dramatic three-point lighting, cinematic lens flares, "
            "professional color science with rich shadows and controlled highlights"
        ),
    },
    "Music Video": {
        "description": "Dynamic MTV-style visuals",
        "prompt": (
            "High-energy music video aesthetic, vibrant saturated colors, high contrast, "
            "stylized dramatic lighting with colored gels, dynamic compositions, "
            "bold visual style, punchy color grading, studio lighting setups, "
            "clean sharp focus, contemporary professional look"
        ),
    },
    "Vintage Film": {
        "description": "Classic 35mm film look",
        "prompt": (
            "Shot on 35mm Kodak Vision3 500T film stock, warm analog color palette, "
            "visible film grain texture, slightly faded blacks, subtle color shifts, "
            "natural halation on highlights, vintage lens characteristics with soft edges, "
            "nostalgic golden hour warmth, organic film imperfections"
        ),
    },
    "Anime/Animation": {
        "description": "Japanese animation style",
        "prompt": (
            "High-quality anime art style, cel-shaded rendering, vibrant saturated colors, "
            "clean bold linework, dramatic manga-inspired compositions, "
            "detailed anime backgrounds with atmospheric perspective, "
            "expressive character art, Studio Ghibli and Makoto Shinkai inspired visuals"
        ),
    },
    "Documentary": {
        "description": "Natural authentic look",
        "prompt": (
            "Documentary cinematography style, natural available lighting, "
            "shallow depth of field with bokeh, neutral realistic color grading, "
            "intimate handheld camera feel, authentic candid compositions, "
            "Sony Venice or Canon C500 look, observational cinema verite aesthetic"
        ),
    },
    "Film Noir": {
        "description": "Classic noir shadows",
        "prompt": (
            "Classic film noir cinematography, high contrast black and white or desaturated, "
            "dramatic chiaroscuro lighting with deep shadows, venetian blind shadows, "
            "fog and atmospheric haze, low-key lighting setups, "
            "1940s Hollywood noir aesthetic, mysterious moody atmosphere"
        ),
    },
    "Ethereal/Dreamy": {
        "description": "Soft dreamlike quality",
        "prompt": (
            "Ethereal dreamy cinematography, soft diffused lighting, "
            "pastel color palette with lifted shadows, subtle lens flare and bloom, "
            "shallow focus with creamy bokeh, overexposed highlights, "
            "hazy atmospheric glow, romantic soft focus effects, magical realism"
        ),
    },
    "Neon Cyberpunk": {
        "description": "Blade Runner neon aesthetic",
        "prompt": (
            "Cyberpunk neon aesthetic, vibrant pink/cyan/purple neon lighting, "
            "rain-slicked reflective surfaces, atmospheric fog with colored light, "
            "high contrast dark scenes with neon accents, Blade Runner 2049 inspired, "
            "futuristic urban nightscapes, Roger Deakins cinematography style"
        ),
    },
    "Golden Hour": {
        "description": "Warm sunset lighting",
        "prompt": (
            "Golden hour magic hour cinematography, warm orange and amber tones, "
            "long dramatic shadows, lens flare from sun, backlit subjects with rim light, "
            "Terrence Malick inspired natural beauty, soft warm color grading, "
            "romantic sunset atmosphere, Emmanuel Lubezki natural lighting"
        ),
    },
    "High Fashion": {
        "description": "Editorial glamour",
        "prompt": (
            "High fashion editorial photography style, controlled studio lighting, "
            "beauty dish and softbox setups, crisp sharp focus, "
            "clean minimalist compositions, high-end commercial aesthetic, "
            "perfect skin tones, luxury brand visual language, Vogue magazine quality"
        ),
    },
    "Custom": {
        "description": "Define your own style",
        "prompt": "",  # User provides their own
    },
}


def render_generate_page() -> None:
    """Render the video generation page."""
    state = get_state()

    st.header("Generate Your Music Video")

    # Check prerequisites
    if not state.transcript:
        st.warning("Please upload and process your audio first.")
        if st.button("Go to Upload"):
            go_to_step(WorkflowStep.UPLOAD)
            st.rerun()
        return

    # Check if demo mode
    is_demo_mode = state.audio_path == "demo_mode"

    if is_demo_mode:
        st.warning("**Demo Mode** - Video will be generated with images only (no audio)")

    # Use getattr for backwards compatibility with old session states
    prompts_ready = getattr(state, 'prompts_ready', False)
    storyboard_ready = getattr(state, 'storyboard_ready', False)
    project_dir = getattr(state, 'project_dir', None)

    # Workflow: Setup -> Prompt Review -> Image Generation -> Video
    if state.final_video_path:
        render_video_complete(state)
    elif storyboard_ready and state.scenes:
        render_storyboard_view(state, is_demo_mode)
    elif prompts_ready and state.scenes:
        render_prompt_review(state, is_demo_mode)
    else:
        render_storyboard_setup(state, is_demo_mode)


def render_storyboard_setup(state, is_demo_mode: bool) -> None:
    """Render the initial setup to generate storyboard."""
    st.markdown(
        f"""
        Ready to create your music video!

        - **Song:** {state.lyrics.title if state.lyrics else 'Untitled'}
        - **Duration:** {state.audio_duration:.1f} seconds
        - **Words:** {len(state.transcript.all_words)}
        - **Mode:** {"Demo (images only)" if is_demo_mode else "Full (with audio)"}
        """
    )

    # Generation settings
    with st.expander("Video Settings", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            scenes_per_minute = st.slider(
                "Scenes per minute",
                min_value=2,
                max_value=8,
                value=4,
                help="More scenes = more dynamic video",
                key="scenes_per_minute",
            )

            # Resolution selection
            resolution = st.selectbox(
                "Video Resolution",
                options=list(RESOLUTION_OPTIONS.keys()),
                index=0,
                help="Higher resolution = better quality but slower generation",
                key="video_resolution",
            )

        with col2:
            st.slider(
                "Crossfade duration (s)",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                key="crossfade",
            )

            # FPS selection
            fps_selection = st.selectbox(
                "Frame Rate",
                options=list(FPS_OPTIONS.keys()),
                index=1,  # Default to 30 fps
                help="Higher fps = smoother motion but larger file size",
                key="video_fps",
            )

        # Cinematography style selection
        st.markdown("---")
        st.markdown("**Cinematography Style**")

        # Build options with descriptions
        style_options = [
            f"{name} - {info['description']}"
            for name, info in CINEMATOGRAPHY_STYLES.items()
        ]
        selected_style_display = st.selectbox(
            "Visual Style",
            options=style_options,
            index=0,  # Default to Cinematic Film
            help="Choose a cinematography style preset for consistent visuals",
            key="cinematography_style",
        )
        # Extract just the style name from the display string
        selected_style_name = selected_style_display.split(" - ")[0]

        # Show the style description
        style_info = CINEMATOGRAPHY_STYLES[selected_style_name]
        if selected_style_name != "Custom":
            st.caption(f"*{style_info['prompt'][:100]}...*")

        # Custom style input (only show if Custom is selected)
        if selected_style_name == "Custom":
            st.text_area(
                "Custom Style Description",
                placeholder=(
                    "Describe your visual style, e.g.:\n"
                    "'Shot on RED Komodo, anamorphic flares, "
                    "moody blue/orange grade, shallow DOF...'"
                ),
                key="custom_style",
                height=100,
            )

        # Advanced options
        st.markdown("---")
        st.markdown("**Advanced Options**")

        col3, col4 = st.columns(2)
        with col3:
            show_lyrics = st.checkbox(
                "Show lyrics overlay",
                value=True,
                help="Burn karaoke-style lyrics into the video",
                key="show_lyrics",
            )
        with col4:
            sequential_mode = st.checkbox(
                "Sequential mode (consistency)",
                value=False,
                help="Use each generated image as reference for the next scene. "
                     "This helps maintain character and style consistency but is slower.",
                key="use_sequential_mode",
            )

        # Parallel workers (only when not in sequential mode)
        if not st.session_state.get("use_sequential_mode", False):
            parallel_workers = st.slider(
                "Parallel image workers",
                min_value=1,
                max_value=8,
                value=4,
                help="Number of images to generate simultaneously. Higher = faster but uses more API quota.",
                key="parallel_workers",
            )
        else:
            st.caption("Parallel processing disabled in sequential mode")

    # Generate storyboard button
    if st.button("Generate Scene Prompts", type="primary"):
        # Get cinematography style
        style_display = st.session_state.get(
            "cinematography_style", "Cinematic Film - Hollywood blockbuster look"
        )
        style_name = style_display.split(" - ")[0]

        # Build the style prompt
        if style_name == "Custom":
            style_prompt = st.session_state.get("custom_style", "")
        else:
            style_prompt = CINEMATOGRAPHY_STYLES.get(style_name, {}).get(
                "prompt", ""
            )

        # Save new settings to state
        fps_key = st.session_state.get("video_fps", "30 fps (standard)")
        update_state(
            show_lyrics=st.session_state.get("show_lyrics", True),
            use_sequential_mode=st.session_state.get("use_sequential_mode", False),
            parallel_workers=st.session_state.get("parallel_workers", 4),
            video_fps=FPS_OPTIONS.get(fps_key, 30),
            cinematography_style=style_name,
        )
        generate_scene_prompts(
            state,
            st.session_state.get("scenes_per_minute", 4),
            style_prompt,
            st.session_state.get("video_resolution", "1080p"),
        )

    # Back button
    st.markdown("---")
    if st.button("Back to Upload"):
        go_to_step(WorkflowStep.UPLOAD)
        st.rerun()


def render_prompt_review(state, is_demo_mode: bool) -> None:
    """Render the prompt review and editing page."""
    st.subheader("Review & Edit Scene Prompts")
    st.markdown(
        """
        Review the AI-generated prompts below. You can **edit any prompt** before
        generating images to get exactly the visuals you want.
        """
    )

    # Summary stats
    total_scenes = len(state.scenes)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Scenes", total_scenes)
    with col2:
        st.metric("Duration", f"{state.audio_duration:.1f}s")
    with col3:
        resolution = getattr(state, 'video_resolution', '1080p')
        st.metric("Resolution", resolution)

    # Video settings expander
    with st.expander("Video Settings", expanded=True):
        settings_col1, settings_col2 = st.columns(2)

        with settings_col1:
            # Resolution selection
            current_resolution = getattr(state, 'video_resolution', '1080p')
            resolution_idx = list(RESOLUTION_OPTIONS.keys()).index(current_resolution) if current_resolution in RESOLUTION_OPTIONS else 0
            new_resolution = st.selectbox(
                "Video Resolution",
                options=list(RESOLUTION_OPTIONS.keys()),
                index=resolution_idx,
                help="Higher resolution = better quality but slower generation",
                key="video_resolution_prompt_review",
            )
            if new_resolution != current_resolution:
                update_state(video_resolution=new_resolution)

            # FPS selection
            current_fps = getattr(state, 'video_fps', 30)
            fps_idx = list(FPS_OPTIONS.values()).index(current_fps) if current_fps in FPS_OPTIONS.values() else 1
            fps_selection = st.selectbox(
                "Frame Rate",
                options=list(FPS_OPTIONS.keys()),
                index=fps_idx,
                help="Higher fps = smoother motion but larger file size",
                key="video_fps_prompt_review",
            )
            new_fps = FPS_OPTIONS[fps_selection]
            if new_fps != current_fps:
                update_state(video_fps=new_fps)

        with settings_col2:
            # Crossfade duration
            current_crossfade = getattr(state, 'crossfade', 0.3)
            new_crossfade = st.slider(
                "Crossfade duration (s)",
                min_value=0.0,
                max_value=1.0,
                value=current_crossfade,
                step=0.1,
                key="crossfade_prompt_review",
            )
            if new_crossfade != current_crossfade:
                update_state(crossfade=new_crossfade)

            # Show lyrics toggle
            current_show_lyrics = getattr(state, 'show_lyrics', True)
            new_show_lyrics = st.checkbox(
                "Show lyrics overlay",
                value=current_show_lyrics,
                help="Display karaoke-style lyrics on the video",
                key="show_lyrics_prompt_review",
            )
            if new_show_lyrics != current_show_lyrics:
                update_state(show_lyrics=new_show_lyrics)

            # Extension mode for animated clips (advanced)
            extension_options = {
                "Ken Burns (all scenes)": "all",
                "Ken Burns (end only)": "end_only",
                "No Ken Burns": "none",
            }
            current_ext_mode = getattr(state, 'extension_mode', 'all')
            ext_labels = list(extension_options.keys())
            ext_values = list(extension_options.values())
            ext_idx = ext_values.index(current_ext_mode) if current_ext_mode in ext_values else 0
            new_ext_label = st.selectbox(
                "Animation extension mode",
                options=ext_labels,
                index=ext_idx,
                help="How to fill gaps when animations are shorter than scene duration. "
                     "'End only' packs animations and adds Ken Burns at the end.",
                key="extension_mode_prompt_review",
            )
            new_ext_mode = extension_options[new_ext_label]
            if new_ext_mode != current_ext_mode:
                update_state(extension_mode=new_ext_mode)

        # Image Generation Settings (full width)
        st.markdown("---")
        st.markdown("**Image Generation Settings**")
        gen_col1, gen_col2 = st.columns(2)

        with gen_col1:
            # Sequential mode (character consistency)
            current_sequential = getattr(state, 'use_sequential_mode', False)
            new_sequential = st.checkbox(
                "Sequential mode (character consistency)",
                value=current_sequential,
                help="Use each generated image as reference for the next scene. "
                     "This helps maintain character and style consistency but is slower.",
                key="sequential_mode_prompt_review",
            )
            if new_sequential != current_sequential:
                update_state(use_sequential_mode=new_sequential)

        with gen_col2:
            # Parallel workers (only when not in sequential mode)
            if not new_sequential:
                current_workers = getattr(state, 'parallel_workers', 4)
                new_workers = st.slider(
                    "Parallel image workers",
                    min_value=1,
                    max_value=8,
                    value=current_workers,
                    help="Number of images to generate simultaneously. Higher = faster but uses more API quota.",
                    key="parallel_workers_prompt_review",
                )
                if new_workers != current_workers:
                    update_state(parallel_workers=new_workers)
            else:
                st.caption("Parallel processing disabled in sequential mode")

        # Hero Image Upload (optional - for visual consistency)
        st.markdown("**Hero Image (Optional)**")
        hero_help = (
            "Upload a reference image to maintain visual consistency across scenes. "
            "In parallel mode: used as reference for ALL scenes. "
            "In sequential mode: seeds the first scene."
        )
        st.caption(hero_help)

        hero_col1, hero_col2 = st.columns([2, 1])
        with hero_col1:
            hero_file = st.file_uploader(
                "Upload hero image",
                type=["png", "jpg", "jpeg", "webp"],
                key="hero_image_uploader",
                label_visibility="collapsed",
            )
            if hero_file is not None:
                # Save to project directory
                project_dir = state.project_dir
                if project_dir:
                    hero_path = Path(project_dir) / "hero_image.png"
                    from PIL import Image as PILImage
                    hero_img = PILImage.open(hero_file)
                    hero_path.parent.mkdir(parents=True, exist_ok=True)
                    hero_img.save(str(hero_path))
                    if str(hero_path) != getattr(state, 'hero_image_path', None):
                        update_state(hero_image_path=str(hero_path))
                    st.success(f"Hero image saved!")

        with hero_col2:
            current_hero = getattr(state, 'hero_image_path', None)
            if current_hero and Path(current_hero).exists():
                st.image(current_hero, caption="Current hero image", width=150)
                if st.button("Clear hero image", key="clear_hero_image"):
                    update_state(hero_image_path=None)
                    st.rerun()

    st.markdown("---")

    # Display each scene's prompt with editing capability
    scenes = state.scenes
    modified_scenes = list(scenes)  # Create a copy to track modifications

    for i, scene in enumerate(scenes):
        with st.container():
            col1, col2 = st.columns([1, 4])

            with col1:
                st.markdown(f"**Scene {i + 1}**")
                st.caption(f"{scene.start_time:.1f}s - {scene.end_time:.1f}s")

                # Ken Burns effect selector
                effect_options = [e.value for e in KenBurnsEffect]
                current_effect_idx = effect_options.index(scene.effect.value)
                new_effect = st.selectbox(
                    "Effect",
                    options=effect_options,
                    index=current_effect_idx,
                    key=f"effect_{i}",
                    label_visibility="collapsed",
                )
                if new_effect != scene.effect.value:
                    modified_scenes[i].effect = KenBurnsEffect(new_effect)

                st.caption(f"Mood: {scene.mood}")

            with col2:
                # Audio preview with adjustable start point
                if state.audio_path and state.audio_path != "demo_mode":
                    # Show word-level timing if words exist
                    if scene.words:
                        with st.expander(f"Word timing ({len(scene.words)} words)", expanded=False):
                            # Show all words in a formatted table-like view
                            timing_lines = []
                            for w in scene.words:
                                timing_lines.append(f"**{w.word}** ({w.start:.2f}s - {w.end:.2f}s)")
                            st.markdown(" | ".join(timing_lines))

                    # Per-scene transcription controls
                    trans_col1, trans_col2 = st.columns(2)
                    with trans_col1:
                        if st.button("Redo Transcription", key=f"preview_redo_trans_{i}", type="secondary", help="Re-transcribe just this scene's audio clip"):
                            with st.spinner("Transcribing scene audio..."):
                                if _redo_scene_transcription(state, i):
                                    st.success("Scene transcribed!")
                                    st.rerun()
                    with trans_col2:
                        if st.button("Re-sync Lyrics", key=f"preview_resync_{i}", type="secondary", help="Re-sync from existing transcript"):
                            _resync_single_scene_lyrics(state, i)
                            st.success("Lyrics re-synced!")
                            st.rerun()

                    # Slider to pick start point within scene
                    scene_duration = scene.end_time - scene.start_time

                    # Default to where first word starts (relative to scene start)
                    default_offset = 0.0
                    if scene.words:
                        first_word_offset = scene.words[0].start - scene.start_time
                        default_offset = max(0.0, min(first_word_offset, scene_duration - 0.5))

                    preview_cols = st.columns([2, 3])
                    with preview_cols[0]:
                        preview_start_offset = st.slider(
                            "Start from",
                            min_value=0.0,
                            max_value=max(0.1, scene_duration - 0.5),
                            value=default_offset,
                            step=0.1,
                            key=f"preview_start_{i}",
                        )
                        actual_start = scene.start_time + preview_start_offset
                        st.caption(f"Song: {actual_start:.1f}s")

                    with preview_cols[1]:
                        audio_clip = _get_audio_clip(
                            str(state.audio_path),
                            actual_start,
                            scene.end_time
                        )
                        if audio_clip:
                            st.audio(audio_clip, format="audio/mp3")

                # Editable lyrics for this scene (allow adding if none detected)
                current_lyrics = " ".join(w.word for w in scene.words) if scene.words else ""
                new_lyrics = st.text_area(
                    "Lyrics" + (" (none detected)" if not scene.words else ""),
                    value=current_lyrics,
                    key=f"lyrics_{i}",
                    placeholder="Type lyrics for this scene..." if not scene.words else "",
                    height=80,
                )
                # Update using helper function
                if new_lyrics != current_lyrics and new_lyrics.strip():
                    _update_scene_lyrics(modified_scenes[i], new_lyrics)

                # Editable prompt
                new_prompt = st.text_area(
                    "Visual Prompt",
                    value=scene.visual_prompt,
                    key=f"prompt_{i}",
                    height=100,
                    label_visibility="collapsed",
                )
                if new_prompt != scene.visual_prompt:
                    modified_scenes[i].visual_prompt = new_prompt

        st.markdown("---")

    # Transcription options (in expander to save space)
    with st.expander("Transcription Options", expanded=False):
        lyrics_hint = st.text_area(
            "Lyrics Hint (improves transcription accuracy):",
            value=state.lyrics if hasattr(state, 'lyrics') and state.lyrics else "",
            height=150,
            key="lyrics_hint_input",
            help="Providing the actual lyrics helps WhisperX recognize words more accurately, especially for music with instruments.",
        )
        use_demucs = st.checkbox(
            "Use Demucs vocal separation",
            value=config.use_demucs,
            key="use_demucs_checkbox",
            help="Separates vocals from music before transcription for better word recognition. Takes longer but improves accuracy for songs with heavy instrumentation.",
        )

    # Action buttons
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("Redo Transcription", type="secondary", help="Re-run WhisperX to get new word timestamps"):
            with st.spinner("Re-transcribing audio..."):
                if _redo_transcription(state, lyrics_hint=lyrics_hint if lyrics_hint else None, use_demucs=use_demucs):
                    st.success("Transcription updated! Lyrics re-synced.")
                    st.rerun()

    with col2:
        if st.button("Re-sync Lyrics", type="secondary", help="Re-assign lyrics from transcript to scene boundaries"):
            _resync_lyrics_to_scenes(state)
            st.success("Lyrics re-synced from transcript!")
            st.rerun()

    with col3:
        if st.button("Regenerate Prompts", type="secondary"):
            # Reset and regenerate
            update_state(
                scenes=[],
                prompts_ready=False,
            )
            st.rerun()

    with col4:
        if st.button("Save Changes", type="secondary"):
            # Save any modified prompts/effects
            update_state(scenes=modified_scenes)
            st.success("Changes saved!")
            st.rerun()

    with col5:
        if st.button("Generate Images", type="primary"):
            # Save modifications first
            update_state(scenes=modified_scenes)
            # Then generate images
            generate_images_from_prompts(state, is_demo_mode)


def render_storyboard_view(state, is_demo_mode: bool) -> None:
    """Render the storyboard preview with all generated images."""
    st.subheader("Storyboard Preview")
    st.markdown(
        """
        Review your storyboard below. You can regenerate individual images
        if you're not satisfied with them, then proceed to create the video.
        """
    )

    # Summary stats
    total_scenes = len(state.scenes)
    scenes_with_images = sum(1 for s in state.scenes if s.image_path and Path(s.image_path).exists())
    missing_count = total_scenes - scenes_with_images

    # Animation stats - break down by type
    from src.models.schemas import AnimationType

    # Count scenes by animation type
    scenes_by_type = {}
    for anim_type in AnimationType:
        if anim_type != AnimationType.NONE:
            scenes_by_type[anim_type] = [
                s for s in state.scenes
                if getattr(s, 'animation_type', None) == anim_type
            ]

    # Also count legacy animated=True scenes without animation_type set
    legacy_animated = [
        s for s in state.scenes
        if getattr(s, 'animated', False)
        and getattr(s, 'animation_type', None) in (None, AnimationType.NONE)
    ]

    # Total marked for animation
    scenes_marked_for_animation = sum(len(scenes) for scenes in scenes_by_type.values()) + len(legacy_animated)

    # Count scenes that already have animations
    scenes_with_animation = sum(
        1 for s in state.scenes
        if getattr(s, 'animated', False) or getattr(s, 'animation_type', None) not in (None, AnimationType.NONE)
        if getattr(s, 'video_path', None) and Path(s.video_path).exists()
    )

    # Pending by type (scenes marked but no video file yet)
    pending_by_type = {}
    for anim_type, scenes in scenes_by_type.items():
        pending = [s for s in scenes if not (getattr(s, 'video_path', None) and Path(s.video_path).exists())]
        if pending:
            pending_by_type[anim_type] = len(pending)

    # Legacy pending
    legacy_pending = [s for s in legacy_animated if not (getattr(s, 'video_path', None) and Path(s.video_path).exists())]
    if legacy_pending:
        pending_by_type[AnimationType.LIP_SYNC] = pending_by_type.get(AnimationType.LIP_SYNC, 0) + len(legacy_pending)

    pending_animations = sum(pending_by_type.values())

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Scenes", total_scenes)
    with col2:
        if missing_count > 0:
            st.metric("Images Ready", scenes_with_images, delta=f"-{missing_count} missing", delta_color="inverse")
        else:
            st.metric("Images Ready", scenes_with_images)
    with col3:
        if scenes_marked_for_animation > 0:
            if pending_animations > 0:
                st.metric("Animated", f"{scenes_with_animation}/{scenes_marked_for_animation}", delta=f"{pending_animations} pending")
            else:
                st.metric("Animated", f"{scenes_with_animation}/{scenes_marked_for_animation}", delta="✓ Ready")
        else:
            st.metric("Animated", "0")
    with col4:
        st.metric("Duration", f"{state.audio_duration:.1f}s")

    if missing_count > 0:
        st.error(
            f"{missing_count} scene(s) are missing images. "
            "Click 'Generate Missing' to retry, or regenerate individual scenes below."
        )

    st.markdown("---")

    # Quick animation control - Apply to All Scenes
    with st.expander("🎬 Quick Animation Control", expanded=False):
        st.markdown("**Apply Animation Type to All Scenes**")
        st.caption("Set all scenes to the same animation type at once.")

        apply_col1, apply_col2 = st.columns([3, 1])

        with apply_col1:
            animation_options = {
                "Static (Ken Burns only)": AnimationType.NONE,
                "Wan S2V Lip Sync (FREE, HF)": AnimationType.LIP_SYNC,
                "Wan TI2V Motion (FREE, HF)": AnimationType.PROMPT,
                "Wan 2.5 I2V (AtlasCloud)": AnimationType.ATLASCLOUD,
                "Seedance 1.5 (AtlasCloud)": AnimationType.SEEDANCE,
                "Veo 3.1 (Google)": AnimationType.VEO,
                "Kling Lip Sync (fal.ai)": AnimationType.KLING,
                "Wan S2V Lip (fal.ai)": AnimationType.WAN_S2V,
                "Seedance+Kling (Atlas+fal)": AnimationType.SEEDANCE_LIPSYNC,
            }
            selected_animation = st.selectbox(
                "Animation Type",
                options=list(animation_options.keys()),
                index=0,
                key="apply_all_animation_type",
                label_visibility="collapsed",
            )

        with apply_col2:
            if st.button("Apply to All", key="apply_all_animation_btn", type="primary"):
                new_anim_type = animation_options[selected_animation]
                scenes = list(state.scenes)

                # Map AnimationType to radio button label for session state sync
                anim_type_to_label = {
                    AnimationType.NONE: "Static",
                    AnimationType.LIP_SYNC: "Wan S2V Lip (FREE, HF)",
                    AnimationType.PROMPT: "Wan TI2V (FREE, HF)",
                    AnimationType.VEO: "Veo 3.1 (Google)",
                    AnimationType.ATLASCLOUD: "Wan 2.5 I2V (AtlasCloud)",
                    AnimationType.SEEDANCE: "Seedance 1.5 (AtlasCloud)",
                    AnimationType.KLING: "Kling Lip (fal.ai)",
                    AnimationType.WAN_S2V: "Wan S2V (fal.ai)",
                    AnimationType.SEEDANCE_LIPSYNC: "Seedance+Kling (Atlas+fal)",
                }
                new_label = anim_type_to_label.get(new_anim_type, "Static")

                for scene in scenes:
                    scene.animation_type = new_anim_type
                    scene.animated = new_anim_type != AnimationType.NONE
                    # Clear existing video path when changing animation type
                    if hasattr(scene, 'video_path'):
                        scene.video_path = None
                    # Update session state for the radio button so it syncs
                    st.session_state[f"anim_type_{scene.index}"] = new_label

                update_state(scenes=scenes)
                st.success(f"Applied '{selected_animation}' to all {len(scenes)} scenes!")
                st.rerun()

    # Display storyboard grid
    render_storyboard_grid(state)

    st.markdown("---")

    # Animation controls (if any scenes are marked for animation)
    if scenes_marked_for_animation > 0:
        st.subheader("🎬 Scene Animations")

        # Build info message based on pending animation types
        if pending_animations > 0:
            type_labels = {
                AnimationType.LIP_SYNC: "Wan S2V Lip (FREE, HF)",
                AnimationType.PROMPT: "Wan TI2V (FREE, HF)",
                AnimationType.ATLASCLOUD: "Wan 2.5 I2V (AtlasCloud)",
                AnimationType.VEO: "Veo 3.1 (Google)",
                AnimationType.SEEDANCE: "Seedance 1.5 (AtlasCloud)",
                AnimationType.KLING: "Kling Lip (fal.ai)",
                AnimationType.WAN_S2V: "Wan S2V (fal.ai)",
                AnimationType.SEEDANCE_LIPSYNC: "Seedance+Kling (Atlas+fal)",
            }
            pending_parts = []
            for anim_type, count in pending_by_type.items():
                label = type_labels.get(anim_type, anim_type.value)
                pending_parts.append(f"{count}× {label}")

            st.info(f"**{pending_animations} pending animation(s):** {', '.join(pending_parts)}")
        else:
            st.success(f"✓ All {scenes_marked_for_animation} animations ready!")

        # Show animation controls if there are any pending animations
        if pending_animations > 0:
            # Check if Wan2.2 (free tier) animations need gradio_client
            wan_pending = pending_by_type.get(AnimationType.LIP_SYNC, 0) + pending_by_type.get(AnimationType.PROMPT, 0)
            if wan_pending > 0 and not check_lip_sync_available():
                st.warning("Lip sync/Prompt animation requires `gradio_client`. Install with: `pip install gradio_client`")

            anim_col1, anim_col2, anim_col3, anim_col4 = st.columns([2, 1, 1, 1])
            with anim_col1:
                resolution = st.selectbox(
                    "Animation Resolution",
                    options=["720P", "480P"],
                    index=0,
                    help="Higher resolution takes longer to generate",
                    key="animation_resolution",
                )
            with anim_col2:
                parallel_workers = st.slider(
                    "Parallel Jobs",
                    min_value=1,
                    max_value=8,
                    value=3,
                    help="Number of animations to generate in parallel. Higher values are faster but use more API quota.",
                    key="animation_parallel_workers",
                )
            with anim_col3:
                if st.button("🎬 Generate Animations", type="primary"):
                    generate_animations(state, resolution, is_demo_mode, max_workers=parallel_workers)
            with anim_col4:
                if scenes_with_animation > 0:
                    if st.button("🔄 Regenerate All"):
                        # Clear existing animations for all types
                        scenes = state.scenes
                        for s in scenes:
                            anim_type = getattr(s, 'animation_type', None)
                            if anim_type not in (None, AnimationType.NONE):
                                s.video_path = None
                        update_state(scenes=scenes)
                        generate_animations(state, resolution, is_demo_mode, max_workers=parallel_workers)

        st.markdown("---")

    # Transcription options (in expander to save space)
    with st.expander("Transcription Options", expanded=False):
        storyboard_lyrics_hint = st.text_area(
            "Lyrics Hint (improves transcription accuracy):",
            value=state.lyrics if hasattr(state, 'lyrics') and state.lyrics else "",
            height=150,
            key="storyboard_lyrics_hint_input",
            help="Providing the actual lyrics helps WhisperX recognize words more accurately, especially for music with instruments.",
        )
        storyboard_use_demucs = st.checkbox(
            "Use Demucs vocal separation",
            value=config.use_demucs,
            key="storyboard_use_demucs_checkbox",
            help="Separates vocals from music before transcription for better word recognition. Takes longer but improves accuracy for songs with heavy instrumentation.",
        )

    # Video output settings (right before Create Video button)
    with st.expander("Video Output Settings", expanded=False):
        vo_col1, vo_col2, vo_col3, vo_col4, vo_col5 = st.columns(5)
        with vo_col1:
            # Resolution
            current_resolution = getattr(state, 'video_resolution', '1080p')
            resolution_idx = list(RESOLUTION_OPTIONS.keys()).index(current_resolution) if current_resolution in RESOLUTION_OPTIONS else 0
            new_resolution = st.selectbox(
                "Resolution",
                options=list(RESOLUTION_OPTIONS.keys()),
                index=resolution_idx,
                help="Final video output resolution",
                key="video_resolution_storyboard",
            )
            if new_resolution != current_resolution:
                update_state(video_resolution=new_resolution)
        with vo_col2:
            # FPS
            current_fps = getattr(state, 'video_fps', 30)
            fps_idx = list(FPS_OPTIONS.values()).index(current_fps) if current_fps in FPS_OPTIONS.values() else 1
            fps_selection = st.selectbox(
                "Frame Rate",
                options=list(FPS_OPTIONS.keys()),
                index=fps_idx,
                help="Higher fps = smoother but larger file",
                key="video_fps_storyboard",
            )
            new_fps = FPS_OPTIONS[fps_selection]
            if new_fps != current_fps:
                update_state(video_fps=new_fps)
        with vo_col3:
            # Extension mode
            extension_options = {
                "Ken Burns (all scenes)": "all",
                "Ken Burns (end only)": "end_only",
                "No Ken Burns": "none",
            }
            current_ext_mode = getattr(state, 'extension_mode', 'all')
            ext_labels = list(extension_options.keys())
            ext_values = list(extension_options.values())
            ext_idx = ext_values.index(current_ext_mode) if current_ext_mode in ext_values else 0
            new_ext_label = st.selectbox(
                "Animation Extension",
                options=ext_labels,
                index=ext_idx,
                help="How to fill gaps when animations are shorter than scene duration",
                key="extension_mode_storyboard",
            )
            new_ext_mode = extension_options[new_ext_label]
            if new_ext_mode != current_ext_mode:
                update_state(extension_mode=new_ext_mode)
        with vo_col4:
            # Crossfade
            crossfade_val = st.slider(
                "Crossfade (sec)",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get("crossfade", 0.3),
                step=0.1,
                help="Transition duration between scenes",
                key="crossfade_storyboard",
            )
            st.session_state["crossfade"] = crossfade_val
        with vo_col5:
            # Show lyrics toggle
            current_show_lyrics = getattr(state, 'show_lyrics', True)
            new_show_lyrics = st.checkbox(
                "Show Lyrics",
                value=current_show_lyrics,
                help="Display karaoke-style lyrics on the video",
                key="show_lyrics_storyboard",
            )
            if new_show_lyrics != current_show_lyrics:
                update_state(show_lyrics=new_show_lyrics)

    # Action buttons
    if missing_count > 0:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            if st.button("Generate Missing", type="primary"):
                regenerate_missing_images(state)
        with col2:
            if st.button("Regenerate All", type="secondary"):
                regenerate_all_images(state)
        with col3:
            if st.button("Redo Transcription", type="secondary", help="Re-run WhisperX"):
                with st.spinner("Re-transcribing audio..."):
                    if _redo_transcription(state, lyrics_hint=storyboard_lyrics_hint if storyboard_lyrics_hint else None, use_demucs=storyboard_use_demucs):
                        st.success("Transcription updated!")
                        st.rerun()
        with col4:
            if st.button("Re-sync Lyrics", type="secondary", help="Re-assign lyrics from transcript"):
                _resync_lyrics_to_scenes(state)
                st.success("Lyrics re-synced!")
                st.rerun()
        with col5:
            if st.button("Edit Prompts", type="secondary"):
                update_state(storyboard_ready=False)
                st.rerun()
        with col6:
            if st.button("Create Video Anyway"):
                crossfade = st.session_state.get("crossfade", 0.3)
                generate_video_from_storyboard(state, crossfade, is_demo_mode)
    else:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            if st.button("Regenerate All", type="secondary"):
                regenerate_all_images(state)
        with col2:
            if st.button("Redo Transcription", type="secondary", help="Re-run WhisperX"):
                with st.spinner("Re-transcribing audio..."):
                    if _redo_transcription(state, lyrics_hint=storyboard_lyrics_hint if storyboard_lyrics_hint else None, use_demucs=storyboard_use_demucs):
                        st.success("Transcription updated!")
                        st.rerun()
        with col3:
            if st.button("Re-sync Lyrics", type="secondary", help="Re-assign lyrics from transcript"):
                _resync_lyrics_to_scenes(state)
                st.success("Lyrics re-synced!")
                st.rerun()
        with col4:
            if st.button("Edit Prompts", type="secondary"):
                update_state(storyboard_ready=False)
                st.rerun()
        with col5:
            if st.button("Start Over", type="secondary"):
                update_state(
                    scenes=[],
                    generated_images=[],
                    prompts_ready=False,
                    storyboard_ready=False,
                    project_dir=None,
                )
                st.rerun()
        with col6:
            if st.button("Create Video", type="primary"):
                crossfade = st.session_state.get("crossfade", 0.3)
                generate_video_from_storyboard(state, crossfade, is_demo_mode)

    # Motion prompts row
    motion_col1, motion_col2 = st.columns([1, 4])
    with motion_col1:
        if st.button("🤖 Redo All Motion Prompts (AI)", help="Regenerate motion prompts for all scenes using AI image analysis"):
            # Create progress UI elements
            status_text = st.empty()
            progress_bar = st.progress(0.0)
            status_text.text("Starting motion prompt generation...")

            count = _regenerate_all_motion_prompts(state, progress_bar=progress_bar, status_text=status_text)

            if count > 0:
                st.success(f"Generated {count} motion prompts!")
                st.rerun()
            else:
                progress_bar.empty()
                status_text.empty()
                st.warning("No scenes with images to analyze")

    # Bulk re-animation row
    selected_indices = _get_selected_scene_indices(state)
    if selected_indices:
        reanimate_col1, reanimate_col2 = st.columns([1, 4])
        with reanimate_col1:
            if st.button(
                f"🎬 Reanimate Selected ({len(selected_indices)} scenes)",
                type="primary",
                help="Clear existing animations and re-animate all selected scenes"
            ):
                # Clear video_path for selected scenes to force re-animation
                scenes = list(state.scenes)
                project_dir = getattr(state, 'project_dir', None)
                cleared_count = 0

                for idx in selected_indices:
                    scene = scenes[idx]
                    if scene.video_path:
                        # Delete existing animation file
                        video_path = Path(scene.video_path)
                        if video_path.exists():
                            video_path.unlink()
                        scenes[idx].video_path = None
                        cleared_count += 1

                update_state(scenes=scenes)
                _clear_selected_scene_checkboxes(state)

                if cleared_count > 0:
                    st.success(f"Cleared {cleared_count} animations. Use individual Animate buttons or the mass animate feature to regenerate.")
                else:
                    st.info("No existing animations to clear. Use individual Animate buttons to animate selected scenes.")
                st.rerun()
        with reanimate_col2:
            st.caption(f"Selected scenes: {', '.join(str(i+1) for i in selected_indices)}")

    # Back button
    st.markdown("---")
    if st.button("Back to Upload"):
        go_to_step(WorkflowStep.UPLOAD)
        st.rerun()


def render_storyboard_grid(state) -> None:
    """Render the storyboard as a grid of images with scene info."""
    scenes = state.scenes

    # Display in rows of 3
    cols_per_row = 3

    for row_start in range(0, len(scenes), cols_per_row):
        row_scenes = scenes[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for col, scene in zip(cols, row_scenes):
            with col:
                render_scene_card(state, scene)


def _run_scene_animation_inline(state, scene_index: int, resolution: str) -> None:
    """Run animation for a scene inline within the fragment."""
    project_dir = getattr(state, 'project_dir', None)
    if not project_dir:
        st.error("No project directory found.")
        return

    scenes = state.scenes
    if scene_index >= len(scenes):
        st.error("Invalid scene index.")
        return

    scene = scenes[scene_index]
    if not scene.image_path or not Path(scene.image_path).exists():
        st.error("Scene has no image to animate.")
        return

    # Get animation type
    animation_type = getattr(scene, 'animation_type', AnimationType.LIP_SYNC)

    # For lip sync, we need audio
    audio_path = getattr(state, 'audio_path', None)
    if animation_type == AnimationType.LIP_SYNC:
        if not audio_path or audio_path == "demo_mode":
            st.error("No audio file found for lip sync animation.")
            return

    # For prompt animation, we need a motion prompt
    if animation_type == AnimationType.PROMPT:
        motion_prompt = _get_motion_prompt(scene)
        if not motion_prompt:
            st.error("No motion prompt found for prompt animation.")
            return

    # For Veo animation, we need a motion prompt
    if animation_type == AnimationType.VEO:
        motion_prompt = _get_motion_prompt(scene)
        if not motion_prompt:
            st.error("No motion prompt found for Veo animation.")
            return

    animations_dir = Path(project_dir) / "animations"
    animations_dir.mkdir(parents=True, exist_ok=True)

    output_path = animations_dir / f"animated_scene_{scene_index:03d}.mp4"

    # Delete existing animation file if it exists
    if output_path.exists():
        output_path.unlink()
        scenes[scene_index].video_path = None
        update_state(scenes=scenes)

    # Show inline progress within the fragment
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    anim_type_labels = {
        AnimationType.LIP_SYNC: "Wan S2V Lip (HF)",
        AnimationType.PROMPT: "Wan TI2V (HF)",
        AnimationType.VEO: "Veo 3.1 (Google)",
        AnimationType.ATLASCLOUD: "Wan 2.5 (AtlasCloud)",
        AnimationType.SEEDANCE: "Seedance 1.5 (AtlasCloud)",
        AnimationType.KLING: "Kling Lip (fal.ai)",
        AnimationType.WAN_S2V: "Wan S2V (fal.ai)",
        AnimationType.SEEDANCE_LIPSYNC: "Seedance+Kling (Atlas+fal)",
    }
    anim_type_label = anim_type_labels.get(animation_type, "animation")
    status_placeholder.info(f"Animating ({anim_type_label}) at {resolution}...")
    progress_bar = progress_placeholder.progress(0.0)

    # Track error messages from the animator
    error_msg_holder = {"msg": None}

    def progress_callback(msg: str, prog: float):
        progress_bar.progress(prog)
        status_placeholder.info(msg)
        # Capture error messages (progress 0.0 with error keywords indicates failure)
        if prog == 0.0 and ("failed" in msg.lower() or "error" in msg.lower() or
                           "exceeded" in msg.lower() or "quota" in msg.lower() or
                           "timeout" in msg.lower() or "invalid" in msg.lower() or
                           "queue too long" in msg.lower()):
            error_msg_holder["msg"] = msg

    try:
        result = None

        if animation_type == AnimationType.LIP_SYNC:
            # Use LipSyncAnimator for audio-driven lip sync
            animator = LipSyncAnimator()
            result = animator.animate_scene(
                image_path=Path(scene.image_path),
                audio_path=Path(audio_path),
                start_time=scene.start_time,
                duration=scene.duration,
                output_path=output_path,
                resolution=resolution,
                progress_callback=progress_callback,
            )
        elif animation_type == AnimationType.PROMPT:
            # Use PromptAnimator for prompt-driven motion
            animator = PromptAnimator()
            motion_prompt = _get_motion_prompt(scene)
            result = animator.animate_scene(
                image_path=Path(scene.image_path),
                prompt=motion_prompt,
                output_path=output_path,
                duration_seconds=min(scene.duration, 5.0),  # TI2V-5B supports 2-5 seconds
                quality_preset="fast",  # 320x576, 8 steps - stays within free tier GPU limits
                progress_callback=progress_callback,
            )
        elif animation_type == AnimationType.VEO:
            # Use VeoAnimator for high-quality Veo 3.1 animation (PAID)
            animator = VeoAnimator()
            motion_prompt = _get_motion_prompt(scene)
            result = animator.animate_scene(
                image_path=Path(scene.image_path),
                prompt=motion_prompt,
                output_path=output_path,
                duration_seconds=scene.duration,  # Veo supports 4, 6, or 8 seconds
                resolution="720p",  # Can be 720p or 1080p
                progress_callback=progress_callback,
            )
        elif animation_type == AnimationType.ATLASCLOUD:
            # Use AtlasCloudAnimator for Wan 2.5 animation (PAID, no GPU limits)
            animator = AtlasCloudAnimator()
            motion_prompt = _get_motion_prompt(scene)
            result = animator.animate_scene(
                image_path=Path(scene.image_path),
                prompt=motion_prompt,
                output_path=output_path,
                duration_seconds=5 if scene.duration < 7.5 else 10,  # AtlasCloud supports 5 or 10 seconds
                resolution="720p",  # Can be 720p or 1080p
                progress_callback=progress_callback,
            )
        elif animation_type == AnimationType.SEEDANCE:
            # Use SeedanceAnimator for Seedance Pro animation (PAID, up to 12s)
            animator = SeedanceAnimator()
            motion_prompt = _get_motion_prompt(scene)
            # Seedance supports 2-12 second durations - use ceil to ensure animation covers full scene
            # (avoids Ken Burns padding at the end due to truncation)
            target_duration = min(12, max(2, math.ceil(scene.duration)))
            result = animator.animate_scene(
                image_path=Path(scene.image_path),
                prompt=motion_prompt,
                output_path=output_path,
                duration_seconds=target_duration,
                resolution="720p",  # Can be 480p, 720p, or 1080p
                progress_callback=progress_callback,
            )

        elif animation_type == AnimationType.KLING:
            # Kling AI image-to-video with lip sync via fal.ai
            if not audio_path or audio_path == "demo_mode":
                status_placeholder.error("Audio required for Kling lip sync animation")
                return

            status_placeholder.info("Generating Kling lip sync animation...")
            animator = KlingAnimator()
            result = animator.animate_scene(
                image_path=Path(scene.image_path),
                audio_path=Path(audio_path),
                start_time=scene.start_time,
                duration=scene.duration,
                output_path=output_path,
                resolution="720p",
                use_i2v=False,  # Use static video as base (cheaper)
                progress_callback=progress_callback,
            )

        elif animation_type == AnimationType.WAN_S2V:
            # Wan 2.2 S2V via fal.ai - full motion + lip sync in one step
            if not audio_path or audio_path == "demo_mode":
                status_placeholder.error("Audio required for Wan S2V animation")
                return

            status_placeholder.info("Generating Wan S2V animation (motion + lip sync)...")
            animator = WanS2VAnimator()
            motion_prompt = _get_motion_prompt(scene)
            result = animator.animate_scene(
                image_path=Path(scene.image_path),
                audio_path=Path(audio_path),
                start_time=scene.start_time,
                duration=scene.duration,
                output_path=output_path,
                prompt=motion_prompt,
                progress_callback=progress_callback,
            )

        elif animation_type == AnimationType.SEEDANCE_LIPSYNC:
            # Two-step workflow: Seedance motion → Kling lip sync
            if not audio_path or audio_path == "demo_mode":
                status_placeholder.error("Audio required for lip sync animation")
                return

            status_placeholder.info("Step 1: Generating motion with Seedance Pro...")
            animator = SeedanceAnimator()
            motion_prompt = _get_motion_prompt(scene)
            # Kling lipsync max 10s - use ceil to ensure animation covers full scene
            target_duration = min(10, max(2, math.ceil(scene.duration)))

            motion_output = output_path.with_suffix(".motion.mp4")
            motion_result = animator.animate_scene(
                image_path=Path(scene.image_path),
                prompt=motion_prompt,
                output_path=motion_output,
                duration_seconds=target_duration,
                resolution="720p",
                progress_callback=progress_callback,
            )

            if motion_result and motion_result.exists():
                status_placeholder.info("Step 2: Applying lip sync with Kling (fal.ai)...")
                lipsync_animator = KlingAnimator()
                result = lipsync_animator.apply_lipsync_to_video(
                    video_path=motion_result,
                    audio_path=Path(audio_path),
                    start_time=scene.start_time,
                    duration=scene.duration,
                    output_path=output_path,
                    progress_callback=progress_callback,
                )
                if result and result.exists():
                    # Lip sync succeeded - remove temp motion file
                    motion_output.unlink(missing_ok=True)
                else:
                    # Lip sync failed - use motion video as fallback
                    status_placeholder.warning("Lip sync failed, using motion video as fallback")
                    # Rename motion file to final output
                    import shutil
                    shutil.move(str(motion_output), str(output_path))
                    result = output_path
            else:
                status_placeholder.error("Motion generation failed")
                result = None

        if result and result.exists():
            scenes[scene_index].video_path = result
            scenes[scene_index].animated = True
            scenes[scene_index].animation_type = animation_type
            update_state(scenes=scenes)
            # Persist to scenes.json so animations are recognized after recovery
            save_scene_metadata(Path(project_dir), scenes)
            status_placeholder.success("Animation complete!")
            # Rerun to show the video in the storyboard
            st.rerun()
        else:
            # Use captured error message if available, otherwise generic message
            if error_msg_holder["msg"]:
                status_placeholder.error(error_msg_holder["msg"])
            else:
                status_placeholder.error("Animation failed - check the logs for details")

    except Exception as e:
        error_str = str(e)
        # Provide user-friendly error messages for common issues
        if "GPU quota" in error_str or "exceeded" in error_str.lower():
            import re
            time_match = re.search(r"Try again in (\d+:\d+:\d+)", error_str)
            wait_time = time_match.group(1) if time_match else "~24 hours"
            user_msg = f"HuggingFace GPU quota exceeded. Try again in {wait_time}."
        elif "billing" in error_str.lower():
            user_msg = "Veo requires billing to be enabled on your Google Cloud account."
        elif "invalid" in error_str.lower() and "api" in error_str.lower():
            user_msg = "Invalid API key. Check your configuration."
        elif "timeout" in error_str.lower():
            user_msg = "Request timed out. The server may be overloaded. Try again."
        elif "queue" in error_str.lower():
            user_msg = "Server is busy. Please try again in a few minutes."
        else:
            user_msg = f"Animation error: {error_str[:150]}"
        status_placeholder.error(user_msg)

    # Clear progress bar after completion
    progress_placeholder.empty()


@st.fragment
def render_scene_card(state, scene: Scene) -> None:
    """Render a single scene card with image and controls. Uses fragment for performance."""
    has_image = scene.image_path and Path(scene.image_path).exists()

    # Check for animation: first from scene.video_path, then check disk for existing file
    animation_path = None
    if getattr(scene, 'video_path', None) and Path(scene.video_path).exists():
        animation_path = Path(scene.video_path)
    elif has_image:
        # Try to find animation based on scene index - derive project dir from image path
        # Image is at: {project_dir}/images/scene_{idx:03d}.png
        # Animation at: {project_dir}/animations/animated_scene_{idx:03d}.mp4
        image_path = Path(scene.image_path)
        project_dir = image_path.parent.parent  # Go up from images/ to project dir
        expected_anim = project_dir / "animations" / f"animated_scene_{scene.index:03d}.mp4"
        if expected_anim.exists():
            animation_path = expected_anim
            # Update the scene's video_path so it's available elsewhere
            scene.video_path = expected_anim

    has_animation = animation_path is not None

    # Scene header with status indicator
    status_icons = ""
    if not has_image:
        status_icons = " :warning:"
    elif getattr(scene, 'animated', False):
        if has_animation:
            status_icons = " :movie_camera:"  # Has animation
        else:
            status_icons = " :hourglass:"  # Pending animation

    # Header row with selection checkbox
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown(f"**Scene {scene.index + 1}**{status_icons}")
    with header_col2:
        # Checkbox for bulk re-animation selection
        st.checkbox(
            "Select",
            key=f"reanimate_select_{scene.index}",
            label_visibility="collapsed",
            help="Select for bulk re-animation"
        )
    st.caption(f"{scene.start_time:.1f}s - {scene.end_time:.1f}s ({scene.duration:.1f}s)")

    # Check for pending variations to select from
    variations_key = f"scene_variations_{scene.index}"
    has_pending_variations = variations_key in st.session_state and st.session_state[variations_key]

    # Image or animation preview - pending variations take priority so user can switch frame
    if has_pending_variations:
        # Show variation selection UI - even if there's an existing animation
        variation_paths = st.session_state[variations_key]
        if has_animation:
            st.warning("Select a variation to replace the current animation's source image:")
        else:
            st.info(f"Select from {len(variation_paths)} variations:")

        # Display variations in columns
        var_cols = st.columns(len(variation_paths))
        for i, (col, var_path) in enumerate(zip(var_cols, variation_paths)):
            with col:
                if Path(var_path).exists():
                    st.image(var_path, use_container_width=True)
                    if st.button(f"Select #{i+1}", key=f"select_var_{scene.index}_{i}", type="primary"):
                        select_scene_variation(state, scene.index, var_path)
                else:
                    st.error(f"Variation {i+1} not found")

        # Cancel button to dismiss variations
        if st.button("Cancel", key=f"cancel_vars_{scene.index}", type="secondary"):
            clear_scene_variations(scene.index)
    elif has_animation:
        st.video(str(animation_path))
    elif has_image:
        st.image(str(scene.image_path), use_container_width=True)
    else:
        st.error("Missing image - click Regenerate")

    # Animation type selector (only show if image exists)
    if has_image:
        # Get current animation type
        current_anim_type = getattr(scene, 'animation_type', AnimationType.NONE)
        # Note: Legacy 'animated' field is no longer auto-converted to LIP_SYNC.
        # Animation type is now persisted in scenes.json metadata.

        anim_options = {
            "Static": AnimationType.NONE,
            "Wan S2V Lip (FREE, HF)": AnimationType.LIP_SYNC,
            "Wan TI2V (FREE, HF)": AnimationType.PROMPT,
            "Wan 2.5 I2V (AtlasCloud)": AnimationType.ATLASCLOUD,
            "Seedance 1.5 (AtlasCloud)": AnimationType.SEEDANCE,
            "Veo 3.1 (Google)": AnimationType.VEO,
            "Kling Lip (fal.ai)": AnimationType.KLING,
            "Wan S2V (fal.ai)": AnimationType.WAN_S2V,
            "Seedance+Kling (Atlas+fal)": AnimationType.SEEDANCE_LIPSYNC,
        }
        anim_labels = list(anim_options.keys())
        current_idx = list(anim_options.values()).index(current_anim_type) if current_anim_type in anim_options.values() else 0

        selected_anim_label = st.radio(
            "Animation",
            options=anim_labels,
            index=current_idx,
            key=f"anim_type_{scene.index}",
            horizontal=True,
            label_visibility="collapsed",
        )
        new_anim_type = anim_options[selected_anim_label]

        # Show motion prompt input if prompt-based animation is selected
        if new_anim_type in (AnimationType.PROMPT, AnimationType.VEO, AnimationType.ATLASCLOUD, AnimationType.SEEDANCE, AnimationType.WAN_S2V, AnimationType.SEEDANCE_LIPSYNC):
            widget_key = f"motion_prompt_{scene.index}"
            ai_result_key = f"_ai_motion_result_{scene.index}"
            scene_motion_prompt = getattr(scene, 'motion_prompt', None)

            # Default to a short motion prompt if no motion_prompt is set
            # Use _get_motion_prompt to get a mood-appropriate short action prompt
            default_prompt = _get_motion_prompt(scene)

            # Check if AI generated a new prompt (stored in temp key from previous render)
            if ai_result_key in st.session_state:
                st.session_state[widget_key] = st.session_state[ai_result_key]
                del st.session_state[ai_result_key]
            # Clear stale "(Recovered from files)" from old code, or initialize from scene
            elif widget_key in st.session_state:
                if st.session_state[widget_key] == "(Recovered from files)":
                    st.session_state[widget_key] = default_prompt
            else:
                # Initialize from scene data or visual prompt as default
                st.session_state[widget_key] = default_prompt

            prompt_col, ai_col = st.columns([4, 1])
            with prompt_col:
                new_motion_prompt = st.text_input(
                    "Motion Prompt",
                    key=widget_key,
                    placeholder="e.g., playing guitar, dancing",
                    help="Describe the motion you want",
                )
            with ai_col:
                # AI Generate button for motion prompt
                if scene.image_path and Path(scene.image_path).exists():
                    if st.button("AI", key=f"ai_motion_{scene.index}", help="Generate motion prompt from image using AI"):
                        with st.spinner("Analyzing..."):
                            from src.services.image_generator import ImageGenerator
                            generator = ImageGenerator()
                            ai_prompt = generator.generate_motion_prompt_from_image(Path(scene.image_path))
                            if ai_prompt:
                                scenes = state.scenes
                                scenes[scene.index].motion_prompt = ai_prompt
                                update_state(scenes=scenes)
                                # Store in temp key - will be transferred to widget on next render
                                st.session_state[ai_result_key] = ai_prompt
                                st.rerun()
                            else:
                                st.error("Failed")
            # Update motion prompt if changed
            if new_motion_prompt != scene_motion_prompt:
                scenes = state.scenes
                scenes[scene.index].motion_prompt = new_motion_prompt
                update_state(scenes=scenes)

        # Update animation type if changed
        if new_anim_type != current_anim_type:
            scenes = state.scenes
            scenes[scene.index].animation_type = new_anim_type
            # Update legacy 'animated' field for compatibility
            scenes[scene.index].animated = new_anim_type != AnimationType.NONE
            if new_anim_type == AnimationType.NONE:
                # Clear video path when disabling animation
                scenes[scene.index].video_path = None
            # Auto-fill motion prompt with a short action description when switching to prompt-based animation
            elif new_anim_type in (AnimationType.PROMPT, AnimationType.VEO, AnimationType.ATLASCLOUD, AnimationType.SEEDANCE, AnimationType.SEEDANCE_LIPSYNC):
                if not getattr(scenes[scene.index], 'motion_prompt', None):
                    short_motion = _get_motion_prompt(scene)
                    scenes[scene.index].motion_prompt = short_motion
                    # Also update the widget session state so it shows immediately
                    widget_key = f"motion_prompt_{scene.index}"
                    if widget_key in st.session_state:
                        st.session_state[widget_key] = short_motion
            update_state(scenes=scenes)
            st.rerun()

        # Animate/Re-animate controls (only if animation type is not NONE)
        if new_anim_type != AnimationType.NONE:
            # Show expected animation duration based on type
            # Use ceil to show the actual duration that will be requested (avoids Ken Burns padding)
            scene_dur = scene.duration
            if new_anim_type in (AnimationType.SEEDANCE, AnimationType.SEEDANCE_LIPSYNC):
                anim_dur = min(12, max(2, math.ceil(scene_dur)))
                st.caption(f"Duration: {anim_dur}s (scene is {scene_dur:.1f}s, Seedance: 2-12s)")
            elif new_anim_type == AnimationType.KLING:
                anim_dur = min(10, max(2, math.ceil(scene_dur)))
                st.caption(f"Duration: {anim_dur}s (scene is {scene_dur:.1f}s, Kling: 2-10s)")
            elif new_anim_type == AnimationType.WAN_S2V:
                # Wan S2V supports video chaining for long scenes (7.5s segments @ 16fps)
                num_segments = int(scene_dur / 7.5) + (1 if scene_dur % 7.5 > 0 else 0)
                if num_segments > 1:
                    st.caption(f"Duration: {scene_dur:.1f}s ({num_segments} segments, chained)")
                else:
                    st.caption(f"Duration: {scene_dur:.1f}s (Wan S2V @ 16fps)")
            elif new_anim_type == AnimationType.ATLASCLOUD:
                anim_dur = 5 if scene_dur < 7.5 else 10
                st.caption(f"Duration: {anim_dur}s (scene is {scene_dur:.1f}s, AtlasCloud: 5 or 10s)")
            elif new_anim_type == AnimationType.VEO:
                st.caption(f"Duration: {scene_dur:.1f}s (Veo chains segments for long scenes)")
            else:
                st.caption(f"Duration: {scene_dur:.1f}s")

            anim_col1, anim_col2 = st.columns([1, 1])
            with anim_col1:
                anim_resolution = st.selectbox(
                    "Resolution",
                    options=["720P", "480P"],
                    index=0,
                    key=f"anim_res_{scene.index}",
                    label_visibility="collapsed",
                )
            with anim_col2:
                button_label = "Re-animate" if has_animation else "Animate"
                anim_type_names = {
                    AnimationType.LIP_SYNC: "Wan S2V Lip (HF)",
                    AnimationType.PROMPT: "Wan TI2V (HF)",
                    AnimationType.VEO: "Veo 3.1 (Google)",
                    AnimationType.ATLASCLOUD: "Wan 2.5 (AtlasCloud)",
                    AnimationType.SEEDANCE: "Seedance 1.5 (AtlasCloud)",
                    AnimationType.KLING: "Kling Lip (fal.ai)",
                    AnimationType.WAN_S2V: "Wan S2V (fal.ai)",
                    AnimationType.SEEDANCE_LIPSYNC: "Seedance+Kling (Atlas+fal)",
                }
                anim_type_name = anim_type_names.get(new_anim_type, "Animation")
                if st.button(button_label, key=f"animate_{scene.index}", help=f"Generate {anim_type_name} animation"):
                    _run_scene_animation_inline(state, scene.index, anim_resolution)

    # Scene details in expander - editable prompt and lyrics
    with st.expander("Edit Scene", expanded=False):
        st.markdown(f"**Mood:** {scene.mood}")

        # Scene timing adjustment
        st.markdown("**Scene Timing**")
        timing_cols = st.columns(2)
        with timing_cols[0]:
            # Ensure max_value is at least 0.0 (handles very short scenes)
            start_max = max(0.0, scene.end_time - 0.5)
            new_start_time = st.number_input(
                "Start (seconds)",
                min_value=0.0,
                max_value=start_max,
                value=min(scene.start_time, start_max),
                step=0.1,
                key=f"timing_start_{scene.index}",
                help="Adjust when this scene starts in the song"
            )
        with timing_cols[1]:
            new_end_time = st.number_input(
                "End (seconds)",
                min_value=new_start_time + 0.5,
                max_value=float(state.audio_duration) if hasattr(state, 'audio_duration') else 999.0,
                value=scene.end_time,
                step=0.1,
                key=f"timing_end_{scene.index}",
                help="Adjust when this scene ends in the song"
            )

        # Apply timing changes button
        if new_start_time != scene.start_time or new_end_time != scene.end_time:
            if st.button("Apply Timing", key=f"apply_timing_{scene.index}", type="secondary"):
                _adjust_scene_timing(state, scene.index, new_start_time, new_end_time)
                st.success(f"Timing updated: {new_start_time:.1f}s - {new_end_time:.1f}s")
                st.rerun()

        st.markdown("---")

        # Audio preview with adjustable start point
        if state.audio_path and state.audio_path != "demo_mode":
            st.markdown("**Audio Preview**")

            # Show word-level timing if words exist
            if scene.words:
                st.markdown(f"**Word timing ({len(scene.words)} words):**")
                # Show all words in a formatted view
                timing_lines = []
                for w in scene.words:
                    timing_lines.append(f"{w.word} ({w.start:.2f}s - {w.end:.2f}s)")
                st.caption(" | ".join(timing_lines))

            # Slider to pick start point within scene
            scene_duration = scene.end_time - scene.start_time

            # Default to where first word starts (relative to scene start)
            default_offset = 0.0
            if scene.words:
                first_word_offset = scene.words[0].start - scene.start_time
                default_offset = max(0.0, min(first_word_offset, scene_duration - 0.5))

            preview_start_offset = st.slider(
                "Start from (seconds into scene)",
                min_value=0.0,
                max_value=max(0.1, scene_duration - 0.5),
                value=default_offset,
                step=0.1,
                key=f"audio_start_{scene.index}",
                help="Adjust to start playback from a specific point"
            )

            # Show the overall song timestamp
            actual_start = scene.start_time + preview_start_offset
            st.caption(f"Playing from **{actual_start:.1f}s** in song (scene ends at {scene.end_time:.1f}s)")

            # Get audio clip from adjusted start point
            audio_clip = _get_audio_clip(
                str(state.audio_path),
                actual_start,
                scene.end_time
            )
            if audio_clip:
                st.audio(audio_clip, format="audio/mp3")

            # Preview with lyrics button
            if has_image:
                if st.button("Preview with Lyrics", key=f"preview_lyrics_{scene.index}"):
                    with st.spinner("Generating preview..."):
                        try:
                            video_bytes = _generate_scene_preview_with_lyrics(state, scene)
                            if video_bytes:
                                st.video(video_bytes)
                        except Exception as e:
                            st.error(f"Preview failed: {e}")

        st.markdown("---")

        # Editable effect
        effect_options = [e.value for e in KenBurnsEffect]
        current_effect_idx = effect_options.index(scene.effect.value)
        new_effect = st.selectbox(
            "Effect",
            options=effect_options,
            index=current_effect_idx,
            key=f"card_effect_{scene.index}",
        )

        # Editable lyrics - fix transcription errors OR add missing lyrics
        current_lyrics = " ".join(w.word for w in scene.words) if scene.words else ""

        # Transcription options (in expander to save space)
        with st.expander("Transcription Options", expanded=False):
            scene_lyrics_hint = st.text_area(
                "Lyrics Hint (improves accuracy):",
                value=current_lyrics,
                height=80,
                key=f"scene_lyrics_hint_{scene.index}",
                help="Provide the expected lyrics for this scene to help WhisperX recognize words more accurately.",
            )
            scene_use_demucs = st.checkbox(
                "Use Demucs vocal separation",
                value=config.use_demucs,
                key=f"scene_use_demucs_{scene.index}",
                help="Separates vocals from music before transcription. Takes longer but improves accuracy.",
            )

        # Transcription buttons for this scene
        trans_col1, trans_col2 = st.columns(2)
        with trans_col1:
            if st.button("Redo Transcription", key=f"redo_trans_{scene.index}", type="secondary", help="Re-transcribe just this scene's audio"):
                with st.spinner("Transcribing scene audio..."):
                    if _redo_scene_transcription(
                        state,
                        scene.index,
                        lyrics_hint=scene_lyrics_hint if scene_lyrics_hint.strip() else None,
                        use_demucs=scene_use_demucs,
                    ):
                        st.success("Scene transcribed!")
                        st.rerun()
        with trans_col2:
            if st.button("Re-sync Lyrics", key=f"resync_{scene.index}", type="secondary", help="Re-sync from existing transcript"):
                _resync_single_scene_lyrics(state, scene.index)
                st.success("Lyrics re-synced!")
                st.rerun()

        new_lyrics = st.text_area(
            "Lyrics" + (" (none detected - add them here)" if not scene.words else ""),
            value=current_lyrics,
            key=f"card_lyrics_{scene.index}",
            help="Edit or add lyrics for this scene. If WhisperX missed words, type them here.",
            placeholder="Type lyrics for this scene..." if not scene.words else "",
            height=80,
        )

        # Editable prompt - full prompt visible
        new_prompt = st.text_area(
            "Visual Prompt",
            value=scene.visual_prompt,
            key=f"card_prompt_{scene.index}",
            height=100,
        )

        # Action buttons
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Save Changes", key=f"save_{scene.index}", type="secondary"):
                # Update scene with new values (no regenerate)
                scenes = state.scenes
                scenes[scene.index].visual_prompt = new_prompt
                scenes[scene.index].effect = KenBurnsEffect(new_effect)
                # Update lyrics
                if new_lyrics and new_lyrics.strip():
                    _update_scene_lyrics(scenes[scene.index], new_lyrics)
                update_state(scenes=scenes)
                st.success("Saved!")
                st.rerun()

        with col_b:
            if st.button("Save & Regenerate", key=f"save_regen_{scene.index}", type="primary"):
                # Update scene with new values
                scenes = state.scenes
                scenes[scene.index].visual_prompt = new_prompt
                scenes[scene.index].effect = KenBurnsEffect(new_effect)
                # Update lyrics
                if new_lyrics and new_lyrics.strip():
                    _update_scene_lyrics(scenes[scene.index], new_lyrics)
                update_state(scenes=scenes)
                regenerate_single_image(state, scene.index)

    # Buttons row - check for existing variations
    # IMPORTANT: Read project_dir fresh from get_state() to handle fragment reruns correctly
    # When st.rerun() is called from within a fragment, the fragment reruns with stale arguments
    current_state = get_state()
    project_dir = getattr(current_state, 'project_dir', None)
    existing_variations = get_existing_variations(project_dir, scene.index) if project_dir else []
    has_saved_variations = len(existing_variations) > 0

    # Number of variations selector
    num_variations_key = f"num_variations_{scene.index}"
    if num_variations_key not in st.session_state:
        st.session_state[num_variations_key] = 3

    if has_saved_variations:
        # 4-column layout when variations exist
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            num_vars = st.number_input(
                "# Vars",
                min_value=1,
                max_value=10,
                value=st.session_state[num_variations_key],
                key=f"num_vars_input_{scene.index}",
                help="Number of variations to generate",
            )
            st.session_state[num_variations_key] = num_vars
        with col2:
            if st.button(f"Regenerate", key=f"regen_{scene.index}", type="primary" if not has_image else "secondary"):
                regenerate_single_image(current_state, scene.index, num_variations=num_vars)
        with col3:
            if st.button(f"Variations ({len(existing_variations)})", key=f"view_vars_{scene.index}", help="View and switch between saved variations"):
                show_existing_variations(project_dir, scene.index)
        with col4:
            if has_image:
                with open(scene.image_path, "rb") as f:
                    st.download_button(
                        "Download",
                        data=f.read(),
                        file_name=f"scene_{scene.index + 1}.png",
                        mime="image/png",
                        key=f"download_{scene.index}",
                    )
    else:
        # 3-column layout when no variations
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            num_vars = st.number_input(
                "# Vars",
                min_value=1,
                max_value=10,
                value=st.session_state[num_variations_key],
                key=f"num_vars_input_{scene.index}",
                help="Number of variations to generate",
            )
            st.session_state[num_variations_key] = num_vars
        with col2:
            if st.button(f"Regenerate", key=f"regen_{scene.index}", type="primary" if not has_image else "secondary"):
                regenerate_single_image(current_state, scene.index, num_variations=num_vars)
        with col3:
            if has_image:
                with open(scene.image_path, "rb") as f:
                    st.download_button(
                        "Download",
                        data=f.read(),
                        file_name=f"scene_{scene.index + 1}.png",
                        mime="image/png",
                        key=f"download_{scene.index}",
                    )


def render_video_complete(state) -> None:
    """Render the video complete view."""
    st.markdown("---")
    st.subheader("Your Music Video is Ready!")

    # Video player
    try:
        st.video(state.final_video_path)
    except Exception:
        st.info(f"Video saved to: {state.final_video_path}")

    col1, col2 = st.columns(2)

    with col1:
        with open(state.final_video_path, "rb") as f:
            st.download_button(
                "Download MP4",
                data=f,
                file_name=Path(state.final_video_path).name,
                mime="video/mp4",
            )

    with col2:
        if st.button("Create Another Video"):
            update_state(
                final_video_path=None,
                scenes=[],
                generated_images=[],
                prompts_ready=False,
                storyboard_ready=False,
                project_dir=None,
            )
            st.rerun()

    if st.button("Complete", type="primary"):
        advance_step()
        st.rerun()

    # Back button
    st.markdown("---")
    if st.button("Back to Storyboard"):
        update_state(final_video_path=None)
        st.rerun()


def generate_scene_prompts(state, scenes_per_minute: int, style_override: str, resolution: str) -> None:
    """Generate scene prompts without generating images yet."""
    config.ensure_directories()

    # Create unique output directory for this project
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = (
        _sanitize_filename(state.lyrics.title)
        if state.lyrics
        else "untitled"
    )
    project_dir = config.output_dir / f"{project_name}_{timestamp}"
    project_dir.mkdir(parents=True, exist_ok=True)

    # Save project directory and resolution
    update_state(project_dir=str(project_dir), video_resolution=resolution)

    with st.status("Creating scene prompts...", expanded=True) as status:
        # Step 1: Plan scenes
        st.write("Planning visual scenes...")
        visual_agent = VisualAgent()

        # Override style if provided
        if style_override and state.concept:
            state.concept.visual_style = style_override

        scenes = visual_agent.plan_video(
            concept=state.concept,
            transcript=state.transcript,
            full_lyrics=state.lyrics.lyrics if state.lyrics else "",
        )

        st.write(f"Created {len(scenes)} scene prompts")

        update_state(
            scenes=scenes,
            prompts_ready=True,
            storyboard_ready=False,
        )

        status.update(label="Scene prompts ready for review!", state="complete")

    st.success("Review and edit your prompts, then generate images!")
    st.rerun()


def generate_images_from_prompts(state, is_demo_mode: bool) -> None:
    """Generate images from the approved prompts."""
    project_dir = getattr(state, 'project_dir', None)

    # Create project directory if it doesn't exist (e.g., when coming from Visual Workshop)
    if not project_dir:
        config.ensure_directories()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = (
            _sanitize_filename(state.lyrics.title)
            if state.lyrics
            else "untitled"
        )
        project_dir = config.output_dir / f"{project_name}_{timestamp}"
        project_dir.mkdir(parents=True, exist_ok=True)
        update_state(project_dir=str(project_dir))

    project_dir = Path(project_dir)
    scenes = state.scenes
    resolution = getattr(state, 'video_resolution', '1080p')
    res_info = RESOLUTION_OPTIONS.get(resolution, RESOLUTION_OPTIONS["1080p"])
    sequential_mode = getattr(state, 'use_sequential_mode', False)
    parallel_workers = getattr(state, 'parallel_workers', 4) if not sequential_mode else 1

    with st.status("Generating scene images...", expanded=True) as status:
        if sequential_mode:
            mode_str = " with sequential consistency"
        elif parallel_workers > 1:
            mode_str = f" in parallel ({parallel_workers} workers)"
        else:
            mode_str = ""
        st.write(f"Generating {len(scenes)} images at {resolution} resolution{mode_str}...")
        image_gen = ImageGenerator()

        images_dir = project_dir / "images"
        images_dir.mkdir(exist_ok=True)

        progress_bar = st.progress(0.0, text="Generating images...")

        prompts = [scene.visual_prompt for scene in scenes]
        style_prefix = (
            state.concept.visual_style
            if state.concept and state.concept.visual_style
            else config.image.default_style
        )
        character_desc = state.concept.character_description if state.concept else None
        visual_world = state.concept.visual_world if state.concept else None

        # Load hero image if set
        hero_image = None
        hero_image_path = getattr(state, 'hero_image_path', None)
        if hero_image_path and Path(hero_image_path).exists():
            from PIL import Image as PILImage
            hero_image = PILImage.open(hero_image_path)
            st.write(f"Using hero image for visual consistency")

        def image_progress(msg: str, prog: float):
            progress_bar.progress(prog, text=msg)

        # Pass resolution, sequential mode, parallel workers, visual_world and hero_image to image generator
        image_paths = image_gen.generate_storyboard(
            scene_prompts=prompts,
            style_prefix=style_prefix,
            character_description=character_desc or "",
            output_dir=images_dir,
            progress_callback=image_progress,
            image_size=res_info["image_size"],
            sequential_mode=sequential_mode,
            visual_world=visual_world,
            max_workers=parallel_workers,
            hero_image=hero_image,
            show_character_flags=[s.show_character for s in scenes],
        )

        # Ensure we have the same number of paths as scenes
        while len(image_paths) < len(scenes):
            image_paths.append(None)

        # Update scenes with image paths
        for i, scene in enumerate(scenes):
            if i < len(image_paths) and image_paths[i]:
                scene.image_path = image_paths[i]
            else:
                scene.image_path = None

        # Count successful images
        successful_images = [p for p in image_paths if p is not None]

        update_state(
            scenes=scenes,
            generated_images=[str(p) for p in successful_images],
            storyboard_ready=True,
        )

        # Save scene metadata (prompts, effects, etc.) for recovery
        save_scene_metadata(Path(project_dir), scenes)

        st.write(f"Generated {len(successful_images)}/{len(scenes)} images")
        status.update(label="Storyboard complete!", state="complete")

    st.success("Storyboard ready for review!")
    st.rerun()


def regenerate_single_image(state, scene_index: int, num_variations: int = 3) -> None:
    """Regenerate a single scene's image with multiple variations for selection."""
    # Read fresh state to handle fragment reruns correctly
    current_state = get_state()
    project_dir = getattr(current_state, 'project_dir', None)
    if not project_dir:
        st.error("No project directory found. Please start over.")
        return

    scenes = current_state.scenes
    if scene_index >= len(scenes):
        st.error(f"Invalid scene index: {scene_index}")
        return

    scene = scenes[scene_index]
    images_dir = Path(project_dir) / "images"
    variations_dir = images_dir / "variations" / f"scene_{scene_index:03d}"
    resolution = getattr(current_state, 'video_resolution', '1080p')
    res_info = RESOLUTION_OPTIONS.get(resolution, RESOLUTION_OPTIONS["1080p"])

    with st.spinner(f"Generating {num_variations} variations for scene {scene_index + 1}..."):
        image_gen = ImageGenerator()

        style_prefix = (
            current_state.concept.visual_style
            if current_state.concept and current_state.concept.visual_style
            else config.image.default_style
        )
        character_desc = current_state.concept.character_description if current_state.concept else None
        visual_world = current_state.concept.visual_world if current_state.concept else None

        # Generate multiple variations
        variations = image_gen.generate_scene_variations(
            prompt=scene.visual_prompt,
            num_variations=num_variations,
            style_prefix=style_prefix,
            character_description=character_desc,
            visual_world=visual_world,
            output_dir=variations_dir,
            image_size=res_info["image_size"],
        )

        if variations:
            # Store variation paths in session state for selection UI
            variation_paths = [str(path) for _, path in variations if path]
            st.session_state[f"scene_variations_{scene_index}"] = variation_paths
            st.success(f"Generated {len(variation_paths)} variations for scene {scene_index + 1}. Select one below!")
        else:
            st.error(f"Failed to generate variations for scene {scene_index + 1}")

    st.rerun()


def select_scene_variation(state, scene_index: int, variation_path: str) -> None:
    """Select a variation as the main scene image."""
    import shutil
    from datetime import datetime

    # Read fresh state to handle fragment reruns correctly
    current_state = get_state()
    project_dir = getattr(current_state, 'project_dir', None)
    if not project_dir:
        st.error("No project directory found.")
        return

    scenes = current_state.scenes
    if scene_index >= len(scenes):
        st.error(f"Invalid scene index: {scene_index}")
        return

    images_dir = Path(project_dir) / "images"
    output_path = images_dir / f"scene_{scene_index:03d}.png"

    # Copy selected variation to main scene image path
    shutil.copy(variation_path, output_path)

    # Update scene image path
    scenes[scene_index].image_path = output_path

    # Archive any existing animation since the source image changed
    # Move old animations to archive folder instead of deleting
    animations_dir = Path(project_dir) / "animations"
    archive_dir = animations_dir / "archive"
    expected_anim = animations_dir / f"animated_scene_{scene_index:03d}.mp4"

    archived_animation = False
    if expected_anim.exists():
        try:
            archive_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = archive_dir / f"scene_{scene_index:03d}_{timestamp}.mp4"
            shutil.move(str(expected_anim), str(archive_path))
            archived_animation = True
        except Exception:
            pass  # Ignore archive errors

    # Also archive if video_path points elsewhere
    if hasattr(scenes[scene_index], 'video_path') and scenes[scene_index].video_path:
        old_video = Path(scenes[scene_index].video_path)
        if old_video.exists() and old_video != expected_anim:
            try:
                archive_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_path = archive_dir / f"scene_{scene_index:03d}_{timestamp}.mp4"
                shutil.move(str(old_video), str(archive_path))
                archived_animation = True
            except Exception:
                pass

    # Reset animation state so user can re-animate with new image
    scenes[scene_index].video_path = None
    scenes[scene_index].animated = False
    scenes[scene_index].animation_type = AnimationType.NONE

    update_state(scenes=scenes)

    # Clear variations from session state
    if f"scene_variations_{scene_index}" in st.session_state:
        del st.session_state[f"scene_variations_{scene_index}"]

    if archived_animation:
        st.success(f"Selected variation applied to scene {scene_index + 1}! Old animation archived to animations/archive/")
    else:
        st.success(f"Selected variation applied to scene {scene_index + 1}!")
    st.rerun()


def clear_scene_variations(scene_index: int) -> None:
    """Clear pending variations for a scene."""
    if f"scene_variations_{scene_index}" in st.session_state:
        del st.session_state[f"scene_variations_{scene_index}"]
    st.rerun()


def get_existing_variations(project_dir: str, scene_index: int) -> list[str]:
    """Get existing variation image paths for a scene from disk."""
    if not project_dir:
        return []

    variations_dir = Path(project_dir) / "images" / "variations" / f"scene_{scene_index:03d}"
    if not variations_dir.exists():
        return []

    # Find all variation_XX.png files
    variation_files = sorted(variations_dir.glob("variation_*.png"))
    return [str(f) for f in variation_files if f.exists()]


def show_existing_variations(project_dir: str, scene_index: int) -> None:
    """Load existing variations into session state for display."""
    variations = get_existing_variations(project_dir, scene_index)
    if variations:
        st.session_state[f"scene_variations_{scene_index}"] = variations
        st.rerun()
    else:
        st.warning(f"No variations found for scene {scene_index + 1}")


def regenerate_missing_images(state) -> None:
    """Regenerate only the missing scene images."""
    project_dir = getattr(state, 'project_dir', None)
    if not project_dir:
        st.error("No project directory found. Please start over.")
        return

    images_dir = Path(project_dir) / "images"
    scenes = state.scenes
    resolution = getattr(state, 'video_resolution', '1080p')
    res_info = RESOLUTION_OPTIONS.get(resolution, RESOLUTION_OPTIONS["1080p"])

    # Find missing images
    missing_indices = [
        i for i, s in enumerate(scenes)
        if not s.image_path or not Path(s.image_path).exists()
    ]

    if not missing_indices:
        st.info("All images are already generated!")
        return

    with st.status(f"Generating {len(missing_indices)} missing images...", expanded=True) as status:
        image_gen = ImageGenerator()

        style_prefix = (
            state.concept.visual_style
            if state.concept and state.concept.visual_style
            else config.image.default_style
        )
        character_desc = state.concept.character_description if state.concept else None
        visual_world = state.concept.visual_world if state.concept else None

        progress_bar = st.progress(0.0, text="Generating missing images...")

        for idx, i in enumerate(missing_indices):
            progress_bar.progress(
                idx / len(missing_indices),
                text=f"Generating scene {i + 1} ({idx + 1}/{len(missing_indices)})..."
            )

            output_path = images_dir / f"scene_{i:03d}.png"

            image = image_gen.generate_scene_image(
                prompt=scenes[i].visual_prompt,
                style_prefix=style_prefix,
                character_description=character_desc,
                visual_world=visual_world,
                output_path=output_path,
                image_size=res_info["image_size"],
                show_character=scenes[i].show_character,
            )

            if image and output_path.exists():
                scenes[i].image_path = output_path

        progress_bar.progress(1.0, text="Done!")
        update_state(scenes=scenes)

        # Count successful regenerations
        still_missing = sum(
            1 for i in missing_indices
            if not scenes[i].image_path or not Path(scenes[i].image_path).exists()
        )
        if still_missing > 0:
            status.update(
                label=f"Completed - {still_missing} still missing",
                state="error"
            )
        else:
            status.update(label="All missing images generated!", state="complete")

    st.rerun()


def generate_single_animation(state, scene_index: int, resolution: str = "720P") -> None:
    """Re-animate a single scene."""
    project_dir = getattr(state, 'project_dir', None)
    if not project_dir:
        st.error("No project directory found.")
        return

    audio_path = getattr(state, 'audio_path', None)
    if not audio_path or audio_path == "demo_mode":
        st.error("No audio file found.")
        return

    scenes = state.scenes
    if scene_index >= len(scenes):
        st.error("Invalid scene index.")
        return

    scene = scenes[scene_index]
    if not scene.image_path or not Path(scene.image_path).exists():
        st.error("Scene has no image to animate.")
        return

    animations_dir = Path(project_dir) / "animations"
    animations_dir.mkdir(parents=True, exist_ok=True)

    # Delete existing animation file if it exists (for re-animation)
    output_path = animations_dir / f"animated_scene_{scene_index:03d}.mp4"
    if output_path.exists():
        output_path.unlink()
        # Also clear the video_path in state so we don't think it's already done
        scenes[scene_index].video_path = None
        update_state(scenes=scenes)

    with st.status(f"Re-animating scene {scene_index + 1} at {resolution}...", expanded=True) as status:
        st.write("Connecting to Wan2.2-S2V...")
        animator = LipSyncAnimator()
        output_path = animations_dir / f"animated_scene_{scene_index:03d}.mp4"

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def progress_callback(msg: str, prog: float):
            progress_bar.progress(prog)
            status_text.text(msg)

        try:
            result = animator.animate_scene(
                image_path=Path(scene.image_path),
                audio_path=Path(audio_path),
                start_time=scene.start_time,
                duration=scene.duration,
                output_path=output_path,
                resolution=resolution,
                progress_callback=progress_callback,
            )

            if result and result.exists():
                scenes[scene_index].video_path = result
                scenes[scene_index].animated = True  # Mark as animated so UI shows video
                update_state(scenes=scenes)
                status.update(label=f"Scene {scene_index + 1} re-animated!", state="complete")
            else:
                status.update(label=f"Animation failed for scene {scene_index + 1}", state="error")

        except Exception as e:
            status.update(label=f"Animation error: {e}", state="error")

    st.rerun()


def _animate_single_scene_worker(
    scene,
    audio_path: Optional[str],
    output_path: Path,
    resolution: str,
) -> dict:
    """
    Thread-safe worker function to animate a single scene.

    This function does NOT call st.write or any Streamlit functions.
    All results are collected and returned for display by the main thread.

    Returns:
        dict with keys:
            - scene_index: int
            - success: bool
            - result_path: Optional[Path]
            - messages: list[str] - log messages for display
            - error: Optional[str] - error message if failed
    """
    import shutil
    import re

    messages = []
    anim_type = getattr(scene, 'animation_type', AnimationType.LIP_SYNC)

    anim_type_names = {
        AnimationType.LIP_SYNC: "Wan S2V Lip (HF)",
        AnimationType.PROMPT: "Wan TI2V (HF)",
        AnimationType.VEO: "Veo 3.1 (Google)",
        AnimationType.ATLASCLOUD: "Wan 2.5 (AtlasCloud)",
        AnimationType.SEEDANCE: "Seedance 1.5 (AtlasCloud)",
        AnimationType.KLING: "Kling Lip (fal.ai)",
        AnimationType.WAN_S2V: "Wan S2V (fal.ai)",
        AnimationType.SEEDANCE_LIPSYNC: "Seedance+Kling (Atlas+fal)",
    }
    anim_type_name = anim_type_names.get(anim_type, "unknown")

    # Thread-safe progress callback that collects messages
    def progress_callback(msg: str, prog: float):
        messages.append(msg)

    try:
        result = None

        if anim_type == AnimationType.LIP_SYNC:
            if not audio_path or audio_path == "demo_mode":
                return {
                    "scene_index": scene.index,
                    "success": False,
                    "result_path": None,
                    "messages": [f"Skipped - no audio for lip sync"],
                    "error": "No audio",
                    "skipped": True,
                }

            messages.append(f"Using Wan2.2-S2V for lip-sync (FREE)")
            animator = LipSyncAnimator()
            result = animator.animate_scene(
                image_path=Path(scene.image_path),
                audio_path=Path(audio_path),
                start_time=scene.start_time,
                duration=scene.duration,
                output_path=output_path,
                resolution=resolution,
                progress_callback=progress_callback,
            )

        elif anim_type == AnimationType.PROMPT:
            messages.append(f"Using Wan2.2-TI2V-5B for prompt animation (FREE)")
            animator = PromptAnimationChainer()
            motion_prompt = _get_motion_prompt(scene)
            result = animator.animate_scene(
                image_path=Path(scene.image_path),
                prompt=motion_prompt,
                output_path=output_path,
                duration_seconds=scene.duration,
                quality_preset="fast",
                progress_callback=progress_callback,
            )

        elif anim_type == AnimationType.VEO:
            messages.append(f"Using Google Veo 3.1 (PAID)")
            animator = VeoAnimationChainer()
            motion_prompt = _get_motion_prompt(scene)
            result = animator.animate_scene(
                image_path=Path(scene.image_path),
                prompt=motion_prompt,
                output_path=output_path,
                duration_seconds=scene.duration,
                resolution="720p",
                progress_callback=progress_callback,
            )

        elif anim_type == AnimationType.ATLASCLOUD:
            messages.append(f"Using AtlasCloud Wan 2.5 (PAID)")
            animator = AtlasCloudAnimator()
            motion_prompt = _get_motion_prompt(scene)
            result = animator.animate_scene(
                image_path=Path(scene.image_path),
                prompt=motion_prompt,
                output_path=output_path,
                duration_seconds=5 if scene.duration < 7.5 else 10,
                resolution="720p",
                progress_callback=progress_callback,
            )

        elif anim_type == AnimationType.SEEDANCE:
            messages.append(f"Using Seedance Pro (PAID)")
            animator = SeedanceAnimator()
            motion_prompt = _get_motion_prompt(scene)
            # Use ceil to ensure animation covers full scene (avoids Ken Burns padding)
            target_duration = min(12, max(2, math.ceil(scene.duration)))
            result = animator.animate_scene(
                image_path=Path(scene.image_path),
                prompt=motion_prompt,
                output_path=output_path,
                duration_seconds=target_duration,
                resolution="720p",
                progress_callback=progress_callback,
            )

        elif anim_type == AnimationType.KLING:
            if not audio_path or audio_path == "demo_mode":
                return {
                    "scene_index": scene.index,
                    "success": False,
                    "result_path": None,
                    "messages": [f"Skipped - no audio for Kling lip sync"],
                    "error": "No audio",
                    "skipped": True,
                }

            messages.append(f"Using Kling (PAID, fal.ai)")
            animator = KlingAnimator()
            result = animator.animate_scene(
                image_path=Path(scene.image_path),
                audio_path=Path(audio_path),
                start_time=scene.start_time,
                duration=scene.duration,
                output_path=output_path,
                resolution="720p",
                use_i2v=False,
                progress_callback=progress_callback,
            )

        elif anim_type == AnimationType.WAN_S2V:
            if not audio_path or audio_path == "demo_mode":
                return {
                    "scene_index": scene.index,
                    "success": False,
                    "result_path": None,
                    "messages": [f"Skipped - no audio for Wan S2V"],
                    "error": "No audio",
                    "skipped": True,
                }

            messages.append(f"Using Wan S2V (PAID, fal.ai)")
            animator = WanS2VAnimator()
            motion_prompt = _get_motion_prompt(scene)
            result = animator.animate_scene(
                image_path=Path(scene.image_path),
                audio_path=Path(audio_path),
                start_time=scene.start_time,
                duration=scene.duration,
                output_path=output_path,
                prompt=motion_prompt,
                progress_callback=progress_callback,
            )

        elif anim_type == AnimationType.SEEDANCE_LIPSYNC:
            if not audio_path or audio_path == "demo_mode":
                return {
                    "scene_index": scene.index,
                    "success": False,
                    "result_path": None,
                    "messages": [f"Skipped - no audio for lip sync"],
                    "error": "No audio",
                    "skipped": True,
                }

            motion_prompt = _get_motion_prompt(scene)
            # Kling lipsync max 10s - use ceil to ensure animation covers full scene
            target_duration = min(10, max(2, math.ceil(scene.duration)))

            motion_output = output_path.with_suffix(".motion.mp4")
            motion_result = None

            if motion_output.exists():
                messages.append(f"Step 1: ✅ Motion video exists, resuming...")
                motion_result = motion_output
            else:
                messages.append(f"Step 1: Generating motion with Seedance Pro...")
                seedance_animator = SeedanceAnimator()
                motion_result = seedance_animator.animate_scene(
                    image_path=Path(scene.image_path),
                    prompt=motion_prompt,
                    output_path=motion_output,
                    duration_seconds=target_duration,
                    resolution="720p",
                    progress_callback=progress_callback,
                )

            if motion_result and motion_result.exists():
                messages.append(f"Step 2: Applying lip sync with Kling...")
                lipsync_animator = KlingAnimator()
                result = lipsync_animator.apply_lipsync_to_video(
                    video_path=motion_result,
                    audio_path=Path(audio_path),
                    start_time=scene.start_time,
                    duration=scene.duration,
                    output_path=output_path,
                    progress_callback=progress_callback,
                )

                if result and result.exists():
                    motion_output.unlink(missing_ok=True)
                else:
                    messages.append(f"Lip sync failed, using motion video as fallback")
                    shutil.move(str(motion_output), str(output_path))
                    result = output_path
            else:
                messages.append(f"Motion generation failed")
                result = None

        if result and result.exists():
            messages.append(f"✅ Animated successfully")
            return {
                "scene_index": scene.index,
                "success": True,
                "result_path": result,
                "messages": messages,
                "error": None,
            }
        else:
            return {
                "scene_index": scene.index,
                "success": False,
                "result_path": None,
                "messages": messages,
                "error": "Animation failed",
            }

    except Exception as e:
        error_str = str(e)
        if "GPU quota" in error_str or "exceeded" in error_str.lower():
            time_match = re.search(r"Try again in (\d+:\d+:\d+)", error_str)
            wait_time = time_match.group(1) if time_match else "~24 hours"
            user_msg = f"GPU quota exceeded. Try again in {wait_time}."
        else:
            user_msg = str(e)[:150]

        messages.append(f"Error: {user_msg}")
        return {
            "scene_index": scene.index,
            "success": False,
            "result_path": None,
            "messages": messages,
            "error": user_msg,
        }


def generate_animations(state, resolution: str = "480P", is_demo_mode: bool = False, max_workers: int = 3) -> None:
    """Generate animations for scenes marked for animation (respects animation type).

    Args:
        state: Application state
        resolution: Video resolution ("480P" or "720P")
        is_demo_mode: Whether we're in demo mode (no real audio)
        max_workers: Number of parallel animation jobs (1 = sequential)
    """
    if is_demo_mode:
        st.warning("Animation generation is not available in demo mode. Please upload real audio.")
        return

    project_dir = getattr(state, 'project_dir', None)
    if not project_dir:
        st.error("No project directory found. Please start over.")
        return

    audio_path = getattr(state, 'audio_path', None)

    animations_dir = Path(project_dir) / "animations"
    animations_dir.mkdir(parents=True, exist_ok=True)

    scenes = state.scenes

    # Find scenes that need animation (check animation_type, not just 'animated' flag)
    pending_scenes = [
        s for s in scenes
        if getattr(s, 'animation_type', AnimationType.NONE) != AnimationType.NONE
        and s.image_path
        and Path(s.image_path).exists()
        and (not getattr(s, 'video_path', None) or not Path(s.video_path).exists())
    ]

    if not pending_scenes:
        # Debug: show why no scenes need animation
        scenes_with_animation = [s for s in scenes if getattr(s, 'animation_type', AnimationType.NONE) != AnimationType.NONE]
        if scenes_with_animation:
            already_done = [s for s in scenes_with_animation if getattr(s, 'video_path', None) and Path(s.video_path).exists()]
            if already_done:
                st.info(f"All {len(already_done)} animated scenes already have videos. Delete the animation files to regenerate.")
            else:
                st.info("No scenes need animation!")
        else:
            st.info("No scenes have animation types set. Select an animation type for scenes you want to animate.")
        return

    # Count by type
    lip_sync_count = sum(1 for s in pending_scenes if getattr(s, 'animation_type', None) == AnimationType.LIP_SYNC)
    prompt_count = sum(1 for s in pending_scenes if getattr(s, 'animation_type', None) == AnimationType.PROMPT)
    veo_count = sum(1 for s in pending_scenes if getattr(s, 'animation_type', None) == AnimationType.VEO)
    atlascloud_count = sum(1 for s in pending_scenes if getattr(s, 'animation_type', None) == AnimationType.ATLASCLOUD)
    seedance_count = sum(1 for s in pending_scenes if getattr(s, 'animation_type', None) == AnimationType.SEEDANCE)
    kling_count = sum(1 for s in pending_scenes if getattr(s, 'animation_type', None) == AnimationType.KLING)
    wan_s2v_count = sum(1 for s in pending_scenes if getattr(s, 'animation_type', None) == AnimationType.WAN_S2V)
    seedance_lipsync_count = sum(1 for s in pending_scenes if getattr(s, 'animation_type', None) == AnimationType.SEEDANCE_LIPSYNC)

    status_label = f"Generating {len(pending_scenes)} animations"
    type_parts = []
    if lip_sync_count > 0:
        type_parts.append(f"{lip_sync_count} lip-sync")
    if prompt_count > 0:
        type_parts.append(f"{prompt_count} prompt")
    if veo_count > 0:
        type_parts.append(f"{veo_count} Veo")
    if atlascloud_count > 0:
        type_parts.append(f"{atlascloud_count} AtlasCloud")
    if seedance_count > 0:
        type_parts.append(f"{seedance_count} Seedance")
    if kling_count > 0:
        type_parts.append(f"{kling_count} Kling")
    if wan_s2v_count > 0:
        type_parts.append(f"{wan_s2v_count} Wan S2V")
    if seedance_lipsync_count > 0:
        type_parts.append(f"{seedance_lipsync_count} Seedance+LipSync")
    if type_parts:
        status_label += f" ({', '.join(type_parts)})"

    with st.status(f"{status_label}...", expanded=True) as status:
        parallel_mode = max_workers > 1
        if parallel_mode:
            st.write(f"Resolution: {resolution} | Parallel jobs: {max_workers}")
        else:
            st.write(f"Resolution: {resolution} | Sequential mode")

        progress_bar = st.progress(0.0, text="Starting animations...")
        last_error_msg = None

        # Build list of tasks to execute
        tasks = []
        for scene in pending_scenes:
            output_path = animations_dir / f"animated_scene_{scene.index:03d}.mp4"
            tasks.append((scene, output_path))

        success_count = 0
        results_by_scene = {}  # scene.index -> result dict

        if parallel_mode and len(tasks) > 1:
            # PARALLEL EXECUTION using ThreadPoolExecutor
            st.write(f"Launching {len(tasks)} animations in parallel...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(
                        _animate_single_scene_worker,
                        scene,
                        audio_path,
                        output_path,
                        resolution,
                    ): scene.index
                    for scene, output_path in tasks
                }

                # Process results as they complete
                completed_count = 0
                for future in as_completed(futures):
                    scene_index = futures[future]
                    completed_count += 1

                    progress_bar.progress(
                        completed_count / len(tasks),
                        text=f"Completed {completed_count}/{len(tasks)} animations..."
                    )

                    try:
                        result = future.result()
                        results_by_scene[scene_index] = result
                    except Exception as e:
                        results_by_scene[scene_index] = {
                            "scene_index": scene_index,
                            "success": False,
                            "result_path": None,
                            "messages": [f"Thread error: {str(e)[:100]}"],
                            "error": str(e)[:100],
                        }

            # Display all results in scene order
            st.write("---")
            st.write("**Results:**")
            for scene, _ in tasks:
                result = results_by_scene.get(scene.index, {})
                scene_label = f"Scene {scene.index + 1}"

                # Show messages for this scene
                for msg in result.get("messages", []):
                    st.write(f"  [{scene_label}] {msg}")

                if result.get("success"):
                    scene.video_path = result.get("result_path")
                    scene.animated = True
                    success_count += 1
                    st.write(f"✅ {scene_label} animated successfully")
                elif result.get("skipped"):
                    st.write(f"⚠️ {scene_label}: {result.get('error', 'Skipped')}")
                else:
                    error = result.get("error", "Unknown error")
                    st.write(f"❌ {scene_label}: {error}")
                    last_error_msg = error

        else:
            # SEQUENTIAL EXECUTION (max_workers=1 or single task)
            for idx, (scene, output_path) in enumerate(tasks):
                anim_type = getattr(scene, 'animation_type', AnimationType.LIP_SYNC)
                anim_type_names = {
                    AnimationType.LIP_SYNC: "Wan S2V Lip (HF)",
                    AnimationType.PROMPT: "Wan TI2V (HF)",
                    AnimationType.VEO: "Veo 3.1 (Google)",
                    AnimationType.ATLASCLOUD: "Wan 2.5 (AtlasCloud)",
                    AnimationType.SEEDANCE: "Seedance 1.5 (AtlasCloud)",
                    AnimationType.KLING: "Kling Lip (fal.ai)",
                    AnimationType.WAN_S2V: "Wan S2V (fal.ai)",
                    AnimationType.SEEDANCE_LIPSYNC: "Seedance+Kling (Atlas+fal)",
                }
                anim_type_name = anim_type_names.get(anim_type, "unknown")

                progress_bar.progress(
                    idx / len(tasks),
                    text=f"Animating scene {scene.index + 1} ({anim_type_name}) ({idx + 1}/{len(tasks)})..."
                )

                st.write(f"**Scene {scene.index + 1}** ({anim_type_name}):")

                # Use the worker function but show messages in real-time
                result = _animate_single_scene_worker(
                    scene,
                    audio_path,
                    output_path,
                    resolution,
                )

                # Display messages
                for msg in result.get("messages", []):
                    st.write(f"  {msg}")

                if result.get("success"):
                    scene.video_path = result.get("result_path")
                    scene.animated = True
                    success_count += 1
                    st.write(f"✅ Scene {scene.index + 1} animated successfully")
                elif result.get("skipped"):
                    st.write(f"⚠️ Scene {scene.index + 1}: {result.get('error', 'Skipped')}")
                else:
                    error = result.get("error", "Unknown error")
                    st.write(f"❌ Scene {scene.index + 1}: {error}")
                    last_error_msg = error

        progress_bar.progress(1.0, text="Done!")
        update_state(scenes=scenes)

        if success_count == len(pending_scenes):
            status.update(
                label=f"All {success_count} animations generated!",
                state="complete"
            )
        elif success_count > 0:
            status.update(
                label=f"{success_count}/{len(pending_scenes)} animations generated",
                state="error"
            )
        else:
            error_hint = f" - {last_error_msg}" if last_error_msg else ""
            status.update(
                label=f"Animation generation failed{error_hint}",
                state="error"
            )

    st.rerun()


def regenerate_all_images(state) -> None:
    """Regenerate all scene images."""
    project_dir = getattr(state, 'project_dir', None)
    if not project_dir:
        st.error("No project directory found. Please start over.")
        return

    images_dir = Path(project_dir) / "images"
    scenes = state.scenes
    resolution = getattr(state, 'video_resolution', '1080p')
    res_info = RESOLUTION_OPTIONS.get(resolution, RESOLUTION_OPTIONS["1080p"])

    with st.status("Regenerating all images...", expanded=True) as status:
        image_gen = ImageGenerator()

        style_prefix = (
            state.concept.visual_style
            if state.concept and state.concept.visual_style
            else config.image.default_style
        )
        character_desc = state.concept.character_description if state.concept else None
        visual_world = state.concept.visual_world if state.concept else None

        progress_bar = st.progress(0.0, text="Regenerating images...")

        for i, scene in enumerate(scenes):
            progress_bar.progress(
                i / len(scenes), text=f"Regenerating scene {i + 1}/{len(scenes)}..."
            )

            output_path = images_dir / f"scene_{i:03d}.png"

            image = image_gen.generate_scene_image(
                prompt=scene.visual_prompt,
                style_prefix=style_prefix,
                character_description=character_desc,
                visual_world=visual_world,
                output_path=output_path,
                image_size=res_info["image_size"],
                show_character=scene.show_character,
            )

            if image:
                scenes[i].image_path = output_path

        progress_bar.progress(1.0, text="Done!")
        update_state(scenes=scenes)
        status.update(label="All images regenerated!", state="complete")

    st.rerun()


def generate_video_from_storyboard(state, crossfade: float, is_demo_mode: bool) -> None:
    """Generate the final video from the approved storyboard."""
    from src.ui.components.state import fix_malformed_project_path

    project_dir = getattr(state, 'project_dir', None)
    if not project_dir:
        st.error("No project directory found. Please start over.")
        return

    project_dir = Path(project_dir)

    # Save scene metadata before creating video (captures any user edits)
    if state.scenes:
        save_scene_metadata(project_dir, state.scenes)

    # Check if project directory exists, try to fix if not
    if not project_dir.exists():
        # Try to fix malformed path
        if fix_malformed_project_path(state):
            project_dir = Path(state.project_dir)
            update_state(project_dir=str(project_dir))
            st.info("Project directory path was fixed.")
        else:
            st.error(f"Project directory not found: {project_dir}. Please start over.")
            return
    scenes = state.scenes
    resolution = getattr(state, 'video_resolution', '1080p')
    res_info = RESOLUTION_OPTIONS.get(resolution, RESOLUTION_OPTIONS["1080p"])
    show_lyrics = getattr(state, 'show_lyrics', True)
    fps = getattr(state, 'video_fps', 30)

    # Verify we have images
    scenes_with_images = [s for s in scenes if s.image_path and Path(s.image_path).exists()]
    if not scenes_with_images:
        st.error("No scene images found. Please regenerate the storyboard.")
        return

    project_name = (
        _sanitize_filename(state.lyrics.title)
        if state.lyrics
        else "untitled"
    )

    with st.status("Creating your music video...", expanded=True) as status:
        subtitle_path = None

        # Step 1: Generate subtitles only if show_lyrics is enabled
        if show_lyrics:
            st.write("Creating karaoke subtitles...")
            subtitle_gen = SubtitleGenerator()

            # Collect all words from scenes (may have been edited by user)
            all_words = []
            for scene in scenes:
                if scene.words:
                    all_words.extend(scene.words)

            # Sort by start time to ensure proper order
            all_words.sort(key=lambda w: w.start)

            subtitle_path = project_dir / "lyrics.ass"
            subtitle_gen.generate_karaoke_ass(
                words=all_words,
                output_path=subtitle_path,
            )
            st.write("   Subtitles created")
        else:
            st.write("   Skipping lyrics overlay (disabled)")

        # Step 2: Generate video
        st.write(
            f"Assembling video at {resolution} "
            f"({res_info['width']}x{res_info['height']}) @ {fps}fps..."
        )
        video_gen = VideoGenerator()

        video_progress = st.progress(0.0, text="Creating video...")

        def video_progress_callback(msg: str, prog: float):
            video_progress.progress(prog, text=msg)

        output_path = project_dir / f"{project_name}.mp4"

        if is_demo_mode:
            # Demo mode: generate slideshow without audio
            st.write("   (Demo mode: creating slideshow without audio)")
            video_gen.generate_slideshow(
                scenes=scenes,
                subtitle_path=subtitle_path if show_lyrics else None,
                output_path=output_path,
                progress_callback=video_progress_callback,
                resolution=(res_info['width'], res_info['height']),
                fps=fps,
            )
        else:
            # Full mode: generate with audio
            ext_mode = getattr(state, 'extension_mode', 'all')
            st.write(f"Extension mode: **{ext_mode}**")
            video_gen.generate_music_video(
                scenes=scenes,
                audio_path=Path(state.audio_path),
                subtitle_path=subtitle_path if show_lyrics else None,
                output_path=output_path,
                progress_callback=video_progress_callback,
                resolution=(res_info['width'], res_info['height']),
                fps=fps,
                extension_mode=ext_mode,
            )

        update_state(final_video_path=str(output_path))
        status.update(label="Video complete!", state="complete")

    st.success(f"Video saved to: {output_path}")
    st.rerun()
