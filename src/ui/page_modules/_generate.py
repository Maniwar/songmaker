"""Video Generation page - Step 4 of the workflow."""

from io import BytesIO
from pathlib import Path
from datetime import datetime

import streamlit as st
from pydub import AudioSegment

from src.config import config
from src.agents.visual_agent import VisualAgent
from src.services.audio_processor import AudioProcessor
from src.services.image_generator import ImageGenerator
from src.services.video_generator import VideoGenerator
from src.services.subtitle_generator import SubtitleGenerator
from src.services.lip_sync_animator import LipSyncAnimator, check_lip_sync_available
from src.ui.components.state import get_state, update_state, advance_step, go_to_step
from src.models.schemas import WorkflowStep, Scene, KenBurnsEffect, Word


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
    lyrics_keys_to_clear = [
        f"lyrics_{scene_index}",           # prompt review
        f"card_lyrics_{scene_index}",      # storyboard card
    ]
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


def _redo_transcription(state) -> bool:
    """
    Re-run WhisperX transcription on the audio file.

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


def _redo_scene_transcription(state, scene_index: int) -> bool:
    """
    Re-run WhisperX transcription on just a single scene's audio clip.

    This extracts the audio for the scene's time range, transcribes it,
    and updates the scene's words with properly offset timestamps.

    Args:
        state: App state with audio path and scenes
        scene_index: Index of the scene to re-transcribe

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
            # These keys match the text_area keys in render_prompt_review and render_scene_card
            lyrics_keys_to_clear = [
                f"lyrics_{scene_index}",           # prompt review
                f"card_lyrics_{scene_index}",      # storyboard card
            ]
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
        # Shift timing so scene starts at 0 (for preview)
        if scene.words:
            # Create shifted words for preview (start from 0)
            shifted_words = []
            for w in scene.words:
                shifted_words.append(Word(
                    word=w.word,
                    start=w.start - scene.start_time,
                    end=w.end - scene.start_time,
                ))
            subtitle_gen.generate_karaoke_ass(
                words=shifted_words,
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

    # Action buttons
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("Redo Transcription", type="secondary", help="Re-run WhisperX to get new word timestamps"):
            with st.spinner("Re-transcribing audio..."):
                if _redo_transcription(state):
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

    # Animation stats
    scenes_marked_for_animation = sum(1 for s in state.scenes if getattr(s, 'animated', False))
    scenes_with_animation = sum(
        1 for s in state.scenes
        if getattr(s, 'animated', False)
        and getattr(s, 'video_path', None)
        and Path(s.video_path).exists()
    )
    pending_animations = scenes_marked_for_animation - scenes_with_animation

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

    # Display storyboard grid
    render_storyboard_grid(state)

    st.markdown("---")

    # Animation controls (if any scenes are marked for animation)
    if scenes_marked_for_animation > 0:
        st.subheader("🎬 Lip Sync Animation")
        if not check_lip_sync_available():
            st.warning("Lip sync animation requires `gradio_client`. Install with: `pip install gradio_client`")
        else:
            st.info(
                f"**{scenes_marked_for_animation} scene(s)** marked for animation using Wan2.2-S2V. "
                "This cloud-based service is FREE and works on all platforms."
            )
            anim_col1, anim_col2, anim_col3 = st.columns([2, 1, 1])
            with anim_col1:
                resolution = st.selectbox(
                    "Animation Resolution",
                    options=["480P", "720P"],
                    index=0,
                    help="Higher resolution takes longer to generate",
                    key="animation_resolution",
                )
            with anim_col2:
                if pending_animations > 0:
                    if st.button("🎬 Generate Animations", type="primary"):
                        generate_animations(state, resolution, is_demo_mode)
                else:
                    st.success("All animations ready!")
            with anim_col3:
                if scenes_with_animation > 0:
                    if st.button("🔄 Regenerate All Animations"):
                        # Clear existing animations
                        scenes = state.scenes
                        for s in scenes:
                            if getattr(s, 'animated', False):
                                s.video_path = None
                        update_state(scenes=scenes)
                        generate_animations(state, resolution, is_demo_mode)

        st.markdown("---")

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
                    if _redo_transcription(state):
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
                    if _redo_transcription(state):
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


@st.fragment
def render_scene_card(state, scene: Scene) -> None:
    """Render a single scene card with image and controls. Uses fragment for performance."""
    has_image = scene.image_path and Path(scene.image_path).exists()
    has_animation = getattr(scene, 'video_path', None) and Path(scene.video_path).exists()

    # Scene header with status indicator
    status_icons = ""
    if not has_image:
        status_icons = " :warning:"
    elif getattr(scene, 'animated', False):
        if has_animation:
            status_icons = " :movie_camera:"  # Has animation
        else:
            status_icons = " :hourglass:"  # Pending animation

    st.markdown(f"**Scene {scene.index + 1}**{status_icons}")
    st.caption(f"{scene.start_time:.1f}s - {scene.end_time:.1f}s ({scene.duration:.1f}s)")

    # Image or animation preview
    if has_animation:
        st.video(str(scene.video_path))
    elif has_image:
        st.image(str(scene.image_path), use_container_width=True)
    else:
        st.error("Missing image - click Regenerate")

    # Animation toggle (only show if image exists)
    if has_image:
        animate_checked = st.checkbox(
            "Animate character (lip-sync)",
            value=getattr(scene, 'animated', False),
            key=f"animate_{scene.index}",
            help="Enable to animate this scene with full-body lip-sync singing using Wan2.2-S2V"
        )
        if animate_checked != getattr(scene, 'animated', False):
            scenes = state.scenes
            scenes[scene.index].animated = animate_checked
            if not animate_checked:
                # Clear video path when disabling animation
                scenes[scene.index].video_path = None
            update_state(scenes=scenes)
            st.rerun()

    # Scene details in expander - editable prompt and lyrics
    with st.expander("Edit Scene", expanded=False):
        st.markdown(f"**Mood:** {scene.mood}")

        # Scene timing adjustment
        st.markdown("**Scene Timing**")
        timing_cols = st.columns(2)
        with timing_cols[0]:
            new_start_time = st.number_input(
                "Start (seconds)",
                min_value=0.0,
                max_value=scene.end_time - 0.5,
                value=scene.start_time,
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

        # Transcription buttons for this scene
        trans_col1, trans_col2 = st.columns(2)
        with trans_col1:
            if st.button("Redo Transcription", key=f"redo_trans_{scene.index}", type="secondary", help="Re-transcribe just this scene's audio"):
                with st.spinner("Transcribing scene audio..."):
                    if _redo_scene_transcription(state, scene.index):
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

    # Buttons row
    col1, col2 = st.columns(2)
    with col1:
        # Regenerate button for this scene
        if st.button(f"Regenerate", key=f"regen_{scene.index}", type="primary" if not has_image else "secondary"):
            regenerate_single_image(state, scene.index)
    with col2:
        # Download button for the image
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

        def image_progress(msg: str, prog: float):
            progress_bar.progress(prog, text=msg)

        # Pass resolution, sequential mode, parallel workers and visual_world to image generator
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

        st.write(f"Generated {len(successful_images)}/{len(scenes)} images")
        status.update(label="Storyboard complete!", state="complete")

    st.success("Storyboard ready for review!")
    st.rerun()


def regenerate_single_image(state, scene_index: int) -> None:
    """Regenerate a single scene's image."""
    project_dir = getattr(state, 'project_dir', None)
    if not project_dir:
        st.error("No project directory found. Please start over.")
        return

    scenes = state.scenes
    if scene_index >= len(scenes):
        st.error(f"Invalid scene index: {scene_index}")
        return

    scene = scenes[scene_index]
    images_dir = Path(project_dir) / "images"
    resolution = getattr(state, 'video_resolution', '1080p')
    res_info = RESOLUTION_OPTIONS.get(resolution, RESOLUTION_OPTIONS["1080p"])

    with st.spinner(f"Regenerating scene {scene_index + 1}..."):
        image_gen = ImageGenerator()

        style_prefix = (
            state.concept.visual_style
            if state.concept and state.concept.visual_style
            else config.image.default_style
        )
        character_desc = state.concept.character_description if state.concept else None
        visual_world = state.concept.visual_world if state.concept else None

        output_path = images_dir / f"scene_{scene_index:03d}.png"

        image = image_gen.generate_scene_image(
            prompt=scene.visual_prompt,
            style_prefix=style_prefix,
            character_description=character_desc,
            visual_world=visual_world,
            output_path=output_path,
            image_size=res_info["image_size"],
        )

        if image:
            scenes[scene_index].image_path = output_path
            update_state(scenes=scenes)
            st.success(f"Scene {scene_index + 1} regenerated!")
        else:
            st.error(f"Failed to regenerate scene {scene_index + 1}")

    st.rerun()


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


def generate_animations(state, resolution: str = "480P", is_demo_mode: bool = False) -> None:
    """Generate lip-sync animations for scenes marked for animation."""
    if is_demo_mode:
        st.warning("Animation generation is not available in demo mode. Please upload real audio.")
        return

    project_dir = getattr(state, 'project_dir', None)
    if not project_dir:
        st.error("No project directory found. Please start over.")
        return

    audio_path = getattr(state, 'audio_path', None)
    if not audio_path or audio_path == "demo_mode":
        st.error("No audio file found. Please upload audio first.")
        return

    animations_dir = Path(project_dir) / "animations"
    animations_dir.mkdir(parents=True, exist_ok=True)

    scenes = state.scenes

    # Find scenes that need animation
    pending_scenes = [
        s for s in scenes
        if getattr(s, 'animated', False)
        and s.image_path
        and Path(s.image_path).exists()
        and (not getattr(s, 'video_path', None) or not Path(s.video_path).exists())
    ]

    if not pending_scenes:
        st.info("No scenes need animation!")
        return

    with st.status(f"Generating {len(pending_scenes)} lip-sync animations...", expanded=True) as status:
        st.write("Using Wan2.2-S2V via Hugging Face Spaces (FREE, cloud-based)")
        st.write(f"Resolution: {resolution}")

        animator = LipSyncAnimator()
        progress_bar = st.progress(0.0, text="Connecting to Wan2.2-S2V...")

        success_count = 0
        for idx, scene in enumerate(pending_scenes):
            progress_bar.progress(
                idx / len(pending_scenes),
                text=f"Animating scene {scene.index + 1} ({idx + 1}/{len(pending_scenes)})..."
            )

            output_path = animations_dir / f"animated_scene_{scene.index:03d}.mp4"

            try:
                result = animator.animate_scene(
                    image_path=Path(scene.image_path),
                    audio_path=Path(audio_path),
                    start_time=scene.start_time,
                    duration=scene.duration,
                    output_path=output_path,
                    resolution=resolution,
                    progress_callback=lambda msg, prog: st.write(f"  {msg}"),
                )

                if result and result.exists():
                    # Update scene with video path
                    scenes[scene.index].video_path = result
                    success_count += 1
                    st.write(f"✅ Scene {scene.index + 1} animated successfully")
                else:
                    st.write(f"❌ Scene {scene.index + 1} animation failed")

            except Exception as e:
                st.write(f"❌ Scene {scene.index + 1} error: {e}")

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
            status.update(
                label="Animation generation failed",
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
            video_gen.generate_music_video(
                scenes=scenes,
                audio_path=Path(state.audio_path),
                subtitle_path=subtitle_path if show_lyrics else None,
                output_path=output_path,
                progress_callback=video_progress_callback,
                resolution=(res_info['width'], res_info['height']),
                fps=fps,
            )

        update_state(final_video_path=str(output_path))
        status.update(label="Video complete!", state="complete")

    st.success(f"Video saved to: {output_path}")
    st.rerun()
