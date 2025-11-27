"""Video Generation page - Step 4 of the workflow."""

from io import BytesIO
from pathlib import Path
from datetime import datetime

import streamlit as st
from pydub import AudioSegment

from src.config import config
from src.agents.visual_agent import VisualAgent
from src.services.image_generator import ImageGenerator
from src.services.video_generator import VideoGenerator
from src.services.subtitle_generator import SubtitleGenerator
from src.ui.components.state import get_state, update_state, advance_step, go_to_step
from src.models.schemas import WorkflowStep, Scene, KenBurnsEffect, Word


def _update_scene_lyrics(scene: Scene, new_lyrics: str) -> None:
    """
    Update a scene's lyrics, creating Word objects if needed.
    Distributes timing evenly across words for the scene duration.
    """
    new_words_text = new_lyrics.split()
    if not new_words_text:
        return

    if scene.words:
        # Update existing words
        for i, word_obj in enumerate(scene.words):
            if i < len(new_words_text):
                word_obj.word = new_words_text[i]
        # If user added more words, append to last word
        if len(new_words_text) > len(scene.words):
            extra = " ".join(new_words_text[len(scene.words):])
            scene.words[-1].word += " " + extra
    else:
        # Create new Word objects with evenly distributed timing
        scene_duration = scene.end_time - scene.start_time
        word_duration = scene_duration / len(new_words_text)
        scene.words = []
        for i, word_text in enumerate(new_words_text):
            start = scene.start_time + (i * word_duration)
            end = start + word_duration - 0.05  # Small gap between words
            scene.words.append(Word(word=word_text, start=start, end=end))


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


# Resolution configurations
RESOLUTION_OPTIONS = {
    "1080p": {"width": 1920, "height": 1080, "image_size": "2K"},
    "2K": {"width": 2560, "height": 1440, "image_size": "2K"},
    "4K": {"width": 3840, "height": 2160, "image_size": "4K"},
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

        st.text_area(
            "Style override (optional)",
            placeholder="Override the visual style, e.g., 'anime style, vibrant colors'",
            key="style_override",
        )

        # New options
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

    # Generate storyboard button
    if st.button("Generate Scene Prompts", type="primary"):
        # Save new settings to state
        update_state(
            show_lyrics=st.session_state.get("show_lyrics", True),
            use_sequential_mode=st.session_state.get("use_sequential_mode", False),
        )
        generate_scene_prompts(
            state,
            st.session_state.get("scenes_per_minute", 4),
            st.session_state.get("style_override", ""),
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
    resolution = getattr(state, 'video_resolution', '1080p')
    res_info = RESOLUTION_OPTIONS.get(resolution, RESOLUTION_OPTIONS["1080p"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Scenes", total_scenes)
    with col2:
        st.metric("Resolution", resolution)
    with col3:
        st.metric("Duration", f"{state.audio_duration:.1f}s")

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
                        word_timing = " | ".join(
                            f"{w.word} ({w.start:.1f}s)" for w in scene.words[:5]
                        )
                        if len(scene.words) > 5:
                            word_timing += f" ... +{len(scene.words) - 5} more"
                        st.caption(f"Word timing: {word_timing}")

                    # Slider to pick start point within scene
                    scene_duration = scene.end_time - scene.start_time
                    preview_cols = st.columns([2, 3])
                    with preview_cols[0]:
                        preview_start_offset = st.slider(
                            "Start from",
                            min_value=0.0,
                            max_value=max(0.1, scene_duration - 0.5),
                            value=0.0,
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
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Regenerate All Prompts", type="secondary"):
            # Reset and regenerate
            update_state(
                scenes=[],
                prompts_ready=False,
            )
            st.rerun()

    with col2:
        if st.button("Save Changes", type="secondary"):
            # Save any modified prompts/effects
            update_state(scenes=modified_scenes)
            st.success("Changes saved!")
            st.rerun()

    with col3:
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

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Scenes", total_scenes)
    with col2:
        if missing_count > 0:
            st.metric("Images Ready", scenes_with_images, delta=f"-{missing_count} missing", delta_color="inverse")
        else:
            st.metric("Images Ready", scenes_with_images)
    with col3:
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

    # Action buttons
    if missing_count > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Generate Missing", type="primary"):
                regenerate_missing_images(state)
        with col2:
            if st.button("Regenerate All", type="secondary"):
                regenerate_all_images(state)
        with col3:
            if st.button("Edit Prompts", type="secondary"):
                update_state(storyboard_ready=False)
                st.rerun()
        with col4:
            if st.button("Create Video Anyway"):
                crossfade = st.session_state.get("crossfade", 0.3)
                generate_video_from_storyboard(state, crossfade, is_demo_mode)
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Regenerate All Images", type="secondary"):
                regenerate_all_images(state)
        with col2:
            if st.button("Edit Prompts", type="secondary"):
                update_state(storyboard_ready=False)
                st.rerun()
        with col3:
            if st.button("Start Over", type="secondary"):
                update_state(
                    scenes=[],
                    generated_images=[],
                    prompts_ready=False,
                    storyboard_ready=False,
                    project_dir=None,
                )
                st.rerun()
        with col4:
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

    # Scene header with status indicator
    if has_image:
        st.markdown(f"**Scene {scene.index + 1}**")
    else:
        st.markdown(f"**Scene {scene.index + 1}** :warning:")

    st.caption(f"{scene.start_time:.1f}s - {scene.end_time:.1f}s ({scene.duration:.1f}s)")

    # Image
    if has_image:
        st.image(str(scene.image_path), use_container_width=True)
    else:
        st.error("Missing image - click Regenerate")

    # Scene details in expander - editable prompt and lyrics
    with st.expander("Edit Scene", expanded=False):
        st.markdown(f"**Mood:** {scene.mood}")

        # Audio preview with adjustable start point
        if state.audio_path and state.audio_path != "demo_mode":
            st.markdown("**Audio Preview**")

            # Show word-level timing if words exist
            if scene.words:
                word_timing = " | ".join(
                    f"{w.word} ({w.start:.1f}s)" for w in scene.words[:6]
                )
                if len(scene.words) > 6:
                    word_timing += f" ... +{len(scene.words) - 6} more"
                st.caption(f"Word timing: {word_timing}")

            # Slider to pick start point within scene
            scene_duration = scene.end_time - scene.start_time
            preview_start_offset = st.slider(
                "Start from (seconds into scene)",
                min_value=0.0,
                max_value=max(0.1, scene_duration - 0.5),
                value=0.0,
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
        state.lyrics.title.replace(" ", "_").lower()[:20]
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
    if not project_dir:
        st.error("No project directory found. Please start over.")
        return

    project_dir = Path(project_dir)
    scenes = state.scenes
    resolution = getattr(state, 'video_resolution', '1080p')
    res_info = RESOLUTION_OPTIONS.get(resolution, RESOLUTION_OPTIONS["1080p"])
    sequential_mode = getattr(state, 'use_sequential_mode', False)

    with st.status("Generating scene images...", expanded=True) as status:
        mode_str = " with sequential consistency" if sequential_mode else ""
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

        def image_progress(msg: str, prog: float):
            progress_bar.progress(prog, text=msg)

        # Pass resolution and sequential mode to image generator
        image_paths = image_gen.generate_storyboard(
            scene_prompts=prompts,
            style_prefix=style_prefix,
            character_description=character_desc or "",
            output_dir=images_dir,
            progress_callback=image_progress,
            image_size=res_info["image_size"],
            sequential_mode=sequential_mode,
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

        output_path = images_dir / f"scene_{scene_index:03d}.png"

        image = image_gen.generate_scene_image(
            prompt=scene.visual_prompt,
            style_prefix=style_prefix,
            character_description=character_desc,
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
    project_dir = getattr(state, 'project_dir', None)
    if not project_dir:
        st.error("No project directory found. Please start over.")
        return

    project_dir = Path(project_dir)
    scenes = state.scenes
    resolution = getattr(state, 'video_resolution', '1080p')
    res_info = RESOLUTION_OPTIONS.get(resolution, RESOLUTION_OPTIONS["1080p"])
    show_lyrics = getattr(state, 'show_lyrics', True)

    # Verify we have images
    scenes_with_images = [s for s in scenes if s.image_path and Path(s.image_path).exists()]
    if not scenes_with_images:
        st.error("No scene images found. Please regenerate the storyboard.")
        return

    project_name = (
        state.lyrics.title.replace(" ", "_").lower()[:20]
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
        st.write(f"Assembling video at {resolution} ({res_info['width']}x{res_info['height']})...")
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
            )

        update_state(final_video_path=str(output_path))
        status.update(label="Video complete!", state="complete")

    st.success(f"Video saved to: {output_path}")
    st.rerun()
