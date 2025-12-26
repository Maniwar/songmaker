"""Movie Mode page - Create animated podcasts, educational videos, and short films."""

import logging
import streamlit as st
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from src.agents.script_agent import ScriptAgent
from src.config import config
from src.models.schemas import (
    Character,
    MovieConfig,
    MovieFormat,
    MovieModeState,
    MovieTone,
    MovieWorkflowStep,
    Script,
    VoiceSettings,
)
from src.ui.components.chat_bubbles import (
    inject_chat_styles,
    render_chat_message,
    render_typing_indicator,
)
from src.ui.components.state import (
    ensure_movie_directories,
    get_movie_project_dir,
    render_movie_project_sidebar,
    save_movie_state,
)


def get_readable_timestamp() -> str:
    """Get a human-readable timestamp for file naming (YYYYMMDD_HHMMSS)."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_image_anti_prompt(visual_style: str) -> str:
    """Get the 'avoid' keywords for image generation based on visual style.

    Note: Gemini's native image generation doesn't have a negative_prompt parameter.
    These are keywords that are implicitly avoided by using the right positive prompts.
    This is shown to the user for transparency.

    Args:
        visual_style: The visual style setting (e.g., "photorealistic", "anime")

    Returns:
        String of keywords to avoid
    """
    style_lower = visual_style.lower() if visual_style else ""

    if any(kw in style_lower for kw in ["photorealistic", "realistic", "photo", "cinematic"]):
        return "CGI, cartoon, anime, 3D render, digital art, stylized, artificial, plastic skin, video game, smooth skin, airbrushed, illustration"
    elif "anime" in style_lower or "manga" in style_lower:
        return "photorealistic, real photo, live action, realistic skin texture, film grain"
    elif "3d" in style_lower or "pixar" in style_lower:
        return "photorealistic, real photo, live action, 2D, flat, anime"
    else:
        # Default/artistic styles
        return "low quality, blurry, distorted, deformed"


def get_video_negative_prompt(visual_style: str, model: str = "wan26") -> str:
    """Get the negative prompt for video generation based on visual style and model.

    Args:
        visual_style: The visual style setting (e.g., "photorealistic", "anime")
        model: The video generation model ("wan26", "seedance15", "veo3")

    Returns:
        Negative prompt string for the model, or empty string if not applicable
    """
    # Veo doesn't use traditional negative prompts (handled via prompt text)
    if "veo" in model.lower():
        return ""

    style_lower = visual_style.lower() if visual_style else ""

    if any(kw in style_lower for kw in ["photorealistic", "realistic", "photo", "cinematic"]):
        # Include face preservation terms
        return "CGI, cartoon, animated, 3D render, digital art, stylized, artificial, plastic skin, video game, unrealistic, smooth skin, airbrushed, different person, face change, morphing face, wrong face"
    elif "anime" in style_lower or "cartoon" in style_lower:
        return "photorealistic, real photo, live action, realistic skin texture"
    else:
        # Default
        return "low quality, blurry, distorted, deformed, ugly"


def get_movie_state() -> MovieModeState:
    """Get or initialize movie mode state."""
    if "movie_state" not in st.session_state:
        st.session_state.movie_state = MovieModeState()
    return st.session_state.movie_state


def get_project_dir(create_if_missing: bool = True) -> Path:
    """Get the project directory.

    Args:
        create_if_missing: If True (default), create the directory if it doesn't exist.
                          Set to False when just querying/reading to avoid creating
                          spurious timestamped directories.

    Returns:
        Path to the project directory, or a fallback path if not set
    """
    from src.config import config

    project_dir = get_movie_project_dir()
    if project_dir is None:
        # Try to derive project name from script title before creating timestamped folder
        state = get_movie_state()
        if state.script and state.script.title:
            # Use script title as project name
            safe_name = "".join(c if c.isalnum() or c in "_- " else "_" for c in state.script.title)
            safe_name = safe_name.strip().replace(" ", "_")[:50]
            project_dir = config.output_dir / "movie" / safe_name
            if create_if_missing:
                project_dir.mkdir(parents=True, exist_ok=True)
                state.project_dir = str(project_dir)
        elif create_if_missing:
            project_dir = ensure_movie_directories()
        else:
            # Return a reasonable fallback without creating directories
            project_dir = config.output_dir / "movie" / "default"
    return project_dir


def update_movie_state(**kwargs) -> None:
    """Update movie mode state."""
    state = get_movie_state()
    for key, value in kwargs.items():
        if hasattr(state, key):
            setattr(state, key, value)


def advance_movie_step() -> None:
    """Advance to the next movie workflow step with auto-save."""
    state = get_movie_state()
    steps = list(MovieWorkflowStep)
    current_idx = steps.index(state.current_step)

    # Check if we should skip VOICES step (video modes generate audio natively)
    gen_method = state.config.generation_method if state.config else "tts_images"
    is_video_mode = gen_method in ("veo3", "wan26", "seedance15")

    if current_idx < len(steps) - 1:
        next_step = steps[current_idx + 1]

        # Skip VOICES step for video modes (they generate audio natively)
        if next_step == MovieWorkflowStep.VOICES and is_video_mode:
            if current_idx + 1 < len(steps) - 1:
                next_step = steps[current_idx + 2]

        # Skip VISUALS step - video generation is now in Scenes
        if next_step == MovieWorkflowStep.VISUALS:
            if current_idx + 2 < len(steps):
                next_step = steps[current_idx + 2]
            # For video modes that already skipped VOICES, need to skip one more
            if next_step == MovieWorkflowStep.VISUALS:
                next_step = MovieWorkflowStep.RENDER

        state.current_step = next_step
        # Auto-save after each major milestone
        try:
            save_movie_state()
        except Exception:
            pass  # Don't block on save failures


def go_to_movie_step(step: MovieWorkflowStep) -> None:
    """Go to a specific movie workflow step."""
    state = get_movie_state()
    state.current_step = step


def render_movie_mode_page() -> None:
    """Render the movie mode page."""
    state = get_movie_state()

    # Render movie project sidebar (save/load)
    render_movie_project_sidebar()

    # Header with exit button
    col1, col2 = st.columns([6, 1])
    with col1:
        st.header("🎬 Movie Mode")
    with col2:
        if st.button("Exit", type="secondary"):
            # Clear movie mode
            if "movie_state" in st.session_state:
                del st.session_state.movie_state
            st.session_state.pop("movie_mode", None)
            st.rerun()

    st.markdown(
        """
        Create animated podcasts, educational videos, and short films with:
        - **Consistent characters** across all scenes
        - **Unique voices** for each character (AI text-to-speech)
        - **Script-based** scene generation
        """
    )

    # Workflow progress
    render_movie_progress(state.current_step)

    # Render current step
    if state.current_step == MovieWorkflowStep.SETUP:
        render_setup_page()
    elif state.current_step == MovieWorkflowStep.SCRIPT:
        render_script_page()
    elif state.current_step == MovieWorkflowStep.CHARACTERS:
        render_characters_page()
    elif state.current_step == MovieWorkflowStep.SCENES:
        render_scenes_page()
    elif state.current_step == MovieWorkflowStep.VOICES:
        render_voices_page()
    elif state.current_step == MovieWorkflowStep.VISUALS:
        render_visuals_page()
    elif state.current_step == MovieWorkflowStep.RENDER:
        render_render_page()
    elif state.current_step == MovieWorkflowStep.COMPLETE:
        render_movie_complete_page()


def render_movie_progress(current_step: MovieWorkflowStep) -> None:
    """Render movie workflow progress indicator with clickable navigation."""
    state = get_movie_state()

    # Check generation method - video modes (veo3, wan26, seedance15) generate their own audio
    gen_method = state.config.generation_method if state.config else "tts_images"
    is_video_mode = gen_method in ("veo3", "wan26", "seedance15")

    # Build steps list - VISUALS is now integrated into SCENES
    all_steps = [
        ("⚙️", "Setup", MovieWorkflowStep.SETUP),
        ("📝", "Script", MovieWorkflowStep.SCRIPT),
        ("👥", "Characters", MovieWorkflowStep.CHARACTERS),
        ("🎬", "Scenes", MovieWorkflowStep.SCENES),
        ("🎙️", "Voices", MovieWorkflowStep.VOICES),
        ("🔧", "Render", MovieWorkflowStep.RENDER),
        ("✅", "Complete", MovieWorkflowStep.COMPLETE),
    ]

    # Filter out VOICES step for video modes (they generate audio natively)
    if is_video_mode:
        steps = [(i, l, s) for i, l, s in all_steps if s != MovieWorkflowStep.VOICES]
    else:
        steps = all_steps

    cols = st.columns(len(steps))
    current_idx = list(MovieWorkflowStep).index(current_step)

    # Track highest step reached for navigation
    if "max_step_reached" not in st.session_state:
        st.session_state.max_step_reached = current_idx
    elif current_idx > st.session_state.max_step_reached:
        st.session_state.max_step_reached = current_idx

    for i, (icon, label, step) in enumerate(steps):
        step_idx = list(MovieWorkflowStep).index(step)
        with cols[i]:
            # Allow navigation to any step up to the highest reached
            if step_idx <= st.session_state.max_step_reached:
                if st.button(
                    f"{icon} {label}",
                    key=f"nav_{step.value}",
                    use_container_width=True,
                    type="primary" if step == current_step else "secondary",
                ):
                    go_to_movie_step(step)
                    st.rerun()
            else:
                # Future steps - show as disabled text
                st.markdown(
                    f"<div style='text-align:center;color:#888;padding:8px;'>"
                    f"{icon} {label}</div>",
                    unsafe_allow_html=True
                )

    st.markdown("---")


def render_setup_page() -> None:
    """Render the project setup/configuration page."""
    state = get_movie_state()

    st.subheader("Project Setup")
    st.markdown(
        """
        Configure your project before diving into script development.
        These settings help the AI understand what you're creating.
        """
    )

    # Project name input (for saving)
    current_project_name = getattr(state, 'project_name', None) or ""
    project_name = st.text_input(
        "Project Name",
        value=current_project_name,
        placeholder="e.g., AI Ethics Podcast, Space Explainer...",
        help="Name your project for easy saving and identification",
        key="setup_project_name",
    )

    st.markdown("---")

    # Use existing config or create default
    current_config = state.config or MovieConfig()

    # Format selection - use session state to persist selection
    st.markdown("##### What are you creating?")
    format_options = {
        MovieFormat.PODCAST: ("🎙️ Podcast", "Discussions, interviews, explainers"),
        MovieFormat.EDUCATIONAL: ("📚 Educational", "Tutorials, lessons, how-to guides"),
        MovieFormat.SHORT_FILM: ("🎬 Short Film", "Stories, skits, animated shorts"),
        MovieFormat.EXPLAINER: ("💡 Explainer", "Product explainers, concept breakdowns"),
        MovieFormat.INTERVIEW: ("🎤 Interview", "Q&A format, interviews"),
    }

    # Initialize format selection in session state
    if "setup_selected_format" not in st.session_state:
        st.session_state.setup_selected_format = current_config.format.value
    if "setup_custom_format" not in st.session_state:
        st.session_state.setup_custom_format = ""

    # Format buttons + Custom option
    format_cols = st.columns(len(format_options) + 1)

    for i, (fmt, (label, desc)) in enumerate(format_options.items()):
        with format_cols[i]:
            is_selected = st.session_state.setup_selected_format == fmt.value
            if st.button(
                label,
                key=f"format_{fmt.value}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
                help=desc,
            ):
                st.session_state.setup_selected_format = fmt.value
                st.session_state.setup_custom_format = ""  # Clear custom when preset selected
                st.rerun()

    # Custom format option
    with format_cols[-1]:
        is_custom = st.session_state.setup_selected_format == "custom"
        if st.button(
            "✏️ Custom",
            key="format_custom",
            use_container_width=True,
            type="primary" if is_custom else "secondary",
            help="Define your own format",
        ):
            st.session_state.setup_selected_format = "custom"
            st.rerun()

    # Show custom format input if custom is selected
    if st.session_state.setup_selected_format == "custom":
        st.session_state.setup_custom_format = st.text_input(
            "Describe your format",
            value=st.session_state.setup_custom_format,
            placeholder="e.g., Documentary, Music video, Animation test, Vlog...",
            key="setup_custom_format_input",
        )

    # Determine the selected format for use in the rest of the page
    if st.session_state.setup_selected_format == "custom":
        selected_format = MovieFormat.SHORT_FILM  # Use as base for scene calculations
        custom_format_text = st.session_state.setup_custom_format
    else:
        selected_format = MovieFormat(st.session_state.setup_selected_format)
        custom_format_text = None

    st.markdown("---")

    # Duration input
    st.markdown("##### Target Duration")
    dur_col1, dur_col2 = st.columns([2, 3])

    with dur_col1:
        selected_minutes = st.number_input(
            "Minutes",
            min_value=0,
            max_value=60,
            value=current_config.target_duration // 60,
            key="setup_duration_min",
        )
        selected_seconds = st.number_input(
            "Seconds",
            min_value=0,
            max_value=59,
            value=current_config.target_duration % 60,
            step=15,
            key="setup_duration_sec",
        )
        selected_duration = int(selected_minutes * 60 + selected_seconds)

    # Calculate scene recommendations (needed for summary too)
    if selected_duration > 0:
        # Estimate scenes: ~20-40 seconds per scene depending on format
        if selected_format == MovieFormat.PODCAST:
            avg_scene_duration = 45  # Longer takes for talking heads
        elif selected_format == MovieFormat.SHORT_FILM:
            avg_scene_duration = 20  # More cuts for drama
        else:
            avg_scene_duration = 30  # Default

        min_scenes = max(2, selected_duration // 60)  # At least 1 per minute
        recommended_scenes = max(3, selected_duration // avg_scene_duration)
        max_scenes = max(min_scenes + 2, selected_duration // 15)  # Max ~4 per minute
    else:
        min_scenes = 3
        recommended_scenes = 5
        max_scenes = 10

    with dur_col2:
        # Scene recommendations based on duration
        if selected_duration > 0:
            st.markdown("**Scene Recommendations**")
            st.markdown(
                f"""
                For a **{selected_minutes}m {selected_seconds}s** {selected_format.value}:
                - Minimum: **{min_scenes}** scenes
                - Recommended: **{recommended_scenes}** scenes
                - Maximum: **{max_scenes}** scenes

                _Fewer scenes = longer takes, more dialogue per scene_
                _More scenes = faster pacing, more visual variety_
                """
            )
        else:
            st.info("Enter a duration to see scene recommendations")

    st.markdown("---")

    # Character count and naming
    st.markdown("##### Characters")
    char_col1, char_col2 = st.columns([1, 2])

    with char_col1:
        char_options = {
            1: "Solo (Narrator)",
            2: "Duo (Dialogue)",
            3: "Trio",
            4: "Ensemble (4+)",
        }
        selected_chars = st.select_slider(
            "Number of Characters",
            options=list(char_options.keys()),
            value=min(current_config.num_characters, 4),
            format_func=lambda x: char_options.get(x, str(x)),
            key="setup_chars",
        )

    # Character naming - initialize from config or use defaults
    # Use getattr for backward compatibility with old MovieConfig objects
    if "setup_character_names" not in st.session_state:
        config_names = getattr(current_config, 'character_names', [])
        if config_names:
            st.session_state.setup_character_names = list(config_names)
        else:
            st.session_state.setup_character_names = []

    # Ensure we have the right number of name slots
    default_names = ["Host", "Guest", "Expert", "Narrator", "Character 5", "Character 6"]
    while len(st.session_state.setup_character_names) < selected_chars:
        idx = len(st.session_state.setup_character_names)
        st.session_state.setup_character_names.append(default_names[idx] if idx < len(default_names) else f"Character {idx + 1}")

    with char_col2:
        st.markdown("**Character Names** _(optional)_")
        name_cols = st.columns(min(selected_chars, 4))
        character_names = []
        for i in range(selected_chars):
            col_idx = i % len(name_cols)
            with name_cols[col_idx]:
                name = st.text_input(
                    f"Character {i + 1}",
                    value=st.session_state.setup_character_names[i] if i < len(st.session_state.setup_character_names) else "",
                    key=f"setup_char_name_{i}",
                    label_visibility="collapsed",
                    placeholder=f"Character {i + 1}",
                )
                character_names.append(name)
        # Update session state
        st.session_state.setup_character_names = character_names

    st.markdown("---")

    # Tone selection - use session state to persist selection
    st.markdown("##### Tone & Style")
    tone_options = {
        MovieTone.CASUAL: ("😊 Casual", "Friendly, conversational"),
        MovieTone.PROFESSIONAL: ("💼 Professional", "Business, formal"),
        MovieTone.EDUCATIONAL: ("🎓 Educational", "Clear, instructive"),
        MovieTone.HUMOROUS: ("😄 Humorous", "Comedy, lighthearted"),
        MovieTone.DRAMATIC: ("🎭 Dramatic", "Intense, emotional"),
    }

    # Initialize tone selection in session state
    if "setup_selected_tone" not in st.session_state:
        st.session_state.setup_selected_tone = current_config.tone.value
    if "setup_custom_tone" not in st.session_state:
        st.session_state.setup_custom_tone = False
    if "setup_custom_tone_text" not in st.session_state:
        custom_tone_val = getattr(current_config, 'custom_tone', None)
        st.session_state.setup_custom_tone_text = custom_tone_val or ""
        if custom_tone_val:
            st.session_state.setup_custom_tone = True

    # Add one more column for Custom
    tone_cols = st.columns(len(tone_options) + 1)

    for i, (tone, (label, desc)) in enumerate(tone_options.items()):
        with tone_cols[i]:
            is_selected = st.session_state.setup_selected_tone == tone.value and not st.session_state.setup_custom_tone
            if st.button(
                label,
                key=f"tone_{tone.value}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
                help=desc,
            ):
                st.session_state.setup_selected_tone = tone.value
                st.session_state.setup_custom_tone = False
                st.rerun()

    # Custom tone button
    with tone_cols[-1]:
        if st.button(
            "✏️ Custom",
            key="tone_custom",
            use_container_width=True,
            type="primary" if st.session_state.setup_custom_tone else "secondary",
            help="Enter your own tone description",
        ):
            st.session_state.setup_custom_tone = True
            st.rerun()

    # Custom tone text input
    custom_tone_text = ""
    if st.session_state.setup_custom_tone:
        custom_tone_text = st.text_input(
            "Describe your tone",
            value=st.session_state.setup_custom_tone_text,
            placeholder="e.g., Mysterious and suspenseful, Warm and nostalgic...",
            key="setup_custom_tone_input",
        )
        st.session_state.setup_custom_tone_text = custom_tone_text

    selected_tone = MovieTone(st.session_state.setup_selected_tone)

    st.markdown("---")

    # Technical settings (collapsible)
    with st.expander("Technical Settings", expanded=False):
        # Generation method - show first since it affects other options
        gen_methods = {
            "tts_images": "TTS + Images (Ken Burns, Cheaper)",
            "veo3": "Veo 3.1 (Google, Premium quality)",
            "wan26": "WAN 2.6 (AtlasCloud, 15s clips, ~$0.05-0.08/s)",
            "seedance15": "Seedance 1.5 Pro (AtlasCloud, Lip-sync, ~$0.015/s)",
            "seedance_fast": "Seedance Fast (AtlasCloud, Faster, ~$0.01/s)",
        }

        # Get current method with fallback
        current_method = current_config.generation_method
        if current_method not in gen_methods:
            current_method = "tts_images"

        selected_gen = st.selectbox(
            "Generation Method",
            options=list(gen_methods.keys()),
            index=list(gen_methods.keys()).index(current_method),
            format_func=lambda x: gen_methods[x],
            key="setup_gen_method",
            help="Choose how to generate video scenes"
        )

        # Show method-specific info
        if selected_gen == "wan26":
            st.info("**WAN 2.6** by Alibaba: Up to 15s per clip, native audio, multi-shot storytelling, character consistency.")
        elif selected_gen == "seedance15":
            st.info("**Seedance 1.5 Pro** by ByteDance: Best quality, millisecond lip-sync, cinematic camera control, 3-12s duration.")
        elif selected_gen == "seedance_fast":
            st.info("**Seedance Fast** by ByteDance: Faster generation, good quality, 3-12s duration. Great for iteration.")

        # Voice provider - only show for TTS + Images mode
        if selected_gen == "tts_images":
            voice_providers = {
                "edge": "Edge TTS (Free)",
                "openai": "OpenAI TTS (Paid)",
                "elevenlabs": "ElevenLabs (Best quality, Paid)",
            }
            selected_voice = st.selectbox(
                "Voice Provider",
                options=list(voice_providers.keys()),
                index=list(voice_providers.keys()).index(current_config.voice_provider),
                format_func=lambda x: voice_providers[x],
                key="setup_voice_provider",
            )
        else:
            # Video models generate their own audio
            selected_voice = selected_gen  # veo3, wan26, or seedance15

        # Veo specific settings (only show if veo3 selected)
        if selected_gen == "veo3":
            st.markdown("**Veo Settings**")

            # Model selection - Veo 3.1 is the latest (3.0 deprecated)
            veo_models = {
                "veo-3.1-generate-preview": "Veo 3.1 Standard (Best quality, audio)",
                "veo-3.1-fast-generate-preview": "Veo 3.1 Fast (Quicker, audio)",
            }
            # Get current veo settings with backward compatibility
            cfg_veo_model = getattr(current_config, 'veo_model', 'veo-3.1-generate-preview')
            cfg_veo_duration = getattr(current_config, 'veo_duration', 8)
            cfg_veo_resolution = getattr(current_config, 'veo_resolution', '720p')

            selected_veo_model = st.selectbox(
                "Veo Model",
                options=list(veo_models.keys()),
                index=list(veo_models.keys()).index(cfg_veo_model)
                if cfg_veo_model in veo_models
                else 0,
                format_func=lambda x: veo_models[x],
                key="setup_veo_model",
            )

            # Model-specific info
            is_veo31 = "3.1" in selected_veo_model
            is_veo2 = "veo-2" in selected_veo_model
            has_audio = "veo-2" not in selected_veo_model

            veo_col1, veo_col2 = st.columns(2)

            with veo_col1:
                # Duration options depend on model
                if is_veo2:
                    veo_durations = {5: "5 sec", 6: "6 sec", 8: "8 sec"}
                    default_dur = 8
                else:
                    veo_durations = {4: "4 sec", 6: "6 sec", 8: "8 sec"}
                    default_dur = 8

                # Get current duration or use default
                current_dur = cfg_veo_duration
                if current_dur not in veo_durations:
                    current_dur = default_dur

                selected_veo_duration = st.selectbox(
                    "Clip Duration",
                    options=list(veo_durations.keys()),
                    index=list(veo_durations.keys()).index(current_dur),
                    format_func=lambda x: veo_durations[x],
                    key="setup_veo_duration",
                )

            with veo_col2:
                # Resolution options
                # Note: 1080p only supports 8s duration for Veo 3.1
                # Veo 2 only supports 720p
                if is_veo2:
                    veo_resolutions = {"720p": "720p"}
                    selected_veo_resolution = "720p"
                    st.selectbox(
                        "Resolution",
                        options=["720p"],
                        index=0,
                        key="setup_veo_resolution",
                        disabled=True,
                    )
                elif selected_veo_duration == 8:
                    veo_resolutions = {"720p": "720p (Faster)", "1080p": "1080p (Better)"}
                    selected_veo_resolution = st.selectbox(
                        "Resolution",
                        options=list(veo_resolutions.keys()),
                        index=list(veo_resolutions.keys()).index(cfg_veo_resolution)
                        if cfg_veo_resolution in veo_resolutions
                        else 0,
                        format_func=lambda x: veo_resolutions[x],
                        key="setup_veo_resolution",
                    )
                else:
                    # Non-8s durations only support 720p
                    selected_veo_resolution = "720p"
                    st.selectbox(
                        "Resolution",
                        options=["720p (1080p requires 8s duration)"],
                        index=0,
                        key="setup_veo_resolution",
                        disabled=True,
                    )

            # Feature indicators
            features = []
            if has_audio:
                features.append("🔊 Audio")
            if is_veo31:
                features.append("🖼️ Reference images")
                features.append("🎬 Video extension")
                features.append("🔄 Interpolation")
            st.caption(" | ".join(features) if features else "Silent video only")

            # Cost estimate with resolution comparison
            # Base cost per second (720p)
            if "fast" in selected_veo_model:
                base_cost = 0.40
            elif is_veo2:
                base_cost = 0.35
            else:
                base_cost = 0.75

            # 1080p costs ~1.5x more than 720p
            resolution_multiplier = 1.5 if selected_veo_resolution == "1080p" else 1.0
            cost_per_second = base_cost * resolution_multiplier

            est_clips = recommended_scenes
            est_total = est_clips * selected_veo_duration * cost_per_second

            # Show cost comparison if 1080p is available
            if selected_veo_duration == 8 and not is_veo2:
                cost_720p = est_clips * selected_veo_duration * base_cost
                cost_1080p = est_clips * selected_veo_duration * base_cost * 1.5
                if selected_veo_resolution == "1080p":
                    st.caption(f"💰 Estimated: ~${est_total:.2f} (1080p) | 720p would be ~${cost_720p:.2f} (save ${cost_1080p - cost_720p:.2f})")
                else:
                    st.caption(f"💰 Estimated: ~${est_total:.2f} (720p) | 1080p would be ~${cost_1080p:.2f} (+${cost_1080p - cost_720p:.2f})")
            else:
                st.caption(f"💰 Estimated cost: ~${est_total:.2f} ({est_clips} clips × {selected_veo_duration}s × ${cost_per_second:.2f}/sec)")

            # Initialize WAN/Seedance defaults when using Veo
            selected_wan_model = getattr(current_config, 'wan_model', 'alibaba/wan-2.6/i2v-720p')
            selected_wan_duration = getattr(current_config, 'wan_duration', 10)
            selected_wan_resolution = getattr(current_config, 'wan_resolution', '720p')
            selected_wan_audio = getattr(current_config, 'wan_enable_audio', True)
            # WAN advanced params
            selected_wan_guidance = getattr(current_config, 'wan_guidance_scale', 6.5)
            selected_wan_flow = getattr(current_config, 'wan_flow_shift', 2.5)
            selected_wan_steps = getattr(current_config, 'wan_inference_steps', 40)
            selected_wan_seed = getattr(current_config, 'wan_seed', 0)
            selected_wan_shot = getattr(current_config, 'wan_shot_type', 'single')
            selected_seedance_model = getattr(current_config, 'seedance_model', 'bytedance/seedance-v1.5-pro-i2v-720p')
            selected_seedance_duration = getattr(current_config, 'seedance_duration', 8)
            selected_seedance_resolution = getattr(current_config, 'seedance_resolution', '720p')
            selected_seedance_lip_sync = getattr(current_config, 'seedance_lip_sync', True)

        # WAN 2.6 settings (AtlasCloud)
        elif selected_gen == "wan26":
            st.markdown("**WAN 2.6 Settings**")

            cfg_wan_model = getattr(current_config, 'wan_model', 'alibaba/wan-2.6/i2v-720p')
            cfg_wan_duration = getattr(current_config, 'wan_duration', 10)
            cfg_wan_resolution = getattr(current_config, 'wan_resolution', '720p')
            cfg_wan_audio = getattr(current_config, 'wan_enable_audio', True)

            wan_models = {
                "alibaba/wan-2.6/i2v-720p": "Image-to-Video 720p (~$0.05/s)",
                "alibaba/wan-2.6/i2v-1080p": "Image-to-Video 1080p (~$0.08/s)",
                "alibaba/wan-2.6/i2v-720p-fast": "Image-to-Video Fast (~$0.03/s)",
                "alibaba/wan-2.6/t2v-720p": "Text-to-Video 720p (~$0.05/s)",
                "alibaba/wan-2.6/t2v-1080p": "Text-to-Video 1080p (~$0.08/s)",
                "alibaba/wan-2.6/v2v-720p-fast": "Video-to-Video (~$0.05/s)",
            }
            selected_wan_model = st.selectbox(
                "WAN Model",
                options=list(wan_models.keys()),
                index=list(wan_models.keys()).index(cfg_wan_model) if cfg_wan_model in wan_models else 0,
                format_func=lambda x: wan_models[x],
                key="setup_wan_model",
            )

            wan_col1, wan_col2 = st.columns(2)
            with wan_col1:
                wan_durations = {5: "5 sec", 10: "10 sec", 15: "15 sec (max)"}
                selected_wan_duration = st.selectbox(
                    "Clip Duration",
                    options=list(wan_durations.keys()),
                    index=list(wan_durations.keys()).index(cfg_wan_duration) if cfg_wan_duration in wan_durations else 1,
                    format_func=lambda x: wan_durations[x],
                    key="setup_wan_duration",
                )
            with wan_col2:
                selected_wan_audio = st.checkbox(
                    "Enable Native Audio",
                    value=cfg_wan_audio,
                    key="setup_wan_audio",
                    help="Generate audio with the video (lip-sync, ambient sounds)"
                )

            # Extract resolution from model name
            selected_wan_resolution = "1080p" if "1080p" in selected_wan_model else "720p"

            # Cost estimate
            cost_per_sec = 0.08 if "1080p" in selected_wan_model else 0.05 if "fast" not in selected_wan_model else 0.03
            est_total = recommended_scenes * selected_wan_duration * cost_per_sec
            st.caption(f"💰 Estimated: ~${est_total:.2f} ({recommended_scenes} clips × {selected_wan_duration}s × ${cost_per_sec:.2f}/sec)")

            # Advanced WAN parameters
            with st.expander("⚙️ Advanced WAN Parameters", expanded=False):
                cfg_guidance = getattr(current_config, 'wan_guidance_scale', 6.5)
                cfg_flow = getattr(current_config, 'wan_flow_shift', 2.5)
                cfg_steps = getattr(current_config, 'wan_inference_steps', 40)
                cfg_seed = getattr(current_config, 'wan_seed', 0)
                cfg_shot = getattr(current_config, 'wan_shot_type', 'single')

                adv_col1, adv_col2 = st.columns(2)
                with adv_col1:
                    selected_wan_guidance = st.slider(
                        "Guidance Scale",
                        min_value=1.0, max_value=10.0, value=float(cfg_guidance), step=0.5,
                        key="setup_wan_guidance",
                        help="5-7 = realistic, 8+ = 'AI look'. Higher = follows prompt more strictly"
                    )
                    selected_wan_steps = st.slider(
                        "Inference Steps",
                        min_value=10, max_value=40, value=int(cfg_steps), step=5,
                        key="setup_wan_steps",
                        help="More steps = higher quality but slower (default 30, max 40)"
                    )
                with adv_col2:
                    selected_wan_flow = st.slider(
                        "Flow Shift",
                        min_value=1.0, max_value=10.0, value=float(cfg_flow), step=0.5,
                        key="setup_wan_flow",
                        help="2-3 = best identity preservation, 4-5 = balanced, 6+ = more motion"
                    )
                    selected_wan_seed = st.number_input(
                        "Seed",
                        min_value=-1, max_value=999999, value=int(cfg_seed),
                        key="setup_wan_seed",
                        help="0 = reproducible, -1 = random"
                    )
                shot_options = {"single": "Single Camera (consistent)", "multi": "Multi Camera (dynamic)"}
                selected_wan_shot = st.selectbox(
                    "Shot Type",
                    options=list(shot_options.keys()),
                    index=0 if cfg_shot == "single" else 1,
                    format_func=lambda x: shot_options[x],
                    key="setup_wan_shot",
                    help="Single = better character consistency, Multi = AI picks camera angles"
                )

            # Initialize Veo/Seedance defaults when using WAN
            selected_veo_model = getattr(current_config, 'veo_model', 'veo-3.1-generate-preview')
            selected_veo_duration = getattr(current_config, 'veo_duration', 8)
            selected_veo_resolution = getattr(current_config, 'veo_resolution', '720p')
            selected_seedance_model = getattr(current_config, 'seedance_model', 'bytedance/seedance-v1.5-pro-i2v-720p')
            selected_seedance_duration = getattr(current_config, 'seedance_duration', 8)
            selected_seedance_resolution = getattr(current_config, 'seedance_resolution', '720p')
            selected_seedance_lip_sync = getattr(current_config, 'seedance_lip_sync', True)

        # Seedance 1.5 Pro settings (AtlasCloud)
        elif selected_gen == "seedance15":
            st.markdown("**Seedance 1.5 Pro Settings**")

            cfg_sd_model = getattr(current_config, 'seedance_model', 'bytedance/seedance-v1.5-pro-i2v-720p')
            cfg_sd_duration = getattr(current_config, 'seedance_duration', 8)
            cfg_sd_lip_sync = getattr(current_config, 'seedance_lip_sync', True)

            seedance_models = {
                "bytedance/seedance-v1.5-pro-i2v-480p": "Image-to-Video 480p (~$0.02/s)",
                "bytedance/seedance-v1.5-pro-i2v-720p": "Image-to-Video 720p (~$0.04/s)",
                "bytedance/seedance-v1.5-pro-i2v-1080p": "Image-to-Video 1080p (~$0.06/s)",
                "bytedance/seedance-v1.5-pro-t2v-480p": "Text-to-Video 480p (~$0.02/s)",
                "bytedance/seedance-v1.5-pro-t2v-720p": "Text-to-Video 720p (~$0.04/s)",
                "bytedance/seedance-v1.5-pro-t2v-1080p": "Text-to-Video 1080p (~$0.06/s)",
            }
            selected_seedance_model = st.selectbox(
                "Seedance Model",
                options=list(seedance_models.keys()),
                index=list(seedance_models.keys()).index(cfg_sd_model) if cfg_sd_model in seedance_models else 1,
                format_func=lambda x: seedance_models[x],
                key="setup_seedance_model",
            )

            sd_col1, sd_col2 = st.columns(2)
            with sd_col1:
                # Seedance supports 3-15 seconds
                selected_seedance_duration = st.slider(
                    "Clip Duration (seconds)",
                    min_value=3,
                    max_value=15,
                    value=cfg_sd_duration,
                    key="setup_seedance_duration",
                    help="Seedance supports flexible 3-15 second clips"
                )
            with sd_col2:
                selected_seedance_lip_sync = st.checkbox(
                    "Enable Lip-Sync",
                    value=cfg_sd_lip_sync,
                    key="setup_seedance_lip_sync",
                    help="2-step process: generates video, then applies lip-sync with bytedance/lipsync model"
                )

            # Extract resolution from model name
            if "1080p" in selected_seedance_model:
                selected_seedance_resolution = "1080p"
                cost_per_sec = 0.06
            elif "480p" in selected_seedance_model:
                selected_seedance_resolution = "480p"
                cost_per_sec = 0.02
            else:
                selected_seedance_resolution = "720p"
                cost_per_sec = 0.04

            # Cost estimate
            est_total = recommended_scenes * selected_seedance_duration * cost_per_sec
            st.caption(f"💰 Estimated: ~${est_total:.2f} ({recommended_scenes} clips × {selected_seedance_duration}s × ${cost_per_sec:.2f}/sec)")

            # Initialize Veo/WAN defaults when using Seedance
            selected_veo_model = getattr(current_config, 'veo_model', 'veo-3.1-generate-preview')
            selected_veo_duration = getattr(current_config, 'veo_duration', 8)
            selected_veo_resolution = getattr(current_config, 'veo_resolution', '720p')
            selected_wan_model = getattr(current_config, 'wan_model', 'alibaba/wan-2.6/i2v-720p')
            selected_wan_duration = getattr(current_config, 'wan_duration', 10)
            selected_wan_resolution = getattr(current_config, 'wan_resolution', '720p')
            selected_wan_audio = getattr(current_config, 'wan_enable_audio', True)
            # WAN advanced params
            selected_wan_guidance = getattr(current_config, 'wan_guidance_scale', 6.5)
            selected_wan_flow = getattr(current_config, 'wan_flow_shift', 2.5)
            selected_wan_steps = getattr(current_config, 'wan_inference_steps', 40)
            selected_wan_seed = getattr(current_config, 'wan_seed', 0)
            selected_wan_shot = getattr(current_config, 'wan_shot_type', 'single')

        else:
            # Default values when not using video gen models (TTS + Images mode)
            selected_veo_model = getattr(current_config, 'veo_model', 'veo-3.1-generate-preview')
            selected_veo_duration = getattr(current_config, 'veo_duration', 8)
            selected_veo_resolution = getattr(current_config, 'veo_resolution', '720p')
            # Initialize WAN/Seedance defaults for config
            selected_wan_model = getattr(current_config, 'wan_model', 'alibaba/wan-2.6/i2v-720p')
            selected_wan_duration = getattr(current_config, 'wan_duration', 10)
            selected_wan_resolution = getattr(current_config, 'wan_resolution', '720p')
            selected_wan_audio = getattr(current_config, 'wan_enable_audio', True)
            # WAN advanced params
            selected_wan_guidance = getattr(current_config, 'wan_guidance_scale', 6.5)
            selected_wan_flow = getattr(current_config, 'wan_flow_shift', 2.5)
            selected_wan_steps = getattr(current_config, 'wan_inference_steps', 40)
            selected_wan_seed = getattr(current_config, 'wan_seed', 0)
            selected_wan_shot = getattr(current_config, 'wan_shot_type', 'single')
            selected_seedance_model = getattr(current_config, 'seedance_model', 'bytedance/seedance-v1.5-pro-i2v-720p')
            selected_seedance_duration = getattr(current_config, 'seedance_duration', 8)
            selected_seedance_resolution = getattr(current_config, 'seedance_resolution', '720p')
            selected_seedance_lip_sync = getattr(current_config, 'seedance_lip_sync', True)

        # Visual style with custom option - photorealistic first for realistic output
        visual_styles = [
            "photorealistic, cinematic lighting, 8K quality",
            "cinematic film, natural lighting",
            "animated digital art",
            "3D animated, Pixar style",
            "anime style",
            "watercolor illustration",
            "comic book style",
            "✏️ Custom...",
        ]

        # Check if current style is custom (not in presets)
        preset_styles = visual_styles[:-1]  # All except "Custom..."
        is_custom_style = current_config.visual_style not in preset_styles

        style_col1, style_col2 = st.columns([2, 3])

        with style_col1:
            selected_style_option = st.selectbox(
                "Visual Style",
                options=visual_styles,
                index=len(visual_styles) - 1 if is_custom_style else visual_styles.index(current_config.visual_style),
                key="setup_visual_style",
            )

        with style_col2:
            if selected_style_option == "✏️ Custom...":
                custom_style_value = current_config.visual_style if is_custom_style else ""
                selected_style = st.text_input(
                    "Custom style description",
                    value=custom_style_value,
                    placeholder="e.g., Oil painting, cyberpunk neon, hand-drawn sketch...",
                    key="setup_custom_visual_style",
                )
                if not selected_style:
                    selected_style = "animated digital art"  # Fallback
            else:
                selected_style = selected_style_option
                st.markdown("")  # Empty space for alignment

        # Show WAN 2.6 negative prompt preview based on style
        style_lower = (selected_style or "").lower()
        if "photorealistic" in style_lower or "realistic" in style_lower or "photo" in style_lower:
            neg_prompt_preview = "CGI, cartoon, animated, 3D render, digital art, stylized..."
            neg_prompt_style = "photorealistic"
        elif "anime" in style_lower or "cartoon" in style_lower:
            neg_prompt_preview = "photorealistic, real photo, live action..."
            neg_prompt_style = "anime/cartoon"
        elif "3d" in style_lower or "pixar" in style_lower or "animated" in style_lower:
            neg_prompt_preview = "photorealistic, real photo, live action, 2D, flat"
            neg_prompt_style = "3D animated"
        else:
            neg_prompt_preview = "(no negative prompt - add 'photorealistic', 'anime', or '3D' to style)"
            neg_prompt_style = "unrecognized"

        st.caption(f"🚫 **WAN 2.6 Negative Prompt** ({neg_prompt_style}): _{neg_prompt_preview}_")

        # Overlap setting
        allow_overlap = st.checkbox(
            "Allow overlapping dialogue (characters can interrupt each other)",
            value=current_config.allow_overlap,
            key="setup_allow_overlap",
        )

    # Summary
    st.markdown("---")
    st.markdown("##### Summary")

    # Format duration string
    if selected_seconds > 0:
        duration_str = f"{selected_minutes}m {selected_seconds}s"
    else:
        duration_str = f"{selected_minutes} min"

    # Format display - handle custom format
    if custom_format_text:
        format_display = f"✏️ {custom_format_text}" if custom_format_text else "✏️ Custom"
    else:
        format_display = format_options[selected_format][0]

    # Character names display
    char_names_display = ", ".join(n for n in character_names if n) if any(character_names) else char_options[selected_chars]

    # Tone display - handle custom tone
    if st.session_state.setup_custom_tone and custom_tone_text:
        tone_display = f"✏️ {custom_tone_text}"
    else:
        tone_display = tone_options[selected_tone][0]

    summary_text = f"""
    - **Format:** {format_display}
    - **Duration:** {duration_str} (~{recommended_scenes} scenes recommended)
    - **Characters:** {char_names_display}
    - **Tone:** {tone_display}
    - **Visual Style:** {selected_style}
    """
    st.markdown(summary_text)

    # Continue button
    st.markdown("---")

    # Check if project name is provided
    if not project_name:
        st.warning("Please enter a **Project Name** above to save your work.")

    if st.button(
        "Continue to Script Workshop →",
        type="primary",
        use_container_width=True,
        disabled=not project_name,  # Disable if no project name
    ):
        # Save project name to state
        state.project_name = project_name

        # Save config to state with all model settings
        new_config = MovieConfig(
            format=selected_format,
            custom_format=custom_format_text,
            target_duration=selected_duration,
            num_characters=selected_chars,
            character_names=[n for n in character_names if n],  # Filter empty names
            tone=selected_tone,
            custom_tone=custom_tone_text if st.session_state.setup_custom_tone and custom_tone_text else None,
            voice_provider=selected_voice,
            generation_method=selected_gen,
            visual_style=selected_style,
            # Veo settings
            veo_model=selected_veo_model,
            veo_duration=selected_veo_duration,
            veo_resolution=selected_veo_resolution,
            # WAN 2.6 settings
            wan_model=selected_wan_model,
            wan_duration=selected_wan_duration,
            wan_resolution=selected_wan_resolution,
            wan_enable_audio=selected_wan_audio,
            # WAN advanced parameters
            wan_guidance_scale=selected_wan_guidance,
            wan_flow_shift=selected_wan_flow,
            wan_inference_steps=selected_wan_steps,
            wan_seed=selected_wan_seed,
            wan_shot_type=selected_wan_shot,
            # Seedance 1.5 Pro settings
            seedance_model=selected_seedance_model,
            seedance_duration=selected_seedance_duration,
            seedance_resolution=selected_seedance_resolution,
            seedance_lip_sync=selected_seedance_lip_sync,
            # Other settings
            allow_overlap=allow_overlap,
            recommended_scenes=recommended_scenes,
        )
        state.config = new_config

        # Update script target_duration if script exists
        if state.script:
            state.script.target_duration = float(selected_duration)

        advance_movie_step()
        st.rerun()


def render_script_page() -> None:
    """Render the script development page."""
    state = get_movie_state()

    # Inject iPhone-style chat CSS
    inject_chat_styles()

    st.subheader("Script Workshop")
    st.markdown(
        """
        Describe your video idea and I'll help you develop a complete script with
        characters, scenes, and dialogue. This works great for:
        - **Podcasts** - Educational discussions, interviews, explainers
        - **Educational videos** - Tutorials, lessons, how-to guides
        - **Short films** - Stories, skits, animated shorts
        """
    )

    # Initialize script agent
    if "script_agent" not in st.session_state:
        st.session_state.script_agent = ScriptAgent()

    agent = st.session_state.script_agent

    # Pass config to agent if available
    if state.config:
        agent.set_project_config(state.config)

    # Show config summary if available
    if state.config:
        cfg = state.config
        format_labels = {
            MovieFormat.PODCAST: "Podcast",
            MovieFormat.EDUCATIONAL: "Educational",
            MovieFormat.SHORT_FILM: "Short Film",
            MovieFormat.EXPLAINER: "Explainer",
            MovieFormat.INTERVIEW: "Interview",
        }
        minutes = cfg.target_duration // 60
        seconds = cfg.target_duration % 60
        duration_str = f"{minutes}m {seconds}s" if seconds else f"{minutes} min"

        # Handle custom format (with backward compatibility)
        custom_format = getattr(cfg, 'custom_format', None)
        format_display = custom_format if custom_format else format_labels.get(cfg.format, cfg.format.value)

        # Handle character names (with backward compatibility)
        character_names = getattr(cfg, 'character_names', [])
        if character_names:
            char_display = ", ".join(character_names)
        else:
            char_display = str(cfg.num_characters)

        # Get recommended scenes (with backward compatibility)
        recommended_scenes = getattr(cfg, 'recommended_scenes', 5)

        st.info(
            f"**Project:** {format_display} | "
            f"**Duration:** {duration_str} | "
            f"**Scenes:** ~{recommended_scenes} | "
            f"**Characters:** {char_display} | "
            f"**Tone:** {cfg.tone.value.title()}"
        )

    # Show getting started help if no messages yet
    if not state.script_messages:
        st.markdown(
            """
            **How to get started:**

            Describe your video idea below. For example:
            - "A discussion about how black holes work"
            - "A debate about the future of AI between a skeptic and an optimist"
            - "A comedy about a cat who thinks he's a dog"

            The AI will create a script that matches your setup settings.
            """
        )

    # Display conversation history with iPhone-style bubbles
    for msg in state.script_messages:
        render_chat_message(
            content=msg["content"],
            role=msg["role"],
            model=msg.get("model"),
            show_model_indicator=True,
        )

    # Chat input
    if user_input := st.chat_input("Describe your video idea..."):
        # Add user message to state
        state.script_messages.append({"role": "user", "content": user_input})

        # Show user message immediately
        render_chat_message(content=user_input, role="user")

        # Show typing indicator while getting response
        typing_placeholder = st.empty()
        with typing_placeholder:
            render_typing_indicator()

        # Get AI response
        response = agent.chat(user_input)

        # Clear typing indicator
        typing_placeholder.empty()

        # Get the model that was used
        model_used = getattr(agent, '_last_model_used', None)

        # Add assistant response to state (including model info)
        state.script_messages.append({
            "role": "assistant",
            "content": response,
            "model": model_used,
        })

        st.rerun()

    # Sidebar with readiness status
    with st.sidebar:
        st.subheader("Script Status")

        status = agent.get_readiness_status()

        if status["ready"]:
            st.success("✅ Script ready!")
            st.markdown("The script includes:")
            st.markdown("- Title and description")
            st.markdown("- Character definitions")
            st.markdown("- Scene breakdown with dialogue")

            if st.button("📝 Finalize Script", type="primary"):
                with st.spinner("Extracting script..."):
                    script = agent.extract_script()
                    if script:
                        update_movie_state(script=script)
                        st.success(f"Script '{script.title}' finalized!")
                        st.write(f"**Characters:** {len(script.characters)}")
                        st.write(f"**Scenes:** {len(script.scenes)}")
                        advance_movie_step()
                        st.rerun()
                    else:
                        st.error("Could not extract script. Please continue refining.")

        elif status["has_content"]:
            st.warning("⏳ Still developing...")
            if status["missing"]:
                st.markdown("**Still needed:**")
                for section in status["missing"]:
                    st.markdown(f"- {section}")
            st.markdown(f"*{status['message_count']} message(s) exchanged*")

            if st.button("Finalize Anyway"):
                with st.spinner("Extracting script..."):
                    script = agent.extract_script()
                    if script:
                        update_movie_state(script=script)
                        st.success(f"Script '{script.title}' extracted!")
                        advance_movie_step()
                        st.rerun()
                    else:
                        st.error("Could not extract script. Please continue the conversation.")
        else:
            st.info("💬 Start by describing your video idea")

        st.markdown("---")

        if st.button("Start Over"):
            agent.reset()
            update_movie_state(script_messages=[], script=None)
            st.rerun()


def render_characters_page() -> None:
    """Render the character editing page."""
    state = get_movie_state()

    if not state.script:
        st.warning("Please create a script first.")
        if st.button("Go to Script"):
            go_to_movie_step(MovieWorkflowStep.SCRIPT)
            st.rerun()
        return

    st.subheader(f"Characters in '{state.script.title}'")
    st.markdown(
        """
        Review and edit your characters. The visual descriptions should be detailed
        enough for consistent AI image generation across all scenes.
        """
    )

    # Display and edit each character
    for i, char in enumerate(state.script.characters):
        with st.expander(f"🎭 {char.name} ({char.id})", expanded=i == 0):
            col1, col2 = st.columns([2, 1])

            with col1:
                new_name = st.text_input(
                    "Name",
                    value=char.name,
                    key=f"char_name_{i}",
                )

                new_description = st.text_area(
                    "Visual Description (for AI image generation)",
                    value=char.description,
                    height=100,
                    key=f"char_desc_{i}",
                    help="Be specific about age, appearance, clothing, etc.",
                )

                new_personality = st.text_input(
                    "Personality",
                    value=char.personality or "",
                    key=f"char_pers_{i}",
                )

            with col2:
                st.markdown("**Voice Settings**")
                new_voice_name = st.text_input(
                    "Voice Type",
                    value=char.voice.voice_name or "neutral",
                    key=f"char_voice_{i}",
                    help="e.g., 'female, 30s, British accent, warm'",
                )

                # Veo Voice Description (for Veo mode continuity)
                is_veo = state.config and state.config.generation_method == "veo3"
                if is_veo:
                    current_veo_voice = getattr(char, 'veo_voice_description', None) or ""
                    new_veo_voice = st.text_input(
                        "Veo Voice Description",
                        value=current_veo_voice,
                        key=f"veo_voice_{i}",
                        placeholder="e.g., 'deep male voice, warm tone, slight British accent'",
                        help="Voice description for Veo 3.1 - keeps voice consistent across scenes",
                    )
                else:
                    new_veo_voice = None

            # Character Portrait Section - full width below character details
            st.markdown("---")
            st.markdown("**Character Portrait**")

            # Get visual style from config (set in Setup) or script
            visual_style = (
                (state.config.visual_style if state.config else None)
                or (state.script.visual_style if state.script else None)
                or "photorealistic, cinematic lighting"
            )

            # Create two-column layout for portrait section
            portrait_left, portrait_right = st.columns([1, 2])

            with portrait_left:
                # Portrait settings
                with st.expander("⚙️ Settings", expanded=True):
                    # Model selection - Gemini supports both generation and transformation
                    portrait_models = {
                        "gemini-2.5-flash-image": "Nano Banana (Fast, 1024px)",
                        "gemini-3-pro-image-preview": "Nano Banana Pro (Best, up to 4K)",
                        "imagen-4.0-generate-001": "Imagen 4.0 Standard (text-to-image only)",
                        "imagen-4.0-ultra-generate-001": "Imagen 4.0 Ultra (text-to-image only)",
                    }
                    portrait_model = st.selectbox(
                        "Model",
                        options=list(portrait_models.keys()),
                        index=0,
                        format_func=lambda x: portrait_models[x],
                        key=f"portrait_model_{i}",
                        help="Nano Banana models support both generation and photo transformation"
                    )

                    port_col1, port_col2 = st.columns(2)
                    with port_col1:
                        portrait_size = st.selectbox(
                            "Size",
                            options=["2K", "4K"],
                            index=1,  # Default to 4K
                            key=f"portrait_size_{i}",
                            help="4K produces higher quality portraits"
                        )
                    with port_col2:
                        portrait_aspect = st.selectbox(
                            "Aspect Ratio",
                            options=["1:1", "3:4", "4:3", "16:9"],
                            index=0,  # Default to 1:1 square
                            key=f"portrait_aspect_{i}",
                            help="1:1 for square portraits, 3:4 for vertical"
                        )

                # Transform section - outside expander for visibility
                st.markdown("**📷 Transform from Photo**")
                st.caption("Upload your photo to transform it into this character's style")
                if "imagen" in portrait_model.lower():
                    st.info("Note: Photo transformation requires Nano Banana models. Will use Nano Banana automatically.")
                source_photo = st.file_uploader(
                    "Upload photo to transform",
                    type=["png", "jpg", "jpeg"],
                    key=f"source_photo_{i}",
                )
                if source_photo:
                    st.image(source_photo, width=100, caption="Source photo")
                    if st.button("🔄 Transform into Character", key=f"transform_{i}"):
                        from src.services.movie_image_generator import MovieImageGenerator
                        from PIL import Image
                        from io import BytesIO
                        import shutil

                        # First, save current portrait as variation (if exists)
                        current_portrait = char.reference_image_path
                        if current_portrait and Path(current_portrait).exists():
                            var_dir = get_project_dir() / "characters" / "variations"
                            var_dir.mkdir(parents=True, exist_ok=True)
                            existing_nums = []
                            for vp in var_dir.glob(f"character_{char.id}_var_*.png"):
                                try:
                                    num = int(vp.stem.split("_")[-1])
                                    existing_nums.append(num)
                                except ValueError:
                                    pass
                            next_idx = max(existing_nums) + 1 if existing_nums else 0
                            backup_path = var_dir / f"character_{char.id}_var_{next_idx}.png"
                            shutil.copy(current_portrait, backup_path)

                        # Save source photo to disk for future variations
                        source_dir = get_project_dir() / "characters" / "sources"
                        source_dir.mkdir(parents=True, exist_ok=True)
                        source_path = source_dir / f"character_{char.id}_source.png"

                        # Load and save source image
                        source_img = Image.open(BytesIO(source_photo.getvalue()))
                        source_img.save(source_path, format="PNG")
                        char.source_image_path = str(source_path)

                        generator = MovieImageGenerator(style=visual_style)
                        output_dir = get_project_dir() / "characters"
                        with st.spinner(f"Transforming photo into {char.name}..."):
                            result = generator.generate_character_reference(
                                character=char,
                                output_dir=output_dir,
                                style=visual_style,
                                image_size=portrait_size,
                                aspect_ratio=portrait_aspect,
                                model=portrait_model,
                                source_image=source_img,
                            )
                            if result:
                                char.reference_image_path = str(result)

                                # Also save the new portrait to variations for browsing
                                var_dir = get_project_dir() / "characters" / "variations"
                                var_dir.mkdir(parents=True, exist_ok=True)
                                existing_nums = []
                                for vp in var_dir.glob(f"character_{char.id}_var_*.png"):
                                    try:
                                        num = int(vp.stem.split("_")[-1])
                                        existing_nums.append(num)
                                    except ValueError:
                                        pass
                                next_idx = max(existing_nums) + 1 if existing_nums else 0
                                var_path = var_dir / f"character_{char.id}_var_{next_idx}.png"
                                shutil.copy(result, var_path)

                                # Auto-save project with source path
                                try:
                                    save_movie_state()
                                except Exception:
                                    pass
                                st.success("Photo transformed and saved to variations!")
                                st.rerun()
                            else:
                                st.error("Failed to transform photo")

            with portrait_right:
                # Check if portrait already exists
                portrait_path = char.reference_image_path
                if portrait_path and Path(portrait_path).exists():
                    # Get actual image dimensions
                    from PIL import Image as PILImage
                    try:
                        with PILImage.open(portrait_path) as img:
                            width, height = img.size
                        resolution_text = f"{width}x{height}"
                    except Exception:
                        resolution_text = "unknown"

                    st.image(str(portrait_path), width="stretch")
                    st.caption(f"AI-generated portrait ({visual_style}) - {resolution_text}")

                    # Full-size view option
                    with st.expander("🔍 View Full Size"):
                        st.image(str(portrait_path), width="stretch")
                        st.markdown(f"**Resolution:** {resolution_text}")
                        st.markdown(f"**File:** `{portrait_path}`")

                    # Check for existing variations on disk
                    var_dir = get_project_dir() / "characters" / "variations"
                    var_dir.mkdir(parents=True, exist_ok=True)
                    existing_variations = list(var_dir.glob(f"character_{char.id}_var_*.png"))

                    # Auto-add current portrait if no variations exist (legacy support)
                    if not existing_variations:
                        import shutil
                        var_path = var_dir / f"character_{char.id}_var_0.png"
                        shutil.copy(portrait_path, var_path)
                        existing_variations.append(var_path)

                    # Action buttons for existing portrait
                    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
                    with btn_col1:
                        # Save project button
                        if st.button("💾 Save", key=f"save_project_{i}"):
                            try:
                                save_path = save_movie_state()
                                st.success(f"Saved to {save_path.name}")
                            except Exception as e:
                                st.error(f"Save failed: {e}")
                    with btn_col2:
                        # Download image button
                        with open(portrait_path, "rb") as f:
                            st.download_button(
                                "⬇️ Export",
                                data=f.read(),
                                file_name=f"{char.name.lower().replace(' ', '_')}_portrait.png",
                                mime="image/png",
                                key=f"download_portrait_{i}",
                            )
                    with btn_col3:
                        if st.button("🔄 Regenerate", key=f"regen_portrait_{i}"):
                            # First, save current portrait as a variation
                            import shutil
                            var_dir = get_project_dir() / "characters" / "variations"
                            var_dir.mkdir(parents=True, exist_ok=True)

                            # Find next available variation index
                            existing_nums = []
                            for vp in var_dir.glob(f"character_{char.id}_var_*.png"):
                                try:
                                    num = int(vp.stem.split("_")[-1])
                                    existing_nums.append(num)
                                except ValueError:
                                    pass
                            next_idx = max(existing_nums) + 1 if existing_nums else 0
                            backup_path = var_dir / f"character_{char.id}_var_{next_idx}.png"
                            shutil.copy(portrait_path, backup_path)

                            # Check for source photo
                            source_img = None
                            source_path = getattr(char, 'source_image_path', None)
                            if source_path and Path(source_path).exists():
                                from PIL import Image as PILImage
                                source_img = PILImage.open(source_path)

                            # Regenerate portrait
                            from src.services.movie_image_generator import MovieImageGenerator
                            generator = MovieImageGenerator(style=visual_style)
                            output_dir = get_project_dir() / "characters"
                            with st.spinner(f"Generating portrait for {char.name}..."):
                                result = generator.generate_character_reference(
                                    character=char,
                                    output_dir=output_dir,
                                    style=visual_style,
                                    image_size=portrait_size,
                                    aspect_ratio=portrait_aspect,
                                    model=portrait_model,
                                    source_image=source_img,  # Use source photo if available
                                )
                                if result:
                                    char.reference_image_path = str(result)

                                    # Also save the new portrait to variations for browsing
                                    next_idx_new = max(existing_nums) + 2 if existing_nums else 1  # +2 because we already saved old as +1
                                    var_path_new = var_dir / f"character_{char.id}_var_{next_idx_new}.png"
                                    shutil.copy(result, var_path_new)

                                    # Auto-save project
                                    try:
                                        save_movie_state()
                                    except Exception:
                                        pass
                                    st.success("Portrait regenerated and saved to variations!")
                                    st.rerun()
                                else:
                                    st.error("Failed to generate portrait")
                    with btn_col4:
                        if st.button("🎲 New Variations", key=f"gen_variations_{i}"):
                            st.session_state[f"generating_variations_{i}"] = True
                            st.rerun()

                    # Show existing variations with fast carousel (no page reload for navigation)
                    if existing_variations:
                        with st.expander(f"📁 Browse Saved Variations ({len(existing_variations)})", expanded=False):
                            sorted_variations = sorted(existing_variations)

                            # Use slider for instant navigation (no rerun needed)
                            if len(sorted_variations) > 1:
                                current_idx = st.slider(
                                    "Browse variations",
                                    min_value=1,
                                    max_value=len(sorted_variations),
                                    value=1,
                                    key=f"var_slider_{i}",
                                    format="%d of " + str(len(sorted_variations)),
                                ) - 1
                            else:
                                current_idx = 0
                                st.caption("1 variation saved")

                            # Display current variation
                            current_var_path = sorted_variations[current_idx]
                            st.image(str(current_var_path), width="stretch")

                            # Action buttons
                            act_col1, act_col2 = st.columns(2)
                            with act_col1:
                                if st.button("✓ Use This Portrait", key=f"use_carousel_{i}", type="primary"):
                                    import shutil
                                    main_path = get_project_dir() / "characters" / f"character_{char.id}_reference.png"
                                    shutil.copy(current_var_path, main_path)
                                    char.reference_image_path = str(main_path)
                                    try:
                                        save_movie_state()
                                    except Exception:
                                        pass
                                    st.success("Portrait updated!")
                                    st.rerun()
                            with act_col2:
                                if st.button("🗑️ Delete This", key=f"delete_var_{i}"):
                                    import os
                                    os.remove(current_var_path)
                                    st.success("Deleted!")
                                    st.rerun()

                    # Generate new variations - with source selection
                    if st.session_state.get(f"generating_variations_{i}", False):
                        st.markdown("---")
                        st.markdown("**Generate Variations From:**")

                        # Build source options
                        source_options = {"description": "Character description (no photo)"}
                        source_path = getattr(char, 'source_image_path', None)
                        if source_path and Path(source_path).exists():
                            source_options["original_photo"] = "Original uploaded photo"
                        if portrait_path and Path(portrait_path).exists():
                            source_options["current_portrait"] = "Current portrait"
                        if existing_variations:
                            source_options["select_variation"] = "Select from variations..."

                        selected_source = st.radio(
                            "Source for new variations:",
                            options=list(source_options.keys()),
                            format_func=lambda x: source_options[x],
                            key=f"var_source_{i}",
                            horizontal=True,
                        )

                        # If selecting from variations, show mini carousel
                        variation_source_path = None
                        if selected_source == "select_variation" and existing_variations:
                            var_select_idx = st.selectbox(
                                "Choose variation:",
                                options=range(len(existing_variations)),
                                format_func=lambda x: f"Variation {x + 1}",
                                key=f"var_select_{i}",
                            )
                            variation_source_path = str(sorted(existing_variations)[var_select_idx])
                            st.image(variation_source_path, width=150)

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🎨 Generate 3 Variations", key=f"do_generate_var_{i}", type="primary"):
                                # Determine source image
                                source_img = None
                                from PIL import Image as PILImage

                                if selected_source == "original_photo" and source_path:
                                    source_img = PILImage.open(source_path)
                                    st.info("Using original uploaded photo...")
                                elif selected_source == "current_portrait" and portrait_path:
                                    source_img = PILImage.open(portrait_path)
                                    st.info("Using current portrait...")
                                elif selected_source == "select_variation" and variation_source_path:
                                    source_img = PILImage.open(variation_source_path)
                                    st.info("Using selected variation...")
                                else:
                                    st.info("Generating from description...")

                                st.session_state[f"do_gen_variations_{i}"] = True
                                st.session_state[f"var_source_img_{i}"] = source_img
                                st.rerun()
                        with col2:
                            if st.button("Cancel", key=f"cancel_gen_var_{i}"):
                                st.session_state[f"generating_variations_{i}"] = False
                                st.rerun()

                    # Actually generate variations
                    if st.session_state.get(f"do_gen_variations_{i}", False):
                        source_img = st.session_state.get(f"var_source_img_{i}")
                        st.session_state[f"do_gen_variations_{i}"] = False
                        st.session_state[f"generating_variations_{i}"] = False

                        from src.services.movie_image_generator import MovieImageGenerator
                        generator = MovieImageGenerator(style=visual_style)
                        output_dir = get_project_dir() / "characters" / "variations"
                        output_dir.mkdir(parents=True, exist_ok=True)

                        # Find next available variation index
                        existing_nums = []
                        for vp in output_dir.glob(f"character_{char.id}_var_*.png"):
                            try:
                                num = int(vp.stem.split("_")[-1])
                                existing_nums.append(num)
                            except ValueError:
                                pass
                        start_idx = max(existing_nums) + 1 if existing_nums else 0

                        new_variations = []
                        progress_bar = st.progress(0)
                        for v_offset in range(3):
                            v_idx = start_idx + v_offset
                            progress_bar.progress((v_offset + 1) / 3)
                            result = generator.generate_character_reference(
                                character=char,
                                output_dir=output_dir,
                                style=visual_style,
                                image_size=portrait_size,
                                aspect_ratio=portrait_aspect,
                                model=portrait_model,
                                source_image=source_img,  # Use source photo if available
                            )
                            if result:
                                # Rename to unique variation name
                                var_path = output_dir / f"character_{char.id}_var_{v_idx}.png"
                                result.rename(var_path)
                                new_variations.append(var_path)
                        progress_bar.empty()

                        # Clear the generating flag
                        st.session_state[f"generating_variations_{i}"] = False

                        if new_variations:
                            st.success(f"Generated {len(new_variations)} new variations!")
                            # Store the new variations for selection
                            st.session_state[f"new_variations_{i}"] = [str(p) for p in new_variations]
                            st.rerun()

                    # Show newly generated variations for selection (carousel)
                    new_var_paths = st.session_state.get(f"new_variations_{i}", [])
                    if new_var_paths:
                        st.markdown("---")
                        st.markdown("**🆕 New Variations - Choose one:**")

                        # Carousel for new variations
                        if f"new_var_idx_{i}" not in st.session_state:
                            st.session_state[f"new_var_idx_{i}"] = 0
                        new_idx = st.session_state[f"new_var_idx_{i}"]
                        new_idx = min(max(0, new_idx), len(new_var_paths) - 1)

                        # Navigation
                        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
                        with nav_col1:
                            if st.button("◀", key=f"prev_new_{i}", disabled=new_idx == 0):
                                st.session_state[f"new_var_idx_{i}"] = new_idx - 1
                                st.rerun()
                        with nav_col2:
                            st.markdown(f"<center><b>Variation {new_idx + 1} / {len(new_var_paths)}</b></center>", unsafe_allow_html=True)
                        with nav_col3:
                            if st.button("▶", key=f"next_new_{i}", disabled=new_idx >= len(new_var_paths) - 1):
                                st.session_state[f"new_var_idx_{i}"] = new_idx + 1
                                st.rerun()

                        # Display current new variation
                        current_new_path = new_var_paths[new_idx]
                        st.image(current_new_path, width="stretch")

                        # Action buttons
                        act_col1, act_col2 = st.columns(2)
                        with act_col1:
                            if st.button("✓ Use This One", key=f"select_new_{i}", type="primary"):
                                import shutil
                                main_path = get_project_dir() / "characters" / f"character_{char.id}_reference.png"
                                shutil.copy(current_new_path, main_path)
                                char.reference_image_path = str(main_path)
                                st.session_state[f"new_variations_{i}"] = []
                                st.session_state[f"new_var_idx_{i}"] = 0
                                try:
                                    save_movie_state()
                                except Exception:
                                    pass
                                st.success("Portrait selected!")
                                st.rerun()
                        with act_col2:
                            if st.button("Keep Current Portrait", key=f"cancel_var_{i}"):
                                st.session_state[f"new_variations_{i}"] = []
                                st.session_state[f"new_var_idx_{i}"] = 0
                                st.rerun()
                else:
                    # Generate portrait buttons
                    gen_col1, gen_col2 = st.columns(2)
                    with gen_col1:
                        if st.button("🎨 Generate Portrait", key=f"gen_portrait_{i}", type="secondary"):
                            from src.services.movie_image_generator import MovieImageGenerator
                            import shutil
                            generator = MovieImageGenerator(style=visual_style)
                            output_dir = get_project_dir() / "characters"
                            with st.spinner(f"Generating portrait for {char.name}..."):
                                result = generator.generate_character_reference(
                                    character=char,
                                    output_dir=output_dir,
                                    style=visual_style,
                                    image_size=portrait_size,
                                    aspect_ratio=portrait_aspect,
                                    model=portrait_model,
                                )
                                if result:
                                    char.reference_image_path = str(result)

                                    # Also save to variations for browsing
                                    var_dir = get_project_dir() / "characters" / "variations"
                                    var_dir.mkdir(parents=True, exist_ok=True)
                                    existing_nums = []
                                    for vp in var_dir.glob(f"character_{char.id}_var_*.png"):
                                        try:
                                            num = int(vp.stem.split("_")[-1])
                                            existing_nums.append(num)
                                        except ValueError:
                                            pass
                                    next_idx = max(existing_nums) + 1 if existing_nums else 0
                                    var_path = var_dir / f"character_{char.id}_var_{next_idx}.png"
                                    shutil.copy(result, var_path)

                                    # Auto-save project
                                    try:
                                        save_movie_state()
                                    except Exception:
                                        pass
                                    st.success("Portrait generated and saved to variations!")
                                    st.rerun()
                                else:
                                    st.error("Failed to generate portrait")
                    with gen_col2:
                        if st.button("🎲 Generate 3 Variations", key=f"gen_3_portraits_{i}", type="secondary"):
                            st.session_state[f"show_variations_{i}"] = True
                            st.rerun()

                    # Also allow direct upload
                    uploaded_ref = st.file_uploader(
                        "Or upload reference directly",
                        type=["png", "jpg", "jpeg"],
                        key=f"char_ref_{i}",
                    )
                    if uploaded_ref:
                        # Save uploaded reference
                        output_dir = get_project_dir() / "characters"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        ref_path = output_dir / f"character_{char.id}_uploaded.png"
                        with open(ref_path, "wb") as f:
                            f.write(uploaded_ref.getvalue())
                        char.reference_image_path = str(ref_path)  # Store as string for JSON
                        # Auto-save project
                        try:
                            save_movie_state()
                        except Exception:
                            pass
                        st.image(uploaded_ref, width=200)
                        st.caption("Uploaded reference")

            # Update character in script
            if new_name != char.name or new_description != char.description:
                char.name = new_name
                char.description = new_description
                char.personality = new_personality if new_personality else None
                char.voice.voice_name = new_voice_name
            # Update Veo voice description if in Veo mode
            if new_veo_voice is not None and new_veo_voice != getattr(char, 'veo_voice_description', ''):
                char.veo_voice_description = new_veo_voice if new_veo_voice else None

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Script"):
            go_to_movie_step(MovieWorkflowStep.SCRIPT)
            st.rerun()
    with col2:
        if st.button("Continue to Scenes →", type="primary"):
            advance_movie_step()
            st.rerun()


# Video generation functions (must be defined before render_scenes_page)
def _generate_single_video_inline(state, scene, generation_method: str, config, use_v2v: bool = False) -> None:
    """Generate video for a single scene inline.

    Args:
        use_v2v: If True, use previous scene's video for continuity (V2V mode)
    """
    scene_model = getattr(scene, 'generation_model', None) or generation_method

    import time as time_module

    # Map model variants to display names
    model_names = {
        "veo3": "Veo 3.1 (T2V)",
        "wan26": "WAN 2.6 I2V",
        "wan26_fast": "WAN 2.5 Fast",
        "wan26_t2v": "WAN 2.6 T2V",
        "seedance15": "Seedance I2V",
        "seedance15_t2v": "Seedance T2V",
    }

    output_dir = get_project_dir() / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use timestamp suffix to create variants instead of overwriting
    timestamp = get_readable_timestamp()
    output_path = output_dir / f"scene_{scene.index:03d}_{timestamp}.mp4"

    # Get base model for duration calculation
    base_model = scene_model.replace("_fast", "").replace("_t2v", "")
    duration = scene.get_clip_duration(base_model)
    resolution = getattr(scene, 'resolution', None) or (
        config.veo_resolution if scene_model == "veo3" and config else
        config.wan_resolution if scene_model.startswith("wan26") and config else
        config.seedance_resolution if scene_model.startswith("seedance15") and config else "720p"
    )
    prompt = getattr(scene, 'video_prompt', None) or f"{scene.direction.setting}. {scene.direction.camera}."

    # Get video consistency option - use parameter or global setting
    use_prev_video = use_v2v or st.session_state.get("use_prev_scene_video", False)

    # Find previous scene's video for continuity (Veo3 and WAN 2.6 V2V)
    previous_video = None
    if use_prev_video and scene.index > 1:
        prev_scene = next((s for s in state.script.scenes if s.index == scene.index - 1), None)
        if prev_scene and prev_scene.video_path and Path(prev_scene.video_path).exists():
            previous_video = Path(prev_scene.video_path)
            logger.info(f"Using previous scene video as reference: {previous_video.name}")

    # Collect character reference images for consistency
    reference_images = []
    if scene.direction.visible_characters:
        for char_id in scene.direction.visible_characters:
            char = state.script.get_character(char_id)
            if char and char.reference_image_path and Path(char.reference_image_path).exists():
                reference_images.append(Path(char.reference_image_path))
        if reference_images:
            logger.info(f"Using {len(reference_images)} character portraits as reference for video")

    # Use scene image as first frame for better consistency
    first_frame = None
    if scene.image_path and Path(scene.image_path).exists():
        first_frame = Path(scene.image_path)
        logger.info(f"Using scene image as first frame: {first_frame.name}")

    with st.status(f"Generating with {model_names.get(scene_model, scene_model)}...", expanded=True) as status:
        try:
            result = None
            if scene_model == "veo3":
                from src.services.veo3_generator import Veo3Generator
                generator = Veo3Generator(
                    model=config.veo_model if config else "veo-3.1-generate-preview",
                    resolution=resolution, duration=duration
                )
                result = generator.generate_scene(
                    scene=scene, script=state.script, output_path=output_path,
                    style=config.visual_style if config else state.script.visual_style,
                    custom_prompt=getattr(scene, 'video_prompt', None),
                    reference_images=reference_images[:3] if reference_images else None,  # Max 3 for Veo 3.1
                    first_frame=first_frame,
                    previous_video=previous_video,
                    use_video_continuity=use_prev_video and previous_video is not None
                )
            else:
                from src.services.atlascloud_animator import AtlasCloudAnimator, WanModel, SeedanceModel

                # Select appropriate model based on variant
                is_t2v = "_t2v" in scene_model  # Text-to-video mode
                has_image = scene.image_path and Path(scene.image_path).exists()

                # Check if we need an image but don't have one
                if not is_t2v and not has_image:
                    status.update(label="Error: No image for I2V mode", state="error")
                    st.error("Image-to-Video requires a scene image. Generate one first or use T2V mode.")
                    return

                # Get visual style for conditional negative prompts
                visual_style = config.visual_style if config else state.script.visual_style

                if scene_model == "seedance15":
                    animator = AtlasCloudAnimator(model=SeedanceModel.IMAGE_TO_VIDEO)
                    result = animator.animate_scene(image_path=Path(scene.image_path), prompt=prompt,
                                                    output_path=output_path, duration_seconds=duration, resolution=resolution,
                                                    visual_style=visual_style)
                elif scene_model in ("seedance_fast", "seedance_fast_i2v"):
                    animator = AtlasCloudAnimator(model=SeedanceModel.IMAGE_TO_VIDEO_FAST)
                    result = animator.animate_scene(image_path=Path(scene.image_path), prompt=prompt,
                                                    output_path=output_path, duration_seconds=duration, resolution=resolution,
                                                    visual_style=visual_style)
                elif scene_model == "seedance15_t2v":
                    animator = AtlasCloudAnimator(model=SeedanceModel.TEXT_TO_VIDEO)
                    # T2V: Pass None for image_path, animator will detect and use text-to-video
                    result = animator.animate_scene(image_path=None, prompt=prompt,
                                                    output_path=output_path, duration_seconds=duration, resolution=resolution,
                                                    visual_style=visual_style)
                elif scene_model == "wan26_t2v":
                    animator = AtlasCloudAnimator(model=WanModel.TEXT_TO_VIDEO)
                    # T2V: Pass None for image_path
                    result = animator.animate_scene(
                        image_path=None, prompt=prompt,
                        output_path=output_path, duration_seconds=duration, resolution=resolution,
                        visual_style=visual_style,
                        guidance_scale=config.wan_guidance_scale if config else None,
                        flow_shift=config.wan_flow_shift if config else None,
                        inference_steps=config.wan_inference_steps if config else None,
                        shot_type=config.wan_shot_type if config else None,
                        seed=config.wan_seed if config else 0,
                    )
                elif scene_model == "wan26_fast":
                    animator = AtlasCloudAnimator(model=WanModel.WAN_25_FAST)
                    result = animator.animate_scene(
                        image_path=Path(scene.image_path), prompt=prompt,
                        output_path=output_path, duration_seconds=duration, resolution=resolution,
                        visual_style=visual_style,
                        guidance_scale=config.wan_guidance_scale if config else None,
                        flow_shift=config.wan_flow_shift if config else None,
                        inference_steps=config.wan_inference_steps if config else None,
                        shot_type=config.wan_shot_type if config else None,
                        seed=config.wan_seed if config else 0,
                    )
                else:  # wan26 (default image-to-video, with optional V2V)
                    # If V2V enabled and we have previous video, use V2V model
                    if previous_video and use_prev_video:
                        animator = AtlasCloudAnimator(model=WanModel.VIDEO_TO_VIDEO)
                        # Note: WAN 2.6 V2V requires video URLs, so we pass source_video
                        # The animator will fall back to I2V if URLs aren't available
                        result = animator.animate_scene(
                            image_path=Path(scene.image_path), prompt=prompt,
                            output_path=output_path, duration_seconds=duration,
                            resolution=resolution, source_video=previous_video,
                            visual_style=visual_style,
                            guidance_scale=config.wan_guidance_scale if config else None,
                            flow_shift=config.wan_flow_shift if config else None,
                            inference_steps=config.wan_inference_steps if config else None,
                            shot_type=config.wan_shot_type if config else None,
                            seed=config.wan_seed if config else 0,
                        )
                    else:
                        animator = AtlasCloudAnimator(model=WanModel.IMAGE_TO_VIDEO)
                        result = animator.animate_scene(
                            image_path=Path(scene.image_path), prompt=prompt,
                            output_path=output_path, duration_seconds=duration, resolution=resolution,
                            visual_style=visual_style,
                            guidance_scale=config.wan_guidance_scale if config else None,
                            flow_shift=config.wan_flow_shift if config else None,
                            inference_steps=config.wan_inference_steps if config else None,
                            shot_type=config.wan_shot_type if config else None,
                            seed=config.wan_seed if config else 0,
                        )

            if result:
                scene.video_path = result
                save_movie_state()
                status.update(label="Done!", state="complete")
                st.rerun()
            else:
                status.update(label="Failed", state="error")
        except Exception as e:
            status.update(label=f"Error: {e}", state="error")


def _generate_all_videos_inline(state, scenes, generation_method: str, config) -> None:
    """Generate videos for multiple scenes."""
    import time as time_module

    model_names = {
        "veo3": "Veo 3.1 (T2V)",
        "wan26": "WAN 2.6 I2V",
        "wan26_fast": "WAN 2.5 Fast",
        "wan26_t2v": "WAN 2.6 T2V",
        "seedance15": "Seedance I2V",
        "seedance15_t2v": "Seedance T2V",
    }
    output_dir = get_project_dir() / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get video consistency option
    use_prev_video = st.session_state.get("use_prev_scene_video", False)

    progress = st.progress(0, text="Starting...")
    total = len(scenes)
    generated = 0
    skipped = 0
    veo_gen = None
    animators = {}  # Cache animators by model type
    last_generated_video = None  # Track last generated video for continuity

    for i, scene in enumerate(scenes):
        scene_model = getattr(scene, 'generation_model', None) or generation_method
        base_model = scene_model.replace("_fast", "").replace("_t2v", "")

        consistency_info = " (with video ref)" if use_prev_video and last_generated_video and scene_model == "veo3" else ""
        progress.progress((i / total), text=f"Scene {scene.index} ({model_names.get(scene_model, scene_model)}{consistency_info})...")

        duration = scene.get_clip_duration(base_model)
        resolution = getattr(scene, 'resolution', None) or (
            config.veo_resolution if scene_model == "veo3" and config else
            config.wan_resolution if scene_model.startswith("wan26") and config else
            config.seedance_resolution if scene_model.startswith("seedance15") and config else "720p"
        )
        prompt = getattr(scene, 'video_prompt', None) or f"{scene.direction.setting}. {scene.direction.camera}."

        # Use timestamp suffix to create variants instead of overwriting
        timestamp = get_readable_timestamp()
        output_path = output_dir / f"scene_{scene.index:03d}_{timestamp}.mp4"

        # Check requirements
        is_t2v = "_t2v" in scene_model
        has_image = scene.image_path and Path(scene.image_path).exists()

        if not is_t2v and not has_image and scene_model != "veo3":
            st.warning(f"Scene {scene.index}: Skipped (no image for I2V mode)")
            skipped += 1
            continue

        # Get previous video for continuity (either from last generated or previous scene)
        previous_video = None
        if use_prev_video and scene_model == "veo3":
            if last_generated_video and last_generated_video.exists():
                previous_video = last_generated_video
                logger.info(f"Using last generated video for continuity: {previous_video.name}")
            elif scene.index > 1:
                prev_scene = next((s for s in state.script.scenes if s.index == scene.index - 1), None)
                if prev_scene and prev_scene.video_path and Path(prev_scene.video_path).exists():
                    previous_video = Path(prev_scene.video_path)
                    logger.info(f"Using previous scene video for continuity: {previous_video.name}")

        try:
            result = None
            if scene_model == "veo3":
                from src.services.veo3_generator import Veo3Generator
                if veo_gen is None:
                    veo_gen = Veo3Generator(model=config.veo_model if config else "veo-3.1-generate-preview",
                                            resolution=resolution, duration=duration)
                result = veo_gen.generate_scene(
                    scene=scene, script=state.script, output_path=output_path,
                    style=config.visual_style if config else state.script.visual_style,
                    custom_prompt=getattr(scene, 'video_prompt', None),
                    previous_video=previous_video,
                    use_video_continuity=use_prev_video and previous_video is not None
                )
            else:
                from src.services.atlascloud_animator import AtlasCloudAnimator, WanModel, SeedanceModel

                # Get or create animator for this model type
                if scene_model not in animators:
                    if scene_model == "seedance15":
                        animators[scene_model] = AtlasCloudAnimator(model=SeedanceModel.IMAGE_TO_VIDEO)
                    elif scene_model in ("seedance_fast", "seedance_fast_i2v"):
                        animators[scene_model] = AtlasCloudAnimator(model=SeedanceModel.IMAGE_TO_VIDEO_FAST)
                    elif scene_model == "seedance15_t2v":
                        animators[scene_model] = AtlasCloudAnimator(model=SeedanceModel.TEXT_TO_VIDEO)
                    elif scene_model == "wan26_t2v":
                        animators[scene_model] = AtlasCloudAnimator(model=WanModel.TEXT_TO_VIDEO)
                    elif scene_model == "wan26_fast":
                        animators[scene_model] = AtlasCloudAnimator(model=WanModel.WAN_25_FAST)
                    else:  # wan26
                        animators[scene_model] = AtlasCloudAnimator(model=WanModel.IMAGE_TO_VIDEO)

                animator = animators[scene_model]
                image_path = Path(scene.image_path) if has_image and not is_t2v else None
                visual_style = config.visual_style if config else state.script.visual_style
                # Pass WAN config params (ignored by non-WAN models)
                result = animator.animate_scene(
                    image_path=image_path, prompt=prompt,
                    output_path=output_path, duration_seconds=duration, resolution=resolution,
                    visual_style=visual_style,
                    guidance_scale=config.wan_guidance_scale if config else None,
                    flow_shift=config.wan_flow_shift if config else None,
                    inference_steps=config.wan_inference_steps if config else None,
                    shot_type=config.wan_shot_type if config else None,
                    seed=config.wan_seed if config else 0,
                )

            if result:
                scene.video_path = result
                generated += 1
                # Track this video for continuity in next scene
                if isinstance(result, Path):
                    last_generated_video = result
                else:
                    last_generated_video = Path(result)
        except Exception as e:
            st.warning(f"Scene {scene.index} failed: {e}")

    progress.progress(1.0, text="Complete!")
    if skipped:
        st.success(f"Generated {generated}/{total} videos! ({skipped} skipped)")
    else:
        st.success(f"Generated {generated}/{total} videos!")
    save_movie_state()
    st.rerun()


def render_scenes_page() -> None:
    """Render the streamlined scene and storyboard editor."""
    from src.models.schemas import Emotion, DialogueLine, SceneDirection, MovieScene
    import shutil

    state = get_movie_state()

    if not state.script:
        st.warning("Please create a script first.")
        return

    script = state.script
    visual_style = (
        (state.config.visual_style if state.config else None)
        or (script.visual_style if script else None)
        or "photorealistic, cinematic lighting"
    )

    # Track unsaved changes for batch saving
    if "scenes_dirty" not in st.session_state:
        st.session_state.scenes_dirty = False

    st.subheader("🎬 Scene Editor")

    # Compact action bar with save indicator
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    with col1:
        if st.button("➕ Add Scene", type="secondary", use_container_width=True):
            new_idx = len(script.scenes) + 1
            script.scenes.append(MovieScene(
                index=new_idx,
                title=f"Scene {new_idx}",
                direction=SceneDirection(setting="New location...", camera="medium shot", mood="neutral", visible_characters=[]),
                dialogue=[]
            ))
            st.session_state.scenes_dirty = True
            st.rerun()
    with col2:
        if st.button("🤖 AI Assistant", use_container_width=True):
            st.session_state["show_scene_assistant"] = not st.session_state.get("show_scene_assistant", False)
            st.rerun()
    with col3:
        save_label = "💾 Save*" if st.session_state.scenes_dirty else "💾 Saved"
        save_type = "primary" if st.session_state.scenes_dirty else "secondary"
        if st.button(save_label, type=save_type, use_container_width=True):
            save_movie_state()
            st.session_state.scenes_dirty = False
            st.toast("Saved!")
    with col4:
        # Quick stats
        total_words = sum(len(d.text.split()) for s in script.scenes for d in s.dialogue)
        est_duration = total_words / 2.5
        st.caption(f"~{int(est_duration // 60)}:{int(est_duration % 60):02d}")

    # AI Scene Assistant Panel
    if st.session_state.get("show_scene_assistant", False):
        with st.container():
            st.markdown("---")
            col_title, col_model = st.columns([3, 1])
            with col_title:
                st.markdown("### 🤖 AI Scene Assistant")
            with col_model:
                # Model selector for AI assistant
                ai_model_options = {
                    "claude-haiku-4-5": "Haiku 4.5 (fast)",
                    "claude-sonnet-4-5": "Sonnet 4.5 (smart)",
                }
                selected_ai_model = st.selectbox(
                    "Model",
                    options=list(ai_model_options.keys()),
                    format_func=lambda x: ai_model_options[x],
                    index=0,  # Default to Haiku
                    key="scene_ai_model",
                    label_visibility="collapsed"
                )
            st.caption("Ask the AI to help edit scenes, rewrite dialogue, add new scenes, or change directions.")

            # Inject iPhone-style chat bubbles
            inject_chat_styles()

            # Initialize chat history for scenes
            if "scene_assistant_messages" not in st.session_state:
                st.session_state.scene_assistant_messages = []

            # Display chat history with styled bubbles
            for msg_idx, msg in enumerate(st.session_state.scene_assistant_messages[-6:]):
                # Show which model was used for this message
                msg_model = msg.get("model", selected_ai_model) if msg["role"] == "assistant" else None
                model_display = "haiku" if msg_model and "haiku" in msg_model else "sonnet" if msg_model else None
                render_chat_message(
                    content=msg["content"],
                    role=msg["role"],
                    model=model_display,
                )

            # Apply Changes button (if there's an assistant response)
            if st.session_state.scene_assistant_messages and st.session_state.scene_assistant_messages[-1]["role"] == "assistant":
                st.markdown("---")
                apply_col1, apply_col2 = st.columns([1, 1])
                with apply_col1:
                    if st.button("✅ Apply Changes to Script", type="primary", key="apply_ai_changes"):
                        last_response = st.session_state.scene_assistant_messages[-1]["content"]

                        with st.spinner("Parsing and applying changes..."):
                            try:
                                from anthropic import Anthropic
                                from src.config import config
                                import json
                                import re

                                client = Anthropic(api_key=config.anthropic_api_key)

                                # Build character mapping
                                char_map = {c.name.lower(): c.id for c in script.characters}
                                char_map.update({c.id.lower(): c.id for c in script.characters})

                                # Ask Claude to extract structured changes (use selected model)
                                ai_model = st.session_state.get("scene_ai_model", "claude-haiku-4-5")
                                ai_max_tokens = 8192 if "haiku" in ai_model else 16384
                                parse_response = client.messages.create(
                                    model=ai_model,
                                    max_tokens=ai_max_tokens,
                                    system=f"""Extract ALL changes from this screenplay suggestion and format as JSON.
Characters available: {json.dumps({c.name: c.id for c in script.characters})}
Scenes available: {json.dumps([{"index": s.index, "title": s.title} for s in script.scenes])}

Return ONLY valid JSON in this format:
{{
  "scenes_to_delete": [1, 3],  // scene indices to remove
  "scene_updates": [
    {{
      "scene_index": 1,
      "title": "New title",
      "setting": "Updated setting description",
      "mood": "tense",
      "lighting": "dramatic",
      "camera": "close-up",
      "visual_prompt": "Full visual description for image generation",
      "video_prompt": "Animation/movement description for video generation",
      "dialogue": [
        {{"character_id": "char_id", "text": "Complete dialogue", "emotion": "neutral", "action": "optional"}}
      ]
    }}
  ],
  "new_scenes": [
    {{
      "title": "Scene Title",
      "setting": "description",
      "mood": "neutral",
      "lighting": "natural",
      "camera": "medium shot",
      "visual_prompt": "Visual description",
      "video_prompt": "Movement description",
      "dialogue": [{{"character_id": "id", "text": "line"}}]
    }}
  ]
}}

IMPORTANT:
- For scene_updates, include the COMPLETE updated dialogue list (not just new lines)
- If combining/merging scenes, put merged content in one scene_update and add other scene(s) to scenes_to_delete
- Always include visual_prompt and video_prompt when updating scenes
- Only include sections that have actual changes
- Return empty object {{}} if no clear changes to apply.""",
                                    messages=[{"role": "user", "content": f"Extract changes from:\n\n{last_response}"}]
                                )

                                # Parse JSON response
                                json_text = parse_response.content[0].text
                                # Try to extract JSON from response
                                json_match = re.search(r'\{[\s\S]*\}', json_text)
                                if json_match:
                                    changes = json.loads(json_match.group())
                                else:
                                    changes = {}

                                changes_made = []
                                from src.models.schemas import MovieScene, SceneDirection, DialogueLine, Emotion

                                # Helper to resolve character ID
                                def resolve_char_id(cid):
                                    if not cid:
                                        return script.characters[0].id if script.characters else "narrator"
                                    if cid.lower() in char_map:
                                        return char_map[cid.lower()]
                                    return cid

                                # 1. Apply comprehensive scene updates FIRST
                                for su in changes.get("scene_updates", []):
                                    scene_idx = su.get("scene_index", 1) - 1
                                    if 0 <= scene_idx < len(script.scenes):
                                        scene = script.scenes[scene_idx]
                                        updated_fields = []

                                        # Update title
                                        if "title" in su:
                                            scene.title = su["title"]
                                            updated_fields.append("title")

                                        # Update direction fields
                                        if "setting" in su:
                                            scene.direction.setting = su["setting"]
                                            updated_fields.append("setting")
                                        if "mood" in su:
                                            scene.direction.mood = su["mood"]
                                            updated_fields.append("mood")
                                        if "lighting" in su:
                                            scene.direction.lighting = su["lighting"]
                                            updated_fields.append("lighting")
                                        if "camera" in su:
                                            scene.direction.camera = su["camera"]
                                            updated_fields.append("camera")

                                        # Update prompts
                                        if "visual_prompt" in su:
                                            try:
                                                scene.visual_prompt = su["visual_prompt"]
                                            except Exception:
                                                object.__setattr__(scene, 'visual_prompt', su["visual_prompt"])
                                            updated_fields.append("visual_prompt")
                                        if "video_prompt" in su:
                                            try:
                                                scene.video_prompt = su["video_prompt"]
                                            except Exception:
                                                object.__setattr__(scene, 'video_prompt', su["video_prompt"])
                                            updated_fields.append("video_prompt")

                                        # Replace dialogue if provided
                                        if "dialogue" in su and su["dialogue"]:
                                            new_dialogue = []
                                            for dl in su["dialogue"]:
                                                char_id = resolve_char_id(dl.get("character_id"))
                                                emotion_str = dl.get("emotion", "neutral").upper()
                                                try:
                                                    emotion = Emotion[emotion_str]
                                                except KeyError:
                                                    emotion = Emotion.NEUTRAL
                                                new_dialogue.append(DialogueLine(
                                                    character_id=char_id,
                                                    text=dl.get("text", ""),
                                                    emotion=emotion,
                                                    action=dl.get("action")
                                                ))
                                            scene.dialogue = new_dialogue
                                            updated_fields.append(f"dialogue ({len(new_dialogue)} lines)")

                                        if updated_fields:
                                            changes_made.append(f"Updated Scene {scene_idx + 1}: {', '.join(updated_fields)}")

                                # 2. Delete scenes (process in reverse order to maintain indices)
                                scenes_to_delete = sorted(changes.get("scenes_to_delete", []), reverse=True)
                                for scene_idx in scenes_to_delete:
                                    # Adjust for 1-based indexing in the JSON
                                    idx = scene_idx - 1 if scene_idx > 0 else scene_idx
                                    if 0 <= idx < len(script.scenes):
                                        deleted_title = script.scenes[idx].title or f"Scene {scene_idx}"
                                        del script.scenes[idx]
                                        changes_made.append(f"Deleted Scene {scene_idx}: {deleted_title}")

                                # 3. Reindex remaining scenes
                                if scenes_to_delete:
                                    for i, scene in enumerate(script.scenes):
                                        scene.index = i + 1

                                # 4. Add new scenes
                                for ns in changes.get("new_scenes", []):
                                    new_scene_idx = len(script.scenes) + 1
                                    new_scene = MovieScene(
                                        index=new_scene_idx,
                                        title=ns.get("title", f"Scene {new_scene_idx}"),
                                        direction=SceneDirection(
                                            setting=ns.get("setting", ""),
                                            mood=ns.get("mood", "neutral"),
                                            lighting=ns.get("lighting", "natural"),
                                            camera=ns.get("camera", "medium shot"),
                                            visible_characters=[]
                                        ),
                                        dialogue=[]
                                    )
                                    # Set prompts
                                    if ns.get("visual_prompt"):
                                        try:
                                            new_scene.visual_prompt = ns["visual_prompt"]
                                        except Exception:
                                            object.__setattr__(new_scene, 'visual_prompt', ns["visual_prompt"])
                                    if ns.get("video_prompt"):
                                        try:
                                            new_scene.video_prompt = ns["video_prompt"]
                                        except Exception:
                                            object.__setattr__(new_scene, 'video_prompt', ns["video_prompt"])

                                    # Add dialogue if provided
                                    for dl in ns.get("dialogue", []):
                                        char_id = resolve_char_id(dl.get("character_id"))
                                        emotion_str = dl.get("emotion", "neutral").upper()
                                        try:
                                            emotion = Emotion[emotion_str]
                                        except KeyError:
                                            emotion = Emotion.NEUTRAL
                                        new_scene.dialogue.append(DialogueLine(
                                            character_id=char_id,
                                            text=dl.get("text", ""),
                                            emotion=emotion,
                                            action=dl.get("action")
                                        ))
                                    script.scenes.append(new_scene)
                                    changes_made.append(f"Added new scene: {ns.get('title', 'Untitled')}")

                                # Legacy support for old format
                                for dc in changes.get("dialogue_changes", []):
                                    scene_idx = dc.get("scene_index", 1) - 1
                                    if 0 <= scene_idx < len(script.scenes):
                                        scene = script.scenes[scene_idx]
                                        char_id = resolve_char_id(dc.get("character_id"))
                                        emotion_str = dc.get("emotion", "neutral").upper()
                                        try:
                                            emotion = Emotion[emotion_str]
                                        except KeyError:
                                            emotion = Emotion.NEUTRAL
                                        new_line = DialogueLine(
                                            character_id=char_id,
                                            text=dc.get("text", ""),
                                            emotion=emotion,
                                            action=dc.get("action")
                                        )
                                        scene.dialogue.append(new_line)
                                        changes_made.append(f"Added dialogue to Scene {scene_idx + 1}")

                                for sc in changes.get("scene_changes", []):
                                    scene_idx = sc.get("scene_index", 1) - 1
                                    if 0 <= scene_idx < len(script.scenes):
                                        scene = script.scenes[scene_idx]
                                        field = sc.get("field", "")
                                        value = sc.get("value", "")
                                        if field == "setting":
                                            scene.direction.setting = value
                                        elif field == "mood":
                                            scene.direction.mood = value
                                        elif field == "lighting":
                                            scene.direction.lighting = value
                                        elif field == "camera":
                                            scene.direction.camera = value
                                        changes_made.append(f"Updated Scene {scene_idx + 1} {field}")

                                if changes_made:
                                    save_movie_state()
                                    st.success(f"Applied {len(changes_made)} changes:\n" + "\n".join(f"• {c}" for c in changes_made))
                                    st.rerun()
                                else:
                                    st.info("No specific changes found to apply. Try asking for more concrete suggestions.")

                            except Exception as e:
                                st.error(f"Failed to apply changes: {e}")

                with apply_col2:
                    if st.button("🗑️ Clear Chat", key="clear_scene_chat"):
                        st.session_state.scene_assistant_messages = []
                        st.rerun()

            # Chat input
            user_input = st.chat_input("e.g., 'Add more tension to scene 2' or 'Rewrite the dialogue to be funnier'")
            if user_input:
                st.session_state.scene_assistant_messages.append({"role": "user", "content": user_input})

                # Build context about current scenes
                scene_context = "Current scenes:\n"
                for s in script.scenes:
                    scene_context += f"\nScene {s.index}: {s.title or 'Untitled'}\n"
                    scene_context += f"  Setting: {s.direction.setting}\n"
                    scene_context += f"  Mood: {s.direction.mood}, Camera: {s.direction.camera}\n"
                    if s.dialogue:
                        scene_context += f"  Dialogue ({len(s.dialogue)} lines):\n"
                        for d in s.dialogue[:3]:  # First 3 lines
                            char = script.get_character(d.character_id)
                            char_name = char.name if char else d.character_id
                            scene_context += f"    {char_name}: {d.text[:50]}...\n" if len(d.text) > 50 else f"    {char_name}: {d.text}\n"

                # Call Claude for assistance (use selected model)
                ai_model = st.session_state.get("scene_ai_model", "claude-haiku-4-5")
                ai_max_tokens = 8192 if "haiku" in ai_model else 16384
                with st.spinner(f"Thinking ({ai_model.split('-')[1]})..."):
                    try:
                        from anthropic import Anthropic
                        from src.config import config

                        client = Anthropic(api_key=config.anthropic_api_key)
                        response = client.messages.create(
                            model=ai_model,
                            max_tokens=ai_max_tokens,
                            system=f"""You are a helpful screenplay assistant. Help the user edit their movie scenes.

{scene_context}

Characters: {', '.join(c.name for c in script.characters)}

When suggesting changes, be specific about which scene and what to change.
Format dialogue suggestions as: CHARACTER: "dialogue text"
Keep responses concise and actionable.""",
                            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.scene_assistant_messages]
                        )
                        assistant_response = response.content[0].text
                        # Store which model was used for this response
                        st.session_state.scene_assistant_messages.append({
                            "role": "assistant",
                            "content": assistant_response,
                            "model": ai_model
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"AI error: {e}")

            st.markdown("---")

    # Handle empty scenes
    if not script.scenes:
        st.info("No scenes yet. Click 'Add Scene' to get started.")
        if st.button("← Back to Characters"):
            go_to_movie_step(MovieWorkflowStep.CHARACTERS)
            st.rerun()
        return

    # Initialize selected scene
    if "selected_scene_idx" not in st.session_state:
        st.session_state.selected_scene_idx = script.scenes[0].index if script.scenes else 1

    # Check if in video mode
    gen_method = state.config.generation_method if state.config else "tts_images"
    is_video_mode = gen_method in ("veo3", "wan26", "seedance15")
    visual_style = state.config.visual_style if state.config else "photorealistic, cinematic lighting"

    # Get current scene for display
    current_scene_idx = st.session_state.selected_scene_idx
    current_scene = next((s for s in script.scenes if s.index == current_scene_idx), None)
    scene_title = current_scene.title if current_scene and current_scene.title else f"Scene {current_scene_idx}"

    # Scene editor FIRST at the top in an expander for easy access
    with st.expander(f"✏️ Edit Scene {current_scene_idx}: {scene_title}", expanded=True):
        _render_scene_editor_form(script, state, visual_style)

    # Storyboard grid below the editor
    st.markdown("### 📋 Storyboard")
    _render_storyboard_grid(script, state, is_video_mode, gen_method)

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Characters"):
            go_to_movie_step(MovieWorkflowStep.CHARACTERS)
            st.rerun()
    with col2:
        if st.button("Continue →", type="primary"):
            if st.session_state.scenes_dirty:
                save_movie_state()
                st.session_state.scenes_dirty = False
            advance_movie_step()
            st.rerun()


def _render_storyboard_grid(script: Script, state: MovieModeState, is_video_mode: bool = False, gen_method: str = "tts_images") -> None:
    """Render a visual storyboard grid with overlay titles and video generation."""
    import base64

    # Quick duration metrics
    target_duration = state.config.target_duration if state.config else 180
    total_words = sum(len(d.text.split()) for s in script.scenes for d in s.dialogue)
    est_duration = total_words / 2.5 if total_words > 0 else 0

    # Count assets
    scenes_with_images = sum(1 for s in script.scenes if s.image_path and Path(s.image_path).exists())
    scenes_with_videos = sum(1 for s in script.scenes if s.video_path and Path(s.video_path).exists())
    total_scenes = len(script.scenes)

    # Top action bar with metrics and batch actions
    if is_video_mode:
        m1, m2, m3, m4, m5 = st.columns([1, 1, 1, 1, 2])
    else:
        m1, m2, m3, m4 = st.columns([1, 1, 1, 2])
        m5 = None

    with m1:
        st.metric("Scenes", total_scenes)
    with m2:
        st.metric("Duration", f"{int(est_duration // 60)}:{int(est_duration % 60):02d}")
    with m3:
        st.metric("Images", f"{scenes_with_images}/{total_scenes}")
    if is_video_mode and m5:
        with m4:
            st.metric("Videos", f"{scenes_with_videos}/{total_scenes}")
        action_col = m5
    else:
        action_col = m4

    with action_col:
        # Batch actions - show different buttons based on mode
        if is_video_mode:
            btn_col1, btn_col2, btn_col3 = st.columns(3)
        else:
            btn_col1, btn_col2 = st.columns(2)
            btn_col3 = None

        with btn_col1:
            if st.button("🤖 Prompts", use_container_width=True,
                        help="AI generates visual prompts for all scenes"):
                st.session_state.show_prompt_generator = True
                st.rerun()
        with btn_col2:
            scenes_without_images = [s for s in script.scenes if not s.image_path or not Path(s.image_path).exists()]
            btn_label = f"🎨 Images ({len(scenes_without_images)})" if scenes_without_images else "🎨 Done"
            if st.button(btn_label, use_container_width=True,
                        disabled=len(scenes_without_images) == 0,
                        help="Generate images for scenes without images"):
                st.session_state.generate_all_images = True
                st.rerun()

        if is_video_mode and btn_col3:
            with btn_col3:
                # Determine which scenes can have videos generated
                scenes_needing_video = [s for s in script.scenes if not (s.video_path and Path(s.video_path).exists())]
                # For I2V modes, filter to scenes with images
                if gen_method in ["wan26", "seedance15"]:
                    scenes_needing_video = [s for s in scenes_needing_video if s.image_path and Path(s.image_path).exists()]

                btn_label = f"🎬 Videos ({len(scenes_needing_video)})" if scenes_needing_video else "🎬 Done"
                if st.button(btn_label, use_container_width=True,
                            disabled=len(scenes_needing_video) == 0,
                            help="Generate videos for scenes"):
                    st.session_state.generate_all_videos = True
                    st.rerun()

    # Batch video generation
    if st.session_state.get("generate_all_videos") and is_video_mode:
        scenes_needing_video = [s for s in script.scenes if not (s.video_path and Path(s.video_path).exists())]
        if gen_method in ["wan26", "seedance15"]:
            scenes_needing_video = [s for s in scenes_needing_video if s.image_path and Path(s.image_path).exists()]
        if scenes_needing_video:
            _generate_all_videos_inline(state, scenes_needing_video, gen_method, state.config)
        st.session_state.generate_all_videos = False

    # Prompt generator modal
    if st.session_state.get("show_prompt_generator"):
        _render_prompt_generator_modal(script, state)

    # Batch image generation
    if st.session_state.get("generate_all_images"):
        _generate_all_scene_images(script, state)
        st.session_state.generate_all_images = False

    # Batch settings - Apply to All Scenes
    with st.expander("⚙️ Batch Settings - Apply to All Scenes", expanded=False):
        batch_col1, batch_col2, batch_col3 = st.columns(3)

        with batch_col1:
            st.markdown("**Image Model**")
            image_models = {
                "Default": None,
                "Pro (4K)": "gemini-3-pro-image-preview",
                "Fast": "gemini-2.5-flash-image",
                "Imagen Ultra": "imagen-4.0-ultra-generate-001",
            }
            batch_img_model = st.selectbox(
                "Set all to:",
                options=list(image_models.keys()),
                key="scenes_batch_img_model",
                label_visibility="collapsed",
            )
            if st.button("Apply to All", key="scenes_apply_img_model_all", use_container_width=True):
                new_model = image_models[batch_img_model]
                for scene in script.scenes:
                    try:
                        scene.image_model = new_model
                    except Exception:
                        object.__setattr__(scene, 'image_model', new_model)
                save_movie_state()
                st.success(f"Set all scenes to {batch_img_model}")
                st.rerun()

        with batch_col2:
            st.markdown("**Video Model**")
            # Combined video model + Veo variant options
            video_models = {
                "Project Default": (None, None),
                "Veo 3.1 Standard (Best)": ("veo3", "veo-3.1-generate-preview"),
                "Veo 3.1 Fast": ("veo3", "veo-3.1-fast-generate-preview"),
                "WAN 2.6 (AtlasCloud)": ("wan26", None),
                "Seedance 1.5 Pro": ("seedance15", None),
                "Seedance Fast": ("seedance_fast", None),
            }
            batch_vid_model = st.selectbox(
                "Set all to:",
                options=list(video_models.keys()),
                key="scenes_batch_vid_model",
                label_visibility="collapsed",
            )
            if st.button("Apply to All", key="scenes_apply_vid_model_all", use_container_width=True):
                model_value, variant_value = video_models[batch_vid_model]
                for scene in script.scenes:
                    try:
                        scene.generation_model = model_value
                    except Exception:
                        object.__setattr__(scene, 'generation_model', model_value)
                    # Also set Veo variant if applicable
                    if variant_value is not None:
                        try:
                            scene.veo_model_variant = variant_value
                        except Exception:
                            object.__setattr__(scene, 'veo_model_variant', variant_value)
                st.session_state.visuals_dirty = True
                save_movie_state()
                st.success(f"Set all scenes to {batch_vid_model}")
                st.rerun()

        with batch_col3:
            st.markdown("**Resolution**")
            resolutions = {
                "720p (Default)": "720p",
                "1080p (Full HD)": "1080p",
                "480p (Fast)": "480p",
            }
            batch_resolution = st.selectbox(
                "Set all to:",
                options=list(resolutions.keys()),
                key="scenes_batch_resolution",
                label_visibility="collapsed",
            )
            if st.button("Apply to All", key="scenes_apply_resolution_all", use_container_width=True):
                new_res = resolutions[batch_resolution]
                for scene in script.scenes:
                    try:
                        scene.resolution = new_res
                    except Exception:
                        object.__setattr__(scene, 'resolution', new_res)
                save_movie_state()
                st.success(f"Set all scenes to {batch_resolution}")
                st.rerun()

        # Regenerate all video prompts
        st.markdown("---")
        regen_col1, regen_col2 = st.columns([2, 1])
        with regen_col1:
            use_img_for_all = st.checkbox(
                "Use scene images for context when regenerating prompts",
                value=True,
                key="scenes_batch_use_img_ctx",
            )
        with regen_col2:
            scenes_missing_prompts = sum(1 for s in script.scenes if not getattr(s, 'video_prompt', None))
            btn_label = f"🔄 Regenerate All Video Prompts ({scenes_missing_prompts} missing)"
            if st.button(btn_label, key="scenes_regen_all_video_prompts", use_container_width=True):
                progress = st.progress(0, text="Regenerating video prompts...")
                for i, scene in enumerate(script.scenes):
                    progress.progress((i + 1) / len(script.scenes), text=f"Scene {scene.index}...")
                    has_scene_image = scene.image_path and Path(scene.image_path).exists()
                    new_prompt = _regenerate_scene_prompt(
                        scene, script, state,
                        prompt_type="video",
                        use_image_context=use_img_for_all and has_scene_image
                    )
                    if new_prompt:
                        try:
                            scene.video_prompt = new_prompt
                        except Exception:
                            object.__setattr__(scene, 'video_prompt', new_prompt)
                save_movie_state()
                st.success("All video prompts regenerated!")
                st.rerun()

    st.markdown("---")

    # Pre-compute all scene card data in one pass (no base64 encoding - use st.image directly)
    scene_data = []
    gen_method = state.config.generation_method if state.config else "tts_images"
    for scene in script.scenes:
        has_image = scene.image_path and Path(scene.image_path).exists()
        title_text = f"Scene {scene.index}: {scene.title or 'Untitled'}"
        clip_duration = scene.get_clip_duration(gen_method)

        # Pre-compute character info with portrait paths (not base64)
        chars_with_portraits = []
        chars_without = []
        for char_id in scene.direction.visible_characters:
            char = script.get_character(char_id)
            if char:
                if char.reference_image_path and Path(char.reference_image_path).exists():
                    chars_with_portraits.append((char.name, char.reference_image_path))
                else:
                    chars_without.append(char.name)

        scene_data.append({
            "scene": scene,
            "has_image": has_image,
            "title_text": title_text,
            "clip_duration": clip_duration,
            "chars_with_portraits": chars_with_portraits,
            "chars_without": chars_without,
        })

    # Scene grid (3 per row like Song Mode)
    cols_per_row = 3
    for row_start in range(0, len(scene_data), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, col in enumerate(cols):
            data_idx = row_start + col_idx
            if data_idx < len(scene_data):
                sd = scene_data[data_idx]
                scene = sd["scene"]
                with col:
                    is_selected = st.session_state.selected_scene_idx == scene.index

                    with st.container(border=True):
                        # Scene title
                        st.markdown(f"**{sd['title_text']}**")

                        # Check if scene has video - either from scene.video_path or from videos directory
                        has_video = scene.video_path and Path(scene.video_path).exists()

                        # Auto-detect video from project videos directory if not set
                        if not has_video:
                            video_dir = get_project_dir() / "videos"
                            if video_dir.exists():
                                video_matches = list(video_dir.glob(f"scene_{scene.index:03d}*.mp4"))
                                if video_matches:
                                    # Use the first match and update scene (don't save during render)
                                    scene.video_path = str(sorted(video_matches)[0])
                                    has_video = True

                        # Show video if available, otherwise show image
                        if has_video:
                            # Tabs to switch between image and video
                            img_tab, vid_tab = st.tabs(["🖼️ Image", "🎬 Video"])
                            with img_tab:
                                if sd["has_image"]:
                                    st.image(scene.image_path, width="stretch")
                                else:
                                    st.caption("No image")
                            with vid_tab:
                                st.video(str(scene.video_path))
                                # Video variants
                                video_dir = get_project_dir() / "videos"
                                video_variants = list(video_dir.glob(f"scene_{scene.index:03d}*.mp4"))
                                if len(video_variants) > 1:
                                    st.caption(f"📹 {len(video_variants)} variants")
                        elif sd["has_image"]:
                            st.image(scene.image_path, width="stretch")
                        else:
                            # Placeholder
                            st.markdown('''
                                <div style="height:100px;background:linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                                    border-radius:8px;display:flex;align-items:center;justify-content:center;">
                                    <span style="color:#555;font-size:32px;">🎬</span>
                                </div>
                            ''', unsafe_allow_html=True)

                        # Scene metadata
                        st.caption(f"📍 {scene.direction.setting[:50]}..." if len(scene.direction.setting) > 50 else f"📍 {scene.direction.setting}")
                        st.caption(f"🎥 {scene.direction.camera} | {scene.direction.mood} | ⏱️ ~{sd['clip_duration']}s")

                        # Character portraits (use st.image for reliability)
                        if sd["chars_with_portraits"] or sd["chars_without"]:
                            num_chars = len(sd["chars_with_portraits"]) + len(sd["chars_without"])
                            char_cols = st.columns(min(num_chars, 6))
                            col_idx = 0
                            for name, portrait_path in sd["chars_with_portraits"][:6]:
                                with char_cols[col_idx]:
                                    st.image(portrait_path, width=40)
                                    st.caption(name[:8])
                                col_idx += 1
                            for name in sd["chars_without"][:6 - len(sd["chars_with_portraits"])]:
                                with char_cols[col_idx]:
                                    st.markdown("👤")
                                    st.caption(name[:8])
                                col_idx += 1

                        # === VARIANT PICKERS (visible on card) ===
                        # Count image variants - check multiple locations
                        scene_img_dir = get_project_dir() / "scenes" / f"scene_{scene.index:03d}" / "images"
                        scenes_dir = get_project_dir() / "scenes"
                        all_img_variants = set()

                        # From scene.image_variants list
                        for var_path in scene.image_variants:
                            if Path(var_path).exists():
                                all_img_variants.add(str(Path(var_path).resolve()))

                        # From per-scene images subdirectory
                        if scene_img_dir.exists():
                            for img_file in list(scene_img_dir.glob("*.png")) + list(scene_img_dir.glob("*.jpg")):
                                all_img_variants.add(str(img_file.resolve()))

                        # From scenes directory directly (batch generation saves here)
                        if scenes_dir.exists():
                            for img_file in list(scenes_dir.glob(f"scene_{scene.index:03d}*.png")) + list(scenes_dir.glob(f"scene_{scene.index:03d}*.jpg")):
                                all_img_variants.add(str(img_file.resolve()))

                        # From per-scene folder directly (old format: scenes/scene_XXX/scene_XXX.png)
                        scene_folder = get_project_dir() / "scenes" / f"scene_{scene.index:03d}"
                        if scene_folder.exists():
                            for img_file in list(scene_folder.glob("*.png")) + list(scene_folder.glob("*.jpg")):
                                if img_file.parent == scene_folder:  # Only direct children, not from images/
                                    all_img_variants.add(str(img_file.resolve()))

                        # Current image path
                        if sd["has_image"]:
                            all_img_variants.add(str(Path(scene.image_path).resolve()))
                        num_img_var = len(all_img_variants)

                        # Count video variants
                        video_dir = get_project_dir() / "videos"
                        video_variants = list(video_dir.glob(f"scene_{scene.index:03d}*.mp4")) if video_dir.exists() else []
                        num_vid_var = len(video_variants)

                        # Always show variant pickers row (image and video)
                        var_col1, var_col2 = st.columns(2)
                        with var_col1:
                            # Image variant picker - always show if we have images
                            if num_img_var >= 1:
                                img_list = sorted(all_img_variants)
                                img_names = {p: Path(p).stem[-8:] for p in img_list}
                                cur_img = str(Path(scene.image_path).resolve()) if sd["has_image"] else ""
                                img_idx = img_list.index(cur_img) if cur_img in img_list else 0
                                new_img = st.selectbox(
                                    f"📷 Image ({num_img_var})",
                                    options=img_list,
                                    index=img_idx,
                                    format_func=lambda x: img_names.get(x, Path(x).stem),
                                    key=f"grid_img_{scene.index}",
                                )
                                if new_img != cur_img and new_img:
                                    try:
                                        scene.image_path = new_img
                                    except Exception:
                                        object.__setattr__(scene, 'image_path', new_img)
                                    save_movie_state()
                            else:
                                st.caption("📷 No images yet")
                        with var_col2:
                            # Video variant picker
                            if num_vid_var >= 1:
                                vid_names = {str(v.resolve()): v.stem.replace(f"scene_{scene.index:03d}", "").lstrip("_") or "v1"
                                             for v in sorted(video_variants)}
                                cur_vid = str(Path(scene.video_path).resolve()) if scene.video_path and Path(scene.video_path).exists() else ""
                                vid_opts = list(vid_names.keys())
                                vid_idx = vid_opts.index(cur_vid) if cur_vid in vid_opts else 0
                                new_vid = st.selectbox(
                                    f"🎬 Video ({num_vid_var})",
                                    options=vid_opts,
                                    index=vid_idx,
                                    format_func=lambda x: vid_names.get(x, Path(x).stem),
                                    key=f"grid_vid_{scene.index}",
                                )
                                if new_vid != cur_vid and new_vid:
                                    scene.video_path = new_vid
                                    save_movie_state()
                            else:
                                st.caption("🎬 No videos yet")

                        # === SETTINGS ROW (visible on card) ===
                        set_c1, set_c2 = st.columns(2)
                        with set_c1:
                            img_models = {None: "Def", "gemini-3-pro-image-preview": "Pro", "gemini-2.5-flash-image": "Fast"}
                            cur_img_model = getattr(scene, 'image_model', None)
                            img_keys = list(img_models.keys())
                            img_m_idx = img_keys.index(cur_img_model) if cur_img_model in img_keys else 0
                            new_img_m = st.selectbox(
                                "Img",
                                options=img_keys,
                                index=img_m_idx,
                                format_func=lambda x: img_models.get(x, str(x)),
                                key=f"grid_imgm_{scene.index}",
                            )
                            if new_img_m != cur_img_model:
                                try:
                                    scene.image_model = new_img_m
                                except Exception:
                                    object.__setattr__(scene, 'image_model', new_img_m)
                                save_movie_state()

                        with set_c2:
                            ratios = {None: "16:9", "16:9": "16:9", "9:16": "9:16", "1:1": "1:1"}
                            cur_ratio = getattr(scene, 'image_aspect_ratio', None)
                            ratio_keys = list(ratios.keys())
                            ratio_idx = ratio_keys.index(cur_ratio) if cur_ratio in ratio_keys else 0
                            new_ratio = st.selectbox(
                                "Ratio",
                                options=ratio_keys,
                                index=ratio_idx,
                                format_func=lambda x: ratios.get(x, str(x)),
                                key=f"grid_ratio_{scene.index}",
                            )
                            if new_ratio != cur_ratio:
                                try:
                                    scene.image_aspect_ratio = new_ratio
                                except Exception:
                                    object.__setattr__(scene, 'image_aspect_ratio', new_ratio)
                                save_movie_state()

                        # Prompts & Settings (expandable - detailed settings)
                        prompt_text = scene.visual_prompt or "No prompts set"
                        prompt_preview = prompt_text[:40] + "..." if len(prompt_text) > 40 else prompt_text
                        with st.expander(f"📝 {prompt_preview}", expanded=False):
                            # Collect all inputs for visualization
                            project_method = state.config.generation_method if state.config else "tts_images"
                            scene_model = scene.generation_model or project_method
                            is_i2v = "i2v" in str(scene_model).lower() or scene_model in ["wan26", "seedance15"]

                            # Collect ALL visible characters (with and without portraits)
                            chars_with_portraits = []
                            chars_without_portraits = []
                            for char_id in scene.direction.visible_characters:
                                char = script.get_character(char_id)
                                if char:
                                    if char.reference_image_path and Path(char.reference_image_path).exists():
                                        chars_with_portraits.append((char.name, char.reference_image_path))
                                    else:
                                        chars_without_portraits.append(char.name)

                            # Get style reference (previous scene)
                            style_ref_path = None
                            if scene.index > 1:
                                prev_scene = next((s for s in script.scenes if s.index == scene.index - 1), None)
                                if prev_scene and prev_scene.image_path and Path(prev_scene.image_path).exists():
                                    style_ref_path = prev_scene.image_path

                            # === IMAGE GENERATION SECTION ===
                            st.markdown("### 🖼️ Image Generation")

                            # Show visible characters status with IDs for debugging
                            total_visible = len(scene.direction.visible_characters)
                            has_portrait = len(chars_with_portraits)
                            missing_portrait = len(chars_without_portraits)
                            st.caption(f"Visible characters: {total_visible} | With portrait: {has_portrait} | Missing portrait: {missing_portrait}")
                            st.caption(f"IDs in scene: {scene.direction.visible_characters}")

                            if chars_without_portraits:
                                st.warning(f"⚠️ Missing portraits: {', '.join(chars_without_portraits)} - generate portraits in Characters tab")

                            st.markdown("**Input Images:**")
                            if chars_with_portraits or style_ref_path:
                                img_cols = st.columns(min(len(chars_with_portraits) + (1 if style_ref_path else 0), 5))
                                col_idx = 0
                                for name, path in chars_with_portraits[:4]:
                                    with img_cols[col_idx]:
                                        st.image(path, width=70)
                                        st.caption(f"📷 {name}")
                                    col_idx += 1
                                if style_ref_path and col_idx < len(img_cols):
                                    with img_cols[col_idx]:
                                        st.image(style_ref_path, width=70)
                                        st.caption("🎨 Style ref")
                            else:
                                st.caption("No reference images (text-only generation)")

                            st.markdown("**Text Prompt:**")
                            # Show context that will be added
                            context_parts = []
                            if script.world_description:
                                context_parts.append(f"Setting/World: {script.world_description}")
                            if state.config and state.config.visual_style:
                                context_parts.append(f"Style: {state.config.visual_style}")
                            if context_parts:
                                st.caption("Context: " + " | ".join(context_parts))

                            # Show image anti-prompt (what to avoid)
                            img_visual_style = state.config.visual_style if state.config else (script.visual_style if script.visual_style else "")
                            img_anti_prompt = get_image_anti_prompt(img_visual_style)
                            st.caption(f"🚫 **Avoid:** _{img_anti_prompt}_")

                            # Visual prompt - use versioned key to sync with edit menu
                            st.markdown("**Image Prompt**")
                            widget_version = st.session_state.get(f"visual_prompt_version_{scene.index}", 0)
                            new_prompt = st.text_area(
                                "Visual Prompt",
                                value=scene.visual_prompt or "",
                                key=f"prompt_{scene.index}_v{widget_version}",
                                height=60,
                                label_visibility="collapsed",
                                placeholder="Describe the visual scene for image generation..."
                            )
                            if new_prompt != (scene.visual_prompt or ""):
                                try:
                                    scene.visual_prompt = new_prompt
                                except Exception:
                                    object.__setattr__(scene, 'visual_prompt', new_prompt)
                                st.session_state.scenes_dirty = True

                            # Editable negative prompt for images
                            st.markdown("**🚫 Negative Prompt (Avoid)**")
                            default_neg = get_image_anti_prompt(img_visual_style)
                            current_img_neg = getattr(scene, 'image_negative_prompt', None) or default_neg
                            new_img_neg = st.text_area(
                                "Image Negative Prompt",
                                value=current_img_neg,
                                key=f"img_neg_{scene.index}",
                                height=40,
                                label_visibility="collapsed",
                                placeholder="Keywords to avoid in image generation..."
                            )
                            if new_img_neg != current_img_neg:
                                try:
                                    scene.image_negative_prompt = new_img_neg
                                except Exception:
                                    object.__setattr__(scene, 'image_negative_prompt', new_img_neg)
                                st.session_state.scenes_dirty = True
                            # Reset to default button
                            if st.button("↩️ Reset to Default", key=f"reset_img_neg_{scene.index}", use_container_width=True):
                                try:
                                    scene.image_negative_prompt = None
                                except Exception:
                                    object.__setattr__(scene, 'image_negative_prompt', None)
                                save_movie_state()
                                st.rerun()

                            # Regenerate image prompt button
                            if st.button("🔄 Regenerate Image Prompt", key=f"regen_img_prompt_{scene.index}",
                                        use_container_width=True):
                                with st.spinner("Generating image prompt..."):
                                    new_img_prompt = _regenerate_scene_prompt(scene, script, state, prompt_type="visual", use_image_context=False)
                                    if new_img_prompt:
                                        try:
                                            scene.visual_prompt = new_img_prompt
                                        except Exception:
                                            object.__setattr__(scene, 'visual_prompt', new_img_prompt)
                                        save_movie_state()
                                        # Increment version to sync widgets across UI
                                        st.session_state[f"visual_prompt_version_{scene.index}"] = widget_version + 1
                                        st.success("Image prompt regenerated!")
                                        st.rerun()

                            # === VIDEO GENERATION SECTION ===
                            st.markdown("---")
                            st.markdown("### 🎬 Video Generation")

                            # Check what inputs are available
                            has_scene_image = scene.image_path and Path(scene.image_path).exists()
                            has_scene_video = scene.video_path and Path(scene.video_path).exists()

                            st.markdown("**Input Image (for I2V):**")
                            if has_scene_image:
                                vid_img_col1, vid_img_col2 = st.columns([1, 3])
                                with vid_img_col1:
                                    st.image(scene.image_path, width=100)
                                with vid_img_col2:
                                    st.success("✅ Scene image ready for I2V")
                            else:
                                st.warning("❌ No scene image - generate image first for I2V models")

                            st.markdown("**Video Prompt:**")
                            # Get video prompt from scene - use session state override if just regenerated
                            regen_key = f"video_prompt_regenerated_{scene.index}"
                            if regen_key in st.session_state and st.session_state[regen_key]:
                                # Use the freshly generated prompt from session state
                                current_video_prompt = st.session_state.get(f"video_prompt_value_{scene.index}", "") or ""
                                st.session_state[regen_key] = False
                            else:
                                current_video_prompt = getattr(scene, 'video_prompt', None) or ""

                            # Use a unique key that changes when we regenerate to force refresh
                            widget_version = st.session_state.get(f"video_prompt_version_{scene.index}", 0)
                            widget_key = f"video_prompt_{scene.index}_v{widget_version}"

                            new_video_prompt = st.text_area(
                                "Video Prompt",
                                value=current_video_prompt,
                                key=widget_key,
                                height=80,
                                label_visibility="collapsed",
                                placeholder="Describe character movements, actions, AND dialogue in quotes..."
                            )
                            if new_video_prompt != current_video_prompt:
                                try:
                                    scene.video_prompt = new_video_prompt
                                except Exception:
                                    object.__setattr__(scene, 'video_prompt', new_video_prompt)
                                # Auto-save to persist the change
                                save_movie_state()
                                st.session_state.scenes_dirty = False

                            # Show video negative prompt for WAN/Seedance (not Veo)
                            scene_gen_model = getattr(scene, 'generation_model', None) or project_method
                            if scene_gen_model in ["wan26", "seedance15"]:
                                vid_visual_style = state.config.visual_style if state.config else state.script.visual_style
                                vid_neg_prompt = get_video_negative_prompt(vid_visual_style, scene_gen_model)
                                if vid_neg_prompt:
                                    st.caption(f"🚫 **Negative prompt:** _{vid_neg_prompt}_")

                            # Regenerate video prompt button
                            regen_col1, regen_col2 = st.columns([1, 2])
                            with regen_col1:
                                use_img_ctx = st.checkbox(
                                    "Use image",
                                    value=has_scene_image,
                                    key=f"regen_img_ctx_{scene.index}",
                                    help="Analyze scene image for context"
                                )
                            with regen_col2:
                                has_video_prompt = bool(current_video_prompt.strip())
                                btn_type = "primary" if not has_video_prompt else "secondary"
                                btn_label = "✨ Generate" if not has_video_prompt else "🔄 Regenerate"
                                if st.button(btn_label, key=f"regen_vprompt_{scene.index}",
                                            type=btn_type, use_container_width=True):
                                    with st.spinner("Generating video prompt..."):
                                        new_prompt = _regenerate_scene_prompt(
                                            scene, script, state,
                                            prompt_type="video",
                                            use_image_context=use_img_ctx and has_scene_image
                                        )
                                        if new_prompt:
                                            try:
                                                scene.video_prompt = new_prompt
                                            except Exception:
                                                object.__setattr__(scene, 'video_prompt', new_prompt)
                                            save_movie_state()
                                            # Store the new prompt value and increment version to force widget refresh
                                            st.session_state[f"video_prompt_value_{scene.index}"] = new_prompt
                                            st.session_state[f"video_prompt_regenerated_{scene.index}"] = True
                                            st.session_state[f"video_prompt_version_{scene.index}"] = st.session_state.get(f"video_prompt_version_{scene.index}", 0) + 1
                                            st.success("Video prompt updated!")
                                            st.rerun()
                                        else:
                                            st.error("Failed to generate prompt")

                            # Video generation settings
                            if project_method != "tts_images":
                                st.markdown("**Settings:**")

                                # Show current status
                                input_status = []
                                if has_scene_image:
                                    input_status.append("✅ Image ready")
                                else:
                                    input_status.append("❌ No image")
                                if has_scene_video:
                                    input_status.append("✅ Has video")
                                st.caption(" | ".join(input_status))

                                # Video model with clear I2V/T2V indication + Veo variants
                                # Format: (display_name, generation_model, veo_variant)
                                model_opts = {
                                    "default": (f"Default ({project_method})", None, None),
                                    # Veo variants (3.1 only - 3.0 deprecated)
                                    "veo31_standard": ("📝→🎬 Veo 3.1 Standard (Best)", "veo3", "veo-3.1-generate-preview"),
                                    "veo31_fast": ("📝→🎬 Veo 3.1 Fast", "veo3", "veo-3.1-fast-generate-preview"),
                                    # I2V models (require scene image)
                                    "wan26_i2v": ("🖼️→🎬 WAN 2.6 I2V (image→video)", "wan26_i2v", None),
                                    "seedance15_i2v": ("🖼️→🎬 Seedance 1.5 I2V (lip-sync)", "seedance15_i2v", None),
                                    "seedance_fast_i2v": ("🖼️→🎬 Seedance Fast I2V", "seedance_fast_i2v", None),
                                    # T2V models (text prompt only)
                                    "wan26_t2v": ("📝→🎬 WAN 2.6 T2V (text→video)", "wan26_t2v", None),
                                    "seedance15_t2v": ("📝→🎬 Seedance 1.5 T2V (text→video)", "seedance15_t2v", None),
                                }

                                # Determine current selection key based on model + variant
                                cur_model = scene.generation_model
                                cur_variant = getattr(scene, 'veo_model_variant', None)
                                cur_key = "default"
                                if cur_model == "veo3":
                                    if cur_variant == "veo-3.1-fast-generate-preview":
                                        cur_key = "veo31_fast"
                                    else:
                                        # Default to 3.1 Standard (3.0 deprecated)
                                        cur_key = "veo31_standard"
                                elif cur_model in model_opts:
                                    cur_key = cur_model
                                elif cur_model:
                                    # Try to find matching key
                                    for k, v in model_opts.items():
                                        if v[1] == cur_model:
                                            cur_key = k
                                            break

                                m_keys = list(model_opts.keys())
                                m_idx = m_keys.index(cur_key) if cur_key in m_keys else 0
                                new_model_key = st.selectbox(
                                    "Video Model",
                                    options=m_keys,
                                    index=m_idx,
                                    format_func=lambda x: model_opts.get(x, (str(x), None, None))[0],
                                    key=f"model_{scene.index}"
                                )

                                # Apply model and variant from selection
                                if new_model_key != cur_key:
                                    display_name, gen_model, veo_variant = model_opts[new_model_key]
                                    try:
                                        scene.generation_model = gen_model
                                    except Exception:
                                        object.__setattr__(scene, 'generation_model', gen_model)
                                    if veo_variant:
                                        try:
                                            scene.veo_model_variant = veo_variant
                                        except Exception:
                                            object.__setattr__(scene, 'veo_model_variant', veo_variant)
                                    st.session_state.scenes_dirty = True
                                    save_movie_state()
                                    # Note: Prompt regeneration removed - use "Regenerate Video Prompt" button

                                # Show warning if I2V selected but no image
                                _, eff_gen_model, _ = model_opts[new_model_key]
                                eff_model = eff_gen_model or project_method
                                is_i2v = "i2v" in str(eff_model).lower() or eff_model in ["wan26", "seedance15", "seedance_fast"]
                                if is_i2v and not has_scene_image:
                                    st.warning("⚠️ I2V requires scene image - generate image first!")

                                v_col1, v_col2 = st.columns(2)
                                with v_col1:
                                    # Duration based on model
                                    if "veo3" in str(eff_model):
                                        dur_options = [4, 6, 8]
                                    elif "wan26" in str(eff_model):
                                        dur_options = [5, 10, 15]
                                    else:  # seedance (Pro or Fast)
                                        dur_options = [3, 5, 8, 10, 15]

                                    auto_dur = scene.get_clip_duration(eff_model)
                                    current_dur = scene.clip_duration
                                    dur_idx = 0 if current_dur is None else (dur_options.index(current_dur) + 1 if current_dur in dur_options else 0)
                                    new_dur = st.selectbox(
                                        "Duration",
                                        options=[None] + dur_options,
                                        index=dur_idx,
                                        format_func=lambda x: f"Auto ({auto_dur}s)" if x is None else f"{x}s",
                                        key=f"dur_{scene.index}"
                                    )
                                    if new_dur != current_dur:
                                        scene.clip_duration = new_dur
                                        st.session_state.scenes_dirty = True

                                with v_col2:
                                    # Resolution
                                    if "veo3" in str(eff_model):
                                        res_options = ["720p", "1080p"]
                                    else:
                                        res_options = ["480p", "720p", "1080p"]

                                    current_res = scene.resolution
                                    res_idx = 0 if current_res is None else (res_options.index(current_res) + 1 if current_res in res_options else 0)
                                    new_res = st.selectbox(
                                        "Resolution",
                                        options=[None] + res_options,
                                        index=res_idx,
                                        format_func=lambda x: "Default" if x is None else x,
                                        key=f"res_{scene.index}"
                                    )
                                    if new_res != current_res:
                                        scene.resolution = new_res
                                        st.session_state.scenes_dirty = True

                                # Lip-sync for Seedance
                                if "seedance" in str(eff_model).lower():
                                    current_lip = scene.enable_lip_sync if scene.enable_lip_sync is not None else None
                                    new_lip = st.checkbox(
                                        "Enable Lip Sync",
                                        value=current_lip if current_lip is not None else True,
                                        key=f"lip_{scene.index}"
                                    )
                                    if new_lip != current_lip:
                                        scene.enable_lip_sync = new_lip
                                        st.session_state.scenes_dirty = True

                                # Show negative prompt that will be used (WAN 2.6 only)
                                if "wan26" in str(eff_model).lower():
                                    # Get visual style from config or script
                                    scene_visual_style = state.config.visual_style if state.config else (state.script.visual_style if state.script else "")
                                    style_lower = (scene_visual_style or "").lower()
                                    if "photorealistic" in style_lower or "realistic" in style_lower or "photo" in style_lower:
                                        neg_prompt = "CGI, cartoon, animated, 3D render, digital art, stylized, artificial, plastic skin, video game, unrealistic, smooth skin, airbrushed"
                                    elif "anime" in style_lower or "cartoon" in style_lower:
                                        neg_prompt = "photorealistic, real photo, live action, realistic skin texture"
                                    elif "3d" in style_lower or "pixar" in style_lower or "animated" in style_lower:
                                        neg_prompt = "photorealistic, real photo, live action, 2D, flat"
                                    else:
                                        neg_prompt = "(none - style not recognized)"

                                    with st.expander("🚫 Negative Prompt", expanded=False):
                                        st.caption(f"**Style:** {scene_visual_style}")
                                        st.code(neg_prompt, language=None)
                                        st.caption("This is auto-generated based on your visual style setting.")

                        # Dialogue lines
                        if scene.dialogue:
                            with st.expander(f"💬 Dialogue ({len(scene.dialogue)} lines)", expanded=True):
                                for d in scene.dialogue:
                                    char = script.get_character(d.character_id)
                                    char_name = char.name if char else d.character_id
                                    st.caption(f"**{char_name}:** {d.text}")

                        # Action buttons - include video for video modes
                        has_image = scene.image_path and Path(scene.image_path).exists()
                        has_video = scene.video_path and Path(scene.video_path).exists()

                        if is_video_mode:
                            btn1, btn2, btn3 = st.columns(3)
                        else:
                            btn1, btn2 = st.columns(2)
                            btn3 = None

                        with btn1:
                            if st.button("✏️", key=f"edit_{scene.index}", use_container_width=True,
                                        type="primary" if is_selected else "secondary",
                                        help="Edit scene"):
                                st.session_state.selected_scene_idx = scene.index
                                st.rerun()
                        with btn2:
                            img_icon = "🔄" if has_image else "🎨"
                            if st.button(img_icon, key=f"img_{scene.index}", use_container_width=True,
                                        help="Generate image" if not has_image else "Regenerate image"):
                                _generate_single_scene_image(scene, script, state)
                                st.rerun()

                        if is_video_mode and btn3:
                            with btn3:
                                # Video button
                                can_gen_video = has_image or gen_method == "veo3"
                                if has_video:
                                    if st.button("🔁", key=f"vid_{scene.index}", use_container_width=True,
                                                help="Regenerate video"):
                                        scene.video_path = None
                                        save_movie_state()
                                        use_v2v = st.session_state.get(f"v2v_{scene.index}", False)
                                        _generate_single_video_inline(state, scene, gen_method, state.config, use_v2v=use_v2v)
                                elif can_gen_video:
                                    if st.button("🎬", key=f"vid_{scene.index}", use_container_width=True,
                                                help="Generate video"):
                                        use_v2v = st.session_state.get(f"v2v_{scene.index}", False)
                                        _generate_single_video_inline(state, scene, gen_method, state.config, use_v2v=use_v2v)
                                else:
                                    st.button("🎬", key=f"vid_{scene.index}", use_container_width=True,
                                             disabled=True, help="Need image first")

                        # V2V continuity option (for scenes after first)
                        if is_video_mode and scene.index > 1:
                            # Check if previous scene has video
                            prev_scene = next((s for s in script.scenes if s.index == scene.index - 1), None)
                            has_prev_video = prev_scene and prev_scene.video_path and Path(prev_scene.video_path).exists()
                            if has_prev_video:
                                st.checkbox(
                                    "🔗 V2V",
                                    value=st.session_state.get(f"v2v_{scene.index}", False),
                                    key=f"v2v_{scene.index}",
                                    help="Use previous scene's video for continuity"
                                )


def _render_prompt_generator_modal(script: Script, state: MovieModeState) -> None:
    """Render the AI scene setup and prompt generator modal."""
    # Use a container with border instead of expander to avoid state issues
    with st.container(border=True):
        st.markdown("### 🤖 AI Scene Setup & Prompts")
        st.markdown("""
        **AI will analyze each scene and set up:**
        - Camera angle and shot type
        - Lighting and mood
        - Which characters should be visible
        - Visual prompt for image generation
        - Video animation prompt (format depends on selected model)
        - **Best video generation model per scene** (if enabled)
        """)

        visual_style = state.config.visual_style if state.config else "photorealistic, cinematic lighting"
        project_method = state.config.generation_method if state.config else "tts_images"

        # Image model selector with clear Fast vs Pro distinction
        st.markdown("#### 🖼️ Image Generation Settings")
        image_models = {
            # Pro models first (best quality) - recommended
            "gemini-3-pro-image-preview": "🏆 Nano Banana Pro (Best, 4K, 14 refs)",
            "imagen-4.0-ultra-generate-001": "🏆 Imagen 4 Ultra (Highest quality)",
            # Standard models
            "imagen-4.0-generate-001": "🎨 Imagen 4 Standard (Good quality)",
            # Fast models
            "gemini-2.5-flash-image": "⚡ Nano Banana Fast (Quick, 1K)",
            "imagen-4.0-fast-generate-001": "⚡ Imagen 4 Fast (Quick)",
        }

        # Get current image model from session state or config
        current_image_model = st.session_state.get("scene_image_model", "gemini-3-pro-image-preview")

        img_col1, img_col2, img_col3 = st.columns([2, 1, 1])
        with img_col1:
            selected_image_model = st.selectbox(
                "Image Generation Model",
                options=list(image_models.keys()),
                index=list(image_models.keys()).index(current_image_model) if current_image_model in image_models else 1,
                format_func=lambda x: image_models[x],
                key="ai_setup_image_model",
                help="Model to use for generating scene images"
            )
        with img_col2:
            image_sizes = {"1K": "1024px", "2K": "2048px", "4K": "4096px"}
            current_size = st.session_state.get("scene_image_size", "2K")
            selected_image_size = st.selectbox(
                "Image Size",
                options=list(image_sizes.keys()),
                index=list(image_sizes.keys()).index(current_size) if current_size in image_sizes else 1,
                format_func=lambda x: image_sizes[x],
                key="ai_setup_image_size",
                help="Output resolution for generated images"
            )
        with img_col3:
            aspect_ratios = {
                "16:9": "16:9 (Widescreen)",
                "9:16": "9:16 (Portrait)",
                "1:1": "1:1 (Square)",
                "4:3": "4:3 (Standard)",
                "3:4": "3:4 (Portrait)",
                "21:9": "21:9 (Ultrawide)",
            }
            current_ar = st.session_state.get("scene_image_aspect_ratio", "16:9")
            selected_aspect_ratio = st.selectbox(
                "Aspect Ratio",
                options=list(aspect_ratios.keys()),
                index=list(aspect_ratios.keys()).index(current_ar) if current_ar in aspect_ratios else 0,
                format_func=lambda x: aspect_ratios[x],
                key="ai_setup_image_aspect_ratio",
                help="Aspect ratio for generated images"
            )

        # Save selections to session state for use during generation
        st.session_state.scene_image_model = selected_image_model
        st.session_state.scene_image_size = selected_image_size
        st.session_state.scene_image_aspect_ratio = selected_aspect_ratio

        # Scene consistency options
        st.markdown("#### 🔗 Scene Consistency")
        consist_col1, consist_col2 = st.columns(2)
        with consist_col1:
            use_prev_image = st.checkbox(
                "Use previous scene image as reference",
                value=True,
                key="ai_setup_use_prev_image",
                help="Pass the previous scene's image to maintain visual consistency"
            )
        with consist_col2:
            use_prev_video = st.checkbox(
                "Use previous scene video as reference",
                value=False,
                key="ai_setup_use_prev_video",
                help="Pass the previous scene's video clip for continuity (uploads via Files API)"
            )

        st.session_state.use_prev_scene_image = use_prev_image
        st.session_state.use_prev_scene_video = use_prev_video

        # Video generation method selector
        st.markdown("#### 🎬 Video Generation Settings")

        # Combined video method + Veo variant options (Veo 3.0 deprecated)
        gen_methods = {
            "tts_images": ("🎙️ TTS + Images (Ken Burns)", None),
            "veo3_standard": ("📝→🎬 Veo 3.1 Standard (Best quality)", "veo-3.1-generate-preview"),
            "veo3_fast": ("📝→🎬 Veo 3.1 Fast", "veo-3.1-fast-generate-preview"),
            "wan26": ("🖼️→🎬 WAN 2.6 (image→video)", None),
            "seedance15": ("🖼️→🎬 Seedance 1.5 Pro (image→video, lip-sync)", None),
        }

        # Map project method to combined key
        if project_method == "veo3":
            default_key = "veo3_standard"
        elif project_method in gen_methods:
            default_key = project_method
        else:
            default_key = "tts_images"

        method_col1, method_col2 = st.columns([2, 1])
        with method_col1:
            selected_method_key = st.selectbox(
                "Default Video Method",
                options=list(gen_methods.keys()),
                index=list(gen_methods.keys()).index(default_key) if default_key in gen_methods else 0,
                format_func=lambda x: gen_methods[x][0],
                key="ai_setup_gen_method",
                help="Default method for all scenes (can be overridden per scene)"
            )
            # Extract base method and variant
            if selected_method_key.startswith("veo3"):
                selected_method = "veo3"
                selected_veo_variant = gen_methods[selected_method_key][1]
            else:
                selected_method = selected_method_key
                selected_veo_variant = None

            # Store Veo variant in session state
            if selected_veo_variant:
                st.session_state.scene_veo_variant = selected_veo_variant

        with method_col2:
            ai_pick_models = st.checkbox(
                "🧠 AI picks best model",
                value=False,
                key="ai_setup_pick_models",
                help="Let AI choose between I2V and T2V models based on scene context"
            )

        video_method_name = gen_methods.get(selected_method_key, (selected_method_key, None))[0].split('(')[0].strip()
        st.info(f"Style: **{visual_style}** | Image: **{image_models.get(selected_image_model, selected_image_model).split('(')[0].strip()}** | Video: **{video_method_name}**")

        # Options with unique keys
        setup_col1, setup_col2 = st.columns(2)
        with setup_col1:
            update_direction = st.checkbox("Update scene direction (camera, lighting, mood)",
                                          value=True, key="ai_setup_direction")
            update_characters = st.checkbox("Update visible characters",
                                           value=True, key="ai_setup_characters")
        with setup_col2:
            update_visual_prompt = st.checkbox("Generate image prompts",
                                              value=True, key="ai_setup_visual")
            update_video_prompt = st.checkbox("Generate video animation prompts",
                                             value=True, key="ai_setup_video")

        col1, col2 = st.columns(2)
        with col1:
            run_setup = st.button("🚀 Set Up All Scenes", type="primary",
                                  use_container_width=True, key="ai_setup_run_btn")

        with col2:
            if st.button("Cancel", use_container_width=True, key="ai_setup_cancel_btn"):
                st.session_state.show_prompt_generator = False
                st.rerun()

        # Run setup after button rendering to avoid widget state issues
        if run_setup:
            _run_ai_scene_setup(script, state, update_direction, update_characters,
                               update_visual_prompt, update_video_prompt,
                               selected_method, ai_pick_models, selected_veo_variant)


def _run_ai_scene_setup(script: Script, state: MovieModeState,
                        update_direction: bool, update_characters: bool,
                        update_visual_prompt: bool, update_video_prompt: bool,
                        gen_method: str = "veo3", ai_pick_models: bool = False,
                        veo_variant: Optional[str] = None) -> None:
    """Run AI scene setup for all scenes in parallel."""
    import json
    import time as time_module
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not script.scenes:
        st.warning("No scenes to set up.")
        return

    visual_style = state.config.visual_style if state.config else "photorealistic, cinematic lighting"
    world_description = script.world_description or ""

    # Build character context with full details
    char_context = []
    for char in script.characters:
        char_info = f"- {char.id}: {char.name}"
        if char.description:
            char_info += f"\n  Visual: {char.description}"
        if char.personality:
            char_info += f"\n  Personality: {char.personality}"
        has_portrait = char.reference_image_path and Path(char.reference_image_path).exists()
        char_info += f"\n  Has reference portrait: {'Yes' if has_portrait else 'No'}"
        char_context.append(char_info)

    # Get AI model from session state
    ai_model = st.session_state.get("scene_ai_model", "claude-haiku-4-5")

    # Use status container for better visibility
    with st.status(f"Setting up {len(script.scenes)} scenes with AI (parallel)...", expanded=True) as status:
        try:
            from anthropic import Anthropic
            from src.config import config

            if not config.anthropic_api_key:
                st.error("ANTHROPIC_API_KEY not set. Please add it to your environment.")
                return

            client = Anthropic(api_key=config.anthropic_api_key)
            st.write(f"Connected to Claude API ({ai_model})...")

            def process_scene(scene):
                """Process a single scene - runs in thread."""
                # Build dialogue context
                dialogue_lines = []
                speaking_chars = set()
                for d in scene.dialogue:
                    char = script.get_character(d.character_id)
                    char_name = char.name if char else d.character_id
                    dialogue_lines.append(f"{char_name}: \"{d.text}\"")
                    speaking_chars.add(d.character_id)

                # Collect ALL visible characters (speaking + already marked visible)
                all_visible_chars = set(speaking_chars)
                if scene.direction.visible_characters:
                    all_visible_chars.update(scene.direction.visible_characters)

                # Build model-specific video prompt guidance and model selection rules
                video_prompt_guidance = ""
                video_prompt_example = ""
                model_selection_guidance = ""
                generation_model_json = ""

                # When AI picks models, provide guidance for all model types
                if ai_pick_models:
                    model_selection_guidance = """
=== VIDEO MODEL SELECTION (choose the best for each scene) ===
Available models and when to use them:

**veo3** (Text-to-Video with Audio) - BEST FOR:
- Scenes with important dialogue that needs natural speech
- Scenes with ambient audio (wind, crowds, music)
- Establishing shots with atmospheric sounds
- Any scene where audio quality matters most

**wan26_i2v** (Image-to-Video) - BEST FOR:
- Character-focused scenes where appearance must match exactly
- Scenes requiring visual consistency with previous scenes
- Complex multi-character compositions
- When you have a perfect reference image to animate

**wan26_t2v** (Text-to-Video) - BEST FOR:
- Dynamic action scenes with lots of movement
- Abstract or stylized visuals
- When no reference image exists yet
- Rapid scene transitions

**seedance15_i2v** (Image-to-Video with Lip-sync) - BEST FOR:
- Close-up dialogue scenes requiring precise lip sync
- Interview or talking-head style scenes
- Emotional performances where facial expressions are key
- Any scene where lips MUST match the dialogue

**seedance15_t2v** (Text-to-Video) - BEST FOR:
- General scenes without precise lip-sync needs
- Wide shots where lip sync isn't visible
- Non-dialogue scenes with music/narration

DECISION LOGIC:
1. Does the scene have dialogue AND close-up faces? → seedance15_i2v
2. Does the scene have dialogue AND need natural speech audio? → veo3
3. Is character appearance critical AND have reference image? → wan26_i2v
4. Is it an action scene without dialogue focus? → wan26_t2v
5. Default fallback: Use the project default ({gen_method})
"""
                    video_prompt_guidance = """VIDEO PROMPT FORMAT (adapt based on chosen model):
- For veo3: Include dialogue in quotes with emotion cues, add ambient audio descriptions
- For seedance: Camera movement, facial expressions, exact dialogue in quotes for lip-sync
- For wan26: Shot breakdowns, character actions, movements, transitions"""
                    video_prompt_example = "Medium shot of Sarah on the couch. She leans forward with interest and says 'I never expected this to happen.' Camera slowly pushes in as her expression shifts from surprise to understanding."
                    generation_model_json = '"generation_model": "veo3",'

                else:
                    # Single model mode - format prompts for that specific model
                    if "veo" in gen_method.lower():
                        video_prompt_guidance = """VIDEO PROMPT FORMAT (Veo 3):
- Include character descriptions with actions
- Put dialogue in quotes with emotion cues: Character (emotion) says "dialogue text"
- Include ambient audio cues if appropriate"""
                        video_prompt_example = "Sarah (curious) leans forward on the couch and says 'I never expected this to happen.' Her expression shifts from surprise to understanding."
                    elif "seedance" in gen_method.lower():
                        video_prompt_guidance = """VIDEO PROMPT FORMAT (Seedance 1.5 - Lip-sync focus):
- Camera: Specify camera movement (dolly in, pan, tracking shot, static)
- Expression: Describe facial expressions and emotions
- Include exact dialogue in quotes for lip-sync generation
- Lighting and mood transitions if any"""
                        video_prompt_example = "Camera: Medium shot with slow dolly in. Sarah sits on couch, expression shifting from curiosity to understanding. She says 'I never expected this to happen.' Warm ambient lighting."
                    elif "wan" in gen_method.lower():
                        video_prompt_guidance = """VIDEO PROMPT FORMAT (Wan 2.6 - Multi-shot):
- Can include shot breakdowns: Shot 1 [time]: description
- Include character actions and movements
- Include dialogue in quotes
- Describe camera movements and transitions"""
                        video_prompt_example = "Medium shot of Sarah on the couch in warm-lit living room. She leans forward with interest and says 'I never expected this to happen.' Camera slowly pushes in as her expression shifts from surprise to understanding."
                    else:
                        video_prompt_guidance = """VIDEO PROMPT FORMAT:
- Describe the action and movement
- Include dialogue in quotes
- Specify camera angles and movements"""
                        video_prompt_example = "Sarah sits on the couch and says 'I never expected this to happen.' She leans forward with interest, her expression shifting from surprise to understanding."

                # Build the JSON schema based on whether AI picks models
                if ai_pick_models:
                    json_schema = f"""{{
  "title": "The Meeting",
  "setting": "Modern tech startup office with glass walls and standing desks",
  "camera": "medium shot",
  "lighting": "warm golden hour",
  "mood": "romantic",
  "visible_characters": ["char_id_1", "char_id_2"],
  "generation_model": "veo3",
  "visual_prompt": "Medium shot of a cozy modern apartment living room with large windows and beige sofa. Sarah (young woman with auburn hair, green eyes, wearing casual blue sweater) sits on the couch looking thoughtful. John (tall man in his 30s, short dark hair, wearing a gray button-up shirt) stands nearby. Warm golden hour lighting through the windows, romantic atmosphere.",
  "video_prompt": "{video_prompt_example}"
}}"""
                    model_field_desc = "9. generation_model: One of 'veo3', 'wan26_i2v', 'wan26_t2v', 'seedance15_i2v', 'seedance15_t2v' - pick the BEST model for this scene"
                else:
                    json_schema = f"""{{
  "title": "The Meeting",
  "setting": "Modern tech startup office with glass walls and standing desks",
  "camera": "medium shot",
  "lighting": "warm golden hour",
  "mood": "romantic",
  "visible_characters": ["char_id_1", "char_id_2"],
  "visual_prompt": "Medium shot of a cozy modern apartment living room with large windows and beige sofa. Sarah (young woman with auburn hair, green eyes, wearing casual blue sweater) sits on the couch looking thoughtful. John (tall man in his 30s, short dark hair, wearing a gray button-up shirt) stands nearby. Warm golden hour lighting through the windows, romantic atmosphere.",
  "video_prompt": "{video_prompt_example}"
}}"""
                    model_field_desc = ""

                system_prompt = f"""You are a professional film director and cinematographer setting up scenes for video production.

=== PRODUCTION STYLE ===
Visual Style: {visual_style}
World/Environment: {world_description if world_description else 'Not specified - use setting details from each scene'}
Default Video Generation Method: {gen_method}
{model_selection_guidance}
=== CHARACTERS (include their FULL visual descriptions in every image prompt) ===
{chr(10).join(char_context)}

=== CRITICAL RULES ===
- NEVER use vague references like "same", "same as before", "continues", etc.
- Always provide COMPLETE, EXPLICIT descriptions that stand alone
- ANALYZE the world/environment description above - if it mentions a character by name (e.g., "the device sits alone"), that character MUST be in visible_characters
- EVERY CHARACTER WHO SPEAKS IN THE DIALOGUE MUST APPEAR IN visible_characters AND visual_prompt
- If dialogue references or interacts with another character (e.g., looking at "the device"), add that character to visible_characters
- If there are 3 characters speaking, the visual_prompt MUST describe all 3 characters
- Include the visual style "{visual_style}" in every visual_prompt
- Include world/environment style in prompts when appropriate
- Include specific details: location name, time of day, furniture, objects, atmosphere
- NEVER say a character "sits alone" or is "by themselves" if multiple characters are in the scene

{video_prompt_guidance}

For each scene, provide ALL of the following:
1. title: Short descriptive title (2-5 words)
2. setting: Specific location with details (e.g., "Modern tech startup office with glass walls and standing desks")
3. camera: Camera angle (wide shot, medium shot, close-up, extreme close-up, over-the-shoulder, POV, establishing shot, two-shot)
4. lighting: Lighting style (warm, cool, dramatic, soft, harsh, natural, dim, bright, etc.)
5. mood: Scene mood (neutral, tense, happy, sad, mysterious, romantic, action, comedic)
6. visible_characters: List of character IDs who appear in the scene
7. visual_prompt: MUST include camera, setting, lighting, mood, AND full character descriptions for EACH visible character (copy their descriptions from the character list above). NO dialogue/text. Example format: "Medium shot of [setting]. [Character1 name] ([full description]) [action]. [Character2 name] ([full description]) [action]. [Lighting] lighting, [mood] atmosphere."
8. video_prompt: Animation prompt with action AND dialogue - MUST include all dialogue from the scene in quotes
{model_field_desc}

Respond ONLY with valid JSON in this exact format:
{json_schema}"""

                # Get full descriptions of ALL visible characters (speaking + already visible)
                visible_char_descriptions = []
                for char_id in all_visible_chars:
                    char = script.get_character(char_id)
                    if char:
                        speaks = "(SPEAKS)" if char_id in speaking_chars else "(silent/prop)"
                        visible_char_descriptions.append(f"- {char.name} ({char.id}) {speaks}: {char.description}")

                # List all available characters from the script
                all_script_chars = []
                for char in script.characters:
                    all_script_chars.append(f"- {char.name} ({char.id}): {char.description[:100]}...")

                user_prompt = f"""Set up this scene:

Scene {scene.index}: {scene.title or 'Untitled'}
Current setting: {scene.direction.setting}

=== DIALOGUE ===
{chr(10).join(dialogue_lines) if dialogue_lines else 'No dialogue'}

=== ALL AVAILABLE CHARACTERS IN THIS STORY ===
{chr(10).join(all_script_chars)}

=== CURRENTLY MARKED VISIBLE ({len(all_visible_chars)} total) ===
{chr(10).join(visible_char_descriptions) if visible_char_descriptions else 'None - YOU MUST DETERMINE which characters should be visible'}

REQUIREMENTS:
1. ANALYZE the scene setting and world description - if a character is mentioned (like "the device"), ADD them to visible_characters
2. ANALYZE the dialogue - if characters interact with or reference another character, ADD them to visible_characters
3. visible_characters MUST include: speaking characters + characters referenced in setting/dialogue
4. visual_prompt MUST describe ALL visible characters with their full descriptions
5. video_prompt MUST include ALL dialogue in quotes
6. DO NOT say any character is "alone" if other characters should logically be present"""

                # Set max_tokens based on model
                if "haiku" in ai_model:
                    max_tokens = 8192  # Haiku 4.5 max
                else:
                    max_tokens = 16384  # Sonnet 4.5 max

                response = client.messages.create(
                    model=ai_model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )

                # Parse the response
                response_text = response.content[0].text.strip()
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]

                setup_data = json.loads(response_text)

                # Debug: Log what we got for video_prompt
                vp = setup_data.get('video_prompt', None)
                logger.info(f"Scene {scene.index} video_prompt received: {vp[:100] if vp else 'NONE'}...")

                return scene.index, setup_data, all_visible_chars

            # Launch all API calls in parallel
            st.write(f"Launching {len(script.scenes)} parallel API calls...")
            status.update(label=f"Processing {len(script.scenes)} scenes in parallel...")

            results = {}
            errors = []

            with ThreadPoolExecutor(max_workers=min(10, len(script.scenes))) as executor:
                future_to_scene = {executor.submit(process_scene, scene): scene for scene in script.scenes}

                for future in as_completed(future_to_scene):
                    scene = future_to_scene[future]
                    try:
                        scene_idx, setup_data, speaking_chars = future.result()
                        results[scene_idx] = (setup_data, speaking_chars)
                        # Show what was received
                        has_visual = bool(setup_data.get('visual_prompt', '').strip())
                        has_video = bool(setup_data.get('video_prompt', '').strip())
                        prompts_status = f"{'✅' if has_visual else '❌'}img {'✅' if has_video else '❌'}vid"
                        # Debug: show video prompt preview
                        if has_video:
                            vp_preview = setup_data.get('video_prompt', '')[:60] + "..."
                            st.caption(f"   Video prompt: {vp_preview}")
                        else:
                            st.caption(f"   ⚠️ No video_prompt in response. Keys: {list(setup_data.keys())}")
                        # Show AI-selected model if present
                        model_info = ""
                        if setup_data.get('generation_model'):
                            model_info = f" | 🎬 {setup_data['generation_model']}"
                        st.write(f"✓ Scene {scene_idx}: {setup_data.get('camera', 'N/A')}, {setup_data.get('mood', 'N/A')} | {prompts_status}{model_info}")
                    except Exception as e:
                        errors.append((scene.index, str(e)))
                        st.write(f"✗ Scene {scene.index}: {e}")

            # Apply results to scenes
            scenes_updated = 0
            for scene in script.scenes:
                if scene.index in results:
                    setup_data, visible_chars = results[scene.index]

                    if update_direction:
                        if "title" in setup_data:
                            scene.title = setup_data["title"]
                        if "setting" in setup_data:
                            scene.direction.setting = setup_data["setting"]
                        if "camera" in setup_data:
                            scene.direction.camera = setup_data["camera"]
                        if "lighting" in setup_data:
                            scene.direction.lighting = setup_data["lighting"]
                        if "mood" in setup_data:
                            scene.direction.mood = setup_data["mood"]

                    if update_characters and "visible_characters" in setup_data:
                        valid_chars = [c for c in setup_data["visible_characters"]
                                      if any(ch.id == c for ch in script.characters)]
                        # Ensure all visible chars (speaking + silent) are included
                        for char_id in visible_chars:
                            if char_id not in valid_chars:
                                valid_chars.append(char_id)
                        scene.direction.visible_characters = valid_chars

                    if update_visual_prompt:
                        visual_prompt_value = setup_data.get("visual_prompt", "")
                        if visual_prompt_value and visual_prompt_value.strip():
                            try:
                                scene.visual_prompt = visual_prompt_value
                            except Exception:
                                object.__setattr__(scene, 'visual_prompt', visual_prompt_value)
                            logger.info(f"Scene {scene.index} visual_prompt set: {visual_prompt_value[:50]}...")
                        else:
                            logger.warning(f"Scene {scene.index}: No visual_prompt in AI response")

                    if update_video_prompt:
                        video_prompt_value = setup_data.get("video_prompt", "")
                        if video_prompt_value and video_prompt_value.strip():
                            try:
                                scene.video_prompt = video_prompt_value
                                logger.info(f"Scene {scene.index} video_prompt set via property: {video_prompt_value[:50]}...")
                            except Exception as e:
                                logger.warning(f"Scene {scene.index} video_prompt property failed: {e}, trying __setattr__")
                                object.__setattr__(scene, 'video_prompt', video_prompt_value)
                            # Verify it was set
                            actual_value = getattr(scene, 'video_prompt', None)
                            if actual_value:
                                logger.info(f"Scene {scene.index} video_prompt VERIFIED: {actual_value[:50]}...")
                            else:
                                logger.error(f"Scene {scene.index} video_prompt NOT SET despite no exception!")
                        else:
                            logger.warning(f"Scene {scene.index}: No video_prompt in AI response")

                    # Apply AI-selected generation model if provided
                    if "generation_model" in setup_data:
                        model_value = setup_data["generation_model"]
                        valid_models = ["veo3", "wan26_i2v", "wan26_t2v", "seedance15_i2v", "seedance15_t2v", "tts_images"]
                        if model_value in valid_models:
                            try:
                                scene.generation_model = model_value
                            except Exception:
                                object.__setattr__(scene, 'generation_model', model_value)
                            logger.info(f"Scene {scene.index} generation_model set: {model_value}")
                        else:
                            logger.warning(f"Scene {scene.index}: Invalid generation_model '{model_value}', using default")

                    # Apply Veo variant if selected and using Veo
                    if veo_variant and (not "generation_model" in setup_data or setup_data.get("generation_model", "").startswith("veo") or gen_method == "veo3"):
                        try:
                            scene.veo_model_variant = veo_variant
                        except Exception:
                            object.__setattr__(scene, 'veo_model_variant', veo_variant)
                        logger.info(f"Scene {scene.index} veo_model_variant set: {veo_variant}")

                    scenes_updated += 1

            # Save after all scenes processed
            save_movie_state()

            if errors:
                status.update(label=f"Completed {scenes_updated}/{len(script.scenes)} scenes ({len(errors)} errors)", state="complete")
            else:
                status.update(label=f"Completed all {scenes_updated} scenes!", state="complete")

        except Exception as e:
            status.update(label=f"Error: {e}", state="error")
            st.error(f"Error setting up scenes: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    # Brief pause so user sees success
    time_module.sleep(1)
    st.session_state.show_prompt_generator = False
    st.session_state.scenes_dirty = False
    st.rerun()


def _regenerate_scene_prompt(
    scene,
    script: Script,
    state: MovieModeState,
    prompt_type: str = "video",  # "visual" or "video"
    use_image_context: bool = True,
) -> Optional[str]:
    """Regenerate a single prompt for a scene using AI.

    Args:
        scene: The scene to regenerate prompts for
        script: The full script
        state: MovieModeState
        prompt_type: "visual" for image prompt, "video" for animation prompt
        use_image_context: If True and scene has an image, analyze it for context

    Returns:
        The generated prompt, or None if failed
    """
    import json

    visual_style = state.config.visual_style if state.config else "photorealistic, cinematic lighting"

    # Use per-scene generation model if set, otherwise fall back to global config
    scene_gen_model = getattr(scene, 'generation_model', None)
    is_seedance_fast = False  # Track if using Seedance Fast vs Pro
    is_wan_t2v = False  # Track if using WAN Text-to-Video (no input image)
    is_wan_fast = False  # Track if using WAN 2.5 Fast
    if scene_gen_model:
        # Map scene model to gen_method format
        if scene_gen_model == "veo3":
            gen_method = "veo3"
        elif "wan" in scene_gen_model.lower():
            gen_method = "wan26"  # Triggers WAN-specific prompts
            is_wan_t2v = "t2v" in scene_gen_model.lower()
            is_wan_fast = "fast" in scene_gen_model.lower()
        elif "seedance" in scene_gen_model.lower():
            gen_method = "seedance"  # Triggers Seedance-specific prompts
            is_seedance_fast = "fast" in scene_gen_model.lower()
        else:
            gen_method = scene_gen_model
        logger.info(f"Using per-scene model for prompt generation: {scene_gen_model} -> {gen_method} (is_seedance_fast={is_seedance_fast}, is_wan_t2v={is_wan_t2v}, is_wan_fast={is_wan_fast})")
        # Show detected model to user
        if "seedance" in gen_method:
            model_display = "Seedance Fast" if is_seedance_fast else "Seedance 1.5 Pro"
        elif "wan" in gen_method:
            model_display = "WAN 2.6 T2V" if is_wan_t2v else ("WAN 2.5 Fast" if is_wan_fast else "WAN 2.6 I2V")
        else:
            model_display = gen_method
        st.toast(f"🎬 Generating prompt for: {model_display}", icon="🎯")
    else:
        gen_method = state.config.generation_method if state.config else "veo3"
        if gen_method and "seedance" in gen_method.lower():
            is_seedance_fast = "fast" in gen_method.lower()
        if gen_method and "wan" in gen_method.lower():
            is_wan_t2v = "t2v" in gen_method.lower()
            is_wan_fast = "fast" in gen_method.lower()
        logger.info(f"Using global config for prompt generation: {gen_method} (is_seedance_fast={is_seedance_fast}, is_wan_t2v={is_wan_t2v}, is_wan_fast={is_wan_fast})")
        # Show detected model to user
        if gen_method and "seedance" in gen_method.lower():
            model_display = "Seedance Fast" if is_seedance_fast else "Seedance 1.5 Pro"
        elif gen_method and "wan" in gen_method.lower():
            model_display = "WAN 2.6 T2V" if is_wan_t2v else ("WAN 2.5 Fast" if is_wan_fast else "WAN 2.6 I2V")
        else:
            model_display = gen_method if gen_method else "default"
        st.toast(f"🎬 Generating prompt for: {model_display} (project default)", icon="📁")

    # Build dialogue context
    dialogue_lines = []
    for d in scene.dialogue:
        char = script.get_character(d.character_id)
        char_name = char.name if char else d.character_id
        dialogue_lines.append(f'{char_name} ({d.emotion.value}): "{d.text}"')

    # Build character context
    char_descriptions = []
    for char_id in scene.direction.visible_characters:
        char = script.get_character(char_id)
        if char:
            char_descriptions.append(f"- {char.name}: {char.description}")

    # Image context (if available and requested)
    image_context = ""
    if use_image_context and scene.image_path and Path(scene.image_path).exists():
        # Analyze the image using Gemini
        try:
            from google import genai
            from src.config import config

            if config.google_api_key:
                client = genai.Client(api_key=config.google_api_key)
                with open(scene.image_path, "rb") as f:
                    image_data = f.read()

                from google.genai import types
                image = types.Part.from_bytes(data=image_data, mime_type="image/png")

                analysis_prompt = """Analyze this scene image and describe:
1. Character positions and poses
2. What actions or movements would naturally follow
3. The mood and atmosphere
4. Any objects or elements that should be considered for animation

Be specific and concise. Focus on actionable details for video animation."""

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[image, analysis_prompt],
                )
                image_context = f"\n\n=== SCENE IMAGE ANALYSIS ===\n{response.text}\n\nUse this context to make the animation natural and continuous with the image."
                logger.info(f"Scene {scene.index} image analysis: {response.text[:100]}...")
        except Exception as e:
            logger.warning(f"Failed to analyze scene image: {e}")
            image_context = ""

    # Camera continuity context (for video prompts with cinematography plan)
    camera_continuity_context = ""
    if prompt_type == "video" and hasattr(state, 'cinematography_plan') and state.cinematography_plan:
        try:
            from src.services.cinematography import build_camera_continuity_context

            # Get current scene's camera context from plan
            current_ctx = next(
                (c for c in state.cinematography_plan.scenes if c.scene_index == scene.index),
                None
            )

            # Get previous scene and its camera context
            prev_scene = next((s for s in script.scenes if s.index == scene.index - 1), None)
            prev_ctx = next(
                (c for c in state.cinematography_plan.scenes if c.scene_index == scene.index - 1),
                None
            ) if prev_scene else None

            # Get next scene for transition awareness
            next_scene = next((s for s in script.scenes if s.index == scene.index + 1), None)

            # Get camera override if set by user
            camera_override = getattr(scene, 'camera_override', None)

            if current_ctx:
                camera_continuity_context = build_camera_continuity_context(
                    current_scene=scene,
                    previous_scene=prev_scene,
                    next_scene=next_scene,
                    camera_context=current_ctx,
                    previous_camera_context=prev_ctx,
                    camera_override=camera_override,
                    is_multi_shot=True,  # Default to multi-shot for cinematic feel
                )
                logger.info(f"Scene {scene.index} camera context: {current_ctx.camera_angle.value} ({current_ctx.narrative_role})")
        except Exception as e:
            logger.warning(f"Failed to build camera continuity context: {e}")
            camera_continuity_context = ""

    try:
        from anthropic import Anthropic
        from src.config import config

        if not config.anthropic_api_key:
            st.error("ANTHROPIC_API_KEY not set")
            return None

        client = Anthropic(api_key=config.anthropic_api_key)

        if prompt_type == "visual":
            # Detect if photorealistic style
            is_photorealistic = any(kw in visual_style.lower() for kw in ["photorealistic", "realistic", "photo", "cinematic"])

            if is_photorealistic:
                # Optimized for Gemini Native Image Generation (Nano Banana Pro) - photorealistic
                system_prompt = f"""<role>
You are an expert prompt engineer for Gemini Native Image Generation, specializing in photorealistic imagery.
</role>

<output_format>
Generate a single image prompt optimized for photorealistic output. No explanations, just the prompt.
</output_format>

<photorealistic_keywords>
ALWAYS include these for photorealistic results:
- Skin: "natural skin texture", "skin pores visible", "subsurface scattering"
- Lighting: "natural lighting", "soft diffused light", "golden hour", "volumetric lighting"
- Camera: "shot on Sony A7IV", "85mm f/1.4 lens", "shallow depth of field", "bokeh"
- Quality: "8K resolution", "RAW photo", "film grain", "photojournalistic"
- Realism: "hyperrealistic", "lifelike", "unedited photograph", "candid moment"
</photorealistic_keywords>

<constraints>
- NO text, watermarks, or UI elements in the image
- NO CGI, 3D render, digital art, or illustrated look
- NO plastic skin, airbrushed, or uncanny valley
- Characters must look like real humans photographed in real locations
</constraints>

<example>
INPUT: A woman sitting at a cafe
OUTPUT: RAW photograph, candid moment of a woman sitting at a Parisian sidewalk cafe, natural skin texture with visible pores, subsurface scattering on cheeks, soft diffused afternoon light, shot on Sony A7IV with 85mm f/1.4 lens, shallow depth of field, creamy bokeh in background, 8K resolution, film grain, hyperrealistic, photojournalistic style, golden hour warm tones, unedited photograph aesthetic
</example>"""

                user_prompt = f"""<scene>
Title: Scene {scene.index} - {scene.title or scene.direction.setting}
Setting: {scene.direction.setting}
Camera: {scene.direction.camera}
Lighting: {scene.direction.lighting or 'natural lighting'}
Mood: {scene.direction.mood}
</scene>

<characters>
{chr(10).join(char_descriptions) if char_descriptions else 'No specific characters'}
</characters>

<task>
Generate a PHOTOREALISTIC image prompt following the example format. Include:
1. "RAW photograph" or "unedited photo" at the start
2. Natural skin texture keywords (pores, subsurface scattering)
3. Specific camera/lens (Sony A7IV, 85mm f/1.4)
4. Lighting description (soft diffused, golden hour, etc.)
5. Quality markers (8K, film grain, hyperrealistic)
6. All character descriptions exactly as provided
</task>"""

            else:
                # Non-photorealistic styles (anime, 3D, illustrated, etc.)
                system_prompt = f"""<role>
You are an expert prompt engineer for Gemini Native Image Generation.
</role>

<style>
Required visual style: {visual_style}
</style>

<output_format>
Generate a single image prompt. No explanations, just the prompt text.
</output_format>

<constraints>
- START with the visual style
- Include camera angle and shot type
- Include setting with specific details
- Include ALL character descriptions exactly as provided
- Include lighting and atmosphere
- NO text, dialogue, or UI elements in the image
</constraints>"""

                user_prompt = f"""<scene>
Title: Scene {scene.index} - {scene.title or scene.direction.setting}
Setting: {scene.direction.setting}
Camera: {scene.direction.camera}
Lighting: {scene.direction.lighting or 'cinematic lighting'}
Mood: {scene.direction.mood}
</scene>

<characters>
{chr(10).join(char_descriptions) if char_descriptions else 'No specific characters'}
</characters>

<task>
Generate an image prompt in {visual_style} style. Start with the style, then describe the scene with all characters.
</task>"""

        else:  # video prompt
            # Determine video model format
            if gen_method == "veo3":
                format_guidance = """Format for Veo 3 (text-to-video with dialogue):
- Include dialogue in quotes with character names
- Describe character actions and movements
- Include emotion cues in parentheses
- Example: 'Sarah (curious) leans forward and says "I never expected this." Her expression shifts to surprise.'"""
            elif "seedance" in gen_method:
                seedance_model_name = "Seedance Fast" if is_seedance_fast else "Seedance 1.5 Pro"
                format_guidance = f"""Format for {seedance_model_name} (image-to-video with lip-sync) - MULTI-SHOT CINEMATIC:

CRITICAL I2V RULES:
1. PHOTOREALISTIC - output must match source image exactly, NO CGI look
2. NO STYLE DESCRIPTION - style is in the image
3. MAX 50 WORDS total - shorter = more faithful
4. Reference "the person in the image"
5. Focus on: lip movements, subtle expressions, natural dialogue delivery
6. Dialogue format: speaks: "line"
7. MULTI-SHOT CAMERA: Include cinematic variety - "camera work transitions from wide to medium to close-up"
8. ALWAYS END WITH: "Photorealistic. Preserve exact appearance."

GOOD EXAMPLE:
"The person in the image speaks: 'I've been waiting for this moment.' Subtle expression shifts. Camera work: wide establishing shot, transitions to medium shot, pushes in to close-up on emotional beats. Photorealistic. Preserve exact appearance."
"""
            else:  # wan26 (I2V, T2V, or Fast)
                # Determine WAN model variant name
                if is_wan_t2v:
                    wan_model_name = "WAN 2.6 T2V (Text-to-Video)"
                elif is_wan_fast:
                    wan_model_name = "WAN 2.5 Fast"
                else:
                    wan_model_name = "WAN 2.6 I2V (Image-to-Video)"

                # Different rules for T2V (no input image) vs I2V (has input image)
                if is_wan_t2v:
                    format_guidance = f"""Format for {wan_model_name} - MULTI-SHOT CINEMATIC:

WAN 2.6 T2V generates video from TEXT ONLY (no input image).
You MUST fully describe characters, setting, and appearance in the prompt.

PROMPT STRUCTURE (shot breakdown format):
Wide establishing shot: [setting description, character appearance, positioning]
Medium shot: [character interaction, dialogue, body language]
Close-up: [emotional expressions, reactions, facial details]

RULES:
1. DESCRIBE CHARACTERS FULLY: Include appearance, clothing, hair, build
2. DIALOGUE: Include ALL lines using: [Character] speaks: "line"
3. MOTION: Use concrete verbs with speed adverbs (gently sways, slowly tilts)
4. MULTI-SHOT: Provide descriptions for wide/medium/close-up angles
5. STYLE: "Cinematic, photorealistic, natural skin texture"
6. Keep total under 120 words

EXAMPLE PROMPT:
"Wide shot: Marcus, a 30-year-old man with short dark hair and stubble, wearing a navy sweater, sits in a warm-lit living room.
Medium shot: Marcus speaks: 'I've been waiting for this moment.' Natural body language, gentle eye movement.
Close-up: His expression shifts from anticipation to understanding, subtle facial movements.
Cinematic, photorealistic, natural skin texture."
"""
                else:
                    format_guidance = f"""Format for {wan_model_name} - MULTI-SHOT CINEMATIC:

WAN 2.6 I2V uses MULTI-SHOT mode by default, generating varied camera angles for cinematic storytelling.
The input image shows the character - reference them as "the person in the image".

PROMPT STRUCTURE (shot breakdown format):
Wide establishing shot: [main action/setting description]
Medium shot: [character interaction, dialogue, body language]
Close-up: [emotional expressions, reactions, facial movements]

RULES:
1. DIALOGUE: Include ALL lines using: [Character] speaks: "line"
2. MOTION: Use concrete verbs with speed adverbs (gently sways, slowly tilts)
3. MULTI-SHOT: Provide descriptions for wide/medium/close-up angles
4. STYLE: "Cinematic, photorealistic, natural skin texture"
5. END WITH: "Preserve identity, maintain appearance"
6. Keep total under 100 words

EXAMPLE PROMPT:
"Wide shot: The person in the image in a warm-lit living room, soft key light.
Medium shot: They speak: 'I've been waiting for this moment.' Natural body language, gentle eye movement.
Close-up: Expression shifts from anticipation to understanding, subtle facial movements.
Cinematic, photorealistic, natural skin texture. Preserve identity, maintain appearance."
"""

            # Different system prompts for different video models
            if gen_method == "veo3":
                # Veo 3 can handle longer narrative prompts with dialogue
                system_prompt = f"""You are a professional director creating video animation prompts for Veo 3.

REQUIRED Visual Style: {visual_style}

{format_guidance}
{image_context}
{camera_continuity_context}

RULES:
1. START with: "VISUAL STYLE: {visual_style}."
2. Include ALL dialogue in quotes with character names and emotions
3. Describe character actions, expressions, and movements
4. Characters are already positioned in scene - describe how they move
5. FOLLOW the camera continuity guidance above for professional cinematography

Respond with ONLY the video prompt text, no JSON or explanations."""
            else:
                # WAN 2.6 and Seedance - MULTI-SHOT cinematic prompts
                if "seedance" in gen_method:
                    model_name = "Seedance Fast" if is_seedance_fast else "Seedance 1.5 Pro"
                    prompt_type_desc = "image-to-video"
                else:
                    # WAN variant names
                    if is_wan_t2v:
                        model_name = "WAN 2.6 T2V"
                        prompt_type_desc = "text-to-video"
                    elif is_wan_fast:
                        model_name = "WAN 2.5 Fast"
                        prompt_type_desc = "image-to-video"
                    else:
                        model_name = "WAN 2.6 I2V"
                        prompt_type_desc = "image-to-video"

                # Adjust rules based on T2V vs I2V
                if is_wan_t2v:
                    identity_rule = "DESCRIBE CHARACTERS: Include full physical description (appearance, clothing, hair)"
                    max_words = 120
                else:
                    identity_rule = "END WITH: \"Preserve identity, maintain appearance\""
                    max_words = 100

                system_prompt = f"""You create MULTI-SHOT CINEMATIC {prompt_type_desc} prompts for {model_name}. OUTPUT MUST BE PHOTOREALISTIC.

{format_guidance}
{image_context}
{camera_continuity_context}

MULTI-SHOT MODE (DEFAULT):
The video will use multiple camera angles for cinematic variety.
FOLLOW the camera continuity guidance above for scene-to-scene flow.

YOUR PROMPT SHOULD:
1. DIALOGUE (CRITICAL): Include ALL dialogue lines using format: [Character] speaks: "exact line"
2. SHOT BREAKDOWN: Start with the planned camera angle, progress through the shot sequence
3. MOTION: Concrete verbs + speed adverbs (gently sways, slowly tilts)
4. {identity_rule}
5. CAMERA CONTINUITY: Follow the planned camera angle and transitions for professional cinematography

RULES:
- If dialogue provided, include ALL lines - do not skip!
- Use shot breakdown format: "Wide shot: ... Medium shot: ... Close-up: ..."
- MAX {max_words} words - dialogue takes priority

Output ONLY the prompt."""

            # Different user prompts for different models
            if gen_method == "veo3":
                user_prompt = f"""Scene {scene.index}: {scene.title or scene.direction.setting}

Setting: {scene.direction.setting}
Camera: {scene.direction.camera}
Mood: {scene.direction.mood}

Characters:
{chr(10).join(char_descriptions) if char_descriptions else 'No characters'}

Dialogue (MUST be included in quotes):
{chr(10).join(dialogue_lines) if dialogue_lines else 'No dialogue'}

Generate a video animation prompt that:
1. STARTS with "VISUAL STYLE: {visual_style}."
2. Includes ALL dialogue in quotes with emotions
3. Describes character movements and expressions
4. Flows naturally from the scene image"""
            else:
                # WAN 2.6 / Seedance - MULTI-SHOT cinematic prompts with all elements
                # Include ALL dialogue lines for the scene
                all_dialogue = dialogue_lines if dialogue_lines else []
                dialogue_text = "\n".join(all_dialogue) if all_dialogue else None
                if "seedance" in gen_method:
                    model_label = "Seedance Fast" if is_seedance_fast else "Seedance 1.5 Pro"
                else:
                    # WAN variant labels
                    if is_wan_t2v:
                        model_label = "WAN 2.6 T2V"
                    elif is_wan_fast:
                        model_label = "WAN 2.5 Fast"
                    else:
                        model_label = "WAN 2.6 I2V"

                # Build user prompt with emphasis on dialogue and MULTI-SHOT format
                if dialogue_text:
                    user_prompt = f"""Create a MULTI-SHOT CINEMATIC I2V prompt for {model_label}. MAX 100 WORDS.

**DIALOGUE (MUST INCLUDE ALL LINES):**
{dialogue_text}

Scene mood: {scene.direction.mood}
Setting: {scene.direction.setting}

USE MULTI-SHOT FORMAT:
Wide shot: [establishing - setting, character position, lighting]
Medium shot: [dialogue, body language] - Include ALL dialogue here as: Character speaks: "line"
Close-up: [emotional reactions, facial expressions]

End with: "Cinematic, photorealistic. Preserve identity, maintain appearance."

CRITICAL: Include ALL dialogue lines verbatim using format: [Character] speaks: "line"

Output ONLY the prompt text."""
                else:
                    user_prompt = f"""Create a MULTI-SHOT CINEMATIC I2V prompt for {model_label}. MAX 80 WORDS.

Scene mood: {scene.direction.mood}
Setting: {scene.direction.setting}
(No dialogue in this scene)

USE MULTI-SHOT FORMAT:
Wide shot: [establishing - setting, atmosphere]
Medium shot: [character action, body language]
Close-up: [emotional expressions, subtle movements]

Include motion verbs (sway, tilt, blink) with speed adverbs (gently, slowly).
End with: "Cinematic, photorealistic. Preserve identity, maintain appearance."

Output ONLY the prompt text."""

        # Use configured model and max tokens from sidebar settings
        model_to_use = config.claude_model if hasattr(config, 'claude_model') else "claude-sonnet-4-5-20250929"
        max_tokens = config.claude_max_tokens if hasattr(config, 'claude_max_tokens') else 8192

        response = client.messages.create(
            model=model_to_use,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            system=system_prompt,
        )

        generated_prompt = response.content[0].text.strip()

        # Log the full generated prompt
        logger.info("=" * 60)
        logger.info(f"GENERATED {prompt_type.upper()} PROMPT (Scene {scene.index}):")
        logger.info("-" * 60)
        for line in generated_prompt.split('\n'):
            logger.info(line)
        logger.info("=" * 60)

        return generated_prompt

    except Exception as e:
        logger.error(f"Failed to regenerate {prompt_type} prompt: {e}")
        st.error(f"Failed to regenerate prompt: {e}")
        return None


def _generate_all_scene_images(script: Script, state: MovieModeState) -> None:
    """Generate images for all scenes that don't have one."""
    from src.services.movie_image_generator import MovieImageGenerator

    visual_style = state.config.visual_style if state.config else "photorealistic, cinematic lighting"
    generator = MovieImageGenerator(style=visual_style)
    output_dir = get_project_dir() / "scenes"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get image model settings from session state
    image_model = st.session_state.get("scene_image_model", "gemini-3-pro-image-preview")
    image_size = st.session_state.get("scene_image_size", "2K")
    aspect_ratio = st.session_state.get("scene_image_aspect_ratio", "16:9")

    # Get consistency options
    use_prev_image = st.session_state.get("use_prev_scene_image", True)

    scenes_to_generate = [s for s in script.scenes if not s.image_path or not Path(s.image_path).exists()]

    if not scenes_to_generate:
        st.info("All scenes already have images.")
        return

    consistency_info = "with previous scene refs" if use_prev_image else "without scene refs"
    progress = st.progress(0, text=f"Generating scene images with {image_model} ({consistency_info})...")

    for i, scene in enumerate(scenes_to_generate):
        progress.progress((i + 1) / len(scenes_to_generate), text=f"Generating Scene {scene.index} ({image_model}, {image_size}, {aspect_ratio})...")

        # Find style reference image (previous scene for continuity) - only if enabled
        reference_image = None
        if use_prev_image and scene.index > 1:
            prev_scene = next((s for s in script.scenes if s.index == scene.index - 1), None)
            if prev_scene and prev_scene.image_path and Path(prev_scene.image_path).exists():
                reference_image = Path(prev_scene.image_path)
                logger.info(f"Using previous scene image as reference: {reference_image.name}")

        # Collect ALL character portraits for visible characters
        character_portraits = []
        if scene.direction.visible_characters:
            for char_id in scene.direction.visible_characters:
                char = script.get_character(char_id)
                if char and char.reference_image_path and Path(char.reference_image_path).exists():
                    character_portraits.append(Path(char.reference_image_path))

        try:
            result = generator.generate_scene_image(
                scene=scene,
                script=script,
                output_dir=output_dir,
                reference_image=reference_image,
                character_portraits=character_portraits if character_portraits else None,
                model=image_model,
                image_size=image_size,
                aspect_ratio=aspect_ratio,
            )
            if result:
                scene.image_path = str(result)
        except Exception as e:
            st.warning(f"Failed to generate image for Scene {scene.index}: {e}")

    save_movie_state()
    st.session_state.scenes_dirty = False
    progress.empty()
    st.success(f"Generated {len(scenes_to_generate)} scene images!")


def _generate_single_scene_image(scene, script: Script, state: MovieModeState) -> None:
    """Generate image for a single scene, storing as variant."""
    from src.services.movie_image_generator import MovieImageGenerator
    import time

    visual_style = state.config.visual_style if state.config else "photorealistic, cinematic lighting"
    generator = MovieImageGenerator(style=visual_style)

    # Get image model settings - per-scene overrides global session state
    global_model = st.session_state.get("scene_image_model", "gemini-3-pro-image-preview")
    global_size = st.session_state.get("scene_image_size", "2K")
    global_ar = st.session_state.get("scene_image_aspect_ratio", "16:9")

    # Use per-scene settings if available
    image_model = getattr(scene, 'image_model', None) or global_model
    image_size = getattr(scene, 'image_size', None) or global_size
    aspect_ratio = getattr(scene, 'image_aspect_ratio', None) or global_ar

    # Get consistency options
    use_prev_image = st.session_state.get("use_prev_scene_image", True)

    # Create per-scene directory structure for variants
    scene_dir = get_project_dir() / "scenes" / f"scene_{scene.index:03d}" / "images"
    scene_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename for this variant
    variant_num = len(scene.image_variants) + 1
    timestamp = get_readable_timestamp()
    output_path = scene_dir / f"take_{variant_num:03d}_{timestamp}.png"

    # Find style reference image (previous scene for continuity) - only if enabled
    reference_image = None
    if use_prev_image and scene.index > 1:
        prev_scene = next((s for s in script.scenes if s.index == scene.index - 1), None)
        if prev_scene:
            prev_img = prev_scene.get_selected_image()
            if prev_img and prev_img.exists():
                reference_image = prev_img
                logger.info(f"Using previous scene image as reference: {prev_img.name}")

    # Collect ALL character portraits for visible characters
    character_portraits = []
    if scene.direction.visible_characters:
        for char_id in scene.direction.visible_characters:
            char = script.get_character(char_id)
            if char and char.reference_image_path and Path(char.reference_image_path).exists():
                character_portraits.append(Path(char.reference_image_path))

    with st.spinner(f"Generating image variant {variant_num} for Scene {scene.index} ({image_model}, {image_size}, {aspect_ratio})..."):
        try:
            # Pass the scenes directory (not scene_dir which already includes scene_XXX/images)
            scenes_dir = get_project_dir() / "scenes"
            result = generator.generate_scene_image(
                scene=scene,
                script=script,
                output_dir=scenes_dir,  # Generator will create scene_XXX/images/take_XXX.png
                reference_image=reference_image,
                character_portraits=character_portraits if character_portraits else None,
                model=image_model,
                image_size=image_size,
                aspect_ratio=aspect_ratio,
            )
            if result:
                # Add to variants and select it
                idx = scene.add_image_variant(str(result))
                scene.select_image_variant(idx)
                save_movie_state()
                st.toast(f"Generated image variant {variant_num} for Scene {scene.index}")
        except Exception as e:
            st.error(f"Failed: {e}")


def _render_scene_image_manager(scene, script: Script, state: MovieModeState, compact: bool = False) -> None:
    """Render scene image manager with variant carousel, model selection, and full-size viewing.

    Args:
        scene: The MovieScene to manage images for
        script: The full Script
        state: MovieModeState
        compact: If True, show minimal controls (for grid view). If False, show full manager.
    """
    import os

    has_image = scene.image_path and Path(scene.image_path).exists()

    # Collect all existing variants (from variants list + check directory)
    # Use resolved paths to avoid duplicates from different path formats
    existing_variants = []
    seen_paths = set()
    scene_dir = get_project_dir() / "scenes" / f"scene_{scene.index:03d}" / "images"

    # Helper to add variant if not already seen
    def add_variant(path: Path):
        resolved = str(path.resolve())
        if resolved not in seen_paths and path.exists():
            seen_paths.add(resolved)
            existing_variants.append(path)
            return True
        return False

    # First add the main scene image (so it's first)
    if has_image:
        add_variant(Path(scene.image_path))

    # Add variants from the scene's variant list
    for var_path in scene.image_variants:
        var_p = Path(var_path)
        if add_variant(var_p):
            pass  # Already added

    # Also check directory for any orphaned images not in variants list
    if scene_dir.exists():
        for img_file in scene_dir.glob("*.png"):
            if add_variant(img_file):
                # Add to variants list for future
                scene.add_image_variant(str(img_file))
        for img_file in scene_dir.glob("*.jpg"):
            if add_variant(img_file):
                scene.add_image_variant(str(img_file))

    # Check old format: scenes/scene_XXX/scene_XXX.png (direct children of scene folder)
    scene_folder = get_project_dir() / "scenes" / f"scene_{scene.index:03d}"
    if scene_folder.exists():
        for img_file in scene_folder.glob("*.png"):
            if img_file.parent == scene_folder:  # Only direct children, not from images/
                if add_variant(img_file):
                    scene.add_image_variant(str(img_file))
        for img_file in scene_folder.glob("*.jpg"):
            if img_file.parent == scene_folder:
                if add_variant(img_file):
                    scene.add_image_variant(str(img_file))

    num_variants = len(existing_variants)

    if compact:
        # Compact mode - minimal controls for grid view
        if has_image:
            # Show current image with click to expand
            st.image(str(scene.image_path), width="stretch")

            # Show variant count if multiple
            if num_variants > 1:
                st.caption(f"📷 {num_variants} variants")
        else:
            st.info("No image", icon="📷")

        # Compact action buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            icon = "🔄" if has_image else "🎨"
            if st.button(icon, key=f"img_gen_{scene.index}", use_container_width=True,
                        help="Generate new image variant"):
                _generate_single_scene_image(scene, script, state)
                st.rerun()
        with btn_col2:
            if num_variants > 1:
                if st.button("📂", key=f"img_browse_{scene.index}", use_container_width=True,
                            help=f"Browse {num_variants} variants"):
                    st.session_state[f"show_image_manager_{scene.index}"] = True
                    st.rerun()

        # Show expanded manager if requested
        if st.session_state.get(f"show_image_manager_{scene.index}", False):
            with st.expander("🖼️ Image Manager", expanded=True):
                _render_scene_image_manager(scene, script, state, compact=False)
                if st.button("Close", key=f"close_img_mgr_{scene.index}"):
                    st.session_state[f"show_image_manager_{scene.index}"] = False
                    st.rerun()
    else:
        # Full mode - complete image manager
        st.markdown(f"#### 🖼️ Scene {scene.index} Image Manager")

        # Image model selection
        with st.expander("⚙️ Image Generation Settings", expanded=False):
            image_models = {
                "gemini-3-pro-image-preview": "🏆 Nano Banana Pro (Best, 4K, 14 refs)",
                "imagen-4.0-ultra-generate-001": "🏆 Imagen 4 Ultra (Highest quality)",
                "imagen-4.0-generate-001": "🎨 Imagen 4 Standard (Good quality)",
                "gemini-2.5-flash-image": "⚡ Nano Banana Fast (Quick, 1K)",
                "imagen-4.0-fast-generate-001": "⚡ Imagen 4 Fast (Quick)",
            }

            # Per-scene image model (falls back to global setting)
            scene_image_model = getattr(scene, 'image_model', None)
            global_model = st.session_state.get("scene_image_model", "gemini-3-pro-image-preview")
            current_model = scene_image_model or global_model

            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                model_list = list(image_models.keys())
                model_idx = model_list.index(current_model) if current_model in model_list else 0
                new_model = st.selectbox(
                    "Model",
                    options=model_list,
                    index=model_idx,
                    format_func=lambda x: image_models[x],
                    key=f"scene_img_model_{scene.index}",
                )
                if new_model != scene_image_model:
                    try:
                        scene.image_model = new_model
                    except Exception:
                        object.__setattr__(scene, 'image_model', new_model)
                    save_movie_state()

            with col2:
                image_sizes = {"1K": "1024px", "2K": "2048px", "4K": "4096px"}
                scene_size = getattr(scene, 'image_size', None)
                global_size = st.session_state.get("scene_image_size", "2K")
                current_size = scene_size or global_size

                size_list = list(image_sizes.keys())
                size_idx = size_list.index(current_size) if current_size in size_list else 1
                new_size = st.selectbox(
                    "Size",
                    options=size_list,
                    index=size_idx,
                    format_func=lambda x: image_sizes[x],
                    key=f"scene_img_size_{scene.index}",
                )
                if new_size != scene_size:
                    try:
                        scene.image_size = new_size
                    except Exception:
                        object.__setattr__(scene, 'image_size', new_size)
                    save_movie_state()

            with col3:
                aspect_ratios = {"16:9": "Widescreen", "9:16": "Portrait", "1:1": "Square", "4:3": "Standard"}
                scene_ar = getattr(scene, 'image_aspect_ratio', None)
                global_ar = st.session_state.get("scene_image_aspect_ratio", "16:9")
                current_ar = scene_ar or global_ar

                ar_list = list(aspect_ratios.keys())
                ar_idx = ar_list.index(current_ar) if current_ar in ar_list else 0
                new_ar = st.selectbox(
                    "Aspect Ratio",
                    options=ar_list,
                    index=ar_idx,
                    format_func=lambda x: aspect_ratios[x],
                    key=f"scene_img_ar_{scene.index}",
                )
                if new_ar != scene_ar:
                    try:
                        scene.image_aspect_ratio = new_ar
                    except Exception:
                        object.__setattr__(scene, 'image_aspect_ratio', new_ar)
                    save_movie_state()

            # Show image anti-prompt (what to avoid) based on visual style
            mgr_visual_style = state.config.visual_style if state.config else (script.visual_style if script.visual_style else "")
            mgr_anti_prompt = get_image_anti_prompt(mgr_visual_style)
            st.caption(f"🚫 **Avoid (based on visual style):** _{mgr_anti_prompt}_")

        # Current image display with full-size viewer
        if has_image:
            # Full-size viewer
            with st.expander("🔍 View Full Size", expanded=False):
                st.image(str(scene.image_path), width="stretch")
                img = Path(scene.image_path)
                if img.exists():
                    from PIL import Image as PILImage
                    try:
                        pil_img = PILImage.open(img)
                        st.caption(f"**{pil_img.width}x{pil_img.height}** | {img.name}")
                    except Exception:
                        st.caption(f"{img.name}")

            # Thumbnail preview
            st.image(str(scene.image_path), width=300, caption=f"Current image ({num_variants} total)")
        else:
            st.info("No image generated yet. Click 'Generate New Variant' below.")

        # Variant carousel (if multiple variants)
        if num_variants > 1:
            st.markdown("---")
            st.markdown(f"**📂 Browse Variants ({num_variants})**")

            # Slider for navigation
            sorted_variants = sorted(existing_variants, key=lambda x: x.name)

            # Find current selection index
            current_selection = scene.get_selected_image()
            current_var_idx = 0
            if current_selection:
                for idx, v in enumerate(sorted_variants):
                    if str(v) == str(current_selection) or v == current_selection:
                        current_var_idx = idx
                        break

            selected_idx = st.slider(
                "Browse variants",
                min_value=1,
                max_value=num_variants,
                value=current_var_idx + 1,
                key=f"var_slider_{scene.index}",
                format="%d of " + str(num_variants),
            ) - 1

            # Display selected variant
            selected_var = sorted_variants[selected_idx]
            st.image(str(selected_var), width="stretch")

            # Show file info
            st.caption(f"**File:** {selected_var.name}")

            # Action buttons
            act_col1, act_col2, act_col3 = st.columns(3)

            with act_col1:
                is_current = str(selected_var) == str(scene.image_path)
                if st.button(
                    "✓ Use This" if not is_current else "✓ Current",
                    key=f"use_var_{scene.index}",
                    type="primary" if not is_current else "secondary",
                    disabled=is_current,
                    use_container_width=True,
                ):
                    # Find index in scene.image_variants and select it
                    for idx, v in enumerate(scene.image_variants):
                        if str(v) == str(selected_var):
                            scene.select_image_variant(idx)
                            save_movie_state()
                            st.success("Image updated!")
                            st.rerun()
                            break
                    else:
                        # Not in list yet, add it
                        new_idx = scene.add_image_variant(str(selected_var))
                        scene.select_image_variant(new_idx)
                        save_movie_state()
                        st.success("Image updated!")
                        st.rerun()

            with act_col2:
                if st.button("🔍 Full Size", key=f"fullsize_{scene.index}", use_container_width=True):
                    st.session_state[f"fullsize_var_{scene.index}"] = str(selected_var)
                    st.rerun()

            with act_col3:
                # Don't allow deleting the currently used image
                can_delete = not is_current and num_variants > 1
                if st.button(
                    "🗑️ Delete",
                    key=f"delete_var_{scene.index}",
                    disabled=not can_delete,
                    use_container_width=True,
                    help="Can't delete currently used image" if not can_delete else "Delete this variant"
                ):
                    try:
                        os.remove(selected_var)
                        # Remove from variants list
                        scene.image_variants = [v for v in scene.image_variants if str(v) != str(selected_var)]
                        save_movie_state()
                        st.success("Deleted!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete: {e}")

            # Full-size popup
            if st.session_state.get(f"fullsize_var_{scene.index}"):
                with st.container(border=True):
                    st.markdown("**Full Size Preview**")
                    st.image(st.session_state[f"fullsize_var_{scene.index}"], width="stretch")
                    if st.button("Close", key=f"close_fullsize_{scene.index}"):
                        del st.session_state[f"fullsize_var_{scene.index}"]
                        st.rerun()

        # Generate new variant button
        st.markdown("---")
        gen_col1, gen_col2 = st.columns([2, 1])
        with gen_col1:
            if st.button("🎨 Generate New Variant", key=f"gen_new_var_{scene.index}",
                        type="primary", use_container_width=True):
                _generate_single_scene_image(scene, script, state)
                st.rerun()
        with gen_col2:
            if num_variants > 0:
                st.metric("Variants", num_variants)


def _render_scene_video_manager(scene: "MovieScene", state: "MovieModeState", compact: bool = False) -> None:
    """Render scene video manager with variant carousel, similar to image manager.

    Args:
        scene: The MovieScene to manage videos for
        state: MovieModeState
        compact: Deprecated, kept for compatibility. Full mode is always used.
    """
    has_video = scene.video_path and Path(scene.video_path).exists()

    # Collect all video variants
    video_dir = get_project_dir() / "videos"
    existing_variants = []
    seen_paths = set()

    # Helper to add variant if not already seen
    def add_variant(path: Path):
        resolved = str(path.resolve())
        if resolved not in seen_paths and path.exists():
            seen_paths.add(resolved)
            existing_variants.append(path)
            return True
        return False

    # First add the main scene video (so it's first)
    if has_video:
        add_variant(Path(scene.video_path))

    # Add videos from video directory matching this scene
    if video_dir.exists():
        for vid_file in video_dir.glob(f"scene_{scene.index:03d}*.mp4"):
            add_variant(vid_file)

    num_variants = len(existing_variants)

    # Full mode - complete video manager
    st.markdown(f"#### 🎬 Scene {scene.index} Video Manager")

    if has_video:
        # Current video preview
        st.video(str(scene.video_path))
        st.caption(f"**Current:** {Path(scene.video_path).name}")
    else:
        st.info("No video generated yet.")

    # Variant carousel (if multiple variants)
    if num_variants > 1:
        st.markdown("---")
        st.markdown(f"**📂 Browse Video Variants ({num_variants})**")

        # Slider for navigation
        sorted_variants = sorted(existing_variants, key=lambda x: x.name)

        # Find current selection index
        current_var_idx = 0
        if has_video:
            for idx, v in enumerate(sorted_variants):
                if str(v) == str(scene.video_path) or str(v.resolve()) == str(Path(scene.video_path).resolve()):
                    current_var_idx = idx
                    break

        selected_idx = st.slider(
            "Browse video variants",
            min_value=1,
            max_value=num_variants,
            value=current_var_idx + 1,
            key=f"vid_var_slider_{scene.index}",
            format="%d of " + str(num_variants),
        ) - 1

        # Display selected variant
        selected_var = sorted_variants[selected_idx]
        st.video(str(selected_var))

        # Show file info
        st.caption(f"**File:** {selected_var.name}")

        # Action buttons
        act_col1, act_col2, act_col3 = st.columns(3)

        with act_col1:
            is_current = str(selected_var.resolve()) == str(Path(scene.video_path).resolve()) if has_video else False
            if st.button(
                "✓ Use This" if not is_current else "✓ Current",
                key=f"use_vid_var_{scene.index}",
                type="primary" if not is_current else "secondary",
                disabled=is_current,
                use_container_width=True,
            ):
                scene.video_path = str(selected_var)
                save_movie_state()
                st.success("Video updated!")
                st.rerun()

        with act_col2:
            # Get video duration info
            try:
                import subprocess
                result = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(selected_var)],
                    capture_output=True, text=True, timeout=5
                )
                duration = float(result.stdout.strip())
                st.metric("Duration", f"{duration:.1f}s")
            except Exception:
                st.empty()

        with act_col3:
            # Don't allow deleting the currently used video
            import os
            can_delete = not is_current and num_variants > 1
            if st.button(
                "🗑️ Delete",
                key=f"delete_vid_var_{scene.index}",
                disabled=not can_delete,
                use_container_width=True,
                help="Can't delete currently used video" if not can_delete else "Delete this variant"
            ):
                try:
                    os.remove(selected_var)
                    save_movie_state()
                    st.success("Deleted!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete: {e}")

    # Stats
    if num_variants > 0:
        st.markdown("---")
        st.metric("Video Variants", num_variants)


def _render_scene_editor_form(script: Script, state: MovieModeState, visual_style: str) -> None:
    """Render the streamlined scene editor form."""
    from src.models.schemas import Emotion, DialogueLine, SceneDirection, MovieScene

    # Ensure selected_scene_idx is valid
    scene_indices = [s.index for s in script.scenes]
    if st.session_state.selected_scene_idx not in scene_indices:
        st.session_state.selected_scene_idx = scene_indices[0] if scene_indices else 1

    current_idx = st.session_state.selected_scene_idx
    current_pos = scene_indices.index(current_idx) if current_idx in scene_indices else 0

    # Scene navigation bar
    nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
    with nav_col1:
        # Previous scene button
        can_go_prev = current_pos > 0
        if st.button("⬅️ Prev", use_container_width=True, disabled=not can_go_prev, key="nav_prev"):
            if can_go_prev:
                st.session_state.selected_scene_idx = scene_indices[current_pos - 1]
                st.rerun()

    with nav_col2:
        # Scene selector dropdown
        scene_options = {s.index: f"Scene {s.index}: {s.title or 'Untitled'}" for s in script.scenes}
        selected_idx = st.selectbox(
            "Scene",
            options=list(scene_options.keys()),
            format_func=lambda x: scene_options.get(x, f"Scene {x}"),
            index=current_pos,
            key="scene_selector",
            label_visibility="collapsed"
        )

        if selected_idx != current_idx:
            st.session_state.selected_scene_idx = selected_idx
            st.rerun()

    with nav_col3:
        # Next scene button
        can_go_next = current_pos < len(scene_indices) - 1
        if st.button("Next ➡️", use_container_width=True, disabled=not can_go_next, key="nav_next"):
            if can_go_next:
                st.session_state.selected_scene_idx = scene_indices[current_pos + 1]
                st.rerun()

    scene = next((s for s in script.scenes if s.index == selected_idx), None)
    if not scene:
        st.warning("Scene not found.")
        return

    st.markdown("---")

    # Scene title
    col1, col2 = st.columns([3, 1])
    with col1:
        new_title = st.text_input("Title", value=scene.title or "", key=f"title_{scene.index}")
        if new_title != scene.title:
            scene.title = new_title
            st.session_state.scenes_dirty = True
    with col2:
        if len(script.scenes) > 1 and st.button("🗑️ Delete", key=f"del_{scene.index}"):
            script.scenes.remove(scene)
            st.session_state.scenes_dirty = True
            st.rerun()

    # Scene direction (2 columns)
    col1, col2 = st.columns(2)
    with col1:
        new_setting = st.text_area("Setting", value=scene.direction.setting, height=80, key=f"set_{scene.index}")
        if new_setting != scene.direction.setting:
            scene.direction.setting = new_setting
            st.session_state.scenes_dirty = True

        cameras = ["wide shot", "medium shot", "close-up", "extreme close-up", "over-the-shoulder", "POV"]
        cam_idx = cameras.index(scene.direction.camera) if scene.direction.camera in cameras else 1
        new_cam = st.selectbox("Camera", cameras, index=cam_idx, key=f"cam_{scene.index}")
        if new_cam != scene.direction.camera:
            scene.direction.camera = new_cam
            st.session_state.scenes_dirty = True

    with col2:
        new_lighting = st.text_input("Lighting", value=scene.direction.lighting or "", key=f"light_{scene.index}")
        if new_lighting != (scene.direction.lighting or ""):
            scene.direction.lighting = new_lighting or None
            st.session_state.scenes_dirty = True

        moods = ["neutral", "tense", "happy", "sad", "mysterious", "romantic", "action", "comedic"]
        mood_idx = moods.index(scene.direction.mood) if scene.direction.mood in moods else 0
        new_mood = st.selectbox("Mood", moods, index=mood_idx, key=f"mood_{scene.index}")
        if new_mood != scene.direction.mood:
            scene.direction.mood = new_mood
            st.session_state.scenes_dirty = True

        # Characters in scene
        all_chars = [c.id for c in script.characters]
        char_names = {c.id: c.name for c in script.characters}
        visible = [c for c in (scene.direction.visible_characters or []) if c in all_chars]
        new_visible = st.multiselect("Characters", all_chars, default=visible,
                                     format_func=lambda x: char_names.get(x, x), key=f"chars_{scene.index}")
        if set(new_visible) != set(visible):
            scene.direction.visible_characters = new_visible
            st.session_state.scenes_dirty = True

    # Prompts section with regenerate buttons
    st.markdown("##### Prompts")

    # Check if prompts exist
    has_visual_prompt = bool(scene.visual_prompt and scene.visual_prompt.strip())
    has_video_prompt = bool(getattr(scene, 'video_prompt', None) and scene.video_prompt.strip())
    has_scene_image = scene.image_path and Path(scene.image_path).exists()

    # Status indicator
    prompt_status = []
    if has_visual_prompt:
        prompt_status.append("✅ Image prompt")
    else:
        prompt_status.append("❌ Image prompt missing")
    if has_video_prompt:
        prompt_status.append("✅ Video prompt")
    else:
        prompt_status.append("❌ Video prompt missing")

    st.caption(" | ".join(prompt_status))

    prompt_col1, prompt_col2 = st.columns(2)
    with prompt_col1:
        st.markdown("**Image Generation Prompt**")

        # Get visual prompt - use session state override if just regenerated
        regen_key = f"visual_prompt_regenerated_{scene.index}"
        if regen_key in st.session_state and st.session_state[regen_key]:
            current_visual_prompt = st.session_state.get(f"visual_prompt_value_{scene.index}", "") or ""
            st.session_state[regen_key] = False
        else:
            current_visual_prompt = scene.visual_prompt or ""

        # Use versioned key to force widget refresh on regeneration
        widget_version = st.session_state.get(f"visual_prompt_version_{scene.index}", 0)
        widget_key = f"edit_visual_prompt_{scene.index}_v{widget_version}"

        new_visual_prompt = st.text_area(
            "Image Prompt",
            value=current_visual_prompt,
            height=100,
            key=widget_key,
            placeholder="Describe the visual scene for image generation...",
            label_visibility="collapsed",
        )
        if new_visual_prompt != current_visual_prompt:
            try:
                scene.visual_prompt = new_visual_prompt
            except Exception:
                object.__setattr__(scene, 'visual_prompt', new_visual_prompt)
            st.session_state.scenes_dirty = True

        # Regenerate button for visual prompt
        if st.button("🔄 Regenerate Image Prompt", key=f"regen_visual_{scene.index}",
                    help="Use AI to generate a new image prompt based on scene settings"):
            with st.spinner("Generating image prompt..."):
                new_prompt = _regenerate_scene_prompt(scene, script, state, prompt_type="visual", use_image_context=False)
                if new_prompt:
                    try:
                        scene.visual_prompt = new_prompt
                    except Exception:
                        object.__setattr__(scene, 'visual_prompt', new_prompt)
                    save_movie_state()
                    # Store new value and increment version to force widget refresh
                    st.session_state[f"visual_prompt_value_{scene.index}"] = new_prompt
                    st.session_state[f"visual_prompt_regenerated_{scene.index}"] = True
                    st.session_state[f"visual_prompt_version_{scene.index}"] = widget_version + 1
                    st.success("Image prompt generated!")
                    st.rerun()

        # Image generation settings - right next to prompt
        st.markdown("**Settings**")
        img_set_col1, img_set_col2 = st.columns(2)
        with img_set_col1:
            image_models = {
                None: "Default",
                "gemini-3-pro-image-preview": "Pro 4K",
                "gemini-2.5-flash-image": "Flash",
                "imagen-4.0-ultra-generate-001": "Imagen Ultra",
            }
            current_img_model = getattr(scene, 'image_model', None)
            img_model_keys = list(image_models.keys())
            img_model_idx = img_model_keys.index(current_img_model) if current_img_model in img_model_keys else 0
            new_img_model = st.selectbox(
                "Model",
                options=img_model_keys,
                index=img_model_idx,
                format_func=lambda x: image_models.get(x, str(x)),
                key=f"img_model_inline_{scene.index}"
            )
            if new_img_model != current_img_model:
                try:
                    scene.image_model = new_img_model
                except Exception:
                    object.__setattr__(scene, 'image_model', new_img_model)
                st.session_state.scenes_dirty = True

        with img_set_col2:
            aspect_ratios = {
                None: "16:9",
                "16:9": "16:9",
                "9:16": "9:16",
                "1:1": "1:1",
                "4:3": "4:3",
            }
            current_aspect = getattr(scene, 'image_aspect_ratio', None)
            aspect_keys = list(aspect_ratios.keys())
            aspect_idx = aspect_keys.index(current_aspect) if current_aspect in aspect_keys else 0
            new_aspect = st.selectbox(
                "Ratio",
                options=aspect_keys,
                index=aspect_idx,
                format_func=lambda x: aspect_ratios.get(x, str(x)),
                key=f"img_aspect_inline_{scene.index}"
            )
            if new_aspect != current_aspect:
                try:
                    scene.image_aspect_ratio = new_aspect
                except Exception:
                    object.__setattr__(scene, 'image_aspect_ratio', new_aspect)
                st.session_state.scenes_dirty = True

    with prompt_col2:
        st.markdown("**Video Animation Prompt**")
        current_video_prompt = getattr(scene, 'video_prompt', None) or ""

        # Use version-based key to force widget refresh after regeneration
        vid_prompt_version = st.session_state.get(f"edit_video_prompt_version_{scene.index}", 0)
        vid_prompt_key = f"edit_video_prompt_{scene.index}_v{vid_prompt_version}"

        new_video_prompt = st.text_area(
            "Video Prompt",
            value=current_video_prompt,
            height=100,
            key=vid_prompt_key,
            placeholder="Describe character movements, actions, dialogue in quotes...",
            label_visibility="collapsed",
        )
        if new_video_prompt != current_video_prompt:
            try:
                scene.video_prompt = new_video_prompt
            except Exception:
                object.__setattr__(scene, 'video_prompt', new_video_prompt)
            st.session_state.scenes_dirty = True

        # Show video negative prompt for WAN/Seedance (not Veo)
        scene_gen_model = getattr(scene, 'generation_model', None) or (state.config.generation_method if state.config else "veo3")
        if scene_gen_model in ["wan26", "seedance15"]:
            vid_visual_style = state.config.visual_style if state.config else state.script.visual_style
            vid_neg_prompt = get_video_negative_prompt(vid_visual_style, scene_gen_model)
            if vid_neg_prompt:
                st.caption(f"🚫 **Negative prompt:** _{vid_neg_prompt}_")

        # Regenerate button for video prompt - PROMINENT
        use_img_context = st.checkbox(
            "📷 Use scene image as context for video prompt",
            value=has_scene_image,
            key=f"use_img_ctx_{scene.index}",
            help="Analyze the generated scene image to create a video prompt that flows naturally from the image"
        )

        if st.button("🔄 Regenerate Video Prompt" + (" (with image analysis)" if use_img_context and has_scene_image else ""),
                    key=f"regen_video_{scene.index}",
                    type="primary" if not has_video_prompt else "secondary",
                    use_container_width=True):
            with st.spinner("Generating video prompt..." + (" (analyzing image)" if use_img_context and has_scene_image else "")):
                new_prompt = _regenerate_scene_prompt(scene, script, state,
                                                     prompt_type="video",
                                                     use_image_context=use_img_context)
                if new_prompt:
                    try:
                        scene.video_prompt = new_prompt
                    except Exception:
                        object.__setattr__(scene, 'video_prompt', new_prompt)
                    save_movie_state()
                    # Increment version to force widget to refresh with new value
                    st.session_state[f"edit_video_prompt_version_{scene.index}"] = vid_prompt_version + 1
                    st.success("Video prompt generated!")
                    st.rerun()

        if has_scene_image and not has_video_prompt:
            st.warning("⚠️ Video prompt missing! Click the button above to generate one using the scene image for context.")

    # Video generation settings
    project_gen_method = state.config.generation_method if state.config else "tts_images"
    if project_gen_method != "tts_images":
        st.markdown("##### Video Generation Settings")
        st.caption("Override project defaults for this scene only")

        # Model selector (first row)
        model_options = {
            None: f"Project Default ({project_gen_method})",
            "veo3": "Veo 3.1 (Google)",
            "wan26": "WAN 2.6 (AtlasCloud)",
            "seedance15": "Seedance 1.5 Pro (AtlasCloud)",
            "seedance_fast": "Seedance Fast (AtlasCloud)",
        }
        current_model = scene.generation_model
        model_keys = list(model_options.keys())
        model_idx = model_keys.index(current_model) if current_model in model_keys else 0
        new_model = st.selectbox(
            "Video Model",
            options=model_keys,
            index=model_idx,
            format_func=lambda x: model_options.get(x, str(x)),
            key=f"edit_model_{scene.index}"
        )
        if new_model != current_model:
            scene.generation_model = new_model
            st.session_state.scenes_dirty = True
            save_movie_state()  # Persist the model change

        # Determine effective model for duration/resolution options
        effective_model = new_model or project_gen_method

        # Veo variant selector (when Veo is selected)
        if effective_model == "veo3":
            veo_variant_options = {
                None: "Project Default",
                "veo-3.1-generate-preview": "Veo 3.1 Standard (Best quality)",
                "veo-3.1-fast-generate-preview": "Veo 3.1 Fast (Quicker)",
            }
            current_veo_variant = getattr(scene, 'veo_model_variant', None)
            veo_keys = list(veo_variant_options.keys())
            veo_idx = veo_keys.index(current_veo_variant) if current_veo_variant in veo_keys else 0
            new_veo_variant = st.selectbox(
                "Veo Variant",
                options=veo_keys,
                index=veo_idx,
                format_func=lambda x: veo_variant_options.get(x, str(x)),
                key=f"edit_veo_variant_{scene.index}",
                help="Choose between Standard (best quality) and Fast (quicker generation)"
            )
            if new_veo_variant != current_veo_variant:
                try:
                    scene.veo_model_variant = new_veo_variant
                except Exception:
                    object.__setattr__(scene, 'veo_model_variant', new_veo_variant)
                st.session_state.scenes_dirty = True
                save_movie_state()

        # Camera angle override (from cinematography plan)
        state = get_movie_state()
        if hasattr(state, 'cinematography_plan') and state.cinematography_plan:
            # Get planned camera for this scene
            planned_ctx = next(
                (c for c in state.cinematography_plan.scenes if c.scene_index == scene.index),
                None
            )
            planned_camera = planned_ctx.camera_angle.value if planned_ctx else "medium_shot"
            planned_display = planned_camera.replace("_", " ").title()

            camera_options = {
                None: f"Auto ({planned_display})",
                "wide_shot": "Wide Shot",
                "medium_shot": "Medium Shot",
                "close_up": "Close-up",
                "over_shoulder": "Over-the-shoulder",
                "two_shot": "Two Shot",
                "extreme_close": "Extreme Close-up",
            }
            current_camera = getattr(scene, 'camera_override', None)
            camera_keys = list(camera_options.keys())
            camera_idx = camera_keys.index(current_camera) if current_camera in camera_keys else 0
            new_camera = st.selectbox(
                "Camera Angle",
                options=camera_keys,
                index=camera_idx,
                format_func=lambda x: camera_options.get(x, str(x)),
                key=f"edit_camera_{scene.index}",
                help="Override the auto-planned camera angle for this scene"
            )
            if new_camera != current_camera:
                try:
                    scene.camera_override = new_camera
                except Exception:
                    object.__setattr__(scene, 'camera_override', new_camera)
                st.session_state.scenes_dirty = True
                save_movie_state()

        # Transition override controls (for context-aware rendering)
        trans_col1, trans_col2 = st.columns(2)
        with trans_col1:
            transition_options = {
                None: "Auto (context-aware)",
                "none": "Hard Cut",
                "crossfade": "Crossfade",
                "fade": "Fade to Black",
                "dissolve": "Dissolve",
            }
            current_trans = getattr(scene, 'transition_override', None)
            trans_keys = list(transition_options.keys())
            trans_idx = trans_keys.index(current_trans) if current_trans in trans_keys else 0
            new_trans = st.selectbox(
                "Transition (after scene)",
                options=trans_keys,
                index=trans_idx,
                format_func=lambda x: transition_options.get(x, str(x)),
                key=f"edit_transition_{scene.index}",
                help="Override the transition effect after this scene"
            )
            if new_trans != current_trans:
                try:
                    scene.transition_override = new_trans
                except Exception:
                    object.__setattr__(scene, 'transition_override', new_trans)
                st.session_state.scenes_dirty = True
                save_movie_state()

        with trans_col2:
            current_trans_dur = getattr(scene, 'transition_duration_override', None)
            new_trans_dur = st.number_input(
                "Transition Duration (s)",
                min_value=0.1,
                max_value=2.0,
                value=current_trans_dur if current_trans_dur else 0.5,
                step=0.1,
                key=f"edit_trans_dur_{scene.index}",
                help="Override transition duration (only used if transition type is set)",
                disabled=(new_trans is None)  # Disable if using auto
            )
            # Only save if not auto and value changed
            if new_trans is not None:
                effective_dur = new_trans_dur if new_trans is not None else None
                if effective_dur != current_trans_dur:
                    try:
                        scene.transition_duration_override = effective_dur
                    except Exception:
                        object.__setattr__(scene, 'transition_duration_override', effective_dur)
                    st.session_state.scenes_dirty = True
                    save_movie_state()

        v_col1, v_col2, v_col3 = st.columns(3)

        with v_col1:
            # Duration - options depend on effective model
            if effective_model == "veo3":
                dur_options = [4, 6, 8]
            elif effective_model == "wan26":
                dur_options = [5, 10, 15]
            else:  # seedance15
                dur_options = [3, 5, 8, 10, 15]

            # Calculate auto duration from dialogue
            auto_dur = scene.get_clip_duration(effective_model)

            current_dur = scene.clip_duration
            dur_idx = 0 if current_dur is None else (dur_options.index(current_dur) + 1 if current_dur in dur_options else 0)
            new_dur = st.selectbox(
                "Duration",
                options=[None] + dur_options,
                index=dur_idx,
                format_func=lambda x: f"Auto ({auto_dur}s)" if x is None else f"{x}s",
                key=f"edit_dur_{scene.index}"
            )
            if new_dur != current_dur:
                scene.clip_duration = new_dur
                st.session_state.scenes_dirty = True

        with v_col2:
            # Resolution - options depend on effective model
            if effective_model == "veo3":
                res_options = ["720p", "1080p"]
            else:
                res_options = ["480p", "720p", "1080p"]

            current_res = scene.resolution
            res_idx = 0 if current_res is None else (res_options.index(current_res) + 1 if current_res in res_options else 0)
            new_res = st.selectbox(
                "Resolution",
                options=[None] + res_options,
                index=res_idx,
                format_func=lambda x: "Default" if x is None else x,
                key=f"edit_res_{scene.index}"
            )
            if new_res != current_res:
                scene.resolution = new_res
                st.session_state.scenes_dirty = True

        with v_col3:
            # Lip-sync for Seedance (both Pro and Fast support 2-step lip sync)
            if "seedance" in str(effective_model).lower():
                current_lip = scene.enable_lip_sync
                new_lip = st.checkbox(
                    "Enable Lip Sync",
                    value=current_lip if current_lip is not None else True,
                    key=f"edit_lip_{scene.index}",
                    help="2-step: generates video, then applies lip-sync with bytedance/lipsync model"
                )
                if new_lip != current_lip:
                    scene.enable_lip_sync = new_lip
                    st.session_state.scenes_dirty = True
            else:
                st.empty()  # Placeholder

        # Scene Video section
        st.markdown("---")
        st.markdown("##### Scene Video")
        has_video = scene.video_path and Path(scene.video_path).exists()
        has_video_prompt = bool(getattr(scene, 'video_prompt', None) and scene.video_prompt.strip())

        # Show negative prompt / quality guidance info based on model
        if "wan26" in str(effective_model).lower() or "wan" in str(effective_model).lower():
            edit_visual_style = state.config.visual_style if state.config else (state.script.visual_style if state.script else "")
            edit_style_lower = (edit_visual_style or "").lower()
            if "photorealistic" in edit_style_lower or "realistic" in edit_style_lower or "photo" in edit_style_lower:
                edit_neg_prompt = "CGI, cartoon, animated, 3D render, digital art, stylized, artificial, plastic skin..."
            elif "anime" in edit_style_lower or "cartoon" in edit_style_lower:
                edit_neg_prompt = "photorealistic, real photo, live action, realistic skin texture"
            elif "3d" in edit_style_lower or "pixar" in edit_style_lower or "animated" in edit_style_lower:
                edit_neg_prompt = "photorealistic, real photo, live action, 2D, flat"
            else:
                edit_neg_prompt = "(no negative prompt - unrecognized style)"
            st.caption(f"🚫 **Negative prompt:** {edit_neg_prompt}")
        elif "seedance" in str(effective_model).lower():
            # Seedance uses embedded quality guidance in prompt, not a separate negative_prompt
            st.caption("ℹ️ **Seedance:** Quality guidance embedded in prompt (camera behavior, style, anti-drift)")

        # Collect video variants (same logic as video manager)
        video_dir = get_project_dir() / "videos"
        video_variants = []
        video_seen = set()
        if has_video:
            video_variants.append(Path(scene.video_path))
            video_seen.add(str(Path(scene.video_path).resolve()))
        if video_dir.exists():
            for vid_file in video_dir.glob(f"scene_{scene.index:03d}*.mp4"):
                resolved = str(vid_file.resolve())
                if resolved not in video_seen:
                    video_variants.append(vid_file)
                    video_seen.add(resolved)
        num_video_variants = len(video_variants)

        vid_col1, vid_col2 = st.columns([1, 2])
        with vid_col1:
            # Generate video button
            btn_disabled = not has_scene_image and effective_model not in ["wan26_t2v", "seedance15_t2v", "veo3"]
            btn_label = "🎬 Regenerate Video" if has_video else "🎬 Generate Video"
            if not has_video_prompt:
                btn_label += " (no prompt)"

            if st.button(btn_label, key=f"gen_video_edit_{scene.index}",
                        disabled=btn_disabled,
                        type="primary" if not has_video else "secondary",
                        use_container_width=True):
                # Build defaults dict for the generator
                defaults = {
                    "resolution": getattr(scene, 'resolution', None) or (
                        state.config.wan_resolution if state.config else "720p"
                    )
                }
                _generate_single_scene_video(state, scene, effective_model, defaults)
                st.rerun()

            if btn_disabled:
                st.caption("⚠️ Generate an image first (or use T2V mode)")

            # Browse Videos button (matching image section layout)
            if num_video_variants > 1:
                if st.button(f"📂 Browse {num_video_variants} Variants", key=f"vid_browse_{scene.index}", use_container_width=True):
                    st.session_state[f"show_video_manager_{scene.index}"] = True
                    st.rerun()

        with vid_col2:
            # Video display only (matching image section layout)
            if has_video:
                st.video(str(scene.video_path))
                if num_video_variants > 1:
                    st.caption(f"🎬 {num_video_variants} variants available")
            else:
                st.info("No video generated yet")

        # Show full video manager popup if requested (matching image section pattern)
        if st.session_state.get(f"show_video_manager_{scene.index}", False):
            with st.container(border=True):
                _render_scene_video_manager(scene, state, compact=False)
                if st.button("✕ Close Video Manager", key=f"close_vid_mgr_{scene.index}"):
                    st.session_state[f"show_video_manager_{scene.index}"] = False
                    st.rerun()

    # Scene image with variant management
    st.markdown("##### Scene Image")

    has_scene_image = scene.image_path and Path(scene.image_path).exists()

    # Collect all unique variants - check multiple locations
    scene_img_dir = get_project_dir() / "scenes" / f"scene_{scene.index:03d}" / "images"
    scenes_dir = get_project_dir() / "scenes"
    all_variants = set()

    # Add variants from the scene's variant list
    for var_path in scene.image_variants:
        if Path(var_path).exists():
            all_variants.add(str(Path(var_path).resolve()))

    # Add files from per-scene images subdirectory
    if scene_img_dir.exists():
        for img_file in scene_img_dir.glob("*.png"):
            all_variants.add(str(img_file.resolve()))
        for img_file in scene_img_dir.glob("*.jpg"):
            all_variants.add(str(img_file.resolve()))

    # Add files from scenes directory directly (batch generation saves here)
    if scenes_dir.exists():
        for img_file in list(scenes_dir.glob(f"scene_{scene.index:03d}*.png")) + list(scenes_dir.glob(f"scene_{scene.index:03d}*.jpg")):
            all_variants.add(str(img_file.resolve()))

    # Check old format: scenes/scene_XXX/scene_XXX.png (direct children of scene folder)
    scene_folder = get_project_dir() / "scenes" / f"scene_{scene.index:03d}"
    if scene_folder.exists():
        for img_file in list(scene_folder.glob("*.png")) + list(scene_folder.glob("*.jpg")):
            if img_file.parent == scene_folder:  # Only direct children, not from images/
                all_variants.add(str(img_file.resolve()))

    # Add the main scene image
    if has_scene_image:
        all_variants.add(str(Path(scene.image_path).resolve()))

    num_scene_variants = len(all_variants)

    img_col1, img_col2 = st.columns([1, 2])
    with img_col1:
        if st.button("🎨 Generate New Image", key=f"genimg_{scene.index}", use_container_width=True):
            _generate_single_scene_image(scene, script, state)
            st.rerun()

        if num_scene_variants > 1:
            if st.button(f"📂 Browse {num_scene_variants} Variants", key=f"browse_editor_{scene.index}", use_container_width=True):
                st.session_state[f"show_editor_image_manager_{scene.index}"] = True
                st.rerun()

    with img_col2:
        if has_scene_image:
            # Show image with full-size option
            st.image(str(scene.image_path), width="stretch")
            if num_scene_variants > 1:
                st.caption(f"📷 {num_scene_variants} variants available")
        else:
            st.info("No image generated yet")

    # Show full image manager popup if requested
    if st.session_state.get(f"show_editor_image_manager_{scene.index}", False):
        with st.container(border=True):
            _render_scene_image_manager(scene, script, state, compact=False)
            if st.button("✕ Close Image Manager", key=f"close_editor_mgr_{scene.index}"):
                st.session_state[f"show_editor_image_manager_{scene.index}"] = False
                st.rerun()

    # Dialogue
    st.markdown("##### Dialogue")
    for d_idx, d in enumerate(scene.dialogue):
        cols = st.columns([2, 4, 2, 1])
        with cols[0]:
            char_ids = [c.id for c in script.characters]
            char_idx = char_ids.index(d.character_id) if d.character_id in char_ids else 0
            new_char = st.selectbox("Char", char_ids, index=char_idx,
                                    format_func=lambda x: char_names.get(x, x),
                                    key=f"dchar_{scene.index}_{d_idx}", label_visibility="collapsed")
            if new_char != d.character_id:
                d.character_id = new_char
                st.session_state.scenes_dirty = True
        with cols[1]:
            new_text = st.text_area("Text", value=d.text, height=60,
                                   key=f"dtext_{scene.index}_{d_idx}", label_visibility="collapsed")
            if new_text != d.text:
                d.text = new_text
                st.session_state.scenes_dirty = True
        with cols[2]:
            emotions = [e.value for e in Emotion]
            em_idx = emotions.index(d.emotion.value) if d.emotion.value in emotions else 0
            new_em = st.selectbox("Emotion", emotions, index=em_idx,
                                 key=f"dem_{scene.index}_{d_idx}", label_visibility="collapsed")
            if new_em != d.emotion.value:
                d.emotion = Emotion(new_em)
                st.session_state.scenes_dirty = True
        with cols[3]:
            if st.button("🗑️", key=f"ddel_{scene.index}_{d_idx}"):
                scene.dialogue.remove(d)
                st.session_state.scenes_dirty = True
                st.rerun()

    if st.button("➕ Add Dialogue", key=f"adddial_{scene.index}"):
        default_char = script.characters[0].id if script.characters else "narrator"
        scene.dialogue.append(DialogueLine(character_id=default_char, text="", emotion=Emotion.NEUTRAL))
        st.session_state.scenes_dirty = True
        st.rerun()


def render_voices_page() -> None:
    """Render the voice generation page."""
    state = get_movie_state()

    if not state.script:
        st.warning("Please create a script first.")
        return

    # Check if Veo 3.1 is available for alternative path
    from src.services.veo3_generator import check_veo3_available
    veo3_available = check_veo3_available()

    st.subheader("Generate Character Voices")

    # Show path options
    if veo3_available:
        st.info(
            """
            **Two ways to add voices to your movie:**

            1. **TTS Mode** (below): Generate voices now, then create images in the next step.
               Best for: More control over individual voices, lower cost.

            2. **Veo 3.1 Mode**: Skip this step and generate video clips with built-in dialogue.
               Best for: Higher quality, characters actually speak on screen, but costs more.
            """
        )

        if st.button("Skip to Veo 3.1 (Video with Dialogue)", type="secondary"):
            go_to_movie_step(MovieWorkflowStep.VISUALS)
            st.rerun()

        st.markdown("---")

    st.markdown(
        """
        Generate AI voices for each character's dialogue. The voices will be
        used to create the audio track for your video.
        """
    )

    # Check available TTS providers
    from src.services.tts_service import (
        check_elevenlabs_available,
        check_openai_tts_available,
        check_edge_tts_available,
        get_available_providers,
    )

    available = get_available_providers()

    if not available:
        st.error(
            """
            No TTS providers available. Please configure one:
            - **ElevenLabs:** Set `ELEVENLABS_API_KEY` environment variable
            - **OpenAI:** Set `OPENAI_API_KEY` environment variable
            - **Edge TTS:** Install with `pip install edge-tts` (free, no API key needed)
            """
        )
        return

    # Provider selection - default to first available
    # Check if we have a stored provider preference
    if "tts_provider" not in st.session_state:
        st.session_state.tts_provider = available[0]

    # Make sure stored provider is still available
    if st.session_state.tts_provider not in available:
        st.session_state.tts_provider = available[0]

    provider = st.selectbox(
        "TTS Provider",
        options=available,
        index=available.index(st.session_state.tts_provider),
        format_func=lambda x: {
            "elevenlabs": "ElevenLabs (Best Quality)",
            "openai": "OpenAI TTS",
            "edge": "Edge TTS (Free)",
        }.get(x, x),
        key="tts_provider_select",
    )

    # Update session state and character voices when provider changes
    if provider != st.session_state.tts_provider:
        st.session_state.tts_provider = provider
        # Update all character voice providers to use the selected provider
        for char in state.script.characters:
            char.voice.provider = provider
        st.rerun()

    # Ensure all characters use the selected provider
    for char in state.script.characters:
        if char.voice.provider != provider:
            char.voice.provider = provider

    # Show voice options per character
    st.markdown("### Character Voice Assignments")

    from src.services.tts_service import get_available_voices_for_provider, get_rotated_voice

    # Get available voices for selected provider
    available_voices = get_available_voices_for_provider(provider)
    voice_ids = list(available_voices.keys())
    voice_labels = available_voices

    for char_idx, char in enumerate(state.script.characters):
        with st.expander(f"🎭 {char.name}", expanded=char_idx == 0):
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.markdown(f"**Description:** {char.voice.voice_name or 'Not set'}")

                # Voice picker dropdown
                # Auto-assign a rotated voice if none set
                current_voice_id = char.voice.voice_id
                if not current_voice_id or current_voice_id not in voice_ids:
                    # Auto-assign based on character index for variety
                    current_voice_id = get_rotated_voice(provider, char_idx)
                    char.voice.voice_id = current_voice_id

                current_idx = voice_ids.index(current_voice_id) if current_voice_id in voice_ids else 0

                new_voice = st.selectbox(
                    "Voice",
                    options=voice_ids,
                    index=current_idx,
                    format_func=lambda x: voice_labels.get(x, x),
                    key=f"voice_select_{char.id}",
                )

                if new_voice != char.voice.voice_id:
                    char.voice.voice_id = new_voice

            with col2:
                # Speed slider
                new_speed = st.slider(
                    "Speed",
                    min_value=0.5,
                    max_value=2.0,
                    value=char.voice.speed,
                    step=0.1,
                    key=f"voice_speed_{char.id}",
                )
                if new_speed != char.voice.speed:
                    char.voice.speed = new_speed

            with col3:
                # Voice preview button - generates a short sample
                if st.button(f"🔊 Preview", key=f"preview_{char.id}"):
                    from src.services.tts_service import TTSService
                    import tempfile

                    preview_text = f"Hello, I am {char.name}."
                    if char.personality:
                        preview_text = f"Hello, I am {char.name}. {char.personality.split('.')[0]}."

                    with st.spinner(f"Generating preview..."):
                        try:
                            tts = TTSService(default_provider=provider)
                            # Generate to temp file
                            preview_path = Path(tempfile.mktemp(suffix=".mp3"))
                            tts.generate_speech(
                                text=preview_text,
                                voice_settings=char.voice,
                                output_path=preview_path,
                            )
                            # Play the audio
                            st.audio(str(preview_path), format="audio/mp3")
                        except Exception as e:
                            st.error(f"Preview failed: {e}")

    # Dialogue count
    total_dialogue = state.script.total_dialogue_count
    st.markdown(f"**Total dialogue lines:** {total_dialogue}")

    # Check if voices have already been generated
    voices_generated = sum(
        1 for scene in state.script.scenes
        for d in scene.dialogue
        if d.audio_path and Path(d.audio_path).exists()
    )

    st.markdown("---")

    # Show status if voices already generated
    if voices_generated > 0:
        st.success(f"**{voices_generated}/{total_dialogue} voice clips generated**")

        # Calculate total duration from existing audio
        last_scene = state.script.scenes[-1] if state.script.scenes else None
        if last_scene and last_scene.end_time:
            total_duration = last_scene.end_time
            minutes = int(total_duration // 60)
            seconds = int(total_duration % 60)
            st.info(f"Total duration: {minutes}m {seconds}s")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Regenerate Voices", use_container_width=True):
                st.session_state.regenerate_voices = True
                st.rerun()
        with col2:
            if st.button("Continue to Visuals →", type="primary", use_container_width=True):
                advance_movie_step()
                st.rerun()

    # Generate button (show if no voices or regenerating)
    should_generate = voices_generated == 0 or st.session_state.get("regenerate_voices", False)

    if should_generate:
        if st.button("🎙️ Generate All Voices", type="primary", use_container_width=True):
            # Clear regenerate flag
            st.session_state.pop("regenerate_voices", None)

            from src.services.tts_service import TTSService

            tts = TTSService(default_provider=provider)
            output_dir = get_project_dir() / "audio"
            output_dir.mkdir(parents=True, exist_ok=True)

            progress_bar = st.progress(0, text="Generating voices...")
            total = total_dialogue
            completed = 0

            # Track cumulative time for dialogue timing
            running_time = 0.0
            pause_between_lines = 0.3  # 300ms pause between dialogue lines
            pause_between_scenes = 1.0  # 1s pause between scenes
            failed_count = 0

            for scene_idx, scene in enumerate(state.script.scenes):
                # Mark scene start time
                scene.start_time = running_time

                for dialogue in scene.dialogue:
                    char = state.script.get_character(dialogue.character_id)
                    if not char:
                        st.warning(
                            f"Character '{dialogue.character_id}' not found, skipping"
                        )
                        failed_count += 1
                        continue

                    try:
                        audio_path = tts.generate_dialogue_audio(
                            dialogue=dialogue,
                            character=char,
                            output_dir=output_dir,
                        )

                        # Get duration and set timing
                        duration = tts.get_audio_duration(audio_path)
                        dialogue.audio_path = audio_path
                        dialogue.start_time = running_time
                        dialogue.end_time = running_time + duration
                        running_time = dialogue.end_time + pause_between_lines

                        completed += 1
                        progress_bar.progress(
                            completed / total,
                            text=f"Generated voice for {char.name} ({completed}/{total})"
                        )

                    except Exception as e:
                        st.error(f"Failed to generate voice for {char.name}: {e}")
                        failed_count += 1
                        # Clear timing for failed dialogue
                        dialogue.audio_path = None
                        dialogue.start_time = None
                        dialogue.end_time = None

                # Mark scene end time
                scene.end_time = running_time
                # Add pause between scenes (except for last scene)
                if scene_idx < len(state.script.scenes) - 1:
                    running_time += pause_between_scenes

            progress_bar.progress(1.0, text="Voice generation complete!")

            if failed_count > 0:
                st.warning(f"Generated {completed} clips ({failed_count} failed)")
            else:
                st.success(f"Generated {completed} voice clips!")

            # Auto-save after voice generation
            try:
                save_path = save_movie_state()
                st.info(f"💾 Auto-saved to {save_path.name}")
            except Exception as e:
                st.warning(f"Auto-save failed: {e}")

            # Rerun to show the "Continue" button
            st.rerun()

    # Navigation
    st.markdown("---")
    if st.button("← Back to Scenes"):
        go_to_movie_step(MovieWorkflowStep.SCENES)
        st.rerun()


def render_visuals_page() -> None:
    """Render the visual generation page with support for all video models."""
    state = get_movie_state()

    if not state.script:
        st.warning("Please create a script first.")
        return

    # Get project config for default settings
    project_config = state.config
    generation_method = project_config.generation_method if project_config else "tts_images"

    # Auto-generate cinematography plan if not exists or stale
    if generation_method != "tts_images" and state.script.scenes:
        plan_needed = False
        if not hasattr(state, 'cinematography_plan') or state.cinematography_plan is None:
            plan_needed = True
        elif len(state.cinematography_plan.scenes) != len(state.script.scenes):
            plan_needed = True

        if plan_needed:
            try:
                from src.services.cinematography import generate_cinematography_plan
                from src.models.schemas import CinematographyPlan
                plan = generate_cinematography_plan(state.script.scenes)
                state.cinematography_plan = plan
                save_movie_state()
                logger.info(f"Auto-generated cinematography plan for {len(plan.scenes)} scenes")
            except Exception as e:
                logger.warning(f"Failed to auto-generate cinematography plan: {e}")

    st.subheader("Generate Scene Videos")

    # Model info based on generation method
    model_info = {
        "veo3": {
            "name": "Google Veo 3.1",
            "icon": "🎬",
            "features": ["Native dialogue audio", "Cinematic motion", "Sound effects"],
            "durations": [4, 6, 8],
            "resolutions": ["720p", "1080p"],
            "cost_per_sec": 0.75,
        },
        "wan26": {
            "name": "WAN 2.6",
            "icon": "🎥",
            "features": ["Native audio", "Up to 15s clips", "Video-to-video"],
            "durations": [5, 10, 15],
            "resolutions": ["480p", "720p", "1080p"],
            "cost_per_sec": 0.075,
        },
        "seedance15": {
            "name": "Seedance 1.5 Pro",
            "icon": "💃",
            "features": ["Precision lip-sync", "Audio input", "Multi-language"],
            "durations": list(range(3, 16)),
            "resolutions": ["480p", "720p", "1080p"],
            "cost_per_sec": 0.0147,
        },
        "tts_images": {
            "name": "Traditional (Images + TTS)",
            "icon": "🖼️",
            "features": ["Scene images", "TTS audio", "Post-processing"],
            "durations": [],
            "resolutions": [],
            "cost_per_sec": 0,
        },
    }

    current_model = model_info.get(generation_method, model_info["tts_images"])
    st.markdown(f"### {current_model['icon']} {current_model['name']}")

    # Show features
    st.info(" • ".join(current_model["features"]))

    # Collect reference images for character consistency
    character_portraits = []
    for char in state.script.characters:
        if char.reference_image_path and Path(char.reference_image_path).exists():
            character_portraits.append(Path(char.reference_image_path))

    # Show character portraits available
    if character_portraits:
        st.success(f"✅ {len(character_portraits)} character portrait(s) available for reference")
    else:
        st.warning("⚠️ No character portraits. Generate them in the Characters step for better consistency.")

    # Generation settings from project config
    if generation_method != "tts_images":
        st.markdown("### Default Settings")
        st.caption("These are the project defaults from Setup. Individual scenes can override them.")

        col1, col2, col3 = st.columns(3)

        if generation_method == "veo3":
            default_duration = project_config.veo_duration if project_config else 8
            default_resolution = project_config.veo_resolution if project_config else "720p"
            with col1:
                st.metric("Duration", f"{default_duration}s")
            with col2:
                st.metric("Resolution", default_resolution)
            with col3:
                num_scenes = len(state.script.scenes)
                total_cost = num_scenes * default_duration * 0.75
                st.metric("Est. Cost", f"${total_cost:.2f}")

        elif generation_method == "wan26":
            default_duration = project_config.wan_duration if project_config else 10
            default_resolution = project_config.wan_resolution if project_config else "720p"
            with col1:
                st.metric("Duration", f"{default_duration}s")
            with col2:
                st.metric("Resolution", default_resolution)
            with col3:
                num_scenes = len(state.script.scenes)
                total_cost = num_scenes * default_duration * 0.075
                st.metric("Est. Cost", f"${total_cost:.2f}")

        elif generation_method == "seedance15":
            default_duration = project_config.seedance_duration if project_config else 8
            default_resolution = project_config.seedance_resolution if project_config else "720p"
            with col1:
                st.metric("Duration", f"{default_duration}s")
            with col2:
                st.metric("Resolution", default_resolution)
            with col3:
                num_scenes = len(state.script.scenes)
                total_cost = num_scenes * default_duration * 0.0147
                st.metric("Est. Cost", f"${total_cost:.2f}")

    # Continuity options
    with st.expander("🔗 Continuity Options", expanded=True):
        st.markdown("**Reference Images** help maintain visual consistency across clips.")
        use_char_refs = st.checkbox(
            "Use character portraits as references",
            value=True,
            help="Uses character portraits to maintain visual consistency (up to 3 images)",
        )
        use_scene_continuity = st.checkbox(
            "Use scene images for clip continuity",
            value=True,
            help="Uses the previous scene's image as first frame for smooth transitions",
        )
        # Video-to-video continuity - works with all models
        v2v_help = {
            "veo3": "Analyze previous video to maintain character/scene consistency",
            "wan26": "Use previous scene's video as input for smoother transitions",
            "seedance15": "Use previous scene's video as input for smoother transitions",
        }
        use_v2v_continuity = st.checkbox(
            "🎬 Use previous video for scene continuity",
            value=False,
            help=v2v_help.get(generation_method, "Use previous scene's video for continuity"),
        )

    # Cinematography Planning section
    if generation_method != "tts_images":
        with st.expander("🎬 Cinematography Planning", expanded=False):
            st.markdown("**Camera continuity** ensures natural shot flow between scenes.")

            # Show enable/disable toggle
            enable_continuity = st.checkbox(
                "Enable context-aware camera work",
                value=getattr(state, 'enable_camera_continuity', True),
                help="Automatically plan camera angles that flow naturally between scenes",
                key="enable_camera_continuity_checkbox"
            )
            if enable_continuity != getattr(state, 'enable_camera_continuity', True):
                state.enable_camera_continuity = enable_continuity
                save_movie_state()

            if enable_continuity and hasattr(state, 'cinematography_plan') and state.cinematography_plan:
                # Show current plan summary
                plan = state.cinematography_plan
                st.caption(f"📷 Camera plan generated: {len(plan.scenes)} scenes ({plan.pacing} pacing)")

                # Show camera angles in a compact grid
                cam_cols = st.columns(min(6, len(plan.scenes)))
                for i, ctx in enumerate(plan.scenes[:6]):  # Show first 6
                    with cam_cols[i % len(cam_cols)]:
                        camera_display = ctx.camera_angle.value.replace("_", " ").title()
                        role_icon = {"opening": "🎬", "climax": "🔥", "resolution": "🌅"}.get(ctx.narrative_role, "📷")
                        st.metric(f"S{ctx.scene_index + 1}", f"{role_icon}", camera_display)

                if len(plan.scenes) > 6:
                    st.caption(f"... and {len(plan.scenes) - 6} more scenes")

                # Regenerate button
                if st.button("🔄 Regenerate Camera Plan", key="regen_camera_plan"):
                    try:
                        from src.services.cinematography import generate_cinematography_plan
                        plan = generate_cinematography_plan(state.script.scenes)
                        state.cinematography_plan = plan
                        save_movie_state()
                        st.success(f"Regenerated camera plan for {len(plan.scenes)} scenes!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to regenerate plan: {e}")
            elif enable_continuity:
                st.info("Camera plan will be generated automatically when you regenerate video prompts.")

    # Scene list with per-scene settings
    st.markdown("### Scenes to Generate")

    # Track if any changes were made
    if "visuals_dirty" not in st.session_state:
        st.session_state.visuals_dirty = False

    # Get default values for the current generation method
    def _get_scene_defaults(method: str, cfg) -> dict:
        if method == "veo3":
            return {
                "duration": cfg.veo_duration if cfg else 8,
                "resolution": cfg.veo_resolution if cfg else "720p",
                "lip_sync": False,
            }
        elif method == "wan26":
            return {
                "duration": cfg.wan_duration if cfg else 10,
                "resolution": cfg.wan_resolution if cfg else "720p",
                "lip_sync": cfg.wan_enable_audio if cfg else True,
            }
        elif method == "seedance15":
            return {
                "duration": cfg.seedance_duration if cfg else 8,
                "resolution": cfg.seedance_resolution if cfg else "720p",
                "lip_sync": cfg.seedance_lip_sync if cfg else True,
            }
        return {"duration": 8, "resolution": "720p", "lip_sync": False}

    defaults = _get_scene_defaults(generation_method, project_config)

    # Storyboard grid - 3 columns like Song Mode
    _render_movie_storyboard_grid(state, generation_method, current_model, defaults)

    # Save button for per-scene changes
    if st.session_state.visuals_dirty:
        if st.button("💾 Save Scene Settings", type="secondary"):
            save_movie_state()
            st.session_state.visuals_dirty = False
            st.success("Scene settings saved!")
            st.rerun()

    # Generate All button
    st.markdown("---")

    # Count scenes by model
    model_counts = {"veo3": 0, "wan26": 0, "seedance15": 0}
    scenes_needing_video = []
    for scene in state.script.scenes:
        if not (scene.video_path and Path(scene.video_path).exists()):
            model = getattr(scene, 'generation_model', None) or generation_method
            if model in model_counts:
                model_counts[model] += 1
            scenes_needing_video.append(scene)

    if scenes_needing_video:
        # Show count summary
        model_names = {"veo3": "Veo 3.1", "wan26": "WAN 2.6", "seedance15": "Seedance 1.5"}
        counts_text = " | ".join(f"{model_names[m]}: {c}" for m, c in model_counts.items() if c > 0)
        st.info(f"**{len(scenes_needing_video)} scenes to generate:** {counts_text}")

        # Button text based on dominant model or mixed
        if sum(1 for c in model_counts.values() if c > 0) > 1:
            btn_text = f"🎬 Generate All {len(scenes_needing_video)} Scenes (Mixed Models)"
        else:
            dominant = next((m for m, c in model_counts.items() if c > 0), generation_method)
            btn_text = f"🎬 Generate All {len(scenes_needing_video)} Scenes ({model_names.get(dominant, dominant)})"

        if st.button(btn_text, type="primary", use_container_width=True):
            _generate_all_movie_scenes(state, project_config, use_char_refs, use_scene_continuity, use_v2v_continuity, character_portraits)
    else:
        st.success("✅ All scenes have videos generated!")

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Scenes"):
            go_to_movie_step(MovieWorkflowStep.SCENES)
            st.rerun()
    with col2:
        if st.button("Continue to Render →", type="primary"):
            advance_movie_step()
            st.rerun()


def _render_movie_storyboard_grid(state, generation_method: str, model_info: dict, defaults: dict) -> None:
    """Render the storyboard as a grid of scene cards (like Song Mode)."""
    scenes = state.script.scenes

    # Summary stats
    total_scenes = len(scenes)
    scenes_with_images = sum(1 for s in scenes if s.image_path and Path(s.image_path).exists())
    scenes_with_videos = sum(1 for s in scenes if s.video_path and Path(s.video_path).exists())

    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Scenes", total_scenes)
    with col2:
        st.metric("Images", f"{scenes_with_images}/{total_scenes}")
    with col3:
        if scenes_with_videos > 0:
            st.metric("Videos", f"{scenes_with_videos}/{total_scenes}", delta="✓" if scenes_with_videos == total_scenes else None)
        else:
            st.metric("Videos", "0")
    with col4:
        total_words = sum(s.word_count for s in scenes)
        est_duration = total_words / 2.5  # ~2.5 words per second
        st.metric("Est. Duration", f"{int(est_duration // 60)}:{int(est_duration % 60):02d}")

    # Batch settings - Apply to All Scenes
    with st.expander("⚙️ Batch Settings - Apply to All Scenes", expanded=False):
        batch_col1, batch_col2, batch_col3 = st.columns(3)

        with batch_col1:
            st.markdown("**Image Model**")
            image_models = {
                "Default": None,
                "Pro (4K)": "gemini-3-pro-image-preview",
                "Fast": "gemini-2.5-flash-image",
                "Imagen Ultra": "imagen-4.0-ultra-generate-001",
            }
            batch_img_model = st.selectbox(
                "Set all to:",
                options=list(image_models.keys()),
                key="batch_img_model",
                label_visibility="collapsed",
            )
            if st.button("Apply to All", key="apply_img_model_all", use_container_width=True):
                new_model = image_models[batch_img_model]
                for scene in scenes:
                    try:
                        scene.image_model = new_model
                    except Exception:
                        object.__setattr__(scene, 'image_model', new_model)
                save_movie_state()
                st.success(f"Set all scenes to {batch_img_model}")
                st.rerun()

        with batch_col2:
            st.markdown("**Video Model**")
            # Combined video model + Veo variant options (Veo 3.0 deprecated)
            video_models = {
                "Project Default": (None, None),
                "Veo 3.1 Standard (Best)": ("veo3", "veo-3.1-generate-preview"),
                "Veo 3.1 Fast": ("veo3", "veo-3.1-fast-generate-preview"),
                "WAN 2.6 (AtlasCloud)": ("wan26", None),
                "Seedance 1.5 Pro": ("seedance15", None),
            }
            batch_vid_model = st.selectbox(
                "Set all to:",
                options=list(video_models.keys()),
                key="batch_vid_model",
                label_visibility="collapsed",
            )
            if st.button("Apply to All", key="apply_vid_model_all", use_container_width=True):
                model_value, variant_value = video_models[batch_vid_model]
                for scene in scenes:
                    try:
                        scene.generation_model = model_value
                    except Exception:
                        object.__setattr__(scene, 'generation_model', model_value)
                    # Also set Veo variant if applicable
                    if variant_value is not None:
                        try:
                            scene.veo_model_variant = variant_value
                        except Exception:
                            object.__setattr__(scene, 'veo_model_variant', variant_value)
                st.session_state.visuals_dirty = True
                save_movie_state()
                st.success(f"Set all scenes to {batch_vid_model}")
                st.rerun()

        with batch_col3:
            st.markdown("**Resolution**")
            resolutions = {
                "720p (Default)": "720p",
                "1080p (Full HD)": "1080p",
                "480p (Fast)": "480p",
            }
            batch_resolution = st.selectbox(
                "Set all to:",
                options=list(resolutions.keys()),
                key="batch_resolution",
                label_visibility="collapsed",
            )
            if st.button("Apply to All", key="apply_resolution_all", use_container_width=True):
                new_res = resolutions[batch_resolution]
                for scene in scenes:
                    try:
                        scene.resolution = new_res
                    except Exception:
                        object.__setattr__(scene, 'resolution', new_res)
                save_movie_state()
                st.success(f"Set all scenes to {batch_resolution}")
                st.rerun()

        # Regenerate all video prompts
        st.markdown("---")
        regen_col1, regen_col2 = st.columns([2, 1])
        with regen_col1:
            use_img_for_all = st.checkbox(
                "Use scene images for context when regenerating prompts",
                value=True,
                key="batch_use_img_ctx",
            )
        with regen_col2:
            scenes_missing_prompts = sum(1 for s in scenes if not getattr(s, 'video_prompt', None))
            btn_label = f"🔄 Regenerate All Video Prompts ({scenes_missing_prompts} missing)"
            if st.button(btn_label, key="regen_all_video_prompts", use_container_width=True):
                progress = st.progress(0, text="Regenerating video prompts...")
                for i, scene in enumerate(scenes):
                    progress.progress((i + 1) / len(scenes), text=f"Scene {scene.index}...")
                    has_scene_image = scene.image_path and Path(scene.image_path).exists()
                    new_prompt = _regenerate_scene_prompt(
                        scene, state.script, state,
                        prompt_type="video",
                        use_image_context=use_img_for_all and has_scene_image
                    )
                    if new_prompt:
                        try:
                            scene.video_prompt = new_prompt
                        except Exception:
                            object.__setattr__(scene, 'video_prompt', new_prompt)
                save_movie_state()
                st.success("All video prompts regenerated!")
                st.rerun()

    st.markdown("---")

    # Display in rows of 3
    cols_per_row = 3
    for row_start in range(0, len(scenes), cols_per_row):
        row_scenes = scenes[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for col, scene in zip(cols, row_scenes):
            with col:
                _render_movie_scene_card(state, scene, generation_method, model_info, defaults)


@st.cache_data(ttl=30, show_spinner=False)
def _count_scene_variants(scene_index: int, image_variants_count: int) -> int:
    """Cache variant count to avoid repeated directory globbing."""
    count = image_variants_count
    # Check new location: scenes/scene_XXX/images/
    scene_img_dir = get_project_dir() / "scenes" / f"scene_{scene_index:03d}" / "images"
    if scene_img_dir.exists():
        count = max(count, len(list(scene_img_dir.glob("*.png"))) + len(list(scene_img_dir.glob("*.jpg"))))
    # Check old location: scenes/scene_XXX/ (direct children only)
    scene_folder = get_project_dir() / "scenes" / f"scene_{scene_index:03d}"
    if scene_folder.exists():
        direct_images = [f for f in scene_folder.glob("*.png") if f.parent == scene_folder]
        direct_images += [f for f in scene_folder.glob("*.jpg") if f.parent == scene_folder]
        count = max(count, len(direct_images))
    return count


def _render_movie_scene_card(state, scene, generation_method: str, model_info: dict, defaults: dict) -> None:
    """Render a single movie scene card with video/image preview and controls."""
    has_image = scene.image_path and Path(scene.image_path).exists()
    has_video = scene.video_path and Path(scene.video_path).exists()

    # Count image variants (cached)
    num_img_variants = _count_scene_variants(scene.index, len(scene.image_variants))

    # Count video variants
    video_dir = get_project_dir() / "videos"
    video_variants = list(video_dir.glob(f"scene_{scene.index:03d}*.mp4")) if video_dir.exists() else []
    num_vid_variants = len(video_variants)

    # Status icons
    if has_video:
        status_icon = "🎬"
    elif has_image:
        status_icon = "📷"
    else:
        status_icon = "⚠️"

    # Scene header with variant counts
    variant_info = []
    if num_img_variants > 1:
        variant_info.append(f"{num_img_variants}📷")
    if num_vid_variants > 1:
        variant_info.append(f"{num_vid_variants}🎬")
    variant_str = f" ({', '.join(variant_info)})" if variant_info else ""

    st.markdown(f"**Scene {scene.index}** {status_icon}{variant_str}")
    duration = scene.get_clip_duration(generation_method)
    title_text = scene.title or scene.direction.setting[:30] if scene.direction.setting else "Untitled"
    st.caption(f"{title_text} ({duration}s)")

    # Show video if available, otherwise image
    # Use tabs to switch between image and video when both exist (like Scenes page)
    if has_video and has_image:
        img_tab, vid_tab = st.tabs(["🖼️ Image", "🎬 Video"])
        with img_tab:
            st.image(str(scene.image_path), width="stretch")
        with vid_tab:
            st.video(str(scene.video_path))
    elif has_video:
        st.video(str(scene.video_path))
    elif has_image:
        st.image(str(scene.image_path), width="stretch")
    else:
        st.info("No image yet", icon="📷")

    # === VARIANT PICKERS (always visible when variants exist) ===
    if num_vid_variants > 1 or num_img_variants > 1:
        var_col1, var_col2 = st.columns(2)

        # Video variant picker
        with var_col1:
            if num_vid_variants > 1:
                video_names = {str(v.resolve()): v.stem.replace(f"scene_{scene.index:03d}", "").lstrip("_") or "v1"
                               for v in sorted(video_variants)}
                current_vid = str(Path(scene.video_path).resolve()) if scene.video_path and Path(scene.video_path).exists() else ""
                vid_options = list(video_names.keys())
                vid_idx = vid_options.index(current_vid) if current_vid in vid_options else 0
                new_vid = st.selectbox(
                    f"🎬 Video ({num_vid_variants})",
                    options=vid_options,
                    index=vid_idx,
                    format_func=lambda x: video_names.get(x, Path(x).stem),
                    key=f"pick_vid_{scene.index}",
                )
                if new_vid != current_vid and new_vid:
                    scene.video_path = new_vid
                    save_movie_state()

        # Image variant picker
        with var_col2:
            if num_img_variants > 1:
                # Collect all image variants - check multiple locations
                scene_img_dir = get_project_dir() / "scenes" / f"scene_{scene.index:03d}" / "images"
                scenes_dir = get_project_dir() / "scenes"
                all_img_variants = set()
                for var_path in scene.image_variants:
                    if Path(var_path).exists():
                        all_img_variants.add(str(Path(var_path).resolve()))
                if scene_img_dir.exists():
                    for img_file in list(scene_img_dir.glob("*.png")) + list(scene_img_dir.glob("*.jpg")):
                        all_img_variants.add(str(img_file.resolve()))
                # Also check scenes directory directly (batch generation)
                if scenes_dir.exists():
                    for img_file in list(scenes_dir.glob(f"scene_{scene.index:03d}*.png")) + list(scenes_dir.glob(f"scene_{scene.index:03d}*.jpg")):
                        all_img_variants.add(str(img_file.resolve()))
                # Check per-scene folder directly (old format: scenes/scene_XXX/scene_XXX.png)
                scene_folder = get_project_dir() / "scenes" / f"scene_{scene.index:03d}"
                if scene_folder.exists():
                    for img_file in list(scene_folder.glob("*.png")) + list(scene_folder.glob("*.jpg")):
                        if img_file.parent == scene_folder:  # Only direct children, not from images/
                            all_img_variants.add(str(img_file.resolve()))
                if has_image:
                    all_img_variants.add(str(Path(scene.image_path).resolve()))

                img_list = sorted(all_img_variants)
                img_names = {p: Path(p).stem[-10:] for p in img_list}  # Last 10 chars of name
                current_img = str(Path(scene.image_path).resolve()) if has_image else ""
                img_idx = img_list.index(current_img) if current_img in img_list else 0

                new_img = st.selectbox(
                    f"📷 Image ({num_img_variants})",
                    options=img_list,
                    index=img_idx,
                    format_func=lambda x: img_names.get(x, Path(x).stem),
                    key=f"pick_img_{scene.index}",
                )
                if new_img != current_img and new_img:
                    try:
                        scene.image_path = new_img
                    except Exception:
                        object.__setattr__(scene, 'image_path', new_img)
                    save_movie_state()

    # === SETTINGS ROW (Image Model + Aspect Ratio) ===
    set_col1, set_col2 = st.columns(2)
    with set_col1:
        image_models = {
            None: "Default",
            "gemini-3-pro-image-preview": "Pro 4K",
            "gemini-2.5-flash-image": "Flash",
            "imagen-4.0-ultra-generate-001": "Imagen",
        }
        scene_img_model = getattr(scene, 'image_model', None)
        img_model_keys = list(image_models.keys())
        img_model_idx = img_model_keys.index(scene_img_model) if scene_img_model in img_model_keys else 0
        new_img_model = st.selectbox(
            "Model",
            options=img_model_keys,
            index=img_model_idx,
            format_func=lambda x: image_models.get(x, str(x)),
            key=f"card_img_model_{scene.index}",
        )
        if new_img_model != scene_img_model:
            try:
                scene.image_model = new_img_model
            except Exception:
                object.__setattr__(scene, 'image_model', new_img_model)
            save_movie_state()

    with set_col2:
        aspect_ratios = {None: "16:9", "16:9": "16:9", "9:16": "9:16", "1:1": "1:1", "4:3": "4:3"}
        current_aspect = getattr(scene, 'image_aspect_ratio', None)
        aspect_keys = list(aspect_ratios.keys())
        aspect_idx = aspect_keys.index(current_aspect) if current_aspect in aspect_keys else 0
        new_aspect = st.selectbox(
            "Ratio",
            options=aspect_keys,
            index=aspect_idx,
            format_func=lambda x: aspect_ratios.get(x, str(x)),
            key=f"card_img_ratio_{scene.index}",
        )
        if new_aspect != current_aspect:
            try:
                scene.image_aspect_ratio = new_aspect
            except Exception:
                object.__setattr__(scene, 'image_aspect_ratio', new_aspect)
            save_movie_state()

    # === GENERATE BUTTONS ===
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        icon = "🔄" if has_image else "🎨"
        if st.button(f"{icon} Image", key=f"gen_img_{scene.index}", use_container_width=True):
            _generate_single_scene_image(scene, state.script, state)
            st.rerun()
    with btn_col2:
        if num_img_variants > 1:
            if st.button("📂 Browse", key=f"browse_vars_{scene.index}", use_container_width=True):
                st.session_state[f"show_full_image_manager_{scene.index}"] = True
                st.rerun()

    # Show full image manager popup if requested
    if st.session_state.get(f"show_full_image_manager_{scene.index}", False):
        with st.container(border=True):
            _render_scene_image_manager(scene, state.script, state, compact=False)
            if st.button("✕ Close Manager", key=f"close_full_mgr_{scene.index}"):
                st.session_state[f"show_full_image_manager_{scene.index}"] = False
                st.rerun()

    # Video model selector (for video modes)
    if generation_method != "tts_images":
        model_options = {
            "Project Default": None,
            "Veo 3.1 (Google)": "veo3",
            "WAN 2.6 (AtlasCloud)": "wan26",
            "Seedance 1.5 Pro": "seedance15",
            "Seedance Fast": "seedance_fast",
        }
        current_model = getattr(scene, 'generation_model', None)
        current_idx = list(model_options.values()).index(current_model) if current_model in model_options.values() else 0

        new_model_label = st.selectbox(
            "Video Model",
            options=list(model_options.keys()),
            index=current_idx,
            key=f"v_model_{scene.index}",
            label_visibility="collapsed",
        )
        new_model = model_options[new_model_label]

        if new_model != current_model:
            try:
                scene.generation_model = new_model
            except Exception:
                object.__setattr__(scene, 'generation_model', new_model)
            st.session_state.visuals_dirty = True
            save_movie_state()

    # Video prompt regeneration (with image context option)
    if generation_method != "tts_images":
        current_prompt = getattr(scene, 'video_prompt', None)
        has_video_prompt = bool(current_prompt and current_prompt.strip())
        has_scene_image = scene.image_path and Path(scene.image_path).exists()

        # Expander title shows prompt status
        prompt_status = "✅" if has_video_prompt else "⚠️ Missing"
        with st.expander(f"📝 Video Prompt {prompt_status}", expanded=not has_video_prompt):
            # Show current video prompt (truncated but more visible)
            if current_prompt:
                # Show more of the prompt for visibility
                display_len = 200 if len(current_prompt) > 200 else len(current_prompt)
                st.markdown(f"```\n{current_prompt[:display_len]}{'...' if len(current_prompt) > display_len else ''}\n```")
            else:
                st.warning("No video prompt set - click Generate Prompt below", icon="⚠️")

            use_img_ctx = st.checkbox(
                "Use image context",
                value=has_scene_image,
                key=f"card_img_ctx_{scene.index}",
                help="Analyze scene image to generate better animation prompts"
            )

            btn_type = "primary" if not has_video_prompt else "secondary"
            btn_label = "🔄 Regenerate Prompt" if has_video_prompt else "✨ Generate Prompt"
            if has_scene_image and use_img_ctx:
                btn_label += " (with image)"

            if st.button(btn_label, key=f"card_regen_vprompt_{scene.index}",
                        type=btn_type, use_container_width=True):
                with st.spinner("Generating video prompt..."):
                    new_prompt = _regenerate_scene_prompt(
                        scene, state.script, state,
                        prompt_type="video",
                        use_image_context=use_img_ctx and has_scene_image
                    )
                    if new_prompt:
                        try:
                            scene.video_prompt = new_prompt
                        except Exception:
                            object.__setattr__(scene, 'video_prompt', new_prompt)
                        save_movie_state()
                        # Store the new prompt value and increment version to force widget refresh
                        st.session_state[f"video_prompt_value_{scene.index}"] = new_prompt
                        st.session_state[f"video_prompt_regenerated_{scene.index}"] = True
                        st.session_state[f"video_prompt_version_{scene.index}"] = st.session_state.get(f"video_prompt_version_{scene.index}", 0) + 1
                        st.success(f"Video prompt updated!")
                        st.rerun()
                    else:
                        st.error("Failed to generate prompt - check logs")

    # Video continuity settings (for scenes after #1)
    scene_model = getattr(scene, 'generation_model', None) or generation_method
    if scene.index > 1 and scene_model == "veo3":
        # Check if previous scene has video
        prev_scene = next((s for s in state.script.scenes if s.index == scene.index - 1), None)
        prev_has_video = prev_scene and prev_scene.video_path and Path(prev_scene.video_path).exists()

        if prev_has_video:
            use_prev_vid = st.checkbox(
                f"🔗 Use scene {scene.index - 1} video for continuity",
                value=st.session_state.get("use_prev_scene_video", False),
                key=f"use_prev_video_{scene.index}",
                help="Analyze previous scene's video to maintain character/voice consistency"
            )
            st.session_state.use_prev_scene_video = use_prev_vid
        else:
            st.caption(f"💡 Generate scene {scene.index - 1} video first for continuity")

    # Video action buttons
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        can_gen = has_image or generation_method == "veo3"
        if can_gen and st.button("🎬 Gen Video", key=f"v_gen_{scene.index}", use_container_width=True,
                                  help="Generate video for this scene"):
            _generate_single_scene_video(state, scene, generation_method, defaults)
            st.rerun()
        elif not can_gen:
            st.button("🎬 Gen Video", key=f"v_gen_{scene.index}", use_container_width=True,
                     disabled=True, help="Need image first")
    with btn_col2:
        if has_video and st.button("🔄 Regen", key=f"v_regen_{scene.index}", use_container_width=True,
                                   help="Regenerate video"):
            scene.video_path = None
            save_movie_state()
            _generate_single_scene_video(state, scene, generation_method, defaults)
            st.rerun()


def _generate_single_scene_video(state, scene, generation_method: str, defaults: dict) -> None:
    """Generate video for a single scene."""
    import time as time_module

    scene_model = getattr(scene, 'generation_model', None) or generation_method

    output_dir = get_project_dir() / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use timestamp suffix to create variants instead of overwriting
    timestamp = get_readable_timestamp()
    output_path = output_dir / f"scene_{scene.index:03d}_{timestamp}.mp4"

    duration = scene.get_clip_duration(scene_model)
    resolution = getattr(scene, 'resolution', None) or defaults.get("resolution", "720p")
    prompt = getattr(scene, 'video_prompt', None) or f"{scene.direction.setting}. {scene.direction.camera}."

    # Get video consistency option
    use_prev_video = st.session_state.get("use_prev_scene_video", False)

    # Find previous scene's video for continuity (Veo3 and WAN 2.6 V2V)
    previous_video = None
    if use_prev_video and scene.index > 1:
        prev_scene = next((s for s in state.script.scenes if s.index == scene.index - 1), None)
        if prev_scene and prev_scene.video_path and Path(prev_scene.video_path).exists():
            previous_video = Path(prev_scene.video_path)
            logger.info(f"Using previous scene video as reference: {previous_video.name}")

    with st.status(f"Generating scene {scene.index} with {scene_model}...", expanded=True) as status:
        try:
            if scene_model == "veo3":
                from src.services.veo3_generator import Veo3Generator
                from src.config import config as app_config
                # Use per-scene Veo variant if set, otherwise fall back to config/default
                veo_model = getattr(scene, 'veo_model_variant', None) or \
                           (app_config.veo_model if hasattr(app_config, 'veo_model') else "veo-3.1-generate-preview")
                generator = Veo3Generator(
                    model=veo_model,
                    resolution=resolution,
                    duration=duration,
                )
                # Get custom video prompt if set
                custom_video_prompt = getattr(scene, 'video_prompt', None)
                result = generator.generate_scene(
                    scene=scene,
                    script=state.script,
                    output_path=output_path,
                    style=state.config.visual_style if state.config else state.script.visual_style,
                    custom_prompt=custom_video_prompt,
                    previous_video=previous_video,
                    use_video_continuity=use_prev_video and previous_video is not None,
                )
            else:
                # WAN or Seedance
                from src.services.atlascloud_animator import AtlasCloudAnimator, WanModel, SeedanceModel
                visual_style = state.config.visual_style if state.config else state.script.visual_style

                if scene_model == "seedance15":
                    animator = AtlasCloudAnimator(model=SeedanceModel.IMAGE_TO_VIDEO)
                    result = animator.animate_scene(
                        image_path=Path(scene.image_path),
                        prompt=prompt,
                        output_path=output_path,
                        duration_seconds=duration,
                        resolution=resolution,
                        visual_style=visual_style,
                    )
                elif scene_model in ("seedance_fast", "seedance_fast_i2v"):
                    animator = AtlasCloudAnimator(model=SeedanceModel.IMAGE_TO_VIDEO_FAST)
                    result = animator.animate_scene(
                        image_path=Path(scene.image_path),
                        prompt=prompt,
                        output_path=output_path,
                        duration_seconds=duration,
                        resolution=resolution,
                        visual_style=visual_style,
                    )
                elif previous_video and use_prev_video:
                    # WAN 2.6 V2V mode
                    animator = AtlasCloudAnimator(model=WanModel.VIDEO_TO_VIDEO)
                    result = animator.animate_scene(
                        image_path=Path(scene.image_path),
                        prompt=prompt,
                        output_path=output_path,
                        duration_seconds=duration,
                        resolution=resolution,
                        source_video=previous_video,
                        visual_style=visual_style,
                        guidance_scale=state.config.wan_guidance_scale if state.config else None,
                        flow_shift=state.config.wan_flow_shift if state.config else None,
                        inference_steps=state.config.wan_inference_steps if state.config else None,
                        shot_type=state.config.wan_shot_type if state.config else None,
                        seed=state.config.wan_seed if state.config else 0,
                    )
                else:
                    # WAN 2.6 I2V mode
                    animator = AtlasCloudAnimator(model=WanModel.IMAGE_TO_VIDEO)
                    result = animator.animate_scene(
                        image_path=Path(scene.image_path),
                        prompt=prompt,
                        output_path=output_path,
                        duration_seconds=duration,
                        resolution=resolution,
                        visual_style=visual_style,
                        guidance_scale=state.config.wan_guidance_scale if state.config else None,
                        flow_shift=state.config.wan_flow_shift if state.config else None,
                        inference_steps=state.config.wan_inference_steps if state.config else None,
                        shot_type=state.config.wan_shot_type if state.config else None,
                        seed=state.config.wan_seed if state.config else 0,
                    )

            if result:
                scene.video_path = result
                save_movie_state()
                status.update(label=f"Scene {scene.index} complete!", state="complete")
            else:
                status.update(label=f"Scene {scene.index} failed", state="error")

        except Exception as e:
            import traceback
            status.update(label=f"Error: {e}", state="error")
            st.error(f"Video generation failed: {e}")
            st.code(traceback.format_exc())


def _generate_all_movie_scenes(state, config, use_char_refs: bool, use_scene_continuity: bool, use_v2v: bool, portraits: list) -> None:
    """Generate videos for all scenes that don't have one, respecting per-scene model settings."""
    import time as time_module
    from src.services.veo3_generator import Veo3Generator
    from src.services.atlascloud_animator import AtlasCloudAnimator, WanModel, SeedanceModel

    project_default = config.generation_method if config else "veo3"

    output_dir = get_project_dir() / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter scenes that need videos
    scenes_to_generate = [
        s for s in state.script.scenes
        if not (s.video_path and Path(s.video_path).exists())
    ]

    if not scenes_to_generate:
        st.success("All scenes already have videos!")
        return

    # Initialize generators lazily
    veo_generator = None
    wan_animator = None
    seedance_animator = None
    last_generated_video = None  # Track for video continuity

    progress_bar = st.progress(0, text="Starting generation...")
    total = len(scenes_to_generate)
    generated_count = 0

    for i, scene in enumerate(scenes_to_generate):
        scene_model = getattr(scene, 'generation_model', None) or project_default
        model_names = {"veo3": "Veo 3.1", "wan26": "WAN 2.6", "seedance15": "Seedance 1.5"}
        v2v_info = " (v2v)" if use_v2v and last_generated_video and scene_model == "veo3" else ""
        progress_bar.progress((i / total), text=f"Scene {scene.index} ({model_names.get(scene_model, scene_model)}{v2v_info})...")

        # Check image requirement for WAN/Seedance
        if scene_model in ["wan26", "seedance15"] and not (scene.image_path and Path(scene.image_path).exists()):
            st.warning(f"Scene {scene.index} has no image - skipping")
            continue

        duration = scene.get_clip_duration(scene_model)
        resolution = getattr(scene, 'resolution', None) or (
            config.veo_resolution if scene_model == "veo3" else
            config.wan_resolution if scene_model == "wan26" else
            config.seedance_resolution if scene_model == "seedance15" else "720p"
        ) if config else "720p"
        prompt = getattr(scene, 'video_prompt', None) or f"{scene.direction.setting}. {scene.direction.camera}."

        # Use timestamp suffix to create variants instead of overwriting
        timestamp = get_readable_timestamp()
        output_path = output_dir / f"scene_{scene.index:03d}_{timestamp}.mp4"

        # Get previous video for continuity (from last generated or previous scene)
        previous_video = None
        if use_v2v and scene_model == "veo3":
            if last_generated_video and last_generated_video.exists():
                previous_video = last_generated_video
                logger.info(f"Using last generated video for v2v continuity: {previous_video.name}")
            elif scene.index > 1:
                prev_scene = next((s for s in state.script.scenes if s.index == scene.index - 1), None)
                if prev_scene and prev_scene.video_path and Path(prev_scene.video_path).exists():
                    previous_video = Path(prev_scene.video_path)
                    logger.info(f"Using previous scene video for v2v continuity: {previous_video.name}")

        try:
            result = None
            if scene_model == "veo3":
                if veo_generator is None:
                    veo_generator = Veo3Generator(
                        model=config.veo_model if config else "veo-3.1-generate-preview",
                        resolution=resolution,
                        duration=duration,
                    )
                result = veo_generator.generate_scene(
                    scene=scene,
                    script=state.script,
                    output_path=output_path,
                    style=config.visual_style if config else state.script.visual_style,
                    custom_prompt=getattr(scene, 'video_prompt', None),
                    reference_images=portraits[:3] if use_char_refs and portraits else None,
                    first_frame=Path(scene.image_path) if use_scene_continuity and scene.image_path else None,
                    previous_video=previous_video,
                    use_video_continuity=use_v2v and previous_video is not None,
                )
            elif scene_model == "wan26":
                if wan_animator is None:
                    wan_animator = AtlasCloudAnimator(model=WanModel.IMAGE_TO_VIDEO)
                result = wan_animator.animate_scene(
                    image_path=Path(scene.image_path),
                    prompt=prompt,
                    output_path=output_path,
                    duration_seconds=duration,
                    resolution=resolution,
                    visual_style=config.visual_style if config else None,
                    guidance_scale=config.wan_guidance_scale if config else None,
                    flow_shift=config.wan_flow_shift if config else None,
                    inference_steps=config.wan_inference_steps if config else None,
                    shot_type=config.wan_shot_type if config else None,
                    seed=config.wan_seed if config else 0,
                )
            elif scene_model == "seedance15":
                if seedance_animator is None:
                    seedance_animator = AtlasCloudAnimator(model=SeedanceModel.IMAGE_TO_VIDEO)
                # Check if lip sync is enabled
                audio_path = None
                lip_sync = getattr(scene, 'enable_lip_sync', None)
                if lip_sync is None:
                    lip_sync = config.seedance_lip_sync if config else True
                if lip_sync and scene.dialogue:
                    for d in scene.dialogue:
                        if d.audio_path and Path(d.audio_path).exists():
                            audio_path = Path(d.audio_path)
                            break
                # Generate video first (without lip sync)
                result = seedance_animator.animate_scene(
                    image_path=Path(scene.image_path),
                    prompt=prompt,
                    output_path=output_path,
                    duration_seconds=duration,
                    resolution=resolution,
                )
                # Apply lip sync as post-processing if audio available
                if result and audio_path:
                    lipsync_output = output_path.parent / f"lipsync_{output_path.name}"
                    lipsync_result = seedance_animator.apply_lipsync(
                        video_path=Path(result),
                        audio_path=audio_path,
                        output_path=lipsync_output,
                    )
                    if lipsync_result:
                        # Replace original with lip-synced version
                        import shutil
                        shutil.move(str(lipsync_result), str(output_path))
                        result = output_path
            elif scene_model in ("seedance_fast", "seedance_fast_i2v"):
                # Seedance Fast (quicker, lower quality)
                seedance_fast_animator = AtlasCloudAnimator(model=SeedanceModel.IMAGE_TO_VIDEO_FAST)
                # Check if lip sync is enabled
                audio_path = None
                lip_sync = getattr(scene, 'enable_lip_sync', None)
                if lip_sync is None:
                    lip_sync = config.seedance_lip_sync if config else True
                if lip_sync and scene.dialogue:
                    for d in scene.dialogue:
                        if d.audio_path and Path(d.audio_path).exists():
                            audio_path = Path(d.audio_path)
                            break
                # Generate video first (without lip sync)
                result = seedance_fast_animator.animate_scene(
                    image_path=Path(scene.image_path),
                    prompt=prompt,
                    output_path=output_path,
                    duration_seconds=duration,
                    resolution=resolution,
                )
                # Apply lip sync as post-processing if audio available
                if result and audio_path:
                    lipsync_output = output_path.parent / f"lipsync_{output_path.name}"
                    lipsync_result = seedance_fast_animator.apply_lipsync(
                        video_path=Path(result),
                        audio_path=audio_path,
                        output_path=lipsync_output,
                    )
                    if lipsync_result:
                        # Replace original with lip-synced version
                        import shutil
                        shutil.move(str(lipsync_result), str(output_path))
                        result = output_path

            if result:
                scene.video_path = result
                generated_count += 1
                # Track this video for continuity in next scene
                if isinstance(result, Path):
                    last_generated_video = result
                else:
                    last_generated_video = Path(result)

        except Exception as e:
            st.error(f"Scene {scene.index} failed: {e}")

    progress_bar.progress(1.0, text="Complete!")
    st.success(f"Generated {generated_count}/{total} videos!")
    save_movie_state()
    st.rerun()


# Legacy functions kept for backward compatibility
def _render_mixed_model_generation(state, config, use_char_refs: bool, use_scene_continuity: bool, use_v2v: bool, portraits: list) -> None:
    """Legacy: Render video generation with per-scene model selection."""
    import time as time_module
    from src.services.veo3_generator import Veo3Generator
    from src.services.atlascloud_animator import AtlasCloudAnimator, WanModel, SeedanceModel

    project_default = config.generation_method if config else "veo3"
    model_counts = {"veo3": 0, "wan26": 0, "seedance15": 0}
    for scene in state.script.scenes:
        model = getattr(scene, 'generation_model', None) or project_default
        if model in model_counts:
            model_counts[model] += 1

    model_names = {"veo3": "Veo 3.1", "wan26": "WAN 2.6", "seedance15": "Seedance 1.5"}
    counts = " | ".join(f"{model_names[m]}: {c}" for m, c in model_counts.items() if c > 0)
    st.info(f"**Per-scene models:** {counts}")

    if st.button("🎬 Generate All Scenes", type="primary", use_container_width=True):
        output_dir = get_project_dir() / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize generators lazily
        veo_generator = None
        wan_animator = None
        seedance_animator = None

        progress_bar = st.progress(0, text="Generating scenes...")
        total_scenes = len(state.script.scenes)
        generated_videos = []
        previous_video = None

        for i, scene in enumerate(state.script.scenes):
            # Determine model for this scene
            scene_model = getattr(scene, 'generation_model', None) or project_default
            progress_bar.progress((i / total_scenes), text=f"Scene {i + 1}/{total_scenes} ({scene_model})...")

            # Check if scene has an image (required for WAN/Seedance)
            if scene_model in ["wan26", "seedance15"]:
                if not scene.image_path or not Path(scene.image_path).exists():
                    st.warning(f"Scene {i + 1} has no image. Skipping...")
                    continue

            # Get per-scene settings - use get_clip_duration() for auto calculation
            duration = scene.get_clip_duration(scene_model)  # Auto-calculates based on dialogue
            resolution = scene.resolution

            # Use timestamp suffix to create variants instead of overwriting
            timestamp = get_readable_timestamp()
            output_path = output_dir / f"scene_{scene.index:03d}_{timestamp}.mp4"

            try:
                if scene_model == "veo3":
                    # Use Veo 3.1
                    if veo_generator is None:
                        veo_generator = Veo3Generator(
                            model=config.veo_model if config else "veo-3.1-generate-preview",
                            resolution=resolution or (config.veo_resolution if config else "720p"),
                            duration=duration,
                        )

                    result = veo_generator.generate_scene(
                        scene=scene,
                        script=state.script,
                        output_path=output_path,
                        style=config.visual_style if config else state.script.visual_style,
                        custom_prompt=getattr(scene, 'video_prompt', None),
                        reference_images=portraits[:3] if use_char_refs and portraits else None,
                        first_frame=Path(scene.image_path) if use_scene_continuity and scene.image_path else None,
                        previous_video=previous_video if use_v2v and previous_video and previous_video.exists() else None,
                        use_video_continuity=use_v2v and previous_video is not None and previous_video.exists(),
                    )

                elif scene_model == "wan26":
                    # Use WAN 2.6
                    if wan_animator is None:
                        wan_animator = AtlasCloudAnimator(model=WanModel.IMAGE_TO_VIDEO)

                    res = resolution or (config.wan_resolution if config else "720p")
                    prompt = getattr(scene, 'video_prompt', None) or f"{scene.direction.setting}. {scene.direction.camera}."

                    source_video = previous_video if use_v2v and previous_video and previous_video.exists() else None

                    result = wan_animator.animate_scene(
                        image_path=Path(scene.image_path),
                        prompt=prompt,
                        output_path=output_path,
                        duration_seconds=duration,
                        resolution=res,
                        source_video=source_video,
                        first_frame=Path(scene.image_path) if use_scene_continuity else None,
                        visual_style=config.visual_style if config else None,
                        guidance_scale=config.wan_guidance_scale if config else None,
                        flow_shift=config.wan_flow_shift if config else None,
                        inference_steps=config.wan_inference_steps if config else None,
                        shot_type=config.wan_shot_type if config else None,
                        seed=config.wan_seed if config else 0,
                    )

                elif scene_model == "seedance15":
                    # Use Seedance 1.5 Pro
                    if seedance_animator is None:
                        seedance_animator = AtlasCloudAnimator(model=SeedanceModel.IMAGE_TO_VIDEO)

                    res = resolution or (config.seedance_resolution if config else "720p")
                    prompt = getattr(scene, 'video_prompt', None) or f"{scene.direction.setting}. {scene.direction.camera}."

                    # Check if lip sync is enabled
                    audio_path = None
                    lip_sync = scene.enable_lip_sync if scene.enable_lip_sync is not None else (config.seedance_lip_sync if config else True)
                    if lip_sync and scene.dialogue:
                        for d in scene.dialogue:
                            if d.audio_path and Path(d.audio_path).exists():
                                audio_path = Path(d.audio_path)
                                break

                    source_video = previous_video if use_v2v and previous_video and previous_video.exists() else None

                    # Generate video first (without lip sync)
                    result = seedance_animator.animate_scene(
                        image_path=Path(scene.image_path),
                        prompt=prompt,
                        output_path=output_path,
                        duration_seconds=duration,
                        resolution=res,
                        source_video=source_video,
                        first_frame=Path(scene.image_path) if use_scene_continuity else None,
                    )
                    # Apply lip sync as post-processing if audio available
                    if result and audio_path:
                        lipsync_output = output_path.parent / f"lipsync_{output_path.name}"
                        lipsync_result = seedance_animator.apply_lipsync(
                            video_path=Path(result),
                            audio_path=audio_path,
                            output_path=lipsync_output,
                        )
                        if lipsync_result:
                            import shutil
                            shutil.move(str(lipsync_result), str(output_path))
                            result = output_path

                elif scene_model in ("seedance_fast", "seedance_fast_i2v"):
                    # Use Seedance Fast (quicker, lower quality)
                    seedance_fast_animator = AtlasCloudAnimator(model=SeedanceModel.IMAGE_TO_VIDEO_FAST)

                    res = resolution or (config.seedance_resolution if config else "720p")
                    prompt = getattr(scene, 'video_prompt', None) or f"{scene.direction.setting}. {scene.direction.camera}."

                    # Check if lip sync is enabled
                    audio_path = None
                    lip_sync = scene.enable_lip_sync if scene.enable_lip_sync is not None else (config.seedance_lip_sync if config else True)
                    if lip_sync and scene.dialogue:
                        for d in scene.dialogue:
                            if d.audio_path and Path(d.audio_path).exists():
                                audio_path = Path(d.audio_path)
                                break

                    source_video = previous_video if use_v2v and previous_video and previous_video.exists() else None

                    # Generate video first (without lip sync)
                    result = seedance_fast_animator.animate_scene(
                        image_path=Path(scene.image_path),
                        prompt=prompt,
                        output_path=output_path,
                        duration_seconds=duration,
                        resolution=res,
                        source_video=source_video,
                        first_frame=Path(scene.image_path) if use_scene_continuity else None,
                    )
                    # Apply lip sync as post-processing if audio available
                    if result and audio_path:
                        lipsync_output = output_path.parent / f"lipsync_{output_path.name}"
                        lipsync_result = seedance_fast_animator.apply_lipsync(
                            video_path=Path(result),
                            audio_path=audio_path,
                            output_path=lipsync_output,
                        )
                        if lipsync_result:
                            import shutil
                            shutil.move(str(lipsync_result), str(output_path))
                            result = output_path
                else:
                    result = None

                if result:
                    scene.video_path = result
                    generated_videos.append(result)
                    previous_video = result

            except Exception as e:
                st.error(f"Scene {i + 1} ({scene_model}) failed: {e}")

        progress_bar.progress(1.0, text="Generation complete!")
        st.success(f"Generated {len(generated_videos)} video clips!")

        # Show generated videos
        for i, video_path in enumerate(generated_videos):
            with st.expander(f"Scene {i + 1}", expanded=i == 0):
                st.video(str(video_path))

        # Save state
        save_movie_state()


def _render_veo3_generation(state, config, use_char_refs: bool, use_scene_continuity: bool, portraits: list) -> None:
    """Render Veo 3.1 video generation."""
    from src.services.veo3_generator import Veo3Generator

    if st.button("🎬 Generate All Scenes with Veo 3.1", type="primary", use_container_width=True):
        default_duration = config.veo_duration if config else 8
        default_resolution = config.veo_resolution if config else "720p"

        generator = Veo3Generator(
            model=config.veo_model if config else "veo-3.1-generate-preview",
            resolution=default_resolution,
            duration=default_duration,
        )
        output_dir = get_project_dir() / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)

        progress_bar = st.progress(0, text="Generating scenes with Veo 3.1...")

        def progress_callback(msg, progress):
            progress_bar.progress(progress, text=msg)

        try:
            # Use character portraits as references
            ref_images = portraits[:3] if use_char_refs else None

            videos = generator.generate_all_scenes(
                script=state.script,
                output_dir=output_dir,
                style=state.config.visual_style if state.config else state.script.visual_style,
                use_character_references=use_char_refs,
                use_scene_continuity=use_scene_continuity,
                progress_callback=progress_callback,
            )

            progress_bar.progress(1.0, text="Generation complete!")
            st.success(f"Generated {len(videos)} video clips with dialogue!")

            # Show generated videos
            for i, video_path in enumerate(videos):
                with st.expander(f"Scene {i + 1}", expanded=i == 0):
                    st.video(str(video_path))

            # Save state
            save_movie_state()

        except Exception as e:
            st.error(f"Veo 3.1 generation failed: {e}")


def _render_wan26_generation(state, config, use_char_refs: bool, use_scene_continuity: bool, use_v2v: bool, portraits: list) -> None:
    """Render WAN 2.6 video generation."""
    import time as time_module
    from src.services.atlascloud_animator import AtlasCloudAnimator, WanModel

    if st.button("🎥 Generate All Scenes with WAN 2.6", type="primary", use_container_width=True):
        default_duration = config.wan_duration if config else 10
        default_resolution = config.wan_resolution if config else "720p"

        animator = AtlasCloudAnimator(model=WanModel.IMAGE_TO_VIDEO)
        output_dir = get_project_dir() / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)

        progress_bar = st.progress(0, text="Generating scenes with WAN 2.6...")
        total_scenes = len(state.script.scenes)
        generated_videos = []
        previous_video = None

        for i, scene in enumerate(state.script.scenes):
            progress_bar.progress((i / total_scenes), text=f"Generating scene {i + 1}/{total_scenes}...")

            # Check if scene has an image
            if not scene.image_path or not Path(scene.image_path).exists():
                st.warning(f"Scene {i + 1} has no image. Skipping...")
                continue

            # Get per-scene settings or use defaults (use getattr for backward compatibility)
            duration = getattr(scene, 'clip_duration', None) or default_duration
            resolution = getattr(scene, 'resolution', None) or default_resolution

            # Build prompt from video_prompt or fallback to scene direction
            prompt = getattr(scene, 'video_prompt', None) or f"{scene.direction.setting}. {scene.direction.camera}."

            # Use timestamp suffix to create variants instead of overwriting
            timestamp = get_readable_timestamp()
            output_path = output_dir / f"scene_{scene.index:03d}_{timestamp}.mp4"

            # Determine input for video-to-video continuity
            source_video = None
            if use_v2v and previous_video and previous_video.exists():
                source_video = previous_video

            try:
                result = animator.animate_scene(
                    image_path=Path(scene.image_path),
                    prompt=prompt,
                    output_path=output_path,
                    duration_seconds=duration,
                    resolution=resolution,
                    source_video=source_video,
                    first_frame=Path(scene.image_path) if use_scene_continuity else None,
                    visual_style=config.visual_style if config else None,
                    guidance_scale=config.wan_guidance_scale if config else None,
                    flow_shift=config.wan_flow_shift if config else None,
                    inference_steps=config.wan_inference_steps if config else None,
                    shot_type=config.wan_shot_type if config else None,
                    seed=config.wan_seed if config else 0,
                )

                if result:
                    scene.video_path = result
                    generated_videos.append(result)
                    previous_video = result

            except Exception as e:
                st.error(f"Scene {i + 1} failed: {e}")

        progress_bar.progress(1.0, text="Generation complete!")
        st.success(f"Generated {len(generated_videos)} video clips!")

        # Show generated videos
        for i, video_path in enumerate(generated_videos):
            with st.expander(f"Scene {i + 1}", expanded=i == 0):
                st.video(str(video_path))

        # Save state
        save_movie_state()


def _render_seedance_generation(state, config, use_char_refs: bool, use_scene_continuity: bool, use_v2v: bool, portraits: list) -> None:
    """Render Seedance 1.5 Pro video generation."""
    import time as time_module
    from src.services.atlascloud_animator import AtlasCloudAnimator, SeedanceModel

    if st.button("💃 Generate All Scenes with Seedance 1.5 Pro", type="primary", use_container_width=True):
        default_duration = config.seedance_duration if config else 8
        default_resolution = config.seedance_resolution if config else "720p"
        default_lip_sync = config.seedance_lip_sync if config else True

        animator = AtlasCloudAnimator(model=SeedanceModel.IMAGE_TO_VIDEO)
        output_dir = get_project_dir() / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)

        progress_bar = st.progress(0, text="Generating scenes with Seedance 1.5 Pro...")
        total_scenes = len(state.script.scenes)
        generated_videos = []
        previous_video = None

        for i, scene in enumerate(state.script.scenes):
            progress_bar.progress((i / total_scenes), text=f"Generating scene {i + 1}/{total_scenes}...")

            # Check if scene has an image
            if not scene.image_path or not Path(scene.image_path).exists():
                st.warning(f"Scene {i + 1} has no image. Skipping...")
                continue

            # Get per-scene settings or use defaults (use getattr for backward compatibility)
            duration = getattr(scene, 'clip_duration', None) or default_duration
            resolution = getattr(scene, 'resolution', None) or default_resolution
            scene_lip_sync = getattr(scene, 'enable_lip_sync', None)
            lip_sync = scene_lip_sync if scene_lip_sync is not None else default_lip_sync

            # Build prompt from video_prompt or fallback to scene direction
            prompt = getattr(scene, 'video_prompt', None) or f"{scene.direction.setting}. {scene.direction.camera}."

            # Use timestamp suffix to create variants instead of overwriting
            timestamp = get_readable_timestamp()
            output_path = output_dir / f"scene_{scene.index:03d}_{timestamp}.mp4"

            # Get audio for lip sync if enabled
            audio_path = None
            if lip_sync and scene.dialogue:
                # Find audio for this scene's dialogue
                for d in scene.dialogue:
                    if d.audio_path and Path(d.audio_path).exists():
                        audio_path = Path(d.audio_path)
                        break

            # Determine input for video-to-video continuity
            source_video = None
            if use_v2v and previous_video and previous_video.exists():
                source_video = previous_video

            try:
                # Generate video first (without lip sync)
                result = animator.animate_scene(
                    image_path=Path(scene.image_path),
                    prompt=prompt,
                    output_path=output_path,
                    duration_seconds=duration,
                    resolution=resolution,
                    source_video=source_video,
                    first_frame=Path(scene.image_path) if use_scene_continuity else None,
                )

                # Apply lip sync as post-processing if audio available
                if result and audio_path:
                    lipsync_output = output_path.parent / f"lipsync_{output_path.name}"
                    lipsync_result = animator.apply_lipsync(
                        video_path=Path(result),
                        audio_path=audio_path,
                        output_path=lipsync_output,
                    )
                    if lipsync_result:
                        import shutil
                        shutil.move(str(lipsync_result), str(output_path))
                        result = output_path

                if result:
                    scene.video_path = result
                    generated_videos.append(result)
                    previous_video = result

            except Exception as e:
                st.error(f"Scene {i + 1} failed: {e}")

        progress_bar.progress(1.0, text="Generation complete!")
        st.success(f"Generated {len(generated_videos)} video clips!")

        # Show generated videos
        for i, video_path in enumerate(generated_videos):
            with st.expander(f"Scene {i + 1}", expanded=i == 0):
                st.video(str(video_path))

        # Save state
        save_movie_state()


def _render_traditional_generation(state) -> None:
    """Render traditional image generation mode."""
    st.markdown(
        f"""
        Generate images for each scene in '{state.script.title}'.
        Character descriptions will be included for visual consistency.
        You'll need to generate TTS audio in the Voices step.
        """
    )

    if st.button("🎨 Generate All Scene Images", type="primary", use_container_width=True):
        from src.services.movie_image_generator import MovieImageGenerator

        generator = MovieImageGenerator(style=state.config.visual_style if state.config else state.script.visual_style)
        output_dir = get_project_dir() / "images"
        output_dir.mkdir(parents=True, exist_ok=True)

        progress_bar = st.progress(0, text="Generating images...")

        def progress_callback(msg, progress):
            progress_bar.progress(progress, text=msg)

        try:
            images = generator.generate_all_scenes(
                script=state.script,
                output_dir=output_dir,
                use_sequential_mode=True,
                use_character_references=True,  # Use character portraits as references
                progress_callback=progress_callback,
            )

            # Update scene.image_path for each generated image
            for i, img_path in enumerate(images):
                if i < len(state.script.scenes):
                    state.script.scenes[i].image_path = str(img_path)

            progress_bar.progress(1.0, text="Image generation complete!")
            st.success(f"Generated {len(images)} scene images!")

            # Show generated images
            cols = st.columns(3)
            for i, img_path in enumerate(images):
                with cols[i % 3]:
                    st.image(str(img_path), caption=f"Scene {i + 1}")

            # Save state
            save_movie_state()

        except Exception as e:
            st.error(f"Image generation failed: {e}")


def _get_encoding_params(quality: str) -> dict:
    """Get encoding parameters based on quality preset.

    Args:
        quality: One of "draft", "standard", "professional"

    Returns:
        Dictionary with preset, crf, and audio_bitrate settings
    """
    presets = {
        "draft": {
            "preset": "ultrafast",
            "crf": 23,
            "audio_bitrate": "128k",
            "description": "Fast preview (lower quality)"
        },
        "standard": {
            "preset": "fast",
            "crf": 18,
            "audio_bitrate": "192k",
            "description": "Good balance of quality and speed"
        },
        "professional": {
            "preset": "slow",
            "crf": 15,
            "audio_bitrate": "320k",
            "description": "Best quality (slower encoding)"
        },
    }
    return presets.get(quality, presets["standard"])


def _concatenate_with_transitions(
    clip_paths: list[Path],
    output_path: Path,
    transition_type: Optional[str] = None,
    transition_duration: float = 0.5,
    keep_audio: bool = True,
    quality: str = "standard",
    normalize_audio: bool = False,
) -> bool:
    """Concatenate video clips with optional transitions and audio preservation.

    Args:
        clip_paths: List of video clip paths to concatenate
        output_path: Output video path
        transition_type: None for hard cut, "xfade" for crossfade, "fade" for fade to black
        transition_duration: Duration of transition in seconds
        keep_audio: If True, preserve audio from video clips
        quality: Quality preset - "draft", "standard", or "professional"
        normalize_audio: If True, apply loudnorm filter to normalize audio levels

    Returns:
        True if successful, False otherwise
    """
    import subprocess
    import tempfile

    if not clip_paths:
        return False

    if len(clip_paths) == 1:
        # Single clip, just copy it
        import shutil
        shutil.copy(clip_paths[0], output_path)
        return True

    # Get encoding parameters based on quality
    enc_params = _get_encoding_params(quality)

    try:
        # Create a concat file for FFmpeg
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            concat_file = f.name
            for clip in clip_paths:
                f.write(f"file '{clip}'\n")

        if transition_type is None or transition_duration <= 0:
            # Simple concatenation without transitions
            # Use filter_complex with concat filter instead of demuxer for better first-frame handling
            inputs_cmd = []
            for clip in clip_paths:
                inputs_cmd.extend(["-i", str(clip)])

            # Build concat filter
            n = len(clip_paths)
            stream_labels = "".join(f"[{i}:v][{i}:a]" if keep_audio else f"[{i}:v]" for i in range(n))
            if keep_audio:
                if normalize_audio:
                    # Output concat audio to intermediate, then apply loudnorm
                    filter_complex = f"{stream_labels}concat=n={n}:v=1:a=1[vout][aout_raw];[aout_raw]loudnorm=I=-16:TP=-1.5:LRA=11[aout]"
                else:
                    filter_complex = f"{stream_labels}concat=n={n}:v=1:a=1[vout][aout]"
            else:
                filter_complex = f"{stream_labels}concat=n={n}:v=1:a=0[vout]"

            cmd = ["ffmpeg", "-y", *inputs_cmd, "-filter_complex", filter_complex]
            cmd.extend(["-map", "[vout]"])
            if keep_audio:
                cmd.extend(["-map", "[aout]"])
            # Use quality preset encoding parameters with consistent color space
            cmd.extend(["-c:v", "libx264", "-preset", enc_params["preset"], "-crf", str(enc_params["crf"]), "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-x264-params", "keyint=30:min-keyint=30"])
            if keep_audio:
                cmd.extend(["-c:a", "aac", "-b:a", enc_params["audio_bitrate"]])
            cmd.append(str(output_path))

            subprocess.run(cmd, check=True, capture_output=True)

        elif transition_type == "xfade":
            # Crossfade transitions using xfade filter
            # This is more complex - need to chain xfade filters
            # For now, use simpler approach with concat demuxer + fade

            # Get clip durations first
            durations = []
            for clip in clip_paths:
                probe_cmd = [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0", str(clip)
                ]
                result = subprocess.run(probe_cmd, capture_output=True, text=True)
                try:
                    durations.append(float(result.stdout.strip()))
                except ValueError:
                    durations.append(5.0)  # Default duration

            # Build xfade filter chain
            filter_parts = []
            audio_parts = []
            inputs_cmd = []

            for i, clip in enumerate(clip_paths):
                inputs_cmd.extend(["-i", str(clip)])

            # Build video xfade chain
            # Calculate offsets for video transitions (where each clip starts appearing)
            video_offsets = [0.0]  # First clip starts at 0
            for i in range(len(clip_paths) - 1):
                # Next clip starts appearing at: previous start + duration - overlap
                next_offset = video_offsets[-1] + durations[i] - transition_duration
                video_offsets.append(next_offset)

            if len(clip_paths) == 2:
                offset = durations[0] - transition_duration
                filter_parts.append(f"[0:v][1:v]xfade=transition=fade:duration={transition_duration}:offset={offset}[vout]")
            else:
                # Chain multiple xfades for video
                prev_video_label = "[0:v]"
                for i in range(1, len(clip_paths)):
                    offset = video_offsets[i]
                    out_label = f"[v{i}]" if i < len(clip_paths) - 1 else "[vout]"
                    filter_parts.append(f"{prev_video_label}[{i}:v]xfade=transition=fade:duration={transition_duration}:offset={offset}{out_label}")
                    prev_video_label = out_label

            # For audio: use adelay to position each clip at its video offset, then amix
            # This matches the exact timing of video xfade offsets
            if keep_audio:
                # Position each audio clip at its video start offset using adelay
                for i, offset_ms in enumerate([int(o * 1000) for o in video_offsets]):
                    if offset_ms > 0:
                        audio_parts.append(f"[{i}:a]adelay={offset_ms}|{offset_ms}[ad{i}]")
                    else:
                        audio_parts.append(f"[{i}:a]acopy[ad{i}]")

                # Mix all delayed audio streams together
                mix_inputs = "".join(f"[ad{i}]" for i in range(len(clip_paths)))

                # Apply loudnorm if requested
                if normalize_audio:
                    audio_parts.append(f"{mix_inputs}amix=inputs={len(clip_paths)}:dropout_transition=0:normalize=0[amix];[amix]loudnorm=I=-16:TP=-1.5:LRA=11[aout]")
                else:
                    audio_parts.append(f"{mix_inputs}amix=inputs={len(clip_paths)}:dropout_transition=0:normalize=0[aout]")

            filter_complex = ";".join(filter_parts + audio_parts)

            cmd = ["ffmpeg", "-y", *inputs_cmd, "-filter_complex", filter_complex]
            cmd.extend(["-map", "[vout]"])
            if keep_audio and audio_parts:
                cmd.extend(["-map", "[aout]"])
            cmd.extend(["-c:v", "libx264", "-preset", enc_params["preset"], "-crf", str(enc_params["crf"]), "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"])
            if keep_audio:
                cmd.extend(["-c:a", "aac", "-b:a", enc_params["audio_bitrate"]])
            cmd.append(str(output_path))

            subprocess.run(cmd, check=True, capture_output=True)

        elif transition_type == "fade":
            # Fade to black transitions
            # Each clip fades out to black at end, next clip fades in from black at start
            durations = []
            for clip in clip_paths:
                probe_cmd = [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0", str(clip)
                ]
                result = subprocess.run(probe_cmd, capture_output=True, text=True)
                try:
                    durations.append(float(result.stdout.strip()))
                except ValueError:
                    durations.append(5.0)

            inputs_cmd = []
            for clip in clip_paths:
                inputs_cmd.extend(["-i", str(clip)])

            # Apply fade in/out to each clip
            filter_parts = []
            fade_duration = min(transition_duration, 1.0)

            for i, dur in enumerate(durations):
                fade_out_start = max(0, dur - fade_duration)
                if i == 0:
                    # First clip: only fade out
                    filter_parts.append(f"[{i}:v]fade=t=out:st={fade_out_start}:d={fade_duration}[v{i}]")
                elif i == len(clip_paths) - 1:
                    # Last clip: only fade in
                    filter_parts.append(f"[{i}:v]fade=t=in:st=0:d={fade_duration}[v{i}]")
                else:
                    # Middle clips: both fade in and out
                    filter_parts.append(f"[{i}:v]fade=t=in:st=0:d={fade_duration},fade=t=out:st={fade_out_start}:d={fade_duration}[v{i}]")

            # Concatenate the faded clips
            stream_labels = "".join(f"[v{i}][{i}:a]" if keep_audio else f"[v{i}]" for i in range(len(clip_paths)))
            if keep_audio:
                if normalize_audio:
                    filter_parts.append(f"{stream_labels}concat=n={len(clip_paths)}:v=1:a=1[vout][aout_raw];[aout_raw]loudnorm=I=-16:TP=-1.5:LRA=11[aout]")
                else:
                    filter_parts.append(f"{stream_labels}concat=n={len(clip_paths)}:v=1:a=1[vout][aout]")
            else:
                filter_parts.append(f"{stream_labels}concat=n={len(clip_paths)}:v=1:a=0[vout]")

            filter_complex = ";".join(filter_parts)

            cmd = ["ffmpeg", "-y", *inputs_cmd, "-filter_complex", filter_complex]
            cmd.extend(["-map", "[vout]"])
            if keep_audio:
                cmd.extend(["-map", "[aout]"])
            cmd.extend(["-c:v", "libx264", "-preset", enc_params["preset"], "-crf", str(enc_params["crf"]), "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"])
            if keep_audio:
                cmd.extend(["-c:a", "aac", "-b:a", enc_params["audio_bitrate"]])
            cmd.append(str(output_path))

            subprocess.run(cmd, check=True, capture_output=True)

        elif transition_type == "dissolve":
            # Dissolve transitions using xfade with dissolve effect
            durations = []
            for clip in clip_paths:
                probe_cmd = [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0", str(clip)
                ]
                result = subprocess.run(probe_cmd, capture_output=True, text=True)
                try:
                    durations.append(float(result.stdout.strip()))
                except ValueError:
                    durations.append(5.0)

            inputs_cmd = []
            for clip in clip_paths:
                inputs_cmd.extend(["-i", str(clip)])

            # Calculate offsets for video transitions (where each clip starts appearing)
            video_offsets = [0.0]  # First clip starts at 0
            for i in range(len(clip_paths) - 1):
                # Next clip starts appearing at: previous start + duration - overlap
                next_offset = video_offsets[-1] + durations[i] - transition_duration
                video_offsets.append(next_offset)

            filter_parts = []
            audio_parts = []

            if len(clip_paths) == 2:
                offset = durations[0] - transition_duration
                filter_parts.append(f"[0:v][1:v]xfade=transition=dissolve:duration={transition_duration}:offset={offset}[vout]")
            else:
                # Chain multiple xfades for video (dissolve effect)
                prev_video_label = "[0:v]"
                for i in range(1, len(clip_paths)):
                    offset = video_offsets[i]
                    out_label = f"[v{i}]" if i < len(clip_paths) - 1 else "[vout]"
                    filter_parts.append(f"{prev_video_label}[{i}:v]xfade=transition=dissolve:duration={transition_duration}:offset={offset}{out_label}")
                    prev_video_label = out_label

            # For audio: use adelay to position each clip at its video offset, then amix
            # This matches the exact timing of video xfade offsets
            if keep_audio:
                # Position each audio clip at its video start offset using adelay
                for i, offset_ms in enumerate([int(o * 1000) for o in video_offsets]):
                    if offset_ms > 0:
                        audio_parts.append(f"[{i}:a]adelay={offset_ms}|{offset_ms}[ad{i}]")
                    else:
                        audio_parts.append(f"[{i}:a]acopy[ad{i}]")

                # Mix all delayed audio streams together
                mix_inputs = "".join(f"[ad{i}]" for i in range(len(clip_paths)))

                # Apply loudnorm if requested
                if normalize_audio:
                    audio_parts.append(f"{mix_inputs}amix=inputs={len(clip_paths)}:dropout_transition=0:normalize=0[amix];[amix]loudnorm=I=-16:TP=-1.5:LRA=11[aout]")
                else:
                    audio_parts.append(f"{mix_inputs}amix=inputs={len(clip_paths)}:dropout_transition=0:normalize=0[aout]")

            filter_complex = ";".join(filter_parts + audio_parts)

            cmd = ["ffmpeg", "-y", *inputs_cmd, "-filter_complex", filter_complex]
            cmd.extend(["-map", "[vout]"])
            if keep_audio and audio_parts:
                cmd.extend(["-map", "[aout]"])
            cmd.extend(["-c:v", "libx264", "-preset", enc_params["preset"], "-crf", str(enc_params["crf"]), "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"])
            if keep_audio:
                cmd.extend(["-c:a", "aac", "-b:a", enc_params["audio_bitrate"]])
            cmd.append(str(output_path))

            subprocess.run(cmd, check=True, capture_output=True)

        else:
            # Default to simple concat for unsupported transitions
            # Use filter_complex for better first-frame handling (same as None transition)
            inputs_cmd = []
            for clip in clip_paths:
                inputs_cmd.extend(["-i", str(clip)])

            n = len(clip_paths)
            stream_labels = "".join(f"[{i}:v][{i}:a]" if keep_audio else f"[{i}:v]" for i in range(n))
            if keep_audio:
                filter_complex = f"{stream_labels}concat=n={n}:v=1:a=1[vout][aout]"
            else:
                filter_complex = f"{stream_labels}concat=n={n}:v=1:a=0[vout]"

            cmd = ["ffmpeg", "-y", *inputs_cmd, "-filter_complex", filter_complex]
            cmd.extend(["-map", "[vout]"])
            if keep_audio:
                cmd.extend(["-map", "[aout]"])
            cmd.extend(["-c:v", "libx264", "-preset", enc_params["preset"], "-crf", str(enc_params["crf"]), "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-x264-params", "keyint=30:min-keyint=30"])
            if keep_audio:
                cmd.extend(["-c:a", "aac", "-b:a", enc_params["audio_bitrate"]])
            cmd.append(str(output_path))

            subprocess.run(cmd, check=True, capture_output=True)

        # Cleanup temp file
        Path(concat_file).unlink(missing_ok=True)
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg concatenation failed: {e.stderr.decode() if e.stderr else e}")
        return False
    except Exception as e:
        logger.error(f"Concatenation failed: {e}")
        return False


def _mix_audio_tracks(
    base_audio_path: Optional[Path],
    tracks: list,
    output_path: Path,
    video_duration: float,
) -> Path:
    """Mix multiple audio tracks with support for loop, fade, and positioning.

    Args:
        base_audio_path: Optional base audio (e.g., TTS dialogue) to mix with tracks
        tracks: List of AudioTrack objects to mix
        output_path: Path to save the mixed audio
        video_duration: Total video duration (for looping calculation)

    Returns:
        Path to the mixed audio file
    """
    import subprocess

    if not tracks:
        # No tracks to mix, return base audio path or create silence
        if base_audio_path and base_audio_path.exists():
            return base_audio_path
        # Create silent audio
        silence_cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=r=44100:cl=stereo:d={video_duration}",
            "-c:a", "aac",
            str(output_path)
        ]
        subprocess.run(silence_cmd, check=True, capture_output=True)
        return output_path

    # Build complex filter for multi-track mixing
    inputs_cmd = []
    filter_parts = []
    track_labels = []

    input_idx = 0

    # Add base audio as first input if provided
    if base_audio_path and base_audio_path.exists():
        inputs_cmd.extend(["-i", str(base_audio_path)])
        filter_parts.append(f"[{input_idx}:a]acopy[base]")
        track_labels.append("[base]")
        input_idx += 1

    # Process each audio track
    for i, track in enumerate(tracks):
        if not track.file_path or not Path(track.file_path).exists():
            continue

        inputs_cmd.extend(["-i", str(track.file_path)])
        track_label = f"[track{i}]"

        # Build filter chain for this track
        track_filters = []

        # Calculate how long this track needs to be
        track_duration = track.duration or video_duration
        needed_duration = video_duration - track.start_time

        # Step 1: Loop if needed
        if track.loop and track_duration < needed_duration:
            # Calculate number of loops needed
            loops_needed = int(needed_duration / track_duration) + 2
            track_filters.append(f"aloop=loop={loops_needed}:size={int(track_duration * 44100)}")
            # Trim to exact needed duration
            track_filters.append(f"atrim=0:{needed_duration}")
            track_filters.append("asetpts=PTS-STARTPTS")

        # Step 2: Apply volume
        if track.volume != 1.0:
            track_filters.append(f"volume={track.volume}")

        # Step 3: Apply fade in
        if track.fade_in > 0:
            track_filters.append(f"afade=t=in:st=0:d={track.fade_in}")

        # Step 4: Apply fade out (calculate from end of track)
        if track.fade_out > 0:
            if track.loop:
                # For looped tracks, fade out at end of video
                fade_start = max(0, needed_duration - track.fade_out)
            else:
                # For non-looped, fade out at end of track
                fade_start = max(0, min(track_duration, needed_duration) - track.fade_out)
            track_filters.append(f"afade=t=out:st={fade_start}:d={track.fade_out}")

        # Step 5: Delay to start position
        if track.start_time > 0:
            delay_ms = int(track.start_time * 1000)
            track_filters.append(f"adelay={delay_ms}|{delay_ms}")

        # Combine all filters for this track
        if track_filters:
            filter_str = ",".join(track_filters)
            filter_parts.append(f"[{input_idx}:a]{filter_str}{track_label}")
        else:
            filter_parts.append(f"[{input_idx}:a]acopy{track_label}")

        track_labels.append(track_label)
        input_idx += 1

    # Mix all tracks together
    if len(track_labels) == 1:
        # Only one track, just use it directly
        final_label = track_labels[0]
        filter_parts[-1] = filter_parts[-1].replace(track_labels[0], "[aout]")
    else:
        # Multiple tracks - use amix
        mix_inputs = "".join(track_labels)
        filter_parts.append(f"{mix_inputs}amix=inputs={len(track_labels)}:dropout_transition=0:normalize=0[aout]")

    filter_complex = ";".join(filter_parts)

    # Build and run FFmpeg command
    cmd = [
        "ffmpeg", "-y",
        *inputs_cmd,
        "-filter_complex", filter_complex,
        "-map", "[aout]",
        "-ac", "2",
        "-ar", "44100",
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Mixed {len(tracks)} audio tracks to {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Audio mixing failed: {e.stderr.decode() if e.stderr else e}")
        # Fallback: return base audio or first track
        if base_audio_path and base_audio_path.exists():
            return base_audio_path
        if tracks and tracks[0].file_path:
            return Path(tracks[0].file_path)
        raise


def _concatenate_with_context_aware_transitions(
    clip_paths: list[Path],
    output_path: Path,
    scenes: list,
    base_duration: float = 0.5,
    keep_audio: bool = True,
    quality: str = "standard",
    normalize_audio: bool = False,
) -> bool:
    """Concatenate video clips with context-aware per-scene transitions.

    Uses cinematography planning to determine optimal transition type and
    duration for each scene based on emotional intensity and narrative role.

    Args:
        clip_paths: List of video clip paths to concatenate
        output_path: Output video path
        scenes: List of MovieScene objects (for context analysis)
        base_duration: Base transition duration (actual varies by context)
        keep_audio: If True, preserve audio from video clips
        quality: Quality preset - "draft", "standard", or "professional"
        normalize_audio: If True, apply loudnorm filter

    Returns:
        True if successful, False otherwise
    """
    import subprocess
    from src.services.cinematography import (
        generate_cinematography_plan,
        generate_transition_plan,
    )

    if not clip_paths:
        return False

    if len(clip_paths) == 1:
        import shutil
        shutil.copy(clip_paths[0], output_path)
        return True

    enc_params = _get_encoding_params(quality)

    try:
        # Generate cinematography plan and transition plan
        cinematography_plan = generate_cinematography_plan(scenes)
        transition_plan = generate_transition_plan(cinematography_plan, base_duration)

        # Apply per-scene overrides from MovieScene.transition_override
        for i, scene in enumerate(scenes):
            if i < len(transition_plan):
                trans_override = getattr(scene, 'transition_override', None)
                dur_override = getattr(scene, 'transition_duration_override', None)
                if trans_override is not None:
                    # Map override string to transition type
                    override_map = {
                        "none": (None, 0.0),
                        "crossfade": ("xfade", dur_override or base_duration),
                        "fade": ("fade", dur_override or base_duration),
                        "dissolve": ("dissolve", dur_override or base_duration),
                    }
                    if trans_override in override_map:
                        transition_plan[i] = override_map[trans_override]
                    elif dur_override:
                        # Just duration override, keep transition type
                        trans_type, _ = transition_plan[i]
                        transition_plan[i] = (trans_type, dur_override)

        # Get clip durations
        durations = []
        for clip in clip_paths:
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0", str(clip)
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            try:
                durations.append(float(result.stdout.strip()))
            except ValueError:
                durations.append(5.0)

        # Build filter graph with per-scene transitions
        inputs_cmd = []
        for clip in clip_paths:
            inputs_cmd.extend(["-i", str(clip)])

        # Calculate video offsets based on per-scene transition durations
        video_offsets = [0.0]
        for i in range(len(clip_paths) - 1):
            trans_type, trans_dur = transition_plan[i]
            # For hard cuts (None), no overlap
            overlap = trans_dur if trans_type else 0.0
            next_offset = video_offsets[-1] + durations[i] - overlap
            video_offsets.append(next_offset)

        # Build video filter chain
        filter_parts = []
        audio_parts = []

        # For video: chain transitions between clips
        prev_video_label = "[0:v]"
        for i in range(1, len(clip_paths)):
            trans_type, trans_dur = transition_plan[i - 1]
            offset = video_offsets[i]
            out_label = f"[v{i}]" if i < len(clip_paths) - 1 else "[vout]"

            if trans_type is None:
                # Hard cut - just concat at this point (handled via timing)
                # For hard cuts, we need to use concat filter instead
                # This is complex with mixed transitions, so treat None as very short xfade
                trans_type = "xfade"
                trans_dur = 0.05  # Near-instant crossfade for hard cut effect

            # Map transition types to xfade effects
            xfade_effect = {
                "xfade": "fade",
                "fade": "fade",
                "dissolve": "dissolve",
            }.get(trans_type, "fade")

            filter_parts.append(
                f"{prev_video_label}[{i}:v]xfade=transition={xfade_effect}:duration={trans_dur}:offset={offset}{out_label}"
            )
            prev_video_label = out_label

        # For audio: use adelay to position each clip at its video offset, then amix
        if keep_audio:
            for i, offset_ms in enumerate([int(o * 1000) for o in video_offsets]):
                if offset_ms > 0:
                    audio_parts.append(f"[{i}:a]adelay={offset_ms}|{offset_ms}[ad{i}]")
                else:
                    audio_parts.append(f"[{i}:a]acopy[ad{i}]")

            mix_inputs = "".join(f"[ad{i}]" for i in range(len(clip_paths)))
            if normalize_audio:
                audio_parts.append(
                    f"{mix_inputs}amix=inputs={len(clip_paths)}:dropout_transition=0:normalize=0[amix];[amix]loudnorm=I=-16:TP=-1.5:LRA=11[aout]"
                )
            else:
                audio_parts.append(
                    f"{mix_inputs}amix=inputs={len(clip_paths)}:dropout_transition=0:normalize=0[aout]"
                )

        filter_complex = ";".join(filter_parts + audio_parts)

        cmd = ["ffmpeg", "-y", *inputs_cmd, "-filter_complex", filter_complex]
        cmd.extend(["-map", "[vout]"])
        if keep_audio and audio_parts:
            cmd.extend(["-map", "[aout]"])
        cmd.extend([
            "-c:v", "libx264",
            "-preset", enc_params["preset"],
            "-crf", str(enc_params["crf"]),
            "-pix_fmt", "yuv420p",
            "-colorspace", "bt709",
            "-color_primaries", "bt709",
            "-color_trc", "bt709"
        ])
        if keep_audio:
            cmd.extend(["-c:a", "aac", "-b:a", enc_params["audio_bitrate"]])
        cmd.append(str(output_path))

        subprocess.run(cmd, check=True, capture_output=True)

        logger.info(f"Context-aware transitions applied: {len(transition_plan)} transitions")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg context-aware concatenation failed: {e.stderr.decode() if e.stderr else e}")
        return False
    except Exception as e:
        logger.error(f"Context-aware concatenation failed: {e}")
        return False


def render_render_page() -> None:
    """Render the final video rendering page."""
    state = get_movie_state()

    if not state.script:
        st.warning("Please create a script first.")
        return

    st.subheader("Render Final Video")
    st.markdown(
        f"""
        Combine all generated assets into the final video for '{state.script.title}'.
        """
    )

    # Render settings
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        resolution = st.selectbox(
            "Resolution",
            options=["1080p", "2K", "4K"],
            index=0,
        )
        fps = st.selectbox(
            "Frame Rate",
            options=[24, 30, 60],
            index=1,
        )
        # Quality preset
        quality_preset = st.selectbox(
            "Quality",
            options=["Draft", "Standard", "Professional"],
            index=1,  # Default to Standard
            help="Draft: Fast preview. Standard: Good balance. Professional: Best quality (slow)."
        )
    with col2:
        # Transition settings
        transition_type = st.selectbox(
            "Transition",
            options=["None", "Crossfade", "Fade to Black", "Dissolve", "Context-aware"],
            index=1,  # Default to Crossfade
            help="Transition effect between scenes. Context-aware picks transitions based on scene emotion."
        )
        if transition_type == "Context-aware":
            st.caption("📊 Transitions vary by scene emotion and narrative role")
            transition_duration = st.slider(
                "Base Duration (s)",
                min_value=0.3,
                max_value=1.5,
                value=0.5,
                step=0.1,
                help="Base transition duration (varies by context)"
            )
        elif transition_type != "None":
            transition_duration = st.slider(
                "Duration (s)",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1,
                help="Transition duration in seconds"
            )
        else:
            transition_duration = 0.0
    # Determine generation mode early for conditional UI
    gen_method = state.config.generation_method if state.config else "tts_images"
    is_tts_mode = gen_method == "tts_images"

    with col3:
        show_subtitles = st.checkbox("Show Subtitles", value=True)
        normalize_audio = st.checkbox(
            "Normalize Audio",
            value=False,
            help="Apply loudness normalization for consistent audio levels (recommended for final export)"
        )
        # Only show video audio option in video mode
        if not is_tts_mode:
            keep_video_audio = st.checkbox(
                "Keep Video Audio",
                value=True,
                help="Preserve audio from generated videos (Veo, WAN, Seedance)"
            )
        else:
            keep_video_audio = False  # No video audio in TTS mode
    with col4:
        # Only show lip sync in TTS mode (doesn't apply to pre-generated videos)
        if is_tts_mode:
            use_lip_sync = st.checkbox(
                "Lip Sync Animation",
                value=False,
                help="Animate character faces with lip sync (slower, uses Wan2.2-S2V)"
            )
            if use_lip_sync:
                lip_sync_resolution = st.selectbox(
                    "Lip Sync Quality",
                    options=["480P", "720P"],
                    index=0,
                    help="Higher quality = longer processing time"
                )
        else:
            use_lip_sync = False
            st.caption("ℹ️ Using pre-generated videos")

    # Audio Timeline Editor
    st.markdown("### 🎵 Audio Timeline")

    # Calculate estimated video duration
    video_duration = sum(s.duration for s in state.script.scenes if s.duration > 0)
    if video_duration <= 0:
        video_duration = len(state.script.scenes) * 5.0  # Estimate 5s per scene
    st.caption(f"Estimated video duration: {video_duration:.1f}s")

    # Initialize audio tracks if needed
    from src.models.schemas import AudioTrack
    import uuid

    # Track colors for visual distinction
    track_colors = ["#4A90D9", "#50C878", "#FFB347", "#FF6B6B", "#9B59B6", "#3498DB"]

    # Verify audio track file paths and attempt recovery if files moved
    audio_dir = get_project_dir() / "audio"
    tracks_updated = False
    for track in state.audio_tracks:
        if track.file_path:
            file_path = Path(track.file_path)
            if not file_path.exists():
                # Try to find the file in the current project's audio folder
                # The file might be named "track_{id}_{original_name}"
                if audio_dir.exists():
                    # Search by track id
                    matching_files = list(audio_dir.glob(f"track_{track.id}_*"))
                    if matching_files:
                        # Found a matching file - update the path
                        track.file_path = str(matching_files[0])
                        tracks_updated = True
                    else:
                        # Also try matching by original filename
                        original_name = file_path.name
                        if (audio_dir / original_name).exists():
                            track.file_path = str(audio_dir / original_name)
                            tracks_updated = True

    if tracks_updated:
        save_movie_state()

    with st.expander("📼 Audio Tracks", expanded=len(state.audio_tracks) > 0):
        # Add new track button
        if st.button("➕ Add Audio Track", key="add_audio_track"):
            new_track = AudioTrack(
                id=str(uuid.uuid4())[:8],
                name=f"Track {len(state.audio_tracks) + 1}",
                color=track_colors[len(state.audio_tracks) % len(track_colors)]
            )
            state.audio_tracks.append(new_track)
            save_movie_state()
            st.rerun()

        if not state.audio_tracks:
            st.info("No audio tracks. Click 'Add Audio Track' to add background music or sound effects.")
        else:
            # Display each track with controls
            tracks_to_remove = []
            for idx, track in enumerate(state.audio_tracks):
                with st.container():
                    st.markdown(f"---")
                    track_col1, track_col2 = st.columns([3, 1])

                    with track_col1:
                        # Track name
                        new_name = st.text_input(
                            "Track Name",
                            value=track.name,
                            key=f"track_name_{track.id}",
                            label_visibility="collapsed"
                        )
                        if new_name != track.name:
                            track.name = new_name
                            save_movie_state()

                    with track_col2:
                        if st.button("🗑️ Remove", key=f"remove_track_{track.id}"):
                            tracks_to_remove.append(idx)

                    # Check if track already has a file
                    has_existing_file = track.file_path and Path(track.file_path).exists()

                    if has_existing_file:
                        # Show existing file info prominently
                        file_name = Path(track.file_path).name
                        duration_str = f" ({track.duration:.1f}s)" if track.duration else ""
                        st.info(f"🎵 **{file_name}**{duration_str}")

                        # Show replace button
                        replace_key = f"replace_audio_{track.id}"
                        if replace_key not in st.session_state:
                            st.session_state[replace_key] = False

                        if st.session_state[replace_key]:
                            # Show uploader for replacement
                            uploaded_file = st.file_uploader(
                                f"Select new audio for {track.name}",
                                type=["mp3", "wav", "m4a", "aac", "ogg"],
                                key=f"track_file_replace_{track.id}",
                                label_visibility="collapsed"
                            )
                            col_cancel, col_space = st.columns([1, 3])
                            with col_cancel:
                                if st.button("Cancel", key=f"cancel_replace_{track.id}"):
                                    st.session_state[replace_key] = False
                                    st.rerun()
                        else:
                            uploaded_file = None
                            if st.button("🔄 Replace Audio", key=f"show_replace_{track.id}"):
                                st.session_state[replace_key] = True
                                st.rerun()
                    else:
                        # No file or file missing
                        if track.file_path:
                            # File path was saved but file is missing
                            st.warning(f"⚠️ Audio file not found: {Path(track.file_path).name}")
                        # Show uploader
                        uploaded_file = st.file_uploader(
                            f"Audio File for {track.name}",
                            type=["mp3", "wav", "m4a", "aac", "ogg"],
                            key=f"track_file_{track.id}",
                            label_visibility="collapsed"
                        )

                    # Handle new file upload (both new and replacement)
                    if uploaded_file:
                        # Save uploaded file to project directory
                        audio_dir = get_project_dir() / "audio"
                        audio_dir.mkdir(parents=True, exist_ok=True)
                        file_path = audio_dir / f"track_{track.id}_{uploaded_file.name}"
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        track.file_path = str(file_path)

                        # Get audio duration using ffprobe
                        try:
                            import subprocess
                            probe_cmd = [
                                "ffprobe", "-v", "error",
                                "-show_entries", "format=duration",
                                "-of", "csv=p=0", str(file_path)
                            ]
                            result = subprocess.run(probe_cmd, capture_output=True, text=True)
                            track.duration = float(result.stdout.strip())
                        except Exception:
                            track.duration = None

                        # Clear replace mode if it was active
                        replace_key = f"replace_audio_{track.id}"
                        if replace_key in st.session_state:
                            st.session_state[replace_key] = False

                        save_movie_state()
                        st.success(f"✅ Saved: {uploaded_file.name}")
                        st.rerun()

                    if has_existing_file:
                        # Track controls in columns
                        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)

                        with ctrl_col1:
                            new_start = st.number_input(
                                "Start (s)",
                                min_value=0.0,
                                max_value=max(0.0, video_duration - 1),
                                value=float(track.start_time),
                                step=0.5,
                                key=f"track_start_{track.id}",
                                help="When to start playing this track"
                            )
                            if new_start != track.start_time:
                                track.start_time = new_start
                                save_movie_state()

                        with ctrl_col2:
                            new_volume = st.slider(
                                "Volume",
                                min_value=0.0,
                                max_value=1.0,
                                value=float(track.volume),
                                step=0.05,
                                key=f"track_volume_{track.id}"
                            )
                            if new_volume != track.volume:
                                track.volume = new_volume
                                save_movie_state()

                        with ctrl_col3:
                            new_loop = st.checkbox(
                                "🔁 Loop",
                                value=track.loop,
                                key=f"track_loop_{track.id}",
                                help="Loop this track to fill video duration"
                            )
                            if new_loop != track.loop:
                                track.loop = new_loop
                                save_movie_state()

                        with ctrl_col4:
                            fade_col1, fade_col2 = st.columns(2)
                            with fade_col1:
                                new_fade_in = st.number_input(
                                    "Fade In",
                                    min_value=0.0,
                                    max_value=5.0,
                                    value=float(track.fade_in),
                                    step=0.5,
                                    key=f"track_fadein_{track.id}"
                                )
                                if new_fade_in != track.fade_in:
                                    track.fade_in = new_fade_in
                                    save_movie_state()
                            with fade_col2:
                                new_fade_out = st.number_input(
                                    "Fade Out",
                                    min_value=0.0,
                                    max_value=5.0,
                                    value=float(track.fade_out),
                                    step=0.5,
                                    key=f"track_fadeout_{track.id}"
                                )
                                if new_fade_out != track.fade_out:
                                    track.fade_out = new_fade_out
                                    save_movie_state()

                        # Visual timeline bar for this track
                        if track.duration and track.duration > 0:
                            # Calculate track end time (with looping consideration)
                            if track.loop:
                                track_end = video_duration
                            else:
                                track_end = min(track.start_time + track.duration, video_duration)

                            # Calculate percentages for CSS
                            start_pct = (track.start_time / video_duration) * 100 if video_duration > 0 else 0
                            width_pct = ((track_end - track.start_time) / video_duration) * 100 if video_duration > 0 else 0

                            st.markdown(
                                f"""
                                <div style="
                                    background: #333;
                                    height: 24px;
                                    border-radius: 4px;
                                    position: relative;
                                    margin: 8px 0;
                                ">
                                    <div style="
                                        position: absolute;
                                        left: {start_pct}%;
                                        width: {width_pct}%;
                                        height: 100%;
                                        background: {track.color};
                                        border-radius: 4px;
                                        opacity: 0.8;
                                    "></div>
                                    <span style="
                                        position: absolute;
                                        left: 50%;
                                        transform: translateX(-50%);
                                        color: white;
                                        font-size: 11px;
                                        line-height: 24px;
                                    ">{track.name}: {track.start_time:.1f}s - {track_end:.1f}s{' 🔁' if track.loop else ''}</span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

            # Remove tracks marked for deletion
            if tracks_to_remove:
                for idx in sorted(tracks_to_remove, reverse=True):
                    state.audio_tracks.pop(idx)
                save_movie_state()
                st.rerun()

        # Timeline overview (video + all audio tracks)
        if state.audio_tracks:
            st.markdown("#### Timeline Overview")

            # Video bar
            st.markdown(
                f"""
                <div style="margin-bottom: 4px;">
                    <span style="font-size: 12px; color: #888;">Video ({video_duration:.1f}s)</span>
                    <div style="
                        background: linear-gradient(90deg, #666 0%, #888 50%, #666 100%);
                        height: 20px;
                        border-radius: 4px;
                    "></div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Audio track bars
            for track in state.audio_tracks:
                if track.file_path and track.duration:
                    if track.loop:
                        track_end = video_duration
                    else:
                        track_end = min(track.start_time + track.duration, video_duration)

                    start_pct = (track.start_time / video_duration) * 100 if video_duration > 0 else 0
                    width_pct = ((track_end - track.start_time) / video_duration) * 100 if video_duration > 0 else 0

                    st.markdown(
                        f"""
                        <div style="margin-bottom: 4px;">
                            <span style="font-size: 12px; color: #888;">{track.name}</span>
                            <div style="
                                background: #333;
                                height: 16px;
                                border-radius: 4px;
                                position: relative;
                            ">
                                <div style="
                                    position: absolute;
                                    left: {start_pct}%;
                                    width: {width_pct}%;
                                    height: 100%;
                                    background: {track.color};
                                    border-radius: 4px;
                                    opacity: 0.7;
                                "></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    # Asset check
    st.markdown("### Asset Check")

    # Count scenes with videos (check both scene.video_path and video directory)
    video_dir = get_project_dir(create_if_missing=False) / "videos"
    scenes_with_videos = 0
    scenes_with_images_only = 0

    for s in state.script.scenes:
        has_video = s.video_path and Path(s.video_path).exists()

        # Also check video directory for scene videos (auto-detect)
        if not has_video and video_dir.exists():
            video_matches = list(video_dir.glob(f"scene_{s.index:03d}*.mp4"))
            if video_matches:
                # Update scene.video_path to first match
                s.video_path = str(sorted(video_matches)[0])
                has_video = True

        if has_video:
            scenes_with_videos += 1
        elif s.image_path and Path(s.image_path).exists():
            scenes_with_images_only += 1

    scenes_ready = scenes_with_videos + scenes_with_images_only
    total_scenes = len(state.script.scenes)

    if is_tts_mode:
        # TTS mode: check for images and voice clips
        dialogues_with_audio = sum(
            1 for s in state.script.scenes
            for d in s.dialogue
            if d.audio_path and Path(d.audio_path).exists()
        )

        col1, col2 = st.columns(2)
        with col1:
            if scenes_with_images_only == total_scenes:
                st.success(f"✅ All {scenes_with_images_only} scene images ready")
            else:
                st.warning(f"⚠️ {scenes_with_images_only}/{total_scenes} scene images")

        with col2:
            total_dialogue = state.script.total_dialogue_count
            if dialogues_with_audio == total_dialogue:
                st.success(f"✅ All {dialogues_with_audio} voice clips ready")
            else:
                st.warning(f"⚠️ {dialogues_with_audio}/{total_dialogue} voice clips")
    else:
        # Video mode: check for videos
        if scenes_with_videos == total_scenes:
            st.success(f"✅ All {scenes_with_videos} scene videos ready")
        elif scenes_ready == total_scenes:
            st.info(f"ℹ️ {scenes_with_videos} videos + {scenes_with_images_only} images (will use Ken Burns)")
        else:
            missing = total_scenes - scenes_ready
            st.warning(f"⚠️ {scenes_with_videos} videos, {scenes_with_images_only} images, {missing} missing")

    # Render button
    st.markdown("---")

    if st.button("🎬 Render Video", type="primary", use_container_width=True):
        import subprocess
        import tempfile
        from datetime import datetime
        from src.services.video_generator import VideoGenerator
        from src.models.schemas import KenBurnsEffect

        video_gen = VideoGenerator()
        output_dir = get_project_dir() / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect valid audio tracks from timeline
        audio_tracks_to_mix = [
            track for track in state.audio_tracks
            if track.file_path and Path(track.file_path).exists()
        ]

        # Resolution mapping
        res_map = {
            "1080p": (1920, 1080),
            "2K": (2560, 1440),
            "4K": (3840, 2160),
        }
        target_res = res_map.get(resolution, (1920, 1080))

        progress_bar = st.progress(0, text="Preparing assets...")
        status_text = st.empty()

        try:
            # Step 1: Create scene video clips
            scene_clips = []
            ken_burns_effects = [
                KenBurnsEffect.ZOOM_IN, KenBurnsEffect.PAN_RIGHT,
                KenBurnsEffect.ZOOM_OUT, KenBurnsEffect.PAN_LEFT,
                KenBurnsEffect.PAN_UP, KenBurnsEffect.PAN_DOWN,
            ]

            # Initialize lip sync animator if enabled
            lip_sync_animator = None
            if use_lip_sync:
                from src.services.lip_sync_animator import Wan2S2VAnimator
                lip_sync_animator = Wan2S2VAnimator()
                status_text.text("Creating lip-synced scene animations (this may take a while)...")

                # First, create master audio track for lip sync
                audio_clips_for_sync = []
                for scene in state.script.scenes:
                    for dialogue in scene.dialogue:
                        if dialogue.audio_path and Path(dialogue.audio_path).exists():
                            audio_clips_for_sync.append((Path(dialogue.audio_path), dialogue.start_time or 0))

                if audio_clips_for_sync:
                    # Build master audio with FFmpeg
                    master_audio_for_sync = output_dir / f"master_audio_sync_{datetime.now().strftime('%H%M%S')}.mp3"
                    filter_parts = []
                    inputs_cmd = []

                    for idx, (audio_path, start_time) in enumerate(audio_clips_for_sync):
                        inputs_cmd.extend(["-i", str(audio_path)])
                        delay_ms = int(start_time * 1000)
                        filter_parts.append(f"[{idx}]adelay={delay_ms}|{delay_ms}[a{idx}]")

                    mix_inputs = "".join(f"[a{i}]" for i in range(len(audio_clips_for_sync)))
                    filter_parts.append(f"{mix_inputs}amix=inputs={len(audio_clips_for_sync)}:normalize=0[aout]")
                    filter_complex = ";".join(filter_parts)

                    audio_cmd = [
                        "ffmpeg", "-y", *inputs_cmd,
                        "-filter_complex", filter_complex,
                        "-map", "[aout]", "-ac", "2", "-ar", "44100",
                        str(master_audio_for_sync)
                    ]
                    subprocess.run(audio_cmd, check=True, capture_output=True)
            else:
                status_text.text("Creating scene video clips with Ken Burns effects...")

            total_scenes = len(state.script.scenes)

            # Check if we're in video mode (any scene has video) or TTS/image mode
            # Also check video directory for scene videos
            any_scene_has_video = False
            for s in state.script.scenes:
                if s.video_path and Path(s.video_path).exists():
                    any_scene_has_video = True
                    break
                # Check directory for video files
                video_matches = list(output_dir.glob(f"scene_{s.index:03d}*.mp4"))
                video_matches = [v for v in video_matches if "_render" not in v.name]
                if video_matches:
                    any_scene_has_video = True
                    break

            for i, scene in enumerate(state.script.scenes):
                # Check if scene has a pre-generated video (from storyboard or directory)
                has_video = scene.video_path and Path(scene.video_path).exists()

                # Auto-detect video from project videos directory if not set
                if not has_video:
                    video_matches = list(output_dir.glob(f"scene_{scene.index:03d}*.mp4"))
                    # Filter out render intermediates
                    video_matches = [v for v in video_matches if "_render" not in v.name]
                    if video_matches:
                        scene.video_path = str(sorted(video_matches)[0])
                        has_video = True
                        logger.info(f"Auto-detected video for scene {scene.index}: {scene.video_path}")

                has_image = scene.image_path and Path(scene.image_path).exists()

                # In video mode: only use scenes with videos, skip others
                # In TTS/image mode: use Ken Burns on images
                if any_scene_has_video:
                    # Video mode - only include scenes that have videos
                    if has_video:
                        scene_clips.append(Path(scene.video_path))
                        status_text.text(f"Using video for scene {scene.index}")
                        progress_bar.progress(
                            0.3 * (i + 1) / total_scenes,
                            text=f"Using video for scene {scene.index}"
                        )
                    # Skip scenes without videos in video mode
                    continue

                # TTS/image mode - use Ken Burns on images
                if not has_image:
                    continue  # Skip scenes without images

                # Calculate scene duration from dialogue timing
                scene_duration = scene.end_time - scene.start_time
                if scene_duration <= 0:
                    scene_duration = 5.0

                clip_path = output_dir / f"scene_{scene.index:03d}_render.mp4"

                if use_lip_sync and lip_sync_animator and audio_clips_for_sync:
                    # Use lip sync animation
                    def lip_sync_progress(msg, prog):
                        progress_bar.progress(
                            0.1 + 0.5 * (i + prog) / total_scenes,
                            text=f"Scene {scene.index}: {msg}"
                        )

                    result = lip_sync_animator.animate_scene(
                        image_path=Path(scene.image_path),
                        audio_path=master_audio_for_sync,
                        start_time=scene.start_time,
                        duration=scene_duration,
                        output_path=clip_path,
                        resolution=lip_sync_resolution if 'lip_sync_resolution' in dir() else "480P",
                        progress_callback=lip_sync_progress,
                    )

                    if result and result.exists():
                        scene_clips.append(result)
                    else:
                        # Fallback to Ken Burns if lip sync fails
                        st.warning(f"Lip sync failed for scene {scene.index}, using Ken Burns fallback")
                        effect = ken_burns_effects[i % len(ken_burns_effects)]
                        video_gen.create_scene_clip(
                            image_path=Path(scene.image_path),
                            duration=scene_duration,
                            effect=effect,
                            output_path=clip_path,
                            resolution=target_res,
                            fps=fps,
                        )
                        scene_clips.append(clip_path)
                else:
                    # Standard Ken Burns effect
                    effect = ken_burns_effects[i % len(ken_burns_effects)]
                    video_gen.create_scene_clip(
                        image_path=Path(scene.image_path),
                        duration=scene_duration,
                        effect=effect,
                        output_path=clip_path,
                        resolution=target_res,
                        fps=fps,
                    )
                    scene_clips.append(clip_path)

                # Update progress (only for non-lip-sync, lip sync has its own progress)
                if not use_lip_sync:
                    progress_bar.progress(
                        0.3 * (i + 1) / total_scenes,
                        text=f"Created clip for scene {scene.index}"
                    )

            if not scene_clips:
                st.error("No scene clips could be created. Please generate scene images first.")
                return

            # Step 2: Concatenate scene clips with selected transition
            status_text.text("Concatenating video clips...")
            progress_bar.progress(0.4, text="Concatenating video clips...")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_only_path = output_dir / f"video_only_{timestamp}.mp4"

            # Map quality preset to parameter name (lowercase)
            quality_map = {"Draft": "draft", "Standard": "standard", "Professional": "professional"}

            if transition_type == "Context-aware":
                # Use context-aware per-scene transitions
                status_text.text("Analyzing scenes for optimal transitions...")
                _concatenate_with_context_aware_transitions(
                    clip_paths=scene_clips,
                    output_path=video_only_path,
                    scenes=state.script.scenes,
                    base_duration=transition_duration,
                    keep_audio=keep_video_audio,
                    quality=quality_map.get(quality_preset, "standard"),
                    normalize_audio=normalize_audio,
                )
            else:
                # Map transition type to FFmpeg filter
                transition_map = {
                    "None": None,
                    "Crossfade": "xfade",
                    "Fade to Black": "fade",
                    "Dissolve": "dissolve",
                }

                # Use custom concatenation with uniform transitions
                _concatenate_with_transitions(
                    clip_paths=scene_clips,
                    output_path=video_only_path,
                    transition_type=transition_map.get(transition_type),
                    transition_duration=transition_duration,
                    keep_audio=keep_video_audio,
                    quality=quality_map.get(quality_preset, "standard"),
                    normalize_audio=normalize_audio,
                )

            # Step 3: Handle audio
            # In video mode with keep_audio=True, the audio is already in the video
            # In TTS mode, we need to add the dialogue audio clips
            final_output = output_dir / f"{state.script.title.replace(' ', '_')}_{timestamp}.mp4"

            # Check if we're in TTS mode (no pre-generated videos) and have TTS audio
            audio_clips = []
            if not keep_video_audio or is_tts_mode:
                status_text.text("Processing TTS audio...")
                progress_bar.progress(0.5, text="Processing TTS audio...")

                # Collect all dialogue audio paths in order
                for scene in state.script.scenes:
                    for dialogue in scene.dialogue:
                        if dialogue.audio_path and Path(dialogue.audio_path).exists():
                            audio_clips.append((Path(dialogue.audio_path), dialogue.start_time))

            if audio_clips:
                # Create master audio with proper timing using FFmpeg
                master_audio_path = output_dir / f"master_audio_{timestamp}.mp3"

                # Build FFmpeg filter for audio mixing with delays
                # This places each audio clip at its start_time
                filter_parts = []
                inputs_cmd = []

                for idx, (audio_path, start_time) in enumerate(audio_clips):
                    inputs_cmd.extend(["-i", str(audio_path)])
                    delay_ms = int(start_time * 1000)
                    filter_parts.append(f"[{idx}]adelay={delay_ms}|{delay_ms}[a{idx}]")

                # Mix all delayed audio streams
                mix_inputs = "".join(f"[a{i}]" for i in range(len(audio_clips)))
                filter_parts.append(f"{mix_inputs}amix=inputs={len(audio_clips)}:normalize=0[aout]")

                filter_complex = ";".join(filter_parts)

                audio_cmd = [
                    "ffmpeg", "-y",
                    *inputs_cmd,
                    "-filter_complex", filter_complex,
                    "-map", "[aout]",
                    "-ac", "2",
                    "-ar", "44100",
                    str(master_audio_path)
                ]

                subprocess.run(audio_cmd, check=True, capture_output=True)

                # Mix with audio tracks from timeline if provided
                if audio_tracks_to_mix:
                    status_text.text(f"Mixing {len(audio_tracks_to_mix)} audio track(s) with dialogue...")
                    progress_bar.progress(0.65, text="Mixing audio tracks...")

                    # Build FFmpeg filter for multi-track mixing
                    mixed_audio_path = output_dir / f"mixed_audio_{timestamp}.mp3"
                    master_audio_path = _mix_audio_tracks(
                        base_audio_path=master_audio_path,
                        tracks=audio_tracks_to_mix,
                        output_path=mixed_audio_path,
                        video_duration=video_duration,
                    )

                # Step 4: Combine video and audio
                status_text.text("Combining video and audio...")
                progress_bar.progress(0.7, text="Combining video and audio...")

                final_output = output_dir / f"{state.script.title.replace(' ', '_')}_{timestamp}.mp4"

                combine_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(video_only_path),
                    "-i", str(master_audio_path),
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-shortest",
                    str(final_output)
                ]

                subprocess.run(combine_cmd, check=True, capture_output=True)
            else:
                # No TTS audio to add
                if audio_tracks_to_mix:
                    # Add audio tracks from timeline to video
                    status_text.text(f"Adding {len(audio_tracks_to_mix)} audio track(s) to video...")
                    progress_bar.progress(0.65, text="Adding audio tracks...")

                    if keep_video_audio:
                        # First mix all tracks together, then mix with video audio
                        tracks_audio_path = output_dir / f"tracks_mixed_{timestamp}.mp3"
                        _mix_audio_tracks(
                            base_audio_path=None,  # No base audio
                            tracks=audio_tracks_to_mix,
                            output_path=tracks_audio_path,
                            video_duration=video_duration,
                        )

                        # Mix tracks audio with video audio
                        combine_cmd = [
                            "ffmpeg", "-y",
                            "-i", str(video_only_path),
                            "-i", str(tracks_audio_path),
                            "-filter_complex",
                            "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=2[aout]",
                            "-map", "0:v",
                            "-map", "[aout]",
                            "-c:v", "copy",
                            "-c:a", "aac",
                            "-b:a", "192k",
                            "-shortest",
                            str(final_output)
                        ]
                    else:
                        # Use only audio tracks (no video audio)
                        tracks_audio_path = output_dir / f"tracks_mixed_{timestamp}.mp3"
                        _mix_audio_tracks(
                            base_audio_path=None,
                            tracks=audio_tracks_to_mix,
                            output_path=tracks_audio_path,
                            video_duration=video_duration,
                        )

                        combine_cmd = [
                            "ffmpeg", "-y",
                            "-i", str(video_only_path),
                            "-i", str(tracks_audio_path),
                            "-map", "0:v",
                            "-map", "1:a",
                            "-c:v", "copy",
                            "-c:a", "aac",
                            "-b:a", "192k",
                            "-shortest",
                            str(final_output)
                        ]
                    subprocess.run(combine_cmd, check=True, capture_output=True)
                    status_text.text("Audio tracks added successfully!")
                    progress_bar.progress(0.7, text="Audio mixed with video")
                elif keep_video_audio:
                    # Video already has audio, just copy/rename it
                    import shutil
                    shutil.copy(video_only_path, final_output)
                    status_text.text("Using original video audio...")
                    progress_bar.progress(0.7, text="Video with audio ready")
                else:
                    # No audio at all - just use video
                    import shutil
                    shutil.copy(video_only_path, final_output)

            # Step 5: Generate subtitles if enabled
            if show_subtitles and audio_clips:
                status_text.text("Generating subtitles...")
                progress_bar.progress(0.85, text="Generating subtitles...")

                # Create ASS subtitle file
                subtitle_path = output_dir / f"subtitles_{timestamp}.ass"
                with open(subtitle_path, "w", encoding="utf-8") as f:
                    f.write("[Script Info]\n")
                    f.write(f"Title: {state.script.title}\n")
                    f.write("ScriptType: v4.00+\n")
                    f.write(f"PlayResX: {target_res[0]}\n")
                    f.write(f"PlayResY: {target_res[1]}\n\n")

                    f.write("[V4+ Styles]\n")
                    f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
                    f.write("Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,20,20,40,1\n\n")

                    f.write("[Events]\n")
                    f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

                    for scene in state.script.scenes:
                        for dialogue in scene.dialogue:
                            if dialogue.start_time is not None and dialogue.end_time is not None:
                                char = state.script.get_character(dialogue.character_id)
                                char_name = char.name if char else dialogue.character_id

                                # Convert seconds to ASS timestamp format (H:MM:SS.CC)
                                def to_ass_time(seconds):
                                    h = int(seconds // 3600)
                                    m = int((seconds % 3600) // 60)
                                    s = int(seconds % 60)
                                    cs = int((seconds % 1) * 100)
                                    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

                                start = to_ass_time(dialogue.start_time)
                                end = to_ass_time(dialogue.end_time)
                                text = dialogue.text.replace("\n", "\\N")

                                f.write(f"Dialogue: 0,{start},{end},Default,{char_name},0,0,0,,{char_name}: {text}\n")

                # Burn subtitles into video
                from src.services.video_generator import _escape_ffmpeg_path
                subtitled_output = output_dir / f"{state.script.title.replace(' ', '_')}_subtitled_{timestamp}.mp4"
                escaped_subtitle_path = _escape_ffmpeg_path(subtitle_path)
                sub_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(final_output),
                    "-vf", f"ass='{escaped_subtitle_path}'",
                    "-c:a", "copy",
                    str(subtitled_output)
                ]

                try:
                    subprocess.run(sub_cmd, check=True, capture_output=True)
                    final_output = subtitled_output
                except subprocess.CalledProcessError:
                    st.warning("Subtitle overlay failed, using video without burned-in subtitles.")

            progress_bar.progress(1.0, text="Render complete!")
            status_text.text("")

            # Save final video path to state
            state.final_video_path = str(final_output)

            st.success(f"Video rendered successfully!")
            st.info(f"Saved to: `{final_output}`")

            # Clean up only intermediate files created during render
            # DO NOT delete original scene videos (scene_XXX.mp4) - only delete render-specific files
            if video_only_path.exists() and video_only_path != final_output:
                video_only_path.unlink()
            # Delete Ken Burns clips created during this render (they have _render suffix)
            for clip in scene_clips:
                if clip.exists() and clip != final_output and "_render" in clip.name:
                    clip.unlink()

            # Advance to complete
            if st.button("View Results →", type="primary"):
                advance_movie_step()
                st.rerun()

        except subprocess.CalledProcessError as e:
            st.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
        except Exception as e:
            st.error(f"Render failed: {e}")

    # Navigation
    st.markdown("---")
    if st.button("← Back to Visuals"):
        go_to_movie_step(MovieWorkflowStep.VISUALS)
        st.rerun()


def render_movie_complete_page() -> None:
    """Render the completion page."""
    state = get_movie_state()

    st.subheader("🎉 Your Movie is Ready!")

    if state.script:
        st.markdown(f"**Title:** {state.script.title}")
        if state.script.description:
            st.markdown(f"**Description:** {state.script.description}")

        # Calculate total duration
        if state.script.scenes:
            last_scene = state.script.scenes[-1]
            total_duration = last_scene.end_time
            minutes = int(total_duration // 60)
            seconds = int(total_duration % 60)
            st.markdown(f"**Duration:** {minutes}m {seconds}s")
            st.markdown(f"**Scenes:** {len(state.script.scenes)}")
            st.markdown(f"**Characters:** {len(state.script.characters)}")

    # Video preview
    if state.final_video_path and Path(state.final_video_path).exists():
        st.markdown("### Video Preview")
        st.video(state.final_video_path)

        # Download button
        with open(state.final_video_path, "rb") as f:
            video_bytes = f.read()

        file_name = Path(state.final_video_path).name
        st.download_button(
            label="📥 Download Video",
            data=video_bytes,
            file_name=file_name,
            mime="video/mp4",
            type="primary",
            use_container_width=True,
        )

        st.success(f"Video saved to: `{state.final_video_path}`")
    else:
        st.warning("Video not found. Please go back to Render and generate the video.")
        if st.button("← Back to Render"):
            go_to_movie_step(MovieWorkflowStep.RENDER)
            st.rerun()

    # Project assets summary
    with st.expander("📁 Project Assets", expanded=False):
        project_dir = get_project_dir()

        # Audio files
        audio_dir = project_dir / "audio"
        if audio_dir.exists():
            audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav"))
            st.markdown(f"**Voice clips:** {len(audio_files)} files in `{audio_dir}`")

        # Image files
        images_dir = project_dir / "images"
        if images_dir.exists():
            image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
            st.markdown(f"**Scene images:** {len(image_files)} files in `{images_dir}`")

        # Character portraits
        chars_dir = project_dir / "characters"
        if chars_dir.exists():
            char_files = list(chars_dir.glob("*.png")) + list(chars_dir.glob("*.jpg"))
            st.markdown(f"**Character portraits:** {len(char_files)} files in `{chars_dir}`")

        # Videos
        videos_dir = project_dir / "videos"
        if videos_dir.exists():
            video_files = list(videos_dir.glob("*.mp4"))
            st.markdown(f"**Videos:** {len(video_files)} files in `{videos_dir}`")

    st.markdown("---")

    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Edit & Re-render"):
            go_to_movie_step(MovieWorkflowStep.SCENES)
            st.rerun()
    with col2:
        if st.button("Create Another Movie", type="primary"):
            if "script_agent" in st.session_state:
                st.session_state.script_agent.reset()
            st.session_state.movie_state = MovieModeState()
            st.rerun()
