"""Movie Mode page - Create animated podcasts, educational videos, and short films."""

import streamlit as st
from pathlib import Path

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
from src.ui.components.state import render_movie_project_sidebar, save_movie_state


def get_movie_state() -> MovieModeState:
    """Get or initialize movie mode state."""
    if "movie_state" not in st.session_state:
        st.session_state.movie_state = MovieModeState()
    return st.session_state.movie_state


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
    if current_idx < len(steps) - 1:
        state.current_step = steps[current_idx + 1]
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

    # Check generation method to determine if VOICES step should be shown
    is_veo_mode = state.config and state.config.generation_method == "veo3"

    # Build steps list - skip VOICES for Veo mode
    all_steps = [
        ("⚙️", "Setup", MovieWorkflowStep.SETUP),
        ("📝", "Script", MovieWorkflowStep.SCRIPT),
        ("👥", "Characters", MovieWorkflowStep.CHARACTERS),
        ("🎬", "Scenes", MovieWorkflowStep.SCENES),
        ("🎙️", "Voices", MovieWorkflowStep.VOICES),
        ("🎨", "Visuals", MovieWorkflowStep.VISUALS),
        ("🔧", "Render", MovieWorkflowStep.RENDER),
        ("✅", "Complete", MovieWorkflowStep.COMPLETE),
    ]

    # Filter out VOICES step for Veo mode
    if is_veo_mode:
        steps = [(i, l, s) for i, l, s in all_steps if s != MovieWorkflowStep.VOICES]
    else:
        steps = all_steps

    cols = st.columns(len(steps))
    current_idx = list(MovieWorkflowStep).index(current_step)

    for i, (icon, label, step) in enumerate(steps):
        step_idx = list(MovieWorkflowStep).index(step)
        with cols[i]:
            # Make completed and current steps clickable
            if step_idx <= current_idx:
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
            "tts_images": "TTS + Images (Cheaper)",
            "veo3": "Veo 3.1 Video (High quality, Expensive)",
        }
        selected_gen = st.selectbox(
            "Generation Method",
            options=list(gen_methods.keys()),
            index=list(gen_methods.keys()).index(current_config.generation_method),
            format_func=lambda x: gen_methods[x],
            key="setup_gen_method",
        )

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
            # Veo generates its own audio
            selected_voice = "veo"  # Placeholder - not used for Veo mode

        # Veo specific settings (only show if veo3 selected)
        if selected_gen == "veo3":
            st.markdown("**Veo Settings**")

            # Model selection with grouped options
            veo_models = {
                "veo-3.1-generate-preview": "Veo 3.1 Standard (Best quality, audio)",
                "veo-3.1-fast-generate-preview": "Veo 3.1 Fast (Quicker, audio)",
                "veo-3-generate-preview": "Veo 3 Standard (High quality, audio)",
                "veo-3-fast-generate-preview": "Veo 3 Fast (Quick, audio)",
                "veo-2-generate-preview": "Veo 2 (No audio, cheaper)",
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
        else:
            # Default values when not using Veo (with backward compatibility)
            selected_veo_model = getattr(current_config, 'veo_model', 'veo-3.1-generate-preview')
            selected_veo_duration = getattr(current_config, 'veo_duration', 8)
            selected_veo_resolution = getattr(current_config, 'veo_resolution', '720p')

        # Visual style with custom option
        visual_styles = [
            "animated digital art",
            "photorealistic",
            "3D animated",
            "anime style",
            "watercolor illustration",
            "comic book style",
            "minimalist flat design",
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

        # Save config to state
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
            veo_model=selected_veo_model,
            veo_duration=selected_veo_duration,
            veo_resolution=selected_veo_resolution,
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

            # Character Portrait Section - full width below character details
            st.markdown("---")
            st.markdown("**Character Portrait**")

            # Get visual style from config (set in Setup) or script
            visual_style = (
                (state.config.visual_style if state.config else None)
                or (state.script.visual_style if state.script else None)
                or "cinematic digital art"
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
                            var_dir = config.output_dir / "movie" / "characters" / "variations"
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
                        source_dir = config.output_dir / "movie" / "characters" / "sources"
                        source_dir.mkdir(parents=True, exist_ok=True)
                        source_path = source_dir / f"character_{char.id}_source.png"

                        # Load and save source image
                        source_img = Image.open(BytesIO(source_photo.getvalue()))
                        source_img.save(source_path, format="PNG")
                        char.source_image_path = str(source_path)

                        generator = MovieImageGenerator(style=visual_style)
                        output_dir = config.output_dir / "movie" / "characters"
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
                                var_dir = config.output_dir / "movie" / "characters" / "variations"
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

                    st.image(str(portrait_path), use_container_width=True)
                    st.caption(f"AI-generated portrait ({visual_style}) - {resolution_text}")

                    # Full-size view option
                    with st.expander("🔍 View Full Size"):
                        st.image(str(portrait_path), use_container_width=True)
                        st.markdown(f"**Resolution:** {resolution_text}")
                        st.markdown(f"**File:** `{portrait_path}`")

                    # Check for existing variations on disk
                    var_dir = config.output_dir / "movie" / "characters" / "variations"
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
                            var_dir = config.output_dir / "movie" / "characters" / "variations"
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
                            output_dir = config.output_dir / "movie" / "characters"
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
                            st.image(str(current_var_path), use_container_width=True)

                            # Action buttons
                            act_col1, act_col2 = st.columns(2)
                            with act_col1:
                                if st.button("✓ Use This Portrait", key=f"use_carousel_{i}", type="primary"):
                                    import shutil
                                    main_path = config.output_dir / "movie" / "characters" / f"character_{char.id}_reference.png"
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
                        output_dir = config.output_dir / "movie" / "characters" / "variations"
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
                        st.image(current_new_path, use_container_width=True)

                        # Action buttons
                        act_col1, act_col2 = st.columns(2)
                        with act_col1:
                            if st.button("✓ Use This One", key=f"select_new_{i}", type="primary"):
                                import shutil
                                main_path = config.output_dir / "movie" / "characters" / f"character_{char.id}_reference.png"
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
                            output_dir = config.output_dir / "movie" / "characters"
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
                                    var_dir = config.output_dir / "movie" / "characters" / "variations"
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
                        output_dir = config.output_dir / "movie" / "characters"
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


def render_scenes_page() -> None:
    """Render the scene and dialogue editor page."""
    from src.models.schemas import Emotion, DialogueLine, SceneDirection

    state = get_movie_state()

    if not state.script:
        st.warning("Please create a script first.")
        return

    st.subheader("Scene & Dialogue Editor")
    st.markdown(
        """
        Review and edit your scenes, dialogue, and visual descriptions.
        Make sure everything is perfect before generating voices and visuals.
        """
    )

    script = state.script

    # Scene tabs for easy navigation
    if not script.scenes:
        st.info("No scenes in script yet. Go back and add scenes to your script.")
        if st.button("← Back to Script"):
            go_to_movie_step(MovieWorkflowStep.SCRIPT)
            st.rerun()
        return

    scene_tabs = st.tabs([f"Scene {s.index}: {s.title or 'Untitled'}" for s in script.scenes])

    for tab_idx, (tab, scene) in enumerate(zip(scene_tabs, script.scenes)):
        with tab:
            # Scene header with title edit
            col1, col2 = st.columns([3, 1])
            with col1:
                new_title = st.text_input(
                    "Scene Title",
                    value=scene.title or "",
                    key=f"scene_title_{scene.index}",
                    placeholder="e.g., INT. OFFICE - NIGHT"
                )
                if new_title != scene.title:
                    scene.title = new_title

            with col2:
                # Delete scene button (if more than 1 scene)
                if len(script.scenes) > 1:
                    if st.button("🗑️ Delete", key=f"delete_scene_{scene.index}"):
                        script.scenes.remove(scene)
                        st.rerun()

            # Scene Direction (visual settings)
            with st.expander("🎬 Scene Direction", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    new_setting = st.text_area(
                        "Setting/Location",
                        value=scene.direction.setting,
                        key=f"scene_setting_{scene.index}",
                        height=80,
                        help="Describe the location and environment"
                    )
                    if new_setting != scene.direction.setting:
                        scene.direction.setting = new_setting

                    camera_options = [
                        "wide shot", "medium shot", "close-up",
                        "extreme close-up", "over-the-shoulder", "POV",
                        "establishing shot", "two-shot"
                    ]
                    current_camera = scene.direction.camera if scene.direction.camera in camera_options else camera_options[1]
                    new_camera = st.selectbox(
                        "Camera Angle",
                        options=camera_options,
                        index=camera_options.index(current_camera),
                        key=f"scene_camera_{scene.index}"
                    )
                    if new_camera != scene.direction.camera:
                        scene.direction.camera = new_camera

                with col2:
                    new_lighting = st.text_input(
                        "Lighting",
                        value=scene.direction.lighting or "",
                        key=f"scene_lighting_{scene.index}",
                        placeholder="e.g., warm sunset glow, harsh fluorescent"
                    )
                    if new_lighting != (scene.direction.lighting or ""):
                        scene.direction.lighting = new_lighting if new_lighting else None

                    mood_options = ["neutral", "tense", "happy", "sad", "mysterious", "romantic", "action", "comedic"]
                    current_mood = scene.direction.mood if scene.direction.mood in mood_options else mood_options[0]
                    new_mood = st.selectbox(
                        "Mood",
                        options=mood_options,
                        index=mood_options.index(current_mood),
                        key=f"scene_mood_{scene.index}"
                    )
                    if new_mood != scene.direction.mood:
                        scene.direction.mood = new_mood

                # Visible characters
                all_char_ids = [c.id for c in script.characters]
                all_char_names = {c.id: c.name for c in script.characters}

                current_visible = scene.direction.visible_characters or []
                new_visible = st.multiselect(
                    "Characters in Scene",
                    options=all_char_ids,
                    default=[c for c in current_visible if c in all_char_ids],
                    format_func=lambda x: all_char_names.get(x, x),
                    key=f"scene_chars_{scene.index}"
                )
                if set(new_visible) != set(current_visible):
                    scene.direction.visible_characters = new_visible

            # Visual Prompt Preview
            with st.expander("🎨 Visual Prompt", expanded=False):
                # Build and show what the image generation prompt will look like
                prompt_parts = []
                prompt_parts.append(f"Scene: {scene.direction.setting}")
                if scene.direction.lighting:
                    prompt_parts.append(f"Lighting: {scene.direction.lighting}")
                prompt_parts.append(f"Mood: {scene.direction.mood}")
                prompt_parts.append(f"Camera: {scene.direction.camera}")

                # Add visible characters
                for char_id in scene.direction.visible_characters:
                    char = script.get_character(char_id)
                    if char:
                        prompt_parts.append(f"Character - {char.name}: {char.description}")

                auto_prompt = "\n".join(prompt_parts)

                st.text_area(
                    "Auto-generated prompt (for reference)",
                    value=auto_prompt,
                    height=120,
                    disabled=True,
                    key=f"auto_prompt_{scene.index}"
                )

                # Custom override
                new_visual_prompt = st.text_area(
                    "Custom Visual Prompt (optional - overrides auto-generated)",
                    value=scene.visual_prompt or "",
                    key=f"scene_visual_prompt_{scene.index}",
                    height=100,
                    placeholder="Leave empty to use auto-generated prompt"
                )
                if new_visual_prompt != (scene.visual_prompt or ""):
                    scene.visual_prompt = new_visual_prompt if new_visual_prompt else None

            # Dialogue Editor
            st.markdown("##### 💬 Dialogue")

            if not scene.dialogue:
                st.info("No dialogue in this scene yet.")

            for d_idx, dialogue in enumerate(scene.dialogue):
                with st.container():
                    cols = st.columns([2, 4, 2, 1])

                    with cols[0]:
                        # Character selection
                        char_options = [c.id for c in script.characters]
                        char_names = {c.id: c.name for c in script.characters}
                        current_char_idx = char_options.index(dialogue.character_id) if dialogue.character_id in char_options else 0
                        new_char = st.selectbox(
                            "Character",
                            options=char_options,
                            index=current_char_idx,
                            format_func=lambda x: char_names.get(x, x),
                            key=f"dialogue_char_{scene.index}_{d_idx}",
                            label_visibility="collapsed"
                        )
                        if new_char != dialogue.character_id:
                            dialogue.character_id = new_char

                    with cols[1]:
                        # Dialogue text
                        new_text = st.text_area(
                            "Line",
                            value=dialogue.text,
                            key=f"dialogue_text_{scene.index}_{d_idx}",
                            height=68,
                            label_visibility="collapsed"
                        )
                        if new_text != dialogue.text:
                            dialogue.text = new_text

                    with cols[2]:
                        # Emotion
                        emotion_options = [e.value for e in Emotion]
                        current_emotion_idx = emotion_options.index(dialogue.emotion.value) if dialogue.emotion.value in emotion_options else 0
                        new_emotion = st.selectbox(
                            "Emotion",
                            options=emotion_options,
                            index=current_emotion_idx,
                            key=f"dialogue_emotion_{scene.index}_{d_idx}",
                            label_visibility="collapsed"
                        )
                        if new_emotion != dialogue.emotion.value:
                            dialogue.emotion = Emotion(new_emotion)

                        # Action/stage direction
                        new_action = st.text_input(
                            "Action",
                            value=dialogue.action or "",
                            key=f"dialogue_action_{scene.index}_{d_idx}",
                            placeholder="e.g., sighs, looks away",
                            label_visibility="collapsed"
                        )
                        if new_action != (dialogue.action or ""):
                            dialogue.action = new_action if new_action else None

                    with cols[3]:
                        # Delete dialogue button
                        if st.button("🗑️", key=f"delete_dialogue_{scene.index}_{d_idx}"):
                            scene.dialogue.remove(dialogue)
                            st.rerun()

                    st.markdown("---")

            # Add new dialogue button
            if st.button("➕ Add Dialogue Line", key=f"add_dialogue_{scene.index}"):
                default_char = script.characters[0].id if script.characters else "narrator"
                new_dialogue = DialogueLine(
                    character_id=default_char,
                    text="",
                    emotion=Emotion.NEUTRAL
                )
                scene.dialogue.append(new_dialogue)
                st.rerun()

    # Add new scene
    st.markdown("---")
    if st.button("➕ Add New Scene"):
        new_scene_idx = len(script.scenes) + 1
        from src.models.schemas import MovieScene
        new_scene = MovieScene(
            index=new_scene_idx,
            title=f"Scene {new_scene_idx}",
            direction=SceneDirection(
                setting="Describe the location...",
                camera="medium shot",
                mood="neutral",
                visible_characters=[]
            ),
            dialogue=[]
        )
        script.scenes.append(new_scene)
        st.rerun()

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Characters"):
            go_to_movie_step(MovieWorkflowStep.CHARACTERS)
            st.rerun()
    with col2:
        if st.button("Continue to Voices →", type="primary"):
            advance_movie_step()
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
            output_dir = config.output_dir / "movie" / "audio"
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
    """Render the visual generation page."""
    state = get_movie_state()

    if not state.script:
        st.warning("Please create a script first.")
        return

    st.subheader("Generate Scene Visuals")

    # Check if Veo 3.1 is available
    from src.services.veo3_generator import check_veo3_available, get_veo3_pricing_estimate

    veo3_available = check_veo3_available()

    # Generation mode selection
    st.markdown("### Generation Mode")

    generation_modes = ["traditional"]
    mode_labels = {"traditional": "Traditional (Images + TTS Audio)"}

    if veo3_available:
        generation_modes.insert(0, "veo3")
        mode_labels["veo3"] = "Veo 3.1 (Video with Dialogue - Recommended)"

    generation_mode = st.radio(
        "Choose generation method:",
        options=generation_modes,
        format_func=lambda x: mode_labels.get(x, x),
        horizontal=True,
        help="Veo 3.1 generates video clips with dialogue audio directly from prompts. "
             "Traditional mode generates images and uses TTS for voices separately.",
    )

    if generation_mode == "veo3":
        st.info(
            """
            **Veo 3.1 Mode** generates video clips with:
            - Synchronized dialogue audio (characters speak their lines!)
            - Sound effects and ambient audio
            - Cinematic motion and camera movement

            No need for separate TTS or lip sync - everything is generated together.
            """
        )

        # Show cost estimate
        num_scenes = len(state.script.scenes)
        estimate = get_veo3_pricing_estimate(num_scenes, duration=8)
        st.caption(
            f"Estimated cost: **${estimate['estimated_total']:.2f}** "
            f"({num_scenes} scenes × 8 seconds × $0.75/sec)"
        )
    else:
        st.markdown(
            f"""
            Generate images for each scene in '{state.script.title}'.
            Character descriptions will be included for visual consistency.
            You'll need to generate TTS audio separately.
            """
        )

    # Visual style settings
    col1, col2 = st.columns(2)
    with col1:
        visual_style = st.text_area(
            "Visual Style",
            value=state.script.visual_style,
            height=100,
            help="Art style for all generated visuals",
        )
    with col2:
        world_desc = st.text_area(
            "World Description",
            value=state.script.world_description or "",
            height=100,
            help="Consistent setting/world across scenes",
        )

    # Update script
    state.script.visual_style = visual_style
    state.script.world_description = world_desc if world_desc else None

    # Veo 3.1 specific settings
    if generation_mode == "veo3":
        col1, col2 = st.columns(2)
        with col1:
            veo_duration = st.selectbox(
                "Clip Duration",
                options=[4, 6, 8],
                index=2,
                format_func=lambda x: f"{x} seconds",
                help="Duration of each generated clip",
            )
        with col2:
            veo_resolution = st.selectbox(
                "Resolution",
                options=["720p", "1080p"],
                index=0,
                help="1080p only available for 8-second clips",
            )

    # Scene preview
    st.markdown("### Scenes to Generate")
    for scene in state.script.scenes:
        with st.expander(f"Scene {scene.index + 1}: {scene.title or scene.direction.setting[:50]}"):
            st.markdown(f"**Setting:** {scene.direction.setting}")
            st.markdown(f"**Camera:** {scene.direction.camera}")
            st.markdown(f"**Characters:** {', '.join(scene.direction.visible_characters)}")
            if scene.dialogue:
                st.markdown(f"**Dialogue lines:** {len(scene.dialogue)}")
                for d in scene.dialogue:
                    char = state.script.get_character(d.character_id)
                    char_name = char.name if char else d.character_id
                    st.caption(f'  {char_name}: "{d.text}"')

    # Generate button
    st.markdown("---")

    if generation_mode == "veo3":
        if st.button("🎬 Generate All Scenes with Veo 3.1", type="primary", use_container_width=True):
            from src.services.veo3_generator import Veo3Generator

            generator = Veo3Generator(
                model="veo-3.1-generate-preview",
                resolution=veo_resolution,
                duration=veo_duration,
            )
            output_dir = config.output_dir / "movie" / "videos"

            progress_bar = st.progress(0, text="Generating scenes with Veo 3.1...")

            def progress_callback(msg, progress):
                progress_bar.progress(progress, text=msg)

            try:
                videos = generator.generate_all_scenes(
                    script=state.script,
                    output_dir=output_dir,
                    style=visual_style,
                    use_character_references=True,
                    progress_callback=progress_callback,
                )

                progress_bar.progress(1.0, text="Generation complete!")
                st.success(f"Generated {len(videos)} video clips with dialogue!")

                # Show generated videos
                for i, video_path in enumerate(videos):
                    with st.expander(f"Scene {i + 1}", expanded=i == 0):
                        st.video(str(video_path))

                # Skip to render step since videos already have audio
                st.info("Videos include dialogue audio. You can skip straight to rendering!")
                if st.button("Continue to Render →", type="primary"):
                    # Skip voices step since Veo 3.1 generates audio
                    go_to_movie_step(MovieWorkflowStep.RENDER)
                    st.rerun()

            except Exception as e:
                st.error(f"Veo 3.1 generation failed: {e}")

    else:
        # Traditional image generation
        if st.button("🎨 Generate All Scene Images", type="primary", use_container_width=True):
            from src.services.movie_image_generator import MovieImageGenerator

            generator = MovieImageGenerator(style=visual_style)
            output_dir = config.output_dir / "movie" / "images"

            progress_bar = st.progress(0, text="Generating images...")

            def progress_callback(msg, progress):
                progress_bar.progress(progress, text=msg)

            try:
                images = generator.generate_all_scenes(
                    script=state.script,
                    output_dir=output_dir,
                    use_sequential_mode=True,  # For consistency
                    progress_callback=progress_callback,
                )

                progress_bar.progress(1.0, text="Image generation complete!")
                st.success(f"Generated {len(images)} scene images!")

                # Show generated images
                cols = st.columns(3)
                for i, img_path in enumerate(images):
                    with cols[i % 3]:
                        st.image(str(img_path), caption=f"Scene {i + 1}")

            except Exception as e:
                st.error(f"Image generation failed: {e}")

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Voices"):
            go_to_movie_step(MovieWorkflowStep.VOICES)
            st.rerun()
    with col2:
        if st.button("Continue to Render →", type="primary"):
            advance_movie_step()
            st.rerun()


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
    col1, col2, col3 = st.columns(3)
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
    with col2:
        show_subtitles = st.checkbox("Show Subtitles", value=True)
        add_music = st.checkbox("Add Background Music", value=False)
    with col3:
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

    # Asset check
    st.markdown("### Asset Check")

    scenes_with_images = sum(
        1 for s in state.script.scenes
        if s.image_path and Path(s.image_path).exists()
    )
    dialogues_with_audio = sum(
        1 for s in state.script.scenes
        for d in s.dialogue
        if d.audio_path and Path(d.audio_path).exists()
    )

    col1, col2 = st.columns(2)
    with col1:
        if scenes_with_images == len(state.script.scenes):
            st.success(f"✅ All {scenes_with_images} scene images ready")
        else:
            st.warning(f"⚠️ {scenes_with_images}/{len(state.script.scenes)} scene images")

    with col2:
        total_dialogue = state.script.total_dialogue_count
        if dialogues_with_audio == total_dialogue:
            st.success(f"✅ All {dialogues_with_audio} voice clips ready")
        else:
            st.warning(f"⚠️ {dialogues_with_audio}/{total_dialogue} voice clips")

    # Render button
    st.markdown("---")

    if st.button("🎬 Render Video", type="primary", use_container_width=True):
        import subprocess
        import tempfile
        from datetime import datetime
        from src.services.video_generator import VideoGenerator
        from src.models.schemas import KenBurnsEffect

        video_gen = VideoGenerator()
        output_dir = config.output_dir / "movie" / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)

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
            for i, scene in enumerate(state.script.scenes):
                if not scene.image_path or not Path(scene.image_path).exists():
                    st.warning(f"Scene {scene.index} missing image, skipping...")
                    continue

                # Calculate scene duration from dialogue timing
                scene_duration = scene.end_time - scene.start_time
                if scene_duration <= 0:
                    scene_duration = 5.0

                clip_path = output_dir / f"scene_{scene.index:03d}.mp4"

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

            # Step 2: Concatenate scene clips
            status_text.text("Concatenating video clips...")
            progress_bar.progress(0.4, text="Concatenating video clips...")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_only_path = output_dir / f"video_only_{timestamp}.mp4"
            video_gen.concatenate_clips(
                clip_paths=scene_clips,
                output_path=video_only_path,
                crossfade_duration=0.5,
            )

            # Step 3: Concatenate audio clips
            status_text.text("Assembling audio track...")
            progress_bar.progress(0.5, text="Assembling audio track...")

            # Collect all dialogue audio paths in order
            audio_clips = []
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
                # No audio - just use video
                final_output = video_only_path

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

            # Clean up intermediate files
            for clip in scene_clips:
                if clip.exists() and clip != final_output:
                    clip.unlink()
            if video_only_path.exists() and video_only_path != final_output:
                video_only_path.unlink()

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
        output_dir = config.output_dir / "movie"

        # Audio files
        audio_dir = output_dir / "audio"
        if audio_dir.exists():
            audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav"))
            st.markdown(f"**Voice clips:** {len(audio_files)} files in `{audio_dir}`")

        # Image files
        images_dir = output_dir / "images"
        if images_dir.exists():
            image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
            st.markdown(f"**Scene images:** {len(image_files)} files in `{images_dir}`")

        # Character portraits
        chars_dir = output_dir / "characters"
        if chars_dir.exists():
            char_files = list(chars_dir.glob("*.png")) + list(chars_dir.glob("*.jpg"))
            st.markdown(f"**Character portraits:** {len(char_files)} files in `{chars_dir}`")

        # Videos
        videos_dir = output_dir / "videos"
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
