"""Movie Mode page - Create animated podcasts, educational videos, and short films."""

import streamlit as st
from pathlib import Path

from src.agents.script_agent import ScriptAgent
from src.config import config
from src.models.schemas import (
    Character,
    MovieModeState,
    MovieWorkflowStep,
    Script,
    VoiceSettings,
)


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
    """Advance to the next movie workflow step."""
    state = get_movie_state()
    steps = list(MovieWorkflowStep)
    current_idx = steps.index(state.current_step)
    if current_idx < len(steps) - 1:
        state.current_step = steps[current_idx + 1]


def go_to_movie_step(step: MovieWorkflowStep) -> None:
    """Go to a specific movie workflow step."""
    state = get_movie_state()
    state.current_step = step


def render_movie_mode_page() -> None:
    """Render the movie mode page."""
    state = get_movie_state()

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
    if state.current_step == MovieWorkflowStep.SCRIPT:
        render_script_page()
    elif state.current_step == MovieWorkflowStep.CHARACTERS:
        render_characters_page()
    elif state.current_step == MovieWorkflowStep.VOICES:
        render_voices_page()
    elif state.current_step == MovieWorkflowStep.VISUALS:
        render_visuals_page()
    elif state.current_step == MovieWorkflowStep.RENDER:
        render_render_page()
    elif state.current_step == MovieWorkflowStep.COMPLETE:
        render_movie_complete_page()


def render_movie_progress(current_step: MovieWorkflowStep) -> None:
    """Render movie workflow progress indicator."""
    steps = [
        ("📝", "Script", MovieWorkflowStep.SCRIPT),
        ("👥", "Characters", MovieWorkflowStep.CHARACTERS),
        ("🎙️", "Voices", MovieWorkflowStep.VOICES),
        ("🎨", "Visuals", MovieWorkflowStep.VISUALS),
        ("🎬", "Render", MovieWorkflowStep.RENDER),
        ("✅", "Complete", MovieWorkflowStep.COMPLETE),
    ]

    cols = st.columns(len(steps))
    for i, (icon, label, step) in enumerate(steps):
        with cols[i]:
            if step == current_step:
                st.markdown(f"**{icon} {label}**")
            elif list(MovieWorkflowStep).index(step) < list(MovieWorkflowStep).index(current_step):
                st.markdown(f"~~{icon} {label}~~")
            else:
                st.markdown(f"{icon} {label}")

    st.markdown("---")


def render_script_page() -> None:
    """Render the script development page."""
    state = get_movie_state()

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

    # Show getting started help if no messages yet
    if not state.script_messages:
        st.info(
            """
            **How to get started:**

            Type your video idea in the chat box below. For example:
            - "A 5-minute explainer about how black holes work, with two scientists discussing"
            - "A podcast-style discussion about the future of AI between a skeptic and an optimist"
            - "A short animated comedy about a cat who thinks he's a dog"

            I'll help you develop:
            - Memorable characters with distinct appearances and voices
            - Scene-by-scene breakdown with dialogue
            - Visual style and setting descriptions
            """
        )

    # Display conversation history
    for msg in state.script_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Describe your video idea..."):
        # Add user message to state
        state.script_messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Writing..."):
                response = agent.chat(user_input)
            st.markdown(response)

        # Add assistant response to state
        state.script_messages.append({"role": "assistant", "content": response})

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

                # Reference image upload
                uploaded_ref = st.file_uploader(
                    "Reference Image (optional)",
                    type=["png", "jpg", "jpeg"],
                    key=f"char_ref_{i}",
                )

                if uploaded_ref:
                    st.image(uploaded_ref, width=150)

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
        if st.button("Continue to Voices →", type="primary"):
            advance_movie_step()
            st.rerun()


def render_voices_page() -> None:
    """Render the voice generation page."""
    state = get_movie_state()

    if not state.script:
        st.warning("Please create a script first.")
        return

    st.subheader("Generate Character Voices")
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

    # Provider selection
    provider = st.selectbox(
        "TTS Provider",
        options=available,
        format_func=lambda x: {
            "elevenlabs": "ElevenLabs (Best Quality)",
            "openai": "OpenAI TTS",
            "edge": "Edge TTS (Free)",
        }.get(x, x),
    )

    # Show voice options per character
    st.markdown("### Character Voice Assignments")

    for char in state.script.characters:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**{char.name}:** {char.voice.voice_name or 'Not set'}")
        with col2:
            # Voice preview button (placeholder)
            if st.button(f"🔊 Preview", key=f"preview_{char.id}"):
                st.info("Voice preview coming soon!")

    # Dialogue count
    total_dialogue = state.script.total_dialogue_count
    st.markdown(f"**Total dialogue lines:** {total_dialogue}")

    # Generate button
    st.markdown("---")

    if st.button("🎙️ Generate All Voices", type="primary", use_container_width=True):
        from src.services.tts_service import TTSService

        tts = TTSService(default_provider=provider)
        output_dir = config.output_dir / "movie" / "audio"
        output_dir.mkdir(parents=True, exist_ok=True)

        progress_bar = st.progress(0, text="Generating voices...")
        total = total_dialogue
        completed = 0

        for scene in state.script.scenes:
            for dialogue in scene.dialogue:
                char = state.script.get_character(dialogue.character_id)
                if not char:
                    continue

                try:
                    audio_path = tts.generate_dialogue_audio(
                        dialogue=dialogue,
                        character=char,
                        output_dir=output_dir,
                    )

                    # Get duration
                    duration = tts.get_audio_duration(audio_path)
                    dialogue.audio_path = audio_path

                    completed += 1
                    progress_bar.progress(
                        completed / total,
                        text=f"Generated voice for {char.name} ({completed}/{total})"
                    )

                except Exception as e:
                    st.error(f"Failed to generate voice for {char.name}: {e}")

        progress_bar.progress(1.0, text="Voice generation complete!")
        st.success(f"Generated {completed} voice clips!")

        # Continue button
        if st.button("Continue to Visuals →", type="primary"):
            advance_movie_step()
            st.rerun()

    # Navigation
    st.markdown("---")
    if st.button("← Back to Characters"):
        go_to_movie_step(MovieWorkflowStep.CHARACTERS)
        st.rerun()


def render_visuals_page() -> None:
    """Render the visual generation page."""
    state = get_movie_state()

    if not state.script:
        st.warning("Please create a script first.")
        return

    st.subheader("Generate Scene Images")
    st.markdown(
        f"""
        Generate images for each scene in '{state.script.title}'.
        Character descriptions will be included for visual consistency.
        """
    )

    # Visual style settings
    col1, col2 = st.columns(2)
    with col1:
        visual_style = st.text_area(
            "Visual Style",
            value=state.script.visual_style,
            height=100,
            help="Art style for all generated images",
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

    # Scene preview
    st.markdown("### Scenes to Generate")
    for scene in state.script.scenes:
        with st.expander(f"Scene {scene.index + 1}: {scene.title or scene.direction.setting[:50]}"):
            st.markdown(f"**Setting:** {scene.direction.setting}")
            st.markdown(f"**Camera:** {scene.direction.camera}")
            st.markdown(f"**Characters:** {', '.join(scene.direction.visible_characters)}")
            if scene.dialogue:
                st.markdown(f"**Dialogue lines:** {len(scene.dialogue)}")

    # Generate button
    st.markdown("---")

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
    col1, col2 = st.columns(2)
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
        st.info(
            """
            **Video rendering coming soon!**

            This will:
            1. Combine scene images with Ken Burns effects
            2. Sync dialogue audio with visuals
            3. Add subtitles and transitions
            4. Export final video
            """
        )

        # Placeholder for actual rendering
        progress_bar = st.progress(0, text="Preparing assets...")

        # Simulate progress
        import time
        for i in range(100):
            time.sleep(0.05)
            progress_bar.progress(i / 100, text=f"Rendering... {i}%")

        progress_bar.progress(1.0, text="Render complete!")

        # For now, just advance to complete
        advance_movie_step()
        st.rerun()

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

    st.info(
        """
        **Coming soon:** Video preview and download!

        For now, your generated assets are saved in the `output/movie/` directory:
        - `audio/` - Generated voice clips
        - `images/` - Generated scene images
        """
    )

    # Start over button
    if st.button("Create Another Movie", type="primary"):
        if "script_agent" in st.session_state:
            st.session_state.script_agent.reset()
        st.session_state.movie_state = MovieModeState()
        st.rerun()
