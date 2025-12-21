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
    """Render movie workflow progress indicator."""
    steps = [
        ("📝", "Script", MovieWorkflowStep.SCRIPT),
        ("👥", "Characters", MovieWorkflowStep.CHARACTERS),
        ("🎬", "Scenes", MovieWorkflowStep.SCENES),
        ("🎙️", "Voices", MovieWorkflowStep.VOICES),
        ("🎨", "Visuals", MovieWorkflowStep.VISUALS),
        ("🔧", "Render", MovieWorkflowStep.RENDER),
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

                # Character Portrait Section
                st.markdown("**Character Portrait**")

                # Check if portrait already exists
                portrait_path = char.reference_image_path
                if portrait_path and Path(portrait_path).exists():
                    st.image(str(portrait_path), width=150)
                    st.caption("AI-generated portrait")
                    if st.button("🔄 Regenerate", key=f"regen_portrait_{i}"):
                        # Regenerate portrait
                        from src.services.movie_image_generator import MovieImageGenerator
                        generator = MovieImageGenerator(style=state.script.visual_style or "cinematic digital art")
                        output_dir = config.output_dir / "movie" / "characters"
                        with st.spinner(f"Generating portrait for {char.name}..."):
                            result = generator.generate_character_reference(
                                character=char,
                                output_dir=output_dir,
                                style=state.script.visual_style,
                            )
                            if result:
                                char.reference_image_path = result
                                st.success("Portrait regenerated!")
                                st.rerun()
                            else:
                                st.error("Failed to generate portrait")
                else:
                    # Generate portrait button
                    if st.button("🎨 Generate Portrait", key=f"gen_portrait_{i}", type="secondary"):
                        from src.services.movie_image_generator import MovieImageGenerator
                        generator = MovieImageGenerator(style=state.script.visual_style or "cinematic digital art")
                        output_dir = config.output_dir / "movie" / "characters"
                        with st.spinner(f"Generating portrait for {char.name}..."):
                            result = generator.generate_character_reference(
                                character=char,
                                output_dir=output_dir,
                                style=state.script.visual_style,
                            )
                            if result:
                                char.reference_image_path = result
                                st.success("Portrait generated!")
                                st.rerun()
                            else:
                                st.error("Failed to generate portrait")

                    # Also allow upload
                    uploaded_ref = st.file_uploader(
                        "Or upload reference",
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
                        char.reference_image_path = ref_path
                        st.image(uploaded_ref, width=150)
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
                    st.warning(f"Character '{dialogue.character_id}' not found, skipping dialogue")
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
                    # Clear timing for failed dialogue so it's excluded from video
                    dialogue.audio_path = None
                    dialogue.start_time = None
                    dialogue.end_time = None

            # Mark scene end time
            scene.end_time = running_time
            # Add pause between scenes (except for last scene)
            if scene_idx < len(state.script.scenes) - 1:
                running_time += pause_between_scenes

        progress_bar.progress(1.0, text="Voice generation complete!")
        total_duration = running_time
        minutes = int(total_duration // 60)
        seconds = int(total_duration % 60)

        if failed_count > 0:
            st.warning(f"Generated {completed} voice clips ({failed_count} failed). Total duration: {minutes}m {seconds}s")
        else:
            st.success(f"Generated {completed} voice clips! Total duration: {minutes}m {seconds}s")

        # Continue button
        if st.button("Continue to Visuals →", type="primary"):
            advance_movie_step()
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
