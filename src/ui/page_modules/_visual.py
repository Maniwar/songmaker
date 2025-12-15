"""Visual Style Workshop page - Step 4 of the workflow."""

import streamlit as st

from src.agents.visual_style_agent import VisualStyleAgent
from src.models.schemas import WorkflowStep, Scene, KenBurnsEffect
from src.ui.components.state import get_state, update_state, advance_step, go_to_step


def render_visual_page() -> None:
    """Render the visual style workshop page."""
    state = get_state()

    st.header("Visual Style Workshop")

    # Check prerequisites
    if not state.transcript:
        st.warning("Please upload and process your audio first.")
        if st.button("Go to Upload"):
            go_to_step(WorkflowStep.UPLOAD)
            st.rerun()
        return

    # Check if user has skipped or approved the workshop
    if getattr(state, 'visual_skipped', False):
        st.info("Visual Workshop was skipped. Proceeding to Generate step.")
        advance_step()
        st.rerun()
        return

    if getattr(state, 'visual_approved', False) and state.visual_plan:
        st.success("Visual plan approved! Ready to generate.")
        advance_step()
        st.rerun()
        return

    st.markdown(
        """
        Let's develop the visual style for your music video together!
        I'll help you create:
        - A consistent **visual world** for all scenes
        - **Character descriptions** for AI image generation
        - **Scene-by-scene prompts** matched to your lyrics
        """
    )

    # Calculate default scene count based on song duration
    total_duration = state.transcript.duration
    default_num_scenes = max(4, int(total_duration / 15))  # ~4 scenes per minute
    min_scenes = max(4, int(total_duration / 30))  # Minimum ~2 per minute
    max_scenes = min(100, int(total_duration / 3))  # Maximum ~20 per minute

    # Get visual messages from state or initialize
    visual_messages = getattr(state, 'visual_messages', [])

    # Initialize visual style agent in session state
    if "visual_agent" not in st.session_state:
        st.session_state.visual_agent = VisualStyleAgent(
            concept=state.concept,
            lyrics=state.lyrics,
            transcript=state.transcript,
        )
        # Restore conversation history if loading a project with existing messages
        if visual_messages:
            st.session_state.visual_agent.conversation_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in visual_messages
            ]

    agent = st.session_state.visual_agent

    # Show getting started help if no messages yet
    if not visual_messages:
        st.info(
            """
            **How to get started:**

            First, configure how many scenes you want in your video.
            Then the AI will analyze your lyrics and propose visual style options.
            You can:
            - Discuss your preferred visual world (cyberpunk, fantasy, realistic, etc.)
            - Describe how you want the main character to look
            - Choose cinematography style and color palette
            - Refine individual scene prompts

            **Tip:** Look at the sidebar to track what sections are ready!
            """
        )

        # Scene count configuration
        st.subheader("Scene Configuration")
        col1, col2 = st.columns([2, 1])
        with col1:
            num_scenes = st.slider(
                "Number of Scenes",
                min_value=min_scenes,
                max_value=max_scenes,
                value=default_num_scenes,
                help=f"Your song is {total_duration:.0f}s long. More scenes = more variety, fewer scenes = longer each scene."
            )
        with col2:
            scene_duration = total_duration / num_scenes
            st.metric("Avg Scene Length", f"{scene_duration:.1f}s")

        # Update agent with selected scene count
        if num_scenes != agent.num_scenes:
            agent.set_num_scenes(num_scenes)

        # Show preview of scene breakdown
        with st.expander("Preview Scene Breakdown", expanded=True):
            for i, seg in enumerate(agent.scene_segments[:8]):  # Show first 8
                lyrics_preview = seg["lyrics_segment"][:50] + "..." if len(seg["lyrics_segment"]) > 50 else seg["lyrics_segment"]
                st.caption(f"**Scene {i+1}** ({seg['start_time']:.1f}s - {seg['end_time']:.1f}s): \"{lyrics_preview}\"")
            if num_scenes > 8:
                st.caption(f"... and {num_scenes - 8} more scenes")

        st.markdown("---")

        # Auto-start the conversation
        if st.button("Start Visual Workshop", type="primary"):
            with st.spinner("Analyzing your song..."):
                response = agent.start_conversation()
                visual_messages.append({"role": "assistant", "content": response})
                update_state(visual_messages=visual_messages)
                st.rerun()

    # Display conversation history
    for msg in visual_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input (only if conversation has started)
    if visual_messages:
        if user_input := st.chat_input("Describe your visual preferences..."):
            # Add user message to state
            visual_messages.append({"role": "user", "content": user_input})

            with st.chat_message("user"):
                st.markdown(user_input)

            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = agent.chat(user_input)
                st.markdown(response)

            # Add assistant response to state
            visual_messages.append({"role": "assistant", "content": response})
            update_state(visual_messages=visual_messages)
            st.rerun()

    # Sidebar with readiness status
    with st.sidebar:
        st.subheader("Visual Status")

        # Get readiness status
        status = agent.get_readiness_status()

        if status["ready"]:
            st.success("Ready to finalize!")
            st.markdown("All sections complete:")
            st.markdown("- Visual World")
            st.markdown("- Character Description")
            st.markdown("- Cinematography Style")
            st.markdown("- Scene Prompts")

            if st.button("Finalize Visual Plan", type="primary"):
                with st.spinner("Extracting visual plan..."):
                    visual_plan = agent.extract_visual_plan()
                    if visual_plan:
                        # Convert visual plan to scenes
                        scenes = _convert_visual_plan_to_scenes(visual_plan, state)

                        # Update concept with visual info
                        if state.concept:
                            state.concept.visual_world = visual_plan.visual_world
                            state.concept.character_description = visual_plan.character_description
                            state.concept.visual_style = visual_plan.cinematography_style

                        update_state(
                            visual_plan=visual_plan,
                            visual_approved=True,
                            prompts_ready=True,  # Scenes from workshop have prompts ready
                            scenes=scenes,
                            concept=state.concept,
                        )
                        st.success("Visual plan finalized!")
                        advance_step()
                        st.rerun()
                    else:
                        st.error("Could not extract visual plan. Please try again.")

        elif status["has_content"]:
            st.warning("Still developing...")
            if status["missing"]:
                st.markdown("**Still needed:**")
                for section in status["missing"]:
                    st.markdown(f"- {section}")
            st.markdown(f"*{status['message_count']} message(s) exchanged*")

            # Allow manual finalize even if not "ready"
            if st.button("Finalize Anyway"):
                with st.spinner("Extracting visual plan..."):
                    visual_plan = agent.extract_visual_plan()
                    if visual_plan:
                        scenes = _convert_visual_plan_to_scenes(visual_plan, state)
                        if state.concept:
                            state.concept.visual_world = visual_plan.visual_world
                            state.concept.character_description = visual_plan.character_description
                            state.concept.visual_style = visual_plan.cinematography_style
                        update_state(
                            visual_plan=visual_plan,
                            visual_approved=True,
                            prompts_ready=True,  # Scenes from workshop have prompts ready
                            scenes=scenes,
                            concept=state.concept,
                        )
                        st.success("Visual plan finalized!")
                        advance_step()
                        st.rerun()
                    else:
                        st.error("Could not extract visual plan.")
        else:
            st.info("Click 'Start Visual Workshop' to begin")

        st.markdown("---")

        # Scene reference
        st.subheader("Scene Reference")
        _render_scene_reference(state)

        st.markdown("---")

        # Quick actions
        st.subheader("Quick Actions")

        if st.button("Skip Workshop"):
            # Skip the workshop and use auto-generated prompts
            update_state(visual_skipped=True)
            st.info("Skipping workshop - will use auto-generated prompts")
            advance_step()
            st.rerun()

        if st.button("Start Over"):
            agent.reset()
            update_state(
                visual_messages=[],
                visual_plan=None,
                visual_approved=False,
                visual_skipped=False,
            )
            # Reset agent in session state
            del st.session_state.visual_agent
            st.rerun()

        st.markdown("---")

        if st.button("Back to Upload"):
            go_to_step(WorkflowStep.UPLOAD)
            st.rerun()


def _render_scene_reference(state) -> None:
    """Render a compact scene reference showing lyrics timing."""
    if not state.transcript:
        return

    total_duration = state.transcript.duration
    num_scenes = max(4, int(total_duration / 15))  # ~4 scenes per minute
    scene_duration = total_duration / num_scenes

    with st.expander("Scene Lyrics", expanded=False):
        for i in range(min(num_scenes, 8)):  # Show max 8 scenes in sidebar
            start_time = i * scene_duration
            end_time = (i + 1) * scene_duration

            # Get words in this time range
            words = [
                w for w in state.transcript.all_words
                if w.start >= start_time and w.end <= end_time
            ]

            if words:
                lyrics_segment = " ".join(w.word for w in words)[:50]
                if len(lyrics_segment) > 50:
                    lyrics_segment += "..."
                st.markdown(f"**Scene {i+1}** ({start_time:.0f}s-{end_time:.0f}s)")
                st.caption(f'"{lyrics_segment}"')

        if num_scenes > 8:
            st.caption(f"... and {num_scenes - 8} more scenes")


def _convert_visual_plan_to_scenes(visual_plan, state) -> list[Scene]:
    """Convert a VisualPlan to Scene objects."""
    scenes = []

    # If we have scene prompts from the plan, use them
    if visual_plan.scene_prompts:
        for sp in visual_plan.scene_prompts:
            # Get words for this time range from transcript
            words = []
            if state.transcript:
                words = [
                    w for w in state.transcript.all_words
                    if w.start >= sp.start_time and w.end <= sp.end_time
                ]

            # Build the full prompt with visual world and character
            full_prompt = _build_scene_prompt(
                sp.visual_prompt,
                visual_plan.visual_world,
                visual_plan.character_description,
                visual_plan.cinematography_style,
            )

            scenes.append(Scene(
                index=sp.index,
                start_time=sp.start_time,
                end_time=sp.end_time,
                visual_prompt=full_prompt,
                mood=sp.mood,
                effect=sp.effect,
                words=words,
                motion_prompt=sp.motion_prompt,  # LLM-generated motion prompt for animation
                show_character=sp.show_character,  # Whether character appears in this scene
            ))
    else:
        # Fall back to auto-generating scenes based on transcript
        if state.transcript:
            total_duration = state.transcript.duration
            num_scenes = max(4, int(total_duration / 15))
            scene_duration = total_duration / num_scenes

            for i in range(num_scenes):
                start_time = i * scene_duration
                end_time = (i + 1) * scene_duration

                words = [
                    w for w in state.transcript.all_words
                    if w.start >= start_time and w.end <= end_time
                ]

                lyrics_segment = " ".join(w.word for w in words) if words else ""

                # Build a basic prompt
                full_prompt = _build_scene_prompt(
                    f"Scene depicting: {lyrics_segment}" if lyrics_segment else "Cinematic scene",
                    visual_plan.visual_world,
                    visual_plan.character_description,
                    visual_plan.cinematography_style,
                )

                scenes.append(Scene(
                    index=i,
                    start_time=start_time,
                    end_time=end_time,
                    visual_prompt=full_prompt,
                    mood="neutral",
                    effect=KenBurnsEffect.ZOOM_IN,
                    words=words,
                ))

    return scenes


def _build_scene_prompt(
    scene_specific: str,
    visual_world: str,
    character_description: str,
    cinematography_style: str,
) -> str:
    """Build a complete scene prompt combining all visual elements."""
    parts = []

    if scene_specific:
        parts.append(scene_specific)

    if visual_world:
        parts.append(f"Setting: {visual_world}")

    if character_description:
        parts.append(f"Character: {character_description}")

    if cinematography_style:
        parts.append(f"Style: {cinematography_style}")

    return ". ".join(parts)
