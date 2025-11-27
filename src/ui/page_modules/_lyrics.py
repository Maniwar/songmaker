"""Lyrics Generation page - Step 2 of the workflow."""

import streamlit as st

from src.agents.lyrics_agent import LyricsAgent
from src.ui.components.state import get_state, update_state, advance_step, go_to_step
from src.models.schemas import WorkflowStep, GeneratedLyrics


def render_lyrics_page() -> None:
    """Render the lyrics generation page."""
    state = get_state()

    st.header("Generate Lyrics")

    # Check if we have a concept
    if not state.concept:
        st.warning("Please complete the concept workshop first.")
        if st.button("Go to Concept Workshop"):
            go_to_step(WorkflowStep.CONCEPT)
            st.rerun()
        return

    st.markdown(
        f"""
        **Song Concept:**
        - Genre: {state.concept.genre}
        - Mood: {state.concept.mood}
        - Themes: {', '.join(state.concept.themes)}
        """
    )

    # Initialize lyrics agent
    if "lyrics_agent" not in st.session_state:
        st.session_state.lyrics_agent = LyricsAgent()

    agent = st.session_state.lyrics_agent

    # Check if we have draft lyrics from the concept workshop
    if not state.lyrics and state.concept.draft_lyrics and state.concept.suno_tags:
        st.info("✨ Lyrics from your concept workshop are ready! Review them below or regenerate.")
        # Create GeneratedLyrics from the concept's draft
        draft = GeneratedLyrics(
            title=f"{state.concept.genre.title()} Song",  # Placeholder title
            lyrics=state.concept.draft_lyrics,
            suno_tags=state.concept.suno_tags,
            structure=["From Concept Workshop"],
        )
        update_state(lyrics=draft)
        st.rerun()

    # Generate lyrics button
    col1, col2 = st.columns([2, 1])

    with col1:
        if not state.lyrics:
            if st.button("Generate Lyrics", type="primary"):
                with st.spinner("Generating lyrics..."):
                    lyrics = agent.generate_lyrics(state.concept)
                    update_state(lyrics=lyrics)
                    st.rerun()

    # Display current lyrics
    if state.lyrics:
        st.subheader(f"Title: {state.lyrics.title}")

        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Lyrics", "Suno Prompt", "Refine"])

        with tab1:
            st.text_area(
                "Lyrics",
                value=state.lyrics.lyrics,
                height=400,
                disabled=True,
            )
            st.markdown(f"**Style Tags:** {state.lyrics.suno_tags}")
            st.markdown(f"**Structure:** {' → '.join(state.lyrics.structure)}")

        with tab2:
            st.markdown("Copy this to Suno:")
            suno_prompt = agent.generate_suno_prompt(state.lyrics)
            st.code(suno_prompt, language=None)

            if st.button("Copy to Clipboard"):
                st.write("Use Ctrl/Cmd+C to copy the text above")

        with tab3:
            st.markdown("Not happy with the lyrics? Provide feedback to refine them.")
            feedback = st.text_area(
                "What would you like to change?",
                placeholder="e.g., Make the chorus more catchy, add more metaphors...",
            )
            if st.button("Refine Lyrics") and feedback:
                with st.spinner("Refining lyrics..."):
                    refined = agent.refine_lyrics(state.lyrics, feedback)
                    update_state(lyrics=refined)
                    st.rerun()

        # Action buttons
        st.markdown("---")

        # Prominent next step guidance
        st.info(
            """
            **Ready to create your song?**

            1. Copy the **Suno Prompt** from the tab above
            2. Go to [suno.com](https://suno.com) and paste it to generate your song
            3. Download the MP3 file
            4. Click **Continue to Upload** below, then upload your MP3

            *The video generation needs your audio to sync lyrics and create scene timing.*
            """
        )

        # Main action button - prominent
        if st.button("✅ Continue to Upload MP3", type="primary", use_container_width=True):
            update_state(lyrics_approved=True)
            advance_step()
            st.rerun()

        st.markdown("")  # Spacing

        # Secondary actions
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Regenerate Lyrics"):
                with st.spinner("Regenerating..."):
                    lyrics = agent.generate_lyrics(state.concept)
                    update_state(lyrics=lyrics)
                    st.rerun()

        with col2:
            if st.button("← Back to Concept"):
                go_to_step(WorkflowStep.CONCEPT)
                st.rerun()
