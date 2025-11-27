"""Completion page - Step 5 of the workflow."""

from pathlib import Path

import streamlit as st

from src.ui.components.state import get_state, reset_state, go_to_step
from src.models.schemas import WorkflowStep


def render_complete_page() -> None:
    """Render the completion page."""
    state = get_state()

    st.header("Your Music Video is Complete!")

    st.balloons()

    if state.final_video_path:
        st.subheader(state.lyrics.title if state.lyrics else "Your Song")

        # Video player
        try:
            st.video(state.final_video_path)
        except Exception:
            st.info(f"Video saved to: {state.final_video_path}")

        # Download section
        st.markdown("---")
        st.subheader("Download Your Video")

        col1, col2, col3 = st.columns(3)

        with col1:
            try:
                with open(state.final_video_path, "rb") as f:
                    video_data = f.read()
                st.download_button(
                    "Download MP4",
                    data=video_data,
                    file_name=Path(state.final_video_path).name,
                    mime="video/mp4",
                )
            except Exception:
                st.error("Could not prepare download")

        with col2:
            if state.lyrics:
                st.download_button(
                    "Download Lyrics",
                    data=state.lyrics.lyrics,
                    file_name=f"{state.lyrics.title.replace(' ', '_')}_lyrics.txt",
                    mime="text/plain",
                )

        # Project summary
        st.markdown("---")
        st.subheader("Project Summary")

        if state.concept:
            st.markdown(
                f"""
                **Genre:** {state.concept.genre}

                **Mood:** {state.concept.mood}

                **Themes:** {', '.join(state.concept.themes)}
                """
            )

        if state.lyrics:
            st.markdown(f"**Title:** {state.lyrics.title}")
            st.markdown(f"**Suno Tags:** {state.lyrics.suno_tags}")

        st.markdown(f"**Duration:** {state.audio_duration:.1f} seconds")
        st.markdown(f"**Scenes:** {len(state.scenes)}")
        st.markdown(f"**Words synced:** {len(state.transcript.all_words) if state.transcript else 0}")

    else:
        st.warning("No video found. Please complete the generation step.")
        if st.button("Go to Generation"):
            go_to_step(WorkflowStep.GENERATE)
            st.rerun()

    # Start over
    st.markdown("---")
    st.subheader("Create Another Song")

    if st.button("Start New Project", type="primary"):
        reset_state()
        st.rerun()

    st.markdown(
        """
        ---
        *Created with Songmaker - AI-powered music video generation*
        """
    )
