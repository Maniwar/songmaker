"""Main Streamlit application for Songmaker."""

import logging
import streamlit as st

# Configure logging to show INFO level for our services
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Reduce noise from other loggers
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("gradio_client").setLevel(logging.WARNING)

from src.config import config
from src.models.schemas import WorkflowStep
from src.ui.components.state import (
    init_session_state,
    get_state,
    render_project_sidebar,
)
from src.ui.components.wizard import render_wizard_progress
from src.ui.page_modules._concept import render_concept_page
from src.ui.page_modules._lyrics import render_lyrics_page
from src.ui.page_modules._upload import render_upload_page
from src.ui.page_modules._visual import render_visual_page
from src.ui.page_modules._generate import render_generate_page
from src.ui.page_modules._complete import render_complete_page
from src.ui.page_modules._movie import render_movie_mode_page


def main():
    """Main application entry point."""
    # Page config
    st.set_page_config(
        page_title="Songmaker - AI Music Video Generator",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    init_session_state()

    # Header
    st.title("🎵 Songmaker")
    st.markdown("*Transform your song ideas into music videos with synchronized lyrics*")

    # Sidebar for project management
    render_project_sidebar()

    # Validate configuration
    errors = config.validate()
    if errors:
        st.error("Configuration errors:")
        for error in errors:
            st.error(f"- {error}")
        st.info("Please set the required environment variables in your .env file")
        st.stop()

    # Get current state
    state = get_state()

    # Check for special modes (bypass normal workflow)
    # Check both session state flag and AppState for compatibility
    upscale_mode = st.session_state.get('upscale_only_mode', False) or getattr(state, 'upscale_only_mode', False)
    movie_mode = st.session_state.get('movie_mode', False) or getattr(state, 'movie_mode', False)

    if movie_mode:
        # Movie Mode - animated podcasts, educational videos, short films
        render_movie_mode_page()
    elif upscale_mode:
        from src.ui.page_modules._generate import render_upscale_only_page
        render_upscale_only_page(state)
    else:
        # Render wizard progress
        render_wizard_progress(state.current_step)

        # Render current page
        if state.current_step == WorkflowStep.CONCEPT:
            render_concept_page()
        elif state.current_step == WorkflowStep.LYRICS:
            render_lyrics_page()
        elif state.current_step == WorkflowStep.UPLOAD:
            render_upload_page()
        elif state.current_step == WorkflowStep.VISUAL:
            render_visual_page()
        elif state.current_step == WorkflowStep.GENERATE:
            render_generate_page()
        elif state.current_step == WorkflowStep.COMPLETE:
            render_complete_page()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 0.8em;'>
        Songmaker uses Claude for concept development, Gemini for image generation,
        and WhisperX for audio transcription.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
