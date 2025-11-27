"""Concept Workshop page - Step 1 of the workflow."""

import streamlit as st

from src.agents.concept_agent import ConceptAgent
from src.ui.components.state import get_state, update_state, advance_step, go_to_step


def render_concept_page() -> None:
    """Render the concept workshop page."""
    state = get_state()

    st.header("Song Concept Workshop")
    st.markdown(
        """
        Let's develop your song idea together! Describe what you're thinking,
        and I'll help you refine the concept with genre, mood, and theme suggestions.
        """
    )

    # Initialize concept agent in session state
    if "concept_agent" not in st.session_state:
        st.session_state.concept_agent = ConceptAgent()

    agent = st.session_state.concept_agent

    # Show getting started help if no messages yet
    if not state.concept_messages:
        st.info(
            """
            **How to get started:**

            Type your song idea in the chat box below. For example:
            - "I want to write an upbeat pop song about summer road trips"
            - "A melancholic indie folk song about lost love"
            - "An epic power metal anthem about conquering your fears"

            I'll ask a few questions to understand your vision, then create:
            - Complete lyrics with structure markers
            - Suno AI style tags
            - Visual description for your music video

            **Tip:** Look at the sidebar for status updates and quick actions!
            """
        )

    # Display conversation history
    for msg in state.concept_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Describe your song idea..."):
        # Add user message to state
        state.concept_messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent.chat(user_input)
            st.markdown(response)

        # Add assistant response to state
        state.concept_messages.append({"role": "assistant", "content": response})

        st.rerun()

    # Sidebar with readiness status
    with st.sidebar:
        # Get readiness status
        status = agent.get_readiness_status()

        # Show readiness indicator
        st.subheader("Concept Status")

        if status["ready"]:
            st.success("✅ Ready to finalize!")
            st.markdown("The AI has provided all required sections:")
            st.markdown("- Song Concept Summary")
            st.markdown("- Suno Style Tags")
            st.markdown("- Complete Lyrics")
            st.markdown("- Visual Description")

            if st.button("🎵 Finalize & Continue", type="primary"):
                with st.spinner("Extracting concept..."):
                    concept = agent.extract_concept()
                    if concept:
                        update_state(concept=concept)
                        st.success("Concept finalized!")
                        advance_step()
                        st.rerun()
                    else:
                        st.error("Could not extract concept. Please try again.")

        elif status["has_content"]:
            st.warning("⏳ Still developing...")
            if status["missing"]:
                st.markdown("**Still needed:**")
                for section in status["missing"]:
                    st.markdown(f"- {section}")
            st.markdown(f"*{status['message_count']} message(s) exchanged*")

            # Allow manual finalize even if not "ready" (user might want to override)
            if st.button("Finalize Anyway"):
                with st.spinner("Extracting concept..."):
                    concept = agent.extract_concept()
                    if concept:
                        update_state(concept=concept)
                        st.success("Concept finalized!")
                        st.json(concept.model_dump())
                    else:
                        st.error("Could not extract concept. Please continue the conversation.")
        else:
            st.info("💬 Start by describing your song idea")

        st.markdown("---")

        # Quick actions
        st.subheader("Quick Actions")

        if st.button("Get Genre Suggestions"):
            if state.song_idea or status["has_content"]:
                # Use the first user message as the idea if song_idea not set
                idea = state.song_idea
                if not idea and status["has_content"]:
                    for msg in agent.conversation_history:
                        if msg["role"] == "user":
                            idea = msg["content"]
                            break
                if idea:
                    genres = agent.get_genre_suggestions(idea)
                    st.write("**Suggested Genres:**")
                    for genre in genres:
                        st.write(f"- {genre}")

        st.markdown("---")

        # Show current concept if available
        if state.concept:
            st.subheader("Current Concept")
            st.write(f"**Genre:** {state.concept.genre}")
            st.write(f"**Mood:** {state.concept.mood}")
            st.write(f"**Themes:** {', '.join(state.concept.themes)}")
            if state.concept.suno_tags:
                st.write(f"**Suno Tags:** {state.concept.suno_tags}")

            if st.button("Proceed to Lyrics", type="primary"):
                advance_step()
                st.rerun()

        if st.button("Start Over"):
            agent.reset()
            update_state(concept_messages=[], concept=None, song_idea="")
            st.rerun()

        st.markdown("---")
        st.subheader("Quick Start")
        st.markdown("Already have your song ready?")
        if st.button("Skip to Upload", type="secondary"):
            # Create a minimal concept for skipping
            from src.models.schemas import SongConcept, WorkflowStep, GeneratedLyrics
            minimal_concept = SongConcept(
                idea="Custom song",
                genre="Various",
                mood="Mixed",
                themes=["music video"],
                visual_style="cinematic digital art, dramatic lighting",
            )
            # Create placeholder lyrics so user doesn't need to go through lyrics step
            placeholder_lyrics = GeneratedLyrics(
                title="Custom Song",
                lyrics="[Lyrics will be extracted from audio]",
                suno_tags="custom",
                structure=["Custom"],
            )
            update_state(
                concept=minimal_concept,
                lyrics=placeholder_lyrics,
                lyrics_approved=True,  # Skip lyrics requirement
            )
            go_to_step(WorkflowStep.UPLOAD)
            st.rerun()
