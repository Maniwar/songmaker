"""Audio Upload page - Step 3 of the workflow."""

import tempfile
from pathlib import Path

import streamlit as st

from src.config import config
from src.services.audio_processor import AudioProcessor, get_available_backends
from src.ui.components.state import get_state, update_state, advance_step, go_to_step
from src.models.schemas import WorkflowStep


def render_upload_page() -> None:
    """Render the audio upload page."""
    state = get_state()

    st.header("Upload Your Song")

    # Check prerequisites
    if not state.lyrics_approved:
        st.warning("Please complete the lyrics step first.")
        if st.button("Go to Lyrics"):
            go_to_step(WorkflowStep.LYRICS)
            st.rerun()
        return

    st.markdown(
        """
        Upload the MP3 you created on Suno. We'll extract word-level timestamps
        for synchronized lyrics display in your music video.
        """
    )

    # File uploader - PROMINENT at the top
    st.subheader("📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose your MP3 file",
        type=["mp3", "wav", "m4a", "ogg"],
        help="Upload the song you generated on Suno",
    )

    # Demo mode option - at the bottom as alternative
    st.markdown("---")
    st.markdown("**Or use Demo Mode for testing:**")
    if st.button("🧪 Use Demo Mode (skip audio upload)"):
        from src.models.schemas import Transcript, Segment, Word

        # Create simulated transcript based on lyrics
        demo_duration = 180.0  # 3 minutes
        words = []

        if state.lyrics:
            # Parse lyrics into words
            import re
            lyrics_text = re.sub(r'\[.*?\]', '', state.lyrics.lyrics)  # Remove section markers
            lyrics_words = lyrics_text.split()

            # Distribute words over duration
            word_duration = demo_duration / max(len(lyrics_words), 1)
            for i, word_text in enumerate(lyrics_words):
                start = i * word_duration
                end = start + word_duration * 0.8  # 80% of slot for the word
                words.append(Word(word=word_text, start=start, end=end))
        else:
            # Fallback if no lyrics
            for i in range(50):
                words.append(Word(word=f"word{i}", start=i * 3.0, end=i * 3.0 + 2.5))

        # Create transcript
        demo_transcript = Transcript(
            segments=[
                Segment(
                    text=" ".join(w.word for w in words),
                    start=0.0,
                    end=demo_duration,
                    words=words,
                )
            ],
            language="en",
            duration=demo_duration,
        )

        update_state(
            audio_path="demo_mode",
            audio_duration=demo_duration,
            transcript=demo_transcript,
        )

        st.success(f"Demo mode activated! Simulated {len(words)} words over {demo_duration:.0f}s")
        st.rerun()

    st.caption("Demo mode creates simulated word timestamps from your lyrics for testing without real audio.")

    if uploaded_file:
        # Save to temporary file
        config.ensure_directories()
        temp_path = config.songs_dir / uploaded_file.name

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File uploaded: {uploaded_file.name}")

        # Audio player
        st.audio(uploaded_file)

        # Lyrics hint section - for improving transcription accuracy
        st.subheader("🎵 Lyrics (Optional but Recommended)")

        # Check if we already have lyrics from the workflow
        existing_lyrics = ""
        if state.lyrics and state.lyrics.lyrics:
            existing_lyrics = state.lyrics.lyrics
            st.success("Lyrics detected from previous step - will use them to improve transcription!")
            with st.expander("View/Edit Lyrics", expanded=False):
                manual_lyrics = st.text_area(
                    "Edit lyrics if needed:",
                    value=existing_lyrics,
                    height=200,
                    key="lyrics_hint_edit",
                    help="These lyrics help WhisperX accurately identify words in your song",
                )
        else:
            st.info(
                "**Tip:** Paste your song lyrics below to dramatically improve transcription accuracy. "
                "This is especially helpful for music where vocals are mixed with instruments."
            )
            manual_lyrics = st.text_area(
                "Paste your lyrics here:",
                value="",
                height=200,
                key="lyrics_hint_paste",
                placeholder="[Verse 1]\nYour lyrics here...\n\n[Chorus]\nChorus lyrics...",
                help="Copy lyrics from Suno or your original lyrics. Section markers like [Verse] are automatically removed.",
            )

        # Store the lyrics to use
        lyrics_for_transcription = manual_lyrics if manual_lyrics else existing_lyrics

        st.markdown("---")

        # Transcription backend selector
        available_backends = get_available_backends()
        backend_options = {label: value for value, label in available_backends}

        selected_label = st.selectbox(
            "Transcription Engine",
            options=list(backend_options.keys()),
            index=0,
            help="Choose the transcription backend. WhisperX runs locally, AssemblyAI uses cloud API (requires API key).",
        )
        selected_backend = backend_options[selected_label]

        # Demucs vocal separation toggle (only for WhisperX)
        use_demucs = False
        if selected_backend == "whisperx":
            use_demucs = st.checkbox(
                "Use Demucs vocal separation",
                value=config.use_demucs,
                help="Separate vocals from instruments before transcription. "
                     "Improves accuracy for songs with heavy instrumentation. "
                     "Requires 'pip install demucs' (adds ~1-2 min processing time).",
            )

        # Process audio button
        if st.button("Process Audio", type="primary"):
            processor = AudioProcessor(backend=selected_backend)

            # Progress display
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def progress_callback(message: str, progress: float):
                status_text.text(message)
                progress_bar.progress(progress)

            try:
                # Show info about lyrics usage
                if lyrics_for_transcription:
                    st.info("Using your lyrics to improve transcription accuracy...")

                with st.spinner("Processing audio..."):
                    transcript = processor.transcribe(
                        temp_path,
                        progress_callback=progress_callback,
                        lyrics_hint=lyrics_for_transcription if lyrics_for_transcription else None,
                        language="en",  # Most Suno songs are English
                        use_demucs=use_demucs,
                    )

                # Update state
                update_state(
                    audio_path=str(temp_path),
                    audio_duration=transcript.duration,
                    transcript=transcript,
                )

                st.success(
                    f"Audio processed! Duration: {transcript.duration:.1f}s, "
                    f"Words detected: {len(transcript.all_words)}"
                )

                # Show sample of detected words
                with st.expander("Preview detected words"):
                    sample_words = transcript.all_words[:20]
                    for word in sample_words:
                        st.write(
                            f"{word.start:.2f}s - {word.end:.2f}s: {word.word}"
                        )
                    if len(transcript.all_words) > 20:
                        st.write(f"... and {len(transcript.all_words) - 20} more words")

            except Exception as e:
                st.error(f"Error processing audio: {e}")

            finally:
                processor.cleanup()

    # Show current transcript status
    if state.transcript:
        st.markdown("---")
        st.subheader("Audio Ready")
        st.write(f"**Duration:** {state.audio_duration:.1f} seconds")
        st.write(f"**Words detected:** {len(state.transcript.all_words)}")
        st.write(f"**Segments:** {len(state.transcript.segments)}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Re-upload Audio"):
                update_state(audio_path=None, audio_duration=0.0, transcript=None)
                st.rerun()

        with col2:
            if st.button("Continue to Video Generation", type="primary"):
                advance_step()
                st.rerun()

    # Back button
    st.markdown("---")
    if st.button("Back to Lyrics"):
        go_to_step(WorkflowStep.LYRICS)
        st.rerun()
