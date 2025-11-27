# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An agentic AI system that transforms song ideas into complete music videos with synchronized lyrics for sing-along capability. Users collaborate with AI to develop song concepts, generate professional lyrics, create songs via Suno, and produce cohesive videos with dynamically-generated scenes and word-level synced karaoke lyrics.

## Architecture

```
USER WORKFLOW
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1: Song Concept Workshop (Iterative AI collaboration)                 │
│  Step 2: Generate Lyrics + Suno Tags → User creates song on suno.com        │
│  Step 3: Upload MP3 → WhisperX extracts word-level timestamps               │
│  Step 4: Generate Music Video → Dynamic scenes + Ken Burns + synced lyrics  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Technology Stack

- **Audio Processing**: WhisperX (word-level timestamps via wav2vec2 alignment)
- **Image Generation**: Gemini 2.5 Flash Image (Nano Banana)
- **Video Processing**: FFmpeg with Ken Burns effects (zoompan filter)
- **Lyrics Sync**: ASS subtitle format with karaoke highlighting
- **UI**: Streamlit with session state management
- **AI Reasoning**: Claude (Anthropic) for lyrics and concept development

## Project Structure

```
src/
├── agents/           # concept_agent, lyrics_agent, visual_agent
├── services/         # audio_processor, image_generator, video_generator, lyrics_sync, subtitle_generator
├── models/           # Pydantic schemas
└── ui/
    ├── app.py        # Main Streamlit app
    ├── components/   # wizard, progress, player
    └── pages/        # concept, lyrics, upload, generate

output/               # songs/, images/, videos/, subtitles/
tests/                # test_audio.py, test_video.py, test_sync.py
```

## Common Commands

```bash
# Install dependencies
pip install -e .

# Run the Streamlit app
streamlit run src/ui/app.py

# Or use the CLI launcher
python -m src.main ui

# Run tests
python -m pytest tests/ -v

# Run single test file
python -m pytest tests/test_models.py -v

# Generate video from CLI
python -m src.main generate --audio song.mp3 --lyrics lyrics.txt --output video.mp4

# Transcribe audio only
python -m src.main transcribe audio.mp3 --output transcript.json
```

## Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...     # Claude for reasoning/lyrics
GOOGLE_API_KEY=...               # Gemini for images

# Optional
WHISPER_MODEL=large-v3
WHISPER_DEVICE=cuda              # or cpu for Mac
VIDEO_RESOLUTION=1920x1080
VIDEO_FPS=30
DEFAULT_ART_STYLE="cinematic digital art, dramatic lighting, 8k quality"
```

## Key Implementation Details

### Character Consistency
Include identical character description in ALL scene prompts. Use first generated scene as style reference for subsequent scenes.

### Dynamic Scene Generation
- Target: ~4 scenes per minute
- Calculate scene count based on song duration (min 4s, max 12s per scene)
- Identify natural break points (silences, section changes)
- Ensure full song coverage - no gaps allowed

### Ken Burns Effects
Use FFmpeg zoompan filter with these effect types:
- ZOOM_IN, ZOOM_OUT, PAN_LEFT, PAN_RIGHT, PAN_UP, PAN_DOWN

### Karaoke ASS Format
- Group words into lines (~8 words max)
- Use `\t` transform tags for color animation
- Primary color white, highlight yellow
- Timestamps in H:MM:SS.CC format

### Streamlit Session State
- Use centralized `AppState` dataclass
- Track workflow steps: CONCEPT → LYRICS → UPLOAD → GENERATE → COMPLETE
- Use `st.status()` for long-running operations with detailed feedback

## System Requirements

- Python 3.11+
- FFmpeg with libx264
- CUDA 12+ (optional, for GPU acceleration)

## Dependencies

Key packages: anthropic, google-genai, whisperx, faster-whisper, streamlit, pydub, Pillow, pydantic
