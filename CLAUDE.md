# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An agentic AI system with two modes:

1. **Song Mode**: Transform song ideas into complete music videos with synchronized lyrics for sing-along capability. Users collaborate with AI to develop song concepts, generate professional lyrics, create songs via Suno, and produce cohesive videos with dynamically-generated scenes and word-level synced karaoke lyrics.

2. **Movie Mode**: Create animated podcasts, educational videos, and short films with consistent characters and unique AI-generated voices. Script-based workflow with TTS voice generation.

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
├── agents/           # concept_agent, lyrics_agent, visual_agent, script_agent
├── services/         # audio_processor, image_generator, video_generator, tts_service, mps_upscaler
├── models/           # Pydantic schemas (Song + Movie mode)
└── ui/
    ├── app.py        # Main Streamlit app
    ├── components/   # wizard, progress, player
    └── pages/        # concept, lyrics, upload, generate, movie

output/               # songs/, images/, videos/, subtitles/, movie/
tests/                # test_audio.py, test_video.py, test_sync.py
```

## Common Commands

```bash
# Install dependencies
pip install -e .

# Run the Streamlit app (recommended - uses correct venv)
./run.sh

# Or manually with venv
source venv/bin/activate
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

# Optional - Song Mode
WHISPER_MODEL=large-v3
WHISPER_DEVICE=cuda              # or cpu for Mac
VIDEO_RESOLUTION=1920x1080
VIDEO_FPS=30
DEFAULT_ART_STYLE="cinematic digital art, dramatic lighting, 8k quality"

# Optional - Movie Mode TTS (pick one)
OPENAI_API_KEY=...               # OpenAI TTS (good quality)
ELEVENLABS_API_KEY=...           # ElevenLabs TTS (best quality)
# Edge TTS is free and requires no API key
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

## Movie Mode

### Workflow

SCRIPT → CHARACTERS → VOICES → VISUALS → RENDER → COMPLETE

### Key Components

- **ScriptAgent**: AI-powered screenplay development (like ConceptAgent for songs)
- **TTSService**: Multi-provider TTS (OpenAI, ElevenLabs, Edge TTS)
- **MovieImageGenerator**: Scene generation with character consistency

### Character Consistency in Movies

Include full character descriptions in every scene prompt:

```python
for char_id in scene.visible_characters:
    character = script.get_character(char_id)
    prompt += f"{character.name}: {character.description}"
```

### TTS Providers

| Provider | Quality | Cost | Notes |
|----------|---------|------|-------|
| ElevenLabs | Best | $5+/mo | Most natural voices |
| OpenAI | Good | Pay-per-use | Uses existing API key |
| Edge TTS | OK | Free | Microsoft Azure voices |

## System Requirements

- Python 3.11+
- FFmpeg with libx264
- CUDA 12+ (optional, for GPU acceleration)

## Dependencies

Key packages: anthropic, google-genai, whisperx, faster-whisper, streamlit, pydub, Pillow, pydantic
