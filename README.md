# Songmaker

AI-powered music video generator with synchronized lyrics for sing-along capability.

Transform song ideas into complete music videos with karaoke-style word highlighting, or create animated podcasts and short films with AI-generated voices.

## Features

### Song Mode
- **AI Song Concept Workshop** - Collaborate with Claude to develop song ideas, themes, and emotions
- **Professional Lyrics Generation** - AI-crafted lyrics with Suno-compatible tags
- **Word-Level Sync** - WhisperX extracts precise word timestamps from your audio
- **AI Scene Generation** - Gemini creates beautiful, cohesive visuals for each section
- **Ken Burns Effects** - Dynamic zoom/pan animations bring static images to life
- **Karaoke Lyrics** - Real-time word highlighting synced to the music
- **4K AI Upscaling** - Real-ESRGAN upscaling optimized for Apple Silicon (MPS)

### Movie Mode
- **Script Development** - AI-powered screenplay creation
- **Character Voices** - Multiple TTS providers (ElevenLabs, OpenAI, Edge TTS)
- **Consistent Characters** - AI maintains character appearance across scenes
- **Lip Sync Animation** - Optional lip sync via Kling AI or Wan2.2-S2V

## Quick Start

### Prerequisites

- Python 3.11+
- FFmpeg with libx264
- Git

**macOS:**
```bash
brew install ffmpeg python@3.11
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg python3.11 python3.11-venv
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/songmaker.git
cd songmaker
```

2. **Create and activate virtual environment:**
```bash
python3.11 -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows
```

3. **Install dependencies:**
```bash
pip install -e .
```

4. **Set up API keys:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Required API Keys

| Key | Service | Required | Purpose |
|-----|---------|----------|---------|
| `ANTHROPIC_API_KEY` | [Anthropic](https://console.anthropic.com/) | Yes | Claude AI for lyrics & concepts |
| `GOOGLE_API_KEY` | [Google AI Studio](https://aistudio.google.com/) | Yes | Gemini for image generation |

### Optional API Keys

| Key | Service | Purpose |
|-----|---------|---------|
| `ELEVENLABS_API_KEY` | [ElevenLabs](https://elevenlabs.io/) | Best quality TTS for Movie Mode |
| `OPENAI_API_KEY` | [OpenAI](https://platform.openai.com/) | OpenAI TTS (alternative) |
| `FAL_KEY` | [fal.ai](https://fal.ai/) | Kling AI lip sync (faster) |
| `HF_TOKEN` | [Hugging Face](https://huggingface.co/) | Speaker diarization |
| `ASSEMBLYAI_API_KEY` | [AssemblyAI](https://www.assemblyai.com/) | Cloud transcription |

### Run the App

```bash
./run.sh
```

Or manually:
```bash
source venv/bin/activate
streamlit run src/ui/app.py
```

Then open http://localhost:8501 in your browser.

## Usage

### Creating a Music Video

1. **Concept Workshop** - Chat with Claude to develop your song idea
2. **Generate Lyrics** - Get professional lyrics with Suno tags
3. **Create Song** - Use [Suno](https://suno.com) to generate your track
4. **Upload Audio** - Upload your MP3 for word-level timestamp extraction
5. **Generate Video** - AI creates scenes and assembles your music video

### Movie Mode

1. **Develop Script** - Work with AI to create your screenplay
2. **Define Characters** - Set character descriptions and voice settings
3. **Generate Voices** - TTS converts dialogue to audio
4. **Create Visuals** - AI generates consistent character scenes
5. **Render** - Assemble final video with lip sync (optional)

## Project Structure

```
songmaker/
├── src/
│   ├── agents/          # AI agents (concept, lyrics, visual, script)
│   ├── services/        # Core services (audio, video, TTS, upscaling)
│   ├── models/          # Pydantic schemas
│   └── ui/              # Streamlit UI
├── tests/               # Test suite
├── output/              # Generated content
├── .env.example         # Environment template
├── pyproject.toml       # Project config
└── run.sh              # Launch script
```

## Configuration

### Environment Variables

```bash
# Transcription: "whisperx" (local) or "assemblyai" (cloud)
TRANSCRIPTION_BACKEND=whisperx

# Lip Sync: "wan2s2v" (free, slow) or "kling" (paid, fast)
LIP_SYNC_BACKEND=wan2s2v

# WhisperX settings
WHISPER_MODEL=large-v3
WHISPER_DEVICE=cpu  # or "cuda" for NVIDIA GPU

# Video settings
VIDEO_RESOLUTION=1920x1080
VIDEO_FPS=30

# Image generation style
DEFAULT_ART_STYLE=cinematic digital art, dramatic lighting, 8k quality
```

### Apple Silicon Optimization

The MPS upscaler is optimized for Apple Silicon Macs:

| Mac Model | Tile Size | Batch Size |
|-----------|-----------|------------|
| M1 (8-16GB) | 384-512 | 2-4 |
| M1 Pro/Max | 768-896 | 8-12 |
| M2/M3 Ultra | 1024 | 12-16 |

Adjust these in the UI's Performance Settings when upscaling.

## Development

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_audio.py -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Code Structure

- **Agents** handle AI interactions (Claude for concept/lyrics/script)
- **Services** provide core functionality (audio processing, video generation, TTS)
- **Models** define data schemas with Pydantic
- **UI** provides the Streamlit web interface

## Troubleshooting

### FFmpeg not found
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### WhisperX installation issues
```bash
# Install with GPU support (NVIDIA)
pip install whisperx torch torchaudio

# CPU only (Apple Silicon uses this automatically)
pip install whisperx
```

### MPS upscaling slow
- Increase **Tile Size** (768-1024 for M1 Max)
- Increase **Batch Size** (8-12 for M1 Max)
- Both settings are in the UI's Performance Settings

### Edge TTS not working
```bash
pip install edge-tts
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
