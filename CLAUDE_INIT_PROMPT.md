# CLAUDE.md - Suno AI Song Maker & Music Video Generator

## Project Overview

An agentic AI system that transforms song ideas into complete music videos with synchronized lyrics for sing-along capability. The system uses an iterative workflow where users collaborate with AI to develop song concepts, generate professional lyrics, and create cohesive videos with dynamically-generated scenes that match the song's full duration.

### Core Philosophy
- **Iterative Collaboration**: Users work conversationally with the AI to refine song concepts before generation
- **Full Song Coverage**: Every second of the song has corresponding visual content
- **Precise Lyrics Sync**: Word-level timestamp alignment for sing-along karaoke-style display
- **Dynamic Scene Generation**: Scene count adapts to song length automatically
- **Smooth UX**: Streamlit session state management for seamless multi-step workflow

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER WORKFLOW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Step 1: Song Concept Workshop (Iterative)                                  │
│    └─> AI researches genres, suggests options, user picks/refines           │
│  Step 2: Generate Lyrics + Suno Tags                                        │
│    └─> Copy to suno.com, generate song, download MP3                        │
│  Step 3: Upload MP3                                                         │
│    └─> WhisperX extracts word-level timestamps                              │
│  Step 4: Generate Full Music Video                                          │
│    └─> Dynamic scenes + Ken Burns effects + synced lyrics overlay           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack (2025 Best Practices)

### Audio Processing & Lyrics Sync

#### Primary: WhisperX (Recommended)
```python
# WhisperX provides the most accurate word-level timestamps via wav2vec2 alignment
# Released: Oct 2025 | Requires: Python >=3.9, <3.14

import whisperx
import gc

device = "cuda"  # or "cpu" for Mac
batch_size = 16
compute_type = "float16"  # "int8" for low GPU memory

# 1. Transcribe with faster-whisper backend
model = whisperx.load_model("large-v3", device, compute_type=compute_type)
audio = whisperx.load_audio("song.mp3")
result = model.transcribe(audio, batch_size=batch_size)

# 2. Align with wav2vec2 for word-level timestamps
model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device)

# Result contains word-level timestamps:
# {"word": "hello", "start": 1.234, "end": 1.567}

gc.collect()
torch.cuda.empty_cache()
del model  # Free memory
```

#### Alternative: stable-ts (Good for CPU-only)
```python
import stable_whisper

model = stable_whisper.load_model('large-v3')
result = model.transcribe('audio.mp3')

# Export with word-level timing
result.to_ass('karaoke.ass')  # ASS format for karaoke effects
result.to_srt_vtt('subtitles.srt', word_level=True)

# Align existing lyrics to audio (faster)
text = "Your known lyrics here..."
result = model.align('audio.mp3', text)
```

#### For Known Lyrics: lyrics-transcriber
```python
# Best when you already have the lyrics (from Suno generation)
# Uses AudioShake API or Whisper + correction algorithms

from lyrics_transcriber import LyricsTranscriber
from lyrics_transcriber.core.controller import TranscriberConfig, OutputConfig

transcriber = LyricsTranscriber(
    audio_filepath="song.mp3",
    transcriber_config=TranscriberConfig(
        audioshake_api_token="...",  # Preferred for accuracy
    ),
    output_config=OutputConfig(
        output_dir="./output",
        video_resolution="1080p",
        add_countdown=True
    ),
)
result = transcriber.process()
# Outputs: ASS, LRC, and video with word-level karaoke highlighting
```

### Image Generation: Gemini 2.5 Flash Image (Nano Banana)

```python
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Character-consistent storyboard generation
def generate_scene_image(
    prompt: str,
    style_prefix: str,
    character_description: str,
    reference_image: Image = None,
    aspect_ratio: str = "16:9"
) -> Image:
    """
    Generate scene with character consistency.
    
    Key: Include character description in EVERY prompt for consistency.
    """
    full_prompt = f"""
    {style_prefix}
    
    Character: {character_description}
    
    Scene: {prompt}
    
    Technical: Wide cinematic shot, {aspect_ratio} aspect ratio, 
    dramatic lighting, high detail, photorealistic.
    """
    
    contents = [full_prompt]
    if reference_image:
        contents.insert(0, reference_image)  # Use as style reference
    
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",  # Production ready, $0.039/image
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
            )
        )
    )
    
    for part in response.parts:
        if part.inline_data is not None:
            return Image.open(BytesIO(part.inline_data.data))
    
    return None

# Supported aspect ratios: 1:1, 3:4, 4:3, 9:16, 16:9, and more
# Pricing: ~$0.039 per image (1290 output tokens)
```

#### Character Consistency Strategy
```python
# CRITICAL: Use same character description in ALL scene prompts

CHARACTER_DESC = """
A young woman in her late 20s with shoulder-length auburn hair, 
bright green eyes, wearing a vintage blue sundress with small 
white flowers, silver pendant necklace.
"""

STYLE_PREFIX = """
Cinematic digital art style, dramatic chiaroscuro lighting,
moody color palette with warm highlights and cool shadows,
film grain texture, depth of field, 8K quality.
"""

# Generate first scene and use as reference
first_scene = generate_scene_image(
    prompt="Standing on a cliff overlooking a stormy ocean at sunset",
    style_prefix=STYLE_PREFIX,
    character_description=CHARACTER_DESC
)

# Use first scene as style reference for consistency
second_scene = generate_scene_image(
    prompt="Walking through a misty forest path in early morning",
    style_prefix=STYLE_PREFIX,
    character_description=CHARACTER_DESC,
    reference_image=first_scene  # Maintains visual consistency
)
```

### Video Generation: FFmpeg + Ken Burns Effects

```python
import subprocess
from dataclasses import dataclass
from typing import List
from enum import Enum

class KenBurnsEffect(Enum):
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    PAN_UP = "pan_up"
    PAN_DOWN = "pan_down"

@dataclass
class VideoScene:
    image_path: str
    start_time: float
    end_time: float
    effect: KenBurnsEffect
    lyrics: List[dict]  # [{"word": "...", "start": ..., "end": ...}]

def create_ken_burns_clip(
    image_path: str,
    duration: float,
    effect: KenBurnsEffect,
    output_path: str,
    resolution: str = "1920x1080",
    fps: int = 30
) -> str:
    """Create a Ken Burns effect clip from a single image."""
    
    frames = int(duration * fps)
    width, height = map(int, resolution.split('x'))
    
    # FFmpeg zoompan filter configurations
    effects = {
        KenBurnsEffect.ZOOM_IN: 
            f"zoompan=z='min(zoom+0.001,1.3)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={resolution}:fps={fps}",
        KenBurnsEffect.ZOOM_OUT: 
            f"zoompan=z='if(lte(zoom,1.0),1.3,max(1.001,zoom-0.001))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={resolution}:fps={fps}",
        KenBurnsEffect.PAN_LEFT: 
            f"zoompan=z='1.15':x='(iw-iw/zoom)*(1-on/{frames})':y='(ih-ih/zoom)/2':d={frames}:s={resolution}:fps={fps}",
        KenBurnsEffect.PAN_RIGHT: 
            f"zoompan=z='1.15':x='(iw-iw/zoom)*on/{frames}':y='(ih-ih/zoom)/2':d={frames}:s={resolution}:fps={fps}",
        KenBurnsEffect.PAN_UP: 
            f"zoompan=z='1.15':x='(iw-iw/zoom)/2':y='(ih-ih/zoom)*(1-on/{frames})':d={frames}:s={resolution}:fps={fps}",
        KenBurnsEffect.PAN_DOWN: 
            f"zoompan=z='1.15':x='(iw-iw/zoom)/2':y='(ih-ih/zoom)*on/{frames}':d={frames}:s={resolution}:fps={fps}",
    }
    
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", image_path,
        "-vf", effects[effect],
        "-t", str(duration),
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-preset", "medium",
        output_path
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path
```

### ASS Subtitle Generation for Karaoke Effects

```python
def generate_karaoke_ass(
    words: List[dict],
    output_path: str,
    video_width: int = 1920,
    video_height: int = 1080,
    font_name: str = "Arial",
    font_size: int = 48,
    primary_color: str = "&H00FFFFFF",  # White (AABBGGRR format)
    highlight_color: str = "&H0000FFFF"  # Yellow highlight
):
    """
    Generate ASS subtitle file with word-by-word karaoke highlighting.
    
    Words format: [{"word": "hello", "start": 1.23, "end": 1.56}, ...]
    """
    
    header = f"""[Script Info]
Title: Karaoke Lyrics
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_color},{highlight_color},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,2,20,20,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    # Group words into lines (max ~8 words per line for readability)
    lines = []
    current_line = []
    
    for word in words:
        current_line.append(word)
        if len(current_line) >= 8 or word['word'].endswith(('.', '!', '?', ',')):
            lines.append(current_line)
            current_line = []
    
    if current_line:
        lines.append(current_line)
    
    events = []
    
    for line_words in lines:
        if not line_words:
            continue
            
        line_start = line_words[0]['start']
        line_end = line_words[-1]['end']
        
        # Build karaoke text with \t animation tags
        text_parts = []
        for i, word in enumerate(line_words):
            # Calculate timing relative to line start (in milliseconds for ASS)
            word_start_ms = int((word['start'] - line_start) * 1000)
            word_end_ms = int((word['end'] - line_start) * 1000)
            
            # Use \t (transform) for color change animation
            # Start white, transform to yellow at word timing
            text_parts.append(
                f"{{\\1c{primary_color}\\t({word_start_ms},{word_start_ms},\\1c{highlight_color})"
                f"\\t({word_end_ms},{word_end_ms},\\1c{primary_color})}}{word['word']} "
            )
        
        line_text = "".join(text_parts).strip()
        
        # Format timestamps as H:MM:SS.CC
        start_ts = format_ass_time(line_start)
        end_ts = format_ass_time(line_end + 0.5)  # Small buffer
        
        events.append(f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{line_text}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header + '\n'.join(events))
    
    return output_path

def format_ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp format (H:MM:SS.CC)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"
```

### Dynamic Scene Generation

```python
from dataclasses import dataclass
from typing import List
import math

@dataclass
class SongSection:
    type: str  # verse, chorus, bridge, intro, outro
    start_time: float
    end_time: float
    lyrics: str
    mood: str
    visual_prompt: str

def calculate_dynamic_scenes(
    song_duration: float,
    lyrics_with_timestamps: List[dict],
    min_scene_duration: float = 4.0,
    max_scene_duration: float = 12.0,
    target_scenes_per_minute: float = 4.0
) -> List[SongSection]:
    """
    Dynamically calculate scene count and timing based on song length.
    
    Ensures:
    - Full song coverage (no gaps)
    - Natural scene breaks (preferring section boundaries)
    - Appropriate pacing (not too fast or slow)
    """
    
    # Calculate target scene count
    target_count = max(
        math.ceil(song_duration / max_scene_duration),
        min(
            math.ceil(song_duration * target_scenes_per_minute / 60),
            math.floor(song_duration / min_scene_duration)
        )
    )
    
    # Identify natural break points (silences, section changes)
    break_points = identify_section_breaks(lyrics_with_timestamps)
    
    # Distribute scenes across song
    scenes = []
    current_time = 0
    scene_duration = song_duration / target_count
    
    for i in range(target_count):
        # Find nearest natural break point
        ideal_end = current_time + scene_duration
        actual_end = find_nearest_break(break_points, ideal_end, tolerance=2.0)
        
        if actual_end is None or actual_end > song_duration:
            actual_end = min(ideal_end, song_duration)
        
        # Get lyrics for this segment
        segment_lyrics = get_lyrics_in_range(
            lyrics_with_timestamps, 
            current_time, 
            actual_end
        )
        
        # Analyze mood from lyrics using Claude
        mood = analyze_mood(segment_lyrics)
        
        scenes.append(SongSection(
            type=detect_section_type(segment_lyrics),
            start_time=current_time,
            end_time=actual_end,
            lyrics=segment_lyrics,
            mood=mood,
            visual_prompt=generate_visual_prompt(segment_lyrics, mood, i, target_count)
        ))
        
        current_time = actual_end
    
    # Ensure last scene extends to song end
    if scenes and scenes[-1].end_time < song_duration:
        scenes[-1].end_time = song_duration
    
    return scenes

def generate_visual_prompt(lyrics: str, mood: str, scene_index: int, total_scenes: int) -> str:
    """Generate a visual prompt for the scene using Claude."""
    # This would call Claude API to generate contextual visual descriptions
    # based on lyrics content, emotional arc, and narrative position
    pass
```

---

## Streamlit UI/UX Best Practices

### Session State Management

```python
import streamlit as st
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

class WorkflowStep(Enum):
    CONCEPT = 1
    LYRICS = 2
    UPLOAD = 3
    GENERATE = 4
    COMPLETE = 5

@dataclass
class AppState:
    """Centralized state management for the entire workflow."""
    current_step: WorkflowStep = WorkflowStep.CONCEPT
    
    # Step 1: Concept Workshop
    song_idea: str = ""
    genre_options: List[str] = field(default_factory=list)
    selected_genre: str = ""
    mood_options: List[str] = field(default_factory=list)
    selected_mood: str = ""
    concept_iterations: List[dict] = field(default_factory=list)
    
    # Step 2: Lyrics
    generated_lyrics: str = ""
    suno_style_tags: str = ""
    lyrics_approved: bool = False
    
    # Step 3: Upload
    uploaded_audio_path: Optional[str] = None
    audio_duration: float = 0.0
    word_timestamps: List[dict] = field(default_factory=list)
    
    # Step 4: Generation
    scenes: List[dict] = field(default_factory=list)
    generated_images: List[str] = field(default_factory=list)
    generation_progress: float = 0.0
    current_task: str = ""
    
    # Step 5: Complete
    final_video_path: Optional[str] = None

def init_session_state():
    """Initialize session state with defaults."""
    if 'app_state' not in st.session_state:
        st.session_state.app_state = AppState()
    
    # Ensure all attributes exist (for app updates)
    state = st.session_state.app_state
    for field_name, field_def in AppState.__dataclass_fields__.items():
        if not hasattr(state, field_name):
            setattr(state, field_name, field_def.default_factory() 
                    if field_def.default_factory != field_def.default 
                    else field_def.default)

def get_state() -> AppState:
    """Get current app state."""
    init_session_state()
    return st.session_state.app_state

def update_state(**kwargs):
    """Update state attributes."""
    state = get_state()
    for key, value in kwargs.items():
        if hasattr(state, key):
            setattr(state, key, value)
```

### Multi-Step Wizard UI

```python
import streamlit as st

def render_wizard():
    """Render the multi-step wizard interface."""
    state = get_state()
    
    # Progress indicator
    st.markdown("---")
    cols = st.columns(5)
    steps = [
        ("💡", "Concept"),
        ("🎵", "Lyrics"),
        ("📤", "Upload"),
        ("🎬", "Generate"),
        ("✅", "Complete")
    ]
    
    for i, (col, (icon, label)) in enumerate(zip(cols, steps), 1):
        step_enum = WorkflowStep(i)
        with col:
            if state.current_step.value > i:
                st.markdown(f"### ✅ {label}")
            elif state.current_step == step_enum:
                st.markdown(f"### 🔵 **{label}**")
            else:
                st.markdown(f"### ⚪ {label}")
    
    st.markdown("---")
    
    # Render current step
    if state.current_step == WorkflowStep.CONCEPT:
        render_concept_workshop()
    elif state.current_step == WorkflowStep.LYRICS:
        render_lyrics_generator()
    elif state.current_step == WorkflowStep.UPLOAD:
        render_audio_upload()
    elif state.current_step == WorkflowStep.GENERATE:
        render_video_generator()
    elif state.current_step == WorkflowStep.COMPLETE:
        render_completion()

def render_concept_workshop():
    """Step 1: Iterative song concept development."""
    state = get_state()
    
    st.header("🎨 Song Concept Workshop")
    st.markdown("""
    Let's develop your song idea together. Describe what you're thinking,
    and I'll research options and help you refine the concept.
    """)
    
    # Chat-style interface for iterative development
    if 'concept_messages' not in st.session_state:
        st.session_state.concept_messages = []
    
    # Display conversation history
    for msg in st.session_state.concept_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # User input
    if user_input := st.chat_input("Describe your song idea..."):
        st.session_state.concept_messages.append({
            "role": "user",
            "content": user_input
        })
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # AI researches and responds with options
        with st.chat_message("assistant"):
            with st.status("Researching genres and styles..."):
                response = research_and_suggest(user_input, state.concept_iterations)
            st.markdown(response["message"])
            
            # Show selectable options if provided
            if response.get("genre_options"):
                st.markdown("**Suggested Genres:**")
                for genre in response["genre_options"]:
                    if st.button(genre, key=f"genre_{genre}"):
                        update_state(selected_genre=genre)
            
            if response.get("mood_options"):
                st.markdown("**Suggested Moods:**")
                for mood in response["mood_options"]:
                    if st.button(mood, key=f"mood_{mood}"):
                        update_state(selected_mood=mood)
        
        st.session_state.concept_messages.append({
            "role": "assistant",
            "content": response["message"]
        })
    
    # Proceed when concept is finalized
    col1, col2 = st.columns([3, 1])
    with col2:
        if state.selected_genre and state.selected_mood:
            if st.button("✨ Generate Lyrics", type="primary"):
                update_state(current_step=WorkflowStep.LYRICS)
                st.rerun()

def render_video_generator():
    """Step 4: Generate video with progress feedback."""
    state = get_state()
    
    st.header("🎬 Generating Your Music Video")
    
    # Use st.status for detailed progress
    with st.status("Creating your music video...", expanded=True) as status:
        
        # Calculate scenes
        st.write("📊 Analyzing song structure...")
        scenes = calculate_dynamic_scenes(
            state.audio_duration,
            state.word_timestamps
        )
        update_state(scenes=scenes)
        st.write(f"   → Created {len(scenes)} scenes for {state.audio_duration:.1f}s song")
        
        # Progress bar for image generation
        progress_bar = st.progress(0, text="Generating scene images...")
        
        generated_images = []
        for i, scene in enumerate(scenes):
            st.write(f"🎨 Generating scene {i+1}/{len(scenes)}: {scene.mood}")
            
            image_path = generate_scene_image(
                prompt=scene.visual_prompt,
                style_prefix=get_style_prefix(),
                character_description=get_character_description()
            )
            generated_images.append(image_path)
            
            progress = (i + 1) / len(scenes)
            progress_bar.progress(progress, text=f"Generated {i+1}/{len(scenes)} images")
        
        update_state(generated_images=generated_images)
        
        # Video assembly
        st.write("🎥 Assembling video with Ken Burns effects...")
        video_path = assemble_video(
            scenes=scenes,
            images=generated_images,
            audio_path=state.uploaded_audio_path,
            word_timestamps=state.word_timestamps
        )
        
        st.write("📝 Adding synchronized lyrics overlay...")
        final_path = add_lyrics_overlay(
            video_path=video_path,
            word_timestamps=state.word_timestamps
        )
        
        update_state(final_video_path=final_path)
        status.update(label="✅ Video complete!", state="complete")
    
    # Auto-advance to completion
    update_state(current_step=WorkflowStep.COMPLETE)
    st.rerun()
```

---

## Project Structure

```
suno-music-video-generator/
├── CLAUDE.md                    # This file - project context for Claude Code
├── .claude/
│   └── commands/
│       ├── generate-song.md     # /project:generate-song command
│       ├── fix-sync.md          # /project:fix-sync command
│       └── regenerate-scene.md  # /project:regenerate-scene command
├── pyproject.toml               # Python project config
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── main.py                  # Entry point
│   ├── config.py                # Configuration management
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── concept_agent.py     # Iterative song concept development
│   │   ├── lyrics_agent.py      # Lyrics generation with Suno tags
│   │   └── visual_agent.py      # Scene planning and image prompts
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── audio_processor.py   # WhisperX integration
│   │   ├── image_generator.py   # Gemini/Nano Banana integration
│   │   ├── video_generator.py   # FFmpeg Ken Burns + assembly
│   │   ├── lyrics_sync.py       # Word-level alignment
│   │   └── subtitle_generator.py # ASS karaoke generation
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py           # Pydantic models
│   │
│   └── ui/
│       ├── __init__.py
│       ├── app.py               # Main Streamlit app
│       ├── components/          # Reusable UI components
│       │   ├── wizard.py
│       │   ├── progress.py
│       │   └── player.py
│       └── pages/               # Multi-page structure
│           ├── concept.py
│           ├── lyrics.py
│           ├── upload.py
│           └── generate.py
│
├── output/
│   ├── songs/                   # Uploaded MP3s
│   ├── images/                  # Generated scene images
│   ├── videos/                  # Final videos
│   └── subtitles/               # ASS/LRC files
│
└── tests/
    ├── test_audio.py
    ├── test_video.py
    └── test_sync.py
```

---

## Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...        # Claude for reasoning/lyrics
GOOGLE_API_KEY=...                   # Gemini/Nano Banana for images

# Optional
USE_NANO_BANANA_PRO=false            # Use Gemini 3 Pro Image (higher quality)
DEFAULT_ART_STYLE="cinematic digital art, dramatic lighting, 8k quality"
VIDEO_RESOLUTION=1920x1080
VIDEO_FPS=30
WHISPER_MODEL=large-v3               # large-v3 recommended for accuracy
WHISPER_DEVICE=cuda                  # or cpu for Mac
```

---

## Dependencies

```toml
[project]
dependencies = [
    # Core
    "anthropic>=0.40.0",
    "google-genai>=1.52.0",
    "pydantic>=2.9.0",
    
    # Audio Processing
    "whisperx>=3.1.0",           # Word-level timestamps
    "faster-whisper>=1.1.0",     # Backend for WhisperX
    "pydub>=0.25.1",             # Audio manipulation
    "torchaudio>=2.1.0",
    
    # Video Processing
    "Pillow>=10.4.0",
    
    # UI
    "streamlit>=1.40.0",
    
    # Utilities
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
gpu = [
    "torch>=2.1.0",
    "ctranslate2>=4.4.0",
]
```

### System Requirements
- Python 3.11+
- FFmpeg with libx264 (for video encoding)
- CUDA 12+ (for GPU acceleration, optional)

---

## Key Workflows

### Iterative Song Concept Workflow

1. User describes initial idea
2. AI researches genres, artists, and styles
3. AI presents 3-5 options with explanations
4. User selects or requests alternatives
5. AI refines based on feedback
6. Repeat until user is satisfied
7. Generate final lyrics and Suno tags

### Full Song Lyrics Synchronization

1. User uploads MP3 from Suno
2. WhisperX transcribes with word-level timestamps
3. Match transcription against known lyrics (from generation step)
4. Correct any misalignments using alignment model
5. Generate ASS file with karaoke highlighting
6. Sync scenes to section boundaries
7. Overlay synchronized lyrics on video

### Dynamic Scene Generation

1. Calculate song duration from audio
2. Identify natural break points (silences, sections)
3. Calculate optimal scene count (4 scenes/minute target)
4. Assign scenes to time ranges
5. Analyze lyrics mood for each scene
6. Generate visual prompts with character consistency
7. Create images with Nano Banana
8. Apply Ken Burns effects based on mood
9. Assemble with crossfades

---

## Testing Checklist

- [ ] Audio processing works on CPU (Mac compatibility)
- [ ] Word timestamps align within 100ms of actual
- [ ] All scenes cover full song duration (no gaps)
- [ ] Character consistency maintained across scenes
- [ ] Lyrics overlay readable at all times
- [ ] Video exports without errors
- [ ] Session state persists through workflow
- [ ] Progress feedback updates in real-time
- [ ] Error handling shows user-friendly messages

---

## Common Commands

```bash
# Run the Streamlit app
streamlit run src/ui/app.py

# Test audio processing
python -m pytest tests/test_audio.py -v

# Generate video from existing assets
python src/main.py --audio song.mp3 --lyrics lyrics.txt --output video.mp4

# Debug word timestamps
python -c "import whisperx; ..."
```

---

## Notes for Claude Code

- Always read this entire CLAUDE.md before starting work
- Use `st.status()` for long-running operations with detailed feedback
- Maintain session state across all user interactions
- Generate dynamic scene counts based on actual song duration
- Never leave gaps in video - every second must have content
- Test word-level sync accuracy before full video generation
- Keep character descriptions consistent across all image prompts
- Use the iterative workflow for concept development - don't rush to generation

### Code Style
- Type hints on all functions
- Pydantic models for data validation
- Async where beneficial for I/O
- Clear error messages for users
- Progress callbacks for long operations
