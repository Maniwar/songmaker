#!/usr/bin/env python3
"""Quick test of Wan2.2-S2V API."""
import sys
import tempfile
import time
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from pydub import AudioSegment
from gradio_client import Client, handle_file

print("Testing Wan2.2-S2V API...")

# Use existing files
image_path = Path("output/untitled_20251127_130738/images/scene_000.png")
audio_path = Path("output/songs/kawaii kitten v3.mp3")

if not image_path.exists():
    print(f"Error: Image not found: {image_path}")
    sys.exit(1)
if not audio_path.exists():
    print(f"Error: Audio not found: {audio_path}")
    sys.exit(1)

print(f"Image: {image_path}")
print(f"Audio: {audio_path}")

# Extract 5 seconds of audio
with tempfile.TemporaryDirectory() as temp_dir:
    temp_dir = Path(temp_dir)
    audio_clip_path = temp_dir / "audio_clip.wav"

    print("Extracting 5-second audio clip...")
    audio = AudioSegment.from_file(str(audio_path))
    clip = audio[0:5000]  # First 5 seconds
    clip.export(str(audio_clip_path), format="wav")
    print(f"Audio clip saved to: {audio_clip_path}")

    print("Connecting to Wan2.2-S2V...")
    client = Client("Wan-AI/Wan2.2-S2V")
    print("Connected!")

    print("Calling API (this may take 10+ minutes)...")
    start = time.time()
    try:
        result = client.predict(
            ref_img=handle_file(str(image_path)),
            audio=handle_file(str(audio_clip_path)),
            resolution="480P",
            api_name="/predict"
        )
        elapsed = time.time() - start
        print(f"API call completed in {elapsed:.1f}s")
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")

        # Check result
        if isinstance(result, dict):
            video_path = result.get('video')
            if video_path:
                print(f"Video path: {video_path}")
                print(f"Video exists: {Path(video_path).exists()}")
        else:
            print(f"Unexpected result type: {type(result)}")
    except Exception as e:
        elapsed = time.time() - start
        print(f"API call failed after {elapsed:.1f}s")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
