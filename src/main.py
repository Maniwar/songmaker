"""Main entry point for Songmaker CLI."""

import argparse
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Songmaker - AI-powered music video generator"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # UI command
    ui_parser = subparsers.add_parser("ui", help="Launch the Streamlit UI")

    # Generate command
    gen_parser = subparsers.add_parser(
        "generate", help="Generate video from existing assets"
    )
    gen_parser.add_argument("--audio", type=Path, required=True, help="Path to audio file")
    gen_parser.add_argument("--lyrics", type=Path, help="Path to lyrics file")
    gen_parser.add_argument("--output", type=Path, required=True, help="Output video path")
    gen_parser.add_argument("--style", type=str, help="Visual style description")

    # Transcribe command
    trans_parser = subparsers.add_parser("transcribe", help="Transcribe audio to get timestamps")
    trans_parser.add_argument("audio", type=Path, help="Path to audio file")
    trans_parser.add_argument("--output", type=Path, help="Output JSON path")

    args = parser.parse_args()

    if args.command == "ui":
        run_ui()
    elif args.command == "generate":
        run_generate(args.audio, args.lyrics, args.output, args.style)
    elif args.command == "transcribe":
        run_transcribe(args.audio, args.output)
    else:
        parser.print_help()


def run_ui():
    """Launch the Streamlit UI."""
    import subprocess
    import sys
    from pathlib import Path

    app_path = Path(__file__).parent / "ui" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


def run_generate(audio_path: Path, lyrics_path: Path, output_path: Path, style: str):
    """Generate video from existing assets."""
    from src.config import config
    from src.services.audio_processor import AudioProcessor
    from src.services.image_generator import ImageGenerator
    from src.services.video_generator import VideoGenerator
    from src.services.subtitle_generator import SubtitleGenerator
    from src.agents.visual_agent import VisualAgent
    from src.models.schemas import SongConcept

    print(f"Processing audio: {audio_path}")

    # Transcribe audio
    processor = AudioProcessor()
    transcript = processor.transcribe(audio_path)
    print(f"Transcribed {len(transcript.all_words)} words")
    processor.cleanup()

    # Load lyrics if provided
    lyrics_text = ""
    if lyrics_path and lyrics_path.exists():
        lyrics_text = lyrics_path.read_text()

    # Create a basic concept
    concept = SongConcept(
        idea="Generated from CLI",
        genre="various",
        mood="dynamic",
        themes=["music"],
        visual_style=style or config.image.default_style,
    )

    # Plan scenes
    print("Planning scenes...")
    visual_agent = VisualAgent()
    scenes = visual_agent.plan_video(
        concept=concept,
        transcript=transcript,
        full_lyrics=lyrics_text,
    )
    print(f"Created {len(scenes)} scenes")

    # Generate images
    print("Generating images...")
    config.ensure_directories()
    image_gen = ImageGenerator()

    def progress(msg, p):
        print(f"  {msg}")

    image_paths = image_gen.generate_storyboard(
        scene_prompts=[s.visual_prompt for s in scenes],
        style_prefix=concept.visual_style,
        character_description="",
        output_dir=config.images_dir,
        progress_callback=progress,
    )

    for scene, path in zip(scenes, image_paths):
        scene.image_path = path

    # Generate subtitles
    print("Creating subtitles...")
    subtitle_gen = SubtitleGenerator()
    subtitle_path = config.subtitles_dir / "lyrics.ass"
    subtitle_gen.generate_from_transcript(transcript, subtitle_path)

    # Generate video
    print("Assembling video...")
    video_gen = VideoGenerator()
    video_gen.generate_music_video(
        scenes=scenes,
        audio_path=audio_path,
        subtitle_path=subtitle_path,
        output_path=output_path,
        progress_callback=progress,
    )

    print(f"Video saved to: {output_path}")


def run_transcribe(audio_path: Path, output_path: Path = None):
    """Transcribe audio and output timestamps."""
    import json

    from src.services.audio_processor import AudioProcessor

    print(f"Transcribing: {audio_path}")

    processor = AudioProcessor()
    transcript = processor.transcribe(audio_path)
    processor.cleanup()

    print(f"Duration: {transcript.duration:.2f}s")
    print(f"Language: {transcript.language}")
    print(f"Words: {len(transcript.all_words)}")
    print(f"Segments: {len(transcript.segments)}")

    if output_path:
        data = {
            "duration": transcript.duration,
            "language": transcript.language,
            "segments": [
                {
                    "text": seg.text,
                    "start": seg.start,
                    "end": seg.end,
                    "words": [
                        {"word": w.word, "start": w.start, "end": w.end}
                        for w in seg.words
                    ],
                }
                for seg in transcript.segments
            ],
        }
        output_path.write_text(json.dumps(data, indent=2))
        print(f"Saved to: {output_path}")
    else:
        print("\nFirst 10 words:")
        for word in transcript.all_words[:10]:
            print(f"  {word.start:.2f}s - {word.end:.2f}s: {word.word}")


if __name__ == "__main__":
    main()
