"""Session state management for Streamlit."""

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

from src.models.schemas import AppState, WorkflowStep


# Default projects directory
PROJECTS_DIR = Path("projects")

# Scene metadata filename
SCENE_METADATA_FILE = "scenes.json"


def save_scene_metadata(project_path: Path, scenes: list) -> bool:
    """
    Save scene metadata (prompts, effects, etc.) to a JSON file.

    This allows prompts and other scene data to be recovered when loading a project.

    Args:
        project_path: Path to the project directory
        scenes: List of Scene objects

    Returns:
        True if save succeeded
    """
    try:
        metadata = []
        for scene in scenes:
            scene_data = {
                "index": scene.index,
                "start_time": scene.start_time,
                "end_time": scene.end_time,
                "visual_prompt": scene.visual_prompt,
                "mood": scene.mood,
                "effect": scene.effect.value if hasattr(scene.effect, 'value') else scene.effect,
                "motion_prompt": getattr(scene, 'motion_prompt', None),
                "animation_type": scene.animation_type.value if hasattr(scene.animation_type, 'value') else str(scene.animation_type),
                "animated": getattr(scene, 'animated', False),
            }
            metadata.append(scene_data)

        metadata_path = project_path / SCENE_METADATA_FILE
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return True
    except Exception as e:
        print(f"Failed to save scene metadata: {e}")
        return False


def load_scene_metadata(project_path: Path) -> Optional[dict]:
    """
    Load scene metadata from JSON file.

    Returns:
        Dictionary mapping scene index to metadata, or None if file doesn't exist
    """
    metadata_path = project_path / SCENE_METADATA_FILE
    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path, 'r') as f:
            metadata_list = json.load(f)

        # Convert to dict keyed by index for easy lookup
        return {item['index']: item for item in metadata_list}
    except Exception as e:
        print(f"Failed to load scene metadata: {e}")
        return None


def _sanitize_path_component(name: str) -> str:
    """Sanitize a single path component (directory or file name)."""
    # Replace spaces with underscores
    sanitized = name.replace(" ", "_")
    # Remove problematic characters: / \ : * ? " < > |
    sanitized = re.sub(r'[/\\:*?"<>|]', '_', sanitized)
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized


def fix_malformed_project_path(state: AppState) -> bool:
    """
    Fix malformed project paths that contain special characters.

    Some older projects may have been created with slashes or other
    special characters in the project name, creating nested directories.
    This function detects and fixes such paths.

    Args:
        state: The app state to check and fix

    Returns:
        True if path was fixed, False if no fix needed
    """
    if not state.project_dir:
        return False

    project_path = Path(state.project_dir)

    # Check if the path exists as-is
    if project_path.exists():
        return False

    # The path doesn't exist - could be malformed or just missing
    # Try to find if it was split into nested directories
    # e.g., "output/epic_power_metal_/_d_..." should be "output/epic_power_metal_d_..."

    parent = project_path.parent
    if not parent.exists():
        return False

    # Look for directories that might match the expected pattern
    project_name = project_path.name
    sanitized_name = _sanitize_path_component(project_name)

    # Check if a sanitized version exists
    sanitized_path = parent / sanitized_name
    if sanitized_path.exists():
        state.project_dir = str(sanitized_path)
        return True

    # Check for nested directory structure caused by slashes in name
    # e.g., looking for "output/epic_power_metal_/_d_timestamp" where
    # "epic_power_metal_" and "_d_timestamp" are separate directories
    for subdir in parent.iterdir():
        if subdir.is_dir():
            # Look for nested directories that together match the pattern
            for nested in subdir.iterdir():
                if nested.is_dir():
                    # Reconstruct what the combined name would be
                    combined = f"{subdir.name}{nested.name}"
                    if combined.startswith(sanitized_name[:10]):
                        # Found a match - rename it
                        new_path = parent / _sanitize_path_component(combined)
                        try:
                            shutil.move(str(nested), str(new_path))
                            # Try to remove the now-empty parent
                            if not list(subdir.iterdir()):
                                subdir.rmdir()
                            state.project_dir = str(new_path)
                            return True
                        except Exception:
                            pass

    return False


def init_session_state() -> None:
    """Initialize session state with defaults."""
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState()


def get_state() -> AppState:
    """Get the current app state."""
    init_session_state()
    return st.session_state.app_state


def update_state(**kwargs) -> None:
    """
    Update state attributes.

    Args:
        **kwargs: Key-value pairs to update
    """
    state = get_state()
    for key, value in kwargs.items():
        if hasattr(state, key):
            setattr(state, key, value)


def advance_step() -> None:
    """Advance to the next workflow step."""
    state = get_state()
    steps = list(WorkflowStep)
    current_index = steps.index(state.current_step)
    if current_index < len(steps) - 1:
        state.current_step = steps[current_index + 1]


def go_to_step(step: WorkflowStep) -> None:
    """Go to a specific workflow step."""
    state = get_state()
    state.current_step = step


def reset_state() -> None:
    """Reset the entire app state."""
    st.session_state.app_state = AppState()


def get_projects_dir() -> Path:
    """Get the projects directory, creating it if needed."""
    PROJECTS_DIR.mkdir(exist_ok=True)
    return PROJECTS_DIR


def list_saved_projects() -> list[Path]:
    """List all saved project files."""
    projects_dir = get_projects_dir()
    return sorted(projects_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def save_state(project_name: Optional[str] = None) -> Path:
    """
    Save the current app state to a JSON file.

    Args:
        project_name: Optional name for the project. If not provided,
                     uses timestamp or song idea.

    Returns:
        Path to the saved file.
    """
    state = get_state()
    projects_dir = get_projects_dir()

    # Generate filename
    if project_name:
        filename = project_name
    elif state.concept and state.concept.idea:
        # Use first few words of the idea
        words = state.concept.idea.split()[:4]
        filename = "_".join(words)
    else:
        filename = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Clean filename
    filename = "".join(c if c.isalnum() or c in "_-" else "_" for c in filename)
    filepath = projects_dir / f"{filename}.json"

    # Serialize state using Pydantic's model_dump
    state_dict = state.model_dump(mode="json")

    # Add metadata
    save_data = {
        "version": "1.0",
        "saved_at": datetime.now().isoformat(),
        "state": state_dict,
    }

    with open(filepath, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    return filepath


def load_state(filepath: Path) -> bool:
    """
    Load app state from a saved JSON file.

    Args:
        filepath: Path to the saved project file.

    Returns:
        True if loaded successfully, False otherwise.
    """
    try:
        with open(filepath) as f:
            save_data = json.load(f)

        state_dict = save_data.get("state", save_data)

        # Create new AppState from loaded data
        loaded_state = AppState.model_validate(state_dict)

        # Fix any malformed project paths from older projects
        if fix_malformed_project_path(loaded_state):
            st.info("Project path was fixed automatically.")

        st.session_state.app_state = loaded_state
        return True
    except Exception as e:
        st.error(f"Failed to load project: {e}")
        return False


def delete_project(filepath: Path) -> bool:
    """Delete a saved project file."""
    try:
        filepath.unlink()
        return True
    except Exception:
        return False


def get_project_info(filepath: Path) -> dict:
    """Get summary info about a saved project."""
    try:
        with open(filepath) as f:
            save_data = json.load(f)

        state_dict = save_data.get("state", save_data)
        saved_at = save_data.get("saved_at", "Unknown")

        # Extract key info
        concept = state_dict.get("concept")
        idea = concept.get("idea", "Unknown") if concept else "No concept yet"
        step = state_dict.get("current_step", "concept")

        return {
            "name": filepath.stem,
            "idea": idea[:50] + "..." if len(idea) > 50 else idea,
            "step": step,
            "saved_at": saved_at,
        }
    except Exception:
        return {
            "name": filepath.stem,
            "idea": "Unable to read",
            "step": "unknown",
            "saved_at": "Unknown",
        }


def scan_recoverable_projects() -> list[dict]:
    """
    Scan the output directory for projects that can be recovered.

    A recoverable project is a directory with an images/ subfolder containing
    scene images. Returns info about each recoverable project.
    """
    from src.config import config

    recoverable = []
    output_dir = config.output_dir

    if not output_dir.exists():
        return []

    for item in output_dir.iterdir():
        if not item.is_dir():
            continue
        # Skip special directories
        if item.name in ("images", "videos", "subtitles", "songs"):
            continue

        # Check for images directory with scene images
        images_dir = item / "images"
        if not images_dir.exists():
            continue

        scene_images = sorted(images_dir.glob("scene_*.png"))
        if not scene_images:
            continue

        # Check for other assets
        animations_dir = item / "animations"
        animations = list(animations_dir.glob("animated_scene_*.mp4")) if animations_dir.exists() else []
        lyrics_file = item / "lyrics.ass"
        final_videos = list(item.glob("*.mp4"))
        # Filter out animation files from final videos
        final_videos = [v for v in final_videos if "animated_scene" not in v.name]

        recoverable.append({
            "path": item,
            "name": item.name,
            "image_count": len(scene_images),
            "animation_count": len(animations),
            "has_lyrics": lyrics_file.exists(),
            "has_final_video": len(final_videos) > 0,
            "final_video": final_videos[0] if final_videos else None,
            "modified": item.stat().st_mtime,
        })

    # Sort by modification time, newest first
    recoverable.sort(key=lambda x: x["modified"], reverse=True)
    return recoverable


def recover_project(project_path: Path, audio_path: Optional[Path] = None) -> bool:
    """
    Recover a project from its output directory.

    This reconstructs minimal state from the saved files, allowing the user
    to resume from the storyboard view.

    Args:
        project_path: Path to the project directory
        audio_path: Optional path to the audio file

    Returns:
        True if recovery succeeded
    """
    from src.models.schemas import Scene, KenBurnsEffect, AnimationType, Word, Transcript, Segment

    images_dir = project_path / "images"
    animations_dir = project_path / "animations"
    lyrics_file = project_path / "lyrics.ass"

    # Find scene images
    scene_images = sorted(images_dir.glob("scene_*.png"))
    if not scene_images:
        return False

    # Load scene metadata if available
    scene_metadata = load_scene_metadata(project_path)

    # Create scenes from images
    scenes = []
    for img_path in scene_images:
        # Extract scene index from filename (scene_XXX.png)
        try:
            idx = int(img_path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue

        # Check for matching animation
        animation_path = animations_dir / f"animated_scene_{idx:03d}.mp4"
        has_animation = animation_path.exists()

        # Get metadata for this scene if available
        meta = scene_metadata.get(idx, {}) if scene_metadata else {}

        # Parse effect from metadata, or cycle through effects for variety
        effect = None
        if meta.get("effect"):
            try:
                effect = KenBurnsEffect(meta["effect"])
            except ValueError:
                pass

        # If no effect from metadata (legacy project), cycle through effects based on index
        if effect is None:
            all_effects = list(KenBurnsEffect)
            effect = all_effects[idx % len(all_effects)]

        # Parse animation_type from metadata
        animation_type = AnimationType.NONE  # default
        if meta.get("animation_type"):
            try:
                animation_type = AnimationType(meta["animation_type"])
            except ValueError:
                pass

        # Create scene with metadata values or placeholders
        scene = Scene(
            index=idx,
            start_time=meta.get("start_time", 0.0),
            end_time=meta.get("end_time", 0.0),
            visual_prompt=meta.get("visual_prompt", "(Recovered from files)"),
            mood=meta.get("mood", "unknown"),
            effect=effect,
            image_path=img_path,
            words=[],
            animated=meta.get("animated", has_animation),
            animation_type=animation_type,
            motion_prompt=meta.get("motion_prompt"),
            video_path=animation_path if has_animation else None,
        )
        scenes.append(scene)

    # Sort scenes by index
    scenes.sort(key=lambda s: s.index)

    # Try to parse lyrics from .ass file if present
    words_by_scene = {}
    if lyrics_file.exists():
        try:
            words_by_scene = _parse_ass_file(lyrics_file)
        except Exception:
            pass

    # Estimate timing based on audio duration or scene count
    audio_duration = 0.0
    if audio_path and audio_path.exists():
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(audio_path))
            audio_duration = len(audio) / 1000.0
        except Exception:
            pass

    # Assign timing to scenes
    if audio_duration > 0 and scenes:
        scene_duration = audio_duration / len(scenes)
        for i, scene in enumerate(scenes):
            scene.start_time = i * scene_duration
            scene.end_time = (i + 1) * scene_duration
            # Assign words from parsed .ass file if available
            if scene.index in words_by_scene:
                scene.words = words_by_scene[scene.index]

    # Create minimal transcript for compatibility
    all_words = []
    for scene in scenes:
        all_words.extend(scene.words)

    transcript = Transcript(
        segments=[Segment(
            text=" ".join(w.word for w in all_words),
            start=0.0,
            end=audio_duration,
            words=all_words,
        )] if all_words else [],
        language="en",
        duration=audio_duration,
    )

    # Create new state with recovered data
    new_state = AppState(
        current_step=WorkflowStep.GENERATE,
        scenes=scenes,
        project_dir=str(project_path),
        transcript=transcript,
        audio_path=str(audio_path) if audio_path else None,
        audio_duration=audio_duration,
        prompts_ready=True,
        storyboard_ready=True,
        generated_images=[str(s.image_path) for s in scenes if s.image_path],
    )

    st.session_state.app_state = new_state
    return True


def _parse_ass_file(ass_path: Path) -> dict[int, list]:
    """
    Parse an ASS subtitle file to extract word timing.

    Returns a dict mapping scene index to list of Word objects.
    This is a best-effort parser for karaoke-style ASS files.
    """
    from src.models.schemas import Word

    words_by_scene = {}

    try:
        with open(ass_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Find Dialogue lines
        import re
        dialogue_pattern = r"Dialogue:\s*\d+,(\d+:\d+:\d+\.\d+),(\d+:\d+:\d+\.\d+),.*?,,.*?,(.*)"

        for match in re.finditer(dialogue_pattern, content):
            start_str, end_str, text = match.groups()

            # Parse timestamps (H:MM:SS.CC format)
            def parse_time(t):
                parts = t.replace(".", ":").split(":")
                h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
                cs = int(parts[3]) if len(parts) > 3 else 0
                return h * 3600 + m * 60 + s + cs / 100.0

            start = parse_time(start_str)
            end = parse_time(end_str)

            # Clean text of ASS tags
            clean_text = re.sub(r"\{[^}]*\}", "", text).strip()
            if not clean_text:
                continue

            # Split into words and distribute timing
            word_list = clean_text.split()
            if not word_list:
                continue

            duration = end - start
            word_duration = duration / len(word_list)

            # Estimate which scene this belongs to (will be refined later)
            # For now, just collect all words
            for i, word_text in enumerate(word_list):
                word = Word(
                    word=word_text,
                    start=start + i * word_duration,
                    end=start + (i + 1) * word_duration,
                )
                # Scene index will be determined by timing later
                # For now, use scene 0 as placeholder
                if 0 not in words_by_scene:
                    words_by_scene[0] = []
                words_by_scene[0].append(word)

    except Exception:
        pass

    return words_by_scene


def list_audio_files() -> list[Path]:
    """List available audio files in output/songs directory."""
    from src.config import config

    songs_dir = config.output_dir / "songs"
    if not songs_dir.exists():
        return []

    audio_files = []
    for ext in ["*.mp3", "*.wav", "*.m4a", "*.ogg"]:
        audio_files.extend(songs_dir.glob(ext))

    return sorted(audio_files, key=lambda p: p.stat().st_mtime, reverse=True)


def render_project_sidebar() -> None:
    """Render the project management sidebar section."""
    with st.sidebar:
        st.header("Project")

        # Save current project
        with st.expander("Save Project", expanded=False):
            project_name = st.text_input(
                "Project name (optional)",
                placeholder="Auto-generated from song idea",
                key="save_project_name",
            )
            if st.button("Save", key="save_btn", use_container_width=True):
                filepath = save_state(project_name if project_name else None)
                st.success(f"Saved to {filepath.name}")

        # Load existing project
        saved_projects = list_saved_projects()
        if saved_projects:
            with st.expander("Load Project", expanded=False):
                for project_path in saved_projects[:10]:  # Show last 10
                    info = get_project_info(project_path)
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        if st.button(
                            f"{info['name']}\n{info['step']}",
                            key=f"load_{project_path.stem}",
                            use_container_width=True,
                        ):
                            if load_state(project_path):
                                st.success("Loaded!")
                                st.rerun()
                    with col2:
                        if st.button("X", key=f"del_{project_path.stem}"):
                            if delete_project(project_path):
                                st.rerun()

        # Recover project from files
        recoverable_projects = scan_recoverable_projects()
        if recoverable_projects:
            with st.expander("Recover Project", expanded=False):
                st.caption("Recover from output files")

                # List available audio files for pairing
                audio_files = list_audio_files()
                audio_options = ["(No audio)"] + [f.name for f in audio_files]

                for proj in recoverable_projects[:8]:  # Show last 8
                    from datetime import datetime
                    modified = datetime.fromtimestamp(proj["modified"]).strftime("%m/%d %H:%M")

                    # Status icons
                    status = f"{proj['image_count']} imgs"
                    if proj["animation_count"] > 0:
                        status += f", {proj['animation_count']} anims"
                    if proj["has_final_video"]:
                        status += " [video]"

                    st.markdown(f"**{proj['name'][:25]}**")
                    st.caption(f"{status} - {modified}")

                    # Audio selection for this project
                    selected_audio = st.selectbox(
                        "Audio file",
                        options=audio_options,
                        key=f"recover_audio_{proj['name']}",
                        label_visibility="collapsed",
                    )

                    if st.button("Recover", key=f"recover_{proj['name']}", use_container_width=True):
                        audio_path = None
                        if selected_audio != "(No audio)":
                            for af in audio_files:
                                if af.name == selected_audio:
                                    audio_path = af
                                    break

                        if recover_project(proj["path"], audio_path):
                            st.success("Project recovered!")
                            st.rerun()
                        else:
                            st.error("Recovery failed")

                    st.markdown("---")

        # New project
        st.divider()
        if st.button("New Project", key="new_project_btn", use_container_width=True):
            reset_state()
            st.rerun()
