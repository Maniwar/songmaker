"""Session state management for Streamlit."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

from src.models.schemas import AppState, WorkflowStep


# Default projects directory
PROJECTS_DIR = Path("projects")


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

        # New project
        st.divider()
        if st.button("New Project", key="new_project_btn", use_container_width=True):
            reset_state()
            st.rerun()
