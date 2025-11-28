"""Wizard progress indicator component."""

import streamlit as st

from src.models.schemas import WorkflowStep


STEPS = [
    (WorkflowStep.CONCEPT, "Concept", "Develop your song idea"),
    (WorkflowStep.LYRICS, "Lyrics", "Generate lyrics for Suno"),
    (WorkflowStep.UPLOAD, "Upload", "Upload your MP3 from Suno"),
    (WorkflowStep.VISUAL, "Visual", "Design your visual style"),
    (WorkflowStep.GENERATE, "Generate", "Create your music video"),
    (WorkflowStep.COMPLETE, "Complete", "Download and share"),
]


def render_wizard_progress(current_step: WorkflowStep) -> None:
    """
    Render the wizard progress indicator.

    Args:
        current_step: The current workflow step
    """
    st.markdown("---")

    cols = st.columns(len(STEPS))

    # Find current step index for proper comparison
    current_idx = next(
        (i for i, (step, _, _) in enumerate(STEPS) if step == current_step),
        0
    )

    for i, (col, (step, label, _)) in enumerate(zip(cols, STEPS)):
        with col:
            if i < current_idx:
                # Completed step
                st.markdown(f"### :white_check_mark: {label}")
            elif current_step == step:
                # Current step
                st.markdown(f"### :large_blue_circle: **{label}**")
            else:
                # Future step
                st.markdown(f"### :white_circle: {label}")

    st.markdown("---")


def get_step_description(step: WorkflowStep) -> str:
    """Get the description for a workflow step."""
    for s, _, desc in STEPS:
        if s == step:
            return desc
    return ""
