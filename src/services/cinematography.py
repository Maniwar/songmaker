"""Cinematography planning service for context-aware camera work.

This service generates and manages cinematography plans that ensure:
- Natural camera angle transitions between scenes
- Narrative-aware shot selection (opening wide, climax close-ups, etc.)
- Emotional intensity awareness (high emotion = closer shots)
- Scene-to-scene continuity for professional cinematography
"""

import logging
from typing import Optional

from src.models.schemas import (
    CameraAngle,
    CinematographyPlan,
    Emotion,
    MovieScene,
    SceneCameraContext,
    SceneType,
)

logger = logging.getLogger(__name__)

# Camera transition rules: natural shot progressions
# Format: {from_camera: [list of natural next cameras]}
CAMERA_TRANSITION_RULES = {
    CameraAngle.EXTREME_WIDE: [CameraAngle.WIDE, CameraAngle.MEDIUM_WIDE],
    CameraAngle.WIDE: [CameraAngle.MEDIUM_WIDE, CameraAngle.MEDIUM, CameraAngle.TWO_SHOT],
    CameraAngle.MEDIUM_WIDE: [CameraAngle.MEDIUM, CameraAngle.CLOSE_UP, CameraAngle.OVER_SHOULDER],
    CameraAngle.MEDIUM: [CameraAngle.CLOSE_UP, CameraAngle.MEDIUM_CLOSE, CameraAngle.OVER_SHOULDER],
    CameraAngle.MEDIUM_CLOSE: [CameraAngle.CLOSE_UP, CameraAngle.EXTREME_CLOSE, CameraAngle.MEDIUM],
    CameraAngle.CLOSE_UP: [CameraAngle.MEDIUM, CameraAngle.EXTREME_CLOSE, CameraAngle.POV, CameraAngle.WIDE],
    CameraAngle.EXTREME_CLOSE: [CameraAngle.CLOSE_UP, CameraAngle.MEDIUM],  # Pull back
    CameraAngle.OVER_SHOULDER: [CameraAngle.CLOSE_UP, CameraAngle.OVER_SHOULDER, CameraAngle.TWO_SHOT],
    CameraAngle.POV: [CameraAngle.CLOSE_UP, CameraAngle.MEDIUM],
    CameraAngle.TWO_SHOT: [CameraAngle.OVER_SHOULDER, CameraAngle.CLOSE_UP, CameraAngle.MEDIUM],
}

# Emotion to intensity mapping
EMOTION_INTENSITY = {
    Emotion.NEUTRAL: 0.3,
    Emotion.HAPPY: 0.5,
    Emotion.SAD: 0.6,
    Emotion.ANGRY: 0.9,
    Emotion.EXCITED: 0.7,
    Emotion.THOUGHTFUL: 0.4,
    Emotion.SURPRISED: 0.8,
    Emotion.SCARED: 0.9,
    Emotion.SARCASTIC: 0.5,
    Emotion.WHISPER: 0.7,
}

# For multi-shot prompts, typical ending angles based on starting angle
# Multi-shot usually progresses: Wide -> Medium -> Close-up
MULTI_SHOT_ENDING_ANGLES = {
    CameraAngle.EXTREME_WIDE: CameraAngle.MEDIUM,      # Wide → Medium
    CameraAngle.WIDE: CameraAngle.CLOSE_UP,            # Wide → Medium → Close-up
    CameraAngle.MEDIUM_WIDE: CameraAngle.CLOSE_UP,     # Medium → Close-up
    CameraAngle.MEDIUM: CameraAngle.CLOSE_UP,          # Medium → Close-up
    CameraAngle.MEDIUM_CLOSE: CameraAngle.EXTREME_CLOSE,
    CameraAngle.CLOSE_UP: CameraAngle.EXTREME_CLOSE,   # Close-up → Extreme close
    CameraAngle.EXTREME_CLOSE: CameraAngle.CLOSE_UP,   # Pull back slightly
    CameraAngle.OVER_SHOULDER: CameraAngle.CLOSE_UP,   # Conversation → reaction
    CameraAngle.POV: CameraAngle.CLOSE_UP,
    CameraAngle.TWO_SHOT: CameraAngle.OVER_SHOULDER,
}

# Scene type to preferred camera angles mapping
SCENE_TYPE_CAMERA_PREFERENCES = {
    SceneType.ESTABLISHING: [CameraAngle.EXTREME_WIDE, CameraAngle.WIDE],
    SceneType.DIALOGUE: [CameraAngle.MEDIUM, CameraAngle.OVER_SHOULDER, CameraAngle.TWO_SHOT],
    SceneType.ACTION: [CameraAngle.WIDE, CameraAngle.MEDIUM_WIDE, CameraAngle.POV],
    SceneType.MONTAGE: [CameraAngle.MEDIUM, CameraAngle.CLOSE_UP, CameraAngle.EXTREME_CLOSE],
    SceneType.CLIMAX: [CameraAngle.CLOSE_UP, CameraAngle.EXTREME_CLOSE],
    SceneType.TRANSITION: [CameraAngle.WIDE, CameraAngle.MEDIUM_WIDE],
    SceneType.REACTION: [CameraAngle.CLOSE_UP, CameraAngle.EXTREME_CLOSE],
}


def detect_location_change(current_setting: str, previous_setting: Optional[str]) -> bool:
    """Detect if the location/setting has meaningfully changed between scenes.

    Uses heuristics to detect location changes:
    - Completely different text = change
    - Same key words (office, home, street) = same location
    - INT/EXT changes = different location

    Args:
        current_setting: The current scene's setting description
        previous_setting: The previous scene's setting description (None if first scene)

    Returns:
        True if location has changed, False otherwise
    """
    if previous_setting is None:
        return True  # First scene is always a "new" location

    current = current_setting.lower().strip()
    previous = previous_setting.lower().strip()

    if current == previous:
        return False

    # Extract location keywords
    location_words = {
        'office', 'home', 'house', 'street', 'park', 'car', 'restaurant',
        'kitchen', 'bedroom', 'living', 'room', 'bathroom', 'hallway',
        'outside', 'interior', 'exterior', 'int', 'ext', 'beach', 'forest',
        'city', 'apartment', 'building', 'school', 'hospital', 'store',
        'bar', 'club', 'church', 'library', 'studio', 'warehouse', 'alley',
    }

    def extract_location(text: str) -> set:
        words = set(text.split())
        return words.intersection(location_words)

    current_loc = extract_location(current)
    previous_loc = extract_location(previous)

    # If no overlap in location words, it's a change
    if current_loc and previous_loc and not current_loc.intersection(previous_loc):
        return True

    # Check for INT/EXT changes (screenplay format)
    if ('int' in current and 'ext' in previous) or ('ext' in current and 'int' in previous):
        return True

    # Check for explicit location indicators
    if 'interior' in current and 'exterior' in previous:
        return True
    if 'exterior' in current and 'interior' in previous:
        return True

    return False


def detect_character_introductions(
    current_visible: list[str],
    seen_characters: set[str]
) -> tuple[bool, list[str]]:
    """Detect if any characters in the current scene haven't appeared before.

    Args:
        current_visible: Character IDs visible in current scene
        seen_characters: Set of character IDs that have appeared in previous scenes

    Returns:
        Tuple of (is_introduction, list_of_new_character_ids)
    """
    new_chars = [c for c in current_visible if c not in seen_characters]
    return (len(new_chars) > 0, new_chars)


def get_natural_next_camera(
    previous_ending_camera: Optional[CameraAngle],
    narrative_role: str,
    emotional_intensity: float,
    dialogue_density: float,
    num_characters: int = 1,
) -> CameraAngle:
    """Determine the natural next camera angle based on context.

    Args:
        previous_ending_camera: The ENDING camera angle of the previous scene
                                (for multi-shot, this is the final angle, not starting)
        narrative_role: "opening", "buildup", "climax", "resolution", "standard"
        emotional_intensity: 0.0-1.0 scale
        dialogue_density: 0.0-1.0 scale (words per second ratio)
        num_characters: Number of visible characters in scene

    Returns:
        Recommended camera angle for this scene's START
    """
    # Opening scenes should start wide to establish location
    if narrative_role == "opening":
        return CameraAngle.WIDE

    # Climax scenes - push in for intensity
    if narrative_role == "climax":
        if emotional_intensity > 0.7:
            return CameraAngle.EXTREME_CLOSE
        return CameraAngle.CLOSE_UP

    # Resolution - pull back, let it breathe
    if narrative_role == "resolution":
        return CameraAngle.MEDIUM_WIDE

    # High emotion - push closer
    if emotional_intensity > 0.8:
        return CameraAngle.CLOSE_UP

    # Dialogue-heavy with 2+ characters - conversation coverage
    if dialogue_density > 0.7 and num_characters >= 2:
        # Alternate between over-shoulder shots for conversation
        if previous_ending_camera == CameraAngle.OVER_SHOULDER:
            return CameraAngle.CLOSE_UP  # Cut to reaction
        return CameraAngle.OVER_SHOULDER

    # If previous scene ended on close-up, pull back for variety
    if previous_ending_camera in (CameraAngle.CLOSE_UP, CameraAngle.EXTREME_CLOSE):
        return CameraAngle.MEDIUM  # Give breathing room

    # If previous scene ended wide, move in
    if previous_ending_camera in (CameraAngle.WIDE, CameraAngle.EXTREME_WIDE):
        return CameraAngle.MEDIUM

    # Follow natural transition rules
    if previous_ending_camera and previous_ending_camera in CAMERA_TRANSITION_RULES:
        options = CAMERA_TRANSITION_RULES[previous_ending_camera]
        if options:
            return options[0]

    # Default to medium shot
    return CameraAngle.MEDIUM


def get_multi_shot_ending_camera(starting_camera: CameraAngle) -> CameraAngle:
    """Get the expected ending camera for a multi-shot sequence.

    Multi-shot prompts typically progress: Wide → Medium → Close-up
    This helps the next scene know where the previous one ENDED.
    """
    return MULTI_SHOT_ENDING_ANGLES.get(starting_camera, CameraAngle.CLOSE_UP)


def calculate_scene_context(
    scene: MovieScene,
    scene_index: int,
    total_scenes: int,
    previous_context: Optional[SceneCameraContext] = None,
    is_multi_shot: bool = True,
    previous_scene: Optional[MovieScene] = None,
    seen_characters: Optional[set[str]] = None,
) -> SceneCameraContext:
    """Calculate camera context for a scene based on its narrative position.

    Args:
        scene: The MovieScene to analyze
        scene_index: Index of this scene (0-based)
        total_scenes: Total number of scenes in the movie
        previous_context: Camera context from the previous scene
        is_multi_shot: Whether video generation uses multi-shot mode
        previous_scene: The previous MovieScene (for location change detection)
        seen_characters: Set of character IDs that have appeared in previous scenes

    Returns:
        SceneCameraContext with camera angle and context information
    """
    # Get scene type from direction (default to DIALOGUE)
    scene_type = getattr(scene.direction, 'scene_type', SceneType.DIALOGUE)
    if scene_type is None:
        scene_type = SceneType.DIALOGUE

    # Detect location change
    is_location_change = False
    if previous_scene:
        previous_setting = previous_scene.direction.setting if previous_scene.direction else None
        current_setting = scene.direction.setting if scene.direction else ""
        is_location_change = detect_location_change(current_setting, previous_setting)
    elif scene_index == 0:
        # First scene is always a "new" location
        is_location_change = True

    # Detect character introductions
    is_character_introduction = False
    new_characters: list[str] = []
    if seen_characters is not None:
        visible = scene.direction.visible_characters if scene.direction else []
        is_character_introduction, new_characters = detect_character_introductions(visible, seen_characters)

    # Check if establishing shot is suggested/overridden
    suggest_establishing = getattr(scene.direction, 'suggest_establishing', False)
    establishing_override = getattr(scene.direction, 'establishing_override', None)

    # Determine narrative role based on position
    if scene_index == 0:
        narrative_role = "opening"
    elif scene_index == total_scenes - 1:
        narrative_role = "resolution"
    elif total_scenes > 5:
        # For longer movies, identify climax around 2/3 mark
        climax_zone = total_scenes * 0.6
        if abs(scene_index - climax_zone) < 2:
            narrative_role = "climax"
        elif scene_index < climax_zone:
            narrative_role = "buildup"
        else:
            narrative_role = "resolution"
    else:
        # Short movie - middle is climax
        if scene_index == total_scenes // 2:
            narrative_role = "climax"
        else:
            narrative_role = "standard"

    # Calculate dialogue density (words per second)
    words = sum(len(d.text.split()) for d in scene.dialogue)
    duration = scene.duration if scene.duration > 0 else 5.0  # Default 5s if not calculated
    dialogue_density = min(1.0, words / (duration * 3))  # ~3 words/sec = max density

    # Calculate emotional intensity from dialogue emotions
    if scene.dialogue:
        intensities = []
        for d in scene.dialogue:
            intensity = EMOTION_INTENSITY.get(d.emotion, 0.5)
            intensities.append(intensity)
        emotional_intensity = sum(intensities) / len(intensities)
    else:
        # No dialogue - use mood-based intensity
        mood_lower = scene.direction.mood.lower() if scene.direction.mood else ""
        if "intense" in mood_lower or "tense" in mood_lower or "angry" in mood_lower:
            emotional_intensity = 0.8
        elif "sad" in mood_lower or "melancholic" in mood_lower:
            emotional_intensity = 0.6
        elif "calm" in mood_lower or "peaceful" in mood_lower:
            emotional_intensity = 0.3
        else:
            emotional_intensity = 0.5

    # Get previous scene's ENDING camera (for multi-shot, the final angle)
    previous_ending_camera = None
    if previous_context:
        if is_multi_shot:
            # Multi-shot scenes end at a different angle than they started
            previous_ending_camera = get_multi_shot_ending_camera(previous_context.camera_angle)
        else:
            previous_ending_camera = previous_context.camera_angle

    # Number of visible characters
    num_characters = len(scene.direction.visible_characters)

    # Determine camera angle based on multiple factors
    camera_angle: CameraAngle

    # Priority 1: User override for establishing shot
    if establishing_override is True:
        # User explicitly wants establishing shot
        camera_angle = CameraAngle.WIDE
    elif establishing_override is False:
        # User explicitly doesn't want establishing shot - use normal logic
        camera_angle = get_natural_next_camera(
            previous_ending_camera=previous_ending_camera,
            narrative_role=narrative_role,
            emotional_intensity=emotional_intensity,
            dialogue_density=dialogue_density,
            num_characters=num_characters,
        )
    # Priority 2: Scene type is ESTABLISHING
    elif scene_type == SceneType.ESTABLISHING:
        preferred = SCENE_TYPE_CAMERA_PREFERENCES.get(scene_type, [CameraAngle.WIDE])
        camera_angle = preferred[0] if preferred else CameraAngle.WIDE
    # Priority 3: Location change with suggested establishing shot
    elif is_location_change and suggest_establishing:
        camera_angle = CameraAngle.WIDE
    # Priority 4: Scene type preferences (if not opening/climax/resolution which have special logic)
    elif scene_type != SceneType.DIALOGUE and narrative_role == "standard":
        preferred = SCENE_TYPE_CAMERA_PREFERENCES.get(scene_type, [])
        if preferred:
            # Pick from preferred angles, considering previous camera
            if previous_ending_camera and previous_ending_camera in preferred:
                # Avoid same angle, pick next in preference list
                idx = preferred.index(previous_ending_camera)
                camera_angle = preferred[(idx + 1) % len(preferred)]
            else:
                camera_angle = preferred[0]
        else:
            camera_angle = get_natural_next_camera(
                previous_ending_camera=previous_ending_camera,
                narrative_role=narrative_role,
                emotional_intensity=emotional_intensity,
                dialogue_density=dialogue_density,
                num_characters=num_characters,
            )
    # Priority 5: Default natural progression
    else:
        camera_angle = get_natural_next_camera(
            previous_ending_camera=previous_ending_camera,
            narrative_role=narrative_role,
            emotional_intensity=emotional_intensity,
            dialogue_density=dialogue_density,
            num_characters=num_characters,
        )

    # Determine camera movement suggestion
    if scene_type == SceneType.ESTABLISHING or is_location_change:
        movement = "slow establishing dolly or crane down"
    elif scene_type == SceneType.ACTION:
        movement = "dynamic tracking or handheld for energy"
    elif scene_type == SceneType.CLIMAX or narrative_role == "climax":
        movement = "dynamic push-in or handheld for intensity"
    elif scene_type == SceneType.REACTION:
        movement = "subtle push-in for emotional emphasis"
    elif narrative_role == "opening":
        movement = "slow establishing dolly or crane down"
    elif narrative_role == "resolution":
        movement = "gentle pull-back or static"
    elif emotional_intensity > 0.7:
        movement = "subtle push-in for emphasis"
    elif dialogue_density > 0.7:
        movement = "stable framing for dialogue clarity"
    else:
        movement = "subtle natural movement"

    # Build transition description
    transition_from_previous = None
    if previous_context:
        prev_end = get_multi_shot_ending_camera(previous_context.camera_angle) if is_multi_shot else previous_context.camera_angle
        if is_location_change:
            transition_from_previous = "cut to new location establishing shot"
        elif is_character_introduction:
            transition_from_previous = "cut to introduce new character(s)"
        elif prev_end in (CameraAngle.CLOSE_UP, CameraAngle.EXTREME_CLOSE) and camera_angle in (CameraAngle.WIDE, CameraAngle.MEDIUM_WIDE):
            transition_from_previous = "pull back to re-establish after intimate moment"
        elif prev_end in (CameraAngle.WIDE, CameraAngle.EXTREME_WIDE) and camera_angle in (CameraAngle.CLOSE_UP, CameraAngle.EXTREME_CLOSE):
            transition_from_previous = "cut to close-up for emotional focus"
        elif prev_end == camera_angle:
            transition_from_previous = "match cut at same angle"
        else:
            transition_from_previous = "standard cut"

    return SceneCameraContext(
        scene_index=scene_index,
        camera_angle=camera_angle,
        movement=movement,
        transition_from_previous=transition_from_previous,
        narrative_role=narrative_role,
        emotional_intensity=emotional_intensity,
        dialogue_density=dialogue_density,
        scene_type=scene_type,
        is_location_change=is_location_change,
        is_character_introduction=is_character_introduction,
        new_characters=new_characters,
    )


def generate_cinematography_plan(
    scenes: list[MovieScene],
    style: str = "classical",
    is_multi_shot: bool = True,
) -> CinematographyPlan:
    """Generate a complete cinematography plan for all scenes.

    Args:
        scenes: List of MovieScene objects
        style: Cinematography style ("classical", "dynamic", "documentary")
        is_multi_shot: Whether video generation uses multi-shot mode

    Returns:
        CinematographyPlan with camera context for each scene
    """
    from datetime import datetime

    plan_scenes = []
    previous_context = None
    previous_scene = None
    seen_characters: set[str] = set()

    for i, scene in enumerate(scenes):
        context = calculate_scene_context(
            scene=scene,
            scene_index=i,
            total_scenes=len(scenes),
            previous_context=previous_context,
            is_multi_shot=is_multi_shot,
            previous_scene=previous_scene,
            seen_characters=seen_characters,
        )
        plan_scenes.append(context)
        previous_context = context
        previous_scene = scene

        # Update seen characters with this scene's visible characters
        if scene.direction and scene.direction.visible_characters:
            seen_characters.update(scene.direction.visible_characters)

    # Determine pacing based on average scene duration
    if scenes:
        avg_duration = sum(s.duration for s in scenes) / len(scenes)
        if avg_duration < 4:
            pacing = "fast"
        elif avg_duration > 8:
            pacing = "slow"
        else:
            pacing = "moderate"
    else:
        pacing = "moderate"

    logger.info(f"Generated cinematography plan for {len(plan_scenes)} scenes ({style} style, {pacing} pacing)")

    return CinematographyPlan(
        scenes=plan_scenes,
        style=style,
        pacing=pacing,
        generated_at=datetime.now().isoformat(),
    )


def build_camera_continuity_context(
    current_scene: MovieScene,
    previous_scene: Optional[MovieScene],
    next_scene: Optional[MovieScene],
    camera_context: SceneCameraContext,
    previous_camera_context: Optional[SceneCameraContext] = None,
    camera_override: Optional[str] = None,
    is_multi_shot: bool = True,
) -> str:
    """Build camera continuity instructions for prompt generation.

    Args:
        current_scene: The scene being generated
        previous_scene: The previous scene (for continuity)
        next_scene: The next scene (for transition awareness)
        camera_context: Planned camera context for current scene
        previous_camera_context: Camera context from previous scene
        camera_override: User's manual camera override (if any)
        is_multi_shot: Whether using multi-shot generation

    Returns:
        String with camera continuity instructions for the prompt
    """
    parts = []

    # Use override if provided, otherwise use planned camera
    effective_camera = camera_override if camera_override else camera_context.camera_angle.value

    # Previous scene context
    if previous_scene and previous_camera_context:
        prev_start = previous_camera_context.camera_angle.value.replace("_", " ")
        prev_end = get_multi_shot_ending_camera(previous_camera_context.camera_angle).value.replace("_", " ") if is_multi_shot else prev_start

        parts.append(f"""=== CAMERA CONTINUITY FROM PREVIOUS SCENE ===
Previous scene camera: Started at {prev_start}, ended at {prev_end}
Previous scene mood: {previous_scene.direction.mood}
Transition: {camera_context.transition_from_previous or 'standard cut'}
NOTE: This scene should flow naturally from where the previous camera ended.""")

    # Current scene camera plan
    camera_display = effective_camera.replace("_", " ").title()
    intensity_label = "high" if camera_context.emotional_intensity > 0.7 else "medium" if camera_context.emotional_intensity > 0.4 else "low"
    density_label = "heavy dialogue" if camera_context.dialogue_density > 0.7 else "moderate dialogue" if camera_context.dialogue_density > 0.4 else "light/no dialogue"

    parts.append(f"""
=== PLANNED CAMERA FOR THIS SCENE ===
Starting camera: {camera_display}
{"(User override)" if camera_override else "(From cinematography plan)"}
Narrative role: {camera_context.narrative_role}
Emotional intensity: {intensity_label}
Scene type: {density_label}
Movement suggestion: {camera_context.movement}""")

    # Multi-shot progression guidance
    if is_multi_shot:
        ending_camera = get_multi_shot_ending_camera(
            CameraAngle(effective_camera) if effective_camera in [e.value for e in CameraAngle] else camera_context.camera_angle
        ).value.replace("_", " ").title()
        parts.append(f"""
MULTI-SHOT PROGRESSION:
Start: {camera_display}
Progress through: varied angles for cinematic feel
End at: {ending_camera} (for smooth transition to next scene)""")

    # Next scene preview (for transition awareness)
    if next_scene:
        parts.append(f"""
=== PREPARING FOR NEXT SCENE ===
Next scene setting: {next_scene.direction.setting}
Next scene mood: {next_scene.direction.mood}
Consider ending this scene in a way that allows smooth transition.""")

    return "\n".join(parts)


def get_transition_for_scene(
    current_context: SceneCameraContext,
    next_context: Optional[SceneCameraContext] = None,
    base_duration: float = 0.5,
) -> tuple[Optional[str], float]:
    """Determine the best transition type and duration for a scene.

    Context-aware transition selection based on:
    - Emotional intensity: High emotion scenes use faster crossfades
    - Narrative role: Scene breaks (climax/resolution) may use fade to black
    - Camera angle changes: Large jumps may use dissolve for smoothness

    Args:
        current_context: Camera context for the scene ending
        next_context: Camera context for the next scene (if any)
        base_duration: Base transition duration (default 0.5s)

    Returns:
        Tuple of (transition_type, duration) where transition_type is:
        - None: Hard cut (no transition)
        - "xfade": Crossfade (smooth blend)
        - "fade": Fade to black (scene break)
        - "dissolve": Dissolve effect (dreamy/soft transition)
    """
    # No next scene - end of movie, fade to black
    if next_context is None:
        return ("fade", base_duration * 2)  # Longer fade at end

    # Calculate emotional intensity difference
    intensity_change = abs(next_context.emotional_intensity - current_context.emotional_intensity)

    # High emotion to calm = fade to black (scene break feel)
    if current_context.emotional_intensity > 0.7 and next_context.emotional_intensity < 0.4:
        return ("fade", base_duration * 1.5)  # Slightly longer fade

    # Climax scenes - hard cut for impact
    if current_context.narrative_role == "climax":
        if next_context.narrative_role == "resolution":
            return ("fade", base_duration * 2)  # Climax to resolution = dramatic fade
        return (None, 0)  # Hard cuts during climax for intensity

    # Opening scenes - use dissolve for soft entrance
    if next_context.narrative_role == "opening" or current_context.narrative_role == "opening":
        return ("dissolve", base_duration * 1.5)

    # Resolution - gentle fade
    if current_context.narrative_role == "resolution":
        return ("fade", base_duration)

    # Large camera angle changes - dissolve for smoothness
    camera_jump = _calculate_camera_distance(current_context.camera_angle, next_context.camera_angle)
    if camera_jump > 2:  # More than 2 steps in camera distance
        return ("dissolve", base_duration)

    # Default - use crossfade for smooth transition
    # Duration varies by intensity: higher intensity = faster transition
    adjusted_duration = base_duration * (1.5 - current_context.emotional_intensity * 0.5)
    return ("xfade", max(0.3, adjusted_duration))


def _calculate_camera_distance(from_camera: CameraAngle, to_camera: CameraAngle) -> int:
    """Calculate the 'distance' between two camera angles.

    Distance represents how jarring the cut would be:
    - 0: Same angle
    - 1: Adjacent angles (natural progression)
    - 2+: Larger jumps

    Returns:
        Integer distance (0-5 scale)
    """
    # Define camera "closeness" scale (closer to subject = higher number)
    CAMERA_SCALE = {
        CameraAngle.EXTREME_WIDE: 0,
        CameraAngle.WIDE: 1,
        CameraAngle.MEDIUM_WIDE: 2,
        CameraAngle.TWO_SHOT: 2.5,
        CameraAngle.MEDIUM: 3,
        CameraAngle.OVER_SHOULDER: 3.5,
        CameraAngle.MEDIUM_CLOSE: 4,
        CameraAngle.CLOSE_UP: 5,
        CameraAngle.EXTREME_CLOSE: 6,
        CameraAngle.POV: 4,  # POV is usually close
    }

    from_val = CAMERA_SCALE.get(from_camera, 3)
    to_val = CAMERA_SCALE.get(to_camera, 3)

    return int(abs(from_val - to_val))


def generate_transition_plan(
    cinematography_plan: CinematographyPlan,
    base_duration: float = 0.5,
) -> list[tuple[Optional[str], float]]:
    """Generate a complete transition plan based on cinematography plan.

    Args:
        cinematography_plan: The cinematography plan for all scenes
        base_duration: Base transition duration

    Returns:
        List of (transition_type, duration) tuples for each scene transition
    """
    transitions = []
    scenes = cinematography_plan.scenes

    for i, current_ctx in enumerate(scenes):
        next_ctx = scenes[i + 1] if i < len(scenes) - 1 else None
        transition = get_transition_for_scene(current_ctx, next_ctx, base_duration)
        transitions.append(transition)

    return transitions


def format_camera_for_model(
    camera_context: SceneCameraContext,
    model: str,
    camera_override: Optional[str] = None,
    is_multi_shot: bool = True,
) -> str:
    """Format camera instructions for specific video generation models.

    Args:
        camera_context: The camera context for this scene
        model: The video generation model ("wan26", "seedance", "veo3")
        camera_override: User's manual camera override (if any)
        is_multi_shot: Whether using multi-shot generation

    Returns:
        Model-specific camera instructions for the prompt
    """
    effective_camera = camera_override if camera_override else camera_context.camera_angle.value
    camera_name = effective_camera.replace("_", " ").title()
    ending_camera = get_multi_shot_ending_camera(
        CameraAngle(effective_camera) if effective_camera in [e.value for e in CameraAngle] else camera_context.camera_angle
    ).value.replace("_", " ").title()

    model_lower = model.lower()

    if "wan" in model_lower:
        # WAN uses shot breakdown format
        if is_multi_shot:
            return f"""Shot breakdown for this scene:
{camera_name}: Main establishing action
Medium shot: Character interaction and dialogue
{ending_camera}: Emotional resolution and exit"""
        else:
            return f"""{camera_name}: Full scene action with {camera_context.movement}"""

    elif "seedance" in model_lower:
        # Seedance uses camera work descriptions
        if is_multi_shot:
            return f"""Camera work: Start with {camera_name}, {camera_context.movement}.
Progress through varied angles for cinematic feel.
End on {ending_camera} for smooth scene transition."""
        else:
            return f"""Camera: {camera_name}. {camera_context.movement}.
Maintain stable framing throughout."""

    elif "veo" in model_lower:
        # Veo uses narrative camera descriptions
        if is_multi_shot:
            return f"""Cinematic multi-shot sequence: Open on {camera_name},
transition through medium shots for dialogue,
end on {ending_camera} for emotional impact.
{camera_context.movement}."""
        else:
            return f"""Cinematic {camera_name}. {camera_context.movement}.
Professional cinematography throughout."""

    return f"Camera: {camera_name}"
