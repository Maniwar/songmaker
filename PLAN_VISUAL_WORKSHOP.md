# Implementation Plan: Visual Workshop & Animation Preview

## Overview

Two major enhancements:
1. **Animation Preview with Lyrics** - Preview animated scenes with karaoke lyrics overlay
2. **Visual Style Workshop** - Agentic conversation to collaboratively develop visual style and scene prompts

---

## Feature 1: Animation Preview with Lyrics

### Current State
- `_generate_scene_preview_with_lyrics()` in `_generate.py` creates a preview using Ken Burns effect on a static image
- Uses `VideoGenerator.create_scene_preview()` which takes image_path, audio_path, subtitle_path

### Changes Needed

**File: `src/ui/page_modules/_generate.py`**
1. Modify `_generate_scene_preview_with_lyrics()` to detect if scene has animation
2. If animated, use the existing animation video and overlay subtitles instead of creating Ken Burns preview
3. Create helper function `_add_lyrics_to_animation()` that overlays ASS subtitles on existing video

**File: `src/services/video_generator.py`**
1. Add new method `add_subtitles_to_video()` that takes an existing video and overlays subtitles
2. This is simpler than `create_scene_preview` - just overlays on existing video

### Implementation Steps
1. Add `add_subtitles_to_existing_video()` method to VideoGenerator
2. Modify `_generate_scene_preview_with_lyrics()` to check `scene.has_animation`
3. If animated: use animation video + overlay subtitles
4. If not animated: use existing Ken Burns preview generation

---

## Feature 2: Visual Style Workshop

### Current State
- After UPLOAD step, the app automatically generates scene prompts using `VisualAgent.plan_video()`
- User has limited control over visual style - only cinematography presets
- `VisualAgent.extract_visual_world()` runs automatically without user input

### Proposed New Workflow
```
CONCEPT → LYRICS → UPLOAD → VISUAL_WORKSHOP → GENERATE → COMPLETE
                              ↑ NEW STEP
```

### New Step: VISUAL_WORKSHOP
A conversational interface (like Concept Workshop) where:
1. AI analyzes the lyrics and song concept
2. Discusses visual world/setting with user
3. Proposes character descriptions
4. Suggests cinematography style
5. Creates scene-by-scene prompts collaboratively
6. User can refine each scene prompt before image generation

### Files to Create

**1. `src/agents/visual_style_agent.py`** - New agent for visual style development
```python
class VisualStyleAgent:
    """Agent for collaborative visual style and scene prompt development."""

    REQUIRED_SECTIONS = [
        "VISUAL WORLD",
        "CHARACTER DESCRIPTION",
        "CINEMATOGRAPHY STYLE",
        "SCENE PROMPTS"
    ]

    def __init__(self, config, concept, lyrics, transcript):
        # Initialize with song context

    def chat(self, user_message: str) -> str:
        # Back-and-forth conversation

    def is_ready_to_finalize(self) -> bool:
        # Check if all sections complete

    def extract_visual_plan(self) -> VisualPlan:
        # Extract structured visual plan with scene prompts
```

**2. `src/ui/page_modules/_visual.py`** - New page for visual workshop
- Chat interface similar to `_concept.py`
- Shows scene boundaries and lyrics for reference
- Sidebar with readiness status
- Scene prompt preview/editing

**3. `src/models/schemas.py`** - Updates
- Add `WorkflowStep.VISUAL` between UPLOAD and GENERATE
- Add `VisualPlan` schema to store visual style decisions
- Add state fields: `visual_messages`, `visual_plan`, `visual_approved`

### New Data Model: VisualPlan
```python
class VisualPlan(BaseModel):
    """Visual plan extracted from workshop conversation."""
    visual_world: str  # Consistent visual universe description
    character_description: str  # Main character(s) appearance
    cinematography_style: str  # Camera/lighting style
    color_palette: Optional[str]  # Color direction
    scene_prompts: list[ScenePrompt]  # Individual scene prompts

class ScenePrompt(BaseModel):
    """Visual prompt for a single scene."""
    index: int
    start_time: float
    end_time: float
    lyrics_segment: str  # The lyrics for this scene
    visual_prompt: str  # AI-generated or user-edited prompt
    mood: str
    effect: KenBurnsEffect
    user_notes: Optional[str]  # User additions/modifications
```

### Agent System Prompt (Draft)
```
You are a visual creative director for music videos. You're working with a user
to develop the visual style for their song's music video.

Context you have:
- Song concept: {genre, mood, themes}
- Complete lyrics with timestamps
- Transcript showing word-level timing

Your goal is to collaboratively develop:
1. A consistent VISUAL WORLD - the unified setting/universe for all scenes
2. CHARACTER DESCRIPTION - how the main character(s) should look
3. CINEMATOGRAPHY STYLE - lighting, camera work, color palette
4. SCENE-BY-SCENE PROMPTS - specific visuals for each scene

Start by:
1. Analyzing the lyrics and themes
2. Proposing 2-3 visual world options
3. Asking user preferences

After 3-4 exchanges, provide the complete visual plan with all scene prompts.
Each scene prompt should:
- Match the specific lyrics/words being sung
- Maintain visual world consistency
- Include mood and camera movement suggestions
```

### UI Layout for Visual Workshop Page
```
┌─────────────────────────────────────────────────────────────────────┐
│ Visual Style Workshop                                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [Chat conversation with AI creative director]                       │
│                                                                      │
│  User: I want a dark cyberpunk feel                                  │
│  AI: Great choice! For cyberpunk, I suggest...                      │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│ Scene Reference (collapsible)                                        │
│ ┌──────────┬──────────┬──────────┐                                  │
│ │Scene 1   │Scene 2   │Scene 3   │                                  │
│ │0-8s      │8-16s     │16-24s    │                                  │
│ │"In the..."│"We run..."│"Light..." │                                │
│ └──────────┴──────────┴──────────┘                                  │
├─────────────────────────────────────────────────────────────────────┤
│ [Chat input: "Type your visual ideas..."]                           │
└─────────────────────────────────────────────────────────────────────┘

SIDEBAR:
┌─────────────────┐
│ Visual Status   │
│ ✓ Visual World  │
│ ✓ Character     │
│ ⏳ Scene Prompts│
│                 │
│ [Finalize]      │
│                 │
│ ─────────────── │
│ Quick Actions   │
│ [Edit Scenes]   │
│ [Skip Workshop] │
└─────────────────┘
```

### Implementation Steps

1. **Update WorkflowStep enum** - Add VISUAL between UPLOAD and GENERATE

2. **Create VisualStyleAgent** - Similar pattern to ConceptAgent
   - Takes concept, lyrics, transcript as context
   - Conversational chat method
   - Readiness checking for required sections
   - Extraction of structured VisualPlan

3. **Create _visual.py page** - Visual workshop UI
   - Chat interface
   - Scene reference display (showing lyrics per scene)
   - Readiness sidebar
   - Edit individual scene prompts capability

4. **Update AppState** - Add visual workshop state fields

5. **Update app.py routing** - Add new page to workflow

6. **Update _generate.py** - Use VisualPlan for scene prompts instead of auto-generating

7. **Add skip option** - Allow users to skip workshop and use auto-generated prompts

---

## Implementation Order

### Phase 1: Animation Preview with Lyrics (simpler, faster)
1. Add `add_subtitles_to_existing_video()` to VideoGenerator
2. Modify preview function to handle animated scenes
3. Test with existing animations

### Phase 2: Visual Style Workshop (larger change)
1. Create schemas (VisualPlan, ScenePrompt)
2. Update WorkflowStep enum
3. Create VisualStyleAgent
4. Create _visual.py page
5. Update state management
6. Update app routing
7. Modify generate page to use VisualPlan
8. Add skip/bypass options

---

## Estimated Scope
- Phase 1: ~30 minutes
- Phase 2: ~2-3 hours

## Questions for User
1. Should the Visual Workshop be optional (can skip)?
2. Should users be able to edit individual scene prompts after AI generates them?
3. For animation preview - should we add the audio track as well, or just visual + subtitle?
