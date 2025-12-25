# Atlas Cloud API - Video Generation Documentation

## Quick Reference for Claude Code Project

This document provides Atlas Cloud-specific API schemas, Python code examples, and model configurations for **Seedance 1.5** and **Wan 2.6** video generation models.

---

## Atlas Cloud API Overview

- **Base URL**: `https://api.atlascloud.ai/api/v1`
- **OpenAI-compatible**: Yes (for LLM endpoints)
- **Authentication**: Bearer token via `Authorization` header
- **Video Generation**: Asynchronous (create task → poll for result)

---

## Authentication

```python
import os

ATLASCLOUD_API_KEY = os.environ.get("ATLASCLOUD_API_KEY")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {ATLASCLOUD_API_KEY}"
}
```

Get your API key from: https://console.atlascloud.ai/settings

---

## Video Generation Workflow

Atlas Cloud video generation is **asynchronous**:

1. **POST** `/api/v1/model/generateVideo` → Returns `prediction_id`
2. **GET** `/api/v1/model/prediction/{prediction_id}` → Poll until `status` is `completed` or `succeeded`

### Response Schema

```json
{
  "id": "7hm7nbwarxrm80cnq1yvmxz4m4",
  "urls": {
    "cancel": "http://...",
    "result": "http://..."
  },
  "model": "string",
  "status": "processing|completed|succeeded|failed",
  "outputs": ["https://...video-url.mp4"],
  "created_at": "2025-12-24T14:15:22Z",
  "has_nsfw_contents": [false]
}
```

---

## Wan 2.6 Models on Atlas Cloud

### Available Models

| Model ID | Type | Resolution | Price |
|----------|------|------------|-------|
| `alibaba/wan-2.6/t2v-720p` | Text-to-Video | 720p | ~$0.05/s |
| `alibaba/wan-2.6/t2v-1080p` | Text-to-Video | 1080p | ~$0.08/s |
| `alibaba/wan-2.6/i2v-720p` | Image-to-Video | 720p | ~$0.05/s |
| `alibaba/wan-2.6/i2v-1080p` | Image-to-Video | 1080p | ~$0.08/s |
| `alibaba/wan-2.6/i2v-720p-fast` | Image-to-Video (Fast) | 720p | ~$0.03/s |
| `alibaba/wan-2.6/v2v-720p-fast` | Video-to-Video | 720p | ~$0.05/s |

### Wan 2.6 Key Features
- **Duration**: Up to 15 seconds
- **Multi-shot storytelling**: Automatic storyboard generation
- **Audio sync**: Native audio-visual synchronization
- **Character consistency**: Clone-level preservation

### Text-to-Video (Wan 2.6)

```python
import requests
import time

ATLASCLOUD_API_KEY = "your-api-key"
BASE_URL = "https://api.atlascloud.ai/api/v1"

def wan26_text_to_video(
    prompt: str,
    resolution: str = "720p",  # "720p" or "1080p"
    duration: int = 10,
    seed: int = -1,
    enable_audio: bool = True
) -> str:
    """Generate video from text prompt using Wan 2.6"""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ATLASCLOUD_API_KEY}"
    }
    
    # Select model based on resolution
    model = f"alibaba/wan-2.6/t2v-{resolution}"
    
    data = {
        "model": model,
        "prompt": prompt,
        "duration": duration,
        "seed": seed,
        "audio": enable_audio  # Enable native audio generation
    }
    
    # Step 1: Create generation task
    response = requests.post(
        f"{BASE_URL}/model/generateVideo",
        headers=headers,
        json=data
    )
    result = response.json()
    prediction_id = result["data"]["id"]
    
    # Step 2: Poll for completion
    while True:
        poll_response = requests.get(
            f"{BASE_URL}/model/prediction/{prediction_id}",
            headers={"Authorization": f"Bearer {ATLASCLOUD_API_KEY}"}
        )
        poll_result = poll_response.json()
        status = poll_result["data"]["status"]
        
        if status in ["completed", "succeeded"]:
            return poll_result["data"]["outputs"][0]
        elif status == "failed":
            raise Exception(f"Generation failed: {poll_result}")
        
        time.sleep(5)  # Poll every 5 seconds

# Usage
video_url = wan26_text_to_video(
    prompt="""
    A singer performs emotionally on stage.
    Shot 1 [0-5s]: Close-up of face, dramatic lighting, singing passionately.
    Shot 2 [5-10s]: Wide shot revealing full stage, audience in background.
    Shot 3 [10-15s]: Medium shot, camera slowly dollies in as song reaches climax.
    """,
    resolution="1080p",
    duration=15
)
print(f"Generated video: {video_url}")
```

### Image-to-Video (Wan 2.6)

```python
def wan26_image_to_video(
    image_url: str,
    prompt: str,
    resolution: str = "720p",
    duration: int = 10,
    seed: int = -1,
    enable_audio: bool = True
) -> str:
    """Animate an image using Wan 2.6"""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ATLASCLOUD_API_KEY}"
    }
    
    model = f"alibaba/wan-2.6/i2v-{resolution}"
    
    data = {
        "model": model,
        "image": image_url,
        "prompt": prompt,
        "duration": duration,
        "seed": seed,
        "audio": enable_audio
    }
    
    response = requests.post(
        f"{BASE_URL}/model/generateVideo",
        headers=headers,
        json=data
    )
    result = response.json()
    prediction_id = result["data"]["id"]
    
    # Poll for completion
    return poll_for_result(prediction_id)

# Usage for music video storyboards
video_url = wan26_image_to_video(
    image_url="https://your-nano-banana-storyboard.png",
    prompt="The scene comes to life with gentle camera movement, soft lighting transitions",
    resolution="1080p",
    duration=10
)
```

---

## Seedance 1.5 Models on Atlas Cloud

### Available Models

| Model ID | Type | Resolution | Price |
|----------|------|------------|-------|
| `bytedance/seedance-v1.5-pro-t2v-480p` | Text-to-Video | 480p | ~$0.02/s |
| `bytedance/seedance-v1.5-pro-t2v-720p` | Text-to-Video | 720p | ~$0.04/s |
| `bytedance/seedance-v1.5-pro-t2v-1080p` | Text-to-Video | 1080p | ~$0.06/s |
| `bytedance/seedance-v1.5-pro-i2v-480p` | Image-to-Video | 480p | ~$0.02/s |
| `bytedance/seedance-v1.5-pro-i2v-720p` | Image-to-Video | 720p | ~$0.04/s |
| `bytedance/seedance-v1.5-pro-i2v-1080p` | Image-to-Video | 1080p | ~$0.06/s |

### Seedance 1.5 Key Features
- **Lip-sync**: Millisecond-precision phoneme-level accuracy
- **Joint audio-video**: Generated simultaneously (not cascaded)
- **Camera controls**: Dolly zoom, pans, tracking shots
- **Languages**: EN, JA, KO, ES, PT, ID, ZH (+ dialects)

### Seedance 1.5 Pro - Text-to-Video

```python
def seedance15_text_to_video(
    prompt: str,
    resolution: str = "720p",  # "480p", "720p", "1080p"
    duration: int = 8,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    generate_audio: bool = True
) -> str:
    """Generate video with audio using Seedance 1.5 Pro"""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ATLASCLOUD_API_KEY}"
    }
    
    model = f"bytedance/seedance-v1.5-pro-t2v-{resolution}"
    
    data = {
        "model": model,
        "prompt": prompt,
        "width": width,
        "height": height,
        "duration": duration,
        "fps": fps,
        "generate_audio": generate_audio
    }
    
    response = requests.post(
        f"{BASE_URL}/model/generateVideo",
        headers=headers,
        json=data
    )
    result = response.json()
    prediction_id = result["data"]["id"]
    
    return poll_for_result(prediction_id)

# Usage - dialogue scene with lip-sync
video_url = seedance15_text_to_video(
    prompt="""
    A singer performs an emotional ballad on stage.
    Camera: Slow dolly in from medium shot to close-up.
    Lighting: Dramatic spotlight with soft ambient glow.
    Expression: Deep emotional intensity, eyes closed during climax.
    Audio: Powerful vocals with ambient reverb.
    """,
    resolution="1080p",
    duration=10,
    generate_audio=True
)
```

### Seedance 1.5 Pro - Image-to-Video

```python
def seedance15_image_to_video(
    image_url: str,
    prompt: str,
    resolution: str = "720p",
    duration: int = 8,
    fps: int = 24,
    fixed_lens: bool = False,
    generate_audio: bool = True
) -> str:
    """Animate image with Seedance 1.5 Pro (includes audio)"""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ATLASCLOUD_API_KEY}"
    }
    
    model = f"bytedance/seedance-v1.5-pro-i2v-{resolution}"
    
    data = {
        "model": model,
        "image": image_url,  # Can also be base64 or local file path
        "prompt": prompt,
        "duration": duration,
        "fps": fps,
        "fixed_lens": fixed_lens,  # True = static camera
        "generate_audio": generate_audio
    }
    
    response = requests.post(
        f"{BASE_URL}/model/generateVideo",
        headers=headers,
        json=data
    )
    result = response.json()
    prediction_id = result["data"]["id"]
    
    return poll_for_result(prediction_id)
```

---

## Wan 2.2 Models (Also Available)

For budget-conscious generation or faster iteration:

| Model ID | Type | Price |
|----------|------|-------|
| `alibaba/wan-2.2/t2v-480p-ultra-fast` | T2V | $0.0085/s |
| `alibaba/wan-2.2/t2v-5b-720p` | T2V | $0.0085/s |
| `alibaba/wan-2.2/i2v-480p-ultra-fast` | I2V | $0.0085/s |
| `alibaba/wan-2.2/i2v-5b-720p` | I2V | $0.0085/s |
| `alibaba/wan-2.2/i2v-720p-ultra-fast` | I2V | $0.0170/s |
| `alibaba/wan-2.2/animate` | Character Animation | $0.0387/s |

### Wan 2.2 Animate (Character Animation)

```python
def wan22_animate(
    image_url: str,
    video_url: str,
    mode: str = "animate",  # "animate" or "replace"
    prompt: str = "",
    resolution: str = "720p"
) -> str:
    """
    Animate a character from image using motion from video reference.
    
    Modes:
    - animate: Apply movements from video to character in image
    - replace: Replace character in video with character from image
    """
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ATLASCLOUD_API_KEY}"
    }
    
    data = {
        "model": "alibaba/wan-2.2/animate",
        "image": image_url,
        "video": video_url,
        "mode": mode,
        "prompt": prompt,
        "resolution": resolution
    }
    
    response = requests.post(
        f"{BASE_URL}/model/generateVideo",
        headers=headers,
        json=data
    )
    result = response.json()
    prediction_id = result["data"]["id"]
    
    return poll_for_result(prediction_id)
```

---

## Complete AtlasCloudVideoClient Class

```python
"""
Atlas Cloud Video Generation Client
For Suno Music Video Generator Claude Code Project
"""

import requests
import time
import os
from typing import Optional, Literal
from dataclasses import dataclass

@dataclass
class VideoResult:
    url: str
    prediction_id: str
    status: str
    has_nsfw: bool

class AtlasCloudVideoClient:
    """Unified client for Atlas Cloud video generation APIs"""
    
    BASE_URL = "https://api.atlascloud.ai/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ATLASCLOUD_API_KEY")
        if not self.api_key:
            raise ValueError("ATLASCLOUD_API_KEY required")
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _poll_for_result(
        self,
        prediction_id: str,
        poll_interval: int = 5,
        max_wait: int = 300
    ) -> VideoResult:
        """Poll until video generation completes"""
        elapsed = 0
        
        while elapsed < max_wait:
            response = requests.get(
                f"{self.BASE_URL}/model/prediction/{prediction_id}",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            result = response.json()
            data = result.get("data", result)
            status = data.get("status", "")
            
            if status in ["completed", "succeeded"]:
                return VideoResult(
                    url=data["outputs"][0],
                    prediction_id=prediction_id,
                    status=status,
                    has_nsfw=data.get("has_nsfw_contents", [False])[0]
                )
            elif status == "failed":
                raise Exception(f"Generation failed: {result}")
            
            time.sleep(poll_interval)
            elapsed += poll_interval
        
        raise TimeoutError(f"Generation timed out after {max_wait}s")
    
    def _generate(self, data: dict) -> VideoResult:
        """Submit generation request and poll for result"""
        response = requests.post(
            f"{self.BASE_URL}/model/generateVideo",
            headers=self.headers,
            json=data
        )
        response.raise_for_status()
        result = response.json()
        
        # Handle different response structures
        if "data" in result:
            prediction_id = result["data"]["id"]
        else:
            prediction_id = result["id"]
        
        return self._poll_for_result(prediction_id)
    
    # ==================== WAN 2.6 ====================
    
    def wan26_t2v(
        self,
        prompt: str,
        resolution: Literal["720p", "1080p"] = "720p",
        duration: int = 10,
        seed: int = -1,
        audio: bool = True
    ) -> VideoResult:
        """Wan 2.6 Text-to-Video with multi-shot support"""
        return self._generate({
            "model": f"alibaba/wan-2.6/t2v-{resolution}",
            "prompt": prompt,
            "duration": duration,
            "seed": seed,
            "audio": audio
        })
    
    def wan26_i2v(
        self,
        image_url: str,
        prompt: str,
        resolution: Literal["720p", "1080p"] = "720p",
        duration: int = 10,
        seed: int = -1,
        audio: bool = True
    ) -> VideoResult:
        """Wan 2.6 Image-to-Video"""
        return self._generate({
            "model": f"alibaba/wan-2.6/i2v-{resolution}",
            "image": image_url,
            "prompt": prompt,
            "duration": duration,
            "seed": seed,
            "audio": audio
        })
    
    def wan26_i2v_fast(
        self,
        image_url: str,
        prompt: str,
        duration: int = 10
    ) -> VideoResult:
        """Wan 2.6 Image-to-Video (Fast/Budget)"""
        return self._generate({
            "model": "alibaba/wan-2.6/i2v-720p-fast",
            "image": image_url,
            "prompt": prompt,
            "duration": duration
        })
    
    # ==================== SEEDANCE 1.5 ====================
    
    def seedance15_t2v(
        self,
        prompt: str,
        resolution: Literal["480p", "720p", "1080p"] = "720p",
        duration: int = 8,
        width: int = 1280,
        height: int = 720,
        fps: int = 24,
        generate_audio: bool = True
    ) -> VideoResult:
        """Seedance 1.5 Pro Text-to-Video with joint audio"""
        return self._generate({
            "model": f"bytedance/seedance-v1.5-pro-t2v-{resolution}",
            "prompt": prompt,
            "width": width,
            "height": height,
            "duration": duration,
            "fps": fps,
            "generate_audio": generate_audio
        })
    
    def seedance15_i2v(
        self,
        image_url: str,
        prompt: str,
        resolution: Literal["480p", "720p", "1080p"] = "720p",
        duration: int = 8,
        fps: int = 24,
        fixed_lens: bool = False,
        generate_audio: bool = True
    ) -> VideoResult:
        """Seedance 1.5 Pro Image-to-Video with lip-sync"""
        return self._generate({
            "model": f"bytedance/seedance-v1.5-pro-i2v-{resolution}",
            "image": image_url,
            "prompt": prompt,
            "duration": duration,
            "fps": fps,
            "fixed_lens": fixed_lens,
            "generate_audio": generate_audio
        })
    
    # ==================== WAN 2.2 (BUDGET) ====================
    
    def wan22_t2v_fast(
        self,
        prompt: str,
        resolution: Literal["480p", "720p"] = "480p"
    ) -> VideoResult:
        """Wan 2.2 Ultra-Fast Text-to-Video (budget option)"""
        model = f"alibaba/wan-2.2/t2v-{resolution}-ultra-fast"
        return self._generate({
            "model": model,
            "prompt": prompt
        })
    
    def wan22_i2v_fast(
        self,
        image_url: str,
        prompt: str,
        resolution: Literal["480p", "720p"] = "480p"
    ) -> VideoResult:
        """Wan 2.2 Ultra-Fast Image-to-Video (budget option)"""
        model = f"alibaba/wan-2.2/i2v-{resolution}-ultra-fast"
        return self._generate({
            "model": model,
            "image": image_url,
            "prompt": prompt
        })
    
    def wan22_animate(
        self,
        image_url: str,
        video_url: str,
        mode: Literal["animate", "replace"] = "animate",
        prompt: str = "",
        resolution: str = "720p"
    ) -> VideoResult:
        """Wan 2.2 Character Animation"""
        return self._generate({
            "model": "alibaba/wan-2.2/animate",
            "image": image_url,
            "video": video_url,
            "mode": mode,
            "prompt": prompt,
            "resolution": resolution
        })


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    client = AtlasCloudVideoClient()
    
    # Example 1: Wan 2.6 multi-shot music video scene
    result = client.wan26_t2v(
        prompt="""
        A singer performs emotionally under dramatic stage lighting.
        Shot 1 [0-5s]: Close-up face, soft spotlight, eyes closed.
        Shot 2 [5-10s]: Wide shot, full stage visible, audience silhouettes.
        Shot 3 [10-15s]: Medium shot, camera slowly dollies in.
        """,
        resolution="1080p",
        duration=15,
        audio=True
    )
    print(f"Wan 2.6 video: {result.url}")
    
    # Example 2: Seedance 1.5 with precise lip-sync
    result = client.seedance15_i2v(
        image_url="https://your-character-image.png",
        prompt="""
        Character sings with deep emotion.
        Camera: Slow zoom from medium to close-up.
        Expression: Eyes closed, powerful vocals.
        """,
        resolution="1080p",
        duration=10,
        generate_audio=False  # Use external Suno audio
    )
    print(f"Seedance 1.5 video: {result.url}")
    
    # Example 3: Budget option for quick iteration
    result = client.wan22_i2v_fast(
        image_url="https://your-storyboard.png",
        prompt="Gentle pan across the scene with soft lighting",
        resolution="480p"
    )
    print(f"Quick preview: {result.url}")
```

---

## Environment Setup

Add to your `.env` file:

```bash
# Atlas Cloud API
ATLASCLOUD_API_KEY=your-api-key-here
```

Add to `requirements.txt`:

```txt
requests>=2.28.0
```

---

## Model Selection Guide

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| **Lip-sync scenes** | Seedance 1.5 Pro | Millisecond precision |
| **Multi-shot narrative** | Wan 2.6 | Auto storyboarding |
| **Quick previews** | Wan 2.2 Ultra-Fast | Lowest cost |
| **Character consistency** | Wan 2.6 | Clone-level preservation |
| **Cinematic camera** | Seedance 1.5 Pro | Hitchcock zoom support |
| **15-second clips** | Wan 2.6 | Longest duration |
| **Budget production** | Wan 2.2 | ~$0.0085/s |

---

## Pricing Summary (December 2025)

| Model | Resolution | Price per Second |
|-------|------------|------------------|
| Wan 2.2 Ultra-Fast | 480p | $0.0085 |
| Wan 2.2 Standard | 720p | $0.0170 |
| Wan 2.6 | 720p | ~$0.05 |
| Wan 2.6 | 1080p | ~$0.08 |
| Seedance 1.5 Pro | 720p | ~$0.04 |
| Seedance 1.5 Pro | 1080p | ~$0.06 |

---

## Integration with Suno Music Video Generator

```python
# Updated project structure addition
# src/services/atlas_video_generator.py

from .atlas_cloud_client import AtlasCloudVideoClient

class MusicVideoSceneGenerator:
    """Generate video scenes for music videos using Atlas Cloud"""
    
    def __init__(self):
        self.client = AtlasCloudVideoClient()
    
    async def generate_scene_from_storyboard(
        self,
        storyboard_image_url: str,
        lyrics_section: dict,
        visual_style: str,
        use_lip_sync: bool = False
    ) -> str:
        """
        Generate video from Nano Banana storyboard image.
        
        Args:
            storyboard_image_url: URL of generated storyboard image
            lyrics_section: {text, start_time, end_time, mood}
            visual_style: Style description from lyrics analysis
            use_lip_sync: Use Seedance 1.5 for singing scenes
        """
        duration = min(15, lyrics_section["end_time"] - lyrics_section["start_time"])
        
        prompt = f"""
        {visual_style}
        Mood: {lyrics_section.get("mood", "emotional")}
        Camera: Slow cinematic movement matching the music tempo.
        """
        
        if use_lip_sync and "singing" in lyrics_section.get("type", ""):
            # Use Seedance for lip-sync heavy scenes
            result = self.client.seedance15_i2v(
                image_url=storyboard_image_url,
                prompt=prompt,
                resolution="1080p",
                duration=int(duration),
                generate_audio=False  # Use original Suno audio
            )
        else:
            # Use Wan 2.6 for general scenes
            result = self.client.wan26_i2v(
                image_url=storyboard_image_url,
                prompt=prompt,
                resolution="1080p",
                duration=int(duration),
                audio=False
            )
        
        return result.url
```

---

*Documentation compiled: December 24, 2025*
*For: Suno Music Video Generator - Claude Code Project*
*Platform: Atlas Cloud (atlascloud.ai)*
