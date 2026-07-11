from __future__ import annotations

import base64
import os
from pathlib import Path

from video_intelligence.domain.models import Scene
from video_intelligence.domain.settings import LlmSettings


class SceneDescriptor:
    def __init__(self, settings: LlmSettings) -> None:
        self._settings = settings

    def describe(self, scene: Scene, frame_paths: list[Path], objects: list[str], mood: list[str]) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install openai for LLM scene descriptions.") from exc

        client = OpenAI(
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url="https://integrate.api.nvidia.com/v1",
        )
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": (
                    "Describe this video scene in fewer than 100 words. "
                    "Use the frames, object labels, and mood words. "
                    f"Scene time: {scene.timerange.start:.2f}-{scene.timerange.end:.2f}s. "
                    f"Objects: {', '.join(objects) or 'unknown'}. "
                    f"Mood: {', '.join(mood) or 'unknown'}."
                ),
            }
        ]
        for frame_path in frame_paths[:3]:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self._encode(frame_path)}"
                    }
                }
            )

        response = client.chat.completions.create(
            model=self._settings.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=140,
        )
        return self._trim_to_100_words(response.choices[0].message.content.strip())

    @staticmethod
    def _encode(path: Path) -> str:
        return base64.b64encode(path.read_bytes()).decode("ascii")

    @staticmethod
    def _trim_to_100_words(text: str) -> str:
        words = text.split()
        if len(words) <= 100:
            return text
        return " ".join(words[:100])
