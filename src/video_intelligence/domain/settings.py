from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class VideoSettings:
    max_resolution: int = 720
    working_filename: str = "source.mp4"


@dataclass(frozen=True)
class SegmentationSettings:
    shot_threshold: float = 18.0
    scene_threshold: float = 30.0
    min_scene_seconds: float = 2.0
    audio_boundary_window_seconds: float = 0.75


@dataclass(frozen=True)
class FaceSettings:
    detector_backend: str = "opencv"
    emotion_backend: str = "opencv"
    similarity_threshold: float = 0.72
    frame_sample_rate_seconds: float = 1.0


@dataclass(frozen=True)
class ObjectSettings:
    model_name: str = "yolov8n.pt"
    confidence: float = 0.35
    frame_sample_rate_seconds: float = 1.0


@dataclass(frozen=True)
class LlmSettings:
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct"


@dataclass(frozen=True)
class AudioSettings:
    music_min_duration_seconds: float = 3.0
    music_energy_percentile: int = 65


@dataclass(frozen=True)
class AppSettings:
    video: VideoSettings = VideoSettings()
    segmentation: SegmentationSettings = SegmentationSettings()
    faces: FaceSettings = FaceSettings()
    objects: ObjectSettings = ObjectSettings()
    llm: LlmSettings = LlmSettings()
    audio: AudioSettings = AudioSettings()

    @staticmethod
    def from_yaml(path: Path | None) -> "AppSettings":
        if path is None:
            return AppSettings()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return AppSettings(
            video=VideoSettings(**data.get("video", {})),
            segmentation=SegmentationSettings(**data.get("segmentation", {})),
            faces=FaceSettings(**data.get("faces", {})),
            objects=ObjectSettings(**data.get("objects", {})),
            llm=LlmSettings(**data.get("llm", {})),
            audio=AudioSettings(**data.get("audio", {})),
        )

