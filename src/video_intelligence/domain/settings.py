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
    shot_detector: str = "adaptive"
    scene_detector: str = "semantic"
    shot_threshold: float = 18.0
    shot_adaptive_threshold: float = 3.0
    shot_min_content_value: float = 15.0
    scene_threshold: float = 30.0
    min_scene_seconds: float = 2.0
    audio_boundary_window_seconds: float = 0.75
    scene_visual_similarity_threshold: float = 0.55
    scene_audio_boundary_threshold: float = 0.65


@dataclass(frozen=True)
class FaceSettings:
    detector_backend: str = "opencv"
    emotion_backend: str = "opencv"
    yolo_model_name: str = "yolov8n-face.pt"
    yolo_confidence: float = 0.55
    crop_padding_ratio: float = 0.20
    min_crop_size_pixels: int = 64
    similarity_threshold: float = 0.72
    frame_sample_rate_seconds: float = 1.0


@dataclass(frozen=True)
class ObjectSettings:
    model_name: str = "yolov8n.pt"
    confidence: float = 0.35
    frame_sample_rate_seconds: float = 1.0


@dataclass(frozen=True)
class LlmSettings:
    model: str = "qwen/qwen3.5-397b-a17b"


@dataclass(frozen=True)
class AudioSettings:
    music_min_duration_seconds: float = 3.0
    music_energy_percentile: int = 65
    mood_sample_rate: int = 22_050
    mood_minimum_score: float = 0.25
    mood_max_labels: int = 2
    yamnet_enabled: bool = True
    yamnet_sample_rate: int = 16_000
    yamnet_model_url: str = "https://tfhub.dev/google/yamnet/1"
    yamnet_confidence_threshold: float = 0.20
    yamnet_max_events: int = 8


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
