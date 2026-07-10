from __future__ import annotations

from pathlib import Path
from typing import Protocol

from video_intelligence.domain.models import (
    FaceCrop,
    FaceGroup,
    MusicSegment,
    Scene,
    Shot,
    TimeRange,
    VideoAnalysis,
    VideoAsset,
)


class VideoProvider(Protocol):
    def fetch(self, url: str, output_dir: Path) -> VideoAsset:
        """Download or resolve a video into a local asset."""


class ShotSegmenter(Protocol):
    def detect_shots(self, video: VideoAsset) -> list[Shot]:
        """Return audio-aware shot boundaries."""


class SceneSegmenter(Protocol):
    def detect_scenes(self, video: VideoAsset, shots: list[Shot]) -> list[Scene]:
        """Return audio-aware scene boundaries."""


class SceneFrameSampler(Protocol):
    def sample_frames(self, video_path: Path, timerange: TimeRange, every_seconds: float) -> list[Path]:
        """Persist representative frames and return frame paths."""


class SceneDescriptionGenerator(Protocol):
    def describe(self, scene: Scene, frame_paths: list[Path], objects: list[str], mood: list[str]) -> str:
        """Generate a concise scene description."""


class FaceAnalyzer(Protocol):
    def extract_faces(self, video: VideoAsset, scenes: list[Scene], output_dir: Path) -> list[Scene]:
        """Extract face crops and emotions scene-wise."""

    def group_faces(self, faces: list[FaceCrop]) -> list[FaceGroup]:
        """Group similar faces using embeddings."""


class MoodAnalyzer(Protocol):
    def detect_mood(self, video: VideoAsset, scene: Scene) -> list[str]:
        """Return scene mood/sentiment as words."""


class ObjectDetector(Protocol):
    def detect_objects(self, video: VideoAsset, scene: Scene) -> list[str]:
        """Return object labels from a scene."""


class MusicDetector(Protocol):
    def detect_background_music(self, video: VideoAsset, scenes: list[Scene]) -> list[MusicSegment]:
        """Detect music ranges across scenes."""


class AnalysisRepository(Protocol):
    def save(self, analysis: VideoAnalysis, output_dir: Path) -> Path:
        """Persist analysis output and return report path."""

