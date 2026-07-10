from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TimeRange:
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class VideoAsset:
    source_url: str
    path: Path
    title: str | None = None
    duration: float | None = None


@dataclass(frozen=True)
class Shot:
    index: int
    timerange: TimeRange


@dataclass(frozen=True)
class FaceCrop:
    path: Path
    scene_index: int
    timestamp: float
    embedding: list[float]
    emotion: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class FaceGroup:
    group_id: str
    faces: list[FaceCrop]
    scene_emotions: dict[int, list[str]]


@dataclass(frozen=True)
class Scene:
    index: int
    timerange: TimeRange
    description: str | None = None
    mood: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    faces: list[FaceCrop] = field(default_factory=list)


@dataclass(frozen=True)
class MusicSegment:
    timerange: TimeRange
    confidence: float


@dataclass(frozen=True)
class VideoAnalysis:
    video: VideoAsset
    shots: list[Shot]
    scenes: list[Scene]
    face_groups: list[FaceGroup]
    background_music: list[MusicSegment]

