from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AudioClip:
    samples: Any
    sample_rate: int

    @property
    def is_empty(self) -> bool:
        return bool(getattr(self.samples, "size", 0) == 0)


@dataclass(frozen=True)
class AudioFeatures:
    energy: float
    brightness: float
    tempo_bpm: float
    onset_strength: float
    silence_ratio: float


@dataclass(frozen=True)
class AudioEvent:
    label: str
    confidence: float


@dataclass(frozen=True)
class MoodScore:
    label: str
    score: float
    evidence: list[str] = field(default_factory=list)