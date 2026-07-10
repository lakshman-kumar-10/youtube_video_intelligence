from __future__ import annotations

from typing import Protocol

from video_intelligence.domain.models import Scene, VideoAsset

from .models import AudioClip, AudioEvent, AudioFeatures, MoodScore


class AudioClipLoader(Protocol):
    def load(self, video: VideoAsset, scene: Scene, sample_rate: int) -> AudioClip:
        """Load one mono scene clip at the requested sample rate."""


class AudioFeatureExtractor(Protocol):
    def extract(self, clip: AudioClip) -> AudioFeatures:
        """Extract deterministic low-level audio features."""


class AudioEventDetector(Protocol):
    def detect(self, clip: AudioClip) -> list[AudioEvent]:
        """Detect semantic audio events such as music, speech, laughter, or sirens."""


class MoodScorer(Protocol):
    def score(self, features: AudioFeatures, events: list[AudioEvent]) -> list[MoodScore]:
        """Fuse low-level and semantic evidence into mood scores."""