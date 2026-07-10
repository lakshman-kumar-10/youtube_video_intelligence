from __future__ import annotations

from video_intelligence.domain.models import Scene, VideoAsset
from video_intelligence.domain.settings import AudioSettings

from .events import NullAudioEventDetector, YamnetAudioEventDetector
from .features import LibrosaAudioFeatureExtractor
from .loaders import LibrosaAudioClipLoader
from .mood import WeightedMoodScorer
from .models import AudioEvent, AudioFeatures
from .protocols import AudioClipLoader, AudioEventDetector, AudioFeatureExtractor, MoodScorer


class HybridAudioMoodAnalyzer:
    def __init__(
        self,
        settings: AudioSettings,
        clip_loader: AudioClipLoader | None = None,
        feature_extractor: AudioFeatureExtractor | None = None,
        event_detector: AudioEventDetector | None = None,
        mood_scorer: MoodScorer | None = None,
    ) -> None:
        self._settings = settings
        self._clip_loader = clip_loader or LibrosaAudioClipLoader()
        self._feature_extractor = feature_extractor or LibrosaAudioFeatureExtractor()
        self._event_detector = event_detector or self._build_event_detector(settings)
        self._mood_scorer = mood_scorer or WeightedMoodScorer(
            minimum_score=settings.mood_minimum_score,
            max_labels=settings.mood_max_labels,
        )

    def detect_mood(self, video: VideoAsset, scene: Scene) -> list[str]:
        try:
            feature_clip = self._clip_loader.load(video, scene, self._settings.mood_sample_rate)
            if feature_clip.is_empty:
                return ["calm"]

            features = self._feature_extractor.extract(feature_clip)
        except RuntimeError:
            return ["unknown"]

        events: list[AudioEvent] = []
        try:
            event_clip = feature_clip
            if self._settings.yamnet_enabled:
                event_clip = self._clip_loader.load(video, scene, self._settings.yamnet_sample_rate)
            events = self._event_detector.detect(event_clip)
        except RuntimeError:
            events = []

        return [score.label for score in self._mood_scorer.score(features, events)]

    @staticmethod
    def _build_event_detector(settings: AudioSettings) -> AudioEventDetector:
        if not settings.yamnet_enabled:
            return NullAudioEventDetector()
        return YamnetAudioEventDetector(
            model_url=settings.yamnet_model_url,
            confidence_threshold=settings.yamnet_confidence_threshold,
            max_events=settings.yamnet_max_events,
        )
