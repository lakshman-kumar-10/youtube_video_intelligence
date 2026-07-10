from __future__ import annotations

from video_intelligence.domain.models import MusicSegment, Scene, TimeRange, VideoAsset
from video_intelligence.domain.settings import AudioSettings

from .events import NullAudioEventDetector, YamnetAudioEventDetector
from .loaders import LibrosaAudioClipLoader
from .models import AudioEvent
from .protocols import AudioClipLoader, AudioEventDetector


class MusicDetector:
    _MUSIC_LABEL_KEYWORDS = (
        "music",
        "song",
        "singing",
        "choir",
        "chant",
        "humming",
        "instrument",
        "guitar",
        "piano",
        "drum",
        "violin",
        "flute",
        "saxophone",
        "trumpet",
        "orchestra",
        "synthesizer",
        "hip hop",
        "rock",
        "pop music",
        "classical music",
        "electronic music",
        "jazz",
        "blues",
        "reggae",
        "country",
        "soundtrack",
        "background music",
        "theme music",
    )

    def __init__(
        self,
        settings: AudioSettings,
        clip_loader: AudioClipLoader | None = None,
        event_detector: AudioEventDetector | None = None,
    ) -> None:
        self._settings = settings
        self._clip_loader = clip_loader or LibrosaAudioClipLoader()
        self._event_detector = event_detector or self._build_event_detector(settings)

    def detect_background_music(self, video: VideoAsset, scenes: list[Scene]) -> list[MusicSegment]:
        if self._settings.yamnet_enabled and scenes:
            try:
                return self._detect_with_yamnet(video, scenes)
            except RuntimeError:
                pass

        return self._detect_with_librosa(video, scenes)

    def _detect_with_yamnet(self, video: VideoAsset, scenes: list[Scene]) -> list[MusicSegment]:
        segments: list[MusicSegment] = []
        for scene in scenes:
            if scene.timerange.duration < self._settings.music_min_duration_seconds:
                continue
            clip = self._clip_loader.load(video, scene, self._settings.yamnet_sample_rate)
            events = self._event_detector.detect(clip)
            confidence = self._music_confidence(events)
            if confidence is None:
                continue
            segments.append(MusicSegment(timerange=scene.timerange, confidence=confidence))

        return self._merge_adjacent_segments(segments)

    def _detect_with_librosa(self, video: VideoAsset, scenes: list[Scene]) -> list[MusicSegment]:
        try:
            import librosa
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Install librosa for background music detection.") from exc

        audio, sample_rate = librosa.load(str(video.path), sr=22_050, mono=True)
        if audio.size == 0:
            return []

        hop_length = 512
        harmonic, _ = librosa.effects.hpss(audio)
        harmonic_rms = librosa.feature.rms(y=harmonic, hop_length=hop_length)[0]
        threshold = float(np.percentile(harmonic_rms, self._settings.music_energy_percentile))
        times = librosa.frames_to_time(
            np.arange(harmonic_rms.size), sr=sample_rate, hop_length=hop_length
        )
        active = harmonic_rms >= threshold

        segments: list[MusicSegment] = []
        start: float | None = None
        values: list[float] = []
        for time, is_active, score in zip(times, active, harmonic_rms, strict=False):
            if is_active and start is None:
                start = float(time)
                values = []
            if is_active:
                values.append(float(score))
            if not is_active and start is not None:
                self._append_segment(segments, start, float(time), values, threshold)
                start = None
        if start is not None:
            self._append_segment(segments, start, float(times[-1]), values, threshold)

        return self._clip_to_scene_span(segments, scenes)

    def _music_confidence(self, events: list[AudioEvent]) -> float | None:
        music_events = [
            event
            for event in events
            if self._is_music_label(event.label)
            and event.confidence >= self._settings.yamnet_confidence_threshold
        ]
        if not music_events:
            return None
        return max(event.confidence for event in music_events)

    @classmethod
    def _is_music_label(cls, label: str) -> bool:
        normalized = label.casefold()
        return any(keyword in normalized for keyword in cls._MUSIC_LABEL_KEYWORDS)

    def _append_segment(
        self,
        segments: list[MusicSegment],
        start: float,
        end: float,
        values: list[float],
        threshold: float,
    ) -> None:
        if end - start < self._settings.music_min_duration_seconds:
            return
        confidence = min(1.0, (sum(values) / max(len(values), 1)) / max(threshold, 1e-6))
        segments.append(MusicSegment(timerange=TimeRange(start, end), confidence=float(confidence)))

    @staticmethod
    def _clip_to_scene_span(segments: list[MusicSegment], scenes: list[Scene]) -> list[MusicSegment]:
        if not scenes:
            return segments
        start = scenes[0].timerange.start
        end = scenes[-1].timerange.end
        return [
            MusicSegment(
                timerange=TimeRange(max(segment.timerange.start, start), min(segment.timerange.end, end)),
                confidence=segment.confidence,
            )
            for segment in segments
            if segment.timerange.end >= start and segment.timerange.start <= end
        ]

    @staticmethod
    def _merge_adjacent_segments(segments: list[MusicSegment]) -> list[MusicSegment]:
        if not segments:
            return []

        merged: list[MusicSegment] = [segments[0]]
        for segment in segments[1:]:
            previous = merged[-1]
            if segment.timerange.start <= previous.timerange.end + 0.25:
                merged[-1] = MusicSegment(
                    timerange=TimeRange(previous.timerange.start, segment.timerange.end),
                    confidence=max(previous.confidence, segment.confidence),
                )
                continue
            merged.append(segment)
        return merged

    @staticmethod
    def _build_event_detector(settings: AudioSettings) -> AudioEventDetector:
        if not settings.yamnet_enabled:
            return NullAudioEventDetector()
        return YamnetAudioEventDetector(
            model_url=settings.yamnet_model_url,
            confidence_threshold=settings.yamnet_confidence_threshold,
            max_events=settings.yamnet_max_events,
        )
