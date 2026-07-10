from __future__ import annotations

from video_intelligence.domain.models import MusicSegment, Scene, TimeRange, VideoAsset
from video_intelligence.domain.settings import AudioSettings


class LibrosaMusicDetector:
    def __init__(self, settings: AudioSettings) -> None:
        self._settings = settings

    def detect_background_music(self, video: VideoAsset, scenes: list[Scene]) -> list[MusicSegment]:
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

