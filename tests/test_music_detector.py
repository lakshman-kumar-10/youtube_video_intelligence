from __future__ import annotations

from pathlib import Path

from video_intelligence.domain.models import Scene, TimeRange, VideoAsset
from video_intelligence.domain.settings import AudioSettings
from video_intelligence.infrastructure.audio.music_detector import MusicDetector
from video_intelligence.infrastructure.audio.models import AudioClip, AudioEvent


class StubClipLoader:
    def load(self, video: VideoAsset, scene: Scene, sample_rate: int) -> AudioClip:
        return AudioClip(samples=[scene.index], sample_rate=sample_rate)


class StubEventDetector:
    def __init__(self, events_by_scene_index: dict[int, list[AudioEvent]]) -> None:
        self._events_by_scene_index = events_by_scene_index

    def detect(self, clip: AudioClip) -> list[AudioEvent]:
        return self._events_by_scene_index.get(clip.samples[0], [])


def test_yamnet_music_detector_merges_adjacent_music_scenes() -> None:
    detector = MusicDetector(
        AudioSettings(music_min_duration_seconds=1.0, yamnet_confidence_threshold=0.2),
        clip_loader=StubClipLoader(),
        event_detector=StubEventDetector(
            {
                1: [AudioEvent(label="Music", confidence=0.82)],
                2: [AudioEvent(label="Rock music", confidence=0.74)],
            }
        ),
    )

    segments = detector.detect_background_music(
        VideoAsset(source_url="https://example.com", path=Path("source.mp4")),
        [
            Scene(index=1, timerange=TimeRange(0.0, 4.0)),
            Scene(index=2, timerange=TimeRange(4.0, 8.0)),
        ],
    )

    assert len(segments) == 1
    assert segments[0].timerange == TimeRange(0.0, 8.0)
    assert segments[0].confidence == 0.82


def test_yamnet_music_detector_ignores_non_music_events() -> None:
    detector = MusicDetector(
        AudioSettings(music_min_duration_seconds=1.0, yamnet_confidence_threshold=0.2),
        clip_loader=StubClipLoader(),
        event_detector=StubEventDetector(
            {
                1: [AudioEvent(label="Speech", confidence=0.95)],
                2: [AudioEvent(label="Vehicle", confidence=0.88)],
            }
        ),
    )

    segments = detector.detect_background_music(
        VideoAsset(source_url="https://example.com", path=Path("source.mp4")),
        [
            Scene(index=1, timerange=TimeRange(0.0, 4.0)),
            Scene(index=2, timerange=TimeRange(4.0, 8.0)),
        ],
    )

    assert segments == []
