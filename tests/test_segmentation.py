from video_intelligence.domain.models import Shot, TimeRange
from video_intelligence.domain.settings import SegmentationSettings
from video_intelligence.infrastructure.video.scenedetect_segmenter import (
    AudioAwareSceneDetectSegmenter,
)


def test_semantic_scene_grouping_merges_similar_quiet_shots() -> None:
    segmenter = AudioAwareSceneDetectSegmenter(SegmentationSettings(min_scene_seconds=1.0))
    shots = [
        Shot(index=1, timerange=TimeRange(0.0, 2.0)),
        Shot(index=2, timerange=TimeRange(2.0, 4.0)),
        Shot(index=3, timerange=TimeRange(4.0, 6.0)),
    ]

    ranges = segmenter._group_shots_by_evidence(
        shots,
        similarities=[0.9, 0.8],
        audio_scores=[0.1, 0.2],
    )

    assert ranges == [TimeRange(0.0, 6.0)]


def test_semantic_scene_grouping_splits_on_visual_or_audio_boundaries() -> None:
    segmenter = AudioAwareSceneDetectSegmenter(SegmentationSettings(min_scene_seconds=1.0))
    shots = [
        Shot(index=1, timerange=TimeRange(0.0, 2.0)),
        Shot(index=2, timerange=TimeRange(2.0, 4.0)),
        Shot(index=3, timerange=TimeRange(4.0, 6.0)),
        Shot(index=4, timerange=TimeRange(6.0, 8.0)),
    ]

    ranges = segmenter._group_shots_by_evidence(
        shots,
        similarities=[0.9, 0.2, 0.9],
        audio_scores=[0.1, 0.1, 0.8],
    )

    assert ranges == [
        TimeRange(0.0, 4.0),
        TimeRange(4.0, 6.0),
        TimeRange(6.0, 8.0),
    ]
