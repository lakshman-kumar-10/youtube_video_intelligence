from pathlib import Path

from video_intelligence.domain.models import (
    MusicSegment,
    Scene,
    Shot,
    TimeRange,
    VideoAnalysis,
    VideoAsset,
)
from video_intelligence.infrastructure.persistence.json_repository import JsonAnalysisRepository


def test_repository_writes_json(tmp_path: Path) -> None:
    analysis = VideoAnalysis(
        video=VideoAsset(source_url="https://example.com", path=tmp_path / "source.mp4"),
        shots=[Shot(index=1, timerange=TimeRange(0, 1))],
        scenes=[Scene(index=1, timerange=TimeRange(0, 1), mood=["calm"], objects=["person"])],
        face_groups=[],
        background_music=[MusicSegment(timerange=TimeRange(0, 1), confidence=0.9)],
    )

    path = JsonAnalysisRepository().save(analysis, tmp_path)

    assert path.exists()
    assert '"mood": [' in path.read_text(encoding="utf-8")

