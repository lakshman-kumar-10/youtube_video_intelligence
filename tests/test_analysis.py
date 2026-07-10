from __future__ import annotations

import json
from pathlib import Path

from video_intelligence.domain.models import FaceCrop, FaceGroup, MusicSegment, Scene, Shot, TimeRange, VideoAnalysis, VideoAsset

def time_range_from_dict(data: dict) -> TimeRange:
    return TimeRange(
        start=data["start"],
        end=data["end"],
    )


def video_asset_from_dict(data: dict) -> VideoAsset:
    return VideoAsset(
        source_url=data["source_url"],
        path=Path(data["path"]),
        title=data.get("title"),
        duration=data.get("duration"),
    )


def shot_from_dict(data: dict) -> Shot:
    return Shot(
        index=data["index"],
        timerange=time_range_from_dict(data["timerange"]),
    )


def face_crop_from_dict(data: dict) -> FaceCrop:
    return FaceCrop(
        path=Path(data["path"]),
        scene_index=data["scene_index"],
        timestamp=data["timestamp"],
        embedding=data["embedding"],
        emotion=data.get("emotion"),
        confidence=data.get("confidence"),
    )


def face_group_from_dict(data: dict) -> FaceGroup:
    return FaceGroup(
        group_id=data["group_id"],
        faces=[face_crop_from_dict(face) for face in data["faces"]],
        scene_emotions={
            int(k): v
            for k, v in data["scene_emotions"].items()
        },
    )


def scene_from_dict(data: dict) -> Scene:
    return Scene(
        index=data["index"],
        timerange=time_range_from_dict(data["timerange"]),
        description=data.get("description"),
        mood=data.get("mood", []),
        objects=data.get("objects", []),
        faces=[
            face_crop_from_dict(face)
            for face in data.get("faces", [])
        ],
    )


def music_segment_from_dict(data: dict) -> MusicSegment:
    return MusicSegment(
        timerange=time_range_from_dict(data["timerange"]),
        confidence=data["confidence"],
    )


def video_analysis_from_dict(data: dict) -> VideoAnalysis:
    return VideoAnalysis(
        video=video_asset_from_dict(data["video"]),
        shots=[shot_from_dict(s) for s in data["shots"]],
        scenes=[scene_from_dict(s) for s in data["scenes"]],
        face_groups=[
            face_group_from_dict(g)
            for g in data["face_groups"]
        ],
        background_music=[
            music_segment_from_dict(m)
            for m in data["background_music"]
        ],
    )


def load_video_analysis(json_file: str | Path) -> VideoAnalysis:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return video_analysis_from_dict(data)


def get_analysis() -> VideoAnalysis:
    TEST_DATA_DIR = Path(__file__).parent / "data"
    json_file = TEST_DATA_DIR / "analysis.json"
    analysis = load_video_analysis(json_file)
    
    return analysis