from __future__ import annotations

from pathlib import Path

from video_intelligence.application.use_cases import AnalyzeYoutubeVideoUseCase
from video_intelligence.domain.models import FaceCrop, FaceGroup, Scene, TimeRange


class StubFaceAnalyzer:
    def __init__(self) -> None:
        self.grouped_face_batches: list[list[FaceCrop]] = []

    def group_faces(self, faces: list[FaceCrop]) -> list[FaceGroup]:
        self.grouped_face_batches.append(faces)
        if not faces:
            return []

        scene_index = faces[0].scene_index
        return [
            FaceGroup(
                group_id="person_001",
                faces=faces,
                scene_emotions={scene_index: ["happy"]},
            )
        ]


def face(scene_index: int, timestamp: float) -> FaceCrop:
    return FaceCrop(
        path=Path(f"scene_{scene_index}/face_{timestamp}.jpg"),
        scene_index=scene_index,
        timestamp=timestamp,
        embedding=[timestamp],
        emotion="happy",
    )


def test_group_faces_by_scene_keeps_grouping_scene_scoped() -> None:
    face_analyzer = StubFaceAnalyzer()
    use_case = AnalyzeYoutubeVideoUseCase(
        video_provider=None,
        shot_segmenter=None,
        scene_segmenter=None,
        frame_sampler=None,
        description_generator=None,
        face_analyzer=face_analyzer,
        mood_analyzer=None,
        object_detector=None,
        music_detector=None,
        repository=None,
    )
    scene_one_faces = [face(1, 0.0), face(1, 1.0)]
    scene_two_faces = [face(2, 5.0)]

    face_groups = use_case._group_faces_by_scene(
        [
            Scene(index=1, timerange=TimeRange(0, 2), faces=scene_one_faces),
            Scene(index=2, timerange=TimeRange(5, 6), faces=scene_two_faces),
        ]
    )

    assert face_analyzer.grouped_face_batches == [scene_one_faces, scene_two_faces]
    assert [group.group_id for group in face_groups] == [
        "scene_001_person_001",
        "scene_002_person_001",
    ]
    assert [face.scene_index for face in face_groups[0].faces] == [1, 1]
    assert [face.scene_index for face in face_groups[1].faces] == [2]
