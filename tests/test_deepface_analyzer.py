from __future__ import annotations

import numpy as np
import pytest

from video_intelligence.domain.models import Scene, TimeRange
from video_intelligence.domain.settings import FaceSettings
from video_intelligence.infrastructure.vision.deepface_analyzer import DeepFaceAnalyzer


class FakeBox:
    def __init__(self, xyxy: list[float], confidence: float) -> None:
        self.xyxy = np.array([xyxy], dtype=np.float32)
        self.conf = np.array([confidence], dtype=np.float32)


class FakeResult:
    def __init__(self, boxes: list[FakeBox]) -> None:
        self.boxes = boxes


class FakeYoloModel:
    def __init__(self, boxes: list[FakeBox]) -> None:
        self._boxes = boxes

    def predict(self, frame: np.ndarray, conf: float, verbose: bool) -> list[FakeResult]:
        return [FakeResult(self._boxes)]


class FakeDeepFace:
    @staticmethod
    def represent(img_path: str, model_name: str, enforce_detection: bool) -> list[dict]:
        return [{"embedding": [0.1, 0.2, 0.3]}]

    @staticmethod
    def analyze(
        img_path: str,
        actions: list[str],
        detector_backend: str,
        enforce_detection: bool,
    ) -> dict:
        return {"dominant_emotion": "neutral"}


def test_extract_faces_from_frame_uses_yolo_boxes_and_filters_tiny_crops(tmp_path) -> None:
    analyzer = DeepFaceAnalyzer(
        FaceSettings(
            yolo_confidence=0.5,
            crop_padding_ratio=0.0,
            min_crop_size_pixels=64,
        )
    )
    analyzer._face_model = FakeYoloModel(
        [
            FakeBox([10, 10, 120, 140], 0.9),
            FakeBox([150, 20, 180, 50], 0.95),
            FakeBox([200, 20, 300, 120], 0.2),
        ]
    )
    scene = Scene(index=1, timerange=TimeRange(start=0, end=1))
    frame = np.full((240, 320, 3), 127, dtype=np.uint8)

    faces = analyzer._extract_faces_from_frame(FakeDeepFace, frame, scene, 0.0, tmp_path)

    assert len(faces) == 1
    assert faces[0].confidence == pytest.approx(0.9)
    assert faces[0].embedding == [0.1, 0.2, 0.3]
    assert faces[0].emotion == "neutral"
    assert faces[0].path.exists()
