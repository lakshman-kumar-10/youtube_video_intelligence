from __future__ import annotations

from collections import Counter

import cv2

from video_intelligence.domain.models import Scene, VideoAsset
from video_intelligence.domain.settings import ObjectSettings


class YoloObjectDetector:
    def __init__(self, settings: ObjectSettings) -> None:
        self._settings = settings
        self._model = None

    def detect_objects(self, video: VideoAsset, scene: Scene) -> list[str]:
        model = self._load_model()
        capture = cv2.VideoCapture(str(video.path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        labels: Counter[str] = Counter()

        timestamp = scene.timerange.start
        while timestamp <= scene.timerange.end:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * fps))
            ok, frame = capture.read()
            if ok:
                result = model.predict(frame, conf=self._settings.confidence, verbose=False)[0]
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    labels[result.names[class_id]] += 1
            timestamp += self._settings.frame_sample_rate_seconds

        capture.release()
        return [label for label, _ in labels.most_common()]

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install ultralytics for object detection.") from exc
        self._model = YOLO(self._settings.model_name)
        return self._model

