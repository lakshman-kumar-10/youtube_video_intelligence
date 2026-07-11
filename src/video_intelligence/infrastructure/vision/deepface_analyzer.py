from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

from video_intelligence.domain.models import FaceCrop, FaceGroup, Scene, VideoAsset
from video_intelligence.domain.settings import FaceSettings


class DeepFaceAnalyzer:
    def __init__(self, settings: FaceSettings) -> None:
        self._settings = settings
        self._face_model = None

    def extract_faces(self, video: VideoAsset, scenes: list[Scene], output_dir: Path) -> list[Scene]:
        try:
            from deepface import DeepFace
        except ImportError as exc:
            raise RuntimeError("Install deepface for face extraction, embeddings, and emotion analysis.") from exc

        output_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video.path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        enriched: list[Scene] = []

        for scene in scenes:
            scene_dir = output_dir / f"scene_{scene.index:03d}"
            scene_dir.mkdir(parents=True, exist_ok=True)
            faces = self._extract_scene_faces(DeepFace, capture, fps, scene, scene_dir)
            enriched.append(
                Scene(
                    index=scene.index,
                    timerange=scene.timerange,
                    description=scene.description,
                    mood=scene.mood,
                    objects=scene.objects,
                    faces=faces,
                )
            )
        capture.release()
        return enriched

    def group_faces(self, faces: list[FaceCrop]) -> list[FaceGroup]:
        if not faces:
            return []

        embeddings = np.array([face.embedding for face in faces], dtype=np.float32)
        distance_matrix = cosine_distances(embeddings)
        clustering = DBSCAN(
            eps=1.0 - self._settings.similarity_threshold,
            min_samples=1,
            metric="precomputed",
        ).fit(distance_matrix)

        grouped: dict[int, list[FaceCrop]] = defaultdict(list)
        for face, label in zip(faces, clustering.labels_, strict=False):
            grouped[int(label)].append(face)

        groups: list[FaceGroup] = []
        for label, grouped_faces in sorted(grouped.items()):
            scene_emotions: dict[int, list[str]] = defaultdict(list)
            for face in grouped_faces:
                if face.emotion:
                    scene_emotions[face.scene_index].append(face.emotion)
            groups.append(
                FaceGroup(
                    group_id=f"person_{label + 1:03d}",
                    faces=grouped_faces,
                    scene_emotions={key: sorted(set(value)) for key, value in scene_emotions.items()},
                )
            )
        return groups

    def _extract_scene_faces(
        self,
        deepface,
        capture: cv2.VideoCapture,
        fps: float,
        scene: Scene,
        scene_dir: Path,
    ) -> list[FaceCrop]:
        faces: list[FaceCrop] = []
        timestamp = scene.timerange.start
        while timestamp <= scene.timerange.end:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * fps))
            ok, frame = capture.read()
            if ok:
                faces.extend(self._extract_faces_from_frame(deepface, frame, scene, timestamp, scene_dir))
            timestamp += self._settings.frame_sample_rate_seconds
        return faces

    def _extract_faces_from_frame(
        self,
        deepface,
        frame: np.ndarray,
        scene: Scene,
        timestamp: float,
        scene_dir: Path,
    ) -> list[FaceCrop]:
        crops: list[FaceCrop] = []
        try:
            face_boxes = self._detect_face_boxes(frame)
        except Exception as e:
            print(repr(e))
            return crops

        for x1, y1, x2, y2, confidence in face_boxes:
            face_image = frame[y1:y2, x1:x2]
            if face_image is None:
                continue
            if face_image.size == 0:
                continue
            face_path = scene_dir / f"face_{timestamp:.2f}_{uuid4().hex[:8]}.jpg"
            cv2.imwrite(str(face_path), face_image)
            embedding = self._embedding(deepface, face_path)
            if not embedding:
                continue
            emotion = self._emotion(deepface, face_path)
            crops.append(
                FaceCrop(
                    path=face_path,
                    scene_index=scene.index,
                    timestamp=timestamp,
                    embedding=embedding,
                    emotion=emotion,
                    confidence=float(confidence) if confidence is not None else None,
                )
            )
        return crops

    def _detect_face_boxes(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        model = self._load_face_model()
        result = model.predict(frame, conf=self._settings.yolo_confidence, verbose=False)[0]
        height, width = frame.shape[:2]
        boxes: list[tuple[int, int, int, int, float]] = []

        for box in result.boxes:
            confidence = self._box_confidence(box)
            if confidence < self._settings.yolo_confidence:
                continue
            x1, y1, x2, y2 = self._box_xyxy(box)
            crop_box = self._expand_box(x1, y1, x2, y2, width, height)
            if crop_box is None:
                continue
            boxes.append((*crop_box, confidence))
        return boxes

    def _load_face_model(self):
        if self._face_model is not None:
            return self._face_model
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install ultralytics for YOLO face detection.") from exc
        self._face_model = YOLO(self._settings.yolo_model_name)
        return self._face_model

    def _expand_box(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int] | None:
        box_width = x2 - x1
        box_height = y2 - y1
        min_size = self._settings.min_crop_size_pixels
        if box_width < min_size or box_height < min_size:
            return None

        padding = max(box_width, box_height) * self._settings.crop_padding_ratio
        left = max(0, int(round(x1 - padding)))
        top = max(0, int(round(y1 - padding)))
        right = min(frame_width, int(round(x2 + padding)))
        bottom = min(frame_height, int(round(y2 + padding)))
        if right - left < min_size or bottom - top < min_size:
            return None
        return left, top, right, bottom

    @staticmethod
    def _box_xyxy(box) -> tuple[float, float, float, float]:
        values = DeepFaceAnalyzer._as_flat_array(box.xyxy)
        return float(values[0]), float(values[1]), float(values[2]), float(values[3])

    @staticmethod
    def _box_confidence(box) -> float:
        values = DeepFaceAnalyzer._as_flat_array(box.conf)
        return float(values[0])

    @staticmethod
    def _as_flat_array(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=np.float32).reshape(-1)

    @staticmethod
    def _embedding(deepface, face_path: Path) -> list[float]:
        try:
            representations = deepface.represent(
                img_path=str(face_path),
                model_name="Facenet",
                enforce_detection=False,
            )
        except Exception:
            return []
        if not representations:
            return []
        return [float(value) for value in representations[0]["embedding"]]

    def _emotion(self, deepface, face_path: Path) -> str | None:
        try:
            result = deepface.analyze(
                img_path=str(face_path),
                actions=["emotion"],
                detector_backend=self._settings.emotion_backend,
                enforce_detection=False,
            )
        except Exception:
            return None
        analysis = result[0] if isinstance(result, list) else result
        return str(analysis.get("dominant_emotion")) if analysis.get("dominant_emotion") else None
