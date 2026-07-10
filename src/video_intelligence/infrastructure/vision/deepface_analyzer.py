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
            extracted = deepface.extract_faces(
                img_path=frame,
                detector_backend=self._settings.detector_backend,
                enforce_detection=False,
            )
        except Exception as e:
            print(repr(e))
            return crops

        for item in extracted:
            face_image = item.get("face")
            confidence = item.get("confidence")
            if face_image is None:
                continue
            if face_image.dtype != np.uint8:
                face_image = np.clip(face_image * 255, 0, 255).astype(np.uint8)
            face_path = scene_dir / f"face_{timestamp:.2f}_{uuid4().hex[:8]}.jpg"
            cv2.imwrite(str(face_path), cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
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

