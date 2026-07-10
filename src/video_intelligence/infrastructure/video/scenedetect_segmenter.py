from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from video_intelligence.domain.models import Scene, Shot, TimeRange, VideoAsset
from video_intelligence.domain.settings import SegmentationSettings
from video_intelligence.infrastructure.video.audio_boundaries import AudioBoundaryAligner


class AudioAwareSceneDetectSegmenter:
    def __init__(self, settings: SegmentationSettings) -> None:
        self._settings = settings
        self._aligner = AudioBoundaryAligner(settings.audio_boundary_window_seconds)

    def detect_shots(self, video: VideoAsset) -> list[Shot]:
        ranges = self._detect_ranges(
            video.path,
            self._settings.shot_detector,
            self._settings.shot_threshold,
        )
        return [
            Shot(index=index, timerange=timerange)
            for index, timerange in enumerate(ranges, start=1)
        ]

    def detect_scenes(self, video: VideoAsset, shots: list[Shot]) -> list[Scene]:
        if self._settings.scene_detector.lower() == "semantic" and shots:
            ranges = self._group_shots_into_scenes(video.path, shots)
        else:
            ranges = self._detect_ranges(
                video.path,
                self._settings.scene_detector,
                self._settings.scene_threshold,
            )
            ranges = [item for item in ranges if item.duration >= self._settings.min_scene_seconds]
        if not ranges:
            ranges = self._merge_shots_by_duration(shots) if shots else [
                TimeRange(start=0.0, end=self._video_duration(video.path))
            ]
        return [
            Scene(index=index, timerange=timerange)
            for index, timerange in enumerate(ranges, start=1)
        ]

    def _detect_ranges(self, video_path: Path, detector_name: str, threshold: float) -> list[TimeRange]:
        try:
            from scenedetect import SceneManager, open_video
        except ImportError as exc:
            raise RuntimeError("Install scenedetect[opencv] for shot and scene detection.") from exc

        video = open_video(str(video_path), backend="pyav")
        manager = SceneManager()
        manager.add_detector(self._build_detector(detector_name, threshold))
        manager.detect_scenes(video)
        raw_scenes = manager.get_scene_list()
        if not raw_scenes:
            return [TimeRange(start=0.0, end=self._video_duration(video_path))]

        starts = [scene[0].get_seconds() for scene in raw_scenes]
        ends = [scene[1].get_seconds() for scene in raw_scenes]
        interior_boundaries = [end for end in ends[:-1]]
        aligned = self._aligner.align(video_path, interior_boundaries)

        ranges: list[TimeRange] = []
        range_start = starts[0]
        for boundary in aligned:
            if boundary > range_start:
                ranges.append(TimeRange(start=range_start, end=boundary))
                range_start = boundary
        final_end = ends[-1]
        if final_end >= range_start:
            ranges.append(TimeRange(start=range_start, end=final_end))
        return ranges

    def _build_detector(self, detector_name: str, threshold: float) -> Any:
        detector = detector_name.lower()
        if detector == "adaptive":
            try:
                from scenedetect import AdaptiveDetector
            except ImportError:
                from scenedetect.detectors import AdaptiveDetector

            return AdaptiveDetector(
                adaptive_threshold=self._settings.shot_adaptive_threshold,
                min_content_val=self._settings.shot_min_content_value,
            )
        else:
            try:
                from scenedetect import ContentDetector
            except ImportError:
                from scenedetect.detectors import ContentDetector

            return ContentDetector(threshold=threshold)

    def _group_shots_into_scenes(self, video_path: Path, shots: list[Shot]) -> list[TimeRange]:
        if not shots:
            return []

        similarities = self._shot_visual_similarities(video_path, shots)
        boundaries = [shot.timerange.end for shot in shots[:-1]]
        audio_scores = self._audio_transition_scores(video_path, boundaries)
        return self._group_shots_by_evidence(shots, similarities, audio_scores)

    def _group_shots_by_evidence(
        self,
        shots: list[Shot],
        similarities: list[float | None],
        audio_scores: list[float],
    ) -> list[TimeRange]:
        if not shots:
            return []

        ranges: list[TimeRange] = []
        current_start = shots[0].timerange.start
        for index, next_shot in enumerate(shots[1:]):
            previous_shot = shots[index]
            current_duration = previous_shot.timerange.end - current_start
            if current_duration < self._settings.min_scene_seconds:
                continue

            visual_similarity = similarities[index] if index < len(similarities) else None
            audio_score = audio_scores[index] if index < len(audio_scores) else 0.0
            visual_boundary = (
                visual_similarity is not None
                and visual_similarity < self._settings.scene_visual_similarity_threshold
            )
            audio_boundary = audio_score >= self._settings.scene_audio_boundary_threshold
            if visual_boundary or audio_boundary:
                ranges.append(TimeRange(start=current_start, end=previous_shot.timerange.end))
                current_start = next_shot.timerange.start

        ranges.append(TimeRange(start=current_start, end=shots[-1].timerange.end))
        return ranges

    def _shot_visual_similarities(self, video_path: Path, shots: list[Shot]) -> list[float | None]:
        try:
            import cv2
        except ImportError:
            return [None for _ in shots[:-1]]

        capture = cv2.VideoCapture(str(video_path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        histograms: list[np.ndarray | None] = []
        for shot in shots:
            midpoint = shot.timerange.start + (shot.timerange.duration / 2)
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(midpoint * fps))
            ok, frame = capture.read()
            if not ok:
                histograms.append(None)
                continue
            histograms.append(self._frame_histogram(frame, cv2))
        capture.release()

        similarities: list[float | None] = []
        for left, right in zip(histograms, histograms[1:], strict=False):
            if left is None or right is None:
                similarities.append(None)
                continue
            similarities.append(float(np.minimum(left, right).sum()))
        return similarities

    @staticmethod
    def _frame_histogram(frame: Any, cv2: Any) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        hist = hist.astype(np.float32).flatten()
        total = float(hist.sum())
        if total <= 0:
            return hist
        return hist / total

    def _audio_transition_scores(self, video_path: Path, boundaries: list[float]) -> list[float]:
        if not boundaries:
            return []
        try:
            import librosa
        except ImportError:
            return [0.0 for _ in boundaries]

        audio, sample_rate = librosa.load(str(video_path), sr=22_050, mono=True)
        if audio.size == 0:
            return [0.0 for _ in boundaries]

        hop_length = 512
        envelope = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        delta = np.abs(np.diff(envelope, prepend=envelope[0]))
        if not delta.any():
            return [0.0 for _ in boundaries]

        times = librosa.frames_to_time(np.arange(delta.size), sr=sample_rate, hop_length=hop_length)
        normalizer = float(np.percentile(delta, 95)) or float(delta.max()) or 1.0
        scores: list[float] = []
        for boundary in boundaries:
            mask = (times >= boundary - self._settings.audio_boundary_window_seconds) & (
                times <= boundary + self._settings.audio_boundary_window_seconds
            )
            if not mask.any():
                scores.append(0.0)
                continue
            score = float(delta[mask].max() / normalizer)
            scores.append(min(score, 1.0))
        return scores

    def _merge_shots_by_duration(self, shots: list[Shot]) -> list[TimeRange]:
        if not shots:
            return []
        ranges: list[TimeRange] = []
        current_start = shots[0].timerange.start
        current_end = shots[0].timerange.end
        for shot in shots[1:]:
            if current_end - current_start >= self._settings.min_scene_seconds:
                ranges.append(TimeRange(current_start, current_end))
                current_start = shot.timerange.start
            current_end = shot.timerange.end
        ranges.append(TimeRange(current_start, current_end))
        return ranges

    @staticmethod
    def _video_duration(video_path: Path) -> float:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 1
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        capture.release()
        return float(frame_count / fps)
