from __future__ import annotations

from pathlib import Path

from video_intelligence.domain.models import Scene, Shot, TimeRange, VideoAsset
from video_intelligence.domain.settings import SegmentationSettings
from video_intelligence.infrastructure.video.audio_boundaries import AudioBoundaryAligner


class AudioAwareSceneDetectSegmenter:
    def __init__(self, settings: SegmentationSettings) -> None:
        self._settings = settings
        self._aligner = AudioBoundaryAligner(settings.audio_boundary_window_seconds)

    def detect_shots(self, video: VideoAsset) -> list[Shot]:
        ranges = self._detect_ranges(video.path, self._settings.shot_threshold)
        return [Shot(index=index, timerange=timerange) for index, timerange in enumerate(ranges, start=1)]

    def detect_scenes(self, video: VideoAsset, shots: list[Shot]) -> list[Scene]:
        ranges = self._detect_ranges(video.path, self._settings.scene_threshold)
        ranges = [item for item in ranges if item.duration >= self._settings.min_scene_seconds]
        if not ranges and shots:
            ranges = self._merge_shots_into_scenes(shots)
        return [Scene(index=index, timerange=timerange) for index, timerange in enumerate(ranges, start=1)]

    def _detect_ranges(self, video_path: Path, threshold: float) -> list[TimeRange]:
        try:
            from scenedetect import ContentDetector, SceneManager, open_video
        except ImportError as exc:
            raise RuntimeError("Install scenedetect[opencv] for shot and scene detection.") from exc

        video = open_video(str(video_path), backend="pyav")
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=threshold))
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

    def _merge_shots_into_scenes(self, shots: list[Shot]) -> list[TimeRange]:
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

