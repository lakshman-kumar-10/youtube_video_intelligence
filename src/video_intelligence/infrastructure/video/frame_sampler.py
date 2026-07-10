from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2

from video_intelligence.domain.models import TimeRange


class OpenCvFrameSampler:
    def sample_frames(self, video_path: Path, timerange: TimeRange, every_seconds: float) -> list[Path]:
        output_dir = video_path.parent / "frames" / f"{timerange.start:.2f}_{timerange.end:.2f}"
        output_dir.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(video_path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        paths: list[Path] = []
        timestamp = timerange.start
        while timestamp <= timerange.end:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * fps))
            ok, frame = capture.read()
            if ok:
                frame_path = output_dir / f"frame_{timestamp:.2f}_{uuid4().hex[:8]}.jpg"
                cv2.imwrite(str(frame_path), frame)
                paths.append(frame_path)
            timestamp += every_seconds
        capture.release()
        return paths

