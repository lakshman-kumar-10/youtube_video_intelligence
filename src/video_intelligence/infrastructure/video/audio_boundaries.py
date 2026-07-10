from __future__ import annotations

from pathlib import Path

import numpy as np


class AudioBoundaryAligner:
    """Moves visual boundaries to nearby audio transitions."""

    def __init__(self, window_seconds: float) -> None:
        self._window_seconds = window_seconds

    def align(self, video_path: Path, candidate_seconds: list[float]) -> list[float]:
        if not candidate_seconds:
            return []
        try:
            import librosa
        except ImportError:
            return candidate_seconds

        audio, sample_rate = librosa.load(str(video_path), sr=22_050, mono=True)
        if audio.size == 0:
            return candidate_seconds

        hop_length = 512
        envelope = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        delta = np.abs(np.diff(envelope, prepend=envelope[0]))
        times = librosa.frames_to_time(np.arange(delta.size), sr=sample_rate, hop_length=hop_length)

        aligned: list[float] = []
        for candidate in candidate_seconds:
            mask = (times >= candidate - self._window_seconds) & (
                times <= candidate + self._window_seconds
            )
            if not mask.any():
                aligned.append(candidate)
                continue
            local_indices = np.where(mask)[0]
            best_index = local_indices[int(np.argmax(delta[local_indices]))]
            aligned.append(float(times[best_index]))
        return aligned

