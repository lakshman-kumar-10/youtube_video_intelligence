from functools import cached_property
from pathlib import Path
import csv

from .models import AudioClip, AudioEvent


class NullAudioEventDetector:
    def detect(self, clip: AudioClip) -> list[AudioEvent]:
        return []


class YamnetAudioEventDetector:
    _TARGET_SAMPLE_RATE = 16_000

    def __init__(self, model_url: str, confidence_threshold: float, max_events: int) -> None:
        self._model_url = model_url
        self._confidence_threshold = confidence_threshold
        self._max_events = max_events

    def detect(self, clip: AudioClip) -> list[AudioEvent]:
        if clip.is_empty:
            return []
        if clip.sample_rate != self._TARGET_SAMPLE_RATE:
            raise ValueError("YAMNet clips must be loaded at 16 kHz.")

        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Install numpy for YAMNet audio event detection.") from exc

        scores, _, _ = self._model(np.asarray(clip.samples, dtype=np.float32))
        mean_scores = np.mean(scores.numpy(), axis=0)
        indexes = np.argsort(mean_scores)[::-1]

        events: list[AudioEvent] = []
        for index in indexes:
            confidence = float(mean_scores[index])
            if confidence < self._confidence_threshold:
                break
            events.append(AudioEvent(label=self._labels[int(index)], confidence=confidence))
            if len(events) >= self._max_events:
                break
        return events

    @cached_property
    def _model(self) -> object:
        try:
            import tensorflow_hub as hub
        except ImportError as exc:
            raise RuntimeError("Install tensorflow-hub to enable YAMNet audio events.") from exc
        return hub.load(self._model_url)

    @cached_property
    def _labels(self) -> list[str]:
        class_map_path = Path(self._model.class_map_path().numpy().decode("utf-8"))
        with class_map_path.open(newline="", encoding="utf-8") as file:
            return [row["display_name"] for row in csv.DictReader(file)]