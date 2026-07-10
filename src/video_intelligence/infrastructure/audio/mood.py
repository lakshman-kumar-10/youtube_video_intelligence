from .models import AudioEvent, AudioFeatures, MoodScore

class WeightedMoodScorer:
    _EVENT_WEIGHTS: dict[str, dict[str, float]] = {
        "happy": {
            "laughter": 0.50,
            "giggle": 0.45,
            "applause": 0.30,
            "cheering": 0.35,
            "music": 0.20,
        },
        "sad": {
            "crying": 0.55,
            "sobbing": 0.55,
            "sad music": 0.45,
            "silence": 0.20,
            "speech": 0.10,
        },
        "tense": {
            "scream": 0.55,
            "siren": 0.50,
            "alarm": 0.45,
            "gunshot": 0.60,
            "explosion": 0.60,
            "crash": 0.45,
            "shout": 0.30,
        },
        "calm": {
            "silence": 0.40,
            "inside, small room": 0.15,
            "speech": 0.10,
            "wind": 0.20,
            "rain": 0.20,
            "water": 0.20,
        },
    }

    def __init__(self, minimum_score: float, max_labels: int) -> None:
        self._minimum_score = minimum_score
        self._max_labels = max_labels

    def score(self, features: AudioFeatures, events: list[AudioEvent]) -> list[MoodScore]:
        scorecard = {
            "happy": MoodScore("happy", 0.0, []),
            "sad": MoodScore("sad", 0.0, []),
            "tense": MoodScore("tense", 0.0, []),
            "calm": MoodScore("calm", 0.0, []),
        }
        self._apply_feature_evidence(scorecard, features)
        self._apply_event_evidence(scorecard, events)

        ranked = sorted(scorecard.values(), key=lambda item: item.score, reverse=True)
        confident = [item for item in ranked if item.score >= self._minimum_score]
        return confident[: self._max_labels] or [ranked[0]]

    def _apply_feature_evidence(
        self, scorecard: dict[str, MoodScore], features: AudioFeatures
    ) -> None:
        if features.silence_ratio > 0.75 or features.energy < 0.015:
            self._add(scorecard, "calm", 0.45, "low energy")
            self._add(scorecard, "sad", 0.15, "low energy")
        if features.energy > 0.08:
            self._add(scorecard, "tense", 0.25, "high energy")
        if features.tempo_bpm >= 125:
            self._add(scorecard, "happy", 0.25, "fast tempo")
            self._add(scorecard, "tense", 0.20, "fast tempo")
        elif 75 <= features.tempo_bpm < 125:
            self._add(scorecard, "calm", 0.15, "steady tempo")
        elif 0 < features.tempo_bpm < 75:
            self._add(scorecard, "sad", 0.20, "slow tempo")
            self._add(scorecard, "calm", 0.15, "slow tempo")
        if features.brightness > 2_500:
            self._add(scorecard, "happy", 0.15, "bright spectrum")
        elif 0 < features.brightness < 1_200:
            self._add(scorecard, "sad", 0.15, "dark spectrum")
        if features.onset_strength > 1.5:
            self._add(scorecard, "tense", 0.20, "sharp onsets")

    def _apply_event_evidence(
        self, scorecard: dict[str, MoodScore], events: list[AudioEvent]
    ) -> None:
        for event in events:
            label = event.label.lower()
            for mood, weights in self._EVENT_WEIGHTS.items():
                for keyword, weight in weights.items():
                    if keyword in label:
                        self._add(
                            scorecard,
                            mood,
                            weight * event.confidence,
                            f"{event.label} ({event.confidence:.2f})",
                        )

    @staticmethod
    def _add(
        scorecard: dict[str, MoodScore], mood: str, amount: float, evidence: str
    ) -> None:
        current = scorecard[mood]
        scorecard[mood] = MoodScore(
            label=current.label,
            score=current.score + amount,
            evidence=[*current.evidence, evidence],
        )