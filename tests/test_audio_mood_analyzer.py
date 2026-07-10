from __future__ import annotations

from video_intelligence.infrastructure.audio.analyzer import (
    AudioEvent,
    AudioFeatures,
    WeightedMoodScorer,
)


def test_weighted_mood_scorer_uses_yamnet_events_for_happy_mood() -> None:
    scorer = WeightedMoodScorer(minimum_score=0.25, max_labels=2)

    scores = scorer.score(
        AudioFeatures(
            energy=0.04,
            brightness=3_000.0,
            tempo_bpm=132.0,
            onset_strength=0.4,
            silence_ratio=0.1,
        ),
        [AudioEvent(label="Laughter", confidence=0.92)],
    )

    assert scores[0].label == "happy"


def test_weighted_mood_scorer_uses_sharp_danger_events_for_tense_mood() -> None:
    scorer = WeightedMoodScorer(minimum_score=0.25, max_labels=2)

    scores = scorer.score(
        AudioFeatures(
            energy=0.11,
            brightness=2_100.0,
            tempo_bpm=148.0,
            onset_strength=2.0,
            silence_ratio=0.05,
        ),
        [AudioEvent(label="Siren", confidence=0.88)],
    )

    assert scores[0].label == "tense"


def test_weighted_mood_scorer_keeps_low_energy_scenes_calm() -> None:
    scorer = WeightedMoodScorer(minimum_score=0.25, max_labels=2)

    scores = scorer.score(
        AudioFeatures(
            energy=0.005,
            brightness=800.0,
            tempo_bpm=0.0,
            onset_strength=0.0,
            silence_ratio=0.9,
        ),
        [],
    )

    assert scores[0].label == "calm"
