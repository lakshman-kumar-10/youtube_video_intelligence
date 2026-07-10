from __future__ import annotations

from video_intelligence.domain.models import Scene, VideoAsset


class SimpleAudioMoodAnalyzer:
    def detect_mood(self, video: VideoAsset, scene: Scene) -> list[str]:
        try:
            import librosa
            import numpy as np
        except ImportError:
            return ["unknown"]

        duration = scene.timerange.duration
        audio, sample_rate = librosa.load(
            str(video.path), sr=22_050, mono=True, offset=scene.timerange.start, duration=duration
        )
        if audio.size == 0:
            return ["silent"]

        rms = librosa.feature.rms(y=audio)[0]
        tempo = librosa.beat.tempo(y=audio, sr=sample_rate, aggregate=None)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]

        energy = float(np.mean(rms))
        brightness = float(np.mean(centroid))
        bpm = float(np.mean(tempo)) if tempo.size else 0.0

        words: list[str] = []
        words.append("intense" if energy > 0.08 else "calm")
        words.append("bright" if brightness > 2_000 else "dark")
        if bpm > 130:
            words.append("fast")
        elif bpm > 80:
            words.append("steady")
        else:
            words.append("slow")
        return sorted(set(words))

