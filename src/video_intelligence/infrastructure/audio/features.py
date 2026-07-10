from .models import AudioClip, AudioFeatures


class LibrosaAudioFeatureExtractor:
    def extract(self, clip: AudioClip) -> AudioFeatures:
        try:
            import librosa
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Install librosa and numpy for audio feature extraction.") from exc

        if clip.is_empty:
            return AudioFeatures(
                energy=0.0,
                brightness=0.0,
                tempo_bpm=0.0,
                onset_strength=0.0,
                silence_ratio=1.0,
            )

        samples = clip.samples
        rms = librosa.feature.rms(y=samples)[0]
        centroid = librosa.feature.spectral_centroid(y=samples, sr=clip.sample_rate)[0]
        tempo = librosa.beat.tempo(y=samples, sr=clip.sample_rate, aggregate=None)
        onset_envelope = librosa.onset.onset_strength(y=samples, sr=clip.sample_rate)

        energy = float(np.mean(rms))
        silence_floor = max(float(np.percentile(rms, 20)), 1e-6)
        silence_ratio = float(np.mean(rms <= silence_floor * 1.25))
        return AudioFeatures(
            energy=energy,
            brightness=float(np.mean(centroid)),
            tempo_bpm=float(np.mean(tempo)) if tempo.size else 0.0,
            onset_strength=float(np.mean(onset_envelope)) if onset_envelope.size else 0.0,
            silence_ratio=silence_ratio,
        )