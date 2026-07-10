from video_intelligence.domain.models import Scene, VideoAsset

from .models import AudioClip


class LibrosaAudioClipLoader:
    def load(self, video: VideoAsset, scene: Scene, sample_rate: int) -> AudioClip:
        try:
            import librosa
        except ImportError as exc:
            raise RuntimeError("Install librosa for audio mood analysis.") from exc

        samples, actual_sample_rate = librosa.load(
            str(video.path),
            sr=sample_rate,
            mono=True,
            offset=scene.timerange.start,
            duration=scene.timerange.duration,
        )
        return AudioClip(samples=samples, sample_rate=int(actual_sample_rate))