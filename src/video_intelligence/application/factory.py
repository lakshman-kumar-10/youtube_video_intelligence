from __future__ import annotations

from video_intelligence.application.use_cases import AnalyzeYoutubeVideoUseCase
from video_intelligence.domain.settings import AppSettings
from video_intelligence.infrastructure.audio.analyzer import HybridAudioMoodAnalyzer
from video_intelligence.infrastructure.audio.music_detector import MusicDetector
from video_intelligence.infrastructure.llm.scene_descriptor import SceneDescriptor
from video_intelligence.infrastructure.persistence.json_repository import JsonAnalysisRepository
from video_intelligence.infrastructure.video.frame_sampler import OpenCvFrameSampler
from video_intelligence.infrastructure.video.scenedetect_segmenter import AudioAwareSceneDetectSegmenter
from video_intelligence.infrastructure.video.youtube_provider import YtDlpVideoProvider
from video_intelligence.infrastructure.vision.deepface_analyzer import DeepFaceAnalyzer
from video_intelligence.infrastructure.vision.yolo_object_detector import YoloObjectDetector


def build_pipeline(settings: AppSettings) -> AnalyzeYoutubeVideoUseCase:
    segmenter = AudioAwareSceneDetectSegmenter(settings.segmentation)
    return AnalyzeYoutubeVideoUseCase(
        video_provider=YtDlpVideoProvider(settings.video),
        shot_segmenter=segmenter,
        scene_segmenter=segmenter,
        frame_sampler=OpenCvFrameSampler(),
        description_generator=SceneDescriptor(settings.llm),
        face_analyzer=DeepFaceAnalyzer(settings.faces),
        mood_analyzer=HybridAudioMoodAnalyzer(settings.audio),
        object_detector=YoloObjectDetector(settings.objects),
        music_detector=MusicDetector(settings.audio),
        repository=JsonAnalysisRepository(),
    )
