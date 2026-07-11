from pathlib import Path

from test_analysis import get_analysis
from video_intelligence.application.use_cases import AnalyzeYoutubeVideoUseCase
from video_intelligence.domain.models import VideoAnalysis
from video_intelligence.domain.settings import AppSettings, AudioSettings
from video_intelligence.infrastructure.audio.analyzer import HybridAudioMoodAnalyzer
from video_intelligence.infrastructure.audio.music_detector import MusicDetector
from video_intelligence.infrastructure.llm.scene_descriptor import SceneDescriptor
from video_intelligence.infrastructure.persistence.json_repository import JsonAnalysisRepository
from video_intelligence.infrastructure.video.frame_sampler import OpenCvFrameSampler
from video_intelligence.infrastructure.video.scenedetect_segmenter import AudioAwareSceneDetectSegmenter
from video_intelligence.infrastructure.video.youtube_provider import YtDlpVideoProvider
from video_intelligence.infrastructure.vision.deepface_analyzer import DeepFaceAnalyzer
from video_intelligence.infrastructure.vision.yolo_object_detector import YoloObjectDetector


analysis = get_analysis()
# faceAnalyzer = DeepFaceAnalyzer(AppSettings.from_yaml(None).faces)
# faceAnalyzer = faceAnalyzer.extract_faces(analysis.video, analysis.scenes, Path("runs/sample/faces"))
# settings = AppSettings.from_yaml(None)
# segmenter = AudioAwareSceneDetectSegmenter(settings.segmentation)
# usecase = AnalyzeYoutubeVideoUseCase(
#         video_provider=YtDlpVideoProvider(settings.video),
#         shot_segmenter=segmenter,
#         scene_segmenter=segmenter,
#         frame_sampler=OpenCvFrameSampler(),
#         description_generator=SceneDescriptor(settings.llm),
#         face_analyzer=DeepFaceAnalyzer(settings.faces),
#         mood_analyzer=HybridAudioMoodAnalyzer(settings.audio),
#         object_detector=YoloObjectDetector(settings.objects),
#         music_detector=MusicDetector(settings.audio),
#         repository=JsonAnalysisRepository(),
#     )
# face_groups = usecase._group_faces_by_scene(analysis.scenes)
# print(usecase)



music = MusicDetector(AudioSettings())

background_music = music.detect_background_music(analysis.video, analysis.scenes)

analysis = VideoAnalysis(
            video=analysis.video,
            shots=analysis.shots,
            scenes=analysis.scenes,
            face_groups=analysis.face_groups,
            background_music=background_music
        )
repository = JsonAnalysisRepository()
repository.save(analysis, Path("test/data/output.json"))


# repo= JsonAnalysisRepository()
# repo.save(faceAnalyzer, Path("test/data/output.json"))
# from pathlib import Path
# import cv2

# print(cv2.__version__)
# print(cv2.data.haarcascades)

# for p in Path(cv2.data.haarcascades).iterdir():
#     print(p.name)