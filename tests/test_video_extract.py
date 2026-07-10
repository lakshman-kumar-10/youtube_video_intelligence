from pathlib import Path

from test_analysis import get_analysis
from video_intelligence.domain.settings import AppSettings
from video_intelligence.infrastructure.persistence.json_repository import JsonAnalysisRepository
from video_intelligence.infrastructure.vision.deepface_analyzer import DeepFaceAnalyzer


analysis = get_analysis()
analysis.scenes
faceAnalyzer = DeepFaceAnalyzer(AppSettings.from_yaml(None).faces)
faceAnalyzer = faceAnalyzer.extract_faces(analysis.video, analysis.scenes, Path("tests/data/faces"))


print(faceAnalyzer)

repo= JsonAnalysisRepository()
repo.save(faceAnalyzer, Path("test/data/output.json"))
# from pathlib import Path
# import cv2

# print(cv2.__version__)
# print(cv2.data.haarcascades)

# for p in Path(cv2.data.haarcascades).iterdir():
#     print(p.name)