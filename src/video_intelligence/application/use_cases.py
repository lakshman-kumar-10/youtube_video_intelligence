from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from video_intelligence.domain.models import FaceGroup, Scene, VideoAnalysis, VideoAsset
from video_intelligence.domain.ports import (
    AnalysisRepository,
    FaceAnalyzer,
    MoodAnalyzer,
    MusicDetector,
    ObjectDetector,
    SceneDescriptionGenerator,
    SceneFrameSampler,
    SceneSegmenter,
    ShotSegmenter,
    VideoProvider,
)


@dataclass(frozen=True)
class AnalyzeYoutubeVideoUseCase:
    video_provider: VideoProvider
    shot_segmenter: ShotSegmenter
    scene_segmenter: SceneSegmenter
    frame_sampler: SceneFrameSampler
    description_generator: SceneDescriptionGenerator
    face_analyzer: FaceAnalyzer
    mood_analyzer: MoodAnalyzer
    object_detector: ObjectDetector
    music_detector: MusicDetector
    repository: AnalysisRepository

    def execute(self, url: str, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        video = self.video_provider.fetch(url, output_dir)

        shots = self.shot_segmenter.detect_shots(video)
        scenes = self.scene_segmenter.detect_scenes(video, shots)
        scenes = self._add_scene_intelligence(video, scenes, output_dir)
        scenes = self.face_analyzer.extract_faces(video, scenes, output_dir / "faces")

        face_groups = self._group_faces_by_scene(scenes)
        background_music = self.music_detector.detect_background_music(video, scenes)

        analysis = VideoAnalysis(
            video=video,
            shots=shots,
            scenes=scenes,
            face_groups=face_groups,
            background_music=background_music,
        )
        return self.repository.save(analysis, output_dir)

    def _group_faces_by_scene(self, scenes: list[Scene]) -> list[FaceGroup]:
        face_groups: list[FaceGroup] = []

        for scene in scenes:
            scene_face_groups = self.face_analyzer.group_faces(scene.faces)
            face_groups.extend(
                self._scope_face_group_to_scene(scene.index, group)
                for group in scene_face_groups
            )

        return face_groups

    @staticmethod
    def _scope_face_group_to_scene(scene_index: int, group: FaceGroup) -> FaceGroup:
        return FaceGroup(
            group_id=f"scene_{scene_index:03d}_{group.group_id}",
            faces=group.faces,
            scene_emotions=group.scene_emotions,
        )

    def _add_scene_intelligence(
        self, video: VideoAsset, scenes: list[Scene], output_dir: Path
    ) -> list[Scene]:
        enriched: list[Scene] = []
        output_dir.mkdir(parents=True, exist_ok=True)

        for scene in scenes:
            objects = self.object_detector.detect_objects(video, scene)
            mood = self.mood_analyzer.detect_mood(video, scene)
            frames = self.frame_sampler.sample_frames(video.path, scene.timerange, every_seconds=2.0)
            description = self.description_generator.describe(scene, frames, objects, mood)
            enriched.append(
                Scene(
                    index=scene.index,
                    timerange=scene.timerange,
                    description=description,
                    mood=mood,
                    objects=objects,
                    faces=scene.faces,
                )
            )
        return enriched
