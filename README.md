# YouTube Video Intelligence

Python project for selecting one YouTube video and producing:

- Audio-aware shot start and stop times.
- Audio-aware scene start and stop times.
- LLM scene descriptions under 100 words.
- Scene-wise face crops.
- Similar-face groups with scene-wise emotions.
- Scene mood/sentiment as word lists.
- Scene objects as word lists.
- Background music start and end times across scenes.

## Architecture

The project uses SOLID-friendly ports and adapters:

- `domain`: business models and protocol interfaces.
- `application`: orchestration use case.
- `infrastructure`: library-specific adapters for YouTube, scene detection, audio, LLM, faces, objects, and persistence.
- `interfaces`: CLI entrypoint.

Design patterns used:

- Strategy: interchangeable segmentation, LLM, face, object, and audio analyzers.
- Factory: `build_pipeline` wires production adapters.
- Repository: `JsonAnalysisRepository` persists output.
- Facade / Use Case: `AnalyzeYoutubeVideoUseCase` coordinates the workflow.
- Dependency inversion: application depends on domain protocols.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

For LLM descriptions, set:

```bash
set OPENAI_API_KEY=your_key
```

## Run

```bash
video-intel analyze "https://www.youtube.com/watch?v=VIDEO_ID" --output-dir runs/sample
```

Or:

```bash
python -m video_intelligence.interfaces.cli analyze "https://www.youtube.com/watch?v=VIDEO_ID" --output-dir runs/sample
```

The final report is written to `runs/sample/analysis.json`; face crops are under
`runs/sample/faces/scene_<n>/`.

## Notes

- Shot and scene boundaries are refined using both visual cuts and audio transitions so start/stop times remain aligned with the soundtrack.
- The default object detector uses YOLOv8, face/emotion grouping uses DeepFace, and descriptions use the OpenAI API.
- Heavy ML libraries may download model weights on first run.

