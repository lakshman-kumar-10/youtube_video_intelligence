from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from video_intelligence.domain.models import VideoAnalysis


class JsonAnalysisRepository:
    def save(self, analysis: VideoAnalysis, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "analysis.json"
        path.write_text(
            json.dumps(self._to_jsonable(analysis), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def _to_jsonable(self, value: Any) -> Any:
        if is_dataclass(value):
            return self._to_jsonable(asdict(value))
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): self._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._to_jsonable(item) for item in value]
        return value

