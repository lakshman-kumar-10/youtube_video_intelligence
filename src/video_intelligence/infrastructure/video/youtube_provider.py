from __future__ import annotations

from pathlib import Path

from video_intelligence.domain.models import VideoAsset
from video_intelligence.domain.settings import VideoSettings


class YtDlpVideoProvider:
    def __init__(self, settings: VideoSettings) -> None:
        self._settings = settings

    def fetch(self, url: str, output_dir: Path) -> VideoAsset:
        try:
            from yt_dlp import YoutubeDL
        except ImportError as exc:
            raise RuntimeError("Install yt-dlp to download YouTube videos.") from exc

        output_path = output_dir / self._settings.working_filename
        format_selector = (
            f"bv*[height<={self._settings.max_resolution}]+ba/"
            f"b[height<={self._settings.max_resolution}]/b"
        )
        options = {
            "format": format_selector,
            "outtmpl": str(output_path),
            "merge_output_format": "mp4",
            "quiet": False,
            "noplaylist": True,
        }
        with YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
        return VideoAsset(
            source_url=url,
            path=output_path,
            title=info.get("title"),
            duration=float(info["duration"]) if info.get("duration") else None,
        )

