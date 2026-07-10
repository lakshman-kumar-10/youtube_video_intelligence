from __future__ import annotations

from pathlib import Path

import click

from video_intelligence.application.factory import build_pipeline
from video_intelligence.domain.settings import AppSettings


@click.group()
def main() -> None:
    """YouTube video intelligence CLI."""


@main.command()
@click.argument("youtube_url")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=Path("runs/default"),
    show_default=True,
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, file_okay=True, dir_okay=False),
    default=None,
)
def analyze(youtube_url: str, output_dir: Path, config_path: Path | None) -> None:
    """Analyze one YouTube video."""
    settings = AppSettings.from_yaml(config_path)
    report_path = build_pipeline(settings).execute(youtube_url, output_dir)
    click.echo(f"Analysis written to: {report_path}")


if __name__ == "__main__":
    main()

