"""Configuration for API documentation generation."""

from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass
class ApiGenConfig:
    """Configuration for API documentation generation."""

    package_name: str = "qamomile"
    split_threshold: int = 3
    github_base_url: str = "https://github.com/Jij-Inc/Qamomile/blob/main"
    docstring_style: str = "google"
    multiline_sig_width: int = 80

    @property
    def repo_root(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    @property
    def package_dir(self) -> Path:
        return self.repo_root / self.package_name

    @property
    def output_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent / "api"

    @property
    def docs_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent
