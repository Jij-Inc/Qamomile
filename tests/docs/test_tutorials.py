"""Test that tutorial files execute without errors."""

import os
import runpy
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).parent.parent.parent

TUTORIAL_PATTERNS = [
    "docs/en/tutorial/*.py",
    "docs/ja/tutorial/*.py",
    "docs/en/transpile/*.py",
    "docs/ja/transpile/*.py",
]


def discover_tutorial_files() -> list[Path]:
    tutorial_files = []
    for pattern in TUTORIAL_PATTERNS:
        tutorial_files.extend(PROJECT_ROOT.glob(pattern))
    return sorted(tutorial_files)


def get_test_id(file_path: Path) -> str:
    relative = file_path.relative_to(PROJECT_ROOT / "docs")
    return str(relative.with_suffix(""))


TUTORIAL_FILES = discover_tutorial_files()


@pytest.mark.docs
@pytest.mark.parametrize(
    "tutorial_file",
    TUTORIAL_FILES,
    ids=[get_test_id(f) for f in TUTORIAL_FILES],
)
def test_tutorial_executes_without_error(tutorial_file: Path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")

    assert tutorial_file.exists(), f"Tutorial file not found: {tutorial_file}"

    try:
        runpy.run_path(str(tutorial_file), run_name="__main__")
    except SystemExit as e:
        if e.code != 0 and e.code is not None:
            pytest.fail(f"Tutorial exited with code {e.code}")
    except Exception as e:
        pytest.fail(f"Tutorial raised an exception: {type(e).__name__}: {e}")
