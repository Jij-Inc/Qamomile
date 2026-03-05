"""Test that tutorial files execute without errors."""

import os
import runpy
from pathlib import Path

import pytest

try:
    import nbformat
    from nbclient import NotebookClient
    NBCLIENT_AVAILABLE = True
except ImportError:
    NBCLIENT_AVAILABLE = False

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).parent.parent.parent

TUTORIAL_PATTERNS = [
    "docs/en/tutorial/**/*.py",
    "docs/ja/tutorial/**/*.py",
    "docs/en/transpile/**/*.py",
    "docs/ja/transpile/**/*.py",
    "docs/en/tutorial/**/*.ipynb",
    "docs/ja/tutorial/**/*.ipynb",
    "docs/en/transpile/**/*.ipynb",
    "docs/ja/transpile/**/*.ipynb",
]


def discover_tutorial_files() -> list[Path]:
    tutorial_files = []
    for pattern in TUTORIAL_PATTERNS:
        for f in PROJECT_ROOT.glob(pattern):
            # Skip Jupyter checkpoint files
            if ".ipynb_checkpoints" in str(f):
                continue
            tutorial_files.append(f)
    return sorted(tutorial_files)


def get_test_id(file_path: Path) -> str:
    relative = file_path.relative_to(PROJECT_ROOT / "docs")
    return str(relative.with_suffix(""))


def execute_notebook(notebook_path: Path) -> None:
    """Execute a Jupyter notebook file."""
    if not NBCLIENT_AVAILABLE:
        pytest.skip("nbformat and nbclient are required to test .ipynb files")

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()


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
        if tutorial_file.suffix == ".ipynb":
            execute_notebook(tutorial_file)
        else:
            runpy.run_path(str(tutorial_file), run_name="__main__")
    except SystemExit as e:
        if e.code != 0 and e.code is not None:
            pytest.fail(f"Tutorial exited with code {e.code}")
    except Exception as e:
        pytest.fail(f"Tutorial raised an exception: {type(e).__name__}: {e}")
