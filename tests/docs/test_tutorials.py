"""Test that tutorial files execute without errors."""

import runpy
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

import pytest

try:
    import nbformat
    from nbclient import NotebookClient

    NBCLIENT_AVAILABLE = True
except ImportError:
    NBCLIENT_AVAILABLE = False

PROJECT_ROOT = Path(__file__).parent.parent.parent

TUTORIAL_PATTERNS = [
    "docs/en/tutorial/**/*.py",
    "docs/ja/tutorial/**/*.py",
    "docs/en/optimization/**/*.py",
    "docs/ja/optimization/**/*.py",
    "docs/en/tutorial/**/*.ipynb",
    "docs/ja/tutorial/**/*.ipynb",
    "docs/en/optimization/**/*.ipynb",
    "docs/ja/optimization/**/*.ipynb",
    "docs/en/vqa/**/*.py",
    "docs/ja/vqa/**/*.py",
    "docs/en/vqa/**/*.ipynb",
    "docs/ja/vqa/**/*.ipynb",
    # We will not execute the following directories:
    # - collaboration: they may require API keys and may have side effects.
    # - release_notes: they may be quite version specific
    #   and may not follow the same structure as other tutorials.
]

# Tutorials that require optional dependency groups (e.g. chemistry)
# and should be skipped when those dependencies are not installed.
OPTIONAL_SKIP_MODULES = {
    "vqe_for_hydrogen": "openfermion",
}


def discover_tutorial_files() -> list[Path]:
    tutorial_files = []
    for pattern in TUTORIAL_PATTERNS:
        for f in PROJECT_ROOT.glob(pattern):
            # Skip Jupyter checkpoint files
            if ".ipynb_checkpoints" in str(f):
                continue
            # Skip notebooks when a paired .py tutorial exists.
            # The paired Python file exercises the same content without
            # requiring notebook-kernel orchestration.
            if f.suffix == ".ipynb" and f.with_suffix(".py").exists():
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
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    monkeypatch.setenv("QAMOMILE_DOCS_TEST", "1")

    assert tutorial_file.exists(), f"Tutorial file not found: {tutorial_file}"

    for stem, module in OPTIONAL_SKIP_MODULES.items():
        if stem in tutorial_file.stem:
            pytest.importorskip(module)

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
