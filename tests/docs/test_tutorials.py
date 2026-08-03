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

# Tutorials that require credentials or remote side effects and are skipped in CI.
SKIP_TUTORIALS: dict[str, str] = {
    "en/integration/qbraid_executor": "Requires a qBraid API key.",
    "ja/integration/qbraid_executor": "Requires a qBraid API key.",
}

TUTORIAL_PATTERNS = [
    "docs/en/tutorial/**/*.py",
    "docs/ja/tutorial/**/*.py",
    "docs/en/tutorial/**/*.ipynb",
    "docs/ja/tutorial/**/*.ipynb",
    "docs/en/algorithm/**/*.py",
    "docs/ja/algorithm/**/*.py",
    "docs/en/algorithm/**/*.ipynb",
    "docs/ja/algorithm/**/*.ipynb",
    "docs/en/usage/**/*.py",
    "docs/ja/usage/**/*.py",
    "docs/en/usage/**/*.ipynb",
    "docs/ja/usage/**/*.ipynb",
    "docs/en/integration/**/*.py",
    "docs/ja/integration/**/*.py",
    "docs/en/integration/**/*.ipynb",
    "docs/ja/integration/**/*.ipynb",
    # We will not execute the following directories:
    # - release_notes: markdown-only; nothing to execute.
]

# Tutorials that require optional dependency groups (e.g. chemistry)
# and should be skipped when those dependencies are not installed.
OPTIONAL_SKIP_MODULES: dict[str, tuple[str, ...]] = {
    "vqe_for_hydrogen": ("openfermion",),
    "cudaq_support": ("cudaq",),
    "qsci": ("quri_parts",),
    "quri_parts_support": ("quri_parts.qulacs",),
    "hybrid_qnn": ("torch",),
    "ommx_quantum_benchmarks_qaoa": (
        "ommx_quantum_benchmarks",
        "ommx_pyscipopt_adapter",
    ),
}


def discover_tutorial_files() -> list[Path]:
    """Discover runnable documentation pages.

    Returns:
        list[Path]: Runnable Python or unpaired notebook documentation paths.
    """
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


def select_tutorial_files(
    tutorial_files: list[Path], changed_files: list[str] | None
) -> list[Path]:
    """Select runnable pages corresponding to changed documentation files.

    A changed paired notebook selects its Python authoring source because the
    Python file is the canonical executable used by documentation tests.

    Args:
        tutorial_files (list[Path]): All runnable documentation files in the
            current checkout.
        changed_files (list[str] | None): Repository-relative changed paths.
            None selects every runnable page, while an empty list selects none.

    Returns:
        list[Path]: Sorted runnable pages selected for execution.
    """
    if changed_files is None:
        return sorted(tutorial_files)

    files_by_changed_path: dict[str, Path] = {}
    for tutorial_file in tutorial_files:
        relative_path = tutorial_file.relative_to(PROJECT_ROOT).as_posix()
        files_by_changed_path[relative_path] = tutorial_file
        paired_suffix = ".ipynb" if tutorial_file.suffix == ".py" else ".py"
        paired_path = Path(relative_path).with_suffix(paired_suffix).as_posix()
        files_by_changed_path[paired_path] = tutorial_file

    selected_files = {
        files_by_changed_path[changed_file]
        for changed_file in changed_files
        if changed_file in files_by_changed_path
    }
    return sorted(selected_files)


def get_test_id(file_path: Path) -> str:
    """Build a stable pytest identifier for a documentation page.

    Args:
        file_path (Path): Absolute documentation page path.

    Returns:
        str: Extension-free path relative to the docs directory.
    """
    relative = file_path.relative_to(PROJECT_ROOT / "docs")
    return str(relative.with_suffix(""))


def execute_notebook(notebook_path: Path) -> None:
    """Execute a Jupyter notebook file.

    Args:
        notebook_path (Path): Notebook path to execute.
    """
    if not NBCLIENT_AVAILABLE:
        pytest.skip("nbformat and nbclient are required to test .ipynb files")

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()


TUTORIAL_FILES = discover_tutorial_files()


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize tutorial execution from the documentation selection options.

    Args:
        metafunc (pytest.Metafunc): Test function metadata and pytest
            configuration.
    """
    if "tutorial_file" not in metafunc.fixturenames:
        return

    changed_files = None
    if metafunc.config.getoption("changed_docs"):
        changed_files = metafunc.config.getoption("docs_file")
    tutorial_files = select_tutorial_files(TUTORIAL_FILES, changed_files)
    metafunc.parametrize(
        "tutorial_file",
        tutorial_files,
        ids=[get_test_id(file_path) for file_path in tutorial_files],
    )


@pytest.mark.docs
def test_tutorial_executes_without_error(tutorial_file: Path, tmp_path, monkeypatch):
    """Verify that a selected documentation page executes without errors."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    monkeypatch.setenv("QAMOMILE_DOCS_TEST", "1")

    assert tutorial_file.exists(), f"Tutorial file not found: {tutorial_file}"

    test_id = get_test_id(tutorial_file)
    if test_id in SKIP_TUTORIALS:
        pytest.skip(SKIP_TUTORIALS[test_id])

    for stem, modules in OPTIONAL_SKIP_MODULES.items():
        if stem in tutorial_file.stem:
            for module in modules:
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
