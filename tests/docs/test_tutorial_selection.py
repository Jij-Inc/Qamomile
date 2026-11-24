"""Test changed-page selection for documentation execution tests."""

from pathlib import Path

import matplotlib
import pytest

from tests.docs import test_tutorials
from tests.docs.test_tutorials import PROJECT_ROOT, select_tutorial_files

pytestmark = pytest.mark.docs


def test_docs_use_noninteractive_matplotlib_backend() -> None:
    """Keep local documentation tests independent of GUI interaction."""
    assert matplotlib.get_backend().lower() == "agg"


@pytest.mark.parametrize(
    ("changed_files", "expected_files"),
    [
        (None, ["docs/en/tutorial/example.py", "docs/en/usage/notebook.ipynb"]),
        ([], []),
        (
            ["docs/en/tutorial/example.py"],
            ["docs/en/tutorial/example.py"],
        ),
        (
            ["docs/en/tutorial/example.ipynb"],
            ["docs/en/tutorial/example.py"],
        ),
        (
            ["docs/ja/tutorial/example.py"],
            ["docs/en/tutorial/example.py"],
        ),
        (
            ["docs/ja/tutorial/example.ipynb"],
            ["docs/en/tutorial/example.py"],
        ),
        (
            ["docs/en/usage/notebook.ipynb"],
            ["docs/en/usage/notebook.ipynb"],
        ),
        (
            ["docs/en/usage/notebook.py"],
            ["docs/en/usage/notebook.ipynb"],
        ),
        (["docs/en/release_notes/v1_0_0.md", "docs/assets/logo.svg"], []),
        (
            [
                "docs/en/tutorial/example.py",
                "docs/en/tutorial/example.ipynb",
            ],
            ["docs/en/tutorial/example.py"],
        ),
    ],
)
def test_select_tutorial_files(
    changed_files: list[str] | None, expected_files: list[str]
) -> None:
    """Verify changed paths select canonical runnable documentation files."""
    tutorial_files = [
        PROJECT_ROOT / "docs/en/tutorial/example.py",
        PROJECT_ROOT / "docs/en/usage/notebook.ipynb",
    ]

    selected_files = select_tutorial_files(tutorial_files, changed_files)

    assert selected_files == [PROJECT_ROOT / path for path in expected_files]


def test_reviewed_localizations_run_in_both_languages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full and focused runs execute each reviewed program exactly once."""
    monkeypatch.setattr(test_tutorials, "PROJECT_ROOT", tmp_path)
    inputs = [
        "docs/en/algorithm/example.py",
        "docs/en/algorithm/example.ipynb",
        "docs/ja/algorithm/example.py",
        "docs/ja/algorithm/example.ipynb",
        ".github/docs-localization/algorithm/example.json",
    ]
    for relative in inputs:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    pages = test_tutorials.discover_tutorial_files()
    expected = [tmp_path / inputs[0], tmp_path / inputs[2]]
    assert select_tutorial_files(pages, None) == expected
    assert select_tutorial_files(pages, inputs) == expected
    for relative in inputs:
        assert select_tutorial_files(pages, [relative]) == expected
