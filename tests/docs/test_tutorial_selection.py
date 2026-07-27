"""Test changed-page selection for documentation execution tests."""

import pytest

from tests.docs.test_tutorials import PROJECT_ROOT, select_tutorial_files

pytestmark = pytest.mark.docs


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
