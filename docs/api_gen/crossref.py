"""Cross-reference registry for linking between API documentation pages."""

from __future__ import annotations

from pathlib import PurePosixPath


class CrossRefRegistry:
    """Tracks which symbols are defined on which pages for cross-referencing."""

    def __init__(self) -> None:
        self._symbols: dict[str, tuple[str, str]] = {}  # canonical -> (page, anchor)

    def register(self, canonical_path: str, page_path: str, anchor: str) -> None:
        """Register a symbol's location.

        Args:
            canonical_path: Full dotted path (e.g. "qamomile.core.converters.qaoa.QAOAConverter")
            page_path: Relative page path (e.g. "api/core.md")
            anchor: Anchor ID on the page (e.g. "QAOAConverter")
        """
        self._symbols[canonical_path] = (page_path, anchor)

    def resolve(self, canonical_path: str, from_page: str) -> str | None:
        """Get a relative markdown link to a symbol from a given page.

        Returns None if the symbol is not registered.
        """
        if canonical_path not in self._symbols:
            return None
        page_path, anchor = self._symbols[canonical_path]
        if page_path == from_page:
            return f"#{anchor}"
        # Compute relative path
        from_dir = PurePosixPath(from_page).parent
        target = PurePosixPath(page_path)
        try:
            rel = target.relative_to(from_dir)
        except ValueError:
            # Navigate up
            parts_from = from_dir.parts
            parts_to = target.parts
            common = 0
            for a, b in zip(parts_from, parts_to):
                if a == b:
                    common += 1
                else:
                    break
            up = len(parts_from) - common
            rel = PurePosixPath(
                "/".join([".."] * up + list(parts_to[common:]))
            )
        return f"{rel}#{anchor}"

    def make_link(
        self,
        display_name: str,
        canonical_path: str,
        from_page: str,
    ) -> str:
        """Create a markdown link for a symbol, or return plain text if unresolved."""
        ref = self.resolve(canonical_path, from_page)
        if ref is None:
            return f"`{display_name}`"
        return f"[`{display_name}`]({ref})"
