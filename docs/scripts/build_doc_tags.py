"""Generate doc-tag indexes, per-tag pages, and inline tag chips.

This script is the source of truth for tag-based discovery UX across the
documentation. It scans every tagged article under

    docs/<lang>/{tutorial,algorithm,optimization,collaboration}/*.py

(reading the MyST ``title`` / ``tags`` frontmatter inside the first
``# %% [markdown]`` cell), and writes:

* ``docs/<lang>/tags/<tag>.md`` — one page per tag, listing every
  article that carries that tag, grouped by section.
* ``docs/<lang>/tags/index.md`` — a tag-cloud landing page.
* ``docs/<lang>/algorithm/index.md`` — algorithm landing page (this
  section keeps its auto-generated index for tag-first browsing).
* The auto-managed ``Tags`` block inside ``docs/<lang>/myst.yml``,
  bracketed by sentinel comments.
* An auto-managed ``<!-- BEGIN auto-tags --> ... <!-- END auto-tags -->``
  chip block inside each tagged ``.py`` file's first markdown cell, so
  every rendered article shows clickable tag chips at the top.

Run from anywhere::

    uv run python docs/scripts/build_doc_tags.py

The script is idempotent: re-running it produces no diff if nothing
changed. Release notes (``docs/<lang>/release_notes/``) are deliberately
out of scope and never scanned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_ROOT = REPO_ROOT / "docs"

# Sections whose .py files participate in tagging. Adding a new section
# is a matter of appending its directory name here.
SECTIONS: tuple[str, ...] = (
    "tutorial",
    "algorithm",
    "optimization",
    "collaboration",
)

# Sections that get an auto-generated index.md (tag-first landing page).
# Other sections keep their hand-written index.md (with custom intro
# text), and contributors are expected to maintain those manually.
AUTO_INDEX_SECTIONS: tuple[str, ...] = ("algorithm",)

# Locale-aware copy. Keep the taxonomy identical across locales; only
# display strings differ. Adding a locale = adding an entry here.
STRINGS: dict[str, dict[str, object]] = {
    "en": {
        "algorithm_title": "Algorithms",
        "algorithm_slug": "algorithm",
        "algorithm_lead": (
            "Concrete quantum algorithm examples built with Qamomile. "
            "Click a tag below to filter, or browse all algorithms."
        ),
        "tags_index_title": "Tags",
        "tags_index_slug": "tags",
        "tags_index_lead": (
            "Browse the documentation by tag. Click a tag to see every "
            "article that carries it."
        ),
        "browse_by_tag": "Browse by tag",
        "all_in_section": "All articles",
        "tags_label": "Tags",
        "tag_page_lead": "Articles tagged **`{tag}`** ({count}).",
        "tag_page_back": "← Back to all tags",
        "tag_page_title_fmt": "Tag: {tag}",
        "tag_page_heading_fmt": "`{tag}`",
        "section_titles": {
            "tutorial": "Tutorials",
            "algorithm": "Algorithms",
            "optimization": "Optimization",
            "collaboration": "Collaboration",
        },
    },
    "ja": {
        "algorithm_title": "アルゴリズム",
        "algorithm_slug": "algorithm",
        "algorithm_lead": (
            "Qamomileで実装した具体的な量子アルゴリズム例です。"
            "下のタグをクリックして絞り込むか、全アルゴリズムから選んでください。"
        ),
        "tags_index_title": "タグ",
        "tags_index_slug": "tags",
        "tags_index_lead": (
            "タグからドキュメントを探せます。"
            "タグをクリックすると、そのタグが付いた全記事が表示されます。"
        ),
        "browse_by_tag": "タグで探す",
        "all_in_section": "すべての記事",
        "tags_label": "タグ",
        "tag_page_lead": "**`{tag}`** タグが付いた記事 ({count} 件)。",
        "tag_page_back": "← タグ一覧へ戻る",
        "tag_page_title_fmt": "タグ: {tag}",
        "tag_page_heading_fmt": "`{tag}`",
        "section_titles": {
            "tutorial": "チュートリアル",
            "algorithm": "アルゴリズム",
            "optimization": "最適化",
            "collaboration": "コラボレーション",
        },
    },
}


@dataclass(frozen=True)
class Article:
    """A tagged documentation article scanned from a ``.py`` source."""

    section: str  # one of SECTIONS
    slug: str  # filename stem, e.g. "qaoa_maxcut"
    title: str
    tags: tuple[str, ...]
    summary: str
    py_path: Path  # absolute path to the source .py


# --------------------------------------------------------------------- #
# .py parsing helpers                                                   #
# --------------------------------------------------------------------- #


def _extract_first_markdown_cell(py_text: str) -> tuple[int, int, str]:
    """Locate and decode the first ``# %% [markdown]`` cell.

    Returns ``(cell_start, cell_end, markdown_body)`` where the bounds
    are byte offsets into ``py_text`` covering the cell body (between
    the marker and the next ``# %%`` marker / EOF), and the body has
    been stripped of the leading ``# `` per-line comment prefix.
    Returns ``(-1, -1, "")`` when no markdown cell exists.
    """
    marker = "# %% [markdown]\n"
    start = py_text.find(marker)
    if start == -1:
        return -1, -1, ""
    body_start = start + len(marker)
    end = py_text.find("\n# %%", body_start)
    body_end = len(py_text) if end == -1 else end + 1  # include final newline
    body_lines = py_text[body_start:body_end].splitlines()
    decoded: list[str] = []
    for line in body_lines:
        if line.startswith("# "):
            decoded.append(line[2:])
        elif line == "#":
            decoded.append("")
        else:
            decoded.append(line)
    return body_start, body_end, "\n".join(decoded).strip()


def _parse_article_frontmatter(cell_md: str) -> tuple[dict, str]:
    """Split a markdown cell into ``(frontmatter_dict, remainder)``.

    Recognises a leading ``---\\n...---\\n`` block. Returns ``({}, cell_md)``
    when no frontmatter is present or YAML parsing fails.
    """
    if not cell_md.startswith("---"):
        return {}, cell_md
    end_idx = cell_md.find("\n---", 3)
    if end_idx == -1:
        return {}, cell_md
    yaml_block = cell_md[3:end_idx].strip()
    rest = cell_md[end_idx + len("\n---") :].lstrip()
    try:
        fm = yaml.safe_load(yaml_block) or {}
    except yaml.YAMLError:
        fm = {}
    return fm, rest


def _extract_summary(body_md: str) -> str:
    """Return the first non-H1 paragraph of the markdown body."""
    lines = body_md.splitlines()
    i = 0
    while i < len(lines) and not lines[i].startswith("# "):
        i += 1
    if i < len(lines):
        i += 1  # skip past H1
    while i < len(lines) and not lines[i].strip():
        i += 1
    buf: list[str] = []
    while i < len(lines) and lines[i].strip():
        buf.append(lines[i].strip())
        i += 1
    return " ".join(buf)


def _load_article(py_path: Path, section: str) -> Article | None:
    """Read ``py_path`` and return an :class:`Article`, or ``None``.

    Returns ``None`` when the file has no frontmatter / no ``tags``.
    """
    text = py_path.read_text(encoding="utf-8")
    _, _, cell = _extract_first_markdown_cell(text)
    fm, body = _parse_article_frontmatter(cell)
    if not fm or not fm.get("tags"):
        return None
    title = str(fm.get("title") or py_path.stem.replace("_", " ").title())
    raw_tags = fm.get("tags") or []
    if not isinstance(raw_tags, list):
        return None
    tags = tuple(str(t) for t in raw_tags)
    summary = _extract_summary(body)
    return Article(
        section=section,
        slug=py_path.stem,
        title=title,
        tags=tags,
        summary=summary,
        py_path=py_path,
    )


def _walk_articles(lang: str) -> tuple[list[Article], list[Path]]:
    """Walk every section under ``docs/<lang>/``.

    Returns ``(tagged_articles, untagged_py_paths)``. The untagged list
    is used to strip stale chip blocks from files whose ``tags:``
    frontmatter has been removed.
    """
    tagged: list[Article] = []
    untagged: list[Path] = []
    for section in SECTIONS:
        sec_dir = DOCS_ROOT / lang / section
        if not sec_dir.is_dir():
            continue
        for py in sorted(sec_dir.glob("*.py")):
            art = _load_article(py, section)
            if art is not None:
                tagged.append(art)
            else:
                untagged.append(py)
    return tagged, untagged


def _tag_map(articles: Iterable[Article]) -> dict[str, list[Article]]:
    """Group articles by tag, sorted by (section order, title)."""
    out: dict[str, list[Article]] = {}
    for a in articles:
        for t in a.tags:
            out.setdefault(t, []).append(a)
    section_order = {s: i for i, s in enumerate(SECTIONS)}
    for t in out:
        out[t].sort(key=lambda a: (section_order.get(a.section, 99), a.title))
    return out


# --------------------------------------------------------------------- #
# Markdown rendering                                                    #
# --------------------------------------------------------------------- #


def _chip_from_section(tag: str) -> str:
    """Render a tag chip linking from a section landing page (e.g. algorithm/index.md)."""
    return f"[`{tag}`](../tags/{tag}.md)"


def _chip_from_tags_dir(tag: str) -> str:
    """Render a tag chip linking from inside docs/<lang>/tags/."""
    return f"[`{tag}`](./{tag}.md)"


def _chip_from_article(tag: str) -> str:
    """Render a tag chip linking from a section article (sibling of section dir)."""
    return f"[`{tag}`](../tags/{tag}.md)"


def _render_article_card(
    article: Article,
    strings: dict[str, object],
    chip_renderer,
    href_prefix: str,
) -> str:
    chips = " ".join(chip_renderer(t) for t in article.tags)
    summary = article.summary or ""
    return (
        f"### [{article.title}]({href_prefix}{article.slug}.ipynb)\n"
        f"\n"
        f"**{strings['tags_label']}:** {chips}\n"
        f"\n"
        f"{summary}\n"
    )


def _render_algorithm_index(
    articles: list[Article],
    tag_map: dict[str, list[Article]],
    strings: dict[str, object],
) -> str:
    """Render ``docs/<lang>/algorithm/index.md`` (algorithm-only listing)."""
    parts: list[str] = ["---"]
    parts.append(f"slug: {strings['algorithm_slug']}")
    parts.append(f"title: {strings['algorithm_title']}")
    parts.append("---")
    parts.append("")
    parts.append(f"# {strings['algorithm_title']}")
    parts.append("")
    parts.append(str(strings["algorithm_lead"]))
    parts.append("")
    parts.append(f"## {strings['browse_by_tag']}")
    parts.append("")
    # Only include tags that have at least one algorithm article.
    algo_tags = sorted(
        t for t, arts in tag_map.items()
        if any(a.section == "algorithm" for a in arts)
    )
    if algo_tags:
        chip_line = " · ".join(
            f"{_chip_from_section(t)} ({sum(1 for a in tag_map[t] if a.section == 'algorithm')})"
            for t in algo_tags
        )
        parts.append(chip_line)
        parts.append("")
    parts.append(f"## {strings['all_in_section']}")
    parts.append("")
    for a in articles:
        parts.append(_render_article_card(a, strings, _chip_from_section, ""))
    return "\n".join(parts).rstrip() + "\n"


def _render_tags_index(
    tag_map: dict[str, list[Article]],
    strings: dict[str, object],
) -> str:
    """Render ``docs/<lang>/tags/index.md`` (global tag cloud)."""
    parts: list[str] = ["---"]
    parts.append(f"slug: {strings['tags_index_slug']}")
    parts.append(f"title: {strings['tags_index_title']}")
    parts.append("---")
    parts.append("")
    parts.append(f"# {strings['tags_index_title']}")
    parts.append("")
    parts.append(str(strings["tags_index_lead"]))
    parts.append("")
    if tag_map:
        chip_line = " · ".join(
            f"{_chip_from_tags_dir(t)} ({len(tag_map[t])})"
            for t in sorted(tag_map)
        )
        parts.append(chip_line)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _render_tag_page(
    tag: str,
    tag_articles: list[Article],
    strings: dict[str, object],
    all_tags: list[str],
) -> str:
    """Render a single ``docs/<lang>/tags/<tag>.md`` page."""
    parts: list[str] = ["---"]
    title_text = str(strings["tag_page_title_fmt"]).format(tag=tag).replace('"', '\\"')
    # YAML rejects plain scalars containing `: ` or starting with a
    # backtick, so always wrap title in double quotes.
    parts.append(f'title: "{title_text}"')
    parts.append(f"tags: [{tag}]")
    parts.append("---")
    parts.append("")
    parts.append(f"# {str(strings['tag_page_heading_fmt']).format(tag=tag)}")
    parts.append("")
    parts.append(
        str(strings["tag_page_lead"]).format(tag=tag, count=len(tag_articles))
    )
    parts.append("")
    parts.append(f"[{strings['tag_page_back']}](./index.md)")
    parts.append("")
    # Other tags for hopping.
    other_tags = [t for t in all_tags if t != tag]
    if other_tags:
        parts.append("---")
        parts.append("")
        parts.append(f"**{strings['browse_by_tag']}:** ")
        parts.append(" · ".join(_chip_from_tags_dir(t) for t in other_tags))
        parts.append("")
    parts.append("---")
    parts.append("")
    section_titles = strings["section_titles"]
    assert isinstance(section_titles, dict)
    by_section: dict[str, list[Article]] = {}
    for a in tag_articles:
        by_section.setdefault(a.section, []).append(a)
    for section in SECTIONS:
        section_articles = by_section.get(section, [])
        if not section_articles:
            continue
        parts.append(f"## {section_titles[section]}")
        parts.append("")
        for a in section_articles:
            chips = " ".join(
                _chip_from_tags_dir(t) for t in a.tags if t != tag
            )
            parts.append(f"### [{a.title}](../{a.section}/{a.slug}.ipynb)")
            parts.append("")
            if chips:
                parts.append(f"**{strings['tags_label']}:** {chips}")
                parts.append("")
            if a.summary:
                parts.append(a.summary)
                parts.append("")
    return "\n".join(parts).rstrip() + "\n"


# --------------------------------------------------------------------- #
# .py chip injection                                                    #
# --------------------------------------------------------------------- #

CHIP_BEGIN = "# <!-- BEGIN auto-tags -->"
CHIP_END = "# <!-- END auto-tags -->"

# Match an existing chip block plus any surrounding blank ``#`` lines
# (one or more on each side). Replacing this with the canonical
# ``#\n<block>\n#\n`` collapses accidental double-blanks left over from
# earlier insertions.
CHIP_BLOCK_RE = re.compile(
    r"(?:^#\n)+"
    + re.escape(CHIP_BEGIN)
    + r"\n[\s\S]*?\n"
    + re.escape(CHIP_END)
    + r"\n(?:^#\n)+",
    re.MULTILINE,
)


def _build_chip_block(article: Article, strings: dict[str, object]) -> str:
    """Build the per-article chip block (already prefixed with ``# `` for .py)."""
    chips = " · ".join(_chip_from_article(t) for t in article.tags)
    return (
        f"{CHIP_BEGIN}\n"
        f"# **{strings['tags_label']}:** {chips}\n"
        f"{CHIP_END}"
    )


def _inject_tag_chips(article: Article, strings: dict[str, object]) -> Path | None:
    """Insert or refresh the auto-tags block in ``article.py_path``.

    Returns the path if the file was modified, otherwise ``None``.
    """
    text = article.py_path.read_text(encoding="utf-8")
    new_block = _build_chip_block(article, strings)
    canonical = "#\n" + new_block + "\n#\n"

    # Case 1: a sentinel block already exists — replace it in place,
    # collapsing any accidental surrounding blank lines.
    if CHIP_BLOCK_RE.search(text):
        new_text = CHIP_BLOCK_RE.sub(canonical, text, count=1)
        if new_text == text:
            return None
        article.py_path.write_text(new_text, encoding="utf-8")
        return article.py_path

    # Case 2: no sentinel yet — find the H1 in the first markdown cell
    # and insert the block right after it. Drop any existing blank `#`
    # lines that already sit between the H1 and the next content; the
    # canonical block carries its own surrounding blanks.
    cell_start, cell_end, _ = _extract_first_markdown_cell(text)
    if cell_start < 0:
        return None
    cell = text[cell_start:cell_end]
    cell_lines = cell.splitlines(keepends=True)
    h1_idx: int | None = None
    for i, line in enumerate(cell_lines):
        if line.startswith("# # "):
            h1_idx = i
            break
    if h1_idx is None:
        return None
    after = h1_idx + 1
    while after < len(cell_lines) and cell_lines[after].rstrip("\n") == "#":
        after += 1
    new_cell = (
        "".join(cell_lines[: h1_idx + 1])
        + canonical
        + "".join(cell_lines[after:])
    )
    new_text = text[:cell_start] + new_cell + text[cell_end:]
    if new_text == text:
        return None
    article.py_path.write_text(new_text, encoding="utf-8")
    return article.py_path


def _strip_chip_block(py_path: Path) -> Path | None:
    """Remove a leftover auto-tags chip block when an article is untagged.

    When a contributor removes the ``tags:`` frontmatter from a ``.py``
    source, the article no longer needs an inline chip line. This
    routine detects an existing sentinel block (with its surrounding
    canonical blank ``#`` lines) and collapses it down to a single
    blank ``#`` line so the file lands in a clean state. Returns the
    path if the file was modified, otherwise ``None``.
    """
    text = py_path.read_text(encoding="utf-8")
    if not CHIP_BLOCK_RE.search(text):
        return None
    new_text = CHIP_BLOCK_RE.sub("#\n", text, count=1)
    if new_text == text:
        return None
    py_path.write_text(new_text, encoding="utf-8")
    return py_path


# --------------------------------------------------------------------- #
# myst.yml auto-managed Tags region                                     #
# --------------------------------------------------------------------- #

TAGS_TOC_BEGIN = "# --- BEGIN doc tags (auto-generated) ---"
TAGS_TOC_END = "# --- END doc tags (auto-generated) ---"
# Legacy sentinel (left over from the algorithm-only generator) — the
# script accepts either pair on input but always writes the new pair.
LEGACY_TAGS_TOC_BEGIN = "# --- BEGIN algorithm tags (auto-generated) ---"
LEGACY_TAGS_TOC_END = "# --- END algorithm tags (auto-generated) ---"


def _render_tags_toc_block(all_tags: list[str]) -> str:
    """Render the auto-managed Tags toc block for ``myst.yml``.

    Emits the global ``tags/index.md`` and every per-tag page as a flat
    list of toc entries, each marked ``hidden: true``. This keeps the
    pages buildable (so the URLs resolve) without ever showing a
    ``Tag: <name>`` entry in the rendered navigation. Earlier versions
    used a parent ``Tags`` group with ``hidden: true``, but that
    setting did not propagate to children in the rendered sidebar.
    """
    lines = [TAGS_TOC_BEGIN]
    lines.append("    - file: tags/index.md")
    lines.append("      hidden: true")
    for tag in all_tags:
        lines.append(f"    - file: tags/{tag}.md")
        lines.append("      hidden: true")
    lines.append("    " + TAGS_TOC_END)
    return "\n".join(lines)


def _update_myst_yml(lang: str, all_tags: list[str]) -> Path | None:
    """Rewrite the auto-managed Tags block in ``docs/<lang>/myst.yml``.

    Recognises both the new ``# --- BEGIN doc tags ...`` sentinels and
    the legacy ``# --- BEGIN algorithm tags ...`` ones; writes back the
    new sentinels. Returns the path if the file was modified.
    """
    myst_path = DOCS_ROOT / lang / "myst.yml"
    if not myst_path.is_file():
        return None
    text = myst_path.read_text(encoding="utf-8")

    has_new = TAGS_TOC_BEGIN in text and TAGS_TOC_END in text
    has_legacy = LEGACY_TAGS_TOC_BEGIN in text and LEGACY_TAGS_TOC_END in text
    if not (has_new or has_legacy):
        return None

    if has_new:
        begin, end = TAGS_TOC_BEGIN, TAGS_TOC_END
    else:
        begin, end = LEGACY_TAGS_TOC_BEGIN, LEGACY_TAGS_TOC_END

    pattern = re.compile(
        r"[ \t]*" + re.escape(begin) + r"[\s\S]*?" + re.escape(end),
        re.MULTILINE,
    )
    new_block = _render_tags_toc_block(all_tags)
    new_text = pattern.sub(new_block, text, count=1)
    if new_text == text:
        return None
    myst_path.write_text(new_text, encoding="utf-8")
    return myst_path


# --------------------------------------------------------------------- #
# Stale-file cleanup                                                    #
# --------------------------------------------------------------------- #


def _clean_stale(directory: Path, keep: set[Path]) -> list[Path]:
    """Remove any ``*.md`` under ``directory`` that is not in ``keep``."""
    removed: list[Path] = []
    if not directory.is_dir():
        return removed
    for existing in directory.glob("*.md"):
        if existing not in keep:
            existing.unlink()
            removed.append(existing)
    return removed


# --------------------------------------------------------------------- #
# Per-locale build                                                      #
# --------------------------------------------------------------------- #


def _build_for_locale(lang: str) -> tuple[list[Path], list[Path]]:
    """Generate everything for one locale.

    Returns ``(written, removed)`` paths for caller logging.
    """
    if lang not in STRINGS:
        raise ValueError(f"unknown locale {lang!r}")
    strings = STRINGS[lang]

    articles, untagged_paths = _walk_articles(lang)
    tag_map = _tag_map(articles)
    all_tags = sorted(tag_map)

    written: list[Path] = []
    removed: list[Path] = []

    # 1a. Inject / refresh inline chip blocks in each tagged .py source.
    for art in articles:
        modified = _inject_tag_chips(art, strings)
        if modified is not None:
            written.append(modified)

    # 1b. Strip stale chip blocks from .py sources that no longer carry
    # ``tags:`` frontmatter — otherwise an old chip line would survive
    # the un-tagging.
    for py in untagged_paths:
        stripped = _strip_chip_block(py)
        if stripped is not None:
            written.append(stripped)

    # 2. Generate global tags/ pages.
    tags_dir = DOCS_ROOT / lang / "tags"
    tags_dir.mkdir(exist_ok=True)
    keep = {tags_dir / "index.md"} | {tags_dir / f"{t}.md" for t in all_tags}
    removed.extend(_clean_stale(tags_dir, keep))

    tags_index = tags_dir / "index.md"
    tags_index.write_text(_render_tags_index(tag_map, strings), encoding="utf-8")
    written.append(tags_index)

    for tag in all_tags:
        page = tags_dir / f"{tag}.md"
        page.write_text(
            _render_tag_page(tag, tag_map[tag], strings, all_tags),
            encoding="utf-8",
        )
        written.append(page)

    # 3. Generate algorithm/index.md (only this section is auto-indexed).
    if "algorithm" in AUTO_INDEX_SECTIONS:
        algo_dir = DOCS_ROOT / lang / "algorithm"
        if algo_dir.is_dir():
            algo_articles = sorted(
                (a for a in articles if a.section == "algorithm"),
                key=lambda a: a.title,
            )
            index_path = algo_dir / "index.md"
            index_path.write_text(
                _render_algorithm_index(algo_articles, tag_map, strings),
                encoding="utf-8",
            )
            written.append(index_path)

    # 4. Remove any leftover docs/<lang>/algorithm/tags/ from the old
    # layout so the build doesn't pick them up alongside the new path.
    legacy_dir = DOCS_ROOT / lang / "algorithm" / "tags"
    if legacy_dir.is_dir():
        for old in legacy_dir.glob("*.md"):
            old.unlink()
            removed.append(old)
        try:
            legacy_dir.rmdir()
        except OSError:
            pass

    # 5. Update myst.yml auto-managed Tags region.
    myst_updated = _update_myst_yml(lang, all_tags)
    if myst_updated is not None:
        written.append(myst_updated)

    return written, removed


def main() -> None:
    """Regenerate every locale and print a one-line audit per file."""
    for lang in ("en", "ja"):
        written, removed = _build_for_locale(lang)
        for p in written:
            print(f"wrote {p.relative_to(REPO_ROOT)}")
        for p in removed:
            print(f"removed {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
