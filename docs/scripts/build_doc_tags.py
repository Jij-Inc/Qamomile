"""Generate doc-tag indexes, per-tag pages, and inline tag chips.

This script is the source of truth for tag-based discovery UX across the
documentation. It scans every tagged article under

    docs/<lang>/{tutorial,algorithm,usage,integration}/*.py

(reading the MyST ``title`` / ``tags`` frontmatter inside the first
``# %% [markdown]`` cell), and writes:

* ``docs/<lang>/tags/<tag>.md`` — one page per tag, listing every
  article that carries that tag, grouped by section.
* ``docs/<lang>/tags/index.md`` — a tag-cloud landing page.
* The auto-managed ``Tags`` block inside ``docs/<lang>/myst.yml``,
  bracketed by sentinel comments.
* An auto-managed ``<!-- BEGIN auto-tags --> ... <!-- END auto-tags -->``
  chip block inside each tagged ``.py`` file's first markdown cell, so
  every rendered article shows clickable tag chips at the top.
* An auto-managed ``<!-- BEGIN browse-by-tag --> ... <!-- END browse-by-tag -->``
  region inside each section's hand-written ``index.md``, populated
  with a proximity-grouped tag cloud (same level / subsections /
  parent sections / other sections).

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
    "usage",
    "integration",
)

# Whitelist of tags every article is allowed to carry. The taxonomy is
# deliberately small and contributor-controlled; expanding it is a
# maintainer decision, not something a docs PR should do as a side
# effect. CI enforces the whitelist via tests/docs/test_tag_whitelist.py
# — this script itself does not validate, so a stray tag does not crash
# the build, only the test fails on the PR.
ALLOWED_TAGS: frozenset[str] = frozenset({
    # Domain
    "chemistry",
    "differential-equation",
    "error-correction",
    "finance",
    "linear-system",
    "machine-learning",
    "optimization",
    # Method family
    "oracle-based",
    "sample-based",
    "simulation",
    "variational",
    # Article type
    "primitives",
    # Technique
    "circuit-compilation",
    "encoding",
    # Other
    "resource-estimation",
    # Section
    "integration",
})

# Locale-aware copy. Keep the taxonomy identical across locales; only
# display strings differ. Adding a locale = adding an entry here.
STRINGS: dict[str, dict[str, object]] = {
    "en": {
        "tags_index_title": "Tags",
        "tags_index_slug": "tags",
        "tags_index_lead": (
            "Browse the documentation by tag. Click a tag to see every "
            "article that carries it."
        ),
        "browse_by_tag": "Browse by tag",
        "tags_label": "Tags",
        "tag_page_lead": "Articles tagged **`{tag}`** ({count}).",
        "tag_page_back": "← Back to all tags",
        "tag_page_title_fmt": "Tag: {tag}",
        "tag_page_heading_fmt": "`{tag}`",
        "section_titles": {
            "tutorial": "Tutorials",
            "algorithm": "Algorithms",
            "usage": "Usage",
            "integration": "Integration",
        },
        "bucket_labels": {
            "same": "In this section",
            "cousin": "In other sections",
        },
    },
    "ja": {
        "tags_index_title": "タグ",
        "tags_index_slug": "tags",
        "tags_index_lead": (
            "タグからドキュメントを探せます。"
            "タグをクリックすると、そのタグが付いた全記事が表示されます。"
        ),
        "browse_by_tag": "タグで探す",
        "tags_label": "タグ",
        "tag_page_lead": "**`{tag}`** タグが付いた記事 ({count} 件)。",
        "tag_page_back": "← タグ一覧へ戻る",
        "tag_page_title_fmt": "タグ: {tag}",
        "tag_page_heading_fmt": "`{tag}`",
        "section_titles": {
            "tutorial": "チュートリアル",
            "algorithm": "アルゴリズム",
            "usage": "使い方",
            "integration": "インテグレーション",
        },
        "bucket_labels": {
            "same": "このセクション",
            "cousin": "他のセクション",
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


def _extract_h1(body: str) -> str | None:
    """Return the first H1 heading text in a markdown ``body``, or ``None``.

    Scans line-by-line for the first ``# `` heading (rejecting ``##``+).
    Used as a fallback for the article title when frontmatter ``title:``
    is omitted, which is the recommended style — the body's H1 is what
    MyST already renders as the page heading, so duplicating it in
    frontmatter is unnecessary.
    """
    for line in body.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("# ") and not stripped.startswith("##"):
            return stripped[2:].strip()
    return None


def _load_article(py_path: Path, section: str) -> Article | None:
    """Read ``py_path`` and return an :class:`Article`, or ``None``.

    Returns ``None`` when the file has no frontmatter / no ``tags``.

    Title resolution falls back through three sources, in order:

    1. ``title:`` in the article frontmatter (legacy / explicit override).
    2. The first H1 heading in the markdown body — the recommended
       source. Authors only need to write the H1 once.
    3. The filename stem, title-cased (e.g. ``qaoa_maxcut`` →
       ``Qaoa Maxcut``). A last-resort safety net so tag pages still
       render something meaningful for an article missing both.
    """
    text = py_path.read_text(encoding="utf-8")
    _, _, cell = _extract_first_markdown_cell(text)
    fm, body = _parse_article_frontmatter(cell)
    if not fm or not fm.get("tags"):
        return None
    title = str(
        fm.get("title")
        or _extract_h1(body)
        or py_path.stem.replace("_", " ").title()
    )
    raw_tags = fm.get("tags") or []
    if not isinstance(raw_tags, list):
        return None
    tags = tuple(str(t) for t in raw_tags)
    return Article(
        section=section,
        slug=py_path.stem,
        title=title,
        tags=tags,
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


def _chip_html(tag: str, href: str) -> str:
    """Render a single ``<a class="tag-chip">`` chip.

    The chip is plain HTML that MyST passes through verbatim; styling
    lives in ``docs/assets/custom-theme.css`` (``a.tag-chip``).
    """
    return f'<a class="tag-chip" href="{href}">{tag}</a>'


def _chip_from_section(tag: str) -> str:
    """Render a tag chip linking from a section landing page (e.g. algorithm/index.md)."""
    return _chip_html(tag, f"../tags/{tag}.md")


def _chip_from_tags_dir(tag: str) -> str:
    """Render a tag chip linking from inside docs/<lang>/tags/."""
    return _chip_html(tag, f"./{tag}.md")


def _chip_from_article(tag: str) -> str:
    """Render a tag chip linking from a section article (sibling of section dir)."""
    return _chip_html(tag, f"../tags/{tag}.md")


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
        chip_line = " ".join(
            _chip_from_tags_dir(t) for t in sorted(tag_map)
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
        parts.append(" ".join(_chip_from_tags_dir(t) for t in other_tags))
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
            # (chips already space-separated; tag-chip CSS handles its own
            # margin so no extra ` · ` separator is needed)
            parts.append(f"### [{a.title}](../{a.section}/{a.slug}.ipynb)")
            parts.append("")
            if chips:
                parts.append(f"**{strings['tags_label']}:** {chips}")
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
    chips = " ".join(_chip_from_article(t) for t in article.tags)
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
# Section index.md browse-by-tag injection                              #
# --------------------------------------------------------------------- #

BROWSE_BEGIN = "<!-- BEGIN browse-by-tag -->"
BROWSE_END = "<!-- END browse-by-tag -->"

BROWSE_BLOCK_RE = re.compile(
    re.escape(BROWSE_BEGIN) + r"[\s\S]*?" + re.escape(BROWSE_END),
)

# Bucket order = increasing distance from the index.md that hosts the
# chip cloud. Each tag is shown in its closest non-empty bucket only
# (the "highest-only" rule), which keeps the cloud compact when a tag
# spans sections. The current flat layout produces only ``same`` and
# ``cousin``; descendant / ancestor buckets are intentionally absent
# until a section grows nested children.
_BUCKET_ORDER: tuple[str, ...] = ("same", "cousin")
_BUCKET_PRIORITY: dict[str, int] = {b: i for i, b in enumerate(_BUCKET_ORDER)}


def _classify_for_index(article_section: str, index_section: str) -> str:
    """Classify ``article_section`` relative to ``index_section``.

    Returns ``same`` when the article lives in the same section as the
    index, ``cousin`` otherwise. (If we ever introduce nested sections,
    this is the place to teach the classifier about ``descendant`` and
    ``ancestor`` again.)
    """
    if article_section == index_section:
        return "same"
    return "cousin"


def _render_browse_by_tag_block(
    tag_map: dict[str, list[Article]],
    index_section: str,
    strings: dict[str, object],
) -> str:
    """Render the proximity-grouped chip cloud for one section.

    Each tag is placed in its closest non-empty bucket (``same`` first,
    then ``descendant``, ``ancestor``, ``cousin``). The count shown is
    the number of articles inside that bucket — articles that fall into
    farther buckets do not contribute to the displayed count, since the
    tag is suppressed from those buckets.
    """
    bucket_labels = strings["bucket_labels"]
    assert isinstance(bucket_labels, dict)

    bucketed: dict[str, dict[str, int]] = {b: {} for b in _BUCKET_ORDER}
    for tag, articles in tag_map.items():
        if not articles:
            continue
        # Find the closest bucket this tag reaches, count only articles
        # in that bucket.
        closest = min(
            (_classify_for_index(a.section, index_section) for a in articles),
            key=lambda b: _BUCKET_PRIORITY[b],
        )
        count = sum(
            1 for a in articles
            if _classify_for_index(a.section, index_section) == closest
        )
        bucketed[closest][tag] = count

    lines: list[str] = []
    for bucket in _BUCKET_ORDER:
        tag_counts = bucketed[bucket]
        if not tag_counts:
            continue
        chip_line = " ".join(
            _chip_from_section(t) for t in sorted(tag_counts)
        )
        lines.append(f"**{bucket_labels[bucket]}:** {chip_line}")
    return "\n\n".join(lines)


def _inject_browse_by_tag(
    index_path: Path,
    index_section: str,
    tag_map: dict[str, list[Article]],
    strings: dict[str, object],
) -> Path | None:
    """Refresh the auto-managed browse-by-tag block in a section index.

    Looks for a sentinel block::

        <!-- BEGIN browse-by-tag -->
        ...
        <!-- END browse-by-tag -->

    inside ``index_path`` and rewrites its body with the proximity-
    grouped tag cloud (see :func:`_render_browse_by_tag_block`). Section
    index files without the sentinels are left alone — that's how a
    section opts out.

    Returns the path if the file was modified, otherwise ``None``.
    """
    if not index_path.is_file():
        return None
    text = index_path.read_text(encoding="utf-8")
    if not BROWSE_BLOCK_RE.search(text):
        return None
    block = _render_browse_by_tag_block(tag_map, index_section, strings)
    canonical = f"{BROWSE_BEGIN}\n{block}\n{BROWSE_END}"
    new_text = BROWSE_BLOCK_RE.sub(canonical, text, count=1)
    if new_text == text:
        return None
    index_path.write_text(new_text, encoding="utf-8")
    return index_path


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

    # 3. Refresh the auto-managed browse-by-tag block inside each
    # section's hand-written index.md. Sections that haven't opted in
    # (no sentinel block in their index.md) are silently skipped.
    for section in SECTIONS:
        index_path = DOCS_ROOT / lang / section / "index.md"
        modified = _inject_browse_by_tag(index_path, section, tag_map, strings)
        if modified is not None:
            written.append(modified)

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
