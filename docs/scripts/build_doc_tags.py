"""Generate doc-tag indexes, per-tag pages, inline tag chips, and card metadata.

This script is the source of truth for tag-based discovery UX across the
documentation. It scans every tagged article under

    docs/<lang>/{tutorial,algorithm,usage,integration}/*.py

(reading the MyST ``title`` / ``tags`` frontmatter inside the first
``# %% [markdown]`` cell), and writes:

* ``docs/<lang>/tags/<tag>.md`` — one page per tag, rendering every
  article that carries that tag as a card, grouped by section.
* ``docs/<lang>/tags/index.md`` — a tag-cloud landing page.
* A chip block inserted right after the first H1 inside each tagged
  ``.py`` file's first markdown cell, so every rendered article shows
  clickable tag chips at the top.
* A "Browse by tag" section (heading + chip cloud) inserted before
  the article card grid in each section's ``index.md``, presenting a
  proximity-grouped tag cloud (this section / other sections).
* Tag chips plus a thumbnail slot inserted into each section
  ``index.md`` card in the build-dir copy. Cards keep their existing
  descriptions, while article navigation moves to the card header so
  tag chips can be independent links. When an article has no
  ``thumbnail:`` frontmatter, the card uses the shared Qamomile logo.
  Per-tag result cards reuse the same descriptions from the matching
  section ``index.md`` card.

The per-tag pages are picked up by mystmd via a
``- pattern: "tags/*.md"`` toc entry in ``myst.yml`` — the script
does **not** enumerate them in toc itself.

Run from anywhere::

    uv run python docs/scripts/build_doc_tags.py

The script is idempotent: re-running it produces no diff if nothing
changed. Release notes (``docs/<lang>/release_notes/``) are deliberately
out of scope and never scanned.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from html import escape
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# Allow build.sh to point the script at a build-dir copy of ``docs/`` via
# an env var, so auto-managed content (chip blocks, browse-by-tag
# clouds, per-tag pages) is injected into a gitignored scratch tree at
# build time instead of being committed alongside the hand-written
# source. Falls back to the in-repo ``docs/`` so contributors can still
# run the script ad hoc against a working tree.
DOCS_ROOT = Path(os.environ.get("DOCS_ROOT_OVERRIDE", REPO_ROOT / "docs")).resolve()

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
ALLOWED_TAGS: frozenset[str] = frozenset(
    {
        # Section (1:1 with directory layout — every article carries the
        # tag that matches its containing section)
        "tutorial",
        "algorithm",
        "usage",
        "integration",
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
        "primitive",
        # Technique
        "circuit-compilation",
        "encoding",
        # Other
        "resource-estimation",
    }
)

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
    thumbnail: str | None
    py_path: Path  # absolute path to the source .py
    card_body: str | None = None  # section index card body, if available


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
        fm: dict[str, Any] = yaml.safe_load(yaml_block) or {}
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
        # ``startswith("# ")`` already excludes ``"## "`` etc. since the
        # second char must be a space; no extra ``startswith("##")`` guard
        # is needed.
        if stripped.startswith("# "):
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
        fm.get("title") or _extract_h1(body) or py_path.stem.replace("_", " ").title()
    )
    raw_tags: Any = fm.get("tags") or []
    if not isinstance(raw_tags, list):
        return None
    tags = tuple(str(t) for t in raw_tags)
    raw_thumbnail: Any = fm.get("thumbnail")
    thumbnail = str(raw_thumbnail) if raw_thumbnail else None
    return Article(
        section=section,
        slug=py_path.stem,
        title=title,
        tags=tags,
        thumbnail=thumbnail,
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


def _markdown_link_text(text: str) -> str:
    """Escape ``text`` for use as Markdown link text.

    Only bracket characters need escaping for the generated article
    titles we place inside ``[text](href)`` links. Other Markdown
    affordances in hand-written card headers, such as ``**bold**``, are
    intentionally left alone so existing header emphasis survives.
    """
    return text.replace("[", r"\[").replace("]", r"\]")


def _card_header_link_target(header: str) -> str | None:
    """Return the target from a simple Markdown link header, if present."""
    match = re.search(r"\]\((?P<href>[^)]+)\)\s*$", header)
    if match is None:
        return None
    return match.group("href")


def _linked_card_header(header: str, href: str) -> str:
    """Render ``header`` as a Markdown link unless it is already linked."""
    if _card_header_link_target(header) is not None:
        return header
    return f"[{_markdown_link_text(header)}]({href})"


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
        chip_line = " ".join(_chip_from_tags_dir(t) for t in sorted(tag_map))
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
    parts.append(str(strings["tag_page_lead"]).format(tag=tag, count=len(tag_articles)))
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
        parts.append("::::{grid} 1 1 1 1")
        parts.append("")
        for a in section_articles:
            parts.append(
                _render_article_card(
                    a,
                    f"../{a.section}/{a.slug}.ipynb",
                    _chip_from_tags_dir,
                )
            )
            parts.append("")
        parts.append("::::")
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
    return f"{CHIP_BEGIN}\n# **{strings['tags_label']}:** {chips}\n{CHIP_END}"


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
        return _write_if_changed(article.py_path, new_text)

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
        "".join(cell_lines[: h1_idx + 1]) + canonical + "".join(cell_lines[after:])
    )
    new_text = text[:cell_start] + new_cell + text[cell_end:]
    return _write_if_changed(article.py_path, new_text)


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
    return _write_if_changed(py_path, new_text)


# --------------------------------------------------------------------- #
# Section index.md browse-by-tag injection                              #
# --------------------------------------------------------------------- #

BROWSE_BEGIN = "<!-- BEGIN browse-by-tag -->"
BROWSE_END = "<!-- END browse-by-tag -->"

BROWSE_BLOCK_RE = re.compile(
    re.escape(BROWSE_BEGIN) + r"[\s\S]*?" + re.escape(BROWSE_END),
)
BROWSE_SECTION_RE = re.compile(
    r"\n*^## [^\n]+\n(?:[ \t]*\n)*"
    + re.escape(BROWSE_BEGIN)
    + r"[\s\S]*?"
    + re.escape(BROWSE_END)
    + r"\n*",
    re.MULTILINE,
)
GRID_DIRECTIVE_RE = re.compile(r"^:{3,}\{grid\}(?:\s|$)", re.MULTILINE)
H2_RE = re.compile(r"^## ", re.MULTILINE)

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

    Each tag is placed in its closest non-empty bucket — currently
    ``same`` (article in this section) or ``cousin`` (article elsewhere).
    The count shown is the number of articles inside that bucket;
    articles that fall into farther buckets do not contribute to the
    displayed count, since the tag is suppressed from those buckets.
    (When nested sections come back, ``descendant`` / ``ancestor`` slot
    in here — see ``_BUCKET_ORDER``.)
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
            1
            for a in articles
            if _classify_for_index(a.section, index_section) == closest
        )
        bucketed[closest][tag] = count

    lines: list[str] = []
    for bucket in _BUCKET_ORDER:
        tag_counts = bucketed[bucket]
        if not tag_counts:
            continue
        chip_line = " ".join(_chip_from_section(t) for t in sorted(tag_counts))
        lines.append(f"**{bucket_labels[bucket]}:** {chip_line}")
    return "\n\n".join(lines)


def _strip_browse_by_tag_section(text: str) -> str:
    """Remove an existing auto-managed browse-by-tag section from ``text``."""
    stripped, count = BROWSE_SECTION_RE.subn("\n\n", text, count=1)
    if count:
        return stripped
    return BROWSE_BLOCK_RE.sub("", text, count=1)


def _browse_by_tag_insert_at(text: str) -> int | None:
    """Return the insertion offset for the section browse-by-tag block.

    Prefer the first article card grid so the tag browser appears before
    every article card. If a section still has an ``## All articles`` style
    heading before that grid, place the browser before the heading instead.
    """
    grid_match = GRID_DIRECTIVE_RE.search(text)
    h2_match = H2_RE.search(text)
    if grid_match is not None and (
        h2_match is None or h2_match.start() > grid_match.start()
    ):
        return grid_match.start()
    if h2_match is not None:
        return h2_match.start()
    return None


def _inject_browse_by_tag(
    index_path: Path,
    index_section: str,
    tag_map: dict[str, list[Article]],
    strings: dict[str, object],
) -> Path | None:
    """Inject the auto-managed browse-by-tag block into a section index.

    Existing generated sections are stripped first, then a fresh section
    (heading + sentinels + chip cloud) is inserted before the article
    card grid. This is the supported flow for the build-dir model:
    contributors keep the committed ``index.md`` free of any
    browse-by-tag boilerplate, and ``build_doc_tags.py`` materialises
    the section inside the ``_build_src/`` copy.

    Returns the path if the file was modified, otherwise ``None``.
    """
    if not index_path.is_file():
        return None
    text = index_path.read_text(encoding="utf-8")
    block_body = _render_browse_by_tag_block(tag_map, index_section, strings)
    heading = str(strings["browse_by_tag"])
    section_md = f"## {heading}\n\n{BROWSE_BEGIN}\n{block_body}\n{BROWSE_END}\n\n"

    stripped_text = _strip_browse_by_tag_section(text).strip()
    insert_at = _browse_by_tag_insert_at(stripped_text)
    if insert_at is None:
        new_text = stripped_text.rstrip() + "\n\n" + section_md
    else:
        prefix = stripped_text[:insert_at].rstrip()
        suffix = stripped_text[insert_at:].lstrip()
        if prefix:
            new_text = f"{prefix}\n\n{section_md}{suffix}"
        else:
            new_text = f"{section_md}{suffix}"
    return _write_if_changed(index_path, new_text.rstrip() + "\n")


# --------------------------------------------------------------------- #
# Section index.md card metadata injection                              #
# --------------------------------------------------------------------- #

CARD_THUMB_BEGIN = "<!-- BEGIN auto-card-thumbnail -->"
CARD_THUMB_END = "<!-- END auto-card-thumbnail -->"
CARD_TAGS_BEGIN = "<!-- BEGIN auto-card-tags -->"
CARD_TAGS_END = "<!-- END auto-card-tags -->"
DEFAULT_CARD_THUMBNAIL = "../../assets/qamomile_logo.png"

CARD_DIRECTIVE_RE = re.compile(
    r"(?ms)^:::\{card\}\n(?P<body>.*?)^:::\s*$",
)

CARD_AUTO_BLOCK_RE = re.compile(
    r"\n?(?:"
    + re.escape(CARD_THUMB_BEGIN)
    + r"[\s\S]*?"
    + re.escape(CARD_THUMB_END)
    + r"|"
    + re.escape(CARD_TAGS_BEGIN)
    + r"[\s\S]*?"
    + re.escape(CARD_TAGS_END)
    + r")\n?",
    re.MULTILINE,
)

CARD_INLINE_CODE_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")


def _card_link_slug(link: str) -> str | None:
    """Resolve a card ``:link:`` target to a sibling article slug.

    Returns ``None`` for external links, empty links, or index-like
    targets that do not represent an article in the current section.
    """
    cleaned = link.strip()
    if (
        not cleaned
        or cleaned.startswith("//")
        or re.match(r"^[a-z][a-z0-9+.-]*:", cleaned, re.IGNORECASE)
    ):
        return None
    cleaned = cleaned.split("#", 1)[0].split("?", 1)[0].strip("/")
    if not cleaned:
        return None
    name = Path(cleaned).name
    if name in {"index", "index.md", "index.ipynb"}:
        return None
    return Path(name).stem


def _card_thumbnail_html(article: Article) -> str:
    """Render the thumbnail slot for a section index card."""
    src = escape(article.thumbnail or DEFAULT_CARD_THUMBNAIL, quote=True)
    if article.thumbnail:
        alt = escape(f"{article.title} thumbnail", quote=True)
    else:
        alt = ""
    extra_class = "" if article.thumbnail else " qamomile-section-card-thumb-default"
    inner = (
        f'<img class="qamomile-section-card-thumb{extra_class}" '
        f'src="{src}" alt="{alt}">'
    )
    return (
        f"{CARD_THUMB_BEGIN}\n"
        '<div class="qamomile-section-card-thumb-wrap">\n'
        f"{inner}\n"
        "</div>\n"
        f"{CARD_THUMB_END}"
    )


def _card_tags_html(
    article: Article,
    chip_renderer: Callable[[str], str],
) -> str:
    """Render clickable tag chips for an article card."""
    chips = " ".join(chip_renderer(t) for t in article.tags)
    return (
        f"{CARD_TAGS_BEGIN}\n"
        f'<div class="qamomile-section-card-tags">{chips}</div>\n'
        f"{CARD_TAGS_END}"
    )


def _render_article_card(
    article: Article,
    article_href: str,
    chip_renderer: Callable[[str], str],
) -> str:
    """Render one generated article card.

    The card intentionally does not use the MyST ``:link:`` option:
    article navigation lives on the card title, leaving the tag chips as
    normal independent links.
    """
    title = _markdown_link_text(article.title)
    lines = [
        ":::{card}",
        f":header: [**{title}**]({article_href})",
        _card_thumbnail_html(article),
    ]
    if article.card_body:
        lines.extend(["", article.card_body])
    lines.extend(["", _card_tags_html(article, chip_renderer), ":::"])
    return "\n".join(lines)


def _extract_card_link(body: str) -> str | None:
    """Return a card's article link from ``:link:`` or linked header metadata."""
    link_match = re.search(r"^:link:\s*(?P<link>\S+)\s*$", body, re.MULTILINE)
    if link_match is not None:
        return link_match.group("link")
    header_match = re.search(r"^:header:\s*(?P<header>.*?)\s*$", body, re.MULTILINE)
    if header_match is None:
        return None
    return _card_header_link_target(header_match.group("header"))


def _rewrite_card_options(
    option_lines: list[str],
    article_href: str,
) -> list[str]:
    """Rewrite card options so the title, not the whole card, is linked."""
    rewritten: list[str] = []
    for line in option_lines:
        if re.match(r"^:link(?:-type)?:", line):
            continue
        header_match = re.match(r"^:header:\s*(?P<header>.*?)\s*$", line)
        if header_match is not None:
            header = _linked_card_header(header_match.group("header"), article_href)
            rewritten.append(f":header: {header}")
        else:
            rewritten.append(line)
    return rewritten


def _render_card_body_content(content: str) -> str:
    """Render Markdown-only affordances unsupported in MyST card bodies."""

    def render_inline_code(match: re.Match[str]) -> str:
        return f"<code>{escape(match.group(1), quote=False)}</code>"

    return CARD_INLINE_CODE_RE.sub(render_inline_code, content)


def _extract_section_card_bodies(index_path: Path) -> dict[str, str]:
    """Extract hand-written card body text from a section ``index.md``.

    The returned mapping is keyed by article slug. Auto-managed
    thumbnail/tag blocks are stripped first so the extracted body stays
    stable when this script is re-run against an already generated tree.
    Inline code is rendered the same way as section card enhancement so
    tag result cards and section cards display identical summaries.
    """
    if not index_path.is_file():
        return {}
    text = index_path.read_text(encoding="utf-8")
    bodies: dict[str, str] = {}
    for match in CARD_DIRECTIVE_RE.finditer(text):
        body = CARD_AUTO_BLOCK_RE.sub("\n", match.group("body")).strip("\n")
        article_href = _extract_card_link(body)
        if article_href is None:
            continue
        slug = _card_link_slug(article_href)
        if slug is None:
            continue
        lines = body.splitlines()
        split_at = 0
        while split_at < len(lines) and lines[split_at].startswith(":"):
            split_at += 1
        content = _render_card_body_content("\n".join(lines[split_at:]).strip("\n"))
        if content:
            bodies[slug] = content
    return bodies


def _attach_section_card_bodies(lang: str, articles: list[Article]) -> list[Article]:
    """Attach section-index card bodies to matching articles."""
    body_maps = {
        section: _extract_section_card_bodies(DOCS_ROOT / lang / section / "index.md")
        for section in SECTIONS
    }
    enriched: list[Article] = []
    for article in articles:
        card_body = body_maps.get(article.section, {}).get(article.slug)
        enriched.append(replace(article, card_body=card_body))
    return enriched


def _enhance_card_block(
    match: re.Match[str],
    articles_by_slug: dict[str, Article],
) -> str:
    """Insert thumbnail and tag metadata into one card directive."""
    body = match.group("body")
    article_href = _extract_card_link(body)
    if article_href is None:
        return match.group(0)
    slug = _card_link_slug(article_href)
    if slug is None or slug not in articles_by_slug:
        return match.group(0)

    body = CARD_AUTO_BLOCK_RE.sub("\n", body).strip("\n")
    lines = body.splitlines()
    split_at = 0
    while split_at < len(lines) and lines[split_at].startswith(":"):
        split_at += 1

    option_lines = _rewrite_card_options(lines[:split_at], article_href)
    content = _render_card_body_content("\n".join(lines[split_at:]).strip("\n"))
    article = articles_by_slug[slug]
    enhanced_body = (
        "\n".join(option_lines)
        + "\n"
        + _card_thumbnail_html(article)
        + "\n"
        + content
        + "\n\n"
        + _card_tags_html(article, _chip_from_section)
    ).rstrip()
    return f":::{{card}}\n{enhanced_body}\n:::"


def _inject_section_card_metadata(
    index_path: Path,
    section_articles: list[Article],
) -> Path | None:
    """Add tag chips and thumbnail slots to cards in a section ``index.md``."""
    if not index_path.is_file():
        return None
    articles_by_slug = {a.slug: a for a in section_articles}
    if not articles_by_slug:
        return None
    text = index_path.read_text(encoding="utf-8")
    new_text = CARD_DIRECTIVE_RE.sub(
        lambda m: _enhance_card_block(m, articles_by_slug),
        text,
    )
    return _write_if_changed(index_path, new_text)


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
# Generic write helpers                                                 #
# --------------------------------------------------------------------- #


def _write_if_changed(path: Path, content: str) -> Path | None:
    """Write ``content`` to ``path`` only if it differs from the existing file.

    Returns ``path`` when a write occurred, otherwise ``None``. This is the
    single skip-if-equal helper for every writer in the script — chip
    injection, chip stripping, browse-by-tag injection, tag pages, and the
    tags index — so the audit log only mentions paths that actually changed.
    """
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return None
    path.write_text(content, encoding="utf-8")
    return path


# --------------------------------------------------------------------- #
# Per-locale build                                                      #
# --------------------------------------------------------------------- #


def _build_for_locale(lang: str) -> tuple[list[Path], list[Path]]:
    """Generate everything for one locale.

    Returns ``(written, removed)`` paths for caller logging. When the
    locale's root directory is absent under ``DOCS_ROOT`` (e.g. when
    ``build.sh build-en`` set up only ``_build_src/en/``), returns
    ``([], [])`` so single-locale build pipelines do not crash trying
    to write into a non-existent ``<lang>/tags/`` directory.
    """
    if lang not in STRINGS:
        raise ValueError(f"unknown locale {lang!r}")
    if not (DOCS_ROOT / lang).is_dir():
        return [], []
    strings = STRINGS[lang]

    articles, untagged_paths = _walk_articles(lang)
    articles = _attach_section_card_bodies(lang, articles)
    tag_map = _tag_map(articles)
    all_tags = sorted(tag_map)
    articles_by_section: dict[str, list[Article]] = {}
    for article in articles:
        articles_by_section.setdefault(article.section, []).append(article)

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
    written_index = _write_if_changed(tags_index, _render_tags_index(tag_map, strings))
    if written_index is not None:
        written.append(written_index)

    for tag in all_tags:
        page = tags_dir / f"{tag}.md"
        rendered = _render_tag_page(tag, tag_map[tag], strings, all_tags)
        written_page = _write_if_changed(page, rendered)
        if written_page is not None:
            written.append(written_page)

    # 3. Enhance each section's hand-written card grid with auto-managed
    # tag chips and thumbnail slots. The source cards stay hand-written;
    # only the metadata surface is generated.
    for section in SECTIONS:
        index_path = DOCS_ROOT / lang / section / "index.md"
        modified = _inject_section_card_metadata(
            index_path,
            articles_by_section.get(section, []),
        )
        if modified is not None:
            written.append(modified)

    # 4. Inject (or refresh) the browse-by-tag block in each section's
    # hand-written index.md. Existing generated sections are moved if
    # needed so the chip cloud appears before the article cards.
    for section in SECTIONS:
        index_path = DOCS_ROOT / lang / section / "index.md"
        modified = _inject_browse_by_tag(index_path, section, tag_map, strings)
        if modified is not None:
            written.append(modified)

    # 5. Remove any leftover docs/<lang>/algorithm/tags/ from the old
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

    # myst.yml itself does not need rewriting — it carries a single
    # ``- pattern: "tags/*.md"`` toc entry that mystmd resolves at
    # build time, so the per-tag pages we just wrote are picked up
    # automatically without enumerating them in toc.

    return written, removed


def _audit_path(p: Path) -> str:
    """Format ``p`` as a repo-relative path when possible, else absolute.

    Used by ``main()``'s per-file audit lines so an ad-hoc invocation with
    ``DOCS_ROOT_OVERRIDE`` pointing outside the repo prints a usable path
    instead of crashing on ``Path.relative_to``.
    """
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def main() -> None:
    """Regenerate every locale and print a one-line audit per file."""
    for lang in ("en", "ja"):
        written, removed = _build_for_locale(lang)
        for p in written:
            print(f"wrote {_audit_path(p)}")
        for p in removed:
            print(f"removed {_audit_path(p)}")


if __name__ == "__main__":
    main()
