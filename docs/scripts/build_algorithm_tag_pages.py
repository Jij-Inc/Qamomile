"""Generate the algorithm tag index and per-tag landing pages.

This script is the source of truth for algorithm discovery UX: it reads
the MyST frontmatter (``title`` and ``tags``) embedded in the first
markdown cell of each ``docs/<lang>/algorithm/*.py`` jupytext file, then
writes out:

* ``docs/<lang>/algorithm/index.md`` — the algorithm landing page with
  clickable tag chips at the top and a full article list (each article
  card shows its own tag chips).
* ``docs/<lang>/algorithm/tags/<tag>.md`` — one page per tag, listing
  all articles that carry that tag.

Run from anywhere::

    uv run python docs/scripts/build_algorithm_tag_pages.py

The script is idempotent and safe to re-run; it fully overwrites the
generated files. It does NOT modify the algorithm ``.py`` sources.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_ROOT = REPO_ROOT / "docs"

# Locale-aware copy. Keep the taxonomy identical across locales; only
# display strings differ. Adding a new locale is a matter of adding an
# entry here.
STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "title": "Algorithms",
        "slug": "algorithm",
        "lead": (
            "Concrete quantum algorithm examples built with Qamomile. "
            "Click a tag below to filter, or browse all algorithms."
        ),
        "browse_by_tag": "Browse by tag",
        "all_algorithms": "All algorithms",
        "tags_label": "Tags",
        "tag_page_lead": "Algorithm examples tagged **`{tag}`** ({count}).",
        "tag_page_back": "← Back to all algorithms",
        "tag_page_title_fmt": "Tag: {tag}",
        "tag_page_heading_fmt": "`{tag}`",
    },
    "ja": {
        "title": "アルゴリズム",
        "slug": "algorithm",
        "lead": (
            "Qamomile で実装した具体的な量子アルゴリズム例です。"
            "下のタグをクリックして絞り込むか、全アルゴリズムから選んでください。"
        ),
        "browse_by_tag": "タグで探す",
        "all_algorithms": "全アルゴリズム",
        "tags_label": "タグ",
        "tag_page_lead": "**`{tag}`** タグが付いたアルゴリズム例 ({count} 件)。",
        "tag_page_back": "← 全アルゴリズムへ戻る",
        "tag_page_title_fmt": "タグ: {tag}",
        "tag_page_heading_fmt": "`{tag}`",
    },
}


@dataclass(frozen=True)
class Article:
    slug: str  # e.g. "qaoa_maxcut" — the filename without extension
    title: str
    tags: tuple[str, ...]
    summary: str  # First paragraph after the H1, if present


FRONTMATTER_RE = re.compile(
    r"^# ---\s*\n(?P<body>(?:#.*\n)+?)# ---\s*\n",
    re.MULTILINE,
)


def _extract_first_markdown_cell(py_text: str) -> str:
    """Return the body of the first ``# %% [markdown]`` cell as plain markdown."""
    marker = "# %% [markdown]\n"
    start = py_text.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    # Cell ends at the next "# %%" marker or EOF.
    end = py_text.find("\n# %%", start)
    if end == -1:
        end = len(py_text)
    lines = py_text[start:end].splitlines()
    # Strip leading "# " (or "#" on empty lines).
    out: list[str] = []
    for line in lines:
        if line.startswith("# "):
            out.append(line[2:])
        elif line == "#":
            out.append("")
        else:
            out.append(line)
    return "\n".join(out).strip()


def _parse_article_frontmatter(cell_md: str) -> tuple[dict, str]:
    """Split the first cell into (frontmatter_dict, remaining_markdown)."""
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
    """Return the first non-H1 paragraph of the markdown body, collapsed."""
    lines = body_md.splitlines()
    # Skip the H1.
    i = 0
    while i < len(lines) and not lines[i].startswith("# "):
        i += 1
    if i < len(lines):
        i += 1  # past the H1
    # Skip blank lines.
    while i < len(lines) and not lines[i].strip():
        i += 1
    buf: list[str] = []
    while i < len(lines) and lines[i].strip():
        buf.append(lines[i].strip())
        i += 1
    # Collapse soft wraps into a single paragraph.
    para = " ".join(buf)
    # MyST links remain as-is; that's fine for the summary.
    return para


def _load_article(py_path: Path) -> Article | None:
    text = py_path.read_text(encoding="utf-8")
    cell = _extract_first_markdown_cell(text)
    fm, body = _parse_article_frontmatter(cell)
    if not fm:
        return None
    title = str(fm.get("title") or py_path.stem.replace("_", " ").title())
    raw_tags = fm.get("tags") or []
    tags = tuple(str(t) for t in raw_tags) if isinstance(raw_tags, list) else ()
    summary = _extract_summary(body)
    return Article(slug=py_path.stem, title=title, tags=tags, summary=summary)


def _collect_articles(algo_dir: Path) -> list[Article]:
    articles: list[Article] = []
    for py in sorted(algo_dir.glob("*.py")):
        art = _load_article(py)
        if art is not None:
            articles.append(art)
    return articles


def _tag_map(articles: Iterable[Article]) -> dict[str, list[Article]]:
    out: dict[str, list[Article]] = {}
    for a in articles:
        for t in a.tags:
            out.setdefault(t, []).append(a)
    # Stable article order per tag.
    for t in out:
        out[t].sort(key=lambda a: a.title)
    return out


def _render_tag_chip(tag: str) -> str:
    """Render one tag as a clickable chip linking to its tag page."""
    return f"[`{tag}`](tags/{tag}.md)"


def _render_tag_chip_from_tag_page(tag: str) -> str:
    """Render a tag chip when the caller is itself under algorithm/tags/."""
    return f"[`{tag}`](./{tag}.md)"


def _render_article_card(article: Article, strings: dict[str, str]) -> str:
    chips = " ".join(_render_tag_chip(t) for t in article.tags)
    summary = article.summary or ""
    return (
        f"### [{article.title}]({article.slug}.ipynb)\n"
        f"\n"
        f"**{strings['tags_label']}:** {chips}\n"
        f"\n"
        f"{summary}\n"
    )


def _render_index(
    articles: list[Article],
    tag_map: dict[str, list[Article]],
    strings: dict[str, str],
) -> str:
    parts: list[str] = []
    parts.append("---")
    parts.append(f"slug: {strings['slug']}")
    parts.append(f"title: {strings['title']}")
    parts.append("---")
    parts.append("")
    parts.append(f"# {strings['title']}")
    parts.append("")
    parts.append(strings["lead"])
    parts.append("")
    parts.append(f"## {strings['browse_by_tag']}")
    parts.append("")
    chip_line = " · ".join(
        f"{_render_tag_chip(t)} ({len(tag_map[t])})" for t in sorted(tag_map)
    )
    parts.append(chip_line)
    parts.append("")
    parts.append(f"## {strings['all_algorithms']}")
    parts.append("")
    for a in articles:
        parts.append(_render_article_card(a, strings))
    return "\n".join(parts).rstrip() + "\n"


def _render_tag_page(
    tag: str,
    tag_articles: list[Article],
    strings: dict[str, str],
    all_tags: list[str],
) -> str:
    parts: list[str] = []
    parts.append("---")
    # Quote the title: MyST/YAML rejects plain scalars containing ``: ``
    # or starting with backticks, so always wrap in double quotes.
    title_text = strings["tag_page_title_fmt"].format(tag=tag).replace('"', '\\"')
    parts.append(f'title: "{title_text}"')
    parts.append(f"tags: [{tag}]")
    parts.append("---")
    parts.append("")
    parts.append(f"# {strings['tag_page_heading_fmt'].format(tag=tag)}")
    parts.append("")
    parts.append(
        strings["tag_page_lead"].format(tag=tag, count=len(tag_articles))
    )
    parts.append("")
    parts.append(f"[{strings['tag_page_back']}](../index.md)")
    parts.append("")
    # Other tag chips for easy hopping.
    other_tags = [t for t in all_tags if t != tag]
    if other_tags:
        parts.append("---")
        parts.append("")
        parts.append(f"**{strings['browse_by_tag']}:** ")
        parts.append(" · ".join(_render_tag_chip_from_tag_page(t) for t in other_tags))
        parts.append("")
    parts.append("---")
    parts.append("")
    for a in tag_articles:
        chips = " ".join(
            _render_tag_chip_from_tag_page(t) for t in a.tags if t != tag
        )
        parts.append(f"### [{a.title}](../{a.slug}.ipynb)")
        parts.append("")
        if chips:
            parts.append(f"**{strings['tags_label']}:** {chips}")
            parts.append("")
        if a.summary:
            parts.append(a.summary)
            parts.append("")
    return "\n".join(parts).rstrip() + "\n"


TAGS_TOC_BEGIN = "# --- BEGIN algorithm tags (auto-generated) ---"
TAGS_TOC_END = "# --- END algorithm tags (auto-generated) ---"


def _render_tags_toc_block(all_tags: list[str]) -> str:
    """Render the tag children block for ``myst.yml`` between the sentinels."""
    lines = [
        TAGS_TOC_BEGIN,
        "        - title: Tags",
        "          hidden: true",
        "          children:",
    ]
    for tag in all_tags:
        lines.append(f"            - file: algorithm/tags/{tag}.md")
    lines.append("        " + TAGS_TOC_END)
    return "\n".join(lines)


def _update_myst_yml(lang: str, all_tags: list[str]) -> Path | None:
    """Rewrite the auto-managed Tags block in ``docs/<lang>/myst.yml``.

    Does nothing if the sentinel comments are absent — this keeps the
    script a no-op for YAML files that were set up manually and haven't
    been opted in to auto-management.
    """
    myst_path = DOCS_ROOT / lang / "myst.yml"
    if not myst_path.is_file():
        return None
    text = myst_path.read_text(encoding="utf-8")
    if TAGS_TOC_BEGIN not in text or TAGS_TOC_END not in text:
        return None
    pattern = re.compile(
        r"[ \t]*"
        + re.escape(TAGS_TOC_BEGIN)
        + r"[\s\S]*?"
        + re.escape(TAGS_TOC_END),
        re.MULTILINE,
    )
    new_block = _render_tags_toc_block(all_tags)
    new_text = pattern.sub(new_block, text, count=1)
    if new_text != text:
        myst_path.write_text(new_text, encoding="utf-8")
    return myst_path


def _build_for_locale(lang: str) -> list[Path]:
    """Generate index + tag pages for one locale; return written paths."""
    algo_dir = DOCS_ROOT / lang / "algorithm"
    if not algo_dir.is_dir():
        return []
    strings = STRINGS[lang]
    articles = _collect_articles(algo_dir)
    articles.sort(key=lambda a: a.title)
    tag_map = _tag_map(articles)
    all_tags = sorted(tag_map)

    written: list[Path] = []
    index_path = algo_dir / "index.md"
    index_path.write_text(
        _render_index(articles, tag_map, strings), encoding="utf-8"
    )
    written.append(index_path)

    tags_dir = algo_dir / "tags"
    tags_dir.mkdir(exist_ok=True)
    # Clean stale tag files: remove any .md not in the current tag set.
    current_files = {tags_dir / f"{t}.md" for t in all_tags}
    for existing in tags_dir.glob("*.md"):
        if existing not in current_files:
            existing.unlink()

    for tag in all_tags:
        page = tags_dir / f"{tag}.md"
        page.write_text(
            _render_tag_page(tag, tag_map[tag], strings, all_tags),
            encoding="utf-8",
        )
        written.append(page)

    myst_updated = _update_myst_yml(lang, all_tags)
    if myst_updated is not None:
        written.append(myst_updated)

    return written


def main() -> None:
    for lang in ("en", "ja"):
        written = _build_for_locale(lang)
        for p in written:
            rel = p.relative_to(REPO_ROOT)
            print(f"wrote {rel}")


if __name__ == "__main__":
    main()
