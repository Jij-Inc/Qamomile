#!/usr/bin/env python3
"""Post-build HTML patches for the Qamomile docs.

Originally just a colab-launch script-tag injector; the script has
since absorbed four more passes that all run against the same set of
``docs/<lang>/_build/html/*.html`` (and ``*.json``) files, so they live
together for a single rglob walk.

The five passes:

1. Rewrite build-dir paths in GitHub URLs (HTML + JSON). ``./build.sh
   build`` runs mystmd from ``docs/_build_src/<lang>/`` so the
   committed source tree stays free of auto-managed injections;
   mystmd then derives the project-relative source path as
   ``docs/_build_src/<lang>/<section>/<slug>.ipynb`` and bakes that
   into the "Edit on GitHub" / "View source" anchors in **both** the
   HTML page header and the per-page JSON data layer the SPA hydrates
   from. Those URLs would 404 on ``main`` (``_build_src/`` is
   gitignored) and would also cause ``colab-launch.js`` to fail its
   ``^docs/(en|ja)/`` path check, dropping the Colab button after the
   first SPA navigation. We rewrite them to ``docs/<lang>/...``.

2. Inline ``custom-theme.css`` in ``<head>``. mystmd bundles
   ``docs/assets/custom-theme.css`` into the LAST
   ``<link rel="stylesheet" href="/myst-theme.css">`` of the page;
   that file arrives later than the 160 KB ``app-*.css``, so on first
   paint the browser renders the page through Tailwind's default blue
   palette (links, buttons, focus rings) and snaps into the warm-amber
   theme only once ``myst-theme.css`` finishes downloading. The
   noticeable flash is the ``.text-blue-* / .bg-blue-* /
   .border-blue-*`` overrides repainting nearly every nav element, so
   inlining only the chip subsection isn't enough — we re-emit the
   whole ~6.5 KB stylesheet as an inline ``<style>`` early in
   ``<head>`` so every theme rule is parsed before any external
   stylesheet has arrived.

3. Inject a synchronous theme-init ``<script>`` at the start of
   ``<head>``. mystmd persists ``light``/``dark`` in
   ``localStorage["myst:theme"]``, but its React-driven theme
   detection only runs after hydration; meanwhile the SSR HTML ships
   with ``<html class="">`` and ``<body>`` carries
   ``transition-colors duration-500``, so dark-mode users see the
   light theme paint and then animate to dark over half a second.
   Setting the class synchronously before any external stylesheet has
   loaded eliminates that animation.

4. Inject the colab-launch ``<script>`` tag right before ``</head>``
   on every HTML. The script then renders the "Open in Colab" button
   client-side, gated to ``.ipynb``-derived pages by the path check
   that pass 1 protects.

5. Inject the lightbox ``<script>`` tag right before ``</head>`` on
   every HTML. The script binds a click handler to every cell-output
   ``<img>`` / ``<svg>`` and opens an overlay modal that shows the
   figure at its natural size; the user can then use the browser's
   native Ctrl/Cmd + scroll to zoom further. SPA-aware via a
   MutationObserver so client-side navigations also pick up freshly
   hydrated outputs.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

SCRIPT_TAG_ID = "qamomile-colab-launch-script"
SCRIPT_FILE_NAME = "colab-launch.js"

# Companion lightbox helper. Same copy/inject mechanism as the colab
# launch script: the source lives at docs/assets/lightbox.js, gets
# copied into each language's _build/html/build/, and a defer-loaded
# <script> tag with this id is inserted before </head>. The helper
# turns every <img>/<svg> in a cell output into a click-to-zoom modal
# so wide circuit diagrams stay inspectable on RTD without per-page
# wrapper boilerplate.
LIGHTBOX_TAG_ID = "qamomile-lightbox-script"
LIGHTBOX_FILE_NAME = "lightbox.js"

# Match ``<head>`` even if it grows attributes (e.g. ``<head prefix="og: ...">``).
# We anchor on the literal ``<head`` followed by a word boundary so we don't
# also match ``<header>``, then accept any non-``>`` characters up to the
# closing ``>``. ``re.IGNORECASE`` because HTML tag names are
# case-insensitive in spec, even though mystmd lowercases its output today.
HEAD_OPEN_RE = re.compile(r"<head\b[^>]*>", re.IGNORECASE)

# Optional ``<meta charset=...>`` declaration. The HTML5 spec requires
# the charset declaration to appear within the first 1024 bytes of the
# document — pushing 6.5 KB of inline <style> ahead of it would make
# servers without an explicit ``Content-Type: text/html; charset=utf-8``
# header (some RTD configurations, some local previews) fall back to
# guessing the encoding, which corrupts ja pages. So when a meta-charset
# is present, we insert AFTER it; otherwise (theoretical fallback for
# weird HTML) we keep the old behaviour and insert right after <head>.
META_CHARSET_RE = re.compile(
    r'<meta\s[^>]*charset\s*=\s*["\']?[^"\'\s>]+["\']?[^>]*/?>',
    re.IGNORECASE,
)

# Synchronous theme-init + transition-suppression script. Two
# entangled FOUC sources are addressed in one ``<script>`` block:
#
# (a) Dark-mode flash. mystmd persists the user's theme choice in
#     ``localStorage["myst:theme"]`` (values: "light" / "dark") and
#     falls back to ``window.matchMedia("(prefers-color-scheme:
#     dark)")`` when nothing is stored. The SPA's React tree sets the
#     matching class on <html> AFTER hydration, but the SSR HTML
#     ships with ``<html class="">`` (light) and <body> carries
#     ``transition-colors duration-500``, so dark-mode users see the
#     light theme paint first and then animate to dark over 500 ms
#     once the JS bundle hydrates. Setting ``class="dark"`` on
#     <html> synchronously here, before any external stylesheet has
#     even started loading, eliminates that animation.
#
# (b) "Non-theme" transition flashes. Even after (a), light-mode
#     users still see a faint color flicker on every page load.
#     Cause: mystmd's React hydration toggles many other classes
#     (navbar shadow, sidebar collapse state, focus rings,
#     skip-link visibility, …) and Tailwind's ``transition-colors``
#     / ``transition-shadow`` / ``transition-all`` utilities animate
#     each toggle over 200–700 ms. The user perceives that as a
#     persistent FOUC even when the theme itself is stable. We can't
#     surgically silence each utility, so we adopt the standard
#     ``preload-class`` pattern: add ``html.qamomile-preloading`` at
#     the very start of head, the inline <style> below contains a
#     blanket ``transition: none !important`` rule scoped to that
#     class, and after ``window.load`` we observe DOM mutations and
#     remove the class once 250ms passes without any mutation, i.e.
#     once React hydration has quiesced. Bounded by a 5-second
#     hard-cap after ``load`` and a 10-second wall-clock safety net
#     so a runaway page can never stay hidden forever. See the
#     comment on the reveal logic inside ``THEME_INIT_SCRIPT`` below
#     for the trade-offs that led to this idle-detection scheme.
THEME_INIT_SCRIPT_ID = "qamomile-theme-init"
THEME_INIT_SCRIPT = (
    "(function(){"
    "var d=document.documentElement;"
    'd.classList.add("qamomile-preloading");'
    # Theme detection — must match mystmd's logic exactly so React
    # hydration doesn't flip the class right after our synchronous
    # init runs. mystmd's bundle uses:
    #
    #   matchMedia("(prefers-color-scheme: light)").matches
    #     ? "light"
    #     : "dark"
    #
    # i.e. the system fallback queries "light" (not "dark") and
    # treats any non-match — including the "no preference" case — as
    # dark. If we instead query "(prefers-color-scheme: dark)", a
    # user with no OS preference would have us add NO class (light)
    # while React hydrates with dark, triggering body's
    # ``transition-colors duration-500`` animation. We mirror mystmd
    # bit-for-bit: explicit "dark" / "light" override, otherwise
    # invert ``matches("(prefers-color-scheme: light)")``.
    "try{"
    'var stored=localStorage.getItem("myst:theme");'
    'var dark=stored==="dark"||(stored!=="light"&&'
    '!window.matchMedia("(prefers-color-scheme: light)").matches);'
    'if(dark)d.classList.add("dark");'
    "}catch(e){}"
    # Reveal: lift the preloading class once hydration has settled.
    #
    # Older strategy raced three triggers: ``window.load + 2 RAF``, a
    # 2-second safety timeout, and the first user-driven event
    # (mousemove / click / keydown / touch). All three turned out to be
    # unreliable on pages whose React hydration kept mutating the DOM
    # past the ``load`` event:
    #   - ``load + 2 RAF`` only buys ~32ms after ``load``; on heavy
    #     pages (mottonen, the section landings) React is still doing
    #     hydration-mismatch recovery (client-rendering whole subtrees)
    #     well past that point, so the guard lifts mid-storm and the
    #     user sees cell outputs flicker into place.
    #   - The 2-second safety timeout measures from script start, not
    #     from ``load``; on the same heavy pages ``load`` itself fires
    #     several seconds after the script runs, so the timeout always
    #     fires first and never actually saves the user from anything.
    #   - The user-driven-event trigger is hostile by design: any mouse
    #     movement during the multi-second hidden phase races the guard
    #     down and exposes the in-progress hydration. This is exactly
    #     the ちらつき regression contributors keep reporting.
    #
    # New strategy: after ``load``, install a MutationObserver on
    # documentElement (subtree, attributes, characterData) and reveal
    # once 250ms passes without any mutation — that is, hydration has
    # quiesced. Two safety nets bound the worst case so a misbehaving
    # page can never stay blank forever:
    #   1. ``hardCap``: 5 seconds after ``load``. By that point any
    #      reasonable hydration-mismatch recovery has finished; a page
    #      that's still mutating after 5s past ``load`` is in a runaway
    #      state and the user is better served by seeing it than by an
    #      indefinitely hidden body.
    #   2. ``wallClock``: 10 seconds from script start. Covers the
    #      degenerate case where ``load`` never fires at all (broken
    #      network, blocked resource, etc.). Independent of the first
    #      net because if ``load`` never fires, the inner listener
    #      never runs and ``hardCap`` is never installed.
    "var revealed=false;"
    "function reveal(){"
    "if(revealed)return;"
    "revealed=true;"
    'd.classList.remove("qamomile-preloading");'
    "}"
    "setTimeout(reveal,10000);"
    'window.addEventListener("load",function(){'
    "var lastMut=performance.now();"
    "var hardCap=setTimeout(reveal,5000);"
    "var obs=new MutationObserver(function(){"
    "lastMut=performance.now();"
    "});"
    "obs.observe(d,{"
    "childList:true,"
    "subtree:true,"
    "attributes:true,"
    "characterData:true"
    "});"
    "function checkIdle(){"
    "if(revealed){obs.disconnect();return;}"
    "if(performance.now()-lastMut>=250){"
    "obs.disconnect();"
    "clearTimeout(hardCap);"
    "reveal();"
    "}else{"
    "requestAnimationFrame(checkIdle);"
    "}"
    "}"
    # Skip one rAF before starting the idle probe so React has a
    # chance to start its hydration work — otherwise the very first
    # ``checkIdle`` call sees ``performance.now() - lastMut`` already
    # well above 250ms (because we haven't observed any mutation yet)
    # and we'd reveal before hydration even begins.
    "requestAnimationFrame(function(){"
    "lastMut=performance.now();"
    "requestAnimationFrame(checkIdle);"
    "});"
    "});"
    "})();"
)

# Preload guard: lives inside the inline theme <style> block (see
# :func:`inline_theme_css`) so the rule is parsed in the same first-
# style-pass that interprets the rest of the theme overrides. The
# ``html.qamomile-preloading`` class is set by THEME_INIT_SCRIPT a
# moment earlier (inline <script> at the start of <head>), so the
# first paint already has these rules in effect. Once
# ``qamomile-preloading`` is removed (post-load + 2 RAF, or a 2-second
# safety timeout, whichever comes first), the rules stop matching and
# normal styling / transitions resume.
#
# Two layers of suppression:
#
# 1. ``html.qamomile-preloading body { visibility: hidden !important; }``
#    hides the body until the page is fully wired up. Any remaining
#    FOUC source — late-arriving CDN stylesheet, React hydration
#    re-render, font swap, layout shift from JS-injected colab
#    button — happens off-screen. When the class is removed, the
#    body becomes visible in its already-final state. This is the
#    standard "flash of content" guard used by most React docs sites.
#
# 2. ``transition: none !important`` on every element while preloading
#    is a backstop in case anything DOES paint before body is
#    revealed (e.g. browsers occasionally render with ``visibility:
#    hidden`` lifted briefly during interleaved compositing). With
#    transitions silenced, no animation can become visible even if
#    that happens.
#
# ``!important`` is required on both rules because Tailwind's
# ``transition-colors`` etc. set the relevant properties directly on
# each element with normal specificity.
PRELOAD_GUARD_CSS = (
    "html.qamomile-preloading body{visibility:hidden!important;}"
    "html.qamomile-preloading,"
    "html.qamomile-preloading *,"
    "html.qamomile-preloading *::before,"
    "html.qamomile-preloading *::after"
    "{transition:none!important;animation-duration:0s!important;}"
)

# Inline-CSS pass: avoids FOUC across the whole theme override. mystmd
# bundles docs/assets/custom-theme.css into the last
# <link rel="stylesheet" href="/myst-theme.css"> in <head>. That file
# arrives after the 160 KB app-*.css, so on first paint the browser
# applies the un-themed Tailwind palette (blue links, blue buttons,
# default-styled <a> tags), and only after myst-theme.css finishes
# downloading does the page snap into the warm-amber theme — most
# visibly because custom-theme.css's `.text-blue-* / .bg-blue-* /
# .border-blue-*` overrides re-paint nearly every navigational element.
# Re-emitting the entire stylesheet as an inline <style> block early in
# <head> means the rules are parsed during HTML parse, before any
# external stylesheet has arrived, so first paint already uses the warm
# theme; the external myst-theme.css that follows is then a no-op
# duplicate (identical rules, content cascade ties broken by source
# order to the same outcome).
THEME_CSS_STYLE_ID = "qamomile-theme-css-inline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject Colab launcher script into docs build output."
    )
    parser.add_argument(
        "languages",
        nargs="+",
        choices=("en", "ja"),
        help="Language build directories to patch.",
    )
    return parser.parse_args()


def _read_theme_css(docs_root: Path) -> str:
    """Return ``docs/assets/custom-theme.css`` plus the preload guard rule.

    Two pieces are concatenated into the single inline ``<style>``
    block:

    1. The full content of ``custom-theme.css`` (~6.5 KB). The whole
       file is inlined rather than just one section because the
       biggest FOUC contributor is the ``.text-blue-* / .bg-blue-* /
       .border-blue-*`` overrides that re-paint nearly every nav /
       link / button on the page when ``myst-theme.css`` finally
       arrives — inlining only a single subsection (e.g. just chips)
       leaves the bigger flash unaddressed.
    2. ``PRELOAD_GUARD_CSS`` — the blanket
       ``html.qamomile-preloading * { transition: none !important; }``
       rule that suppresses every transition / animation while the
       page is hydrating, paired with the matching class added by
       ``THEME_INIT_SCRIPT`` synchronously at the start of <head>.

    Raises:
        RuntimeError: when ``custom-theme.css`` is missing — fail loud
            so a refactor of ``docs/assets/`` doesn't silently start
            shipping un-themed builds.
    """
    css_path = docs_root / "assets" / "custom-theme.css"
    if not css_path.exists():
        raise RuntimeError(f"Missing source CSS: {css_path}")
    return css_path.read_text(encoding="utf-8").strip() + "\n" + PRELOAD_GUARD_CSS


def inline_theme_css(html_path: Path, css_text: str) -> bool:
    """Inline ``custom-theme.css`` as a ``<style>`` block in ``<head>``.

    Inserted right after the opening ``<head>`` tag — earlier than the
    external ``<link rel="stylesheet">`` entries — so the rules apply
    on first paint and the page never flashes through Tailwind's
    default blue palette before snapping into the warm-amber theme.
    The block carries a stable ``id`` so the function is idempotent
    across re-runs of the patcher.

    Args:
        html_path: Path to the HTML file under
            ``docs/<lang>/_build/html/``.
        css_text: The full theme-CSS payload (from
            :func:`_read_theme_css`).

    Returns:
        True when the file was patched. False if the page already
        carries the inline ``<style>`` (idempotent re-run) or has no
        ``<head>`` to anchor against (e.g. a 404 stub).
    """
    content = html_path.read_text(encoding="utf-8")
    if f'id="{THEME_CSS_STYLE_ID}"' in content:
        return False
    insert_at = _head_insertion_point(content)
    if insert_at is None:
        return False
    style_block = f'<style id="{THEME_CSS_STYLE_ID}">{css_text}</style>'
    new = content[:insert_at] + style_block + content[insert_at:]
    html_path.write_text(new, encoding="utf-8")
    return True


def _head_insertion_point(content: str) -> int | None:
    """Return the byte offset to insert head-level patches at.

    Prefers the position immediately after a ``<meta charset=...>`` tag
    when one exists, falling back to right after the opening ``<head>``
    tag otherwise. Reason for the preference: HTML5 requires the
    charset declaration in the first 1024 bytes; inserting our 6.5 KB
    of inline <style> before it would push it past that window and
    break encoding sniffing on servers that don't send an explicit
    Content-Type charset header (notably an issue for ja pages).

    Returns ``None`` when neither anchor is found (e.g. a 404 stub
    with no <head>).
    """
    head_match = HEAD_OPEN_RE.search(content)
    if head_match is None:
        return None
    charset_match = META_CHARSET_RE.search(content, head_match.end())
    # Don't trust a charset tag that lives outside the head we just
    # matched — bail back to <head>'s end if it appears suspiciously
    # late (post-</head>). Cheap heuristic: cap the search window at
    # the first </head> closing tag.
    head_close = content.find("</head>", head_match.end())
    if charset_match is not None and (
        head_close < 0 or charset_match.end() <= head_close
    ):
        return charset_match.end()
    return head_match.end()


def init_theme_script(html_path: Path) -> bool:
    """Insert a synchronous theme-init ``<script>`` at the start of ``<head>``.

    See the :data:`THEME_INIT_SCRIPT` docstring for the full rationale —
    in short, mystmd's React-driven theme detection runs only after
    hydration, so dark-mode users see the light theme paint first and
    then animate to dark over 500ms (the ``transition-colors duration-500``
    on ``<body>``). Setting ``<html class="dark">`` synchronously
    before first paint stops that animation entirely.

    Inserted right after the opening ``<head>`` tag, before the inline
    ``<style>`` block, so the class is in place when the browser
    computes its first set of styles. The block carries a stable
    ``id`` for idempotency.

    Args:
        html_path: Path to the HTML file under
            ``docs/<lang>/_build/html/``.

    Returns:
        True when the file was patched. False if the page already
        carries the inline ``<script>`` (idempotent re-run) or has no
        ``<head>`` to anchor against.
    """
    content = html_path.read_text(encoding="utf-8")
    if f'id="{THEME_INIT_SCRIPT_ID}"' in content:
        return False
    insert_at = _head_insertion_point(content)
    if insert_at is None:
        return False
    script_block = f'<script id="{THEME_INIT_SCRIPT_ID}">{THEME_INIT_SCRIPT}</script>'
    new = content[:insert_at] + script_block + content[insert_at:]
    html_path.write_text(new, encoding="utf-8")
    return True


def inject_script_tag(html_path: Path, script_src: str) -> bool:
    """Insert the colab-launch ``<script>`` tag right before ``</head>``.

    Returns True when a tag was added. Returns False if the page already
    carries the tag (idempotency) or if the page has no ``</head>`` to
    anchor against (e.g. a 404 stub).
    """
    content = html_path.read_text(encoding="utf-8")
    if SCRIPT_TAG_ID in content:
        return False

    closing_head = "</head>"
    if closing_head not in content:
        return False

    script_tag = (
        f'<script defer src="{script_src}" id="{SCRIPT_TAG_ID}"></script>{closing_head}'
    )
    updated = content.replace(closing_head, script_tag, 1)
    html_path.write_text(updated, encoding="utf-8")
    return True


def inject_lightbox_script_tag(html_path: Path, script_src: str) -> bool:
    """Insert the lightbox ``<script>`` tag right before ``</head>``.

    Mirrors :func:`inject_script_tag` but for ``lightbox.js``, using a
    distinct id (``LIGHTBOX_TAG_ID``) so the two injections are
    independent and idempotent. Returns True when a tag was added; False
    when the page already carries the tag or has no ``</head>`` to
    anchor against.

    Args:
        html_path (Path): The HTML file to patch in place.
        script_src (str): Relative URL (from the HTML page) at which the
            lightbox script is served.

    Returns:
        bool: True if the file was rewritten, False otherwise.
    """
    content = html_path.read_text(encoding="utf-8")
    if LIGHTBOX_TAG_ID in content:
        return False

    closing_head = "</head>"
    if closing_head not in content:
        return False

    script_tag = (
        f'<script defer src="{script_src}" id="{LIGHTBOX_TAG_ID}"></script>'
        f"{closing_head}"
    )
    updated = content.replace(closing_head, script_tag, 1)
    html_path.write_text(updated, encoding="utf-8")
    return True


# Match ``id="cite-…"`` on the rendered bibliography ``<li>``. mystmd
# derives ``…`` from the citation's label, which for DOI-resolved
# citations is the full URL (e.g.
# ``cite-https://doi.org/10.48550/arxiv.quant-ph/0407010``). The ``/``,
# ``:`` and ``.`` chars in that string cause React's client-side
# hydration to fail to reconcile the SSR ``<li>`` against its computed
# tree, and the failure cascades: a single citation `<li>` produces on
# the order of 150 React #418 errors during hydration, each of which
# is a hydration-mismatch recovery that re-renders the surrounding
# subtree client-side and is visible to the user as a flicker on the
# affected cell outputs. Replacing the unsafe chars with ``-`` so the
# id matches what React would compute on the client eliminates the
# mismatch.
#
# We anchor on the ``id=`` prefix so the regex does not accidentally
# rewrite the citation's visible label text (which legitimately
# contains ``https://`` etc.) or the ``href`` of the citation link
# (which legitimately points at the doi.org URL).
_CITE_ID_RE = re.compile(r'(?P<prefix>\sid=")cite-(?P<value>[^"]+)"', re.IGNORECASE)
# Cross-references back into the bibliography ``<li>`` use a fragment
# of the form ``href="#cite-…"``. mystmd doesn't emit these in our
# tagged-pill citation style today, but the upstream renderer can add
# them, and our sanitization needs to keep the two in sync so anchors
# still resolve.
_CITE_HREF_RE = re.compile(r'(?P<prefix>href=")#cite-(?P<value>[^"]+)"', re.IGNORECASE)
# Chars React's hydration tolerates in attribute values; everything
# else gets collapsed to a single ``-``. Underscore is included
# because it is HTML5-id-safe and appears in some upstream citation
# labels.
_UNSAFE_ID_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


def _sanitize_cite_id(raw: str) -> str:
    """Collapse non-safe chars in a cite ID to ``-``, stripping edges.

    Args:
        raw (str): The original suffix after ``cite-`` (e.g. the URL
            ``https://doi.org/10.48550/arxiv.quant-ph/0407010``).

    Returns:
        str: An id-safe suffix using only ``[A-Za-z0-9_-]`` with no
            leading or trailing ``-``. Always non-empty for non-empty
            input.
    """
    safe = _UNSAFE_ID_CHARS.sub("-", raw).strip("-")
    return safe or "id"


def sanitize_cite_ids(html_path: Path) -> bool:
    """Rewrite mystmd's ``<li id="cite-…">`` ids to React-safe chars.

    Background: mystmd renders bibliography entries with HTML ids of
    the form ``id="cite-<label>"``. For citations resolved by DOI, the
    ``<label>`` is the full URL — containing ``/``, ``:``, and ``.``,
    all of which sit outside what React's hydration is willing to
    treat as equivalent across SSR vs client computation. The result
    on every page that ships a DOI-style citation (mottonen,
    qsci, …) is a cascade of ~150 React #418 hydration-mismatch
    errors, each of which recovers by re-rendering its enclosing
    subtree client-side, which manifests to the user as a multi-second
    flicker of the cell outputs sitting near the bibliography. The
    surrounding preload guard (see ``THEME_INIT_SCRIPT``) is bounded
    to a ~5 s hard cap, so on slow networks the flicker re-emerges
    past the cap and is back in the user's face. Removing it at the
    source is the structural fix.

    Strategy: rewrite ``id="cite-<unsafe>"`` to
    ``id="cite-<unsafe-collapsed-to-A-Za-z0-9_->"`` in place, and
    keep any matching ``href="#cite-…"`` cross-references in sync so
    fragment anchors still resolve. Idempotent: a second run on the
    same HTML is a no-op because the sanitized form contains no
    unsafe chars to collapse.

    Args:
        html_path (Path): The HTML file to patch in place.

    Returns:
        bool: True if at least one ``id`` or ``href`` was rewritten,
            False otherwise (page carries no citation ids, or the ids
            were already sanitized).
    """
    content = html_path.read_text(encoding="utf-8")
    changed = False

    def _id_repl(match: re.Match[str]) -> str:
        nonlocal changed
        original = match.group("value")
        sanitized = _sanitize_cite_id(original)
        if sanitized == original:
            return match.group(0)
        changed = True
        return f'{match.group("prefix")}cite-{sanitized}"'

    def _href_repl(match: re.Match[str]) -> str:
        nonlocal changed
        original = match.group("value")
        sanitized = _sanitize_cite_id(original)
        if sanitized == original:
            return match.group(0)
        changed = True
        return f'{match.group("prefix")}#cite-{sanitized}"'

    new_content = _CITE_ID_RE.sub(_id_repl, content)
    new_content = _CITE_HREF_RE.sub(_href_repl, new_content)

    if not changed:
        return False
    html_path.write_text(new_content, encoding="utf-8")
    return True


def rewrite_build_src_paths(target_path: Path, language: str) -> bool:
    """Rewrite ``docs/_build_src/<lang>/`` → ``docs/<lang>/`` in ``target_path``.

    mystmd builds from ``docs/_build_src/<lang>/`` and bakes the build-dir
    path into the GitHub edit / blob URLs both in the page-header HTML and
    in the per-page JSON data layer that the SPA hydrates from on
    client-side navigation. The canonical source lives at ``docs/<lang>/``
    on the branch; ``_build_src/`` is gitignored. Strip the ``_build_src/``
    segment so the "Edit on GitHub" / "View source" links resolve, and so
    ``colab-launch.js``'s ``^docs/(en|ja)/`` path check accepts the
    embedded edit URL and renders the Colab button.

    Patching only the HTML pages would leave the JSON data layer stale: a
    full page load would pick up the rewritten HTML, but as soon as the
    SPA navigated to another page it would re-hydrate the page header from
    the JSON and re-introduce the ``_build_src/`` URL — so the Colab
    button would silently disappear after the first navigation. The
    caller therefore runs this on the union of ``*.html`` and ``*.json``.

    Returns True when the file was rewritten.
    """
    needle = f"docs/_build_src/{language}/"
    replacement = f"docs/{language}/"
    content = target_path.read_text(encoding="utf-8")
    if needle not in content:
        return False
    target_path.write_text(content.replace(needle, replacement), encoding="utf-8")
    return True


def patch_language_build(
    docs_root: Path, language: str
) -> tuple[int, int, int, int, int, int, int, int]:
    """Run all post-build patches for one language's build output.

    Patches applied:

    1. Rewrite build-dir paths in GitHub URLs across both ``*.html`` and
       the SPA's ``*.json`` data layer (see
       :func:`rewrite_build_src_paths`).
    2. Inline the full theme CSS into every ``*.html``'s ``<head>`` so
       first paint already carries the warm-amber theme overrides
       (see :func:`inline_theme_css`).
    3. Inject a synchronous theme-init ``<script>`` at the start of
       ``<head>`` (see :func:`init_theme_script`) — this runs BEFORE
       first paint so dark-mode users don't see a 500 ms light→dark
       transition animation.
    4. Inject the colab-launch ``<script>`` tag right before
       ``</head>`` on every ``*.html`` (see :func:`inject_script_tag`).
    5. Inject the lightbox ``<script>`` tag right before ``</head>``
       on every ``*.html`` (see :func:`inject_lightbox_script_tag`)
       so wide cell-output images become click-to-zoom modals on the
       rendered page.
    6. Sanitize bibliography ``<li id="cite-…">`` ids to
       ``[A-Za-z0-9_-]`` (see :func:`sanitize_cite_ids`). DOI-resolved
       citations otherwise carry ids whose ``/`` / ``:`` chars cause
       React hydration to fail-and-recover in a cascade visible as a
       multi-second flicker around the bibliography.

    Returns:
        ``(injected_count, rewritten_count, css_inlined_count,
        theme_init_count, lightbox_count, cite_sanitized_count,
        total_html, total_json)``.
    """
    html_root = docs_root / language / "_build" / "html"
    if not html_root.exists():
        raise RuntimeError(f"{language}: build directory not found: {html_root}")

    source_script = docs_root / "assets" / SCRIPT_FILE_NAME
    if not source_script.exists():
        raise RuntimeError(f"Missing source script: {source_script}")

    output_script = html_root / "build" / SCRIPT_FILE_NAME
    output_script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_script, output_script)

    source_lightbox = docs_root / "assets" / LIGHTBOX_FILE_NAME
    if not source_lightbox.exists():
        raise RuntimeError(f"Missing source script: {source_lightbox}")
    output_lightbox = html_root / "build" / LIGHTBOX_FILE_NAME
    shutil.copy2(source_lightbox, output_lightbox)

    theme_css = _read_theme_css(docs_root)

    html_files = sorted(html_root.rglob("*.html"))
    json_files = sorted(html_root.rglob("*.json"))
    injected_count = 0
    rewritten_count = 0
    css_inlined_count = 0
    theme_init_count = 0
    lightbox_count = 0
    cite_sanitized_count = 0

    # Rewrite build-dir paths across HTML + JSON. JSON is the SPA's data
    # layer; without rewriting it, post-navigation re-hydration puts the
    # _build_src/ URL back into the DOM and colab-launch.js drops the
    # button (see rewrite_build_src_paths' docstring).
    for target in (*html_files, *json_files):
        if rewrite_build_src_paths(target, language):
            rewritten_count += 1

    # Inline theme CSS + inject the theme-init script + inject the
    # colab script tag, all into HTML only. JSON pages are not
    # navigated to directly; the SPA loads them via the host HTML,
    # which already carries the patches. The theme-init script runs
    # AFTER inline_theme_css below — both insert at the same offset
    # (right after the opening <head>), so the later insert ends up
    # earlier in the output. That's intentional: we want the script
    # to run before the inline <style> is parsed so the browser
    # computes its first set of styles with the correct dark/light
    # class already on <html>.
    for html_file in html_files:
        if inline_theme_css(html_file, theme_css):
            css_inlined_count += 1
        if init_theme_script(html_file):
            theme_init_count += 1
        relative_script = os.path.relpath(output_script, html_file.parent)
        relative_script = Path(relative_script).as_posix()
        if inject_script_tag(html_file, relative_script):
            injected_count += 1
        relative_lightbox = os.path.relpath(output_lightbox, html_file.parent)
        relative_lightbox = Path(relative_lightbox).as_posix()
        if inject_lightbox_script_tag(html_file, relative_lightbox):
            lightbox_count += 1
        if sanitize_cite_ids(html_file):
            cite_sanitized_count += 1

    return (
        injected_count,
        rewritten_count,
        css_inlined_count,
        theme_init_count,
        lightbox_count,
        cite_sanitized_count,
        len(html_files),
        len(json_files),
    )


def main() -> int:
    """Patch every requested language's build output."""
    args = parse_args()
    docs_root = Path(__file__).resolve().parents[1]

    for language in args.languages:
        (
            injected_count,
            rewritten_count,
            css_inlined_count,
            theme_init_count,
            lightbox_count,
            cite_sanitized_count,
            total_html,
            total_json,
        ) = patch_language_build(docs_root, language)
        print(
            f"{language}: injected script tag into {injected_count}/{total_html} HTML, "
            f"injected lightbox script into {lightbox_count}/{total_html} HTML, "
            f"inlined theme CSS into {css_inlined_count}/{total_html} HTML, "
            f"injected theme-init script into {theme_init_count}/{total_html} HTML, "
            f"sanitized cite ids in {cite_sanitized_count}/{total_html} HTML, "
            f"rewrote build-dir paths in {rewritten_count}/{total_html + total_json} "
            f"({total_html} HTML + {total_json} JSON)"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI fallback
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
