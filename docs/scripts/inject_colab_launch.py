#!/usr/bin/env python3
"""Post-build HTML patches for the Qamomile docs.

Originally just a colab-launch script-tag injector; the script has
since absorbed five more passes that all run against the same set of
``docs/<lang>/_build/html/*.html`` (and ``*.json``) files, so they live
together for a single rglob walk.

The six passes:

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

6. Sanitize bibliography ``<li id="cite-…">`` ids so the SSR ``id``
   matches what React computes on the client. mystmd renders
   DOI-resolved citations with HTML ids containing ``/`` / ``:`` /
   ``.``, which makes SSR disagree with the client computation and
   cascades into ~150 React #418 hydration-mismatch errors and a
   visible flicker around the bibliography. Replacing the unsafe
   chars with ``-`` aligns the two sides.
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
#     class, and once the page has settled the class is removed.
#     "Settled" is detected by observing DOM mutations from script
#     init onward and revealing once BOTH (a) ``DOMContentLoaded``
#     has fired (so deferred / module scripts have executed and
#     hydration has had a chance to start) AND (b) 250 ms has
#     elapsed since the last observed mutation (so React's hydration
#     and any mismatch recovery have quiesced). Bounded by an
#     8-second hard-cap from observer start and a 10-second wall-
#     clock safety net so a runaway page can never stay hidden
#     forever. We deliberately do NOT wait for ``window.load`` (slow
#     image / CDN resources can push it well past hydration end and
#     turn the guard into a multi-second blank screen). See the
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
    # The reveal is gated on TWO conditions:
    #
    #  (a) ``DOMContentLoaded`` has fired. Per the HTML spec, this
    #      means parsing finished AND every deferred / module
    #      ``<script>`` in the document has executed. Since mystmd
    #      ships its React entry as ``type="module"`` (deferred by
    #      default), hydration has had a chance to start before
    #      ``DOMContentLoaded`` fires. Gating on ``DOMContentLoaded``
    #      avoids the "JS bundle still downloading" race where the
    #      ``MutationObserver`` would otherwise see no mutations for
    #      the download window and reveal pre-hydration.
    #
    #  (b) The DOM has been quiet for 250 ms — that is, the
    #      ``MutationObserver`` registered at script init hasn't
    #      observed a mutation in 250 ms. This is the "hydration
    #      finished and React isn't recovering anymore" detector. The
    #      observer is installed BEFORE ``DOMContentLoaded`` so we
    #      catch the head→body parse mutations and the entire
    #      hydration burst, not just the tail.
    #
    # Earlier iterations gated the reveal on ``window.load`` (raced
    # against a few additional triggers). That turned out to be wrong
    # on every axis:
    #   - ``load`` on heavy mystmd pages can fire well past hydration
    #     end (the mottonen page measures ``loadEventEnd`` near 8s in
    #     production thanks to lazy-loaded images and external CDN
    #     assets). Waiting for ``load`` turns the guard into a multi-
    #     second blank screen even after React has already settled.
    #   - The user-driven-event fallback (``mousemove`` / ``click`` /
    #     etc.) was hostile by design: any mouse movement during the
    #     hidden phase raced the guard down and exposed the in-
    #     progress hydration. That's the ちらつき regression we keep
    #     getting reported.
    #   - A 2-second safety timeout measured from script start always
    #     fired before ``load`` did, defeating its own purpose.
    #
    # Two safety nets bound the worst case so a runaway page never
    # stays blank forever:
    #   1. ``hardCap``: 8 seconds from observer start. A page still
    #      mutating past 8 s is in a runaway state and the user is
    #      better served by seeing it than by an indefinitely hidden
    #      body.
    #   2. ``wallClock``: 10 seconds from script start. Covers the
    #      degenerate case where the observer never installs at all
    #      (older browsers without ``MutationObserver`` /
    #      ``requestAnimationFrame`` / ``performance.now``).
    #
    # Feature detection: if any of the three APIs is missing, fall
    # back to a 1-second blanket reveal. Better than waiting for the
    # 10 s wall clock and far better than crashing in flight.
    "var revealed=false;"
    "function reveal(){"
    "if(revealed)return;"
    "revealed=true;"
    'd.classList.remove("qamomile-preloading");'
    "}"
    "setTimeout(reveal,10000);"
    "if(typeof MutationObserver!=='function'"
    "||typeof requestAnimationFrame!=='function'"
    "||typeof performance==='undefined'"
    "||typeof performance.now!=='function'){"
    "setTimeout(reveal,1000);"
    "return;"
    "}"
    # Track whether DOMContentLoaded has fired. The flag is set
    # synchronously by an event listener registered before observer
    # start, so the very first ``checkIdle`` tick after DCL sees
    # ``dclFired === true`` without races.
    "var dclFired=document.readyState!=='loading';"
    "if(!dclFired){"
    "document.addEventListener('DOMContentLoaded',function(){"
    "dclFired=true;"
    "},{once:true});"
    "}"
    "var lastMut=performance.now();"
    "var hardCap=setTimeout(reveal,8000);"
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
    # Both gates must be satisfied: DOMContentLoaded fired AND 250 ms
    # since the last mutation. The DCL gate prevents an early reveal
    # while deferred / module scripts are still loading their bundles
    # (and the MutationObserver therefore sees a quiet DOM that does
    # NOT mean hydration is done).
    "if(dclFired&&performance.now()-lastMut>=250){"
    "obs.disconnect();"
    "clearTimeout(hardCap);"
    "reveal();"
    "}else{"
    "requestAnimationFrame(checkIdle);"
    "}"
    "}"
    "requestAnimationFrame(checkIdle);"
    "})();"
)

# Preload guard: lives inside the inline theme <style> block (see
# :func:`inline_theme_css`) so the rule is parsed in the same first-
# style-pass that interprets the rest of the theme overrides. The
# ``html.qamomile-preloading`` class is set by THEME_INIT_SCRIPT a
# moment earlier (inline <script> at the start of <head>), so the
# first paint already has these rules in effect. The class is later
# removed by THEME_INIT_SCRIPT itself — once both
# ``DOMContentLoaded`` has fired AND the DOM has been quiet for
# 250 ms, or, as safety nets, after 8 s from observer start or
# 10 s wall-clock from script start. Once removed, these rules stop
# matching and normal styling / transitions resume.
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


# mystmd's SSR bibliography lives in ``<section ... class="…myst-
# bibliography…">``. The pair below is the most general "find every
# section" / "extract its class attribute" regex; the actual CSS-
# class-token check happens in Python so we don't false-match
# ``not-myst-bibliography`` (a regex ``\bmyst-bibliography\b`` would,
# since ``\b`` only checks the word/non-word boundary and ``-`` is a
# non-word character).
#
# Known limitation: lazy ``.*?`` matching means a nested ``<section>``
# inside the bibliography would close the regex at the inner
# ``</section>`` instead of the outer one. mystmd's current output
# never nests sections inside ``<section class="myst-bibliography">``
# (the body is a flat ``<header>`` + ``<ol><li>…</li></ol>``), so
# this is theoretical, but if upstream ever changes we'll need to
# switch to a proper HTML parser. The regression is **silent**: any
# cite-id beyond the inner ``</section>`` falls out of our scope
# scan and is left in its DOI-URL form, so the React hydration
# cascade returns for whichever citations sit past the boundary
# (the collision detection only runs on ids we did capture, so it
# does not catch this case). Treat the parser switch as a hard
# requirement if upstream ever nests sections inside the
# bibliography.
_SECTION_RE = re.compile(
    r"<section\b(?P<attrs>[^>]*)>(?P<body>.*?)</section>",
    re.IGNORECASE | re.DOTALL,
)
# ``(?<![\w-])`` is the attribute-name boundary: it forbids
# ``data-class="…"`` from matching as if it were ``class="…"``. The
# default ``\b`` would, since ``\b`` treats the ``-`` in ``data-`` as
# a non-word char and considers the boundary between ``-`` and ``c``
# to be a word break — exactly what we don't want.
_CLASS_ATTR_RE = re.compile(
    r'(?<![\w-])class\s*=\s*(?P<q>["\'])(?P<value>[^"\']*)(?P=q)',
    re.IGNORECASE,
)
# Match ``id="cite-…"`` on the rendered bibliography ``<li>`` (scoped
# by the section / class-token check above). mystmd derives ``…``
# from the citation's label, which for DOI-resolved citations is the
# full URL (e.g.
# ``cite-https://doi.org/10.48550/arxiv.quant-ph/0407010``). The
# ``/``, ``:``, and ``.`` chars in that string cause the SSR ``id``
# attribute to disagree with what React computes on the client
# during hydration, and the failure cascades: a single citation
# ``<li>`` produces on the order of 150 React #418 errors during
# hydration, each of which recovers by re-rendering its enclosing
# subtree client-side and is visible to the user as a flicker on
# the affected cell outputs. Collapsing the unsafe chars to ``-``
# aligns the SSR id with the client computation and eliminates the
# mismatch.
#
# We anchor on the ``id=`` prefix so the regex does not accidentally
# rewrite the citation's visible label text (which legitimately
# contains ``https://`` etc.) or the ``href`` of the citation link
# (which legitimately points at the doi.org URL). Both regexes
# tolerate the HTML5-legal whitespace variations: ``\s+`` for the
# leading attribute separator (mystmd emits one space today but
# pretty-printers can emit a newline + indent), ``\s*`` around the
# ``=``, and a captured quote char so single- or double-quoted
# values both work. The closing quote in the value capture uses a
# backreference so we never cross a quote boundary by accident.
# ``re.IGNORECASE`` because HTML attribute names are case-insensitive
# in the spec, so ``ID=`` and ``Id=`` from an alternate serializer
# still match.
_CITE_ID_IN_BIB_RE = re.compile(
    r'(\s+id\s*=\s*(["\']))cite-([^"\']+)\2',
    re.IGNORECASE,
)
# Cross-references back into the bibliography use a fragment of the
# form ``href="#cite-…"``. mystmd doesn't emit these in our
# tagged-pill citation style today, but the upstream renderer can
# (e.g. footnote-style citation pills), and rewritten ids must stay
# in sync with their cross-references so the fragment anchor still
# resolves to its ``<li>`` after sanitization. Same whitespace /
# quote-style tolerance as ``_CITE_ID_IN_BIB_RE``. The ``\s+`` before
# ``href`` is load-bearing: without it, ``data-href="#cite-…"`` or a
# string in inline JSON / a code block that happens to contain
# ``href="#cite-…"`` would be silently rewritten too.
_CITE_HREF_RE = re.compile(
    r'(\s+href\s*=\s*(["\']))#cite-([^"\']+)\2',
    re.IGNORECASE,
)
# Chars HTML5 + CSS-selector + React all accept in id values without
# special encoding; everything else gets collapsed to a single ``-``.
# Underscore is included because it is HTML5-id-safe and appears in
# some upstream citation labels.
_UNSAFE_ID_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


def _has_class_token(class_value: str, token: str) -> bool:
    """Return True iff ``token`` appears as a CSS class in ``class_value``.

    Splits on whitespace so ``"foo myst-bibliography bar"`` matches
    ``myst-bibliography`` but ``"not-myst-bibliography"`` does not.

    Args:
        class_value (str): The raw value of an HTML ``class``
            attribute (the content between the surrounding quotes).
        token (str): The exact class token to look for. Compared
            case-sensitively because mystmd's emitted classes are
            stable case-wise.

    Returns:
        bool: True if ``token`` is one of the whitespace-separated
            tokens in ``class_value``.
    """
    return token in class_value.split()


def _sanitize_cite_id(raw: str) -> str:
    """Collapse non-safe chars in a cite ID to ``-``, stripping edges.

    The caller is responsible for HTML-entity-decoding ``raw`` first
    so that ``cite-a&amp;b`` and the equivalent literal ``cite-a&b``
    produce the same sanitized form. Without that step the SSR side
    would sanitize to ``a-amp-b`` while React's client side — which
    builds the id from the decoded label string — would sanitize to
    ``a-b``, reintroducing the very mismatch this pass is meant to
    eliminate.

    Args:
        raw (str): The original suffix after ``cite-``, already
            HTML-entity-decoded (e.g. the URL
            ``https://doi.org/10.48550/arxiv.quant-ph/0407010``).

    Returns:
        str: An id-safe suffix using only ``[A-Za-z0-9_-]`` with no
            leading or trailing ``-``. Always non-empty for non-empty
            input.
    """
    safe = _UNSAFE_ID_CHARS.sub("-", raw).strip("-")
    return safe or "id"


def _decode_id_raw(raw: str) -> str:
    """HTML-entity-decode a raw id suffix.

    React computes the bibliography ``<li>`` id from the citation's
    decoded label, not from the SSR-encoded form, so we have to
    normalize entities before sanitizing. ``cite-a&amp;b`` and
    ``cite-a&b`` must therefore collapse to the same sanitized id.

    Args:
        raw (str): Raw id suffix as captured by the regex (may
            contain HTML entities such as ``&amp;``).

    Returns:
        str: The same suffix with HTML entities decoded.
    """
    import html

    return html.unescape(raw)


def _decode_href_raw(raw: str) -> str:
    """Percent-decode then HTML-entity-decode a fragment label.

    A ``href="#cite-…"`` value can carry both ``%xx`` percent-encoding
    (browser-level URL escaping) and HTML entities (HTML-level
    escaping), in either order. Decode both so the resulting key
    matches the ``_decode_id_raw`` output of the corresponding ``<li>``
    id and the cross-reference can be rewritten consistently.

    Args:
        raw (str): Raw fragment suffix as captured by the regex.

    Returns:
        str: ``raw`` after URL percent-decoding followed by HTML-
            entity decoding.
    """
    import html
    from urllib.parse import unquote

    return html.unescape(unquote(raw))


def sanitize_cite_ids(html_path: Path) -> bool:
    """Rewrite mystmd's ``<li id="cite-…">`` ids to match React's client compute.

    Background: mystmd renders bibliography entries with HTML ids of
    the form ``id="cite-<label>"``. For citations resolved by DOI, the
    ``<label>`` is the full URL — containing ``/``, ``:``, and ``.``,
    none of which the SSR'd HTML and React's client computation agree
    on during hydration. The result on every page that ships a
    DOI-style citation (mottonen, qsci, …) is a cascade of ~150 React
    #418 hydration-mismatch errors, each of which recovers by re-
    rendering its enclosing subtree client-side, which manifests to
    the user as a multi-second flicker of the cell outputs sitting
    near the bibliography. Removing it at the source is the
    structural fix.

    Strategy: only touch the SSR ``<section class="…myst-bibliography
    …">`` block, scoped by a proper CSS-class-token check (so
    ``class="not-myst-bibliography"`` does NOT match). Within that
    scope, collect every ``id="cite-<raw>"``, HTML-entity-decode the
    raw label, compute its sanitized form, and build a one-pass
    ``{decoded → sanitized}`` map. Then:

    1. Rewrite the ``id`` attributes inside the bibliography section
       using the map (matched by decoded label, not the SSR raw, so
       both ``cite-a&amp;b`` and the equivalent ``cite-a&b`` collapse
       to one entry).
    2. Rewrite any matching ``href="#cite-…"`` fragment cross-
       references anywhere in the document using the same map. A
       fragment whose decoded label is not in the map is left alone
       — it points at something that isn't a bibliography ``<li>``
       and isn't ours to rewrite. Percent-encoded and HTML-entity-
       encoded forms are both decoded before lookup, so
       ``#cite-https%3A%2F%2F…`` or ``#cite-a&amp;b`` still match the
       original ``<li>`` label.

    Collisions: any path that would produce two elements sharing the
    same ``id`` value on the rendered page fails loudly because the
    resulting HTML would be invalid and fragment navigation would
    pick one of the duplicates arbitrarily. Three such paths are
    detected and all raise ``RuntimeError``:

    1. **Distinct decoded labels** that happen to collapse to the
       same sanitized form (e.g. ``cite-a/b`` and ``cite-a:b`` both
       become ``cite-a-b``).
    2. **Repeated decoded occurrences** — the same decoded label
       appearing as multiple SSR ``id`` attributes. E.g.
       ``cite-a&amp;b`` and ``cite-a&b`` are two ``<li>`` items
       whose ``id`` values decode to the same string (``a&b``), so
       both would rewrite to ``cite-a-b``. The SSR is already
       broken in that case (duplicate ids at the DOM level) but
       otherwise our pass would silently fan the encoded forms into
       the canonical one and hide the upstream issue.
    3. **Out-of-scope ``id="cite-…"`` collision** — an existing
       ``id="cite-…"`` outside the bibliography (e.g. a hand-rolled
       HTML cell, a code example, a custom anchor) that already
       holds the sanitized form of one of our bibliography ids.
       We don't rewrite the out-of-scope id (scoped pass), but
       once the bibliography id is sanitized, both elements would
       carry the same ``id`` value on the rendered page.

    Idempotent: re-running on a sanitized HTML is a no-op — the
    sanitized labels match the unchanged sanitized form, and no
    ``id`` / ``href`` ends up actually rewritten.

    Args:
        html_path (Path): The HTML file to patch in place.

    Returns:
        bool: True if at least one ``id`` or ``href`` was rewritten,
            False otherwise (page carries no bibliography section,
            no citation ids inside it, or the ids were already
            sanitized).

    Raises:
        RuntimeError: If two distinct decoded citation labels collapse
            to the same sanitized form, if the same decoded label
            appears as multiple SSR ``id`` occurrences, or if an
            existing out-of-scope ``id="cite-…"`` already holds the
            sanitized form of a bibliography id. All three would
            create duplicate ``id`` attributes on the rendered page.
    """
    content = html_path.read_text(encoding="utf-8")

    # Pass 1: walk every ``<section>``, keep only those whose class
    # attribute actually contains ``myst-bibliography`` as a CSS
    # token, and collect both the unique decoded labels (for the
    # ``decoded → sanitized`` rewrite map) and every per-id-attribute
    # occurrence (so we can detect duplicate-id outputs in pass 2 of
    # the collision check).
    decoded_to_sanitized: dict[str, str] = {}
    sanitized_origins: dict[str, list[str]] = {}
    sanitized_occurrences: dict[str, list[str]] = {}  # sanitized -> raw[]
    bibliography_spans: list[tuple[int, int]] = []
    for section_match in _SECTION_RE.finditer(content):
        attrs = section_match.group("attrs")
        class_match = _CLASS_ATTR_RE.search(attrs)
        if class_match is None:
            continue
        if not _has_class_token(class_match.group("value"), "myst-bibliography"):
            continue
        bibliography_spans.append((section_match.start(), section_match.end()))
        for id_match in _CITE_ID_IN_BIB_RE.finditer(section_match.group(0)):
            raw = id_match.group(3)
            decoded = _decode_id_raw(raw)
            sanitized = _sanitize_cite_id(decoded)
            sanitized_occurrences.setdefault(sanitized, []).append(raw)
            if decoded in decoded_to_sanitized:
                continue
            decoded_to_sanitized[decoded] = sanitized
            sanitized_origins.setdefault(sanitized, []).append(decoded)

    if not decoded_to_sanitized:
        return False  # no bibliography or no cite ids inside one

    # Collision check before we mutate anything. Two failure modes:
    #   1. Two *distinct* decoded labels collapse to the same
    #      sanitized form (e.g. ``cite-a/b`` and ``cite-a:b`` both
    #      become ``cite-a-b``).
    #   2. The *same* decoded label appears as multiple SSR ``id``
    #      occurrences (e.g. ``cite-a&amp;b`` and ``cite-a&b`` are two
    #      `<li>` items in the source that both decode to ``a&b`` and
    #      would both rewrite to ``cite-a-b``). The SSR is already
    #      broken in that case — duplicate ids at the DOM level — but
    #      our sanitizer otherwise silently fans them into the
    #      canonical form and hides the upstream issue. Fail loud
    #      either way so the build never ships a page with duplicate
    #      bibliography ids.
    distinct_collisions = {s: rs for s, rs in sanitized_origins.items() if len(rs) > 1}
    occurrence_collisions = {
        s: rs for s, rs in sanitized_occurrences.items() if len(rs) > 1
    }
    # Third failure mode: an out-of-scope ``id="cite-…"`` already in
    # the page (e.g. a hand-rolled HTML cell or a code example) that
    # happens to match the sanitized form of one of our bibliography
    # ids. We don't rewrite that out-of-scope id (different scope),
    # but the post-sanitize page would still ship two elements with
    # the same ``id``. Scan every ``id="cite-…"`` outside the
    # bibliography spans we collected above and check none of them
    # collides with the sanitized values we're about to produce.
    sanitized_values = set(decoded_to_sanitized.values())
    out_of_scope_collisions: dict[str, str] = {}
    for id_match in _CITE_ID_IN_BIB_RE.finditer(content):
        start = id_match.start()
        if any(s <= start < e for s, e in bibliography_spans):
            continue  # inside bibliography — already counted above
        raw_existing = id_match.group(3)
        decoded_existing = _decode_id_raw(raw_existing)
        if decoded_existing in sanitized_values:
            out_of_scope_collisions[decoded_existing] = raw_existing
    if distinct_collisions or occurrence_collisions or out_of_scope_collisions:
        parts: list[str] = []
        for sanitized, decoded_raws in sorted(distinct_collisions.items()):
            parts.append(f"{sanitized!r} <- distinct labels {sorted(decoded_raws)!r}")
        for sanitized, raw_occurrences in sorted(occurrence_collisions.items()):
            if sanitized in distinct_collisions:
                continue  # already reported with richer detail
            parts.append(
                f"{sanitized!r} <- repeated occurrences {sorted(raw_occurrences)!r}"
            )
        for sanitized, raw_existing in sorted(out_of_scope_collisions.items()):
            parts.append(
                f"{sanitized!r} <- already used by out-of-scope id {raw_existing!r}"
            )
        details = "; ".join(parts)
        raise RuntimeError(
            f"{html_path}: bibliography citation ids would collapse to "
            f"duplicate sanitized values, which would produce duplicate "
            f"id attributes on the page: {details}"
        )

    # If every decoded label is already its own sanitized form, no
    # rewriting is needed — early-exit to keep idempotency cheap.
    if all(decoded == sanitized for decoded, sanitized in decoded_to_sanitized.items()):
        return False

    # Pass 2: rewrite ids only inside bibliography section spans.
    # ``match.group(1)`` is the captured prefix up to and including
    # the opening quote; ``match.group(2)`` is the quote char alone,
    # which we reuse as the closing quote so single-quoted attributes
    # keep their single quotes.
    def _id_repl(match: re.Match[str]) -> str:
        decoded = _decode_id_raw(match.group(3))
        sanitized = decoded_to_sanitized.get(decoded, match.group(3))
        return f"{match.group(1)}cite-{sanitized}{match.group(2)}"

    pieces: list[str] = []
    cursor = 0
    for start, end in bibliography_spans:
        pieces.append(content[cursor:start])
        pieces.append(_CITE_ID_IN_BIB_RE.sub(_id_repl, content[start:end]))
        cursor = end
    pieces.append(content[cursor:])
    new_content = "".join(pieces)

    # Pass 3: rewrite fragment cross-references anywhere using the
    # same map. Unknown fragments (decoded label not in the map) are
    # passed through — they point at non-bibliography anchors. Same
    # group layout as ``_id_repl``: 1 = prefix-with-opening-quote,
    # 2 = quote char, 3 = raw value after ``#cite-``.
    def _href_repl(match: re.Match[str]) -> str:
        decoded = _decode_href_raw(match.group(3))
        sanitized = decoded_to_sanitized.get(decoded)
        if sanitized is None or sanitized == match.group(3):
            return match.group(0)
        return f"{match.group(1)}#cite-{sanitized}{match.group(2)}"

    new_content = _CITE_HREF_RE.sub(_href_repl, new_content)

    if new_content == content:
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

    Args:
        docs_root (Path): Repository ``docs/`` directory. The function
            reads its source assets from ``docs/assets/`` and writes
            into ``docs/<language>/_build/html/``.
        language (str): Language slug — ``"en"`` or ``"ja"``. Drives
            both the build-output directory and the source-path
            rewrite needle.

    Returns:
        tuple[int, int, int, int, int, int, int, int]: An 8-tuple of
            counters in the form ``(injected_count, rewritten_count,
            css_inlined_count, theme_init_count, lightbox_count,
            cite_sanitized_count, total_html, total_json)``. The
            first six are how many files each pass actually rewrote;
            the last two are the universe sizes the passes iterated
            over (every ``*.html`` / ``*.json`` under the build
            output) and are reported alongside so the audit print in
            ``main`` can render ``X/Y`` ratios.

    Raises:
        RuntimeError: If the language's build directory or one of the
            source asset scripts (``colab-launch.js`` /
            ``lightbox.js``) is missing.
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
