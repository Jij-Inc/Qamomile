// Qamomile docs lightbox.
//
// Targets every <img> and <svg> emitted into a Jupyter cell output by
// matplotlib (or any other plotting library). Default rendering shrinks
// the figure to the column width via the responsive
// `svg/img { max-width: 100% }` rule the docs theme inherits from
// Tailwind, so wide unrolled circuit diagrams appear as readable
// thumbnails on the page. Clicking the thumbnail opens a centred
// overlay with the same content shown at its natural size; the user
// can then use the browser's native Ctrl/Cmd + scroll to zoom further.
// Closing is handled by clicking the backdrop, clicking the image
// again, or pressing Escape.
//
// mystmd is a React-driven SPA, so we (a) bind on first paint via
// DOMContentLoaded, and (b) re-bind on subsequent client-side
// navigations via a MutationObserver on `document.documentElement`.
// We observe `documentElement` (the `<html>` element) rather than
// `document.body` because mystmd's hydration recovery — triggered
// when the SSR HTML doesn't match the hydrated tree, including by the
// very `<script>` tag we inject into `<head>` — replaces
// `document.body` wholesale, which would orphan an observer attached
// to the old body. `documentElement` survives that recovery, so the
// observer keeps firing for the freshly hydrated body subtree.
//
// Each target node is tagged with `data-qamomileLightbox="1"` once
// bound so repeated MutationObserver fires do not stack handlers.
// The observer callback itself is coalesced via requestAnimationFrame
// so a hydration burst that mutates the DOM many times in one frame
// only runs one bindAll() per frame.
(function () {
  "use strict";

  // We intentionally exclude figures that already live inside a
  // lightbox overlay — those are the cloned previews and must not be
  // re-bound recursively. Also exclude `data-slot="icon"` SVGs
  // (mystmd's chrome — theme toggle, search icon, hamburger menu,
  // copy-code button, etc.) so the lightbox only fires on actual
  // notebook outputs.
  //
  // mystmd wraps every cell-output image in
  // `<div data-name="safe-output-image">`, which is the precise
  // anchor for matplotlib outputs. The fallback selectors target
  // classic Jupyter (`.jp-OutputArea`), Jupyter Book / nbsphinx
  // (`.cell_output`), Thebe live-execution (`.thebe-output`), and
  // generic figures, so the script keeps working if mystmd renames
  // the outer wrapper or if a page is ever served from a different
  // notebook renderer.
  var SELECTOR = [
    '[data-name="safe-output-image"] img',
    '[data-name="safe-output-image"] svg',
    '.jp-OutputArea img',
    '.jp-OutputArea svg:not([data-slot="icon"])',
    '.jupyter-output img',
    '.jupyter-output svg:not([data-slot="icon"])',
    '.thebe-output img',
    '.thebe-output svg:not([data-slot="icon"])',
    '.cell_output img',
    '.cell_output svg:not([data-slot="icon"])',
    '.cell-output img',
    '.cell-output svg:not([data-slot="icon"])',
    '.notebook-output img',
    '.notebook-output svg:not([data-slot="icon"])',
    'article figure img',
    'article figure svg:not([data-slot="icon"])',
  ].join(", ");

  var OVERLAY_ID = "qamomile-lightbox-overlay";
  var DATA_FLAG = "qamomileLightbox";
  // Saved across openOverlay() / closeOverlay() so we can return focus
  // to the element the user activated the lightbox from once they
  // dismiss it (a11y requirement for `role="dialog"`).
  var previouslyFocused = null;

  function closeOverlay() {
    var overlay = document.getElementById(OVERLAY_ID);
    if (overlay) {
      overlay.remove();
    }
    document.removeEventListener("keydown", onOverlayKeydown, true);
    if (previouslyFocused && typeof previouslyFocused.focus === "function") {
      try {
        previouslyFocused.focus();
      } catch (_) {
        // Element may have been detached during SPA navigation; the
        // failed restoration is harmless — focus simply falls back to
        // <body>.
      }
    }
    previouslyFocused = null;
  }

  function onOverlayKeydown(event) {
    if (event.key === "Escape") {
      event.preventDefault();
      closeOverlay();
      return;
    }
    if (event.key === "Tab") {
      // The overlay is the only focusable element in its subtree
      // (the cloned <img>/<svg> is not interactive), so trapping
      // focus is just keeping the overlay focused on every Tab.
      event.preventDefault();
      var overlay = document.getElementById(OVERLAY_ID);
      if (overlay) {
        overlay.focus();
      }
    }
  }

  function openOverlay(node) {
    closeOverlay(); // ensure only one overlay at a time
    previouslyFocused =
      document.activeElement && document.activeElement !== document.body
        ? document.activeElement
        : node;

    var overlay = document.createElement("div");
    overlay.id = OVERLAY_ID;
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute(
      "aria-label",
      "Enlarged figure preview. Press Escape or click outside to close."
    );
    // tabindex="-1" so focus() works without injecting the overlay
    // into the natural Tab order; the explicit Tab handler above
    // keeps focus on the overlay while it is open.
    overlay.setAttribute("tabindex", "-1");
    overlay.style.cssText = [
      "position: fixed",
      "inset: 0",
      "background: rgba(15, 15, 15, 0.85)",
      "z-index: 99999",
      "display: flex",
      "align-items: center",
      "justify-content: center",
      "cursor: zoom-out",
      "overflow: auto",
      "padding: 2vmin",
      "outline: none",
    ].join(";");

    // The cloned content sits in a white card so SVG strokes / dark
    // theme axis labels stay legible against the dark backdrop.
    var card = document.createElement("div");
    card.style.cssText = [
      "background: white",
      "padding: 16px",
      "border-radius: 6px",
      "box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4)",
      "max-width: 96vw",
      "max-height: 96vh",
      "overflow: auto",
      "cursor: zoom-out",
    ].join(";");

    var clone = node.cloneNode(true);
    // Force the natural size on the cloned node so the overlay shows
    // the figure at full resolution; the cell-output thumbnail is
    // shrunk by the responsive `max-width: 100%` rule.
    clone.style.maxWidth = "none";
    clone.style.maxHeight = "none";
    clone.style.height = "auto";
    clone.style.width = "auto";
    clone.style.display = "block";
    clone.removeAttribute("id"); // avoid duplicate IDs in the document
    // Re-armed thumbnail data flag would prevent re-binding the
    // original node next time, so strip it from the clone.
    if (clone.dataset) {
      delete clone.dataset[DATA_FLAG];
    }

    card.appendChild(clone);
    overlay.appendChild(card);

    // Backdrop or card click both close. Card click is wired
    // separately so a click *on the image* (which bubbles up) also
    // closes — matching the typical lightbox contract.
    overlay.addEventListener("click", closeOverlay);

    document.body.appendChild(overlay);
    document.addEventListener("keydown", onOverlayKeydown, true);
    // Move focus to the overlay so subsequent keyboard input
    // (Escape / Tab) is captured by the dialog instead of leaking to
    // the underlying page.
    overlay.focus();
  }

  function bindOne(node) {
    if (!node || !node.dataset || node.dataset[DATA_FLAG]) {
      return;
    }
    if (node.closest && node.closest("#" + OVERLAY_ID)) {
      // Don't bind the cloned image inside the overlay.
      return;
    }
    node.dataset[DATA_FLAG] = "1";
    node.style.cursor = "zoom-in";
    node.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
      openOverlay(node);
    });
  }

  function bindAll() {
    var nodes = document.querySelectorAll(SELECTOR);
    for (var i = 0; i < nodes.length; i++) {
      bindOne(nodes[i]);
    }
  }

  // Coalesce MutationObserver fires via requestAnimationFrame so a
  // hydration burst that triggers many subtree mutations in one frame
  // runs `bindAll()` (and its full querySelectorAll) only once per
  // frame. Mirrors the pattern in colab-launch.js.
  var bindScheduled = false;
  function scheduleBind() {
    if (bindScheduled) return;
    bindScheduled = true;
    window.requestAnimationFrame(function () {
      bindScheduled = false;
      bindAll();
    });
  }

  function init() {
    scheduleBind();
    if (typeof MutationObserver === "function") {
      var observer = new MutationObserver(scheduleBind);
      // Observe documentElement (the <html> root), not document.body.
      // mystmd's React hydration can replace document.body wholesale
      // when the SSR HTML doesn't match what React expects (a known
      // failure mode triggered by the very script tag we inject into
      // <head>: SSR includes it, hydrated React tree does not, so
      // React emits the "hydration recovered by client rendering"
      // warning and re-renders the body). An observer attached to
      // the old body becomes orphaned the moment that happens.
      // documentElement is stable across that recovery.
      observer.observe(document.documentElement, {
        childList: true,
        subtree: true,
      });
    }
    // SPA navigation hooks. mystmd is a React SPA; Back/Forward and
    // popstate-driven route changes can swap cell-output content
    // without firing DOMContentLoaded again. Re-running scheduleBind
    // on each of these covers those navigations regardless of which
    // event the SPA dispatches.
    window.addEventListener("popstate", scheduleBind);
    window.addEventListener("hashchange", scheduleBind);
    window.addEventListener("pageshow", scheduleBind);
  }

  // `pageshow` fires for both initial load and bfcache restore, so it
  // also covers the "user hit back" case once we have already wired
  // it inside init(). For the initial load we still need the
  // DOMContentLoaded / immediate path to bring the observer up.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
