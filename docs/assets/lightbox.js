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
// navigations via a MutationObserver on document.body. Each target
// node is tagged with `data-qamomileLightbox="1"` once bound so
// repeated MutationObserver fires do not stack handlers.
(function () {
  "use strict";

  // We intentionally exclude figures that already live inside a
  // lightbox overlay — those are the cloned previews and must not be
  // re-bound recursively.
  var SELECTOR =
    ".cell-output img, .cell-output svg, " +
    ".notebook-output img, .notebook-output svg, " +
    "article figure img, article figure svg";

  var OVERLAY_ID = "qamomile-lightbox-overlay";
  var DATA_FLAG = "qamomileLightbox";

  function closeOverlay() {
    var overlay = document.getElementById(OVERLAY_ID);
    if (overlay) {
      overlay.remove();
    }
    document.removeEventListener("keydown", onEscape, true);
  }

  function onEscape(event) {
    if (event.key === "Escape") {
      closeOverlay();
    }
  }

  function openOverlay(node) {
    closeOverlay(); // ensure only one overlay at a time
    var overlay = document.createElement("div");
    overlay.id = OVERLAY_ID;
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
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
    document.addEventListener("keydown", onEscape, true);
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

  function init() {
    bindAll();
    if (typeof MutationObserver === "function") {
      var observer = new MutationObserver(function () {
        // Coalesce — bindAll is idempotent thanks to the data flag.
        bindAll();
      });
      observer.observe(document.body, {
        childList: true,
        subtree: true,
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
