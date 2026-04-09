/*
 * ReadTheDocs search integration for MyST book-theme.
 *
 * The MyST book-theme search bar shows "Search is not enabled for this site"
 * on static HTML hosts because it requires a running Remix server.
 * This script intercepts clicks on the MyST search bar and opens the
 * ReadTheDocs server-side search modal instead.
 *
 * Only activates when ReadTheDocs addons are present (i.e. on RTD-hosted sites).
 * Remove this script once MyST book-theme supports static-site search natively.
 */
(() => {
  "use strict";

  let rtdReady = false;

  function isOnReadTheDocs() {
    return (
      document.querySelector("readthedocs-flyout") !== null ||
      document.querySelector("readthedocs-notification") !== null ||
      typeof window.READTHEDOCS_DATA !== "undefined"
    );
  }

  function openRtdSearch(e) {
    e.preventDefault();
    e.stopPropagation();
    e.stopImmediatePropagation();
    document.dispatchEvent(new CustomEvent("readthedocs-search-show"));
  }

  function bindSearchButtons() {
    document.querySelectorAll(".myst-search-bar").forEach((btn) => {
      if (btn.dataset.rtdSearchBound) return;
      btn.dataset.rtdSearchBound = "true";
      btn.addEventListener("click", openRtdSearch, { capture: true });
    });
  }

  function activate() {
    if (rtdReady) return;
    if (!isOnReadTheDocs()) return;
    rtdReady = true;
    bindSearchButtons();

    // Re-bind after SPA navigation (MyST uses client-side routing)
    const observer = new MutationObserver(() => {
      window.requestAnimationFrame(bindSearchButtons);
    });
    observer.observe(document.documentElement, {
      childList: true,
      subtree: true,
    });
  }

  // RTD addons load asynchronously — poll briefly then give up
  let attempts = 0;
  const maxAttempts = 20;
  const interval = setInterval(() => {
    attempts++;
    activate();
    if (rtdReady || attempts >= maxAttempts) {
      clearInterval(interval);
    }
  }, 500);

  // Also try on common lifecycle events
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", activate, { once: true });
  } else {
    activate();
  }
  window.addEventListener("load", activate, { once: true });
})();
