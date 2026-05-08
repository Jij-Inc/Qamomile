/*
 * Temporary workaround:
 * Jupyter Book 2 / MyST currently does not provide a built-in Google Colab launch button.
 * Remove this script once Colab launch is officially supported by the upstream theme/runtime.
 */
(() => {
  "use strict";

  const BUTTON_ID = "qamomile-colab-button";
  const SOURCE_REPO = "Jij-Inc/Qamomile";
  const SOURCE_BRANCH = "main";

  function extractNotebookPath(editHref) {
    try {
      const url = new URL(editHref, window.location.href);
      const match = url.pathname.match(/^\/[^/]+\/[^/]+\/edit\/[^/]+\/(.+\.ipynb)$/);
      if (!match) return null;
      const notebookPath = decodeURIComponent(match[1]);
      if (!/^docs\/(en|ja)\//.test(notebookPath)) return null;
      return notebookPath;
    } catch (_error) {
      return null;
    }
  }

  function buildColabUrl(notebookPath) {
    return `https://colab.research.google.com/github/${SOURCE_REPO}/blob/${SOURCE_BRANCH}/${notebookPath}`;
  }

  function createButton(colabUrl) {
    const button = document.createElement("a");
    button.id = BUTTON_ID;
    button.className = "qamomile-colab-button";
    button.href = colabUrl;
    button.target = "_blank";
    button.rel = "noopener noreferrer";
    button.title = "Open in Google Colab";
    button.setAttribute("aria-label", "Open in Google Colab");
    button.innerHTML =
      '<span class="qamomile-colab-icon" aria-hidden="true">C</span>' +
      '<span class="sr-only">Open in Google Colab</span>';
    return button;
  }

  function removeButton() {
    document.querySelectorAll(`#${BUTTON_ID}`).forEach((node) => node.remove());
  }

  function syncColabButton() {
    const header = document.querySelector(".myst-fm-block-header");
    if (!header) {
      removeButton();
      return;
    }

    const editLink = header.querySelector(".myst-fm-edit-link[href]");
    if (!editLink) {
      removeButton();
      return;
    }

    const notebookPath = extractNotebookPath(editLink.href);
    if (!notebookPath) {
      removeButton();
      return;
    }

    const colabUrl = buildColabUrl(notebookPath);
    let button = header.querySelector(`#${BUTTON_ID}`);
    if (!button) {
      button = createButton(colabUrl);
    }
    button.href = colabUrl;

    const launchButton = header.querySelector(".myst-fm-launch-button");
    const downloadsDropdown = header.querySelector(".myst-fm-downloads-dropdown");

    if (launchButton) {
      launchButton.insertAdjacentElement("afterend", button);
      return;
    }
    if (downloadsDropdown) {
      downloadsDropdown.insertAdjacentElement("afterend", button);
      return;
    }
    if (button.parentElement !== header) {
      header.appendChild(button);
    }
  }

  let syncScheduled = false;
  function scheduleSync() {
    if (syncScheduled) return;
    syncScheduled = true;
    window.requestAnimationFrame(() => {
      syncScheduled = false;
      syncColabButton();
    });
  }

  const observer = new MutationObserver(scheduleSync);
  observer.observe(document.documentElement, { childList: true, subtree: true });

  window.addEventListener("popstate", scheduleSync);
  window.addEventListener("hashchange", scheduleSync);
  window.addEventListener("pageshow", scheduleSync);

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", scheduleSync, { once: true });
  } else {
    scheduleSync();
  }
})();
