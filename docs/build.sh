#!/usr/bin/env bash
# Build script for Qamomile Documentation
# Alternative to Makefile for non-Make environments

set -e  # Exit on error

# Move to the script's directory to ensure relative paths work
cd "$(dirname "$0")"

# Languages and target directories
LANGS=(en ja)
# integration is excluded because those notebooks may require API keys
# and can't be automatically synced/executed.
# release_notes is excluded because it is markdown-only; nothing to sync or execute.
TARGET_DIRS=(tutorial algorithm usage)

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}✓${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; exit 1; }
warn()  { echo -e "${YELLOW}!${NC} $1"; }

show_help() {
    echo "Qamomile Documentation Build System"
    echo "===================================="
    echo ""
    echo "Usage: ./build.sh [command]"
    echo ""
    echo "Available commands:"
    echo "  build          - Build both English and Japanese documentation (no sync)"
    echo "  build-en       - Build English documentation only (no sync)"
    echo "  build-ja       - Build Japanese documentation only (no sync)"
    echo "  sync           - Convert all .py files to .ipynb (both languages)"
    echo "  sync-en        - Convert English .py files to .ipynb"
    echo "  sync-ja        - Convert Japanese .py files to .ipynb"
    echo "  execute        - Execute all .ipynb notebooks (both languages)"
    echo "  execute-en     - Execute English .ipynb notebooks"
    echo "  execute-ja     - Execute Japanese .ipynb notebooks"
    echo "  sync-build     - Sync, execute, and build both languages"
    echo "  sync-build-en  - Sync, execute, and build English documentation"
    echo "  sync-build-ja  - Sync, execute, and build Japanese documentation"
    echo "  clean          - Remove generated .ipynb files and build outputs"
    echo "  clean-all      - Remove everything including execution cache"
    echo "  serve-en       - Sync, build (if needed), and serve English docs (port 8000)"
    echo "  serve-ja       - Sync, build (if needed), and serve Japanese docs (port 8000)"
    echo "  fresh-en       - Clean, sync, rebuild, and serve English docs"
    echo "  fresh-ja       - Clean, sync, rebuild, and serve Japanese docs"
    echo "  help           - Show this help message"
    echo ""
}

# Check if running on ReadTheDocs
is_rtd() {
    [ "${READTHEDOCS:-}" = "True" ] || [ -n "${READTHEDOCS_CANONICAL_URL:-}" ]
}

generate_api() {
    echo "Generating API reference..."
    uv run python generate_api.py
    info "API reference generated"
}

generate_doc_tags() {
    # Inject auto-managed regions (chip blocks, browse-by-tag clouds,
    # per-tag pages) into the build-dir copy of the docs tree pointed
    # at by $1. Defaults to the current dir when invoked without an
    # argument (back-compat for ad-hoc runs).
    local docs_root="${1:-$(pwd)}"
    echo "Generating doc tag indexes (docs root: ${docs_root})..."
    DOCS_ROOT_OVERRIDE="$docs_root" uv run python scripts/build_doc_tags.py
    info "Doc tag pages generated"
}

setup_build_src() {
    # Build everything inside docs/_build_src/ so the committed source
    # tree never receives auto-managed injections. Sequence:
    #   1. rm + recreate docs/_build_src/<lang>/ as a copy of docs/<lang>/
    #   2. run build_doc_tags.py against _build_src/ (injects chip
    #      blocks and browse-by-tag clouds; generates per-tag pages)
    #   3. jupytext --update so chip-injected .py propagates into .ipynb
    #      cells (preserves committed outputs). Includes integration/
    #      because jupytext only syncs cell sources — no execution, no
    #      API-key dependency.
    local build_src_root="$(pwd)/_build_src"
    local _lang _dir _py_files
    echo "Setting up _build_src/ ..."
    rm -rf "$build_src_root"
    mkdir -p "$build_src_root"
    for _lang in "${LANGS[@]}"; do
        # cp -R is portable across the Linux RTD image and local macOS
        # (rsync isn't installed in the RTD build env). Copy the whole
        # lang dir then prune the build artifacts that we never want
        # in the scratch tree.
        cp -R "${_lang}" "${build_src_root}/"
        rm -rf "${build_src_root}/${_lang}/_build"
    done
    # myst.yml inside each <lang>/ references ../assets/custom-theme.css
    # (the chip-pill CSS, the logo, the colab-launch JS). When mystmd
    # builds from _build_src/<lang>/, that ../assets/ resolves to
    # _build_src/assets/, so the assets dir has to exist there too.
    cp -R assets "${build_src_root}/"

    generate_doc_tags "$build_src_root"

    local sync_dirs=("${TARGET_DIRS[@]}" "integration")
    for _lang in "${LANGS[@]}"; do
        for _dir in "${sync_dirs[@]}"; do
            shopt -s nullglob
            _py_files=("${build_src_root}/${_lang}/${_dir}"/*.py)
            shopt -u nullglob
            [ ${#_py_files[@]} -eq 0 ] && continue
            uv run jupytext --to ipynb --update "${_py_files[@]}"
        done
    done

    info "_build_src/ ready"
}

copy_api() {
    echo "Copying API reference to language directories..."
    for lang in "${LANGS[@]}"; do
        mkdir -p "${lang}/api"
        cp -r api/. "${lang}/api/"
    done
    info "API reference copied"
}

# Sync .py -> .ipynb for a single language
sync_lang() {
    local lang="$1"
    echo "Converting ${lang} .py files to .ipynb..."
    # Include integration/ here even though TARGET_DIRS excludes it.
    # Rationale: jupytext only rewrites cell sources — it does not
    # execute the notebook, so syncing integration/ has no API-key
    # dependency. Without this include, ``./build.sh sync`` would
    # leave docs/<lang>/integration/*.ipynb stale relative to its
    # .py source whenever a contributor updated only the .py and ran
    # the sync target. This list must stay in lock-step with the
    # ``sync_dirs`` defined inside ``setup_build_src`` (which already
    # includes integration/ for the same reason).
    local sync_dirs=("${TARGET_DIRS[@]}" "integration")
    for dir in "${sync_dirs[@]}"; do
        local py_files=()
        shopt -s nullglob
        py_files=("${lang}/${dir}"/*.py)
        shopt -u nullglob
        if [ ${#py_files[@]} -eq 0 ]; then
            warn "No .py files in ${lang}/${dir}, skipping"
            continue
        fi
        uv run jupytext --to ipynb "${py_files[@]}"
    done
    # execute_lang still excludes integration/ because executing those
    # notebooks does need API keys.
    info "${lang} notebooks synced"
}

# Execute .ipynb notebooks for a single language
execute_lang() {
    local lang="$1"
    echo "Executing ${lang} notebooks..."
    for dir in "${TARGET_DIRS[@]}"; do
        for nb in "${lang}/${dir}"/*.ipynb; do
            [ -f "$nb" ] || continue
            info "Executing ${nb}..."
            uv run jupyter nbconvert --to notebook --execute --inplace "$nb"
        done
    done
    info "${lang} notebooks executed"
}

# Build a single language from docs/_build_src/<lang>/ (assumes
# setup_build_src already ran).
_build_lang_from_build_src() {
    local lang="$1"
    echo "Building ${lang} documentation..."
    cd "_build_src/${lang}"
    if is_rtd && [[ -n "${READTHEDOCS_VERSION:-}" ]]; then
        local base_url="/${READTHEDOCS_VERSION}/${lang}"
        info "Read the Docs detected. Using BASE_URL=${base_url}"
        BASE_URL="$base_url" uv run jupyter-book build --html
    else
        if is_rtd; then
            warn "Read the Docs detected but READTHEDOCS_VERSION is unset or empty; building without BASE_URL"
        fi
        uv run jupyter-book build --html
    fi
    cd ../..
    # Move the html output back to docs/<lang>/_build/ so the existing
    # post-build step (colab-launch injection) and the .readthedocs.yaml
    # copy step both find it where they used to.
    rm -rf "${lang}/_build"
    cp -r "_build_src/${lang}/_build" "${lang}/_build"
    uv run python scripts/inject_colab_launch.py "$lang"
    info "${lang} documentation built: ${lang}/_build/html/index.html"
}

# Build documentation for a single language (no sync). Public entry
# point — runs generate_api + copy_api + setup_build_src then builds
# just this lang. We always run the API generation pair so the build
# is self-contained: a contributor running ``./build.sh build-en`` on
# a fresh clone (where ``docs/api/`` is gitignored and absent) does
# not get a missing-toc-entry error from mystmd. The pair is fast and
# idempotent, so re-running them on every single-locale build is an
# acceptable cost; ``build_all`` calls them once up front and then
# delegates to ``_build_lang_from_build_src`` directly so we don't
# double-run.
build_lang() {
    local lang="$1"
    generate_api
    copy_api
    setup_build_src
    _build_lang_from_build_src "$lang"
}

# Sync, execute, and build documentation for a single language
sync_build_lang() {
    local lang="$1"
    sync_lang "$lang"
    execute_lang "$lang"
    build_lang "$lang"
}

sync_build_all() {
    # build_lang inside each sync_build_lang now runs generate_api +
    # copy_api itself (so single-locale builds are self-contained), so
    # we don't need to call them up front here.
    sync_build_lang en
    sync_build_lang ja
    info "Both English and Japanese documentation synced and built successfully"
}

# Serve documentation for a single language
serve_lang() {
    local lang="$1"
    if [ ! -d "${lang}/_build/html" ]; then
        warn "${lang} documentation not built. Syncing and building now..."
        generate_api
        copy_api
        sync_build_lang "$lang"
    fi
    echo "Serving ${lang} documentation at http://localhost:8000"
    echo "Press Ctrl+C to stop the server"
    cd "${lang}/_build/html"
    uv run python -m http.server 8000
}

# Clean, rebuild, and serve documentation for a single language
fresh_lang() {
    local lang="$1"
    clean
    generate_api
    copy_api
    sync_build_lang "$lang"
    echo "Serving ${lang} documentation at http://localhost:8000"
    echo "Press Ctrl+C to stop the server"
    cd "${lang}/_build/html"
    uv run python -m http.server 8000
}

build_all() {
    generate_api
    copy_api
    setup_build_src
    _build_lang_from_build_src en
    _build_lang_from_build_src ja
    info "Both English and Japanese documentation built successfully"
}

clean() {
    echo "Cleaning generated files..."
    for lang in "${LANGS[@]}"; do
        for dir in "${TARGET_DIRS[@]}"; do
            rm -f "${lang}/${dir}"/*.ipynb
        done
        rm -rf "${lang}/_build"
        rm -rf "${lang}/api"
    done
    rm -rf "_build_src"
    info "Cleaned generated .ipynb files and build outputs"
}

clean_all() {
    clean
    echo "Cleaning execution cache..."
    for lang in "${LANGS[@]}"; do
        rm -rf "${lang}/_build/.jupyter_cache"
    done
    info "All generated files and cache removed"
}

# Main command dispatcher
case "${1:-help}" in
    build)      build_all ;;
    build-en)   build_lang en ;;
    build-ja)   build_lang ja ;;
    sync)       sync_lang en; sync_lang ja
                info "All Python scripts converted to notebooks" ;;
    sync-en)    sync_lang en ;;
    sync-ja)    sync_lang ja ;;
    execute)    execute_lang en; execute_lang ja
                info "All notebooks executed" ;;
    execute-en) execute_lang en ;;
    execute-ja) execute_lang ja ;;
    sync-build)     sync_build_all ;;
    sync-build-en)  sync_build_lang en ;;
    sync-build-ja)  sync_build_lang ja ;;
    clean)      clean ;;
    clean-all)  clean_all ;;
    serve-en)   serve_lang en ;;
    serve-ja)   serve_lang ja ;;
    fresh-en)   fresh_lang en ;;
    fresh-ja)   fresh_lang ja ;;
    help|--help|-h) show_help ;;
    *)          error "Unknown command: $1. Run './build.sh help' for available commands." ;;
esac
