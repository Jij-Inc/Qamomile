#!/usr/bin/env bash
# Build script for Qamomile Documentation
# Alternative to Makefile for non-Make environments

set -e  # Exit on error

# Move to the script's directory to ensure relative paths work
cd "$(dirname "$0")"

# Languages and tutorial directories (excluding collaboration/)
LANGS=(en ja)
TUTORIAL_DIRS=(tutorial optimization vqa)

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
    echo "  sync-build     - Sync and build both languages"
    echo "  sync-build-en  - Sync and build English documentation"
    echo "  sync-build-ja  - Sync and build Japanese documentation"
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

copy_api() {
    echo "Copying API reference to language directories..."
    for lang in "${LANGS[@]}"; do
        cp -r api/ "${lang}/api/"
    done
    info "API reference copied"
}

# Sync .py -> .ipynb for a single language
sync_lang() {
    local lang="$1"
    echo "Converting ${lang} .py files to .ipynb..."
    for dir in "${TUTORIAL_DIRS[@]}"; do
        uv run jupytext --to ipynb "${lang}/${dir}"/*.py 2>/dev/null || true
    done
    # We don't convert collaboration/ because those notebooks need API-KEYs.
    info "${lang} notebooks synced"
}

# Build documentation for a single language (no sync)
build_lang() {
    local lang="$1"
    echo "Building ${lang} documentation..."
    cd "$lang"
    if is_rtd; then
        local base_url="${READTHEDOCS_CANONICAL_URL%/}/${lang}"
        info "Read the Docs detected. Using BASE_URL=${base_url}"
        BASE_URL="$base_url" MPLBACKEND=agg uv run jupyter-book build --html
    else
        MPLBACKEND=agg uv run jupyter-book build --html
    fi
    cd ..
    uv run python scripts/inject_colab_launch.py "$lang"
    info "${lang} documentation built: ${lang}/_build/html/index.html"
}

# Sync and build documentation for a single language
sync_build_lang() {
    local lang="$1"
    sync_lang "$lang"
    build_lang "$lang"
}

sync_build_all() {
    generate_api
    copy_api
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
    build_lang en
    build_lang ja
    info "Both English and Japanese documentation built successfully"
}

clean() {
    echo "Cleaning generated files..."
    for lang in "${LANGS[@]}"; do
        for dir in "${TUTORIAL_DIRS[@]}"; do
            rm -f "${lang}/${dir}"/*.ipynb
        done
        rm -rf "${lang}/_build"
        rm -rf "${lang}/api"
    done
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
