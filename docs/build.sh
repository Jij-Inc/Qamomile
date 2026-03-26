#!/usr/bin/env bash
# Build script for Qamomile Documentation (docs2)
# Alternative to Makefile for non-Make environments

set -e  # Exit on error

# Move to the script's directory to ensure relative paths work
cd "$(dirname "$0")"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
info() {
    echo -e "${GREEN}✓${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
    exit 1
}

warn() {
    echo -e "${YELLOW}!${NC} $1"
}

# Function to show help
show_help() {
    echo "Qamomile Documentation Build System"
    echo "===================================="
    echo ""
    echo "Usage: ./build.sh [command]"
    echo ""
    echo "Available commands:"
    echo "  build       - Build both English and Japanese documentation"
    echo "  build-en    - Build English documentation only"
    echo "  build-ja    - Build Japanese documentation only"
    echo "  sync        - Convert all .py files to .ipynb (both languages)"
    echo "  sync-en     - Convert English .py files to .ipynb"
    echo "  sync-ja     - Convert Japanese .py files to .ipynb"
    echo "  clean       - Remove generated .ipynb files and build outputs"
    echo "  clean-all   - Remove everything including execution cache"
    echo "  serve-en    - Build (if needed) and serve English docs (port 8000)"
    echo "  serve-ja    - Build (if needed) and serve Japanese docs (port 8000)"
    echo "  fresh-en    - Clean, rebuild with execution, and serve English docs"
    echo "  fresh-ja    - Clean, rebuild with execution, and serve Japanese docs"
    echo "  help        - Show this help message"
    echo ""
}

# Function to generate API reference
generate_api() {
    echo "Generating API reference..."
    uv run python generate_api.py
    info "API reference generated"
}

# Function to check if running on ReadTheDocs
is_rtd() {
    [ "${READTHEDOCS:-}" = "True" ] || [ -n "${READTHEDOCS_CANONICAL_URL:-}" ]
}

# Function to build documentation with optional BASE_URL for ReadTheDocs
build_with_optional_base_url() {
    local lang="$1"

    if is_rtd; then
        local base_url
        info "Read the Docs detected. Using BASE_URL=${base_url}"
        BASE_URL="${READTHEDOCS_CANONICAL_URL%/}/${lang}" MPLBACKEND=agg uv run jupyter-book build --html
    else
        MPLBACKEND=agg uv run jupyter-book build --html
    fi
}

# Function to copy API reference to language directories
copy_api() {
    echo "Copying API reference to language directories..."
    cp -r api/ en/api/
    cp -r api/ ja/api/
    info "API reference copied"
}

# Function to sync English notebooks
sync_en() {
    echo "Converting English .py files to .ipynb..."
    uv run jupytext --to ipynb en/tutorial/*.py 2>/dev/null || true
    uv run jupytext --to ipynb en/optimization/*.py 2>/dev/null || true
    uv run jupytext --to ipynb en/transpile/*.py 2>/dev/null || true
    info "English notebooks synced"
}

# Function to sync Japanese notebooks
sync_ja() {
    echo "Converting Japanese .py files to .ipynb..."
    uv run jupytext --to ipynb ja/tutorial/*.py 2>/dev/null || true
    uv run jupytext --to ipynb ja/optimization/*.py 2>/dev/null || true
    uv run jupytext --to ipynb ja/transpile/*.py 2>/dev/null || true
    info "Japanese notebooks synced"
}

# Function to sync all notebooks
sync_all() {
    sync_en
    sync_ja
    info "All Python scripts converted to notebooks"
}

# Function to build English documentation
build_en() {
    sync_en
    echo "Building English documentation..."
    cd en
    build_with_optional_base_url en
    cd ..
    # uv run python scripts/inject_colab_launch.py en
    info "English documentation built: en/_build/html/index.html"
}

# Function to build Japanese documentation
build_ja() {
    sync_ja
    echo "Building Japanese documentation..."
    cd ja
    build_with_optional_base_url ja
    cd ..
    # uv run python scripts/inject_colab_launch.py ja
    info "Japanese documentation built: ja/_build/html/index.html"
}

# Function to build all documentation
build_all() {
    generate_api
    copy_api
    build_en
    build_ja
    info "Both English and Japanese documentation built successfully"
}

# Function to clean generated files
clean() {
    echo "Cleaning generated files..."
    rm -f en/tutorial/*.ipynb
    rm -f en/optimization/*.ipynb
    rm -f ja/tutorial/*.ipynb
    rm -f en/transpile/*.ipynb
    rm -f ja/optimization/*.ipynb
    rm -f ja/transpile/*.ipynb
    rm -rf en/_build
    rm -rf ja/_build
    rm -rf en/api
    rm -rf ja/api
    info "Cleaned generated .ipynb files and build outputs"
}

# Function to clean everything including cache
clean_all() {
    clean
    echo "Cleaning execution cache..."
    rm -rf en/_build/.jupyter_cache
    rm -rf ja/_build/.jupyter_cache
    info "All generated files and cache removed"
}

# Function to serve English documentation (builds if needed)
serve_en() {
    if [ ! -d "en/_build/html" ]; then
        warn "English documentation not built. Building now..."
        generate_api
        copy_api
        build_en
    fi
    echo "Serving English documentation at http://localhost:8000"
    echo "Press Ctrl+C to stop the server"
    cd en/_build/html
    uv run python -m http.server 8000
}

# Function to serve Japanese documentation (builds if needed)
serve_ja() {
    if [ ! -d "ja/_build/html" ]; then
        warn "Japanese documentation not built. Building now..."
        generate_api
        copy_api
        build_ja
    fi
    echo "Serving Japanese documentation at http://localhost:8000"
    echo "Press Ctrl+C to stop the server"
    cd ja/_build/html
    uv run python -m http.server 8000
}

# Function to clean, rebuild, and serve English documentation
fresh_en() {
    clean
    generate_api
    copy_api
    build_en
    echo "Serving English documentation at http://localhost:8000"
    echo "Press Ctrl+C to stop the server"
    cd en/_build/html
    uv run python -m http.server 8000
}

# Function to clean, rebuild, and serve Japanese documentation
fresh_ja() {
    clean
    generate_api
    copy_api
    build_ja
    echo "Serving Japanese documentation at http://localhost:8000"
    echo "Press Ctrl+C to stop the server"
    cd ja/_build/html
    uv run python -m http.server 8000
}

# Main command dispatcher
case "${1:-help}" in
    build)
        build_all
        ;;
    build-en)
        build_en
        ;;
    build-ja)
        build_ja
        ;;
    sync)
        sync_all
        ;;
    sync-en)
        sync_en
        ;;
    sync-ja)
        sync_ja
        ;;
    clean)
        clean
        ;;
    clean-all)
        clean_all
        ;;
    serve-en)
        serve_en
        ;;
    serve-ja)
        serve_ja
        ;;
    fresh-en)
        fresh_en
        ;;
    fresh-ja)
        fresh_ja
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        error "Unknown command: $1. Run './build.sh help' for available commands."
        ;;
esac
