#!/usr/bin/env bash
# Build script for Qamomile Documentation (docs2)
# Alternative to Makefile for non-Make environments

set -e  # Exit on error

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
    echo "  serve-en    - Serve English docs locally (port 8000)"
    echo "  serve-ja    - Serve Japanese docs locally (port 8000)"
    echo "  help        - Show this help message"
    echo ""
}

# Function to sync English notebooks
sync_en() {
    echo "Converting English .py files to .ipynb..."
    uv run jupytext --to ipynb en/tutorial/*.py 2>/dev/null || true
    uv run jupytext --to ipynb en/transpile/*.py 2>/dev/null || true
    info "English notebooks synced"
}

# Function to sync Japanese notebooks
sync_ja() {
    echo "Converting Japanese .py files to .ipynb..."
    uv run jupytext --to ipynb ja/tutorial/*.py 2>/dev/null || true
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
    uv run jupyter-book build --html
    cd ..
    info "English documentation built: en/_build/html/index.html"
}

# Function to build Japanese documentation
build_ja() {
    sync_ja
    echo "Building Japanese documentation..."
    cd ja
    uv run jupyter-book build --html
    cd ..
    info "Japanese documentation built: ja/_build/html/index.html"
}

# Function to build all documentation
build_all() {
    sync_all
    build_en
    build_ja
    info "Both English and Japanese documentation built successfully"
}

# Function to clean generated files
clean() {
    echo "Cleaning generated files..."
    rm -f en/tutorial/*.ipynb
    rm -f en/transpile/*.ipynb
    rm -f ja/tutorial/*.ipynb
    rm -f ja/transpile/*.ipynb
    rm -rf en/_build
    rm -rf ja/_build
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

# Function to serve English documentation
serve_en() {
    if [ ! -d "en/_build/html" ]; then
        error "English documentation not built. Run './build.sh build-en' first."
    fi
    echo "Serving English documentation at http://localhost:8000"
    echo "Press Ctrl+C to stop the server"
    cd en/_build/html
    python -m http.server 8000
}

# Function to serve Japanese documentation
serve_ja() {
    if [ ! -d "ja/_build/html" ]; then
        error "Japanese documentation not built. Run './build.sh build-ja' first."
    fi
    echo "Serving Japanese documentation at http://localhost:8000"
    echo "Press Ctrl+C to stop the server"
    cd ja/_build/html
    python -m http.server 8000
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
    help|--help|-h)
        show_help
        ;;
    *)
        error "Unknown command: $1. Run './build.sh help' for available commands."
        ;;
esac
