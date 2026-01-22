# Makefile for Qamomile Documentation (docs2)
# Uses jupytext to convert .py -> .ipynb and Jupyter Book to build

.PHONY: help build build-en build-ja sync sync-en sync-ja clean clean-all serve-en serve-ja build-site serve-site

# Default target
help:
	@echo "Qamomile Documentation Build System"
	@echo "===================================="
	@echo ""
	@echo "Available targets:"
	@echo "  make build       - Build both English and Japanese documentation"
	@echo "  make build-en    - Build English documentation only"
	@echo "  make build-ja    - Build Japanese documentation only"
	@echo "  make sync        - Convert all .py files to .ipynb (both languages)"
	@echo "  make sync-en     - Convert English .py files to .ipynb"
	@echo "  make sync-ja     - Convert Japanese .py files to .ipynb"
	@echo "  make clean       - Remove generated .ipynb files and build outputs"
	@echo "  make clean-all   - Remove everything including execution cache"
	@echo "  make serve-en    - Serve English docs locally (port 8000)"
	@echo "  make serve-ja    - Serve Japanese docs locally (port 8000)"
	@echo "  make build-site  - Build unified site for GitHub Pages"
	@echo "  make serve-site  - Serve unified site locally (port 8000)"
	@echo ""

# Build both languages
build: sync build-en build-ja
	@echo "✓ Both English and Japanese documentation built successfully"

# Build English documentation
build-en: sync-en
	@echo "Building English documentation..."
	cd en && uv run jupyter-book build --html
	@echo "✓ English documentation built: en/_build/html/index.html"

# Build Japanese documentation
build-ja: sync-ja
	@echo "Building Japanese documentation..."
	cd ja && uv run jupyter-book build --html
	@echo "✓ Japanese documentation built: ja/_build/html/index.html"

# Sync all .py files to .ipynb (both languages)
sync: sync-en sync-ja
	@echo "✓ All Python scripts converted to notebooks"

# Sync English .py files to .ipynb
sync-en:
	@echo "Converting English .py files to .ipynb..."
	@uv run jupytext --to ipynb en/tutorial/*.py 2>/dev/null || true
	@uv run jupytext --to ipynb en/transpile/*.py 2>/dev/null || true
	@echo "✓ English notebooks synced"

# Sync Japanese .py files to .ipynb
sync-ja:
	@echo "Converting Japanese .py files to .ipynb..."
	@uv run jupytext --to ipynb ja/tutorial/*.py 2>/dev/null || true
	@uv run jupytext --to ipynb ja/transpile/*.py 2>/dev/null || true
	@echo "✓ Japanese notebooks synced"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@rm -f en/tutorial/*.ipynb
	@rm -f en/transpile/*.ipynb
	@rm -f ja/tutorial/*.ipynb
	@rm -f ja/transpile/*.ipynb
	@rm -rf en/_build
	@rm -rf ja/_build
	@rm -rf _site
	@echo "✓ Cleaned generated .ipynb files and build outputs"

# Clean everything including cache
clean-all: clean
	@echo "Cleaning execution cache..."
	@rm -rf en/_build/.jupyter_cache
	@rm -rf ja/_build/.jupyter_cache
	@echo "✓ All generated files and cache removed"

# Serve English documentation locally
serve-en:
	@if [ ! -d "en/_build/html" ]; then \
		echo "Error: English documentation not built. Run 'make build-en' first."; \
		exit 1; \
	fi
	@echo "Serving English documentation at http://localhost:8000"
	@echo "Press Ctrl+C to stop the server"
	@cd en/_build/html && uv run python -m http.server 8000

# Serve Japanese documentation locally
serve-ja:
	@if [ ! -d "ja/_build/html" ]; then \
		echo "Error: Japanese documentation not built. Run 'make build-ja' first."; \
		exit 1; \
	fi
	@echo "Serving Japanese documentation at http://localhost:8000"
	@echo "Press Ctrl+C to stop the server"
	@cd ja/_build/html && uv run python -m http.server 8000

# Unified site output directory
SITE_DIR := _site

# Build unified site for GitHub Pages
build-site: build
	@echo "Creating unified site structure..."
	@rm -rf $(SITE_DIR)
	@mkdir -p $(SITE_DIR)/en
	@mkdir -p $(SITE_DIR)/ja
	@cp -r en/_build/html/* $(SITE_DIR)/en/
	@cp -r ja/_build/html/* $(SITE_DIR)/ja/
	@cp index.html $(SITE_DIR)/ 2>/dev/null || echo "Creating default index.html..."
	@if [ ! -f $(SITE_DIR)/index.html ]; then \
		echo '<!DOCTYPE html><html><head><meta http-equiv="refresh" content="0;url=en/"></head></html>' > $(SITE_DIR)/index.html; \
	fi
	@echo "✓ Unified site created: $(SITE_DIR)/"

# Serve unified site locally
serve-site: build-site
	@echo "Serving unified site at http://localhost:8000"
	@echo "Press Ctrl+C to stop the server"
	@cd $(SITE_DIR) && uv run python -m http.server 8000
