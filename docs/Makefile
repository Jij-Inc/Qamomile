# Makefile for Qamomile Documentation
# Uses jupytext to convert .py -> .ipynb and Jupyter Book to build

# Languages and tutorial directories (excluding collaboration/)
LANGS := en ja
TUTORIAL_DIRS := tutorial optimization vqa

.PHONY: help build build-en build-ja sync sync-en sync-ja sync-build sync-build-en sync-build-ja generate-api copy-api clean clean-all serve-en serve-ja fresh-en fresh-ja build-site serve-site

# Default target
help:
	@echo "Qamomile Documentation Build System"
	@echo "===================================="
	@echo ""
	@echo "Available targets:"
	@echo "  make build          - Build both languages (no sync)"
	@echo "  make build-en       - Build English documentation only (no sync)"
	@echo "  make build-ja       - Build Japanese documentation only (no sync)"
	@echo "  make sync           - Convert all .py files to .ipynb (both languages)"
	@echo "  make sync-en        - Convert English .py files to .ipynb"
	@echo "  make sync-ja        - Convert Japanese .py files to .ipynb"
	@echo "  make sync-build     - Sync and build both languages"
	@echo "  make sync-build-en  - Sync and build English documentation"
	@echo "  make sync-build-ja  - Sync and build Japanese documentation"
	@echo "  make clean          - Remove generated .ipynb files and build outputs"
	@echo "  make clean-all      - Remove everything including execution cache"
	@echo "  make serve-en       - Sync, build (if needed), and serve English docs (port 8000)"
	@echo "  make serve-ja       - Sync, build (if needed), and serve Japanese docs (port 8000)"
	@echo "  make fresh-en       - Clean, sync, rebuild, and serve English docs"
	@echo "  make fresh-ja       - Clean, sync, rebuild, and serve Japanese docs"
	@echo "  make build-site     - Build unified site for GitHub Pages"
	@echo "  make serve-site     - Serve unified site locally (port 8000)"
	@echo ""

# Generate API reference from docstrings
generate-api:
	@echo "Generating API reference..."
	@uv run python generate_api.py
	@echo "✓ API reference generated"

# Copy API reference to language directories
copy-api: generate-api
	@for lang in $(LANGS); do cp -r api/ $$lang/api/; done
	@echo "✓ API reference copied to language directories"

# Sync .py -> .ipynb for a single language (used via sync-en, sync-ja)
define sync-lang
	@echo "Converting $(1) .py files to .ipynb..."
	@for dir in $(TUTORIAL_DIRS); do \
		uv run jupytext --to ipynb $(1)/$$dir/*.py 2>/dev/null || true; \
	done
	@echo "✓ $(1) notebooks synced"
endef

# Sync targets
sync-en:
	$(call sync-lang,en)

sync-ja:
	$(call sync-lang,ja)

sync: sync-en sync-ja
	@echo "✓ All Python scripts converted to notebooks"

# Build targets (no sync)
build-en: copy-api
	@echo "Building English documentation..."
	cd en && MPLBACKEND=agg uv run jupyter-book build --html
	@uv run python scripts/inject_colab_launch.py en
	@echo "✓ English documentation built: en/_build/html/index.html"

build-ja: copy-api
	@echo "Building Japanese documentation..."
	cd ja && MPLBACKEND=agg uv run jupyter-book build --html
	@uv run python scripts/inject_colab_launch.py ja
	@echo "✓ Japanese documentation built: ja/_build/html/index.html"

build: copy-api build-en build-ja
	@echo "✓ Both English and Japanese documentation built successfully"

# Sync + build targets
sync-build-en: sync-en build-en

sync-build-ja: sync-ja build-ja

sync-build: copy-api sync-build-en sync-build-ja
	@echo "✓ Both English and Japanese documentation synced and built successfully"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@for lang in $(LANGS); do \
		for dir in $(TUTORIAL_DIRS); do \
			rm -f $$lang/$$dir/*.ipynb; \
		done; \
		rm -rf $$lang/_build; \
		rm -rf $$lang/api; \
	done
	@rm -rf _site
	@echo "✓ Cleaned generated .ipynb files and build outputs"

# Clean everything including cache
clean-all: clean
	@echo "Cleaning execution cache..."
	@for lang in $(LANGS); do \
		rm -rf $$lang/_build/.jupyter_cache; \
	done
	@echo "✓ All generated files and cache removed"

# Serve targets (sync + build if needed)
serve-en: sync-build-en
	@echo "Serving English documentation at http://localhost:8000"
	@echo "Press Ctrl+C to stop the server"
	@cd en/_build/html && uv run python -m http.server 8000

serve-ja: sync-build-ja
	@echo "Serving Japanese documentation at http://localhost:8000"
	@echo "Press Ctrl+C to stop the server"
	@cd ja/_build/html && uv run python -m http.server 8000

# Fresh targets (clean + sync + build + serve)
fresh-en: clean sync-build-en
	@echo "Serving English documentation at http://localhost:8000"
	@echo "Press Ctrl+C to stop the server"
	@cd en/_build/html && uv run python -m http.server 8000

fresh-ja: clean sync-build-ja
	@echo "Serving Japanese documentation at http://localhost:8000"
	@echo "Press Ctrl+C to stop the server"
	@cd ja/_build/html && uv run python -m http.server 8000

# Unified site output directory
SITE_DIR := _site

# Build unified site for GitHub Pages
build-site: sync-build
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
