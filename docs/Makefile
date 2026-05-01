# Makefile for Qamomile Documentation
# Uses jupytext to convert .py -> .ipynb and Jupyter Book to build

# Languages and tutorial directories
LANGS := en ja
# integration is excluded because those notebooks may require API keys
# and can't be automatically synced/executed.
# release_notes is excluded because it is markdown-only; nothing to sync or execute.
TARGET_DIRS := tutorial algorithm usage

.PHONY: help build build-en build-ja sync sync-en sync-ja execute execute-en execute-ja sync-build sync-build-en sync-build-ja generate-api copy-api clean clean-all serve-en serve-ja fresh-en fresh-ja build-site serve-site

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
	@echo "  make execute        - Execute all .ipynb notebooks (both languages)"
	@echo "  make execute-en     - Execute English .ipynb notebooks"
	@echo "  make execute-ja     - Execute Japanese .ipynb notebooks"
	@echo "  make sync-build     - Sync, execute, and build both languages"
	@echo "  make sync-build-en  - Sync, execute, and build English documentation"
	@echo "  make sync-build-ja  - Sync, execute, and build Japanese documentation"
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
	@for lang in $(LANGS); do mkdir -p $$lang/api; cp -r api/. $$lang/api/; done
	@echo "✓ API reference copied to language directories"

# Auto-managed regions (chip blocks, browse-by-tag clouds, per-tag
# pages) are generated into a build-dir copy of the docs tree, never
# into the committed source. The actual injection runs inside
# build.sh's setup_build_src; this target is a thin wrapper for
# ad-hoc invocation against the working tree.
generate-doc-tags:
	@echo "Generating doc tag pages (in-place; usually only useful for previewing)..."
	@uv run python scripts/build_doc_tags.py
	@echo "✓ Doc tag pages generated"

# Sync .py -> .ipynb for a single language (used via sync-en, sync-ja)
# Includes integration/ even though TARGET_DIRS excludes it: jupytext
# only rewrites cell sources (no execution), so it has no API-key
# dependency. Excluding integration/ here would leave its .ipynb stale
# whenever a contributor edits only the .py and runs ``make sync``.
# execute-* still excludes integration/ because executing it needs API
# keys.
define sync-lang
	@echo "Converting $(1) .py files to .ipynb..."
	@for dir in $(TARGET_DIRS) integration; do \
		py_files=$$(find $(1)/$$dir -maxdepth 1 -name '*.py' 2>/dev/null); \
		if [ -z "$$py_files" ]; then \
			echo "! No .py files in $(1)/$$dir, skipping"; \
			continue; \
		fi; \
		uv run jupytext --to ipynb $$py_files; \
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

# Execute .ipynb notebooks for a single language
define execute-lang
	@echo "Executing $(1) notebooks..."
	@for dir in $(TARGET_DIRS); do \
		for nb in $(1)/$$dir/*.ipynb; do \
			[ -f "$$nb" ] || continue; \
			echo "✓ Executing $$nb..."; \
			uv run jupyter nbconvert --to notebook --execute --inplace "$$nb"; \
		done; \
	done
	@echo "✓ $(1) notebooks executed"
endef

# Execute targets
execute-en:
	$(call execute-lang,en)

execute-ja:
	$(call execute-lang,ja)

execute: execute-en execute-ja
	@echo "✓ All notebooks executed"

# Build targets — delegate to build.sh so the build-dir setup, tag-page
# injection, and post-build colab-launch step stay defined in one place.
# build.sh handles _build_src/ copy, build_doc_tags.py injection,
# jupytext --update, mystmd build, and the final cp back to <lang>/_build/.
build-en:
	@./build.sh build-en

build-ja:
	@./build.sh build-ja

build:
	@./build.sh build

# Sync + execute + build targets
sync-build-en: sync-en execute-en build-en

sync-build-ja: sync-ja execute-ja build-ja

sync-build: copy-api sync-build-en sync-build-ja
	@echo "✓ Both English and Japanese documentation synced and built successfully"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@for lang in $(LANGS); do \
		for dir in $(TARGET_DIRS); do \
			rm -f $$lang/$$dir/*.ipynb; \
		done; \
		rm -rf $$lang/_build; \
		rm -rf $$lang/api; \
	done
	@rm -rf _build_src
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
