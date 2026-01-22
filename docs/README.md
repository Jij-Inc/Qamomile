# Qamomile Documentation

This directory contains the documentation for Qamomile, built with **Jupyter Book 2** and **jupytext**.

## Overview

This documentation system uses a modern workflow where:

- **Source files**: Python scripts (`.py`) with `# %% [markdown]` format (jupytext percent format)
- **Build-time conversion**: Automatically converted to Jupyter Notebooks (`.ipynb`) during build
- **Execution caching**: Notebook outputs are cached for fast rebuilds
- **Version control**: Only `.py` files are committed to git; `.ipynb` files are auto-generated

### Why This Approach?

1. **Git-friendly**: Python scripts are easier to diff, merge, and review than JSON notebooks
2. **IDE support**: Edit with full Python IDE features (linting, type checking, refactoring)
3. **Automatic sync**: No manual conversion needed - build process handles everything
4. **Performance**: Execution caching means notebooks only re-run when code changes
5. **Jupyter Book 2.x**: Access to latest features and improvements

## Directory Structure

```
docs/
├── README.md                    # This file
├── Makefile                     # Main build system
├── build.sh                     # Alternative build script (for non-Make environments)
├── .gitignore                   # Ignores generated .ipynb files and caches
├── assets/                      # Shared images and resources
│   └── qamomile_logo.png        # Project logo
├── en/                          # English documentation
│   ├── myst.yml                 # Jupyter Book 2 configuration and TOC
│   ├── _build/                  # Build output (gitignored)
│   │   ├── html/                # Generated HTML documentation
│   │   └── .jupyter_cache/      # Execution cache (English)
│   ├── index.md                 # Landing page
│   ├── tutorial/
│   │   ├── qaoa.py              # Source: QAOA tutorial (percent format)
│   │   ├── qaoa.ipynb           # Auto-generated (gitignored)
│   │   ├── qpe.py               # Source: QPE tutorial (percent format)
│   │   └── qpe.ipynb            # Auto-generated (gitignored)
│   └── transpile/
│       ├── transpile_flow.py    # Source: Transpiler flow (percent format)
│       └── transpile_flow.ipynb # Auto-generated (gitignored)
└── ja/                          # Japanese documentation (same structure as en/)
    ├── myst.yml                 # Jupyter Book 2 configuration and TOC
    ├── _build/
    │   ├── html/
    │   └── .jupyter_cache/
    ├── index.md
    ├── tutorial/
    └── transpile/
```

## Quick Start

### Prerequisites

Ensure you have the development dependencies installed:

```bash
uv sync
```

This installs `jupyter-book`, `jupytext`, and other required packages.

### Building Documentation

#### Build all languages (English + Japanese):

```bash
make build
```

Or using the shell script:

```bash
./build.sh build
```

#### Build only English:

```bash
make build-en
```

#### Build only Japanese:

```bash
make build-ja
```

#### Clean generated files:

```bash
make clean
```

This removes:
- All generated `.ipynb` files
- Build output in `en/_build/` and `ja/_build/`
- Execution caches

#### Sync notebooks without building:

```bash
make sync
```

This converts `.py` → `.ipynb` without running Jupyter Book build.

### Viewing Documentation Locally

After building, you can view the documentation in your browser:

#### English version:

```bash
make serve-en
```

Then open: http://localhost:8000

#### Japanese version:

```bash
make serve-ja
```

Then open: http://localhost:8000

## Development Workflow

### Writing/Editing Documentation

1. **Edit Python files** (`.py`) in the appropriate directory:
   - English: `docs/en/tutorial/*.py` or `docs/en/transpile/*.py`
   - Japanese: `docs/ja/tutorial/*.py` or `docs/ja/transpile/*.py`

2. **Use jupytext percent format**:
   ```python
   # %% [markdown]
   # # Section Title
   # This is markdown content
   
   # %%
   import qamomile.circuit as qmc
   
   # Your Python code here
   ```

3. **Build and preview**:
   ```bash
   make build-en  # or build-ja for Japanese
   make serve-en  # or serve-ja for Japanese
   ```

4. **Commit only `.py` files** - DO NOT commit `.ipynb` files

### Creating New Pages

1. Create a new `.py` file with jupytext header:

   ```python
   # ---
   # jupyter:
   #   jupytext:
   #     cell_metadata_filter: -all
   #     text_representation:
   #       extension: .py
   #       format_name: percent
   #       format_version: '1.3'
   #       jupytext_version: 1.18.1
   #   kernelspec:
   #     display_name: qamomile
   #     language: python
   #     name: qamomile
   # ---
   
   # %% [markdown]
   # # Your Page Title
   # Content goes here...
   ```

2. Add the page to the `toc:` section in `myst.yml`:
   ```yaml
   toc:
     - file: your_folder/your_page
   ```

3. Build and verify:
   ```bash
   make build-en
   ```

### Jupyter Execution Cache

The system uses Jupyter's execution cache to avoid re-running unchanged notebooks:

- **Cache location**: `docs/_build/.jupyter_cache/` (shared across en/ja)
- **Behavior**: Notebooks only re-execute if their code changes
- **Clean cache**: `make clean` removes the cache

This makes rebuilds fast while ensuring outputs are always up-to-date.

## Jupytext Format

All Python source files use the **percent format**:

```python
# %% [markdown]
# Markdown cells start with this marker

# %%
# Python code cells start with this marker
print("Hello, Quantum World!")

# %% [markdown]
# You can mix markdown and code cells freely
```

### Key Features:

- **Metadata in header**: Jupytext and kernel configuration at top of file
- **Cell markers**: `# %%` for code, `# %% [markdown]` for text
- **Clean diffs**: Git sees normal Python file changes
- **IDE friendly**: Full Python tooling support

## Configuration Files

### `myst.yml` (Jupyter Book 2 Configuration)

Jupyter Book 2 uses a single `myst.yml` file that combines configuration and table of contents. Key sections:

```yaml
version: 1

project:
  title: Qamomile Documentation

  # Jupyter notebook execution settings
  jupyter:
    execute:
      - pattern: "*.ipynb"

  # Table of contents
  toc:
    - file: index
    - file: tutorial/qaoa
      title: QAOA Tutorial
    - file: tutorial/qpe
      title: QPE Tutorial
    - file: transpile/transpile_flow
      title: Transpiler Flow
```

This unified format replaces the separate `_config.yml` and `_toc.yml` files from Jupyter Book 1.

## Troubleshooting

### Problem: "No module named 'qamomile'"

**Solution**: Ensure you're in the project environment and qamomile is installed:

```bash
uv sync
```

### Problem: Notebooks not updating after code changes

**Solution**: Clear the execution cache:

```bash
make clean
make build
```

### Problem: Build errors about missing `.ipynb` files

**Solution**: Run the sync step first:

```bash
make sync
```

### Problem: Port 8000 already in use

**Solution**: Kill the existing server or use a different port:

```bash
# Kill existing server
pkill -f "python -m http.server"

# Or manually serve on different port
cd _build/html/en && python -m http.server 8001
```

### Problem: Jupyter Book not found

**Solution**: Install development dependencies:

```bash
uv sync
```

## About Jupyter Book 2

This documentation uses **Jupyter Book 2**, which is built on top of the MyST Document Engine. Key features:

- **Single config file**: `myst.yml` contains both configuration and table of contents
- **Python-based**: No Node.js/npm dependencies required
- **Notebook execution**: Built-in Jupyter notebook execution and caching
- **MyST Markdown**: Full support for MyST-flavored Markdown with directives and roles

Jupyter Book 2 wraps the MyST Document Engine with a Python CLI (`jupyter-book`), making it easy to install via pip/uv without needing npm.

## Contributing

When contributing documentation:

1. **Edit only `.py` files** - Never edit `.ipynb` directly
2. **Test your changes**: Run `make build-en` or `make build-ja`
3. **Check outputs**: Use `make serve-en` or `make serve-ja` to preview
4. **Commit only source**: Git should only see changes to `.py`, `.md`, or `myst.yml`

## Additional Resources

- [Jupyter Book Documentation](https://jupyterbook.org/)
- [Jupytext Documentation](https://jupytext.readthedocs.io/)
- [MyST Markdown Guide](https://myst-parser.readthedocs.io/)
- [Qamomile GitHub](https://github.com/Jij-Inc/Qamomile)

## License

Same license as the main Qamomile project.
