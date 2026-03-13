# Qamomile Documentation

This directory contains the documentation for Qamomile, built with **Jupyter Book 2** and **jupytext**.

## Overview

This documentation system uses a modern workflow where:

- **Source files**: Python scripts (`.py`) with `# %% [markdown]` format (jupytext percent format)
- **Build-time conversion**: Automatically converted to Jupyter Notebooks (`.ipynb`) via jupytext
- **Execution caching**: Notebook outputs are cached for fast rebuilds
- **API reference**: Auto-generated from docstrings using `griffe`
- **Bilingual**: English (`en/`) and Japanese (`ja/`) documentation with identical structure

### Why This Approach?

1. **Git-friendly**: Python scripts are easier to diff, merge, and review than JSON notebooks
2. **IDE support**: Edit with full Python IDE features (linting, type checking, refactoring)
3. **Automatic sync**: Build process handles `.py` → `.ipynb` conversion
4. **Performance**: Execution caching means notebooks only re-run when code changes
5. **Jupyter Book 2.x**: Single `myst.yml` config, Python-based build (no Node.js)

## Directory Structure

```
docs/
├── README.md                    # This file
├── Makefile                     # Main build system
├── build.sh                     # Alternative build script (for non-Make environments)
├── generate_api.py              # API reference generator entry point
├── index.html                   # Language selector landing page (for GitHub Pages)
├── myst.yml                     # Top-level MyST stub
├── .gitignore                   # Ignores _build/, _site/, caches
│
├── assets/                      # Shared images and resources
│   ├── qamomile_logo.png        # Project logo
│   └── custom-theme.css         # Custom site theme
│
├── api_gen/                     # API doc generation package (uses griffe)
│   ├── __init__.py              # main() entry point
│   ├── config.py                # ApiGenConfig dataclass
│   ├── discovery.py             # Module discovery
│   ├── pages.py                 # Page generation
│   ├── toc.py                   # TOC generation
│   └── ...                      # Other helpers
│
├── api/                         # Generated API reference (shared source)
│   ├── index.md
│   ├── observable.md
│   ├── circuit/                 # Circuit module API (10 pages)
│   ├── optimization/            # Optimization module API (7 pages)
│   ├── qiskit/                  # Qiskit backend API (5 pages)
│   └── quri_parts/              # QuriParts backend API (6 pages)
│
├── en/                          # English documentation
│   ├── myst.yml                 # Jupyter Book 2 config + TOC
│   ├── index.md                 # Landing page
│   ├── _build/                  # Build output (gitignored)
│   ├── api/                     # Copied from docs/api/ at build time
│   ├── tutorial/                # 11 tutorials (jupytext .py sources)
│   │   ├── 01_introduction.py
│   │   ├── 02_type_system.py
│   │   ├── 03_gates.py
│   │   ├── 04_superposition_entanglement.py
│   │   ├── 05_stdlib.py
│   │   ├── 06_composite_gate.py
│   │   ├── 07_first_algorithm.py
│   │   ├── 08_parametric_circuits.py
│   │   ├── 09_resource_estimation.py
│   │   ├── 10_transpile.py
│   │   └── 11_custom_executor.py
│   └── optimization/            # Optimization tutorials
│       ├── qaoa.py
│       ├── fqaoa.py
│       ├── qrao31.py
│       └── custom_converter.py
│
└── ja/                          # Japanese documentation (mirrors en/ structure)
    ├── myst.yml
    ├── index.md
    ├── _build/
    ├── api/
    ├── tutorial/                # Japanese translations of all 11 tutorials
    └── optimization/            # Japanese translations of all 4 optimization tutorials
```

## Quick Start

### Prerequisites

Ensure you have the development dependencies installed:

```bash
uv sync
```

This installs `jupyter-book`, `jupytext`, `griffe`, and other required packages.

### Building Documentation

#### Build all languages (English + Japanese):

```bash
make build
```

This runs the full pipeline: generate API docs → copy to language dirs → sync `.py` → `.ipynb` → build HTML.

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

#### Clean and rebuild from scratch:

```bash
make fresh-en   # clean + build + serve English
make fresh-ja   # clean + build + serve Japanese
```

#### Clean generated files:

```bash
make clean
```

This removes:
- Generated `.ipynb` files in `en/` and `ja/`
- Build output in `en/_build/` and `ja/_build/`
- Copied API docs in `en/api/` and `ja/api/`
- Unified site output in `_site/`

#### Clean everything including execution cache:

```bash
make clean-all
```

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

## API Reference Generation

API reference pages are auto-generated from `qamomile` docstrings.

### How it works

1. `generate_api.py` uses the `api_gen/` package (built on `griffe`) to introspect the `qamomile` package
2. Generates MyST-compatible Markdown files into `docs/api/`
3. During build, `make copy-api` copies `docs/api/` → `en/api/` and `ja/api/`

### Regenerating API docs

```bash
make generate-api   # regenerate docs/api/ from docstrings
make copy-api       # copy to en/api/ and ja/api/ (also runs generate-api)
```

API regeneration is automatically included in `make build`, `make build-en`, and `make build-ja`.

## GitHub Pages Deployment

The documentation supports deployment as a unified bilingual site:

```bash
make build-site   # build unified _site/ with en/ and ja/ subdirectories
make serve-site   # build and serve locally at http://localhost:8000
```

The `_site/` directory structure:
```
_site/
├── index.html    # Language selector (redirects to en/ by default)
├── en/           # English HTML documentation
└── ja/           # Japanese HTML documentation
```

## Development Workflow

### Writing/Editing Documentation

1. **Edit Python files** (`.py`) in the appropriate directory:
   - English: `docs/en/tutorial/*.py` or `docs/en/optimization/*.py`
   - Japanese: `docs/ja/tutorial/*.py` or `docs/ja/optimization/*.py`

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

### Creating New Tutorial Pages

1. Create a new `.py` file following the numbered convention:

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

   Place it in the appropriate directory:
   - `en/tutorial/NN_topic_name.py` for tutorials
   - `en/optimization/topic_name.py` for optimization guides

2. Add the page to the `toc:` section in the corresponding `myst.yml`:
   ```yaml
   toc:
     - title: Tutorials
       children:
         - title: Section Name
           children:
             - file: tutorial/NN_topic_name.ipynb
   ```

3. Build and verify:
   ```bash
   make build-en
   ```

## Configuration Files

### `myst.yml` (Jupyter Book 2 Configuration)

Each language directory has its own `myst.yml` combining configuration and table of contents. Key structure:

```yaml
version: 1

project:
  title: Qamomile Documentation
  jupyter: true
  execute:
    cache: true

  toc:
    - file: index.md
    - title: Tutorials
      children:
        - title: Foundations
          children:
            - file: tutorial/01_introduction.ipynb
            - file: tutorial/02_type_system.ipynb
            - file: tutorial/03_gates.ipynb
            - file: tutorial/04_superposition_entanglement.ipynb
        - title: Standard Library & Algorithms
          children:
            - file: tutorial/05_stdlib.ipynb
            - ...
        - title: Advanced Topics
          children:
            - file: tutorial/09_resource_estimation.ipynb
            - ...
    - title: Optimization
      children:
        - file: optimization/qaoa.ipynb
        - ...
    - title: API Reference
      file: api/index
      children:
        - file: api/circuit/index
        - file: api/optimization/index
        - ...

site:
  template: book-theme
  options:
    logo: ../assets/qamomile_logo.png
    style: ../assets/custom-theme.css
```

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

## All Makefile Targets

| Target | Description |
|---|---|
| `make help` | Print usage information |
| `make build` | Full build (both languages) |
| `make build-en` | Build English documentation |
| `make build-ja` | Build Japanese documentation |
| `make sync` | Convert all `.py` → `.ipynb` (both languages) |
| `make sync-en` | Convert English `.py` → `.ipynb` |
| `make sync-ja` | Convert Japanese `.py` → `.ipynb` |
| `make generate-api` | Regenerate API reference from docstrings |
| `make copy-api` | Copy API docs to language directories |
| `make clean` | Remove generated files and build outputs |
| `make clean-all` | Remove everything including execution cache |
| `make serve-en` | Build and serve English docs (port 8000) |
| `make serve-ja` | Build and serve Japanese docs (port 8000) |
| `make fresh-en` | Clean, rebuild, and serve English docs |
| `make fresh-ja` | Clean, rebuild, and serve Japanese docs |
| `make build-site` | Build unified site for GitHub Pages |
| `make serve-site` | Serve unified site locally (port 8000) |

## Troubleshooting

### Problem: "No module named 'qamomile'"

**Solution**: Ensure you're in the project environment and qamomile is installed:

```bash
uv sync
```

### Problem: Notebooks not updating after code changes

**Solution**: Clear the execution cache:

```bash
make clean-all
make build
```

### Problem: Port 8000 already in use

**Solution**: Kill the existing server or use a different port:

```bash
# Kill existing server
pkill -f "python -m http.server"

# Or manually serve on different port
cd en/_build/html && python -m http.server 8001
```

### Problem: Jupyter Book not found

**Solution**: Install development dependencies:

```bash
uv sync
```

## Contributing

When contributing documentation:

1. **Edit `.py` source files** — not `.ipynb` directly
2. **Follow the numbered convention** for tutorials (`NN_topic_name.py`)
3. **Test your changes**: Run `make build-en` or `make build-ja`
4. **Check outputs**: Use `make serve-en` or `make serve-ja` to preview
5. **API changes**: If you modified docstrings, run `make generate-api` to update API reference

## Additional Resources

- [Jupyter Book Documentation](https://jupyterbook.org/)
- [Jupytext Documentation](https://jupytext.readthedocs.io/)
- [MyST Markdown Guide](https://myst-parser.readthedocs.io/)
- [Qamomile GitHub](https://github.com/Jij-Inc/Qamomile)

## License

Same license as the main Qamomile project.
