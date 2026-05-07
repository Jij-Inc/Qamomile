# Qamomile Documentation

This directory contains the documentation for Qamomile, built with **Jupyter Book 2** and **jupytext**.

## Overview

This documentation system uses a modern workflow where:

- **Source files**: Python scripts (`.py`) with `# %% [markdown]` format (jupytext percent format)
- **Build-time conversion**: Automatically converted to Jupyter Notebooks (`.ipynb`) via jupytext
- **API reference**: Auto-generated from docstrings using `griffe`
- **Bilingual**: English (`en/`) and Japanese (`ja/`) documentation with identical structure

### Why This Approach?

1. **Git-friendly**: Python scripts are easier to diff, merge, and review than JSON notebooks
2. **IDE support**: Edit with full Python IDE features (linting, type checking, refactoring)
3. **Jupyter Book 2.x**: Single `myst.yml` config, Python-based build (no Node.js)

### Notebook Execution Strategy

ReadTheDocs has a build time limit that is too short to execute all notebooks during hosting.
Therefore, notebooks must be **pre-executed before pushing** to the branch:

```
.py (source) → jupytext → .ipynb (notebook) → jupyter execute → .ipynb (with outputs)
```

ReadTheDocs builds with `execute.enabled: false`, assuming that executed `.ipynb` files with outputs already exist in the branch.

> **Important — do NOT set `QAMOMILE_DOCS_TEST` when executing docs**
>
> Several tutorials (e.g. `optimization/qaoa_graph_partition.py`,
> `vqa/qaoa_maxcut.py`, `vqa/vqe_for_hydrogen.py`,
> `tutorial/05_classical_flow_patterns.py`) read the
> `QAMOMILE_DOCS_TEST` environment variable and switch to a **reduced
> workload** (fewer optimizer iterations, fewer shots) when it equals
> `"1"`. This flag exists only so that `tests/docs/test_tutorials.py`
> can run quickly in CI — it is **not** the configuration we want to
> ship in the rendered docs.
>
> When you execute notebooks for the actual docs build (locally,
> via `./build.sh execute*` / `sync-build*`, or in any pre-push
> workflow), make sure `QAMOMILE_DOCS_TEST` is **unset** (or not
> equal to `"1"`) so the notebooks run with the full settings and
> produce the high-quality outputs intended for readers.

**Future plan**: We are considering a GitHub Actions workflow that automatically detects `.py` changes on merge to `main`, converts and executes the notebooks, and commits the resulting `.ipynb` files back — eliminating the need for manual notebook management.

## Directory Structure

```
docs/
├── README.md                    # This file
├── Makefile                     # Build system (make targets)
├── build.sh                     # Build script (shell alternative)
├── generate_api.py              # API reference generator entry point
├── index.html                   # Redirects to ./en/
├── myst.yml                     # Top-level MyST stub
│
├── assets/                      # Shared images and resources
├── api_gen/                     # API doc generation package (uses griffe)
├── api/                         # Generated API reference (shared source)
├── scripts/                     # Build helper scripts (e.g., inject_colab_launch.py)
├── qamomile-lp/                 # Landing page assets
│
├── en/                          # English documentation
│   ├── myst.yml                 # Jupyter Book 2 config + TOC
│   ├── index.md                 # Landing page
│   ├── _build/                  # Build output (gitignored)
│   ├── api/                     # Copied from docs/api/ at build time
│   ├── tutorial/                # Tutorials (.py sources + .ipynb)
│   ├── optimization/            # Optimization guides
│   ├── vqa/                     # VQA examples
│   └── collaboration/           # External integration tutorials (API keys required)
│
└── ja/                          # Japanese documentation (mirrors en/ structure)
    ├── myst.yml
    ├── index.md
    ├── _build/
    ├── api/
    ├── tutorial/
    ├── optimization/
    ├── vqa/
    └── collaboration/
```

## Quick Start

### Prerequisites

Ensure you have the development dependencies installed:

```bash
uv sync
```

This installs `jupyter-book`, `jupytext`, `griffe`, and other required packages.

If you would like to develop/execute notebooks requiring optional dependencies such as quri-parts, please install those optionals:

```bash
uv sync --extra OPTIONAL_DEPENDENCY
```


### Building Documentation

We recommend using `build.sh` as the primary build tool. A `Makefile` with equivalent targets is also available (`make <command>`).

#### Available commands

| Command | Description |
|---------|-------------|
| `./build.sh build` | Build both languages (no sync) |
| `./build.sh build-en` | Build English only (no sync) |
| `./build.sh build-ja` | Build Japanese only (no sync) |
| `./build.sh sync` | Convert all `.py` → `.ipynb` (both languages) |
| `./build.sh sync-en` | Convert English `.py` → `.ipynb` |
| `./build.sh sync-ja` | Convert Japanese `.py` → `.ipynb` |
| `./build.sh execute` | Execute all `.ipynb` notebooks (both languages) |
| `./build.sh execute-en` | Execute English `.ipynb` notebooks |
| `./build.sh execute-ja` | Execute Japanese `.ipynb` notebooks |
| `./build.sh sync-build` | Sync, execute, and build both languages |
| `./build.sh sync-build-en` | Sync, execute, and build English |
| `./build.sh sync-build-ja` | Sync, execute, and build Japanese |
| `./build.sh clean` | Remove generated `.ipynb`, build outputs, and copied API docs |
| `./build.sh clean-all` | Remove everything including execution cache |
| `./build.sh serve-en` | Sync, build (if needed), and serve English docs (port 8000) |
| `./build.sh serve-ja` | Sync, build (if needed), and serve Japanese docs (port 8000) |
| `./build.sh fresh-en` | Clean, sync, rebuild, and serve English docs |
| `./build.sh fresh-ja` | Clean, sync, rebuild, and serve Japanese docs |

#### Typical workflow

```bash
# 1. First time or after cleaning: target .py → .ipynb and build
uv run jupytext --to ipynb target.py

# 2. Execute the notebook locally
uv run jupyter nbconvert --to notebook --execute --inplace target.ipynb

# 3. Rebuild without re-syncing (e.g., after config changes)
./build.sh build

# 4. Preview locally
./build.sh serve-en
```

Note that, if you run `sync` command at the first step, almost all the `.py` files will be converted into `.ipynb` without outputs. Use `sync-build` instead to sync, execute, and build in one step.

### Viewing Documentation Locally

After building, serve the documentation in your browser:

```bash
./build.sh serve-en   # English at http://localhost:8000
./build.sh serve-ja   # Japanese at http://localhost:8000
```

Then open: http://localhost:8000

## API Reference Generation

API reference pages are auto-generated from `qamomile` docstrings.

### How it works

1. `generate_api.py` uses the `api_gen/` package (built on `griffe`) to introspect the `qamomile` package
2. Generates MyST-compatible Markdown files into `docs/api/`
3. During build, the generated files are copied to `en/api/` and `ja/api/`

API generation and copying are automatically included in `./build.sh build`. No separate command is needed.

## Development Workflow

### Writing/Editing Documentation

1. **Edit Python files** (`.py`) in the appropriate directory.

2. **Use jupytext percent format**:
   ```python
   # %% [markdown]
   # # Section Title
   # This is markdown content

   # %%
   import qamomile.circuit as qmc

   # Your Python code here
   ```

3. **Build, execute and preview**:
   ```bash
   uv run jupytext --to notebook target.py
   uv run jupyter nbconvert --to notebook --execute --inplace target.ipynb
   ./build.sh serve-en       # or serve-ja for Japanese
   ```

### Creating New Pages

1. **Create the `.py` file** in both `en/` and `ja/` following the numbered convention:

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

   Place it in the appropriate directory.

2. **Add to TOC** in both `en/myst.yml` and `ja/myst.yml`:
   ```yaml
   toc:
     - title: Target Section
       children:
         - file: target_directory/target.ipynb
   ```
   The example above is 

3. **Add to test patterns** if the new page is in a directory not yet covered by `tests/docs/test_tutorials.py` `TUTORIAL_PATTERNS`. Here, collaboration is explicitly not included since they could require additional settings.

4. **Generate, execute and commit the `.ipynb`**: Convert to notebook and execute it so that output cells are populated:
   ```bash
   uv run jupytext --to notebook target.py
   uv run jupyter nbconvert --to notebook --execute --inplace target.ipynb
   ```
   Repeat for the `ja/` counterpart.

5. **Build and verify**:
   ```bash
   ./build.sh build
   ./build.sh serve-en
   ```

#### Checklist for new pages

- [ ] `.py` file created in both `en/` and `ja/`
- [ ] Page added to `toc:` in both `en/myst.yml` and `ja/myst.yml`
- [ ] Page linked from both `en/index.md` and `ja/index.md`
- [ ] `.ipynb` generated and executed (outputs present)
- [ ] Test patterns cover the new directory (if applicable)
- [ ] Build succeeds and page renders correctly

## Configuration Files

### `myst.yml` (Jupyter Book 2 Configuration)

Each language directory has its own `myst.yml` combining configuration and table of contents. Key structure:

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

## All Build Commands

See the [Building Documentation](#building-documentation) section for the full command table. All `./build.sh <command>` targets are also available as `make <command>`.

## Troubleshooting

### Problem: "No module named 'qamomile'"

**Solution**: Ensure you're in the project environment and qamomile is installed:

```bash
uv sync
```

### Problem: Notebooks not updating after code changes

**Solution**: Clear the execution cache:

```bash
./build.sh clean-all
./build.sh sync-build
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
3. **Test your changes**: Run `./build.sh sync-build-en` or `sync-build-ja`
4. **Check outputs**: Use `./build.sh serve-en` or `serve-ja` to preview
5. **API changes**: API reference is auto-regenerated on every build

## Additional Resources

- [Jupyter Book Documentation](https://jupyterbook.org/)
- [Jupytext Documentation](https://jupytext.readthedocs.io/)
- [MyST Markdown Guide](https://myst-parser.readthedocs.io/)
- [Qamomile GitHub](https://github.com/Jij-Inc/Qamomile)

## License

Same license as the main Qamomile project.
