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

ReadTheDocs has a build time limit that is too short to execute all notebooks during hosting. Therefore, notebooks must be **pre-executed before pushing** to the branch:

```
.py (source) → jupytext → .ipynb (notebook) → jupyter execute → .ipynb (with outputs)
```

ReadTheDocs builds with `execute.enabled: false`, assuming that executed `.ipynb` files with outputs already exist in the branch.

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
├── assets/                      # Shared images and resources (incl. custom-theme.css)
├── api_gen/                     # API doc generation package (uses griffe)
├── api/                         # Generated API reference (shared source)
├── scripts/                     # Build helper scripts
│                                #   - build_doc_tags.py: regenerate tag pages, chip blocks, myst.yml auto-tags region
│                                #   - inject_colab_launch.py: post-build "open in Colab" button
│
├── en/                          # English documentation
│   ├── myst.yml                 # Jupyter Book 2 config + TOC
│   ├── index.md                 # Landing page
│   ├── _build/                  # Build output (gitignored)
│   ├── api/                     # Copied from docs/api/ at build time
│   ├── tutorial/                # SDK fundamentals (kernels, parameters, execution, …)
│   ├── algorithm/               # Algorithm walkthroughs (QAOA, VQE, QEC, Hamiltonian sim, …)
│   ├── usage/                   # Per-module how-to guides (BinaryModel, …)
│   ├── integration/             # External-library / platform integration notes (qBraid; needs API key)
│   ├── release_notes/           # Per-version changelog
│   └── tags/                    # Auto-generated tag pages (gitignored)
│
└── ja/                          # Japanese documentation (mirrors en/ structure)
```

## Quick Start

### Prerequisites

```bash
uv sync
```

This installs `jupyter-book`, `jupytext`, `griffe`, and the rest of the dev dependencies. To execute notebooks that need optional extras:

```bash
uv sync --extra OPTIONAL_DEPENDENCY    # e.g. quri_parts, cudaq-cu13
```

### Building Documentation

`build.sh` is the primary entry point; an equivalent `Makefile` is also available (`make <command>`).

#### Typical workflow

After editing a single `.py` file, re-sync the paired notebook, execute it, and serve the docs locally:

```bash
uv run jupytext --to ipynb --update docs/en/<section>/foo.py
uv run jupyter nbconvert --to notebook --execute --inplace docs/en/<section>/foo.ipynb
./build.sh serve-en    # browse at http://localhost:8000
```

That's the day-to-day loop. `build.sh` handles tag-page regeneration and API-reference generation/copy as part of `build`, so you don't need to run those steps yourself.

#### All `build.sh` commands

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

`./build.sh sync` alone produces `.ipynb` without outputs. Use `sync-build` (or run `execute` after `sync`) when you actually need executed notebooks.

## API Reference Generation

API generation and copying are automatically included in `./build.sh build` — no separate command is needed under normal use.

For reference, the underlying flow:

1. `generate_api.py` uses the `api_gen/` package (built on `griffe`) to introspect the `qamomile` package
2. Generates MyST-compatible Markdown files into `docs/api/`
3. During build, the generated files are copied to `en/api/` and `ja/api/`

## Development Workflow

### Editing an existing page

1. Edit the `.py` source — never the `.ipynb` directly.
2. Re-sync and re-execute:
   ```bash
   uv run jupytext --to ipynb --update docs/en/<section>/foo.py
   uv run jupyter nbconvert --to notebook --execute --inplace docs/en/<section>/foo.ipynb
   ```
3. Preview: `./build.sh serve-en` (or `serve-ja`).

### Creating a new page

1. **Create the `.py` file** in both `en/` and `ja/` with the standard
   jupytext header, a frontmatter block carrying `tags`, then the H1
   and content:

   ```python
   # ---
   # jupyter:
   #   jupytext:
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
   # ---
   # tags: [optimization, variational]
   # ---
   #
   # # Your Page Title
   # Content goes here...
   ```

   `tags:` values are language-agnostic — keep them identical between
   `en/` and `ja/`. Tags must be in `ALLOWED_TAGS` (see
   [Tags](#tags) below).

2. **Add the page to the main `toc:` in both `en/myst.yml` and
   `ja/myst.yml`**, under the matching section:

   ```yaml
   - title: Algorithms
     file: algorithm/index.md
     children:
       - file: algorithm/qaoa_maxcut.ipynb
       - file: algorithm/your_new_topic.ipynb   # ← add this
   ```

   Do NOT hand-edit the `# --- BEGIN doc tags ... # --- END doc tags ---`
   region — it is regenerated by the script.

3. **Update the section landing page** by adding a bullet/link in the
   matching `<section>/index.md` (each section's `index.md` is
   hand-written).

4. **Add to test patterns** if the new page is in a directory not yet
   covered by `tests/docs/test_tutorials.py` `TUTORIAL_PATTERNS`.
   `integration/` is intentionally excluded since those notebooks
   may need an API key.

5. **Regenerate auto-managed regions** (chip blocks, tag pages,
   `myst.yml` auto-tags region):

   ```bash
   uv run python docs/scripts/build_doc_tags.py
   ```

6. **Generate and execute the `.ipynb`**:

   ```bash
   uv run jupytext --to ipynb --update docs/en/<section>/your_new_topic.py
   uv run jupyter nbconvert --to notebook --execute --inplace docs/en/<section>/your_new_topic.ipynb
   # repeat for ja/
   ```

7. **`git add` + commit + push.**

8. **Build and verify**:

   ```bash
   ./build.sh build
   ./build.sh serve-en
   ```

### Tags

Articles under `{tutorial,algorithm,usage,integration}/` (in both
`en/` and `ja/`) are tag-filterable. Each `.py` declares its tags in
the frontmatter at the top of its first markdown cell:

```python
# %% [markdown]
# ---
# tags: [optimization, variational]
# ---
#
# # My Article
```

Tags are language-agnostic (same string for `en` and `ja`). From these
declarations, `docs/scripts/build_doc_tags.py` regenerates:

| Output | Path | In git? |
|---|---|---|
| Tag landing page | `docs/<lang>/tags/index.md` | gitignored |
| Per-tag pages | `docs/<lang>/tags/<tag>.md` | gitignored |
| Inline tag chips at the top of each article | sentinel block inside the `.py` | committed |
| Browse-by-tag chip cloud on each section's `index.md` | sentinel block in `index.md` | committed |
| `Tags` toc block in `myst.yml` | sentinel region in `myst.yml` | committed |

Section index pages are hand-written; the script only refreshes the
chip line *inside* the sentinel block. To opt a section into the chip
cloud, drop the sentinel pair in the right place:

```markdown
## Browse by tag

<!-- BEGIN browse-by-tag -->
<!-- END browse-by-tag -->
```

Sentinels:

| Where | Begin / End |
|---|---|
| Inline chip block inside an article `.py` | `# <!-- BEGIN auto-tags -->` / `# <!-- END auto-tags -->` |
| Browse-by-tag block inside a section `index.md` | `<!-- BEGIN browse-by-tag -->` / `<!-- END browse-by-tag -->` |
| Auto-managed `Tags` toc block in `myst.yml` | `# --- BEGIN doc tags (auto-generated) ---` / `# --- END doc tags (auto-generated) ---` |

`./build.sh build-{en,ja}` runs the generator before MyST builds, so
RTD and local builds stay in sync without manual steps. Run the
generator yourself if you want the auto-managed regions to be current
in your working tree before commit.

#### Tag whitelist

Allowed tags live in `ALLOWED_TAGS` inside
`docs/scripts/build_doc_tags.py`. The taxonomy is intentionally small.
The whitelist is **enforced by CI** via
`tests/docs/test_tag_whitelist.py` — any PR that uses a tag outside
the set fails the unit-test job. There is no pre-commit hook and no
build-time validation: a stray tag in your local working tree will
not crash the build, only the test fails on the PR.

**Adding a new tag is a deliberate maintainer decision**, not a
side-effect of writing an article. Stay within the existing set
unless the project owner has explicitly approved the new tag — see
`CLAUDE.md` for the full policy. Release notes (`release_notes/`) are
intentionally out of scope and never tagged.

## Jupytext Format

All Python source files use the **percent format**:

```python
# %% [markdown]
# Markdown cells start with this marker

# %%
# Python code cells start with this marker
print("Hello, Quantum World!")
```

- **Metadata in header**: jupytext + kernelspec at top of file
- **Cell markers**: `# %%` for code, `# %% [markdown]` for text
- **Clean diffs**: git sees normal Python file changes
- **IDE friendly**: full Python tooling support

## Troubleshooting

### "No module named 'qamomile'"

Ensure dev dependencies are installed in the active env:

```bash
uv sync
```

### Notebooks not updating after code changes

Clear the execution cache, then re-sync:

```bash
./build.sh clean-all
./build.sh sync-build
```

### Port 8000 already in use

```bash
pkill -f "python -m http.server"
# or use a different port
cd en/_build/html && python -m http.server 8001
```

### `test_tag_whitelist` fails on a PR

The article carries a tag that isn't in `ALLOWED_TAGS`. Either fix
the typo in the article's frontmatter, or — if the tag is genuinely
new and approved — extend `ALLOWED_TAGS` in
`docs/scripts/build_doc_tags.py` (see policy in `CLAUDE.md`).

## Additional Resources

- [Jupyter Book Documentation](https://jupyterbook.org/)
- [Jupytext Documentation](https://jupytext.readthedocs.io/)
- [MyST Markdown Guide](https://myst-parser.readthedocs.io/)
- [Qamomile GitHub](https://github.com/Jij-Inc/Qamomile)

## License

Same license as the main Qamomile project.
