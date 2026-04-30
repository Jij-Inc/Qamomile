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
│   ├── tutorial/                # SDK tutorials (.py sources + .ipynb)
│   ├── algorithm/               # Algorithm examples (flat, tag-filterable)
│   ├── usage/                  # Module usage guides (BinaryModel, etc.)
│   └── collaboration/           # External integration tutorials (API keys required)
│
└── ja/                          # Japanese documentation (mirrors en/ structure)
    ├── myst.yml
    ├── index.md
    ├── _build/
    ├── api/
    ├── tutorial/
    ├── algorithm/
    ├── usage/
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

The pre-commit hook handles tag chips, `tags/*.md`, `algorithm/index.md`,
and `myst.yml`'s auto-tags region for you (see [Tags on documentation
pages](#tags-on-documentation-pages) below). What you still need to do
by hand:

1. **Create the `.py` file** in both `en/` and `ja/` with the standard
   jupytext header, a frontmatter block carrying `title` and `tags`,
   then the H1 and content:

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
   # title: Your Page Title
   # tags: [some-tag, another-tag, intermediate]
   # ---
   #
   # # Your Page Title
   # Content goes here...
   ```

   Tag values are language-agnostic — keep them identical between
   `en/` and `ja/`. Only `title` differs.

2. **Add the page to the main `toc:` in both `en/myst.yml` and
   `ja/myst.yml`**, under the matching section:

   ```yaml
   - title: Algorithms
     file: algorithm/index.md
     children:
       - file: algorithm/qaoa_maxcut.ipynb
       - file: algorithm/your_new_topic.ipynb   # ← add this
   ```

   The auto-managed `# --- BEGIN doc tags ... # --- END doc tags ---`
   region elsewhere in `myst.yml` is regenerated by the hook — do not
   hand-edit it.

3. **Update the section landing page** (only for sections with a
   hand-written `index.md`):

   - `tutorial/index.md`, `usage/index.md`,
     `collaboration/index.md` → add a bullet/link to the new page.
   - `algorithm/index.md` is auto-generated, so nothing to do here.

4. **Add to test patterns** if the new page is in a directory not yet
   covered by `tests/docs/test_tutorials.py` `TUTORIAL_PATTERNS`.
   Collaboration is explicitly not included since those tutorials may
   require API keys.

5. **Generate, execute, and commit the `.ipynb`**:

   ```bash
   uv run jupytext --to notebook docs/en/<section>/your_new_topic.py
   uv run jupyter nbconvert --to notebook --execute --inplace docs/en/<section>/your_new_topic.ipynb
   # repeat for ja/
   ```

   Once the pre-commit hook is installed, every subsequent edit to the
   `.py` re-syncs the `.ipynb` automatically (via
   `jupytext --to ipynb --update`, which preserves outputs).

6. **`git add` + commit.** The pre-commit hook may modify tracked
   files (chip block in your `.py`, `myst.yml` auto-tags region) — if
   so, pre-commit prints "Files were modified, please re-stage";
   `git add` and re-run `git commit`.

7. **Build and verify**:

   ```bash
   ./build.sh build
   ./build.sh serve-en
   ```

#### Tags on documentation pages

Articles under `en/{tutorial,algorithm,usage,collaboration}/` and
their `ja/` counterparts are tag-filterable. Give each `.py` a MyST
frontmatter block at the top of its first markdown cell, e.g.:

```python
# %% [markdown]
# ---
# title: My New Algorithm
# tags: [qaoa, optimization, variational]
# ---
#
# # My New Algorithm
# ...
```

Tags are language-agnostic (the same string for `en` and `ja`); only
`title` differs per locale.

##### Allowed tags (whitelist)

The script enforces a whitelist of allowed tags — anything outside it
fails the build with an `UnknownTagError` pointing at the offending
file. The current set lives in `ALLOWED_TAGS` inside
`docs/scripts/build_doc_tags.py`. The taxonomy is intentionally small;
adding a new tag is a deliberate two-line patch:

1. Append the tag to `ALLOWED_TAGS`.
2. Use it in the article's `tags:` frontmatter.

Both go in the same commit. We avoid free-form tagging because tag
soup makes the cloud noisy and makes navigation worse than no tags.

From these frontmatter blocks, `docs/scripts/build_doc_tags.py` regenerates:

| Output | Path | In git? |
|---|---|---|
| Tag landing page | `docs/<lang>/tags/index.md` | gitignored |
| Per-tag pages | `docs/<lang>/tags/<tag>.md` | gitignored |
| Inline tag chips | sentinel block inside each tagged `.py` | committed |
| Browse-by-tag chip cloud | sentinel block inside each section's `index.md` | committed |
| `Tags` toc block | sentinel region inside each `myst.yml` | committed |

Section index pages (`tutorial/index.md`, `algorithm/index.md`, …) are
hand-written and tracked in git. The script only refreshes the chip
line *inside* the sentinel block; it never touches the surrounding
prose. To opt a section into the chip cloud, drop the sentinel pair in
the right place:

```markdown
## Browse by tag

<!-- BEGIN browse-by-tag -->
<!-- END browse-by-tag -->
```

Sentinels:

- Inline chip block in `.py`: `# <!-- BEGIN auto-tags -->` / `# <!-- END auto-tags -->`
- Browse-by-tag block in `index.md`: `<!-- BEGIN browse-by-tag -->` / `<!-- END browse-by-tag -->`
- `Tags` toc block in `myst.yml`: `# --- BEGIN doc tags (auto-generated) ---` / `# --- END doc tags (auto-generated) ---`

`./build.sh build-{en,ja}` (and `make build-{en,ja}`) run the generator
before the MyST build, so RTD and local builds stay in sync without any
manual step.

##### Pre-commit hook (recommended)

A pre-commit hook (`.pre-commit-config.yaml`) runs the generator and
`jupytext --to ipynb --update` whenever a tagged `.py` is staged, so the
chip blocks, `myst.yml`, and `.ipynb` shadow stay current automatically.

Set it up once per clone:

```bash
uv sync
uv run pre-commit install
```

After that every `git commit` runs the hook on staged docs sources. If
the hook modifies any tracked files, pre-commit asks you to re-stage
and commit again.

To run the generator manually (e.g. after editing tags without
committing yet):

```bash
uv run python docs/scripts/build_doc_tags.py
```

Release notes (`release_notes/`) are intentionally out of scope and
never tagged.

#### Checklist for new pages

- [ ] `.py` file created in both `en/` and `ja/`, with frontmatter
      (`title`, `tags`) at the top of the first markdown cell
- [ ] Page added to the main `toc:` in both `en/myst.yml` and
      `ja/myst.yml` (the auto-tags region is regenerated by the hook)
- [ ] Page linked from `<section>/index.md` (for `tutorial/`,
      `usage/`, `collaboration/`) and from the top-level
      `en/index.md` / `ja/index.md` if appropriate. `algorithm/index.md`
      is auto-generated.
- [ ] `.ipynb` generated and executed (outputs present)
- [ ] Test patterns cover the new directory (if applicable)
- [ ] Pre-commit hook installed (`uv run pre-commit install`) so the
      tag chips, `myst.yml` auto-tags region, and `.ipynb` shadows stay
      in sync on every commit
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
