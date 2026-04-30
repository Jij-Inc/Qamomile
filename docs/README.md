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
│                                #   - build_doc_tags.py: generate per-tag pages, inject chip blocks, inject section browse-by-tag clouds
│                                #   - inject_colab_launch.py: post-build "open in Colab" button on every rendered HTML page
│
├── _build_src/                  # Build-time scratch tree (gitignored)
│                                #   build.sh copies en/ and ja/ here, runs
│                                #   build_doc_tags.py + jupytext --update against
│                                #   the copy, then mystmd builds from this dir.
│                                #   Source files never receive auto-managed
│                                #   injections (chip blocks, browse-by-tag, etc.).
│
├── en/                          # English documentation (committed source)
│   ├── myst.yml                 # Jupyter Book 2 config + TOC
│   ├── index.md                 # Landing page
│   ├── _build/                  # Build output, populated by build.sh from
│   │                            #   _build_src/en/_build/ (gitignored)
│   ├── api/                     # Copied from docs/api/ at build time (gitignored)
│   ├── tutorial/                # SDK fundamentals (kernels, parameters, execution, …)
│   ├── algorithm/               # Algorithm walkthroughs (QAOA, VQE, QEC, Hamiltonian sim, …)
│   ├── usage/                   # Per-module how-to guides (BinaryModel, …)
│   ├── integration/             # External-library / platform integration notes (qBraid; needs API key)
│   ├── release_notes/           # Per-version changelog
│   └── tags/                    # Auto-generated tag pages (gitignored; canonical
│                                #   copies live in _build_src/<lang>/tags/)
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

That's the day-to-day loop. `build.sh` copies your source into a gitignored `_build_src/` scratch tree, injects tag-related auto-managed content (chip blocks, browse-by-tag clouds, per-tag pages) there, and runs mystmd from the scratch tree — so your committed source files never gain tag-related build-time content. API generation and copying are also bundled into `build`, so you don't need to run those steps yourself. (`generate_api.py` does write the API Reference TOC entries back into committed `myst.yml`; that's a separate, intentional channel — `build_doc_tags.py` itself does not touch `myst.yml`.)

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

2. **Update the section landing page** by adding a bullet/link in
   the matching `<section>/index.md`. Each section's `index.md` is
   hand-written and contains only the intro paragraph plus the
   article list; the `## Browse by tag` heading and chip cloud are
   synthesised into the build-dir copy automatically, so don't add
   them yourself. The article ordering inside the section's
   sidebar nav is whatever your filename sorts to alphabetically
   (mystmd discovers the .ipynb files via a glob in `myst.yml`); use
   `01_`, `02_`, … prefixes if you need a curated order.

3. **You don't need to touch `myst.yml`.** The toc uses
   `pattern: <section>/*.ipynb`, so any new `.ipynb` you drop into
   the section is auto-discovered. (Per-tag pages and the per-tag
   listing are similarly discovered via `pattern: tags/*.md`.)

4. **Add to test patterns** if the new page is in a directory not
   yet covered by `tests/docs/test_tutorials.py`
   `TUTORIAL_PATTERNS`. `integration/` is intentionally excluded
   since those notebooks may need an API key.

5. **Generate and execute the `.ipynb`**:

   ```bash
   uv run jupytext --to ipynb --update docs/en/<section>/your_new_topic.py
   uv run jupyter nbconvert --to notebook --execute --inplace docs/en/<section>/your_new_topic.ipynb
   # repeat for ja/
   ```

6. **`git add` + commit + push.**

7. **Build and verify**:

   ```bash
   ./build.sh build
   ./build.sh serve-en
   ```

   `build.sh` will copy your source into `_build_src/`, inject chip
   blocks / browse-by-tag clouds / per-tag pages there, and build from
   that scratch tree — your committed source stays clean.

### Adding a new section directory

The four sections currently registered (`tutorial/`, `algorithm/`,
`usage/`, `integration/`) live at the top level under
`docs/<lang>/`. Adding a new one — at the same level — is supported
and only needs config changes. Adding a nested subsection (a
section inside another section) is **not** currently supported and
would require code changes; details below.

#### Adding a sibling section (e.g. a new `theory/`)

Update each of the following:

1. **Create the section directories**: `docs/en/<new>/` and
   `docs/ja/<new>/`. Drop the first article's `.py` in.
2. **Create `<new>/index.md`** in both en/ja with the standard
   landing-page shape. Just intro paragraph + `## All articles` (or
   localised equivalent) — **do not** add a `## Browse by tag`
   heading; the build pipeline synthesises it from `tags:`
   frontmatter:
   ```markdown
   ---
   slug: <new>
   ---

   # <Section Title>

   One-line description of the section.

   ## All articles

   - [First Article Title](first_article) — short description
   ```
3. **Register the section in `docs/scripts/build_doc_tags.py`**:
   - Append `"<new>"` to `SECTIONS`.
   - Add `"<new>": "<English title>"` to `STRINGS["en"]["section_titles"]`
     and `"<new>": "<Japanese title>"` to `STRINGS["ja"]["section_titles"]`
     so the section's name renders correctly on tag pages.
4. **Add the section to each `myst.yml` toc** (en/ja), under the
   hand-written part. Use a `pattern:` child so any future article
   in the section is auto-discovered without further toc edits:
   ```yaml
   - title: <English title>
     file: <new>/index.md
     children:
       - pattern: <new>/*.ipynb
   ```
5. **Add a top-level link in `docs/{en,ja}/index.md`** alongside
   the other section bullets.
6. **Decide whether `build.sh` should auto-sync / auto-execute the
   notebooks under this section**:
   - *Yes* (default — articles are runnable in CI without external
     resources): append `"<new>"` to `TARGET_DIRS` in both
     `docs/build.sh` and `docs/Makefile`.
   - *No* (notebooks need API keys, network, or other side effects
     — like `integration/`): leave `TARGET_DIRS` alone. Update the
     comment block above `TARGET_DIRS` in both files to mention
     `<new>` as another excluded directory. Also list `<new>/` in
     the "We will not execute the following directories" comment in
     `tests/docs/test_tutorials.py`.
7. **Update `tests/docs/test_tutorials.py`'s `TUTORIAL_PATTERNS`**
   if the section's notebooks should be exercised by the docs
   tests (mirrors the `TARGET_DIRS` decision above):
   ```python
   "docs/en/<new>/**/*.py",
   "docs/ja/<new>/**/*.py",
   "docs/en/<new>/**/*.ipynb",
   "docs/ja/<new>/**/*.ipynb",
   ```
8. **Update the `Directory Structure` section of this README** to
   mention the new directory.

After step 4, run `./build.sh build` to verify the new section
appears in the rendered nav and that auto-managed content (chip
blocks on articles, browse-by-tag cloud on the new `index.md`)
shows up correctly inside `_build_src/`. The Tags toc itself is
discovered via the existing `pattern: "tags/*.md"` entry in
`myst.yml`, so it does not need touching.

#### Adding a nested subsection (currently unsupported)

The script's section model is flat:
`docs/scripts/build_doc_tags.py` walks `<lang>/<section>/*.py` only
at the top level — no recursion. The browse-by-tag classifier in
`_classify_for_index` returns `same` or `cousin`; the legacy
`descendant` / `ancestor` buckets were dropped when the layout
flattened, but the comment notes "this is the place to teach the
classifier about descendant and ancestor again".

To enable nested sections you would need at least:

1. Make `_walk_articles` recurse (`rglob("*.py")` instead of
   `glob("*.py")`), and represent `Article.section` as a path-like
   breadcrumb instead of a single string.
2. Restore `descendant` / `ancestor` buckets in
   `_classify_for_index` and `_BUCKET_ORDER`, plus the matching
   localised labels in `STRINGS["en"]["bucket_labels"]` /
   `STRINGS["ja"]["bucket_labels"]`.
3. Update `myst.yml` toc to be nested (mystmd already supports
   nested children).

If you genuinely need nested sections, treat it as a small feature
on `build_doc_tags.py` rather than a pure docs change.

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

Tags are language-agnostic (same string for `en` and `ja`). The build
pipeline runs `docs/scripts/build_doc_tags.py` against `_build_src/`
(the scratch copy) and turns those declarations into:

| Output | Where it ends up | In git? |
|---|---|---|
| Tag landing page | `_build_src/<lang>/tags/index.md` | no |
| Per-tag pages | `_build_src/<lang>/tags/<tag>.md` (one per tag) | no |
| Inline tag chips at the top of each article | injected into the `.py`/`.ipynb` inside `_build_src/<lang>/<section>/` | no |
| Browse-by-tag chip cloud on each section's `index.md` | injected into `_build_src/<lang>/<section>/index.md` | no |

Where the script injects in the build-dir copy:

| Where | How the script finds the spot |
|---|---|
| Article `.py` body | inserted right after the first H1 |
| Section `index.md` | a whole `## Browse by tag` section is synthesised and inserted right before the first H2 (e.g. before `## All articles`) |

The per-tag pages are picked up by mystmd via a single
`- pattern: "tags/*.md"` toc entry in `myst.yml` (with
`hidden: true`), so the script does **not** maintain any region
inside `myst.yml`. The committed source therefore has zero
auto-managed regions to track — articles, section index pages,
and `myst.yml` are all hand-written.

`./build.sh build-{en,ja}` runs `setup_build_src` (copy → inject →
jupytext sync) before MyST builds, so RTD and local builds stay in
sync without manual steps. PRs that change tag taxonomy or article
tags only diff the actual `tags:` frontmatter line, not the
chip-block churn that comes with it.

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
