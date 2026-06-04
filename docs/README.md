# Qamomile Documentation

This directory contains the documentation for Qamomile, built with **Jupyter Book 2** and **jupytext**.

## Overview

This documentation system uses a modern workflow where:

- **Source files**: Python scripts (`.py`) in jupytext format
- **Paired notebooks**: `.ipynb` files are generated with jupytext and executed before pushing, so rendered docs can ship with committed outputs
- **Build-dir synchronization**: [docs/build.sh](build.sh) `build` copies the docs into `_build_src/` and runs `jupytext --update` there after tag injection; this syncs cell sources in the build copy but does not execute notebooks or replace the pre-push `.ipynb` workflow
- **API reference**: Generated from docstrings with `griffe` through
  [docs/generate_api.py](generate_api.py). The generated Markdown is written
  to gitignored `docs/api/`, copied into `docs/{en,ja}/api/` during builds,
  and the API TOC region in each language's `myst.yml` is refreshed.
- **Bilingual**: English (`en/`) and Japanese (`ja/`) documentation with identical structure

### Why This Approach?

1. **Reviewable source**: `.py` files are the main authoring surface, so content and code changes can be reviewed without relying only on notebook JSON diffs
2. **IDE support**: Edit source pages with full Python IDE features (linting, type checking, refactoring)
3. **Jupyter Book 2.x**: MyST config files
   ([en](en/myst.yml), [ja](ja/myst.yml)), Python-driven docs workflow

## Authoring Policy

The docs are reader-first, Qamomile-first, and testable. Each article should
teach one clear purpose: a Qamomile workflow, an algorithm implemented with
Qamomile, an integration boundary, or one usage detail. If a page needs two
independent goals, split it into two pages.

### Content principles

- **One notebook for one purpose**: keep each notebook focused on a single outcome.
- **Reader-first explanations**: prefer the shortest path that helps a user
  understand and run the example. Move detailed mechanics into `usage/` pages.
- **Verified claims only**: avoid unreviewed academic claims. When an algorithm
  page needs theory, keep the statement narrow, cite or link the source, and
  verify the implemented behavior with assertions or reference values.
- **Qamomile-first examples**: use Qamomile qkernels, observables, converters,
  drawings, resource estimates, and execution APIs wherever possible.
- **Use MyST figures**: authored figures and screenshots should use MyST
  `figure` directives so captions, alt text, and layout stay explicit.
- **Commit static images as files**: when adding an image that is not generated
  by Python during notebook execution (for example, not a matplotlib output),
  add it as an image file and reference that file. Do not embed static images
  as notebook blobs or base64 payloads.
- **Docs as tests**: every runnable notebook should include assertions for the
  important shapes, counts, or result properties. Keep runtime small enough for
  [tests/docs/test_tutorials.py](../tests/docs/test_tutorials.py).
- **No manual References section**: use MyST cross-references, citations, or
  external links instead of hand-maintaining a References section.
- **Use MyST notes for remarks**: tips, notes, and remarks should use
  `:::{note}` blocks.

### Skeletons

Authoring templates live in [docs/skeletons/](skeletons/). They are
intentionally outside [docs/en/](en/) and [docs/ja/](ja/), so they are not
rendered, synced, or executed by the
docs build. Page shape and section-specific outlines live in the skeletons
themselves. See [docs/skeletons/README.md](skeletons/README.md) before creating a new
notebook-style page.

### Notebook Execution Strategy

ReadTheDocs has a build time limit that is too short to execute all notebooks during hosting. Therefore, notebooks must be **pre-executed before pushing** to the branch:

```
.py (source) → jupytext → .ipynb (notebook) → jupyter execute → .ipynb (with outputs)
```

ReadTheDocs builds with `execute.enabled: false`, assuming that executed `.ipynb` files with outputs already exist in the branch.

> **Important — do NOT set `QAMOMILE_DOCS_TEST` when executing docs**
>
> Several tutorials (e.g. `algorithm/qaoa_graph_partition.py`,
> `algorithm/qaoa_maxcut.py`, `algorithm/vqe_for_hydrogen.py`,
> `tutorial/07_classical_flow_patterns.py`) read the
> `QAMOMILE_DOCS_TEST` environment variable and switch to a **reduced
> workload** (fewer optimizer iterations, fewer shots) when it equals
> `"1"`. This flag exists only so that
> [tests/docs/test_tutorials.py](../tests/docs/test_tutorials.py)
> can run quickly in CI — it is **not** the configuration we want to
> ship in the rendered docs.
>
> When you execute notebooks for the actual docs build (locally, via
> `./docs/build.sh page-build`, `./docs/build.sh execute`,
> `./docs/build.sh execute-en`, `./docs/build.sh execute-ja`,
> `./docs/build.sh sync-build`, `./docs/build.sh sync-build-en`,
> `./docs/build.sh sync-build-ja`, `./docs/build.sh fresh-en`,
> `./docs/build.sh fresh-ja`, or `./docs/build.sh serve-en` /
> `./docs/build.sh serve-ja` when build output is missing, or in any
> pre-push workflow), make sure `QAMOMILE_DOCS_TEST` is **unset** (or not
> equal to `"1"`) so the notebooks run with the full settings and produce
> the high-quality outputs intended for readers.

## Directory Guide

| Path | Purpose |
| --- | --- |
| [docs/en/](en/) | English documentation source. |
| [docs/ja/](ja/) | Japanese documentation source mirroring `docs/en/`. |
| [docs/en/tutorial/](en/tutorial/) / [docs/ja/tutorial/](ja/tutorial/) | New-user tutorials for learning Qamomile workflows. |
| [docs/en/algorithm/](en/algorithm/) / [docs/ja/algorithm/](ja/algorithm/) | Algorithm walkthroughs implemented with Qamomile. |
| [docs/en/usage/](en/usage/) / [docs/ja/usage/](ja/usage/) | Focused usage notes for Qamomile features and API patterns. |
| [docs/en/integration/](en/integration/) / [docs/ja/integration/](ja/integration/) | Integration guides for external SDKs, services, or packages. |
| [docs/en/release_notes/](en/release_notes/) / [docs/ja/release_notes/](ja/release_notes/) | Versioned release notes. |
| [docs/skeletons/](skeletons/) | Non-rendered authoring templates for new notebook-style pages. |
| [docs/assets/](assets/) | Shared static assets such as images, CSS, and logos. |
| `docs/api/` | Gitignored generated API reference Markdown produced by [docs/generate_api.py](generate_api.py). |
| [docs/api_gen/](api_gen/) | API reference generation code built on `griffe`. |
| [docs/scripts/](scripts/) | Build helper scripts for tag pages, injected UI, and post-build tweaks. |
| `docs/_build_src/` | Gitignored build scratch tree populated by [docs/build.sh](build.sh). |

## Development Workflow

### Setup

Run this once before editing or building docs:

```bash
uv sync
```

To execute notebooks that need optional extras:

```bash
uv sync --extra OPTIONAL_DEPENDENCY    # e.g. quri_parts, cudaq-cu13
```

### Editing an existing page

1. Edit the `.py` source — never the `.ipynb` directly.
2. Sync, execute, and build the edited page.

   English page:

   ```bash
   ./docs/build.sh page-build docs/en/<section>/foo.py
   ./docs/build.sh serve-en    # browse English docs at http://localhost:8000
   ```

   Japanese page:

   ```bash
   ./docs/build.sh page-build docs/ja/<section>/foo.py
   ./docs/build.sh serve-ja    # browse Japanese docs at http://localhost:8000
   ```

   `page-build` is for runnable pages. It updates and executes only the
   specified paired `.ipynb` files, then rebuilds the affected language from
   `_build_src/`, where tag-related content is injected and all section
   sources are synced with jupytext. Do not use it for pages that need API
   keys or other credentials unless those credentials are configured.

### Creating a new page

1. **Start from the closest skeleton** in [docs/skeletons/](skeletons/) and copy it
   into both `en/` and `ja/` under the matching section. Follow
   [docs/skeletons/README.md](skeletons/README.md) for the page shape, tag placement, and
   authoring conventions.

2. **Update the section landing page** by adding a bullet/link in
   the matching `<section>/index.md`. Section index pages are
   hand-written, so update the visible article list and any nearby
   section-specific guidance as needed. Do not add a `## Browse by tag`
   heading or tag chip cloud yourself; those are synthesised into the
   build-dir copy automatically. The article ordering inside the
   sidebar nav is whatever your filename sorts to alphabetically
   (mystmd discovers the .ipynb files via a glob in each language's
   `myst.yml`: [en](en/myst.yml), [ja](ja/myst.yml)); use
   `01_`, `02_`, … prefixes if you need a curated order.

3. **You don't need to touch `myst.yml`** ([en](en/myst.yml) /
   [ja](ja/myst.yml)). The toc uses
   `pattern: <section>/*.ipynb`, so any new `.ipynb` you drop into
   the section is auto-discovered. (Per-tag pages and the per-tag
   listing are similarly discovered via `pattern: tags/*.md`.)

4. **Check docs-test coverage.** Pages under
   `tutorial/`, `algorithm/`, `usage/`, and `integration/` are already
   discovered by [tests/docs/test_tutorials.py](../tests/docs/test_tutorials.py)'s
   `TUTORIAL_PATTERNS`.
   If a page cannot run in CI because it needs credentials or remote
   side effects, add that specific page to `SKIP_TUTORIALS` with a
   reason. If it only needs optional packages, add the import module
   names to `OPTIONAL_SKIP_MODULES` instead of skipping the whole
   directory.

5. **Sync, execute, and build the new runnable pages**:

   ```bash
   ./docs/build.sh page-build docs/en/<section>/your_new_topic.py docs/ja/<section>/your_new_topic.py
   ```

   `page-build` executes the specified pages. For integration pages that
   need API keys or remote side effects, add an explicit docs-test skip and
   execute the notebook only in an environment with the required credentials.

6. **Preview locally**:

   ```bash
   ./docs/build.sh serve-en    # browse English docs at http://localhost:8000
   # or
   ./docs/build.sh serve-ja    # browse Japanese docs at http://localhost:8000
   ```

   `page-build` has already executed the specified page sources and rebuilt
   the affected language from `_build_src/`, where chip blocks,
   browse-by-tag clouds, and per-tag pages are injected — your committed
   source stays clean.

7. **`git add` + commit + push.**

### Adding a new section directory

The four sections currently registered (`tutorial/`, `algorithm/`,
`usage/`, `integration/`) live at the top level under
`docs/<lang>/`. Adding a new one — at the same level — is supported
and only needs config changes.

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
3. **Register the section in [docs/scripts/build_doc_tags.py](scripts/build_doc_tags.py)**:
   - Append `"<new>"` to `SECTIONS`.
   - Add `"<new>"` to `ALLOWED_TAGS` as the section tag.
   - Add `"<new>": "<English title>"` to `STRINGS["en"]["section_titles"]`
     and `"<new>": "<Japanese title>"` to `STRINGS["ja"]["section_titles"]`
     so the section's name renders correctly on tag pages.
4. **Add the section to each `myst.yml` toc** ([en](en/myst.yml) /
   [ja](ja/myst.yml)), under the
   hand-written part. Use a `pattern:` child so any future article
   in the section is auto-discovered without further toc edits:
   ```yaml
   - title: <English title>
     file: <new>/index.md
     children:
       - pattern: <new>/*.ipynb
   ```
5. **Add a top-level link in [docs/en/index.md](en/index.md) and
   [docs/ja/index.md](ja/index.md)** alongside the other section bullets.
6. **Register the section in [docs/build.sh](build.sh)**:
   - Add `"<new>"` to `SYNC_DIRS` if its `.py` sources should be synced
     to `.ipynb` during `sync`, `build`, and `page-build` workflows. This is
     usually safe even for credentialed pages because jupytext sync does not
     execute code.
   - Add `"<new>"` to `TARGET_DIRS` only if the section's notebooks can be
     bulk-executed without credentials, network access, or other side effects.
     Sections like `integration/` stay out of `TARGET_DIRS` and rely on
     page-level execution or docs-test skips for credentialed pages.
7. **Update [tests/docs/test_tutorials.py](../tests/docs/test_tutorials.py)'s
   `TUTORIAL_PATTERNS`**
   if the section's notebooks should be exercised by the docs tests:
   ```python
   "docs/en/<new>/**/*.py",
   "docs/ja/<new>/**/*.py",
   "docs/en/<new>/**/*.ipynb",
   "docs/ja/<new>/**/*.ipynb",
   ```
   Prefer adding the directory pattern and then skipping only the
   specific credentialed or side-effecting pages via `SKIP_TUTORIALS`.
8. **Update the `Directory Guide` section of this README** to
   mention the new directory.

After these steps, run `./docs/build.sh build` to verify the new section
appears in the rendered nav and that auto-managed content (chip
blocks on articles, browse-by-tag cloud on the new `index.md`)
shows up correctly inside `_build_src/`. The Tags toc itself is
discovered via the existing `pattern: "tags/*.md"` entry in each
language's `myst.yml` ([en](en/myst.yml), [ja](ja/myst.yml)), so it
does not need touching.

### Tags

Articles under `{tutorial,algorithm,usage,integration}/` (in both
`en/` and `ja/`) are tag-filterable. Tags are declared in each article's
jupytext frontmatter as shown in the skeletons.

Tags are language-agnostic (same string for `en` and `ja`). The build
pipeline runs [docs/scripts/build_doc_tags.py](scripts/build_doc_tags.py)
against `_build_src/`
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
`- pattern: "tags/*.md"` toc entry in each language's `myst.yml`
([en](en/myst.yml), [ja](ja/myst.yml)) with `hidden: true`, so the
script does **not** maintain any region inside those files. The committed
source therefore has no tag-managed regions to track — articles and
section index pages stay hand-written. The API reference has a separate
auto-generated TOC region in `myst.yml`, managed by
[docs/generate_api.py](generate_api.py).

`./docs/build.sh build`, `./docs/build.sh build-en`, and
`./docs/build.sh build-ja` run `setup_build_src` (copy → inject →
jupytext sync) before MyST builds, so RTD and local builds stay in sync
without manual steps. PRs that change tag taxonomy or article tags only
diff the actual `tags:` frontmatter line, not the chip-block churn that
comes with it.

#### Tag whitelist

Allowed tags live in `ALLOWED_TAGS` inside
[docs/scripts/build_doc_tags.py](scripts/build_doc_tags.py). The taxonomy is intentionally small.
The whitelist is **enforced by CI** via
[tests/docs/test_tag_whitelist.py](../tests/docs/test_tag_whitelist.py) — any PR that uses a tag outside
the set fails the unit-test job. If `test_tag_whitelist` fails on a PR,
fix the typo in the article frontmatter, or extend `ALLOWED_TAGS` only
when the new tag is intentional and approved.

**Adding a new tag is a deliberate maintainer decision**, not a
side-effect of writing an article. Stay within the existing set
unless the project owner has explicitly approved the new tag. Release
notes (`release_notes/`) are intentionally out of scope and never tagged.

## Troubleshooting

### "No module named 'qamomile'"

Ensure dev dependencies are installed in the active env:

```bash
uv sync
```

### Port 8000 already in use

```bash
cd docs/en/_build/html
uv run python -m http.server 8001
```

## Additional Resources

- [Jupyter Book Documentation](https://jupyterbook.org/)
- [Jupytext Documentation](https://jupytext.readthedocs.io/)
- [MyST Markdown Guide](https://myst-parser.readthedocs.io/)
- [Qamomile GitHub](https://github.com/Jij-Inc/Qamomile)

## License

Same license as the main Qamomile project.
