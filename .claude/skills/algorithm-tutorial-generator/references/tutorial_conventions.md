# Qamomile tutorial conventions

Non-negotiable conventions every Qamomile tutorial (`docs/en/`) obeys. Read this before drafting and re-check against it before delivering. The section guide (`references/section_guide.md`) covers per-section content; this file covers the file-level shape.

## Table of contents

- [Jupytext header](#jupytext-header)
- [Tags frontmatter](#tags-frontmatter)
- [Topic-tag vocabulary](#topic-tag-vocabulary)
- [First two cells](#first-two-cells)
- [Import tiers by destination category](#import-tiers-by-destination-category)
- [End-to-end Qamomile workflow shape](#end-to-end-qamomile-workflow-shape)
- [Heavy-compute cells (`QAMOMILE_DOCS_TEST`)](#heavy-compute-cells-qamomile_docs_test)
- [Markdown and heading conventions](#markdown-and-heading-conventions)
- [Paper citation conventions (MyST)](#paper-citation-conventions-myst)

For the percent-format mechanics themselves (cell separators, markdown-cell `# ` prefix, etc.) see `references/jupytext_format.md`.

## Jupytext header

Every tutorial file starts with this YAML block, verbatim:

```python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
```

## Tags frontmatter

Every tutorial declares its tags inside a YAML block at the very top of the first markdown cell, **before** the H1 title. MyST consumes this block to generate the auto-rendered tag landing pages under `docs/en/tags/` (registered in `myst.yml` as `pattern: "tags/*.md", hidden: true`). The exact shape is:

```python
# %% [markdown]
# ---
# tags: [<category>, <topic1>, <topic2>]
# ---
#
# # <Tutorial title>
#
# <Abstract paragraph(s) go here — no list, see "First two cells" below.>
```

Rules for picking tags:

- **First tag = the directory category**, always. One of: `tutorial`, `algorithm`, `usage`, `integration`. This is the tag the category landing page filters on.
- **Add 0–3 topic tags** from the vocabulary below. Don't invent a new tag — the auto-generated tag pages only group what already exists, and a brand-new tag becomes a one-entry page that adds navigational noise.
- The total should be 1–4 tags.

Before delivering, check that the same tag set is reachable from at least one existing tutorial — that confirms the tag is in the vocabulary and the new file will share its landing page with kin. If a genuinely new tag is needed, surface the proposed name and rationale to the user before committing.

## Topic-tag vocabulary

Live tree as of writing. Re-scan with `grep -h "^# tags:" docs/en/**/*.py` to confirm.

| Tag | Meaning | Used in |
|---|---|---|
| `optimization` | Combinatorial or continuous optimization workflow (QUBO/Ising, JijModeling/OMMX problems). | `algorithm/qaoa_maxcut`, `algorithm/qaoa_graph_partition`, `algorithm/pce_maxcut`, `usage/binary_model`, `integration/qbraid_executor`. |
| `variational` | Variational quantum algorithm — parameterised ansatz + classical optimiser loop (QAOA, VQE, PCE). | `algorithm/qaoa_maxcut`, `algorithm/qaoa_graph_partition`, `algorithm/pce_maxcut`, `algorithm/vqe_for_hydrogen`. |
| `simulation` | Hamiltonian / time-evolution simulation (Trotter–Suzuki, exponentiation of $H$). | `algorithm/hamiltonian_simulation`, `tutorial/07_hermitian_decomposition`. |
| `encoding` | State preparation, amplitude encoding, Pauli encodings, problem-to-Hamiltonian maps. | `algorithm/mottonen_amplitude_encoding`, `tutorial/07_hermitian_decomposition`. |
| `error-correction` | QEC codes, stabilisers, syndrome decoding, transversal gates. | `algorithm/quantum_error_correction`, `algorithm/steane_code`. |
| `chemistry` | Quantum chemistry — molecular Hamiltonians, fermionic problems, VQE for molecules. | `algorithm/vqe_for_hydrogen`. |
| `primitive` | Low-level building blocks not tied to one algorithm (Trotter, multiplexers, Hermitian decomposition). | `algorithm/hamiltonian_simulation`, `algorithm/mottonen_amplitude_encoding`, `tutorial/07_hermitian_decomposition`. |
| `sample-based` | Methods that classically post-process measurement samples (QSCI, sample-based diagonalisation). | `algorithm/qsci`. |
| `resource-estimation` | Symbolic / pre-execution resource counts (gate, qubit, depth). | `tutorial/03_resource_estimation`. |

## First two cells

1. **Abstract markdown cell.** Opens with the `tags:` YAML block, then the H1 tutorial title, then a short **prose** summary of what the reader will build — typically 2–5 sentences across 1–2 short paragraphs that name the algorithm, the problem instance, and the target Qamomile API. **No numbered list, no bullet list** — the abstract is prose-only. See `references/section_guide.md` → "1. Abstract" for the full rule.
2. **Install-stub-and-imports code cell.** All `import` statements the tutorial uses go here, after the install line. The install line goes first, exactly as other tutorials have it:

   ```python
   # Install the latest Qamomile through pip!
   # # !pip install qamomile
   ```

   (The `# #` prefix keeps the `pip install` line commented out in the `.py` source but rendered as an executable magic in the `.ipynb` after `jupytext --to ipynb`.)

   Imports are *not* scattered next to the code cells that first use them — readers should see the full dependency footprint of the article in one place, at the top. See `references/section_guide.md` → "Imports go in a single up-front cell".

## Import tiers by destination category

The tutorial ships with Qamomile and is built by `jupyter-book`. Which imports are allowed depends on the destination directory.

**Always allowed (Qamomile runtime dependencies, per `pyproject.toml`):**

- `qamomile` (every public submodule: `qamomile.circuit`, `qamomile.optimization`, `qamomile.observable`, `qamomile.qiskit`, …)
- `jijmodeling`
- `ommx` / `ommx.v1`
- `qiskit`, `qiskit_aer`
- `sympy`
- Python standard library
- `numpy` — pulled in transitively by `qiskit`, used freely in existing tutorials.

**Allowed in `tutorial/`, `algorithm/`, `usage/` (doc-build dev dependencies, per `[dependency-groups].dev`):**

- `matplotlib`, `networkx`, `scipy`
- These are in the dev environment used by `jupyter-book build .`, so existing tutorials (e.g. `qaoa_maxcut.py`) import them freely. Use them where a plot or a graph layout teaches something the numbers cannot.

**Only allowed in `integration/` (gated on an optional extra):**

- `qamomile.cudaq` + `cudaq` → requires `qamomile[cudaq-cu13]`
- `qamomile.quri_parts` + `quri_parts.*` → requires `qamomile[quri_parts]`
- `qamomile.qbraid` + `qbraid` → requires `qamomile[qbraid]`

When a tutorial in `integration/` uses an optional backend, the abstract cell must tell the reader which extra to install. For tutorials in the other three categories, **do not** import optional backends — use `qamomile.qiskit.QiskitTranspiler` as the default backend, which ships with the core install.

## End-to-end Qamomile workflow shape

Every algorithm tutorial walks the reader through this pipeline, in order:

```
problem (math / JijModeling / BinaryModel)
  → encode / formulate (QUBO, Ising, or custom Hamiltonian via qamomile.observable)
  → @qkernel ansatz (or a Qamomile converter that builds one for you)
  → transpiler.transpile(kernel, bindings=..., parameters=[...])
  → executable.sample(executor, shots=..., bindings={...}).result()
    and/or
    executable.run(executor, observable=..., bindings={...}).result()
  → decode (converter.decode / BinaryModel.decode_from_sampleresult / custom)
  → interpret (print counts, visualise partition, compare to brute force)
```

The reader should see `transpiler.transpile(...)` and `executable.sample(...)` / `executable.run(...)` called with their real names in code cells — never hidden behind a helper.

## Heavy-compute cells (`QAMOMILE_DOCS_TEST`)

For any cell that runs a classical optimiser or samples many shots, gate the expensive settings so the CI build stays fast:

```python
import os
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
sample_shots = 256 if docs_test_mode else 2048
maxiter = 20 if docs_test_mode else 500
```

This is the pattern used in `docs/en/algorithm/qaoa_maxcut.py` and `docs/en/algorithm/qaoa_graph_partition.py`. Copy that shape; do not invent another knob.

## Markdown and heading conventions

- Every markdown line starts with `# ` (hash + space). An un-prefixed line breaks the cell.
- Use `$...$` for inline math and `$$...$$` for display math. Hamiltonians, objectives, and update rules should be rendered, not code-fenced as plain text.
- Heading hierarchy is **strict**:
  - H1 (`# <title>`) — the tutorial title, once, inside the Abstract cell.
  - H2 — exactly these five labels, in order, and nothing else: `## Background`, `## Algorithm`, `## Implementation`, `## Result`, `## Conclusion`. The Abstract has no H2 heading.
  - H3 — natural topic names (`### What is MaxCut?`, `### Create the Graph`, `### QAOA Ansatz`, `### Decode and Analyze Results`, …). **Exception:** every H3 directly under `## Implementation` MUST be prefixed with `Step N:` (numbered sequentially from `Step 1:`), e.g. `### Step 1: Build the BinaryModel and PCEConverter`, `### Step 2: Transpile to an Executable Circuit`, … This signals the reader is following a fixed pipeline; the `Step N:` prefix does not appear under any other H2.
  - H4 — finer subsections inside an H3 when needed (`#### Step 1: …`, `#### Feasibility Check`, …).

## Paper citation conventions (MyST)

The docs are built with MyST (`myst.yml`), which auto-renders bare DOI URLs into rich citation cards. **When citing an academic paper in body text, paste the DOI URL as a bare link — do not wrap it in `[text](url)` markdown link syntax, do not hand-type "(Farhi et al. 2014)", and do not use BibTeX/`{cite}` roles.** Qamomile tutorials currently rely solely on MyST's bare-DOI auto-citation; do not introduce a `references.bib` workflow.

The accepted form is:

```markdown
# QAOA was originally proposed by Farhi, Goldstone, and Gutmann
# (https://doi.org/10.48550/arXiv.1411.4028) as a hybrid quantum-classical
# heuristic for combinatorial optimization.
```

Notes:

- Always use the canonical `https://doi.org/<doi>` form. For arXiv preprints without a journal DOI, use the arXiv DOI (`https://doi.org/10.48550/arXiv.<id>`). Do not paste raw `arxiv.org/abs/...` URLs — MyST renders DOI URLs as citations but treats arXiv URLs as plain links.
- One DOI per cited claim; place it in parentheses right after the author names or claim it supports, like the example above.
- Where to cite: primarily in `## Background` (when the problem itself comes from a paper) and `## Algorithm` (for the original method paper, encoding schemes, or compression-rate results). Do **not** sprinkle citations through `## Implementation` — that section is about the Qamomile API, not the literature. The `## Conclusion` section is prose-only and does not carry citations.
- **Plain markdown links remain correct** for non-academic references — blog posts, GitHub repos, Qamomile docs, vendor documentation. The bare-URL rule applies only to DOI-backed papers.
- **Never invent a DOI** for a paper the user did not provide and you did not fetch. If you cannot confirm a DOI from the source the user supplied, omit the citation rather than guess.
