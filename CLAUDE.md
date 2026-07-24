# Qamomile Develping Guide

This file provides guidance to AI Agent when working with code in this repository.

## The Code Context Compass

- Code explains the "How".
- Tests explain the "What".
- Commits explain the "Why".
- Comments explain the "Why not".

## This document (CLAUDE.md) is simple.

`CLAUDE.md` should be keeped as a simple guideline.
If you want to add some context, firstly consider using skills to write the context.
However when the context must be referred by AI agent whenever, write the context in `CLAUDE.md`.

This document therefore holds only what an agent cannot recover by reading the
code, tests, or commits: design constraints, project rules, and conventions.
How the pipeline, IR, and modules actually work is documented in the code and
its docstrings — read those, don't duplicate them here. Task-specific
conventions live in skills:

- **Writing/editing code under `qamomile/`** → the `docstrings` skill (mandatory Google-style docstring convention).
- **Adding/changing an algorithm or stdlib qkernel, or a new SDK backend** → the `cross-backend-testing` skill (mandatory transpile-and-execute coverage across backends).
- **Translating docs** → the `/translate` skill.

## Build and Development Commands

```bash
# Install dependencies (using uv)
uv sync

# Run all tests
uv run pytest tests/

# Unit tests only (default, skips docs)
uv run pytest

# Docs tests only
uv run pytest -m docs -v

# All tests (unit + docs)
uv run pytest -m "" -v

# Specific tutorial
uv run pytest -m docs -k "en/tutorial/qaoa" -v

# Run a single test file
uv run pytest tests/core/test_qaoa.py

# Run a specific test
uv run pytest tests/core/test_qaoa.py::test_qaoa_converter -v

# Lint and isort with ruff
uv run ruff check qamomile/

# Format with ruff
uv run ruff format qamomile/

# Type checking with zuban
uv run zuban check qamomile/

# Build documentation (from docs/en or docs/ja directory)
jupyter-book build .
```

## Architecture Constraints

Qamomile is a quantum programming language built around a circuit-first compiler
core. The central abstraction is `@qkernel`: users write quantum programs in
Python, and the compiler pipeline transforms them into executable circuits for
multiple backends. The pass sequence, IR types, and module layout are described
in the code — walk `Transpiler.transpile()` and the `qamomile/circuit/` tree
directly. The rules below are the constraints that code review enforces but that
the code does not state for itself.

### Design Center — dependency direction

**`qamomile.circuit`** is the compiler core. All other modules depend on it,
never the reverse. Dependency direction is `optimization → circuit ← backends`;
there are **no reverse dependencies**.

- `qamomile/optimization/` — domain helpers for quantum optimization (QAOA, QRAO, FQAOA); consumes circuit's public transpiler/algorithm APIs.
- `qamomile/observable/` — backend-independent Hamiltonians and Pauli observables.
- `qamomile/optimization/binary_model/` — Ising/QUBO/HUBO modeling utilities.
- `qamomile/{qiskit,quri_parts,cudaq,...}/` — backend packages implementing emit passes and executors.

Before editing each modules, you have to read `__init__.py`.
Because a design principal is written at docstring of `__init__.py` for each modules.

### Placement of `algorithm` and `stdlib`

Treat the boundary between `qamomile/circuit/algorithm/` and
`qamomile/circuit/stdlib/` as a practical guideline rather than a strict
semantic classification.

- Put basic, reusable, or very widely used building blocks that are suitable as quantum-kernel subroutines in `stdlib`.
- Put other algorithm-oriented implementations in `algorithm`.
- Promote a component to `stdlib` when experience shows that it is broadly reused across multiple algorithms.

### IR Abstraction Principle

**Keep the IR as abstract as possible; delegate concretization to the transpile
target.** The IR should express *what the program means*, not *how a particular
backend realizes it*. Per-qubit instruction encoding, native composite-gate
equivalents, and runtime control-flow lowering are backend concerns — push them
down to the backend's emit pass / `GateEmitter`.

When introducing a new IR op or pass:

1. Prefer **a single abstract op** over expanding into multiple low-level ops at IR level.
2. Push lowering as **late as possible** — `emit` for backend-specific concretization, `plan` only when segmentation forces a split (HYBRID → pure-quantum + pure-classical).
3. When IR-level lowering IS needed, keep each resulting op as abstract as the next stage allows.

Pre-expanding an abstract concept into per-element / per-qubit / per-step
concretization at IR level — without a stated reason — is a design regression.

### Binding vs. Parameter Contract

**`bindings` and `parameters` MUST be strictly disjoint** — a kernel argument
name must never appear in both. This is a hard project-level constraint; treat it
as inviolable when writing kernels, helpers, or wrappers around `transpile()`.
The API enforces it by raising `ValueError` on any overlap; the absence of this
check historically caused silent miscompilation of control-flow predicates
depending on parameter-array elements (see #354).

- `bindings={...}` — values resolved at compile time and baked into the emitted circuit.
- `parameters=[...]` — argument names that survive the pipeline as runtime parameters in the emitted backend circuit.
- **Arguments driving a classical-value `if` branch (a condition that is not a measurement-backed `Bit`), or any other compile-time structural decision such as `qmc.range(...)` bounds, MUST be in `bindings`** so the condition can be resolved at compile time. Per the no-overlap rule they therefore cannot also appear in `parameters`. Measurement-backed `if bit:` / `while bit:` is unrelated — that is runtime control flow handled at emit time.

## Documentation

When editing anything under `docs/`, **read `docs/README.md` first** — it is the
source of truth for the docs build pipeline and page conventions. Two rules the
build does not state for itself:

- **Do NOT modify `ALLOWED_TAGS`** (`docs/scripts/build_doc_tags.py`) unless the user explicitly asks to add/remove a tag in the current conversation. A new tag is a deliberate taxonomy decision — stop and ask first.
- **Never hand-commit auto-managed content** (chip blocks, `## Browse by tag` sections, per-tag pages). They are synthesised into the gitignored build copy; the committed source stays clean.

## Commits, Pull Requests, and Issues

These rules apply to any text that lands in the project's permanent record or on
GitHub — commit messages, PR/issue titles and bodies, PR/code review comments and
replies, and inline source code comments. Consult them **before** creating any
commit, PR, or issue.

- **Use English.** All checked-in / GitHub-tracked text must be in English. Japanese (or other languages) belongs only in private chat / Slack / live discussion.
- **Run `/local-review` before opening a PR.** Address every finding, re-run, and repeat until it reports no remaining issues. A clean run is a precondition for PR creation.
- **Reply to GitHub review threads when addressing feedback.** Every review comment should end with an explicit reply describing the resolution (fixed in `<sha>` / out of scope / disagreed because …). Do not rely on the reviewer inferring it from the diff.
- **Sync PR branches with `git merge origin/main`, not `rebase` + force-push.** Force-pushing rewrites SHAs that reviewers anchored comments to; a merge commit on a feature branch is harmless.
- **No `@`-mentions.** Never write bare `@username` / `@org/team` tokens in GitHub-tracked text — they trigger notifications. This applies to Python decorators in prose too: a bare decorator reference to the qkernel decorator matches a real GitHub user named `qkernel`, so writing it un-wrapped anywhere on GitHub sends that stranger a notification. Refer to decorators descriptively ("the qkernel decorator"); show a literal `@…` only inside a fenced code block or inline code span (GitHub does not parse mentions there).
- **No unsolicited external links or private references.** Do not add external URLs, or identifiers/titles of private/internal sources (tracker keys, backlog IDs, Notion/Slack links), unless the user provided them in the current conversation. In-repo references (e.g. `#354`) are fine. The one exception is published academic preprints/papers — arXiv links/IDs, DOIs, and paper titles may be cited directly.
