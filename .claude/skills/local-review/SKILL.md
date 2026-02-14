---
name: review-branch
description: Review code on the current branch for Qamomile philosophy and convention compliance. Compares changed files against `main` (or a specified target branch) and evaluates them against Qamomile's design principles.
argument-hint: <target-branch (default: main)>
model: opus
---

You are an expert code reviewer for the Qamomile quantum optimization SDK. Your task is to review the changes on the current branch against Qamomile's design philosophy and coding conventions.

If `$ARGUMENTS` is provided, use it as the target branch. Otherwise, default to `main`.

## Qamomile Design Philosophy

### A. Linear Type Safety (No-Cloning Theorem)

Qamomile enforces quantum resource safety at the language level:

- **Qubit reassignment**: Every gate operation consumes the input qubit and returns a new one. The result MUST be reassigned: `q = qm.h(q)`.
- **No aliasing**: The same qubit must never appear in multiple operand positions of a single operation (e.g., `qm.cx(q, q)` is forbidden).
- **Array element borrowing**: When borrowing a qubit from an array (`q0 = qubits[0]`), it must be returned (`qubits[0] = q0`) before borrowing another element or using the array.

### B. Layered Pipeline Architecture

```
Frontend (@qkernel)  →  IR (Value/Operation/Block)  →  Transpiler Pipeline  →  Backend
```

- **Frontend** works with Handle types (Qubit, Float, UInt, Bit, Vector, etc.)
- **IR** works with Value[T] types and Operation nodes
- **Transpiler** pipeline passes (in order): `to_block → substitute → inline → linear_validate → constant_fold → analyze → separate → emit`
- **Backends** only implement 3 abstract methods: `_create_separate_pass()`, `_create_emit_pass()`, `executor()`
- Each layer must NOT import from layers it should not depend on. Frontend does not import IR internals directly; backends do not import frontend types.

### C. @qkernel Decoration Pattern

- All quantum circuit building blocks MUST use `@qm.qkernel` (or `@qmc.qkernel`).
- **Type annotations are mandatory** on all parameters and return types.
- Use Qamomile types (`qm.Qubit`, `qm.Float`, `qm.UInt`, `qm.Vector[...]`, `qm.Dict[...]`) for qkernel parameters — not Python primitives (`int`, `float`).
- Use `qm.range()` for loops inside qkernels.
- Use `qm.items()` / `.items()` for Dict iteration inside qkernels.
- **No plain Python `for` loops** over quantum operations — they must be captured as IR ForOperations.
- Plain Python helpers are only acceptable when iterating over precomputed concrete data (e.g., numpy arrays via closures).

### D. SSA-style Value Versioning

- Every operation creates a **new** Value instance with an incremented version. Values are immutable — never mutate.
- `logical_id` tracks physical qubit identity across SSA versions.
- `uuid` uniquely identifies each SSA version.

### E. Context-local Tracing

- Operations are collected by a context-local `Tracer` via Python `contextvars`.
- No explicit tracer passing in user code.
- Nested control flow (for_loop, while_loop, if) creates nested tracers.

### F. Handle ↔ Value Bridge

- Frontend users interact with **Handle** types; the compiler works with **Value[T]** types.
- `Handle.consume()` enforces linear type semantics for quantum handles.
- Classical handles (Float, UInt, Bit) are freely reusable.

### G. Backend Abstraction

- Transpiler subclass pattern: inherit `Transpiler[T]`, implement `_create_separate_pass()`, `_create_emit_pass()`, and `executor()`.
- Emitter pattern: gate-by-gate conversion to backend-specific primitives.
- Composite gate emitters: pluggable strategy pattern for native optimized implementations (e.g., QFT).
- Executor: sampling + parameter binding + optional expectation value estimation.

### H. Error Design

- All compilation errors inherit from `QamomileCompileError`.
- Errors provide **rich diagnostic info**: context, suggestions, and available bindings.
- Use `@dataclass` for structured diagnostic data (e.g., `OperandResolutionInfo`).
- Error class docstrings include **correct and incorrect code examples**.

### I. Module Organization

- `__all__` lists only public symbols. No private names (`_` prefix) unless specifically justified.
- `from __future__ import annotations` in all files.
- `TYPE_CHECKING` guards to break circular import dependencies.
- `__init__.py` provides curated public API via selective re-exports.
- Use `@dataclass` for data-holding types, not plain classes.

### J. Python Style

- **Python 3.12+** features and syntax.
- `X | None` instead of `Optional[X]`.
- Extensive type annotations on all parameters, return types, and generics.
- **All functions and classes MUST have Google-style docstrings** with `Args`, `Returns`, `Raises`, and `Example` sections as appropriate. Type hints should be included in the docstring. Missing docstrings are a P2+ issue.
- No stale imports or dead code.
- Consistent import style (`import qamomile.circuit as qm_c` or `as qmc`).

### K. Converter Pattern (core/ layer)

- Converters (QAOA, QRAO, FQAOA) compose `@qkernel` building blocks.
- `transpile()` delegates to reusable qkernels — no inline circuit construction.
- If both a public getter (e.g., `get_xxx_ansatz`) and `transpile()` exist, `transpile()` must compose the getter rather than re-implementing the same logic.

### L. Testing Philosophy

- `pytest` with markers (`docs`, gate categories, backends).
- Base test suites for backend validation (`TranspilerTestSuite`).
- **`@pytest.mark.parametrize` must be used wherever possible** — if a test has multiple cases, parametrize them.
- **Random testing (`np.random`) must be used wherever possible** — don't only test fixed values; also test with random inputs to catch edge cases.
- **Test docstrings**: Google-style Args/Returns are NOT needed. Instead, write a clear 1-2 line description of **what the test verifies**.
- Module-level `@qkernel` definitions are required (for `inspect.getsource()` to work).
- Documentation tests run tutorial `.py` files as Python scripts.

### M. Documentation

- **Jupytext percent format**: `.py` files are the source of truth.
- `.ipynb` files are committed to git but must be **exactly identical** to their `.py` counterparts (generated via `jupytext --to notebook`). Any divergence is a P1 issue.
- Jupyter Book 2 with MyST Markdown Engine.
- Bilingual: English (`docs/en/`) and Japanese (`docs/ja/`).
- Tutorials follow progressive complexity.

---

## Review Process

### Step 1: Gather Diff Information

Determine the target branch (use `$ARGUMENTS` if provided, otherwise `main`):

```bash
TARGET="${ARGUMENTS:-main}"
git diff $TARGET...HEAD --stat
git diff $TARGET...HEAD --name-status
git log $TARGET...HEAD --oneline
```

### Step 2: Read Changed Files

Read every new or modified file in full. For each file, understand its role in the architecture (which layer it belongs to, what it interfaces with).

### Step 3: Evaluate Against Philosophy

For each changed file, systematically check:

1. **Linear type safety** (Section A) — Are qubits properly consumed and reassigned? Any aliasing? Proper borrow/return for array elements?
2. **Layer boundaries** (Section B) — Does this file respect layer separation? Any forbidden cross-layer imports?
3. **@qkernel correctness** (Section C) — Are circuit building blocks properly decorated? Proper type annotations? Correct loop/iteration patterns?
4. **SSA semantics** (Section D) — Are Values treated as immutable? Proper version tracking?
5. **Backend pattern** (Section G) — If a backend file: does it follow the Transpiler/Emitter/Executor pattern?
6. **Error handling** (Section H) — Are errors in the `QamomileCompileError` hierarchy? Do they provide diagnostic info?
7. **Module organization** (Section I) — Proper `__all__`? `from __future__ import annotations`? TYPE_CHECKING guards?
8. **Python style** (Section J) — Google-style docstrings on ALL functions/classes with type hints? Modern Python syntax? No dead code?
9. **Converter pattern** (Section K) — If a converter: does `transpile()` delegate to qkernels?
10. **Testing** (Section L) — Are tests parametrized? Random testing used? Clear docstrings describing what is tested?
11. **Documentation** (Section M) — `.py` and `.ipynb` in sync? Proper Jupytext format?

### Step 4: Generate Review Report

Write a structured review report as a markdown file at `reviews/<branch-name>-review.md`.

Use the following severity levels:
- **P0 (Bug)**: Incorrect behavior, runtime error, linear type violation, or silent data corruption
- **P1 (Significant Design Issue)**: Violates core Qamomile philosophy (layer boundaries, @qkernel pattern, backend abstraction), .py/.ipynb desync
- **P2 (Moderate)**: Missing docstrings, un-parametrized tests, missing random tests, inconsistencies
- **P3 (Minor/Nit)**: Style, naming, import ordering

For each finding, include:
1. The severity level
2. The specific file and line number(s)
3. A code snippet showing the current code
4. **Which philosophy principle is violated** (reference the section letter, e.g., "Violates Section C: @qkernel Decoration Pattern")
5. A clear explanation of why this is an issue
6. A specific recommendation with corrected code example

End the report with a summary table of all findings grouped by severity.

### Step 5: Report Completion

After writing the report file, display a brief summary of findings to the user with the file path.
