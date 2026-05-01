# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
uv run zuban qamomile/

# Build documentation (from docs/en or docs/ja directory)
jupyter-book build .
```

## Architecture Overview

Qamomile is a quantum programming language built around a circuit-first compiler core. The central abstraction is `@qkernel`: users write quantum programs in Python, and the compiler pipeline transforms them into executable circuits for multiple backends.

### Design Center

**`qamomile.circuit`** is the compiler core. All other modules depend on it, never the reverse.

- `qamomile/optimization/` is a domain-specific helper module for quantum optimization algorithms (QAOA, QRAO, FQAOA). It consumes circuit's public transpiler and algorithm APIs.
- `qamomile/core/` provides mathematical modeling utilities (Pauli operators, Ising/QUBO conversion).
- `qamomile/{qiskit,quri_parts,cudaq,...}/` are backend packages implementing emit passes and executors.

Dependency direction: `optimization → circuit ← backends`. No reverse dependencies.

### IR Abstraction Principle

**Keep the IR as abstract as possible; delegate concretization to the transpile target.** The IR should express *what the program means*, not *how a particular backend realizes it*. Per-qubit instruction encoding, native composite-gate equivalents, and runtime control-flow lowering are backend concerns — push them down to the backend's emit pass / `GateEmitter`.

How the codebase already follows this:

- **Vector measurement** is a single `MeasureVectorOperation` ([qamomile/circuit/ir/operation/gate.py#L410](qamomile/circuit/ir/operation/gate.py#L410)) — never expanded into N per-qubit `MeasureOperation`s at IR level. How a vector measurement turns into actual measurement instructions is delegated entirely to emit time: a backend with a native vector-measurement primitive can emit it as one operation, while backends without one can iterate per-qubit. The IR does not commit to per-qubit semantics.
- **`MeasureQFixedOperation`** lives at an even higher abstraction (HYBRID quantum measurement + classical decode). At `plan`'s pre-segmentation lowering, it is split into `MeasureVectorOperation + DecodeQFixedOperation` so segmentation can route the halves into the right (quantum / classical) segment — but **each half stays abstract**: `MeasureVectorOperation` still represents "measure this whole Vector" (not per-qubit), `DecodeQFixedOperation` is a clean classical op.
- **Composite gates** (QFT / QPE / IQFT) stay as `CompositeGateOperation` boxes; backends with a native `CompositeGateEmitter` emit a single high-level gate, others fall back to library decomposition. The IR is identical either way.
- **Loops** (`qmc.range(...)`) stay as `ForOperation`s with symbolic bounds when possible; `LoopAnalyzer` decides unroll vs. runtime loop at emit time.

When introducing a new IR op or pass:

1. Prefer **a single abstract op** over expanding into multiple low-level ops at IR level.
2. Push lowering as **late as possible** — `emit` for backend-specific concretization, `plan` only when segmentation forces a split (HYBRID → pure-quantum + pure-classical).
3. When IR-level lowering IS needed, keep each resulting op as abstract as the next stage allows.

Pre-expanding an abstract concept into per-element / per-qubit / per-step concretization at IR level — without a stated reason — is a design regression.

### Core Pipeline Flow

This is a high-level overview of the layers a `@qkernel` flows through.
For the canonical, ordered pass sequence — including every transformation
the pipeline runs and the `BlockKind` boundary each pass advances — see
"Transpiler Pipeline Stages" below.

```
@qkernel Python function
         ↓
    Frontend (AST transform → tracing → Block with operations)
         ↓
    Transpiler Pipeline (multi-pass IR rewriting; canonical sequence below)
         ↓
    Backend Execution (Qiskit, QuriParts, CUDA-Q, etc.)
```

### Key Module Structure

**qamomile/circuit/** - Compiler core:
- `frontend/`: Python decorator-based API (`@qm.qkernel`) with handle types (Qubit, Float, UInt, Bit), AST transformation, tracing
- `ir/`: Intermediate representation — `Block` (staged via `BlockKind`), `Value` (SSA-like versioning), `Operation` hierarchy
- `transpiler/`: Multi-pass compilation pipeline with `BlockKind` preconditions on each pass
- `transpiler/decompositions.py`: Shared gate decomposition recipes for backends
- `transpiler/value_resolver.py`: Unified value resolution for passes
- `stdlib/`: Built-in algorithms (QFT, IQFT, QPE)
- `estimator/`: Symbolic resource estimation (gate count, qubit count)

### Key Design Patterns

**Value Versioning (SSA-like)**: Each operation creates new Value instances with incremented versions. Metadata is intentionally preserved across versions for parameter binding continuity.

**Operation.all_input_values() / replace_values()**: Generic value access protocol. Subclasses (e.g. ControlledUOperation) override to include extra Value fields. Eliminates special-case handling in passes.

**HasNestedOps protocol**: Control flow operations (For, ForItems, If, While) implement `nested_op_lists()` / `rebuild_nested()`. Passes use this instead of isinstance chains, preventing missed control flow types.

**GateOperation factory constructors**: `GateOperation.rotation()` / `GateOperation.fixed()` ensure theta is always in operands for rotation gates. Property accessors (`theta`, `qubit_operands`) provide typed read access.

**BlockKind state machine**: `HIERARCHICAL → AFFINE → ANALYZED`. Each pass validates its expected input kind.

**SegmentationStrategy**: Pluggable execution model. `NisqSegmentationStrategy` enforces single quantum segment. Future strategies (JIT, distributed) can be added without core changes.

**MeasurementMode enum**: Formalizes backend measurement handling (NATIVE / STATIC / RUNNABLE) in the GateEmitter protocol.

**Composite Gate Emitters**: Pluggable pattern allowing backends to provide native implementations for composite gates (QFT, QPE).

### Frontend API Example

```python
import qamomile.circuit as qm

@qm.qkernel
def my_circuit(q: qm.Qubit, theta: qm.Float) -> qm.Qubit:
    q = qm.h(q)
    q = qm.rx(q, theta)
    return q

# Build to IR and transpile
transpiler = QiskitTranspiler()
executable = transpiler.transpile(my_circuit, bindings={"theta": 0.5})
```

### Transpiler Pipeline Stages

`Transpiler.transpile()` runs the following passes in order. `BlockKind`
advances as preconditions are met. Every pass other than
`validate_entrypoint` is idempotent and exposed as a public method on
`Transpiler` for step-by-step debugging; `validate_entrypoint`
(implemented by `EntrypointValidationPass`, whose `.name` is
`"validate_entrypoint"`) runs inline inside `transpile()` as a structural
check (no public method today).

```
QKernel
   │  to_block                    (trace Python AST → IR)
   │  validate_entrypoint         (internal: EntrypointValidationPass — requires classical I/O on entrypoint kernels)
   ▼
Block [HIERARCHICAL]
   │  substitute                  (optional rule-based block / strategy replacement)
   │  resolve_parameter_shapes    (concretize Vector shape dims from bindings)
   │  inline                      (remove CallBlockOperations)
   ▼
Block [AFFINE]
   │  unroll_recursion            (iterated inline ↔ partial_eval for self-recursive kernels)
   │  affine_validate             (enforce "each quantum value used at most once")
   │  partial_eval                (constant fold + CompileTimeIfLoweringPass)
   │  analyze                     (dependency graph + classical/quantum operand check)
   ▼
Block [ANALYZED]
   │  classical_lowering          (rewrite measurement-derived classical ops to RuntimeClassicalExpr)
   │  validate_symbolic_shapes    (reject unresolved Vector dims at ForOperation bounds)
   │  plan                        (segment into C→Q→C; pre-segmentation lowering of MeasureQFixed etc.)
   ▼
ProgramPlan
   │  emit                        (backend-specific codegen; LoopAnalyzer decides unroll vs runtime loop)
   ▼
ExecutableProgram[T]
```

See [docs/en/tutorial/09_compilation_and_transpilation.py](docs/en/tutorial/09_compilation_and_transpilation.py) for a step-by-step walk-through with IR dumps after each pass.

### Binding vs. Parameter Contract

**Project rule: `bindings` and `parameters` MUST be strictly disjoint** —
a kernel argument name must never appear in both. This rule is a hard
project-level constraint; treat it as inviolable when writing kernels,
helpers, or higher-level wrappers around `transpile()`. The API also
enforces it at the function boundary by raising `ValueError` immediately
on any overlap ([transpiler.py][overlap-check]); historically the absence
of this check caused silent miscompilation of control-flow predicates
depending on parameter-array elements (see #354 / 7198bfe9).

- **Never specify the same name in both `bindings` and `parameters`.**
  - `bindings={...}` — values resolved at compile time, substituted into
    the IR by `resolve_parameter_shapes` / `partial_eval`. The value is
    baked into the emitted circuit.
  - `parameters=[...]` — argument names that survive the pipeline as
    runtime parameters in the emitted backend circuit.
- **Required classical arguments (those without Python defaults) must be
  resolved exactly one way.** Bind them via `bindings`, list them in an
  explicit `parameters=[...]`, or rely on `parameters=None` to let
  `QKernel.build()` auto-detect them — auto-detect picks up classical
  arguments that have neither a `bindings` value nor a Python default and
  treats them as runtime parameters. Classical arguments with Python
  defaults may be omitted from both `bindings` and `parameters`, in which
  case the default is used.
- **Arguments driving a classical-value `if` branch (one whose condition
  is not a measurement-backed `Bit`) must be in `bindings` so
  `CompileTimeIfLoweringPass` can resolve the condition at compile time.**
  Per the no-overlap rule above, such arguments therefore cannot
  simultaneously appear in `parameters`; leaving them symbolic causes
  compilation to fail. The same applies to any other compile-time
  structural decision such as `qmc.range(...)` bounds. Measurement-backed
  `if bit:` / `while bit:` (where `bit = qmc.measure(q)`) is unrelated —
  that is runtime control flow handled at emit time by backends whose
  emitters report support for the corresponding constructs
  (`GateEmitter.supports_if_else()` / `supports_while_loop()`), with the
  appropriate measurement handling on the same backend (`MeasurementMode`)
  when mid-circuit measurement is required.

[overlap-check]: qamomile/circuit/transpiler/transpiler.py#L475-L487

## Docstring Convention (MANDATORY)

All functions, methods, and classes in `qamomile/` — **public and private alike** — MUST carry a **Google-style docstring** with the appropriate sections filled in, not just a one-line summary. This is enforced by `/local-review` (missing docstrings are P2+).

Required sections, in this order:

1. **One-line summary** (imperative mood, ending with a period).
2. *(Optional)* A longer description paragraph after a blank line.
3. **`Args:`** — one entry per parameter. Include the type in the docstring even though it is also in the signature; describe meaning, units, valid range, and default behavior.
4. **`Returns:`** — describe the returned value's type and meaning. For tuple returns, name each element. Omit this section only for functions that truly return `None`.
5. **`Raises:`** — list every exception the function can raise with the condition that triggers it. Omit only if the function genuinely cannot raise.
6. *(When helpful)* **`Example:`** — a minimal runnable snippet, especially for public API surfaces and `@qkernel` building blocks. Error classes MUST include both correct and incorrect examples.

Example:

```python
def transpile(
    self,
    kernel: QKernel,
    bindings: dict[str, Any] | None = None,
    parameters: list[str] | None = None,
) -> ExecutableProgram[T]:
    """Transpile a qkernel into a backend-specific executable program.

    Runs the full compilation pipeline (to_block → inline → partial_eval →
    analyze → plan → emit) and returns an executable bound to this backend.

    Args:
        kernel: The `@qkernel`-decorated function to compile. Must be an
            entry-point kernel with concrete (non-symbolic) shapes once
            `bindings` are applied.
        bindings: Compile-time parameter bindings, keyed by parameter name.
            Values are coerced to the parameter's declared handle type.
            Also resolves array shapes. Defaults to None, meaning no
            bindings — the kernel must then have no free parameters.
        parameters: Names of kernel parameters to preserve as backend
            runtime parameters rather than binding at compile time. Each
            name must refer to a non-array parameter of the kernel.
            Defaults to None, meaning all unbound parameters are treated
            as compile-time fixed.

    Returns:
        An `ExecutableProgram[T]` wrapping the backend circuit and the
        parameter metadata needed to re-bind runtime parameters.

    Raises:
        QamomileCompileError: If analyze/plan detects a dependency or shape
            violation in the IR.
        KeyError: If `bindings` or `parameters` contains a name that is not
            a kernel parameter.

    Example:
        >>> transpiler = QiskitTranspiler()
        >>> exe = transpiler.transpile(my_kernel, parameters=["theta"])
        >>> counts = exe.sample(
        ...     transpiler.executor(),
        ...     shots=1024,
        ...     bindings={"theta": 0.5},
        ... ).result()
    """
```

Additional rules:

- **Private helpers** (`_foo`, `__bar`) follow the same rule — a Google-style docstring with `Args`/`Returns`/`Raises` as applicable. The sections may be compact, but they must be present whenever they apply.
- **Tests** do NOT need `Args`/`Returns` sections — a clear 1–2 line description of **what the test verifies** is sufficient (see test philosophy).
- **`X | None` syntax** is the project standard in both signatures and docstrings — no `Optional[X]`.
- **Error-class docstrings** must include both a correct-usage example and an incorrect example that triggers the error.

## Test

`tests/` contains unit tests, and docs tests under `tests/docs` must also be run for documentation-impacting changes.

### Cross-Backend Execution Tests for Algorithms and Stdlib (MANDATORY)

Any addition or non-trivial modification under `qamomile/circuit/algorithm/` or `qamomile/circuit/stdlib/` **MUST** ship with tests that transpile and **actually execute** the new/changed qkernel on every supported quantum SDK backend — not just build the IR. Missing cross-backend execution coverage is treated as a P1 issue by `/local-review`.

**The supported backend matrix**:

| Backend | Transpiler | Simulator/executor |
|---|---|---|
| Qiskit | `qamomile.qiskit.QiskitTranspiler` | `qiskit.providers.basic_provider.BasicSimulator` / `qiskit_aer.AerSimulator` |
| QuriParts | `qamomile.quri_parts.QuriPartsTranspiler` | QuriParts state-vector sampler/estimator |
| CUDA-Q | `qamomile.cudaq.CudaqTranspiler` | `cudaq` state-vector simulator |

> **Note on qBraid**: `qamomile.qbraid` provides only `QBraidExecutor` (no standalone transpiler — it wraps Qiskit circuits) and requires a qBraid API key / `QbraidProvider` to execute. qBraid is **out of scope** for this mandatory matrix. Optional qBraid coverage should gate on a `QBRAID_API_KEY` env var and skip otherwise.

**Each backend test must**:

1. Guard the optional SDK dependency with `pytest.importorskip("qiskit")` (or the equivalent) at module or test scope so environments without that SDK skip rather than error.
2. Call `transpiler.transpile(my_algo_kernel, bindings=...)` — verify it returns a real `ExecutableProgram` / backend circuit.
3. **Execute both sampling AND expectation-value paths**: Run `executable.sample(shots=...)` (or equivalent) **and** `executable.run(observable=...)` / `estimate()`. Sampling and expval go through different backend primitives (sampler vs estimator) and regress independently — exercising only one is insufficient. Building the IR without running it does not satisfy this rule at all.
4. Verify each result against a reference — either an analytic expected statevector/probability distribution, or a cross-backend equivalence check (`np.allclose(result_qiskit, result_quri_parts, atol=1e-8)` for expval; shot-noise tolerance for sampling).

**Large-problem escape hatch**: When the algorithm intrinsically requires a size that cannot be locally simulated (e.g., 30+ qubits, dense Hamiltonians that blow up statevector memory), the execution requirement is relaxed — verifying that `transpile()` succeeds and the emitted circuit has the expected structure (gate/qubit counts) is sufficient. Document the reason in the test docstring. This exception applies **only** when local simulation is genuinely infeasible; prefer adding a small-scale parametrization (e.g., `n ∈ {2, 3, 4}`) that IS simulatable and reserve the transpile-only check for the large scale.

**Randomized inputs are required wherever the algorithm admits them**:

- Parametrize over seeds: `@pytest.mark.parametrize("seed", [0, 1, 2, 42])` with `rng = np.random.default_rng(seed)` inside the test body.
- Randomize every exposed degree of freedom: rotation angles, phase parameters, initial bitstrings, Hamiltonian coefficients.
- Parametrize over a small set of register sizes (e.g., `n ∈ {1, 2, 3, 5}`) — never only one size.
- Always seed randomness. Never use bare `np.random.rand()` / `np.random.randn()`.
- Include boundary inputs alongside random ones: angles `0`, `π`, `2π`; empty/single-qubit registers where supported.

**What counts as "algorithm or stdlib"**:

- New file under `qamomile/circuit/algorithm/` (e.g., new ansatz, VQE component, QAOA variant).
- New file under `qamomile/circuit/stdlib/` (e.g., QFT, QPE, IQFT, or future additions).
- Any change to an existing algorithm/stdlib qkernel that alters its IR shape, parameter interface, or gate sequence.

Pure refactors that provably preserve IR output are exempt but still encouraged to re-run cross-backend tests.

**Reverse direction — when a new SDK backend is added**: When introducing a new quantum SDK backend (a new directory under `qamomile/` with its own `Transpiler`), the PR MUST also extend the existing algorithm and stdlib test suites to include the new backend. Specifically:

- Add the new backend to every backend-parametrized test in `tests/circuit/algorithm/` and the stdlib tests (`tests/circuit/test_qft.py`, `test_qpe.py`, etc.), with an `importorskip` guard.
- If existing tests are not yet parametrized over backends, refactor them to be so as part of the SDK-addition PR — then add all current backends (Qiskit, QuriParts, CUDA-Q) plus the new one.
- The same sampling + expval + randomization requirements apply to the new backend.

Shipping a new backend without retro-actively extending algorithm/stdlib coverage leaves the backend silently unvalidated against real quantum programs.

## Documentation Translation

To translate English docs (`docs/en/`) into Japanese (`docs/ja/`), use the `/translate` skill:

```bash
/translate docs/en/tutorial/01_your_first_quantum_kernel.py
```

Translation rules (tone, spacing, terminology, soft line breaks, etc.) are defined in `.claude/skills/translate/SKILL.md`. Always use this skill when translating documentation.

## Commits, Pull Requests, and Issues

The rules below apply to any text Claude writes that lands in the project's
permanent record or on GitHub — commit messages, PR titles / bodies, issue
titles / bodies, PR review comments, code review replies, and inline source
code comments. Consult this section **before** creating any commit, PR, or
issue.

### Use English

All text that lands in the project's permanent record or on GitHub — commit
messages, PR titles / bodies, issue titles / bodies, PR / code review
comments and replies, and inline source code comments — MUST be written in
**English**. English is the project's lingua franca so that contributors
regardless of native-language background can read, search, and respond to
the shared record. Japanese (or other languages) is appropriate only in
private chat / Slack / live verbal discussion, never in checked-in text or
GitHub-tracked artifacts.

### Run `/local-review` before opening a PR

Before opening a pull request, run the `/local-review` skill against the
current branch. Address every finding it reports, then re-run
`/local-review`. Repeat the review-and-fix cycle until the skill reports
**no remaining issues** — only then create the PR. A clean `/local-review`
run is a precondition for PR creation, not a post-merge polish step.

### Reply to GitHub review threads when addressing feedback

Whenever a PR review comment is addressed — whether by a code change, a
documentation update, or a reasoned disagreement that leaves the line as
is — post a reply directly in that comment's GitHub thread describing the
resolution. A short, specific note is enough:

- "Fixed in `<commit-sha>`: `<one-line summary of the change>`."
- "Out of scope for this PR; tracked as a follow-up at `<reference>`."
- "Disagreed because `<reason>` — left as is."

Do **not** rely on the reviewer (or a review bot) to infer that a new
commit resolves their comment by reading the diff alone. Every individual
review comment should end up with at least one explicit reply explaining
the resolution. This keeps the discussion self-contained, makes the
conversation easy to audit later, and prevents the same concern from
being re-raised in the next review pass.

### Sync PR branches with `merge`, not `rebase` + force-push

When a PR branch needs to incorporate the latest `main` (for example, when
the documentation or code in this PR depends on a recent commit on `main`
that has not yet propagated to the branch's parent), use a regular
`git merge origin/main` on the PR branch, **not** `git rebase origin/main`
followed by a force-push. Force-pushing rewrites the commit SHAs that
reviewers and review bots may have already anchored their comments to,
disturbs the chronological view of the PR, and creates extra noise for
everyone re-reading the diff. A merge commit on a feature branch is
harmless and preserves the existing review thread.

### No `@`-mentions

Never include **bare** `@username` or `@org/team` mention tokens in
normal GitHub-tracked text — commit messages, PR titles / bodies, issue
titles / bodies, **and PR / code review comments and replies posted to
those threads** — because they trigger unintended GitHub notifications.
The rule covers the entire scope listed at the top of this section;
review-thread replies are not an exception, even when the reply is
short or only quotes a previous reviewer comment. This rule applies to
bare Python decorators in running prose too: refer to them
descriptively (e.g., "the qkernel decorator") instead of typing
`@qkernel` directly in prose. If you must show a literal `@…` string
(decorator syntax, an actual user handle being discussed, etc.), do so
**only** inside a fenced code block or an inline code span — GitHub
does not parse mentions in either, so a code-wrapped `@qkernel` is fine
and is the only permitted way to render the literal symbol. In normal
prose, write "the qkernel decorator" instead.

- ✅ "Update the qkernel decorator so metadata survives `next_version`."
- ❌ "Update `@qkernel` so metadata survives `next_version`." (the
  `@qkernel` is wrapped in inline code here only so this example itself
  doesn't render as a mention; outside a fenced code block or inline code
  span, never type the literal `@qkernel` symbol in prose)

### No unsolicited external links

Do not add external URLs (arXiv, blog posts, docs sites, vendor pages,
etc.) to commits, PRs, issues, **or PR / code review comments and
replies** unless the user has explicitly provided that URL in the
current conversation. When in doubt, omit the link or ask the user to
supply one. Internal references to other issues / PRs in this repo
(e.g., `#354`) are fine when factually relevant.

- ✅ "Implements the Trotter circuit (see #337 for the design discussion)."
- ❌ "Implements the Trotter circuit (see https://arxiv.org/abs/... )."
