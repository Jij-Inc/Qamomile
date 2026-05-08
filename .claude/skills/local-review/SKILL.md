---
name: local-review
description: Review code on the current branch for Qamomile philosophy and convention compliance. Compares changed files against `main` (or a specified target branch) and evaluates them against Qamomile's design principles.
argument-hint: <target-branch (default: main)>
model: opus
---

You are an expert code reviewer for the Qamomile quantum optimization SDK. Review the changes on the current branch against Qamomile's design philosophy and coding conventions.

If `$ARGUMENTS` is provided, use it as the target branch. Otherwise, default to `main`.

## Severity Levels

Use these labels consistently when reporting findings (referenced by rules below and re-applied in Step 6):

- **P0 — Bug**: incorrect behavior, runtime error, linear-type violation, silent data corruption, breaking change in unchanged files, mutable default arg, bare `except` swallowing, float `==` in production code, missing exception chaining.
- **P1 — Significant Design Issue**: layer-boundary / `@qkernel` / backend-pattern violation; missing / stale / unexecuted `.ipynb`; new docs not in test patterns; incomplete modifications (changed interface, callers not updated, `__all__` stale); missing tolerance in numerical asserts; missing tests or docs for a new feature; OMMX parity break; missing cross-backend execution coverage for new algorithm/stdlib.
- **P2 — Moderate**: missing docstrings, un-parametrized tests, missing randomization, missing edge / negative tests, missing deterministic-test justification or completeness, raw-string dispatch on closed sets, unnecessary copies in hot paths, unjustified incidental HUBO/QUBO guard, plain `range(...)` in a qkernel body (use `qmc.range(...)` — `qmc` is the preferred `qamomile.circuit` alias; `qm.range(...)` works but is discouraged in new code).
- **P3 — Minor**: style, naming, import ordering, per-call dict construction, suboptimal generator/list choice.

## Qamomile Review Rules

### A. Linear Type Safety (No-Cloning Theorem)

- **Qubit reassignment**: every gate consumes the input qubit and returns a new one. The result MUST be reassigned: `q = qm.h(q)`.
- **No aliasing**: the same qubit must never appear in multiple operand positions of one operation (e.g., `qm.cx(q, q)` is forbidden).
- **Array element use**: in-place forms are canonical — `qubits[i] = qm.h(qubits[i])` for single-qubit gates, `qubits[i], qubits[j] = qm.cx(qubits[i], qubits[j])` (tuple assignment) for multi-qubit gates. These are always OK. If you take an element into a local name (`q0 = qubits[0]`), you must return it (`qubits[0] = q0`) before re-borrowing the same index **into a local name** or before using the array as a whole. Flag **double-borrow** of the same element into different locals, not the tuple-assignment form.

### B. Layer Dependency Direction

`Frontend (@qkernel) → IR → Transpiler Pipeline → Backend`

Dependencies flow only downstream. The practical rules:

- **IR** depends on nothing — it is the shared data model.
- **Frontend** depends only on IR. It MUST NOT import from Transpiler or Backend.
- **Transpiler passes** are IR-centric: each pass primarily reads / writes IR and must not depend on Frontend **implementation details** (AST machinery, tracer internals, frontend-only helpers). Passes at the compile-entry / config boundary may legitimately accept or type-reference Frontend surface types such as `QKernel` and `DecompositionConfig` as configuration inputs — e.g., `SubstitutionPass` takes `QKernel` as a substitution target and imports it under `TYPE_CHECKING` + runtime-late import (see `qamomile/circuit/transpiler/passes/substitution.py:34,276`). That pattern is NOT a layer violation. Flag only passes that reach into Frontend behavior or non-boundary internals.
- **Backend** depends on IR and Transpiler public APIs. Backend MUST NOT import from Frontend.

### B-bis. IR Abstraction Level

Qamomile prefers to **keep the IR as abstract as possible and delegate concretization to the transpile target** (backend emit / runtime). The IR encodes *what the program means*; how a backend realizes it (per-qubit instruction encoding, native composite-gate equivalents, runtime loop / branch lowering) is the backend's job. Concretizing too early at the IR layer locks Qamomile into a single backend's view and bypasses the emit-time extension surfaces — `GateEmitter` (per-gate emission), `CompositeGateEmitter` (native lowering of composite gates), and the emit pass itself when something needs structural customization.

Existing examples of the principle in code:

- **Vector measurement** is a single `MeasureVectorOperation` (`qamomile/circuit/ir/operation/gate.py#L410`) — not expanded into N per-qubit `MeasureOperation`s at IR level. How it turns into actual measurement instructions is left to emit time: backends with a native vector primitive can emit one operation, others can iterate per-qubit. The IR does not commit to per-qubit semantics.
- **`MeasureQFixedOperation`** is HYBRID (quantum measurement + classical decode). It is split into `MeasureVectorOperation + DecodeQFixedOperation` only at `plan`'s pre-segmentation lowering — late enough that the IR keeps the highest abstraction that still admits a clean classical/quantum segmentation boundary, and each resulting half remains as abstract as possible (no per-qubit expansion).
- **Composite gates** (QFT / QPE / IQFT) stay as `CompositeGateOperation`; native lowering is opt-in via `CompositeGateEmitter`.
- **Symbolic loop bounds** stay as `ForOperation`; `LoopAnalyzer` decides unroll-vs-runtime-loop at emit time.

Reviewer rules:

- A new IR op or transpiler pass that pre-expands an abstract concept into per-element / per-qubit / per-step concretization at IR level — **without** a stated reason such as enabling segmentation or breaking a HYBRID kind into pure halves — is a **P1** design regression.
- A new pass that performs concretization (per-qubit expansion, native-gate substitution, loop unrolling) at a stage **earlier than necessary** — when the same lowering could happen later (typically `plan` or `emit`) — is **P1**.
- A new pass or op that hard-codes backend-specific lowering inside the IR layer — instead of leaving it to emit time where `GateEmitter`, `CompositeGateEmitter`, or the emit pass itself can express the lowering — is **P1** (also a Section B violation).
- Splitting a HYBRID op into pure-quantum + pure-classical halves at IR level **is** acceptable when needed for segmentation (`MeasureQFixed → MeasureVector + DecodeQFixed` is the canonical example). The check is whether each resulting half stays as abstract as possible.

When ambiguous, prefer (a) a single abstract op over multiple low-level ones, and (b) lowering at the **latest stage** where the IR still cleanly expresses the abstraction.

### C. @qkernel & Converter Pattern

- Quantum building blocks exposed for composition by algorithm or optimization converters (QAOA / QRAO / FQAOA ansatz pieces, stdlib algorithms like QFT / QPE / IQFT, mixer / cost-Hamiltonian prep, etc.) MUST use `@qm.qkernel` (or `@qmc.qkernel`). `CompositeGate._decompose()` methods and emitter-side decomposition strategies are separate patterns (class method / strategy protocol) and are out of scope for this rule.
- **Type annotations are mandatory** on all parameters and return types, using Qamomile types (`qm.Qubit`, `qm.Float`, `qm.UInt`, `qm.Vector[...]`, `qm.Dict[...]`) — not Python primitives.
- Loops inside a qkernel body over quantum operations MUST use `qmc.range(...)` (project convention — `qmc` is the preferred `qamomile.circuit` alias). The AST transformer technically also accepts plain `range(...)` and the legacy `qm.range(...)` alias, but the convention is to make quantum-side iteration explicit via `qmc`; older code using `qm.range(...)` is tolerated but new code should use `qmc.range(...)`. Dict iteration can use **either** the native `for k, v in mapping.items()` form or the Qamomile form `for k, v in qmc.items(mapping)` — both are accepted (see `qamomile/circuit/algorithm/fqaoa.py:115,118` for the `qmc.items` style). Plain Python iteration over arbitrary iterables (`for x in some_list:`) is **not** captured as an IR `ForOperation` — restrict that form to iteration over precomputed concrete data used via closures (e.g., numpy arrays referenced from within the kernel body).
- Converters (QAOA, QRAO, FQAOA) compose `@qkernel` building blocks; `transpile()` delegates rather than inlining circuit construction. If both a public getter (`get_xxx_ansatz`) and `transpile()` exist, `transpile()` must compose the getter.

### D. Backend Abstraction

Applies to backend files under `qamomile/{qiskit,quri_parts,cudaq,qbraid,...}/`.

- Transpiler subclass: inherit `Transpiler[T]`, implement `_create_segmentation_pass()`, `_create_emit_pass()`, `executor()`.
- Emitter pattern: gate-by-gate conversion to backend primitives.
- Composite gate emitters: pluggable strategy for native optimized implementations (e.g., QFT).

### E. Error Design & Exception Safety

- All compilation errors inherit from `QamomileCompileError` and carry rich diagnostics (context, suggestions, available bindings). Use `@dataclass` for structured diagnostic payloads.
- **Exception chaining**: `raise NewError(...) from e` when re-raising.
- **No bare `except`** and no unconditional `except Exception:`. Catch specific types; graceful fallbacks must `warnings.warn()` the reason.
- **Exception safety**: stateful operations (tracer state, value versioning) must clean up on error paths.
- Error-class docstrings include both a correct and an incorrect usage example.

### F. Module Organization

- `__init__.py` provides the curated public API via `__all__` (no `_`-prefixed names in `__all__` without justification). `TYPE_CHECKING` guards break circular imports.
- Use `@dataclass` for state-oriented types with field-wise equality; use a regular class when invariants, encapsulation, or custom equality matter.

### G. Python Style

- **Python 3.11+** (per `pyproject.toml`'s `requires-python`), `X | None` (not `Optional[X]`), extensive type annotations. `enum.StrEnum` is available from 3.11 and is the preferred pattern for closed-set parameters (see later in this section).
- **No stale imports** left behind from a rename / refactor (dead code at function / class level is covered in Step 5.5's root-cause consolidation).
- **Google-style docstrings on ALL functions, methods, and classes** (public and private), with `Args` / `Returns` / `Raises` as applicable and `Example` where helpful. Private helpers may use compact sections but must keep the structure. Tests are exempt from `Args` / `Returns` — a 1–2 line description of what is verified suffices. See CLAUDE.md's "Docstring Convention (MANDATORY)". Missing docstrings are P2+.
- **Closed-set parameters as Enum**: a public parameter with a finite set of valid strings (mode, method, algorithm variant, backend key) MUST be defined as `enum.StrEnum` and dispatched via the Enum internally. Signatures accept `str | MyEnum` and normalize at the entry (`method = MyEnum(method)`; on failure `raise ValueError` listing the valid values). Internal dispatch via `match` (small sets, see Section L) or a `ClassVar[dict[MyEnum, ...]]` (8+ variants). **Never** construct a `dict[str, callable]` inside a hot-path method every call. Raw string `==` dispatch on closed sets is P2; per-call dict construction is P3.

### H. Testing Philosophy

- `pytest` with markers (`docs`, gate categories, backends); reuse base test suites (`TranspilerTestSuite`). Module-level `@qkernel` definitions are **recommended** for qkernels that are reused across tests or that rely on `inspect.getsource` at runtime; function-local `@qkernel` definitions (inside a test method) are acceptable for one-off test helpers.
- **Parametrize** (`@pytest.mark.parametrize`) and **randomize** (`np.random.default_rng(seed)` with fixed seeds) wherever inputs admit it.
- **Test docstrings**: a 1–2 line description of what is verified — not Google-style `Args` / `Returns`.
- **Edge cases** (empty, single-element, `0` / `π` / `2π`) and **negative tests** (`pytest.raises(ExpectedType)`) are mandatory.
- **Numerical assertions**: `np.allclose` / `np.testing.assert_allclose` with explicit `atol`/`rtol`. Never element-wise `==` on floats.
- **Deterministic-test justification**: when an asserted value requires derivation (energy formulas, known optima, hand-rolled reference outputs), the docstring or a nearby comment MUST show how it was derived. A bare `assert np.isclose(energy, -8.0)` without justification is P2. Trivial cases (`f(0) == 0`) are exempt.
- **Deterministic-test completeness**: on deterministic code paths, assert ALL deterministic outputs. If a function returns `(sample, energy, count)` and the test checks only `energy`, regressions in the others go unnoticed. Missing asserts on deterministic outputs are P2. For parametrized cases with per-case deterministic properties, assert those per-case — do not collapse to the weakest shared assertion.
- **Test isolation**: no shared mutable state; use `@pytest.fixture`.

### H-bis. Algorithm / Stdlib Cross-Backend Execution Tests (MANDATORY)

Any addition or non-trivial modification under `qamomile/circuit/algorithm/` or `qamomile/circuit/stdlib/` MUST ship with tests that **transpile AND execute** the new/changed qkernel on every supported backend — not just build the IR. Pure refactors that provably preserve IR output (same gate sequence, same shapes) are exempt — still advisable to re-run but not enforced.

**Backend matrix**:

| Backend | Transpiler | Executor |
|---|---|---|
| Qiskit | `qamomile.qiskit.QiskitTranspiler` | `BasicSimulator` / `qiskit_aer.AerSimulator` |
| QuriParts | `qamomile.quri_parts.QuriPartsTranspiler` | QuriParts state-vector sampler/estimator |
| CUDA-Q | `qamomile.cudaq.CudaqTranspiler` | `cudaq` state-vector simulator |

qBraid is out of scope (executor-only wrapper around Qiskit, requires API key). Optional coverage may gate on `QBRAID_API_KEY`.

**Required per backend** (parametrize or one test each):

1. `transpiler.transpile(kernel, bindings=...)` returns a real `ExecutableProgram`.
2. **Exercise BOTH paths**: a sampling path (`executable.sample(executor, shots=...)`) AND an expval path. The expval path goes through `qmc.expval(q, hamiltonian)` inside the kernel — `executable.run(executor, ...)` then returns an `ExpvalJob` — or, when working directly with the emitted circuit, the backend executor's `estimate(circuit, hamiltonian)` entry point. Sampler and estimator regress independently; exercising only one is insufficient.
3. Verify against an analytic reference or cross-backend agreement (`np.allclose` for expval; statistical tolerance for sampling).
4. Guard SDK deps with `pytest.importorskip("<sdk>")`.

**Randomize** where the algorithm admits it: parametrize over seeds, register sizes (`n ∈ {1, 2, 3, 5}` — not a single `n`), and every degree of freedom (angles, phases, bitstrings, coefficients). Always seed via `np.random.default_rng(seed)`. Include boundary inputs (`0`, `π`, `2π`; single-qubit registers where applicable).

**Large-problem escape hatch**: when local simulation is genuinely infeasible (30+ qubits, dense Hamiltonians that blow up statevector memory), `transpile()` success + structural checks (gate / qubit count) suffice. Document the reason in the docstring. Prefer a small-scale parametrization that IS simulatable and reserve the escape hatch for the large scale.

**Reverse direction**: when a new backend SDK is added, extend every existing `tests/circuit/algorithm/` and stdlib test (`test_qft.py`, `test_qpe.py`, ...) to cover it — refactoring the tests to be backend-parametrized if needed — with the same sample + expval + randomization requirements.

**Severity**: missing cross-backend execution is **P1**; fixed-inputs-only is **P2**; sampler-only or estimator-only when simulatable is **P2**; missing reverse-direction retro-extension in a new-backend PR is **P1**.

The **documentation** counterpart of this rule lives in Section I-bis below — every new algorithm / stdlib / backend that this section requires tests for *also* requires a corresponding doc article. Apply both checks together when reviewing such a PR.

### I. Documentation

- **Jupytext percent-format `.py` is the source of truth.** Every tutorial `.py` must have a committed `.ipynb` that (a) exists, (b) stays in sync when its `.py` changes, (c) contains execution outputs. Any of these failing is **P1**. An `.ipynb`-only change (no corresponding `.py` update) is **P2** — it bypasses the source-of-truth.
- **Docs test coverage**: new tutorial paths (outside `integration/`) must be in `TUTORIAL_PATTERNS` in `tests/docs/test_tutorials.py`. Missing is **P1**.
- **en/ja parity**: `docs/en/` and `docs/ja/` must share file structure and content — only the natural language differs. Missing or outdated counterpart is **P1**.
- **Tag whitelist (`ALLOWED_TAGS` in `docs/scripts/build_doc_tags.py`)**: every `tags:` entry in an article's MyST frontmatter MUST already be in `ALLOWED_TAGS`. Adding a new tag to the whitelist as a side-effect of introducing an article is **P1** — the taxonomy is deliberately small and curated, and an unannounced expansion of it is precisely what the whitelist exists to prevent. If a new tag is genuinely needed, the PR should call it out explicitly and the `ALLOWED_TAGS` change should be a separate, documented decision (not a quiet line in a docs PR).
- **Google Colab pip install cell**: every article under `docs/{en,ja}/{tutorial,algorithm,usage,integration}/` MUST have a commented-out pip install cell as its **first code cell** (right after the intro markdown, before any `import qamomile…`), in the project standard form:

  ```python
  # %%
  # Install the latest Qamomile through pip!
  # # !pip install qamomile
  ```

  Readers open these notebooks via the "Open in Colab" button, where Qamomile is not pre-installed. The leading `# ` keeps the cell a no-op for the local docs build; Colab users uncomment it. Rules:
  - The install line MUST always include `qamomile` so the cell works on a fresh Colab VM.
  - If the article uses an optional Qamomile extra (a backend transpiler/executor behind a `[…]` extra in `pyproject.toml`), install with that extra: `qamomile[quri_parts]` for `qamomile.quri_parts.QuriPartsTranspiler` / `QuriPartsExecutor`, `qamomile[qbraid]` for `qamomile.qbraid.QBraidExecutor`, `qamomile[cudaq-cu13]` (or `cudaq-cu12`) for `qamomile.cudaq.CudaqTranspiler`.
  - If the article needs additional non-Qamomile packages (e.g. `openfermion`, `pyscf`, `openfermionpyscf`), put them on the same `pip install` line so the cell is self-contained.
  - JA mirror localises the comment but keeps the shell command identical.

  Missing pip install cell, or an install line whose extras / additional packages do not cover what the notebook actually imports, is **P2**. (See `docs/README.md` "Creating a new page" step 2 for the full convention.)
- Jupyter Book 2 with MyST.

### I-bis. New-Feature Documentation Coverage (MANDATORY)

Mirrors Section H-bis on the documentation side. Tests prove a feature *works*; docs prove a feature *exists* for users. Any addition or non-trivial modification that introduces a new public-facing surface MUST ship with corresponding documentation, **in both `docs/en/` and `docs/ja/`**.

**What needs a doc article**:

| Change in `qamomile/` | Required doc surface |
|---|---|
| New file under `qamomile/circuit/algorithm/` (new ansatz, VQE component, QAOA / QRAO / FQAOA variant, optimization helper) | New article under `docs/{en,ja}/algorithm/<name>.py` with a runnable example |
| New file under `qamomile/circuit/stdlib/` (composite gate, primitive routine like QFT / QPE / IQFT) | A dedicated article under `docs/{en,ja}/tutorial/` (or `algorithm/` if its primary use is algorithmic), **or** a clearly added section in an existing article that covers the family |
| New SDK backend (new directory under `qamomile/` with its own `Transpiler` / `Executor`) | Article under `docs/{en,ja}/integration/<backend>.py`, mirroring `integration/qbraid_executor.py` |
| New top-level public module (e.g. a peer of `qamomile.linalg`, `qamomile.observable`) | Dedicated article under `docs/{en,ja}/usage/` |
| Large new public-API surface inside an existing module (new public class, new public top-level function exported via `__all__`) | At minimum, an existing tutorial / usage / algorithm article touched to introduce the surface |
| New optional extra in `pyproject.toml`'s `[project.optional-dependencies]` | Consuming article(s) install with the right `qamomile[<extra>]`; `docs/README.md`'s extras list and Section I above are extended |

**What counts as "non-trivial modification"** (mirrors H-bis): IR shape change, parameter / signature change, gate-sequence change, semantic change observable from the public API. Pure refactors that preserve public behavior are exempt — but if the existing article was already wrong / outdated, this is the moment to fix it.

**Required content per new article**:

- Standard jupytext percent-format header.
- MyST `tags:` frontmatter with the section tag (`tutorial` / `algorithm` / `usage` / `integration`) plus topical tags, all within `ALLOWED_TAGS`.
- Google Colab pip install cell as the first code cell, with the right extra(s) per Section I.
- A complete runnable example exercising the new feature, with executed `.ipynb` outputs committed.
- en/ja parity: the JA mirror has the same structure / cells / outputs, with prose translated.
- (For new tutorial / algorithm / usage paths) the path added to `TUTORIAL_PATTERNS` in `tests/docs/test_tutorials.py`.

**Severity**:

- New algorithm/stdlib file with **no** corresponding docs article (en or ja or both): **P1**.
- New SDK backend without `integration/<backend>.py` article: **P1**.
- Article exists in `en` but not `ja` (or vice versa): **P1** (also a Section I "en/ja parity" violation — report once with both sections cited).
- Existing article using outdated parameter signatures / removed APIs / removed extras after the diff: **P1**.
- New top-level public module without a `usage/` article: **P1**.
- New large public-API surface inside an existing module documented only in docstrings, with no tutorial / algorithm / usage article touching it: **P2**.

When the PR description explicitly defers documentation to a follow-up tracked elsewhere (a linked issue, a written plan), downgrade by one level (P1 → P2, P2 → P3) but still report — the deferral is part of the public record, not a silent omission.

### J. Numerical Correctness

- **No `==` / `!=` on floats**: use `math.isclose()`, `np.isclose()`, or the project's `is_close_zero()` utility. In tests, use `np.allclose` / `np.testing.assert_allclose` with explicit tolerances (Section H).
- Document tolerance assumptions when the computation chain is long. Docstrings should note behavior for extreme values (very small / large coefficients, near-zero) when relevant.

### K. Performance & Memory

- **No mutable default arguments**: `def f(x: list[int] | None = None): x = x if x is not None else []`.
- Avoid redundant `.copy()` in hot paths; prefer views or safe in-place ops.
- Prefer generators over materialized lists when the result is consumed once.

### L. Defensive Programming

- **Exhaustive branching**: `if-else` and `match` MUST include an `else` / default branch. When no action is needed, leave an explicit comment or `assert False, "unreachable"`.
- **`assert` vs `raise`**: `assert` for internal invariants (programmer error / specification unreachables). `raise` (with a `QamomileCompileError`-family type when in the compile pipeline) for conditions triggered by user input or external data.

### M. Optimization Module Conventions

Applies to changes under `qamomile/optimization/`.

- **OMMX parity for `BinaryModel`-touching changes**: if a change introduces or modifies code that produces, consumes, or factory-constructs a `BinaryModel` (new constructors, new algorithms returning `BinarySampleSet`, new converters), verify the OMMX ingest/emit path is preserved or extended.
  - New build surface → OMMX `Instance` ingestion parity. Silent divergence is **P1**.
  - New result emitter → output must round-trip into OMMX (`Solution` / `SampleSet`, directly or via `BinarySampleSet` / `decode_from_sampleresult`). A terminal output with no OMMX representation is **P1**.
  - If OMMX is intentionally out of scope, the PR description or a code comment MUST state why.
- **HUBO/QUBO restriction audit**: when code rejects higher-order terms (`if model.order > 2: raise ValueError(...)`) or restricts itself to quadratic Ising, classify the restriction:
  - *Fundamental*: the formula genuinely cannot generalize. Keep the guard; the code or docstring must state why.
  - *Incidental*: the core computation (`calc_energy`, `ΔE_k = -2·s_k·Σ_{inds∋k} coeff·Π s_i`, etc.) generalizes in one line and the restriction is only there because a quadratic-specific helper was written. Propose lifting, or at minimum document that it is a convenience, not a mathematical limit. Unjustified incidental guards are **P2**.
  - If the guard stays, a negative test asserting the correct exception on HUBO input is required. If lifted, randomized HUBO coverage per Section H.

---

## Review Process

### Step 1: Gather Diff

```bash
TARGET="${ARGUMENTS:-main}"
git diff $TARGET...HEAD --stat
git diff $TARGET...HEAD --name-status
git log $TARGET...HEAD --oneline
```

### Step 2: Read Changed Files

Read every new or modified file in full. Understand each file's role in the architecture.

### Step 3: Explore Related Files

Scale exploration to diff size:

- **1–3 files**: check direct imports both ways.
- **4–10 files**: additionally, Grep for call sites of any added / removed / renamed / re-signatured symbol.
- **11+ files**: full import graph + symbol usage + interface consumer trace + pipeline-pass interaction check.

Always: if a public symbol was added, removed, or renamed, verify `__init__.py` / `__all__` were updated.

### Step 4: Evaluate Against Rules

For each changed file, walk through Sections A–M and apply the ones relevant to that file. Each scoped section (D, H-bis, M) declares its scope in its opening lines; the remaining sections apply broadly.

### Step 5: Impact & Completeness

After rules review, look for:

- **Incomplete propagation**: signature changes not reflected in every caller; `__all__` / `__init__.py` not updated; new behavior not covered by tests; `.py` / `.ipynb` / en / ja not kept in sync.
- **Latent bugs in unchanged files**: unchanged code relying on behavior the diff silently altered (return-type change, default-value change, formerly-`None` now raises); enum / constant additions that existing `match` / `if-elif` chains don't handle; pipeline pass ordering invalidated by a new pass.
- **Performance / compilation-time impact**: for transpiler-pass, emitter, or IR-node changes, flag any change that plausibly affects compilation time, emitted circuit size, or runtime execution — even if not a regression in correctness.

### Step 5.5: Root-Cause Consolidation

Mentally simulate applying all proposed fixes simultaneously and re-evaluate the finding set:

- **Moot**: a finding that becomes irrelevant when another fix lands (e.g., style nits on code that's actually dead). Drop or replace with the root cause.
- **Dependent**: several findings stemming from one underlying issue. Report the root cause with a **"Root cause of:"** annotation listing the surface issues.
- **Emergent**: a new issue revealed only after the simulated fixes (e.g., removed dead code leaves a stale `__all__` re-export). Add it.

Severity of a consolidated finding inherits the highest severity among root cause and dependents. Three or more P3 findings under one architectural issue elevate to at least P2. If consolidation is still shifting after a second pass, stop and note it in the final report.

Common root-cause patterns to watch for: **dead code** (multiple nits on code never called — root cause is deletion); **missing abstraction** (repeated logic across findings — root cause is extract into a shared helper / qkernel); **incomplete rename / refactor** (mismatched names, broken references, stale imports); **wrong layer** (one misplaced import cascading into Section B violations); **type / signature ripple** (a type change not propagated to callers).

### Step 6: Report

For each finding, give: severity (using the **Severity Levels** rubric at the top of this document), `file:line`, code snippet, the violated section, a short explanation, a concrete recommendation with corrected code, and — for consolidated findings — a `Root cause of:` line listing subsumed surface issues. End with a severity-grouped summary table.

If Step 5.5 did not stabilize, append: "Note: some findings may have deeper interdependencies warranting further investigation."
