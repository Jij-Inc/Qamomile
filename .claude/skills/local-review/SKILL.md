---
name: local-review
description: Review code on the current branch for Qamomile philosophy and convention compliance. Compares changed files against `main` (or a specified target branch) and evaluates them against Qamomile's design principles.
argument-hint: <target-branch (default: main)>
model: opus
---

You are an expert code reviewer for the Qamomile quantum optimization SDK. Your task is to review the changes on the current branch against Qamomile's design philosophy and coding conventions.

If `$ARGUMENTS` is provided, use it as the target branch. Otherwise, default to `main`.

## Qamomile Review Rules

### A. Linear Type Safety (No-Cloning Theorem)

Qamomile enforces quantum resource safety at the language level:

- **Qubit reassignment**: Every gate operation consumes the input qubit and returns a new one. The result MUST be reassigned: `q = qm.h(q)`.
- **No aliasing**: The same qubit must never appear in multiple operand positions of a single operation (e.g., `qm.cx(q, q)` is forbidden).
- **Array element borrowing**: When borrowing a qubit from an array (`q0 = qubits[0]`), it must be returned (`qubits[0] = q0`) before borrowing another element or using the array.

### B. Layer Dependency Direction

```
Frontend (@qkernel)  →  IR (Value/Operation/Block)  →  Transpiler Pipeline  →  Backend
```

Dependencies flow only downstream (left to right). Upstream (reverse) dependencies are forbidden:

- **IR** depends on nothing — it is the shared data model.
- **Frontend** depends only on IR. It MUST NOT import from Transpiler or Backend.
- **Transpiler** depends only on IR. It MUST NOT import from Frontend.
- **Backend** depends on IR and Transpiler public APIs. It MUST NOT import from Frontend.

### C. @qkernel & Converter Pattern

- All quantum circuit building blocks MUST use `@qm.qkernel` (or `@qmc.qkernel`).
- **Type annotations are mandatory** on all parameters and return types.
- Use Qamomile types (`qm.Qubit`, `qm.Float`, `qm.UInt`, `qm.Vector[...]`, `qm.Dict[...]`) for qkernel parameters — not Python primitives (`int`, `float`).
- Use `qm.range()` for loops inside qkernels.
- Use `qm.items()` / `.items()` for Dict iteration inside qkernels.
- **No plain Python `for` loops** over quantum operations — they must be captured as IR ForOperations.
- Plain Python helpers are only acceptable when iterating over precomputed concrete data (e.g., numpy arrays via closures).
- Converters (QAOA, QRAO, FQAOA) compose `@qkernel` building blocks.
- `transpile()` delegates to reusable qkernels — no inline circuit construction.
- If both a public getter (e.g., `get_xxx_ansatz`) and `transpile()` exist, `transpile()` must compose the getter rather than re-implementing the same logic.

### D. Backend Abstraction

- Transpiler subclass pattern: inherit `Transpiler[T]`, implement `_create_separate_pass()`, `_create_emit_pass()`, and `executor()`.
- Emitter pattern: gate-by-gate conversion to backend-specific primitives.
- Composite gate emitters: pluggable strategy pattern for native optimized implementations (e.g., QFT).

### E. Error Design & Exception Safety

- All compilation errors inherit from `QamomileCompileError`.
- Errors provide **rich diagnostic info**: context, suggestions, and available bindings.
- Use `@dataclass` for structured diagnostic data (e.g., `OperandResolutionInfo`).
- Error class docstrings include **correct and incorrect code examples**.
- **Exception chaining**: When re-raising exceptions, MUST use `raise NewError(...) from e` to preserve the original traceback.
- **No bare `except`**: Never use bare `except:` or `except Exception:` without re-raising. Catch specific exception types only.
- **Fallback pattern**: When catching an exception for graceful degradation, use `warnings.warn()` to notify the user and document the fallback behavior.
- **Exception safety**: Verify that stateful operations (e.g., tracer state, value versioning) are properly cleaned up on error paths.

### F. Module Organization

- `__all__` lists only public symbols. No private names (`_` prefix) unless specifically justified.
- `TYPE_CHECKING` guards to break circular import dependencies.
- `__init__.py` provides curated public API via selective re-exports.
- Use `@dataclass` for simple state-oriented types where field-wise equality is appropriate; use a regular class when initialization has substantial logic, invariants or encapsulation are important, or equality needs custom semantics.

### G. Python Style

- **Python 3.12+** features and syntax.
- `X | None` instead of `Optional[X]`.
- Extensive type annotations on all parameters, return types, and generics.
- **All functions and classes MUST have Google-style docstrings** with `Args`, `Returns`, `Raises`, and `Example` sections as appropriate. Type hints should be included in the docstring. Missing docstrings are a P2+ issue.
- No stale imports or dead code.
- Consistent import style (`import qamomile.circuit as qmc`).

### H. Testing Philosophy

- `pytest` with markers (`docs`, gate categories, backends).
- Base test suites for backend validation (`TranspilerTestSuite`).
- **`@pytest.mark.parametrize` must be used wherever possible** — if a test has multiple cases, parametrize them.
- **Random testing (`np.random`) must be used wherever possible** — don't only test fixed values; also test with random inputs to catch edge cases. Fix seeds for reproducibility.
- **Test docstrings**: Google-style Args/Returns are NOT needed. Instead, write a clear 1-2 line description of **what the test verifies**.
- Module-level `@qkernel` definitions are required (for `inspect.getsource()` to work).
- Documentation tests run tutorial `.py` files as Python scripts.
- **Edge case coverage**: Tests MUST include edge cases — empty inputs, single-element arrays, and boundary values (e.g., angles 0, π, 2π).
- **Negative testing**: Tests MUST verify that invalid inputs raise the correct exception types (use `pytest.raises`).
- **Test isolation**: No shared mutable state between tests. Use `@pytest.fixture` for setup.
- **Numerical assertions**: Use `np.allclose()` or `np.testing.assert_allclose()` with explicit `atol`/`rtol` — never `==` for floating-point arrays.

### H-bis. Algorithm & Stdlib Cross-Backend Execution Tests (MANDATORY)

Any addition or non-trivial modification under `qamomile/circuit/algorithm/` or `qamomile/circuit/stdlib/` MUST ship with tests that **transpile the new/changed qkernel to every supported quantum SDK backend and actually execute it** — not just build the IR. This guards against silent emitter regressions and ensures algorithms remain portable across the backend matrix.

**Supported backends (the full matrix)**:

| Backend | Transpiler | Typical executor |
|---|---|---|
| Qiskit | `qamomile.qiskit.QiskitTranspiler` | `BasicSimulator` / `qiskit_aer.AerSimulator` |
| QuriParts | `qamomile.quri_parts.QuriPartsTranspiler` | QuriParts state-vector sampler/estimator |
| CUDA-Q | `qamomile.cudaq.CudaqTranspiler` | `cudaq` state-vector simulator |

> **Note on qBraid**: `qamomile.qbraid` provides only `QBraidExecutor` (no transpiler of its own — it wraps Qiskit circuits) and requires a qBraid API key / `QbraidProvider` to actually execute. It is **out of scope** for this mandatory cross-backend matrix. Coverage for qBraid is optional and, if added, should gate on `QBRAID_API_KEY` env var with a skip.

**Required test shape** (one test per backend, or a parametrized suite over backends):

1. **Transpile**: `transpiler.transpile(my_algo_kernel, bindings=...)` succeeds and returns an `ExecutableProgram` / backend circuit.
2. **Execute both sampling AND expectation-value**: Actually run the program on that backend's simulator in **both** execution modes:
   - **Sampling**: `executable.sample(shots=...)` / equivalent — verifies measurement / counts path.
   - **Expectation value**: `executable.run(observable=...)` / `estimate()` — verifies the expval / estimator path.
   Dry-builds without execution do NOT satisfy this rule. Both modes must be exercised because sampling and expval go through different backend code paths (sampler vs estimator primitives) and regress independently.
3. **Verify**: Compare the backend result against a reference (either an analytic expected value, or a cross-backend equivalence check where backend A's result ≈ backend B's result within `np.allclose` tolerance for expval, or a statistical tolerance for shot-based sampling).
4. **Guard missing deps**: Use `pytest.importorskip("<sdk>")` at the top so the test is skipped (not errored) on environments without that SDK installed.

**Large-problem escape hatch**: If the algorithm intrinsically requires a problem size that cannot be locally simulated (e.g., 30+ qubit circuits, dense Hamiltonians that blow up statevector memory), the cross-backend **execution** requirement is relaxed — in that case it is sufficient to verify that `transpile()` succeeds on every backend and that the emitted circuit has the expected structure (gate count, qubit count, shape). Document the reason in the test docstring (e.g., `"Skipping execution: statevector simulation infeasible for n=30"`). This exception applies **only** when local simulation is genuinely infeasible, not as a shortcut to avoid writing execution tests. Prefer adding a small-scale parametrization (e.g., `n ∈ {2, 3, 4}`) that IS simulatable, and reserve the "transpile-only" check for the large scale.

**Randomized inputs are MANDATORY where applicable**. Do not rely solely on fixed inputs:

- Parametrize over random seeds (`@pytest.mark.parametrize("seed", [0, 1, 2, 42])`) and use `np.random.default_rng(seed)` inside.
- Randomize every input degree of freedom the algorithm/gate exposes: rotation angles, phase parameters, initial bitstrings, control/target wiring, Hamiltonian coefficients, register sizes (across a small set like `n ∈ {1, 2, 3, 5}` — NOT only a single `n`).
- Seeds must be fixed. Never use a bare `np.random.rand()` — always go through a seeded `Generator`.
- Include boundary inputs alongside random ones: angles `0`, `π`, `2π`; empty/single-qubit registers where the algorithm allows.

**What counts as "algorithm or stdlib"**:

- New file under `qamomile/circuit/algorithm/` (e.g., a new variational ansatz, VQE component, new QAOA variant).
- New file under `qamomile/circuit/stdlib/` (e.g., QFT, QPE, IQFT, future additions).
- Any change to an existing algorithm/stdlib qkernel that alters its IR shape, parameter interface, or gate sequence.

**What does NOT count**: pure refactors that provably preserve IR output (still advisable, but not enforced by this rule).

**Reverse direction — new backend SDK added**: Symmetrically, when a new quantum SDK / backend is introduced (a new directory under `qamomile/` with its own `Transpiler` implementation), the existing algorithm and stdlib test suites MUST be extended to cover the new backend. Adding a new `Transpiler` without wiring it into the cross-backend algorithm/stdlib tests leaves the backend silently unvalidated against real quantum programs. Concretely:

- Every existing test under `tests/circuit/algorithm/` and the stdlib tests (`tests/circuit/test_qft.py`, `test_qpe.py`, etc.) that parametrizes over backends MUST include the new backend in its `@pytest.mark.parametrize` list (with `importorskip` guard).
- If the test is not yet parametrized over backends, that is a signal the existing tests need to be refactored first — do the refactor as part of the SDK-addition PR.
- Same randomization and sampling+expval execution requirements apply.

Missing this reverse-direction extension is a **P1** finding in a new-backend PR.

**Severity**: Missing cross-backend execution tests for a new algorithm/stdlib is a **P1** finding. Missing randomized coverage (only fixed inputs tested) is a **P2** finding. Testing only sampling but not expval (or vice versa) when the problem is simulatable is a **P2** finding.

### I. Documentation

- **Jupytext percent format**: `.py` files are the source of truth.
- **ipynb existence**: Every tutorial `.py` file must have a corresponding `.ipynb` file committed to git. A missing `.ipynb` is a P1 issue.
- **ipynb sync on update**: If a `.py` file is modified in the diff, its corresponding `.ipynb` must also be updated. A stale `.ipynb` (unchanged while its `.py` was modified) is a P1 issue.
- **ipynb-only change warning**: If an `.ipynb` file is modified but its corresponding `.py` file is not, this is a P2 warning — it suggests the notebook was edited directly instead of through the source `.py` file. The `.py` file is the source of truth and should be updated first.
- **ipynb must be executed**: Committed `.ipynb` files must contain execution outputs — code cells that produce output must have their output cells populated. An `.ipynb` with empty/missing outputs is a P1 issue.
- **Docs test coverage**: New tutorial files or directories (outside `collaboration/`) must be included in `TUTORIAL_PATTERNS` in `tests/docs/test_tutorials.py`. A new docs path not covered by tests is a P1 issue.
- Jupyter Book 2 with MyST Markdown Engine.
- Bilingual: English (`docs/en/`) and Japanese (`docs/ja/`).
- **en/ja parity**: `docs/en/` and `docs/ja/` must have the same file structure and cover the same content — only the natural language differs. If a tutorial is added or modified in one language, the corresponding file in the other language must also be added or updated. A missing or outdated counterpart is a P1 issue.

### J. Numerical Correctness

- **No `==`/`!=` for floats**: Floating-point comparisons MUST use `math.isclose()` or the project's `is_close_zero()` utility — never `==` or `!=`.
- **Array comparisons in tests**: Use `np.allclose()` or `np.testing.assert_allclose()` — never element-wise `==`.
- **Tolerance propagation**: Be aware that chained floating-point computations accumulate error. Document tolerance assumptions when the computation chain is long.
- **Numerical edge cases**: Docstrings should note behavior for extreme values (very small/large coefficients, near-zero values) when relevant.

### K. Performance & Memory Awareness

- **Unnecessary copies**: Avoid redundant `numpy.copy()` or `.copy()` in hot paths. Prefer views or in-place operations where mutation is safe.
- **Mutable default arguments**: NEVER use mutable defaults (`def f(x=[])`). Use `def f(x: list[int] | None = None): x = x if x is not None else []`.
- **Generators vs lists**: For large iterations, prefer generators or iterators over materialized lists when the result is consumed only once.

### L. Defensive Programming

- **Exhaustive branching**: `if-else` and `match` statements MUST always include an `else` (or default) branch, even when it seems unnecessary. If the else branch has no action, add an explicit comment explaining why (e.g., `# No action needed: ...`) or a defensive `assert` (e.g., `assert False, "unreachable"`).
- **Defensive assertions**: When guarding against cases that should never occur according to Qamomile's specification but are added "just in case", prefer `assert` over `raise`. This clearly communicates the intent: "this is a specification invariant, not an expected error path."
- **`assert` vs `raise`**: Use `assert` for internal invariants that indicate programmer error if violated. Use `raise` (with proper exception types from `QamomileCompileError` hierarchy) for conditions that could be triggered by user input or external data.

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

### Step 3: Explore Related Files

**Do not limit your review to changed files.** Scale your exploration to the size of the diff:

**Small diff (1-3 files):** Check direct imports (files that import from changed files, and files that changed files import from).

**Medium diff (4-10 files):** Additionally, use Grep to find call sites and references for any added, removed, renamed, or signature-changed symbols.

**Large diff (11+ files):** Full exploration:
1. **Import graph**: Identify all files that import from changed files, and all files that changed files import from. Read them.
2. **Symbol usage**: For every function, class, method, or type that was added, removed, renamed, or had its signature changed, use Grep to find **all call sites and references** across the entire codebase. Read those files.
3. **Interface consumers**: If a public API was changed (arguments added/removed, return type changed, behavior altered), trace all consumers of that API and verify they are compatible with the new interface.
4. **Pipeline interactions**: If a transpiler pass, operation type, or IR node was changed, check all other passes and emitters that interact with it.

**Always (any size):** If a new public symbol was added or an existing one renamed/removed, check whether `__init__.py` and `__all__` were updated accordingly.

### Step 4: Evaluate Against Rules

For each changed file, systematically check:

1. **Linear type safety** (Section A) — Are qubits properly consumed and reassigned? Any aliasing? Proper borrow/return for array elements?
2. **Layer dependency direction** (Section B) — Does this file respect the dependency direction? Any upstream imports?
3. **@qkernel & converter correctness** (Section C) — Are circuit building blocks properly decorated? Proper type annotations? Correct loop/iteration patterns? Does `transpile()` delegate to qkernels?
4. **Backend pattern** (Section D) — If a backend file: does it follow the Transpiler/Emitter/Executor pattern?
5. **Error handling & exception safety** (Section E) — Are errors in the `QamomileCompileError` hierarchy? Do they provide diagnostic info? Exception chaining with `from`? No bare `except`? Fallback patterns use `warnings.warn`?
6. **Module organization** (Section F) — Proper `__all__`? TYPE_CHECKING guards?
7. **Python style** (Section G) — Google-style docstrings on ALL functions/classes with type hints? Modern Python syntax? No dead code?
8. **Testing** (Section H) — Are tests parametrized? Random testing used? Clear docstrings describing what is tested? Edge cases and negative tests covered? Numerical assertions use `np.allclose`?
8-bis. **Algorithm/Stdlib cross-backend execution (Section H-bis)** — If the diff adds or materially changes a file under `qamomile/circuit/algorithm/` or `qamomile/circuit/stdlib/`, are there tests that **transpile AND execute** the new kernel on every supported SDK (Qiskit, QuriParts, CUDA-Q) with `importorskip` guards? (qBraid is out of scope — it is an executor-only wrapper around Qiskit and needs an API key.) Are inputs randomized over seeded `np.random.default_rng` across multiple register sizes and angles? Flag as **P1** if cross-backend execution coverage is missing, **P2** if only fixed inputs are used.
9. **Documentation** (Section I) — Every `.py` has a corresponding `.ipynb`? If `.py` was modified, is `.ipynb` also updated? Are `.ipynb` files executed (outputs present)? New docs paths covered in test patterns? Do `docs/en/` and `docs/ja/` have matching file structures?
10. **Numerical correctness** (Section J) — Float comparisons use `isclose`/`is_close_zero`? Tests use `np.allclose`? Tolerance assumptions documented?
11. **Performance & memory** (Section K) — Unnecessary copies? Mutable default arguments? Large iterations use generators?
12. **Defensive programming** (Section L) — Do `if-else`/`match` statements include `else`/default branches? Are defensive checks using `assert` for internal invariants? Are unreachable branches clearly marked?

### Step 5: Impact Analysis and Potential Bug Detection

After the rules review, perform a deeper analysis of the changes' consequences:

#### 5a. Change Impact Prediction

For each significant change, describe:
- **What behavior changes**: How does the change alter inputs, outputs, side effects, or error behavior?
- **Performance implications**: Does the change affect compilation time, circuit size, or execution performance?
- **Existing test impact**: Will existing tests still pass? Are any tests now testing outdated behavior?

#### 5b. Modification Completeness Check

Verify that all necessary changes were made — not just the primary change but all dependent updates:
- **Feature addition completeness**: If a new feature was added, are there sufficient tests covering the new functionality (including edge cases and negative tests)? Is documentation (docstrings, tutorials) provided?
- **Feature modification completeness**: If existing behavior was modified, were related tests updated to reflect the new behavior? Were related documentation and tutorials updated accordingly?
- **Call site updates**: If a function signature changed, were ALL callers updated?
- **`__init__.py` / `__all__` updates**: If a public symbol was added, removed, or renamed, were exports updated?
- **Test coverage**: Were tests added or updated to cover the new/changed behavior? Are there missing test cases? Do tests cover edge cases (empty inputs, boundary values) and negative cases (invalid inputs raise correct exceptions)?
- **Numerical tolerance**: If new floating-point comparisons were added, do they use `isclose`/`allclose` with appropriate tolerance?
- **Documentation updates**: If behavior changed, were docs/tutorials updated to reflect it?
- **Bilateral docs**: If English docs were updated, were Japanese docs also updated (and vice versa)?

#### 5c. Potential Bugs in Related (Unchanged) Files

Look for latent bugs that the change may introduce in files that were **not** modified:
- **Type mismatches**: Does unchanged code use types or interfaces that the change altered?
- **Broken assumptions**: Does unchanged code depend on behavior that the change silently modified? (e.g., a function that previously returned `None` now raises, or a default parameter value changed)
- **Pipeline pass ordering**: If a pass was added or modified, does the overall pipeline order in `Transpiler.transpile()` still make sense?
- **Enum/constant additions**: If a new `OperationKind`, `GateOperationType`, or similar enum value was added, do all `match`/`if-elif` chains handle it?

### Step 5.5: Root-Cause Consolidation via Fix Simulation

After collecting all findings from Steps 4 and 5, perform a mental fix simulation to identify root causes and eliminate surface-level noise. This step does NOT modify any files — it is a reasoning exercise over the finding set.

#### Process

**Iteration 1 — Propose fix directions**

For each finding collected so far, mentally propose the most natural fix direction (do not write code):
- "Wrong base class" → change the parent class
- "Missing docstring on `foo`" → add docstring
- "Unused import of `Bar`" → remove the import

**Iteration 2 — Simultaneous-fix re-evaluation**

Assume ALL proposed fixes are applied simultaneously. Under that assumption, re-evaluate the full finding set:

1. **Moot findings**: Would any finding become irrelevant?
   - Example: Finding A says "`ErrorFoo` has wrong base class" and Finding B says "`ErrorFoo` is never referenced anywhere". Fixing A is pointless because the real issue is B (dead code). A is moot — replace it with B as the root cause.
   - Example: Three style findings in a function, plus a fourth saying the function duplicates existing logic. Root cause is "extract and reuse" — style fixes become moot if the function is removed.

2. **Emergent findings**: Would applying all fixes reveal a new issue not previously identified?
   - Example: Removing dead code reveals that `__init__.py` re-exports a now-deleted symbol.
   - Example: Fixing a type annotation reveals the corrected type is incompatible with a caller.

3. **Fix-dependency chains**: Does fixing A require also fixing B, C, D?
   - If a single root cause spawns 3+ dependent findings, consolidate them under the root cause.

**Iteration 3 (if needed)**

If Iteration 2 produced changes (moot findings removed, new findings added, or findings consolidated), repeat the re-evaluation once more. If still changing after this iteration, stop and note it in the report.

#### Convergence

- **Maximum 3 iterations** (initial proposal + 2 re-evaluations).
- Stop early if the finding set stabilizes (no moot findings, no new findings, no consolidations).
- If still changing after 3 iterations, append to the report: "Note: Some findings may have deeper interdependencies that warrant further investigation."

#### Consolidation Rules

1. **Replace surface findings with root causes.** If finding A is moot because of root-cause B, remove A. If B was not already in the set, add it.
2. **Merge dependent findings under a single root cause.** Report the root cause as the primary finding with "Root cause of:" listing the dependent issues, and a single recommendation addressing the root cause.
3. **Preserve independent findings as-is.**
4. **Severity adjustment.** The consolidated finding inherits the **highest severity** among the root cause and its dependents. If multiple P3 findings stem from a single architectural issue, elevate to at least P2.

#### Specific Patterns to Detect

- **Dead code umbrella**: Multiple style/correctness findings in code that is never called → root cause: dead code (Section G). Severity: at least P2.
- **Missing abstraction**: Multiple findings about repeated logic → root cause: missing shared utility or qkernel. Severity: P1 if it violates Section C.
- **Incomplete rename/refactor**: Mismatched names, broken references, stale imports from an incomplete rename → root cause: the incomplete rename. Severity: P0 if runtime errors result.
- **Wrong layer, cascading violations**: A file imported from the wrong layer causes multiple downstream dependency violations → root cause: the single misplaced import (Section B). Severity: P1.
- **Type change ripple**: A type/signature change not propagated to callers → root cause: incomplete propagation. Severity: P0 if callers would fail at runtime.

### Step 6: Report

Display a structured review report directly in the conversation.

Use the following severity levels:
- **P0 (Bug)**: Incorrect behavior, runtime error, linear type violation, silent data corruption, **breaking changes in related (unchanged) files** caused by the modifications, mutable default arguments, bare `except` swallowing errors, floating-point `==` in production code, missing exception chaining causing lost diagnostics
- **P1 (Significant Design Issue)**: Violates core Qamomile philosophy (layer boundaries, @qkernel pattern, backend abstraction), .py/.ipynb missing or stale, unexecuted .ipynb, new docs not in test patterns, **incomplete modifications** (e.g., changed interface but callers not updated, missing `__all__` update), missing tolerance in numerical test assertions, **missing tests or documentation for new/modified features**
- **P2 (Moderate)**: Missing docstrings, un-parametrized tests, missing random tests, inconsistencies, missing test coverage for new behavior, missing edge case or negative tests, unnecessary copies in hot paths
- **P3 (Minor/Nit)**: Style, naming, import ordering, unfixed random seed, suboptimal generator vs list choice

For each finding, include:
1. The severity level
2. The specific file and line number(s)
3. A code snippet showing the current code
4. **Which rule is violated** (reference the section letter, e.g., "Violates Section C: @qkernel & Converter Pattern")
5. A clear explanation of why this is an issue
6. A specific recommendation with corrected code example
7. For consolidated root-cause findings (from Step 5.5): a **"Root cause of:"** annotation listing the surface-level issues this finding subsumes

End the report with a summary table of all findings grouped by severity.

If the finding set did not converge within 3 iterations in Step 5.5, append: "**Note:** Some findings may have deeper interdependencies that warrant further investigation."
