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
- Use `@dataclass` for data-holding types, not plain classes.

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

### I. Documentation

- **Jupytext percent format**: `.py` files are the source of truth.
- `.ipynb` files are committed to git but must be **exactly identical** to their `.py` counterparts (generated via `jupytext --to notebook`). Any divergence is a P1 issue.
- Jupyter Book 2 with MyST Markdown Engine.
- Bilingual: English (`docs/en/`) and Japanese (`docs/ja/`).
- Tutorials follow progressive complexity.

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
9. **Documentation** (Section I) — `.py` and `.ipynb` in sync? Proper Jupytext format?
10. **Numerical correctness** (Section J) — Float comparisons use `isclose`/`is_close_zero`? Tests use `np.allclose`? Tolerance assumptions documented?
11. **Performance & memory** (Section K) — Unnecessary copies? Mutable default arguments? Large iterations use generators?
<<<<<<< HEAD
=======
12. **Defensive programming** (Section L) — Do `if-else`/`match` statements include `else`/default branches? Are defensive checks using `assert` for internal invariants? Are unreachable branches clearly marked?
>>>>>>> new_qamomile

### Step 5: Impact Analysis and Potential Bug Detection

After the rules review, perform a deeper analysis of the changes' consequences:

#### 5a. Change Impact Prediction

For each significant change, describe:
- **What behavior changes**: How does the change alter inputs, outputs, side effects, or error behavior?
- **Performance implications**: Does the change affect compilation time, circuit size, or execution performance?
- **Existing test impact**: Will existing tests still pass? Are any tests now testing outdated behavior?

#### 5b. Modification Completeness Check

Verify that all necessary changes were made — not just the primary change but all dependent updates:
<<<<<<< HEAD
=======
- **Feature addition completeness**: If a new feature was added, are there sufficient tests covering the new functionality (including edge cases and negative tests)? Is documentation (docstrings, tutorials) provided?
- **Feature modification completeness**: If existing behavior was modified, were related tests updated to reflect the new behavior? Were related documentation and tutorials updated accordingly?
>>>>>>> new_qamomile
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

### Step 6: Report

Display a structured review report directly in the conversation.

Use the following severity levels:
- **P0 (Bug)**: Incorrect behavior, runtime error, linear type violation, silent data corruption, **breaking changes in related (unchanged) files** caused by the modifications, mutable default arguments, bare `except` swallowing errors, floating-point `==` in production code, missing exception chaining causing lost diagnostics
<<<<<<< HEAD
- **P1 (Significant Design Issue)**: Violates core Qamomile philosophy (layer boundaries, @qkernel pattern, backend abstraction), .py/.ipynb desync, **incomplete modifications** (e.g., changed interface but callers not updated, missing `__all__` update), missing tolerance in numerical test assertions
=======
- **P1 (Significant Design Issue)**: Violates core Qamomile philosophy (layer boundaries, @qkernel pattern, backend abstraction), .py/.ipynb desync, **incomplete modifications** (e.g., changed interface but callers not updated, missing `__all__` update), missing tolerance in numerical test assertions, **missing tests or documentation for new/modified features**
>>>>>>> new_qamomile
- **P2 (Moderate)**: Missing docstrings, un-parametrized tests, missing random tests, inconsistencies, missing test coverage for new behavior, missing edge case or negative tests, unnecessary copies in hot paths
- **P3 (Minor/Nit)**: Style, naming, import ordering, unfixed random seed, suboptimal generator vs list choice

For each finding, include:
1. The severity level
2. The specific file and line number(s)
3. A code snippet showing the current code
4. **Which rule is violated** (reference the section letter, e.g., "Violates Section C: @qkernel & Converter Pattern")
5. A clear explanation of why this is an issue
6. A specific recommendation with corrected code example

<<<<<<< HEAD
End the report with a summary table of all findings grouped by severity.
=======
End the report with a summary table of all findings grouped by severity.
>>>>>>> new_qamomile
