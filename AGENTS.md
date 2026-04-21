
# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

```bash
# Install dependencies (using uv)
uv sync

# Run all tests
uv run pytest tests/

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

# Run docs tests
uv run pytest -m docs tests/docs -v

# Build documentation
cd docs
make build        # Build both language sites
# or: make build-en / make build-ja
```

## Architecture Overview

Qamomile is a quantum optimization SDK that converts mathematical models (JijModeling) into executable quantum circuits across multiple backends (Qiskit, QuriParts, CUDA-Q, etc.).

### Core Pipeline Flow

```
Mathematical Model (JijModeling)
         ↓
    Core Layer (converters: QAOA, QRAO)
         ↓
    Circuit Layer (Frontend @qkernel → IR Block)
         ↓
    Transpiler Pipeline (inline → partial_eval → analyze → plan → emit)
         ↓
    Backend Execution (Qiskit, QuriParts, etc.)
```

### Key Module Structure

**qamomile/circuit/** - Circuit abstraction layer:
- `frontend/`: Python decorator-based API (`@qm.qkernel`) with handle types (Qubit, Float, UInt, Bit)
- `ir/`: Intermediate representation with `Block`, `Value`, and `Operation`
- `transpiler/`: Multi-pass compilation (inline, partial-eval, analyze, plan, emit)
- `stdlib/`: Built-in algorithms (QFT, IQFT, QPE)

**qamomile/core/** - Mathematical modeling layer:
- `converters/`: QAOA, QRAO, FQAOA converters from JijModeling problems
- `operator.py`: Pauli operators and Hamiltonian representation
- `ising_qubo.py`: Ising/QUBO model conversion utilities

**qamomile/{qiskit,quri_parts,cudaq,pennylane,qutip,udm}/** - Backend transpilers implementing emit passes and executors

### Key Design Patterns

**Value Versioning (SSA-like)**: Each operation creates new Value instances with incremented versions for dependency tracking.

**Operation Classification**: Operations are classified as QUANTUM, CLASSICAL, HYBRID, or CONTROL to enable intelligent segment separation.

**Tracing Pattern**: Quantum operations are traced via context-local Tracer that collects emitted operations during kernel build.

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

1. **to_block()**: `QKernel` → `Block`
2. **inline()**: Inlines `CallBlockOperation` → affine `Block`
3. **partial_eval()**: Constant folds and lowers compile-time control flow
4. **analyze()**: Validates dependencies → analyzed `Block`
5. **plan()**: Builds a `ProgramPlan`
6. **emit()**: Backend-specific circuit generation → `ExecutableProgram`

## Docstring Convention (MANDATORY)

All **public** functions, methods, and classes in `qamomile/` MUST carry a **Google-style docstring** with the appropriate sections filled in — not just a one-line summary. This is enforced by `/local-review` (missing docstrings are P2+).

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
    *,
    bindings: dict[str, Any] | None = None,
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
            Defaults to None, meaning no bindings — the kernel must then have
            no free parameters.

    Returns:
        An `ExecutableProgram[T]` wrapping the backend circuit and the
        parameter metadata needed to re-bind runtime parameters.

    Raises:
        QamomileCompileError: If analyze/plan detects a dependency or shape
            violation in the IR.
        KeyError: If `bindings` contains a name that is not a kernel
            parameter.

    Example:
        >>> transpiler = QiskitTranspiler()
        >>> exe = transpiler.transpile(my_kernel, bindings={"theta": 0.5})
        >>> counts = exe.sample(shots=1024)
    """
```

Additional rules:

- **Private helpers** (`_foo`, `__bar`) should have a docstring when their purpose is not obvious from the name; a one-line summary is acceptable.
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
