---
name: cross-backend-testing
description: Qamomile's mandatory cross-backend execution test policy for qamomile/circuit/algorithm/ and qamomile/circuit/stdlib/. Use whenever adding or modifying an algorithm/stdlib qkernel, or when adding a new SDK backend, so the change ships tests that actually transpile AND execute on every supported backend (Qiskit, QuriParts, CUDA-Q). Missing coverage is a P1 in /local-review.
---

# Cross-Backend Execution Tests for Algorithms and Stdlib (MANDATORY)

`tests/` contains unit tests, and docs tests under `tests/docs` must also be run for documentation-impacting changes.

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
