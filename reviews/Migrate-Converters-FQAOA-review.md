# Code Review: Migrate/Converters-FQAOA

**Branch:** `Migrate/Converters-FQAOA` vs `new_qamomile`
**Reviewer:** Claude Code
**Date:** 2026-02-13

## Scope

| Status | File | Description |
|--------|------|-------------|
| A | `qamomile/circuit/algorithm/fqaoa.py` | FQAOA circuit building blocks |
| M | `qamomile/circuit/algorithm/__init__.py` | Algorithm exports |
| A | `qamomile/optimization/fqaoa.py` | FQAOA converter |
| D→A | `optimization/converter.py` → `optimization/converters/converter.py` | Base class moved |
| A | `qamomile/optimization/converters/__init__.py` | Empty package init |
| M | `qamomile/optimization/qaoa.py` | QAOA converter updated |
| M | `qamomile/optimization/qrao/qrao31.py` | Import path update |
| D | `tests/core/converters/test_fqaoa.py` | Old tests removed |
| A | `tests/optimization/test_fqaoa.py` | New tests |
| M | `.github/workflows/python-test.yml` | CI addition |

---

## Findings

### F1 — `__all__` contains nonexistent symbols

**Severity:** P0 (Bug)
**File:** `qamomile/circuit/algorithm/__init__.py`, lines 38–39

```python
__all__ = [
    ...
    "hardware_efficient_ansatz",   # ← does not exist
    "num_parameters",              # ← does not exist
    ...
]
```

**Reference:** In `qaoa.py`, every symbol in `__all__` is imported at the top of the file.

**Impact:** `from qamomile.circuit.algorithm import hardware_efficient_ansatz` will raise `ImportError`.

**Recommendation:** Remove both entries from `__all__`, or (if they are planned) add the corresponding imports from `basic.py` once the functions exist.

---

### F2 — `transpile()` duplicates `get_fqaoa_ansatz()` logic instead of composing it

**Severity:** P1 (Significant Design Issue)
**File:** `qamomile/optimization/fqaoa.py`, lines 315–378

```python
# transpile() re-implements init + givens + layers + measurement inline:
def transpile(self, transpiler, *, p, hopping=1.0):
    ...
    givens_rotation_data = self._givens_decomposition(unitary_rows)
    ...
    @qm_c.qkernel
    def fqaoa_sampling(betas, gammas, linear, quad):
        q = qm_c.qubit_array(num_qubits, name="q")
        for i in qm_c.range(num_fermions):
            q[i] = qm_c.x(q[i])
        q = givens_rotations(q, givens_rotation_data)
        for layer in qm_c.range(p):
            # inline cost + mixer ...
```

**Reference (`optimization/qaoa.py`):**
QAOA `transpile()` delegates to the reusable `qaoa_state` kernel from `algorithm/qaoa.py`:

```python
def transpile(self, transpiler, *, p):
    @qmc.qkernel
    def qaoa_sampling(p, quad, linear, gammas, betas, n):
        q = qaoa_state(p, quad, linear, n, gammas, betas)
        return qmc.measure(q)
```

**Impact:** The same circuit-building logic exists in both `get_fqaoa_ansatz()` and `transpile()`. If a bug is fixed in one, the other may be missed. The cost layer in `transpile()` uses `2 * hi * gammas[layer]` while `cost_layer()` in `algorithm/fqaoa.py` also uses `2 * hi * gamma` — the duplication creates divergence risk.

**Recommendation:** Create a reusable `fqaoa_state` kernel in `algorithm/fqaoa.py` (like `qaoa_state`) and have `transpile()` call it, following the QAOA pattern.

---

### F3 — FQAOA `cost_layer` duplicates QAOA `ising_cost_circuit`

**Severity:** P2 (Moderate)
**File:** `qamomile/circuit/algorithm/fqaoa.py`, lines 89–106

```python
def cost_layer(q, gamma, linear, quad):
    for i, hi in linear.items():
        q[i] = qm_c.rz(q[i], 2 * hi * gamma)
    for (i, j), Jij in quad.items():
        q[i], q[j] = qm_c.rzz(q[i], q[j], 2 * Jij * gamma)
    return q
```

**Reference (`algorithm/qaoa.py`):**

```python
@qmc.qkernel
def ising_cost_circuit(quad, linear, q, gamma):
    for (i, j), Jij in quad.items():
        q[i], q[j] = qmc.rzz(q[i], q[j], angle=Jij * gamma)
    for i, hi in linear.items():
        q[i] = qmc.rz(q[i], angle=hi * gamma)
    return q
```

**Impact:** Nearly identical logic. FQAOA embeds a factor of 2 in the gate angles (`2 * hi * gamma`), which could instead be normalized at the converter level. This is an opportunity for code reuse.

**Recommendation:** Either reuse `ising_cost_circuit` from `algorithm/qaoa.py` (pre-multiplying the spin model coefficients by 2 at the converter level), or document why the 2x factor necessitates a separate function.

---

### F4 — `_givens_decomposition` missing type annotation on parameter

**Severity:** P2 (Moderate)
**File:** `qamomile/optimization/fqaoa.py`, line 179

```python
def _givens_decomposition(self, fermi_orbital):
```

**Reference:** Other methods in the same class annotate their parameters:

```python
def get_fermi_orbital(self) -> np.ndarray:
def cyclic_mapping(self) -> dict[tuple[int, int], int]:
```

**Recommendation:** Add annotation: `def _givens_decomposition(self, fermi_orbital: np.ndarray) -> list[list]:` (or a more precise type for the return value).

---

### F5 — `converters/__init__.py` is empty

**Severity:** P2 (Moderate)
**File:** `qamomile/optimization/converters/__init__.py`

The file exists but is empty. This means `from qamomile.optimization.converters import MathematicalProblemConverter` fails — users must use the full deep path.

**Reference:** Other `__init__.py` files in the project re-export key symbols for convenient access.

**Recommendation:** Add:
```python
from .converter import MathematicalProblemConverter
```

---

### F6 — Unused imports in test file

**Severity:** P3 (Minor/Nit)
**File:** `tests/optimization/test_fqaoa.py`, lines 2–4

```python
import numpy as np      # unused
import ommx.v1          # unused
```

**Recommendation:** Remove unused imports. `ruff check` will flag these.

---

### F7 — Typo in test function name

**Severity:** P3 (Minor/Nit)
**File:** `tests/optimization/test_fqaoa.py`, line 41

```python
def test_initializaiton(simple_problem):
#             ^^^^ "aiton" should be "ation"
```

**Recommendation:** Rename to `test_initialization`.

---

### F8 — Import alias inconsistency (`qm_c` vs `qmc`)

**Severity:** P3 (Minor/Nit)
**Files:**
- `optimization/fqaoa.py` and `algorithm/fqaoa.py` use `import qamomile.circuit as qm_c`
- `optimization/qaoa.py` and `algorithm/qaoa.py` use `import qamomile.circuit as qmc`

**Recommendation:** Standardize on one alias across the codebase. The existing QAOA files use `qmc`; the new FQAOA files use `qm_c`. Either is fine, but consistency within the module set is preferred.

---

### F9 — Docstring mentions `jijmodeling_transpiler` (stale reference)

**Severity:** P3 (Minor/Nit)
**File:** `qamomile/optimization/fqaoa.py`, line 28

```python
Note:
        This module requires `jijmodeling` and `jijmodeling_transpiler` for problem representation.
```

**Impact:** The code no longer imports or uses `jijmodeling_transpiler`. This is a leftover from the core migration.

**Recommendation:** Update the docstring to only mention `jijmodeling` and `ommx`.

---

### F10 — Docstring example uses wrong kwarg name `num_fermion`

**Severity:** P3 (Minor/Nit)
**File:** `qamomile/optimization/fqaoa.py`, line 70

```python
fqaoa_converter = FQAOAConverter(compiled_instance, num_fermion=4)
#                                                   ^^^^^^^^^^
# Should be: num_fermions=4
```

**Recommendation:** Fix to `num_fermions=4`.

---

### F11 — `boundary hop` workaround lacks explanation

**Severity:** P3 (Minor/Nit)
**File:** `qamomile/optimization/fqaoa.py`, lines 368–370

```python
# Boundary hop (wrap in control flow to avoid segment splits)
for _ in qm_c.range(1):
    q = hopping_gate(q, 0, last_qubit, betas[layer], hopping)
```

This `for _ in qm_c.range(1)` loop runs exactly once. The comment says "avoid segment splits" but doesn't explain the mechanism or why a regular call would cause splits.

**Recommendation:** Add a brief explanation of the IR segmentation issue this works around, or link to an issue/doc.

---

## Summary Table

| ID | Severity | File | Issue |
|----|----------|------|-------|
| F1 | **P0** | `algorithm/__init__.py:38-39` | `__all__` references `hardware_efficient_ansatz` and `num_parameters` which don't exist |
| F2 | **P1** | `optimization/fqaoa.py:315-378` | `transpile()` duplicates `get_fqaoa_ansatz()` logic instead of composing a reusable kernel |
| F3 | **P2** | `algorithm/fqaoa.py:89-106` | `cost_layer` nearly duplicates `ising_cost_circuit` from QAOA |
| F4 | **P2** | `optimization/fqaoa.py:179` | `_givens_decomposition` missing parameter/return type annotations |
| F5 | **P2** | `converters/__init__.py` | Empty `__init__.py` — no re-export of `MathematicalProblemConverter` |
| F6 | **P3** | `test_fqaoa.py:2-4` | Unused imports: `numpy`, `ommx.v1` |
| F7 | **P3** | `test_fqaoa.py:41` | Typo: `test_initializaiton` → `test_initialization` |
| F8 | **P3** | FQAOA files | Import alias `qm_c` vs `qmc` inconsistency with QAOA files |
| F9 | **P3** | `optimization/fqaoa.py:28` | Stale docstring reference to `jijmodeling_transpiler` |
| F10 | **P3** | `optimization/fqaoa.py:70` | Docstring example uses `num_fermion` instead of `num_fermions` |
| F11 | **P3** | `optimization/fqaoa.py:368-370` | `for _ in qm_c.range(1)` workaround lacks explanation |

**Overall:** 1 P0, 1 P1, 3 P2, 6 P3
