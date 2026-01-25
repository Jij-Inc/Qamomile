# Resource Estimation Design

Based on the survey paper "Quantum algorithms: A survey of applications and end-to-end complexities" (arXiv:2310.03011v2).

## Overview

This module provides algebraic resource estimation for quantum circuits built with Qamomile.
All estimates are expressed using SymPy symbolic expressions, allowing dependency on problem size parameters.

## Core Metrics

### 1. Qubit Count (Existing: `qubits_counter.py`)
- Number of logical qubits required
- Already implemented with SymPy support
- Handles parametric dimensions (e.g., `n`, `n+3`)

### 2. Gate Count (`gate_counter.py`)
- **Total Gates**: Total number of quantum gates
- **Single-qubit Gates**: Count of 1-qubit gates
- **Two-qubit Gates**: Count of 2-qubit gates (CNOT, CZ, etc.)
- **T Gates**: Count of T gates (dominant cost in fault-tolerant implementations)
- **Clifford Gates**: H, S, CNOT, CZ gates

Returns: `GateCount(total, single_qubit, two_qubit, t_gates, clifford_gates)`

### 3. Circuit Depth (`depth_estimator.py`)
- **Total Depth**: Maximum depth of the circuit
- **T-Depth**: Depth considering only T gates (critical for fault-tolerant execution)
- **Two-qubit Depth**: Depth of two-qubit gates (bottleneck in many architectures)

Returns: `CircuitDepth(total_depth, t_depth, two_qubit_depth)`

### 4. Resource Summary (`resource_estimator.py`)
Unified interface combining all metrics:

```python
@dataclass
class ResourceEstimate:
    qubits: sp.Expr  # Logical qubit count
    gates: GateCount  # Gate counts by type
    depth: CircuitDepth  # Circuit depths
    parameters: dict[str, sp.Symbol]  # Problem size parameters
```

## Algorithm-Specific Estimators

### Directory: `estimator/algorithmic/`

These provide **theoretical** resource estimates based on asymptotic complexity formulas from literature.

#### 1. QAOA (`algorithmic/qaoa.py`)

For $p$-layer QAOA on $n$ qubits with Ising Hamiltonian:

```python
def estimate_qaoa(
    n: sp.Expr,  # Number of qubits
    p: sp.Expr,  # Number of QAOA layers
    num_edges: sp.Expr,  # Number of edges in interaction graph
) -> ResourceEstimate:
    """
    Returns:
        qubits: n
        single_qubit_gates: 2*p*n (Rx gates for mixer)
        two_qubit_gates: p * num_edges (RZZ gates for cost)
        depth: O(p * (num_edges + n))
    """
```

#### 2. Quantum Phase Estimation (`algorithmic/qpe.py`)

Based on qubitization/QSVT methods:

```python
def estimate_qpe(
    n_system: sp.Expr,  # System qubits
    precision: sp.Expr,  # Bits of precision (ε = 2^(-precision))
    hamiltonian_norm: sp.Expr,  # ||H|| normalization
    evolution_time: sp.Expr | None = None,
) -> ResourceEstimate:
    """
    Returns (qubitization-based):
        qubits: n_system + precision + O(log n_system)
        gates: O(hamiltonian_norm * 2^precision) block-encoding calls

    Based on Section 13 of arXiv:2310.03011
    """
```

#### 3. Hamiltonian Simulation (`algorithmic/hamiltonian_simulation.py`)

Multiple methods with different tradeoffs:

```python
def estimate_trotter(
    n: sp.Expr,  # Number of qubits
    L: sp.Expr,  # Number of Hamiltonian terms
    time: sp.Expr,  # Evolution time
    error: sp.Expr,  # Target error
    order: int = 2,  # Trotter order (2, 4, ...)
) -> ResourceEstimate:
    """
    Product formula approach (Section 11.1).

    Returns:
        gates: O(L * t^(1+1/order) / ε^(1/order))
    """

def estimate_qsvt(
    n: sp.Expr,
    hamiltonian_norm: sp.Expr,  # α in block-encoding
    time: sp.Expr,
    error: sp.Expr,
) -> ResourceEstimate:
    """
    QSVT-based simulation (Section 11.4).

    Returns:
        gates: O(α*t + log(1/ε)/log(log(1/ε)))
    """

def estimate_qdrift(
    L: sp.Expr,  # Number of terms
    hamiltonian_1norm: sp.Expr,  # ||H||_1
    time: sp.Expr,
    error: sp.Expr,
) -> ResourceEstimate:
    """
    qDRIFT method (Section 11.2).

    Returns:
        samples: O(||H||_1^2 * t^2 / ε)
    """
```

## Usage Examples

### Example 1: Analyze a QKernel

```python
import qamomile.circuit as qm
from qamomile.circuit.estimator import estimate_resources

@qm.qkernel
def my_circuit(n: qm.UInt, theta: qm.Float) -> qm.Vector[qm.Qubit]:
    q = qm.qubit_array(n)
    for i in qm.range(n):
        q[i] = qm.h(q[i])
        q[i] = qm.rz(q[i], theta)
    return q

# Estimate with symbolic n
estimate = estimate_resources(my_circuit.block)
print(estimate.qubits)  # n
print(estimate.gates.total)  # 2*n
print(estimate.depth.total_depth)  # n

# Substitute concrete value
concrete = estimate.substitute(n=100)
print(concrete.qubits)  # 100
```

### Example 2: Theoretical QAOA Estimate

```python
from qamomile.circuit.estimator.algorithmic import estimate_qaoa
import sympy as sp

n, p, E = sp.symbols('n p E', positive=True, integer=True)
est = estimate_qaoa(n, p, num_edges=E)

print(est.qubits)  # n
print(est.gates.two_qubit)  # p*E
print(est.depth.total_depth)  # O(p*(E + n))

# MaxCut on complete graph K_10 with p=3
concrete = est.substitute(n=10, p=3, E=45)
print(concrete.gates.two_qubit)  # 135
```

### Example 3: Compare Hamiltonian Simulation Methods

```python
from qamomile.circuit.estimator.algorithmic import (
    estimate_trotter,
    estimate_qsvt,
    estimate_qdrift
)
import sympy as sp

n, L, t = sp.symbols('n L t', positive=True)
eps = sp.Symbol('eps', positive=True)

# Second-order Trotter
trotter2 = estimate_trotter(n, L, t, eps, order=2)
print(trotter2.gates.total)  # O(L * (t^1.5 / eps^0.5))

# QSVT
qsvt = estimate_qsvt(n, hamiltonian_norm=L, time=t, error=eps)
print(qsvt.gates.total)  # O(L*t + log(1/eps))

# qDRIFT
qdrift = estimate_qdrift(L, hamiltonian_1norm=L, time=t, error=eps)
print(qdrift.gates.total)  # O(L^2 * t^2 / eps)
```

## Implementation Architecture

```
qamomile/circuit/estimator/
├── __init__.py                    # Public API
├── DESIGN.md                      # This file
├── qubits_counter.py             # Existing: qubit count
├── gate_counter.py               # New: gate counting
├── depth_estimator.py            # New: depth estimation
├── resource_estimator.py         # New: unified interface
└── algorithmic/                   # Theoretical estimates
    ├── __init__.py
    ├── qaoa.py
    ├── qpe.py
    └── hamiltonian_simulation.py
```

## References

- arXiv:2310.03011v2 - "Quantum algorithms: A survey of applications and end-to-end complexities"
- Section 10-14: Quantum algorithmic primitives
- Section 25-27: Fault-tolerant quantum computation
- Application sections: Resource estimate tables

## Future Extensions

1. **Backend-specific estimates**: Account for native gate sets (e.g., Qiskit uses U3+CX)
2. **Error budget**: Distribute error across subroutines
3. **Physical resource estimates**: Map to physical qubits using error correction codes
4. **Optimization**: Suggest parameter choices to minimize resources
5. **Classical resources**: Track classical pre/post-processing costs
