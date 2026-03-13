
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

# Lint with ruff
uv run ruff check qamomile/

# Format with ruff
uv run ruff format qamomile/

# Type checking with zuban
uv run zuban qamomile/

# Build documentation (legacy - from docs/en or docs/ja directory)
jupyter-book build .

# Build docs2 documentation (recommended for new work)
cd docs2
make build        # Build both English and Japanese
make build-en     # Build English only
make build-ja     # Build Japanese only
make serve-en     # Serve English docs locally
make serve-ja     # Serve Japanese docs locally
```

## Architecture Overview

Qamomile is a quantum optimization SDK that converts mathematical models (JijModeling) into executable quantum circuits across multiple backends (Qiskit, QuriParts, CUDA-Q, etc.).

### Core Pipeline Flow

```
Mathematical Model (JijModeling)
         ↓
    Core Layer (converters: QAOA, QRAO)
         ↓
    Circuit Layer (Frontend @qkernel → IR Graph)
         ↓
    Transpiler Pipeline (inline → constant_fold → analyze → separate → emit)
         ↓
    Backend Execution (Qiskit, QuriParts, etc.)
```

### Key Module Structure

**qamomile/circuit/** - Circuit abstraction layer:
- `frontend/`: Python decorator-based API (`@qm.qkernel`) with handle types (Qubit, Float, UInt, Bit)
- `ir/`: Intermediate representation with Value nodes, Operations, and Graph/Block structures
- `transpiler/`: Multi-pass compilation (inline, constant fold, analyze, separate, emit)
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

1. **to_block()**: QKernel → Block (HIERARCHICAL)
2. **inline()**: Inlines CallBlockOperation → Block (LINEAR)
3. **constant_fold()**: Evaluates constant expressions
4. **analyze()**: Validates dependencies → Block (ANALYZED)
5. **separate()**: Splits into quantum/classical segments → SeparatedProgram
6. **emit()**: Backend-specific circuit generation → ExecutableProgram

## Test

tests/ directory contains unit tests.
and you have to run docs tests too.
