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

### Core Pipeline Flow

```
@qkernel Python function
         ↓
    Frontend (AST transform → tracing → Block with operations)
         ↓
    Transpiler Pipeline (to_block → inline → partial_eval → analyze → plan → emit)
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

1. **to_block()**: `QKernel` → `Block`
2. **inline()**: Inlines `CallBlockOperation` → affine `Block`
3. **partial_eval()**: Constant folds and lowers compile-time control flow
4. **analyze()**: Validates dependencies → analyzed `Block`
5. **plan()**: Builds a `ProgramPlan`
6. **emit()**: Backend-specific circuit generation → `ExecutableProgram`

## Test

`tests/` contains unit tests, and docs tests under `tests/docs` must also be run for documentation-impacting changes.

## Documentation Translation

To translate English docs (`docs/en/`) into Japanese (`docs/ja/`), use the `/translate` skill:

```
/translate docs/en/tutorial/qaoa.py
```

Translation rules (tone, spacing, terminology, soft line breaks, etc.) are defined in `.claude/skills/translate/SKILL.md`. Always use this skill when translating documentation.
