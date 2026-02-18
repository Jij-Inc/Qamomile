# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: qamomile
#     language: python
#     name: qamomile
# ---

# %% [markdown]
# # Composite Gates
#
# In the [previous tutorial](05_stdlib.ipynb) we saw that QFT and IQFT
# appear as single labelled boxes in circuit diagrams. These are
# **composite gates** — multi-qubit operations bundled into a single
# named unit. Qamomile lets us define our own composite gates using the
# `CompositeGate` base class or the `@composite_gate` decorator.
#
# ## What We Will Learn
# - Writing custom composite gates by subclassing `CompositeGate`
# - Attaching resource metadata for analysis
# - Using the `@composite_gate` decorator for simpler cases
# - Creating stub gates for resource estimation without gate-level implementation
#
# ## Why CompositeGate?
#
# - **Encapsulation**: Bundle multiple gates into a single named operation
# - **Reusability**: Use the gate in multiple kernels
# - **Backend optimization**: Backends can provide native implementations
# - **Resource estimation**: Attach resource metadata for analysis
# - **Decomposition strategies**: Support multiple implementations

# %%
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 1. Subclassing `CompositeGate`
#
# To create a custom gate, subclass `CompositeGate` and implement:
#
# 1. `num_target_qubits` (property): How many qubits the gate acts on
# 2. `_decompose(qubits)`: The gate logic using frontend operations
# 3. `_resources()` (optional): Resource metadata

# %%
from qamomile.circuit import CompositeGate
from qamomile.circuit.ir.operation.composite_gate import ResourceMetadata


class MyTwoQubitGate(CompositeGate):
    """A custom 2-qubit gate: H on first qubit, then CNOT."""

    custom_name = "my_gate"

    def __init__(self):
        pass

    @property
    def num_target_qubits(self) -> int:
        return 2

    def _decompose(self, qubits):
        q0, q1 = qubits
        q0 = qmc.h(q0)
        q0, q1 = qmc.cx(q0, q1)
        return q0, q1

    def _resources(self):
        return ResourceMetadata(
            t_gate_count=0,
            custom_metadata={
                "num_h_gates": 1,
                "num_cx_gates": 1,
                "total_gates": 2,
            },
        )


# %% [markdown]
# ## 2. Using the Custom Gate in a QKernel
#
# Instantiate the gate and call it like a function inside a `@qkernel`.
# It accepts individual qubits as arguments and returns a tuple of qubits.


# %%
@qmc.qkernel
def use_custom_gate() -> tuple[qmc.Bit, qmc.Bit]:
    """Use MyTwoQubitGate inside a circuit."""
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")

    gate = MyTwoQubitGate()
    q0, q1 = gate(q0, q1)

    return qmc.measure(q0), qmc.measure(q1)


use_custom_gate.draw()

# %% [markdown]
# With `expand_composite=True`, we can see the gates inside the box:

# %%
use_custom_gate.draw(expand_composite=True)

# %% [markdown]
# ### Checking Resources

# %%
gate = MyTwoQubitGate()
resources = gate.get_resource_metadata()

print("=== MyTwoQubitGate Resources ===")
print(f"  Custom metadata: {resources.custom_metadata}")

# %% [markdown]
# ## 3. The `@composite_gate` Decorator
#
# For simpler cases, Qamomile provides a `@composite_gate` decorator
# that wraps a `@qkernel` function as a `CompositeGate`. This avoids
# the need to write a full class.


# %%
@qmc.composite_gate
@qmc.qkernel
def bell_gate(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Create a Bell state: H on q0, then CNOT."""
    q0 = qmc.h(q0)
    q0, q1 = qmc.cx(q0, q1)
    return q0, q1


# %%
@qmc.qkernel
def use_bell_gate() -> tuple[qmc.Bit, qmc.Bit]:
    """Use the decorator-based bell_gate."""
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")

    q0, q1 = bell_gate(q0, q1)

    return qmc.measure(q0), qmc.measure(q1)


use_bell_gate.draw()

# %% [markdown]
# ### Choosing Between Class and Decorator
#
# | Feature | `CompositeGate` subclass | `@composite_gate` decorator |
# |---------|--------------------------|----------------------------|
# | Resource metadata | Full control via `_resources()` | Not supported |
# | Decomposition strategies | Via `_strategies` registry | Not supported |
# | Parameterized construction | Via `__init__` arguments | Via closure/constants |
# | Simplicity | More boilerplate | Minimal code |
# | Best for | Library gates, configurable gates | Quick one-off gates |

# %% [markdown]
# ## 4. Stub Gates for Resource Estimation
#
# Sometimes we need to estimate resources for a component whose gate-level
# implementation is not yet available — for example, an oracle in Grover's
# algorithm. `@composite_gate(stub=True, ...)` creates a placeholder gate
# with resource annotations but **no decomposition**.


# %%
@qmc.composite_gate(
    stub=True,
    name="oracle",
    num_qubits=5,
    query_complexity=1,
    t_gate_count=100,
)
def oracle():
    pass


# %%
@qmc.qkernel
def grover_iteration() -> qmc.Vector[qmc.Bit]:
    """A single Grover iteration using a stub oracle."""
    q = qmc.qubit_array(5, name="q")

    # Superposition
    for i in qmc.range(5):
        q[i] = qmc.h(q[i])

    # Oracle (stub — no gate-level implementation)
    q[0], q[1], q[2], q[3], q[4] = oracle(q[0], q[1], q[2], q[3], q[4])

    return qmc.measure(q)


grover_iteration.draw()

# %%
stub_resources = oracle.get_resource_metadata()

print("=== Stub Oracle Resources ===")
print(f"  Query complexity: {stub_resources.query_complexity}")
print(f"  T-gate count:     {stub_resources.t_gate_count}")

# %% [markdown]
# The stub gate appears as a labelled box in the circuit diagram and
# carries resource metadata for estimation, but has no gate-level
# decomposition. This is useful for top-down circuit design where we
# define the algorithm structure first and fill in implementations later.

# %% [markdown]
# ## 5. Summary
#
# ### Key Classes
#
# | Class | Module | Purpose |
# |-------|--------|---------|
# | `CompositeGate` | `qamomile.circuit` | Base class for custom gates |
# | `ResourceMetadata` | `qamomile.circuit.ir.operation.composite_gate` | Resource estimation data |
#
# ### Custom CompositeGate Pattern
#
# ```python
# class MyGate(CompositeGate):
#     custom_name = "my_gate"
#
#     def __init__(self, ...):
#         ...
#
#     @property
#     def num_target_qubits(self) -> int:
#         return N
#
#     def _decompose(self, qubits):
#         q0, q1 = qubits
#         # ... gate operations ...
#         return q0, q1
#
#     def _resources(self):
#         return ResourceMetadata(t_gate_count=0)
# ```
#
# ### Next Tutorials
#
# - [Our First Quantum Algorithm](07_first_algorithm.ipynb): The Deutsch-Jozsa algorithm
# - [Resource Estimation](09_resource_estimation.ipynb): Estimate gate counts and circuit depth
# - [QAOA](../optimization/qaoa.ipynb): Solve combinatorial optimization problems with QAOA

# %% [markdown]
# ## What We Learned
#
# - **Writing custom composite gates with `CompositeGate`** — Subclass `CompositeGate` to define reusable multi-qubit operations with pluggable decomposition strategies.
# - **Attaching resource metadata** — Override `_resources()` to return `ResourceMetadata` for gate-count analysis.
# - **The `@composite_gate` decorator** — A lightweight alternative to the full class approach for quick one-off composite gates.
# - **Stub gates for resource estimation** — Use `@composite_gate(stub=True, ...)` to create placeholder gates with resource annotations but no decomposition, enabling top-down circuit design.
