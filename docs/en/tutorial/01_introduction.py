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
# # Introduction to Qamomile
#
# Welcome to Qamomile, a quantum computing SDK for building, analyzing,
# and executing quantum circuits in Python.
#
# ## What We Will Learn
# - What Qamomile is and where it fits in the quantum ecosystem
# - Creating and running our first quantum circuit
# - The affine type system (no-cloning enforcement)
# - Execution with QiskitTranspiler
# - How Qamomile traces and transpiles quantum programs
# - Brief introduction to parametric circuits
# - Brief introduction to resource estimation
# - Brief introduction to the standard library and algorithm library
# - Overview of optimization features

# %% [markdown]
# ## 1. What is Qamomile?
#
# **Qamomile** is a Python SDK for quantum computing, designed to bridge
# the path from Noisy Intermediate-Scale Quantum (NISQ) computing to
# Fault-Tolerant Quantum Computing (FTQC). It provides a single framework
# for writing quantum programs that work across both paradigms.
#
# ### Positioning
#
# Qamomile supports both paradigms of quantum computing:
#
# - **NISQ algorithms**: Variational algorithms like QAOA and VQE
# - **Fault-tolerant algorithms**: Exact algorithms like QPE through the
#   standard library, with algebraic resource estimation for planning
#
# ### Key Features
#
# | Feature | Description |
# |---------|-------------|
# | **Pythonic syntax** | Define quantum program with `@qkernel` decorator |
# | **Type safety** | All parameters and returns require type annotations |
# | **Affine types** | Enforces quantum no-cloning before execution |
# | **Multi-backend** | Currently Qiskit; CUDA-Q and QURI Parts coming soon, more planned |
# | **Standard library** | Built-in QFT, IQFT, QPE with decomposition strategies (more planned) |
# | **Resource estimation** | Symbolic gate counts and depth analysis |
# | **Optimization** | QAOA, FQAOA, QRAO converters with [ommx](https://jij-inc.github.io/ommx/en/introduction.html) integration (more planned) |

# %% [markdown]
# ## 2. Our First Quantum Circuit
#
# ### Quantum Programs as `@qkernel` Functions
#
# In Qamomile, **every quantum program (quantum circuit) is written as
# a Python function decorated with `@qmc.qkernel`**. This is the single
# entry point for defining quantum computation — there is no other way
# to create circuits.
#
# A `@qkernel` function has a few minimum requirements:
#
# 1. **The `@qmc.qkernel` decorator** — marks the function as a quantum
#    kernel that Qamomile can trace, visualize, and transpile.
# 2. **Type annotations on all parameters and the return type** — Qamomile
#    uses these to determine qubit allocation, parameter handling, and
#    measurement decoding. Omitting them will raise an error at trace time.
# 3. **A return value** — the function must return something (measurement
#    results, qubits, or an expectation value). The return type annotation
#    tells Qamomile how to interpret the output.
#
# ```python
# @qmc.qkernel
# def my_circuit(param: qmc.Float) -> qmc.Bit:   # annotations required
#     q = qmc.qubit(name="q")      # allocate qubits inside the function
#     q = qmc.ry(q, param)         # apply gates (reassign to respect affine types)
#     return qmc.measure(q)        # return measurement result
# ```
#
# See [02_type_system](02_type_system.ipynb) for the full catalogue of
# available types.
#
# ### Qubits and Gates
#
# A **qubit** is the basic unit of quantum information. Unlike a classical
# bit (always 0 or 1), a qubit can exist in a **superposition** of
# $|0\rangle$ and $|1\rangle$ until measured.
#
# Quantum **gates** transform qubit states. The simplest gate is the
# **X gate** (NOT gate), which flips $|0\rangle \to |1\rangle$ and vice versa.
#
# Let's create our first circuit.

# %%
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler


@qmc.qkernel
def x_gate_circuit() -> qmc.Bit:
    """Apply X gate to flip |0> to |1>."""
    q = qmc.qubit(name="q")
    q = qmc.x(q)
    return qmc.measure(q)


x_gate_circuit.draw()

# %% [markdown]
# ### Code Breakdown
#
# 1. **`@qmc.qkernel`**: Marks this function as a quantum kernel
# 2. **`-> qmc.Bit`**: Return type annotation (measurement result)
# 3. **`qmc.qubit(name="q")`**: Creates one qubit, initialized to $|0\rangle$
# 4. **`q = qmc.x(q)`**: Applies the X gate. Note the reassignment!
# 5. **`qmc.measure(q)`**: Measures the qubit, returning a classical `Bit`
# 6. **`x_gate_circuit.draw()`**: Visualizes the circuit diagram. Every `@qkernel` has a `.draw()` method that renders the circuit using Matplotlib.

# %% [markdown]
# ## 3. The Linear Type System
#
# We may have noticed the pattern `q = qmc.x(q)` — why must we reassign?
#
# In quantum mechanics, qubits cannot be copied (the **no-cloning theorem**).
# Qamomile enforces this through its **affine type system**: once a qubit
# enters a gate, the old handle is consumed and a new handle is returned.
# We must always capture the return value.
#
# ```python
# # Correct
# q = qmc.h(q)      # captures the new handle
# q = qmc.x(q)      # uses the updated handle
#
# # Wrong — will cause an error
# qmc.h(q)           # ignores return value
# qmc.x(q)           # tries to use consumed handle
# ```


# %%
# Error example: using a qubit twice
@qmc.qkernel
def bad_example() -> tuple[qmc.Bit, qmc.Bit]:
    q = qmc.qubit(name="q")
    q1 = qmc.h(q)  # consumes q
    q2 = qmc.x(q)  # ERROR: q was already consumed
    return qmc.measure(q1), qmc.measure(q2)


try:
    bad_example.draw()
except Exception as e:
    print(f"Error (expected): {type(e).__name__}: {e}")

# %% [markdown]
# ### Linear Type Rules
#
# | Code | Valid? | Reason |
# |------|--------|--------|
# | `q = qmc.h(q)` | OK | Reassigns the return value |
# | `qmc.h(q)` | NG | Ignores the return value |
# | `q1 = qmc.h(q); q2 = qmc.x(q)` | NG | Uses q twice |
# | `q = qmc.h(q); q = qmc.x(q)` | OK | Sequential updates |
#
# For multi-qubit gates, both qubits are returned:
# ```python
# q0, q1 = qmc.cx(q0, q1)   # CNOT returns both qubits
# ```

# %% [markdown]
# ## 4. Execution with Qiskit
#
# Qamomile circuits are backend-agnostic. To execute them, we use a
# **transpiler** that converts the circuit to a specific backend format.
#
# Currently, `QiskitTranspiler` is the supported backend.
# Support for CUDA-Q and QURI Parts is under active development,
# with more backends planned. Because circuits are defined as
# backend-agnostic `@qkernel` functions, our code will work
# unchanged as new backends become available.
#
# The execution pipeline has two modes depending on what the kernel returns:
# ```
# @qkernel (returns Bit/Vector[Bit]/Float via measure)
#   → transpile() → ExecutableProgram → sample(executor, shots=N) → SampleResult
#
# @qkernel (returns Float via expval)
#   → transpile() → ExecutableProgram → run(executor) → Float
# ```
#
# - **`sample()`**: Shot-based measurement. Returns a `SampleResult` with
#   `(value, count)` pairs — one entry per distinct outcome.
# - **`run()`**: Expectation value computation. Returns a `Float` value
#   directly (the expectation $\langle\psi|H|\psi\rangle$). Used with
#   `qmc.expval()` in variational algorithms (see
#   [08_parametric_circuits](08_parametric_circuits.ipynb)).
#
# In this tutorial we use `sample()` exclusively. We will encounter
# `run()` when working with observables and variational circuits.

# %%
# Create transpiler
transpiler = QiskitTranspiler()

# Compile the circuit
executable = transpiler.transpile(x_gate_circuit)

# Execute on simulator (1000 shots)
job = executable.sample(transpiler.executor(), shots=1000)
result = job.result()

print("=== X Gate Circuit Results ===")
for value, count in result.results:
    print(f"  {value}: {count}")

# %% [markdown]
# The X gate flips $|0\rangle$ to $|1\rangle$, so all 1000 measurements
# should yield `1`.
#
# We can also inspect the transpiled Qiskit circuit:

# %%
qiskit_circuit = executable.get_first_circuit()
qiskit_circuit.draw(output="mpl")

# %% [markdown]
# ### Qubit Ordering Convention
#
# When working with multiple qubits, Qamomile uses the following convention:
#
# - **Ket notation** is big-endian: the leftmost bit is the
#   **highest-indexed** qubit. For example, $|110\rangle$ for 3 qubits
#   means `q[2]=1, q[1]=1, q[0]=0`.
# - **Tuple results** follow array order: `(q[0], q[1], ..., q[n-1])`.
#   So the state $|110\rangle$ appears as `(0, 1, 1)` in the measurement
#   result.
#
# Let's see this in action with a 3-qubit circuit.


# %%
@qmc.qkernel
def ordering_demo() -> tuple[qmc.Bit, qmc.Bit, qmc.Bit]:
    """Demonstrate qubit ordering: q0=0, q1=1, q2=1 → ket |110>."""
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")
    q2 = qmc.qubit(name="q2")
    # q0 stays |0>
    q1 = qmc.x(q1)  # q1 → |1>
    q2 = qmc.x(q2)  # q2 → |1>
    return qmc.measure(q0), qmc.measure(q1), qmc.measure(q2)


ordering_demo.draw()

# %%
exec_ord = transpiler.transpile(ordering_demo)
result_ord = exec_ord.sample(transpiler.executor(), shots=100).result()

print("=== Qubit Ordering Demo ===")
for value, count in result_ord.results:
    print(f"  {value}: {count}")

# %% [markdown]
# The state is `q0=0, q1=1, q2=1`. In ket notation (big-endian: q2 q1 q0)
# this is $|110\rangle$, but the tuple result is `(0, 1, 1)` following
# array order `(q0, q1, q2)`.

# %% [markdown]
# ## 5. Tracing and Compilation
#
# When we define a `@qmc.qkernel` function, Qamomile does not execute
# the quantum operations immediately. Instead, it uses a two-phase approach:
#
# 1. **Tracing**: When we call `.draw()` or `transpile()`, Qamomile
#    **traces** the function body to build an intermediate representation
#    (IR) — a directed graph of operations and data dependencies.
# 2. **Transpilation**: The IR graph is processed through a multi-pass
#    pipeline that optimizes and converts it into a backend-specific
#    circuit.
#
# ```
# @qkernel function
#     ↓  trace
# IR Graph (operations + dependencies)
#     ↓  inline → constant_fold → analyze → separate → emit
# Backend Circuit (e.g., Qiskit QuantumCircuit)
# ```
#
# This architecture provides two key benefits:
#
# - **Backend independence**: Define a circuit once as a `@qkernel`,
#   then transpile it to any supported backend without modification.
# - **Optimization opportunities**: The multi-pass pipeline can inline
#   subroutines, fold constants, and analyze dependencies before
#   generating the final circuit.
#
# See [10_transpile](10_transpile.ipynb) for a deep dive into the transpiler pipeline.

# %% [markdown]
# ## 6. Parametric Circuits
#
# Many quantum algorithms use circuits with tunable parameters.
# In Qamomile, we use `qmc.Float` as the parameter type.

# %%
import math


@qmc.qkernel
def rotation_circuit(theta: qmc.Float) -> qmc.Bit:
    """Parameterized rotation around the Y-axis."""
    q = qmc.qubit(name="q")
    q = qmc.ry(q, theta)
    return qmc.measure(q)


rotation_circuit.draw()

# %% [markdown]
# We can also pass parameter values directly to `draw()` to see the
# circuit with concrete values filled in:

# %%
rotation_circuit.draw(theta=math.pi / 4)

# %% [markdown]
# ### bindings vs parameters
#
# When transpiling, we can provide values in two ways:
#
# - **`bindings`**: Values fixed at transpile time (circuit structure may depend on them)
# - **`parameters`**: Values that remain free and can be changed between executions
#   without retranspiling

# %%
# Fix theta at transpile time
exec_fixed = transpiler.transpile(rotation_circuit, bindings={"theta": math.pi / 2})
result_fixed = exec_fixed.sample(transpiler.executor(), shots=1000).result()

print("=== RY(pi/2) — fixed at transpile time ===")
for value, count in result_fixed.results:
    percentage = count / 1000 * 100
    print(f"  {value}: {count} ({percentage:.1f}%)")

# %%
# Keep theta as a free parameter
exec_param = transpiler.transpile(rotation_circuit, parameters=["theta"])

# Execute multiple times with different values without retranspiling
for angle, name in [(0, "0"), (math.pi / 4, "pi/4"), (math.pi, "pi")]:
    res = exec_param.sample(
        transpiler.executor(), bindings={"theta": angle}, shots=1000
    ).result()
    counts = {str(v): c for v, c in res.results}
    print(f"RY({name}): {counts}")

# %% [markdown]
# This is the foundation of variational quantum algorithms (such as VQE and QAOA),
# where parameters are optimized in a classical-quantum loop.

# %% [markdown]
# ## 7. Resource Estimation
#
# Before running on real hardware, we may want to estimate how many qubits
# and gates our circuit requires. Qamomile provides **algebraic resource
# estimation** that works with symbolic parameters.
#
# When a circuit uses `qmc.range()` with a symbolic `UInt` bound, the
# estimator produces SymPy expressions that describe how resources scale
# with problem size.


# %%
from qamomile.circuit.estimator import estimate_resources


@qmc.qkernel
def ghz_circuit(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """GHZ state with symbolic size n."""
    q = qmc.qubit_array(n, name="q")
    q[0] = qmc.h(q[0])
    for i in qmc.range(n - 1):
        q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])
    return q


est_ghz = estimate_resources(ghz_circuit.block)

print("=== GHZ State Resources (symbolic) ===")
print(f"  Qubits:        {est_ghz.qubits}")
print(f"  Total gates:   {est_ghz.gates.total}")
print(f"  Two-qubit:     {est_ghz.gates.two_qubit}")
print(f"  Circuit depth: {est_ghz.depth.total_depth}")

# %% [markdown]
# The results contain the symbol **n** — closed-form expressions that
# describe how resources grow with problem size. See [09_resource_estimation](09_resource_estimation.ipynb)
# for the full resource estimation tutorial.

# %% [markdown]
# ## 8. Standard Library & Algorithm Library
#
# Qamomile provides two libraries of pre-built circuit components:
#
# **Standard library** (`qamomile.circuit.stdlib`): Commonly used building
# blocks for quantum algorithms.
#
# | Component | Description |
# |-----------|-------------|
# | `qft` / `iqft` | Quantum Fourier Transform (with multiple decomposition strategies) |
# | `qpe` | Quantum Phase Estimation |
#
# **Algorithm library** (`qamomile.circuit.algorithm`): More specific circuit
# patterns for variational and optimization algorithms.
#
# | Component | Description |
# |-----------|-------------|
# | `qaoa_circuit`, `qaoa_state` | QAOA circuit construction |
# | `fqaoa_layers`, `fqaoa_state` | Fermionic QAOA components (Givens rotations, hopping gates) |
# | `rx_layer`, `ry_layer`, `rz_layer` | Parameterized rotation layers |
# | `cz_entangling_layer` | CZ entangling layer |
#
# Both libraries are under active development. We welcome feedback on which
# circuit patterns and algorithms to add next — please open an issue on
# [GitHub](https://github.com/Jij-Inc/Qamomile).
#
# See [05_stdlib](05_stdlib.ipynb) for a detailed standard library tutorial.

# %% [markdown]
# ## 9. Optimization Features
#
# Qamomile includes converters for solving combinatorial optimization
# problems on quantum computers.
#
# The optimization pipeline:
# ```
# Mathematical Model (JijModeling)
#      ↓  Interpreter + ommx.v1.Instance
# Converter (QAOAConverter, FQAOAConverter, QRAO31Converter)
#      ↓  get_cost_hamiltonian() / transpile()
# Quantum Circuit + Classical Optimization Loop
#      ↓  decode()
# Solution
# ```
#
# Available converters:
#
# | Converter | Algorithm | Use Case |
# |-----------|-----------|----------|
# | `QAOAConverter` | QAOA | General combinatorial optimization |
# | `FQAOAConverter` | Fermionic QAOA | Constrained optimization (exact constraint enforcement) |
# | `QRAO31Converter` | QRAO 3-to-1 | Qubit-efficient encoding (3 variables per qubit) |
#
# More converters are planned as new quantum optimization algorithms are developed.
# We actively welcome community feedback on which algorithms and converters
# to prioritize next. If we have a use case that would benefit from a new
# converter, please open an issue on [GitHub](https://github.com/Jij-Inc/Qamomile).
#
# For more information about the mathematical modeling layer, see the
# [ommx documentation](https://jij-inc.github.io/ommx/en/introduction.html)
# and the [JijModeling tutorials](https://jij-inc-jijmodeling-tutorials-en.readthedocs-hosted.com/en/latest/introduction.html).
#
# See the optimization tutorials ([QAOA](../optimization/qaoa.ipynb), [FQAOA](../optimization/fqaoa.ipynb), [QRAO](../optimization/qrao31.ipynb), [Custom Converter](../optimization/custom_converter.ipynb)) for details.

# %% [markdown]
# ## 10. Summary
#
# This tutorial covered the essential concepts of Qamomile:
#
# 1. **`@qmc.qkernel`**: Define quantum circuits as Python functions
# 2. **Affine types**: Always reassign after gates (`q = qmc.h(q)`)
# 3. **Execution**: `QiskitTranspiler` → `transpile()` → `sample()`
# 4. **Tracing and transpilation**: `@qkernel` is traced into an IR graph, then transpiled through a multi-pass pipeline
# 5. **Parametric circuits**: `qmc.Float` parameters with `bindings` / `parameters`
# 6. **Resource estimation**: `estimate_resources(kernel.block)` for symbolic gate counts
# 7. **Standard library & algorithm library**: Pre-built QFT, QPE, QAOA, FQAOA components
# 8. **Optimization**: Converters for QAOA, FQAOA, QRAO (more planned)
#
# ### What's Next
#
# | Tutorial | Topic |
# |----------|-------|
# | `02_type_system.ipynb` | Full type system: Qubit, Float, UInt, Bit, Vector, Dict |
# | `03_gates.ipynb` | Complete gate reference (all 11 gates) |
# | `04_superposition_entanglement.ipynb` | Superposition, interference, Bell/GHZ states |
# | `05_stdlib.ipynb` | QFT, QPE, algorithm module |
# | `06_composite_gate.ipynb` | CompositeGate, `@composite_gate`, stub gates |
# | `07_first_algorithm.ipynb` | Deutsch-Jozsa algorithm |
# | `08_parametric_circuits.ipynb` | Parametric circuits and QAOA from scratch |
# | `09_resource_estimation.ipynb` | Algebraic resource estimation |
# | `10_transpile.ipynb` | Transpiler pipeline internals |
# | `11_custom_executor.ipynb` | Custom backend integration |
# | `optimization/qaoa.ipynb` | QAOA for combinatorial optimization |
# | `optimization/fqaoa.ipynb` | Fermionic QAOA with constraint enforcement |
# | `optimization/qrao31.ipynb` | Quantum Random Access Optimization |
# | `optimization/custom_converter.ipynb` | Building our own converter |

# %% [markdown]
# ## What We Learned
#
# - **What Qamomile is and where it fits in the quantum ecosystem** — Qamomile bridges the path from NISQ to FTQC, providing a single framework for writing quantum programs across both paradigms.
# - **Creating and running our first quantum circuit** — Used `@qmc.qkernel` with `qmc.qubit()`, gates like `qmc.x()`, and `qmc.measure()` to build and visualize circuits.
# - **The affine type system (no-cloning enforcement)** — Gates consume and return qubits; always reassign (`q = qmc.h(q)`) to enforce the quantum no-cloning theorem at transpile time.
# - **Execution with QiskitTranspiler** — `QiskitTranspiler` transpiles kernels via `transpile()` and runs them via `sample()` to obtain measurement results.
# - **How Qamomile traces and transpiles quantum programs** — `@qkernel` functions are traced into an IR graph, then transpiled through a multi-pass pipeline (inline, constant fold, analyze, separate, emit) to produce backend-specific circuits.
# - **Brief introduction to parametric circuits** — `qmc.Float` parameters can be fixed at transpile time with `bindings=` or kept free with `parameters=` for variational algorithms.
# - **Brief introduction to resource estimation** — `estimate_resources(kernel.block)` produces symbolic gate counts and circuit depth, scaling with problem size via SymPy expressions.
# - **Brief introduction to the standard library and algorithm library** - Pre-built components for QFT, QPE and more (even more planned).
# - **Overview of optimization features** — QAOA, FQAOA, and QRAO converters turn JijModeling problems into executable quantum circuits through a model → converter → circuit pipeline.
