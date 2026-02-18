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
# # Writing Custom Optimization Converters
#
# This tutorial explains how to write our own optimization converter by
# subclassing `MathematicalProblemConverter`.  A converter is the bridge between
# a mathematical optimization problem and the quantum circuit that solves it:
# it turns the problem into a cost Hamiltonian and decodes measurement results
# back into classical solutions.
#
# This tutorial focuses on `MathematicalProblemConverter`
# (`qamomile.optimization.converter`) because it handles
# the boilerplate of converting a `BinaryModel` (or `ommx.v1.Instance`) into
# an internal spin model and provides a built-in `decode()` method.  We only
# need to implement one abstract method: `get_cost_hamiltonian()`.

# %% [markdown]
# ## Understanding the Base Class
#
# The key structure of `MathematicalProblemConverter` is:
#
# ```
# MathematicalProblemConverter
#     __init__(instance: ommx.v1.Instance | BinaryModel)
#         -> converts to spin model internally (self.spin_model)
#         -> calls __post_init__()
#
#     get_cost_hamiltonian() -> Hamiltonian    [abstract, we implement this]
#     decode(samples: SampleResult) -> BinarySampleSet   [built-in]
#     __post_init__()                          [optional override]
# ```
#
# When we create a converter instance:
#
# 1. The constructor accepts an `ommx.v1.Instance` or a `BinaryModel`.
# 2. It internally converts the problem to a spin (Ising) model stored in `self.spin_model`.
# 3. It calls `__post_init__()`, which we can override for custom initialization.
# 4. Our `get_cost_hamiltonian()` reads from `self.spin_model` and returns a `Hamiltonian`.
# 5. After quantum execution, `decode()` converts measurement bitstrings back to a `BinarySampleSet` in the original variable type.
#
# The behavior in step 2 differs depending on the input type:
#
# | Input type | Conversion to spin model | Sense handling |
# |---|---|---|
# | `ommx.v1.Instance` | Calls `instance.to_qubo()` internally, then converts QUBO → SPIN | Maximization problems are automatically negated to minimization form. Decoded energies will be negative for maximization problems. |
# | `BinaryModel` | Calls `change_vartype(SPIN)` directly | No sense concept. Coefficients are used as-is -- the user is responsible for the sign convention. |
#
# Let's import the key classes.

# %%
import qamomile.observable as qm_o
from qamomile.circuit.transpiler.job import SampleResult
from qamomile.optimization.binary_model import (
    BinaryModel,
)
from qamomile.optimization.converter import MathematicalProblemConverter

# %% [markdown]
# ## The Observable Module
#
# Before building a converter, let's review the `qamomile.observable` module
# that we use to construct Hamiltonians.
#
# The module provides:
#
# - `Hamiltonian` -- a sum of weighted Pauli operator products plus a constant term
# - `PauliOperator(pauli, index)` -- a single Pauli gate on a specific qubit
# - `Pauli` -- an enum with values `X`, `Y`, `Z`, `I`
# - Shorthand factory functions: `X(i)`, `Y(i)`, `Z(i)` that each return a single-term `Hamiltonian`
#
# Hamiltonians support arithmetic (`+`, `-`, `*`, scalar multiplication), so
# we can build them up naturally.

# %%
# Create single Pauli-Z operators on qubits 0 and 1
Z0 = qm_o.Z(0)
Z1 = qm_o.Z(1)

# Build a Hamiltonian: -1.0 * Z0*Z1 + 0.5 * Z0
H_example = -1.0 * Z0 * Z1 + 0.5 * Z0
print("Example Hamiltonian:", H_example)
print("Number of qubits:", H_example.num_qubits)
print("Constant term:", H_example.constant)

# %% [markdown]
# We can also build Hamiltonians term-by-term using `add_term()`:

# %%
H_manual = qm_o.Hamiltonian()
H_manual.add_term(
    (qm_o.PauliOperator(qm_o.Pauli.Z, 0), qm_o.PauliOperator(qm_o.Pauli.Z, 1)),
    -1.0,
)
H_manual.add_term(
    (qm_o.PauliOperator(qm_o.Pauli.Z, 0),),
    0.5,
)
print("Manual Hamiltonian:", H_manual)
print("Are they equal?", H_example == H_manual)

# %% [markdown]
# ## Example: A Simple Ising Converter
#
# Now let's create a custom converter.  The simplest case is the standard Ising
# encoding: map each spin variable to a Pauli-Z operator.  This is essentially
# what `QAOAConverter` does internally, but writing it from scratch illustrates
# the pattern clearly.
#
# The spin model (`self.spin_model`) is a `BinaryModel` in SPIN vartype.
# It has the following properties:
#
# - `linear` -- `dict[int, float]` mapping qubit index to its linear coefficient
# - `quad` -- `dict[tuple[int, int], float]` mapping qubit pairs to quadratic coefficients
# - `higher` -- `dict[tuple[int, ...], float]` for higher-order terms
# - `constant` -- `float` constant offset
# - `num_bits` -- `int` number of spin variables


# %%
class SimpleIsingConverter(MathematicalProblemConverter):
    """A simple converter that creates a Z-only Hamiltonian from the spin model.

    This converter maps each spin variable s_i to a Pauli Z_i operator.
    The resulting Hamiltonian is:

        H = sum_{(i,j)} J_ij Z_i Z_j  +  sum_i h_i Z_i  +  constant
    """

    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        hamiltonian = qm_o.Hamiltonian()

        # Add Z-Z interaction terms (quadratic)
        for (i, j), Jij in self.spin_model.quad.items():
            hamiltonian += Jij * qm_o.Z(i) * qm_o.Z(j)

        # Add single-Z field terms (linear)
        for i, hi in self.spin_model.linear.items():
            hamiltonian += hi * qm_o.Z(i)

        # Add constant energy offset
        hamiltonian += self.spin_model.constant

        return hamiltonian


# %% [markdown]
# That is the entire converter.  The `decode()` method is inherited from
# `MathematicalProblemConverter` and handles the conversion from measurement
# bitstrings back to the original variable type automatically.

# %% [markdown]
# ## Defining a Problem with BinaryModel
#
# To test our converter we need a problem instance.  The simplest way is to use
# `BinaryModel` directly, avoiding the need for JijModeling.
#
# A `BinaryModel` is constructed from a `BinaryExpr` that holds the variable
# type (BINARY or SPIN), a constant offset, and a dictionary of coefficients
# keyed by index tuples.  There are also convenience constructors like
# `from_ising()` and `from_qubo()`.
#
# Let's create a 3-spin Ising problem:
#
# $$
# E(s) = -1.0 \, s_0 s_1  -0.5 \, s_1 s_2  + 0.3 \, s_0
# $$
#
# where $s_i \in \{+1, -1\}$.

# %%
model = BinaryModel.from_ising(
    linear={0: 0.3},
    quad={(0, 1): -1.0, (1, 2): -0.5},
    constant=0.0,
)

print("Variable type:", model.vartype)
print("Number of spins:", model.num_bits)
print("Linear terms:", model.linear)
print("Quadratic terms:", model.quad)
print("Constant:", model.constant)

# %% [markdown]
# ## Creating the Converter and Inspecting the Hamiltonian
#
# Pass the `BinaryModel` to our `SimpleIsingConverter`.  The base class
# internally converts it to a spin model (which is already in SPIN form here).

# %%
converter = SimpleIsingConverter(model)
cost_hamiltonian = converter.get_cost_hamiltonian()

print("Cost Hamiltonian:", cost_hamiltonian)
print("Number of qubits:", cost_hamiltonian.num_qubits)
print("Constant:", cost_hamiltonian.constant)
print()
print("Terms:")
for ops, coeff in cost_hamiltonian:
    print(f"  {ops} -> {coeff}")

# %% [markdown]
# ## Building a Variational Circuit
#
# Now let's build a simple variational quantum eigensolver (VQE) circuit
# that uses our Hamiltonian.  The circuit takes the Hamiltonian as an
# `Observable` parameter and computes its expectation value using `qmc.expval()`.

# %%
import qamomile.circuit as qmc


@qmc.qkernel
def variational_circuit(
    n_qubits: qmc.UInt,
    theta: qmc.Vector[qmc.Float],
    H: qmc.Observable,
) -> qmc.Float:
    """A simple variational ansatz: Ry rotations + CNOT entangling layer."""
    q = qmc.qubit_array(n_qubits, name="q")

    # Apply parameterized Ry rotations
    for i in qmc.range(n_qubits):
        q[i] = qmc.ry(q[i], theta[i])

    # Apply CNOT entangling layer
    for i in qmc.range(n_qubits - 1):
        q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])

    # Compute and return the expectation value
    return qmc.expval(q, H)


# %% [markdown]
# ## End-to-End Workflow
#
# Let's put everything together: define a problem, create a converter,
# build the Hamiltonian, transpile the circuit, execute it, and decode
# the results.

# %%
import numpy as np

from qamomile.qiskit import QiskitTranspiler

# Step 1: Define the problem (reuse the model from above)
print("Step 1: Problem defined")
print(f"  {model.num_bits} spins, linear={model.linear}, quad={model.quad}")

# Step 2: Create the converter
converter = SimpleIsingConverter(model)
print("\nStep 2: Converter created")

# Step 3: Get the cost Hamiltonian
cost_hamiltonian = converter.get_cost_hamiltonian()
print(f"\nStep 3: Cost Hamiltonian ({cost_hamiltonian.num_qubits} qubits)")
for ops, coeff in cost_hamiltonian:
    print(f"  {ops} : {coeff}")

# Step 4: Transpile the variational circuit with the Hamiltonian
transpiler = QiskitTranspiler()
n_qubits = cost_hamiltonian.num_qubits

executable = transpiler.transpile(
    variational_circuit,
    bindings={
        "n_qubits": n_qubits,
        "H": cost_hamiltonian,
    },
    parameters=["theta"],
)
print("\nStep 4: Circuit transpiled")

# Step 5: Execute with some initial parameters
np.random.seed(901)
theta_init = np.random.uniform(0, np.pi, size=n_qubits)
job = executable.run(
    transpiler.executor(),
    bindings={"theta": theta_init},
)
expval_result = job.result()
print(f"\nStep 5: Expectation value = {expval_result:.4f}")

# %% [markdown]
# ### Optimizing the Parameters
#
# We can use a classical optimizer to find the parameters that minimize
# the expectation value of our cost Hamiltonian.

# %%
from scipy.optimize import minimize

energy_history = []


def objective(params, transpiler, executable):
    job = executable.run(
        transpiler.executor(),
        bindings={"theta": params},
    )
    energy = job.result()
    energy_history.append(energy)
    return energy


init_params = np.random.uniform(0, np.pi, size=n_qubits)

energy_history = []
result_opt = minimize(
    objective,
    init_params,
    args=(transpiler, executable),
    method="COBYLA",
    options={"maxiter": 100},
)

print(f"Optimized energy: {result_opt.fun:.4f}")
print(f"Optimal theta: {result_opt.x}")

# %% [markdown]
# ### Visualizing Convergence

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(energy_history, marker="o", markersize=3)
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("VQE Optimization Convergence")
plt.grid(True)
plt.tight_layout()

# %% [markdown]
# ### Decoding Results via Sampling
#
# To extract a classical solution we switch to sampling mode.  We build
# a measurement circuit, sample bitstrings, and decode them using the
# converter's `decode()` method.


# %%
@qmc.qkernel
def sampling_circuit(
    n_qubits: qmc.UInt,
    theta: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    """Same ansatz as above, but ending with measurement."""
    q = qmc.qubit_array(n_qubits, name="q")
    for i in qmc.range(n_qubits):
        q[i] = qmc.ry(q[i], theta[i])
    for i in qmc.range(n_qubits - 1):
        q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])
    return qmc.measure(q)


sample_executable = transpiler.transpile(
    sampling_circuit,
    bindings={"n_qubits": n_qubits},
    parameters=["theta"],
)

# Sample with the optimized parameters
sample_job = sample_executable.sample(
    transpiler.executor(),
    bindings={"theta": result_opt.x},
    shots=1024,
)
sample_result = sample_job.result()

print("Top measurement results:")
for bitstring, count in sample_result.most_common(5):
    print(f"  {bitstring} : {count} counts")

# %% [markdown]
# Now decode the measurement results back into the original variable domain
# using `converter.decode()`.  Since our input model used SPIN variables,
# the decoded samples will also be in SPIN form ($+1$ / $-1$).

# %%
decoded = converter.decode(sample_result)

print(f"Variable type: {decoded.vartype}")
print(f"Number of samples: {len(decoded.samples)}")

# Show the lowest-energy solution
best_sample, best_energy, best_count = decoded.lowest()
print("\nBest solution:")
print(f"  Sample: {best_sample}")
print(f"  Energy: {best_energy:.4f}")
print(f"  Occurrences: {best_count}")

print(f"\nMean energy: {decoded.energy_mean():.4f}")

# %% [markdown]
# ## Working with BINARY Variables
#
# The converter handles variable-type conversion automatically.  If we
# provide a BINARY model, the base class converts it to SPIN internally,
# and `decode()` converts the results back to BINARY (0/1).

# %%
binary_model = BinaryModel.from_qubo(
    qubo={
        (0, 1): -2.0,
        (1, 2): -1.0,
        (0, 0): 1.0,  # diagonal = linear term in QUBO
        (1, 1): -0.5,
    },
    constant=0.0,
)

print("Input vartype:", binary_model.vartype)

binary_converter = SimpleIsingConverter(binary_model)
H_binary = binary_converter.get_cost_hamiltonian()
print("Cost Hamiltonian:", H_binary)
print("Num qubits:", H_binary.num_qubits)

# %% [markdown]
# When we decode, results come back in BINARY form:

# %%
# Simulate a fake sample result for demonstration
fake_result = SampleResult(
    results=[
        ([0, 0, 0], 100),  # all |0>
        ([1, 1, 0], 200),  # |110>
        ([1, 1, 1], 150),  # |111>
    ],
    shots=450,
)

decoded_binary = binary_converter.decode(fake_result)
print(f"Decoded vartype: {decoded_binary.vartype}")
for sample, energy in zip(decoded_binary.samples, decoded_binary.energy):
    print(f"  {sample} -> energy = {energy:.4f}")

# %% [markdown]
# ## Using JijModeling and `ommx.v1.Instance`
#
# In practice, optimization problems are usually defined symbolically using
# JijModeling and compiled into an `ommx.v1.Instance`.  The base class
# `MathematicalProblemConverter` accepts either a `BinaryModel` or an
# `ommx.v1.Instance`, so our `SimpleIsingConverter` works with both
# -- no code changes needed.
#
# Let's demonstrate this with a small Max-Cut problem on a 4-node graph.

# %%
import jijmodeling as jm
import networkx as nx

# Define the Max-Cut problem symbolically
problem = jm.Problem("Maxcut", sense=jm.ProblemSense.MAXIMIZE)


@problem.update
def _(problem: jm.DecoratedProblem):
    V = problem.Dim()
    E = problem.Graph()
    x = problem.BinaryVar(shape=(V,))
    obj = (
        E.rows()
        .map(lambda e: 1 / 2 * (1 - (2 * x[e[0]] - 1) * (2 * x[e[1]] - 1)))
        .sum()
    )
    problem += obj


# Create a small graph instance
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3)])

data = {"V": G.number_of_nodes(), "E": list(G.edges())}
instance = problem.eval(data)

# %% [markdown]
# Now pass the `ommx.v1.Instance` directly to our custom converter:

# %%
ommx_converter = SimpleIsingConverter(instance)
H_ommx = ommx_converter.get_cost_hamiltonian()

print("Cost Hamiltonian from OMMX instance:")
print(H_ommx)
print(f"Number of qubits: {H_ommx.num_qubits}")

# %% [markdown]
# The converter works identically -- the base class internally converts
# the `ommx.v1.Instance` to a QUBO, then to a spin model.  We can
# sample and decode results exactly as before:

# %%
ommx_sample_exec = transpiler.transpile(
    sampling_circuit,
    bindings={"n_qubits": H_ommx.num_qubits},
    parameters=["theta"],
)

ommx_sample_job = ommx_sample_exec.sample(
    transpiler.executor(),
    bindings={"theta": np.random.uniform(0, np.pi, size=H_ommx.num_qubits)},
    shots=512,
)
ommx_decoded = ommx_converter.decode(ommx_sample_job.result())

print(f"Decoded vartype: {ommx_decoded.vartype}")
best_sample, best_energy, best_count = ommx_decoded.lowest()
print(f"Best solution: {best_sample}")
print(f"Best energy: {best_energy:.4f}")

# %% [markdown]
# ## Advanced: Overriding `__post_init__`
#
# The `MathematicalProblemConverter` base class calls `__post_init__()` at the
# end of `__init__()`, after `self.spin_model` is available.  We can override
# this hook to perform custom initialization that depends on the spin model.
#
# For example, the built-in `QRAC31Converter` uses `__post_init__` to perform
# graph coloring on the interaction graph, which determines how spins are
# packed into qubits:
#
# ```python
# class QRAC31Converter(MathematicalProblemConverter):
#     def __post_init__(self) -> None:
#         _, color_group = greedy_graph_coloring(
#             graph=self.spin_model.quad.keys(),
#             max_color_group_size=3,
#         )
#         self.color_group = color_group
#         self.encoded_ope = color_group_to_qrac_encode(color_group)
#         # ...
# ```
#
# Here is a practical example: precomputing the **interaction graph** from
# the spin model.  This is useful when the converter needs structural
# information about variable interactions (e.g., for circuit layout
# optimization or variable ordering heuristics).


# %%
class GraphAwareIsingConverter(MathematicalProblemConverter):
    """Ising converter that precomputes the interaction graph structure."""

    def __post_init__(self) -> None:
        # Build adjacency list and degree information from the spin model
        self.adjacency: dict[int, list[int]] = {
            i: [] for i in range(self.spin_model.num_bits)
        }
        for i, j in self.spin_model.quad:
            self.adjacency[i].append(j)
            self.adjacency[j].append(i)

        self.degree = {
            node: len(neighbors) for node, neighbors in self.adjacency.items()
        }

        # Identify isolated variables (no interactions) and hub variables
        self.isolated = [i for i, d in self.degree.items() if d == 0]
        self.max_degree_node = (
            max(self.degree, key=self.degree.get) if self.degree else None
        )

    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        hamiltonian = qm_o.Hamiltonian()
        for (i, j), Jij in self.spin_model.quad.items():
            hamiltonian += Jij * qm_o.Z(i) * qm_o.Z(j)
        for i, hi in self.spin_model.linear.items():
            hamiltonian += hi * qm_o.Z(i)
        hamiltonian += self.spin_model.constant
        return hamiltonian


# Demonstrate the precomputed structure
graph_converter = GraphAwareIsingConverter(model)

print("Adjacency list:", graph_converter.adjacency)
print("Degree:", graph_converter.degree)
print("Isolated nodes:", graph_converter.isolated)
print("Highest-degree node:", graph_converter.max_degree_node)

# The Hamiltonian itself is the same as SimpleIsingConverter
H_graph = graph_converter.get_cost_hamiltonian()
print("\nCost Hamiltonian:", H_graph)

# %% [markdown]
# The `__post_init__` hook lets us analyze the problem structure once at
# construction time.  This information could then be used for:
#
# - **Circuit layout**: mapping high-degree variables to well-connected qubits
# - **Variable ordering**: processing hub nodes first in custom ansatz designs
# - **Problem diagnostics**: detecting disconnected subproblems
#
# The built-in converters use this pattern extensively:
#
# - `QRAC31Converter`: graph coloring to determine QRAC encoding
# - `FQAOAConverter`: cyclic variable mapping for fermionic encoding

# %% [markdown]
# ## Summary
#
# In this tutorial we covered:
#
# 1. **The `MathematicalProblemConverter` base class** -- accepts a
#    `BinaryModel` or `ommx.v1.Instance`, converts it to a spin model
#    internally, and provides a built-in `decode()` method.
#
# 2. **Implementing `get_cost_hamiltonian()`** -- the one abstract method
#    we must implement.  Read from `self.spin_model` (linear, quad, higher,
#    constant) and return a `qm_o.Hamiltonian`.
#
# 3. **End-to-end workflow** -- define a problem, create a converter, build
#    a variational circuit with `qmc.expval()`, optimize parameters, sample,
#    and decode results.
#
# 4. **Using `ommx.v1.Instance`** -- define problems symbolically with
#    JijModeling, compile to an instance, and pass it directly to the
#    converter.  No converter changes needed.
#
# 5. **Overriding `__post_init__()`** -- for custom initialization logic
#    that runs after the spin model is available (e.g., precomputing
#    interaction graph structure, graph coloring).
#
# ### Built-in converters for reference
#
# - `QAOAConverter` (`qamomile.optimization.qaoa`) -- standard QAOA with
#   Z-only Hamiltonian, includes a `transpile()` method for the QAOA ansatz.
#
# - `QRAC31Converter` (`qamomile.optimization.qrao`) -- Quantum Random Access
#   Code encoding, packs up to 3 spins per qubit using X/Y/Z operators.
#
# - `FQAOAConverter` (`qamomile.optimization.fqaoa`) -- Fermionic QAOA with
#   particle-number conservation for constrained optimization.
