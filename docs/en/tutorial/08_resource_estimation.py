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
# # Algebraic Resource Estimation
#
# How many qubits does your algorithm need? How many gates? How does
# the cost scale with problem size? These are the questions that
# **resource estimation** answers -- and Qamomile lets you answer them
# *symbolically*, before committing to any concrete parameter values.
#
# ## What You Will Learn
# - Using `estimate_resources()` to obtain qubit counts, gate counts, and circuit depth as SymPy expressions
# - Iterating over all available metrics with `.to_dict()`
# - Working with symbolic (parametric) circuits and using `.substitute()` and `.simplify()`
# - Analyzing the resource profile of Quantum Phase Estimation (QPE)
# - Comparing decomposition strategies for composite gates
# - Defining custom composite gates with strategies, and using stub gates for black-box resource estimation
#
# **Prerequisites:** Tutorials 01--03 (basic circuit construction).
# Familiarity with QPE is helpful but not required.

# %%
import math

import qamomile.circuit as qmc
from qamomile.circuit.estimator import estimate_resources

# %% [markdown]
# ## 1. What is Algebraic Resource Estimation?
#
# Traditional resource estimation works on a *fixed* circuit: you build
# the circuit for a specific size, then count the gates. This is fine
# for small experiments, but it does not tell you how your algorithm
# *scales*.
#
# Qamomile's algebraic estimator takes a different approach:
#
# 1. You write a `@qkernel` whose parameters may include symbolic
#    sizes (e.g. `n: qmc.UInt`).
# 2. You call `estimate_resources(kernel.block)`.
# 3. The estimator walks the circuit IR -- including loops, function
#    calls, and controlled operations -- and returns qubit counts,
#    gate counts, and depth as **SymPy expressions** that may contain
#    the symbolic parameters.
#
# This lets you explore the full design space without building a
# separate circuit for every size.

# %% [markdown]
# ## 2. Resource Metrics Reference
#
# The object returned by `estimate_resources()` is a `ResourceEstimate`
# with the following fields:
#
# | Attribute                   | Description                               |
# |-----------------------------|-------------------------------------------|
# | `est.qubits`               | Number of logical qubits                  |
# | `est.gates.total`          | Total gate count                          |
# | `est.gates.single_qubit`   | Single-qubit gates (H, X, RZ, P, ...)    |
# | `est.gates.two_qubit`      | Two-qubit gates (CX, CZ, CP, SWAP, ...)  |
# | `est.gates.t_gates`        | T and Tdg gates (critical for fault tolerance) |
# | `est.gates.clifford_gates` | Clifford gates (H, S, CX, CZ, SWAP, ...) |
# | `est.depth.total_depth`    | Circuit depth (sequential upper bound)    |
# | `est.depth.t_depth`        | T-depth                                   |
# | `est.depth.two_qubit_depth`| Two-qubit gate depth                      |
# | `est.parameters`           | Dict mapping symbol names to SymPy symbols|
#
# All values are **SymPy expressions**. Three convenience methods are
# available:
#
# - **`est.simplify()`** -- returns a new `ResourceEstimate` with all
#   expressions simplified via `sympy.simplify()`.
# - **`est.substitute(**kwargs)`** -- returns a new `ResourceEstimate`
#   with the named symbols replaced by concrete values.
# - **`est.to_dict()`** -- returns a nested dictionary with all fields
#   as strings, suitable for serialisation or iteration.
#
# The field set may grow in future releases (e.g. rotation gate counts).
# For forward-compatible code, iterate over `to_dict()` rather than
# hard-coding field names.
#
# ### Qamomile-level vs Backend-level Estimation
#
# **Important:** These estimates reflect the gates defined at the
# **Qamomile IR level**. When transpiled to a backend (Qiskit,
# CUDA-Q, etc.), gate counts may change due to:
#
# - Backend-specific decompositions (e.g. SWAP $\rightarrow$ 3 CX)
# - Optimization passes that reduce gate count or depth
# - Native gate set conversions
#
# Use these estimates for **algorithm-level analysis** and
# **design comparison**, not as exact hardware costs.

# %% [markdown]
# ## 3. Simple Examples
#
# ### 3.1 Bell State -- Concrete Resource Count
#
# Let's start with the simplest possible case: a Bell state circuit
# with no symbolic parameters. The estimator should return plain
# integers.


# %%
@qmc.qkernel
def bell_state() -> qmc.Vector[qmc.Qubit]:
    """Create a Bell state |Phi+> = (|00> + |11>) / sqrt(2)."""
    q = qmc.qubit_array(2, name="q")
    q[0] = qmc.h(q[0])
    q[0], q[1] = qmc.cx(q[0], q[1])
    return q


bell_state.draw()

# %%
est = estimate_resources(bell_state.block)

# Iterate over all fields using to_dict()
data = est.to_dict()
print("Bell State Resource Estimate:")
for section, values in data.items():
    if isinstance(values, dict):
        print(f"  {section}:")
        for key, val in values.items():
            print(f"    {key}: {val}")
    else:
        print(f"  {section}: {values}")

# %% [markdown]
# The `to_dict()` approach shows every metric at a glance and is
# forward-compatible -- if new fields are added in a future release,
# they will appear automatically.
#
# You can also access fields directly:
# ```python
# est.qubits          # 2
# est.gates.total      # 2
# est.gates.two_qubit  # 1
# ```

# %% [markdown]
# ### 3.2 GHZ State -- Parametric Estimation
#
# Now consider a GHZ state whose size `n` is left symbolic.


# %%
@qmc.qkernel
def ghz_state(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Create an n-qubit GHZ state (|0...0> + |1...1>) / sqrt(2)."""
    q = qmc.qubit_array(n, name="q")
    q[0] = qmc.h(q[0])
    for i in qmc.range(n - 1):
        q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])
    return q


ghz_state.draw(n=4, fold_loops=False)

# %%
est_ghz = estimate_resources(ghz_state.block)

print("GHZ State Resource Estimate (symbolic):")
print(f"  Qubits:            {est_ghz.qubits}")
print(f"  Total gates:       {est_ghz.gates.total}")
print(f"  Single-qubit gates:{est_ghz.gates.single_qubit}")
print(f"  Two-qubit gates:   {est_ghz.gates.two_qubit}")
print(f"  T gates:           {est_ghz.gates.t_gates}")
print(f"  Clifford gates:    {est_ghz.gates.clifford_gates}")
print(f"  Total depth:       {est_ghz.depth.total_depth}")
print(f"  T-depth:           {est_ghz.depth.t_depth}")
print(f"  Two-qubit depth:   {est_ghz.depth.two_qubit_depth}")
print(f"  Parameters:        {est_ghz.parameters}")

# %% [markdown]
# The results contain the symbol **n**. This is the key insight of
# algebraic estimation: you get closed-form expressions that describe
# how resources grow with the problem size.
#
# ### 3.3 Substituting Concrete Values
#
# Use `.substitute()` to plug in specific values for the symbolic
# parameters.

# %%
for size in [10, 50, 100]:
    concrete = est_ghz.substitute(n=size)
    print(
        f"  n={size:>3}:  qubits={concrete.qubits}, "
        f"gates={concrete.gates.total}, "
        f"two_qubit={concrete.gates.two_qubit}"
    )

# %% [markdown]
# You can also serialise the estimate as JSON using `to_dict()`:

# %%
import json

data = est_ghz.to_dict()
print(json.dumps(data, indent=2))

# %% [markdown]
# ## 4. QPE Resource Analysis
#
# Quantum Phase Estimation (QPE) is one of the most important
# subroutines in quantum computing. It estimates the phase $\theta$
# of a unitary operator:
#
# $$U|\psi\rangle = e^{2\pi i\theta}|\psi\rangle$$
#
# QPE has four stages:
#
# 1. Prepare $m$ counting qubits in superposition
# 2. Apply controlled-$U^{2^k}$ operations
# 3. Apply the inverse Quantum Fourier Transform (IQFT)
# 4. Measure the counting register
#
# Let's implement each component from scratch and analyze its
# resource profile.
#
# ### 4.1 Inverse QFT (IQFT)
#
# The IQFT is the final step of QPE. It converts phase information
# encoded in the counting register into basis states that can be
# measured.


# %%
@qmc.qkernel
def iqft(qubits: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Inverse Quantum Fourier Transform."""
    n = qubits.shape[0]
    # Swap qubits (reverse order)
    for j in qmc.range(n // 2):
        qubits[j], qubits[n - j - 1] = qmc.swap(qubits[j], qubits[n - j - 1])
    # Apply inverse QFT gates
    for j in qmc.range(n):
        for k in qmc.range(j):
            angle = -math.pi / (2 ** (j - k))
            qubits[j], qubits[k] = qmc.cp(qubits[j], qubits[k], theta=angle)
        qubits[j] = qmc.h(qubits[j])
    return qubits


iqft.draw(qubits=4, fold_loops=False)

# %% [markdown]
# Now let's wrap the IQFT in a kernel that allocates its own qubits
# so that the estimator can track the qubit count and express
# everything in terms of the symbolic size `n`.

# %%


@qmc.qkernel
def iqft_n(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """IQFT on n freshly-allocated qubits."""
    qubits = qmc.qubit_array(n, name="q")
    return iqft(qubits)


est_iqft = estimate_resources(iqft_n.block)

print("IQFT Resource Estimate (symbolic n):")
for section, values in est_iqft.to_dict().items():
    if isinstance(values, dict):
        print(f"  {section}:")
        for key, val in values.items():
            print(f"    {key}: {val}")
    else:
        print(f"  {section}: {values}")

# %% [markdown]
# The IQFT requires $O(n^2)$ gates due to the nested loops: the outer
# loop runs $n$ times and the inner loop runs up to $j$ times, giving
# $\sum_{j=0}^{n-1} j = n(n-1)/2$ controlled-phase gates plus $n$
# Hadamard gates and $\lfloor n/2 \rfloor$ SWAP gates.
#
# Let's verify by substituting a few concrete values:

# %%
for size in [4, 8, 16]:
    concrete = est_iqft.substitute(n=size)
    print(
        f"  n={size:>2}:  qubits={concrete.qubits}, "
        f"gates={concrete.gates.total}, depth={concrete.depth.total_depth}"
    )

# %% [markdown]
# ### 4.2 Target Unitary
#
# For this tutorial we use a simple phase gate $P(\theta)$ as the
# target unitary. Its eigenstate is $|1\rangle$ with eigenvalue
# $e^{i\theta}$.


# %%
@qmc.qkernel
def phase_gate(q: qmc.Qubit, theta: qmc.Float, iter: qmc.UInt) -> qmc.Qubit:
    """Apply P(theta) a total of `iter` times."""
    for i in qmc.range(iter):
        q = qmc.p(q, theta)
    return q


phase_gate.draw(iter=4, fold_loops=False)

# %% [markdown]
# ### 4.3 Full Manual QPE
#
# Now we assemble the full QPE algorithm using the building blocks
# defined above.


# %%
@qmc.qkernel
def qpe_manual(theta: qmc.Float, m: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """QPE with m-bit precision, implemented from basic gates."""
    # Allocate qubits
    counting = qmc.qubit_array(m, name="counting")
    target = qmc.qubit(name="target")

    # Prepare target in eigenstate |1>
    target = qmc.x(target)

    # Step 1: Put counting qubits in superposition
    for i in qmc.range(m):
        counting[i] = qmc.h(counting[i])

    # Step 2: Apply controlled-U^(2^k) operations
    controlled_phase = qmc.controlled(phase_gate)
    for i in qmc.range(m):
        iterations = 2**i
        counting[i], target = controlled_phase(
            counting[i], target, theta=theta, iter=iterations
        )

    # Step 3: Apply IQFT
    counting = iqft(counting)

    # Step 4: Measure
    bits = qmc.measure(counting)
    return bits


qpe_manual.draw(theta=math.pi / 2, m=3, fold_loops=False, inline=True)

# %% [markdown]
# ### 4.4 Simplified vs Unsimplified Expressions
#
# When circuits involve nested loops and function calls, the raw
# SymPy expressions can become unwieldy. The `.simplify()` method
# runs SymPy's simplification engine on every field of the estimate.
# Let's see the difference:

# %%
est_qpe_raw = estimate_resources(qpe_manual.block)
est_qpe_simplified = est_qpe_raw.simplify()

print("Manual QPE -- Before simplify():")
print(f"  Total gates: {est_qpe_raw.gates.total}")
print(f"  Two-qubit:   {est_qpe_raw.gates.two_qubit}")
print(f"  Depth:       {est_qpe_raw.depth.total_depth}")
print()
print("Manual QPE -- After simplify():")
print(f"  Total gates: {est_qpe_simplified.gates.total}")
print(f"  Two-qubit:   {est_qpe_simplified.gates.two_qubit}")
print(f"  Depth:       {est_qpe_simplified.depth.total_depth}")

# %% [markdown]
# **Observations:**
#
# - **Qubits**: $m + 1$ -- $m$ counting qubits plus 1 target qubit.
#   This scales linearly.
# - **Gates**: Grows rapidly with precision due to the $2^m$
#   controlled-phase operations.
# - **Depth**: Also increases significantly because the controlled
#   unitaries are applied sequentially.
#
# Let's substitute a few values to see the concrete numbers:

# %%
for precision in [4, 8, 12]:
    concrete = est_qpe_raw.substitute(m=precision).simplify()
    print(
        f"  m={precision:>2}:  qubits={concrete.qubits}, "
        f"gates={concrete.gates.total}, depth={concrete.depth.total_depth}"
    )

# %% [markdown]
# ### 4.5 Built-in QPE
#
# Qamomile provides a built-in `qmc.qpe()` primitive that handles the
# controlled operations and IQFT internally. Let's compare its
# resource profile with our manual implementation.


# %%
@qmc.qkernel
def simple_phase_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Single application of the phase gate.

    When used with qmc.qpe(), the repetitions (2^k) are handled
    internally by the power parameter.
    """
    return qmc.p(q, theta)


@qmc.qkernel
def qpe_builtin(theta: qmc.Float, n: qmc.UInt) -> qmc.Float:
    """QPE using Qamomile's built-in qpe function."""
    counting = qmc.qubit_array(n, name="counting")
    target = qmc.qubit(name="target")
    target = qmc.x(target)  # Prepare eigenstate

    # qmc.qpe() handles controlled operations and IQFT internally
    phase = qmc.qpe(target, counting, simple_phase_gate, theta=theta)
    return qmc.measure(phase)


qpe_builtin.draw(theta=math.pi / 2, n=3, fold_loops=False, inline=True)

# %%
est_builtin = estimate_resources(qpe_builtin.block)
est_builtin = est_builtin.simplify()

print("Built-in QPE Resource Estimate (symbolic n):")
for section, values in est_builtin.to_dict().items():
    if isinstance(values, dict):
        print(f"  {section}:")
        for key, val in values.items():
            print(f"    {key}: {val}")
    else:
        print(f"  {section}: {values}")

# %% [markdown]
# The built-in QPE produces the same asymptotic scaling as the manual
# implementation. The key advantage of the built-in version is
# convenience and the ability for backends to substitute optimized
# composite-gate implementations.

# %% [markdown]
# ## 5. Comparing QFT Strategies
#
# Qamomile's `QFT` composite gate supports multiple decomposition
# **strategies**. Each strategy offers a different trade-off between
# gate count (cost) and approximation error.
#
# The standard QFT uses $O(n^2)$ gates, while the approximate QFT
# truncates small-angle rotations to reduce the gate count to
# $O(nk)$, where $k$ is the truncation depth.
#
# ### 5.1 Listing Available Strategies

# %%
from qamomile.circuit.stdlib.qft import QFT

print("Available QFT strategies:")
for name in QFT.list_strategies():
    print(f"  - {name}")

# %% [markdown]
# ### 5.2 Comparing Gate Counts
#
# Let's compare the standard and approximate strategies for an
# 8-qubit QFT.

# %%
qft_gate = QFT(8)

standard = qft_gate.get_resources_for_strategy("standard")
approx = qft_gate.get_resources_for_strategy("approximate")  # k=3

print("8-qubit QFT -- Standard Strategy:")
print(f"  H gates:    {standard.custom_metadata['num_h_gates']}")
print(f"  CP gates:   {standard.custom_metadata['num_cp_gates']}")
print(f"  SWAP gates: {standard.custom_metadata['num_swap_gates']}")
print(f"  Total:      {standard.custom_metadata['total_gates']}")

print()
print("8-qubit QFT -- Approximate Strategy (k=3):")
print(f"  H gates:    {approx.custom_metadata['num_h_gates']}")
print(f"  CP gates:   {approx.custom_metadata['num_cp_gates']}")
print(f"  SWAP gates: {approx.custom_metadata['num_swap_gates']}")
print(f"  Total:      {approx.custom_metadata['total_gates']}")

# %% [markdown]
# The approximate strategy significantly reduces the number of
# controlled-phase gates while keeping all Hadamard and SWAP gates.
# The error introduced by truncation scales as $O(n / 2^k)$, so
# increasing $k$ improves accuracy at the cost of more gates.

# %%
# Compare across several sizes
print(f"{'n':>4}  {'Standard':>10}  {'Approx k=3':>12}  {'Savings':>8}")
print("-" * 42)
for n in [4, 8, 16, 32, 64]:
    qft_n = QFT(n)
    std = qft_n.get_resources_for_strategy("standard")
    apx = qft_n.get_resources_for_strategy("approximate")
    std_total = std.custom_metadata["total_gates"]
    apx_total = apx.custom_metadata["total_gates"]
    savings = 1 - apx_total / std_total if std_total > 0 else 0
    print(f"{n:>4}  {std_total:>10}  {apx_total:>12}  {savings:>7.1%}")

# %% [markdown]
# As $n$ grows, the savings from the approximate strategy become
# increasingly significant. For $n = 64$, the approximate QFT uses
# roughly half the gates of the standard decomposition.
#
# ### 5.3 Choosing a Strategy at Circuit Level
#
# You can also select a strategy when applying the composite gate
# inside a `@qkernel`. The estimator will then use the corresponding
# resource metadata automatically:
#
# ```python
# @qmc.qkernel
# def my_qft_circuit() -> qmc.Vector[qmc.Qubit]:
#     q = qmc.qubit_array(8, name="q")
#     qft_gate = QFT(8)
#     result = qft_gate(*[q[i] for i in range(8)], strategy="approximate")
#     for i in range(8):
#         q[i] = result[i]
#     return q
# ```
#
# This gives you fine-grained control over the cost/accuracy
# trade-off without changing the rest of your algorithm.

# %% [markdown]
# ## 6. Custom Composite Gates and Strategies
#
# Qamomile lets you define your own composite gates with multiple
# decomposition strategies. This is useful for:
#
# - Comparing implementation trade-offs (precision vs. gate count)
# - Providing resource metadata for oracles or subroutines whose
#   internals are unknown (stub gates)
#
# ### 6.1 Defining a Custom Composite Gate
#
# Let's create a simple "marking oracle" that applies a phase to
# marked states. We'll provide two strategies: one using
# $P(\pi/4)$ rotations (T-gate equivalent, higher precision) and
# one using only $P(\pi/2)$ rotations (S-gate equivalent,
# Clifford-only).

# %%
from dataclasses import dataclass

from qamomile.circuit.frontend.composite_gate import CompositeGate
from qamomile.circuit.ir.operation.composite_gate import ResourceMetadata


class MarkingOracle(CompositeGate):
    """A custom composite gate demonstrating strategy selection.

    The gate applies a marking pattern to each qubit using
    H-P-H rotations, followed by entangling CZ gates.
    """

    custom_name = "marking_oracle"

    def __init__(self, num_qubits: int):
        self._num_qubits = num_qubits

    @property
    def num_target_qubits(self) -> int:
        return self._num_qubits

    def _decompose(self, qubits):
        """Standard decomposition using P(pi/4) (T-equivalent) rotations."""
        n = self._num_qubits
        qubits_list = list(qubits)
        for i in range(n):
            qubits_list[i] = qmc.h(qubits_list[i])
            qubits_list[i] = qmc.p(qubits_list[i], math.pi / 4)  # T-equivalent
            qubits_list[i] = qmc.h(qubits_list[i])
        for i in range(n - 1):
            qubits_list[i], qubits_list[i + 1] = qmc.cz(
                qubits_list[i], qubits_list[i + 1]
            )
        return tuple(qubits_list)

    def _resources(self) -> ResourceMetadata:
        n = self._num_qubits
        return ResourceMetadata(
            t_gate_count=n,
            custom_metadata={
                "num_h_gates": 2 * n,
                "num_p_gates": n,
                "num_cz_gates": n - 1,
                "total_gates": 3 * n + (n - 1),
            },
        )


# %% [markdown]
# ### 6.2 Adding a Strategy
#
# Now let's define a Clifford-only strategy that replaces $P(\pi/4)$
# with $P(\pi/2)$ (S-gate equivalent). This removes T-gate cost
# entirely -- useful for fault-tolerant settings where T gates are
# expensive.


# %%
@dataclass
class CliffordOracleStrategy:
    """Clifford-only strategy: replace P(pi/4) with P(pi/2)."""

    @property
    def name(self) -> str:
        return "clifford_only"

    def decompose(self, qubits):
        n = len(qubits)
        qubits_list = list(qubits)
        for i in range(n):
            qubits_list[i] = qmc.h(qubits_list[i])
            qubits_list[i] = qmc.p(qubits_list[i], math.pi / 2)  # S-equivalent
            qubits_list[i] = qmc.h(qubits_list[i])
        for i in range(n - 1):
            qubits_list[i], qubits_list[i + 1] = qmc.cz(
                qubits_list[i], qubits_list[i + 1]
            )
        return tuple(qubits_list)

    def resources(self, num_qubits):
        n = num_qubits
        return ResourceMetadata(
            t_gate_count=0,  # No T gates!
            custom_metadata={
                "num_h_gates": 2 * n,
                "num_p_gates": n,
                "num_cz_gates": n - 1,
                "total_gates": 3 * n + (n - 1),
            },
        )


# Register the strategy
MarkingOracle.register_strategy("clifford_only", CliffordOracleStrategy())

# %% [markdown]
# ### 6.3 Comparing Strategy Resources

# %%
oracle = MarkingOracle(5)

std_res = oracle.get_resources_for_strategy()  # default: from _resources()
cliff_res = oracle.get_resources_for_strategy("clifford_only")

print("MarkingOracle (5 qubits) -- Strategy comparison:")
print()
print(f"  {'Metric':<20} {'Standard':>10} {'Clifford':>10}")
print(f"  {'-' * 40}")
print(f"  {'T gates':<20} {std_res.t_gate_count:>10} {cliff_res.t_gate_count:>10}")
print(
    f"  {'Total gates':<20} {std_res.custom_metadata['total_gates']:>10}"
    f" {cliff_res.custom_metadata['total_gates']:>10}"
)

# %% [markdown]
# The total gate count is the same, but the Clifford-only strategy
# eliminates all T gates -- a significant advantage for fault-tolerant
# quantum computing where T gates require costly magic state
# distillation.

# %% [markdown]
# ### 6.4 Using the Custom Gate in a Circuit


# %%
@qmc.qkernel
def algorithm_with_oracle() -> qmc.Vector[qmc.Qubit]:
    """A simple circuit using our custom composite gate."""
    q = qmc.qubit_array(4, name="q")
    # Apply Hadamard layer
    for i in qmc.range(4):
        q[i] = qmc.h(q[i])
    # Apply our custom oracle
    oracle_gate = MarkingOracle(4)
    q[0], q[1], q[2], q[3] = oracle_gate(q[0], q[1], q[2], q[3])
    return q


est_algo = estimate_resources(algorithm_with_oracle.block)
print("Algorithm with MarkingOracle:")
for section, values in est_algo.to_dict().items():
    if isinstance(values, dict):
        print(f"  {section}:")
        for key, val in values.items():
            print(f"    {key}: {val}")
    else:
        print(f"  {section}: {values}")

# %% [markdown]
# ### 6.5 Stub Gates for Black-Box Estimation
#
# Sometimes you need to estimate resources for an algorithm that uses
# a subroutine whose internal structure is unknown or irrelevant.
# Qamomile supports **stub gates** -- composite gates with resource
# metadata but no gate-level implementation.

# %%
from qamomile.circuit.frontend.composite_gate import composite_gate


@composite_gate(
    stub=True,
    name="black_box_oracle",
    num_qubits=3,
    t_gate_count=10,
    query_complexity=1,
)
def black_box_oracle():
    """A stub gate: resource metadata only, no implementation."""
    pass


# Inspect metadata
meta = black_box_oracle.get_resource_metadata()
print("Stub gate metadata:")
print(f"  T-gate count:      {meta.t_gate_count}")
print(f"  Query complexity:  {meta.query_complexity}")


# %%
# Use the stub gate in a circuit
@qmc.qkernel
def grover_iteration() -> qmc.Vector[qmc.Qubit]:
    """Simplified Grover iteration using a black-box oracle."""
    q = qmc.qubit_array(3, name="q")
    for i in qmc.range(3):
        q[i] = qmc.h(q[i])
    # Apply the black-box oracle (no implementation needed)
    q[0], q[1], q[2] = black_box_oracle(q[0], q[1], q[2])
    # Diffuser
    for i in qmc.range(3):
        q[i] = qmc.h(q[i])
        q[i] = qmc.x(q[i])
    q[0], q[1] = qmc.cz(q[0], q[1])
    q[1], q[2] = qmc.cz(q[1], q[2])
    for i in qmc.range(3):
        q[i] = qmc.x(q[i])
        q[i] = qmc.h(q[i])
    return q


est_grover = estimate_resources(grover_iteration.block)
print("Grover iteration with stub oracle:")
for section, values in est_grover.to_dict().items():
    if isinstance(values, dict):
        print(f"  {section}:")
        for key, val in values.items():
            print(f"    {key}: {val}")
    else:
        print(f"  {section}: {values}")

# %% [markdown]
# The estimator picks up the stub's T-gate count (10) and adds it to
# the surrounding gates. This allows you to estimate total algorithm
# cost even when the oracle internals are not available.
#
# Note that **query complexity** does not appear in the
# `estimate_resources()` output -- it is metadata on the composite
# gate itself, not a circuit-level metric. To access it, use
# `get_resource_metadata()` on the gate directly:

# %%
meta = black_box_oracle.get_resource_metadata()
print(f"Oracle query complexity: {meta.query_complexity}")

# %% [markdown]
# **Next steps:**
#
# - [Transpile](09_transpile.ipynb) -- learn about the transpilation pipeline
# - [Custom Executor](10_custom_executor.ipynb) -- run circuits on cloud quantum hardware
# - [QAOA](../optimization/qaoa.ipynb) -- optimization with variational circuits

# %% [markdown]
# ## What We Learned
#
# - **Using `estimate_resources()` to obtain qubit counts, gate counts, and circuit depth as SymPy expressions** -- `estimate_resources(kernel.block)` walks the circuit IR and returns a `ResourceEstimate` with all metrics as symbolic expressions.
# - **Iterating over all available metrics with `.to_dict()`** -- The nested dictionary is forward-compatible and suitable for serialisation; new fields added in future releases appear automatically.
# - **Working with symbolic (parametric) circuits and using `.substitute()` and `.simplify()`** -- `.substitute()` plugs in concrete values; `.simplify()` reduces expression complexity via SymPy.
# - **Analyzing the resource profile of Quantum Phase Estimation (QPE)** -- Both a manual implementation (IQFT + controlled unitaries) and the built-in `qmc.qpe()` primitive give the same asymptotic scaling.
# - **Comparing decomposition strategies for composite gates** -- QFT's standard vs approximate strategies demonstrate how truncating small-angle rotations trades accuracy for gate count savings.
# - **Defining custom composite gates with strategies, and using stub gates for black-box resource estimation** -- Custom `CompositeGate` classes support multiple strategies; stub gates provide resource metadata without gate-level implementation, enabling algorithm-level cost analysis.

# %%
