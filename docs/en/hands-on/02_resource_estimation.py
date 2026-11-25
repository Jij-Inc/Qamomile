# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# tags: [tutorial]
# ---
#
# # Resource Estimation
#
# When designing a quantum algorithm, it is important to know what resources it will require before running it, such as the number of qubits and gates and the circuit depth. An approximate resource estimate helps you understand what problem sizes current quantum computers can handle with the algorithm. It also lets you use current roadmaps to estimate when particular problem sizes might become feasible and compare the resource requirements of algorithms that solve the same problem. [Beverland et al.](https://doi.org/10.48550/arXiv.2211.07629)
#
# This notebook covers the basics of Qamomile's symbolic resource estimation. Calling `resources.logical()` with a qkernel estimates the algorithm-level resources required from the structure of the qkernel, without executing the circuit. Here, algorithm-level resources **do not include** the physical qubits or gates needed to run the quantum algorithm on a particular quantum computer, or the resources required for error correction.
#
# We use the [Greenberger–Horne–Zeilinger (GHZ) state](https://doi.org/10.1007/978-94-017-0849-4_10)
#
# $$
# |\mathrm{GHZ}_n\rangle
# =\frac{|0\rangle^{\otimes n}+|1\rangle^{\otimes n}}{\sqrt2}
# $$
#
# as a common example to learn how to:
#
# - Pass a qkernel to `resources.logical()` to perform resource estimation.
# - Estimate and compare the resources required by two different implementations.
# - Estimate the resources of a qkernel containing an opaque oracle with a fixed cost.
# - Define a symbolic cost rule and estimate the resources of a qkernel containing an oracle whose cost depends on its input.

# %%
# Install Qamomile and the packages used in this notebook.
# # !pip install qamomile matplotlib

# %%
from __future__ import annotations

import matplotlib.pyplot as plt

import qamomile as qm
from qamomile import resources

# %% [markdown]
# ## 1. Symbolic Resource Estimation for a GHZ State
#
# We prepare an `n`-qubit GHZ state. We start with a simple implementation that applies `CX` gates in a chain. [Cruz et al.](https://doi.org/10.1002/qute.201900015)


# %%
@qm.qkernel
def linear_ghz(n: qm.Const[qm.Int]) -> qm.Q[qm.Bits[n]]:  # noqa: F821
    """Prepare a GHZ state with a nearest-neighbor CX chain."""
    q = qm.qubits(n)
    q = qm.h(q, target=0)
    for index in range(n - 1):
        q = qm.cx(q, control=index, target=index + 1)
    return q


# %% [markdown]
# Leave the qubit count unspecified to get a formula in terms of `n`. Both GHZ implementations allocate their qubits once and keep them until they return, so the circuit width is also the peak qubit count.

# %%
linear_symbolic = resources.logical(linear_ghz)


circuit_qubits = resources.LOGICAL_CIRCUIT_QUBITS
peak_qubits = resources.LOGICAL_PEAK_QUBITS

ghz_metrics = {
    "#single-qubit gates": resources.LOGICAL_SINGLE_QUBIT_GATES,
    "#two-qubit gates": resources.LOGICAL_TWO_QUBIT_GATES,
    "#total gates": resources.LOGICAL_GATES,
    "gate depth": resources.LOGICAL_GATE_DEPTH,
}

print("=== Resource required ===")
print("#qubits:", linear_symbolic.metric(circuit_qubits).expression)
for label, metric in ghz_metrics.items():
    print(f"{label}: {linear_symbolic.metric(metric)}")

# %% [markdown]
# The circuit uses $n$ qubits. The symbolic estimate gives $n$ total gates, including the Hadamard gate, and $n-1$ CX gates. Since this qkernel applies a Hadamard gate to qubit 0, the formulas are valid for $n\geq1$.

# %% [markdown]
# ## 2. Comparing Different Implementations
#
# The example above prepares a GHZ state using a straightforward method. Here, we prepare a GHZ state using a different implementation and compare the resources required by each implementation. In the following implementation, we pass the number of qubits `n` and the number of stages `m` separately, with $n=2^m$. [Cruz et al.](https://doi.org/10.1002/qute.201900015)


# %%
@qm.qkernel
def tree_ghz(m: qm.Const[qm.Int], n: qm.Const[qm.Int]) -> qm.Q[qm.Bits[n]]:  # noqa: F821
    """Prepare a GHZ state using m stages and n = 2**m qubits."""
    q = qm.qubits(n)
    q = qm.h(q, target=0)

    for stage in range(m):
        width = 2**stage
        for control in range(width):
            target = width + control
            q = qm.cx(q, control=control, target=target)
    return q


# %%
tree_symbolic = resources.logical(tree_ghz)

print("=== Resource required ===")
print("#qubits:", tree_symbolic.metric(circuit_qubits).expression)
for label, metric in ghz_metrics.items():
    print(f"{label}: {tree_symbolic.metric(metric)}")

# %% [markdown]
# We compare the resource requirements using the formulas obtained from both implementations. Here, we look at the number of quantum gates and the depth for the same number of qubits. Use `study.evaluate(...)` to substitute values without analyzing the qkernel again. The tree's symbolic depth is bounded because the dependencies between gates depend on the register indices. To get an exact depth, specialize the tree kernel for each size and analyze the specialized program.

# %%
m_values = list(range(1, 9))
n_values = [2**m_value for m_value in m_values]

linear_resources_list = [linear_symbolic.evaluate(n=n) for n in n_values]
tree_resources_list = [
    tree_symbolic.evaluate(m=m, n=n) for m, n in zip(m_values, n_values, strict=True)
]
tree_specialized = [
    resources.logical(qm.specialize(tree_ghz, m=m, n=n))
    for m, n in zip(m_values, n_values, strict=True)
]

linear_gate_counts = [
    study.metric(resources.LOGICAL_GATES).numeric_enclosure()
    for study in linear_resources_list
]
tree_gate_counts = [
    study.metric(resources.LOGICAL_GATES).numeric_enclosure()
    for study in tree_resources_list
]
linear_gate_depths = [
    study.metric(resources.LOGICAL_GATE_DEPTH).numeric_enclosure()
    for study in linear_resources_list
]
tree_gate_depths = [
    study.metric(resources.LOGICAL_GATE_DEPTH).numeric_enclosure()
    for study in tree_specialized
]

for n, linear_study, tree_study in zip(
    n_values, linear_resources_list, tree_resources_list, strict=True
):
    assert linear_study.metric(peak_qubits).numeric_enclosure() == (n, n)
    assert tree_study.metric(peak_qubits).numeric_enclosure() == (n, n)
assert linear_gate_counts == tree_gate_counts == [(n, n) for n in n_values]
assert linear_gate_depths == [(n, n) for n in n_values]
assert tree_gate_depths == [(m + 1, m + 1) for m in m_values]

# Plot gate counts on the first axis.
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
for values, label, marker, linestyle in [
    (linear_gate_counts, "Linear GHZ", "o", "-"),
    (tree_gate_counts, "Tree GHZ", "x", "--"),
]:
    axes[0].plot(
        n_values,
        [lower for lower, _ in values],
        marker=marker,
        linestyle=linestyle,
        label=label,
    )
# Plot circuit depths on the second axis.
for values, label, marker, linestyle in [
    (linear_gate_depths, "Linear GHZ", "o", "-"),
    (tree_gate_depths, "Tree GHZ", "x", "--"),
]:
    axes[1].plot(
        n_values,
        [lower for lower, _ in values],
        marker=marker,
        linestyle=linestyle,
        label=label,
    )
# Set the labels.
axes[0].set(
    xlabel=r"Number of qubits ($n=2^m$)",
    ylabel="Total gates",
    title="Total gate count",
)
axes[1].set(
    xlabel=r"Number of qubits ($n=2^m$)",
    ylabel="Depth",
    title="Depth",
)
# Configure the plots.
for axis in axes:
    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=2)
    axis.set_xticks(n_values)
    axis.set_xticklabels([str(value) for value in n_values])
    axis.grid(alpha=0.25)
    axis.legend()
# Show the plots.
plt.tight_layout()
plt.show()

# %% [markdown]
# The total gate counts are the same, so the curves overlap. In contrast, the depth is $n$ for the straightforward implementation and $\log_2 n+1$ for the other implementation, so the difference grows as the number of qubits increases. [Cruz et al.](https://doi.org/10.1002/qute.201900015)

# %% [markdown]
# ## 3. Resource Estimation for a qkernel Containing an Oracle with a Fixed Cost
#
# Here, we perform resource estimation using an oracle. In Qamomile, a qkernel whose body is `...` declares an operation without an implementation. This lets users estimate the overall resources without fully implementing, for example, problem-specific operations or subroutines with several possible implementations. We first give the oracle a fixed cost using `ResourceRule` and estimate the resources of a qkernel that uses it.


# %%
@qm.qkernel(kind="unitary")
def fixed_state_preparation(q: qm.Q[qm.Bits[4]]) -> qm.Q[qm.Bits[4]]: ...


fixed_linear_ghz_cost = tuple(
    resources.ResourceRule(
        identity=f"hands-on.fixed-ghz.{metric.identifier}",
        callable=fixed_state_preparation,
        metric=metric,
        relation=resources.Relation.EQUALITY,
        expression=resources.literal(cost),
        provenance="hands-on.linear-ghz-cost",
        scheduling_footprint=(
            resources.SchedulingFootprint(("q",))
            if metric == resources.LOGICAL_GATE_DEPTH
            else None
        ),
    )
    for metric, cost in (
        (resources.LOGICAL_GATES, 4),
        (resources.LOGICAL_SINGLE_QUBIT_GATES, 1),
        (resources.LOGICAL_TWO_QUBIT_GATES, 3),
        (resources.LOGICAL_GATE_DEPTH, 4),
    )
)


# %%
@qm.qkernel
def fixed_cost_algorithm(theta: qm.Param[qm.Real]) -> qm.Shot[qm.Bits[4]]:
    """Apply a fixed-cost state-preparation oracle followed by an RZ gate."""
    q = qm.qubits(4)
    q = fixed_state_preparation(q)
    q = qm.rz(q, theta, target=3)
    return qm.measure(q)


# %%
fixed_resources = resources.logical(fixed_cost_algorithm, rules=fixed_linear_ghz_cost)
fixed_calls = resources.algorithmic_call_count(fixed_state_preparation.symbol)

print("=== Resource required ===")
print("#qubits:", fixed_resources.metric(peak_qubits).expression)
for label, metric in ghz_metrics.items():
    print(f"{label}: {fixed_resources.metric(metric)}")
print(f"#measurements: {fixed_resources.metric(resources.LOGICAL_MEASUREMENTS)}")
print(f"Oracle calls: {fixed_resources.metric(fixed_calls)}")

assert fixed_resources.metric(resources.LOGICAL_GATES).numeric_enclosure() == (5, 5)
assert fixed_resources.metric(resources.LOGICAL_MEASUREMENTS).numeric_enclosure() == (
    4,
    4,
)

# %% [markdown]
# The resource estimate above includes the fixed cost assigned to the oracle. The results depend on the supplied rules because the oracle has no implementation. The depth rule states that the oracle occupies all four qubits for four gate layers. The RZ gate acts on one of those qubits and starts after those layers.

# %% [markdown]
# ## 4. Specifying Input-Dependent Costs with a Rule
#
# In the example above, we defined a fixed cost for the oracle. In some cases, such as the state preparation used here, we may want to specify an oracle cost that depends on the input size rather than a fixed resource cost. Qamomile supports this through a symbolic expression in a `ResourceRule`. Use `rule_parameter` to reference the number of input qubits `n` in the expression. Below, we construct an oracle with an input-dependent cost and estimate the resources of a qkernel containing this oracle.


# %%
@qm.qkernel(kind="unitary")
def parameterized_state_preparation(
    q: qm.Q[qm.Bits[n]],  # noqa: F821
    n: qm.Int,
) -> qm.Q[qm.Bits[n]]: ...  # noqa: F821


oracle_width = resources.rule_parameter(
    parameterized_state_preparation,
    "n",
    domain=resources.Domain.NATURAL,
    constraints=("n >= 1",),
)
parameterized_linear_ghz_cost = tuple(
    resources.ResourceRule(
        identity=f"hands-on.parameterized-ghz.{metric.identifier}",
        callable=parameterized_state_preparation,
        metric=metric,
        relation=resources.Relation.EQUALITY,
        expression=cost,
        provenance="hands-on.linear-ghz-cost",
        scheduling_footprint=(
            resources.SchedulingFootprint(("q",))
            if metric == resources.LOGICAL_GATE_DEPTH
            else None
        ),
    )
    for metric, cost in (
        (resources.LOGICAL_GATES, oracle_width),
        (resources.LOGICAL_SINGLE_QUBIT_GATES, resources.literal(1)),
        (resources.LOGICAL_GATE_DEPTH, oracle_width),
    )
)


# %%
@qm.qkernel
def parameterized_algorithm(
    n: qm.Const[qm.Int], theta: qm.Param[qm.Real]
) -> qm.Shot[qm.Bits[n]]:  # noqa: F821
    """Estimate an algorithm with an input-dependent state-preparation cost."""
    q = qm.qubits(n)
    q = parameterized_state_preparation(q, n)
    q = qm.rz(q, theta, target=n - 1)
    return qm.measure(q)


# %%
parameterized_resources = resources.logical(
    parameterized_algorithm, rules=parameterized_linear_ghz_cost
)
parameterized_calls = resources.algorithmic_call_count(
    parameterized_state_preparation.symbol
)

print("=== Resource required ===")
print("#qubits:", oracle_width)
for label, metric in ghz_metrics.items():
    print(f"{label}: {parameterized_resources.metric(metric)}")
print(f"Oracle calls: {parameterized_resources.metric(parameterized_calls)}")

for width in (4, 8):
    estimate = parameterized_resources.evaluate(n=width)
    assert estimate.metric(peak_qubits).numeric_enclosure() == (width, width)
    assert estimate.metric(resources.LOGICAL_GATES).numeric_enclosure() == (
        width + 1,
        width + 1,
    )
    assert estimate.metric(parameterized_calls).numeric_enclosure() == (1, 1)

# %% [markdown]
# The model uses the $n$ qubits passed to the oracle. The resource estimate above includes the cost assigned to the oracle, which depends on that input width. As in the fixed-cost example, the RZ gate is applied after the oracle to a qubit used by the oracle. We supply rules for total gates, single-qubit gates, and depth. The analysis reports bounds for the two-qubit gate count.

# %% [markdown]
# ## Summary
#
# We passed qkernels to `resources.logical` to perform symbolic resource estimation without specifying concrete input values. We then substituted concrete values into the gate-count formulas obtained from two different implementations and specialized the tree kernel to get its exact gate depth. Finally, we defined fixed and parameterized costs for oracles and examined the resource estimates of qkernels that use them, confirming that the costs specified for the oracles are reflected in the results.
