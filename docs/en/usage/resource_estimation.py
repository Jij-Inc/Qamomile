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
# tags: [usage, resource-estimation]
# ---
#
# # Resource Estimation in Detail
#
# This guide explains individual resource metrics, illustrates how they are counted, and shows how to save estimates and interpret their assumptions. For the basic workflow, symbolic estimates, and controlled-operation models, see [Resource Estimation](../tutorial/04_resource_estimation.ipynb).
#
# All examples below include their own definitions and can be run in order without running the tutorial first.

# %% [markdown]
# ## Setup
#
# Install Qamomile if needed, then import it. The estimates are obtained by analyzing kernels without executing them. The later persistence example uses Qamomile's built-in simulator to run a restored study.

# %%
# Install the latest Qamomile through pip.
# # !pip install qamomile

# %%
from __future__ import annotations

import qamomile as qm

# %% [markdown]
# ## A Study to Inspect
#
# We start with a kernel that prepares a Bell pair: two entangled qubits. It applies a Hadamard (H) gate followed by a controlled-X (CX) gate, then measures both qubits. `qm.resources.logical(...)` analyzes the kernel and returns a `ResourceStudy`.


# %%
@qm.qkernel
def bell_pair() -> qm.Shot[qm.Bits[2]]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.cx(state, control=0, target=1)
    return qm.measure(state)


# %%
bell_study = qm.resources.logical(bell_pair)

# %% [markdown]
# ## Accessing Results in a `ResourceStudy`
#
# A `ResourceStudy` contains the estimates produced by the analysis. Its `metric(...)` method retrieves the result for a particular metric as a `ResourceValue`; it does not perform the analysis again.

# %%
gate_estimate = bell_study.metric(qm.resources.LOGICAL_GATES)
assert gate_estimate.numeric_enclosure() == (2, 2)

# %% [markdown]
# Here, `gate_estimate` contains the gate-count estimate together with information about its assumptions and how it was derived. Calling `gate_estimate.numeric_enclosure()` returns its numerical bounds—in this example, `(2, 2)`.
#
# When you supply a `ResourceRule`, you provide information used in the analysis. The resulting `ResourceValue` retains the relevant assumptions and evidence so that you can see what the estimate depends on.
#
# A `ResourceValue` provides the following information:
#
# - `numeric_enclosure()` returns numerical lower and upper bounds, using `None` when a numerical bound is unavailable.
# - `relation` indicates whether the result is an equality, a bound, an enclosure, or an incomplete estimate.
# - `assumptions` records conditions on which the estimate depends, including conditions supplied with a resource rule or introduced by a model.
# - `evidence` records how the estimate was obtained, such as analysis of the kernel or a supplied cost assumption.
# - `unknown_contributors` identifies unresolved contributions to the estimate.
#
# To see which metrics are present, inspect `study.metrics`, which contains pairs of metric identifiers and `ResourceValue` instances. The available metrics depend on the analysis and any additional models applied. Requesting a metric that is absent raises a `KeyError`; an absent metric does not imply a value of zero.
#
# The following methods help explain individual estimates:
#
# - `study.contributions(metric)` lists the contributing operations, their estimated costs, and their source locations.
# - `study.hotspots(metric)` ranks contributions by numerical upper bound, with the largest first. Contributions without a numerical upper bound appear afterward.
# - `study.explain(metric)` returns a text explanation of the estimate and its contributions.
#
# A study can also be saved with `to_bytes()` and restored with `ResourceStudy.from_bytes(...)`. A later example demonstrates this.

# %% [markdown]
# ## Qubit Counts
#
# Qamomile distinguishes the qubits supplied to a kernel, those it allocates, and how many are in use at the same time:
#
# - `LOGICAL_INPUT_QUBITS`: the number of qubits passed to the kernel through its arguments.
# - `LOGICAL_ALLOCATED_QUBITS`: the number of qubits allocated within the analyzed kernel, including allocations in kernels it calls.
# - `LOGICAL_PEAK_QUBITS`: the maximum number of logical qubits in use at the same time.
# - `LOGICAL_CIRCUIT_QUBITS`: the static circuit width, computed as the input-qubit count plus the allocated-qubit count. This count does not decrease when a qubit’s lifetime ends.
#
# The peak count can therefore be smaller than the static circuit width: a kernel may finish using some qubits before allocating others. The examples in “Worked Metric Examples” illustrate this distinction.
#
# Ancillas explicitly allocated in a kernel are included in these logical metrics. Additional ancillas introduced by `clean_ancilla_controls(...)` are reported in `CIRCUIT_CLEAN_ANCILLA_QUBITS` and included in the model’s total qubit count, `CIRCUIT_VIRTUAL_QUBITS`.

# %% [markdown]
# ## Gate Counts
#
# `LOGICAL_GATES` gives the total logical gate count. Measurements and resets are counted separately.
#
# The following metrics group gates by the number of qubits they act on:
#
# - `LOGICAL_SINGLE_QUBIT_GATES`: gates acting on one qubit.
# - `LOGICAL_TWO_QUBIT_GATES`: gates acting on two qubits.
# - `LOGICAL_MULTI_QUBIT_GATES`: gates acting on three or more qubits.
#
# :::{note}
# Clifford gates are unitary operations that can be built from H, S, and CX gates, up to an overall phase. The S gate leaves $|0\rangle$ unchanged and maps $|1\rangle$ to $i|1\rangle$.
#
# Non-Clifford gates are also unitary, but cannot be built from those gates alone. One example is T, which leaves $|0\rangle$ unchanged and maps $|1\rangle$ to $e^{i\pi/4}|1\rangle$.
# :::
#
# Qamomile also groups gates by operation type:
#
# - `LOGICAL_CLIFFORD_GATES`: Clifford gates, such as H and CX.
# - `LOGICAL_ROTATION_GATES`: parameterized rotation and phase gates, such as RX, RY, and RZ.
# - `LOGICAL_T_GATES`: T gates and their adjoints.
# - `LOGICAL_TOFFOLI_GATES`: Toffoli (CCX) gates.
# - `LOGICAL_NON_CLIFFORD_GATES`: operations classified as non-Clifford by the logical analysis, including T, Toffoli, and parameterized rotation gates.
#
# Qamomile's resource analysis counts rotation gates such as RX, RY, and RZ in `LOGICAL_NON_CLIFFORD_GATES` regardless of their angles. Some angles make these gates Clifford operations, but this count does not check for those cases.
#
# These groups overlap. For example, a T gate contributes to both `LOGICAL_T_GATES` and `LOGICAL_NON_CLIFFORD_GATES`, so the gate-family counts should not all be added together.
#
# To count a specific operation, use `qm.resources.logical_operation_count(...)` with its internal operation name. For example, `"quantum.h"` identifies the H-gate operation applied by `qm.h`:

# %%
h_estimate = bell_study.metric(qm.resources.logical_operation_count("quantum.h"))

# %% [markdown]
# `h_estimate.numeric_enclosure()` returns `(1, 1)`, because `bell_pair` contains one H gate.

# %% [markdown]
# ## Measurements and Resets
#
# These metrics count measurements and resets of individual qubits:
#
# - `LOGICAL_MEASUREMENTS`: the number of individual qubit measurements.
# - `LOGICAL_RESETS`: the number of individual qubit resets to $|0\rangle$.
#
# Each measured or reset qubit contributes one to the corresponding count, even when a single function call operates on an entire register.
#
# The counts describe one execution of the kernel. Increasing `shots` does not multiply the values reported by the resource analysis.

# %% [markdown]
# ## Depth
#
# These metrics measure depth in dependency steps:
#
# - `LOGICAL_SPAN`: the overall dependency depth across operations included in the logical analysis.
# - `LOGICAL_GATE_DEPTH`: depth counting only gates.
# - `LOGICAL_CLIFFORD_DEPTH`: the number of sequential layers counting only Clifford gates.
# - `LOGICAL_NON_CLIFFORD_DEPTH`: the number of sequential layers counting operations classified as non-Clifford by the logical analysis.
# - `LOGICAL_ROTATION_DEPTH`, `LOGICAL_T_DEPTH`, and `LOGICAL_TOFFOLI_DEPTH`: depth counting only gates in the corresponding family.
# - `LOGICAL_MEASUREMENT_DEPTH` and `LOGICAL_RESET_DEPTH`: depth counting only measurements or resets, respectively.
#
# For a category-specific metric, operations outside that category contribute no depth themselves, but their dependencies are still respected. For example, two Clifford gates separated by a rotation on the same qubit contribute two layers to the Clifford depth.

# %% [markdown]
# ## Kernel Calls and Oracle Queries
#
# The following `oracle` is an opaque unitary kernel: its body is `...`, so its gate sequence is not available for analysis. `use_oracle` calls it `rounds` times, then measures the qubits.


# %%
@qm.qkernel(kind="unitary")
def oracle(state: qm.Q[qm.Bits[2]], rounds: qm.Int) -> qm.Q[qm.Bits[2]]: ...


@qm.qkernel
def use_oracle(rounds: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[2]]:
    state = qm.qubits(2)
    for _ in range(rounds):
        state = oracle(state, rounds)
    return qm.measure(state)


# %% [markdown]
# For this example, we assume that each oracle call uses `2 * rounds` logical gates. `rule_parameter(...)` represents the oracle's `rounds` argument in the rule expression. `Relation.EQUALITY` states that the gate count equals that expression under our assumption; Qamomile does not verify it against an implementation.
#
# We specialize `use_oracle` with `rounds=2` and analyze it with this rule. The two calls each use four gates under the assumption, giving eight in total. The rule also records the condition `rounds >= 0` with the result.

# %%
oracle_rounds = qm.resources.rule_parameter(
    oracle,
    "rounds",
    domain=qm.resources.Domain.NATURAL,
    constraints=("rounds >= 0",),
)
oracle_gates = qm.resources.ResourceRule(
    identity="tutorial.oracle.gates.v1",
    callable=oracle,
    metric=qm.resources.LOGICAL_GATES,
    relation=qm.resources.Relation.EQUALITY,
    expression=qm.resources.multiply(qm.resources.literal(2), oracle_rounds),
    assumptions=("rounds >= 0",),
    validity_domain=("rounds is an integer",),
    provenance="tutorial.oracle.contract.v1",
)

# %%
oracle_program = qm.specialize(use_oracle, rounds=2)
oracle_study = qm.resources.logical(
    oracle_program,
    rules=oracle_gates,
    strict=True,
)

# %% [markdown]
#
# These metrics count how often a kernel is called or an oracle is queried:
#
# - `qm.resources.algorithmic_call_count(oracle.symbol)` defines a metric for calls to `oracle`.
# - `qm.resources.algorithmic_query_count(oracle.symbol)` defines a metric for queries to that opaque quantum kernel.
#
# Here, `oracle.symbol` identifies the kernel. Pass the resulting metric to `oracle_study.metric(...)` to retrieve its estimate.
#
# With `rounds=2`, each call to `oracle` counts as one query, so both estimates have bounds of `(2, 2)`. The number of gates required by those calls is reported separately by `LOGICAL_GATES`.

# %%
oracle_calls = qm.resources.algorithmic_call_count(oracle.symbol)
oracle_queries = qm.resources.algorithmic_query_count(oracle.symbol)

print(
    "logical gates:",
    oracle_study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure(),
)
print("oracle calls:", oracle_study.metric(oracle_calls).numeric_enclosure())

assert oracle_study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (8, 8)
assert oracle_study.metric(oracle_calls).numeric_enclosure() == (2, 2)
assert oracle_study.metric(oracle_queries).numeric_enclosure() == (2, 2)
assert qm.resources.EvidenceAuthority.TRUSTED_AXIOM in {
    item.authority for item in oracle_study.metric(qm.resources.LOGICAL_GATES).evidence
}

# %% [markdown]
# ## Worked Metric Examples
#
# The following examples illustrate the differences between qubit counts, operation counts, and depth. The kernel below prepares a GHZ (Greenberger–Horne–Zeilinger) state: an equal superposition of the all-zero and all-one states. We use `width=4`, then examine two small kernels that highlight how these metrics behave.
#


# %%
@qm.qkernel
def ghz_state(
    width: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[width]]:  # noqa: F821
    state = qm.qubits(width)
    state = qm.h(state, target=0)
    for index in range(width - 1):
        state = qm.cx(state, control=index, target=index + 1)
    return qm.measure(state)


# %% [markdown]
# ### Qubit Counts
#
# We first compare the four qubit-count metrics for `ghz_state` with `width=4`.

# %%
ghz_program = qm.specialize(ghz_state, width=4)
ghz_study = qm.resources.logical(ghz_program)
input_qubits = ghz_study.metric(qm.resources.LOGICAL_INPUT_QUBITS)
allocated_qubits = ghz_study.metric(qm.resources.LOGICAL_ALLOCATED_QUBITS)
ghz_peak = ghz_study.metric(qm.resources.LOGICAL_PEAK_QUBITS)
circuit_qubits = ghz_study.metric(qm.resources.LOGICAL_CIRCUIT_QUBITS)

print("input quantum bits:", input_qubits.numeric_enclosure())
print("allocated quantum bits:", allocated_qubits.numeric_enclosure())
print("peak quantum bits:", ghz_peak.numeric_enclosure())
print("static circuit width:", circuit_qubits.numeric_enclosure())

assert input_qubits.numeric_enclosure() == (0, 0)
assert allocated_qubits.numeric_enclosure() == (4, 4)
assert ghz_peak.numeric_enclosure() == (4, 4)
assert circuit_qubits.numeric_enclosure() == (4, 4)

# %% [markdown]
# The kernel receives no qubits as arguments and allocates four qubits internally. All four are in use at the same time, so the allocated-qubit count, peak count, and static circuit width are all four.
#
# These values need not be equal. The next kernel measures one qubit before allocating another:


# %%
@qm.qkernel
def released_qubit_example() -> tuple[qm.Shot[qm.Bits[1]], qm.Q[qm.Bits[1]]]:
    first = qm.qubits(1)
    measured = qm.measure(first)
    second = qm.qubits(1)
    return measured, second


# %%
released_study = qm.resources.logical(released_qubit_example)
released_peak = released_study.metric(qm.resources.LOGICAL_PEAK_QUBITS)
released_allocated = released_study.metric(qm.resources.LOGICAL_ALLOCATED_QUBITS)
released_circuit = released_study.metric(qm.resources.LOGICAL_CIRCUIT_QUBITS)

print("peak quantum bits:", released_peak.numeric_enclosure())
print("allocated quantum bits:", released_allocated.numeric_enclosure())
print("static circuit width:", released_circuit.numeric_enclosure())

assert released_peak.numeric_enclosure() == (1, 1)
assert released_allocated.numeric_enclosure() == (2, 2)
assert released_circuit.numeric_enclosure() == (2, 2)

# %% [markdown]
# The kernel allocates two qubits in total, but their lifetimes do not overlap. The peak count is therefore one, while the allocated-qubit count and static circuit width are both two.
#
# ### Gate, Measurement, and Reset Counts
#
# We now examine the gates in the same four-qubit `ghz_state`, grouping them by the number of qubits they act on.

# %%
ghz_gates = ghz_study.metric(qm.resources.LOGICAL_GATES)
single_qubit_gates = ghz_study.metric(qm.resources.LOGICAL_SINGLE_QUBIT_GATES)
two_qubit_gates = ghz_study.metric(qm.resources.LOGICAL_TWO_QUBIT_GATES)
multi_qubit_gates = ghz_study.metric(qm.resources.LOGICAL_MULTI_QUBIT_GATES)
ghz_measurements = ghz_study.metric(qm.resources.LOGICAL_MEASUREMENTS)
ghz_resets = ghz_study.metric(qm.resources.LOGICAL_RESETS)

print("total gates:", ghz_gates.numeric_enclosure())
print("single-quantum-bit gates:", single_qubit_gates.numeric_enclosure())
print("two-quantum-bit gates:", two_qubit_gates.numeric_enclosure())
print("multi-quantum-bit gates:", multi_qubit_gates.numeric_enclosure())

assert ghz_gates.numeric_enclosure() == (4, 4)
assert single_qubit_gates.numeric_enclosure() == (1, 1)
assert two_qubit_gates.numeric_enclosure() == (3, 3)
assert multi_qubit_gates.numeric_enclosure() == (0, 0)

# %% [markdown]
# The kernel applies one single-qubit Hadamard gate and three two-qubit CX gates. No gate acts on three or more qubits, so the counts are one, three, and zero, respectively, giving four gates in total.
#
# Measurements and resets have their own counts:

# %%
print("measurements:", ghz_measurements.numeric_enclosure())
print("resets:", ghz_resets.numeric_enclosure())

assert ghz_measurements.numeric_enclosure() == (4, 4)
assert ghz_resets.numeric_enclosure() == (0, 0)

# %% [markdown]
# The final `qm.measure(state)` measures all four qubits, giving a measurement count of four. The reset count is zero because the kernel performs no resets. These counts describe one execution of the kernel and do not change with `shots`.
#
# ### Depth
#
# Gate count tells us how many gates are applied. Gate depth tells us how many sequential gate layers are needed while respecting dependencies. We can also examine measurement and reset depth separately.

# %%
ghz_gate_depth = ghz_study.metric(qm.resources.LOGICAL_GATE_DEPTH)
measurement_depth = ghz_study.metric(qm.resources.LOGICAL_MEASUREMENT_DEPTH)
reset_depth = ghz_study.metric(qm.resources.LOGICAL_RESET_DEPTH)

print("gate depth:", ghz_gate_depth.numeric_enclosure())
print("measurement depth:", measurement_depth.numeric_enclosure())
print("reset depth:", reset_depth.numeric_enclosure())

assert ghz_gate_depth.numeric_enclosure() == (4, 4)
assert measurement_depth.numeric_enclosure() == (1, 1)
assert reset_depth.numeric_enclosure() == (0, 0)

# %% [markdown]
# In this example, the Hadamard gate and three CX gates form a chain. The first CX gate depends on the Hadamard gate, and each later CX gate uses the preceding gate’s target qubit as its control. The four gates therefore occupy four successive layers.
#
# The four measurements act on separate qubits and fit in one measurement layer. There are no resets, so the reset depth is zero.

# %% [markdown]
# ### Category-Specific Depth
#
# Clifford depth counts the sequential layers of Clifford gates. Clifford gates that can run in parallel count as one layer. A non-Clifford gate between two Clifford gates can require them to occupy separate layers, even though the non-Clifford gate itself is not counted.
#
# RZZ is a two-qubit rotation that changes the relative phase between basis states with matching bits and those with different bits. For an angle $\theta$, it multiplies $|00\rangle$ and $|11\rangle$ by $e^{-i\theta/2}$, and $|01\rangle$ and $|10\rangle$ by $e^{i\theta/2}$.
#
# The following kernel applies an H gate to qubit 0, an RZZ rotation with angle $0.25$ radians to both qubits, and another H gate to qubit 1:


# %%
@qm.qkernel
def category_depth_example() -> qm.Q[qm.Bits[2]]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.rzz(state, 0.25, left=0, right=1)
    state = qm.h(state, target=1)
    return state


# %%
category_study = qm.resources.logical(category_depth_example)
category_gate_depth = category_study.metric(qm.resources.LOGICAL_GATE_DEPTH)
category_clifford_depth = category_study.metric(qm.resources.LOGICAL_CLIFFORD_DEPTH)
category_rotation_depth = category_study.metric(qm.resources.LOGICAL_ROTATION_DEPTH)
category_non_clifford_depth = category_study.metric(
    qm.resources.LOGICAL_NON_CLIFFORD_DEPTH
)

assert category_gate_depth.numeric_enclosure() == (3, 3)
assert category_clifford_depth.numeric_enclosure() == (2, 2)
assert category_rotation_depth.numeric_enclosure() == (1, 1)
assert category_non_clifford_depth.numeric_enclosure() == (1, 1)

# %% [markdown]
# The RZZ gate depends on the first H gate, and the second H gate depends on the RZZ gate. The two H gates therefore belong to separate Clifford layers, even though they act on different qubits.
#
# The resulting depths are:
#
# - **Gate depth: 3**, counting all three gates.
# - **Clifford depth: 2**, counting the two H gates while preserving their dependency through RZZ.
# - **Rotation depth: 1**, counting the RZZ gate.
# - **Non-Clifford depth: 1**, also counting the RZZ gate.
#
# The rotation and non-Clifford categories overlap here: both include RZZ. Adding all the category depths would therefore count that gate twice.

# %% [markdown]
# ## Saving and Loading a `ResourceStudy`

# %% [markdown]
# You can convert a `ResourceStudy` to bytes with `to_bytes()` and restore it with `ResourceStudy.from_bytes(...)`. The serialized data includes the resource estimates, their assumptions and evidence, and the program representation retained by the study.
#
# These bytes can be saved to a file for later use. The example below restores them immediately in memory, then compiles and runs the restored study. For this workflow, we first specialize `bell_pair` and analyze the resulting `Program`.

# %%
bell_program = qm.specialize(bell_pair)
executable_bell_study = qm.resources.logical(bell_program)
restored_bell_study = qm.resources.ResourceStudy.from_bytes(
    executable_bell_study.to_bytes()
)
bell_artifact = qm.compile(restored_bell_study, to=qm.Reference())
bell_record = qm.run(bell_artifact, on=qm.Reference(), shots=4, seed=7)

assert restored_bell_study == executable_bell_study
assert restored_bell_study.digest == executable_bell_study.digest
assert sum(bell_record.result.counts.values()) == 4
assert all(left == right for left, right in bell_record.result.counts)

# %% [markdown]
# The first two assertions check that the restored study equals the original and has the same content fingerprint, `digest`. The remaining assertions check that execution produced four samples, each with matching bits.
#
# :::{note}
# A study created directly from a quantum kernel can also be saved and restored. To compile a `ResourceStudy`, however, it must retain a specialized `Program`, even when the kernel has no `Const` inputs. A study returned by `evaluate(...)` cannot be compiled.
#
# This requirement concerns passing a study to `qm.compile(...)`; you can still pass a quantum kernel directly without calling `qm.specialize(...)` separately.
# :::

# %% [markdown]
# ## Estimating Physical Resources
#
# Quantum error correction encodes each logical qubit across multiple physical qubits. The surface code is one such scheme: it repeatedly measures information about errors without reading out the encoded logical information itself, then uses the measurement results to determine corrections.
#
# This protection requires additional qubits and operations. Qamomile provides a simplified surface-code model for estimating the resulting physical-qubit requirements and duration.
#
# `qm.resources.surface_code(...)` applies this model to a `ResourceStudy`. It returns a new study containing the original logical estimates and additional physical estimates, including the physical-qubit count and duration in seconds.
#
# Here, we apply the model with its default parameters to `category_study` from the preceding RZZ example. The model uses the logical peak-qubit count and non-Clifford operation count, together with assumptions about error rates and the time required for an error-correction cycle.
#
# A result’s `evidence` is a tuple of supporting records and may contain more than one entry. The checks below use `evidence[0]` to access the first entry.

# %%
category_physical = qm.resources.surface_code(category_study)

physical_qubits = category_physical.metric(qm.resources.PHYSICAL_QUBIT_COUNT)
physical_duration = category_physical.metric(qm.resources.PHYSICAL_DURATION)

print("estimated physical qubits:", physical_qubits.numeric_enclosure())
print("estimated duration (seconds):", physical_duration.numeric_enclosure())

assert physical_qubits.numeric_enclosure()[0] > 0
assert physical_duration.numeric_enclosure()[0] > 0
assert (
    physical_qubits.evidence[0].authority
    is qm.resources.EvidenceAuthority.HEURISTIC_MODEL
)
assert not physical_qubits.evidence[0].verified
assert category_physical.semantic_digest == category_study.semantic_digest

# %% [markdown]
# Under the default assumptions, the model estimates 72 physical qubits and a duration of 3 microseconds. The built-in model is illustrative and does not capture all costs of a physical implementation.
#
# The results carry `HEURISTIC_MODEL` evidence with `verified=False`, reflecting their dependence on an approximate physical model. Their assumptions remain available through `physical_qubits.assumptions` and `physical_duration.assumptions`.
#
# The final assertion checks that `semantic_digest`, the fingerprint of the program being analyzed, is unchanged. Applying the model adds resource estimates without changing that program.

# %% [markdown]
# ## Checking Confidence in an Estimate
#
# A numerical result alone does not tell you how an estimate was obtained or which assumptions it depends on. Each `ResourceValue` includes information that helps you interpret it:
#
# - **`evidence`** records the basis for the estimate. Each entry has an `authority` indicating whether it comes from the kernel’s structure, a supplied rule or model accepted as an assumption, or a heuristic model.
# - **`relation`** states whether the result gives an equality, a bound, an interval, or an unresolved quantity.
# - **`assumptions`** records conditions under which the estimate applies.
# - **`unknown_contributors`** identifies unresolved contributions, such as an opaque kernel without a gate-count rule.
#
# The examples in this guide and the [Resource Estimation tutorial](../tutorial/04_resource_estimation.ipynb) illustrate these distinctions:
#
# - **Bell pair:** the gate count is exactly two, derived from the kernel’s structure. Its evidence has authority `STRUCTURALLY_DERIVED`, and its relation is `EQUALITY`.
# - **Fixed-cost oracle in the tutorial:** the gate count is five under the supplied rule. The result includes `TRUSTED_AXIOM` evidence because the analysis accepts that cost without inspecting the oracle’s implementation.
# - **Controlled operation in the tutorial:** the clean-ancilla model gives five circuit gates under its stated counting assumptions, including counting the singly controlled H as one gate. These estimates also carry `TRUSTED_AXIOM` evidence.
# - **Oracle without a cost:** the gate-count bounds are `(0, None)`, with relation `PARTIAL_ENCLOSURE`. The missing rule appears in `unknown_contributors`.
# - **Physical estimate:** the surface-code model gives 72 physical qubits under its default assumptions. Its evidence has authority `HEURISTIC_MODEL`.
#
# An `EQUALITY` result can therefore depend on a supplied rule or an approximate model. For the surface-code example, it means that the model produces a specific value under its assumptions; it does not make that value an exact prediction for physical hardware.

# %% [markdown]
# To check the unresolved case, we analyze `oracle_program` without a cost rule. Its gate-count bounds are `(0, None)`: the count cannot be negative, but its upper bound is unknown. The assertions below check this result alongside the evidence and assumptions from the other examples.

# %%
unpriced_study = qm.resources.logical(oracle_program)
unpriced_gates = unpriced_study.metric(qm.resources.LOGICAL_GATES)
assert unpriced_gates.numeric_enclosure() == (0, None)
assert unpriced_gates.relation is qm.resources.Relation.PARTIAL_ENCLOSURE

# %%
assert (
    gate_estimate.evidence[0].authority
    is qm.resources.EvidenceAuthority.STRUCTURALLY_DERIVED
)
assert gate_estimate.relation is qm.resources.Relation.EQUALITY
assert not gate_estimate.assumptions
assert "rounds >= 0" in oracle_study.metric(qm.resources.LOGICAL_GATES).assumptions
assert unpriced_gates.unknown_contributors
assert (
    physical_qubits.evidence[0].authority
    is qm.resources.EvidenceAuthority.HEURISTIC_MODEL
)

# %% [markdown]
# ### Keeping Assumptions with the Results
#
# Assumptions are stored with the estimate. For example, the parameter-dependent oracle rule records `rounds >= 0`, while the surface-code model records which implementation costs it omits.
#
# Keep these assumptions alongside the values when reporting or comparing estimates. If part of an estimate remains unresolved, include its `unknown_contributors` as well.

# %% [markdown]
# ## Summary
#
# - Read each metric's bounds together with its relation, assumptions, evidence, and unresolved contributions.
# - Distinguish total allocations, peak usage, and static circuit width; also distinguish gate counts from sequential layers.
# - Save a `ResourceStudy` with `to_bytes()` and restore it with `from_bytes(...)`. A study must retain a specialized `Program` to be compiled.
# - Physical estimates depend on the selected model and its assumptions. Keep those assumptions with the reported values.
