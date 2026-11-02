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
# tags: [tutorial, resource-estimation]
# ---
#
# # Resource Estimation
#
# This chapter introduces resource estimation in Qamomile: how to estimate the qubits and quantum operations a kernel requires without running it. We will focus on logical resources, including the maximum number of qubits in use at one time, gate counts, and gate depth—the number of sequential gate layers required by the operations’ dependencies.
#
# We will start with a Bell-state circuit, then extend it to prepare a GHZ state: an equal superposition of the all-zero and all-one states. By varying the number of qubits, we will see how resource requirements change. Along the way, we will learn to interpret exact values and bounds, evaluate symbolic estimates, and identify the operations that contribute to a total.
#
# We will also show how to supply costs for operations declared without an implementation and estimate controlled operations under an explicit decomposition model. These examples also show how to distinguish results derived from the kernel from values that depend on supplied costs or modeling assumptions.

# %% [markdown]
# ## Setup
#
# Install Qamomile if needed, then import it. Logical resource estimation analyzes a kernel without running it, so no execution engine is required.

# %%
# Install the latest Qamomile through pip.
# # !pip install qamomile

# %%
from __future__ import annotations

import qamomile as qm

# %% [markdown]
# ## Your First Resource Estimate
#
# We will start with a kernel that prepares a Bell pair: two entangled qubits. It applies a Hadamard (H) gate to the first qubit, followed by a controlled-X (CX) gate with the first qubit as the control and the second as the target. It then measures both qubits.


# %%
@qm.qkernel
def bell_pair() -> qm.Shot[qm.Bits[2]]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.cx(state, control=0, target=1)
    return qm.measure(state)


# %% [markdown]
# Pass the kernel directly to `qm.resources.logical(...)` to analyze its logical resources. You do not need to call `qm.specialize(...)` separately or compile the kernel for a particular quantum computer.
#
# The analysis returns a `ResourceStudy`. Its `metric(...)` method retrieves a `ResourceValue` for a particular metric, such as the gate count or the maximum number of qubits in use at one time.
#
# `numeric_enclosure()` returns the numerical lower and upper bounds as a pair, `(lower, upper)`. When the bounds are equal, the estimate gives a single value: for example, `(2, 2)` means two.

# %%
bell_study = qm.resources.logical(bell_pair)

peak = bell_study.metric(qm.resources.LOGICAL_PEAK_QUBITS)
gates = bell_study.metric(qm.resources.LOGICAL_GATES)
measurements = bell_study.metric(qm.resources.LOGICAL_MEASUREMENTS)
resets = bell_study.metric(qm.resources.LOGICAL_RESETS)
gate_depth = bell_study.metric(qm.resources.LOGICAL_GATE_DEPTH)

print("peak logical qubits:", peak.numeric_enclosure())
print("logical gates:", gates.numeric_enclosure())
print("logical measurements:", measurements.numeric_enclosure())
print("logical resets:", resets.numeric_enclosure())
print("logical gate depth:", gate_depth.numeric_enclosure())

assert peak.numeric_enclosure() == (2, 2)
assert gates.numeric_enclosure() == (2, 2)
assert measurements.numeric_enclosure() == (2, 2)
assert resets.numeric_enclosure() == (0, 0)
assert gate_depth.numeric_enclosure() == (2, 2)
assert gates.relation is qm.resources.Relation.EQUALITY

# %% [markdown]
# All five estimates are exact in this example. The kernel uses two qubits at the same time and contains two gates, two qubit measurements, and no resets. The gate depth is two because the CX gate must follow the H gate on the first qubit. Measurements are counted separately from gates.
#
# These values come from analyzing the kernel; no quantum computation or measurement has been performed.

# %% [markdown]
# ## Understanding the Metrics
#
# The previous example reported qubit usage, operation counts, and gate depth. These metrics describe different aspects of a kernel’s resource requirements.
#
# ### Peak Qubit Usage
#
# `LOGICAL_PEAK_QUBITS` is the maximum number of logical qubits in use at the same time. In `bell_pair`, both qubits remain in use until they are measured, so the peak is two.
#
# This is different from counting every qubit allocated throughout a kernel. If one qubit’s lifetime ends before another is allocated, they do not both contribute to the peak. [Resource Estimation in Detail](../usage/resource_estimation.ipynb) explains the related metrics for input qubits, allocated qubits, and static circuit width.
#
# ### Gate, Measurement, and Reset Counts
#
# `LOGICAL_GATES` counts logical gate applications. A gate counts as one application regardless of how many qubits it acts on. In `bell_pair`, the H gate and the CX gate each contribute one, giving a total of two.
#
# Qamomile also provides a breakdown by gate size:
#
# - `LOGICAL_SINGLE_QUBIT_GATES`: gates acting on one qubit.
# - `LOGICAL_TWO_QUBIT_GATES`: gates acting on two qubits.
# - `LOGICAL_MULTI_QUBIT_GATES`: gates acting on three or more qubits.
#
# Measurements and resets are counted separately, per qubit, using `LOGICAL_MEASUREMENTS` and `LOGICAL_RESETS`. Although `bell_pair` calls `qm.measure(state)` only once, it measures two qubits, so the measurement count is two. There are no resets, so the reset count is zero.
#
# These counts describe one execution of the kernel; they are not multiplied by the number of shots.
#
# ### Gate Depth
#
# Gate count tells us how many gates are applied. `LOGICAL_GATE_DEPTH` tells us how many sequential gate layers are needed, respecting the dependencies between operations. Gates acting on different qubits can share a layer when neither depends on the other.
#
# In `bell_pair`, the CX gate uses the first qubit after the H gate, so the two gates occupy separate layers. Its gate count and gate depth are both two.
#
# For comparison, the following kernel applies an H gate independently to each qubit:


# %%
@qm.qkernel
def parallel_hadamards() -> qm.Shot[qm.Bits[2]]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.h(state, target=1)
    return qm.measure(state)


# %%
parallel_study = qm.resources.logical(parallel_hadamards)

parallel_gates = parallel_study.metric(qm.resources.LOGICAL_GATES)
parallel_depth = parallel_study.metric(qm.resources.LOGICAL_GATE_DEPTH)

print("logical gates:", parallel_gates.numeric_enclosure())
print("logical gate depth:", parallel_depth.numeric_enclosure())

assert parallel_gates.numeric_enclosure() == (2, 2)
assert parallel_depth.numeric_enclosure() == (1, 1)
assert parallel_study.metric(
    qm.resources.LOGICAL_MEASUREMENT_DEPTH
).numeric_enclosure() == (1, 1)

# %% [markdown]
# The two H gates appear on separate lines in the source, but they have no dependency on each other. They therefore fit in one gate layer.
#
# :::{note}
# Logical gate depth is a dependency-based layer count, not an estimate of elapsed time on a particular device. Measurements and resets have their own depth metrics and do not add gate layers.
# :::

# %% [markdown]
# ## Comparing Problem Sizes
#
# Resource requirements often depend on the size of a problem. Qamomile can express these requirements in terms of a kernel’s `Const` parameters, allowing you to compare several sizes without running the kernel.

# %% [markdown]
# ### Evaluating a Symbolic Resource Estimate
#
# The GHZ (Greenberger–Horne–Zeilinger) state on $n=\texttt{width}$ qubits is
#
# $$
# |\mathrm{GHZ}_n\rangle
# =\frac{|0\rangle^{\otimes n}+|1\rangle^{\otimes n}}{\sqrt{2}}.
# $$
#
# For example, with four qubits, it is $(|0000\rangle+|1111\rangle)/\sqrt{2}$.
#
# The kernel applies a Hadamard gate to the first qubit, followed by a chain of CX gates connecting adjacent qubits. For `width=2`, this produces the same Bell state as the earlier example.
#
# `width` is a `Const` input because it determines the number of qubits, the number of loop iterations, and the size of the measurement result.


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
# Passing this kernel to `qm.resources.logical(...)` without specifying `width` returns a `ResourceStudy` containing symbolic resource estimates—expressions that can still depend on `width`.
#
# Use `study.evaluate(...)` to substitute a value into those expressions. Each call returns a new study, leaving the original available for other values.

# %%
symbolic_study = qm.resources.logical(ghz_state)

# Compare logical gate counts and gate depths at different register widths.
for width in (2, 4, 8):
    estimate = symbolic_study.evaluate(width=width)
    width_gates = estimate.metric(qm.resources.LOGICAL_GATES)
    width_depth = estimate.metric(qm.resources.LOGICAL_GATE_DEPTH)

    print(
        f"width={width}: "
        f"gates={width_gates.numeric_enclosure()}, "
        f"depth={width_depth.numeric_enclosure()}"
    )

    assert width_gates.numeric_enclosure() == (width, width)
    assert width_depth.numeric_enclosure() == (width, width)
    assert estimate.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure() == (
        width,
        width,
    )
    assert estimate.metric(qm.resources.LOGICAL_MEASUREMENTS).numeric_enclosure() == (
        width,
        width,
    )

# %% [markdown]
# For each width shown here, the kernel uses one H gate and `width - 1` CX gates, giving `width` gates in total.
#
# The first CX gate must follow the H gate on qubit 0. Each subsequent CX gate uses the previous gate’s target qubit as its control, so it must follow that gate. The H gate and `width - 1` CX gates therefore occupy `width` sequential layers, giving a gate depth of `width`.
#
# All `width` qubits are in use together and are measured at the end. The peak qubit usage and measurement count are therefore also `width`. Evaluating these estimates does not execute the quantum circuit.

# %% [markdown]
# ### When Specialization Gives a More Precise Estimate
#
# `study.evaluate(...)` substitutes values into resource expressions that have already been derived. It does not analyze the kernel again.
#
# To analyze the kernel with particular `Const` values fixed, pass the `Program` returned by `qm.specialize(...)` to `qm.resources.logical(...)`. This can give a more precise estimate when those values determine which operations depend on one another.
#
# For example, the following kernel applies an H gate to qubit 0 and an X gate to qubit `i`:


# %%
@qm.qkernel
def symbolic_access(i: qm.Const[qm.Int]) -> qm.Q[qm.Bits[2]]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.x(state, target=i)
    return state


# %% [markdown]
# If `i=0`, both gates act on the same qubit and occupy two layers. If `i=1`, they act on different qubits and fit in one layer.

# %%
access_symbolic = qm.resources.logical(symbolic_access)
access_depth = access_symbolic.metric(qm.resources.LOGICAL_GATE_DEPTH)

access_evaluated = access_symbolic.evaluate(i=1)
evaluated_depth = access_evaluated.metric(qm.resources.LOGICAL_GATE_DEPTH)

access_i0 = qm.resources.logical(qm.specialize(symbolic_access, i=0))
access_i1 = qm.resources.logical(qm.specialize(symbolic_access, i=1))

depth_i0 = access_i0.metric(qm.resources.LOGICAL_GATE_DEPTH)
depth_i1 = access_i1.metric(qm.resources.LOGICAL_GATE_DEPTH)

# Compare logical gate-depth bounds before and after evaluation or specialization.
print("gate depth before specifying i:", access_depth.numeric_enclosure())
print("gate depth after evaluate(i=1):", evaluated_depth.numeric_enclosure())
print(
    "gate depth after specializing with i=0:",
    depth_i0.numeric_enclosure(),
)
print(
    "gate depth after specializing with i=1:",
    depth_i1.numeric_enclosure(),
)

assert access_depth.numeric_enclosure() == (1, 2)
assert evaluated_depth.numeric_enclosure() == (1, 2)
assert depth_i0.numeric_enclosure() == (2, 2)
assert depth_i1.numeric_enclosure() == (1, 1)

# %% [markdown]
# Without a specific value for `i`, the analysis reports a depth between one and two. Calling `evaluate(i=1)` leaves those bounds unchanged because it does not reconsider the dependencies between the gates. Analyzing the specialized program resolves which qubit the X gate acts on and gives an exact depth.
#
# Use `evaluate(...)` to compare values of an existing resource expression. Use specialization followed by resource analysis when you need the analysis to account for a particular kernel configuration.

# %% [markdown]
# ## Tracing Resource Costs
#
# A total gate count tells us how many gates a kernel requires. To understand where that count comes from, we can examine the individual operations that contribute to it.
#
# A `ResourceStudy` provides three methods for this:
#
# - `contributions(...)` lists individual operation contributions, including their values and source locations.
# - `hotspots(...)` returns up to ten contributions, ordered by numerical upper bounds from largest to smallest. Entries without a numerical upper bound appear afterward.
# - `explain(...)` returns a text explanation of the selected metric’s total and the operations that contribute to it.
#
# We will use the `bell_study` created earlier to examine its logical gate count:

# %%
gate_contributions = bell_study.contributions(qm.resources.LOGICAL_GATES)
gate_hotspots = bell_study.hotspots(qm.resources.LOGICAL_GATES)
explanation = bell_study.explain(qm.resources.LOGICAL_GATES)

print(explanation)

assert [item.operation for item in gate_contributions] == ["quantum.h", "quantum.cx"]
assert set(gate_hotspots) == set(gate_contributions)
assert all(item.value.numeric_enclosure() == (1, 1) for item in gate_contributions)
assert "quantum.h" in explanation
assert "quantum.cx" in explanation

# %% [markdown]
# The explanation reports two logical gates in total. The entries `quantum.h` and `quantum.cx` correspond to the H and CX gates in `bell_pair`, each contributing one gate. Measurements are absent from this breakdown because we requested `LOGICAL_GATES`; they are counted separately.
#
# Both gates have the same cost here, so neither contributes more than the other. In a larger kernel, `hotspots(...)` can help identify operations with larger estimated contributions to the selected metric.

# %% [markdown]
# ## Assigning a Cost to an Opaque Kernel
#
# You may want to estimate an algorithm’s resources before implementing every kernel it calls. Qamomile supports this through an *opaque quantum kernel*: a kernel that declares its inputs and outputs without providing an implementation.
#
# You can use a `ResourceRule` to specify assumptions about an opaque kernel’s resource requirements. Qamomile uses these assumptions together with the surrounding code to estimate the resources required by the larger kernel.
#
# :::{note}
# Qamomile cannot infer an opaque kernel’s internal gate count without its implementation or a supplied resource rule. It can still determine other facts, such as how many times the kernel is called, from the surrounding code. A rule for the gate count does not also determine gate depth or the types of gates used.
# :::
#
# ### Providing a Fixed Cost
#
# The following `oracle` declares a unitary operation on two qubits. Its body is `...`, so no gate sequence is specified. The `use_oracle` kernel calls it `rounds` times, then measures the qubits.


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
# Suppose each call to `oracle` requires exactly five logical gates. The rule below expresses this assumption: **logical gate count per call to `oracle` = 5**. `LOGICAL_GATES` identifies the quantity, `Relation.EQUALITY` specifies equality, and `literal(5)` supplies the value. We pass this rule to `qm.resources.logical(...)` to use it in the analysis.
#
# `identity` names the rule. `provenance` is a bookkeeping note that records the assumption or reference behind the specified cost. Here, it labels our five-gate assumption.
#
# Here, `strict=True` raises an error if the oracle’s resource cost remains unresolved. The supplied rule provides its gate cost; it does not determine every resource metric exactly. We will examine this setting in more detail later in this chapter.

# %%
fixed_oracle_gates = qm.resources.ResourceRule(
    identity="tutorial.oracle.fixed-gates.v1",
    callable=oracle,
    metric=qm.resources.LOGICAL_GATES,
    relation=qm.resources.Relation.EQUALITY,
    expression=qm.resources.literal(5),
    provenance="tutorial.oracle.fixed-contract.v1",
)

fixed_oracle_program = qm.specialize(use_oracle, rounds=1)
fixed_oracle_study = qm.resources.logical(
    fixed_oracle_program,
    rules=fixed_oracle_gates,
    strict=True,
)

print(
    "logical gates:",
    fixed_oracle_study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure(),
)

assert fixed_oracle_study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (
    5,
    5,
)

# %% [markdown]
# With `rounds=1`, the kernel calls `oracle` once, giving five logical gates under the supplied rule. The final measurements are counted separately.
#
# The result is exact under our assumption; Qamomile has not verified the five-gate cost against an implementation of `oracle`.

# %% [markdown]
# ### Computing a Parameter-Dependent Cost
#
# A cost can also depend on an argument passed to the opaque kernel. As an alternative to the fixed cost above, suppose one call to `oracle` uses `2 * rounds` logical gates, where `rounds` is the argument passed to that call.
#
# To write this formula, we use `rule_parameter(oracle, "rounds", ...)` to create the symbol `oracle_rounds` for that argument. The expression `multiply(literal(2), oracle_rounds)` then represents `2 * rounds`. The domain and constraints specify that `rounds` is a nonnegative integer.


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

# %% [markdown]
# In this example, `rounds` has two roles: it determines how many times `use_oracle` calls the oracle, and it is also passed to each call. Because it is a `Const` parameter of `use_oracle`, we fix its value when specializing that kernel.
#
# With `rounds=2`, the rule assigns four gates to each oracle call. The two calls therefore contribute eight gates in total. This analysis uses the new rule instead of the fixed five-gate rule.

# %%
oracle_program = qm.specialize(use_oracle, rounds=2)
oracle_study = qm.resources.logical(
    oracle_program,
    rules=oracle_gates,
    strict=True,
)

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
# The call count comes from analyzing `use_oracle`. Each call to this opaque unitary also counts as an oracle query, so both counts are two here.
#
# The gate cost per call, however, comes from our rule. `TRUSTED_AXIOM` records that the estimate relies on a supplied assumption rather than a gate sequence that Qamomile has analyzed.
#
# :::{note}
# Specifying a resource cost does not make the opaque kernel executable; execution still requires an implementation. Resource rules cannot override costs derived from a kernel whose implementation is available for analysis.
# :::

# %% [markdown]
# ## Estimating Controlled Operations
#
# A controlled operation can be implemented using different gate sequences. To estimate the resources needed for a particular decomposition, we first analyze the kernel, then apply a model that specifies the decomposition and how its gates are counted.
#
# We will use a Hadamard gate controlled by three qubits as an example.
#
# ### Starting with Logical Analysis
#
# `qm.control` currently requires a unitary quantum kernel. Although the H gate is unitary, the built-in function `qm.h` is not a quantum kernel and cannot be passed directly. We therefore define `apply_hadamard`, a quantum kernel that calls `qm.h`, and pass it to `qm.control`. The kernel applies an ordinary H gate; `qm.control` adds the controls.
#
# In `three_controlled_hadamard`, we supply three control qubits and one target qubit. The H gate is applied to the target when all three controls are in state $|1\rangle$.


# %%
@qm.qkernel
def apply_hadamard(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.h(state)


@qm.qkernel
def three_controlled_hadamard() -> qm.Q[qm.Bits[4]]:
    controls = qm.qubits(3)
    target = qm.qubits(1)
    controls, target = qm.control(apply_hadamard, controls, target)
    return qm.concat(controls, target)


# %% [markdown]
# Finally, `qm.concat(controls, target)` combines the three control qubits and the target into one four-qubit register, with the controls first. The kernel returns this quantum state without measuring it.
#
# We analyze the kernel directly with `qm.resources.logical(...)`. This function also accepts a `Program` returned by `qm.specialize(...)` when we want to analyze a particular specialization.
#
# To count controlled kernel calls, we use the metric `qm.resources.logical_operation_count("qcore.control_call")`. The string `"qcore.control_call"` is Qamomile’s internal identifier for the operation represented by `qm.control`. We pass this metric to `abstract_control_study.metric(...)` to retrieve the estimated count.
#
# This counts controlled calls, rather than the gates needed to implement them.

# %%
abstract_control_study = qm.resources.logical(three_controlled_hadamard)
abstract_control_calls = abstract_control_study.metric(
    qm.resources.logical_operation_count("qcore.control_call")
)

print("controlled calls:", abstract_control_calls.numeric_enclosure())
print(
    "logical gate-count bounds:",
    abstract_control_study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure(),
)

assert abstract_control_calls.numeric_enclosure() == (1, 1)
assert abstract_control_study.metric(
    qm.resources.LOGICAL_PEAK_QUBITS
).numeric_enclosure() == (4, 4)
assert abstract_control_study.metric(
    qm.resources.LOGICAL_GATES
).numeric_enclosure() == (1, None)

# %% [markdown]
# The analysis identifies one controlled call and four logical qubits. The gate-count result `(1, None)` gives a lower bound of one, with no known upper bound. We have not yet selected a decomposition for the controlled operation.
#
# ### Choosing a Decomposition Model
#
# `qm.resources.clean_ancilla_controls(...)` estimates the resources needed for a decomposition using *clean ancillas*: temporary qubits initialized to $|0\rangle$ and restored to $|0\rangle$ after use.
#
# The decomposition uses Toffoli (CCX) gates. A Toffoli gate flips its target when both of its control qubits are in state $|1\rangle$.
#
# For three control qubits, this model uses two ancillas. Two Toffoli gates compute the control condition, a singly controlled H gate applies H to the target when that condition is satisfied, and the same two Toffoli gates are applied in reverse order to restore the ancillas. The model counts each Toffoli gate and the singly controlled H as one gate, giving five gates in total. These steps are explained in detail in the appendix, “Deriving the Clean-Ancilla Decomposition,” at the end of this chapter.

# %%
decomposed_control_study = qm.resources.clean_ancilla_controls(abstract_control_study)

print(
    "model gate count:",
    decomposed_control_study.metric(
        qm.resources.CIRCUIT_VIRTUAL_GATES
    ).numeric_enclosure(),
)
print(
    "model Toffoli count:",
    decomposed_control_study.metric(
        qm.resources.CIRCUIT_TOFFOLI_GATES
    ).numeric_enclosure(),
)
print(
    "additional clean ancillas:",
    decomposed_control_study.metric(
        qm.resources.CIRCUIT_CLEAN_ANCILLA_QUBITS
    ).numeric_enclosure(),
)
print(
    "model gate depth:",
    decomposed_control_study.metric(
        qm.resources.CIRCUIT_GATE_DEPTH
    ).numeric_enclosure(),
)
print(
    "model qubit count:",
    decomposed_control_study.metric(
        qm.resources.CIRCUIT_VIRTUAL_QUBITS
    ).numeric_enclosure(),
)

assert decomposed_control_study.metric(
    qm.resources.CIRCUIT_VIRTUAL_GATES
).numeric_enclosure() == (5, 5)
assert decomposed_control_study.metric(
    qm.resources.CIRCUIT_TOFFOLI_GATES
).numeric_enclosure() == (4, 4)
assert decomposed_control_study.metric(
    qm.resources.CIRCUIT_CLEAN_ANCILLA_QUBITS
).numeric_enclosure() == (2, 2)
assert decomposed_control_study.metric(
    qm.resources.CIRCUIT_GATE_DEPTH
).numeric_enclosure() == (5, 5)
assert decomposed_control_study.metric(
    qm.resources.CIRCUIT_VIRTUAL_QUBITS
).numeric_enclosure() == (6, 6)

# %% [markdown]
# The two ancillas bring the qubit count from four to six. In this example, the five gates form a dependency chain, giving a gate depth of five.
#
# `clean_ancilla_controls(...)` returns a new `ResourceStudy` containing these additional `CIRCUIT_*` metrics. It retains the original logical metrics and does not modify or compile the kernel.
#
# ### Checking the Resource Estimate with an Explicit Circuit
#
# We can also write this decomposition explicitly and compare its estimated resources with the model’s results.
#
# In the following kernel, qubits 0–2 are the controls, qubit 3 is the target, and qubits 4–5 are the ancillas $a_0$ and $a_1$. The numbered comments identify the five gate operations. The appendix explains why the final two operations restore the ancillas to $|0\rangle$.


# %%
@qm.qkernel
def three_controlled_hadamard_manually() -> qm.Q[qm.Bits[6]]:
    state = qm.qubits(6)

    # Qubits 0–2: controls; qubit 3: target; qubits 4–5: ancillas a0 and a1.

    # Step 1: Compute the AND of control qubits 0 and 1 into a0.
    state = qm.ccx(state, control1=0, control2=1, target=4)

    # Step 2: Compute the AND of control qubit 2 and a0 into a1.
    state = qm.ccx(state, control1=2, control2=4, target=5)

    controls, remainder = qm.split_at(state, 3)
    target, ancillas = qm.split_at(remainder, 1)
    first_ancilla, second_ancilla = qm.split_at(ancillas, 1)

    # Step 3: Apply H to the target, controlled by a1.
    second_ancilla, target = qm.control(
        apply_hadamard,
        second_ancilla,
        target,
    )

    # Combine the registers in their original index order.
    state = qm.concat(
        controls,
        qm.concat(target, first_ancilla, second_ancilla),
    )

    # Step 4: Undo step 2 and restore a1 to |0>.
    state = qm.ccx(state, control1=2, control2=4, target=5)

    # Step 5: Undo step 1 and restore a0 to |0>.
    state = qm.ccx(state, control1=0, control2=1, target=4)

    return state


# %% [markdown]
# The singly controlled H in step 3 is still represented by `qm.control`, so we apply the same model to count it as one circuit gate. The assertions below compare the total gate count, Toffoli count, gate depth, and qubit count.

# %%
manual_control_study = qm.resources.logical(three_controlled_hadamard_manually)
manual_control_projection = qm.resources.clean_ancilla_controls(manual_control_study)

assert (
    manual_control_study.metric(qm.resources.LOGICAL_TOFFOLI_GATES).numeric_enclosure()[
        0
    ]
    == 4
)
assert (
    manual_control_study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure()[0] == 5
)
assert manual_control_study.metric(
    qm.resources.LOGICAL_PEAK_QUBITS
).numeric_enclosure() == (6, 6)

for metric in (
    qm.resources.CIRCUIT_VIRTUAL_GATES,
    qm.resources.CIRCUIT_TOFFOLI_GATES,
    qm.resources.CIRCUIT_GATE_DEPTH,
    qm.resources.CIRCUIT_VIRTUAL_QUBITS,
):
    assert manual_control_projection.metric(metric).numeric_enclosure() == (
        decomposed_control_study.metric(metric).numeric_enclosure()
    )

# %% [markdown]
# Both approaches give five gates, including four Toffoli gates, a gate depth of five, and six qubits. In the explicit kernel, the two ancillas are allocated in the code. In the earlier estimate, their cost was introduced by the decomposition model.

# %% [markdown]
# ## Handling Incomplete Estimates
#
# In the earlier oracle example, we supplied a resource rule to describe the oracle’s gate cost. What happens when no such rule is available?
#
# When cost information is missing, `qm.resources.logical(...)` returns a partial estimate by default, retaining the bounds it can establish and recording what remains unresolved. Here, we leave the cost of `unpriced_step` unspecified.


# %%
@qm.qkernel(kind="unitary")
def unpriced_step(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]: ...


@qm.qkernel
def with_unpriced_oracle() -> qm.Q[qm.Bits[1]]:
    state = qm.qubits(1)
    return unpriced_step(state)


# %%
unpriced_study = qm.resources.logical(with_unpriced_oracle)
unpriced_gates = unpriced_study.metric(qm.resources.LOGICAL_GATES)

print("gate-count bounds:", unpriced_gates.numeric_enclosure())
print("unresolved costs:", unpriced_gates.unknown_contributors)

assert unpriced_gates.relation is qm.resources.Relation.PARTIAL_ENCLOSURE
assert unpriced_gates.numeric_enclosure() == (0, None)
assert unpriced_gates.unknown_contributors

# %% [markdown]
# The gate-count bounds are `(0, None)`. Zero is the lower bound because gate counts cannot be negative; no upper bound is known. The oracle’s actual gate count remains unspecified.
#
# `Relation.PARTIAL_ENCLOSURE` marks the estimate as incomplete, and `unknown_contributors` explains what remains unresolved—in this case, the missing gate-cost rule for `unpriced_step`.
#
# If your workflow requires unresolved costs to be treated as errors, pass `strict=True`. The same analysis then raises a `ValueError`:

# %%
try:
    qm.resources.logical(with_unpriced_oracle, strict=True)
except ValueError as error:
    assert "unknown contributors" in str(error)
    print("Strict analysis rejected the unresolved resource cost.")
else:
    raise AssertionError("strict analysis must reject unresolved work")

# %% [markdown]
# The exception is caught here so that the notebook can continue. To provide the missing cost, supply a `ResourceRule` as in the earlier examples. If zero is the intended cost, state it explicitly in the rule.

# %% [markdown]
# ## Further Details
#
# See [Resource Estimation in Detail](../usage/resource_estimation.ipynb) for the full metric descriptions, additional worked examples, saving and loading a `ResourceStudy`, and interpreting physical estimates and their evidence. The guide includes its own setup and kernel definitions.

# %% [markdown]
# ## Summary
#
# - Use `qm.resources.logical(...)` to estimate logical qubit usage, gate counts, and gate depth without executing the quantum circuit.
# - Retrieve each metric as a `ResourceValue`, and read its value or bounds together with its evidence, assumptions, and unresolved contributions.
# - Use `study.evaluate(...)` to substitute values into existing symbolic estimates. To analyze the kernel again with specific `Const` values, specialize it and analyze the resulting `Program`.
# - Use `contributions(...)`, `hotspots(...)`, and `explain(...)` to trace an estimate back to the operations that contribute to it.
# - Supply a `ResourceRule` when an opaque kernel’s cost is known or assumed. Without sufficient information, retain the partial estimate or use `strict=True` to reject unresolved contributions.
# - Estimates based on a circuit-decomposition or physical model depend on that model’s assumptions. Keep those assumptions with the results when reporting or comparing them.

# %% [markdown]
# ## Appendix: Deriving the Clean-Ancilla Decomposition
#
# The decomposition in *Estimating Controlled Operations* implements a Hadamard gate controlled by three qubits using two clean ancillas. Here we show why it applies H only when all three controls are 1, and why both ancillas return to $|0\rangle$.
#
# Let $c_0,c_1,c_2\in\{0,1\}$ denote the control values in a computational-basis component. The target is in state $|t\rangle$, and the two ancillas, $a_0$ and $a_1$, start in $|00\rangle$.
#
# Define
#
# $$
# p=c_0c_1,\qquad q=c_0c_1c_2.
# $$
#
# For binary values, multiplication represents AND. Thus, $q=1$ only when all three controls are 1. In the notation below, $\operatorname{CCX}(x,y;z)$ has controls $x,y$ and target $z$.
#
# The five steps correspond to the numbered comments in `three_controlled_hadamard_manually`.
#
# 1. **Compute the first condition into $a_0$.**
#
#    Apply a Toffoli gate controlled by $c_0$ and $c_1$:
#
#    $$
#    \operatorname{CCX}(c_0,c_1;a_0):
#    \qquad a_0:\ 0\longrightarrow p.
#    $$
#
# 2. **Combine it with the third control in $a_1$.**
#
#    Apply a Toffoli gate controlled by $c_2$ and $a_0$:
#
#    $$
#    \operatorname{CCX}(c_2,a_0;a_1):
#    \qquad a_1:\ 0\longrightarrow c_2p=q.
#    $$
#
# 3. **Use $a_1$ to control H on the target.**
#
#    $$
#    |t\rangle\longrightarrow H^q|t\rangle
#    =
#    \begin{cases}
#    |t\rangle,&q=0,\\
#    H|t\rangle,&q=1.
#    \end{cases}
#    $$
#
#    This operation leaves the control values and both ancilla values unchanged.
#
# 4. **Undo step 2, restoring $a_1$.**
#
#    Applying the same Toffoli gate again gives
#
#    $$
#    a_1:\ q\longrightarrow q\oplus c_2p=q\oplus q=0.
#    $$
#
# 5. **Undo step 1, restoring $a_0$.**
#
#    Applying the first Toffoli gate again gives
#
#    $$
#    a_0:\ p\longrightarrow p\oplus c_0c_1=p\oplus p=0.
#    $$
#
# Here, $\oplus$ denotes XOR. A binary value XOR itself is always zero, so repeating the two Toffoli gates in reverse order restores both ancillas. This reversal of the temporary computation is called *uncomputation*.
#
# The overall transformation is therefore
#
# $$
# |c_0c_1c_2\rangle|t\rangle|00\rangle_a
# \longrightarrow
# |c_0c_1c_2\rangle
# H^{c_0c_1c_2}|t\rangle
# |00\rangle_a.
# $$
#
# The target receives H only when all three controls are 1, and both ancillas return to $|0\rangle$. These equations also describe superpositions by linearity; no control qubit is measured.
#
# In this model, each Toffoli gate and the singly controlled H count as one gate:
#
# $$
# \underbrace{2\ \mathrm{CCX}}_{\text{compute}}
# +\underbrace{1\ \mathrm{controlled}\ H}_{\text{apply}}
# +\underbrace{2\ \mathrm{CCX}}_{\text{uncompute}}
# =5\text{ gates}.
# $$
#
# The three controls, one target, and two ancillas require six qubits. Each gate depends on the preceding gate in this decomposition, so its gate depth is five.

# %% [markdown]
# **Next**: [Execution, Statistics, and Observable Estimation](05_execution_models.ipynb) covers running kernels, summarizing measured samples, and estimating expectation values of observables.
