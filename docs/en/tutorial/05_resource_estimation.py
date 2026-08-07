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
# Before running a quantum kernel on real hardware, you may want to know its qubit width, gate count, measurement and reset events, and depth. Qamomile's `estimate_resources()` fills this need **without executing the qkernel**. It currently reports one target-neutral logical algorithmic resource model rather than hardware-native gates. By default, coherent controls use the clean-ancilla Toffoli decomposition, without applying target-specific optimization. It works with both concrete and symbolic (parameterized) qkernels.
#
# This chapter covers:
#
# - Basic resource estimation for fixed qkernels
# - Choosing how coherent controls are decomposed
# - How controls, inverse calls, SELECT, Pauli evolution, and control flow compose
# - Separating gate, measurement, and reset resources
# - Symbolic resource estimation for parameterized qkernels
# - Structural requirements, opaque boundaries, traces, and JSON-friendly output
# - Scaling analysis with `.substitute()`
# - Applying body-derived estimates to Shor order finding

# %%
# Install the latest Qamomile through pip!
# # !pip install "qamomile[qiskit,visualization]"

# %%
from dataclasses import replace

import qamomile.circuit as qmc

# %% [markdown]
# ## Estimating Resources of a Fixed QKernel
#
# For a qkernel with no parameters, `estimate_resources()` returns concrete numbers.


# %%
@qmc.qkernel
def fixed_circuit() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(3, name="q")

    q[0] = qmc.h(q[0])
    q[0], q[1] = qmc.cx(q[0], q[1])
    q[1], q[2] = qmc.cx(q[1], q[2])

    return qmc.measure(q)


# %%
fixed_circuit.draw()

# %%
est = fixed_circuit.estimate_resources()
print("qubits:", est.qubits)
assert est.qubits == 3
print("total gates:", est.gates.total)
assert est.gates.total == 3
print("single-qubit gates:", est.gates.single_qubit)
assert est.gates.single_qubit == 1
print("two-qubit gates:", est.gates.two_qubit)
assert est.gates.two_qubit == 2
print("measurements:", est.measurements.total)
assert est.measurements.total == 3
assert est.resets.total == 0
assert est.depth.depth == 4
assert est.depth.gate_depth == 3
assert est.depth.measurement_depth == 1
assert est.depth.reset_depth == 0

# %% [markdown]
# ## Choosing the Control Decomposition
#
# `control_decomposition` selects how coherent controls are represented. The default is `ControlDecomposition.CLEAN_ANCILLA_TOFFOLI`. Called qkernels are recursively inspected in either control model. For example, the default model represents a three-controlled H with four Toffoli gates, one controlled-H gate, and two clean ancillas.


# %%
@qmc.qkernel
def one_h(target: qmc.Qubit) -> qmc.Qubit:
    return qmc.h(target)


@qmc.qkernel
def controlled_h() -> qmc.Qubit:
    controls = qmc.qubit_array(3, name="controls")
    target = qmc.qubit("target")
    *_, target = qmc.control(one_h, num_controls=3)(controls, target)
    return target


# %%
decomposed_controls = controlled_h.estimate_resources()
assert (
    decomposed_controls.control_decomposition
    is qmc.ControlDecomposition.CLEAN_ANCILLA_TOFFOLI
)
assert decomposed_controls.gates.total == 5
assert decomposed_controls.gates.toffoli == 4
assert decomposed_controls.width.clean_ancilla_qubits == 2
assert decomposed_controls.qubits == 6
assert decomposed_controls.derivation is qmc.EstimateDerivation.STRUCTURAL
assert decomposed_controls.quality is qmc.EstimateQuality.CONSERVATIVE

abstract_controls = controlled_h.estimate_resources(
    control_decomposition=qmc.ControlDecomposition.ABSTRACT,
)
assert abstract_controls.control_decomposition is qmc.ControlDecomposition.ABSTRACT
assert abstract_controls.gates.total == 1
assert abstract_controls.width.clean_ancilla_qubits == 0
assert abstract_controls.qubits == 4

# %% [markdown]
# `CLEAN_ANCILLA_TOFFOLI` is a fixed algorithmic model used by the resource estimator for coherent controls. Each ordinary gate contributes one unit of active work. When two or more controls surround at least two units of active work, the model computes their conjunction once and shares it across the body; a one-operation or one-control body retains a conservative per-primitive decomposition. The work count follows resolved qkernel calls, inverse blocks, branches, and concrete loops, so merely factoring the same body into another qkernel does not change the result. Nested control boundaries, active SELECT cases, and Pauli evolution may represent enough internal controlled work to select sharing as one structural leaf. This resource model is deliberately independent from an engine's emission policy: an engine may retain a direct path or use native multi-controlled gates without changing the estimate. Select `ABSTRACT` when you intentionally want each controlled source primitive to remain one operation regardless of control count. The estimator currently exposes only this logical algorithmic gate model; target-native, fault-tolerant, or other resource models can be added as separate interfaces when they are implemented. Neither control setting performs routing, hardware-native optimization, or error-correction costing.


# %%
@qmc.qkernel
def identity_body(target: qmc.Qubit) -> qmc.Qubit:
    return target


@qmc.qkernel
def controlled_phase(theta: qmc.Float) -> qmc.Qubit:
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    _, target = qmc.control(identity_body)(
        control,
        target,
        global_phase=theta,
    )
    return target


# %%
phase_est = controlled_phase.estimate_resources()
assert phase_est.substitute(theta=0).gates.total == 0
nonzero_phase = phase_est.substitute(theta=0.3)
assert nonzero_phase.gates.total == 1
assert nonzero_phase.gates.rotation_gates == 1
assert nonzero_phase.approximation is qmc.ApproximationStatus.EXACT

# %% [markdown]
# A standalone global phase is unobservable and costs zero in this target-neutral model. Under coherent control it becomes a relative phase, as the example shows. An angle that is not the identity modulo `2π` contributes one logical rotation; this estimate does not claim any particular synthesized gate sequence.
#
# Higher-level operations are interpreted from their executable meaning rather than counted as one opaque box:
#
# | Construct | Estimation behavior |
# |---|---|
# | Body-backed qkernel calls and `qmc.inverse(...)` | Recursively traverse the selected implementation body. Inversion preserves gate counts while reversing the unitary semantics; it does not collapse a large body to one call. |
# | `qmc.control(...)` | Propagate every surrounding control into nested bodies. With `CLEAN_ANCILLA_TOFFOLI`, multi-control decomposition gates and reusable clean ancillas are included. With `ABSTRACT`, each controlled primitive remains one abstract operation. Open controls also include their X brackets. |
# | `qmc.select(...)` | Sum every emitted controlled case body, including index-bit controls, open-control brackets, vector broadcast, and surrounding controls. The declared index width must match its operands and address all cases. |
# | `qmc.pauli_evolve(...)` | For a supplied Hermitian Hamiltonian, count Pauli-basis changes, parity ladders, axial rotations, and any controlled constant-term phase. A noncommuting Pauli sum is one first-order Lie–Trotter step in Hamiltonian term order and is marked `approximate`. The target register must cover the Hamiltonian support. |
# | `if`, `for`, `for_items`, and `while` | Keep compile-time conditions and loop bounds symbolic when possible. Gate and width metrics for a parameter branch use `Piecewise`; a measurement-backed branch uses a conservative maximum. Dependency-depth quality can remain `conservative` when symbolic aliases or multi-wire boundaries prevent an exact schedule. Loop work accumulates while reusable width follows liveness. Unsupported carried recurrences fail explicitly or add a visible assumption. |
# | `qmc.expval(...)` | Record one abstract expectation query and one measurement layer. Observable grouping, basis rotations, shot count, and executor sampling policy remain engine/executor-dependent, so the result has `derivation=MODELED` and `quality=UNKNOWN`, with a visible assumption rather than a fabricated gate count. |
# | LCU block encodings | Preserve exact signal- and system-register requirements through direct, controlled, inverse, descriptor-bound, and serialized forms. |

# %% [markdown]
# ### Resource-count derivation, quality, and mathematical approximation
#
# Three independent fields answer different questions. `estimate.derivation` reports whether the estimator derived the counts from visible IR and the selected estimator decomposition rules (`STRUCTURAL`) or used a declared/fallback model (`MODELED`). `estimate.quality` reports whether those counts are `EXACT`, `CONSERVATIVE`, or `UNKNOWN`. `CONSERVATIVE` means that the counts may overestimate but must not underestimate the selected circuit model. `estimate.approximation` records mathematical approximations that the estimator recognizes in the selected circuit. Its `EXACT` value means that no known approximation was recorded; it is not a proof about semantics hidden behind an opaque boundary or already lowered by an external helper. Consequently, `CONSERVATIVE` does not imply `APPROXIMATE`, and a modeled estimate may also be conservative.


# %%
import qamomile.observable as qmo


@qmc.qkernel
def pauli_sum_evolution(
    hamiltonian: qmc.Observable,
    time: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    target = qmc.qubit_array(2, name="target")
    return qmc.pauli_evolve(target, hamiltonian, time)


noncommuting_hamiltonian = qmo.X(0) + qmo.Z(0)
noncommuting_est = pauli_sum_evolution.estimate_resources(
    inputs={"hamiltonian": noncommuting_hamiltonian, "time": 0.25},
)
zero_time_est = pauli_sum_evolution.estimate_resources(
    inputs={"hamiltonian": noncommuting_hamiltonian, "time": 0.0},
)
commuting_est = pauli_sum_evolution.estimate_resources(
    inputs={"hamiltonian": qmo.Z(0) + qmo.Z(1), "time": 0.25},
)

assert noncommuting_est.derivation is qmc.EstimateDerivation.STRUCTURAL
assert noncommuting_est.quality is qmc.EstimateQuality.EXACT
assert noncommuting_est.approximation is qmc.ApproximationStatus.APPROXIMATE
assert zero_time_est.approximation is qmc.ApproximationStatus.EXACT
assert commuting_est.approximation is qmc.ApproximationStatus.EXACT

# %% [markdown]
# The noncommuting result has exact counts for the selected one-step circuit but approximates the ideal exponential. Its assumption names the first-order Lie–Trotter formula. Zero evolution time removes that operation, and commuting Pauli terms need no product-formula approximation, so both report `ApproximationStatus.EXACT`.

# %% [markdown]
# ## Symbolic Resource Estimation
#
# When a qkernel has unbound parameters (like `n: qmc.UInt`), `estimate_resources()` returns **SymPy expressions** that show how costs scale with the parameter. This lets you analyze scaling without picking a specific value.


# %%
@qmc.qkernel
def scalable_circuit(n: qmc.UInt, theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    q = qmc.h(q)
    q = qmc.ry(q, theta)

    for i in qmc.range(n - 1):
        q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])

    return qmc.measure(q)


# %%
scalable_circuit.draw(n=4, fold_loops=False)

# %%
est = scalable_circuit.estimate_resources()
print("qubits:", est.qubits)
assert str(est.qubits) == "n"
print("total gates:", est.gates.total)
assert str(est.gates.total) == "2*n + Max(0, n - 1)"
print("single-qubit gates:", est.gates.single_qubit)
assert str(est.gates.single_qubit) == "2*n"
print("two-qubit gates:", est.gates.two_qubit)
assert str(est.gates.two_qubit) == "Max(0, n - 1)"
print("rotation gates:", est.gates.rotation_gates)
assert str(est.gates.rotation_gates) == "n"
print("parameters:", est.parameters)
assert set(est.parameters.keys()) == {"n"}

# %% [markdown]
# The output contains SymPy expressions like `n` for qubits and `2*n + Max(0, n - 1)` for total gates. Inspect `est.derivation` to see whether an explicit/fallback model contributed and `est.quality` to distinguish exact, conservative, and directionally unknown counts.
#
# The `Max(0, ...)` comes from the trip count of `qmc.range(n - 1)`. Since `n` is unbound, the estimator cannot assume `n >= 1`, so it clamps the count at zero rather than letting `n = 0` contribute `-1` iterations. Substituting any concrete `n >= 1` collapses the guard, which is why the totals below come out as plain integers.

# %% [markdown]
# ### Supplying input shapes and validating requirements
#
# Use `inputs={...}` when a resource depends on a concrete qkernel input. Classical vectors accept ordinary array-like values, whose shapes specialize the estimate. For a one-dimensional `Vector[Qubit]` input, pass its width directly as an integer; no dummy qubit objects are needed.


# %%
@qmc.qkernel
def vector_input(register: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    return qmc.h(register)


# %%
vector_est = vector_input.estimate_resources(inputs={"register": 8})
assert vector_est.width.input_qubits == 8
assert vector_est.width.allocated_qubits == 0
assert vector_est.gates.total == 8
assert vector_est.parameters == {}

# %% [markdown]
# Widths, indices, and arities are resource requirements, not hints. The estimator retains requirements for nonnegative integer allocations, array accesses and views, control indices, SELECT index width and nonempty target width, Pauli support, and block-encoding signal/system registers. A descriptor-produced block encoding carries its exact one-dimensional quantum-port widths, so `encoding.unitary.estimate_resources()` applies them automatically. Supplying `inputs={"signal": encoding.num_signal_qubits, "system": encoding.num_system_qubits}` remains useful for wrapper qkernels and explicitly validates the same contract. Requirements are checked both when `inputs` are supplied and after `.substitute()`, including through controlled and inverse forms. Invalid values raise `ValueError` instead of being clamped into a plausible-looking estimate. Serialized output exposes these checks in the `requirements` list.
#
# Supply concrete structural values through `inputs` in the original `estimate_resources()` call whenever possible. This lets the dependency scheduler resolve physical array indices, views, and loop-carried wire identities before it builds the depth expression. Calling `.substitute()` later safely evaluates the symbolic estimate and validates retained requirements, but it cannot rebuild an already conservative alias schedule; its depth quality may therefore remain `conservative` even after every symbol is concrete.

# %% [markdown]
# ## `ResourceEstimate` Fields Reference
#
# | Field | Description |
# |-------|------------|
# | `est.qubits` | Peak live logical width, including decomposition ancillas |
# | `est.circuit_qubits` | Conservative static circuit width |
# | `est.width.input_qubits` | Qubits supplied by the caller |
# | `est.width.allocated_qubits` | Qubits allocated by the qkernel body |
# | `est.width.clean_ancilla_qubits` | Reusable clean decomposition ancillas |
# | `est.gates.total` | Total gate count |
# | `est.gates.single_qubit` | Single-qubit gates |
# | `est.gates.two_qubit` | Two-qubit gates |
# | `est.gates.multi_qubit` | Multi-qubit gates (3+ qubits) |
# | `est.gates.t` | Logical T-gate family count |
# | `est.gates.toffoli` | Logical Toffoli-gate family count |
# | `est.gates.clifford` | Logical Clifford-gate family count |
# | `est.gates.rotation` | Logical rotation-gate family count |
# | `est.gates.non_clifford` | Logical non-Clifford-gate family count |
# | `est.measurements.total` | Per-qubit measurement events in one logical execution |
# | `est.resets.total` | Explicit per-qubit reset events in one logical execution |
# | `est.depth.depth` | Complete dependency-aware algorithmic depth |
# | `est.depth.gate_depth` | Gate-only algorithmic depth |
# | `est.depth.measurement_depth` | Measurement-only algorithmic depth |
# | `est.depth.reset_depth` | Reset-only algorithmic depth |
# | `est.calls.calls_by_name` | Unexpanded named semantic-boundary calls |
# | `est.calls.queries_by_name` | Algorithmic query complexity by name |
# | `est.parameters` | Dict of symbol names → SymPy symbols |
# | `est.control_decomposition` | Selected coherent-control model (`abstract` or `clean_ancilla_toffoli`) |
# | `est.derivation` | `structural` or `modeled`: how the counts were obtained |
# | `est.quality` | `exact`, `conservative`, or `unknown`: how the counts relate to the selected circuit cost |
# | `est.approximation` | `exact` or `approximate`: whether the estimator recognized a mathematical approximation |
# | `est.assumptions` | Active modeling assumptions made by the estimator |
# | `est.trace` / `est.explain()` | Optional explanation tree and its text rendering |
#
# Numeric resource fields are SymPy expressions. For fixed qkernels they evaluate to plain integers. The `clifford`, `t`, `rotation`, `toffoli`, and `non_clifford` fields are actively computed classifications inside the current logical algorithmic model. They are not another arity partition: T, Toffoli, and rotation counts are included in `non_clifford`, so the family fields must not be summed to reconstruct `total`. They neither describe a lowering into another gate set nor exist as scaffolding for a future resource model. A future basis-specific model can define its own result contract without depending on these fields remaining in the logical result. `derivation` says how the resource counts were obtained, `quality` says what relation they have to the selected circuit cost, and `approximation` independently records mathematical approximations recognized by the estimator. Its `exact` value means “no known approximation was recorded,” not a proof about opaque semantics. Measurements and resets are not gates: measuring an `N`-qubit vector contributes `N` to `measurements.total`, while parallel readout can still contribute only one `measurement_depth` layer. Counts describe one logical qkernel execution and are not multiplied by shots. `qmc.expval` leaves `measurements.total` at zero because observable grouping, basis rotations, and shots are executor-dependent; here zero means “not included in this estimate,” not “no measurement is required.” Its modeled derivation, unknown quality, assumption, abstract query, and measurement layer expose that uncertainty. `resets.total` counts explicit `qmc.reset` operations, not fresh `|0>` allocation or target-dependent resets inserted by an engine. The category depths are scheduled independently, so they must not be subtracted from or summed to reconstruct `depth.depth`.
#
# `calls_by_name` deliberately does **not** count ordinary body-backed qkernel calls: those bodies have already been expanded into gates, width, depth, measurements, and resets. It records named boundaries that remain unexpanded in the selected model. These include explicit or unknown opaque calls and modeled semantic boundaries such as `qmc.expval`, whose concrete sampling implementation is executor-dependent. One `expval` therefore records `calls_by_name={"expval": 1}` and `queries_by_name={"expval": 1}` in addition to its modeled measurement layer.

# %% [markdown]
# ## Opaque Boundaries, Model Settings, and Traces
#
# A bodyless callable has no honest gate cost unless you provide one. The default `UnknownResourcePolicy.ERROR` therefore raises. Prefer declaring an explicit `ResourceEstimate` cost on `qmc.opaque(...)` when one is known. For exploratory work, `OPAQUE_CALL` records a named call and query with `derivation=MODELED` and `quality=UNKNOWN`, while `ZERO_WITH_WARNING` records a zero-cost assumption with the same metadata. Neither policy pretends the unknown body was decomposed.
#
# A fixed `ResourceEstimate` and a context-dependent `cost(ctx)` callback share one contract: each completely describes the base cost of applying the Oracle definition once. That base cost already includes controls declared by `qmc.opaque(..., num_control_qubits=...)`, but it excludes controls added later with `qmc.control(oracle, ...)` and controls inherited from a surrounding controlled qkernel. The estimator applies inverse and those external controls after receiving either form of base cost. The cost author must include any phase-relevant work that those later coherent controls need to transform; the estimator does not infer an omitted global-phase contribution.
#
# `GateResources` has no separate scalar field for an opaque definition's hidden phase. If a nontrivial phase is part of a bodyless Oracle and that Oracle may later be coherently controlled, represent the phase-relevant work as a logical primitive in the declared profile: increment `total` and its arity bucket (normally `single_qubit`), and use `rotation` when that family is known. The controlled aggregate envelope then includes that entry. Using the one-qubit bucket for an intrinsically target-free phase is deliberately an upper-bound representative, not an exact reconstruction: it may apply one more level of control than an angle-aware phase lowering would need. This is one reason the complete aggregate result is `CONSERVATIVE`. An identity phase needs no entry. If the exact phase angle must determine the logical operation, use a body-backed `qmc.global_phase(...)` path instead of an angle-free aggregate profile.
#
# The callback receives an `OpaqueCostContext`, not the complete call site. It can read `target_qubits`, per-target `target_shapes`, `definition_control_qubits`, `control_decomposition`, and an optional base `strategy`. Added and inherited controls, inverse, and open-control values are deliberately absent: the callback describes only the definition-level base cost, while the estimator has sole responsibility for those call-site transforms. As with every user-supplied cost, the callback author remains responsible for returning a base cost that follows this contract.
#
# A fixed cost containing gate-model-sensitive fields also fixes the `control_decomposition` under which those numbers were calculated. Estimating that Oracle with another control decomposition raises an error instead of silently relabeling the same gate numbers. A cost containing only calls and queries can be reused across control models because there is no gate breakdown to reinterpret. A callback can follow the active control model by constructing its result with `control_decomposition=ctx.control_decomposition`.
#
# | Control source | Visible to the callback? | Who includes its cost? |
# |---|---|---|
# | Declared by `num_control_qubits` when the Oracle is defined | `ctx.definition_control_qubits` | The fixed cost or callback includes it in the base cost. |
# | Added later by `qmc.control(oracle, ...)` | No | The estimator applies it after receiving the base cost. |
# | Inherited from a surrounding controlled qkernel | No | The estimator applies it after receiving the base cost. |
#
# With either explicit cost form, only the `calls_by_name` and `queries_by_name` entries supplied in that cost are recorded; the estimator does not add another Oracle call or query. A base cost containing measurements or resets can be used only by an uncontrolled, non-inverted definition. Declared, added, or inherited coherent controls and inverse all reject such a cost because counts alone do not define those transformed implementations.
#
# Under `ABSTRACT`, external controls keep `total` unchanged and move known arity buckets: a one-qubit gate becomes two-qubit under one control and multi-qubit under two or more. Unclassified arity remains unclassified. Under `CLEAN_ANCILLA_TOFFOLI`, a logical profile with known `single_qubit` or `two_qubit` gates supports a decomposed estimate. With at least two modeled operations and at least two external controls, this estimation model computes the controls' AND once, projects every known primitive under that one effective control, and uncomputes the shared ladder after the body. A one-operation or one-control profile keeps the per-primitive decomposition. `CLEAN_ANCILLA_TOFFOLI` names a fixed resource-estimation model; its formulas do not automatically change when an engine's emission policy changes.
#
# Aggregate profiles have no gate names or original schedule. Their arity and gate-family fields are therefore independent field-wise bounds and need not sum to `total`; unclassified gates are not mislabeled as `multi_qubit`. A logical profile is complete for `CLEAN_ANCILLA_TOFFOLI` when `total == single_qubit + two_qubit` and no measurement or reset is present. The estimator applies an upper envelope over every supported gate in each declared arity and serializes the projected depth, so a nonempty complete profile whose source quality is `EXACT` or `CONSERVATIVE` has `derivation=MODELED` and `quality=CONSERVATIVE`. A source `quality=UNKNOWN` is never strengthened and remains `UNKNOWN` after the same projection. For `ABSTRACT`, completeness instead means `total == single_qubit + two_qubit + multi_qubit`, because this mode shifts a declared multi-qubit operation without decomposing it. A positive remainder or an undecomposed multi-qubit gate under `CLEAN_ANCILLA_TOFFOLI` leaves the result `UNKNOWN`. A calls/query-only cost has no gate profile to transform, so those counters remain unchanged with a visible assumption. Use a body-backed callable when gate-specific transformed costs are required.


# %%
complete_profile = qmc.ResourceEstimate(
    gates=qmc.GateResources(
        total=3,
        single_qubit=2,
        two_qubit=1,
    ),
)
complete_controlled = complete_profile.controlled(2)

assert complete_controlled.gates.total == 7
assert complete_controlled.depth.depth == 7
assert complete_controlled.width.clean_ancilla_qubits == 2
assert complete_controlled.derivation is qmc.EstimateDerivation.MODELED
assert complete_controlled.quality is qmc.EstimateQuality.CONSERVATIVE

unknown_source = replace(
    complete_profile,
    quality=qmc.EstimateQuality.UNKNOWN,
)
assert unknown_source.controlled(2).quality is qmc.EstimateQuality.UNKNOWN


# %% [markdown]
# This complete profile uses one shared two-control ladder. Its compute and uncompute steps cost two Toffolis, the two one-qubit entries cost two controlled gates, and the conservative two-qubit envelope costs three gates, for `2 + 2 + 3 = 7`. Gate names and the original schedule are still absent, so the result is conservative rather than exact.


# %%
costed_oracle = qmc.opaque(
    "costed_oracle",
    num_qubits=2,
    cost=qmc.ResourceEstimate(
        gates=qmc.GateResources(
            total=5,
            single_qubit=2,
            two_qubit=1,
        ),
        calls=qmc.CallResources(
            queries_by_name={"costed_oracle": 1},
        ),
    ),
)


@qmc.qkernel
def controlled_costed_oracle() -> tuple[qmc.Qubit, qmc.Qubit]:
    control_0 = qmc.qubit("control_0")
    control_1 = qmc.qubit("control_1")
    target_0 = qmc.qubit("target_0")
    target_1 = qmc.qubit("target_1")
    *_, target_0, target_1 = qmc.control(
        costed_oracle,
        num_controls=2,
    )(control_0, control_1, target_0, target_1)
    return target_0, target_1


# %%
costed_est = controlled_costed_oracle.estimate_resources()
assert costed_est.gates.total == 9
assert costed_est.depth.depth == 9
assert costed_est.width.clean_ancilla_qubits == 2
assert costed_est.calls.queries_by_name == {"costed_oracle": 1}
assert costed_est.derivation is qmc.EstimateDerivation.MODELED
assert costed_est.quality is qmc.EstimateQuality.UNKNOWN
assert any(
    "unresolved portion may include gates with unclassified arity" in assumption.message
    for assumption in costed_est.assumptions
)


# %% [markdown]
# This profile has five modeled operations and two controls, so the fixed recipe uses one shared ladder. Computing and uncomputing the controls' AND costs two Toffolis. Beneath the resulting effective control, the two one-qubit entries cost two gates, the conservative two-qubit envelope costs three gates, and the two unresolved placeholders remain two unit-cost operations. The total is therefore `2 + 2 + 3 + 2 = 9`. The two clean ancillas combine one ladder ancilla with the largest workspace needed inside the conservative two-qubit envelope.


# %%
oracle = qmc.opaque("conditional_oracle", num_qubits=1)


@qmc.qkernel
def conditional_resource_branch(flag: qmc.UInt) -> qmc.Qubit:
    target = qmc.qubit("target")
    if flag:
        target = qmc.h(target)
    else:
        (target,) = oracle(target)
    return target


# %%
conditional_est = conditional_resource_branch.estimate_resources(
    unknown_policy=qmc.UnknownResourcePolicy.OPAQUE_CALL,
    trace=True,
)
exact_branch = conditional_est.substitute(flag=1)
opaque_branch = conditional_est.substitute(flag=0)

assert exact_branch.calls.calls_by_name == {}
assert exact_branch.derivation is qmc.EstimateDerivation.STRUCTURAL
assert exact_branch.quality is qmc.EstimateQuality.EXACT
assert "conditional_oracle" not in exact_branch.explain()
assert opaque_branch.calls.calls_by_name == {"conditional_oracle": 1}
assert opaque_branch.calls.queries_by_name == {"conditional_oracle": 1}
assert opaque_branch.derivation is qmc.EstimateDerivation.MODELED
assert opaque_branch.quality is qmc.EstimateQuality.UNKNOWN
assert "conditional_oracle" in opaque_branch.explain()

# %% [markdown]
# Assumptions, derivation/quality facts, call summaries, and trace nodes carry the same symbolic branch guards as the numeric metrics. Once `inputs` or `.substitute()` selects a branch, inactive warnings and opaque calls disappear, as above. Set `trace=True` only when you need the explanation tree; estimates stay compact by default. `est.explain()` renders the retained recursive body/primitive/opaque explanation.

# %% [markdown]
# ### JSON-friendly output
#
# `to_dict()` produces a JSON-friendly report snapshot. Symbolic expressions and structural requirements are stored as strings, alongside the control decomposition, derivation, quality, approximation status, and assumptions. The opt-in trace is intentionally rendered separately with `explain()` rather than embedded in this compact payload. This snapshot is not a round-trip `ResourceEstimate` serialization format: its strings can contain Qamomile-specific symbolic nodes and should not be evaluated with `sympy.sympify()`. To export a concrete report, specialize the original estimate with `.substitute(...)` first. Persist an unbound qkernel, including supported fixed opaque costs, with `qamomile.circuit.serialization.serialize()` instead.


# %%
import json

payload = json.loads(json.dumps(conditional_est.to_dict()))
assert payload["control_decomposition"] == "clean_ancilla_toffoli"
assert payload["derivation"] == "modeled"
assert payload["quality"] == "unknown"
assert payload["approximation"] == "exact"
assert "requirements" in payload

# %% [markdown]
# ## Scaling Analysis with `.substitute()`
#
# The symbolic expressions tell you the *formula*, but often you want concrete numbers at specific sizes. Use `.substitute()` to evaluate an existing estimate. If a concrete value affects indexing, views, loop structure, or another scheduling decision, prefer passing it as `inputs` while constructing the estimate, as described above.

# %%
for n_val in [4, 8, 16, 32]:
    c = est.substitute(n=n_val)
    print(
        f"n={n_val:2d}: {int(c.gates.total):>3} gates total, {int(c.gates.two_qubit):>2} two-qubit"
    )
    assert int(c.gates.total) == 3 * n_val - 1
    assert int(c.gates.two_qubit) == n_val - 1

# %% [markdown]
# ## Deriving Shor resources from the circuit body
#
# As a practical example, consider order finding, the quantum part of Shor's algorithm.
# `qmc.shor_order_finding()` accepts a base and modulus and returns one qkernel that can both run and estimate resources. The register width comes from `modulus.bit_length()`, so the returned kernel has no artificial `n` argument.

# %%
order_finding = qmc.shor_order_finding(base=2, modulus=15)
shor_est = order_finding.estimate_resources()

print("default peak qubits:", shor_est.qubits)
print("default total gates:", shor_est.gates.total)
print("measurements:", shor_est.measurements.total)
print("resets:", shor_est.resets.total)
print("estimate derivation:", shor_est.derivation)
print("estimate quality:", shor_est.quality)

assert shor_est.parameters == {}
assert shor_est.width.allocated_qubits == 21
assert shor_est.width.clean_ancilla_qubits == 2
assert shor_est.qubits == 23
assert shor_est.gates.total == 4585
assert shor_est.measurements.total == 80
assert shor_est.resets.total == 80
assert shor_est.derivation is qmc.EstimateDerivation.STRUCTURAL
assert shor_est.quality is qmc.EstimateQuality.CONSERVATIVE

# %% [markdown]
# This implementation does not keep a `2*n`-qubit counting register at once. It measures and resets one phase qubit for reuse, applying semiclassical inverse-QFT phase corrections from the previously observed bits. Of the 80 measurement/reset events above, 8 come from that phase readout and reuse, while 72 come from measurement-assisted carry venting inside the arithmetic. All explicit resets appear in `resets.total`, not in `gates.total`.
#
# With a fixed lookup-window width `w`, peak-live allocation in the circuit body gives `3*n + w + 7` logical qubits before control decomposition.
#
# | Purpose | Width |
# |---|---:|
# | Reused phase qubit | `1` |
# | Modular-value work register | `n` |
# | Modular-multiplication accumulator | `n` |
# | Window-lookup output | `n` |
# | Lookup address | `w` |
# | Carry, vent, overflow, reduction, domain, and enable | `6` |
#
# The default is `w=2`, giving a body allocation of `3*n + 9`. For modulus 15, `n=4`, so the body contributes 21 qubits. The default `CLEAN_ANCILLA_TOFFOLI` model also reports two reusable clean ancillas, producing a peak and static circuit width of 23. Selecting `ABSTRACT` keeps the pre-decomposition width of 21. These values come from the executed qkernel body and its transforms, not an external cost formula registered with `estimate_resources()`.

# %%
import sympy as sp
from IPython.display import Math, display

symbolic_n, symbolic_w = sp.symbols("n w", integer=True, positive=True)
shor_body_width = 3 * symbolic_n + symbolic_w + 7
shor_decomposed_width = shor_body_width + 2
display(Math(rf"N_\mathrm{{body}} = {sp.latex(shor_body_width)}"))
display(Math(rf"N_\mathrm{{decomposed}} = {sp.latex(shor_decomposed_width)}"))
assert (
    shor_body_width.subs({symbolic_n: 4, symbolic_w: 2})
    == shor_est.width.allocated_qubits
)
assert shor_decomposed_width.subs({symbolic_n: 4, symbolic_w: 2}) == shor_est.qubits

# %% [markdown]
# The gate breakdown for modulus 15 is derived by traversing that same body to completion.

# %%
print("single-qubit gates:", shor_est.gates.single_qubit)
print("two-qubit gates:", shor_est.gates.two_qubit)
print("multi-qubit gates:", shor_est.gates.multi_qubit)
print("Toffoli gates:", shor_est.gates.toffoli)

assert shor_est.gates.total == (
    shor_est.gates.single_qubit + shor_est.gates.two_qubit + shor_est.gates.multi_qubit
)

# %% [markdown]
# `quality` is `conservative` because branches selected by mid-circuit measurements and classical feed-forward are counted conservatively, and `CLEAN_ANCILLA_TOFFOLI` is a conservative multi-control model. The derivation remains `structural`, and this bound does not by itself make the circuit mathematically approximate. These remain algorithmic circuit resources: they do not include decomposition to device-native gates, routing, error correction, or magic-state production.

# %% [markdown]
# ### Why the gate count is `O(n^3)`
#
# `qmc.modmul_const()` reads the source `w` bits at a time, looks up a classical multiple, and adds it into an accumulator. One modular multiplication has about `n / w` windows. The ripple-carry addition, constant subtraction, comparison, and conditional restoration used by every window are each `O(n)` gates. Constant addition and subtraction use [Gidney's carry-venting adder](https://arxiv.org/abs/2507.23079), which borrows the work register as dirty workspace rather than allocating another `n`-qubit register.
#
# Therefore, at fixed `w`, one modular multiplication is `O(n^2)`, and the default order-finding schedule performs `2*n` controlled multiplications for `O(n^3)` total cost. Semiclassical inverse-QFT feed-forward is `O(n^2)` and does not change the leading order. Including the lookup dependence gives approximately `O(2^w * n^3 / w)`.

# %% [markdown]
# ### Estimating a modular multiplication from the same body
#
# `qmc.modmul_const()` is the public primitive. FTQC arithmetic is specialized to a concrete problem instance before construction, so here we estimate the width-four kernel directly.


# %%
@qmc.qkernel
def modular_multiplier() -> qmc.Vector[qmc.Qubit]:
    reg = qmc.qubit_array(4, name="reg")
    return qmc.modmul_const(
        reg,
        multiplier=2,
        modulus=15,
        window_size=2,
    )


# %%
window_est = modular_multiplier.estimate_resources()
print("windowed arithmetic qubits:", window_est.qubits)
print("windowed arithmetic gates:", window_est.gates.total)
assert window_est.width.allocated_qubits == 3 * 4 + 2 + 7
assert window_est.width.clean_ancilla_qubits == 2
assert window_est.qubits == 3 * 4 + 2 + 9
assert window_est.gates.total == 2272
assert window_est.measurements.total == 36
assert window_est.resets.total == 36

# %% [markdown]
# In the standalone primitive, an internal control for the unconditional case takes the phase-qubit role, so its body has the same `3*n + w + 7` allocation as the full order-finding circuit. The default peak is two qubits larger because of clean control-decomposition ancillas. For `x < modulus`, `modmul_const()` implements `|x> -> |a*x mod modulus>`; it leaves basis states outside that domain unchanged to preserve unitarity.

# %% [markdown]
# ### Ekerå–Håstad short-exponent schedule
#
# `qmc.ekera_hastad_factoring()` constructs the short-discrete-logarithm quantum stage of the [Ekerå–Håstad method](https://arxiv.org/abs/1702.00249) for factoring a product of two similarly sized primes. Qamomile sets `m = ceil(n / 2) + 1` and measures schedules of lengths `2*m` and `m` sequentially.
#
# Rather than retain two exponent registers coherently, it reuses the same phase qubit and arithmetic workspace. Its body allocation is consequently the same `3*n + w + 7` as Shor's order finding, with the same additional clean ancillas under `CLEAN_ANCILLA_TOFFOLI`; the number of controlled modular multiplications differs. The returned `Vector[Bit]` contains the first `2*m` little-endian bits from the long schedule and the remaining `m` bits from the short schedule.
#
# The modulus-5 instance below is a compact resource-estimation fixture, not a complete factoring example. A factoring application supplies the composite modulus whose two prime factors are to be recovered.

# %%
short_dlp = qmc.ekera_hastad_factoring(
    generator=2,
    modulus=5,
    window_size=2,
)
short_dlp_est = short_dlp.estimate_resources()

print("Ekerå–Håstad default qubits:", short_dlp_est.qubits)
print("Ekerå–Håstad default gates:", short_dlp_est.gates.total)
assert short_dlp_est.width.allocated_qubits == 3 * 3 + 2 + 7
assert short_dlp_est.width.clean_ancilla_qubits == 2
assert short_dlp_est.qubits == 3 * 3 + 2 + 9
assert short_dlp_est.gates.total == 4950
assert short_dlp_est.measurements.total == 81
assert short_dlp_est.resets.total == 81
assert short_dlp.output_types == [qmc.Vector[qmc.Bit]]

# %% [markdown]
# ## Summary
#
# - `estimate_resources()` reports logical algorithmic width, gates, measurement/reset events, and depth without executing. Other resource models can be added through separate interfaces when implemented.
# - Coherent controls use `clean_ancilla_toffoli` by default; select `abstract` to retain controlled source primitives without decomposition.
# - Calls, inverse calls, SELECT, global phase, Pauli evolution, and control flow compose from their bodies and semantics instead of collapsing to one gate.
# - `expval` is reported as a modeled abstract query and measurement layer; grouping, basis-change, and shot costs are deliberately left to the selected executor.
# - For parameterized qkernels, results are SymPy expressions showing scaling within the selected model.
# - `inputs` can supply classical values, array shapes, and an integer width for a one-dimensional quantum Vector; retained requirements reject invalid widths and indices.
# - Check `control_decomposition`, `derivation`, `quality`, `approximation`, `assumptions`, and opt-in traces before interpreting a result. `derivation` identifies structural versus modeled counts, `quality` classifies their relation to the logical circuit cost, and `approximation` independently records a mathematical approximation known to the estimator. Condition selection removes inactive metadata.
# - `calls_by_name` describes unexpanded named boundaries, including opaque calls and modeled semantic operations such as `expval`. Body-backed calls are recursively expanded. Fixed and callback opaque costs both describe one base Oracle application; the estimator applies later-added and inherited controls, projects the known one-/two-qubit portion through the selected control decomposition, and keeps any remaining gates as visible modeled placeholders.
# - `to_dict()` exports a display/report snapshot; use `.substitute(...)` on the original estimate before exporting concrete values.
# - Use `.substitute(n=...)` to evaluate an existing estimate at specific sizes and check feasibility; use initial `inputs` when concrete structure should sharpen dependency scheduling.
# - The FTQC Shor and Ekerå–Håstad factories share the same `O(n^2)` windowed modular-multiplication body and one reused phase qubit.
# - At fixed window width, the circuit body allocates `3*n + w + 7` qubits; the default clean-ancilla Toffoli decomposition adds up to two reusable clean ancillas, and Shor's logical gate count is `O(n^3)`.
# - Problem-specialized FTQC factories expose concrete width and gate estimates derived from their executable bodies.
#
# **Next**: [Execution Models](06_execution_models.ipynb) — `sample()` vs `run()`, observables, and bit ordering.
