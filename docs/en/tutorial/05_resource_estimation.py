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
# Before running a quantum kernel on real hardware, you may want to know its required resources, such as qubit count and gate count. Qamomile's `estimate_resources()` fills this need **without executing the qkernel**. It reports algorithmic resources rather than hardware-native gates: the default `portable` basis follows Qamomile's backend-neutral control decomposition, but does not apply target-specific optimization. It works with both concrete and symbolic (parameterized) qkernels.
#
# This chapter covers:
#
# - Basic resource estimation for fixed qkernels
# - Choosing between `portable`, `logical`, and `clifford_t` models
# - How controls, inverse calls, SELECT, Pauli evolution, and control flow compose
# - Symbolic resource estimation for parameterized qkernels
# - Structural requirements, opaque boundaries, traces, and JSON-friendly output
# - Scaling analysis with `.substitute()`
# - Applying body-derived estimates to Shor order finding

# %%
# Install the latest Qamomile through pip!
# # !pip install "qamomile[qiskit,visualization]"

# %%
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

# %% [markdown]
# ## Choosing the Estimation Model
#
# The default `portable` basis recursively inspects called qkernels. When coherent controls are present, it applies Qamomile's backend-neutral fallback to each primitive and includes reusable clean ancillas. For example, a three-controlled H uses four Toffoli gates, one controlled-H gate, and two clean ancillas.


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
portable = controlled_h.estimate_resources()
assert portable.basis is qmc.GateBasis.PORTABLE
assert portable.gates.total == 5
assert portable.gates.toffoli == 4
assert portable.width.clean_ancilla_qubits == 2
assert portable.qubits == 6
assert portable.quality is qmc.EstimateQuality.UPPER_BOUND

abstract = controlled_h.estimate_resources(basis=qmc.GateBasis.LOGICAL)
assert abstract.gates.total == 1
assert abstract.width.clean_ancilla_qubits == 0
assert abstract.qubits == 4

# %% [markdown]
# `portable` mirrors Qamomile's backend-neutral control fallback at the algorithmic level. When a concrete controlled body contains enough work, its gates share one computed control conjunction; isolated primitives and symbolic structures retain a conservative per-primitive fallback. A backend with a native multi-controlled gate may use fewer resources. Select `logical` when you intentionally want the higher-level view in which each source primitive remains one abstract gate regardless of control count. Select `clifford_t` for aggregate Clifford+T resources of supported operations; `precision` controls arbitrary-rotation synthesis. Unsupported Clifford+T lowering raises an error instead of inventing a cost. None of these models performs routing, hardware-native optimization, or error-correction costing.


# %%
import math


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
phase_est = controlled_phase.estimate_resources(
    basis=qmc.GateBasis.CLIFFORD_T,
    precision=1e-3,
)
assert phase_est.substitute(theta=0).gates.total == 0
assert phase_est.substitute(theta=math.pi / 4).gates.t == 1
assert phase_est.substitute(theta=0.3).quality is qmc.EstimateQuality.UPPER_BOUND

# %% [markdown]
# A standalone global phase is unobservable and costs zero in this target-neutral model. Under coherent control it becomes a relative phase, as the example shows. Substitution classifies canonical angles such as zero, Z, S, and T exactly; an arbitrary angle uses the requested synthesis model.
#
# Higher-level operations are interpreted from their executable meaning rather than counted as one opaque box:
#
# | Construct | Estimation behavior |
# |---|---|
# | Body-backed qkernel calls and `qmc.inverse(...)` | Recursively traverse the selected implementation body. Inversion preserves gate counts while reversing the unitary semantics; it does not collapse a large body to one call. |
# | `qmc.control(...)` | Propagate every surrounding control into nested bodies. In `portable`, multi-control fallback gates and reusable clean ancillas are included. Open controls also include their X brackets. |
# | `qmc.select(...)` | Sum every emitted controlled case body, including index-bit controls, open-control brackets, vector broadcast, and surrounding controls. The declared index width must match its operands and address all cases. |
# | `qmc.pauli_evolve(...)` | For a supplied Hermitian Hamiltonian, count Pauli-basis changes, parity ladders, axial rotations, and any controlled constant-term phase. The target register must cover the Hamiltonian support. |
# | `if`, `for`, `for_items`, and `while` | Keep compile-time conditions and loop bounds symbolic when possible. Gate and width metrics for a parameter branch use `Piecewise`; a measurement-backed branch uses a conservative maximum. Dependency depth can remain an `upper_bound` when symbolic aliases or multi-wire boundaries prevent an exact schedule. Loop work accumulates while reusable width follows liveness. Unsupported carried recurrences fail explicitly or add a visible assumption. |
# | `qmc.expval(...)` | Record one abstract expectation query and one measurement layer. Observable grouping, basis rotations, shot count, and executor sampling policy remain backend/executor-dependent, so the result is `modeled` with a visible assumption rather than a fabricated gate count. |
# | LCU block encodings | Preserve exact signal- and system-register requirements through direct, controlled, inverse, descriptor-bound, and serialized forms. |

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
# The output contains SymPy expressions like `n` for qubits and `2*n + Max(0, n - 1)` for total gates. These expressions are exact within the selected estimation model. Inspect `est.quality` for estimates that use a conservative bound or an explicit model.
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
# Widths, indices, and arities are resource requirements, not hints. The estimator retains requirements for nonnegative integer allocations, array accesses and views, control indices, SELECT index width, Pauli support, and block-encoding signal/system registers. A descriptor-produced block encoding carries its exact one-dimensional quantum-port widths, so `encoding.unitary.estimate_resources()` applies them automatically. Supplying `inputs={"signal": encoding.num_signal_qubits, "system": encoding.num_system_qubits}` remains useful for wrapper qkernels and explicitly validates the same contract. Requirements are checked both when `inputs` are supplied and after `.substitute()`, including through controlled and inverse forms. Invalid values raise `ValueError` instead of being clamped into a plausible-looking estimate. Serialized output exposes these checks in the `requirements` list.
#
# Supply concrete structural values through `inputs` in the original `estimate_resources()` call whenever possible. This lets the dependency scheduler resolve physical array indices, views, and loop-carried wire identities before it builds the depth expression. Calling `.substitute()` later safely evaluates the symbolic estimate and validates retained requirements, but it cannot rebuild an already conservative alias schedule; its depth may therefore remain an `upper_bound` even after every symbol is concrete.

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
# | `est.gates.t_gates` | T-gate count |
# | `est.gates.clifford_gates` | Clifford gate count |
# | `est.gates.rotation_gates` | Rotation gate count |
# | `est.depth.depth` | Dependency-aware algorithmic depth |
# | `est.calls.calls_by_name` | Bodyless/opaque boundary calls by name |
# | `est.calls.queries_by_name` | Opaque query complexity by name |
# | `est.parameters` | Dict of symbol names → SymPy symbols |
# | `est.basis` | Selected gate basis (`portable`, `logical`, or `clifford_t`) |
# | `est.precision` | Rotation-synthesis precision for `clifford_t` |
# | `est.quality` | `exact`, `upper_bound`, or `modeled` |
# | `est.assumptions` | Active modeling assumptions made by the estimator |
# | `est.trace` / `est.explain()` | Optional explanation tree and its text rendering |
#
# Numeric resource fields are SymPy expressions. For fixed qkernels they evaluate to plain integers. `calls_by_name` deliberately does **not** count ordinary body-backed qkernel calls: those bodies have already been expanded into gates, width, and depth. It records only opaque boundaries, either from an explicit opaque cost or from an unknown-call policy.

# %% [markdown]
# ## Opaque Boundaries, Condition-Aware Provenance, and Traces
#
# A bodyless callable has no honest gate cost unless you provide one. The default `UnknownResourcePolicy.ERROR` therefore raises. Prefer declaring an explicit `ResourceEstimate` cost on `qmc.opaque(...)` when one is known. For exploratory work, `OPAQUE_CALL` records a named call and query with `modeled` quality, while `ZERO_WITH_WARNING` records a zero-cost assumption. Neither policy pretends the unknown body was decomposed.
#
# A fixed opaque cost can still support a useful controlled estimate when its `portable` gate total is completely partitioned into `single_qubit` and `two_qubit` counts. Qamomile then controls every counted primitive with a conservative gate-kind upper bound, serializes their depth through the shared controls, and reports additional clean ancillas. Arity alone cannot distinguish, for example, X from H or CX from SWAP, nor can it reveal an undeclared global phase that becomes observable under control. The result therefore remains `modeled` and records these assumptions. If the arity profile is incomplete or contains three-or-more-qubit gates, the declared cost stays unchanged and the missing controlled overhead is explicit in `assumptions`; use a context-dependent `cost(ctx)` model when gate-specific control costs or phase behavior are known. A callback is authoritative for the complete invocation and can price all coherent controls exactly once through `ctx.total_controls`.


# %%
costed_oracle = qmc.opaque(
    "costed_oracle",
    num_qubits=2,
    cost=qmc.ResourceEstimate(
        gates=qmc.GateResources(
            total=3,
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
assert costed_est.gates.total == 13
assert costed_est.width.clean_ancilla_qubits == 2
assert costed_est.calls.queries_by_name == {"costed_oracle": 1}
assert costed_est.quality is qmc.EstimateQuality.MODELED
assert any(
    "complete one- and two-qubit counts" in assumption.message
    for assumption in costed_est.assumptions
)


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
assert exact_branch.quality is qmc.EstimateQuality.EXACT
assert "conditional_oracle" not in exact_branch.explain()
assert opaque_branch.calls.calls_by_name == {"conditional_oracle": 1}
assert opaque_branch.calls.queries_by_name == {"conditional_oracle": 1}
assert opaque_branch.quality is qmc.EstimateQuality.MODELED
assert "conditional_oracle" in opaque_branch.explain()

# %% [markdown]
# Assumptions, quality, call summaries, and trace nodes carry the same symbolic branch guards as the numeric metrics. Once `inputs` or `.substitute()` selects a branch, inactive warnings and opaque calls disappear, as above. Set `trace=True` only when you need the explanation tree; estimates stay compact by default. `est.explain()` renders the retained recursive body/primitive/opaque provenance.

# %% [markdown]
# ### JSON-friendly output
#
# `to_dict()` produces a JSON-friendly snapshot. Symbolic expressions and structural requirements are stored as strings, alongside basis, precision, quality, and assumptions. The opt-in trace is intentionally rendered separately with `explain()` rather than embedded in this compact payload.


# %%
import json

payload = json.loads(json.dumps(conditional_est.to_dict()))
assert payload["basis"] == "portable"
assert payload["quality"] == "modeled"
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

print("portable peak qubits:", shor_est.qubits)
print("portable total gates:", shor_est.gates.total)
print("estimate quality:", shor_est.quality)

assert shor_est.parameters == {}
assert shor_est.width.allocated_qubits == 21
assert shor_est.width.clean_ancilla_qubits == 2
assert shor_est.qubits == 23
assert shor_est.gates.total == 4665
assert str(shor_est.quality) == "upper_bound"

# %% [markdown]
# This implementation does not keep a `2*n`-qubit counting register at once. It measures and resets one phase qubit for reuse, applying semiclassical inverse-QFT phase corrections from the previously observed bits.
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
# The default is `w=2`, giving a body allocation of `3*n + 9`. For modulus 15, `n=4`, so the body contributes 21 qubits. The default `portable` model also reports two reusable clean ancillas needed by its multi-control fallback, producing a peak and static circuit width of 23. The explicit `logical` model keeps the pre-decomposition width of 21. These values come from the executed qkernel body and its transforms, not an external cost formula registered with `estimate_resources()`.

# %%
import sympy as sp
from IPython.display import Math, display

symbolic_n, symbolic_w = sp.symbols("n w", integer=True, positive=True)
shor_body_width = 3 * symbolic_n + symbolic_w + 7
shor_portable_width = shor_body_width + 2
display(Math(rf"N_\mathrm{{body}} = {sp.latex(shor_body_width)}"))
display(Math(rf"N_\mathrm{{portable}} = {sp.latex(shor_portable_width)}"))
assert (
    shor_body_width.subs({symbolic_n: 4, symbolic_w: 2})
    == shor_est.width.allocated_qubits
)
assert shor_portable_width.subs({symbolic_n: 4, symbolic_w: 2}) == shor_est.qubits

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
# `quality` is `upper_bound` because branches selected by mid-circuit measurements and classical feed-forward are counted conservatively, and `portable` uses a conservative multi-control fallback. These remain algorithmic circuit resources: they do not include decomposition to device-native gates, routing, error correction, or magic-state production.

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
assert window_est.gates.total == 2308

# %% [markdown]
# In the standalone primitive, an internal control for the unconditional case takes the phase-qubit role, so its body has the same `3*n + w + 7` allocation as the full order-finding circuit. The default `portable` peak is two qubits larger because of clean control-decomposition ancillas. For `x < modulus`, `modmul_const()` implements `|x> -> |a*x mod modulus>`; it leaves basis states outside that domain unchanged to preserve unitarity.

# %% [markdown]
# ### Ekerå–Håstad short-exponent schedule
#
# `qmc.ekera_hastad_factoring()` constructs the short-discrete-logarithm quantum stage of the [Ekerå–Håstad method](https://arxiv.org/abs/1702.00249) for factoring a product of two similarly sized primes. Qamomile sets `m = ceil(n / 2) + 1` and measures schedules of lengths `2*m` and `m` sequentially.
#
# Rather than retain two exponent registers coherently, it reuses the same phase qubit and arithmetic workspace. Its body allocation is consequently the same `3*n + w + 7` as Shor's order finding, with the same additional clean ancillas in `portable`; the number of controlled modular multiplications differs. The returned `Vector[Bit]` contains the first `2*m` little-endian bits from the long schedule and the remaining `m` bits from the short schedule.
#
# The modulus-5 instance below is a compact resource-estimation fixture, not a complete factoring example. A factoring application supplies the composite modulus whose two prime factors are to be recovered.

# %%
short_dlp = qmc.ekera_hastad_factoring(
    generator=2,
    modulus=5,
    window_size=2,
)
short_dlp_est = short_dlp.estimate_resources()

print("Ekerå–Håstad portable qubits:", short_dlp_est.qubits)
print("Ekerå–Håstad portable gates:", short_dlp_est.gates.total)
assert short_dlp_est.width.allocated_qubits == 3 * 3 + 2 + 7
assert short_dlp_est.width.clean_ancilla_qubits == 2
assert short_dlp_est.qubits == 3 * 3 + 2 + 9
assert short_dlp_est.gates.total == 5031
assert short_dlp.output_types == [qmc.Vector[qmc.Bit]]

# %% [markdown]
# ## Summary
#
# - `estimate_resources()` reports algorithmic qubit and gate costs without executing.
# - The default `portable` basis recursively expands coherent controls and reports required clean ancillas; `logical` retains the abstract source-gate view, and `clifford_t` applies the supported synthesis model.
# - Calls, inverse calls, SELECT, global phase, Pauli evolution, and control flow compose from their bodies and semantics instead of collapsing to one gate.
# - `expval` is reported as a modeled abstract query and measurement layer; grouping, basis-change, and shot costs are deliberately left to the selected executor.
# - For parameterized qkernels, results are SymPy expressions showing scaling within the selected model.
# - `inputs` can supply classical values, array shapes, and an integer width for a one-dimensional quantum Vector; retained requirements reject invalid widths and indices.
# - Check `basis`, `quality`, `assumptions`, and opt-in traces before interpreting a result as an exact implementation cost. Condition selection removes inactive provenance.
# - `calls_by_name` describes opaque boundaries only. Body-backed calls are recursively expanded; a fixed opaque cost with a complete one-/two-qubit profile receives a modeled controlled estimate, while other unknown bodies use an explicit policy or context-dependent cost.
# - `to_dict()` exports symbolic metrics and requirements in a JSON-friendly form.
# - Use `.substitute(n=...)` to evaluate an existing estimate at specific sizes and check feasibility; use initial `inputs` when concrete structure should sharpen dependency scheduling.
# - The FTQC Shor and Ekerå–Håstad factories share the same `O(n^2)` windowed modular-multiplication body and one reused phase qubit.
# - At fixed window width, the circuit body allocates `3*n + w + 7` qubits; the current default `portable` fallback adds up to two reusable clean ancillas, and Shor's default-precision gate count is `O(n^3)`.
# - Problem-specialized FTQC factories expose concrete width and gate estimates derived from their executable bodies.
#
# **Next**: [Execution Models](06_execution_models.ipynb) — `sample()` vs `run()`, observables, and bit ordering.
