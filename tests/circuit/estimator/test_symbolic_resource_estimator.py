"""Tests for the symbolic ResourceEstimator compiler service."""

from __future__ import annotations

import pytest
import sympy as sp

import qamomile.circuit as qm


@qm.qkernel
def _basis_probe(theta: qm.Float) -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
    """Apply one exact Toffoli and one approximate axial rotation."""
    left = qm.qubit("left")
    right = qm.qubit("right")
    target = qm.qubit("target")
    left, right, target = qm.ccx(left, right, target)
    target = qm.rx(target, theta)
    return left, right, target


@qm.qkernel
def _allocation_helper(q: qm.Qubit) -> qm.Qubit:
    """Allocate and measure one helper-local qubit."""
    fresh = qm.qubit("fresh")
    qm.measure(fresh)
    return q


@qm.qkernel
def _distinct_helper_callsites() -> tuple[qm.Bit, qm.UInt]:
    """Select two branch-local helper call sites across range iterations."""
    q = qm.qubit("q")
    total = qm.uint(0)
    for index in qm.range(2):
        if index == 0:
            q = _allocation_helper(q)
        else:
            q = _allocation_helper(q)
        total = total + 1
    return qm.measure(q), total


@qm.qkernel
def _repeated_helper_callsite() -> tuple[qm.Bit, qm.UInt]:
    """Replay one helper call site across two range iterations."""
    q = qm.qubit("q")
    total = qm.uint(0)
    for _index in qm.range(2):
        q = _allocation_helper(q)
        total = total + 1
    return qm.measure(q), total


def test_clifford_t_basis_lowers_body_gates_and_reports_quality() -> None:
    """Basis lowering reports aggregate Clifford+T counts and approximation."""
    logical = _basis_probe.estimate_resources()
    lowered = _basis_probe.estimate_resources(
        basis=qm.GateBasis.CLIFFORD_T,
        precision=1 / 8,
    )

    assert logical.gates.total == 2
    assert logical.quality is qm.EstimateQuality.EXACT
    assert lowered.gates.total == 26
    assert lowered.gates.single_qubit == 20
    assert lowered.gates.two_qubit == 6
    assert lowered.gates.t == 16
    assert lowered.gates.rotation == 0
    assert lowered.quality is qm.EstimateQuality.UPPER_BOUND


def test_pauli_evolve_depth_tracks_gadget_structure() -> None:
    """Pauli evolution depth grows with term support and basis changes."""
    import qamomile.observable as qm_o

    @qm.qkernel
    def circuit(hamiltonian: qm.Observable) -> qm.Vector[qm.Qubit]:
        """Apply one bound two-term Pauli evolution."""
        qubits = qm.qubit_array(2, "qubits")
        return qm.pauli_evolve(qubits, hamiltonian, qm.float_(0.5))

    estimate = circuit.estimate_resources(
        inputs={"hamiltonian": qm_o.X(0) + qm_o.Z(0) * qm_o.Z(1)}
    )

    assert estimate.gates.total == 6
    assert estimate.depth.depth == 6
    assert estimate.depth.clifford_depth == 4
    assert estimate.depth.rotation_depth == 2


def test_gate_basis_accepts_strings_and_rejects_unknown_values() -> None:
    """The public basis option accepts notebook-friendly strings safely."""
    estimate = _basis_probe.estimate_resources(basis="logical")
    assert estimate.gates.total == 2

    with pytest.raises(ValueError, match="expected one of: logical, clifford_t"):
        _basis_probe.estimate_resources(basis="surface_code")


def test_clifford_t_basis_lowers_controlled_toffoli_with_clean_ancilla() -> None:
    """An added control uses the declared clean-ancilla Toffoli ladder."""

    @qm.composite_gate
    def toffoli(
        left: qm.Qubit,
        right: qm.Qubit,
        target: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
        """Apply one Toffoli gate."""
        return qm.ccx(left, right, target)

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit, qm.Qubit]:
        """Apply a Toffoli under one additional control."""
        control = qm.qubit("control")
        left = qm.qubit("left")
        right = qm.qubit("right")
        target = qm.qubit("target")
        controlled_toffoli = qm.control(toffoli)
        return controlled_toffoli(control, left, right, target)

    estimate = circuit.estimate_resources(basis=qm.GateBasis.CLIFFORD_T)

    assert estimate.gates.t == 21
    assert estimate.gates.total == 45
    assert estimate.width.clean_ancilla_qubits == 1
    assert estimate.qubits == 5


def test_clifford_t_basis_rejects_missing_controlled_gate_lowering() -> None:
    """Unsupported controlled gates fail instead of reporting logical counts."""

    @qm.composite_gate
    def hadamard(target: qm.Qubit) -> qm.Qubit:
        """Apply one Hadamard gate."""
        return qm.h(target)

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Apply a controlled Hadamard through a body-backed callable."""
        control = qm.qubit("control")
        target = qm.qubit("target")
        controlled_hadamard = qm.control(hadamard)
        return controlled_hadamard(control, target)

    with pytest.raises(ValueError, match="controlled gate 'h'"):
        circuit.estimate_resources(basis=qm.GateBasis.CLIFFORD_T)


def test_clifford_t_basis_rejects_controlled_rotation_lowering() -> None:
    """Controlled rotations fail when no decomposition contract is defined."""

    @qm.composite_gate
    def rotate(target: qm.Qubit, theta: qm.Float) -> qm.Qubit:
        """Apply one arbitrary Y rotation."""
        return qm.ry(target, theta)

    @qm.qkernel
    def circuit(theta: qm.Float) -> tuple[qm.Qubit, qm.Qubit]:
        """Apply an arbitrary rotation under one control."""
        control = qm.qubit("control")
        target = qm.qubit("target")
        controlled_rotate = qm.control(rotate)
        return controlled_rotate(control, target, theta)

    with pytest.raises(ValueError, match="controlled gate 'ry'"):
        circuit.estimate_resources(basis=qm.GateBasis.CLIFFORD_T)


@qm.qkernel
def _dependency_depth_probe() -> tuple[qm.Qubit, qm.Qubit]:
    """Apply two independent first-layer gates and one dependent gate."""
    left = qm.qubit("left")
    right = qm.qubit("right")
    left = qm.h(left)
    right = qm.x(right)
    left = qm.z(left)
    return left, right


def test_depth_uses_quantum_wire_dependencies() -> None:
    """Independent gates share a layer while dependent gates serialize."""
    estimate = _dependency_depth_probe.estimate_resources()

    assert estimate.gates.total == 3
    assert estimate.depth.depth == 2
    assert estimate.depth.clifford_depth == 2


@qm.qkernel
def _released_qubit_probe() -> qm.Qubit:
    """Measure one allocation before creating a replacement qubit."""
    first = qm.qubit("first")
    _bit = qm.measure(first)
    second = qm.qubit("second")
    return second


def test_width_reuses_affinely_released_qubits() -> None:
    """Measurement ends a wire lifetime before a later allocation."""
    estimate = _released_qubit_probe.estimate_resources()

    assert estimate.width.allocated_qubits == 2
    assert estimate.qubits == 1


def test_trace_is_opt_in_and_honors_the_flag() -> None:
    """Default estimates stay compact while trace=True retains explanations."""

    @qm.qkernel
    def circuit() -> qm.Bit:
        """Apply and measure one H gate."""
        return qm.measure(qm.h(qm.qubit("q")))

    assert circuit.estimate_resources().trace is None
    assert circuit.estimate_resources(trace=False).trace is None
    traced = circuit.estimate_resources(trace=True)
    assert traced.trace is not None
    assert "h" in traced.trace.render()


def test_composite_resources_follow_the_executable_body() -> None:
    """ResourceEstimator derives a named callable's cost from its body."""

    @qm.composite_gate
    @qm.qkernel
    def body(q: qm.Qubit) -> qm.Qubit:
        """Apply two gates."""
        return qm.x(qm.h(q))

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Call the body-backed composite."""
        q = qm.qubit("q")
        q = body(q)
        return q

    estimate = qm.ResourceEstimator().estimate(circuit)
    assert estimate.gates.total == 2


def test_opaque_fixed_cost_counts_calls_and_gates() -> None:
    """Opaque callables may carry an explicit fixed cost."""
    oracle_estimate = qm.ResourceEstimate(
        gates=qm.GateResources(total=7, t=7, non_clifford=7),
        calls=qm.CallResources(
            calls_by_name={"phase_oracle": 1},
            queries_by_name={"phase_oracle": 1},
        ),
    )
    oracle = qm.opaque(
        "phase_oracle",
        num_qubits=1,
        cost=oracle_estimate,
    )

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Call an opaque oracle once."""
        q = qm.qubit("q")
        (q,) = oracle(q)
        return q

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 7
    assert estimate.gates.t == 7
    assert estimate.calls.calls_by_name["phase_oracle"] == 1
    assert estimate.calls.queries_by_name["phase_oracle"] == 1


def test_symbolic_loop_with_loop_dependent_cost_is_summed() -> None:
    """Loop-dependent resource expressions are summed symbolically."""

    @qm.qkernel
    def circuit(n: qm.UInt) -> qm.Qubit:
        """Apply exponentially many gates through explicit loop structure."""
        q = qm.qubit("q")
        for i in qm.range(n):
            for _ in qm.range(2**i):
                q = qm.h(q)
        return q

    estimate = circuit.estimate_resources()
    n = estimate.parameters["n"]

    assert sp.simplify(estimate.gates.total - (2**n - 1)) == 0


def test_zero_trip_offset_loop_has_zero_body_resources() -> None:
    """A UInt value of zero preserves Python's empty-range semantics."""

    @qm.qkernel
    def circuit(n: qm.UInt) -> qm.Qubit:
        """Allocate work qubits only while an offset range executes."""
        q = qm.qubit("q")
        for _ in qm.range(1, n):
            _work = qm.qubit_array(3, "work")
            q = qm.h(q)
        return q

    symbolic = circuit.estimate_resources()
    n = symbolic.parameters["n"]

    assert n.is_nonnegative is True
    assert n.is_positive is None
    assert symbolic.gates.total == sp.Max(0, n - 1)
    assert symbolic.substitute(n=0).gates.total == 0
    assert symbolic.substitute(n=0).qubits == 1
    assert circuit.estimate_resources(inputs={"n": 0}).gates.total == 0
    assert circuit.estimate_resources(inputs={"n": 1}).qubits == 1
    with pytest.raises(ValueError, match="negative"):
        symbolic.substitute(n=-1)
    with pytest.raises(ValueError, match="negative"):
        circuit.estimate_resources(inputs={"n": -1})


def test_direct_zero_trip_loop_disables_reusable_width() -> None:
    """A direct range(n) body contributes no workspace when UInt n is zero."""

    @qm.qkernel
    def circuit(n: qm.UInt) -> qm.Qubit:
        """Allocate temporary workspace only when the loop executes."""
        q = qm.qubit("q")
        for _ in qm.range(n):
            _work = qm.qubit_array(3, "work")
            q = qm.h(q)
        return q

    symbolic = circuit.estimate_resources()
    n = symbolic.parameters["n"]

    assert n.is_nonnegative is True
    assert symbolic.substitute(n=0).width.allocated_qubits == 1
    assert symbolic.substitute(n=0).qubits == 1
    assert circuit.estimate_resources(inputs={"n": 0}).qubits == 1


def test_uint_zero_branch_remains_symbolically_reachable() -> None:
    """A UInt symbol does not simplify its equality-to-zero branch away."""

    @qm.qkernel
    def circuit(n: qm.UInt) -> qm.Qubit:
        """Use different gate counts on the zero and nonzero branches."""
        q = qm.qubit("q")
        if n == 0:
            q = qm.x(q)
        else:
            q = qm.h(q)
            q = qm.z(q)
        return q

    symbolic = circuit.estimate_resources()
    n = symbolic.parameters["n"]

    assert symbolic.gates.total == sp.Piecewise((1, sp.Eq(n, 0)), (2, True))
    assert symbolic.substitute(n=0).gates.total == 1
    assert symbolic.substitute(n=1).gates.total == 2


def test_float_parameter_remains_real_in_symbolic_branches() -> None:
    """A negative Float input can select a branch after symbolic estimation."""

    @qm.qkernel
    def circuit(theta: qm.Float) -> qm.Qubit:
        """Use a real-valued parameter as a compile-time predicate."""
        q = qm.qubit("q")
        if theta > 0:
            q = qm.x(q)
        else:
            q = qm.h(q)
            q = qm.z(q)
        return q

    symbolic = circuit.estimate_resources()

    assert symbolic.parameters["theta"].is_real is True
    assert symbolic.parameters["theta"].is_integer is None
    assert circuit.estimate_resources(inputs={"theta": -0.5}).gates.total == 2


def test_float_region_carry_fallback_remains_real() -> None:
    """An unsupported Float recurrence never becomes an integer condition."""

    @qm.qkernel
    def circuit(n: qm.UInt) -> qm.Qubit:
        """Branch on a nonlinear Float carry after a symbolic loop."""
        total = qm.float_(0.5)
        for _ in qm.range(n):
            total = total * total
        q = qm.qubit("q")
        if total == 0.25:
            q = qm.x(q)
            q = qm.h(q)
            q = qm.z(q)
        else:
            q = qm.x(q)
        return q

    estimate = circuit.estimate_resources()

    # The nonlinear recurrence is intentionally modeled symbolically. Its
    # Float-typed result must keep the equality undecidable, so the estimator
    # preserves the branch instead of simplifying a non-integer equality
    # against an incorrectly integer-typed symbol.
    fallback = estimate.parameters["total_after_loop"]
    assert fallback.is_real is True
    assert fallback.is_integer is None
    assert estimate.gates.total == sp.Piecewise(
        (3, sp.Eq(fallback, sp.Float(0.25))),
        (1, True),
    )
    assert estimate.substitute(total_after_loop=0.25).gates.total == 3
    assert any(
        "recurrence could not be reduced" in a.message for a in estimate.assumptions
    )


def test_float_runtime_if_merge_keeps_reachable_resource_branch() -> None:
    """An undecidable Float merge never collapses a reachable comparison."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Branch on a Float selected by a measurement-backed conditional."""
        predicate = qm.qubit("predicate")
        measured = qm.measure(predicate)
        value = qm.float_(0.5)
        if measured:
            value = qm.float_(0.25)

        q = qm.qubit("q")
        if value == 0.25:
            q = qm.x(q)
            q = qm.h(q)
            q = qm.z(q)
        else:
            q = qm.x(q)
        return q

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 3


def test_same_named_runtime_if_merges_remain_independent() -> None:
    """Independent measurement merges never collapse by display name."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Compare two independently selected but same-named UInt merges."""
        left_predicate = qm.measure(qm.qubit("left_predicate"))
        right_predicate = qm.measure(qm.qubit("right_predicate"))

        left = qm.uint(0)
        if left_predicate:
            left = qm.uint(1)
        right = qm.uint(0)
        if right_predicate:
            right = qm.uint(1)

        q = qm.qubit("q")
        if left != right:
            q = qm.x(q)
            q = qm.h(q)
            q = qm.z(q)
        else:
            q = qm.x(q)
        return q

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 3


def test_runtime_if_uint_merge_symbol_keeps_ir_domain() -> None:
    """An undecidable UInt merge remains a nonnegative integer."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Use a runtime-selected UInt as a later resource loop bound."""
        predicate = qm.qubit("predicate")
        measured = qm.measure(predicate)

        count = qm.uint(0)
        if measured:
            count = qm.uint(2)

        q = qm.qubit("q")
        for _ in qm.range(count):
            q = qm.h(q)
        return q

    estimate = circuit.estimate_resources()
    (count_symbol,) = estimate.parameters.values()

    assert set(estimate.parameters) == {"uint_const_merge_0"}
    assert count_symbol.is_integer is True
    assert count_symbol.is_nonnegative is True
    assert estimate.substitute(uint_const_merge_0=2).gates.total == 2


def test_measurement_provenance_crosses_qkernel_call_boundary() -> None:
    """A helper's runtime merge stays fresh instead of nesting Piecewise."""

    @qm.qkernel
    def select_count(measured: qm.Bit) -> qm.UInt:
        """Select a loop count from a measurement-backed condition."""
        count = qm.uint(0)
        if measured:
            count = qm.uint(2)
        return count

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Use a helper-selected runtime value as a resource loop bound."""
        predicate = qm.measure(qm.qubit("predicate"))
        count = select_count(predicate)
        target = qm.qubit("target")
        for _ in qm.range(count):
            target = qm.h(target)
        return target

    estimate = circuit.estimate_resources()
    (count_symbol,) = estimate.parameters.values()

    assert not estimate.gates.total.has(sp.Piecewise)
    assert estimate.gates.total == count_symbol
    assert count_symbol.is_integer is True
    assert count_symbol.is_nonnegative is True


def test_runtime_if_bit_merge_symbol_keeps_ir_domain() -> None:
    """An undecidable Bit merge remains a nonnegative integer."""
    from qamomile.circuit.estimator._resolver import ExprResolver
    from qamomile.circuit.estimator.resource_estimator import (
        ResourceEstimatorConfig,
        ResourceInterpreter,
        build_if_scopes,
    )
    from qamomile.circuit.ir.operation.control_flow import IfOperation
    from qamomile.circuit.ir.types.primitives import BitType
    from qamomile.circuit.ir.value import Value

    condition = Value(type=BitType(), name="condition")
    true_value = Value(type=BitType(), name="true").with_const(True)
    false_value = Value(type=BitType(), name="false").with_const(False)
    result = Value(type=BitType(), name="merged")
    operation = IfOperation(
        operands=[condition],
        true_operations=[],
        false_operations=[],
    )
    operation.add_merge(true_value, false_value, result)

    resolver = ExprResolver()
    true_resolver, false_resolver = build_if_scopes(operation, resolver)
    interpreter = ResourceInterpreter(
        config=ResourceEstimatorConfig(),
        bindings={},
    )
    interpreter._publish_if_results(
        operation,
        resolver,
        true_resolver,
        false_resolver,
        taken=None,
    )
    merged_symbol = resolver.resolve(result)

    assert merged_symbol.is_integer is True
    assert merged_symbol.is_nonnegative is True


def test_condition_values_never_capture_identity_fresh_fallbacks() -> None:
    """Public condition inputs specialize symbols but not same-named dummies."""
    from qamomile.circuit.estimator.resource_estimator import (
        ResourceEstimatorConfig,
        ResourceInterpreter,
    )

    public = sp.Symbol("flag", integer=True, nonnegative=True)
    internal = sp.Dummy("flag", integer=True, nonnegative=True)
    interpreter = ResourceInterpreter(
        config=ResourceEstimatorConfig(),
        bindings={},
        condition_values={"flag": sp.Integer(1)},
    )

    specialized = interpreter._apply_condition_values(public + internal)
    decision, _note = interpreter._decide_branch(sp.Eq(internal, 1))

    assert public not in specialized.free_symbols
    assert internal in specialized.free_symbols
    assert decision is None


def test_loop_index_domain_preserves_zero_and_negative_branches() -> None:
    """Loop indices are integers rather than assumed-positive parameters."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Take the false branch for indices minus one and zero."""
        q = qm.qubit("q")
        for index in qm.range(-1, 1):
            if index > 0:
                q = qm.x(q)
            else:
                q = qm.h(q)
                q = qm.z(q)
        return q

    assert circuit.estimate_resources().gates.total == 4


def test_nested_shadowed_loop_names_keep_value_identity() -> None:
    """A saved outer index remains distinct from a shadowing inner index."""

    @qm.qkernel
    def circuit(n: qm.UInt) -> qm.Qubit:
        """Apply outer-index times inner-index gates per index pair."""
        q = qm.qubit("q")
        for index in qm.range(n):
            outer_index = index
            for index in qm.range(n):
                for _ in qm.range(outer_index * index):
                    q = qm.h(q)
        return q

    # (0 + 1 + 2) * (0 + 1 + 2) = 9.
    assert circuit.estimate_resources(inputs={"n": 3}).gates.total == 9


def test_for_items_uses_concrete_entries_and_rejects_symbolic_dependency() -> None:
    """Entry-dependent resources are exact when bound and fail closed otherwise."""

    @qm.qkernel
    def circuit(sizes: qm.Dict[qm.UInt, qm.Float]) -> qm.Qubit:
        """Apply one gate for every unit encoded by each dictionary key."""
        q = qm.qubit("q")
        for size, _ in qm.items(sizes):
            for _ in qm.range(size):
                q = qm.h(q)
        return q

    concrete = circuit.estimate_resources(inputs={"sizes": {1: 0.1, 3: 0.2}})

    assert concrete.gates.total == 4
    assert concrete.parameters == {}
    with pytest.raises(NotImplementedError, match="current item key or value"):
        circuit.estimate_resources()


def test_for_items_default_entries_are_evaluated_exactly() -> None:
    """A concrete dictionary default does not leak cardinality or item symbols."""

    @qm.qkernel
    def circuit(
        sizes: qm.Dict[qm.UInt, qm.Float] = {1: 0.1, 3: 0.2},
    ) -> qm.Qubit:
        """Apply a key-sized gate sequence for the default dictionary."""
        q = qm.qubit("q")
        for size, _ in qm.items(sizes):
            for _ in qm.range(size):
                q = qm.h(q)
        return q

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 4
    assert estimate.parameters == {}


def test_for_items_vector_key_elements_are_bound_per_concrete_entry() -> None:
    """Concrete Vector-key elements drive exact per-entry resources."""

    @qm.qkernel
    def circuit(
        data: qm.Dict[qm.Vector[qm.UInt], qm.Float],
    ) -> qm.Qubit:
        """Apply a key-sized gate sequence for every dictionary entry."""
        q = qm.qubit("q")
        for key, _value in qm.items(data):
            for _ in qm.range(key[0]):
                q = qm.h(q)
        return q

    estimate = circuit.estimate_resources(inputs={"data": {(1, 2): 0.1, (3, 4): 0.2}})

    assert estimate.gates.total == 4
    assert estimate.parameters == {}


def test_for_items_dynamic_vector_key_index_fails_closed() -> None:
    """Concrete Vector keys never degrade to an unbound element symbol."""

    @qm.qkernel
    def circuit(
        data: qm.Dict[qm.Vector[qm.UInt], qm.Float],
    ) -> qm.Qubit:
        """Use every dynamically indexed key element as a loop bound."""
        q = qm.qubit("q")
        for key, _value in qm.items(data):
            for index in qm.range(key.shape[0]):
                for _ in qm.range(key[index]):
                    q = qm.h(q)
        return q

    with pytest.raises(NotImplementedError, match="dynamically indexed"):
        circuit.estimate_resources(inputs={"data": {(1, 2): 0.1}})


def test_zero_cardinality_disables_reusable_loop_width() -> None:
    """A symbolic empty dictionary does not allocate its loop-body workspace."""

    @qm.qkernel
    def circuit(coeffs: qm.Dict[qm.UInt, qm.Float]) -> qm.Qubit:
        """Allocate fixed workspace once per dictionary entry."""
        q = qm.qubit("q")
        for _, _ in qm.items(coeffs):
            _work = qm.qubit_array(3, "work")
        return q

    symbolic = circuit.estimate_resources()
    cardinality = symbolic.parameters["|coeffs|"]

    assert cardinality.is_nonnegative is True
    assert symbolic.substitute(**{"|coeffs|": 0}).qubits == 1


def test_for_items_carry_uses_concrete_entry_order() -> None:
    """A carried accumulator receives every concrete dictionary key in order."""

    @qm.qkernel
    def circuit(sizes: qm.Dict[qm.UInt, qm.Float]) -> qm.Qubit:
        """Use the accumulated dictionary keys as a later loop bound."""
        total = qm.uint(0)
        for size, _ in qm.items(sizes):
            total = total + size
        q = qm.qubit("q")
        for _ in qm.range(total):
            q = qm.h(q)
        return q

    estimate = circuit.estimate_resources(inputs={"sizes": {1: 0.1, 3: 0.2}})

    assert estimate.gates.total == 4


def test_symbolic_for_items_carry_sums_each_iteration() -> None:
    """A carried items-loop bound is summed over the symbolic item index."""

    @qm.qkernel
    def circuit(data: qm.Dict[qm.UInt, qm.Float]) -> qm.Qubit:
        """Apply zero, one, two, and so on gates across dictionary entries."""
        total = qm.uint(0)
        q = qm.qubit("q")
        for _key, _value in qm.items(data):
            for _ in qm.range(total):
                q = qm.h(q)
            total = total + 1
        return q

    estimate = circuit.estimate_resources()
    cardinality = estimate.parameters["|data|"]
    expected = cardinality * (cardinality - 1) / 2

    assert sp.simplify(estimate.gates.total - expected) == 0
    assert set(estimate.parameters) == {"|data|"}
    for size, gate_count in ((0, 0), (1, 0), (2, 1), (3, 3), (4, 6)):
        assert estimate.substitute(**{"|data|": size}).gates.total == gate_count


def test_while_trip_count_is_nonnegative_and_zero_disables_width() -> None:
    """A while loop may execute zero times without allocating body workspace."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Allocate and measure body-local work under a measured condition."""
        trigger = qm.qubit("trigger")
        bit = qm.measure(trigger)
        while bit:
            work = qm.qubit_array(3, "work")
            work[0] = qm.h(work[0])
            bit = qm.measure(work[0])
        return qm.qubit("result")

    symbolic = circuit.estimate_resources()
    trip_count = symbolic.parameters["|while|"]
    zero = symbolic.substitute(**{"|while|": 0})

    assert trip_count.is_nonnegative is True
    assert zero.gates.total == 0
    assert zero.qubits == 1


def test_while_classical_recurrence_fails_closed() -> None:
    """A carried classical while value is not modeled from one traced body."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Use a while-carried counter as a later loop bound."""
        trigger = qm.qubit("trigger")
        bit = qm.measure(trigger)
        total = qm.uint(0)
        while bit:
            total = total + 1
            next_trigger = qm.qubit("next_trigger")
            bit = qm.measure(next_trigger)
        q = qm.qubit("q")
        for _ in qm.range(total):
            q = qm.x(q)
        return q

    with pytest.raises(NotImplementedError, match="WhileOperation"):
        circuit.estimate_resources()


def test_resource_substitute_matches_symbols_by_printed_name() -> None:
    """Substitution replaces same-named symbols with different assumptions."""
    positive_n = sp.Symbol("n", integer=True, positive=True)
    nonnegative_n = sp.Symbol("n", integer=True, nonnegative=True)
    estimate = qm.ResourceEstimate(
        width=qm.WidthResources(peak_qubits=positive_n + nonnegative_n),
        gates=qm.GateResources(total=nonnegative_n),
        parameters={"n": positive_n},
    )

    concrete = estimate.substitute(n=2)

    assert concrete.qubits == 4
    assert concrete.gates.total == 2


def test_input_vector_shape_symbol_is_nonnegative() -> None:
    """A public array dimension permits an empty input vector."""

    @qm.qkernel
    def circuit(values: qm.Vector[qm.Float]) -> qm.Vector[qm.Qubit]:
        """Allocate one qubit per input vector element."""
        return qm.qubit_array(values.shape[0], "q")

    estimate = circuit.estimate_resources()
    (dimension,) = estimate.parameters.values()

    assert dimension.is_nonnegative is True


def test_unresolved_uint_and_bit_fallbacks_are_nonnegative() -> None:
    """Identity-qualified UInt and Bit symbols preserve their IR domains."""
    from qamomile.circuit.estimator._resolver import ExprResolver
    from qamomile.circuit.ir.types.primitives import BitType, UIntType
    from qamomile.circuit.ir.value import Value

    resolver = ExprResolver()
    uint_symbol = resolver.resolve(Value(type=UIntType(), name="value"))
    bit_symbol = resolver.resolve(Value(type=BitType(), name="value"))

    assert uint_symbol.is_integer is True
    assert uint_symbol.is_nonnegative is True
    assert bit_symbol.is_integer is True
    assert bit_symbol.is_nonnegative is True


def test_controlled_composite_body_counts_own_control() -> None:
    """A controlled body-backed composite reclassifies its primitives as controlled.

    A body-backed composite (one H gate), invoked with
    an explicit control, must have its exact-body estimate count the H as a
    two-qubit (controlled) gate — the body path honors the invocation's own
    control just like the model path and ``eval_controlled_u`` do.
    """

    @qm.composite_gate(name="one_h")
    def one_h(t: qm.Qubit) -> qm.Qubit:
        """Apply a single H gate to the target."""
        return qm.h(t)

    @qm.composite_gate(name="plain_one_h")
    def plain_one_h(t: qm.Qubit) -> qm.Qubit:
        """Apply a single H gate to the target (no controls)."""
        return qm.h(t)

    @qm.qkernel
    def controlled() -> tuple[qm.Qubit, qm.Qubit]:
        """Apply one_h through the normal higher-order control operator."""
        c = qm.qubit("c")
        t = qm.qubit("t")
        controlled_one_h = qm.control(one_h)
        return controlled_one_h(c, t)

    @qm.qkernel
    def uncontrolled() -> qm.Qubit:
        """Apply plain_one_h directly."""
        t = qm.qubit("t")
        return plain_one_h(t)

    ctrl = controlled.estimate_resources()
    plain = uncontrolled.estimate_resources()

    assert plain.gates.single_qubit == 1
    assert plain.gates.two_qubit == 0
    # The control turns the single-qubit H into a two-qubit controlled gate.
    assert ctrl.gates.single_qubit == 0
    assert ctrl.gates.two_qubit == 1


def test_controlled_three_qubit_primitive_is_not_implicitly_toffoli() -> None:
    """Three-qubit arity alone cannot classify a controlled gate as Toffoli."""

    @qm.composite_gate(name="one_swap")
    def one_swap(
        left: qm.Qubit,
        right: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit]:
        """Swap two target qubits."""
        return qm.swap(left, right)

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
        """Apply SWAP under one surrounding control."""
        control = qm.qubit("control")
        left = qm.qubit("left")
        right = qm.qubit("right")
        return qm.control(one_swap)(control, left, right)

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 1
    assert estimate.gates.multi_qubit == 1
    assert estimate.gates.toffoli == 0


def test_for_items_width_reuses_wires_across_entries() -> None:
    """A bound items loop's width is the per-entry max, not the entry sum.

    Each entry's body allocates one fresh qubit, but the emitted circuit
    resets and reuses dead wires between iterations — exactly like
    ``repeat``. Sequential composition alone would report
    1 (outs) + 2 (one fresh per entry) == 3 qubits; the loop must
    report 2, matching the emitted circuit.
    """

    @qm.qkernel
    def circuit(data: qm.Dict[qm.UInt, qm.Float]) -> qm.Vector[qm.Bit]:
        """Rotate one fresh qubit per entry against a persistent register."""
        outs = qm.qubit_array(1, "outs")
        for _key, val in qm.items(data):
            fresh = qm.qubit("fresh")
            fresh = qm.rx(fresh, val)
            outs[0], fresh = qm.cx(outs[0], fresh)
            qm.measure(fresh)
        return qm.measure(outs)

    data = {1: 0.5, 2: 1.5}
    estimate = circuit.estimate_resources(inputs={"data": data})

    assert estimate.qubits == 2
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    emitted = QiskitTranspiler().transpile(
        circuit,
        bindings={"data": data},
    )
    assert estimate.width.allocated_qubits == 2
    assert emitted.quantum_circuit.num_qubits == estimate.qubits
    # Gates still accumulate sequentially across entries: one rx + one
    # cx per entry.
    assert estimate.gates.total >= 4


def test_concrete_region_for_width_reuses_wires_across_iterations() -> None:
    """A concrete carried range loop matches emitted reusable wire width."""

    @qm.qkernel
    def circuit() -> tuple[qm.Bit, qm.UInt]:
        """Allocate one temporary qubit in each carried loop iteration."""
        output = qm.qubit("output")
        total = qm.uint(0)
        for _index in qm.range(2):
            fresh = qm.qubit("fresh")
            qm.measure(fresh)
            total = total + 1
        return qm.measure(output), total

    estimate = circuit.estimate_resources()

    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    emitted = QiskitTranspiler().transpile(circuit)
    assert estimate.qubits == 2
    assert emitted.quantum_circuit.num_qubits == estimate.qubits


def test_concrete_region_for_items_width_reuses_wires_across_entries() -> None:
    """A concrete carried items loop matches emitted reusable wire width."""

    @qm.qkernel
    def circuit(
        data: qm.Dict[qm.UInt, qm.Float],
    ) -> tuple[qm.Bit, qm.UInt]:
        """Allocate one temporary qubit in each carried items entry."""
        output = qm.qubit("output")
        total = qm.uint(0)
        for key, _value in qm.items(data):
            fresh = qm.qubit("fresh")
            qm.measure(fresh)
            total = total + key
        return qm.measure(output), total

    data = {1: 0.5, 2: 1.5}
    estimate = circuit.estimate_resources(inputs={"data": data})

    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    emitted = QiskitTranspiler().transpile(
        circuit,
        bindings={"data": data},
    )
    assert estimate.qubits == 2
    assert emitted.quantum_circuit.num_qubits == estimate.qubits


def test_concrete_loops_union_distinct_qinit_sites_by_identity() -> None:
    """Concrete replays deduplicate one QInit site but retain distinct sites.

    Each loop selects allocation site ``a`` on its first iteration/entry and
    site ``b`` on its second. The sites are never simultaneously live, so the
    logical peak remains one, but both are static allocations in the emitted
    circuit. A third site ``out`` after the loop makes the identity-aware
    allocation total three for RegionArg range, RegionArg ForItems, and
    carry-less ForItems alike.
    """
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    @qm.qkernel
    def region_range() -> tuple[qm.Bit, qm.UInt]:
        """Select two allocation sites across carried range iterations."""
        total = qm.uint(0)
        for index in qm.range(2):
            if index == 0:
                a = qm.qubit("a")
                qm.measure(a)
            else:
                b = qm.qubit("b")
                qm.measure(b)
            total = total + 1
        out = qm.qubit("out")
        return qm.measure(out), total

    @qm.qkernel
    def region_items(
        data: qm.Dict[qm.UInt, qm.Float],
    ) -> tuple[qm.Bit, qm.UInt]:
        """Select two allocation sites across carried dictionary entries."""
        total = qm.uint(0)
        for key, _value in qm.items(data):
            if key == 0:
                a = qm.qubit("a")
                qm.measure(a)
            else:
                b = qm.qubit("b")
                qm.measure(b)
            total = total + 1
        out = qm.qubit("out")
        return qm.measure(out), total

    @qm.qkernel
    def plain_items(data: qm.Dict[qm.UInt, qm.Float]) -> qm.Bit:
        """Select two allocation sites across carry-less dictionary entries."""
        for key, _value in qm.items(data):
            if key == 0:
                a = qm.qubit("a")
                qm.measure(a)
            else:
                b = qm.qubit("b")
                qm.measure(b)
        out = qm.qubit("out")
        return qm.measure(out)

    data = {0: 0.5, 1: 1.5}
    cases = (
        (region_range, {}),
        (region_items, {"data": data}),
        (plain_items, {"data": data}),
    )
    for kernel, bindings in cases:
        estimate = kernel.estimate_resources(inputs=bindings)
        emitted = QiskitTranspiler().transpile(
            kernel,
            bindings=bindings,
        )

        assert estimate.width.allocated_qubits == 3
        assert estimate.width.peak_qubits == 1
        assert emitted.quantum_circuit.num_qubits == (estimate.width.allocated_qubits)


def test_nested_qinit_identity_is_namespaced_by_helper_callsite() -> None:
    """Distinct helper calls allocate separately while one replayed call reuses."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    distinct = _distinct_helper_callsites.estimate_resources()
    repeated = _repeated_helper_callsite.estimate_resources()
    distinct_emitted = (
        QiskitTranspiler().transpile(_distinct_helper_callsites).quantum_circuit
    )
    repeated_emitted = (
        QiskitTranspiler().transpile(_repeated_helper_callsite).quantum_circuit
    )

    assert distinct.width.allocated_qubits == 3
    assert distinct.width.peak_qubits == 2
    assert distinct_emitted.num_qubits == distinct.width.allocated_qubits
    assert repeated.width.allocated_qubits == 2
    assert repeated.width.peak_qubits == 2
    assert repeated_emitted.num_qubits == repeated.width.allocated_qubits


def test_same_named_range_fallback_symbols_remain_independent() -> None:
    """Separate unsupported range recurrences never collapse by source name."""

    @qm.qkernel
    def circuit(n: qm.UInt, m: qm.UInt) -> qm.Bit:
        """Compare two nonlinear carries created under the same variable name."""
        total = qm.float_(0.5)
        for _index in qm.range(n):
            total = total * total
        left = total

        total = qm.float_(0.25)
        for _index in qm.range(m):
            total = total * total
        right = total

        q = qm.qubit("q")
        if left != right:
            q = qm.x(q)
            q = qm.h(q)
            q = qm.z(q)
        else:
            q = qm.x(q)
        return qm.measure(q)

    estimate = circuit.estimate_resources()
    final_symbols = [
        symbol
        for symbol in estimate.gates.total.free_symbols
        if symbol.name == "total_after_loop"
    ]

    assert len(final_symbols) == 2
    assert final_symbols[0] != final_symbols[1]
    assert estimate.gates.total != 1

    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    emitted = QiskitTranspiler().transpile(
        circuit,
        bindings={"n": 0, "m": 0},
    )
    assert emitted.quantum_circuit.count_ops()["x"] == 1
    assert emitted.quantum_circuit.count_ops()["h"] == 1
    assert emitted.quantum_circuit.count_ops()["z"] == 1


def test_same_named_for_items_fallback_symbols_remain_independent() -> None:
    """Separate unsupported items recurrences never collapse by source name."""

    @qm.qkernel
    def circuit(
        left_data: qm.Dict[qm.UInt, qm.Float],
        right_data: qm.Dict[qm.UInt, qm.Float],
    ) -> qm.Qubit:
        """Compare two nonlinear item carries with one reused variable name."""
        total = qm.float_(0.5)
        for _key, _value in qm.items(left_data):
            total = total * total
        left = total

        total = qm.float_(0.25)
        for _key, _value in qm.items(right_data):
            total = total * total
        right = total

        q = qm.qubit("q")
        if left != right:
            q = qm.x(q)
            q = qm.h(q)
            q = qm.z(q)
        else:
            q = qm.x(q)
        return q

    estimate = circuit.estimate_resources()
    final_symbols = [
        symbol
        for symbol in estimate.gates.total.free_symbols
        if symbol.name == "total_after_items"
    ]

    assert len(final_symbols) == 2
    assert final_symbols[0] != final_symbols[1]
    assert estimate.gates.total != 1


def test_allocation_site_identity_is_internal_to_resource_estimates() -> None:
    """Random QInit UUIDs do not affect equality or public serialization."""
    left = qm.ResourceEstimate(
        width=qm.WidthResources(allocated_qubits=1, peak_qubits=1),
        _allocation_sites={"build-random-left": sp.Integer(1)},
    )
    right = qm.ResourceEstimate(
        width=qm.WidthResources(allocated_qubits=1, peak_qubits=1),
        _allocation_sites={"build-random-right": sp.Integer(1)},
    )

    assert left == right
    assert "_allocation_sites" not in left.to_dict()


def test_unknown_branch_allocations_keep_conditional_width_semantics() -> None:
    """Unknown branch sites remain anonymous and substitute exactly."""
    flag = sp.Symbol("flag", integer=True, nonnegative=True)
    left = qm.ResourceEstimate(
        width=qm.WidthResources(allocated_qubits=1, peak_qubits=1),
        _allocation_sites={"left": sp.Integer(1)},
    )
    right = qm.ResourceEstimate(
        width=qm.WidthResources(allocated_qubits=2, peak_qubits=2),
        _allocation_sites={"right": sp.Integer(2)},
    )

    conditional = left.conditional(right, sp.Eq(flag, 1))
    choice = left.choice(right)

    assert conditional._allocation_sites == {}
    assert conditional.substitute(flag=1).width.allocated_qubits == 1
    assert conditional.substitute(flag=0).width.allocated_qubits == 2
    assert choice._allocation_sites == {}
    assert choice.width.allocated_qubits == 2


def test_zero_trip_repeat_disables_internal_allocation_sites() -> None:
    """A zero-trip repeat contributes neither width nor an active QInit site."""
    body = qm.ResourceEstimate(
        width=qm.WidthResources(allocated_qubits=1, peak_qubits=1),
        _allocation_sites={"body": sp.Integer(1)},
    )

    repeated = body.repeat(0)

    assert repeated.width.allocated_qubits == 0
    assert repeated.width.peak_qubits == 0
    assert repeated._allocation_sites == {"body": sp.Integer(0)}
