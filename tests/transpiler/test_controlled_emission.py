"""Tests for controlled-emission support helpers."""

from typing import Any

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableRef,
    CompositeGateType,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.gate import (
    GateOperation,
    GateOperationType,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.types.hamiltonian import ObservableType
from qamomile.circuit.ir.types.primitives import FloatType, QubitType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.passes.emit_support import (
    controlled_emission,
    inverse_emission,
)
from qamomile.circuit.transpiler.passes.emit_support.controlled_emission import (
    _gate_matches_qubit_count,
    emit_controlled_operations,
    emit_multi_controlled_gate,
)
from qamomile.circuit.transpiler.passes.emit_support.multi_control_ancilla import (
    MultiControlAncillaPool,
)
from qamomile.circuit.transpiler.passes.emit_support.qubit_address import (
    QubitAddress,
)
from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
    ValueResolver,
)


class _ResolverOnlyEmitPass:
    """Minimal emit-pass stand-in exposing only the value resolver."""

    def __init__(self) -> None:
        """Initialize the stand-in with a real ``ValueResolver``."""
        self._resolver = ValueResolver()


class _GateWithoutQubitCount:
    """Backend-gate stand-in with no qubit-count attribute."""


class _GateWithQubitCount:
    """Backend-gate stand-in with a qubit-count attribute."""

    def __init__(self, num_qubits: int | None) -> None:
        """Initialize the fake gate.

        Args:
            num_qubits (int | None): Fake backend gate width.
        """
        self.num_qubits = num_qubits


def test_gate_matches_qubit_count_rejects_unknown_width() -> None:
    """Unknown backend-gate width should force fallback emission."""
    assert not _gate_matches_qubit_count(_GateWithoutQubitCount(), 2)
    assert not _gate_matches_qubit_count(_GateWithQubitCount(None), 2)
    assert _gate_matches_qubit_count(_GateWithQubitCount(2), 2)
    assert not _gate_matches_qubit_count(_GateWithQubitCount(1), 2)


def test_controlled_dispatch_accepts_inverse_block(monkeypatch) -> None:
    """Controlled dispatch resolves inverse-block operands via the map."""
    q = Value(type=QubitType(), name="q")
    q_out = q.next_version()
    op = InverseBlockOperation(
        operands=[q],
        results=[q_out],
        num_target_qubits=1,
        source_block=Block(),
        implementation_block=Block(),
    )
    calls: list[tuple[list[int], list[int]]] = []

    def fake_emit_inverse(
        emit_pass: Any,
        circuit: Any,
        op: InverseBlockOperation,
        control_indices: list[int],
        target_indices: list[int],
        bindings: dict[str, Any],
    ) -> None:
        """Record the resolved controlled inverse call."""
        del emit_pass, circuit, op, bindings
        calls.append((control_indices, target_indices))

    # The dispatch site imports lazily from ``inverse_emission`` at call
    # time, so the patch targets that module rather than
    # ``controlled_emission``.
    monkeypatch.setattr(
        inverse_emission,
        "emit_inverse_block_at_indices",
        fake_emit_inverse,
    )

    qubit_map = {QubitAddress(q.uuid): 3}
    emit_controlled_operations(
        _ResolverOnlyEmitPass(), object(), [op], [7], qubit_map, {}
    )

    assert calls == [([7], [3])]
    assert qubit_map[QubitAddress(q_out.uuid)] == 3


def test_controlled_dispatch_accepts_composite_invocation(monkeypatch) -> None:
    """Controlled dispatch resolves composite invocation operands via the map."""
    q = Value(type=QubitType(), name="q")
    q_out = q.next_version()
    ref = CallableRef(namespace="user.composite", name="custom")
    attrs = {
        "kind": "composite",
        "gate_type": CompositeGateType.CUSTOM.name,
        "num_control_qubits": 0,
        "num_target_qubits": 1,
        "custom_name": "custom",
    }
    op = InvokeOperation(
        operands=[q],
        results=[q_out],
        target=ref,
        attrs=attrs,
        definition=CallableDef(ref=ref, body=Block(), attrs=attrs),
    )
    calls: list[tuple[list[int], list[int]]] = []

    def fake_emit_composite(
        emit_pass: Any,
        circuit: Any,
        op: InvokeOperation,
        control_indices: list[int],
        qubit_indices: list[int],
        bindings: dict[str, Any],
    ) -> None:
        """Record the resolved controlled composite call."""
        del emit_pass, circuit, op, bindings
        calls.append((control_indices, qubit_indices))

    monkeypatch.setattr(
        controlled_emission,
        "emit_controlled_composite_at_indices",
        fake_emit_composite,
    )

    qubit_map = {QubitAddress(q.uuid): 3}
    emit_controlled_operations(
        _ResolverOnlyEmitPass(), object(), [op], [7], qubit_map, {}
    )

    assert calls == [([7], [3])]
    assert qubit_map[QubitAddress(q_out.uuid)] == 3


def test_controlled_dispatch_accepts_pauli_evolve(monkeypatch) -> None:
    """Controlled dispatch routes a PauliEvolveOp to the shared lowering."""
    q = Value(type=QubitType(), name="q")
    q_out = q.next_version()
    observable = Value(type=ObservableType(), name="ham")
    gamma = Value(type=FloatType(), name="gamma")
    op = PauliEvolveOp(operands=[q, observable, gamma], results=[q_out])
    calls: list[list[int]] = []

    def fake_emit_pauli_evolve(
        emit_pass: Any,
        circuit: Any,
        op: PauliEvolveOp,
        control_indices: list[int],
        qubit_map: dict[Any, int],
        bindings: dict[str, Any],
    ) -> None:
        """Record the controls threaded into the controlled Pauli evolution."""
        del emit_pass, circuit, op, qubit_map, bindings
        calls.append(control_indices)

    monkeypatch.setattr(
        controlled_emission,
        "emit_controlled_pauli_evolve",
        fake_emit_pauli_evolve,
    )

    emit_controlled_operations(
        _ResolverOnlyEmitPass(), object(), [op], [7], {QubitAddress(q.uuid): 3}, {}
    )

    assert calls == [[7]]


class _RecordingEmitter:
    """Gate emitter stand-in that records primitive emissions."""

    def __init__(self) -> None:
        """Initialize the empty emission log."""
        self.calls: list[tuple[Any, ...]] = []

    def emit_cx(self, circuit: Any, control: int, target: int) -> None:
        """Record a CX emission."""
        del circuit
        self.calls.append(("cx", control, target))

    def emit_cz(self, circuit: Any, control: int, target: int) -> None:
        """Record a CZ emission."""
        del circuit
        self.calls.append(("cz", control, target))

    def emit_h(self, circuit: Any, qubit: int) -> None:
        """Record an H emission."""
        del circuit
        self.calls.append(("h", qubit))

    def emit_toffoli(
        self, circuit: Any, control1: int, control2: int, target: int
    ) -> None:
        """Record a Toffoli emission."""
        del circuit
        self.calls.append(("toffoli", control1, control2, target))

    def emit_crz(self, circuit: Any, control: int, target: int, angle: Any) -> None:
        """Record a CRZ emission."""
        del circuit
        self.calls.append(("crz", control, target, angle))

    def emit_cry(self, circuit: Any, control: int, target: int, angle: Any) -> None:
        """Record a CRY emission."""
        del circuit
        self.calls.append(("cry", control, target, angle))

    def emit_cp(self, circuit: Any, control: int, target: int, angle: Any) -> None:
        """Record a CP emission."""
        del circuit
        self.calls.append(("cp", control, target, angle))


class _MultiControlEmitPass:
    """Emit-pass stand-in for multi-controlled reduction tests."""

    def __init__(
        self,
        record_hook: bool = False,
        ancilla_pool: MultiControlAncillaPool | None = None,
    ) -> None:
        """Initialize with a recording emitter and optional hook.

        Args:
            record_hook (bool): When True, provide the irreducible
                multi-controlled gate hook and record its invocations
                instead of raising.
            ancilla_pool (MultiControlAncillaPool | None): Clean-ancilla
                pool visible to the base irreducible hook, enabling the
                shared Toffoli-cascade lowering. Defaults to None (no
                pool — the base hook raises).
        """
        self._emitter = _RecordingEmitter()
        self._resolver = ValueResolver()
        self.hook_calls: list[tuple[Any, ...]] = []
        self._record_hook = record_hook
        self._mc_ancilla_pool = ancilla_pool

    def _resolve_angle(self, op: Any, bindings: dict[str, Any]) -> Any:
        """Resolve a rotation angle from the gate's theta constant."""
        del bindings
        theta = op.theta
        assert theta is not None
        return float(theta.get_const())

    def _emit_irreducible_multi_controlled_gate(
        self,
        circuit: Any,
        gate_type: Any,
        control_indices: list[int],
        target_idx: int,
        angle: Any,
    ) -> None:
        """Record or reject an irreducible multi-controlled gate."""
        del circuit
        if not self._record_hook:
            from qamomile.circuit.transpiler.passes.standard_emit import (
                StandardEmitPass,
            )

            StandardEmitPass._emit_irreducible_multi_controlled_gate(
                self, None, gate_type, control_indices, target_idx, angle
            )
            return
        self.hook_calls.append((gate_type, tuple(control_indices), target_idx, angle))


def _fixed_gate(gate_type: "GateOperationType", num_qubits: int) -> "GateOperation":
    """Build a fixed GateOperation over fresh qubit values."""
    qubits = [Value(type=QubitType(), name=f"q{i}") for i in range(num_qubits)]
    results = [q.next_version() for q in qubits]
    return GateOperation.fixed(gate_type, qubits, results)


def _rotation_gate(gate_type: "GateOperationType", angle: float) -> "GateOperation":
    """Build a single-qubit rotation GateOperation with a constant angle."""
    from qamomile.circuit.ir.types.primitives import FloatType

    qubit = Value(type=QubitType(), name="q0")
    theta = Value(type=FloatType(), name="theta").with_const(angle)
    return GateOperation.rotation(gate_type, [qubit], theta, [qubit.next_version()])


def test_multi_controlled_x_two_controls_uses_toffoli() -> None:
    """Two-controlled X reduces to a single Toffoli."""
    emit_pass = _MultiControlEmitPass()
    op = _fixed_gate(GateOperationType.X, 1)
    emit_multi_controlled_gate(emit_pass, object(), op, [4, 5], [9], {})
    assert emit_pass._emitter.calls == [("toffoli", 4, 5, 9)]


def test_multi_controlled_z_two_controls_conjugates_toffoli() -> None:
    """Two-controlled Z emits H–Toffoli–H on the target."""
    emit_pass = _MultiControlEmitPass()
    op = _fixed_gate(GateOperationType.Z, 1)
    emit_multi_controlled_gate(emit_pass, object(), op, [4, 5], [9], {})
    assert emit_pass._emitter.calls == [
        ("h", 9),
        ("toffoli", 4, 5, 9),
        ("h", 9),
    ]


def test_multi_controlled_cx_absorbs_gate_control() -> None:
    """CX under one outer control becomes a Toffoli on the composed set."""
    emit_pass = _MultiControlEmitPass()
    op = _fixed_gate(GateOperationType.CX, 2)
    emit_multi_controlled_gate(emit_pass, object(), op, [7], [2, 3], {})
    assert emit_pass._emitter.calls == [("toffoli", 7, 2, 3)]


def test_multi_controlled_toffoli_routes_to_hook() -> None:
    """TOFFOLI under one outer control needs three controls -> hook."""
    emit_pass = _MultiControlEmitPass(record_hook=True)
    op = _fixed_gate(GateOperationType.TOFFOLI, 3)
    emit_multi_controlled_gate(emit_pass, object(), op, [7], [2, 3, 4], {})
    assert emit_pass.hook_calls == [(GateOperationType.X, (7, 2, 3), 4, None)]


def test_multi_controlled_swap_sandwiches_mc_x() -> None:
    """SWAP under two controls uses the CX–MCX–CX Fredkin form."""
    emit_pass = _MultiControlEmitPass(record_hook=True)
    op = _fixed_gate(GateOperationType.SWAP, 2)
    emit_multi_controlled_gate(emit_pass, object(), op, [7, 8], [2, 3], {})
    assert emit_pass._emitter.calls == [("cx", 3, 2), ("cx", 3, 2)]
    assert emit_pass.hook_calls == [(GateOperationType.X, (7, 8, 2), 3, None)]


def test_multi_controlled_rzz_controls_only_the_rz() -> None:
    """RZZ under one control conjugates a CRZ with uncontrolled CXs."""
    from qamomile.circuit.ir.types.primitives import FloatType

    emit_pass = _MultiControlEmitPass()
    qubits = [Value(type=QubitType(), name=f"q{i}") for i in range(2)]
    theta = Value(type=FloatType(), name="theta").with_const(0.5)
    op = GateOperation.rotation(
        GateOperationType.RZZ, qubits, theta, [q.next_version() for q in qubits]
    )
    emit_multi_controlled_gate(emit_pass, object(), op, [7], [2, 3], {})
    assert emit_pass._emitter.calls == [
        ("cx", 2, 3),
        ("crz", 7, 3, 0.5),
        ("cx", 2, 3),
    ]


def test_multi_controlled_rotation_routes_to_hook_with_angle() -> None:
    """Two-controlled RY forwards the resolved angle to the hook."""
    emit_pass = _MultiControlEmitPass(record_hook=True)
    op = _rotation_gate(GateOperationType.RY, 1.25)
    emit_multi_controlled_gate(emit_pass, object(), op, [4, 5], [9], {})
    assert emit_pass.hook_calls == [(GateOperationType.RY, (4, 5), 9, 1.25)]


def test_multi_controlled_irreducible_without_hook_raises() -> None:
    """Three-controlled X on a backend without an ancilla pool raises EmitError."""
    import pytest

    from qamomile.circuit.transpiler.errors import EmitError

    emit_pass = _MultiControlEmitPass()
    op = _fixed_gate(GateOperationType.X, 1)
    with pytest.raises(EmitError, match="3-controlled X"):
        emit_multi_controlled_gate(emit_pass, object(), op, [4, 5, 6], [9], {})


def test_multi_controlled_x_three_controls_cascades_on_clean_ancillas() -> None:
    """Three-controlled X lowers to a Toffoli cascade on pool ancillas.

    Verifies the arXiv:2307.07478 App. A.3 shape: the control AND is
    computed onto the ancillas, a single CX fires from the last
    ancilla, and the cascade is uncomputed in reverse order.
    """
    emit_pass = _MultiControlEmitPass(
        ancilla_pool=MultiControlAncillaPool(first_index=10, count=2)
    )
    op = _fixed_gate(GateOperationType.X, 1)
    emit_multi_controlled_gate(emit_pass, object(), op, [4, 5, 6], [9], {})
    assert emit_pass._emitter.calls == [
        ("toffoli", 4, 5, 10),
        ("toffoli", 6, 10, 11),
        ("cx", 11, 9),
        ("toffoli", 6, 10, 11),
        ("toffoli", 4, 5, 10),
    ]


def test_multi_controlled_ry_two_controls_cascades_with_angle() -> None:
    """Two-controlled RY forwards its angle to a CRY on the AND ancilla."""
    emit_pass = _MultiControlEmitPass(
        ancilla_pool=MultiControlAncillaPool(first_index=10, count=1)
    )
    op = _rotation_gate(GateOperationType.RY, 1.25)
    emit_multi_controlled_gate(emit_pass, object(), op, [4, 5], [9], {})
    assert emit_pass._emitter.calls == [
        ("toffoli", 4, 5, 10),
        ("cry", 10, 9, 1.25),
        ("toffoli", 4, 5, 10),
    ]


def test_multi_controlled_pool_shortfall_raises_estimation_bug() -> None:
    """A pool smaller than n-1 ancillas raises the demand-bug error."""
    import pytest

    from qamomile.circuit.transpiler.errors import EmitError

    emit_pass = _MultiControlEmitPass(
        ancilla_pool=MultiControlAncillaPool(first_index=10, count=1)
    )
    op = _fixed_gate(GateOperationType.X, 1)
    with pytest.raises(EmitError, match="under-measured"):
        emit_multi_controlled_gate(emit_pass, object(), op, [4, 5, 6], [9], {})


def test_resolve_controlled_u_rejects_control_indices_with_multi_arg_prefix() -> None:
    """control_indices combined with a multi-arg control prefix raises EmitError."""
    import pytest

    from qamomile.circuit.ir.operation.gate import SymbolicControlledU
    from qamomile.circuit.ir.types.primitives import UIntType
    from qamomile.circuit.transpiler.errors import EmitError
    from qamomile.circuit.transpiler.passes.emit_support.controlled_emission import (
        resolve_controlled_u_call,
    )

    c0 = Value(type=QubitType(), name="c0")
    c1 = Value(type=QubitType(), name="c1")
    target = Value(type=QubitType(), name="t")
    num_controls = Value(type=UIntType(), name="nc").with_const(1)
    control_index = Value(type=UIntType(), name="ci").with_const(0)

    # Malformed layout: control_indices selects from a single pool, yet the
    # control prefix carries two operands (num_control_args=2). The frontend
    # rejects this, but a hand-built / deserialized op must still fail loudly
    # rather than silently treating only the first operand as the pool.
    op = SymbolicControlledU(
        operands=[c0, c1, target],
        results=[c0.next_version(), c1.next_version(), target.next_version()],
        num_controls=num_controls,
        control_indices=(control_index,),
        num_control_args=2,
        block=Block(),
    )

    qubit_map = {
        QubitAddress(c0.uuid): 0,
        QubitAddress(c1.uuid): 1,
        QubitAddress(target.uuid): 2,
    }
    with pytest.raises(EmitError, match="exactly one control-pool operand"):
        resolve_controlled_u_call(_ResolverOnlyEmitPass(), op, qubit_map, {})


def test_batched_multi_gate_body_shares_one_and_ladder() -> None:
    """A multi-gate body under >=2 controls emits one shared AND ladder.

    The walker ANDs the three controls onto ancilla 11 once, applies RY
    then RZ under that single control, and uncomputes the ladder once — so
    the recorded sequence is ladder / cry / crz / reverse-ladder, four
    Toffolis rather than one full cascade per gate.
    """
    pool = MultiControlAncillaPool(first_index=10, count=2)
    emit_pass = _MultiControlEmitPass(ancilla_pool=pool)

    q = Value(type=QubitType(), name="q")
    q1 = q.next_version()
    q2 = q1.next_version()
    ry = GateOperation.rotation(
        GateOperationType.RY,
        [q],
        Value(type=FloatType(), name="a").with_const(0.5),
        [q1],
    )
    rz = GateOperation.rotation(
        GateOperationType.RZ,
        [q1],
        Value(type=FloatType(), name="b").with_const(0.7),
        [q2],
    )

    emit_controlled_operations(
        emit_pass, object(), [ry, rz], [0, 1, 2], {QubitAddress(q.uuid): 3}, {}
    )

    assert emit_pass._emitter.calls == [
        ("toffoli", 0, 1, 10),
        ("toffoli", 2, 10, 11),
        ("cry", 11, 3, 0.5),
        ("crz", 11, 3, 0.7),
        ("toffoli", 2, 10, 11),
        ("toffoli", 0, 1, 10),
    ]


def test_batched_two_control_all_native_body_uses_per_gate_toffolis() -> None:
    """A two-control body of native X gates is not batched.

    Under exactly two controls each X is already a native Toffoli, so the
    two-control guard keeps the per-gate path: the recorded sequence is one
    Toffoli per X and no AND ladder / ancilla indirection.
    """
    pool = MultiControlAncillaPool(first_index=10, count=1)
    emit_pass = _MultiControlEmitPass(ancilla_pool=pool)

    qa = Value(type=QubitType(), name="a")
    qb = Value(type=QubitType(), name="b")
    xa = GateOperation.fixed(GateOperationType.X, [qa], [qa.next_version()])
    xb = GateOperation.fixed(GateOperationType.X, [qb], [qb.next_version()])

    emit_controlled_operations(
        emit_pass,
        object(),
        [xa, xb],
        [0, 1],
        {QubitAddress(qa.uuid): 3, QubitAddress(qb.uuid): 4},
        {},
    )

    assert emit_pass._emitter.calls == [
        ("toffoli", 0, 1, 3),
        ("toffoli", 0, 1, 4),
    ]


def test_segment_may_reserve_ancillas_gates_the_counting_dry_run() -> None:
    """The dry-run gate is True only when a segment can reach the cascade."""
    from qamomile.circuit.ir.operation.control_flow import ForOperation
    from qamomile.circuit.ir.operation.gate import ConcreteControlledU
    from qamomile.circuit.ir.types.primitives import UIntType
    from qamomile.circuit.transpiler.passes.standard_emit import (
        _segment_may_reserve_ancillas,
    )

    def _controlled_u() -> ConcreteControlledU:
        """Build a minimal one-control controlled-U over fresh qubits."""
        ctrl = Value(type=QubitType(), name="ctrl")
        target = Value(type=QubitType(), name="tgt")
        return ConcreteControlledU(
            operands=[ctrl, target],
            results=[ctrl.next_version(), target.next_version()],
            num_controls=1,
            block=Block(),
        )

    # Plain gates never reach the multi-control hook.
    assert not _segment_may_reserve_ancillas(
        [_fixed_gate(GateOperationType.X, 1), _fixed_gate(GateOperationType.CX, 2)]
    )
    # A controlled-U does, directly.
    assert _segment_may_reserve_ancillas([_controlled_u()])
    # A controlled-U nested inside a loop body is found recursively.
    loop = ForOperation(
        operands=[
            Value(type=UIntType(), name="start").with_const(0),
            Value(type=UIntType(), name="stop").with_const(3),
            Value(type=UIntType(), name="step").with_const(1),
        ],
        results=[],
        loop_var_value=Value(type=UIntType(), name="i"),
        operations=[_controlled_u()],
    )
    assert _segment_may_reserve_ancillas([loop])
