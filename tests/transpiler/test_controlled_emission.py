"""Tests for controlled-emission support helpers."""

from typing import Any

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    CompositeGateType,
)
from qamomile.circuit.ir.operation.gate import (
    GateOperation,
    GateOperationType,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.types.primitives import QubitType
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


def test_controlled_dispatch_accepts_composite_gate(monkeypatch) -> None:
    """Controlled dispatch resolves composite operands via the map."""
    q = Value(type=QubitType(), name="q")
    q_out = q.next_version()
    op = CompositeGateOperation(
        operands=[q],
        results=[q_out],
        gate_type=CompositeGateType.CUSTOM,
        num_target_qubits=1,
        implementation_block=Block(),
    )
    calls: list[tuple[list[int], list[int]]] = []

    def fake_emit_composite(
        emit_pass: Any,
        circuit: Any,
        op: CompositeGateOperation,
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

    def emit_cp(self, circuit: Any, control: int, target: int, angle: Any) -> None:
        """Record a CP emission."""
        del circuit
        self.calls.append(("cp", control, target, angle))


class _MultiControlEmitPass:
    """Emit-pass stand-in for multi-controlled reduction tests."""

    def __init__(self, record_hook: bool = False) -> None:
        """Initialize with a recording emitter and optional hook.

        Args:
            record_hook (bool): When True, provide the irreducible
                multi-controlled gate hook and record its invocations
                instead of raising.
        """
        self._emitter = _RecordingEmitter()
        self._resolver = ValueResolver()
        self.hook_calls: list[tuple[Any, ...]] = []
        self._record_hook = record_hook

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
    """Three-controlled X on a hookless backend raises EmitError."""
    import pytest

    from qamomile.circuit.transpiler.errors import EmitError

    emit_pass = _MultiControlEmitPass()
    op = _fixed_gate(GateOperationType.X, 1)
    with pytest.raises(EmitError, match="3-controlled X"):
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
