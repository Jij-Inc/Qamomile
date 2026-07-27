"""Lower primitive gates with one or more quantum controls."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from qamomile.circuit.ir.operation.gate import GateOperation, GateOperationType
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.passes.emit_support.gate_emission import (
    reject_duplicate_physical_indices,
)

if TYPE_CHECKING:
    from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass


def emit_controlled_gate(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: GateOperation,
    control_idx: int,
    target_indices: list[int],
    bindings: dict[str, Any],
) -> None:
    """Emit a controlled version of a gate.

    Resolves the rotation angle for rotation-like gates and dispatches
    single-target gate kinds through
    :func:`emit_single_controlled_primitive`; ``SWAP`` is lowered here
    via its Fredkin conjugation because it needs two targets.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        op (GateOperation): Gate operation to control.
        control_idx (int): Physical control qubit.
        target_indices (list[int]): Physical qubits for the gate's own
            operands, in operand order. An empty list is a no-op.
        bindings (dict[str, Any]): Bindings used to resolve rotation
            angles.

    Raises:
        EmitError: If the gate type is unsupported in controlled
            decomposition, or a controlled SWAP has fewer than two
            targets.
    """
    if not target_indices:
        return

    if op.gate_type == GateOperationType.SWAP:
        if len(target_indices) < 2:
            raise EmitError(
                "Controlled-SWAP requires at least 2 target qubits.",
                operation="ControlledGate",
            )
        tgt_a = target_indices[0]
        tgt_b = target_indices[1]
        # Fredkin gate decomposition:
        #   CNOT(tgt_b, tgt_a)
        #   Toffoli(ctrl, tgt_a, tgt_b)
        #   CNOT(tgt_b, tgt_a)
        emit_pass._emitter.emit_cx(circuit, tgt_b, tgt_a)
        emit_pass._emitter.emit_toffoli(circuit, control_idx, tgt_a, tgt_b)
        emit_pass._emitter.emit_cx(circuit, tgt_b, tgt_a)
        return

    angle: Any = None
    if op.gate_type in (
        GateOperationType.P,
        GateOperationType.RX,
        GateOperationType.RY,
        GateOperationType.RZ,
    ):
        angle = emit_pass._resolve_angle(op, bindings)
    emit_single_controlled_primitive(
        emit_pass, circuit, op.gate_type, control_idx, target_indices[0], angle
    )


def emit_single_controlled_primitive(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    gate_type: GateOperationType | None,
    control_idx: int,
    target_idx: int,
    angle: Any,
) -> None:
    """Emit one singly-controlled single-qubit gate from a resolved angle.

    This is the single-control dispatch shared by
    :func:`emit_controlled_gate` (which resolves the angle from the IR
    operation first) and the Toffoli-cascade lowering of irreducible
    multi-controlled gates (which arrives with the angle already
    resolved). Fixed phase-family gates (``S`` / ``T`` and daggers) are
    emitted as controlled phases.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        gate_type (GateOperationType | None): Single-qubit gate kind.
            None (a gate operation without a type) is rejected like any
            other unsupported kind.
        control_idx (int): Physical control qubit.
        target_idx (int): Physical target qubit.
        angle (Any): Resolved rotation angle (concrete number or backend
            parameter expression) for ``P`` / ``RX`` / ``RY`` / ``RZ``;
            ignored for fixed gates.

    Raises:
        EmitError: If ``gate_type`` is not a single-qubit gate kind
            supported in controlled decomposition.
    """
    match gate_type:
        case GateOperationType.H:
            emit_pass._emitter.emit_ch(circuit, control_idx, target_idx)
        case GateOperationType.X:
            emit_pass._emitter.emit_cx(circuit, control_idx, target_idx)
        case GateOperationType.Y:
            emit_pass._emitter.emit_cy(circuit, control_idx, target_idx)
        case GateOperationType.Z:
            emit_pass._emitter.emit_cz(circuit, control_idx, target_idx)
        case GateOperationType.P:
            emit_pass._emitter.emit_cp(circuit, control_idx, target_idx, angle)
        case GateOperationType.RX:
            emit_pass._emitter.emit_crx(circuit, control_idx, target_idx, angle)
        case GateOperationType.RY:
            emit_pass._emitter.emit_cry(circuit, control_idx, target_idx, angle)
        case GateOperationType.RZ:
            emit_pass._emitter.emit_crz(circuit, control_idx, target_idx, angle)
        case GateOperationType.S:
            emit_pass._emitter.emit_cp(circuit, control_idx, target_idx, math.pi / 2)
        case GateOperationType.T:
            emit_pass._emitter.emit_cp(circuit, control_idx, target_idx, math.pi / 4)
        case GateOperationType.SDG:
            emit_pass._emitter.emit_cp(circuit, control_idx, target_idx, -math.pi / 2)
        case GateOperationType.TDG:
            emit_pass._emitter.emit_cp(circuit, control_idx, target_idx, -math.pi / 4)
        case _:
            raise EmitError(
                f"Unsupported gate type {gate_type!r} in controlled "
                f"block decomposition.",
                operation="ControlledGate",
            )


def _and_ladder_steps(
    control_indices: list[int], ancilla_indices: list[int]
) -> list[tuple[int, int, int]]:
    """Build the Toffoli chain that ANDs ``control_indices`` onto ancillas.

    The chain combines the first two controls onto ``ancilla_indices[0]``,
    then folds each subsequent control together with the previous ancilla,
    so the last ancilla (``ancilla_indices[len(control_indices) - 2]``)
    holds the logical AND of every control. Emitting the returned steps in
    order computes the AND; emitting them reversed uncomputes it.

    Args:
        control_indices (list[int]): Physical control qubits; at least two.
        ancilla_indices (list[int]): Clean ancilla qubits; at least
            ``len(control_indices) - 1`` entries.

    Returns:
        list[tuple[int, int, int]]: ``(control_a, control_b, target)``
            triples, one per Toffoli, in compute order.
    """
    steps: list[tuple[int, int, int]] = [
        (control_indices[0], control_indices[1], ancilla_indices[0])
    ]
    for i in range(2, len(control_indices)):
        steps.append(
            (control_indices[i], ancilla_indices[i - 2], ancilla_indices[i - 1])
        )
    return steps


def _emit_toffoli_steps(
    emitter: Any, circuit: Any, steps: Iterable[tuple[int, int, int]]
) -> None:
    """Emit a sequence of Toffoli gates for the given ``(a, b, target)`` steps.

    Args:
        emitter (Any): Backend gate emitter.
        circuit (Any): Backend circuit being emitted into.
        steps (Iterable[tuple[int, int, int]]): Toffoli triples, in the
            order they should be emitted (pass ``reversed(steps)`` to
            uncompute a ladder built by :func:`_and_ladder_steps`).

    Returns:
        None.
    """
    for control_a, control_b, target in steps:
        emitter.emit_toffoli(circuit, control_a, control_b, target)


def emit_multi_controlled_on_clean_ancillas(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    gate_type: GateOperationType,
    control_indices: list[int],
    target_idx: int,
    angle: Any,
    ancilla_indices: list[int],
) -> None:
    """Lower an irreducible multi-controlled gate via a Toffoli cascade.

    Implements the standard clean-ancilla construction (arXiv:2307.07478,
    Appendix A.3): the logical AND of the ``n`` controls is accumulated
    onto ``n - 1`` clean ancillas with a cascade of Toffoli gates, the
    gate is applied once under a single control (the last ancilla)
    through :func:`emit_single_controlled_primitive`, and the cascade is
    uncomputed in reverse order. The cost is ``2 * (n - 1)`` Toffoli
    gates plus one singly-controlled gate, and every ancilla returns to
    ``|0>``, so the same pool may be reused by subsequent gates.

    Unlike a dense ``2**(n+1)`` unitary-matrix lowering, this scales
    linearly in the control count and keeps rotation angles symbolic,
    so runtime-parametric multi-controlled rotations are supported.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        gate_type (GateOperationType): Single-qubit gate kind to apply
            under the controls.
        control_indices (list[int]): Physical control qubits; at least
            two.
        target_idx (int): Physical target qubit.
        angle (Any): Resolved rotation angle (concrete number or backend
            parameter expression) for rotation-like gates, or ``None``
            for fixed gates.
        ancilla_indices (list[int]): Clean (``|0>``) ancilla qubits;
            at least ``len(control_indices) - 1`` entries.

    Raises:
        EmitError: If fewer than two controls or too few ancillas are
            supplied (both indicate a caller bug), or the gate type is
            unsupported by the single-control dispatch.
    """
    num_controls = len(control_indices)
    if num_controls < 2:
        raise EmitError(
            "Toffoli-cascade lowering requires at least two controls; "
            f"got {num_controls}.",
            operation="ControlledGate",
        )
    if len(ancilla_indices) < num_controls - 1:
        raise EmitError(
            f"Toffoli-cascade lowering of a {num_controls}-controlled "
            f"{gate_type.name} needs {num_controls - 1} clean ancilla "
            f"qubit(s) but only {len(ancilla_indices)} were supplied.",
            operation="ControlledGate",
        )

    emitter = emit_pass._emitter
    cascade = _and_ladder_steps(control_indices, ancilla_indices)
    _emit_toffoli_steps(emitter, circuit, cascade)
    emit_single_controlled_primitive(
        emit_pass,
        circuit,
        gate_type,
        ancilla_indices[num_controls - 2],
        target_idx,
        angle,
    )
    _emit_toffoli_steps(emitter, circuit, reversed(cascade))


def emit_multi_controlled_gate(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: GateOperation,
    control_indices: list[int],
    target_indices: list[int],
    bindings: dict[str, Any],
) -> None:
    """Emit one gate under an arbitrary number of accumulated controls.

    Structurally reduces multi-qubit gate types to single-target forms
    by absorbing their own control qubits into the control set
    (``CX -> X``, ``CZ -> Z``, ``CP -> P``, ``TOFFOLI -> X``; ``SWAP``
    and ``RZZ`` via their standard CX conjugations), then emits:

    - one control: via :func:`emit_controlled_gate` (the existing
      single-control dispatch),
    - two controls on X / Z: via ``emit_toffoli`` (Z conjugated by H),
    - anything else: via the backend's
      ``_emit_irreducible_multi_controlled_gate`` hook, whose base
      implementation raises a descriptive ``EmitError``.

    Args:
        emit_pass (StandardEmitPass): Active emit pass. Subclass
            overrides of ``_emit_irreducible_multi_controlled_gate``
            are respected for the irreducible tail.
        circuit (Any): Backend circuit being emitted into.
        op (GateOperation): Gate operation whose operands are already
            resolved to ``target_indices``.
        control_indices (list[int]): Physical control qubits. Must be
            non-empty.
        target_indices (list[int]): Physical qubits for the gate's own
            operands, in operand order.
        bindings (dict[str, Any]): Bindings used to resolve rotation
            angles.

    Raises:
        EmitError: If ``control_indices`` is empty, the gate has fewer
            resolved targets than its type requires, or the reduction
            bottoms out on a backend without multi-control support.
    """
    gate_type = op.gate_type
    if not control_indices:
        raise EmitError(
            "emit_multi_controlled_gate requires at least one control.",
            operation="ControlledGate",
        )

    # Inner gates reached through the fallback (block-decomposition) walker
    # emit directly via ``emit_toffoli`` / ``emit_cx`` etc. and never pass
    # through ``emit_gate``'s aliasing check. This is the single choke point
    # for every controlled inner gate, so re-run the shared check on the
    # combined control + target set: a body ``cx(qs[i], qs[j])`` with i == j at
    # runtime, or an inner control that coincides with a target, is caught here.
    reject_duplicate_physical_indices(
        f"controlled {gate_type.name if gate_type else 'gate'}",
        control_indices + target_indices,
    )

    def _require_targets(count: int) -> None:
        """Validate that the gate received enough resolved targets.

        Args:
            count (int): Minimum number of physical targets required.

        Raises:
            EmitError: If fewer than ``count`` targets were resolved.
        """
        if len(target_indices) < count:
            gate_name = gate_type.name if gate_type is not None else "<unknown>"
            raise EmitError(
                f"Controlled-{gate_name} requires {count} target "
                f"qubit(s); got {len(target_indices)}.",
                operation="ControlledGate",
            )

    match gate_type:
        case GateOperationType.CX:
            _require_targets(2)
            _emit_mc_x(
                emit_pass,
                circuit,
                [*control_indices, target_indices[0]],
                target_indices[1],
            )
            return
        case GateOperationType.CZ:
            _require_targets(2)
            _emit_mc_z(
                emit_pass,
                circuit,
                [*control_indices, target_indices[0]],
                target_indices[1],
            )
            return
        case GateOperationType.TOFFOLI:
            _require_targets(3)
            _emit_mc_x(
                emit_pass,
                circuit,
                [*control_indices, target_indices[0], target_indices[1]],
                target_indices[2],
            )
            return
        case GateOperationType.CP:
            _require_targets(2)
            angle = emit_pass._resolve_angle(op, bindings)
            _emit_mc_rotation(
                emit_pass,
                circuit,
                GateOperationType.P,
                [*control_indices, target_indices[0]],
                target_indices[1],
                angle,
            )
            return
        case GateOperationType.RZZ:
            _require_targets(2)
            angle = emit_pass._resolve_angle(op, bindings)
            # RZZ = CX(t0, t1) . RZ(t1) . CX(t0, t1); the CX pair is
            # self-inverse, so only the RZ needs the controls.
            emit_pass._emitter.emit_cx(circuit, target_indices[0], target_indices[1])
            _emit_mc_rotation(
                emit_pass,
                circuit,
                GateOperationType.RZ,
                control_indices,
                target_indices[1],
                angle,
            )
            emit_pass._emitter.emit_cx(circuit, target_indices[0], target_indices[1])
            return
        case GateOperationType.SWAP:
            _require_targets(2)
            # Generalized Fredkin: SWAP(a, b) = CX(b, a) . CX(a, b)
            # . CX(b, a); controlling only the middle CX controls the
            # whole SWAP.
            emit_pass._emitter.emit_cx(circuit, target_indices[1], target_indices[0])
            _emit_mc_x(
                emit_pass,
                circuit,
                [*control_indices, target_indices[0]],
                target_indices[1],
            )
            emit_pass._emitter.emit_cx(circuit, target_indices[1], target_indices[0])
            return
        case _:
            pass

    if len(control_indices) == 1:
        emit_controlled_gate(
            emit_pass, circuit, op, control_indices[0], target_indices, bindings
        )
        return

    _require_targets(1)
    target_idx = target_indices[0]
    match gate_type:
        case GateOperationType.X:
            _emit_mc_x(emit_pass, circuit, control_indices, target_idx)
        case GateOperationType.Z:
            _emit_mc_z(emit_pass, circuit, control_indices, target_idx)
        case GateOperationType.S:
            _emit_mc_rotation(
                emit_pass,
                circuit,
                GateOperationType.P,
                control_indices,
                target_idx,
                math.pi / 2,
            )
        case GateOperationType.T:
            _emit_mc_rotation(
                emit_pass,
                circuit,
                GateOperationType.P,
                control_indices,
                target_idx,
                math.pi / 4,
            )
        case GateOperationType.SDG:
            _emit_mc_rotation(
                emit_pass,
                circuit,
                GateOperationType.P,
                control_indices,
                target_idx,
                -math.pi / 2,
            )
        case GateOperationType.TDG:
            _emit_mc_rotation(
                emit_pass,
                circuit,
                GateOperationType.P,
                control_indices,
                target_idx,
                -math.pi / 4,
            )
        case (
            GateOperationType.P
            | GateOperationType.RX
            | GateOperationType.RY
            | GateOperationType.RZ
        ):
            angle = emit_pass._resolve_angle(op, bindings)
            _emit_mc_rotation(
                emit_pass,
                circuit,
                gate_type,
                control_indices,
                target_idx,
                angle,
            )
        case _:
            _emit_irreducible(
                emit_pass, circuit, gate_type, control_indices, target_idx, None
            )


def _emit_mc_x(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    control_indices: list[int],
    target_idx: int,
) -> None:
    """Emit an X gate under one or more controls.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        control_indices (list[int]): Physical control qubits (>= 1).
        target_idx (int): Physical target qubit.

    Raises:
        EmitError: If three or more controls are required and the
            backend has no multi-controlled gate hook.
    """
    if len(control_indices) == 1:
        emit_pass._emitter.emit_cx(circuit, control_indices[0], target_idx)
    elif len(control_indices) == 2:
        emit_pass._emitter.emit_toffoli(
            circuit, control_indices[0], control_indices[1], target_idx
        )
    else:
        _emit_irreducible(
            emit_pass, circuit, GateOperationType.X, control_indices, target_idx, None
        )


def _emit_mc_z(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    control_indices: list[int],
    target_idx: int,
) -> None:
    """Emit a Z gate under one or more controls.

    Two-control Z is conjugated into a Toffoli by Hadamards on the
    target (``Z = H X H``).

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        control_indices (list[int]): Physical control qubits (>= 1).
        target_idx (int): Physical target qubit.

    Raises:
        EmitError: If three or more controls are required and the
            backend has no multi-controlled gate hook.
    """
    if len(control_indices) == 1:
        emit_pass._emitter.emit_cz(circuit, control_indices[0], target_idx)
    elif len(control_indices) == 2:
        emit_pass._emitter.emit_h(circuit, target_idx)
        emit_pass._emitter.emit_toffoli(
            circuit, control_indices[0], control_indices[1], target_idx
        )
        emit_pass._emitter.emit_h(circuit, target_idx)
    else:
        _emit_irreducible(
            emit_pass, circuit, GateOperationType.Z, control_indices, target_idx, None
        )


def _emit_mc_rotation(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    gate_type: GateOperationType,
    control_indices: list[int],
    target_idx: int,
    angle: Any,
) -> None:
    """Emit a rotation-like gate under one or more controls.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        gate_type (GateOperationType): One of ``P`` / ``RX`` / ``RY`` /
            ``RZ``.
        control_indices (list[int]): Physical control qubits (>= 1).
        target_idx (int): Physical target qubit.
        angle (Any): Resolved rotation angle (concrete float or backend
            parameter expression).

    Raises:
        EmitError: If two or more controls are required and the backend
            has no multi-controlled gate hook, or ``gate_type`` is not
            rotation-like.
    """
    if len(control_indices) == 1:
        control_idx = control_indices[0]
        match gate_type:
            case GateOperationType.P:
                emit_pass._emitter.emit_cp(circuit, control_idx, target_idx, angle)
            case GateOperationType.RX:
                emit_pass._emitter.emit_crx(circuit, control_idx, target_idx, angle)
            case GateOperationType.RY:
                emit_pass._emitter.emit_cry(circuit, control_idx, target_idx, angle)
            case GateOperationType.RZ:
                emit_pass._emitter.emit_crz(circuit, control_idx, target_idx, angle)
            case _:
                raise EmitError(
                    f"Gate type {gate_type!r} is not rotation-like.",
                    operation="ControlledGate",
                )
        return
    _emit_irreducible(emit_pass, circuit, gate_type, control_indices, target_idx, angle)


def _emit_irreducible(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    gate_type: GateOperationType | None,
    control_indices: list[int],
    target_idx: int,
    angle: Any,
) -> None:
    """Dispatch an irreducible multi-controlled gate to the backend hook.

    Args:
        emit_pass (StandardEmitPass): Active emit pass. Its
            ``_emit_irreducible_multi_controlled_gate`` method (base
            implementation raises; backends may override) receives the
            gate.
        circuit (Any): Backend circuit being emitted into.
        gate_type (GateOperationType | None): Single-qubit gate kind.
        control_indices (list[int]): Physical control qubits.
        target_idx (int): Physical target qubit.
        angle (Any): Resolved rotation angle, or ``None`` for fixed
            gates.

    Raises:
        EmitError: If ``gate_type`` is missing or the backend's hook
            rejects the gate.
    """
    if gate_type is None:
        raise EmitError(
            f"Cannot emit {len(control_indices)}-controlled gate without a gate type.",
            operation="ControlledGate",
        )
    emit_pass._emit_irreducible_multi_controlled_gate(
        circuit, gate_type, control_indices, target_idx, angle
    )
