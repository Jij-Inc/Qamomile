"""Count-only emitter for measuring multi-control ancilla demand.

A quantum segment's backend circuit is created with a fixed qubit count
before emission starts, so the multi-control ancilla pool must be sized in
advance. Rather than mirror the emit-time control-absorption rules in a
separate static walk — which has to stay in lockstep with the real
emitter or it will silently under-reserve — the demand is measured by
running the real emission machinery once against this no-op emitter and a
counting ancilla pool.

Every gate emission is a no-op and every structural query returns a value
that keeps the walk on its cascade-prone fallback path (``circuit_to_gate``
/ ``gate_controlled`` / ``gate_power`` / ``gate_inverse`` return None and
``supports_reusable_gates`` / ``supports_gate_inverse`` are False), so the
count is an upper bound and never touches a real circuit object. Capability
queries (``measurement_mode`` and the control-flow ``supports_*`` hooks)
delegate to the wrapped emitter so the walk takes the same unroll-versus-
native-loop decisions the real emission would.
"""

from __future__ import annotations

from typing import Any

from qamomile.circuit.transpiler.gate_emitter import MeasurementMode


class _CountingCircuit:
    """Placeholder circuit handed to the counting walk; holds no state."""


class _CountingParameter:
    """Arithmetic-absorbing stand-in for a runtime parameter.

    Angle expressions are evaluated (e.g. ``factor * gamma``, ``-angle``,
    linear combinations) before the no-op emit call, so the parameter a
    counting run produces must absorb every arithmetic operator and stay
    hashable — QURI Parts represents symbolic angles as
    ``{parameter: coefficient}`` dicts, so the parameter is used as a dict
    key. Comparison operators are intentionally absent: a counting walk
    that reaches a value-dependent structural branch should fail loudly
    rather than silently miscount.
    """

    def _absorb(self, other: Any) -> "_CountingParameter":
        """Return a fresh counting parameter, discarding ``other``.

        Args:
            other (Any): The other operand, ignored.

        Returns:
            _CountingParameter: A new placeholder parameter.
        """
        del other
        return _CountingParameter()

    def __add__(self, other: Any) -> "_CountingParameter":
        """Absorb ``self + other``."""
        return self._absorb(other)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "_CountingParameter":
        """Absorb ``self - other``."""
        return self._absorb(other)

    __rsub__ = __sub__

    def __mul__(self, other: Any) -> "_CountingParameter":
        """Absorb ``self * other``."""
        return self._absorb(other)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "_CountingParameter":
        """Absorb ``self / other``."""
        return self._absorb(other)

    __rtruediv__ = __truediv__

    def __floordiv__(self, other: Any) -> "_CountingParameter":
        """Absorb ``self // other``."""
        return self._absorb(other)

    __rfloordiv__ = __floordiv__

    def __mod__(self, other: Any) -> "_CountingParameter":
        """Absorb ``self % other``."""
        return self._absorb(other)

    __rmod__ = __mod__

    def __pow__(self, other: Any) -> "_CountingParameter":
        """Absorb ``self ** other``."""
        return self._absorb(other)

    __rpow__ = __pow__

    def __neg__(self) -> "_CountingParameter":
        """Absorb ``-self``."""
        return _CountingParameter()

    def __pos__(self) -> "_CountingParameter":
        """Absorb ``+self``."""
        return _CountingParameter()


class CountingEmitter:
    """A no-op gate emitter that lets a real emission walk run for counting.

    Wraps the real backend emitter, delegating capability queries so
    control-flow lowering matches real emission while turning every gate
    emission into a no-op and every gate/sub-circuit construction into a
    ``None`` that forces the cascade-prone fallback path. Used only by
    ``StandardEmitPass._count_multi_control_ancilla_demand``; the object is
    passed where a ``GateEmitter[T]`` is expected via a cast.
    """

    def __init__(self, real: Any) -> None:
        """Wrap a real emitter for a counting run.

        Args:
            real (Any): The backend emitter whose capability answers should
                be mirrored so the walk makes the same structural choices.
        """
        self._real = real

    @property
    def measurement_mode(self) -> MeasurementMode:
        """Delegate the measurement mode to the wrapped emitter.

        Returns:
            MeasurementMode: The real emitter's measurement mode, so the
                measurement lowering path matches real emission.
        """
        return self._real.measurement_mode

    def create_circuit(self, num_qubits: int, num_clbits: int) -> _CountingCircuit:
        """Return a placeholder circuit.

        Args:
            num_qubits (int): Ignored.
            num_clbits (int): Ignored.

        Returns:
            _CountingCircuit: A stateless placeholder.
        """
        del num_qubits, num_clbits
        return _CountingCircuit()

    def create_parameter(self, name: str) -> _CountingParameter:
        """Return an arithmetic-absorbing placeholder parameter.

        Args:
            name (str): Ignored.

        Returns:
            _CountingParameter: A placeholder runtime parameter.
        """
        del name
        return _CountingParameter()

    def combine_symbolic(self, kind: Any, lhs: Any, rhs: Any) -> _CountingParameter:
        """Return a placeholder for a symbolic angle combination.

        Args:
            kind (Any): Ignored binary-op kind.
            lhs (Any): Ignored left operand.
            rhs (Any): Ignored right operand.

        Returns:
            _CountingParameter: A placeholder symbolic angle.
        """
        del kind, lhs, rhs
        return _CountingParameter()

    def emit_h(self, circuit: Any, qubit: int) -> None:
        """No-op H."""

    def emit_x(self, circuit: Any, qubit: int) -> None:
        """No-op X."""

    def emit_y(self, circuit: Any, qubit: int) -> None:
        """No-op Y."""

    def emit_z(self, circuit: Any, qubit: int) -> None:
        """No-op Z."""

    def emit_s(self, circuit: Any, qubit: int) -> None:
        """No-op S."""

    def emit_t(self, circuit: Any, qubit: int) -> None:
        """No-op T."""

    def emit_sdg(self, circuit: Any, qubit: int) -> None:
        """No-op Sdg."""

    def emit_tdg(self, circuit: Any, qubit: int) -> None:
        """No-op Tdg."""

    def emit_rx(self, circuit: Any, qubit: int, angle: Any) -> None:
        """No-op RX."""

    def emit_ry(self, circuit: Any, qubit: int, angle: Any) -> None:
        """No-op RY."""

    def emit_rz(self, circuit: Any, qubit: int, angle: Any) -> None:
        """No-op RZ."""

    def emit_p(self, circuit: Any, qubit: int, angle: Any) -> None:
        """No-op P."""

    def emit_cx(self, circuit: Any, control: int, target: int) -> None:
        """No-op CX."""

    def emit_cz(self, circuit: Any, control: int, target: int) -> None:
        """No-op CZ."""

    def emit_swap(self, circuit: Any, qubit1: int, qubit2: int) -> None:
        """No-op SWAP."""

    def emit_cp(self, circuit: Any, control: int, target: int, angle: Any) -> None:
        """No-op CP."""

    def emit_rzz(self, circuit: Any, qubit1: int, qubit2: int, angle: Any) -> None:
        """No-op RZZ."""

    def emit_toffoli(
        self, circuit: Any, control1: int, control2: int, target: int
    ) -> None:
        """No-op Toffoli."""

    def emit_ch(self, circuit: Any, control: int, target: int) -> None:
        """No-op CH."""

    def emit_cy(self, circuit: Any, control: int, target: int) -> None:
        """No-op CY."""

    def emit_crx(self, circuit: Any, control: int, target: int, angle: Any) -> None:
        """No-op CRX."""

    def emit_cry(self, circuit: Any, control: int, target: int, angle: Any) -> None:
        """No-op CRY."""

    def emit_crz(self, circuit: Any, control: int, target: int, angle: Any) -> None:
        """No-op CRZ."""

    def emit_measure(self, circuit: Any, qubit: int, clbit: int) -> None:
        """No-op measurement."""

    def emit_barrier(self, circuit: Any, qubits: list[int]) -> None:
        """No-op barrier."""

    def circuit_to_gate(self, circuit: Any, name: str = "U") -> Any:
        """Force the fallback path by declining to build a reusable gate.

        Returns:
            Any: Always None.
        """
        del circuit, name
        return None

    def append_gate(self, circuit: Any, gate: Any, qubits: list[int]) -> None:
        """No-op append."""

    def gate_power(self, gate: Any, power: int) -> Any:
        """Decline to build a real powered gate.

        Returns:
            Any: Always None.
        """
        del gate, power
        return None

    def gate_controlled(self, gate: Any, num_controls: int) -> Any:
        """Decline to build a real controlled gate.

        Returns:
            Any: Always None.
        """
        del gate, num_controls
        return None

    def gate_inverse(self, gate: Any) -> Any:
        """Decline to build a real inverse gate.

        Returns:
            Any: Always None.
        """
        del gate
        return None

    def supports_reusable_gates(self) -> bool:
        """Force the gate-by-gate fallback path.

        Returns:
            bool: Always False.
        """
        return False

    def supports_gate_inverse(self) -> bool:
        """Force the inverse-by-emission fallback path.

        Returns:
            bool: Always False.
        """
        return False

    def supports_for_loop(self) -> bool:
        """Delegate native-for-loop support to the wrapped emitter.

        Returns:
            bool: The real emitter's answer, so loops unroll (or not) the
                same way real emission would.
        """
        return bool(self._real.supports_for_loop())

    def emit_for_loop_start(self, circuit: Any, indexset: range) -> Any:
        """No-op native for-loop start.

        Returns:
            Any: A null context (the counting walk emits the body once).
        """
        del circuit, indexset
        return None

    def emit_for_loop_end(self, circuit: Any, context: Any) -> None:
        """No-op native for-loop end."""

    def supports_if_else(self) -> bool:
        """Delegate native if/else support to the wrapped emitter.

        Returns:
            bool: The real emitter's answer.
        """
        return bool(self._real.supports_if_else())

    def emit_if_start(self, circuit: Any, clbit: int, value: int = 1) -> Any:
        """No-op native if start.

        Returns:
            Any: A null context.
        """
        del circuit, clbit, value
        return None

    def emit_else_start(self, circuit: Any, context: Any) -> None:
        """No-op native else start."""

    def emit_if_end(self, circuit: Any, context: Any) -> None:
        """No-op native if end."""

    def supports_while_loop(self) -> bool:
        """Delegate native while-loop support to the wrapped emitter.

        Returns:
            bool: The real emitter's answer.
        """
        return bool(self._real.supports_while_loop())

    def emit_while_start(self, circuit: Any, clbit: int, value: int = 1) -> Any:
        """No-op native while start.

        Returns:
            Any: A null context.
        """
        del circuit, clbit, value
        return None

    def emit_while_end(self, circuit: Any, context: Any) -> None:
        """No-op native while end."""
