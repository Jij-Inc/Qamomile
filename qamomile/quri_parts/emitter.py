"""QURI Parts GateEmitter implementation.

This module provides QuriPartsGateEmitter, which implements the GateEmitter
protocol for QURI Parts backends.

QURI Parts uses LinearMappedUnboundParametricQuantumCircuit for parametric
circuits. Angles are specified as dictionaries: {param: coeff, CONST: offset}.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

from qamomile.circuit.ir.operation.arithmetic_operations import BinOpKind
from qamomile.circuit.transpiler.decompositions import (  # noqa: F401 -- recipe data
    CH_DECOMPOSITION,
    CP_DECOMPOSITION,
    CRY_DECOMPOSITION,
    CRZ_DECOMPOSITION,
    CY_DECOMPOSITION,
)
from qamomile.circuit.transpiler.gate_emitter import MeasurementMode

from .exceptions import QamomileQuriPartsTranspileError

if TYPE_CHECKING:
    from quri_parts.circuit import (
        LinearMappedUnboundParametricQuantumCircuit,
        Parameter,
    )


def _to_linear_form(angle: Any) -> dict[Any, float]:
    """Normalize an angle operand to QURI Parts linear-combination form.

    Three input shapes occur in the symbolic emit path:

    - ``int`` / ``float``: a concrete numeric offset; lifted to
      ``{CONST: float(angle)}``.
    - QURI Parts ``Parameter``: a single parameter atom; lifted to
      ``{angle: 1.0}``.
    - ``dict``: an existing linear form (typically the result of an
      earlier ``combine_symbolic`` call stored in ``EmitContext._values``);
      returned as a fresh ``dict`` so callers can mutate the result
      without aliasing the stored intermediate.

    Args:
        angle: The operand to normalize.

    Returns:
        A fresh ``dict`` mapping QURI Parts ``Parameter`` (including
        ``CONST``) to ``float`` coefficients.
    """
    if isinstance(angle, (int, float)):
        from quri_parts.circuit import CONST

        return {CONST: float(angle)}
    if isinstance(angle, dict):
        return dict(angle)
    return {angle: 1.0}


def _is_pure_const(form: dict[Any, float]) -> bool:
    """Return True when ``form`` has no parameter atoms (CONST-only or empty).

    Used by ``combine_symbolic`` to detect linear-only multiplication and
    division: ``param * scalar`` is allowed, ``param * param`` is not.
    """
    from quri_parts.circuit import CONST

    return all(k is CONST for k in form)


def _add_forms(
    lhs: dict[Any, float],
    rhs: dict[Any, float],
) -> dict[Any, float]:
    """Coefficient-wise add two linear forms into a fresh dict."""
    out = dict(lhs)
    for k, v in rhs.items():
        out[k] = out.get(k, 0.0) + v
    return out


def _scale_form(form: dict[Any, float], scalar: float) -> dict[Any, float]:
    """Multiply every coefficient of ``form`` by ``scalar`` into a fresh dict."""
    return {k: v * scalar for k, v in form.items()}


def _sub_forms(
    lhs: dict[Any, float],
    rhs: dict[Any, float],
) -> dict[Any, float]:
    """Subtract ``rhs`` from ``lhs`` coefficient-wise into a fresh dict."""
    return _add_forms(lhs, _scale_form(rhs, -1.0))


def _const_value(form: dict[Any, float]) -> float:
    """Extract the CONST coefficient from a CONST-only linear form."""
    from quri_parts.circuit import CONST

    return form.get(CONST, 0.0)


class QuriPartsGateEmitter:
    """GateEmitter implementation for QURI Parts.

    Emits individual quantum gates to QURI Parts circuits.

    QURI Parts parametric circuits accept angles in dictionary form:
    {parameter: coefficient, CONST: constant_offset}
    """

    @property
    def measurement_mode(self) -> MeasurementMode:
        """QURI Parts always uses STATIC measurement (sampler handles it)."""
        return MeasurementMode.STATIC

    def __init__(self) -> None:
        """Initialize the emitter."""
        self._param_map: dict[str, "Parameter"] = {}
        self._current_circuit: "LinearMappedUnboundParametricQuantumCircuit | None" = (
            None
        )

    def create_circuit(
        self, num_qubits: int, num_clbits: int
    ) -> "LinearMappedUnboundParametricQuantumCircuit":
        """Create a new QURI Parts parametric circuit.

        Note: QURI Parts does not support classical bits in circuits.
        The num_clbits parameter is accepted for interface compatibility
        but is not used.
        """
        import quri_parts.circuit as qp_c

        circuit = qp_c.LinearMappedUnboundParametricQuantumCircuit(num_qubits)
        self._current_circuit = circuit
        self._param_map.clear()
        return circuit

    def create_parameter(self, name: str) -> "Parameter":
        """Create a QURI Parts parameter and register it with the circuit."""
        if self._current_circuit is None:
            raise RuntimeError("create_circuit must be called before create_parameter")

        if name not in self._param_map:
            param = self._current_circuit.add_parameter(name)
            self._param_map[name] = param
        return self._param_map[name]

    def _make_angle_dict(self, angle: float | Any) -> "dict[Parameter, float] | float":
        """Convert angle to QURI Parts format.

        Three shapes are accepted:

        - ``int``/``float``: returned as ``float`` for the non-parametric
          gate API (``add_RX_gate`` etc.).
        - ``dict``: a linear-combination form already produced by
          ``combine_symbolic`` (e.g. ``{gamma: 0.5, CONST: 0.1}``). Passed
          through unchanged — QURI Parts'
          ``LinearMappedUnboundParametricQuantumCircuit.add_Parametric*``
          API consumes this form natively.
        - QURI Parts ``Parameter``: a single parameter atom (no BinOp came
          through), wrapped as ``{angle: 1.0}``.

        Args:
            angle: The angle value to convert.

        Returns:
            ``float`` for concrete angles, otherwise a
            ``dict[Parameter, float]`` linear combination.
        """
        if isinstance(angle, (int, float)):
            return float(angle)
        if isinstance(angle, dict):
            return angle
        return {angle: 1.0}

    def combine_symbolic(
        self,
        kind: BinOpKind,
        lhs: Any,
        rhs: Any,
    ) -> "dict[Parameter, float]":
        """Combine two angle operands as a QURI Parts linear-combination dict.

        QURI Parts' ``Parameter`` is Rust-backed and exposes no Python
        arithmetic operators (``param * float`` raises ``TypeError``).
        The only supported representation for parametric angles is the
        linear-combination dict consumed by
        ``LinearMappedUnboundParametricQuantumCircuit.add_Parametric*``,
        which fundamentally cannot express non-linear combinations
        (parameter × parameter, parameter ** n, etc.). This override
        produces such a dict for every linear case and raises a clear
        ``QamomileQuriPartsTranspileError`` for non-linear cases instead
        of letting the underlying ``TypeError`` bubble up.

        Args:
            kind: The ``BinOpKind`` from the IR.
            lhs: Left operand (numeric, QURI Parts ``Parameter``, or an
                existing linear-combination ``dict``).
            rhs: Right operand (same shapes).

        Returns:
            A fresh ``dict[Parameter, float]`` representing the linear
            combination of the operands.

        Raises:
            QamomileQuriPartsTranspileError: When the operation cannot be
                expressed as a linear combination (e.g.
                ``param * param``, ``param / param``, ``param ** n``,
                ``param // n``, division by zero in the symbolic path).
        """
        lf_lhs = _to_linear_form(lhs)
        lf_rhs = _to_linear_form(rhs)

        match kind:
            case BinOpKind.ADD:
                return _add_forms(lf_lhs, lf_rhs)
            case BinOpKind.SUB:
                return _sub_forms(lf_lhs, lf_rhs)
            case BinOpKind.MUL:
                if _is_pure_const(lf_rhs):
                    return _scale_form(lf_lhs, _const_value(lf_rhs))
                if _is_pure_const(lf_lhs):
                    return _scale_form(lf_rhs, _const_value(lf_lhs))
                raise QamomileQuriPartsTranspileError(
                    "QURI Parts backend supports only linear combinations of "
                    "parameters; parameter * parameter is non-linear and "
                    "cannot be expressed as a LinearMappedUnboundParametric "
                    "angle. Bind one side to a concrete value first."
                )
            case BinOpKind.DIV:
                if not _is_pure_const(lf_rhs):
                    raise QamomileQuriPartsTranspileError(
                        "QURI Parts backend supports only linear combinations "
                        "of parameters; division by a parameter is non-linear "
                        "and cannot be expressed as a LinearMappedUnboundParametric "
                        "angle. Bind the divisor to a concrete value first."
                    )
                divisor = _const_value(lf_rhs)
                if divisor == 0:
                    raise QamomileQuriPartsTranspileError(
                        "QURI Parts backend: division by zero in symbolic angle."
                    )
                return _scale_form(lf_lhs, 1.0 / divisor)
            case BinOpKind.POW | BinOpKind.FLOORDIV:
                raise QamomileQuriPartsTranspileError(
                    f"QURI Parts backend supports only linear combinations of "
                    f"parameters; '{kind.name}' is not supported on parametric "
                    f"angles. Bind the parameter to a concrete value first."
                )
            case _:
                raise QamomileQuriPartsTranspileError(
                    f"QURI Parts backend: unsupported BinOpKind '{kind}' on "
                    f"parametric angle."
                )

    # Single-qubit gates (no parameters)
    def emit_h(
        self, circuit: "LinearMappedUnboundParametricQuantumCircuit", qubit: int
    ) -> None:
        """Emit Hadamard gate."""
        circuit.add_H_gate(qubit)

    def emit_x(
        self, circuit: "LinearMappedUnboundParametricQuantumCircuit", qubit: int
    ) -> None:
        """Emit Pauli-X gate."""
        circuit.add_X_gate(qubit)

    def emit_y(
        self, circuit: "LinearMappedUnboundParametricQuantumCircuit", qubit: int
    ) -> None:
        """Emit Pauli-Y gate."""
        circuit.add_Y_gate(qubit)

    def emit_z(
        self, circuit: "LinearMappedUnboundParametricQuantumCircuit", qubit: int
    ) -> None:
        """Emit Pauli-Z gate."""
        circuit.add_Z_gate(qubit)

    def emit_s(
        self, circuit: "LinearMappedUnboundParametricQuantumCircuit", qubit: int
    ) -> None:
        """Emit S (phase) gate."""
        circuit.add_S_gate(qubit)

    def emit_t(
        self, circuit: "LinearMappedUnboundParametricQuantumCircuit", qubit: int
    ) -> None:
        """Emit T gate."""
        circuit.add_T_gate(qubit)

    def emit_sdg(
        self, circuit: "LinearMappedUnboundParametricQuantumCircuit", qubit: int
    ) -> None:
        """Emit S-dagger (inverse S) gate."""
        circuit.add_Sdag_gate(qubit)

    def emit_tdg(
        self, circuit: "LinearMappedUnboundParametricQuantumCircuit", qubit: int
    ) -> None:
        """Emit T-dagger (inverse T) gate."""
        circuit.add_Tdag_gate(qubit)

    # Single-qubit rotation gates
    def emit_rx(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        qubit: int,
        angle: float | Any,
    ) -> None:
        """Emit RX rotation gate."""
        angle_dict = self._make_angle_dict(angle)
        if isinstance(angle_dict, (int, float)):
            circuit.add_RX_gate(qubit, angle_dict)
        else:
            circuit.add_ParametricRX_gate(qubit, angle_dict)

    def emit_ry(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        qubit: int,
        angle: float | Any,
    ) -> None:
        """Emit RY rotation gate."""
        angle_dict = self._make_angle_dict(angle)
        if isinstance(angle_dict, (int, float)):
            circuit.add_RY_gate(qubit, angle_dict)
        else:
            circuit.add_ParametricRY_gate(qubit, angle_dict)

    def emit_rz(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        qubit: int,
        angle: float | Any,
    ) -> None:
        """Emit RZ rotation gate."""
        angle_dict = self._make_angle_dict(angle)
        if isinstance(angle_dict, (int, float)):
            circuit.add_RZ_gate(qubit, angle_dict)
        else:
            circuit.add_ParametricRZ_gate(qubit, angle_dict)

    def emit_p(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        qubit: int,
        angle: float | Any,
    ) -> None:
        """Emit Phase gate using U1.

        P(θ) = U1(θ) = diag(1, e^{iθ}).
        For non-parametric angles we use the native U1 gate which is
        mathematically identical to the Phase gate.
        For parametric angles we fall back to ParametricRZ because QURI Parts
        does not provide a ParametricU1 gate. RZ differs only by a global
        phase: P(θ) = e^{iθ/2} · RZ(θ), which is physically irrelevant for
        single-qubit usage. Controlled-phase (CP) has its own decomposition
        and does not go through this path.
        """
        angle_dict = self._make_angle_dict(angle)
        if isinstance(angle_dict, (int, float)):
            circuit.add_U1_gate(qubit, angle_dict)
        else:
            circuit.add_ParametricRZ_gate(qubit, angle_dict)

    # Two-qubit gates
    def emit_cx(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        control: int,
        target: int,
    ) -> None:
        """Emit CNOT (controlled-X) gate."""
        circuit.add_CNOT_gate(control, target)

    def emit_cz(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        control: int,
        target: int,
    ) -> None:
        """Emit controlled-Z gate."""
        circuit.add_CZ_gate(control, target)

    def emit_swap(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        qubit1: int,
        qubit2: int,
    ) -> None:
        """Emit SWAP gate."""
        circuit.add_SWAP_gate(qubit1, qubit2)

    # Two-qubit rotation gates
    def emit_cp(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        control: int,
        target: int,
        angle: float | Any,
    ) -> None:
        """Emit controlled-Phase gate using decomposition.

        Follows the shared CP_DECOMPOSITION recipe from
        ``qamomile.circuit.transpiler.decompositions``.
        Inlined here because QURI Parts parametric angles use dict representation.
        """
        angle_dict = self._make_angle_dict(angle)
        if isinstance(angle_dict, (int, float)):
            half_angle = angle_dict / 2
            circuit.add_RZ_gate(target, half_angle)
            circuit.add_CNOT_gate(control, target)
            circuit.add_RZ_gate(target, -half_angle)
            circuit.add_CNOT_gate(control, target)
            circuit.add_RZ_gate(control, half_angle)
        else:
            # Parametric case
            half_angle_dict = {k: v / 2 for k, v in angle_dict.items()}
            neg_half_angle_dict = {k: -v / 2 for k, v in angle_dict.items()}
            circuit.add_ParametricRZ_gate(target, half_angle_dict)
            circuit.add_CNOT_gate(control, target)
            circuit.add_ParametricRZ_gate(target, neg_half_angle_dict)
            circuit.add_CNOT_gate(control, target)
            circuit.add_ParametricRZ_gate(control, half_angle_dict)

    def emit_rzz(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        qubit1: int,
        qubit2: int,
        angle: float | Any,
    ) -> None:
        """Emit RZZ gate using ParametricPauliRotation."""
        # QURI Parts pauli_ids: 1=X, 2=Y, 3=Z
        angle_dict = self._make_angle_dict(angle)
        if isinstance(angle_dict, (int, float)):
            circuit.add_PauliRotation_gate([qubit1, qubit2], [3, 3], angle_dict)
        else:
            circuit.add_ParametricPauliRotation_gate(
                [qubit1, qubit2], [3, 3], angle_dict
            )

    # Three-qubit gates
    def emit_toffoli(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        control1: int,
        control2: int,
        target: int,
    ) -> None:
        """Emit Toffoli (CCX) gate."""
        circuit.add_TOFFOLI_gate(control1, control2, target)

    # Controlled single-qubit gates (decomposition required)
    def emit_ch(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        control: int,
        target: int,
    ) -> None:
        """Emit controlled-Hadamard gate using decomposition.

        Follows the shared CH_DECOMPOSITION recipe from
        ``qamomile.circuit.transpiler.decompositions``.
        """
        circuit.add_RY_gate(target, math.pi / 4)
        circuit.add_CNOT_gate(control, target)
        circuit.add_RY_gate(target, -math.pi / 4)

    def emit_cy(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        control: int,
        target: int,
    ) -> None:
        """Emit controlled-Y gate using decomposition.

        Follows the shared CY_DECOMPOSITION recipe from
        ``qamomile.circuit.transpiler.decompositions``.
        """
        circuit.add_Sdag_gate(target)
        circuit.add_CNOT_gate(control, target)
        circuit.add_S_gate(target)

    def emit_crx(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        control: int,
        target: int,
        angle: float | Any,
    ) -> None:
        """Emit controlled-RX gate using decomposition.

        CRX(ctrl, tgt, θ) =
            RZ(tgt, π/2)
            CNOT(ctrl, tgt)
            RY(tgt, -θ/2)
            CNOT(ctrl, tgt)
            RY(tgt, θ/2)
            RZ(tgt, -π/2)
        """
        angle_dict = self._make_angle_dict(angle)
        if isinstance(angle_dict, (int, float)):
            half_angle = angle_dict / 2
            circuit.add_RZ_gate(target, math.pi / 2)
            circuit.add_CNOT_gate(control, target)
            circuit.add_RY_gate(target, -half_angle)
            circuit.add_CNOT_gate(control, target)
            circuit.add_RY_gate(target, half_angle)
            circuit.add_RZ_gate(target, -math.pi / 2)
        else:
            half_angle_dict = {k: v / 2 for k, v in angle_dict.items()}
            neg_half_angle_dict = {k: -v / 2 for k, v in angle_dict.items()}
            circuit.add_RZ_gate(target, math.pi / 2)
            circuit.add_CNOT_gate(control, target)
            circuit.add_ParametricRY_gate(target, neg_half_angle_dict)
            circuit.add_CNOT_gate(control, target)
            circuit.add_ParametricRY_gate(target, half_angle_dict)
            circuit.add_RZ_gate(target, -math.pi / 2)

    def emit_cry(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        control: int,
        target: int,
        angle: float | Any,
    ) -> None:
        """Emit controlled-RY gate using decomposition.

        Follows the shared CRY_DECOMPOSITION recipe from
        ``qamomile.circuit.transpiler.decompositions``.
        Inlined here because QURI Parts parametric angles use dict representation.
        """
        angle_dict = self._make_angle_dict(angle)
        if isinstance(angle_dict, (int, float)):
            half_angle = angle_dict / 2
            circuit.add_RY_gate(target, half_angle)
            circuit.add_CNOT_gate(control, target)
            circuit.add_RY_gate(target, -half_angle)
            circuit.add_CNOT_gate(control, target)
        else:
            half_angle_dict = {k: v / 2 for k, v in angle_dict.items()}
            neg_half_angle_dict = {k: -v / 2 for k, v in angle_dict.items()}
            circuit.add_ParametricRY_gate(target, half_angle_dict)
            circuit.add_CNOT_gate(control, target)
            circuit.add_ParametricRY_gate(target, neg_half_angle_dict)
            circuit.add_CNOT_gate(control, target)

    def emit_crz(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        control: int,
        target: int,
        angle: float | Any,
    ) -> None:
        """Emit controlled-RZ gate using decomposition.

        Follows the shared CRZ_DECOMPOSITION recipe from
        ``qamomile.circuit.transpiler.decompositions``.
        Inlined here because QURI Parts parametric angles use dict representation.
        """
        angle_dict = self._make_angle_dict(angle)
        if isinstance(angle_dict, (int, float)):
            half_angle = angle_dict / 2
            circuit.add_RZ_gate(target, half_angle)
            circuit.add_CNOT_gate(control, target)
            circuit.add_RZ_gate(target, -half_angle)
            circuit.add_CNOT_gate(control, target)
        else:
            half_angle_dict = {k: v / 2 for k, v in angle_dict.items()}
            neg_half_angle_dict = {k: -v / 2 for k, v in angle_dict.items()}
            circuit.add_ParametricRZ_gate(target, half_angle_dict)
            circuit.add_CNOT_gate(control, target)
            circuit.add_ParametricRZ_gate(target, neg_half_angle_dict)
            circuit.add_CNOT_gate(control, target)

    # Measurement - QURI Parts doesn't support mid-circuit measurements
    #
    # When measurement_mode is STATIC, the sampler returns an all-qubit
    # bitstring indexed by qubit position.  The transpiler pipeline
    # needs a clbit->qubit mapping to decode partial measurements
    # correctly.  The measurement_mode property (defined above) tells
    # StandardEmitPass to build that mapping.

    def emit_measure(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        qubit: int,
        clbit: int,
    ) -> None:
        """No-op: QURI Parts circuits don't support measurement gates.

        Measurement is handled separately by samplers/estimators.
        """
        pass

    # Barrier - QURI Parts doesn't support barriers
    def emit_barrier(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        qubits: list[int],
    ) -> None:
        """No-op: QURI Parts doesn't support barrier instructions."""
        pass

    # Sub-circuit support - limited in QURI Parts
    def circuit_to_gate(
        self, circuit: "LinearMappedUnboundParametricQuantumCircuit", name: str = "U"
    ) -> Any:
        """QURI Parts doesn't support converting circuits to gates.

        Returns None to signal that manual decomposition should be used.
        """
        return None

    def append_gate(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        gate: Any,
        qubits: list[int],
    ) -> None:
        """Append a gate to the circuit.

        Since circuit_to_gate returns None, this is only called with
        QURI Parts native gates which can be extended into the circuit.
        """
        if gate is not None:
            try:
                circuit.extend(gate)
            except Exception as e:
                warnings.warn(
                    f"Failed to append gate to QURI Parts circuit: {e}. "
                    f"Falling back to manual decomposition.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def gate_power(self, gate: Any, power: int) -> Any:
        """Create gate raised to a power.

        Returns None since QURI Parts doesn't support this natively.
        """
        return None

    def gate_controlled(self, gate: Any, num_controls: int) -> Any:
        """Create controlled version of a gate.

        Returns None since QURI Parts doesn't support this natively.
        """
        return None

    # Control flow support - not supported by QURI Parts
    def supports_for_loop(self) -> bool:
        """QURI Parts does not support native for loops."""
        return False

    def supports_if_else(self) -> bool:
        """QURI Parts does not support native if/else."""
        return False

    def supports_while_loop(self) -> bool:
        """QURI Parts does not support native while loops."""
        return False
