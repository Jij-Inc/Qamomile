"""QURI Parts GateEmitter implementation.

This module provides QuriPartsGateEmitter, which implements the GateEmitter
protocol for QURI Parts backends.

QURI Parts uses LinearMappedUnboundParametricQuantumCircuit for parametric
circuits. Angles are specified as dictionaries: {param: coeff, CONST: offset}.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, TYPE_CHECKING

from qamomile.circuit.transpiler.value_resolution import is_concrete_real_number

if TYPE_CHECKING:
    from quri_parts.circuit import (
        LinearMappedUnboundParametricQuantumCircuit,
        Parameter,
    )


class QuriPartsParamExpr:
    """Affine combination of QURI Parts Parameters with a constant offset.

    QURI Parts ``Parameter`` objects do not support arithmetic.  This
    wrapper allows ``StandardEmitPass._evaluate_binop`` to compose
    parameter expressions (e.g. ``a + b``, ``params[1] + 0.5``) which
    are later converted to the ``{Parameter: coeff, CONST: offset}``
    dict format that QURI Parts parametric gates accept.
    """

    __slots__ = ("terms", "const")

    def __init__(
        self,
        terms: dict["Parameter", float] | None = None,
        const: float = 0.0,
    ) -> None:
        """Initialize an affine parameter expression."""
        self.terms = self._normalize_terms(terms if terms is not None else {})
        self.const = const

    @staticmethod
    def _normalize_terms(terms: dict["Parameter", float]) -> dict["Parameter", float]:
        """Drop zero-coefficient terms from an affine expression."""
        return {param: coeff for param, coeff in terms.items() if coeff != 0.0}

    def is_constant(self) -> bool:
        """Return whether the expression has no symbolic parameter terms."""
        return not self.terms

    @staticmethod
    def _coerce_constant(other: Any) -> float | None:
        """Convert numeric or constant-only expressions to a float."""
        if is_concrete_real_number(other):
            return float(other)
        if isinstance(other, QuriPartsParamExpr) and other.is_constant():
            return other.const
        return None

    @staticmethod
    def _non_affine_error(detail: str) -> TypeError:
        """Build an error for unsupported nonlinear parameter arithmetic."""
        return TypeError(
            "QURI Parts supports only affine parameter expressions; "
            f"{detail} is not supported."
        )

    # -- arithmetic ----------------------------------------------------------

    def __add__(self, other: Any) -> "QuriPartsParamExpr":
        if isinstance(other, QuriPartsParamExpr):
            new_terms = dict(self.terms)
            for p, c in other.terms.items():
                new_terms[p] = new_terms.get(p, 0.0) + c
            return QuriPartsParamExpr(new_terms, self.const + other.const)
        scalar = self._coerce_constant(other)
        if scalar is not None:
            return QuriPartsParamExpr(dict(self.terms), self.const + scalar)
        return NotImplemented

    def __radd__(self, other: Any) -> "QuriPartsParamExpr":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "QuriPartsParamExpr":
        if isinstance(other, QuriPartsParamExpr):
            new_terms = dict(self.terms)
            for p, c in other.terms.items():
                new_terms[p] = new_terms.get(p, 0.0) - c
            return QuriPartsParamExpr(new_terms, self.const - other.const)
        scalar = self._coerce_constant(other)
        if scalar is not None:
            return QuriPartsParamExpr(dict(self.terms), self.const - scalar)
        return NotImplemented

    def __rsub__(self, other: Any) -> "QuriPartsParamExpr":
        scalar = self._coerce_constant(other)
        if scalar is not None:
            neg = {p: -c for p, c in self.terms.items()}
            return QuriPartsParamExpr(neg, scalar - self.const)
        return NotImplemented

    def __mul__(self, other: Any) -> "QuriPartsParamExpr":
        """Multiply by a constant while preserving affine form."""
        scalar = self._coerce_constant(other)
        if scalar is not None:
            new_terms = {p: c * scalar for p, c in self.terms.items()}
            return QuriPartsParamExpr(new_terms, self.const * scalar)
        if isinstance(other, QuriPartsParamExpr) and self.is_constant():
            return other * self.const
        raise self._non_affine_error("multiplication between parameterized expressions")

    def __rmul__(self, other: Any) -> "QuriPartsParamExpr":
        """Multiply from the right by a constant."""
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "QuriPartsParamExpr":
        """Divide by a constant while preserving affine form."""
        scalar = self._coerce_constant(other)
        if scalar is None:
            raise self._non_affine_error("division by a parameterized expression")
        if scalar == 0.0:
            raise ZeroDivisionError("division by zero")
        new_terms = {p: c / scalar for p, c in self.terms.items()}
        return QuriPartsParamExpr(new_terms, self.const / scalar)

    def __rtruediv__(self, other: Any) -> "QuriPartsParamExpr":
        """Divide a numeric constant by a constant-only expression."""
        scalar = self._coerce_constant(other)
        if scalar is None:
            return NotImplemented
        if not self.is_constant():
            raise self._non_affine_error("division by a parameterized expression")
        if self.const == 0.0:
            raise ZeroDivisionError("division by zero")
        return QuriPartsParamExpr(const=scalar / self.const)

    def __floordiv__(self, other: Any) -> "QuriPartsParamExpr":
        """Floor-divide constant-only expressions."""
        scalar = self._coerce_constant(other)
        if scalar is None:
            raise self._non_affine_error("floor division by a parameterized expression")
        if scalar == 0.0:
            raise ZeroDivisionError("float floor division by zero")
        if not self.is_constant():
            raise self._non_affine_error("floor division of a parameterized expression")
        return QuriPartsParamExpr(const=self.const // scalar)

    def __rfloordiv__(self, other: Any) -> "QuriPartsParamExpr":
        """Floor-divide a numeric constant by a constant-only expression."""
        scalar = self._coerce_constant(other)
        if scalar is None:
            return NotImplemented
        if not self.is_constant():
            raise self._non_affine_error("floor division by a parameterized expression")
        if self.const == 0.0:
            raise ZeroDivisionError("float floor division by zero")
        return QuriPartsParamExpr(const=scalar // self.const)

    def __pow__(self, other: Any) -> "QuriPartsParamExpr":
        """Raise constant-only expressions to a constant power."""
        scalar = self._coerce_constant(other)
        if scalar is None:
            raise self._non_affine_error("power with a parameterized exponent")
        if not self.is_constant():
            raise self._non_affine_error("power of a parameterized expression")
        return QuriPartsParamExpr(const=self.const**scalar)

    def __rpow__(self, other: Any) -> "QuriPartsParamExpr":
        """Raise a numeric constant to a constant-only expression."""
        scalar = self._coerce_constant(other)
        if scalar is None:
            return NotImplemented
        if not self.is_constant():
            raise self._non_affine_error("power with a parameterized exponent")
        return QuriPartsParamExpr(const=scalar**self.const)

    def __neg__(self) -> "QuriPartsParamExpr":
        neg = {p: -c for p, c in self.terms.items()}
        return QuriPartsParamExpr(neg, -self.const)

    def to_angle_dict(self) -> dict[Any, float]:
        """Convert to the ``{Parameter: coeff, CONST: offset}`` dict."""
        from quri_parts.circuit import CONST

        result: dict[Any, float] = dict(self.terms)
        if self.const != 0.0:
            result[CONST] = self.const
        return result

    def __repr__(self) -> str:
        parts = [f"{c}*{p.name}" for p, c in self.terms.items()]
        if self.const != 0.0:
            parts.append(str(self.const))
        return " + ".join(parts) if parts else "0.0"


class QuriPartsGateEmitter:
    """GateEmitter implementation for QURI Parts.

    Emits individual quantum gates to QURI Parts circuits.

    QURI Parts parametric circuits accept angles in dictionary form:
    {parameter: coefficient, CONST: constant_offset}
    """

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

    def create_parameter(self, name: str) -> QuriPartsParamExpr:
        """Create a QURI Parts parameter wrapped in a ``QuriPartsParamExpr``.

        The wrapper supports arithmetic so that ``_evaluate_binop`` can
        compose expressions like ``a + b`` without requiring QURI Parts
        ``Parameter`` objects to support operators directly. Only affine
        symbolic expressions are supported because QURI Parts uses a
        linear parameter mapping model.
        """
        if self._current_circuit is None:
            raise RuntimeError("create_circuit must be called before create_parameter")

        if name not in self._param_map:
            param = self._current_circuit.add_parameter(name)
            self._param_map[name] = param
        return QuriPartsParamExpr({self._param_map[name]: 1.0})

    def _make_angle_dict(self, angle: float | Any) -> dict[Any, float] | float:
        """Convert angle to QURI Parts format.

        If angle is a float, return it directly.
        If angle is a ``QuriPartsParamExpr``, convert to ``{Parameter: coeff}`` dict.
        If angle is a raw ``Parameter``, return ``{parameter: 1.0}``.
        """
        if is_concrete_real_number(angle):
            return float(angle)
        if isinstance(angle, QuriPartsParamExpr):
            return angle.to_angle_dict()
        # Fallback: assume it's a QURI Parts Parameter
        return {angle: 1.0}

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
        if isinstance(angle_dict, float):
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
        if isinstance(angle_dict, float):
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
        if isinstance(angle_dict, float):
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
        if isinstance(angle_dict, float):
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

        CP(ctrl, tgt, θ) =
            RZ(tgt, θ/2)
            CNOT(ctrl, tgt)
            RZ(tgt, -θ/2)
            CNOT(ctrl, tgt)
            RZ(ctrl, θ/2)
        """
        angle_dict = self._make_angle_dict(angle)
        if isinstance(angle_dict, float):
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
        if isinstance(angle_dict, float):
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

        CH(ctrl, tgt) =
            RY(tgt, π/4)
            CNOT(ctrl, tgt)
            RY(tgt, -π/4)
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

        CY(ctrl, tgt) =
            S†(tgt)
            CNOT(ctrl, tgt)
            S(tgt)
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
        if isinstance(angle_dict, float):
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

        CRY(ctrl, tgt, θ) =
            RY(tgt, θ/2)
            CNOT(ctrl, tgt)
            RY(tgt, -θ/2)
            CNOT(ctrl, tgt)
        """
        angle_dict = self._make_angle_dict(angle)
        if isinstance(angle_dict, float):
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

        CRZ(ctrl, tgt, θ) =
            RZ(tgt, θ/2)
            CNOT(ctrl, tgt)
            RZ(tgt, -θ/2)
            CNOT(ctrl, tgt)
        """
        angle_dict = self._make_angle_dict(angle)
        if isinstance(angle_dict, float):
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
    # When emit_measure is a no-op, the sampler returns an all-qubit
    # bitstring indexed by qubit position.  The transpiler pipeline
    # needs a clbit→qubit mapping to decode partial measurements
    # correctly.  Setting this flag causes StandardEmitPass to build
    # that mapping.
    noop_measurement: bool = True

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
