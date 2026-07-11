"""Declare circuit-target capabilities and compilation preferences.

Capabilities describe the complete target-legal input language accepted by a
materializer. Policy selects between multiple legal realizations. Neither
object performs rewriting; :mod:`qamomile.circuit.transpiler.circuit_ir.legalize`
uses both to produce a target-legal :class:`CircuitProgram`.
"""

from __future__ import annotations

import dataclasses
import enum

from qamomile.circuit.transpiler.circuit_ir.model import (
    BinaryOperator,
    CircuitIntrinsic,
    PauliEvolutionRealization,
    ReusableCircuit,
    UnaryOperator,
)
from qamomile.circuit.transpiler.gate_emitter import GateKind

ALL_PRIMITIVE_GATES: frozenset[GateKind] = frozenset(GateKind) - {GateKind.MEASURE}
"""Every gate kind valid in ``GateInstruction``.

Measurement has a dedicated :class:`MeasureInstruction`; accepting the legacy
``GateKind.MEASURE`` marker here would let target verification approve an
instruction no circuit materializer implements.
"""

ARITHMETIC_BINARY_OPERATORS: frozenset[BinaryOperator] = frozenset(
    {
        BinaryOperator.ADD,
        BinaryOperator.SUB,
        BinaryOperator.MUL,
        BinaryOperator.DIV,
        BinaryOperator.FLOORDIV,
        BinaryOperator.MOD,
        BinaryOperator.POW,
    }
)
"""Binary operators that produce arithmetic scalar values."""

ALL_BINARY_OPERATORS: frozenset[BinaryOperator] = frozenset(BinaryOperator)
"""Every binary operator in the circuit scalar vocabulary."""

ALL_UNARY_OPERATORS: frozenset[UnaryOperator] = frozenset(UnaryOperator)
"""Every unary operator in the circuit scalar vocabulary."""


class ScalarAtom(enum.Enum):
    """Enumerate leaf values that may occur in a scalar expression."""

    LITERAL = "literal"
    PARAMETER = "parameter"
    CLASSICAL_BIT = "classical-bit"
    LOOP_VARIABLE = "loop-variable"


class ScalarExpressionForm(enum.Enum):
    """Enumerate permitted runtime-parameter expression shapes."""

    CONCRETE_ONLY = "concrete-only"
    LINEAR = "linear"
    ARBITRARY = "arbitrary"


class CallControlMode(enum.Enum):
    """Enumerate how a target realizes controls on reusable calls."""

    UNSUPPORTED = "unsupported"
    WHOLE_CALL = "whole-call"
    DISTRIBUTE = "distribute"


@dataclasses.dataclass(frozen=True)
class ScalarCapabilities:
    """Declare the scalar language accepted in one instruction context.

    Args:
        atoms (frozenset[ScalarAtom]): Leaf value kinds accepted in the
            expression.
        unary_operators (frozenset[UnaryOperator]): Accepted unary operators.
        binary_operators (frozenset[BinaryOperator]): Accepted binary
            operators.
        parameter_form (ScalarExpressionForm): Maximum algebraic form for
            runtime parameters.
    """

    atoms: frozenset[ScalarAtom]
    unary_operators: frozenset[UnaryOperator]
    binary_operators: frozenset[BinaryOperator]
    parameter_form: ScalarExpressionForm


@dataclasses.dataclass(frozen=True)
class CallTransformCapabilities:
    """Declare reusable-call forms accepted by a target realization.

    Args:
        supports_power (bool): Whether powers other than one are accepted.
        supports_inverse (bool): Whether inverse calls are accepted.
        max_controls (int | None): Maximum added controls. ``None`` means no
            declared limit.
        supports_nonunitary_body (bool): Whether a reusable body may contain
            measurement, reset, or dynamic control flow.
        supports_barrier_body (bool): Whether barriers may remain inside a
            reusable body.
        control_mode (CallControlMode): How added controls are realized.
        controlled_gate_kinds (frozenset[GateKind]): Body gate kinds accepted
            when controls are distributed into the body.
        controlled_pauli_time (ScalarCapabilities | None): Pauli-time scalar
            language accepted under distributed controls, or ``None`` when
            controlled Pauli evolution is unsupported.
    """

    supports_power: bool
    supports_inverse: bool
    max_controls: int | None
    supports_nonunitary_body: bool = False
    supports_barrier_body: bool = False
    control_mode: CallControlMode = CallControlMode.WHOLE_CALL
    controlled_gate_kinds: frozenset[GateKind] = frozenset()
    controlled_pauli_time: ScalarCapabilities | None = None

    def accepts(self, callee: ReusableCircuit) -> bool:
        """Return whether this declaration accepts a concrete call shape.

        Args:
            callee (ReusableCircuit): Reusable body and requested transforms.

        Returns:
            bool: Whether power, inverse, and control transforms are accepted.
        """
        if callee.power != 1 and not self.supports_power:
            return False
        if callee.inverse and not self.supports_inverse:
            return False
        if callee.controls and self.control_mode is CallControlMode.UNSUPPORTED:
            return False
        return self.max_controls is None or callee.controls <= self.max_controls


@dataclasses.dataclass(frozen=True)
class NativeIntrinsicCapabilities:
    """Declare a native intrinsic realization and its accepted transforms.

    Args:
        intrinsic (CircuitIntrinsic): Intrinsic realized natively.
        call_transforms (CallTransformCapabilities): Call shapes supported by
            the native realization.
    """

    intrinsic: CircuitIntrinsic
    call_transforms: CallTransformCapabilities


@dataclasses.dataclass(frozen=True)
class CircuitCapabilities:
    """Declare the complete circuit-IR language accepted by one target.

    Args:
        name (str): Stable target name used in diagnostics.
        primitive_gates (frozenset[GateKind]): Primitive gate kinds accepted
            by the target materializer.
        native_intrinsics (tuple[NativeIntrinsicCapabilities, ...]): Native
            intrinsic realizations and their supported call shapes.
        gate_parameters (ScalarCapabilities): Scalar language accepted by gate
            parameters.
        predicates (ScalarCapabilities): Scalar language accepted by dynamic
            ``if`` and ``while`` predicates.
        pauli_time (ScalarCapabilities): Scalar language accepted by Pauli
            evolution time values.
        global_phase (ScalarCapabilities | None): Scalar language accepted for
            nonzero program-level global phase, or ``None`` when unsupported.
        generic_calls (CallTransformCapabilities): Reusable-call forms accepted
            after intrinsic legalization.
        supports_dynamic_if (bool): Whether runtime ``if`` regions are
            accepted.
        supports_dynamic_while (bool): Whether runtime ``while`` regions are
            accepted.
        supports_reset (bool): Whether reset instructions are accepted.
        pauli_realizations (frozenset[PauliEvolutionRealization]): Concrete
            Pauli-evolution realizations accepted by the materializer.
    """

    name: str
    primitive_gates: frozenset[GateKind]
    native_intrinsics: tuple[NativeIntrinsicCapabilities, ...]
    gate_parameters: ScalarCapabilities
    predicates: ScalarCapabilities
    pauli_time: ScalarCapabilities
    global_phase: ScalarCapabilities | None
    generic_calls: CallTransformCapabilities
    supports_dynamic_if: bool
    supports_dynamic_while: bool
    supports_reset: bool
    pauli_realizations: frozenset[PauliEvolutionRealization]

    def native_intrinsic(
        self,
        intrinsic: CircuitIntrinsic,
    ) -> NativeIntrinsicCapabilities | None:
        """Return the native declaration for one intrinsic.

        Args:
            intrinsic (CircuitIntrinsic): Intrinsic kind to look up.

        Returns:
            NativeIntrinsicCapabilities | None: Matching declaration, or
            ``None`` when the target has no native realization.
        """
        for declaration in self.native_intrinsics:
            if declaration.intrinsic is intrinsic:
                return declaration
        return None


@dataclasses.dataclass(frozen=True)
class CompilationPolicy:
    """Select preferred realizations among target-supported alternatives.

    Args:
        prefer_native_intrinsics (bool): Whether legal native intrinsic
            realizations are preferred over fallback bodies. Defaults to
            ``True``.
        prefer_native_pauli_evolution (bool): Whether native Pauli evolution is
            preferred over a gate gadget. Defaults to ``True``.
    """

    prefer_native_intrinsics: bool = True
    prefer_native_pauli_evolution: bool = True


DEFAULT_POLICY = CompilationPolicy()
"""Policy used when a backend transpiler does not supply one."""
