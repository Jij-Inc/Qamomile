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
    PauliEvolutionRealization,
    ReusableCircuit,
    SemanticOpKey,
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


class StandalonePhaseMode(enum.Enum):
    """Enumerate target treatment of an unconditionally applied phase."""

    PRESERVE = "preserve"
    DISCARD = "discard"


class CallPhaseMode(enum.Enum):
    """Enumerate how a target realizes phase in coherently controlled calls."""

    NATIVE_BODY = "native-body"
    EXPLICIT_CORRECTION = "explicit-correction"
    UNSUPPORTED = "unsupported"


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
class GlobalPhaseCapabilities:
    """Declare standalone global-phase handling and accepted scalar syntax.

    Args:
        scalars (ScalarCapabilities): Scalar language accepted for the phase.
        standalone_mode (StandalonePhaseMode): Whether the final artifact
            preserves the phase or deliberately discards it projectively.
    """

    scalars: ScalarCapabilities
    standalone_mode: StandalonePhaseMode


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
        phase_mode (CallPhaseMode): How a reusable body's phase is realized
            after coherent controls are known. Defaults to ``UNSUPPORTED``.
        controlled_phase_scalars (ScalarCapabilities | None): Scalar language
            accepted for an observable controlled-call phase, or ``None``
            when no such phase is supported. Defaults to ``None``.
    """

    supports_power: bool
    supports_inverse: bool
    max_controls: int | None
    supports_nonunitary_body: bool = False
    supports_barrier_body: bool = False
    control_mode: CallControlMode = CallControlMode.WHOLE_CALL
    controlled_gate_kinds: frozenset[GateKind] = frozenset()
    controlled_pauli_time: ScalarCapabilities | None = None
    phase_mode: CallPhaseMode = CallPhaseMode.UNSUPPORTED
    controlled_phase_scalars: ScalarCapabilities | None = None

    def accepts(
        self,
        callee: ReusableCircuit,
        inherited_controls: int = 0,
    ) -> bool:
        """Return whether this declaration accepts a concrete call shape.

        Args:
            callee (ReusableCircuit): Reusable body and requested transforms.
            inherited_controls (int): Controls physically distributed from an
                enclosing call. Defaults to zero.

        Returns:
            bool: Whether power, inverse, and control transforms are accepted.
        """
        if callee.power != 1 and not self.supports_power:
            return False
        if callee.inverse and not self.supports_inverse:
            return False
        effective_controls = inherited_controls + callee.controls
        if effective_controls and self.control_mode is CallControlMode.UNSUPPORTED:
            return False
        return self.max_controls is None or effective_controls <= self.max_controls


@dataclasses.dataclass(frozen=True)
class NativeSemanticOpCapabilities:
    """Declare a target-native realization of an abstract operation.

    Args:
        key (SemanticOpKey): Backend-independent semantic operation key.
        realization (str): Target-owned realization identifier passed to the
            materializer after legalization.
        call_transforms (CallTransformCapabilities): Call shapes supported by
            the native realization.
        operand_widths (tuple[int | None, ...] | None): Required semantic
            operand grouping. ``None`` accepts any grouping; an integer
            requires that exact width and ``None`` inside the tuple accepts
            any positive width at that position. Defaults to ``None``.
        min_qubits (int): Minimum fallback-body width accepted by the native
            operation. Defaults to zero.
        max_qubits (int | None): Maximum fallback-body width, or ``None`` for
            no limit. Defaults to ``None``.
        required_arguments (frozenset[str]): Semantic argument names required
            by this realization. Defaults to none.
        matching_operand_widths (tuple[tuple[int, int], ...]): Pairs of
            operand positions that must have equal widths. Defaults to none.
    """

    key: SemanticOpKey
    realization: str
    call_transforms: CallTransformCapabilities
    operand_widths: tuple[int | None, ...] | None = None
    min_qubits: int = 0
    max_qubits: int | None = None
    required_arguments: frozenset[str] = frozenset()
    matching_operand_widths: tuple[tuple[int, int], ...] = ()

    def accepts(
        self,
        callee: ReusableCircuit,
        inherited_controls: int = 0,
    ) -> bool:
        """Return whether this realization accepts one semantic call shape.

        Args:
            callee (ReusableCircuit): Reusable call retaining source operand
                grouping and deferred transforms.
            inherited_controls (int): Controls physically distributed from an
                enclosing call. Defaults to zero.

        Returns:
            bool: Whether transform, total-width, and operand-shape contracts
            all accept the call.
        """
        if not self.call_transforms.accepts(callee, inherited_controls):
            return False
        if callee.identity is None:
            return False
        if not self.required_arguments <= callee.identity.arguments.names():
            return False
        if callee.body.num_qubits < self.min_qubits:
            return False
        if self.max_qubits is not None and callee.body.num_qubits > self.max_qubits:
            return False
        widths_match = True
        if self.operand_widths is not None:
            if len(callee.operand_widths) != len(self.operand_widths):
                return False
            widths_match = all(
                actual > 0 and (required is None or actual == required)
                for actual, required in zip(
                    callee.operand_widths,
                    self.operand_widths,
                    strict=True,
                )
            )
        if any(
            left < 0
            or right < 0
            or left >= len(callee.operand_widths)
            or right >= len(callee.operand_widths)
            for left, right in self.matching_operand_widths
        ):
            return False
        return widths_match and all(
            callee.operand_widths[left] == callee.operand_widths[right]
            for left, right in self.matching_operand_widths
        )


@dataclasses.dataclass(frozen=True)
class CircuitCapabilities:
    """Declare the complete circuit-IR language accepted by one target.

    Args:
        name (str): Stable target name used in diagnostics.
        primitive_gates (frozenset[GateKind]): Primitive gate kinds accepted
            by the target materializer.
        native_semantic_ops (tuple[NativeSemanticOpCapabilities, ...]): Native
            realizations keyed by open semantic operation identity.
        gate_parameters (ScalarCapabilities): Scalar language accepted by gate
            parameters.
        predicates (ScalarCapabilities): Scalar language accepted by dynamic
            ``if`` and ``while`` predicates.
        pauli_time (ScalarCapabilities): Scalar language accepted by Pauli
            evolution time values.
        global_phase (GlobalPhaseCapabilities | None): Standalone phase mode
            and accepted scalar language, or ``None`` when unsupported.
        generic_calls (CallTransformCapabilities): Reusable-call forms accepted
            after semantic-call legalization.
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
    native_semantic_ops: tuple[NativeSemanticOpCapabilities, ...]
    gate_parameters: ScalarCapabilities
    predicates: ScalarCapabilities
    pauli_time: ScalarCapabilities
    global_phase: GlobalPhaseCapabilities | None
    generic_calls: CallTransformCapabilities
    supports_dynamic_if: bool
    supports_dynamic_while: bool
    supports_reset: bool
    pauli_realizations: frozenset[PauliEvolutionRealization]

    def native_semantic_op(
        self,
        key: SemanticOpKey,
    ) -> NativeSemanticOpCapabilities | None:
        """Return the native declaration for one semantic operation.

        Args:
            key (SemanticOpKey): Semantic operation key to look up.

        Returns:
            NativeSemanticOpCapabilities | None: Matching declaration, or
            ``None`` when the target has no native realization.
        """
        for declaration in self.native_semantic_ops:
            if declaration.key == key:
                return declaration
        return None


@dataclasses.dataclass(frozen=True)
class CompilationPolicy:
    """Select preferred realizations among target-supported alternatives.

    Args:
        prefer_native_semantic_ops (bool): Whether legal target-native
            realizations are preferred over reusable fallback bodies.
            Defaults to ``True``.
        prefer_native_pauli_evolution (bool): Whether native Pauli evolution is
            preferred over a gate gadget. Defaults to ``True``.
    """

    prefer_native_semantic_ops: bool = True
    prefer_native_pauli_evolution: bool = True


DEFAULT_POLICY = CompilationPolicy()
"""Policy used when a backend transpiler does not supply one."""
