"""Circuit-IR capability declarations, legalization decisions, and legality.

Covers the three-layer contract: capabilities declare, legalization decides
(preserve-native vs lower-to-body per intrinsic call), and target
verification proves the result before any materializer runs.
"""

import math

import pytest

from qamomile.circuit.transpiler.circuit_ir import (
    ALL_BINARY_OPERATORS,
    ALL_PRIMITIVE_GATES,
    ALL_UNARY_OPERATORS,
    ARITHMETIC_BINARY_OPERATORS,
    CallableIdentity,
    CallInstruction,
    CallTransformCapabilities,
    CircuitBuilder,
    CircuitCapabilities,
    CircuitIntrinsic,
    CircuitProgram,
    ClassicalBitExpr,
    CompilationPolicy,
    GateInstruction,
    LiteralExpr,
    NativeIntrinsicCapabilities,
    ParameterExpr,
    PauliEvolutionInstruction,
    PauliEvolutionRealization,
    ReusableCircuit,
    ScalarAtom,
    ScalarCapabilities,
    ScalarExpressionForm,
    UnaryOperator,
    legalize_program,
    verify_circuit,
    verify_target_legal,
)
from qamomile.circuit.transpiler.errors import TargetCapabilityError
from qamomile.circuit.transpiler.gate_emitter import GateKind


def _capabilities(**overrides: object) -> CircuitCapabilities:
    """Build a permissive capability declaration with selective overrides.

    Args:
        **overrides (object): Field values replacing permissive defaults.

    Returns:
        CircuitCapabilities: Immutable test target declaration.
    """
    numeric = ScalarCapabilities(
        atoms=frozenset(
            {ScalarAtom.LITERAL, ScalarAtom.PARAMETER, ScalarAtom.LOOP_VARIABLE}
        ),
        unary_operators=frozenset({UnaryOperator.NEG}),
        binary_operators=ARITHMETIC_BINARY_OPERATORS,
        parameter_form=ScalarExpressionForm.ARBITRARY,
    )
    base = dict(
        name="test-target",
        primitive_gates=ALL_PRIMITIVE_GATES,
        native_intrinsics=(),
        gate_parameters=numeric,
        predicates=ScalarCapabilities(
            atoms=frozenset(ScalarAtom),
            unary_operators=ALL_UNARY_OPERATORS,
            binary_operators=ALL_BINARY_OPERATORS,
            parameter_form=ScalarExpressionForm.ARBITRARY,
        ),
        pauli_time=numeric,
        global_phase=numeric,
        generic_calls=CallTransformCapabilities(True, True, None),
        supports_dynamic_if=True,
        supports_dynamic_while=True,
        supports_reset=True,
        pauli_realizations=frozenset(
            {
                PauliEvolutionRealization.NATIVE,
                PauliEvolutionRealization.GADGET,
            }
        ),
    )
    base.update(overrides)
    return CircuitCapabilities(**base)


def _qft_body() -> CircuitProgram:
    """Build a 2-qubit QFT-shaped fallback body.

    Returns:
        CircuitProgram: Immutable two-qubit fallback body.
    """
    body = CircuitBuilder(2, 0, name="qft")
    body.append_gate(GateKind.H, (1,))
    body.append_gate(GateKind.CP, (1, 0), (LiteralExpr(math.pi / 2),))
    body.append_gate(GateKind.H, (0,))
    body.append_gate(GateKind.SWAP, (0, 1))
    return body.freeze()


def _intrinsic_program(controls: int = 0) -> CircuitProgram:
    """Build a program invoking an intrinsic-tagged QFT call.

    Args:
        controls (int): Added call-control count. Defaults to zero.

    Returns:
        CircuitProgram: Intrinsic-tagged caller program.
    """
    callee = ReusableCircuit(
        body=_qft_body(),
        name="qft",
        controls=controls,
        identity=CallableIdentity("qft", CircuitIntrinsic.QFT),
    )
    num_qubits = 2 + controls
    builder = CircuitBuilder(num_qubits, 2)
    builder.append_call(callee, tuple(range(num_qubits)))
    builder.append_measure(0, 0)
    builder.append_measure(1, 1)
    return builder.freeze()


class TestIntrinsicLegalization:
    """Legalization decisions for intrinsic-tagged calls."""

    def test_native_intrinsic_preserved(self):
        """A declared-native intrinsic call survives legalization intact."""
        capabilities = _capabilities(
            native_intrinsics=(
                NativeIntrinsicCapabilities(
                    CircuitIntrinsic.QFT,
                    CallTransformCapabilities(True, True, None),
                ),
            ),
        )
        legalized = legalize_program(
            _intrinsic_program(),
            capabilities,
            CompilationPolicy(),
        )
        calls = [op for op in legalized.operations if isinstance(op, CallInstruction)]
        assert len(calls) == 1
        assert calls[0].callee.identity is not None
        assert calls[0].callee.identity.intrinsic is CircuitIntrinsic.QFT
        verify_circuit(legalized)
        verify_target_legal(legalized, capabilities)

    def test_native_intrinsic_does_not_require_fallback_gate_support(self):
        """A native intrinsic ignores target legality of its fallback body."""
        capabilities = _capabilities(
            primitive_gates=frozenset({GateKind.H}),
            native_intrinsics=(
                NativeIntrinsicCapabilities(
                    CircuitIntrinsic.QFT,
                    CallTransformCapabilities(True, True, None),
                ),
            ),
        )
        legalized = legalize_program(
            _intrinsic_program(),
            capabilities,
            CompilationPolicy(),
        )

        verify_circuit(legalized)
        verify_target_legal(legalized, capabilities)

    def test_unsupported_native_transform_uses_generic_fallback(self):
        """A native intrinsic falls back when its native form rejects controls."""
        capabilities = _capabilities(
            native_intrinsics=(
                NativeIntrinsicCapabilities(
                    CircuitIntrinsic.QFT,
                    CallTransformCapabilities(True, True, 0),
                ),
            ),
        )
        legalized = legalize_program(
            _intrinsic_program(controls=1),
            capabilities,
            CompilationPolicy(),
        )
        [call] = [
            operation
            for operation in legalized.operations
            if isinstance(operation, CallInstruction)
        ]

        assert call.callee.identity is not None
        assert call.callee.identity.intrinsic is None
        verify_target_legal(legalized, capabilities)

    def test_policy_forces_fallback_body(self):
        """prefer_native_intrinsics=False inlines the body on a capable target."""
        capabilities = _capabilities(
            native_intrinsics=(
                NativeIntrinsicCapabilities(
                    CircuitIntrinsic.QFT,
                    CallTransformCapabilities(True, True, None),
                ),
            ),
        )
        legalized = legalize_program(
            _intrinsic_program(),
            capabilities,
            CompilationPolicy(prefer_native_intrinsics=False),
        )
        kinds = [
            op.kind for op in legalized.operations if isinstance(op, GateInstruction)
        ]
        assert kinds == [GateKind.H, GateKind.CP, GateKind.H, GateKind.SWAP]
        assert not any(isinstance(op, CallInstruction) for op in legalized.operations)
        verify_circuit(legalized)
        verify_target_legal(legalized, capabilities)

    def test_missing_native_support_inlines(self):
        """A target without native support receives the inlined fallback."""
        capabilities = _capabilities()
        legalized = legalize_program(
            _intrinsic_program(),
            capabilities,
            CompilationPolicy(),
        )
        assert not any(isinstance(op, CallInstruction) for op in legalized.operations)
        verify_circuit(legalized)
        verify_target_legal(legalized, capabilities)

    def test_transformed_intrinsic_demotes_to_generic_call(self):
        """A controlled intrinsic on a non-native target keeps its body call."""
        capabilities = _capabilities()
        legalized = legalize_program(
            _intrinsic_program(controls=1),
            capabilities,
            CompilationPolicy(),
        )
        calls = [op for op in legalized.operations if isinstance(op, CallInstruction)]
        assert len(calls) == 1
        assert calls[0].callee.controls == 1
        identity = calls[0].callee.identity
        assert identity is not None
        assert identity.symbol == "qft"
        assert identity.intrinsic is None
        verify_circuit(legalized)
        verify_target_legal(legalized, capabilities)

    def test_inline_inside_structured_region(self):
        """A call spliced inside a for-loop body still verifies."""
        callee = ReusableCircuit(
            body=_qft_body(),
            name="qft",
            identity=CallableIdentity("qft", CircuitIntrinsic.QFT),
        )
        builder = CircuitBuilder(2, 0)
        builder.begin_for(range(3))
        builder.append_call(callee, (0, 1))
        builder.end_for()
        program = builder.freeze()
        legalized = legalize_program(
            program,
            _capabilities(),
            CompilationPolicy(),
        )
        verify_circuit(legalized)
        assert not any(isinstance(op, CallInstruction) for op in legalized.operations)


class TestTargetLegalityVerification:
    """Target verification proves programs against declared capabilities."""

    def test_rejects_dynamic_if(self):
        """Measurement-conditioned branching fails on a static target."""
        builder = CircuitBuilder(1, 1)
        builder.append_measure(0, 0)
        context = builder.begin_if(ClassicalBitExpr(0))
        builder.append_gate(GateKind.X, (0,))
        builder.end_if(context)
        program = builder.freeze()
        capabilities = _capabilities(supports_dynamic_if=False)
        with pytest.raises(
            TargetCapabilityError,
            match="measurement-conditioned branching",
        ) as excinfo:
            verify_target_legal(program, capabilities)
        assert excinfo.value.target == "test-target"

    def test_rejects_symbolic_angle_on_concrete_only_target(self):
        """Runtime parameters fail with a bindings-oriented diagnosis."""
        builder = CircuitBuilder(1, 0)
        builder.append_gate(GateKind.RX, (0,), (ParameterExpr("theta"),))
        program = builder.freeze()
        capabilities = _capabilities(
            gate_parameters=ScalarCapabilities(
                atoms=frozenset({ScalarAtom.LITERAL, ScalarAtom.LOOP_VARIABLE}),
                unary_operators=frozenset({UnaryOperator.NEG}),
                binary_operators=ARITHMETIC_BINARY_OPERATORS,
                parameter_form=ScalarExpressionForm.CONCRETE_ONLY,
            ),
        )
        with pytest.raises(
            TargetCapabilityError,
            match="must be supplied through bindings",
        ):
            verify_target_legal(program, capabilities)

    def test_linear_form_accepts_linear_and_rejects_nonlinear(self):
        """Linear targets accept a*theta+b and reject theta*theta."""
        capabilities = _capabilities(
            gate_parameters=ScalarCapabilities(
                atoms=frozenset(
                    {
                        ScalarAtom.LITERAL,
                        ScalarAtom.PARAMETER,
                        ScalarAtom.LOOP_VARIABLE,
                    }
                ),
                unary_operators=frozenset({UnaryOperator.NEG}),
                binary_operators=ARITHMETIC_BINARY_OPERATORS,
                parameter_form=ScalarExpressionForm.LINEAR,
            )
        )

        linear = CircuitBuilder(1, 0)
        linear.append_gate(
            GateKind.RX,
            (0,),
            (ParameterExpr("theta") * 2.0 + 1.0,),
        )
        verify_target_legal(linear.freeze(), capabilities)

        nonlinear = CircuitBuilder(1, 0)
        nonlinear.append_gate(
            GateKind.RX,
            (0,),
            (ParameterExpr("theta") * ParameterExpr("theta"),),
        )
        with pytest.raises(TargetCapabilityError, match="non-linear"):
            verify_target_legal(nonlinear.freeze(), capabilities)

    def test_rejects_reset_on_target_without_reset(self):
        """Reset fails on a target that cannot represent it."""
        builder = CircuitBuilder(1, 0)
        builder.append_reset(0)
        program = builder.freeze()
        with pytest.raises(TargetCapabilityError, match="reset"):
            verify_target_legal(
                program,
                _capabilities(supports_reset=False),
            )

    def test_rejects_measurement_bit_as_quri_gate_angle(self):
        """QURI scalar capabilities reject measurement-derived gate angles."""
        from qamomile.quri_parts.materializer import QuriPartsMaterializer

        builder = CircuitBuilder(1, 1)
        builder.append_measure(0, 0)
        builder.append_gate(GateKind.RX, (0,), (ClassicalBitExpr(0),))
        capabilities = QuriPartsMaterializer().capabilities
        legalized = legalize_program(
            builder.freeze(),
            capabilities,
            CompilationPolicy(),
        )

        with pytest.raises(TargetCapabilityError, match="classical-bit"):
            verify_target_legal(legalized, capabilities)

    def test_rejects_quration_controlled_reusable_call(self):
        """Quration rejects controlled calls before PyQret materialization."""
        from qamomile.quration.materializer import PyQretMaterializer

        capabilities = PyQretMaterializer().capabilities
        legalized = legalize_program(
            _intrinsic_program(controls=1),
            capabilities,
            CompilationPolicy(),
        )

        with pytest.raises(TargetCapabilityError, match="reusable call transforms"):
            verify_target_legal(legalized, capabilities)

    def test_rejects_unsupported_distributed_control_gate(self):
        """QURI rejects controlled body gates absent from its control table."""
        from qamomile.quri_parts.materializer import QuriPartsMaterializer

        body = CircuitBuilder(2, 0)
        body.append_gate(GateKind.CRX, (0, 1), (LiteralExpr(0.5),))
        caller = CircuitBuilder(3, 0)
        caller.append_call(
            ReusableCircuit(body.freeze(), "controlled-crx", controls=1),
            (0, 1, 2),
        )
        capabilities = QuriPartsMaterializer().capabilities
        legalized = legalize_program(
            caller.freeze(),
            capabilities,
            CompilationPolicy(),
        )

        with pytest.raises(TargetCapabilityError, match="onto CRX"):
            verify_target_legal(legalized, capabilities)

    def test_rejects_symbolic_controlled_pauli_time_on_quri(self):
        """QURI controlled Pauli gadgets require a concrete evolution time."""
        from qamomile.quri_parts.materializer import QuriPartsMaterializer

        body = CircuitBuilder(1, 0)
        body.append_pauli_evolution((0,), object(), ParameterExpr("theta"))
        caller = CircuitBuilder(2, 0)
        caller.append_call(
            ReusableCircuit(body.freeze(), "controlled-pauli", controls=1),
            (0, 1),
        )
        capabilities = QuriPartsMaterializer().capabilities
        legalized = legalize_program(
            caller.freeze(),
            capabilities,
            CompilationPolicy(),
        )

        with pytest.raises(TargetCapabilityError, match="supplied through bindings"):
            verify_target_legal(legalized, capabilities)

    def test_rejects_unsupported_program_global_phase(self):
        """A target without global phase support rejects it before materialization."""
        from qamomile.quri_parts.materializer import QuriPartsMaterializer

        builder = CircuitBuilder(1, 0)
        builder.add_global_phase(0.5)
        capabilities = QuriPartsMaterializer().capabilities
        legalized = legalize_program(
            builder.freeze(),
            capabilities,
            CompilationPolicy(),
        )

        with pytest.raises(TargetCapabilityError, match="global phase"):
            verify_target_legal(legalized, capabilities)

    def test_pauli_realization_is_selected_by_policy(self):
        """Legalization fixes native versus gadget Pauli realization."""
        builder = CircuitBuilder(1, 0)
        builder.append_pauli_evolution((0,), object(), 0.5)
        capabilities = _capabilities()

        native = legalize_program(
            builder.freeze(),
            capabilities,
            CompilationPolicy(prefer_native_pauli_evolution=True),
        )
        gadget = legalize_program(
            builder.freeze(),
            capabilities,
            CompilationPolicy(prefer_native_pauli_evolution=False),
        )
        [native_instruction] = native.operations
        [gadget_instruction] = gadget.operations

        assert isinstance(native_instruction, PauliEvolutionInstruction)
        assert isinstance(gadget_instruction, PauliEvolutionInstruction)
        assert native_instruction.realization is PauliEvolutionRealization.NATIVE
        assert gadget_instruction.realization is PauliEvolutionRealization.GADGET
        verify_target_legal(native, capabilities)
        verify_target_legal(gadget, capabilities)

    def test_intrinsic_residue_is_invariant_violation(self):
        """An unlegalized intrinsic call on a non-native target is rejected."""
        with pytest.raises(
            TargetCapabilityError,
            match="legalization invariant",
        ):
            verify_target_legal(_intrinsic_program(), _capabilities())
