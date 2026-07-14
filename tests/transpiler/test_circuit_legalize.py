"""Circuit-IR capability declarations, legalization decisions, and legality.

Covers the three-layer contract: capabilities declare, legalization decides
(preserve-native vs lower-to-body per semantic call), and target
verification proves the result before any materializer runs.
"""

import dataclasses
import math

import pytest

from qamomile.circuit.transpiler.circuit_ir import (
    ALL_BINARY_OPERATORS,
    ALL_PRIMITIVE_GATES,
    ALL_UNARY_OPERATORS,
    ARITHMETIC_BINARY_OPERATORS,
    QFT_SEMANTIC_KEY,
    CallableIdentity,
    CallControlMode,
    CallInstruction,
    CallPhaseMode,
    CallTransformCapabilities,
    CircuitBuilder,
    CircuitCapabilities,
    CircuitProgram,
    ClassicalBitExpr,
    CompilationPolicy,
    GlobalPhaseCapabilities,
    LiteralExpr,
    NativeSemanticOpCapabilities,
    ParameterExpr,
    PauliEvolutionInstruction,
    PauliEvolutionRealization,
    ReusableCircuit,
    ScalarAtom,
    ScalarCapabilities,
    ScalarExpressionForm,
    SemanticArguments,
    StandalonePhaseMode,
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
        native_semantic_ops=(),
        gate_parameters=numeric,
        predicates=ScalarCapabilities(
            atoms=frozenset(ScalarAtom),
            unary_operators=ALL_UNARY_OPERATORS,
            binary_operators=ALL_BINARY_OPERATORS,
            parameter_form=ScalarExpressionForm.ARBITRARY,
        ),
        pauli_time=numeric,
        global_phase=GlobalPhaseCapabilities(
            numeric,
            StandalonePhaseMode.PRESERVE,
        ),
        generic_calls=CallTransformCapabilities(
            True,
            True,
            None,
            phase_mode=CallPhaseMode.NATIVE_BODY,
            controlled_phase_scalars=numeric,
        ),
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


def _semantic_program(controls: int = 0) -> CircuitProgram:
    """Build a program invoking an semantic-tagged QFT call.

    Args:
        controls (int): Added call-control count. Defaults to zero.

    Returns:
        CircuitProgram: Semantic-tagged caller program.
    """
    callee = ReusableCircuit(
        body=_qft_body(),
        name="qft",
        controls=controls,
        identity=CallableIdentity(key=QFT_SEMANTIC_KEY, symbol="qft"),
        operand_widths=(2,),
    )
    num_qubits = 2 + controls
    builder = CircuitBuilder(num_qubits, 2)
    builder.append_call(callee, tuple(range(num_qubits)))
    builder.append_measure(0, 0)
    builder.append_measure(1, 1)
    return builder.freeze()


def _nested_call_program(
    *,
    outer_controls: int,
    inner_controls: int,
    semantic_inner: bool = False,
) -> CircuitProgram:
    """Build two nested reusable calls with independently added controls."""
    inner_body = CircuitBuilder(1, 0, name="inner-body")
    inner_body.append_gate(GateKind.H, (0,))
    inner = ReusableCircuit(
        inner_body.freeze(),
        "inner",
        controls=inner_controls,
        identity=(
            CallableIdentity(key=QFT_SEMANTIC_KEY, symbol="inner")
            if semantic_inner
            else None
        ),
        operand_widths=(1,),
    )
    outer_body = CircuitBuilder(1 + inner_controls, 0, name="outer-body")
    outer_body.append_call(inner, tuple(range(1 + inner_controls)))
    outer = ReusableCircuit(
        outer_body.freeze(),
        "outer",
        controls=outer_controls,
    )
    caller = CircuitBuilder(1 + inner_controls + outer_controls, 0)
    caller.append_call(
        outer,
        tuple(range(1 + inner_controls + outer_controls)),
    )
    return caller.freeze()


class TestSemanticLegalization:
    """Legalization decisions for semantic-tagged calls."""

    def test_native_semantic_op_preserved(self):
        """A declared-native semantic operation call survives legalization intact."""
        capabilities = _capabilities(
            native_semantic_ops=(
                NativeSemanticOpCapabilities(
                    QFT_SEMANTIC_KEY,
                    "test.qft",
                    CallTransformCapabilities(True, True, None),
                    operand_widths=(None,),
                    min_qubits=1,
                ),
            ),
        )
        legalized = legalize_program(
            _semantic_program(),
            capabilities,
            CompilationPolicy(),
        )
        calls = [op for op in legalized.operations if isinstance(op, CallInstruction)]
        assert len(calls) == 1
        assert calls[0].callee.identity is not None
        assert calls[0].callee.identity.key == QFT_SEMANTIC_KEY
        assert calls[0].callee.native_realization == "test.qft"
        verify_circuit(legalized)
        verify_target_legal(legalized, capabilities)

    def test_native_semantic_op_does_not_require_fallback_gate_support(self):
        """A native semantic operation ignores target legality of its fallback body."""
        capabilities = _capabilities(
            primitive_gates=frozenset({GateKind.H}),
            native_semantic_ops=(
                NativeSemanticOpCapabilities(
                    QFT_SEMANTIC_KEY,
                    "test.qft",
                    CallTransformCapabilities(True, True, None),
                ),
            ),
        )
        legalized = legalize_program(
            _semantic_program(),
            capabilities,
            CompilationPolicy(),
        )

        verify_circuit(legalized)
        verify_target_legal(legalized, capabilities)

    def test_native_semantic_op_with_fallback_phase_is_demoted(self):
        """A native form without phase semantics retains the phased fallback."""
        program = _semantic_program(controls=1)
        call = program.operations[0]
        assert isinstance(call, CallInstruction)
        phased_call = dataclasses.replace(
            call,
            callee=dataclasses.replace(
                call.callee,
                body=dataclasses.replace(
                    call.callee.body,
                    global_phase=LiteralExpr(1e-16),
                ),
            ),
        )
        phased_program = dataclasses.replace(
            program,
            operations=(phased_call, *program.operations[1:]),
        )
        capabilities = _capabilities(
            native_semantic_ops=(
                NativeSemanticOpCapabilities(
                    QFT_SEMANTIC_KEY,
                    "test.qft",
                    CallTransformCapabilities(True, True, None),
                ),
            ),
        )

        legalized = legalize_program(
            phased_program,
            capabilities,
            CompilationPolicy(),
        )
        legalized_call = legalized.operations[0]
        assert isinstance(legalized_call, CallInstruction)
        assert legalized_call.callee.native_realization is None
        assert legalized_call.callee.body.global_phase == LiteralExpr(1e-16)
        verify_circuit(legalized)
        verify_target_legal(legalized, capabilities)

    @pytest.mark.parametrize(
        ("control_mode", "expected_native"),
        [
            (CallControlMode.WHOLE_CALL, "test.inner"),
            (CallControlMode.DISTRIBUTE, None),
        ],
    )
    def test_native_selection_uses_only_physically_distributed_controls(
        self,
        control_mode: CallControlMode,
        expected_native: str | None,
    ) -> None:
        """Whole-call controls do not leak into nested native call shapes."""
        generic = dataclasses.replace(
            _capabilities().generic_calls,
            max_controls=1,
            control_mode=control_mode,
            controlled_gate_kinds=ALL_PRIMITIVE_GATES,
        )
        capabilities = _capabilities(
            generic_calls=generic,
            native_semantic_ops=(
                NativeSemanticOpCapabilities(
                    QFT_SEMANTIC_KEY,
                    "test.inner",
                    CallTransformCapabilities(
                        supports_power=True,
                        supports_inverse=True,
                        max_controls=0,
                    ),
                    operand_widths=(1,),
                ),
            ),
        )

        legalized = legalize_program(
            _nested_call_program(
                outer_controls=1,
                inner_controls=0,
                semantic_inner=True,
            ),
            capabilities,
            CompilationPolicy(),
        )
        [outer] = legalized.operations
        assert isinstance(outer, CallInstruction)
        [inner] = outer.callee.body.operations
        assert isinstance(inner, CallInstruction)
        assert inner.callee.native_realization == expected_native
        verify_circuit(legalized)
        verify_target_legal(legalized, capabilities)

    def test_distributed_nested_controls_respect_effective_limit(self) -> None:
        """Nested distributed calls cannot exceed the cumulative control cap."""
        generic = dataclasses.replace(
            _capabilities().generic_calls,
            max_controls=1,
            control_mode=CallControlMode.DISTRIBUTE,
            controlled_gate_kinds=ALL_PRIMITIVE_GATES,
        )
        capabilities = _capabilities(generic_calls=generic)
        legalized = legalize_program(
            _nested_call_program(outer_controls=1, inner_controls=1),
            capabilities,
            CompilationPolicy(),
        )

        with pytest.raises(TargetCapabilityError, match="reusable call transforms"):
            verify_target_legal(legalized, capabilities)

    def test_native_semantic_op_rejects_incompatible_operand_grouping(self):
        """A backend vector API is not selected for two scalar operands."""
        capabilities = _capabilities(
            native_semantic_ops=(
                NativeSemanticOpCapabilities(
                    QFT_SEMANTIC_KEY,
                    "test.qft",
                    CallTransformCapabilities(True, True, None),
                    operand_widths=(None,),
                    min_qubits=1,
                ),
            ),
        )
        program = _semantic_program()
        [call, *rest] = program.operations
        assert isinstance(call, CallInstruction)
        incompatible = dataclasses.replace(
            call,
            callee=dataclasses.replace(call.callee, operand_widths=(1, 1)),
        )
        legalized = legalize_program(
            dataclasses.replace(program, operations=(incompatible, *rest)),
            capabilities,
            CompilationPolicy(),
        )
        [legalized_call] = [
            operation
            for operation in legalized.operations
            if isinstance(operation, CallInstruction)
        ]

        assert legalized_call.callee.native_realization is None
        verify_target_legal(legalized, capabilities)

    def test_native_declaration_matches_equal_operand_groups_without_shape(self):
        """Equality constraints apply even without an exact operand shape."""
        declaration = NativeSemanticOpCapabilities(
            QFT_SEMANTIC_KEY,
            "test.equal-widths",
            CallTransformCapabilities(True, True, None),
            matching_operand_widths=((0, 1),),
        )
        callee = ReusableCircuit(
            body=_qft_body(),
            name="equal-widths",
            identity=CallableIdentity(
                key=QFT_SEMANTIC_KEY,
                symbol="equal-widths",
                arguments=SemanticArguments.from_mapping({"mode": "exact"}),
            ),
            operand_widths=(1, 1),
        )

        assert declaration.accepts(callee)

    def test_native_declaration_rejects_negative_operand_group_index(self):
        """Negative equality indices cannot accidentally address from the end."""
        declaration = NativeSemanticOpCapabilities(
            QFT_SEMANTIC_KEY,
            "test.invalid-width-index",
            CallTransformCapabilities(True, True, None),
            matching_operand_widths=((-1, 0),),
        )
        callee = ReusableCircuit(
            body=_qft_body(),
            name="invalid-width-index",
            identity=CallableIdentity(key=QFT_SEMANTIC_KEY, symbol="invalid"),
            operand_widths=(1, 1),
        )

        assert not declaration.accepts(callee)

    def test_unsupported_native_transform_uses_generic_fallback(self):
        """A native semantic operation falls back when its native form rejects controls."""
        capabilities = _capabilities(
            native_semantic_ops=(
                NativeSemanticOpCapabilities(
                    QFT_SEMANTIC_KEY,
                    "test.qft",
                    CallTransformCapabilities(True, True, 0),
                ),
            ),
        )
        legalized = legalize_program(
            _semantic_program(controls=1),
            capabilities,
            CompilationPolicy(),
        )
        [call] = [
            operation
            for operation in legalized.operations
            if isinstance(operation, CallInstruction)
        ]

        assert call.callee.identity is not None
        assert call.callee.identity.key == QFT_SEMANTIC_KEY
        assert call.callee.native_realization is None
        verify_target_legal(legalized, capabilities)

    def test_policy_forces_fallback_body(self):
        """prefer_native_semantic_ops=False inlines the body on a capable target."""
        capabilities = _capabilities(
            native_semantic_ops=(
                NativeSemanticOpCapabilities(
                    QFT_SEMANTIC_KEY,
                    "test.qft",
                    CallTransformCapabilities(True, True, None),
                ),
            ),
        )
        legalized = legalize_program(
            _semantic_program(),
            capabilities,
            CompilationPolicy(prefer_native_semantic_ops=False),
        )
        calls = [op for op in legalized.operations if isinstance(op, CallInstruction)]
        assert len(calls) == 1
        assert calls[0].callee.native_realization is None
        verify_circuit(legalized)
        verify_target_legal(legalized, capabilities)

    def test_missing_native_support_inlines(self):
        """A target without native support receives the inlined fallback."""
        capabilities = _capabilities()
        legalized = legalize_program(
            _semantic_program(),
            capabilities,
            CompilationPolicy(),
        )
        calls = [op for op in legalized.operations if isinstance(op, CallInstruction)]
        assert len(calls) == 1
        assert calls[0].callee.native_realization is None
        verify_circuit(legalized)
        verify_target_legal(legalized, capabilities)

    def test_transformed_semantic_op_uses_generic_call(self):
        """A controlled semantic op on a non-native target keeps its body call."""
        capabilities = _capabilities()
        legalized = legalize_program(
            _semantic_program(controls=1),
            capabilities,
            CompilationPolicy(),
        )
        calls = [op for op in legalized.operations if isinstance(op, CallInstruction)]
        assert len(calls) == 1
        assert calls[0].callee.controls == 1
        identity = calls[0].callee.identity
        assert identity is not None
        assert identity.symbol == "qft"
        assert identity.key == QFT_SEMANTIC_KEY
        assert calls[0].callee.native_realization is None
        verify_circuit(legalized)
        verify_target_legal(legalized, capabilities)

    def test_inline_inside_structured_region(self):
        """A call spliced inside a for-loop body still verifies."""
        callee = ReusableCircuit(
            body=_qft_body(),
            name="qft",
            identity=CallableIdentity(key=QFT_SEMANTIC_KEY, symbol="qft"),
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
        assert any(
            isinstance(op, CallInstruction) for op in legalized.operations[0].body
        )


class TestTargetLegalityVerification:
    """Target verification proves programs against declared capabilities."""

    def test_outer_control_preserves_phase_in_nested_reusable_body(self):
        """Inherited controls make a deeply nested reusable phase observable."""
        phase_body = CircuitBuilder(1, 0, name="phase-body")
        phase_body.add_global_phase(ParameterExpr("theta"))
        middle_body = CircuitBuilder(1, 0, name="middle-body")
        middle_body.append_call(
            ReusableCircuit(phase_body.freeze(), "phase-body"),
            (0,),
        )
        caller = CircuitBuilder(2, 0)
        caller.append_call(
            ReusableCircuit(middle_body.freeze(), "outer", controls=1),
            (0, 1),
        )
        program = caller.freeze()
        permissive = _capabilities()
        explicit_phase = dataclasses.replace(
            permissive.generic_calls,
            phase_mode=CallPhaseMode.EXPLICIT_CORRECTION,
        )
        capabilities = _capabilities(generic_calls=explicit_phase)
        legalized = legalize_program(
            program,
            capabilities,
            CompilationPolicy(),
        )

        verify_circuit(legalized)
        verify_target_legal(legalized, capabilities)

        unsupported = _capabilities(
            generic_calls=dataclasses.replace(
                explicit_phase,
                phase_mode=CallPhaseMode.UNSUPPORTED,
                controlled_phase_scalars=None,
            )
        )
        with pytest.raises(
            TargetCapabilityError,
            match="global phase under coherent controls",
        ):
            verify_target_legal(legalized, unsupported)

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
            _semantic_program(controls=1),
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

    def test_quri_preserves_standalone_phase_and_checks_scalar_form(self):
        """A preserved QURI phase obeys its linear-expression capability."""
        from qamomile.quri_parts.materializer import QuriPartsMaterializer

        builder = CircuitBuilder(1, 0)
        theta = ParameterExpr("theta")
        builder.add_global_phase(theta * theta)
        capabilities = QuriPartsMaterializer().capabilities
        legalized = legalize_program(
            builder.freeze(),
            capabilities,
            CompilationPolicy(),
        )

        assert capabilities.global_phase is not None
        assert capabilities.global_phase.standalone_mode is StandalonePhaseMode.PRESERVE
        with pytest.raises(TargetCapabilityError, match="non-linear in runtime"):
            verify_target_legal(legalized, capabilities)

    def test_preserved_phase_can_require_a_physical_carrier_qubit(self):
        """A target rejects a phase when its exact synthesis lacks a carrier."""
        builder = CircuitBuilder(0, 0)
        builder.add_global_phase(0.25)
        base = _capabilities()
        assert base.global_phase is not None
        capabilities = _capabilities(
            global_phase=dataclasses.replace(base.global_phase, min_qubits=1)
        )

        with pytest.raises(TargetCapabilityError, match="at least 1 qubit"):
            verify_target_legal(builder.freeze(), capabilities)

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

    def test_native_residue_is_invariant_violation(self):
        """An unverified native realization on a target is rejected."""
        program = _semantic_program()
        call = program.operations[0]
        assert isinstance(call, CallInstruction)
        malformed = dataclasses.replace(
            program,
            operations=(
                dataclasses.replace(
                    call,
                    callee=dataclasses.replace(
                        call.callee,
                        native_realization="missing.native",
                    ),
                ),
                *program.operations[1:],
            ),
        )
        with pytest.raises(
            TargetCapabilityError,
            match="legalization invariant",
        ):
            verify_target_legal(malformed, _capabilities())
