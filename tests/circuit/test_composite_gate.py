"""Tests for the CompositeGate API."""

from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.composite_gate import CompositeGate
from qamomile.circuit.frontend.constructors import qubit_array
from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.frontend.qkernel import qkernel
from qamomile.circuit.frontend.tracer import Tracer
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import InvokeOperation
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableImplementation,
    CallableRef,
    CallPolicy,
    CallTransform,
    CompositeGateType,
)
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.types import FloatType, QubitType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.passes.emit_support.composite_gate_emission import (
    emit_composite_fallback,
    emit_invoke_operation,
    emit_iqft_with_strategy,
    emit_qft_with_strategy,
    emit_qpe_manual,
    extract_phase_from_params,
)
from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
    ValueResolver,
)
from tests.circuit.conftest import run_statevector

# =============================================================================
# Reusable CompositeGate definitions
# =============================================================================


class HadamardAll(CompositeGate):
    """Apply H to all n qubits.

    Args:
        n: Number of target qubits.
    """

    custom_name = "hadamard_all"

    def __init__(self, n: int):
        self._n = n

    @property
    def num_target_qubits(self) -> int:
        return self._n

    def _decompose(
        self, qubits: Vector[Qubit] | tuple[Qubit, ...]
    ) -> tuple[Qubit, ...]:
        return tuple(qmc.h(q) for q in qubits)


class BellPair(CompositeGate):
    """Apply H + CX on 2 qubits to create a Bell state."""

    custom_name = "bell_pair"

    @property
    def num_target_qubits(self) -> int:
        return 2

    def _decompose(
        self, qubits: Vector[Qubit] | tuple[Qubit, ...]
    ) -> tuple[Qubit, ...]:
        q0, q1 = qubits
        q0 = qmc.h(q0)
        q0, q1 = qmc.cx(q0, q1)
        return (q0, q1)


class DoubleH(CompositeGate):
    """Apply H twice (identity operation) on a single qubit."""

    custom_name = "double_h"

    @property
    def num_target_qubits(self) -> int:
        return 1

    def _decompose(
        self, qubits: Vector[Qubit] | tuple[Qubit, ...]
    ) -> tuple[Qubit, ...]:
        q = qubits[0]
        q = qmc.h(q)
        q = qmc.h(q)
        return (q,)


def apply_gate_to_array(qs: Vector[Qubit], gate: CompositeGate) -> Vector[Qubit]:
    """Apply a CompositeGate to all qubits in an array.

    This is a regular function (not a qkernel) so range() produces Python ints,
    allowing tuple indexing on the gate's return value.
    """
    n = gate.num_target_qubits
    qubit_list = [qs[i] for i in range(n)]
    result = gate(*qubit_list)
    for i in range(n):
        qs[i] = result[i]
    return qs


def _invoke_ops(operations: list[object]) -> list[InvokeOperation]:
    """Return callable invocation operations from an operation list."""
    return [op for op in operations if isinstance(op, InvokeOperation)]


def _assert_composite_invoke(
    op: object,
    *,
    name: str,
    num_targets: int,
    num_controls: int = 0,
    gate_type: CompositeGateType = CompositeGateType.CUSTOM,
) -> InvokeOperation:
    """Assert that an operation is a composite invocation."""
    assert isinstance(op, InvokeOperation)
    assert op.attrs["custom_name"] == name
    assert op.attrs["gate_type"] == gate_type.name
    assert op.target.name == name
    assert op.num_target_qubits == num_targets
    assert op.num_control_qubits == num_controls
    return op


class TestCallableDefinitionSignatures:
    """Verify public callable APIs preserve compiler-facing signatures."""

    def test_qkernel_call_records_callable_signature(self):
        """qkernel calls attach a CallableDef signature derived from the body."""

        @qmc.qkernel
        def helper(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.h(q)

        @qmc.qkernel
        def circuit() -> qmc.Qubit:
            q = qmc.qubit("q")
            q = helper(q)
            return q

        op = _invoke_ops(circuit.build().operations)[0]

        assert op.definition is not None
        assert op.definition.signature is not None
        assert op.signature is op.definition.signature
        assert op.definition.signature.operands[0].name == "q"
        assert isinstance(op.definition.signature.operands[0].type, QubitType)
        assert isinstance(op.definition.signature.results[0].type, QubitType)

    def test_composite_call_records_callable_signature(self):
        """composite calls attach a CallableDef signature derived from the body."""

        @qmc.composite_gate(name="encode")
        @qmc.qkernel
        def encode(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            q[0] = qmc.h(q[0])
            return q

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(2, "q")
            q = encode(q)
            return q

        op = _invoke_ops(circuit.build().operations)[0]

        assert op.definition is not None
        assert op.definition.signature is not None
        assert op.signature is op.definition.signature
        assert op.definition.signature.operands[0].name == "q"
        assert len(op.definition.signature.results) == 1

    def test_composite_decorator_records_implementation_candidates(self):
        """composite decorators attach user-supplied implementation candidates."""
        implementation = CallableImplementation(
            backend="qiskit",
            strategy="native",
        )

        @qmc.composite_gate(name="encode", implementations=[implementation])
        @qmc.qkernel
        def encode(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            return q

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(2, "q")
            q = encode(q)
            return q

        op = _invoke_ops(circuit.build().operations)[0]

        assert op.definition is not None
        assert op.definition.implementations[0] is implementation

    def test_composite_decorator_exposes_qkernel_like_surface(self):
        """composite decorators keep the wrapped qkernel inspection surface."""

        @qmc.composite_gate(name="encode")
        @qmc.qkernel
        def encode(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        assert encode.name == "encode"
        assert encode.qkernel.name == "encode"
        assert "q" in encode.signature.parameters
        assert encode.input_types == encode.qkernel.input_types
        assert encode.output_types == encode.qkernel.output_types
        assert encode.block is encode.qkernel.block
        assert list(encode.build(parameters=[]).label_args) == ["q"]

        @qmc.qkernel
        def circuit() -> qmc.Qubit:
            q = qmc.qubit("q")
            q = encode(q)
            return q

        op = _invoke_ops(circuit.build().operations)[0]
        assert op.default_policy is CallPolicy.PRESERVE_BOX

    def test_composite_decorator_rejects_missing_declared_controls(self):
        """A fixed-control decorator call must not emit inconsistent IR."""

        @qmc.composite_gate(name="controlled_encode", num_controls=1)
        @qmc.qkernel
        def controlled_encode(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.h(q)

        @qmc.qkernel
        def circuit() -> qmc.Qubit:
            q = qmc.qubit("q")
            q = controlled_encode(q)
            return q

        with pytest.raises(ValueError, match="declares 1 control qubits"):
            circuit.build()

    def test_opaque_call_records_callable_signature_without_user_signature(self):
        """opaque calls synthesize a CallableDef signature from call values."""
        black_box = qmc.opaque("black_box", 1)

        @qmc.qkernel
        def circuit(q: qmc.Qubit) -> qmc.Qubit:
            (q,) = black_box(q)
            return q

        op = _invoke_ops(circuit.build().operations)[0]

        assert op.definition is not None
        assert op.definition.signature is not None
        assert op.signature is op.definition.signature
        assert op.definition.signature.operands[0].name == "target_0"
        assert isinstance(op.definition.signature.operands[0].type, QubitType)
        assert isinstance(op.definition.signature.results[0].type, QubitType)

    def test_manual_invoke_auto_definition_records_signature(self):
        """InvokeOperation auto-definitions preserve operand/result types."""
        operand = Value(type=QubitType(), name="q")
        result = operand.next_version()

        op = InvokeOperation(
            operands=[operand],
            results=[result],
            target=CallableRef(namespace="user", name="manual"),
        )

        assert op.definition is not None
        assert op.definition.signature is not None
        assert op.signature is op.definition.signature
        assert op.definition.signature.operands[0].name == "arg_0"
        assert isinstance(op.definition.signature.operands[0].type, QubitType)
        assert isinstance(op.definition.signature.results[0].type, QubitType)


class TestCompositeGate:
    """Test the CompositeGate API: definition, application, and error handling."""

    def test_invoke_stores_body_only_on_definition(self):
        """InvokeOperation exposes callable body without duplicating fields."""
        body = Block()
        ref = CallableRef(namespace="user", name="callable")
        op = InvokeOperation(
            target=ref,
            definition=CallableDef(ref=ref, body=body),
        )

        field_names = set(InvokeOperation.__dataclass_fields__)

        assert "body" not in field_names
        assert "resource" not in field_names
        assert op.definition is not None
        assert op.definition.body is body
        assert op.body is body

    def test_invoke_effective_body_uses_selected_strategy(self):
        """InvokeOperation selects a strategy-specific implementation body."""
        ref = CallableRef(namespace="user", name="selectable")
        attrs = {
            "kind": "composite",
            "gate_type": CompositeGateType.CUSTOM.name,
            "custom_name": "selectable",
            "strategy_name": "fast",
            "num_control_qubits": 0,
            "num_target_qubits": 0,
            "default_policy": CallPolicy.PRESERVE_BOX.name,
        }
        default_body = Block(name="default")
        fast_body = Block(name="fast")
        op = InvokeOperation(
            operands=[],
            results=[],
            target=ref,
            attrs=attrs,
            definition=CallableDef(
                ref=ref,
                body=default_body,
                implementations=[
                    CallableImplementation(strategy="fast", body=fast_body),
                ],
                default_policy=CallPolicy.PRESERVE_BOX,
                attrs=attrs,
            ),
        )

        assert op.effective_body() is fast_body

    def test_invoke_effective_body_does_not_guess_strategy(self):
        """InvokeOperation does not pick a strategy body without strategy_name."""
        ref = CallableRef(namespace="user", name="selectable")
        attrs = {
            "kind": "composite",
            "gate_type": CompositeGateType.CUSTOM.name,
            "custom_name": "selectable",
            "strategy_name": None,
            "num_control_qubits": 0,
            "num_target_qubits": 0,
            "default_policy": CallPolicy.PRESERVE_BOX.name,
        }
        default_body = Block(name="default")
        fast_body = Block(name="fast")
        op = InvokeOperation(
            operands=[],
            results=[],
            target=ref,
            attrs=attrs,
            definition=CallableDef(
                ref=ref,
                body=default_body,
                implementations=[
                    CallableImplementation(strategy="fast", body=fast_body),
                ],
                default_policy=CallPolicy.PRESERVE_BOX,
                attrs=attrs,
            ),
        )

        assert op.effective_body() is default_body

    @pytest.mark.parametrize(
        "transform",
        [CallTransform.INVERSE, CallTransform.CONTROLLED],
    )
    def test_transformed_invoke_does_not_fallback_to_direct_body(
        self,
        transform: CallTransform,
    ):
        """Transformed InvokeOperation requires a matching implementation body."""
        ref = CallableRef(namespace="user", name="selectable")
        attrs = {
            "kind": "composite",
            "gate_type": CompositeGateType.CUSTOM.name,
            "custom_name": "selectable",
            "strategy_name": None,
            "num_control_qubits": 0,
            "num_target_qubits": 0,
            "default_policy": CallPolicy.PRESERVE_BOX.name,
        }
        direct_body = Block(name="direct")
        transformed_body = Block(name="transformed")
        op_without_transform = InvokeOperation(
            operands=[],
            results=[],
            target=ref,
            transform=transform,
            attrs=attrs,
            definition=CallableDef(
                ref=ref,
                body=direct_body,
                default_policy=CallPolicy.PRESERVE_BOX,
                attrs=attrs,
            ),
        )
        op_with_transform = InvokeOperation(
            operands=[],
            results=[],
            target=ref,
            transform=transform,
            attrs=attrs,
            definition=CallableDef(
                ref=ref,
                body=direct_body,
                implementations=[
                    CallableImplementation(
                        transform=transform,
                        body=transformed_body,
                    ),
                ],
                default_policy=CallPolicy.PRESERVE_BOX,
                attrs=attrs,
            ),
        )

        assert op_without_transform.effective_body() is None
        assert op_with_transform.effective_body() is transformed_body

    def test_invoke_vector_target_keeps_parameters_after_quantum_operands(self):
        """InvokeOperation separates vector targets from classical parameters."""
        dim = Value(type=UIntType(), name="n").with_const(3)
        target = ArrayValue(type=QubitType(), name="q", shape=(dim,))
        theta = Value(type=FloatType(), name="theta")
        ref = CallableRef(namespace="user", name="vector_gate")
        op = InvokeOperation(
            operands=[target, theta],
            results=[target.next_version()],
            target=ref,
            attrs={
                "kind": "composite",
                "gate_type": CompositeGateType.CUSTOM.name,
                "custom_name": "vector_gate",
                "num_control_qubits": 0,
                "num_target_qubits": 3,
                "default_policy": CallPolicy.PRESERVE_BOX.name,
            },
        )

        assert op.target_qubits == [target]
        assert op.parameters == [theta]

    def test_emit_fallback_uses_selected_implementation_body(self):
        """Composite fallback emits the selected implementation body."""
        ref = CallableRef(namespace="user", name="selectable")
        attrs = {
            "kind": "composite",
            "gate_type": CompositeGateType.CUSTOM.name,
            "custom_name": "selectable",
            "strategy_name": "fast",
            "num_control_qubits": 0,
            "num_target_qubits": 0,
            "default_policy": CallPolicy.PRESERVE_BOX.name,
        }
        default_body = Block(name="default")
        fast_body = Block(name="fast")
        op = InvokeOperation(
            operands=[],
            results=[],
            target=ref,
            attrs=attrs,
            definition=CallableDef(
                ref=ref,
                body=default_body,
                implementations=[
                    CallableImplementation(strategy="fast", body=fast_body),
                ],
                default_policy=CallPolicy.PRESERVE_BOX,
                attrs=attrs,
            ),
        )

        class EmitPassStub:
            """Capture the implementation passed to custom composite emission."""

            emitted_impl: Block | None = None

            def _emit_custom_composite(
                self,
                circuit: Any,
                op: InvokeOperation,
                impl: Block,
                qubit_indices: list[int],
                bindings: dict[str, Any],
            ) -> None:
                """Record the selected implementation body."""
                del circuit, op, qubit_indices, bindings
                self.emitted_impl = impl

        emit_pass = EmitPassStub()
        emit_composite_fallback(emit_pass, object(), op, [], {})

        assert emit_pass.emitted_impl is fast_body

    def test_emit_invoke_allows_bodyless_oracle_with_native_emitter(self):
        """Bodyless oracle invocation can execute through a native emitter."""
        ref = CallableRef(namespace="user", name="native_oracle")
        attrs = {
            "kind": "oracle",
            "custom_name": "native_oracle",
            "num_control_qubits": 0,
            "num_target_qubits": 0,
            "default_policy": CallPolicy.NATIVE_FIRST.name,
        }

        class NativeEmitter:
            """Capture a native callable emission."""

            emitted: bool = False

            def emit(
                self,
                circuit: Any,
                op: InvokeOperation,
                qubit_indices: list[int],
                bindings: dict[str, Any],
            ) -> bool:
                """Record that native emission was used."""
                del circuit, op, qubit_indices, bindings
                self.emitted = True
                return True

        class EmitPassStub:
            """Provide the minimal fields used by invoke emission."""

            def __init__(self) -> None:
                """Initialize the stub with no global composite emitters."""
                self._composite_emitters: list[Any] = []

        native = NativeEmitter()
        op = InvokeOperation(
            operands=[],
            results=[],
            target=ref,
            attrs=attrs,
            definition=CallableDef(
                ref=ref,
                implementations=[
                    CallableImplementation(emitter=native),
                ],
                default_policy=CallPolicy.NATIVE_FIRST,
                attrs=attrs,
            ),
        )

        emit_invoke_operation(EmitPassStub(), object(), op, {}, {})

        assert native.emitted is True

    def test_emit_invoke_selects_backend_specific_callable_emitter(self):
        """Backend-specific callable emitters are selected by emit backend name."""
        ref = CallableRef(namespace="user", name="native_oracle")
        attrs = {
            "kind": "oracle",
            "custom_name": "native_oracle",
            "num_control_qubits": 0,
            "num_target_qubits": 0,
            "default_policy": CallPolicy.NATIVE_FIRST.name,
        }

        class NativeEmitter:
            """Capture backend-specific native callable emission."""

            emitted: bool = False

            def emit(
                self,
                circuit: Any,
                op: InvokeOperation,
                qubit_indices: list[int],
                bindings: dict[str, Any],
            ) -> bool:
                """Record that native emission was used."""
                del circuit, op, qubit_indices, bindings
                self.emitted = True
                return True

        class EmitPassStub:
            """Provide a qiskit backend name and no global emitters."""

            backend_name = "qiskit"

            def __init__(self) -> None:
                """Initialize the stub with no global composite emitters."""
                self._composite_emitters: list[Any] = []

        native = NativeEmitter()
        op = InvokeOperation(
            operands=[],
            results=[],
            target=ref,
            attrs=attrs,
            definition=CallableDef(
                ref=ref,
                implementations=[
                    CallableImplementation(backend="qiskit", emitter=native),
                ],
                default_policy=CallPolicy.NATIVE_FIRST,
                attrs=attrs,
            ),
        )

        emit_invoke_operation(EmitPassStub(), object(), op, {}, {})

        assert native.emitted is True

    def test_emit_invoke_rejects_bodyless_oracle_when_backend_impl_mismatches(self):
        """Backend-specific callable emitters do not match another backend."""
        ref = CallableRef(namespace="user", name="native_oracle")
        attrs = {
            "kind": "oracle",
            "custom_name": "native_oracle",
            "num_control_qubits": 0,
            "num_target_qubits": 0,
            "default_policy": CallPolicy.NATIVE_FIRST.name,
        }

        class NativeEmitter:
            """Backend-specific native callable emitter stand-in."""

            def emit(
                self,
                circuit: Any,
                op: InvokeOperation,
                qubit_indices: list[int],
                bindings: dict[str, Any],
            ) -> bool:
                """Fail if the mismatched emitter is incorrectly selected."""
                del circuit, op, qubit_indices, bindings
                raise AssertionError("mismatched backend emitter should not run")

        class EmitPassStub:
            """Provide a different backend name from the implementation."""

            backend_name = "quri_parts"

            def __init__(self) -> None:
                """Initialize the stub with no global composite emitters."""
                self._composite_emitters: list[Any] = []

        op = InvokeOperation(
            operands=[],
            results=[],
            target=ref,
            attrs=attrs,
            definition=CallableDef(
                ref=ref,
                implementations=[
                    CallableImplementation(backend="qiskit", emitter=NativeEmitter()),
                ],
                default_policy=CallPolicy.NATIVE_FIRST,
                attrs=attrs,
            ),
        )

        with pytest.raises(EmitError, match="has no implementation"):
            emit_invoke_operation(EmitPassStub(), object(), op, {}, {})

    def test_callable_implementation_emitter_precedes_global_composite_emitter(self):
        """Callable-specific emitters take priority over type-wide emitters."""
        ref = CallableRef(namespace="user", name="custom")
        attrs = {
            "kind": "composite",
            "gate_type": CompositeGateType.CUSTOM.name,
            "custom_name": "custom",
            "num_control_qubits": 0,
            "num_target_qubits": 0,
            "default_policy": CallPolicy.NATIVE_FIRST.name,
        }

        class NativeEmitter:
            """Capture callable-specific emission."""

            emitted: bool = False

            def emit(
                self,
                circuit: Any,
                op: InvokeOperation,
                qubit_indices: list[int],
                bindings: dict[str, Any],
            ) -> bool:
                """Record callable-specific emission."""
                del circuit, op, qubit_indices, bindings
                self.emitted = True
                return True

        class GlobalEmitter:
            """Capture whether the global composite emitter was used."""

            emitted: bool = False

            def can_emit(self, gate_type: CompositeGateType) -> bool:
                """Return true for all composite gate types."""
                del gate_type
                return True

            def emit(
                self,
                circuit: Any,
                op: InvokeOperation,
                qubit_indices: list[int],
                bindings: dict[str, Any],
            ) -> bool:
                """Record global emission."""
                del circuit, op, qubit_indices, bindings
                self.emitted = True
                return True

        class EmitPassStub:
            """Provide a backend name and global composite emitter."""

            backend_name = "qiskit"

            def __init__(self, global_emitter: GlobalEmitter) -> None:
                """Initialize the stub with one global composite emitter."""
                self._composite_emitters = [global_emitter]

        native = NativeEmitter()
        global_emitter = GlobalEmitter()
        op = InvokeOperation(
            operands=[],
            results=[],
            target=ref,
            attrs=attrs,
            definition=CallableDef(
                ref=ref,
                implementations=[
                    CallableImplementation(backend="qiskit", emitter=native),
                ],
                default_policy=CallPolicy.NATIVE_FIRST,
                attrs=attrs,
            ),
        )

        emit_invoke_operation(EmitPassStub(global_emitter), object(), op, {}, {})

        assert native.emitted is True
        assert global_emitter.emitted is False

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_simple_composite_gate_definition(self, n):
        """A simple CompositeGate can be defined with _decompose()."""
        gate = HadamardAll(n)
        assert gate.num_target_qubits == n
        assert gate.num_control_qubits == 0
        assert gate.gate_type == CompositeGateType.CUSTOM
        assert gate.custom_name == "hadamard_all"

    def test_apply_composite_gate_in_qkernel(self):
        """CompositeGate can be used inside a qkernel."""
        double_h = DoubleH()

        @qkernel
        def circuit(q: Qubit) -> Qubit:
            (q,) = double_h(q)
            return q

        block = circuit.build()

        ops = block.operations
        assert len(ops) == 2
        assert isinstance(ops[0], QInitOperation)
        _assert_composite_invoke(ops[1], name="double_h", num_targets=1)

    def test_apply_multi_qubit_composite_gate(self):
        """Multi-qubit CompositeGate works correctly."""
        bell = BellPair()

        @qkernel
        def circuit(q0: Qubit, q1: Qubit) -> tuple[Qubit, Qubit]:
            q0, q1 = bell(q0, q1)
            return q0, q1

        block = circuit.build()

        ops = block.operations
        assert len(ops) == 3
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], QInitOperation)
        _assert_composite_invoke(ops[2], name="bell_pair", num_targets=2)

    def test_wrong_num_qubits_raises_error(self):
        """Passing wrong number of qubits raises ValueError."""

        class TwoQubitGate(CompositeGate):
            @property
            def num_target_qubits(self) -> int:
                return 2

            def _decompose(self, qubits: tuple[Qubit, ...]) -> tuple[Qubit, ...]:
                return qubits

        gate = TwoQubitGate()

        @qkernel
        def circuit(q: Qubit) -> Qubit:
            (q,) = gate(q)
            return q

        with pytest.raises(ValueError, match="requires 2 target qubits"):
            circuit.build()

    def test_composite_without_decomposition_raises(self):
        """CompositeGate without a body points users to Oracle."""

        class MissingBodyGate(CompositeGate):
            custom_name = "oracle"

            @property
            def num_target_qubits(self) -> int:
                return 2

        gate = MissingBodyGate()

        @qkernel
        def circuit() -> Vector[Qubit]:
            qs = qubit_array(2, "qs")
            q0, q1 = gate(qs[0], qs[1])
            qs[0] = q0
            qs[1] = q1
            return qs

        with pytest.raises(ValueError, match="Use Oracle"):
            circuit.build()

    def test_opaque_helper_emits_bodyless_invoke(self):
        """opaque() creates an Oracle backed by a bodyless InvokeOperation."""
        model = qmc.FixedResourceModel(
            width=qmc.WidthResources(clean_ancilla_qubits=2),
            calls=qmc.CallResources(queries_by_name={"black_box": 3}),
        )
        black_box = qmc.opaque("black_box", 1, resource_model=model)

        assert isinstance(black_box, qmc.Oracle)

        @qkernel
        def circuit(q: Qubit) -> Qubit:
            (q,) = black_box(q)
            return q

        block = circuit.build()
        invokes = _invoke_ops(block.operations)

        assert len(invokes) == 1
        op = invokes[0]
        assert op.attrs["kind"] == "oracle"
        assert op.target.name == "black_box"
        assert op.body is None
        assert op.definition is not None
        assert op.definition.resource_models[0].model is model

    def test_controlled_opaque_records_controlled_transform(self):
        """Controlled opaque calls use the controlled Invoke transform."""
        model = qmc.FixedResourceModel(
            gates=qmc.GateResources(t=4),
            calls=qmc.CallResources(queries_by_name={"controlled_black_box": 1}),
        )
        black_box = qmc.Oracle(
            "controlled_black_box",
            num_qubits=1,
            num_control_qubits=1,
            resource_model=model,
        )

        @qkernel
        def circuit(ctrl: Qubit, target: Qubit) -> tuple[Qubit, Qubit]:
            ctrl, target = black_box(target, controls=(ctrl,))
            return ctrl, target

        block = circuit.build()
        invokes = _invoke_ops(block.operations)

        assert len(invokes) == 1
        op = invokes[0]
        assert op.attrs["kind"] == "oracle"
        assert op.transform is CallTransform.CONTROLLED
        assert op.num_control_qubits == 1
        assert op.definition is not None
        assert op.definition.resource_models[0].model is model

    def test_opaque_accepts_callable_signature(self):
        """opaque() accepts a vector CallableSignature without fixed arity."""
        black_box = qmc.opaque(
            "vector_black_box",
            signature=qmc.CallableSignature(
                inputs=[qmc.Vector[qmc.Qubit]],
                outputs=[qmc.Vector[qmc.Qubit]],
            ),
        )

        @qkernel
        def circuit() -> Vector[Qubit]:
            qs = qubit_array(2, "qs")
            qs = black_box(qs)
            return qs

        block = circuit.build()
        invokes = _invoke_ops(block.operations)

        assert len(invokes) == 1
        op = invokes[0]
        assert op.attrs["kind"] == "oracle"
        assert op.num_target_qubits == 2
        assert op.definition is not None
        assert op.definition.signature is not None
        assert op.body is None

    def test_composite_alias_attaches_resource_model(self):
        """Compatibility alias attaches callable resource models."""
        model = qmc.FixedResourceModel(gates=qmc.GateResources(total=1))

        @qmc.composite(name="boxed_h", resource_model=model)
        @qkernel
        def boxed_h(q: Qubit) -> Qubit:
            q = qmc.h(q)
            return q

        @qkernel
        def circuit(q: Qubit) -> Qubit:
            q = boxed_h(q)
            return q

        block = circuit.build()
        invokes = _invoke_ops(block.operations)

        assert len(invokes) == 1
        op = invokes[0]
        assert op.attrs["kind"] == "composite"
        assert op.target.namespace == "user.composite"
        assert op.target.name == "boxed_h"
        assert op.definition is not None
        assert op.definition.resource_models[0].model is model

    def test_composite_gate_accepts_raw_function(self):
        """Decorator can qkernel-wrap a raw function before boxing it."""
        model = qmc.FixedResourceModel(gates=qmc.GateResources(total=1))

        @qmc.composite_gate(name="raw_boxed_h", resource_model=model)
        def raw_boxed_h(q: Qubit) -> Qubit:
            q = qmc.h(q)
            return q

        @qkernel
        def circuit(q: Qubit) -> Qubit:
            q = raw_boxed_h(q)
            return q

        block = circuit.build()
        invokes = _invoke_ops(block.operations)

        assert len(invokes) == 1
        op = invokes[0]
        assert op.definition is not None
        assert op.definition.resource_models[0].model is model
        assert op.attrs["kind"] == "composite"
        assert op.target.namespace == "user.composite"
        assert op.target.name == "raw_boxed_h"
        assert op.body is not None

    def test_composite_gate_direct_form_accepts_raw_function(self):
        """@composite_gate without arguments can box a raw function."""

        @qmc.composite_gate
        def raw_entangle(a: Qubit, b: Qubit) -> tuple[Qubit, Qubit]:
            a, b = qmc.cx(a, b)
            return a, b

        @qkernel
        def circuit(a: Qubit, b: Qubit) -> tuple[Qubit, Qubit]:
            a, b = raw_entangle(a, b)
            return a, b

        block = circuit.build()
        invokes = _invoke_ops(block.operations)

        assert len(invokes) == 1
        op = invokes[0]
        assert op.attrs["kind"] == "composite"
        assert op.target.name == "raw_entangle"
        assert op.body is not None

    def test_decompose_called_during_call(self, mocker):
        """_decompose is called exactly once when gate is invoked."""
        spy = mocker.spy(DoubleH, "_decompose")
        double_h = DoubleH()

        @qkernel
        def circuit(q: Qubit) -> Qubit:
            (q,) = double_h(q)
            return q

        circuit.build()
        spy.assert_called_once()

    def test_strategy_overrides_decompose(self, mocker):
        """When a strategy is registered and selected, _decompose is NOT called."""

        class TestGateForStrategy(CompositeGate):
            custom_name = "test_strat_gate"

            @property
            def num_target_qubits(self) -> int:
                return 1

            def _decompose(self, qubits):
                q = qubits[0]
                q = qmc.h(q)
                return (q,)

        gate = TestGateForStrategy()

        mock_strategy = mocker.MagicMock()
        mock_strategy.name = "mock_strat"
        mock_strategy.decompose.side_effect = lambda qubits: (qmc.x(qubits[0]),)

        TestGateForStrategy.register_strategy("mock_strat", mock_strategy)
        decompose_spy = mocker.spy(TestGateForStrategy, "_decompose")

        @qkernel
        def circuit(q: Qubit) -> Qubit:
            (q,) = gate(q, strategy="mock_strat")
            return q

        circuit.build()

        mock_strategy.decompose.assert_called_once()
        decompose_spy.assert_not_called()

        del TestGateForStrategy._strategies["mock_strat"]

    def test_decomposition_uses_fresh_tracer(self):
        """Decomposition operations don't leak into the outer block."""
        bell = BellPair()

        @qkernel
        def circuit(q0: Qubit, q1: Qubit) -> tuple[Qubit, Qubit]:
            q0, q1 = bell(q0, q1)
            return q0, q1

        block = circuit.build()
        # Block should have QInit ops + 1 InvokeOperation only
        # (no inner H/CX ops leaked from decomposition)
        invoke_ops = _invoke_ops(block.operations)
        assert len(invoke_ops) == 1
        assert invoke_ops[0].attrs["custom_name"] == "bell_pair"

    def test_output_qubit_versions_incremented(self):
        """Output qubits have version = input version + 1 with same logical_id."""
        double_h = DoubleH()

        @qkernel
        def circuit(q: Qubit) -> Qubit:
            (q,) = double_h(q)
            return q

        block = circuit.build()

        ops = block.operations
        assert len(ops) == 2
        assert isinstance(ops[0], QInitOperation)
        op = _assert_composite_invoke(ops[1], name="double_h", num_targets=1)
        # Target operand is the last operand.
        target_operand = op.operands[-1]
        target_result = op.results[-1]
        assert target_result.version == target_operand.version + 1
        assert target_result.logical_id == target_operand.logical_id

    def test_operands_order_in_ir(self):
        """Operands are ordered: [...controls, ...targets]."""

        class ControlledGate(CompositeGate):
            custom_name = "ctrl_gate"

            @property
            def num_target_qubits(self) -> int:
                return 1

            @property
            def num_control_qubits(self) -> int:
                return 1

            def _decompose(self, qubits):
                q = qubits[0]
                q = qmc.h(q)
                return (q,)

        gate = ControlledGate()

        @qkernel
        def circuit(ctrl: Qubit, tgt: Qubit) -> tuple[Qubit, Qubit]:
            ctrl, tgt = gate(tgt, controls=(ctrl,))
            return ctrl, tgt

        block = circuit.build()

        ops = block.operations
        assert len(ops) == 3
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], QInitOperation)
        op = _assert_composite_invoke(
            ops[2],
            name="ctrl_gate",
            num_targets=1,
            num_controls=1,
        )
        assert op.body is not None
        # operands[0] = control qubit value, operands[1] = target qubit value
        ctrl_value = ops[0].results[0]
        tgt_value = ops[1].results[0]
        assert op.operands[0].logical_id == ctrl_value.logical_id
        assert op.operands[1].logical_id == tgt_value.logical_id
        assert op.num_control_qubits == 1
        assert op.num_target_qubits == 1
        assert op.transform is CallTransform.CONTROLLED
        assert op.implementation_for() is not None
        assert op.effective_body() is op.body

    def test_strategy_registry_isolation(self):
        """Strategies registered on one subclass are not visible to another."""

        class GateA(CompositeGate):
            custom_name = "gate_a"

            @property
            def num_target_qubits(self) -> int:
                return 1

        class GateB(CompositeGate):
            custom_name = "gate_b"

            @property
            def num_target_qubits(self) -> int:
                return 1

        class FakeStrategy:
            @property
            def name(self) -> str:
                return "fake"

            def decompose(self, qubits):
                return qubits

            def resources(self, num_qubits):
                return None

        GateA.register_strategy("fake", FakeStrategy())
        assert GateA.get_strategy("fake") is not None
        assert GateB.get_strategy("fake") is None
        del GateA._strategies["fake"]

    def test_tracer_add_operation_called(self, mocker):
        """Gate invocation calls tracer.add_operation with InvokeOperation."""
        spy = mocker.spy(Tracer, "add_operation")
        double_h = DoubleH()

        @qkernel
        def circuit(q: Qubit) -> Qubit:
            (q,) = double_h(q)
            return q

        circuit.build()

        # add_operation is called for QInitOperation + InvokeOperation
        invoke_calls = [
            c for c in spy.call_args_list if isinstance(c.args[1], InvokeOperation)
        ]
        assert len(invoke_calls) == 1
        assert invoke_calls[0].args[1].attrs["custom_name"] == "double_h"

    def test_output_preserves_parent_indices(self):
        """Output qubit handles preserve parent_array from array elements."""
        bell = BellPair()

        @qkernel
        def circuit() -> Vector[Qubit]:
            qs = qubit_array(2, "qs")
            q0, q1 = bell(qs[0], qs[1])
            qs[0] = q0
            qs[1] = q1
            return qs

        block = circuit.build()
        ops = block.operations
        assert len(ops) == 2
        assert isinstance(ops[0], QInitOperation)
        op = _assert_composite_invoke(ops[1], name="bell_pair", num_targets=2)
        # Result values should have parent_array set (from array elements)
        for index, r in enumerate(op.results):
            assert isinstance(r.parent_array, ArrayValue)
            assert len(r.element_indices) == 1
            assert isinstance(r.element_indices[0], Value)
            assert r.element_indices[0].get_const() == index


class TestQFTAndIQFTClasses:
    """Test the built-in QFT and IQFT CompositeGate classes."""

    def test_qft_class_attributes(self):
        """QFT class has correct attributes."""
        from qamomile.circuit.stdlib.qft import QFT

        qft = QFT(4)
        assert qft.num_target_qubits == 4
        assert qft.gate_type == CompositeGateType.QFT
        assert qft.custom_name == "qft"

    def test_iqft_class_attributes(self):
        """IQFT class has correct attributes."""
        from qamomile.circuit.stdlib.qft import IQFT

        iqft = IQFT(3)
        assert iqft.num_target_qubits == 3
        assert iqft.gate_type == CompositeGateType.IQFT
        assert iqft.custom_name == "iqft"

    def test_qft_strategy_is_callable_implementation(self):
        """QFT strategy body is selected as an implementation."""
        from qamomile.circuit.stdlib.qft import QFT

        @qkernel
        def circuit(q0: Qubit, q1: Qubit, q2: Qubit) -> tuple[Qubit, Qubit, Qubit]:
            qft = QFT(3)
            q0, q1, q2 = qft(q0, q1, q2, strategy="approximate_k2")
            return q0, q1, q2

        block = circuit.build()
        invokes = _invoke_ops(block.operations)

        assert len(invokes) == 1
        op = invokes[0]
        assert op.gate_type is CompositeGateType.QFT
        assert op.strategy_name == "approximate_k2"
        impl = op.implementation_for()
        assert impl is not None
        assert impl.strategy == "approximate_k2"
        assert impl.body is op.body
        assert op.effective_body() is op.body

    def test_qft_in_qkernel(self):
        """QFT can be used in a qkernel."""
        from qamomile.circuit.stdlib.qft import QFT

        @qkernel
        def circuit(q0: Qubit, q1: Qubit) -> tuple[Qubit, Qubit]:
            qft = QFT(2)
            q0, q1 = qft(q0, q1)
            return q0, q1

        block = circuit.build()
        assert block is not None

    def test_qft_factory_function(self):
        """qft() factory function works in qkernel."""
        from qamomile.circuit.stdlib.qft import qft

        @qkernel
        def circuit() -> Vector[Qubit]:
            qubits = qubit_array(3, "q")
            qubits = qft(qubits)
            return qubits

        block = circuit.build()
        assert block is not None

    def test_iqft_factory_function(self):
        """iqft() factory function works in qkernel."""
        from qamomile.circuit.stdlib.qft import iqft

        @qkernel
        def circuit() -> Vector[Qubit]:
            qubits = qubit_array(3, "q")
            qubits = iqft(qubits)
            return qubits

        block = circuit.build()
        assert block is not None


class TestCompositeGateTranspilation:
    """Test CompositeGate IR generation and transpilation (requires Qiskit)."""

    def test_composite_decorator_can_be_compiler_entrypoint(self, qiskit_transpiler):
        """composite-decorated qkernels are accepted as qkernel-like entrypoints."""

        @qmc.composite_gate(name="bell_entry")
        @qmc.qkernel
        def bell_entry() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        executable = qiskit_transpiler.transpile(bell_entry)
        circuit = executable.get_first_circuit()

        assert circuit is not None
        assert circuit.num_qubits == 2

    def test_custom_gate_builds_ir(self, qiskit_transpiler):
        """Custom CompositeGate builds correct IR."""
        bell = BellPair()

        @qkernel
        def circuit() -> Vector[Qubit]:
            qs = qubit_array(2, "qs")
            q0 = qs[0]
            q1 = qs[1]
            q0, q1 = bell(q0, q1)
            qs[0] = q0
            qs[1] = q1
            return qs

        block = qiskit_transpiler.to_block(circuit)
        inlined = qiskit_transpiler.inline(block)

        invoke_ops = _invoke_ops(inlined.operations)
        assert len(invoke_ops) == 1
        assert invoke_ops[0].body is not None
        body_op_types = [type(op).__name__ for op in invoke_ops[0].body.operations]
        assert "GateOperation" in body_op_types

    def test_qft_builds_ir(self, qiskit_transpiler):
        """QFT builds correct IR with native gate type."""
        from qamomile.circuit.stdlib.qft import QFT

        @qkernel
        def circuit() -> Vector[Qubit]:
            qs = qubit_array(3, "qs")
            q0 = qs[0]
            q1 = qs[1]
            q2 = qs[2]
            qft = QFT(3)
            q0, q1, q2 = qft(q0, q1, q2)
            qs[0] = q0
            qs[1] = q1
            qs[2] = q2
            return qs

        block = qiskit_transpiler.to_block(circuit)

        invoke_ops = _invoke_ops(block.operations)
        assert len(invoke_ops) == 1
        assert invoke_ops[0].attrs["gate_type"] == CompositeGateType.QFT.name

    def test_iqft_builds_ir(self, qiskit_transpiler):
        """IQFT builds correct IR with native gate type."""
        from qamomile.circuit.stdlib.qft import IQFT

        @qkernel
        def circuit() -> Vector[Qubit]:
            qs = qubit_array(2, "qs")
            q0 = qs[0]
            q1 = qs[1]
            iqft = IQFT(2)
            q0, q1 = iqft(q0, q1)
            qs[0] = q0
            qs[1] = q1
            return qs

        block = qiskit_transpiler.to_block(circuit)

        invoke_ops = _invoke_ops(block.operations)
        assert len(invoke_ops) == 1
        assert invoke_ops[0].attrs["gate_type"] == CompositeGateType.IQFT.name

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_hadamard_all_transpile_circuit(self, qiskit_transpiler, n):
        """HadamardAll transpiles to a circuit with correct qubit count."""
        gate = HadamardAll(n)

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(n, "qs")
            qs = apply_gate_to_array(qs, gate)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(circuit)
        assert qc.num_qubits == n
        assert len(qc.data) == n + 1  # boxed composite + |measurement|
        data = qc.data
        from qiskit.circuit import Measure

        assert data[0].operation.num_qubits == n
        assert not isinstance(data[0].operation, Measure)
        for i in range(1, n + 1):
            assert isinstance(data[i].operation, Measure)

    def test_bell_pair_transpile_circuit(self, qiskit_transpiler):
        """BellPair transpiles to a 2-qubit circuit."""
        bell = BellPair()

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(2, "qs")
            qs = apply_gate_to_array(qs, bell)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(circuit)
        assert qc.num_qubits == 2
        assert len(qc.data) == 3  # boxed composite + |measurement|
        data = qc.data
        from qiskit.circuit import Measure

        assert data[0].operation.num_qubits == 2
        assert not isinstance(data[0].operation, Measure)
        assert isinstance(data[1].operation, Measure)
        assert isinstance(data[2].operation, Measure)

    def test_double_h_transpile_circuit(self, qiskit_transpiler):
        """DoubleH transpiles to a 1-qubit circuit."""
        double_h = DoubleH()

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(1, "qs")
            qs = apply_gate_to_array(qs, double_h)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(circuit)
        assert qc.num_qubits == 1
        assert len(qc.data) == 2  # boxed composite + |measurement|
        data = qc.data
        from qiskit.circuit import Measure

        assert data[0].operation.num_qubits == 1
        assert not isinstance(data[0].operation, Measure)
        assert isinstance(data[1].operation, Measure)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_hadamard_all_statevector(self, qiskit_transpiler, n):
        """HadamardAll on |0...0> produces uniform superposition."""
        gate = HadamardAll(n)

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(n, "qs")
            qs = apply_gate_to_array(qs, gate)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(circuit)
        sv = run_statevector(qc)

        expected_amp = 1.0 / np.sqrt(2**n)
        assert np.allclose(np.abs(sv), expected_amp, atol=1e-8), (
            f"n={n}: expected uniform amplitudes {expected_amp}, got {np.abs(sv)}"
        )

    def test_bell_pair_statevector(self, qiskit_transpiler):
        """BellPair produces (|00> + |11>) / sqrt(2)."""
        bell = BellPair()

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(2, "qs")
            qs = apply_gate_to_array(qs, bell)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(circuit)
        sv = run_statevector(qc)

        # Bell state: |00> and |11> each have amplitude 1/sqrt(2)
        expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        assert np.allclose(np.abs(sv), np.abs(expected), atol=1e-8), (
            f"Expected Bell state, got {sv}"
        )

    def test_double_h_statevector(self, qiskit_transpiler):
        """DoubleH (H*H = I) leaves |0> unchanged."""
        double_h = DoubleH()

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(1, "qs")
            qs = apply_gate_to_array(qs, double_h)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(circuit)
        sv = run_statevector(qc)

        expected = np.array([1, 0], dtype=complex)
        assert np.allclose(np.abs(sv), np.abs(expected), atol=1e-8), (
            f"Expected |0>, got {sv}"
        )

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_hadamard_all_sampling(self, qiskit_transpiler, seeded_executor, n):
        """HadamardAll on |0...0> produces approximately uniform distribution."""
        gate = HadamardAll(n)

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(n, "qs")
            qs = apply_gate_to_array(qs, gate)
            return qmc.measure(qs)

        executable = qiskit_transpiler.transpile(circuit)
        shots = 4096
        job = executable.sample(seeded_executor, shots=shots)
        result = job.result()

        # num_outcomes = number of distinct measurement bitstrings, i.e. 2**n.
        # Under H^⊗n on |0...0> all outcomes are equally likely (probability
        # 1/num_outcomes each), independent of `shots` (the number of shots).
        num_outcomes = 2**n
        counts = {i: 0 for i in range(num_outcomes)}
        for bits, count in result.results:
            val = sum(b << i for i, b in enumerate(bits))
            counts[val] += count

        assert sum(counts.values()) == shots
        assert len([c for c in counts.values() if c > 0]) == num_outcomes

        # Each per-outcome count is a Binomial(shots, 1/num_outcomes) marginal
        # of the multinomial sampling distribution, so the 4 + 8 + 16 = 28
        # checks aggregated across n in {2, 3, 4} are not independent -- but
        # Bonferroni (union bound) is still a valid upper bound. A naive
        # 3-sigma tolerance has a two-sided Gaussian tail of ~0.27% per
        # outcome; multiplied by 28 outcomes this gives an aggregate flake
        # rate of ~7.6% per CI run, which matches the reported flakes. At
        # k_sigma = 5, summing the *exact* binomial two-sided tails over
        # all 28 outcomes gives an aggregate bound of ~2.2e-5 per CI run.
        # The simulator RNG is still pinned in `seeded_executor` (see
        # conftest.py) so any failure is reproducible.
        expected = shots / num_outcomes
        sigma = (expected * (1 - 1 / num_outcomes)) ** 0.5
        k_sigma = 5
        for outcome, count in counts.items():
            assert abs(count - expected) < k_sigma * sigma, (
                f"n={n}, outcome={outcome}: count={count}, "
                f"expected={expected:.0f} +/- {k_sigma * sigma:.0f}"
            )

    def test_bell_pair_sampling(self, qiskit_transpiler, seeded_executor):
        """BellPair produces only |00> and |11> outcomes."""
        bell = BellPair()

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(2, "qs")
            qs = apply_gate_to_array(qs, bell)
            return qmc.measure(qs)

        executable = qiskit_transpiler.transpile(circuit)
        shots = 1024
        job = executable.sample(seeded_executor, shots=shots)
        result = job.result()

        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for bits, count in result.results:
            val = sum(b << i for i, b in enumerate(bits))
            counts[val] += count

        assert sum(counts.values()) == shots
        # Only |00> (0) and |11> (3) should appear
        assert counts[1] == 0, f"Unexpected |01> count: {counts[1]}"
        assert counts[2] == 0, f"Unexpected |10> count: {counts[2]}"
        assert counts[0] > 0, "Expected |00> outcomes"
        assert counts[3] > 0, "Expected |11> outcomes"

    def test_double_h_sampling(self, qiskit_transpiler, seeded_executor):
        """DoubleH (identity) always measures 0."""
        double_h = DoubleH()

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(1, "qs")
            qs = apply_gate_to_array(qs, double_h)
            return qmc.measure(qs)

        executable = qiskit_transpiler.transpile(circuit)
        shots = 1024
        job = executable.sample(seeded_executor, shots=shots)
        result = job.result()

        for bits, count in result.results:
            assert all(b == 0 for b in bits), (
                f"Expected all zeros, got {bits} (count={count})"
            )

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_hadamard_all_statevector_symbolic(self, qiskit_transpiler, n):
        """HadamardAll with symbolic n produces uniform superposition."""
        gate = HadamardAll(n)

        @qkernel
        def circuit(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(num, "qs")
            qs = apply_gate_to_array(qs, gate)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(circuit, bindings={"num": n})
        sv = run_statevector(qc)

        expected_amp = 1.0 / np.sqrt(2**n)
        assert np.allclose(np.abs(sv), expected_amp, atol=1e-8), (
            f"n={n}: expected uniform amplitudes {expected_amp}, got {np.abs(sv)}"
        )

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_hadamard_all_sampling_symbolic(
        self, qiskit_transpiler, seeded_executor, n
    ):
        """HadamardAll with symbolic n produces uniform distribution."""
        gate = HadamardAll(n)

        @qkernel
        def circuit(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(num, "qs")
            qs = apply_gate_to_array(qs, gate)
            return qmc.measure(qs)

        executable = qiskit_transpiler.transpile(circuit, bindings={"num": n})
        shots = 4096
        job = executable.sample(seeded_executor, shots=shots)
        result = job.result()

        # num_outcomes = number of distinct measurement bitstrings, i.e. 2**n.
        # Under H^⊗n on |0...0> all outcomes are equally likely (probability
        # 1/num_outcomes each), independent of `shots` (the number of shots).
        num_outcomes = 2**n
        counts = {i: 0 for i in range(num_outcomes)}
        for bits, count in result.results:
            val = sum(b << i for i, b in enumerate(bits))
            counts[val] += count

        assert sum(counts.values()) == shots
        assert len([c for c in counts.values() if c > 0]) == num_outcomes

    def test_no_phantom_qubits_symbolic(self, qiskit_transpiler):
        """No phantom qubits with symbolic n and CompositeGate."""
        bell = BellPair()

        @qkernel
        def circuit(num: qmc.UInt) -> qmc.Bit:
            qs = qubit_array(num, "qs")
            q0, q1 = bell(qs[0], qs[1])
            qs[0] = q0
            qs[1] = q1
            return qmc.measure(qs[0])

        qc = qiskit_transpiler.to_circuit(circuit, bindings={"num": 2})
        qc.remove_final_measurements()
        assert qc.num_qubits == 2, f"Expected 2 qubits, got {qc.num_qubits}"


class TestNestedQKernelNoPhantomQubits:
    """Verify no phantom qubits when nesting qkernels with array arguments."""

    def test_2_level_nest(self, qiskit_transpiler):
        """No phantom qubits with 2-level nested qkernel passing array."""

        @qkernel
        def inner(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs[0] = qmc.h(qs[0])
            return qs

        @qkernel
        def outer() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(3, "qs")
            qs = inner(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(outer)
        assert qc.num_qubits == 3, f"Expected 3 qubits, got {qc.num_qubits}"

    def test_3_level_nest(self, qiskit_transpiler):
        """No phantom qubits with 3-level nested qkernel passing array."""

        @qkernel
        def level_c(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs[0] = qmc.h(qs[0])
            return qs

        @qkernel
        def level_b(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs = level_c(qs)
            return qs

        @qkernel
        def level_a() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(3, "qs")
            qs = level_b(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(level_a)
        assert qc.num_qubits == 3, f"Expected 3 qubits, got {qc.num_qubits}"

    def test_4_level_nest(self, qiskit_transpiler):
        """No phantom qubits with 4-level nested qkernel passing array."""

        @qkernel
        def d(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs[0] = qmc.h(qs[0])
            return qs

        @qkernel
        def c(qs: Vector[Qubit]) -> Vector[Qubit]:
            return d(qs)

        @qkernel
        def b(qs: Vector[Qubit]) -> Vector[Qubit]:
            return c(qs)

        @qkernel
        def a() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(3, "qs")
            qs = b(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(a)
        assert qc.num_qubits == 3, f"Expected 3 qubits, got {qc.num_qubits}"

    def test_inner_creates_array(self, qiskit_transpiler):
        """No phantom qubits when inner qkernel creates its own array."""

        @qkernel
        def inner_create() -> Vector[Qubit]:
            qs = qubit_array(3, "inner_qs")
            qs[0] = qmc.h(qs[0])
            return qs

        @qkernel
        def outer_create() -> qmc.Vector[qmc.Bit]:
            qs = inner_create()
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(outer_create)
        assert qc.num_qubits == 3, f"Expected 3 qubits, got {qc.num_qubits}"

    def test_3_level_nest_with_composite_gate(self, qiskit_transpiler):
        """No phantom qubits with 3-level nest and CompositeGate (QFT)."""
        from qamomile.circuit.stdlib.qft import qft

        @qkernel
        def apply_qft(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs = qft(qs)
            return qs

        @qkernel
        def middle(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs = apply_qft(qs)
            return qs

        @qkernel
        def top() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(3, "qs")
            qs = middle(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(top)
        assert qc.num_qubits == 3, f"Expected 3 qubits, got {qc.num_qubits}"

    def test_multi_element_access_nested(self, qiskit_transpiler):
        """No phantom qubits when multiple array elements are accessed in nested call."""

        @qkernel
        def process(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs[0] = qmc.h(qs[0])
            qs[1] = qmc.x(qs[1])
            qs[2] = qmc.h(qs[2])
            return qs

        @qkernel
        def wrapper(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs = process(qs)
            return qs

        @qkernel
        def main_circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(4, "qs")
            qs = wrapper(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(main_circuit)
        assert qc.num_qubits == 4, f"Expected 4 qubits, got {qc.num_qubits}"

    def test_same_qkernel_called_twice(self, qiskit_transpiler):
        """No phantom qubits when same qkernel is called twice on same array."""

        @qkernel
        def apply_h_first(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs[0] = qmc.h(qs[0])
            return qs

        @qkernel
        def call_twice() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(3, "qs")
            qs = apply_h_first(qs)
            qs = apply_h_first(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(call_twice)
        assert qc.num_qubits == 3, f"Expected 3 qubits, got {qc.num_qubits}"

    def test_multiple_array_parameters(self, qiskit_transpiler):
        """No phantom qubits with qkernel taking two array parameters."""

        @qkernel
        def entangle_arrays(
            qs1: Vector[Qubit], qs2: Vector[Qubit]
        ) -> tuple[Vector[Qubit], Vector[Qubit]]:
            qs1[0] = qmc.h(qs1[0])
            qs1[0], qs2[0] = qmc.cx(qs1[0], qs2[0])
            return qs1, qs2

        @qkernel
        def two_arrays() -> qmc.Vector[qmc.Bit]:
            a = qubit_array(2, "a")
            b = qubit_array(2, "b")
            a, b = entangle_arrays(a, b)
            return qmc.measure(a)

        qc = qiskit_transpiler.to_circuit(two_arrays)
        assert qc.num_qubits == 4, f"Expected 4 qubits, got {qc.num_qubits}"

    def test_scalar_qubit_and_vector_mixed(self, qiskit_transpiler):
        """No phantom qubits with mixed scalar Qubit and Vector parameters."""

        @qkernel
        def mixed_params(q: Qubit, qs: Vector[Qubit]) -> tuple[Qubit, Vector[Qubit]]:
            q, qs[0] = qmc.cx(q, qs[0])
            return q, qs

        @qkernel
        def scalar_and_vector() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(3, "qs")
            q = qs[0]
            q = qmc.h(q)
            qs[0] = q
            q2 = qs[1]
            q2, qs = mixed_params(q2, qs)
            qs[1] = q2
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(scalar_and_vector)
        assert qc.num_qubits == 3, f"Expected 3 qubits, got {qc.num_qubits}"

    def test_operations_before_and_after_nested_call(self, qiskit_transpiler):
        """No phantom qubits with array ops before and after nested call."""

        @qkernel
        def inner_op(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs[1] = qmc.x(qs[1])
            return qs

        @qkernel
        def before_and_after() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(3, "qs")
            qs[0] = qmc.h(qs[0])
            qs = inner_op(qs)
            qs[2] = qmc.h(qs[2])
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(before_and_after)
        assert qc.num_qubits == 3, f"Expected 3 qubits, got {qc.num_qubits}"

    def test_two_different_nested_calls(self, qiskit_transpiler):
        """No phantom qubits with two different nested qkernel calls."""

        @qkernel
        def apply_h(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs[0] = qmc.h(qs[0])
            return qs

        @qkernel
        def apply_x(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs[1] = qmc.x(qs[1])
            return qs

        @qkernel
        def two_nested() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(3, "qs")
            qs = apply_h(qs)
            qs = apply_x(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(two_nested)
        assert qc.num_qubits == 3, f"Expected 3 qubits, got {qc.num_qubits}"

    def test_nested_call_then_composite_gate(self, qiskit_transpiler):
        """No phantom qubits with nested call followed by CompositeGate."""
        from qamomile.circuit.stdlib.qft import qft

        @qkernel
        def apply_h(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs[0] = qmc.h(qs[0])
            return qs

        @qkernel
        def nested_then_qft() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(3, "qs")
            qs = apply_h(qs)
            qs = qft(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(nested_then_qft)
        assert qc.num_qubits == 3, f"Expected 3 qubits, got {qc.num_qubits}"

    def test_two_independent_arrays_with_nested_calls(self, qiskit_transpiler):
        """No phantom qubits with two independent arrays and separate nested calls."""

        @qkernel
        def apply_h(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs[0] = qmc.h(qs[0])
            return qs

        @qkernel
        def apply_x(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs[1] = qmc.x(qs[1])
            return qs

        @qkernel
        def two_independent() -> qmc.Vector[qmc.Bit]:
            a = qubit_array(2, "a")
            b = qubit_array(3, "b")
            a = apply_h(a)
            b = apply_x(b)
            return qmc.measure(b)

        qc = qiskit_transpiler.to_circuit(two_independent)
        assert qc.num_qubits == 5, f"Expected 5 qubits, got {qc.num_qubits}"


class TestAllocateGateInvariant:
    """Test that _allocate_gate enforces invariants on malformed operations."""

    def test_missing_qinit_for_array_element(self):
        """GateOperation on array element without QInitOperation raises AssertionError."""
        from qamomile.circuit.ir.operation.gate import (
            GateOperation,
            GateOperationType,
        )
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.transpiler.passes.emit_support import ResourceAllocator

        parent_array = ArrayValue(
            type=QubitType(),
            name="q_array",
            shape=(Value(type=UIntType(), name="dim0").with_const(3),),
        )

        idx_value = Value(type=UIntType(), name="idx").with_const(0)

        element_qubit = Value(
            type=QubitType(),
            name="q_element",
            parent_array=parent_array,
            element_indices=(idx_value,),
        )

        result_qubit = element_qubit.next_version()

        gate_op = GateOperation(
            gate_type=GateOperationType.H,
            operands=[element_qubit],
            results=[result_qubit],
        )

        allocator = ResourceAllocator()
        with pytest.raises(AssertionError, match="Array element key"):
            allocator.allocate([gate_op])


class TestBackwardsCompatibility:
    """Test that existing APIs still work."""

    def test_old_iqft_import_still_works(self):
        """Old import path for iqft/qft still works."""
        from qamomile.circuit.stdlib import iqft, qft

        assert callable(iqft)
        assert callable(qft)

    def test_stdlib_exports_classes(self):
        """stdlib exports both class and function APIs."""
        from qamomile.circuit.stdlib import IQFT, QFT, iqft, qft, qpe

        assert QFT is not None
        assert IQFT is not None
        assert qft is not None
        assert iqft is not None
        assert qpe is not None


class TestQPEPhaseExtraction:
    """Regression tests for QPE fallback phase operand resolution."""

    @staticmethod
    def _emit_pass(parameters: set[str] | None = None) -> Any:
        """Create the minimal emit-pass facade needed by phase extraction.

        Args:
            parameters (set[str] | None): Runtime parameter names that the
                resolver must keep symbolic. Defaults to ``None``.

        Returns:
            Any: Object exposing the ``_resolver`` attribute consumed by
                ``extract_phase_from_params``.
        """
        return type(
            "EmitPassStub",
            (),
            {"_resolver": ValueResolver(parameters=parameters)},
        )()

    @staticmethod
    def _qpe_op(phase_operand: Value) -> InvokeOperation:
        """Create a block-free QPE operation with one phase operand.

        Args:
            phase_operand (Value): Classical phase operand to place after the
                target qubit operand.

        Returns:
            InvokeOperation: QPE operation that exercises manual
                fallback phase extraction.
        """
        target = Value(type=QubitType(), name="target")
        ref = CallableRef(namespace="qamomile.stdlib", name="qpe")
        attrs = {
            "kind": "composite",
            "gate_type": CompositeGateType.QPE.name,
            "num_control_qubits": 0,
            "num_target_qubits": 1,
            "custom_name": "qpe",
        }
        return InvokeOperation(
            operands=[target, phase_operand],
            results=[],
            target=ref,
            attrs=attrs,
            definition=CallableDef(ref=ref, attrs=attrs),
        )

    @staticmethod
    def _typed_op(gate_type: CompositeGateType) -> InvokeOperation:
        """Create a minimal operation with the requested composite gate type.

        Args:
            gate_type (CompositeGateType): Composite gate type to assign.

        Returns:
            InvokeOperation: Block-free composite operation for direct
                helper validation tests.
        """
        target = Value(type=QubitType(), name="target")
        phase = Value(type=FloatType(), name="phase").with_const(0.25)
        ref = CallableRef(namespace="qamomile.stdlib", name=gate_type.value)
        attrs = {
            "kind": "composite",
            "gate_type": gate_type.name,
            "num_control_qubits": 0,
            "num_target_qubits": 1,
            "custom_name": gate_type.value,
        }
        return InvokeOperation(
            operands=[target, phase],
            results=[],
            target=ref,
            attrs=attrs,
            definition=CallableDef(ref=ref, attrs=attrs),
        )

    @staticmethod
    def _qpe_op_with_params(params: list[Value]) -> InvokeOperation:
        """Create a block-free QPE operation with explicit parameter operands.

        Args:
            params (list[Value]): Classical parameter operands to append after
                the target qubit operand.

        Returns:
            InvokeOperation: QPE operation with the supplied parameter
                operands.
        """
        target = Value(type=QubitType(), name="target")
        ref = CallableRef(namespace="qamomile.stdlib", name="qpe")
        attrs = {
            "kind": "composite",
            "gate_type": CompositeGateType.QPE.name,
            "num_control_qubits": 0,
            "num_target_qubits": 1,
            "custom_name": "qpe",
        }
        return InvokeOperation(
            operands=[target, *params],
            results=[],
            target=ref,
            attrs=attrs,
            definition=CallableDef(ref=ref, attrs=attrs),
        )

    def test_extract_phase_from_bound_array_element(self):
        """QPE fallback phase extraction reads a bound array element."""
        parent = ArrayValue(type=FloatType(), name="theta")
        index = Value(type=UIntType(), name="idx").with_const(1)
        phase_operand = Value(
            type=FloatType(),
            name="theta_elem",
            parent_array=parent,
            element_indices=(index,),
        )

        phase = extract_phase_from_params(
            self._emit_pass(),
            self._qpe_op(phase_operand),
            {"theta": np.array([0.125, 0.25])},
        )

        assert phase == pytest.approx(0.25)

    def test_extract_phase_from_const_array_element_metadata(self):
        """QPE fallback phase extraction reads frozen array literal metadata."""
        parent = ArrayValue(
            type=FloatType(),
            name="theta",
        ).with_array_runtime_metadata(const_array=[0.125, 0.375])
        index = Value(type=UIntType(), name="idx").with_const(1)
        phase_operand = Value(
            type=FloatType(),
            name="theta_elem",
            parent_array=parent,
            element_indices=(index,),
        )

        phase = extract_phase_from_params(
            self._emit_pass(),
            self._qpe_op(phase_operand),
            {},
        )

        assert phase == pytest.approx(0.375)

    def test_extract_phase_from_bound_vector_view_element(self):
        """QPE fallback phase extraction maps VectorView elements to root data."""
        root = ArrayValue(type=FloatType(), name="gammas")
        view = ArrayValue(
            type=FloatType(),
            name="gammas[slice]",
            slice_of=root,
            slice_start=Value(type=UIntType(), name="start").with_const(1),
            slice_step=Value(type=UIntType(), name="step").with_const(2),
        )
        index = Value(type=UIntType(), name="idx").with_const(1)
        phase_operand = Value(
            type=FloatType(),
            name="gamma_view_elem",
            parent_array=view,
            element_indices=(index,),
        )

        phase = extract_phase_from_params(
            self._emit_pass(),
            self._qpe_op(phase_operand),
            {"gammas": np.array([0.125, 0.25, 0.375, 0.5])},
        )

        assert phase == pytest.approx(0.5)

    def test_extract_phase_from_const_vector_view_element_metadata(self):
        """QPE fallback phase extraction maps VectorView elements to root literals."""
        root = ArrayValue(
            type=FloatType(),
            name="gammas",
        ).with_array_runtime_metadata(const_array=[0.125, 0.25, 0.375, 0.5])
        view = ArrayValue(
            type=FloatType(),
            name="gammas[slice]",
            slice_of=root,
            slice_start=Value(type=UIntType(), name="start").with_const(1),
            slice_step=Value(type=UIntType(), name="step").with_const(2),
        )
        index = Value(type=UIntType(), name="idx").with_const(1)
        phase_operand = Value(
            type=FloatType(),
            name="gamma_view_elem",
            parent_array=view,
            element_indices=(index,),
        )

        phase = extract_phase_from_params(
            self._emit_pass(),
            self._qpe_op(phase_operand),
            {},
        )

        assert phase == pytest.approx(0.5)

    def test_extract_phase_leaves_symbolic_slice_bound_unresolved(self):
        """QPE fallback phase extraction preserves unresolved VectorView bounds."""
        root = ArrayValue(type=FloatType(), name="gammas")
        view = ArrayValue(
            type=FloatType(),
            name="gammas[slice]",
            slice_of=root,
            slice_start=Value(type=UIntType(), name="start"),
            slice_step=Value(type=UIntType(), name="step").with_const(1),
        )
        index = Value(type=UIntType(), name="idx").with_const(0)
        phase_operand = Value(
            type=FloatType(),
            name="gamma_view_elem",
            parent_array=view,
            element_indices=(index,),
        )

        phase = extract_phase_from_params(
            self._emit_pass(),
            self._qpe_op(phase_operand),
            {"gammas": np.array([0.125, 0.25])},
        )

        assert phase is None

    def test_extract_phase_leaves_runtime_parameter_array_unresolved(self):
        """QPE fallback phase extraction preserves runtime parameter arrays."""
        parent = ArrayValue(
            type=FloatType(),
            name="theta",
        ).with_array_runtime_metadata(const_array=[0.125, 0.375])
        index = Value(type=UIntType(), name="idx").with_const(1)
        phase_operand = Value(
            type=FloatType(),
            name="theta_elem",
            parent_array=parent,
            element_indices=(index,),
        )

        phase = extract_phase_from_params(
            self._emit_pass(parameters={"theta"}),
            self._qpe_op(phase_operand),
            {"theta": np.array([0.125, 0.25])},
        )

        assert phase is None

    def test_extract_phase_leaves_symbolic_array_index_unresolved(self):
        """QPE fallback phase extraction preserves unresolved symbolic indices."""
        parent = ArrayValue(
            type=FloatType(),
            name="theta",
        ).with_array_runtime_metadata(const_array=[0.125, 0.375])
        index = Value(type=UIntType(), name="idx")
        phase_operand = Value(
            type=FloatType(),
            name="theta_elem",
            parent_array=parent,
            element_indices=(index,),
        )

        phase = extract_phase_from_params(
            self._emit_pass(),
            self._qpe_op(phase_operand),
            {},
        )

        assert phase is None

    def test_extract_phase_rejects_multiple_concrete_parameters(self):
        """QPE fallback phase extraction rejects ambiguous numeric params."""
        first = Value(type=FloatType(), name="first").with_const(0.125)
        second = Value(type=FloatType(), name="second").with_const(0.25)

        with pytest.raises(EmitError, match="multiple numeric parameters"):
            extract_phase_from_params(
                self._emit_pass(),
                self._qpe_op_with_params([first, second]),
                {},
            )

    def test_extract_phase_rejects_non_qpe_operations(self):
        """QPE phase extraction rejects mismatched composite gate types."""
        with pytest.raises(EmitError, match="only supports QPE"):
            extract_phase_from_params(
                self._emit_pass(), self._typed_op(CompositeGateType.QFT), {}
            )

    def test_extract_phase_rejects_non_composite_operations(self):
        """QPE phase extraction rejects non-composite operations."""
        result = Value(type=QubitType(), name="q")
        non_composite_op: Any = QInitOperation(operands=[], results=[result])

        with pytest.raises(EmitError, match="only supports QPE"):
            extract_phase_from_params(self._emit_pass(), non_composite_op, {})

    def test_emit_qpe_manual_rejects_non_qpe_operations(self):
        """Manual QPE emission rejects mismatched composite gate types."""
        with pytest.raises(EmitError, match="only supports QPE"):
            emit_qpe_manual(
                self._emit_pass(),
                None,
                self._typed_op(CompositeGateType.QFT),
                [],
                {},
            )

    def test_emit_qft_with_strategy_rejects_non_qft_operations(self):
        """QFT strategy emission rejects mismatched composite gate types."""
        with pytest.raises(EmitError, match="only supports QFT"):
            emit_qft_with_strategy(
                self._emit_pass(),
                None,
                self._typed_op(CompositeGateType.IQFT),
                [],
            )

    def test_emit_iqft_with_strategy_rejects_non_iqft_operations(self):
        """IQFT strategy emission rejects mismatched composite gate types."""
        with pytest.raises(EmitError, match="only supports IQFT"):
            emit_iqft_with_strategy(
                self._emit_pass(),
                None,
                self._typed_op(CompositeGateType.QFT),
                [],
            )


class TestVectorQubitCompositeDecorator:
    """Verify vector qkernels can be wrapped as boxed composites."""

    def test_vector_qubit_param_emits_composite_invoke(self):
        """Decorating a Vector[Qubit] qkernel preserves it as a box."""
        import qamomile.circuit as qmc

        model = qmc.FixedResourceModel(gates=qmc.GateResources(total=1))

        @qmc.composite_gate(name="encode", resource_model=model)
        @qmc.qkernel
        def encode(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            q[0] = qmc.h(q[0])
            return q

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(2, "q")
            q = encode(q)
            return q

        block = circuit.build()
        invokes = _invoke_ops(block.operations)
        assert len(invokes) == 1
        op = invokes[0]
        assert op.attrs["kind"] == "composite"
        assert op.target.namespace == "user.composite"
        assert op.target.name == "encode"
        assert op.default_policy is CallPolicy.PRESERVE_BOX
        assert op.num_target_qubits == 2
        assert op.body is not None
        assert op.definition is not None
        assert op.definition.resource_models[0].model is model

    def test_vector_qubit_direct_decorator_form_uses_qkernel_name(self):
        """Direct @composite_gate form supports Vector[Qubit] qkernels."""
        import qamomile.circuit as qmc

        @qmc.composite_gate
        @qmc.qkernel
        def encode(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            return q

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(3, "q")
            q = encode(q)
            return q

        op = _invoke_ops(circuit.build().operations)[0]
        assert op.target.name == "encode"
        assert op.num_target_qubits == 3

    def test_fixed_arity_qubit_params_still_work(self):
        """Regression: fixed-arity Qubit-only signatures still decorate cleanly."""
        import qamomile.circuit as qmc

        @qmc.composite_gate(name="entangle")
        @qmc.qkernel
        def entangle(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            q0, q1 = qmc.cx(q0, q1)
            return q0, q1

        assert entangle.num_target_qubits == 2
        assert entangle.custom_name == "entangle"
