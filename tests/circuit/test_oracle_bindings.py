"""Tests for per-call opaque oracle implementation bindings."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.callable import CallTransform, InvokeOperation
from qamomile.circuit.ir.operation.control_flow import HasNestedOps
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.types import UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.serialization import deserialize, serialize
from qamomile.circuit.transpiler import TranspilerConfig
from qamomile.circuit.transpiler.circuit_ir.lowering import CircuitLoweringPass
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.circuit.transpiler.oracle_bindings import (
    _apply_oracle_bindings as apply_oracle_bindings,
)
from qamomile.circuit.transpiler.passes.emit_support.controlled_emission import (
    _controlled_body_batch_weight,
)
from qamomile.circuit.transpiler.passes.substitution import (
    SubstitutionConfig,
    SubstitutionPass,
    SubstitutionRule,
)

_ORACLE = qmc.opaque(
    "late_bound_oracle",
    num_qubits=1,
    cost=qmc.ResourceEstimate(gates=qmc.GateResources(total=7)),
)
_VECTOR_ORACLE = qmc.opaque("late_bound_vector_oracle", num_qubits=2)
_SIGNATURED_VECTOR_ORACLE = qmc.opaque(
    "late_bound_signature_vector_oracle",
    signature=qmc.CallableSignature(
        inputs=[qmc.Vector[qmc.Qubit]],
        outputs=[qmc.Vector[qmc.Qubit]],
    ),
)
_SIGNATURED_CONTROLLED_ORACLE = qmc.opaque(
    "late_bound_signature_controlled_oracle",
    num_control_qubits=1,
    signature=qmc.CallableSignature(
        inputs=[qmc.Qubit],
        outputs=[qmc.Qubit],
    ),
)
_LOG_WIDTH_VECTOR_ORACLE = qmc.opaque(
    "late_bound_log_width_vector_oracle",
    num_qubits=8,
)
_SECOND_ORACLE = qmc.opaque("second_late_bound_oracle", num_qubits=1)


@qmc.qkernel
def _x_implementation(q: qmc.Qubit) -> qmc.Qubit:
    """Implement the test oracle with an X gate."""
    return qmc.x(q)


@qmc.qkernel
def _z_implementation(q: qmc.Qubit) -> qmc.Qubit:
    """Implement the test oracle with a Z gate."""
    return qmc.z(q)


@qmc.qkernel
def _chained_implementation(q: qmc.Qubit) -> qmc.Qubit:
    """Implement one oracle by invoking a second late-bound oracle."""
    (q,) = _SECOND_ORACLE(q)
    return q


@qmc.qkernel
def _recursive_implementation(q: qmc.Qubit) -> qmc.Qubit:
    """Invoke the same oracle that this body would implement."""
    (q,) = _ORACLE(q)
    return q


@qmc.qkernel
def _reset_implementation(q: qmc.Qubit) -> qmc.Qubit:
    """Provide an intentionally non-unitary oracle implementation."""
    return qmc.reset(q)


@qmc.qkernel
def _static_implementation(
    q: qmc.Qubit,
    encoding: qmc.LCUBlockEncoding,
) -> qmc.Qubit:
    """Use an LCU descriptor through a compile-time static binding."""
    return qmc.rx(q, encoding.normalization)


@qmc.qkernel
def _incompatible_implementation(
    q0: qmc.Qubit,
    q1: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Provide an intentionally incompatible two-qubit implementation."""
    return qmc.cx(q0, q1)


@qmc.qkernel
def _vector_implementation(
    qubits: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Flip every qubit in a vector implementation."""
    for index in qmc.range(qubits.shape[0]):
        qubits[index] = qmc.x(qubits[index])
    return qubits


@qmc.qkernel
def _log_width_vector_implementation(
    qubits: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Flip ``ceil(log2(length))`` qubits in a vector implementation."""
    width = qmc.ceil(qmc.log2(qubits.shape[0]))
    for index in qmc.range(width):
        qubits[index] = qmc.x(qubits[index])
    return qubits


@qmc.qkernel
def _log_width_vector_helper(
    qubits: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Call the log-width vector oracle from a controllable helper."""
    return _LOG_WIDTH_VECTOR_ORACLE(qubits)


@qmc.qkernel
def _symbolic_vector_call(
    qubits: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Call the vector oracle at a symbolic input extent."""
    return _VECTOR_ORACLE(qubits)


@qmc.qkernel
def _oracle_helper(q: qmc.Qubit) -> qmc.Qubit:
    """Call the opaque oracle from a nested qkernel body."""
    (q,) = _ORACLE(q)
    return q


@qmc.qkernel
def _recursive_oracle_helper(depth: qmc.UInt, q: qmc.Qubit) -> qmc.Qubit:
    """Reach the opaque oracle through a terminating self-recursive helper."""
    if depth == 0:
        (q,) = _ORACLE(q)
    else:
        q = _recursive_oracle_helper(depth - 1, q)
    return q


@qmc.qkernel
def _recursive_sample(depth: qmc.UInt) -> qmc.Bit:
    """Measure an oracle reached through a self-recursive definition."""
    q = _recursive_oracle_helper(depth, qmc.qubit("q"))
    return qmc.measure(q)


@qmc.qkernel
def _direct_sample() -> qmc.Bit:
    """Measure one directly bound oracle call."""
    q = qmc.qubit("q")
    (q,) = _ORACLE(q)
    return qmc.measure(q)


@qmc.qkernel
def _nested_sample() -> qmc.Bit:
    """Measure an oracle reached through a qkernel helper."""
    q = _oracle_helper(qmc.qubit("q"))
    return qmc.measure(q)


@qmc.qkernel
def _loop_sample() -> qmc.Bit:
    """Invoke the late-bound oracle inside structured control flow."""
    q = qmc.qubit("q")
    for _ in qmc.range(1):
        (q,) = _ORACLE(q)
    return qmc.measure(q)


@qmc.qkernel
def _controlled_sample() -> qmc.Vector[qmc.Bit]:
    """Measure a controlled late-bound oracle call."""
    qubits = qmc.qubit_array(2, "qubits")
    qubits[0] = qmc.x(qubits[0])
    qubits[0], qubits[1] = qmc.control(_ORACLE)(qubits[0], qubits[1])
    return qmc.measure(qubits)


@qmc.qkernel
def _signatured_controlled_sample() -> qmc.Vector[qmc.Bit]:
    """Measure a controlled oracle carrying an explicit target signature."""
    qubits = qmc.qubit_array(2, "qubits")
    qubits[0], qubits[1] = _SIGNATURED_CONTROLLED_ORACLE(
        qubits[1],
        controls=(qubits[0],),
    )
    return qmc.measure(qubits)


@qmc.qkernel
def _controlled_helper_sample() -> qmc.Vector[qmc.Bit]:
    """Measure control of a helper whose body contains the oracle."""
    qubits = qmc.qubit_array(2, "qubits")
    qubits[0] = qmc.x(qubits[0])
    qubits[0], qubits[1] = qmc.control(_oracle_helper)(
        qubits[0],
        qubits[1],
    )
    return qmc.measure(qubits)


@qmc.qkernel
def _select_sample() -> qmc.Vector[qmc.Bit]:
    """Measure an oracle reached through SELECT case bodies."""
    qubits = qmc.qubit_array(2, "qubits")
    qubits[0], qubits[1] = qmc.select([_oracle_helper, _oracle_helper])(
        qubits[0],
        qubits[1],
    )
    return qmc.measure(qubits)


@qmc.qkernel
def _vector_sample() -> qmc.Vector[qmc.Bit]:
    """Measure a shape-dependent vector oracle implementation."""
    qubits = _VECTOR_ORACLE(qmc.qubit_array(2, "qubits"))
    return qmc.measure(qubits)


@qmc.qkernel
def _signature_vector_sample() -> qmc.Vector[qmc.Bit]:
    """Measure an oracle declared with an explicit vector signature."""
    qubits = _SIGNATURED_VECTOR_ORACLE(qmc.qubit_array(2, "qubits"))
    return qmc.measure(qubits)


@qmc.qkernel
def _log_width_vector_sample() -> qmc.Vector[qmc.Bit]:
    """Measure a fixed-width oracle whose loop uses unary math."""
    qubits = _LOG_WIDTH_VECTOR_ORACLE(qmc.qubit_array(8, "qubits"))
    return qmc.measure(qubits)


@qmc.qkernel
def _controlled_log_width_vector_sample() -> qmc.Vector[qmc.Bit]:
    """Measure unary math inside an oracle under an enclosing control."""
    qubits = qmc.qubit_array(9, "qubits")
    qubits[0] = qmc.x(qubits[0])
    controlled = qmc.control(_log_width_vector_helper)
    qubits[0], targets = controlled(qubits[0], qubits[1:9])
    qubits[1:9] = targets
    return qmc.measure(qubits)


def _invoke_named(block: Any, name: str) -> InvokeOperation:
    """Return a named invocation from a block.

    Args:
        block (Any): Block-like object with an ``operations`` sequence.
        name (str): Callable definition name to find.

    Returns:
        InvokeOperation: Matching invocation.

    Raises:
        StopIteration: If the block has no matching invocation.
    """
    return next(
        operation
        for operation in block.operations
        if isinstance(operation, InvokeOperation) and operation.target.name == name
    )


def _observed_bits(result: Any) -> set[tuple[int, ...]]:
    """Normalize backend-independent sample results.

    Args:
        result (Any): Qamomile sample result.

    Returns:
        set[tuple[int, ...]]: Distinct observed bit strings.
    """
    observed: set[tuple[int, ...]] = set()
    for bits, _ in result.results:
        if isinstance(bits, tuple):
            observed.add(tuple(int(bit) for bit in bits))
        else:
            observed.add((int(bits),))
    return observed


def test_binding_preserves_identity_and_clears_resource_estimate() -> None:
    """A direct binding keeps the opaque identity and removes its cost."""
    source = _direct_sample.block
    transformed = apply_oracle_bindings(
        source,
        {"late_bound_oracle": _x_implementation},
    )

    invocation = _invoke_named(transformed, "late_bound_oracle")
    assert invocation.target.namespace == "user.oracle"
    assert invocation.custom_name == "late_bound_oracle"
    assert invocation.body is not None
    assert invocation.body.name == "_x_implementation"
    assert invocation.definition is not None
    assert invocation.definition.opaque_cost is None
    assert _invoke_named(source, "late_bound_oracle").body is None


def test_binding_uses_exact_target_name_not_display_name() -> None:
    """Binding keys match ``target.name`` rather than a display alias."""
    source = _direct_sample.block
    invocation = _invoke_named(source, "late_bound_oracle")
    aliased = dataclasses.replace(
        source,
        operations=[
            dataclasses.replace(
                invocation,
                attrs={**invocation.attrs, "custom_name": "display_alias"},
            )
        ],
    )

    with pytest.raises(ValueError, match="display_alias"):
        apply_oracle_bindings(
            aliased,
            {"display_alias": _x_implementation},
        )

    transformed = apply_oracle_bindings(
        aliased,
        {"late_bound_oracle": _x_implementation},
    )
    bound = _invoke_named(transformed, "late_bound_oracle")
    assert bound.custom_name == "display_alias"
    assert bound.body is not None


def test_binding_accepts_direct_and_controlled_calls() -> None:
    """The same direct body supplies direct and controlled opaque calls."""
    direct = apply_oracle_bindings(
        _direct_sample.block,
        {"late_bound_oracle": _x_implementation},
    )
    controlled = apply_oracle_bindings(
        _controlled_sample.block,
        {"late_bound_oracle": _x_implementation},
    )

    direct_call = _invoke_named(direct, "late_bound_oracle")
    controlled_call = _invoke_named(controlled, "late_bound_oracle")
    assert direct_call.transform is CallTransform.DIRECT
    assert controlled_call.transform is CallTransform.CONTROLLED
    assert direct_call.body is not None
    assert controlled_call.body is not None
    assert direct_call.body.name == controlled_call.body.name == "_x_implementation"


def test_binding_accepts_controlled_explicit_signature() -> None:
    """A controlled oracle canonicalizes controls around its target signature."""
    source_call = _invoke_named(
        _signatured_controlled_sample.block,
        "late_bound_signature_controlled_oracle",
    )
    assert source_call.definition is not None
    assert source_call.definition.signature is not None
    assert len(source_call.definition.signature.operands) == 2
    assert len(source_call.definition.signature.results) == 2
    assert serialize(_signatured_controlled_sample)

    transformed = apply_oracle_bindings(
        _signatured_controlled_sample.block,
        {"late_bound_signature_controlled_oracle": _x_implementation},
    )
    bound = _invoke_named(
        transformed,
        "late_bound_signature_controlled_oracle",
    )
    assert bound.transform is CallTransform.CONTROLLED
    assert bound.body is not None
    assert bound.body.name == "_x_implementation"

    with pytest.raises(ValidationError, match="Input count mismatch"):
        apply_oracle_bindings(
            _signatured_controlled_sample.block,
            {"late_bound_signature_controlled_oracle": (_incompatible_implementation)},
        )


def test_binding_accepts_control_wrapper_for_explicit_signature() -> None:
    """The control wrapper preserves an explicit target signature."""
    oracle = qmc.opaque(
        "wrapped_signature_controlled_oracle",
        signature=qmc.CallableSignature(
            inputs=[qmc.Qubit],
            outputs=[qmc.Qubit],
        ),
    )
    controlled = qmc.control(oracle)

    @qmc.qkernel
    def sample() -> qmc.Vector[qmc.Bit]:
        """Measure an explicitly signed oracle through ``qmc.control``."""
        qubits = qmc.qubit_array(2, "qubits")
        qubits[0], qubits[1] = controlled(qubits[0], qubits[1])
        return qmc.measure(qubits)

    assert serialize(sample)
    transformed = apply_oracle_bindings(
        sample.block,
        {"wrapped_signature_controlled_oracle": _x_implementation},
    )
    bound = _invoke_named(
        transformed,
        "wrapped_signature_controlled_oracle",
    )
    assert bound.transform is CallTransform.CONTROLLED
    assert bound.body is not None
    assert bound.body.name == "_x_implementation"


def test_binding_rejects_bodyful_callable() -> None:
    """Bindings accept resource-only opaque definitions, not qkernels."""
    with pytest.raises(ValueError, match="resource-only opaque"):
        apply_oracle_bindings(
            _nested_sample.block,
            {"_oracle_helper": _x_implementation},
        )


def test_binding_rejects_inverse_transform() -> None:
    """Generated inverse callables remain outside the binding contract."""
    source = _direct_sample.block
    invocation = _invoke_named(source, "late_bound_oracle")
    inverse = dataclasses.replace(
        source,
        operations=[dataclasses.replace(invocation, transform=CallTransform.INVERSE)],
    )

    with pytest.raises(ValueError, match="direct or controlled"):
        apply_oracle_bindings(
            inverse,
            {"late_bound_oracle": _x_implementation},
        )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(_direct_sample.block, id="direct"),
        pytest.param(_controlled_sample.block, id="controlled"),
        pytest.param(_controlled_helper_sample.block, id="controlled-block"),
        pytest.param(_select_sample.block, id="select"),
        pytest.param(_nested_sample.block, id="nested"),
    ],
)
def test_binding_rejects_nonunitary_implementation(source: qmc.Block) -> None:
    """Opaque implementations remain unitary in every reachable context."""
    with pytest.raises(
        ValidationError,
        match=r"non-unitary kernel effects \[RESET\].*must be unitary",
    ):
        apply_oracle_bindings(
            source,
            {"late_bound_oracle": _reset_implementation},
        )


def test_binding_reaches_nested_callable_definition() -> None:
    """Binding descends into an invoked qkernel definition."""
    transformed = apply_oracle_bindings(
        _nested_sample.block,
        {"late_bound_oracle": _x_implementation},
    )

    helper = _invoke_named(transformed, "_oracle_helper")
    assert helper.body is not None
    nested_oracle = _invoke_named(helper.body, "late_bound_oracle")
    assert nested_oracle.body is not None
    assert nested_oracle.body.name == "_x_implementation"


def test_binding_reaches_supplied_implementation_body() -> None:
    """One binding can resolve another opaque inside its implementation."""
    transformed = apply_oracle_bindings(
        _direct_sample.block,
        {
            "late_bound_oracle": _chained_implementation,
            "second_late_bound_oracle": _x_implementation,
        },
    )

    first = _invoke_named(transformed, "late_bound_oracle")
    assert first.body is not None
    second = _invoke_named(first.body, "second_late_bound_oracle")
    assert second.body is not None
    assert second.body.name == "_x_implementation"


def test_binding_rejects_recursive_implementation_cycle() -> None:
    """A supplied implementation cannot recursively bind itself."""
    with pytest.raises(
        ValueError,
        match="late_bound_oracle -> late_bound_oracle",
    ):
        apply_oracle_bindings(
            _direct_sample.block,
            {"late_bound_oracle": _recursive_implementation},
        )


def test_binding_rejects_cycle_through_active_source_body() -> None:
    """An implementation cannot reuse its currently enclosing source body."""
    with pytest.raises(ValueError, match="active source"):
        apply_oracle_bindings(
            _nested_sample.block,
            {"late_bound_oracle": _oracle_helper},
        )


def test_binding_rejects_cycle_through_configured_replacement(
    qiskit_transpiler: Any,
) -> None:
    """An implementation cannot reuse its enclosing Configure replacement."""
    qiskit_transpiler.set_config(
        TranspilerConfig(
            substitutions=SubstitutionConfig(
                rules=[
                    SubstitutionRule(
                        source_name="_oracle_helper",
                        target=_recursive_implementation,
                    )
                ]
            )
        )
    )

    with pytest.raises(ValueError, match="configured replacement"):
        qiskit_transpiler.substitute(
            _nested_sample.block,
            oracle_bindings={
                "late_bound_oracle": _recursive_implementation,
            },
        )


def test_binding_allows_same_block_with_different_rule_scope() -> None:
    """Rule-enabled and rule-disabled clones do not form a false cycle."""
    substitution = SubstitutionPass(
        SubstitutionConfig(
            rules=[
                SubstitutionRule(
                    source_name="second_late_bound_oracle",
                    target=_recursive_implementation,
                )
            ]
        ),
        oracle_bindings={"late_bound_oracle": _chained_implementation.block},
    )

    transformed = substitution.run(_chained_implementation.block)
    configured = _invoke_named(transformed, "_recursive_implementation")
    assert configured.body is not None
    bound = _invoke_named(configured.body, "late_bound_oracle")
    assert bound.body is not None
    unconfigured = _invoke_named(bound.body, "second_late_bound_oracle")
    assert unconfigured.body is None


def test_binding_reaches_structured_control_flow() -> None:
    """Binding descends through structured control-flow regions."""
    transformed = apply_oracle_bindings(
        _loop_sample.block,
        {"late_bound_oracle": _x_implementation},
    )
    region = next(
        operation
        for operation in transformed.operations
        if isinstance(operation, HasNestedOps)
    )
    nested_oracle = next(
        operation
        for operations in region.nested_op_lists()
        for operation in operations
        if isinstance(operation, InvokeOperation)
        and operation.target.name == "late_bound_oracle"
    )

    assert nested_oracle.body is not None
    assert nested_oracle.body.name == "_x_implementation"


def test_binding_reaches_select_case_blocks() -> None:
    """Binding descends through every SELECT implementation body."""
    transformed = apply_oracle_bindings(
        _select_sample.block,
        {"late_bound_oracle": _x_implementation},
    )
    selection = next(
        operation
        for operation in transformed.operations
        if isinstance(operation, SelectOperation)
    )

    for case_block in selection.case_blocks:
        nested_oracle = _invoke_named(case_block, "late_bound_oracle")
        assert nested_oracle.body is not None
        assert nested_oracle.body.name == "_x_implementation"


def test_binding_handles_self_recursive_source_graph(qiskit_transpiler: Any) -> None:
    """Graph traversal terminates while binding a recursive helper's leaf."""
    executable = qiskit_transpiler.transpile(
        _recursive_sample,
        bindings={"depth": 2},
        oracle_bindings={"late_bound_oracle": _x_implementation},
    )
    result = executable.sample(qiskit_transpiler.executor(), shots=16).result()

    assert _observed_bits(result) == {(1,)}


def test_binding_validates_vector_signature_and_shape() -> None:
    """Vector bindings preserve compatible shapes and reject fixed mismatch."""
    transformed = apply_oracle_bindings(
        _vector_sample.block,
        {"late_bound_vector_oracle": _vector_implementation},
    )
    bound = _invoke_named(transformed, "late_bound_vector_oracle")
    assert bound.body is not None
    assert bound.body.name == "_vector_implementation"

    target = _vector_implementation.block
    target_input = target.input_values[0]
    target_output = target.output_values[0]
    assert isinstance(target_input, ArrayValue)
    assert isinstance(target_output, ArrayValue)
    fixed_dim = Value(type=UIntType(), name="fixed_dim").with_const(3)
    fixed_target = dataclasses.replace(
        target,
        input_values=[dataclasses.replace(target_input, shape=(fixed_dim,))],
        output_values=[dataclasses.replace(target_output, shape=(fixed_dim,))],
    )

    with pytest.raises(
        ValidationError,
        match="source extent is symbolic but target extent is fixed",
    ):
        apply_oracle_bindings(
            _symbolic_vector_call.block,
            {"late_bound_vector_oracle": fixed_target},
        )


def test_binding_validates_bodyless_oracle_signature() -> None:
    """A resource-only oracle still validates its concrete call signature."""
    with pytest.raises(ValidationError, match="Input count mismatch"):
        apply_oracle_bindings(
            _direct_sample.block,
            {"late_bound_oracle": _incompatible_implementation},
        )


def test_binding_validates_explicit_callable_signature() -> None:
    """Explicit oracle signatures accept only shape-compatible bodies."""
    transformed = apply_oracle_bindings(
        _signature_vector_sample.block,
        {"late_bound_signature_vector_oracle": _vector_implementation},
    )
    bound = _invoke_named(transformed, "late_bound_signature_vector_oracle")
    assert bound.body is not None
    assert bound.body.name == "_vector_implementation"

    with pytest.raises(
        ValidationError,
        match=r"Input shape mismatch.*source is an array, target is a scalar",
    ):
        apply_oracle_bindings(
            _signature_vector_sample.block,
            {"late_bound_signature_vector_oracle": _x_implementation},
        )


def test_binding_rejects_declared_signature_callsite_mismatch() -> None:
    """Binding rejects an invocation that violates its declared signature."""
    source = _signature_vector_sample.block
    invocation = _invoke_named(source, "late_bound_signature_vector_oracle")
    assert invocation.definition is not None
    declared = qmc.CallableSignature(
        inputs=[qmc.Vector[qmc.Qubit]],
        outputs=[qmc.Vector[qmc.Bit]],
    ).to_ir_signature()
    malformed_invocation = dataclasses.replace(
        invocation,
        definition=dataclasses.replace(
            invocation.definition,
            signature=declared,
        ),
    )
    malformed = dataclasses.replace(
        source,
        operations=[
            malformed_invocation if operation is invocation else operation
            for operation in source.operations
        ],
    )

    with pytest.raises(
        ValidationError,
        match=r"declared callable signature disagrees.*Return type mismatch",
    ):
        apply_oracle_bindings(
            malformed,
            {"late_bound_signature_vector_oracle": _vector_implementation},
        )


@pytest.mark.parametrize(
    ("bindings", "error_type", "message"),
    [
        pytest.param(
            ["late_bound_oracle"],
            TypeError,
            "must be a mapping",
            id="non-mapping",
        ),
        pytest.param(
            {1: _x_implementation},
            TypeError,
            "keys must be strings",
            id="non-string-key",
        ),
        pytest.param(
            {"": _x_implementation},
            ValueError,
            "must not be empty",
            id="empty-key",
        ),
        pytest.param(
            {"late_bound_oracle": object()},
            TypeError,
            "values must be QKernel or Block",
            id="invalid-target",
        ),
    ],
)
def test_binding_validates_mapping_entries(
    bindings: Any,
    error_type: type[Exception],
    message: str,
) -> None:
    """Invalid mapping keys and values fail before traversal.

    Args:
        bindings (Any): Invalid mapping under test.
        error_type (type[Exception]): Expected exception class.
        message (str): Expected diagnostic fragment.
    """
    with pytest.raises(error_type, match=message):
        apply_oracle_bindings(_direct_sample.block, bindings)


@pytest.mark.parametrize(
    "implementation",
    [
        pytest.param(_static_implementation, id="qkernel"),
        pytest.param(_static_implementation.block, id="block"),
    ],
)
def test_binding_rejects_unresolved_static_implementation(
    implementation: Any,
) -> None:
    """Unresolved static templates fail with concrete recovery guidance."""
    with pytest.raises(
        ValueError,
        match=r"unresolved static bindings \['encoding'\].*implementation\.build",
    ):
        apply_oracle_bindings(
            _direct_sample.block,
            {"late_bound_oracle": implementation},
        )


def test_binding_accepts_specialized_static_block(qiskit_transpiler: Any) -> None:
    """A statically specialized implementation block compiles normally."""
    implementation = _static_implementation.build(
        encoding=qmc.identity_block_encoding(1)
    )

    assert implementation.static_bindings == ()
    executable = qiskit_transpiler.transpile(
        _direct_sample,
        oracle_bindings={"late_bound_oracle": implementation},
    )
    result = executable.sample(qiskit_transpiler.executor(), shots=16).result()

    assert sum(count for _, count in result.results) == 16


def test_unary_math_preserves_multi_control_batch_weight() -> None:
    """Unary loop bounds retain the shared-control batching threshold."""
    block = _log_width_vector_implementation.block
    vector_input = block.input_values[0]
    assert isinstance(vector_input, ArrayValue)

    weight = _controlled_body_batch_weight(
        CircuitLoweringPass(),
        block.operations,
        {vector_input.shape[0].uuid: 8},
    )

    assert weight == 2


def test_binding_rejects_unused_name() -> None:
    """A misspelled binding key is not silently ignored."""
    with pytest.raises(ValueError, match="missing_oracle"):
        apply_oracle_bindings(
            _direct_sample.block,
            {"missing_oracle": _x_implementation},
        )


def test_empty_binding_mapping_is_noop() -> None:
    """An empty binding mapping returns the original block unchanged."""
    source = _direct_sample.block
    assert apply_oracle_bindings(source, {}) is source


def test_bindings_are_nonmutating_and_per_call() -> None:
    """Independent binding calls neither leak state nor mutate their source."""
    source = _direct_sample.block
    x_bound = apply_oracle_bindings(
        source,
        {"late_bound_oracle": _x_implementation},
    )
    z_bound = apply_oracle_bindings(
        source,
        {"late_bound_oracle": _z_implementation},
    )

    source_call = _invoke_named(source, "late_bound_oracle")
    x_call = _invoke_named(x_bound, "late_bound_oracle")
    z_call = _invoke_named(z_bound, "late_bound_oracle")
    assert source_call.body is None
    assert x_call.body is not None
    assert z_call.body is not None
    assert x_call.body.name == "_x_implementation"
    assert z_call.body.name == "_z_implementation"
    assert x_call.definition is not z_call.definition


def test_binding_body_wins_and_configured_strategy_composes(
    qiskit_transpiler: Any,
) -> None:
    """Per-call body precedence retains a configured lowering strategy."""
    config = TranspilerConfig(
        substitutions=SubstitutionConfig(
            rules=[
                SubstitutionRule(
                    source_name="late_bound_oracle",
                    target=_z_implementation,
                    strategy="configured_strategy",
                )
            ]
        )
    )
    qiskit_transpiler.set_config(config)

    transformed = qiskit_transpiler.substitute(
        _direct_sample.block,
        oracle_bindings={"late_bound_oracle": _x_implementation},
    )

    bound = _invoke_named(transformed, "late_bound_oracle")
    assert bound.body is not None
    assert bound.body.name == "_x_implementation"
    assert bound.strategy_name == "configured_strategy"
    assert config.substitutions.rules[0].target is _z_implementation


def test_binding_reaches_configured_replacement_body(
    qiskit_transpiler: Any,
) -> None:
    """An enclosing Configure replacement does not hide nested bindings."""
    qiskit_transpiler.set_config(
        TranspilerConfig(
            substitutions=SubstitutionConfig(
                rules=[
                    SubstitutionRule(
                        source_name="_oracle_helper",
                        target=_oracle_helper,
                    )
                ]
            )
        )
    )

    transformed = qiskit_transpiler.substitute(
        _nested_sample.block,
        oracle_bindings={"late_bound_oracle": _x_implementation},
    )

    helper = _invoke_named(transformed, "_oracle_helper")
    assert helper.body is not None
    nested = _invoke_named(helper.body, "late_bound_oracle")
    assert nested.body is not None
    assert nested.body.name == "_x_implementation"


def test_public_substitute_accepts_oracle_bindings(qiskit_transpiler: Any) -> None:
    """The stepwise substitution API accepts per-call bindings."""
    transformed = qiskit_transpiler.substitute(
        _direct_sample.block,
        oracle_bindings={"late_bound_oracle": _x_implementation},
    )

    assert _invoke_named(transformed, "late_bound_oracle").body is not None


def test_public_to_circuit_accepts_oracle_bindings(qiskit_transpiler: Any) -> None:
    """Circuit extraction forwards per-call bindings."""
    circuit = qiskit_transpiler.to_circuit(
        _direct_sample,
        oracle_bindings={"late_bound_oracle": _x_implementation},
    )

    assert circuit.num_qubits == 1


def test_binding_survives_qkernel_serialization(qiskit_transpiler: Any) -> None:
    """A serialized resource-only skeleton remains late-bindable."""
    restored = deserialize(serialize(_direct_sample))
    executable = qiskit_transpiler.transpile(
        restored,
        oracle_bindings={"late_bound_oracle": _x_implementation},
    )
    result = executable.sample(qiskit_transpiler.executor(), shots=16).result()

    assert _observed_bits(result) == {(1,)}


def test_substitution_pass_resets_state_between_runs() -> None:
    """One substitution pass instance can safely process multiple blocks."""
    substitution = SubstitutionPass(
        SubstitutionConfig(),
        oracle_bindings={"late_bound_oracle": _x_implementation.block},
    )

    direct = substitution.run(_direct_sample.block)
    assert _invoke_named(direct, "late_bound_oracle").body is not None

    with pytest.raises(ValueError, match="late_bound_oracle"):
        substitution.run(_x_implementation.block)

    nested = substitution.run(_nested_sample.block)
    helper = _invoke_named(nested, "_oracle_helper")
    assert helper.body is not None
    assert _invoke_named(helper.body, "late_bound_oracle").body is not None


@pytest.mark.parametrize(
    ("kernel", "bindings", "expected"),
    [
        pytest.param(
            _direct_sample,
            {"late_bound_oracle": _x_implementation},
            (1,),
            id="direct",
        ),
        pytest.param(
            _nested_sample,
            {"late_bound_oracle": _x_implementation},
            (1,),
            id="nested",
        ),
        pytest.param(
            _controlled_sample,
            {"late_bound_oracle": _x_implementation},
            (1, 1),
            id="controlled",
        ),
        pytest.param(
            _controlled_helper_sample,
            {"late_bound_oracle": _x_implementation},
            (1, 1),
            id="controlled-helper",
        ),
        pytest.param(
            _vector_sample,
            {"late_bound_vector_oracle": _vector_implementation},
            (1, 1),
            id="vector",
        ),
        pytest.param(
            _log_width_vector_sample,
            {"late_bound_log_width_vector_oracle": _log_width_vector_implementation},
            (1, 1, 1, 0, 0, 0, 0, 0),
            id="log-width-vector",
        ),
        pytest.param(
            _controlled_log_width_vector_sample,
            {"late_bound_log_width_vector_oracle": _log_width_vector_implementation},
            (1, 1, 1, 1, 0, 0, 0, 0, 0),
            id="controlled-log-width-vector",
        ),
    ],
)
def test_oracle_binding_executes_on_every_backend(
    sdk_transpiler: Any,
    kernel: Any,
    bindings: dict[str, Any],
    expected: tuple[int, ...],
) -> None:
    """Bound direct, nested, controlled, and vector oracles execute.

    Args:
        sdk_transpiler (Any): Supported SDK transpiler fixture.
        kernel (Any): Kernel containing an opaque invocation.
        bindings (dict[str, Any]): Per-call oracle implementations.
        expected (tuple[int, ...]): Deterministic observed bit string.
    """
    executable = sdk_transpiler.transpiler.transpile(
        kernel,
        oracle_bindings=bindings,
    )
    result = executable.sample(
        sdk_transpiler.transpiler.executor(),
        shots=16,
    ).result()

    assert _observed_bits(result) == {expected}, sdk_transpiler.backend_name
