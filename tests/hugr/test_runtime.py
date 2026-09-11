"""Execution-ABI tests for direct HUGR callers and tagged output restoration."""

from __future__ import annotations

import copy
import dataclasses
import math

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.types import BitType, FloatType, UIntType
from qamomile.circuit.ir.value import ArrayValue, DictValue, TupleValue, Value
from qamomile.circuit.transpiler import (
    CompilationMetadata,
    CompiledProgram,
    QamomileCompiler,
)
from qamomile.circuit.transpiler.segments import ProgramABI

pytest.importorskip("hugr")
pytest.importorskip("tket_exts")

from hugr import ops, tys
from hugr.build import Module
from hugr.package import Package
from hugr.std.float import FLOAT_T
from hugr.std.int import int_t

from qamomile.hugr import HugrTranspiler
from qamomile.hugr._runtime import prepare_execution, validate_runtime_bindings
from qamomile.hugr._shapes import resolve_parameter_shapes
from qamomile.hugr.lowerer import HugrTarget

pytestmark = pytest.mark.hugr


@qmc.qkernel
def _runtime_bell(theta: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
    """Return a pair of correlated bits controlled by one runtime angle.

    Args:
        theta (qmc.Float): Rotation angle in radians.

    Returns:
        tuple[qmc.Bit, qmc.Bit]: Correlated measurement results.
    """
    left = qmc.qubit("left")
    right = qmc.qubit("right")
    left = qmc.ry(left, theta)
    left, right = qmc.cx(left, right)
    return qmc.measure(left), qmc.measure(right)


@qmc.qkernel
def _runtime_bit_order() -> qmc.Vector[qmc.Bit]:
    """Return an asymmetric bit pattern in logical qubit-index order.

    Returns:
        qmc.Vector[qmc.Bit]: Three measured bits beginning with one.
    """
    register = qmc.qubit_array(3, "register")
    register[0] = qmc.x(register[0])
    return qmc.measure(register)


@qmc.qkernel
def _runtime_matrix(
    values: qmc.Matrix[qmc.Float],
) -> tuple[qmc.Vector[qmc.Bit], qmc.Matrix[qmc.Float]]:
    """Use runtime matrix elements as rotation angles and return the original matrix.

    Args:
        values (qmc.Matrix[qmc.Float]): Matrix with two rotation-angle rows.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.Matrix[qmc.Float]]: Measurements and input matrix.
    """
    register = qmc.qubit_array(2, "register")
    for index in qmc.range(values.shape[0]):
        register[index] = qmc.ry(register[index], values[index, 1])
    return qmc.measure(register), values


@qmc.qkernel
def _runtime_computed_shape(
    values: qmc.Vector[qmc.Float],
) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Float]]:
    """Allocate from a computed shape while preserving runtime array values.

    Args:
        values (qmc.Vector[qmc.Float]): Runtime rotation angles.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Float]]: Measurements and input.
    """
    register = qmc.qubit_array(values.shape[0] + 1, "register")
    register[values.shape[0]] = qmc.ry(register[values.shape[0]], values[0])
    return qmc.measure(register), values


@qmc.qkernel
def _runtime_inferred_slice(values: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Float]:
    """Infer a strided slice length from a runtime vector's declared dimension.

    Args:
        values (qmc.Vector[qmc.Float]): Runtime values to slice.

    Returns:
        qmc.Vector[qmc.Float]: Odd-indexed input elements.
    """
    return values[1::2]


@qmc.qkernel
def _runtime_nested_computed_shapes(
    values: qmc.Vector[qmc.Float],
) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Float]]:
    """Propagate computed dimensions between multiple callable boundaries.

    Args:
        values (qmc.Vector[qmc.Float]): Runtime rotation angles.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Float]]: Measurements and slice.
    """
    measured, forwarded = _runtime_computed_shape(values)
    return measured, _runtime_inferred_slice(forwarded)


@qmc.qkernel
def _runtime_unary_shape(values: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
    """Resolve structural logarithm and ceiling without consuming array contents.

    Args:
        values (qmc.Vector[qmc.Float]): Runtime rotation angles.

    Returns:
        qmc.Vector[qmc.Bit]: Register measurements.
    """
    register = qmc.qubit_array(qmc.ceil(qmc.log2(values.shape[0])) + 1, "register")
    register[0] = qmc.ry(register[0], values[0])
    return qmc.measure(register)


@qmc.qkernel
def _runtime_content_dependent_shape(
    values: qmc.Vector[qmc.UInt],
) -> qmc.Vector[qmc.Bit]:
    """Require an unavailable runtime array element to determine structure.

    Args:
        values (qmc.Vector[qmc.UInt]): Runtime register-size contributions.

    Returns:
        qmc.Vector[qmc.Bit]: Register measurements.
    """
    register = qmc.qubit_array(values.shape[0] + values[0], "register")
    return qmc.measure(register)


@qmc.qkernel
def _runtime_scalar_dependent_shape(
    values: qmc.Vector[qmc.Float], extra: qmc.UInt
) -> qmc.Vector[qmc.Bit]:
    """Require an unavailable runtime scalar to determine structure.

    Args:
        values (qmc.Vector[qmc.Float]): Runtime vector with a declared dimension.
        extra (qmc.UInt): Runtime register-size contribution.

    Returns:
        qmc.Vector[qmc.Bit]: Register measurements.
    """
    register = qmc.qubit_array(values.shape[0] + extra, "register")
    return qmc.measure(register)


@qmc.qkernel
def _runtime_shape_underflow(values: qmc.Vector[qmc.Float]) -> qmc.UInt:
    """Return a native UInt subtraction derived from an array dimension.

    Args:
        values (qmc.Vector[qmc.Float]): Runtime vector with a declared dimension.

    Returns:
        qmc.UInt: Dimension minus three modulo 2**64.
    """
    return values.shape[0] - 3


@qmc.qkernel
def _runtime_shape_overflow(values: qmc.Vector[qmc.Float]) -> qmc.UInt:
    """Return a native UInt addition derived from an array dimension.

    Args:
        values (qmc.Vector[qmc.Float]): Runtime vector with a declared dimension.

    Returns:
        qmc.UInt: Dimension plus the largest UInt modulo 2**64.
    """
    return values.shape[0] + 18446744073709551615


@qmc.qkernel
def _runtime_mixed_shape_uses(
    values: qmc.Vector[qmc.Float],
) -> tuple[qmc.Vector[qmc.Bit], qmc.UInt]:
    """Share an arithmetic result between structural and native scalar uses.

    Args:
        values (qmc.Vector[qmc.Float]): Runtime vector with a declared dimension.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.UInt]: Measurements and wrapped subtraction.
    """
    difference = values.shape[0] - 3
    register = qmc.qubit_array(difference + 3, "register")
    return qmc.measure(register), difference


@qmc.qkernel
def _runtime_shared_loop_bound(
    values: qmc.Vector[qmc.Float],
) -> tuple[qmc.Vector[qmc.Bit], qmc.UInt]:
    """Share a computed dimension between allocation, loop bounds, and scalar output.

    Args:
        values (qmc.Vector[qmc.Float]): Runtime vector with a declared dimension.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.UInt]: Flipped register and native loop bound.
    """
    count = values.shape[0] + 1
    register = qmc.qubit_array(count, "register")
    for index in qmc.range(count):
        register[index] = qmc.x(register[index])
    return qmc.measure(register), count


@qmc.qkernel
def _runtime_shared_loop_step(
    values: qmc.Vector[qmc.Float],
) -> tuple[qmc.Vector[qmc.Bit], qmc.UInt]:
    """Share a computed scalar between a static loop step and a native output.

    Args:
        values (qmc.Vector[qmc.Float]): Runtime vector with a declared dimension.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.UInt]: Flipped register and native loop step.
    """
    step = values.shape[0] - 1
    register = qmc.qubit_array(3, "register")
    for index in qmc.range(0, 3, step):
        register[index] = qmc.x(register[index])
    return qmc.measure(register), step


@qmc.qkernel
def _runtime_shared_slice_bound(
    values: qmc.Vector[qmc.Float],
) -> tuple[qmc.Vector[qmc.Float], qmc.UInt]:
    """Share a computed scalar between slice metadata and a native output.

    Args:
        values (qmc.Vector[qmc.Float]): Runtime vector to slice.

    Returns:
        tuple[qmc.Vector[qmc.Float], qmc.UInt]: Last input element and slice start.
    """
    start = values.shape[0] - 1
    return values[start:], start


@qmc.qkernel
def _matrix_identity_helper(values: qmc.Matrix[qmc.Float]) -> qmc.Matrix[qmc.Float]:
    """Preserve a matrix through a direct callable boundary.

    Args:
        values (qmc.Matrix[qmc.Float]): Runtime matrix to return.

    Returns:
        qmc.Matrix[qmc.Float]: Unchanged matrix.
    """
    return values


@qmc.qkernel
def _matrix_nested_helper(values: qmc.Matrix[qmc.Float]) -> qmc.Matrix[qmc.Float]:
    """Preserve a matrix through a nested callable boundary.

    Args:
        values (qmc.Matrix[qmc.Float]): Runtime matrix to return.

    Returns:
        qmc.Matrix[qmc.Float]: Unchanged matrix from the inner helper.
    """
    result = _matrix_identity_helper(values)
    return result


@qmc.qkernel
def _matrix_direct_call(values: qmc.Matrix[qmc.Float]) -> qmc.Matrix[qmc.Float]:
    """Call a helper from a public matrix entrypoint.

    Args:
        values (qmc.Matrix[qmc.Float]): Runtime matrix to return.

    Returns:
        qmc.Matrix[qmc.Float]: Matrix from the helper.
    """
    return _matrix_identity_helper(values)


@qmc.qkernel
def _matrix_nested_call(values: qmc.Matrix[qmc.Float]) -> qmc.Matrix[qmc.Float]:
    """Propagate an array shape through multiple helper levels.

    Args:
        values (qmc.Matrix[qmc.Float]): Runtime matrix to return.

    Returns:
        qmc.Matrix[qmc.Float]: Matrix from the nested helper.
    """
    return _matrix_nested_helper(values)


@qmc.qkernel
def _matrix_repeated_call(values: qmc.Matrix[qmc.Float]) -> qmc.Matrix[qmc.Float]:
    """Reuse a shared helper twice without manufacturing conflicting definitions.

    Args:
        values (qmc.Matrix[qmc.Float]): Runtime matrix to return.

    Returns:
        qmc.Matrix[qmc.Float]: Matrix preserved through two helper invocations.
    """
    first = _matrix_identity_helper(values)
    return _matrix_identity_helper(first)


@qmc.qkernel
def _matrix_quantum_call(
    values: qmc.Matrix[qmc.Float],
) -> tuple[qmc.Vector[qmc.Bit], qmc.Matrix[qmc.Float]]:
    """Use a helper-returned matrix as runtime parameters in a quantum helper.

    Args:
        values (qmc.Matrix[qmc.Float]): Matrix containing rotation angles.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.Matrix[qmc.Float]]: Measured bits and matrix.
    """
    forwarded = _matrix_nested_helper(values)
    return _runtime_matrix(forwarded)


@qmc.qkernel
def _matrix_conflicting_calls(
    left: qmc.Matrix[qmc.Float], right: qmc.Matrix[qmc.Float]
) -> tuple[qmc.Matrix[qmc.Float], qmc.Matrix[qmc.Float]]:
    """Call one helper with two matrices whose public shapes may disagree.

    Args:
        left (qmc.Matrix[qmc.Float]): First runtime matrix.
        right (qmc.Matrix[qmc.Float]): Second runtime matrix.

    Returns:
        tuple[qmc.Matrix[qmc.Float], qmc.Matrix[qmc.Float]]: Both helper results.
    """
    return _matrix_identity_helper(left), _matrix_identity_helper(right)


def _run(prepared, shots=2, qubits=2):
    """Execute the reporting package in Selene and decode every tagged shot.

    Args:
        prepared (PreparedExecution): Reporting package and output schema.
        shots (int): Number of simulator executions.
        qubits (int): Simulator qubit capacity.

    Returns:
        list[Any]: Public typed results for all shots.
    """
    selene = pytest.importorskip("selene_sim")
    instance = selene.build(prepared.package.to_bytes())
    return [
        prepared.decode_shot(dict(shot))
        for shot in instance.run_shots(
            selene.Quest(), n_qubits=qubits, n_shots=shots, random_seed=47
        )
    ]


def _identity_program(value, native):
    """Provide an independently built native function with a typed Qamomile ABI.

    Args:
        value (ValueLike): Public scalar or structural ABI descriptor.
        native (Any): Independently specified HUGR type.

    Returns:
        CompiledProgram: Native identity function and its public ABI.
    """
    module = Module()
    main = module.define_function("main", [native], [native], visibility="Public")
    main.set_outputs(*main.inputs())
    return CompiledProgram(
        Package([module.hugr], []),
        ProgramABI(public_inputs={"value": value}, output_values=[value]),
        CompilationMetadata("hugr", "program_graph"),
    )


def _array_value(shape, element_type=None):
    """Construct a public array ABI with explicit compile-time dimensions.

    Args:
        shape (tuple[int, ...]): Concrete array dimensions.
        element_type (ValueType | None): Scalar type, defaulting to Float.

    Returns:
        ArrayValue: Array descriptor with constant dimensions.
    """
    return ArrayValue(
        type=element_type or FloatType(),
        name="value",
        shape=tuple(
            Value(type=UIntType(), name=f"dim{i}").with_const(size)
            for i, size in enumerate(shape)
        ),
    )


def test_runtime_parameter_changes_reuse_the_compiled_function(monkeypatch):
    """Typed callers execute changing angles without retracing or rebinding the kernel."""
    compiled = HugrTranspiler().compile(_runtime_bell, parameters=["theta"])
    original = compiled.artifact.to_bytes()

    def reject_recompile(*args, **kwargs):
        """Fail if submission attempts to invoke Qamomile compilation again.

        Args:
            args (tuple[Any, ...]): Unexpected positional compilation arguments.
            kwargs (dict[str, Any]): Unexpected keyword compilation arguments.

        Raises:
            Failed: Always, because execution cannot retrace the kernel.
        """
        pytest.fail("Runtime submission must retain the already compiled function")

    monkeypatch.setattr(HugrTranspiler, "compile", reject_recompile)
    for angle, expected in [(0.0, (False, False)), (math.pi, (True, True))]:
        prepared = prepare_execution(compiled, {"theta": angle})
        functions = [
            data.op
            for _, data in prepared.package.modules[0].nodes()
            if isinstance(data.op, ops.FuncDefn)
        ]
        native = next(op for op in functions if op.f_name != "main")
        assert native.inputs == [FLOAT_T]
        assert _run(prepared) == [expected, expected]
    assert compiled.artifact.to_bytes() == original


def test_runtime_preserves_public_bit_order():
    """Tagged outputs follow the declared array order, including its leading bit."""
    compiled = HugrTranspiler().compile(_runtime_bit_order)
    assert _run(prepare_execution(compiled), qubits=3) == [(True, False, False)] * 2


@pytest.mark.parametrize(
    "number", [0, 1, 2**31, 2**53 + 1, 2**63, 2**63 + 1, 2**64 - 1]
)
def test_runtime_full_uint64_domain(number):
    """Unsigned result recording preserves high-bit values without signed truncation."""
    value = Value(type=UIntType(), name="value")
    prepared = prepare_execution(_identity_program(value, int_t(6)), {"value": number})
    [actual] = _run(prepared, shots=1, qubits=1)
    assert actual == number
    assert type(actual) is int


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 1, 2), (2, 0)])
def test_runtime_array_shape_roundtrip(shape):
    """Vector, matrix, tensor, and empty axes retain row-major shape and float types."""
    values = np.arange(math.prod(shape), dtype=float).reshape(shape) + 0.125
    native = tys.Tuple(*[FLOAT_T] * math.prod(shape))
    compiled = _identity_program(_array_value(shape), native)
    prepared = prepare_execution(compiled, {"value": values})
    [actual] = _run(prepared, shots=1, qubits=1)
    assert np.asarray(actual).shape == shape
    np.testing.assert_array_equal(actual, values)


def test_runtime_nested_tuple_and_dictionary_roundtrip():
    """Nested heterogeneous outputs preserve structure, public types, and dictionary keys."""
    flag = Value(type=BitType(), name="flag")
    number = Value(type=UIntType(), name="number")
    ratio = Value(type=FloatType(), name="ratio")
    key = Value(type=UIntType(), name="key")
    mapping = DictValue(name="mapping", entries=((key, ratio),))
    value = TupleValue(name="value", elements=(flag, number, mapping))
    native = tys.Tuple(tys.Bool, int_t(6), tys.Tuple(tys.Tuple(int_t(6), FLOAT_T)))
    compiled = _identity_program(value, native)
    prepared = prepare_execution(compiled, {"value": (True, 2**63 + 7, {9: 0.25})})
    assert _run(prepared, shots=1, qubits=1) == [(True, 2**63 + 7, {9: 0.25})]


@pytest.mark.parametrize(
    "bindings,exception,match",
    [
        ({}, ValueError, "Missing"),
        ({"theta": 0.0, "size": 2}, ValueError, "Unexpected"),
        ({"theta": True}, TypeError, "real number"),
        ({"theta": "0.1"}, TypeError, "real number"),
    ],
)
def test_runtime_rejects_invalid_bindings(bindings, exception, match):
    """Binding mistakes fail before local compilation or remote submission."""
    compiled = HugrTranspiler().compile(_runtime_bell, parameters=["theta"])
    with pytest.raises(exception, match=match):
        prepare_execution(compiled, bindings)


@pytest.mark.parametrize("number", [-1, 2**64, True, 1.5])
def test_runtime_rejects_uint_values_outside_the_public_domain(number):
    """UInt runtime parameters never silently wrap, truncate, or accept booleans."""
    compiled = _identity_program(Value(type=UIntType(), name="value"), int_t(6))
    with pytest.raises((TypeError, ValueError), match="UInt"):
        prepare_execution(compiled, {"value": number})


def test_runtime_rejects_array_shape_mismatch():
    """A flat array cannot impersonate a matrix with the same element count."""
    compiled = _identity_program(_array_value((2, 2)), tys.Tuple(*[FLOAT_T] * 4))
    with pytest.raises(ValueError, match="shape"):
        prepare_execution(compiled, {"value": [1.0, 2.0, 3.0, 4.0]})


def test_runtime_rejects_incomplete_and_invalid_output_records():
    """Missing, extra, and incorrectly typed records produce explicit ABI errors."""
    compiled = HugrTranspiler().compile(_runtime_bit_order)
    prepared = prepare_execution(compiled)
    valid = dict(zip(prepared.output_tags, [1, 0, 0], strict=True))
    assert prepared.decode_shot(valid) == (True, False, False)
    with pytest.raises(ValueError, match="Missing"):
        prepared.decode_shot({})
    with pytest.raises(ValueError, match="Unexpected"):
        prepared.decode_shot(valid | {"foreign": 0})
    with pytest.raises(ValueError, match="Bit"):
        prepared.decode_shot(valid | {prepared.output_tags[0]: 2})


def test_runtime_uses_explicit_input_order_and_excludes_compile_time_values():
    """Explicit native input metadata controls binding order independently of ABI order."""
    module = Module()
    main = module.define_function(
        "main", [FLOAT_T, tys.Bool], [FLOAT_T, tys.Bool], visibility="Public"
    )
    main.set_outputs(*main.inputs())
    flag = Value(type=BitType(), name="flag")
    number = Value(type=FloatType(), name="number")
    fixed = Value(type=UIntType(), name="fixed").with_const(3)
    compiled = CompiledProgram(
        Package([module.hugr], []),
        ProgramABI({"flag": flag, "fixed": fixed, "number": number}, [number, flag]),
        CompilationMetadata(
            "hugr", "program_graph", {"runtime_inputs": ("number", "flag")}
        ),
    )
    prepared = prepare_execution(compiled, {"flag": False, "number": 0.5})
    assert _run(prepared, shots=1, qubits=1) == [(0.5, False)]
    with pytest.raises(ValueError, match="Unexpected"):
        prepare_execution(compiled, {"flag": False, "number": 0.5, "fixed": 3})


def test_runtime_identity_expval_validates_parameters_without_a_native_package():
    """Identity-only estimation still enforces the original parameter contract."""
    compiled = HugrTranspiler().compile(_runtime_bell, parameters=["theta"])
    identity = dataclasses.replace(compiled, artifact=())
    validate_runtime_bindings(identity, {"theta": 0.125})
    with pytest.raises(TypeError, match="Float"):
        validate_runtime_bindings(identity, {"theta": False})
    with pytest.raises(ValueError, match="Missing"):
        validate_runtime_bindings(identity)


def test_runtime_native_abi_mismatch_is_rejected_without_mutation():
    """An artifact with stale public output metadata cannot be silently decoded."""
    compiled = HugrTranspiler().compile(_runtime_bell, parameters=["theta"])
    abi = copy.deepcopy(compiled.abi)
    abi.output_values.pop()
    incompatible = dataclasses.replace(compiled, abi=abi)
    with pytest.raises(ValueError, match="output ports"):
        prepare_execution(incompatible, {"theta": 0.0})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_runtime_rejects_nonfinite_float_before_native_serialization(value):
    """Non-finite arguments cannot reach native serializers that abort the process."""
    compiled = HugrTranspiler().compile(_runtime_bell, parameters=["theta"])
    with pytest.raises(ValueError, match="finite"):
        prepare_execution(compiled, {"theta": value})


def test_parameter_shapes_keep_matrix_values_runtime():
    """Dimension specialization permits matrix angles without binding their values."""
    original = QamomileCompiler().prepare(_runtime_matrix, parameters=["values"])
    program = resolve_parameter_shapes(original, {"values": (2, 2)})
    assert not original.abi.public_inputs["values"].shape[0].is_constant()
    assert not program.bindings
    target = HugrTarget()
    compiled = target.compile(program, target.plan(program))
    for values, bits in [
        ([[0.1, math.pi], [0.2, 0.0]], (True, False)),
        ([[0.3, 0.0], [0.4, math.pi]], (False, True)),
    ]:
        prepared = prepare_execution(compiled, {"values": values})
        [actual] = _run(prepared, shots=1)
        assert actual == (bits, tuple(tuple(row) for row in values))


@pytest.mark.parametrize(
    "kernel", [_runtime_computed_shape, _runtime_nested_computed_shapes]
)
@pytest.mark.parametrize("length", [2, 4])
def test_runtime_computed_parameter_shapes_preserve_array_values(kernel, length):
    """Computed allocations cross helper boundaries and respond to changed contents."""
    from qamomile.hugr import HugrExecutor

    executable = HugrTranspiler().transpile(
        kernel, parameters=["values"], parameter_shapes={"values": (length,)}
    )
    executor = HugrExecutor()
    # Ry(0)|0> = |0> and Ry(pi)|0> = |1>, so each expected bit is deterministic.
    for angle, last in [(0.0, False), (math.pi, True)]:
        values = (angle, *[float(index) for index in range(1, length)])
        returned = values if kernel is _runtime_computed_shape else values[1::2]
        bits, actual = executable.run(executor, bindings={"values": values}).result()
        assert bits == (False,) * length + (last,)
        # Copying or striding the input performs no arithmetic, so require exact values.
        assert np.shape(actual) == np.shape(returned)
        np.testing.assert_allclose(actual, returned, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("length", [0, 1, 2, 3, 4])
def test_runtime_inferred_parameter_slice_length(length):
    """Inferred strided slices return the correct elements for odd and even lengths."""
    from qamomile.hugr import HugrExecutor

    executable = HugrTranspiler().transpile(
        _runtime_inferred_slice,
        parameters=["values"],
        parameter_shapes={"values": (length,)},
    )
    executor = HugrExecutor()
    for offset in [0.25, -1.5]:
        values = tuple(offset + index for index in range(length))
        actual = executable.run(executor, bindings={"values": values}).result()
        # Striding the input performs no arithmetic, so require exact values.
        assert np.shape(actual) == np.shape(values[1::2])
        np.testing.assert_allclose(actual, values[1::2], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("length", [1, 3])
def test_runtime_unary_parameter_shape_arithmetic(length):
    """Logarithm and ceiling specialize structural dimensions without freezing angles."""
    from qamomile.hugr import HugrExecutor

    executable = HugrTranspiler().transpile(
        _runtime_unary_shape,
        parameters=["values"],
        parameter_shapes={"values": (length,)},
    )
    executor = HugrExecutor()
    # Ry(0)|0> = |0> and Ry(pi)|0> = |1>; all untouched qubits stay zero.
    for angle, first in [(0.0, False), (math.pi, True)]:
        values = (angle,) * length
        assert executable.run(executor, bindings={"values": values}).result() == (
            first,
        ) + (False,) * math.ceil(math.log2(length))


@pytest.mark.parametrize(
    "kernel,expected",
    [(_runtime_shape_underflow, (1 << 64) - 1), (_runtime_shape_overflow, 1)],
)
def test_runtime_parameter_shape_scalar_arithmetic_preserves_uint64(kernel, expected):
    """Ordinary shape-derived scalar results retain native unsigned wraparound."""
    from qamomile.hugr import HugrExecutor

    # 2 - 3 wraps to 2**64 - 1; 2 + (2**64 - 1) wraps to 1.
    executable = HugrTranspiler().transpile(
        kernel, parameters=["values"], parameter_shapes={"values": (2,)}
    )
    assert (
        executable.run(HugrExecutor(), bindings={"values": (0.0, 1.0)}).result()
        == expected
    )


def test_runtime_parameter_shape_shared_arithmetic_preserves_uint64():
    """Structural mathematical integers never replace a shared native UInt result."""
    from qamomile.hugr import HugrExecutor

    executable = HugrTranspiler().transpile(
        _runtime_mixed_shape_uses,
        parameters=["values"],
        parameter_shapes={"values": (2,)},
    )
    # Allocation uses (-1) + 3 = 2, while the native UInt result wraps -1 to 2**64 - 1.
    assert executable.run(HugrExecutor(), bindings={"values": (0.0, 1.0)}).result() == (
        (False, False),
        (1 << 64) - 1,
    )


@pytest.mark.parametrize(
    "kernel,expected_scalar",
    [(_runtime_shared_loop_bound, 3), (_runtime_shared_loop_step, 1)],
)
def test_runtime_parameter_shape_shared_loop_structure(kernel, expected_scalar):
    """Shared loop bounds and steps remain static while preserving native outputs."""
    from qamomile.hugr import HugrExecutor

    executable = HugrTranspiler().transpile(
        kernel, parameters=["values"], parameter_shapes={"values": (2,)}
    )
    # Both loops apply X exactly once to each of the three initially zero qubits.
    assert executable.run(HugrExecutor(), bindings={"values": (0.0, 1.0)}).result() == (
        (True, True, True),
        expected_scalar,
    )


def test_runtime_parameter_shape_shared_slice_structure():
    """A shared computed slice start selects changed runtime contents exactly."""
    from qamomile.hugr import HugrExecutor

    executable = HugrTranspiler().transpile(
        _runtime_shared_slice_bound,
        parameters=["values"],
        parameter_shapes={"values": (2,)},
    )
    executor = HugrExecutor()
    for values in [(0.25, -1.5), (2.5, 3.75)]:
        actual, start = executable.run(executor, bindings={"values": values}).result()
        assert start == 1
        # Slicing copies values without arithmetic, so require exact roundtrip values.
        assert np.shape(actual) == np.shape(values[1:])
        np.testing.assert_allclose(actual, values[1:], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "kernel,parameters",
    [
        (_runtime_content_dependent_shape, ["values"]),
        (_runtime_scalar_dependent_shape, ["values", "extra"]),
    ],
)
def test_parameter_shapes_reject_runtime_dependent_structure(kernel, parameters):
    """A declared array shape never makes array contents or another argument static."""
    from qamomile.circuit.transpiler import EmitError

    with pytest.raises(EmitError, match="requires static dimensions"):
        HugrTranspiler().transpile(
            kernel, parameters=parameters, parameter_shapes={"values": (2,)}
        )


def test_computed_shape_resolution_preserves_owned_definition_identity():
    """Computed dimensions specialize independently without mutating source bodies."""
    original = QamomileCompiler().prepare(
        _runtime_nested_computed_shapes, parameters=["values"]
    )
    for length in [2, 4]:
        program = resolve_parameter_shapes(original, {"values": (length,)})
        assert not program.bindings
        assert program.entrypoint.output_values[0].shape[0].get_const() == length + 1
        assert program.entrypoint.output_values[1].shape[0].get_const() == length // 2
        assert all(
            len(variants) == 1 for variants in program.definition_variants.values()
        )
    assert not original.entrypoint.output_values[0].shape[0].is_constant()
    for definition in original.definitions.values():
        assert not definition.body.input_values[0].shape[0].is_constant()


@pytest.mark.parametrize(
    "shapes,exception,match",
    [
        ({"unknown": (2, 2)}, ValueError, "Unknown"),
        ({"values": (2,)}, ValueError, "rank"),
        ({"values": (2, -1)}, ValueError, "non-negative"),
        ({"values": (2, True)}, TypeError, "integer"),
        ({"values": (2, 1.5)}, TypeError, "integer"),
    ],
)
def test_parameter_shapes_reject_invalid_shape_metadata(shapes, exception, match):
    """Shape declarations cannot silently change ranks or coerce invalid dimensions."""
    original = QamomileCompiler().prepare(_runtime_matrix, parameters=["values"])
    with pytest.raises(exception, match=match):
        resolve_parameter_shapes(original, shapes)


@pytest.mark.parametrize(
    "kernel", [_matrix_direct_call, _matrix_nested_call, _matrix_repeated_call]
)
def test_runtime_matrix_shapes_cross_callable_boundaries(kernel):
    """Direct, nested, and repeated helper calls retain runtime array values and shapes."""
    executable = HugrTranspiler().transpile(
        kernel, parameters=["values"], parameter_shapes={"values": (2, 2)}
    )
    from qamomile.hugr import HugrExecutor

    executor = HugrExecutor()
    for data in [((0.5, 1.0), (1.5, 2.0)), ((-1.0, 0.0), (0.25, 3.0))]:
        assert executable.run(executor, bindings={"values": data}).result() == data


def test_runtime_matrix_helpers_preserve_quantum_parameter_values():
    """Shape propagation reaches helper loop bodies while matrix angles remain runtime."""
    from qamomile.hugr import HugrExecutor

    executable = HugrTranspiler().transpile(
        _matrix_quantum_call, parameters=["values"], parameter_shapes={"values": (2, 2)}
    )
    executor = HugrExecutor()
    for data, bits in [
        (((0.25, math.pi), (0.5, 0.0)), (True, False)),
        (((0.75, 0.0), (1.0, math.pi)), (False, True)),
    ]:
        assert executable.run(executor, bindings={"values": data}).result() == (
            bits,
            data,
        )


def test_callable_shape_resolution_preserves_source_and_shared_definition_identity():
    """Resolving one shape never changes source helpers or prevents later specialization."""
    original = QamomileCompiler().prepare(_matrix_repeated_call, parameters=["values"])
    first = resolve_parameter_shapes(original, {"values": (2, 2)})
    second = resolve_parameter_shapes(original, {"values": (3, 2)})
    for program, rows in [(first, 2), (second, 3)]:
        assert all(
            len(variants) == 1 for variants in program.definition_variants.values()
        )
        for definition in program.definitions.values():
            assert definition.body.input_values[0].shape[0].get_const() == rows
            assert definition.body.output_values[0].shape[0].get_const() == rows
    for definition in original.definitions.values():
        assert not definition.body.input_values[0].shape[0].is_constant()


def test_conflicting_callable_shapes_fail_with_an_explicit_diagnostic():
    """A shared native helper cannot silently accept two incompatible tuple widths."""
    with pytest.raises(ValueError, match="Conflicting HUGR callable shapes"):
        HugrTranspiler().transpile(
            _matrix_conflicting_calls,
            parameters=["left", "right"],
            parameter_shapes={"left": (2, 2), "right": (3, 2)},
        )


def test_runtime_abi_identity_distinguishes_equal_native_array_carriers():
    """Restoration identities distinguish matrix shapes with identical native tuples."""
    native = tys.Tuple(*[FLOAT_T] * 6)
    first = prepare_execution(
        _identity_program(_array_value((2, 3)), native),
        {"value": [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]},
    )
    second = prepare_execution(
        _identity_program(_array_value((3, 2)), native),
        {"value": [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]},
    )
    assert first.package.to_bytes() == second.package.to_bytes()
    assert first.abi_sha256 != second.abi_sha256


@pytest.mark.parametrize("shape", [(0, 2), (2, 0, 3), (2, 0)])
@pytest.mark.parametrize(
    "element_type,dtype",
    [(BitType(), np.bool_), (UIntType(), np.uint64), (FloatType(), np.float64)],
)
@pytest.mark.parametrize("target", ["selene", "helios"])
def test_empty_array_public_jobs_preserve_shape_and_element_type(
    monkeypatch, tmp_path, shape, element_type, dtype, target
):
    """Run and sample preserve empty-axis shape with typed arrays where tuples cannot."""
    from types import SimpleNamespace

    from hugr.qsystem.result import QsysResult

    from qamomile.circuit.transpiler.execution_handle import CompletedExecutionHandle
    from qamomile.hugr import HugrExecutable, HugrExecutor, SeleneExecutionOptions
    from qamomile.hugr._nexus import _decode_results

    monkeypatch.setattr(
        "qamomile.hugr.execution._NexusTransport",
        lambda options: SimpleNamespace(
            submit=lambda package, shots: CompletedExecutionHandle(
                _decode_results(QsysResult([[] for _ in range(shots)]), shots)
            )
        ),
    )
    compiled = _identity_program(_array_value(shape, element_type), tys.Tuple())
    executable = HugrExecutable(compiled)
    options = SeleneExecutionOptions(build_dir=tmp_path) if target == "selene" else None
    executor = HugrExecutor(target, options=options)
    values = np.empty(shape, dtype=dtype)
    actual = executable.run(executor, bindings={"value": values}).result()
    sampled = executable.sample(executor, shots=3, bindings={"value": values}).result()
    assert sampled.shots == 3
    assert len(sampled.results) == 1
    sampled_value, count = sampled.results[0]
    assert count == 3
    for returned in (actual, sampled_value):
        assert np.asarray(returned).shape == shape
        if 0 in shape[:-1]:
            assert isinstance(returned, np.ndarray)
            assert returned.dtype == np.dtype(dtype)
        else:
            assert isinstance(returned, tuple)


def test_empty_array_abi_identity_preserves_element_type():
    """Empty native tuples cannot erase distinct public array element types."""
    prepared = [
        prepare_execution(
            _identity_program(_array_value((0, 2), element_type), tys.Tuple()),
            {"value": np.empty((0, 2))},
        )
        for element_type in (BitType(), UIntType(), FloatType())
    ]
    assert len({item.package.to_bytes() for item in prepared}) == 1
    assert len({item.abi_sha256 for item in prepared}) == 3
