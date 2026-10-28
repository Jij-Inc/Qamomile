"""Selected branch widths define packed-register casts and measurements."""

from __future__ import annotations

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation import CastOperation
from qamomile.circuit.ir.operation.gate import MeasureQFixedOperation
from qamomile.circuit.ir.types import QFixedType, QUIntType
from qamomile.circuit.ir.value import array_static_length, root_carrier_keys
from qamomile.circuit.serialization import deserialize, serialize
from qamomile.circuit.transpiler.errors import SeparationError


@qmc.qkernel
def _static_qint(flag: qmc.UInt) -> qmc.UInt:
    """Measure a selected static-width register as an unsigned integer.

    Args:
        flag (qmc.UInt): Select the true branch when equal to one.

    Returns:
        qmc.UInt: Selected or transformed quantum result.
    """
    if flag == 1:
        work = qmc.qubit_array(4, "large")
    else:
        work = qmc.qubit_array(2, "small")
    work[0] = qmc.x(work[0])
    return qmc.measure(qmc.cast(work, qmc.QInt))


@qmc.qkernel
def _static_qfixed(flag: qmc.UInt) -> qmc.Float:
    """Measure a selected static-width register with one integer bit.

    Args:
        flag (qmc.UInt): Select the true branch when equal to one.

    Returns:
        qmc.Float: Selected or transformed quantum result.
    """
    if flag == 1:
        work = qmc.qubit_array(4, "large")
    else:
        work = qmc.qubit_array(2, "small")
    work[0] = qmc.x(work[0])
    return qmc.measure(qmc.cast(work, qmc.QFixed, int_bits=1))


@qmc.qkernel
def _symbolic_qint(flag: qmc.UInt, n: qmc.UInt, m: qmc.UInt) -> qmc.UInt:
    """Measure the register allocated with the selected symbolic width.

    Args:
        flag (qmc.UInt): Select the true branch when equal to one.
        n (qmc.UInt): True-branch register width.
        m (qmc.UInt): False-branch register width.

    Returns:
        qmc.UInt: Selected or transformed quantum result.
    """
    if flag == 1:
        work = qmc.qubit_array(n, "large")
    else:
        work = qmc.qubit_array(m, "small")
    work[0] = qmc.x(work[0])
    return qmc.measure(qmc.cast(work, qmc.QInt))


@qmc.qkernel
def _symbolic_qfixed(flag: qmc.UInt, n: qmc.UInt, m: qmc.UInt) -> qmc.Float:
    """Measure the selected symbolic register as a fractional value.

    Args:
        flag (qmc.UInt): Select the true branch when equal to one.
        n (qmc.UInt): True-branch register width.
        m (qmc.UInt): False-branch register width.

    Returns:
        qmc.Float: Selected or transformed quantum result.
    """
    if flag == 1:
        work = qmc.qubit_array(n, "large")
    else:
        work = qmc.qubit_array(m, "small")
    work[0] = qmc.x(work[0])
    return qmc.measure(qmc.cast(work, qmc.QFixed))


def _lower(transpiler, kernel, bindings):
    """Run public pipeline stages through compile-time branch selection.

    Args:
        transpiler (QiskitTranspiler): Engine providing the public pass methods.
        kernel (QKernel | SerializedQKernel): Quantum program to lower.
        bindings (dict[str, int]): Compile-time argument values.

    Returns:
        Block: Lowered block with selected array widths and carrier layouts.
    """
    block = transpiler.to_block(kernel, bindings=bindings)
    block = transpiler.inline(transpiler.substitute(block))
    block = transpiler.affine_validate(block)
    block = transpiler.constant_fold(block, bindings=bindings)
    return transpiler.lower_compile_time_ifs(block, bindings=bindings)


@pytest.mark.parametrize("flag", [0, 1])
@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("symbolic", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
def test_selected_branch_rebuilds_complete_packed_layout(
    flag, serialized, symbolic, fixed
):
    """Direct and round-tripped casts adopt the selected width in every channel."""
    from qamomile.qiskit import QiskitTranspiler

    kernels = {
        (False, False): _static_qint,
        (False, True): _static_qfixed,
        (True, False): _symbolic_qint,
        (True, True): _symbolic_qfixed,
    }
    kernel = kernels[symbolic, fixed]
    if serialized:
        kernel = deserialize(serialize(kernel))
    bindings = {"flag": flag}
    if symbolic:
        bindings.update(n=4, m=2)
    width = 4 if flag else 2
    int_bits = int(fixed and not symbolic)
    transpiler = QiskitTranspiler()
    lowered = _lower(transpiler, kernel, bindings)
    cast_op = next(op for op in lowered.operations if isinstance(op, CastOperation))
    source = cast_op.operands[0]
    result = cast_op.results[0]
    expected_type = (
        QFixedType(integer_bits=int_bits, fractional_bits=width - int_bits)
        if fixed
        else QUIntType(width=width)
    )
    assert array_static_length(source) == width
    assert result.type == expected_type == cast_op.target_type
    carriers, logical = root_carrier_keys(source, width)
    assert result.get_cast_source_uuid() == source.uuid
    assert result.get_cast_source_logical_id() == source.logical_id
    assert result.get_cast_qubit_uuids() == tuple(carriers)
    assert result.get_cast_qubit_logical_ids() == tuple(logical)
    assert cast_op.qubit_mapping == carriers
    if fixed:
        assert result.get_qfixed_qubit_uuids() == tuple(carriers)
        assert result.get_qfixed_num_bits() == width
        assert result.get_qfixed_int_bits() == int_bits
        measurement = next(
            op for op in lowered.operations if isinstance(op, MeasureQFixedOperation)
        )
        assert measurement.num_bits == width
    executable = transpiler.transpile(kernel, bindings=bindings)
    result_counts = executable.sample(transpiler.executor(), shots=4).result()
    expected = 2 ** (int_bits - width) if fixed else 1
    assert len(result_counts.results) == 1
    value, shots = result_counts.results[0]
    assert shots == 4
    if fixed:
        assert value == pytest.approx(expected, rel=0.0, abs=1e-12)
    else:
        assert value == expected


@pytest.mark.parametrize("fixed", [False, True])
def test_runtime_unequal_width_branch_rejected_by_segmentation(fixed):
    """Measurement-selected register allocation has the same plan-time rejection."""
    from qamomile.qiskit import QiskitTranspiler

    @qmc.qkernel
    def integer() -> qmc.UInt:
        """Measure a runtime-selected register as an integer.

        Returns:
            qmc.UInt: Selected or transformed quantum result.
        """
        selector = qmc.measure(qmc.qubit("selector"))
        if selector:
            work = qmc.qubit_array(4, "large")
        else:
            work = qmc.qubit_array(2, "small")
        return qmc.measure(qmc.cast(work, qmc.QInt))

    @qmc.qkernel
    def fixed_point() -> qmc.Float:
        """Measure a runtime-selected register as a fractional value.

        Returns:
            qmc.Float: Selected or transformed quantum result.
        """
        selector = qmc.measure(qmc.qubit("selector"))
        if selector:
            work = qmc.qubit_array(4, "large")
        else:
            work = qmc.qubit_array(2, "small")
        return qmc.measure(qmc.cast(work, qmc.QFixed))

    kernel = fixed_point if fixed else integer
    messages = []
    for candidate in (kernel, deserialize(serialize(kernel))):
        with pytest.raises(
            SeparationError, match="width is still symbolic at plan time"
        ) as caught:
            QiskitTranspiler().transpile(candidate)
        messages.append(str(caught.value))
    assert messages[0] == messages[1]


def test_qfixed_integer_bits_must_fit_selected_width():
    """Only a selected width too narrow for the integer layout is rejected."""
    from qamomile.qiskit import QiskitTranspiler

    @qmc.qkernel
    def kernel(flag: qmc.UInt) -> qmc.Float:
        """Measure the selected register after composing quantum operations.

        Args:
            flag (qmc.UInt): Select the true branch when equal to one.

        Returns:
            qmc.Float: Selected or transformed quantum result.
        """
        if flag == 1:
            work = qmc.qubit_array(4, "large")
        else:
            work = qmc.qubit_array(2, "small")
        return qmc.measure(qmc.cast(work, qmc.QFixed, int_bits=3))

    QiskitTranspiler().transpile(kernel, bindings={"flag": 1})
    with pytest.raises(ValueError, match=r"int_bits \(3\).*qubits \(2\)"):
        QiskitTranspiler().transpile(kernel, bindings={"flag": 0})


@qmc.qkernel
def _select_register(flag: qmc.UInt, n: qmc.UInt, m: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Allocate the register with the selected branch width.

    Args:
        flag (qmc.UInt): Select the true branch when equal to one.
        n (qmc.UInt): True-branch register width.
        m (qmc.UInt): False-branch register width.

    Returns:
        qmc.Vector[qmc.Qubit]: Selected or transformed quantum result.
    """
    if flag == 1:
        work = qmc.qubit_array(n, "large")
    else:
        work = qmc.qubit_array(m, "small")
    return work


@qmc.qkernel
def _nested_register(flag: qmc.UInt, n: qmc.UInt, m: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Select a register through nested compile-time branches.

    Args:
        flag (qmc.UInt): Select the true branch when equal to one.
        n (qmc.UInt): True-branch register width.
        m (qmc.UInt): False-branch register width.

    Returns:
        qmc.Vector[qmc.Qubit]: Selected or transformed quantum result.
    """
    if flag == 1:
        if flag == 1:
            work = qmc.qubit_array(4, "large")
        else:
            work = qmc.qubit_array(3, "middle")
    else:
        work = qmc.qubit_array(2, "small")
    return work


@qmc.qkernel
def _whole_register(work: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Flip every carrier in the register.

    Args:
        work (qmc.Vector[qmc.Qubit]): Register to transform and return.

    Returns:
        qmc.Vector[qmc.Qubit]: Selected or transformed quantum result.
    """
    return qmc.x(work)


@qmc.qkernel
def _first_bit(work: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Set the first register carrier to one.

    Args:
        work (qmc.Vector[qmc.Qubit]): Register to transform and return.

    Returns:
        qmc.Vector[qmc.Qubit]: Selected or transformed quantum result.
    """
    work[0] = qmc.x(work[0])
    return work


def _composed_kernel(fixed, path):
    """Trace the same selected register through layout-preserving operations.

    Args:
        fixed (bool): Use fixed-point measurement instead of integer decoding.
        path (str): Select nested, view, subkernel, or whole-vector composition.

    Returns:
        QKernel: Program exercising the requested composition.
    """
    selector = _nested_register if path == "nested" else _select_register
    transform = {
        "whole": _whole_register,
        "view": _first_bit,
        "nested": _first_bit,
        "subkernel": _first_bit,
    }[path]
    if path == "view":
        if fixed:

            @qmc.qkernel
            def view_kernel(flag: qmc.UInt) -> qmc.Float:
                """Measure the first two carriers of the selected register.

                Args:
                    flag (qmc.UInt): Select the true branch when equal to one.

                Returns:
                    qmc.Float: Selected or transformed quantum result.
                """
                work = selector(flag, 4, 2)
                view = work[0:2]
                view[0] = qmc.x(view[0])
                return qmc.measure(qmc.cast(view, qmc.QFixed))
        else:

            @qmc.qkernel
            def view_kernel(flag: qmc.UInt) -> qmc.UInt:
                """Measure the first two carriers of the selected register.

                Args:
                    flag (qmc.UInt): Select the true branch when equal to one.

                Returns:
                    qmc.UInt: Selected or transformed quantum result.
                """
                work = selector(flag, 4, 2)
                view = work[0:2]
                view[0] = qmc.x(view[0])
                return qmc.measure(qmc.cast(view, qmc.QInt))

        return view_kernel
    if fixed:

        @qmc.qkernel
        def kernel(flag: qmc.UInt) -> qmc.Float:
            """Measure the selected register after composing quantum operations.

            Args:
                flag (qmc.UInt): Select the true branch when equal to one.

            Returns:
                qmc.Float: Selected or transformed quantum result.
            """
            work = selector(flag, 4, 2)
            work = transform(work)
            return qmc.measure(qmc.cast(work, qmc.QFixed))
    else:

        @qmc.qkernel
        def kernel(flag: qmc.UInt) -> qmc.UInt:
            """Measure the selected register after composing quantum operations.

            Args:
                flag (qmc.UInt): Select the true branch when equal to one.

            Returns:
                qmc.UInt: Selected or transformed quantum result.
            """
            work = selector(flag, 4, 2)
            work = transform(work)
            return qmc.measure(qmc.cast(work, qmc.QInt))

    return kernel


@pytest.mark.parametrize("flag", [0, 1])
@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("path", ["nested", "view", "subkernel", "whole"])
def test_selected_width_survives_composition(flag, serialized, fixed, path):
    """Nested choices, views, subkernels and vector gates keep selected carriers."""
    from qamomile.qiskit import QiskitTranspiler

    kernel = _composed_kernel(fixed, path)
    if serialized:
        kernel = deserialize(serialize(kernel))
    width = 2 if path == "view" or not flag else 4
    expected = 2**width - 1 if path == "whole" else 1
    if fixed:
        expected /= 2**width
    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(kernel, bindings={"flag": flag})
    result = executable.sample(transpiler.executor(), shots=4).result()
    assert len(result.results) == 1
    value, shots = result.results[0]
    assert shots == 4
    if fixed:
        assert value == pytest.approx(expected, rel=0.0, abs=1e-12)
    else:
        assert value == expected


def test_qfixed_measure_requires_cast_metadata():
    """A QFixed-only carrier channel cannot bypass the strict plan policy."""
    from qamomile.circuit.ir.types import FloatType
    from qamomile.circuit.ir.value import Value
    from qamomile.circuit.transpiler.passes.separate import lower_measure_qfixed

    register = Value(type=QFixedType(0, 2), name="register").with_qfixed_metadata(
        qubit_uuids=("q0", "q1"), num_bits=2, int_bits=0
    )
    operation = MeasureQFixedOperation(
        operands=[register],
        results=[Value(type=FloatType(), name="result")],
        num_bits=2,
        int_bits=0,
    )
    with pytest.raises(SeparationError, match="requires cast metadata"):
        lower_measure_qfixed(operation)


@qmc.qkernel
def _unequal_qint_views(flag: qmc.UInt) -> tuple[qmc.UInt, qmc.UInt]:
    """Measure a selected view as an unsigned integer.

    Args:
        flag (qmc.UInt): Select four carriers when one and two otherwise.

    Returns:
        tuple[qmc.UInt, qmc.UInt]: Encoded integer and selected view length.
    """
    work = qmc.qubit_array(4, "work")
    work[0] = qmc.x(work[0])
    if flag == 1:
        selected = work[0:4]
    else:
        selected = work[0:2]
    width = selected.shape[0] + flag * 0
    return qmc.measure(qmc.cast(selected, qmc.QInt)), width


@qmc.qkernel
def _unequal_qfixed_views(flag: qmc.UInt) -> tuple[qmc.Float, qmc.UInt]:
    """Measure a selected view as a fractional fixed-point value.

    Args:
        flag (qmc.UInt): Select four carriers when one and two otherwise.

    Returns:
        tuple[qmc.Float, qmc.UInt]: Encoded fraction and selected view length.
    """
    work = qmc.qubit_array(4, "work")
    work[0] = qmc.x(work[0])
    if flag == 1:
        selected = work[0:4]
    else:
        selected = work[0:2]
    width = selected.shape[0] + flag * 0
    return qmc.measure(qmc.cast(selected, qmc.QFixed)), width


@pytest.mark.parametrize("flag", [0, 1])
@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
def test_unequal_branch_views_keep_selected_width(flag, serialized, fixed):
    """Merged views defer their shape and carriers until their branch is selected."""
    from qamomile.qiskit import QiskitTranspiler

    kernel = _unequal_qfixed_views if fixed else _unequal_qint_views
    if serialized:
        kernel = deserialize(serialize(kernel))
    width = 4 if flag else 2
    transpiler = QiskitTranspiler()
    lowered = _lower(transpiler, kernel, {"flag": flag})
    operation = next(op for op in lowered.operations if isinstance(op, CastOperation))
    source = operation.operands[0]
    packed = operation.results[0]
    expected_type = QFixedType(0, width) if fixed else QUIntType(width)
    carriers, logical = root_carrier_keys(source, width)
    assert array_static_length(source) == width
    assert packed.type == expected_type == operation.target_type
    assert operation.qubit_mapping == carriers
    assert packed.get_cast_source_uuid() == source.uuid
    assert packed.get_cast_source_logical_id() == source.logical_id
    assert packed.get_cast_qubit_uuids() == tuple(carriers)
    assert packed.get_cast_qubit_logical_ids() == tuple(logical)
    if fixed:
        assert packed.get_qfixed_num_bits() == width
        assert packed.get_qfixed_int_bits() == 0
        assert packed.get_qfixed_qubit_uuids() == tuple(carriers)
    executable = transpiler.transpile(kernel, bindings={"flag": flag})
    circuit = executable.compiled_quantum[0].circuit
    assert (
        sum(instruction.operation.name == "measure" for instruction in circuit.data)
        == width
    )
    results = executable.sample(transpiler.executor(), shots=4).result().results
    assert len(results) == 1
    (value, observed_width), shots = results[0]
    assert shots == 4
    assert observed_width == width
    if fixed:
        assert value == pytest.approx(2 ** (-width), rel=0.0, abs=1e-12)
    else:
        assert value == 1


@pytest.mark.parametrize("carrier_count", [0, 2])
def test_qfixed_merge_rejects_carrier_count_mismatch(carrier_count):
    """Empty carrier channels cannot conceal a nonzero stored QFixed width."""
    from qamomile.circuit.ir.value import Value

    carriers = tuple(f"q{index}" for index in range(carrier_count))
    register = (
        Value(type=QFixedType(0, 3), name="register")
        .with_cast_metadata(
            source_uuid="source",
            source_logical_id="source_logical",
            qubit_uuids=carriers,
            qubit_logical_ids=carriers,
        )
        .with_qfixed_metadata(qubit_uuids=carriers, num_bits=3, int_bits=0)
    )
    with pytest.raises(TypeError, match="num_bits to match its carrier count"):
        qmc.QFixed(value=register)._wrap_merge_result(register, register)


@pytest.mark.parametrize("symbolic", [False, True])
def test_qfixed_merge_accepts_valid_empty_carrier_layout(symbolic):
    """Known-zero and deferred-symbolic layouts both use zero stored carrier count."""
    from qamomile.circuit.ir.types import UIntType
    from qamomile.circuit.ir.value import Value

    width = Value(type=UIntType(), name="width") if symbolic else 0
    register = (
        Value(type=QFixedType(0, width), name="register")
        .with_cast_metadata(
            source_uuid="source",
            source_logical_id="source_logical",
            qubit_uuids=(),
            qubit_logical_ids=(),
        )
        .with_qfixed_metadata(qubit_uuids=(), num_bits=0, int_bits=0)
    )
    merged = qmc.QFixed(value=register)._wrap_merge_result(register, register)
    assert merged.value.type == register.type
    assert merged.value.get_qfixed_num_bits() == 0
    assert merged.value.get_cast_qubit_uuids() == ()
