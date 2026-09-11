"""Verify selected quantum widths without rebinding shared input dimensions."""

from __future__ import annotations

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.serialization import deserialize, serialize


@qmc.qkernel
def _symbolic_qint(flag: qmc.UInt, n: qmc.UInt, m: qmc.UInt) -> qmc.UInt:
    """Measure a packed integer after selecting its source allocation.

    Args:
        flag (qmc.UInt): Select the n-sized branch when equal to one.
        n (qmc.UInt): True-branch carrier count.
        m (qmc.UInt): False-branch carrier count.

    Returns:
        qmc.UInt: Measured unsigned integer.
    """
    if flag == 1:
        register = qmc.qubit_array(n, "register")
    else:
        register = qmc.qubit_array(m, "register")
    return qmc.measure(qmc.cast(register, qmc.QInt))


@qmc.qkernel
def _symbolic_qfixed(flag: qmc.UInt, n: qmc.UInt, m: qmc.UInt) -> qmc.Float:
    """Measure a fixed-point register after selecting its source allocation.

    Args:
        flag (qmc.UInt): Select the n-sized branch when equal to one.
        n (qmc.UInt): True-branch carrier count.
        m (qmc.UInt): False-branch carrier count.

    Returns:
        qmc.Float: Measured fixed-point value.
    """
    if flag == 1:
        register = qmc.qubit_array(n, "register")
    else:
        register = qmc.qubit_array(m, "register")
    return qmc.measure(qmc.cast(register, qmc.QFixed))


@qmc.qkernel
def _symbolic_vector(flag: qmc.UInt, n: qmc.UInt, m: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Measure an array after selecting its allocation.

    Args:
        flag (qmc.UInt): Select the n-sized branch when equal to one.
        n (qmc.UInt): True-branch carrier count.
        m (qmc.UInt): False-branch carrier count.

    Returns:
        qmc.Vector[qmc.Bit]: Measured selected carriers.
    """
    if flag == 1:
        register = qmc.qubit_array(n, "register")
    else:
        register = qmc.qubit_array(m, "register")
    return qmc.measure(register)


@qmc.qkernel
def _static_qint(flag: qmc.UInt) -> qmc.UInt:
    """Measure an integer selected from four- and two-carrier allocations.

    Args:
        flag (qmc.UInt): Select four carriers when equal to one, otherwise two.

    Returns:
        qmc.UInt: Measured unsigned integer.
    """
    if flag == 1:
        register = qmc.qubit_array(4, "register")
    else:
        register = qmc.qubit_array(2, "register")
    return qmc.measure(qmc.cast(register, qmc.QInt))


@qmc.qkernel
def _static_qfixed(flag: qmc.UInt) -> qmc.Float:
    """Measure a fixed point selected from four- and two-carrier allocations.

    Args:
        flag (qmc.UInt): Select four carriers when equal to one, otherwise two.

    Returns:
        qmc.Float: Measured fixed-point value.
    """
    if flag == 1:
        register = qmc.qubit_array(4, "register")
    else:
        register = qmc.qubit_array(2, "register")
    return qmc.measure(qmc.cast(register, qmc.QFixed, int_bits=1))


@qmc.qkernel
def _static_vector(flag: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Measure an array selected from four- and two-carrier allocations.

    Args:
        flag (qmc.UInt): Select four carriers when equal to one, otherwise two.

    Returns:
        qmc.Vector[qmc.Bit]: Measured selected carriers.
    """
    if flag == 1:
        register = qmc.qubit_array(4, "register")
    else:
        register = qmc.qubit_array(2, "register")
    return qmc.measure(register)


@pytest.mark.parametrize("kernel", [_symbolic_qint, _symbolic_qfixed, _symbolic_vector])
@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("flag, expected", [(1, 4), (0, 2)])
def test_symbolic_quantum_merge_measurement_width(kernel, serialized, flag, expected):
    """Both input specialization paths retain the selected measurement width."""
    source = deserialize(serialize(kernel)) if serialized else kernel
    inputs = {"flag": flag, "n": 4, "m": 2}
    direct = qmc.estimate_resources(source, inputs=inputs)
    symbolic = qmc.estimate_resources(source).substitute(**inputs)
    for estimate in (direct, symbolic):
        assert estimate.measurements.total == expected
        assert estimate.width.allocated_qubits == expected
        assert estimate.width.peak_qubits == expected


@pytest.mark.parametrize("kernel", [_static_qint, _static_qfixed, _static_vector])
@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("flag, expected", [(1, 4), (0, 2)])
def test_static_quantum_merge_measurement_width(kernel, serialized, flag, expected):
    """Static differing branch widths are estimated from the selected source."""
    source = deserialize(serialize(kernel)) if serialized else kernel
    direct = qmc.estimate_resources(source, inputs={"flag": flag})
    symbolic = qmc.estimate_resources(source).substitute(flag=flag)
    for estimate in (direct, symbolic):
        assert estimate.measurements.total == expected
        assert estimate.width.allocated_qubits == expected
        assert estimate.width.peak_qubits == expected


@qmc.qkernel
def _merge_preserves_input_n(
    flag: qmc.UInt, n: qmc.UInt, m: qmc.UInt
) -> tuple[qmc.UInt, qmc.Vector[qmc.Bit]]:
    """Reuse n after consuming a result whose shape aliases the n input.

    Args:
        flag (qmc.UInt): Select the n-sized branch when equal to one.
        n (qmc.UInt): True-branch and independent allocation carrier count.
        m (qmc.UInt): False-branch carrier count.

    Returns:
        tuple[qmc.UInt, qmc.Vector[qmc.Bit]]: Selected integer and independent bits.
    """
    if flag == 1:
        register = qmc.qubit_array(n, "register")
    else:
        register = qmc.qubit_array(m, "register")
    measured = qmc.measure(qmc.cast(register, qmc.QInt))
    independent = qmc.qubit_array(n, "independent")
    return measured, qmc.measure(independent)


@pytest.mark.parametrize("flag, expected", [(1, 8), (0, 6)])
def test_quantum_merge_does_not_rebind_input_dimension(flag, expected):
    """A selected m-sized result must not change later uses of the input n."""
    inputs = {"flag": flag, "n": 4, "m": 2}
    for source in (
        _merge_preserves_input_n,
        deserialize(serialize(_merge_preserves_input_n)),
    ):
        for estimate in (
            qmc.estimate_resources(source, inputs=inputs),
            qmc.estimate_resources(source).substitute(**inputs),
        ):
            assert estimate.measurements.total == expected
            assert estimate.width.allocated_qubits == expected


@pytest.mark.parametrize("flag, expected", [(1, 8), (0, 6)])
def test_raw_quantum_merge_with_aliased_shape_preserves_n(flag, expected):
    """Raw IR may alias n as the result dimension and must retain n's value."""
    import dataclasses

    from qamomile.circuit.ir.operation.control_flow import IfOperation
    from qamomile.circuit.ir.value import ArrayValue
    from qamomile.circuit.ir.value_mapping import ValueSubstitutor

    block = _merge_preserves_input_n.build()
    branch = next(op for op in block.operations if isinstance(op, IfOperation))
    merge = next(
        merge for merge in branch.iter_merges() if isinstance(merge.result, ArrayValue)
    )
    aliased = dataclasses.replace(merge.result, shape=merge.true_value.shape)
    assert aliased.shape[0] is merge.true_value.shape[0]
    substitutor = ValueSubstitutor({merge.result.uuid: aliased})
    block = dataclasses.replace(
        block,
        operations=[substitutor.substitute_operation(op) for op in block.operations],
    )
    inputs = {"flag": flag, "n": 4, "m": 2}
    for estimate in (
        qmc.estimate_resources(block, inputs=inputs),
        qmc.estimate_resources(block).substitute(**inputs),
    ):
        assert estimate.measurements.total == expected
        assert estimate.width.allocated_qubits == expected


@qmc.qkernel
def _pack_register(register: qmc.Vector[qmc.Qubit]) -> qmc.QInt:
    """Pack a caller-selected register without changing its carrier count.

    Args:
        register (qmc.Vector[qmc.Qubit]): Selected source register.

    Returns:
        qmc.QInt: Packed source register.
    """
    return qmc.cast(register, qmc.QInt)


@qmc.qkernel
def _nested_whole_vector_call(
    flag: qmc.UInt, inner: qmc.UInt, n: qmc.UInt, m: qmc.UInt
) -> qmc.UInt:
    """Propagate a nested selection through whole-vector X and a packed call.

    Args:
        flag (qmc.UInt): Select the nested branch when equal to one.
        inner (qmc.UInt): Select n carriers when equal to one, otherwise m.
        n (qmc.UInt): Inner true-branch carrier count.
        m (qmc.UInt): Inner false-branch carrier count.

    Returns:
        qmc.UInt: Measured integer after applying X to each selected carrier.
    """
    if flag == 1:
        if inner == 1:
            register = qmc.qubit_array(n, "register")
        else:
            register = qmc.qubit_array(m, "register")
    else:
        register = qmc.qubit_array(3, "register")
    register = qmc.x(register)
    return qmc.measure(_pack_register(register))


@pytest.mark.parametrize("flag, inner, expected", [(1, 1, 4), (1, 0, 2), (0, 0, 3)])
def test_selected_quantum_width_survives_nested_vector_call(flag, inner, expected):
    """Selected widths govern whole-vector work and the called packed result."""
    inputs = {"flag": flag, "inner": inner, "n": 4, "m": 2}
    for source in (
        _nested_whole_vector_call,
        deserialize(serialize(_nested_whole_vector_call)),
    ):
        for estimate in (
            qmc.estimate_resources(source, inputs=inputs),
            qmc.estimate_resources(source).substitute(**inputs),
        ):
            assert estimate.measurements.total == expected
            assert estimate.gates.total == expected
            assert estimate.width.allocated_qubits == expected
            assert estimate.width.peak_qubits == expected
