"""Compile-bound branch dimensions remain concrete for Python helpers."""

from __future__ import annotations

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.handle.utils import get_size
from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.ir.value import ArrayValue
from qamomile.circuit.serialization import deserialize, serialize


@qmc.qkernel
def _builtin_width(flag: qmc.UInt) -> qmc.UInt:
    """Return the selected literal width through a Python size helper."""
    if flag == 1:
        register = qmc.qubit_array(4, "large")
    else:
        register = qmc.qubit_array(3, "small")
    return qmc.uint(get_size(register))


def _flip_last(register: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Use an ordinary Python loop over the selected concrete width."""
    for index in range(get_size(register)):
        if index == get_size(register) - 1:
            register[index] = qmc.x(register[index])
    return register


@qmc.qkernel
def _python_width_helper(flag: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Set the selected register's most significant bit with a Python helper."""
    if (flag == 1) & ~(flag == 0):
        register = qmc.qubit_array(4, "large")
    else:
        register = qmc.qubit_array(3, "small")
    register = _flip_last(register)
    return qmc.measure(register)


@qmc.qkernel
def _bound_width(flag: qmc.UInt, n: qmc.UInt, m: qmc.UInt) -> qmc.UInt:
    """Keep the unselected input size independent of the selected width."""
    if flag == 1:
        register = qmc.qubit_array(n, "large")
    else:
        register = qmc.qubit_array(m, "small")
    return n + get_size(register)


@qmc.qkernel
def _ir_width(flag: qmc.UInt, n: qmc.UInt, m: qmc.UInt) -> qmc.UInt:
    """Preserve a symbolic shape read for serialization and later binding."""
    if flag == 1:
        register = qmc.qubit_array(n, "large")
    else:
        register = qmc.qubit_array(m, "small")
    return n + register.shape[0]


@qmc.qkernel
def _modmul_width(flag: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Specialize modular multiplication using the selected array width."""
    if flag == 1:
        register = qmc.qubit_array(4, "large")
    else:
        register = qmc.qubit_array(3, "small")
    register[0] = qmc.x(register[0])
    register = qmc.modmul_const(register, multiplier=2, modulus=3)
    return qmc.measure(register)


@qmc.qkernel
def _dead_nested_cast(flag: qmc.UInt) -> qmc.Float:
    """Leave an invalid nested cast symbolic when its outer branch is dead."""
    if flag == 1:
        live = qmc.qubit_array(3, "live")
        live[2] = qmc.x(live[2])
        result = qmc.measure(qmc.cast(live, qmc.QFixed, int_bits=3))
    else:
        if flag == 1:
            work = qmc.qubit_array(2, "unreachable")
        else:
            work = qmc.qubit_array(4, "reachable")
        work[0] = qmc.x(work[0])
        result = qmc.measure(qmc.cast(work, qmc.QFixed, int_bits=3))
    return result


@pytest.mark.parametrize("flag, width", [(1, 4), (0, 3)])
def test_python_size_helpers_use_selected_branch(qiskit_transpiler, flag, width):
    """Python size reads and Python loops observe both selected widths."""
    scalar = qiskit_transpiler.transpile(_builtin_width, bindings={"flag": flag})
    assert scalar.sample(qiskit_transpiler.executor(), shots=1).result().results == [
        (width, 1)
    ]
    measured = qiskit_transpiler.transpile(
        _python_width_helper, bindings={"flag": flag}
    )
    bits, shots = (
        measured.sample(qiskit_transpiler.executor(), shots=1).result().results[0]
    )
    assert tuple(bits) == (0,) * (width - 1) + (1,)
    assert shots == 1


@pytest.mark.parametrize("serialized", [False, True])
def test_selected_width_does_not_stale_or_rebind_inputs(qiskit_transpiler, serialized):
    """Fresh bindings select new widths without overwriting a shared n input."""
    kernel = deserialize(serialize(_ir_width)) if serialized else _bound_width
    for flag, n, m in [(1, 4, 3), (0, 4, 3), (0, 5, 2), (1, 3, 2)]:
        executable = qiskit_transpiler.transpile(
            kernel, bindings={"flag": flag, "n": n, "m": m}
        )
        expected = n + (n if flag else m)
        assert executable.sample(
            qiskit_transpiler.executor(), shots=1
        ).result().results == [(expected, 1)]


@pytest.mark.parametrize("flag, width", [(1, 4), (0, 3)])
def test_concrete_shape_keeps_independent_merge_provenance(flag, width):
    """Exposing a constant dimension retains its own selected SSA merge."""
    block = _bound_width.build(flag=flag, n=4, m=3)
    branch = next(op for op in block.operations if isinstance(op, IfOperation))
    array_merge = next(
        merge for merge in branch.iter_merges() if isinstance(merge.result, ArrayValue)
    )
    dimension = array_merge.result.shape[0]
    assert dimension.get_const() == width
    assert dimension.uuid not in {
        array_merge.true_value.shape[0].uuid,
        array_merge.false_value.shape[0].uuid,
    }
    shape_merge = next(
        merge for merge in branch.iter_merges() if merge.result.uuid == dimension.uuid
    )
    assert shape_merge.true_value.get_const() == 4
    assert shape_merge.false_value.get_const() == 3


@pytest.mark.parametrize("flag, width", [(1, 4), (0, 3)])
def test_modmul_specializes_selected_branch_width(qiskit_transpiler, flag, width):
    """The stdlib's concrete-width requirement accepts either bound branch."""
    executable = qiskit_transpiler.transpile(_modmul_width, bindings={"flag": flag})
    assert len(executable.compiled_quantum) == 1
    assert len(executable.output_values) == 1
    output = executable.output_values[0]
    assert isinstance(output, ArrayValue)
    assert output.shape[0].get_const() == width


def test_unbound_condition_does_not_fabricate_a_python_width(qiskit_transpiler):
    """Unknown predicates cannot choose one branch's fixed Python size."""
    with pytest.raises(ValueError, match="Array must have fixed size"):
        qiskit_transpiler.transpile(_builtin_width, parameters=["flag"])
    with pytest.raises(ValueError, match="flag"):
        qiskit_transpiler.transpile(
            _builtin_width, bindings={"flag": 1}, parameters=["flag"]
        )


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("flag, expected", [(1, 4.0), (0, 0.5)])
def test_unreachable_nested_shape_does_not_fail_eager_cast(
    qiskit_transpiler, serialized, flag, expected
):
    """Direct and restored kernels discard invalid casts in dead outer branches."""
    kernel = (
        deserialize(serialize(_dead_nested_cast)) if serialized else _dead_nested_cast
    )
    executable = qiskit_transpiler.transpile(kernel, bindings={"flag": flag})
    assert executable.sample(
        qiskit_transpiler.executor(), shots=1
    ).result().results == [(expected, 1)]
