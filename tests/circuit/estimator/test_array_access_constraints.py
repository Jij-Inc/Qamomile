"""Structural array-access requirements for logical resource estimation."""

import pytest

import qamomile.circuit as qm


@qm.qkernel
def _direct_array_access(
    length: qm.UInt,
    index: qm.UInt,
) -> qm.Vector[qm.Qubit]:
    """Apply one gate to a dynamically selected array element."""
    register = qm.qubit_array(length, "register")
    target = register[index]
    target = qm.h(target)
    register[index] = target
    return register


@qm.qkernel
def _computed_negative_access(index: qm.UInt) -> qm.Vector[qm.Qubit]:
    """Address a fixed register with an index that can become negative."""
    register = qm.qubit_array(2, "register")
    position = index - 1
    target = register[position]
    target = qm.h(target)
    register[position] = target
    return register


@qm.qkernel
def _loop_array_access(length: qm.UInt) -> qm.Vector[qm.Qubit]:
    """Address one slot beyond the loop index on every iteration."""
    register = qm.qubit_array(length, "register")
    for index in qm.range(length):
        position = index + 1
        target = register[position]
        target = qm.h(target)
        register[position] = target
    return register


@qm.qkernel
def _valid_loop_array_access(length: qm.UInt) -> qm.Vector[qm.Qubit]:
    """Address exactly the slots covered by a symbolic loop range."""
    register = qm.qubit_array(length, "register")
    for index in qm.range(length):
        target = register[index]
        target = qm.h(target)
        register[index] = target
    return register


@qm.composite_gate(name="indexed_h_for_resource_estimation")
def _indexed_h(
    register: qm.Vector[qm.Qubit],
    index: qm.UInt,
) -> qm.Vector[qm.Qubit]:
    """Apply one Hadamard gate at a supplied vector index."""
    target = register[index]
    target = qm.h(target)
    register[index] = target
    return register


@qm.qkernel
def _invoke_array_access(
    length: qm.UInt,
    index: qm.UInt,
) -> qm.Vector[qm.Qubit]:
    """Invoke an indexed body-backed callable directly."""
    register = qm.qubit_array(length, "register")
    return _indexed_h(register, index)


@qm.qkernel
def _control_array_access(
    length: qm.UInt,
    index: qm.UInt,
) -> tuple[qm.Qubit, qm.Vector[qm.Qubit]]:
    """Invoke an indexed body-backed callable under coherent control."""
    control = qm.qubit("control")
    register = qm.qubit_array(length, "register")
    control, register = qm.control(_indexed_h)(control, register, index)
    return control, register


@qm.qkernel
def _inverse_array_access(
    length: qm.UInt,
    index: qm.UInt,
) -> qm.Vector[qm.Qubit]:
    """Invoke the inverse of an indexed body-backed callable."""
    register = qm.qubit_array(length, "register")
    return qm.inverse(_indexed_h)(register, index)


@qm.qkernel
def _compile_time_branch_access(
    length: qm.UInt,
    index: qm.UInt,
    flag: qm.Bit,
) -> qm.Vector[qm.Qubit]:
    """Use a dynamic index only in one compile-time branch."""
    register = qm.qubit_array(length, "register")
    if flag:
        target = register[index]
        target = qm.h(target)
        register[index] = target
    else:
        target = register[0]
        target = qm.x(target)
        register[0] = target
    return register


@qm.qkernel
def _runtime_branch_access(
    length: qm.UInt,
    index: qm.UInt,
) -> qm.Vector[qm.Qubit]:
    """Use a dynamic index in one measurement-selected branch."""
    register = qm.qubit_array(length, "register")
    predicate = qm.measure(qm.qubit("predicate"))
    if predicate:
        target = register[index]
        target = qm.h(target)
        register[index] = target
    else:
        target = register[0]
        target = qm.x(target)
        register[0] = target
    return register


@qm.qkernel
def _out_of_bounds_view(length: qm.UInt) -> qm.Vector[qm.Bit]:
    """Measure a symbolic view whose declared coverage exceeds its parent."""
    register = qm.qubit_array(length, "register")
    return qm.measure(register[0 : length + 1])


@qm.qkernel
def _empty_view(length: qm.UInt) -> qm.Vector[qm.Bit]:
    """Measure an empty view positioned at the parent array boundary."""
    register = qm.qubit_array(length, "register")
    return qm.measure(register[length:length])


@qm.qkernel
def _matrix_access(
    values: qm.Matrix[qm.Float],
    row: qm.UInt,
    column: qm.UInt,
) -> qm.Qubit:
    """Use a two-dimensional classical array element as a gate angle."""
    return qm.rx(qm.qubit("target"), values[row, column])


@qm.qkernel
def _classical_store(index: qm.UInt) -> qm.Vector[qm.Bit]:
    """Store a literal into a dynamically selected classical array slot."""
    bits = qm.measure(qm.qubit_array(1, "register"))
    bits[index] = True
    return bits


@qm.qkernel
def _array_content_index(
    indices: qm.Vector[qm.UInt],
) -> qm.Vector[qm.Qubit]:
    """Use one input-array element as a quantum register index."""
    register = qm.qubit_array(indices.shape[0], "register")
    position = indices[1]
    target = register[position]
    target = qm.h(target)
    register[position] = target
    return register


def test_direct_array_index_constraints_validate_inputs_and_substitution() -> None:
    """Direct symbolic accesses reject upper bounds through both public paths."""
    valid = _direct_array_access.estimate_resources(inputs={"length": 3, "index": 2})
    assert valid.gates.total == 1

    with pytest.raises(ValueError, match="element index 0 upper bound"):
        _direct_array_access.estimate_resources(inputs={"length": 3, "index": 3})

    symbolic = _direct_array_access.estimate_resources()
    with pytest.raises(ValueError, match="element index 0 upper bound"):
        symbolic.substitute(length=3, index=3)


def test_computed_index_constraint_rejects_negative_result() -> None:
    """A nonnegative UInt input can still produce an invalid negative index."""
    with pytest.raises(ValueError, match="element index 0 lower bound"):
        _computed_negative_access.estimate_resources(inputs={"index": 0})

    with pytest.raises(ValueError, match="element index 0 lower bound"):
        _computed_negative_access.estimate_resources().substitute(index=0)


def test_loop_index_constraints_are_quantified_over_executed_iterations() -> None:
    """Loop requirements reject the final OOB access but allow zero trips."""
    empty = _loop_array_access.estimate_resources(inputs={"length": 0})
    assert empty.gates.total == 0

    with pytest.raises(ValueError, match="element index 0 upper bound"):
        _loop_array_access.estimate_resources(inputs={"length": 3})

    valid = _valid_loop_array_access.estimate_resources(inputs={"length": 3})
    assert valid.gates.total == 3


@pytest.mark.parametrize(
    "kernel",
    [_invoke_array_access, _control_array_access, _inverse_array_access],
    ids=["invoke", "control", "inverse"],
)
def test_body_backed_transforms_preserve_array_constraints(kernel: object) -> None:
    """Direct, controlled, and inverse body traversal retain index bounds."""
    with pytest.raises(ValueError, match="element index 0 upper bound"):
        kernel.estimate_resources(inputs={"length": 3, "index": 3})  # type: ignore[attr-defined]


def test_compile_time_branch_validates_only_the_selected_access() -> None:
    """An invalid index in an untaken static branch does not reject estimation."""
    untaken = _compile_time_branch_access.estimate_resources(
        inputs={"length": 3, "index": 3, "flag": 0}
    )
    assert untaken.gates.total == 1

    with pytest.raises(ValueError, match="element index 0 upper bound"):
        _compile_time_branch_access.estimate_resources(
            inputs={"length": 3, "index": 3, "flag": 1}
        )


def test_runtime_branch_validates_every_possible_access() -> None:
    """A measurement-selected branch keeps both branches' structural bounds."""
    with pytest.raises(ValueError, match="element index 0 upper bound"):
        _runtime_branch_access.estimate_resources(inputs={"length": 3, "index": 3})


def test_whole_view_constraint_checks_parent_coverage_and_empty_views() -> None:
    """Whole-array operations validate view coverage while allowing no slots."""
    with pytest.raises(ValueError, match="parent coverage"):
        _out_of_bounds_view.estimate_resources(inputs={"length": 3})

    empty = _empty_view.estimate_resources(inputs={"length": 3})
    assert empty.gates.total == 0
    assert empty.width.circuit_qubits == 3


def test_multidimensional_array_constraints_validate_each_axis() -> None:
    """Matrix accesses bind concrete shapes and reject either overflowing axis."""
    values = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    valid = _matrix_access.estimate_resources(
        inputs={"values": values, "row": 1, "column": 2}
    )
    assert valid.gates.total == 1

    with pytest.raises(ValueError, match="element index 0 upper bound"):
        _matrix_access.estimate_resources(
            inputs={"values": values, "row": 2, "column": 0}
        )
    with pytest.raises(ValueError, match="element index 1 upper bound"):
        _matrix_access.estimate_resources(
            inputs={"values": values, "row": 0, "column": 3}
        )


def test_classical_store_uses_its_separate_index_operands() -> None:
    """A StoreArrayElementOperation validates indices not embedded in Values."""
    valid = _classical_store.estimate_resources(inputs={"index": 0})
    assert valid.width.circuit_qubits == 1

    with pytest.raises(ValueError, match="store index 0 upper bound"):
        _classical_store.estimate_resources(inputs={"index": 1})


@pytest.mark.parametrize(
    ("target_index", "source_index", "role"), [(2, 0, "target"), (0, 2, "source")]
)
def test_deferred_quantum_return_validates_both_index_layouts(
    target_index: int,
    source_index: int,
    role: str,
) -> None:
    """Deferred quantum returns validate separately encoded target and source indices."""
    from qamomile.circuit.ir.operation.classical_ops import (
        ReturnQuantumArrayElementOperation,
    )
    from qamomile.circuit.ir.types import QubitType, UIntType
    from qamomile.circuit.ir.value import ArrayValue, Value

    extent = Value(type=UIntType(), name="extent").with_const(2)
    array = ArrayValue(type=QubitType(), name="register", shape=(extent,))
    returned = Value(type=QubitType(), name="returned")
    target = Value(type=UIntType(), name="target").with_const(target_index)
    source = Value(type=UIntType(), name="source").with_const(source_index)
    operation = ReturnQuantumArrayElementOperation(
        operands=[array, returned, target, source],
        results=[],
    )

    with pytest.raises(ValueError, match=rf"return {role} index 0 upper bound"):
        qm.estimate_resources([operation])


def test_array_content_index_remains_an_explicit_symbolic_requirement() -> None:
    """Concrete array contents remain symbolic while their shape is specialized."""
    estimate = _array_content_index.estimate_resources(inputs={"indices": [0, 2]})

    assert set(estimate.parameters) == {"indices[1]"}
    assert any(
        "indices[1]" in requirement["expression"]
        for requirement in estimate.to_dict()["requirements"]
    )
    assert estimate.substitute(**{"indices[1]": 1}).gates.total == 1
    with pytest.raises(ValueError, match="element index 0 upper bound"):
        estimate.substitute(**{"indices[1]": 2})
