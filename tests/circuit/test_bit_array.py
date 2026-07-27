"""Tests for fixed-length first-class ``Vector[Bit]`` values."""

from typing import Any

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.classical_ops import StoreArrayElementOperation
from qamomile.circuit.transpiler.errors import ValidationError


@qmc.qkernel
def collect_measurements() -> qmc.Vector[qmc.Bit]:
    """Collect a non-palindromic measurement pattern into a Bit array.

    Returns:
        qmc.Vector[qmc.Bit]: Stored measurement results in qubit order.
    """
    qubits = qmc.qubit_array(3, "qubits")
    qubits[0] = qmc.x(qubits[0])
    qubits[1] = qmc.x(qubits[1])
    measured = qmc.measure(qubits)
    output = qmc.bit_array(3, name="output")
    output[0] = measured[0]
    output[1] = measured[1]
    output[2] = measured[2]
    return output


def test_bit_array_store_return_and_order_across_backends(
    sdk_transpiler: Any,
) -> None:
    """Measured values preserve element order on every installed backend."""
    name = sdk_transpiler.backend_name
    transpiler = sdk_transpiler.transpiler
    executor = transpiler.executor()
    executable = transpiler.transpile(collect_measurements)

    sampled = executable.sample(executor, shots=64).result()
    counts = {tuple(int(bit) for bit in bits): count for bits, count in sampled.results}
    assert counts == {(1, 1, 0): 64}, f"{name}: got {counts}"

    single = executable.run(executor).result()
    assert tuple(int(bit) for bit in single) == (1, 1, 0)


@qmc.qkernel
def return_bound_bits(
    bits: qmc.Vector[qmc.Bit],
) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    """Return a bound Bit vector together with a one-bit measurement.

    Args:
        bits (qmc.Vector[qmc.Bit]): Compile-time bound bit vector.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]: Bound vector and
        one-bit measurement vector.
    """
    qubit = qmc.qubit_array(1, "qubit")
    return bits, qmc.measure(qubit)


def test_vector_bit_accepts_compile_time_input_binding(qiskit_transpiler: Any) -> None:
    """A fixed ``Vector[Bit]`` is accepted as a qkernel input and output."""
    executable = qiskit_transpiler.transpile(
        return_bound_bits,
        bindings={"bits": [True, 0, 1]},
    )

    bits, measured = executable.run(qiskit_transpiler.executor()).result()
    assert tuple(bool(bit) for bit in bits) == (True, False, True)
    assert tuple(int(bit) for bit in measured) == (0,)


@pytest.mark.parametrize(
    ("binding", "error_type", "message"),
    [
        ([0, "1"], TypeError, "must be bool, 0, or 1"),
        ([0, 2], ValueError, "must be 0 or 1"),
    ],
)
def test_vector_bit_rejects_invalid_element_bindings(
    binding: list[object],
    error_type: type[Exception],
    message: str,
    qiskit_transpiler: Any,
) -> None:
    """Bit vector bindings reject wrong element types and values."""
    with pytest.raises(error_type, match=message):
        qiskit_transpiler.transpile(
            return_bound_bits,
            bindings={"bits": binding},
        )


@pytest.mark.parametrize(
    ("shape", "error_type", "message"),
    [
        (-1, ValueError, "non-negative"),
        ((1, 2), NotImplementedError, "rank-2"),
        (True, TypeError, "shape must be"),
    ],
)
def test_bit_array_rejects_invalid_shapes(
    shape: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """The constructor rejects negative, multi-rank, and boolean shapes."""

    @qmc.qkernel
    def invalid() -> qmc.Vector[qmc.Bit]:
        return qmc.bit_array(shape)  # type: ignore[arg-type]

    with pytest.raises(error_type, match=message):
        invalid.build()


def test_bit_array_rejects_symbolic_runtime_length() -> None:
    """A runtime UInt cannot size a fixed-length Bit array."""

    @qmc.qkernel
    def symbolic(size: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        return qmc.bit_array(size)

    with pytest.raises(ValueError, match="known at trace time"):
        symbolic.build(parameters=["size"])


def test_bit_array_rejects_out_of_range_store() -> None:
    """A constant store index outside the fixed length raises immediately."""

    @qmc.qkernel
    def out_of_range() -> qmc.Vector[qmc.Bit]:
        bits = qmc.bit_array(2)
        bits[2] = 1
        return bits

    with pytest.raises(IndexError, match="out of range"):
        out_of_range.build()


def test_bit_array_rejects_wrong_stored_handle_type() -> None:
    """A non-Bit handle cannot be assigned to a Bit array element."""

    @qmc.qkernel
    def wrong_type() -> qmc.Vector[qmc.Bit]:
        bits = qmc.bit_array(1)
        bits[0] = qmc.uint(1)  # type: ignore[assignment]
        return bits

    with pytest.raises(TypeError, match="Cannot assign UInt"):
        wrong_type.build()


def test_stored_bit_array_condition_fails_loudly(qiskit_transpiler: Any) -> None:
    """Reading an assigned slot for feed-forward is rejected in v1."""

    @qmc.qkernel
    def unsupported_feed_forward() -> qmc.Bit:
        source = qmc.qubit("source")
        target = qmc.qubit("target")
        measured = qmc.measure(source)
        bits = qmc.bit_array(1)
        bits[0] = measured
        if bits[0]:
            target = qmc.x(target)
        return qmc.measure(target)

    with pytest.raises(
        ValidationError,
        match="storing and returning values only",
    ):
        qiskit_transpiler.transpile(unsupported_feed_forward)


def test_bit_array_store_ir_keeps_source_and_destination_index(
    qiskit_transpiler: Any,
) -> None:
    """Each store records exact measurement provenance for one array slot."""
    block = qiskit_transpiler.inline(qiskit_transpiler.to_block(collect_measurements))
    stores = [
        operation
        for operation in block.operations
        if isinstance(operation, StoreArrayElementOperation)
    ]

    assert len(stores) == 3
    assert [int(store.index_values[0].get_const()) for store in stores] == [0, 1, 2]
    assert all(store.stored_value.parent_array is not None for store in stores)
