"""Exercise physical qubit ownership when HUGR measurements consume array views."""

from pathlib import Path

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.transpiler import QKernelLike
from qamomile.hugr import HugrExecutor, HugrTranspiler, SeleneExecutionOptions

pytestmark = pytest.mark.hugr


@qmc.qkernel
def _measure_strided(flipped: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Measure odd root slots and leave the other qubits for automatic cleanup.

    Args:
        flipped (qmc.UInt): Root index prepared in the one state.

    Returns:
        qmc.Vector[qmc.Bit]: Measurements of root slots 1, 3, 5, and 7.
    """
    qubits = qmc.qubit_array(8, "qubits")
    qubits[flipped] = qmc.x(qubits[flipped])
    return qmc.measure(qubits[1::2])


@qmc.qkernel
def _measure_gated_strided(flipped: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Measure a view after its physical wires advance through broadcast gates.

    Args:
        flipped (qmc.UInt): Root index prepared in the one state before gates.

    Returns:
        qmc.Vector[qmc.Bit]: Inverted measurements of root slots 1, 3, 5, and 7.
    """
    qubits = qmc.qubit_array(8, "qubits")
    qubits[flipped] = qmc.x(qubits[flipped])
    view = qubits[1::2]
    view = qmc.x(view)
    return qmc.measure(view)


@qmc.qkernel
def _measure_nested_strided(flipped: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Measure a nested view using the composition of its two physical strides.

    Args:
        flipped (qmc.UInt): Root index prepared in the one state.

    Returns:
        qmc.Vector[qmc.Bit]: Measurements of root slots 3 and 7.
    """
    qubits = qmc.qubit_array(8, "qubits")
    qubits[flipped] = qmc.x(qubits[flipped])
    view = qubits[1::2]
    return qmc.measure(view[1::2])


@qmc.qkernel
def _measure_strided_and_remaining(
    flipped: qmc.UInt,
) -> tuple[qmc.Vector[qmc.Bit], qmc.Bit]:
    """Measure a disjoint root element after consuming a strided array view.

    Args:
        flipped (qmc.UInt): Root index prepared in the one state.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.Bit]: Odd root slots followed by root slot 0.
    """
    qubits = qmc.qubit_array(8, "qubits")
    qubits[flipped] = qmc.x(qubits[flipped])
    measured = qmc.measure(qubits[1::2])
    return measured, qmc.measure(qubits[0])


@qmc.qkernel
def _measure_scalar_view(flipped: qmc.UInt) -> tuple[qmc.Bit, qmc.Bit]:
    """Measure one view element and one disjoint root element.

    Args:
        flipped (qmc.UInt): Root index prepared in the one state.

    Returns:
        tuple[qmc.Bit, qmc.Bit]: Measurements of root slots 3 and 0.
    """
    qubits = qmc.qubit_array(8, "qubits")
    qubits[flipped] = qmc.x(qubits[flipped])
    view = qubits[1::2]
    return qmc.measure(view[1]), qmc.measure(qubits[0])


@pytest.mark.parametrize("flipped", [0, 3, 7])
@pytest.mark.parametrize(
    "kernel,indices,inverted,remaining",
    [
        (_measure_strided, (1, 3, 5, 7), False, False),
        (_measure_gated_strided, (1, 3, 5, 7), True, False),
        (_measure_nested_strided, (3, 7), False, False),
        (_measure_strided_and_remaining, (1, 3, 5, 7), False, True),
    ],
)
def test_slice_measurement_retires_root_aliases(
    tmp_path: Path,
    kernel: QKernelLike,
    indices: tuple[int, ...],
    inverted: bool,
    remaining: bool,
    flipped: int,
) -> None:
    """Native validation and Selene preserve view order without freeing measured roots twice."""
    executable = HugrTranspiler().transpile(kernel, bindings={"flipped": flipped})
    executor = HugrExecutor(options=SeleneExecutionOptions(build_dir=tmp_path, seed=37))
    # Exactly the flipped root starts in |1>; a broadcast X inverts every
    # measured view slot. Nested strides select physical roots 3 and 7.
    bits = tuple((flipped == index) ^ inverted for index in indices)
    expected = (bits, flipped == 0) if remaining else bits
    assert executable.run(executor).result() == expected
    sampled = executable.sample(executor, shots=3).result()
    assert sampled.shots == 3
    assert sampled.results == [(expected, 3)]


@pytest.mark.parametrize("flipped", [0, 3, 7])
def test_scalar_slice_measurement_preserves_disjoint_root_liveness(
    tmp_path: Path, flipped: int
) -> None:
    """Scalar view measurements preserve disjoint measurements and free untouched roots once."""
    executable = HugrTranspiler().transpile(
        _measure_scalar_view, bindings={"flipped": flipped}
    )
    executor = HugrExecutor(options=SeleneExecutionOptions(build_dir=tmp_path, seed=41))
    # Only the selected root is |1>, while view[1] addresses physical root 3.
    expected = (flipped == 3, flipped == 0)
    assert executable.run(executor).result() == expected
    sampled = executable.sample(executor, shots=3).result()
    assert sampled.shots == 3
    assert sampled.results == [(expected, 3)]
