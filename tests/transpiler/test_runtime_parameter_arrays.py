"""Runtime binding tests for multidimensional classical parameters."""

from __future__ import annotations

import numpy as np
import pytest

import qamomile.circuit as qmc


@qmc.qkernel
def _matrix_rotation_kernel(
    angles: qmc.Matrix[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    """Rotate two qubits using the first matrix column."""
    q = qmc.qubit_array(2, name="q")
    for index in qmc.range(2):
        q[index] = qmc.rx(q[index], angles[index, 0])
    return qmc.measure(q)


@qmc.qkernel
def _reverse_array_use_kernel(angles: qmc.Vector[qmc.Float]) -> qmc.Bit:
    """Use vector elements in reverse order to exercise parameter ABI ordering."""
    qubit = qmc.qubit("q")
    qubit = qmc.rx(qubit, angles[1])
    qubit = qmc.ry(qubit, angles[0])
    return qmc.measure(qubit)


@pytest.mark.parametrize(
    "binding",
    [
        [[0.0, 0.25], [np.pi, 0.5]],
        np.array([[0.0, 0.25], [np.pi, 0.5]]),
    ],
)
def test_matrix_runtime_binding_flattens_to_scalar_parameters(binding) -> None:
    """Nested lists and ndarrays bind Matrix elements by indexed name."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(_matrix_rotation_kernel, parameters=["angles"])
    array_info = executable.compiled_quantum[0].parameter_metadata.arrays["angles"]

    result = executable.sample(
        transpiler.executor(),
        shots=16,
        bindings={"angles": binding},
    ).result()

    assert array_info.rank == 2
    assert array_info.expected_shape == (None, None)
    assert result.results == [((0, 1), 16)]


def test_array_parameter_metadata_uses_index_order() -> None:
    """Positional backend ABIs follow array-index order, not first use."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    executable = QiskitTranspiler().transpile(
        _reverse_array_use_kernel,
        parameters=["angles"],
    )

    names = [
        parameter.name
        for parameter in executable.compiled_quantum[0].parameter_metadata.parameters
    ]
    array_info = executable.compiled_quantum[0].parameter_metadata.arrays["angles"]

    assert names == ["angles[0]", "angles[1]"]
    assert array_info.rank == 1
    assert array_info.expected_shape == (2,)


def test_overlong_vector_runtime_binding_is_rejected() -> None:
    """Extra vector elements must not be silently ignored at execution."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(
        _reverse_array_use_kernel,
        parameters=["angles"],
    )

    with pytest.raises(ValueError, match="beyond the emitted shape"):
        executable.sample(
            transpiler.executor(),
            shots=1,
            bindings={"angles": [0.0, 0.0, 0.0, 0.0]},
        )
