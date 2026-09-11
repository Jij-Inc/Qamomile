"""End-to-end tests for the Amazon Braket transpiler."""

import math

import numpy as np
import pytest

pytestmark = pytest.mark.braket
pytest.importorskip("braket.circuits")

import qamomile.circuit as qmc  # noqa: E402
import qamomile.observable as qm_o  # noqa: E402
from qamomile.braket import BraketTranspiler  # noqa: E402


@qmc.qkernel
def _parameterized_bell(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Prepare and measure a parameterized Bell-like state.

    Args:
        theta (qmc.Float): Rotation angle.

    Returns:
        qmc.Vector[qmc.Bit]: Two measured output bits.
    """
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.ry(q[0], theta)
    q[0], q[1] = qmc.cx(q[0], q[1])
    return qmc.measure(q)


@qmc.qkernel
def _plus_expectation(observable: qmc.Observable) -> qmc.Float:
    """Prepare a plus state and return an observable expectation value.

    Args:
        observable (qmc.Observable): Observable to evaluate.

    Returns:
        qmc.Float: Observable expectation value.
    """
    q = qmc.qubit_array(1, "q")
    q[0] = qmc.h(q[0])
    return qmc.expval(q, observable)


@qmc.qkernel
def _indexed_parameter(phases: qmc.Vector[qmc.Float]) -> qmc.Bit:
    """Rotate with an indexed runtime parameter.

    Args:
        phases (qmc.Vector[qmc.Float]): Runtime rotation angles.

    Returns:
        qmc.Bit: Measured qubit value.
    """
    q = qmc.qubit("q")
    q = qmc.rx(q, phases[0])
    return qmc.measure(q)


@pytest.mark.parametrize("theta", [0.0, math.pi, 2 * math.pi])
def test_parameterized_sampling_executes(theta: float) -> None:
    """Runtime parameters survive transpilation and bind before sampling."""
    transpiler = BraketTranspiler()
    executable = transpiler.transpile(_parameterized_bell, parameters=["theta"])

    result = executable.sample(
        transpiler.executor(),
        shots=32,
        bindings={"theta": theta},
    ).result()

    expected = (
        (0, 0)
        if np.isclose(theta % (2 * math.pi), 0.0, rtol=0.0, atol=1e-12)
        else (1, 1)
    )
    assert result.results == [(expected, 32)]


def test_expectation_execution_uses_native_braket_circuit() -> None:
    """A transpiled expectation program executes on LocalSimulator."""
    transpiler = BraketTranspiler()
    executable = transpiler.transpile(
        _plus_expectation,
        bindings={"observable": qm_o.X(0)},
    )

    result = executable.run(transpiler.executor()).result()

    assert np.isclose(result, 1.0, rtol=0.0, atol=1e-10)
    assert type(executable.quantum_circuit).__module__.startswith("braket.")


def test_indexed_parameter_uses_openqasm_safe_native_name() -> None:
    """Indexed public names remain native inputs without QASM ambiguity."""
    transpiler = BraketTranspiler()
    executable = transpiler.transpile(_indexed_parameter, parameters=["phases"])
    parameter = executable.compiled_quantum[0].parameter_metadata.parameters[0]

    assert parameter.name == "phases[0]"
    assert str(parameter.engine_param).startswith("_qamomile_parameter_")
    result = executable.run(
        transpiler.executor(),
        bindings={"phases": [math.pi]},
    ).result()
    assert result == 1
