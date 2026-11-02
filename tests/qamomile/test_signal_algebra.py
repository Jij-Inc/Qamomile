"""Independent full-complex-matrix witnesses for projected-map constructions."""

from math import acos

import numpy as np
import pytest

import qamomile as qm
from qamomile._native import compiler as native
from qamomile.stdlib import (
    Region,
    Response,
    Signal,
    amplify,
    estimate_success,
    inspect_schedule,
    interfere,
    response,
    sample,
    select,
    sequence,
)

_reflection_angle = 2 * acos(0.3)


@qm.qkernel
def reflection(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    state = qm.z(state, target=0)
    return qm.ry(state, _reflection_angle, target=0)


@qm.qkernel
def complex_process(state: qm.Q[qm.Bits[3]]) -> qm.Q[qm.Bits[3]]:
    state = qm.h(state, target=0)
    state = qm.ry(state, 0.71, target=1)
    state = qm.cx(state, control=1, target=2)
    state = qm.p(state, 0.37, target=2)
    state = qm.cx(state, control=2, target=0)
    return qm.ry(state, 0.41, target=0)


def selected_matrix(signal: Signal) -> np.ndarray:
    """Extract all selected columns, including phases and coordinate order."""
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    from qamomile.qiskit import Qiskit, to_circuit

    width = signal.ambient_width
    process = signal.process

    @qm.qkernel
    def full() -> qm.Q[qm.Bits[width]]:
        return process(qm.qubits(width))

    unitary = Operator(to_circuit(full, target=Qiskit())).data

    def indices(region):
        return [
            i
            for i in range(1 << width)
            if all(
                c == "*" or int(c) == ((i >> lane) & 1)
                for lane, c in enumerate(region.pattern)
            )
        ]

    return (
        signal.normalization
        * unitary[np.ix_(indices(signal.output_region), indices(signal.input_region))]
    )


def test_product_does_not_reuse_a_failed_workspace():
    a = select(reflection, input=Region("0*"), output=Region("0*"))
    np.testing.assert_allclose(selected_matrix(a @ a), 0.09 * np.eye(2), atol=2e-14)
    squared = select(
        sequence(reflection, reflection, width=2),
        input=Region("0*"),
        output=Region("0*"),
    )
    np.testing.assert_allclose(selected_matrix(squared), np.eye(2), atol=2e-14)


def test_rectangular_product_adjoint_and_asymmetric_embeddings():
    a = select(complex_process, input=Region("0**"), output=Region("01*"))
    matrix = selected_matrix(a)
    np.testing.assert_allclose(
        selected_matrix(a.adjoint()), matrix.conj().T, atol=2e-14
    )
    np.testing.assert_allclose(
        selected_matrix(a.adjoint() @ a), matrix.conj().T @ matrix, atol=3e-14
    )
    np.testing.assert_allclose(
        selected_matrix(a @ a.adjoint()), matrix @ matrix.conj().T, atol=3e-14
    )


def test_interference_aligns_rectangular_layouts_and_scales():
    a = select(complex_process, input=Region("0**"), output=Region("01*"))
    b = select(complex_process, input=Region("**1"), output=Region("1*0"))
    a = Signal._checked(a.process, a.input_region, a.output_region, 2.0, 0.01)
    b = Signal._checked(b.process, b.input_region, b.output_region, 3.0, 0.02)
    combined = interfere(((0.25j, a), (-0.5, b)))
    assert combined.normalization == 2.0
    assert combined.error_bound >= 0.0125
    np.testing.assert_allclose(
        selected_matrix(combined),
        0.25j * selected_matrix(a) - 0.5 * selected_matrix(b),
        atol=3e-14,
    )
    cancelled = interfere(((1, a), (-1, a)))
    assert cancelled.normalization == 1.0
    np.testing.assert_allclose(selected_matrix(cancelled), 0, atol=1e-14)


@pytest.mark.parametrize("side", ["input", "output"])
def test_rectangular_even_response_includes_nullspace(side):
    a = select(complex_process, input=Region("0**"), output=Region("01*"))
    matrix = selected_matrix(a)
    gram = matrix.conj().T @ matrix if side == "input" else matrix @ matrix.conj().T
    target = Response.chebyshev((0.1, 0.0, 0.05))
    transformed = response(a, target, side=side)
    expected = 0.05 * np.eye(len(gram)) + 0.1 * gram
    np.testing.assert_allclose(
        selected_matrix(transformed), expected, atol=transformed.error_bound + 2e-13
    )


def test_complex_odd_response_keeps_orientation_and_explicit_scale():
    a = select(complex_process, input=Region("0**"), output=Region("01*"))
    matrix = selected_matrix(a)
    target = Response.chebyshev((0.0, 0.2, 0.0, 0.05), scale=2.0)
    transformed = response(a, target)
    expected = 0.05 * matrix + 0.2 * matrix @ matrix.conj().T @ matrix
    assert transformed.normalization == 2.0
    np.testing.assert_allclose(
        selected_matrix(transformed), expected, atol=transformed.error_bound + 2e-13
    )
    with pytest.raises(ValueError, match="Hermitian"):
        response(a, target, kind="eigenvalue")


def test_constant_response_and_structural_hermitian_evidence():
    a = select(complex_process, input=Region("0**"), output=Region("*0*"))
    constant = response(a, Response.chebyshev((0.5,)), side="input")
    np.testing.assert_allclose(selected_matrix(constant), 0.5 * np.eye(4), atol=1e-12)
    h = a.hermitian_part()
    target = Response.chebyshev((0.0, 0.1))
    transformed = response(h, target, kind="eigenvalue")
    np.testing.assert_allclose(
        selected_matrix(transformed),
        0.1 * selected_matrix(h) / h.normalization,
        atol=1e-10,
    )


@qm.qkernel
def uniform(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    return qm.h(state)


def test_readout_amplification_and_pointwise_schedule_inspection():
    from math import pi

    initial = select(uniform, input=Region("00"), output=Region("11"))
    amplified = amplify(initial, phases=((pi, pi),))
    result = qm.run(sample(amplified), on=qm.Reference(), shots=16, seed=0).result
    assert result.values == {True: 16}
    estimate = estimate_success(result)
    assert estimate.probability == 1
    assert estimate.conditional == 1
    assert estimate.shots == estimate.successes == 16
    assert abs(inspect_schedule(((pi / 3, pi / 3),), 0.5)) ** 2 == pytest.approx(
        37 / 64
    )
    np.testing.assert_allclose(
        selected_matrix(amplify(initial, phases=())),
        selected_matrix(initial),
        atol=1e-13,
    )


def test_success_denominator_payload_order_and_zero_success():
    a = select(complex_process, input=Region("0**"), output=Region("01*"))
    zero = interfere(((1, a), (-1, a)))
    result = qm.run(sample(zero), on=qm.Reference(), shots=8, seed=0).result
    estimate = estimate_success(result, z_indices=(0,))
    assert estimate.successes == 0
    assert estimate.probability == 0
    assert estimate.conditional is None
    assert estimate.conditional_interval == (-1, 1)
    assert estimate.probability_interval[1] > 0
    with pytest.raises(ValueError):
        estimate_success(result, z_indices=(0, 0))


def test_readout_persistence_preserves_typed_success_and_payload():
    a = select(uniform, input=Region("**"), output=Region("**"))
    measurement = sample(a @ a)
    program = measurement.specialize()
    restored = qm.Program.from_bytes(program.to_bytes())
    result = qm.run(restored, on=qm.Reference(), shots=4, seed=0).result
    assert result.values == {(True, (0, 0)): 4}
    estimate = estimate_success(result, z_indices=(0, 1))
    assert estimate.conditional == 1
    assert restored.to_bytes() == program.to_bytes()


def test_explicit_lcu_method_extends_the_solver_region():
    a = select(reflection, input=Region("0*"), output=Region("0*"))
    target = Response.chebyshev((0.0, 0.5, 0.0, 0.5), scale=1.5)
    transformed = response(a, target, method="chebyshev_lcu")
    assert transformed.normalization == 1.5
    np.testing.assert_allclose(
        selected_matrix(transformed),
        target.evaluate(0.3) * np.eye(2),
        atol=transformed.error_bound + 1e-13,
    )


def test_response_input_perturbation_is_not_a_scalar_derivative_claim():
    original = select(complex_process, input=Region("0**"), output=Region("01*"))
    matrix = selected_matrix(original)
    error = np.zeros_like(matrix)
    error[0, -1] = 0.001j
    asserted_target = matrix + error
    approximate = Signal._checked(
        original.process, original.input_region, original.output_region, 1.0, 0.001
    )
    polynomial = Response.chebyshev((0.0, 0.1, 0.0, 0.02))
    transformed = response(approximate, polynomial, max_error=0.01)
    expected = (
        0.04 * asserted_target
        + 0.08 * asserted_target @ asserted_target.conj().T @ asserted_target
    )
    assert (
        np.linalg.norm(selected_matrix(transformed) - expected, ord=2)
        <= transformed.error_bound
    )
    with pytest.raises(ValueError, match="total response error"):
        response(approximate, polynomial, max_error=1e-8)


def test_contract_failures_and_bounded_construction(monkeypatch):
    a = select(complex_process, input=Region("0**"), output=Region("01*"))
    with pytest.raises(ValueError, match="spaces"):
        _ = a @ a
    with pytest.raises(ValueError, match="parity"):
        Response.chebyshev((0.1, 0.1))
    for values in ((), (float("nan"),), (float("inf"),), (0.0,) * 258):
        with pytest.raises(ValueError):
            Response.chebyshev(values)
    with pytest.raises(ValueError, match="even responses require"):
        response(a, Response.chebyshev((0.1,)))
    with pytest.raises(ValueError, match="odd responses"):
        response(a, Response.chebyshev((0.0, 0.1)), side="input")
    with pytest.raises(ValueError, match="scale at least"):
        response(
            a,
            Response.chebyshev((0.0, 0.5, 0.0, 0.5), scale=0.5),
            method="chebyshev_lcu",
        )
    with pytest.raises(ValueError):
        interfere(((1.0, a),) * 257)
    with pytest.raises(ValueError):
        inspect_schedule(((float("inf"), 0.0),), 0.5)
    contract = a._contract
    for _ in range(native.MAX_SIGNAL_DEPTH - 1):
        contract = contract.adjoint()
    with pytest.raises(ValueError, match="budget"):
        contract.adjoint()
    monkeypatch.setattr(native, "signal_contract", None)
    with pytest.raises(TypeError, match="Q7733"):
        select(complex_process, input=Region("0**"), output=Region("01*"))


def exercise_signal_algebra_backend(*, endpoint):
    """Execute the same algebra, nullspace, success, and expectation witnesses."""
    from ._programs import (
        signal_algebra_expectation_state,
        signal_amplified_readout,
        signal_interference_readout,
        signal_product_readout,
        signal_response_readout,
    )

    for program, expected in (
        (signal_product_readout, {(True, (0,)): 4}),
        (signal_response_readout, {(True, (1, 0)): 4}),
        (signal_amplified_readout, {True: 4}),
    ):
        artifact = qm.compile(program, to=endpoint.target)
        result = qm.execute(artifact, on=endpoint, shots=4, seed=0)
        assert result.values == expected
        assert estimate_success(result).probability == 1
    artifact = qm.compile(signal_interference_readout, to=endpoint.target)
    result = qm.execute(artifact, on=endpoint, shots=4, seed=0)
    estimate = estimate_success(result, z_indices=(0,))
    assert estimate.probability == 0
    assert estimate.conditional is None
    assert (
        qm.expectation(
            signal_algebra_expectation_state, qm.Z(1), on=endpoint, shots=4, seed=0
        ).value
        == -1
    )
