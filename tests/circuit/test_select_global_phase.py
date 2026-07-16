"""Cross-backend relative-phase tests for ``qmc.select``."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o


@qmc.qkernel
def _identity(q: qmc.Qubit) -> qmc.Qubit:
    """Return one target unchanged."""
    return q


@qmc.qkernel
def _x(q: qmc.Qubit) -> qmc.Qubit:
    """Apply Pauli X to one target."""
    return qmc.x(q)


@qmc.qkernel
def _phase_pi_identity(q: qmc.Qubit) -> qmc.Qubit:
    """Apply phase pi to an identity Pauli case."""
    return qmc.global_phase(_identity, math.pi)(q)


@qmc.qkernel
def _phase_half_pi_identity(q: qmc.Qubit) -> qmc.Qubit:
    """Apply phase pi/2 to an identity Pauli case."""
    return qmc.global_phase(_identity, math.pi / 2.0)(q)


@qmc.qkernel
def _phase_half_pi_x(q: qmc.Qubit) -> qmc.Qubit:
    """Apply Pauli X with phase pi/2."""
    return qmc.global_phase(qmc.x, math.pi / 2.0)(q)


@qmc.qkernel
def _phase_0(q: qmc.Qubit) -> qmc.Qubit:
    """Apply the first arbitrary eight-case phase."""
    return qmc.global_phase(_identity, 0.0)(q)


@qmc.qkernel
def _phase_1(q: qmc.Qubit) -> qmc.Qubit:
    """Apply the second arbitrary eight-case phase."""
    return qmc.global_phase(_identity, 0.17)(q)


@qmc.qkernel
def _phase_2(q: qmc.Qubit) -> qmc.Qubit:
    """Apply the third arbitrary eight-case phase."""
    return qmc.global_phase(_identity, -0.31)(q)


@qmc.qkernel
def _phase_3(q: qmc.Qubit) -> qmc.Qubit:
    """Apply the fourth arbitrary eight-case phase."""
    return qmc.global_phase(_identity, 0.7)(q)


@qmc.qkernel
def _phase_4(q: qmc.Qubit) -> qmc.Qubit:
    """Apply the fifth arbitrary eight-case phase."""
    return qmc.global_phase(_identity, 1.1)(q)


@qmc.qkernel
def _phase_5(q: qmc.Qubit) -> qmc.Qubit:
    """Apply the sixth arbitrary eight-case phase."""
    return qmc.global_phase(_identity, -1.3)(q)


@qmc.qkernel
def _phase_6(q: qmc.Qubit) -> qmc.Qubit:
    """Apply the seventh arbitrary eight-case phase."""
    return qmc.global_phase(_identity, 2.0)(q)


@qmc.qkernel
def _phase_7(q: qmc.Qubit) -> qmc.Qubit:
    """Apply the eighth arbitrary eight-case phase."""
    return qmc.global_phase(_identity, -2.4)(q)


@qmc.qkernel
def _phase_theta_identity(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Apply a forwarded phase parameter to an identity case."""
    return qmc.global_phase(_identity, theta)(q)


@qmc.qkernel
def _parameterized_identity(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Accept the shared phase parameter while leaving the target unchanged."""
    _ = theta
    return q


@qmc.qkernel
def _select_phase_body(
    index: qmc.Qubit,
    target: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Apply a phase-only two-case SELECT."""
    return qmc.select([_identity, _phase_pi_identity])(index, target)


@qmc.qkernel
def _select_half_phase_body(
    index: qmc.Qubit,
    target: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Apply a non-self-inverse phase-only SELECT."""
    return qmc.select([_identity, _phase_half_pi_identity])(index, target)


@qmc.qkernel
def _select_half_phase_x_body(
    index: qmc.Qubit,
    target: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Apply a non-self-inverse phased-Pauli SELECT."""
    return qmc.select([_identity, _phase_half_pi_x])(index, target)


@qmc.qkernel
def _identity_pair(
    index: qmc.Qubit,
    target: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Return a two-qubit target register unchanged."""
    return index, target


@qmc.qkernel
def _nested_phase_case(
    index: qmc.Qubit,
    target: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Use a phase-only SELECT as another SELECT's case."""
    return qmc.select([_identity, _phase_pi_identity])(index, target)


@qmc.qkernel
def _broadcast_phase_only_select(
    target_size: qmc.UInt,
    observable: qmc.Observable,
) -> qmc.Float:
    """Broadcast a scalar phase-only SELECT case over a vector target."""
    index = qmc.h(qmc.qubit("index"))
    target = qmc.qubit_array(target_size, "target")
    index, target = qmc.select([_identity, _phase_half_pi_identity])(
        index,
        target,
    )
    return qmc.expval(index, observable)


@qmc.qkernel
def _broadcast_phased_x_select(
    target_size: qmc.UInt,
    observable: qmc.Observable,
) -> qmc.Float:
    """Broadcast a phased X case and coherently uncompute its X operations."""
    index = qmc.h(qmc.qubit("index"))
    target = qmc.qubit_array(target_size, "target")
    index, target = qmc.select([_identity, _phase_half_pi_x])(index, target)
    index, target = qmc.select([_identity, _x])(index, target)
    return qmc.expval(index, observable)


def _executor(case: Any, *, runtime_control: bool = False) -> Any:
    """Return a simulator executor for one SDK backend case.

    Args:
        case (Any): Backend fixture containing a transpiler and backend name.
        runtime_control (bool): Whether Qiskit needs a dynamic-control capable
            simulator. Defaults to ``False``.

    Returns:
        Any: Executor for the selected SDK backend.
    """
    if case.backend_name == "qiskit":
        if runtime_control:
            from qiskit_aer import AerSimulator

            return case.transpiler.executor(backend=AerSimulator(method="statevector"))
        from qiskit.providers.basic_provider import BasicSimulator

        return case.transpiler.executor(backend=BasicSimulator())
    return case.transpiler.executor()


def _only_outcome(
    case: Any,
    kernel: Any,
    *,
    runtime_control: bool = False,
) -> Any:
    """Transpile and sample a deterministic kernel on one backend.

    Args:
        case (Any): Backend fixture containing a transpiler and backend name.
        kernel (Any): Deterministic qkernel to transpile and execute.
        runtime_control (bool): Whether the kernel uses dynamic control flow.
            Defaults to ``False``.

    Returns:
        Any: The single sampled classical outcome.
    """
    executable = case.transpiler.transpile(kernel)
    result = executable.sample(
        _executor(case, runtime_control=runtime_control),
        shots=128,
    ).result()
    counts = {bits: count for bits, count in result.results}
    assert sum(counts.values()) == 128
    assert len(counts) == 1, f"{case.backend_name}: got {counts}"
    return next(iter(counts))


def test_identity_case_phase_becomes_relative_phase(sdk_transpiler: Any) -> None:
    """A phase-only identity case remains observable under index control."""

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        index = qmc.h(qmc.qubit("index"))
        target = qmc.qubit("target")
        index, target = qmc.select([_identity, _phase_pi_identity])(index, target)
        index = qmc.h(index)
        return qmc.measure(index)

    assert _only_outcome(sdk_transpiler, circuit) == 1


@pytest.mark.parametrize("target_size", [1, 2, 3])
def test_broadcast_phase_only_case_applies_phase_once(
    sdk_transpiler: Any,
    target_size: int,
) -> None:
    """A broadcast case's relative phase is independent of target length."""
    executable = sdk_transpiler.transpiler.transpile(
        _broadcast_phase_only_select,
        bindings={"target_size": target_size, "observable": qm_o.Y(0)},
    )
    value = executable.run(_executor(sdk_transpiler)).result()

    assert value == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("target_size", [1, 2, 3])
def test_broadcast_phased_x_case_applies_phase_once(
    sdk_transpiler: Any,
    target_size: int,
) -> None:
    """A broadcast phased-X case preserves one phase and every X operation."""
    executable = sdk_transpiler.transpiler.transpile(
        _broadcast_phased_x_select,
        bindings={"target_size": target_size, "observable": qm_o.Y(0)},
    )
    value = executable.run(_executor(sdk_transpiler)).result()

    assert value == pytest.approx(1.0, abs=1e-6)


def test_symbolic_wide_index_preserves_identity_case_phase(
    sdk_transpiler: Any,
) -> None:
    """A bound over-wide register keeps a phase-only case coherent."""

    @qmc.qkernel
    def circuit(width: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        index = qmc.qubit_array(3, "index")
        index[0] = qmc.h(index[0])
        target = qmc.qubit("target")
        index, target = qmc.select(
            [_identity, _phase_pi_identity],
            num_index_qubits=width,
        )(index, target)
        index[0] = qmc.h(index[0])
        return qmc.measure(index)

    executable = sdk_transpiler.transpiler.transpile(
        circuit,
        bindings={"width": 3},
    )
    result = executable.sample(_executor(sdk_transpiler), shots=128).result()
    counts = {bits: count for bits, count in result.results}
    assert set(counts) == {(1, 0, 0)}, f"{sdk_transpiler.backend_name}: got {counts}"


def test_eight_case_phase_select_uses_lsb_zero(sdk_transpiler: Any) -> None:
    """Three-index SELECT preserves phases and treats index qubit zero as LSB."""

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Bit]:
        index = qmc.h(qmc.qubit_array(3, "index"))
        target = qmc.qubit("target")
        cases = [
            _identity,
            _phase_pi_identity,
            _identity,
            _phase_pi_identity,
            _identity,
            _phase_pi_identity,
            _identity,
            _phase_pi_identity,
        ]
        index, target = qmc.select(cases)(index, target)
        index = qmc.h(index)
        return qmc.measure(index)

    assert _only_outcome(sdk_transpiler, circuit) == (1, 0, 0)


def test_outer_control_preserves_select_relative_phase(sdk_transpiler: Any) -> None:
    """An enclosing coherent control composes with the SELECT index controls."""

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        outer = qmc.h(qmc.qubit("outer"))
        index = qmc.x(qmc.qubit("index"))
        target = qmc.qubit("target")
        outer, index, target = qmc.control(_select_phase_body)(
            outer,
            index,
            target,
        )
        outer = qmc.h(outer)
        return qmc.measure(outer)

    assert _only_outcome(sdk_transpiler, circuit) == 1


def test_inverse_and_nested_select_preserve_phase(sdk_transpiler: Any) -> None:
    """Inverse cancels phase cases and nested SELECT keeps their relative phase."""

    @qmc.qkernel
    def inverse_circuit() -> qmc.Bit:
        index = qmc.h(qmc.qubit("index"))
        target = qmc.qubit("target")
        index, target = _select_half_phase_body(index, target)
        index, target = qmc.inverse(_select_half_phase_body)(index, target)
        index = qmc.h(index)
        return qmc.measure(index)

    @qmc.qkernel
    def nested_circuit() -> qmc.Bit:
        outer_index = qmc.h(qmc.qubit("outer_index"))
        inner_index = qmc.x(qmc.qubit("inner_index"))
        target = qmc.qubit("target")
        outer_index, inner_index, target = qmc.select(
            [_identity_pair, _nested_phase_case]
        )(outer_index, inner_index, target)
        outer_index = qmc.h(outer_index)
        return qmc.measure(outer_index)

    assert _only_outcome(sdk_transpiler, inverse_circuit) == 0
    assert _only_outcome(sdk_transpiler, nested_circuit) == 1


def test_inverse_cancels_phased_pauli_case(sdk_transpiler: Any) -> None:
    """SELECT followed by its inverse cancels both Pauli and case phase."""

    @qmc.qkernel
    def circuit() -> tuple[qmc.Bit, qmc.Bit]:
        index = qmc.h(qmc.qubit("index"))
        target = qmc.qubit("target")
        index, target = _select_half_phase_x_body(index, target)
        index, target = qmc.inverse(_select_half_phase_x_body)(index, target)
        index = qmc.h(index)
        return qmc.measure(index), qmc.measure(target)

    assert _only_outcome(sdk_transpiler, circuit) == (0, 0)


def test_runtime_phase_parameter_is_forwarded(sdk_transpiler: Any) -> None:
    """A case's phase parameter survives as a backend runtime parameter."""

    @qmc.qkernel
    def circuit(theta: qmc.Float) -> qmc.Bit:
        index = qmc.h(qmc.qubit("index"))
        target = qmc.qubit("target")
        index, target = qmc.select([_phase_theta_identity, _parameterized_identity])(
            index,
            target,
            theta=theta,
        )
        index = qmc.h(index)
        return qmc.measure(index)

    executable = sdk_transpiler.transpiler.transpile(
        circuit,
        parameters=["theta"],
    )
    result = executable.sample(
        _executor(sdk_transpiler),
        shots=128,
        bindings={"theta": math.pi},
    ).result()
    counts = {bits: count for bits, count in result.results}
    assert set(counts) == {1}, f"{sdk_transpiler.backend_name}: got {counts}"


def test_eight_case_runtime_phase_uses_lsb_zero(
    sdk_transpiler: Any,
) -> None:
    """An eight-case symbolic phase pattern addresses index qubit zero as LSB."""

    @qmc.qkernel
    def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
        index = qmc.h(qmc.qubit_array(3, "index"))
        target = qmc.qubit("target")
        cases = [
            _parameterized_identity,
            _phase_theta_identity,
            _parameterized_identity,
            _phase_theta_identity,
            _parameterized_identity,
            _phase_theta_identity,
            _parameterized_identity,
            _phase_theta_identity,
        ]
        index, target = qmc.select(cases)(index, target, theta=theta)
        index = qmc.h(index)
        return qmc.measure(index)

    executable = sdk_transpiler.transpiler.transpile(
        circuit,
        parameters=["theta"],
    )
    result = executable.sample(
        _executor(sdk_transpiler),
        shots=128,
        bindings={"theta": math.pi},
    ).result()
    counts = {bits: count for bits, count in result.results}
    assert set(counts) == {(1, 0, 0)}, f"{sdk_transpiler.backend_name}: got {counts}"


def test_eight_case_runtime_phase_preserves_signed_lsb_interference(
    sdk_transpiler: Any,
) -> None:
    """A nontrivial runtime phase keeps its sign through the 8-case path."""
    theta = 0.73

    @qmc.qkernel
    def circuit(
        angle: qmc.Float,
        observable: qmc.Observable,
    ) -> qmc.Float:
        index = qmc.h(qmc.qubit_array(3, "index"))
        target = qmc.qubit("target")
        index, target = qmc.select(
            [
                _parameterized_identity,
                _phase_theta_identity,
                _parameterized_identity,
                _phase_theta_identity,
                _parameterized_identity,
                _phase_theta_identity,
                _parameterized_identity,
                _phase_theta_identity,
            ]
        )(index, target, theta=angle)
        return qmc.expval(index[0], observable)

    executable = sdk_transpiler.transpiler.transpile(
        circuit,
        bindings={"observable": qm_o.Y(0)},
        parameters=["angle"],
    )
    value = executable.run(
        _executor(sdk_transpiler),
        bindings={"angle": theta},
    ).result()
    assert value == pytest.approx(math.sin(theta), abs=1e-6)


@pytest.mark.quri_parts
def test_quri_eight_case_runtime_phase_reserves_clean_ancilla() -> None:
    """QURI reserves clean workspace for 3 controls and runtime phase."""
    pytest.importorskip("quri_parts.circuit")
    from qamomile.quri_parts import QuriPartsTranspiler

    @qmc.qkernel
    def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
        index = qmc.h(qmc.qubit_array(3, "index"))
        target = qmc.qubit("target")
        index, target = qmc.select(
            [
                _parameterized_identity,
                _phase_theta_identity,
                _parameterized_identity,
                _phase_theta_identity,
                _parameterized_identity,
                _phase_theta_identity,
                _parameterized_identity,
                _phase_theta_identity,
            ]
        )(index, target, theta=theta)
        return qmc.measure(index)

    executable = QuriPartsTranspiler().transpile(
        circuit,
        parameters=["theta"],
    )
    materialized = executable.compiled_quantum[0].circuit

    # Four program qubits, one control-conjunction ancilla, and one clean
    # runtime-phase carrier are required by this lowering.
    assert materialized.qubit_count == 6


def test_loop_variable_phase_forces_safe_select_unrolling(
    sdk_transpiler: Any,
) -> None:
    """A case phase derived from the loop index is materialized per iteration."""

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        index = qmc.h(qmc.qubit("index"))
        target = qmc.qubit("target")
        for i in qmc.range(2):
            theta = cast(qmc.Float, math.pi * i)
            index, target = qmc.select(
                [_parameterized_identity, _phase_theta_identity]
            )(
                index,
                target,
                theta=theta,
            )
        index = qmc.h(index)
        return qmc.measure(index)

    assert _only_outcome(sdk_transpiler, circuit) == 1


def test_phase_select_survives_mixed_runtime_control_flow(
    sdk_transpiler: Any,
) -> None:
    """A phased SELECT survives nested while, if, and for regions."""
    if sdk_transpiler.backend_name == "quri_parts":
        pytest.skip("QURI Parts has no dynamic if or while primitive")

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        condition = qmc.measure(qmc.x(qmc.qubit("condition")))
        branch = qmc.measure(qmc.x(qmc.qubit("branch")))
        stop = qmc.qubit("stop")
        index = qmc.h(qmc.qubit("index"))
        target = qmc.qubit("target")
        while condition:
            if branch:
                for _iteration in qmc.range(1):
                    index, target = qmc.select([_identity, _phase_pi_identity])(
                        index, target
                    )
            condition = qmc.measure(stop)
        index = qmc.h(index)
        return qmc.measure(index)

    assert (
        _only_outcome(
            sdk_transpiler,
            circuit,
            runtime_control=True,
        )
        == 1
    )


def test_eight_arbitrary_case_phases_match_exact_statevector(
    qiskit_transpiler: Any,
) -> None:
    """Eight distinct complex coefficients retain their index-relative phases."""
    from qiskit.quantum_info import Statevector

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Bit]:
        index = qmc.h(qmc.qubit_array(3, "index"))
        target = qmc.qubit("target")
        index, target = qmc.select(
            [
                _phase_0,
                _phase_1,
                _phase_2,
                _phase_3,
                _phase_4,
                _phase_5,
                _phase_6,
                _phase_7,
            ]
        )(index, target)
        return qmc.measure(index)

    executable = qiskit_transpiler.transpile(circuit)
    quantum_circuit = executable.quantum_circuit.remove_final_measurements(
        inplace=False
    )
    state = np.asarray(Statevector.from_instruction(quantum_circuit).data)
    angles = np.asarray([0.0, 0.17, -0.31, 0.7, 1.1, -1.3, 2.0, -2.4])
    expected = np.exp(1j * angles) / np.sqrt(8.0)

    assert np.allclose(state[:8], expected, rtol=0.0, atol=1e-10)
    assert np.allclose(state[8:], 0.0, rtol=0.0, atol=1e-10)
