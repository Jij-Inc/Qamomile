"""Shared frontend semantics tests for all SDK engines.

These tests cover engine-independent circuit meaning. SDK-specific tests
should keep SDK shape/source assertions, while statevector semantics live here.
"""

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from tests.engine_cases import SDK_ENGINES, EngineCase
from tests.transpiler.gate_test_specs import (
    GATE_SPECS,
    all_zeros_state,
    computational_basis_state,
    compute_expected_statevector,
    statevectors_equal,
)

ROTATION_ANGLES = [
    0.0,
    np.pi / 4,
    np.pi / 2,
    np.pi,
    2 * np.pi,
] + [
    np.random.default_rng(seed).uniform(0, 2 * np.pi)
    for seed in [42, 123, 456]
]


def _single_qubit_initial_state(name: str) -> np.ndarray:
    """Return the expected statevector before the tested gate runs."""
    if name == "zero":
        return all_zeros_state(1)
    if name == "one":
        return compute_expected_statevector(
            all_zeros_state(1),
            GATE_SPECS["X"].matrix_fn(),
        )
    if name == "plus":
        return compute_expected_statevector(
            all_zeros_state(1),
            GATE_SPECS["H"].matrix_fn(),
        )
    raise AssertionError(f"unknown initial state: {name}")


def _observable(num_qubits: int) -> qm_o.Hamiltonian:
    """Return a simple observable for expval wrapper entrypoints."""
    hamiltonian = qm_o.Hamiltonian(num_qubits=num_qubits)
    hamiltonian += qm_o.Z(0)
    return hamiltonian


class TestSingleQubitGateSemantics:
    """Verify single-qubit gate semantics across SDK engines."""

    @pytest.mark.parametrize("engine", SDK_ENGINES)
    @pytest.mark.parametrize(
        ("gate_name", "initial_state"),
        [
            ("H", "zero"),
            ("X", "zero"),
            ("Y", "zero"),
            ("Z", "zero"),
            ("Z", "one"),
            ("Z", "plus"),
            ("S", "zero"),
            ("S", "one"),
            ("S", "plus"),
            ("SDG", "zero"),
            ("SDG", "one"),
            ("SDG", "plus"),
            ("T", "zero"),
            ("T", "one"),
            ("T", "plus"),
            ("TDG", "zero"),
            ("TDG", "one"),
            ("TDG", "plus"),
        ],
    )
    def test_fixed_gate_statevector(
        self,
        engine: EngineCase,
        gate_name: str,
        initial_state: str,
    ) -> None:
        """Check each fixed single-qubit gate against its unitary matrix."""

        @qmc.qkernel
        def prepare() -> qmc.Qubit:
            q = qmc.qubit("q")
            if initial_state == "one":
                q = qmc.x(q)
            if initial_state == "plus":
                q = qmc.h(q)

            if gate_name == "H":
                q = qmc.h(q)
            elif gate_name == "X":
                q = qmc.x(q)
            elif gate_name == "Y":
                q = qmc.y(q)
            elif gate_name == "Z":
                q = qmc.z(q)
            elif gate_name == "S":
                q = qmc.s(q)
            elif gate_name == "SDG":
                q = qmc.sdg(q)
            elif gate_name == "T":
                q = qmc.t(q)
            elif gate_name == "TDG":
                q = qmc.tdg(q)
            return q

        @qmc.qkernel
        def circuit(H: qmc.Observable) -> qmc.Float:
            q = prepare()
            return qmc.expval(q, H)

        sv = engine.statevector(circuit, bindings={"H": _observable(1)})
        expected = compute_expected_statevector(
            _single_qubit_initial_state(initial_state),
            GATE_SPECS[gate_name].matrix_fn(),
        )
        assert statevectors_equal(sv, expected, atol=engine.statevector_atol)

    @pytest.mark.parametrize("engine", SDK_ENGINES)
    @pytest.mark.parametrize(
        ("gate_name", "initial_state"),
        [
            ("RX", "zero"),
            ("RY", "zero"),
            ("RZ", "plus"),
            ("P", "plus"),
        ],
    )
    @pytest.mark.parametrize("angle", ROTATION_ANGLES)
    def test_rotation_gate_statevector(
        self,
        engine: EngineCase,
        gate_name: str,
        initial_state: str,
        angle: float,
    ) -> None:
        """Check each single-qubit rotation gate against its unitary matrix."""

        @qmc.qkernel
        def prepare(theta: qmc.Float) -> qmc.Qubit:
            q = qmc.qubit("q")
            if initial_state == "one":
                q = qmc.x(q)
            if initial_state == "plus":
                q = qmc.h(q)

            if gate_name == "RX":
                q = qmc.rx(q, theta)
            elif gate_name == "RY":
                q = qmc.ry(q, theta)
            elif gate_name == "RZ":
                q = qmc.rz(q, theta)
            elif gate_name == "P":
                q = qmc.p(q, theta)
            return q

        @qmc.qkernel
        def circuit(theta: qmc.Float, H: qmc.Observable) -> qmc.Float:
            q = prepare(theta)
            return qmc.expval(q, H)

        sv = engine.statevector(
            circuit,
            bindings={"theta": angle, "H": _observable(1)},
        )
        expected = compute_expected_statevector(
            _single_qubit_initial_state(initial_state),
            GATE_SPECS[gate_name].matrix_fn(angle),
        )
        assert statevectors_equal(sv, expected, atol=engine.statevector_atol)


class TestMultiQubitGateSemantics:
    """Verify multi-qubit gate semantics across SDK engines."""

    @pytest.mark.parametrize("engine", SDK_ENGINES)
    @pytest.mark.parametrize("gate_name", ["CX", "CZ", "SWAP"])
    @pytest.mark.parametrize("basis_idx", [0, 1, 2, 3])
    def test_fixed_two_qubit_gate_statevector(
        self,
        engine: EngineCase,
        gate_name: str,
        basis_idx: int,
    ) -> None:
        """Check fixed two-qubit gates on all computational basis states."""

        @qmc.qkernel
        def prepare() -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(2, "q")
            if basis_idx & 1:
                q[0] = qmc.x(q[0])
            if basis_idx & 2:
                q[1] = qmc.x(q[1])

            if gate_name == "CX":
                q[0], q[1] = qmc.cx(q[0], q[1])
            elif gate_name == "CZ":
                q[0], q[1] = qmc.cz(q[0], q[1])
            elif gate_name == "SWAP":
                q[0], q[1] = qmc.swap(q[0], q[1])
            return q

        @qmc.qkernel
        def circuit(H: qmc.Observable) -> qmc.Float:
            q = prepare()
            return qmc.expval(q, H)

        sv = engine.statevector(circuit, bindings={"H": _observable(2)})
        expected = compute_expected_statevector(
            computational_basis_state(2, basis_idx),
            GATE_SPECS[gate_name].matrix_fn(),
        )
        assert statevectors_equal(sv, expected, atol=engine.statevector_atol)

    @pytest.mark.parametrize("engine", SDK_ENGINES)
    @pytest.mark.parametrize("gate_name", ["CP", "RZZ"])
    @pytest.mark.parametrize("angle", ROTATION_ANGLES)
    def test_rotation_two_qubit_gate_statevector(
        self,
        engine: EngineCase,
        gate_name: str,
        angle: float,
    ) -> None:
        """Check two-qubit rotation gates on a phase-sensitive input state."""

        @qmc.qkernel
        def prepare(theta: qmc.Float) -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            if gate_name == "CP":
                q[0], q[1] = qmc.cp(q[0], q[1], theta)
            elif gate_name == "RZZ":
                q[0], q[1] = qmc.rzz(q[0], q[1], theta)
            return q

        @qmc.qkernel
        def circuit(theta: qmc.Float, H: qmc.Observable) -> qmc.Float:
            q = prepare(theta)
            return qmc.expval(q, H)

        sv = engine.statevector(
            circuit,
            bindings={"theta": angle, "H": _observable(2)},
        )
        initial = np.ones(4, dtype=complex) / 2
        expected = compute_expected_statevector(
            initial,
            GATE_SPECS[gate_name].matrix_fn(angle),
        )
        assert statevectors_equal(sv, expected, atol=engine.statevector_atol)

    @pytest.mark.parametrize("engine", SDK_ENGINES)
    @pytest.mark.parametrize("basis_idx", list(range(8)))
    def test_ccx_statevector(self, engine: EngineCase, basis_idx: int) -> None:
        """Check Toffoli semantics on all computational basis states."""

        @qmc.qkernel
        def prepare() -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(3, "q")
            if basis_idx & 1:
                q[0] = qmc.x(q[0])
            if basis_idx & 2:
                q[1] = qmc.x(q[1])
            if basis_idx & 4:
                q[2] = qmc.x(q[2])
            q[0], q[1], q[2] = qmc.ccx(q[0], q[1], q[2])
            return q

        @qmc.qkernel
        def circuit(H: qmc.Observable) -> qmc.Float:
            q = prepare()
            return qmc.expval(q, H)

        sv = engine.statevector(circuit, bindings={"H": _observable(3)})
        expected = compute_expected_statevector(
            computational_basis_state(3, basis_idx),
            GATE_SPECS["TOFFOLI"].matrix_fn(),
        )
        assert statevectors_equal(sv, expected, atol=engine.statevector_atol)
