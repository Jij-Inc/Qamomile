"""Controlled ``pauli_evolve`` must preserve the Hamiltonian's constant term.

For ``exp(-i * gamma * H)`` the identity component of ``H`` (its constant
offset ``c``) is an unobservable global phase ``exp(-i * gamma * c)`` when
applied uncontrolled, so every backend drops it.  Under ``qmc.control``
that phase becomes an *observable* relative phase between the control-on
and control-off branches.

CUDA-Q lowers a controlled sub-kernel by wrapping the *uncontrolled*
gadget in ``cudaq.control``, which discards that global phase; the
transpiler must re-apply ``P(-gamma * c)`` on the controls.  These tests
pin the CUDA-Q statevector to the Qiskit reference, which handles the
constant correctly -- before the fix they diverge by a few to tens of
percent (a 0.7 offset already shifts the relative phase visibly).

Runs in ``-m cudaq`` sessions only: cudaq is imported lazily inside the
helper, so a default session never loads it (see tests/_cudaq_isolation.py).
The Qiskit reference uses the Aer-free ``qiskit.quantum_info.Statevector``
so no qiskit-aer simulation runs in the same process as cudaq.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qiskit")

import qamomile.circuit as qmc  # noqa: E402
import qamomile.observable as qm_o  # noqa: E402
from qamomile.qiskit import QiskitTranspiler  # noqa: E402


@qmc.qkernel
def _evolve(
    q: qmc.Vector[qmc.Qubit], ham: qmc.Observable, gamma: qmc.Float
) -> qmc.Vector[qmc.Qubit]:
    """Single uncontrolled Pauli evolution layer."""
    return qmc.pauli_evolve(q, ham, gamma)


@qmc.qkernel
def _evolve_three_steps(
    q: qmc.Vector[qmc.Qubit], ham: qmc.Observable, gamma: qmc.Float
) -> qmc.Vector[qmc.Qubit]:
    """Three Pauli evolution layers via a compile-time ``for`` loop.

    Exercises the unrolled-loop path: the controlled constant phase must
    be re-applied once per iteration.
    """
    for _ in qmc.range(3):
        q = qmc.pauli_evolve(q, ham, gamma)
    return q


def _single_control_kernel(sub):
    """Build a one-control wrapper around ``sub`` with the control in |+>.

    Args:
        sub: A ``Vector[Qubit], Observable, Float`` sub-qkernel.

    Returns:
        A no-argument-qubit qkernel taking ``(ham, gamma)`` bindings.
    """

    @qmc.qkernel
    def kernel(ham: qmc.Observable, gamma: qmc.Float) -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(2, "q")
        # Hadamard on the control exposes the dropped phase as an
        # observable relative phase between the |0>/|1> control branches.
        q[0] = qmc.h(q[0])
        controlled = qmc.control(sub)
        q[0], target = controlled(q[0], q[1:2], ham=ham, gamma=gamma)
        q[1:2] = target
        return qmc.measure(q)

    return kernel


def _two_control_kernel(sub):
    """Build a two-control wrapper around ``sub`` with both controls in |+>.

    Args:
        sub: A ``Vector[Qubit], Observable, Float`` sub-qkernel.

    Returns:
        A no-argument-qubit qkernel taking ``(ham, gamma)`` bindings.
    """

    @qmc.qkernel
    def kernel(ham: qmc.Observable, gamma: qmc.Float) -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(3, "q")
        q[0] = qmc.h(q[0])
        q[1] = qmc.h(q[1])
        controlled = qmc.control(sub, num_controls=2)
        q[0:2], target = controlled(q[0:2], q[2:3], ham=ham, gamma=gamma)
        q[2:3] = target
        return qmc.measure(q)

    return kernel


def _qiskit_statevector(circuit) -> np.ndarray:
    """Aer-free reference statevector (unroll ``for_loop``s, drop measures).

    Args:
        circuit: Compiled Qiskit circuit.

    Returns:
        np.ndarray: Complex amplitudes of the unitary core.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import UnrollForLoops

    stripped = QuantumCircuit(*circuit.qregs)
    for instr in circuit.data:
        if instr.operation.name not in ("measure", "save_statevector"):
            stripped.append(instr)
    unrolled = PassManager([UnrollForLoops()]).run(stripped)
    return np.asarray(Statevector(unrolled).data)


def _cudaq_statevector(artifact) -> np.ndarray:
    """Simulate a fully-bound CUDA-Q STATIC artifact via ``cudaq.get_state``.

    Args:
        artifact: Compiled CUDA-Q kernel artifact.

    Returns:
        np.ndarray: Complex amplitudes of the kernel state.
    """
    cudaq = pytest.importorskip("cudaq")

    return np.array(cudaq.get_state(artifact.kernel_func))


def _fidelity_error(kernel, ham, gamma) -> float:
    """Return ``1 - |<psi_qiskit|psi_cudaq>|`` for ``kernel`` at ``(ham, gamma)``.

    Statevectors may differ by an overall global phase, which is physically
    irrelevant, so the comparison uses the (phase-insensitive) fidelity.
    The constant-term bug is *not* an overall global phase -- it is a
    relative phase on one branch of the control superposition -- so it does
    lower the fidelity.

    Args:
        kernel: The qkernel to transpile on both backends.
        ham: Hamiltonian binding for ``ham``.
        gamma (float): Evolution time binding for ``gamma``.

    Returns:
        float: The fidelity error (``0`` when the backends agree).
    """
    from qamomile.cudaq import CudaqTranspiler

    bindings = {"ham": ham, "gamma": gamma}
    sv_qiskit = _qiskit_statevector(
        QiskitTranspiler()
        .transpile(kernel, bindings=bindings)
        .compiled_quantum[0]
        .circuit
    )
    sv_cudaq = _cudaq_statevector(
        CudaqTranspiler()
        .transpile(kernel, bindings=bindings)
        .compiled_quantum[0]
        .circuit
    )
    overlap = abs(np.vdot(sv_qiskit, sv_cudaq))
    norm = np.linalg.norm(sv_qiskit) * np.linalg.norm(sv_cudaq)
    return 1.0 - overlap / norm


@pytest.mark.cudaq
class TestControlledPauliEvolveConstant:
    """CUDA-Q controlled ``pauli_evolve`` agrees with Qiskit on constant offsets."""

    @pytest.mark.parametrize(
        "ham, gamma",
        [
            (qm_o.X(0) + 0.7, 0.5),
            (qm_o.Z(0) + 1.3, 0.9),
            (qm_o.Y(0) + 0.4, 1.1),
            (qm_o.X(0) - 2.1, 1.7),
        ],
    )
    def test_single_control_matches_qiskit(self, ham, gamma) -> None:
        """One control: ``P(-gamma * c)`` lands on the control qubit."""
        assert _fidelity_error(_single_control_kernel(_evolve), ham, gamma) < 1e-9

    def test_two_controls_matches_qiskit(self) -> None:
        """Two controls: the phase becomes a multi-controlled ``P``."""
        err = _fidelity_error(_two_control_kernel(_evolve), qm_o.X(0) + 0.7, 0.5)
        assert err < 1e-9

    def test_for_loop_body_matches_qiskit(self) -> None:
        """A ``for``-loop body re-applies the constant phase per iteration."""
        err = _fidelity_error(
            _single_control_kernel(_evolve_three_steps), qm_o.X(0) + 0.7, 0.4
        )
        assert err < 1e-9

    def test_zero_constant_is_unchanged(self) -> None:
        """No constant offset -> no extra phase, and still matches Qiskit."""
        err = _fidelity_error(_single_control_kernel(_evolve), qm_o.X(0) + 0.0, 0.5)
        assert err < 1e-9

    def test_complex_constant_is_rejected(self) -> None:
        """A non-real constant term (non-Hermitian) raises EmitError, not a
        silently non-unitary circuit."""
        from qamomile.circuit.transpiler.errors import EmitError
        from qamomile.cudaq import CudaqTranspiler

        ham = qm_o.X(0) + 1j  # complex constant -> exp(-i*gamma*H) non-unitary
        kernel = _single_control_kernel(_evolve)
        with pytest.raises(EmitError):
            CudaqTranspiler().transpile(kernel, bindings={"ham": ham, "gamma": 0.5})
