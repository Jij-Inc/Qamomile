"""CUDA-Q lowers ``pauli_evolve`` natively via ``exp_pauli`` and matches Qiskit.

CUDA-Q realizes each Hamiltonian term ``coeff * P`` of ``exp(-i*gamma*H)``
with a single ``exp_pauli`` call instead of the ``h`` / ``rz`` + CX-ladder
gadget the other backends use.  The risk in that lowering is the Pauli
word / qubit ordering: an asymmetric multi-qubit term such as ``X0 Z1``
silently computes the wrong operator if the word is reversed relative to
the qubit list.  These statevector checks against Qiskit's
``PauliEvolutionGate`` reference pin that ordering down for single-qubit,
multi-qubit asymmetric, three-qubit mixed, and sliced (non-contiguous)
registers.

Runs in ``-m cudaq`` sessions only: cudaq is imported lazily inside the
helper, so a default session never loads it (see tests/_cudaq_isolation.py).
The Qiskit reference uses the Aer-free ``qiskit.quantum_info.Statevector``
so no qiskit-aer simulation runs in the same process as cudaq.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types

import numpy as np
import pytest

pytest.importorskip("qiskit")

import qamomile.circuit as qmc  # noqa: E402
import qamomile.observable as qm_o  # noqa: E402
from qamomile.qiskit import QiskitTranspiler  # noqa: E402


@qmc.qkernel
def _evolve_all(
    ham: qmc.Observable, gamma: qmc.Float, n: qmc.UInt
) -> qmc.Vector[qmc.Bit]:
    """Evolve a fully-superposed ``n``-qubit register under ``ham``."""
    q = qmc.qubit_array(n, "q")
    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
    q = qmc.pauli_evolve(q, ham, gamma)
    return qmc.measure(q)


@qmc.qkernel
def _evolve_even(ham: qmc.Observable, gamma: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Evolve only the even qubits of a 4-qubit register (sliced view)."""
    q = qmc.qubit_array(4, "q")
    for i in qmc.range(4):
        q[i] = qmc.h(q[i])
    q[0::2] = qmc.pauli_evolve(q[0::2], ham, gamma)
    return qmc.measure(q)


@qmc.qkernel
def _padded_evolve_then_x(
    q: qmc.Vector[qmc.Qubit],
    ham: qmc.Observable,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Use the untouched tail slot after a narrower Hamiltonian evolution."""
    q = qmc.pauli_evolve(q, ham, gamma)
    q[1] = qmc.x(q[1])
    return q


@qmc.qkernel
def _controlled_padded_evolve_then_x(
    ham: qmc.Observable,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    """Control a vector helper whose Hamiltonian is narrower than its target."""
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.x(q[0])
    controlled = qmc.control(_padded_evolve_then_x)
    q[0], targets = controlled(q[0], q[1:3], ham=ham, gamma=gamma)
    q[1:3] = targets
    return qmc.measure(q)


def _install_fake_cudaq_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a minimal CUDA-Q module for source-generation tests.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture used to
            install the fake module for the current test only.
    """
    fake_cudaq = types.ModuleType("cudaq")
    fake_cudaq.__spec__ = importlib.machinery.ModuleSpec("cudaq", loader=None)
    fake_cudaq.kernel = lambda func: func

    class qubit:
        """Stand in for the ``cudaq.qubit`` type annotation."""

    fake_cudaq.qubit = qubit
    monkeypatch.setitem(sys.modules, "cudaq", fake_cudaq)


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


def _fidelity_error(kernel, bindings) -> float:
    """Return ``1 - |<psi_qiskit|psi_cudaq>|`` for ``kernel`` at ``bindings``.

    Args:
        kernel: The qkernel to transpile on both backends.
        bindings (dict): Transpile bindings (``ham``, ``gamma``, ``n``).

    Returns:
        float: The fidelity error (``0`` when the backends agree).
    """
    from qamomile.cudaq import CudaqTranspiler

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


def test_controlled_helper_pads_narrow_hamiltonian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA-Q helper emission preserves tail slots after padded evolution."""
    from qamomile.cudaq.transpiler import CudaqTranspiler

    _install_fake_cudaq_module(monkeypatch)

    exe = CudaqTranspiler().transpile(
        _controlled_padded_evolve_then_x,
        bindings={"ham": qm_o.Z(0), "gamma": 0.5},
    )
    source = exe.compiled_quantum[0].circuit.source

    assert 'exp_pauli(-0.5, [q0], "Z")' in source
    assert "x(q1)" in source
    assert "cudaq.control(_qamomile_U_0, q[0], q[1], q[2])" in source


_X, _Y, _Z = qm_o.X, qm_o.Y, qm_o.Z


@pytest.mark.cudaq
class TestCudaqNativePauliEvolve:
    """Native ``exp_pauli`` lowering reproduces the Qiskit statevector."""

    @pytest.mark.parametrize(
        "ham, gamma, n",
        [
            (_X(0), 0.5, 1),
            (_Z(0), 0.9, 1),
            (_X(0) * _Z(1), 0.6, 2),  # asymmetric: catches a reversed word
            (_Z(0) * _Z(1), 0.7, 2),
            (_X(0) * _Y(1) * _Z(2), 0.5, 3),  # three-qubit mixed word
            (_Z(0) * _Z(1) + 1.3 * _Z(0), 0.4, 2),  # multiple commuting terms
        ],
    )
    def test_matches_qiskit(self, ham, gamma, n) -> None:
        """Each representative term structure matches the Qiskit reference."""
        assert _fidelity_error(_evolve_all, {"ham": ham, "gamma": gamma, "n": n}) < 1e-9

    @pytest.mark.parametrize(
        "ham, gamma",
        [
            (_X(0) * _Z(1), 0.5),
            (_Z(0) * _Z(1), 0.7),
        ],
    )
    def test_sliced_register_matches_qiskit(self, ham, gamma) -> None:
        """Evolving a non-contiguous sliced view keeps qubit/word alignment."""
        assert _fidelity_error(_evolve_even, {"ham": ham, "gamma": gamma}) < 1e-9
