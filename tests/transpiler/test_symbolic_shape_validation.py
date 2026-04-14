"""Tests for Layer 3: ``SymbolicShapeValidationPass``.

When a top-level Vector parameter's symbolic shape dimension reaches a
``ForOperation`` loop bound without being folded, transpile must raise a
``QamomileCompileError`` with an actionable message — not silently elide
the loop, not fail cryptically at emit time. The library QAOA pattern
(``p`` bound in bindings, ``gammas.shape`` never queried) must keep
working unchanged.
"""

import pytest

pytest.importorskip("qiskit")

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.basic import superposition_vector
from qamomile.circuit.algorithm.qaoa import qaoa_layers, x_mixer
from qamomile.circuit.transpiler.errors import QamomileCompileError
from qamomile.qiskit.transpiler import QiskitTranspiler


def _make_h() -> qm_o.Hamiltonian:
    H = qm_o.Hamiltonian()
    H.add_term(
        (
            qm_o.PauliOperator(qm_o.Pauli.Z, 0),
            qm_o.PauliOperator(qm_o.Pauli.Z, 1),
        ),
        1.0,
    )
    return H


class TestRejection:
    """Patterns that Layer 3 should catch."""

    def test_flat_kernel_unresolved_shape_raises(self):
        """Flat kernel using ``gamma.shape[0]`` with no binding is rejected."""

        @qmc.qkernel
        def kernel(
            n: qmc.UInt,
            gamma: qmc.Vector[qmc.Float],
            hamiltonian: qmc.Observable,
        ) -> qmc.Float:
            q = superposition_vector(n)
            for i in qmc.range(gamma.shape[0]):
                q = x_mixer(q, gamma[i])
            return qmc.expval(q, hamiltonian)

        H = _make_h()
        tr = QiskitTranspiler()
        with pytest.raises(QamomileCompileError) as exc_info:
            tr.transpile(
                kernel,
                bindings={"n": H.num_qubits, "hamiltonian": H},
                parameters=["gamma"],
            )
        msg = str(exc_info.value)
        assert "gamma" in msg
        assert "shape dimension 0" in msg

    def test_error_suggests_concrete_binding(self):
        """Error message guides users to bind the array concretely."""

        @qmc.qkernel
        def kernel(
            n: qmc.UInt,
            betas: qmc.Vector[qmc.Float],
            hamiltonian: qmc.Observable,
        ) -> qmc.Float:
            q = superposition_vector(n)
            for i in qmc.range(betas.shape[0]):
                q = x_mixer(q, betas[i])
            return qmc.expval(q, hamiltonian)

        H = _make_h()
        tr = QiskitTranspiler()
        with pytest.raises(QamomileCompileError) as exc_info:
            tr.transpile(
                kernel,
                bindings={"n": H.num_qubits, "hamiltonian": H},
                parameters=["betas"],
            )
        msg = str(exc_info.value)
        assert "bindings" in msg
        assert "betas" in msg

    def test_error_suggests_loop_counter(self):
        """Error message also shows the ``p`` counter pattern."""

        @qmc.qkernel
        def kernel(
            n: qmc.UInt,
            gamma: qmc.Vector[qmc.Float],
            hamiltonian: qmc.Observable,
        ) -> qmc.Float:
            q = superposition_vector(n)
            for i in qmc.range(gamma.shape[0]):
                q = x_mixer(q, gamma[i])
            return qmc.expval(q, hamiltonian)

        H = _make_h()
        tr = QiskitTranspiler()
        with pytest.raises(QamomileCompileError) as exc_info:
            tr.transpile(
                kernel,
                bindings={"n": H.num_qubits, "hamiltonian": H},
                parameters=["gamma"],
            )
        msg = str(exc_info.value)
        assert "qm.range" in msg


class TestAcceptance:
    """Patterns that Layer 3 should leave alone."""

    def test_library_qaoa_layers_pattern_passes(self):
        """``qaoa_layers`` with ``p`` bound is the blessed pattern."""

        @qmc.qkernel
        def kernel(
            p: qmc.UInt,
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            n: qmc.UInt,
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = superposition_vector(n)
            q = qaoa_layers(p, quad, linear, q, gammas, betas)
            return qmc.measure(q)

        tr = QiskitTranspiler()
        exe = tr.transpile(
            kernel,
            bindings={
                "p": 2,
                "quad": {(0, 1): 0.5},
                "linear": {0: 0.1},
                "n": 2,
            },
            parameters=["gammas", "betas"],
        )
        assert exe.compiled_quantum[0].circuit.num_parameters >= 2

    def test_concrete_array_binding_passes(self):
        """When ``gamma`` is bound concretely, shape is folded → no error."""

        @qmc.qkernel
        def kernel(
            n: qmc.UInt,
            gamma: qmc.Vector[qmc.Float],
            hamiltonian: qmc.Observable,
        ) -> qmc.Float:
            q = superposition_vector(n)
            for i in qmc.range(gamma.shape[0]):
                q = x_mixer(q, gamma[i])
            return qmc.expval(q, hamiltonian)

        H = _make_h()
        tr = QiskitTranspiler()
        exe = tr.transpile(
            kernel,
            bindings={
                "n": H.num_qubits,
                "hamiltonian": H,
                "gamma": [0.3, 0.5],
            },
        )
        circuit = exe.compiled_quantum[0].circuit
        # 2 H (init) + 2 * 2 = 4 Rx from x_mixer(2*beta) unrolled for 2 layers
        assert circuit.size() >= 2
        assert circuit.num_qubits == H.num_qubits
