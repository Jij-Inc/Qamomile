"""Tests for Layer 3: ``SymbolicShapeValidationPass``.

When a top-level Vector parameter's symbolic shape dimension reaches a
``ForOperation`` loop bound without being folded, transpile must raise a
``QamomileCompileError`` with an actionable message — not silently elide
the loop, not fail cryptically at emit time. The same applies to loop
bounds left as runtime parameters (``parameters=["n"]`` with
``qmc.range(n)``): they must fail here, before segmentation, with the
"Cannot unroll loop" message — not as a misleading
``MultipleQuantumSegmentsError`` at plan or a late emit-time
``ValueError``. The library QAOA pattern (``p`` bound in bindings,
``gammas.shape`` never queried) must keep working unchanged.
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


class TestRuntimeParameterLoopBound:
    """Loop bounds left as runtime parameters fail early and actionably.

    Regression suite for the diagnostic inconsistency where the same user
    mistake (a ``qmc.range`` bound depending on a runtime parameter)
    surfaced either as a misleading ``MultipleQuantumSegmentsError``
    blaming measurement-dependent control flow, or as a late emit-time
    ``ValueError`` — depending on which pass tripped first.
    """

    def test_direct_runtime_parameter_bound_raises_actionable_error(self):
        """``qmc.range(n)`` with ``parameters=["n"]`` fails at validation.

        Pre-fix this case passed segmentation and failed at emit with a
        ``ValueError``; it must now raise ``QamomileCompileError`` with the
        canonical "Cannot unroll loop" wording and a bindings fix.
        """

        @qmc.qkernel
        def kernel(theta: qmc.Float, n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            for i in qmc.range(n):
                q[0] = qmc.rx(q[0], theta)
            return qmc.measure(q)

        tr = QiskitTranspiler()
        with pytest.raises(QamomileCompileError) as exc_info:
            tr.transpile(kernel, bindings={"theta": 0.5}, parameters=["n"])
        msg = str(exc_info.value)
        assert "Cannot unroll loop: bounds could not be resolved at compile time" in msg
        assert "'n'" in msg
        assert "bindings" in msg

    def test_bound_expression_no_longer_multiple_segments_error(self):
        """An arithmetic bound expression gets the same clear error.

        Pre-fix ``qmc.range(num_pairs + num_pairs)`` stranded the bound's
        ``BinOp`` between quantum ops, so segmentation raised
        ``MultipleQuantumSegmentsError`` blaming measurement-dependent
        control flow — the wrong diagnosis. The dataflow walk must trace
        the bound back to ``num_pairs`` and name it. (The raised
        ``QamomileCompileError`` is not a ``MultipleQuantumSegmentsError``
        — the two classes are unrelated in the exception hierarchy.)
        """

        @qmc.qkernel
        def kernel(
            theta: qmc.Vector[qmc.Float], num_pairs: qmc.UInt
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            for pair in qmc.range(num_pairs + num_pairs):
                q0 = q[0]
                q1 = q[1]
                q0, q1 = qmc.cp(q0, q1, theta[pair])
                q[0] = q0
                q[1] = q1
            return qmc.measure(q)

        tr = QiskitTranspiler()
        with pytest.raises(QamomileCompileError) as exc_info:
            tr.transpile(
                kernel,
                bindings={"theta": [0.1, 0.2, 0.3, 0.4]},
                parameters=["num_pairs"],
            )
        msg = str(exc_info.value)
        assert "Cannot unroll loop: bounds could not be resolved at compile time" in msg
        assert "'num_pairs'" in msg

    def test_helper_kernel_repro_raises_actionable_error(self):
        """The reported repro shape (helper + indexed angle) is caught.

        A helper qkernel applying a controlled phase, called in a loop
        whose bound is a runtime parameter, must get the loop-bound
        diagnostic naming ``num_pairs`` — not an emit-time ``ValueError``
        or a segmentation error.
        """

        @qmc.qkernel
        def cphase_helper(
            q: qmc.Vector[qmc.Qubit], angle: qmc.Float
        ) -> qmc.Vector[qmc.Qubit]:
            q0 = q[0]
            q1 = q[1]
            q0, q1 = qmc.cp(q0, q1, angle)
            q[0] = q0
            q[1] = q1
            return q

        @qmc.qkernel
        def kernel(
            theta: qmc.Vector[qmc.Float], num_pairs: qmc.UInt
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            for pair in qmc.range(num_pairs):
                q = cphase_helper(q, theta[pair + pair])
            return qmc.measure(q)

        tr = QiskitTranspiler()
        with pytest.raises(QamomileCompileError) as exc_info:
            tr.transpile(
                kernel,
                bindings={"theta": [0.1, 0.2, 0.3, 0.4]},
                parameters=["num_pairs"],
            )
        msg = str(exc_info.value)
        assert "Cannot unroll loop: bounds could not be resolved at compile time" in msg
        assert "'num_pairs'" in msg

    def test_auto_detected_runtime_parameter_bound_raises(self):
        """A bound auto-detected as a runtime parameter is caught too.

        With no ``parameters`` list, an unbound classical argument without
        a Python default becomes a runtime parameter via auto-detect; a
        loop bound depending on it must get the same diagnostic.
        """

        @qmc.qkernel
        def kernel(theta: qmc.Float, n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            for i in qmc.range(n):
                q[0] = qmc.rx(q[0], theta)
            return qmc.measure(q)

        tr = QiskitTranspiler()
        with pytest.raises(QamomileCompileError) as exc_info:
            tr.transpile(kernel, bindings={"theta": 0.5})
        msg = str(exc_info.value)
        assert "Cannot unroll loop: bounds could not be resolved at compile time" in msg
        assert "'n'" in msg

    def test_runtime_parameter_array_element_bound_raises(self):
        """A bound indexing a runtime parameter array names the array.

        ``qmc.range(idxs[0])`` with ``parameters=["idxs"]`` reaches the
        array through the element's parent-array dataflow edge; the
        diagnostic must name ``idxs``.
        """

        @qmc.qkernel
        def kernel(theta: qmc.Float, idxs: qmc.Vector[qmc.UInt]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            for i in qmc.range(idxs[0]):
                q[0] = qmc.rx(q[0], theta)
            return qmc.measure(q)

        tr = QiskitTranspiler()
        with pytest.raises(QamomileCompileError) as exc_info:
            tr.transpile(kernel, bindings={"theta": 0.5}, parameters=["idxs"])
        msg = str(exc_info.value)
        assert "Cannot unroll loop: bounds could not be resolved at compile time" in msg
        assert "'idxs'" in msg


class TestAcceptance:
    """Patterns that Layer 3 should leave alone."""

    def test_nested_bound_on_outer_loop_var_passes(self):
        """A nested ``qmc.range(i + 1)`` bound resolves during unrolling.

        Bounds derived from an enclosing loop variable are not runtime
        parameters — the dataflow walk must stop at the loop variable and
        let emit-time unrolling supply the concrete value.
        """

        @qmc.qkernel
        def kernel(theta: qmc.Float, p: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            for i in qmc.range(p):
                for j in qmc.range(i + 1):
                    q[0] = qmc.rx(q[0], theta)
            return qmc.measure(q)

        tr = QiskitTranspiler()
        exe = tr.transpile(kernel, bindings={"p": 3}, parameters=["theta"])
        assert exe.get_first_circuit() is not None

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
