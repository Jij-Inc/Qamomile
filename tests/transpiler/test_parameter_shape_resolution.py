"""Tests for ParameterShapeResolutionPass (Layer 2).

This pass substitutes symbolic ``{name}_dim{i}`` Values with concrete
constants whenever ``bindings`` carries a concrete array-like for the
corresponding parameter. That unlocks compile-time loop unrolling for
code that uses ``arr.shape[0]`` as a loop bound, while leaving library
patterns untouched when the shape is not queried.
"""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("qiskit")

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.basic import superposition_vector
from qamomile.circuit.algorithm.qaoa import x_mixer
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.value import ArrayValue
from qamomile.circuit.transpiler.passes.parameter_shape_resolution import (
    ParameterShapeResolutionPass,
    _extract_concrete_shape,
)
from qamomile.qiskit.transpiler import QiskitTranspiler


def _as_hierarchical(block: Block) -> Block:
    """``kernel.build()`` returns a TRACED block; the transpile pipeline
    flips it to HIERARCHICAL in ``to_block``. Do the same for unit tests
    that invoke the pass directly."""
    return dataclasses.replace(block, kind=BlockKind.HIERARCHICAL)


class TestExtractConcreteShape:
    def test_list_binding(self):
        assert _extract_concrete_shape([0.1, 0.2, 0.3]) == (3,)

    def test_tuple_binding(self):
        assert _extract_concrete_shape((1.0, 2.0)) == (2,)

    def test_numpy_array_binding(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert _extract_concrete_shape(arr) == (3, 2)

    def test_scalar_binding_returns_none(self):
        assert _extract_concrete_shape(3.14) is None

    def test_dict_binding_returns_none(self):
        assert _extract_concrete_shape({0: 1.0}) is None

    def test_none_binding_returns_none(self):
        assert _extract_concrete_shape(None) is None


class TestPassDirectInvocation:
    def _make_kernel(self):
        @qmc.qkernel
        def kernel(gamma: qmc.Vector[qmc.Float]) -> qmc.Float:
            acc = gamma[0]
            for i in qmc.range(gamma.shape[0]):
                acc = acc + gamma[i]
            return acc

        return kernel

    def test_binding_folds_symbolic_dim(self):
        kernel = self._make_kernel()
        block = _as_hierarchical(kernel.build(parameters=["gamma"]))

        resolved = ParameterShapeResolutionPass(
            bindings={"gamma": [0.1, 0.2, 0.3]}
        ).run(block)

        gamma = next(
            v for v in resolved.input_values if getattr(v, "name", None) == "gamma"
        )
        assert isinstance(gamma, ArrayValue)
        assert gamma.shape[0].is_constant()
        assert gamma.shape[0].get_const() == 3

    def test_no_binding_leaves_symbolic(self):
        kernel = self._make_kernel()
        block = _as_hierarchical(kernel.build(parameters=["gamma"]))

        resolved = ParameterShapeResolutionPass(bindings={}).run(block)

        gamma = next(
            v for v in resolved.input_values if getattr(v, "name", None) == "gamma"
        )
        assert not gamma.shape[0].is_constant()

    def test_non_array_binding_leaves_symbolic(self):
        kernel = self._make_kernel()
        block = _as_hierarchical(kernel.build(parameters=["gamma"]))

        # Scalar binding should not resolve a Vector dim.
        resolved = ParameterShapeResolutionPass(bindings={"gamma": 42}).run(block)

        gamma = next(
            v for v in resolved.input_values if getattr(v, "name", None) == "gamma"
        )
        assert not gamma.shape[0].is_constant()

    def test_hierarchical_block_kind_required(self):
        kernel = self._make_kernel()
        block = kernel.build(parameters=["gamma"])
        # Simulate a post-inline block to ensure we reject the wrong kind.
        affine = dataclasses.replace(block, kind=BlockKind.AFFINE)

        from qamomile.circuit.transpiler.errors import ValidationError

        with pytest.raises(ValidationError):
            ParameterShapeResolutionPass(bindings={"gamma": [0.0]}).run(affine)


class TestEndToEndTranspile:
    """End-to-end: user's ipynb pattern with concrete gamma/beta bindings.

    Before Layer 2, the transpiler silently produced a ``H H`` circuit with
    the entire QAOA loop elided. After Layer 2, binding concrete arrays for
    ``gamma`` / ``beta`` resolves their shape dims and the loop unrolls.
    """

    def _make_h(self) -> qm_o.Hamiltonian:
        H = qm_o.Hamiltonian()
        H.add_term(
            (
                qm_o.PauliOperator(qm_o.Pauli.Z, 0),
                qm_o.PauliOperator(qm_o.Pauli.Z, 1),
            ),
            1.0,
        )
        return H

    def test_flat_kernel_with_concrete_bindings_emits_qaoa_layers(self):
        @qmc.qkernel
        def qaoa(
            n: qmc.UInt,
            gamma: qmc.Vector[qmc.Float],
            beta: qmc.Vector[qmc.Float],
            hamiltonian: qmc.Observable,
        ) -> qmc.Float:
            q = superposition_vector(n)
            for p in qmc.range(gamma.shape[0]):
                q = qmc.pauli_evolve(q, hamiltonian, gamma[p])
                q = x_mixer(q, beta[p])
            return qmc.expval(q, hamiltonian)

        H = self._make_h()
        tr = QiskitTranspiler()
        exe = tr.transpile(
            qaoa,
            bindings={
                "n": H.num_qubits,
                "hamiltonian": H,
                "gamma": [0.3, 0.5],
                "beta": [0.4, 0.6],
            },
        )
        circuit = exe.compiled_quantum[0].circuit
        gate_names = [inst.operation.name for inst in circuit.data]
        # 2 H (init) + 2 PauliEvolution (cost layers) + 4 Rx (mixer layers)
        assert gate_names.count("h") == 2
        assert gate_names.count("PauliEvolution") == 2
        assert gate_names.count("rx") == 4
        assert circuit.size() == 8

    def test_circuit_depth_scales_with_gamma_length(self):
        @qmc.qkernel
        def qaoa(
            n: qmc.UInt,
            gamma: qmc.Vector[qmc.Float],
            beta: qmc.Vector[qmc.Float],
            hamiltonian: qmc.Observable,
        ) -> qmc.Float:
            q = superposition_vector(n)
            for p in qmc.range(gamma.shape[0]):
                q = qmc.pauli_evolve(q, hamiltonian, gamma[p])
                q = x_mixer(q, beta[p])
            return qmc.expval(q, hamiltonian)

        H = self._make_h()
        tr = QiskitTranspiler()

        def _size(p: int) -> int:
            exe = tr.transpile(
                qaoa,
                bindings={
                    "n": H.num_qubits,
                    "hamiltonian": H,
                    "gamma": [0.1] * p,
                    "beta": [0.2] * p,
                },
            )
            return exe.compiled_quantum[0].circuit.size()

        size1 = _size(1)
        size2 = _size(2)
        size3 = _size(3)
        assert size1 < size2 < size3

    def test_library_pattern_still_works_without_shape_query(self):
        """Library pattern: ``p`` bound, ``gammas.shape`` never queried."""
        from qamomile.circuit.algorithm.qaoa import qaoa_layers

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
        # Should succeed; parameters remain symbolic backend-side.
        assert exe.compiled_quantum[0].circuit.num_parameters >= 2
