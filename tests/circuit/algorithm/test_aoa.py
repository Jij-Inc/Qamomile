"""Tests for qamomile/circuit/algorithm/aoa.py primitives."""

import numpy as np
import pytest

pytest.importorskip("qiskit")

import qamomile.circuit as qmc
from qamomile.circuit.algorithm.aoa import aoa_state, xy_mixer, xy_pair_rotation
from qamomile.qiskit.transpiler import QiskitTranspiler


def _gate_counts(qc):
	counts = {}
	for inst in qc.data:
		name = inst.operation.name
		counts[name] = counts.get(name, 0) + 1
	return counts


@qmc.qkernel
def _wrap_xy_pair_rotation(
	n: qmc.UInt,
	i: qmc.UInt,
	j: qmc.UInt,
	beta: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
	q = qmc.qubit_array(n, name="q")
	q = xy_pair_rotation(q, i, j, beta)
	return qmc.measure(q)


@qmc.qkernel
def _wrap_xy_mixer(
	n: qmc.UInt,
	beta: qmc.Float,
	pair_indices: qmc.Matrix[qmc.UInt],
) -> qmc.Vector[qmc.Bit]:
	q = qmc.qubit_array(n, name="q")
	q = xy_mixer(q, beta, pair_indices)
	return qmc.measure(q)


@qmc.qkernel
def _wrap_aoa_state(
	p: qmc.UInt,
	quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
	linear: qmc.Dict[qmc.UInt, qmc.Float],
	n: qmc.UInt,
	gammas: qmc.Vector[qmc.Float],
	betas: qmc.Vector[qmc.Float],
	pair_indices: qmc.Matrix[qmc.UInt],
) -> qmc.Vector[qmc.Bit]:
	q = aoa_state(
		p=p,
		quad=quad,
		linear=linear,
		n=n,
		gammas=gammas,
		betas=betas,
		pair_indices=pair_indices,
	)
	return qmc.measure(q)


def test_xy_pair_rotation_gate_counts():
	transpiler = QiskitTranspiler()
	exe = transpiler.transpile(
		_wrap_xy_pair_rotation,
		bindings={"n": 2, "i": 0, "j": 1, "beta": 0.3},
	)
	qc = exe.compiled_quantum[0].circuit
	counts = _gate_counts(qc)

	assert counts.get("cx", 0) == 2
	assert counts.get("rx", 0) == 5
	assert counts.get("rz", 0) == 1


def test_xy_mixer_gate_counts_scale_with_pairs():
	pair_indices = np.asarray([(0, 1), (2, 3)], dtype=np.uint64)

	transpiler = QiskitTranspiler()
	exe = transpiler.transpile(
		_wrap_xy_mixer,
		bindings={"n": 4, "beta": 0.2, "pair_indices": pair_indices},
	)
	qc = exe.compiled_quantum[0].circuit
	counts = _gate_counts(qc)

	assert counts.get("cx", 0) == 2 * len(pair_indices)
	assert counts.get("rx", 0) == 5 * len(pair_indices)
	assert counts.get("rz", 0) == len(pair_indices)


def test_aoa_state_includes_superposition_and_cost_layer():
	transpiler = QiskitTranspiler()
	exe = transpiler.transpile(
		_wrap_aoa_state,
		bindings={
			"p": 1,
			"quad": {(0, 1): 0.5},
			"linear": {0: -0.2},
			"n": 2,
			"gammas": [0.3],
			"betas": [0.2],
			"pair_indices": np.asarray([(0, 1)], dtype=np.uint64),
		},
	)
	qc = exe.compiled_quantum[0].circuit
	counts = _gate_counts(qc)

	assert counts.get("h", 0) == 2
	assert counts.get("rzz", 0) == 1
