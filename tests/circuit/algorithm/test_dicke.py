"""Tests for qamomile/circuit/algorithm/dicke.py primitives."""

import numpy as np
import pytest

pytest.importorskip("qiskit")

import qamomile.circuit as qmc
from qamomile.circuit.algorithm.dicke import prepare_dicke, scs_gate_2q, scs_gate_3q
from qamomile.optimization.schedules.dicke import dicke_state_composition_schedule
from qamomile.qiskit.transpiler import QiskitTranspiler


def _gate_counts(qc):
	counts = {}
	for inst in qc.data:
		name = inst.operation.name
		counts[name] = counts.get(name, 0) + 1
	return counts


@qmc.qkernel
def _wrap_scs_gate_2q(
	n: qmc.UInt,
	t: qmc.UInt,
	c: qmc.UInt,
	theta: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
	q = qmc.qubit_array(n, name="q")
	q = scs_gate_2q(q, t, c, theta)
	return qmc.measure(q)


@qmc.qkernel
def _wrap_scs_gate_3q(
	n: qmc.UInt,
	t: qmc.UInt,
	c1: qmc.UInt,
	c2: qmc.UInt,
	theta: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
	q = qmc.qubit_array(n, name="q")
	q = scs_gate_3q(q, t, c1, c2, theta)
	return qmc.measure(q)


@qmc.qkernel
def _wrap_prepare_dicke(
	n: qmc.UInt,
	initial_ones: qmc.Vector[qmc.UInt],
	pair_indices: qmc.Matrix[qmc.UInt],
	triplets_indices: qmc.Matrix[qmc.UInt],
	pair_angles: qmc.Vector[qmc.Float],
	triplets_angles: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
	q = prepare_dicke(
		n,
		initial_ones,
		pair_indices,
		triplets_indices,
		pair_angles,
		triplets_angles,
	)
	return qmc.measure(q)


def test_scs_gate_2q_uses_expected_cnot_and_ry_counts():
	transpiler = QiskitTranspiler()
	exe = transpiler.transpile(
		_wrap_scs_gate_2q,
		bindings={"n": 2, "t": 0, "c": 1, "theta": 0.3},
	)
	qc = exe.compiled_quantum[0].circuit
	counts = _gate_counts(qc)

	assert counts.get("cx", 0) == 4
	assert counts.get("ry", 0) == 2


def test_scs_gate_3q_uses_expected_cnot_and_ry_counts():
	transpiler = QiskitTranspiler()
	exe = transpiler.transpile(
		_wrap_scs_gate_3q,
		bindings={"n": 3, "t": 0, "c1": 1, "c2": 2, "theta": 0.3},
	)
	qc = exe.compiled_quantum[0].circuit
	counts = _gate_counts(qc)

	assert counts.get("cx", 0) == 6
	assert counts.get("ry", 0) == 4


def test_prepare_dicke_applies_basis_initialization_and_scs_rotations():
	(
		initial_ones,
		pair_indices,
		triplets_indices,
		pair_angles,
		triplets_angles,
	) = dicke_state_composition_schedule(n_qubits=3, block_size=3, hamming_weight=2)

	transpiler = QiskitTranspiler()
	exe = transpiler.transpile(
		_wrap_prepare_dicke,
		bindings={
			"n": 3,
			"initial_ones": initial_ones,
			"pair_indices": pair_indices,
			"triplets_indices": triplets_indices,
			"pair_angles": pair_angles,
			"triplets_angles": triplets_angles,
		},
	)
	qc = exe.compiled_quantum[0].circuit
	counts = _gate_counts(qc)

	assert counts.get("x", 0) == len(initial_ones)
	assert counts.get("ry", 0) == 2 * len(pair_indices) + 4 * len(triplets_indices)
	assert counts.get("cx", 0) == 4 * len(pair_indices) + 6 * len(triplets_indices)