"""Tests for qamomile/circuit/algorithm/aoa.py primitives."""

import re

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.aoa import (
	aoa_state_dicke,
	aoa_state_superposition,
	xy_mixer,
	xy_pair_rotation,
)

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

BACKENDS: list[tuple[str, type]] = []
try:
	import qiskit  # noqa: F401
	from qamomile.qiskit.transpiler import QiskitTranspiler
	BACKENDS.append(("qiskit", QiskitTranspiler))
except ImportError:
	pass
try:
	import quri_parts  # noqa: F401
	from qamomile.quri_parts.transpiler import QuriPartsTranspiler
	BACKENDS.append(("quri_parts", QuriPartsTranspiler))
except ImportError:
	pass
try:
	import cudaq  # noqa: F401
	from qamomile.cudaq.transpiler import CudaqTranspiler
	BACKENDS.append(("cudaq", CudaqTranspiler))
except ImportError:
	pass

if not BACKENDS:
	pytest.skip("No quantum backend available", allow_module_level=True)

# ---------------------------------------------------------------------------
# Per-backend gate-count helpers  (all return dict[str, int] with Qiskit-style
# lowercase keys: "cx", "rx", "ry", "rz", "h", "x", "rzz", …)
# ---------------------------------------------------------------------------

def _qiskit_gate_counts(exe) -> dict[str, int]:
	qc = exe.compiled_quantum[0].circuit
	counts: dict[str, int] = {}
	for inst in qc.data:
		name = inst.operation.name
		counts[name] = counts.get(name, 0) + 1
	return counts

_QURI_PARTS_CANONICAL: dict[str, str] = {
	"H": "h", "X": "x", "Y": "y", "Z": "z",
	"S": "s", "Sdag": "sdg", "T": "t", "Tdag": "tdg",
	"CNOT": "cx", "CZ": "cz", "SWAP": "swap",
	"RX": "rx", "ParametricRX": "rx",
	"RY": "ry", "ParametricRY": "ry",
	"RZ": "rz", "ParametricRZ": "rz",
	"PauliRotation": "rzz", "ParametricPauliRotation": "rzz",
}

def _quri_parts_gate_counts(exe) -> dict[str, int]:
	circuit = exe.compiled_quantum[0].circuit
	counts: dict[str, int] = {}
	for gate in circuit.gates:
		canon = _QURI_PARTS_CANONICAL.get(gate.name, gate.name.lower())
		counts[canon] = counts.get(canon, 0) + 1
	return counts

_CUDAQ_PATTERNS: dict[str, re.Pattern] = {
	"cx": re.compile(r"x\.ctrl\("),
	"rx": re.compile(r"\brx\("),
	"ry": re.compile(r"\bry\("),
	"rz": re.compile(r"\brz\("),
	"h":  re.compile(r"\bh\("),
	"x":  re.compile(r"\bx\("),
}

def _cudaq_gate_counts(exe) -> dict[str, int]:
	source = exe.compiled_quantum[0].source
	return {name: len(pat.findall(source)) for name, pat in _CUDAQ_PATTERNS.items()}


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
	pair_indices_mixer: qmc.Matrix[qmc.UInt],
) -> qmc.Vector[qmc.Bit]:
	q = qmc.qubit_array(n, name="q")
	q = xy_mixer(q, beta, pair_indices_mixer)
	return qmc.measure(q)


@qmc.qkernel
def _wrap_aoa_state_superposition(
	p: qmc.UInt,
	quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
	linear: qmc.Dict[qmc.UInt, qmc.Float],
	n: qmc.UInt,
	gammas: qmc.Vector[qmc.Float],
	betas: qmc.Vector[qmc.Float],
	pair_indices_mixer: qmc.Matrix[qmc.UInt],
) -> qmc.Vector[qmc.Bit]:
	q = aoa_state_superposition(
		p=p,
		quad=quad,
		linear=linear,
		n=n,
		gammas=gammas,
		betas=betas,
		pair_indices_mixer=pair_indices_mixer,
	)
	return qmc.measure(q)


@qmc.qkernel
def _wrap_aoa_state_dicke(
	p: qmc.UInt,
	quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
	linear: qmc.Dict[qmc.UInt, qmc.Float],
	n: qmc.UInt,
	gammas: qmc.Vector[qmc.Float],
	betas: qmc.Vector[qmc.Float],
	pair_indices_mixer: qmc.Matrix[qmc.UInt],
	initial_ones: qmc.Vector[qmc.UInt],
	pair_indices_dicke: qmc.Matrix[qmc.UInt],
	triplets_indices_dicke: qmc.Matrix[qmc.UInt],
	pair_angles_dicke: qmc.Vector[qmc.Float],
	triplets_angles_dicke: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
	q = aoa_state_dicke(
		p=p,
		quad=quad,
		linear=linear,
		n=n,
		gammas=gammas,
		betas=betas,
		pair_indices_mixer=pair_indices_mixer,
		initial_ones=initial_ones,
		pair_indices_dicke=pair_indices_dicke,
		triplets_indices_dicke=triplets_indices_dicke,
		pair_angles_dicke=pair_angles_dicke,
		triplets_angles_dicke=triplets_angles_dicke,
	)
	return qmc.measure(q)


@qmc.qkernel
def _wrap_aoa_state_superposition_expval(
	p: qmc.UInt,
	quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
	linear: qmc.Dict[qmc.UInt, qmc.Float],
	n: qmc.UInt,
	gammas: qmc.Vector[qmc.Float],
	betas: qmc.Vector[qmc.Float],
	pair_indices_mixer: qmc.Matrix[qmc.UInt],
	hamiltonian: qmc.Observable,
) -> qmc.Float:
	q = aoa_state_superposition(
		p=p,
		quad=quad,
		linear=linear,
		n=n,
		gammas=gammas,
		betas=betas,
		pair_indices_mixer=pair_indices_mixer,
	)
	return qmc.expval(q, hamiltonian)


@qmc.qkernel
def _wrap_aoa_state_dicke_expval(
	p: qmc.UInt,
	quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
	linear: qmc.Dict[qmc.UInt, qmc.Float],
	n: qmc.UInt,
	gammas: qmc.Vector[qmc.Float],
	betas: qmc.Vector[qmc.Float],
	pair_indices_mixer: qmc.Matrix[qmc.UInt],
	initial_ones: qmc.Vector[qmc.UInt],
	pair_indices_dicke: qmc.Matrix[qmc.UInt],
	triplets_indices_dicke: qmc.Matrix[qmc.UInt],
	pair_angles_dicke: qmc.Vector[qmc.Float],
	triplets_angles_dicke: qmc.Vector[qmc.Float],
	hamiltonian: qmc.Observable,
) -> qmc.Float:
	q = aoa_state_dicke(
		p=p,
		quad=quad,
		linear=linear,
		n=n,
		gammas=gammas,
		betas=betas,
		pair_indices_mixer=pair_indices_mixer,
		initial_ones=initial_ones,
		pair_indices_dicke=pair_indices_dicke,
		triplets_indices_dicke=triplets_indices_dicke,
		pair_angles_dicke=pair_angles_dicke,
		triplets_angles_dicke=triplets_angles_dicke,
	)
	return qmc.expval(q, hamiltonian)

# ---------------------------------------------------------------------------
# Primitive tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
def test_xy_pair_rotation_gate_counts(name, TranspilerCls):
	"""Tests that xy_pair_rotation emits the expected number of CX, RX, and RZ gates."""
	transpiler = TranspilerCls()
	exe = transpiler.transpile(
		_wrap_xy_pair_rotation,
		bindings={"n": 2, "i": 0, "j": 1, "beta": 0.3},
	)

	match name:
		case "qiskit":
			counts = _qiskit_gate_counts(exe)
			assert counts.get("cx", 0) == 2
			assert counts.get("rx", 0) == 5
			assert counts.get("rz", 0) == 1
		case "quri_parts":
			counts = _quri_parts_gate_counts(exe)
			assert counts.get("cx", 0) == 2
			assert counts.get("rx", 0) == 5
			assert counts.get("rz", 0) == 1
		case "cudaq":
			counts = _cudaq_gate_counts(exe)
			assert counts.get("cx", 0) == 2
			assert counts.get("rx", 0) == 5
			assert counts.get("rz", 0) == 1


@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
def test_xy_mixer_gate_counts_scale_with_pairs(name, TranspilerCls):
	"""Tests that xy_mixer gate counts scale linearly with the number of qubit pairs."""
	pair_indices_mixer = np.asarray([(0, 1), (2, 3)], dtype=np.uint64)

	transpiler = TranspilerCls()
	exe = transpiler.transpile(
		_wrap_xy_mixer,
		bindings={"n": 4, "beta": 0.2, "pair_indices_mixer": pair_indices_mixer},
	)

	n_pairs = len(pair_indices_mixer)

	match name:
		case "qiskit":
			counts = _qiskit_gate_counts(exe)
			assert counts.get("cx", 0) == 2 * n_pairs
			assert counts.get("rx", 0) == 5 * n_pairs
			assert counts.get("rz", 0) == n_pairs
		case "quri_parts":
			counts = _quri_parts_gate_counts(exe)
			assert counts.get("cx", 0) == 2 * n_pairs
			assert counts.get("rx", 0) == 5 * n_pairs
			assert counts.get("rz", 0) == n_pairs
		case "cudaq":
			counts = _cudaq_gate_counts(exe)
			assert counts.get("cx", 0) == 2 * n_pairs
			assert counts.get("rx", 0) == 5 * n_pairs
			assert counts.get("rz", 0) == n_pairs


@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
def test_aoa_state_includes_superposition_and_cost_layer(name, TranspilerCls):
	"""Tests that aoa_state_superposition produces Hadamard and ZZ cost-layer gates."""
	transpiler = TranspilerCls()
	exe = transpiler.transpile(
		_wrap_aoa_state_superposition,
		bindings={
			"p": 1,
			"quad": {(0, 1): 0.5},
			"linear": {0: -0.2},
			"n": 2,
			"gammas": [0.3],
			"betas": [0.2],
			"pair_indices_mixer": np.asarray([(0, 1)], dtype=np.uint64),
		},
	)

	match name:
		case "qiskit":
			counts = _qiskit_gate_counts(exe)
			assert counts.get("h", 0) == 2   # superposition layer
			assert counts.get("rzz", 0) == 1  # cost layer
		case "quri_parts":
			counts = _quri_parts_gate_counts(exe)
			assert counts.get("h", 0) == 2    # superposition layer
			assert counts.get("rzz", 0) == 1  # cost layer (ParametricPauliRotation → rzz)
		case "cudaq":
			counts = _cudaq_gate_counts(exe)
			assert counts.get("h", 0) == 2    # superposition layer
			# RZZ decomposes to CX + RZ + CX; XY mixer adds 2 more CX
			assert counts.get("cx", 0) == 4


@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
def test_aoa_state_superposition_expval_z_sum_is_zero(name, TranspilerCls):
	"""Tests that <+|^2 Z_0 + Z_1 |+>^2 = 0 via the estimator (run) path.

	With gammas=[0] and betas=[0] the circuit reduces to the uniform-superposition
	initialisation (Hadamard on every qubit). The expectation of Z on any qubit
	in the |+> state is exactly 0, so <Z_0 + Z_1> = 0.
	This test exercises the expval / estimator code path for the AOA superposition
	initial state.
	"""
	H = qm_o.Z(0) + qm_o.Z(1)

	transpiler = TranspilerCls()
	exe = transpiler.transpile(
		_wrap_aoa_state_superposition_expval,
		bindings={
			"p": 1,
			"quad": {},
			"linear": {},
			"n": 2,
			"gammas": [0.0],
			"betas": [0.0],
			"pair_indices_mixer": np.asarray([(0, 1)], dtype=np.uint64),
			"hamiltonian": H,
		},
	)

	job = exe.run(transpiler.executor())
	result = job.result()

	assert abs(result) < 1e-6


@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
def test_aoa_state_dicke_expval_z_sum_is_zero(name, TranspilerCls):
	"""Tests that <D^2_1|Z_0 + Z_1|D^2_1> = 0 via the estimator (run) path.

	|D^2_1> = (|01> + |10>) / sqrt(2) with gammas=[0] (no cost evolution) and
	betas=[0] (no mixer evolution). By symmetry <Z_0> = <Z_1> = 0 exactly.
	This test exercises the expval / estimator code path for the AOA Dicke
	initial state.
	"""
	from qamomile.optimization.schedules.dicke import dicke_state_composition_schedule

	(
		initial_ones,
		pair_indices_dicke,
		triplets_indices_dicke,
		pair_angles_dicke,
		triplets_angles_dicke,
	) = dicke_state_composition_schedule(n_qubits=2, block_size=2, hamming_weight=1)

	H = qm_o.Z(0) + qm_o.Z(1)

	transpiler = TranspilerCls()
	exe = transpiler.transpile(
		_wrap_aoa_state_dicke_expval,
		bindings={
			"p": 1,
			"quad": {},
			"linear": {},
			"n": 2,
			"gammas": [0.0],
			"betas": [0.0],
			"pair_indices_mixer": np.asarray([(0, 1)], dtype=np.uint64),
			"initial_ones": initial_ones,
			"pair_indices_dicke": pair_indices_dicke,
			"triplets_indices_dicke": triplets_indices_dicke,
			"pair_angles_dicke": pair_angles_dicke,
			"triplets_angles_dicke": triplets_angles_dicke,
			"hamiltonian": H,
		},
	)

	job = exe.run(transpiler.executor())
	result = job.result()

	assert abs(result) < 1e-6

