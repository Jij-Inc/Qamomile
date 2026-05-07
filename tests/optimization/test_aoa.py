import numpy as np
import pytest

from qamomile.optimization.aoa import AOAConverter
from qamomile.optimization.binary_model import BinaryModel

BACKENDS = []
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


def _make_4bit_quadratic_model() -> BinaryModel:
	return BinaryModel.from_hubo(
		{
			(0, 1): 1.0,
			(2, 3): -0.5,
			(0,): 0.2,
			(1,): -0.3,
			(2,): 0.1,
			(3,): -0.4,
		}
	)


def test_ring_pair_indices_block_size_two_has_no_duplicates():
	"""Tests that ring mixer with block_size=2 produces correct non-duplicate adjacent pair indices."""
	converter = AOAConverter(_make_4bit_quadratic_model())

	pair_indices = converter._resolve_pair_indices(
		mixer="ring",
		pair_indices=None,
		block_size=2,
	)

	expected = np.asarray([(0, 1), (2, 3)], dtype=np.uint64)
	np.testing.assert_array_equal(pair_indices, expected)


def test_explicit_pair_indices_warn_when_mixer_is_ignored():
	"""Tests that passing explicit pair_indices alongside a different mixer string emits a UserWarning."""
	converter = AOAConverter(_make_4bit_quadratic_model())
	explicit = np.asarray([(0, 1), (2, 3)], dtype=np.uint64)

	with pytest.warns(UserWarning, match="mixer=.*ignored"):
		resolved = converter._resolve_pair_indices(
			mixer="fully-connected",
			pair_indices=explicit,
			block_size=2,
		)

	assert resolved.dtype == np.uint64
	np.testing.assert_array_equal(resolved, explicit)


def test_invalid_mixer_error_contains_value():
	"""Tests that an unknown mixer string raises ValueError containing the invalid value."""
	converter = AOAConverter(_make_4bit_quadratic_model())

	with pytest.raises(ValueError, match="Unknown mixer 'invalid'"):
		converter._resolve_pair_indices(
			mixer="invalid",  # type: ignore[arg-type]
			pair_indices=None,
			block_size=2,
		)


@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
def test_transpile_with_custom_pair_indices_smoke(name, TranspilerCls):
	"""Tests that transpile with explicit pair_indices runs and returns valid bitstring samples."""
	model = BinaryModel.from_hubo({(0, 1): 1.0, (0,): -0.2})
	converter = AOAConverter(model)
	transpiler = TranspilerCls()
	executable = converter.transpile(
		transpiler,
		p=1,
		initial_state="uniform",
		pair_indices_mixer=np.asarray([(0, 1)], dtype=np.uint64),
	)

	job = executable.sample(
		transpiler.executor(),
		shots=8,
		bindings={"gammas": [0.2], "betas": [0.3]},
	)
	result = job.result()

	assert len(result.results) > 0
	for sample, count in result.results:
		assert len(sample) == model.num_bits
		assert count > 0


@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
def test_transpile_without_block_size_defaults_to_full_register_uniform(name, TranspilerCls):
	"""Tests that transpile with ring mixer and uniform initial state defaults to full-register pairing."""
	model = _make_4bit_quadratic_model()
	converter = AOAConverter(model)
	transpiler = TranspilerCls()

	executable = converter.transpile(
		transpiler,
		p=1,
		initial_state="uniform",
		mixer="ring",
	)

	job = executable.sample(
		transpiler.executor(),
		shots=8,
		bindings={"gammas": [0.2], "betas": [0.3]},
	)
	result = job.result()

	assert len(result.results) > 0
	for sample, count in result.results:
		assert len(sample) == model.num_bits
		assert count > 0


@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
def test_transpile_without_block_size_defaults_to_full_register_dicke(name, TranspilerCls):
	"""Tests that transpile with ring mixer and Dicke initial state defaults to full-register pairing."""
	model = _make_4bit_quadratic_model()
	converter = AOAConverter(model)
	transpiler = TranspilerCls()

	executable = converter.transpile(
		transpiler,
		p=1,
		initial_state="dicke",
		hamming_weight=1,
		mixer="ring",
	)

	job = executable.sample(
		transpiler.executor(),
		shots=8,
		bindings={"gammas": [0.2], "betas": [0.3]},
	)
	result = job.result()

	assert len(result.results) > 0
	for sample, count in result.results:
		assert len(sample) == model.num_bits
		assert count > 0


@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
def test_transpile_without_block_size_defaults_to_full_register_single_basis_state(name, TranspilerCls):
	"""Tests that transpile with ring mixer and single_basis_state initial state defaults to full-register pairing."""
	model = _make_4bit_quadratic_model()
	converter = AOAConverter(model)
	transpiler = TranspilerCls()

	executable = converter.transpile(
		transpiler,
		p=1,
		initial_state="single_basis_state",
		hamming_weight=1,
		mixer="ring",
	)

	job = executable.sample(
		transpiler.executor(),
		shots=8,
		bindings={"gammas": [0.2], "betas": [0.3]},
	)
	result = job.result()

	assert len(result.results) > 0
	for sample, count in result.results:
		assert len(sample) == model.num_bits
		assert count > 0


@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
@pytest.mark.parametrize(
	"initial_state,hamming_weight",
	[
		("uniform", 1),
		("dicke", 1),
		("single_basis_state", 1),
	],
)
def test_transpile_all_initial_states_smoke(name, TranspilerCls, initial_state, hamming_weight):
	"""Tests that all supported initial states transpile and sample successfully."""
	model = _make_4bit_quadratic_model()
	converter = AOAConverter(model)
	transpiler = TranspilerCls()

	executable = converter.transpile(
		transpiler,
		p=1,
		initial_state=initial_state,
		hamming_weight=hamming_weight,
		mixer="ring",
	)

	job = executable.sample(
		transpiler.executor(),
		shots=8,
		bindings={"gammas": [0.2], "betas": [0.3]},
	)
	result = job.result()

	assert len(result.results) > 0
	for sample, count in result.results:
		assert len(sample) == model.num_bits
		assert count > 0


@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
def test_preserve_hamming_weight(name, TranspilerCls):
	"""Tests that all sampled states from a Dicke-initialized AOA circuit have the expected Hamming weight."""
	hamming_weight = 2
	model = _make_4bit_quadratic_model()
	converter = AOAConverter(model)
	transpiler = TranspilerCls()

	executable = converter.transpile(
		transpiler,
		p=1,
		initial_state="dicke",
		hamming_weight=hamming_weight,
		mixer="ring",
	)

	job = executable.sample(
		transpiler.executor(),
		shots=64,
		bindings={"gammas": [0.2], "betas": [0.3]},
	)
	result = job.result()

	assert len(result.results) > 0
	for sample, _count in result.results:
		assert sum(sample) == hamming_weight


@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
def test_preserve_hamming_weight_single_basis_state(name, TranspilerCls):
	"""Tests that all sampled states from a single_basis_state-initialized AOA circuit have the expected Hamming weight."""
	hamming_weight = 2
	model = _make_4bit_quadratic_model()
	converter = AOAConverter(model)
	transpiler = TranspilerCls()

	executable = converter.transpile(
		transpiler,
		p=1,
		initial_state="single_basis_state",
		hamming_weight=hamming_weight,
		mixer="ring",
	)

	job = executable.sample(
		transpiler.executor(),
		shots=64,
		bindings={"gammas": [0.2], "betas": [0.3]},
	)
	result = job.result()

	assert len(result.results) > 0
	for sample, _count in result.results:
		assert sum(sample) == hamming_weight
