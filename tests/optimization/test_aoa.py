import numpy as np
import pytest

pytest.importorskip("qiskit")

from qamomile.optimization.aoa import AOAConverter
from qamomile.optimization.binary_model import BinaryModel
from qamomile.qiskit.transpiler import QiskitTranspiler


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
	converter = AOAConverter(_make_4bit_quadratic_model())

	pair_indices = converter._resolve_pair_indices(
		mixer="ring",
		pair_indices=None,
		block_size=2,
	)

	expected = np.asarray([(0, 1), (2, 3)], dtype=np.uint64)
	np.testing.assert_array_equal(pair_indices, expected)


def test_explicit_pair_indices_warn_when_mixer_is_ignored():
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
	converter = AOAConverter(_make_4bit_quadratic_model())

	with pytest.raises(ValueError, match="Unknown mixer 'invalid'"):
		converter._resolve_pair_indices(
			mixer="invalid",  # type: ignore[arg-type]
			pair_indices=None,
			block_size=2,
		)


def test_transpile_with_custom_pair_indices_smoke():
	model = BinaryModel.from_hubo({(0, 1): 1.0, (0,): -0.2})
	converter = AOAConverter(model)
	transpiler = QiskitTranspiler()

	executable = converter.transpile(
		transpiler,
		p=1,
		pair_indices=np.asarray([(0, 1)], dtype=np.uint64),
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
