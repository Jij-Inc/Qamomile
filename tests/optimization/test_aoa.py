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


def _make_3bit_hubo_model() -> BinaryModel:
    """Create a 3-variable HUBO model with a cubic 3-body term."""
    return BinaryModel.from_hubo(
        {
            (0, 1, 2): 1.0,
            (0, 1): -0.5,
            (0,): 0.2,
        }
    )


def _make_4bit_quadratic_model() -> BinaryModel:
    """Create a 4-variable quadratic model with some positive and negative coefficients."""
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

    pair_indices = converter.resolve_pair_indices(
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
        resolved = converter.resolve_pair_indices(
            mixer="fully-connected",
            pair_indices=explicit,
            block_size=2,
        )

    assert resolved.dtype == np.uint64
    np.testing.assert_array_equal(resolved, explicit)


def test_invalid_mixer_error_contains_value():
    """Tests that an unknown mixer string raises ValueError containing the invalid value."""
    converter = AOAConverter(_make_4bit_quadratic_model())

    with pytest.raises(ValueError, match="invalid.*MixerName|MixerName.*invalid"):
        converter.resolve_pair_indices(
            mixer="invalid",  # type: ignore[arg-type]
            pair_indices=None,
            block_size=2,
        )


def test_invalid_initial_state_raises():
    """Tests that an unknown initial_state string raises ValueError."""
    converter = AOAConverter(_make_4bit_quadratic_model())

    with pytest.raises(ValueError, match="bogus"):
        converter.transpile(
            BACKENDS[0][1](),
            p=1,
            initial_state="bogus",  # type: ignore[arg-type]
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
def test_transpile_without_block_size_defaults_to_full_register_uniform(
    name, TranspilerCls
):
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
def test_transpile_without_block_size_defaults_to_full_register_dicke(
    name, TranspilerCls
):
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
def test_transpile_without_block_size_defaults_to_full_register_single_basis_state(
    name, TranspilerCls
):
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
def test_transpile_all_initial_states_smoke(
    name, TranspilerCls, initial_state, hamming_weight
):
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
@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize(
    "block_size,hamming_weight",
    [(2, 1), (4, 1), (4, 2)],
)
def test_preserve_hamming_weight(
    name, TranspilerCls, seed, p, block_size, hamming_weight
):
    """Tests that Dicke-initialized AOA preserves Hamming weight across seeds, p, and register sizes.

    For n=2*block_size qubits with ring mixer and Dicke initial state, every sample
    must have Hamming weight equal to num_blocks * hamming_weight per block.
    """
    rng = np.random.default_rng(seed)
    n_qubits = 2 * block_size
    gammas = rng.uniform(-np.pi, np.pi, size=p).tolist()
    betas = rng.uniform(-np.pi, np.pi, size=p).tolist()

    model = BinaryModel.from_hubo(
        {(i, (i + 1) % n_qubits): 1.0 for i in range(n_qubits)}
    )
    expected_weight = hamming_weight * (n_qubits // block_size)
    converter = AOAConverter(model)
    transpiler = TranspilerCls()

    executable = converter.transpile(
        transpiler,
        p=p,
        initial_state="dicke",
        hamming_weight=hamming_weight,
        mixer="ring",
        block_size=block_size,
    )

    job = executable.sample(
        transpiler.executor(),
        shots=32,
        bindings={"gammas": gammas, "betas": betas},
    )
    result = job.result()

    assert len(result.results) > 0
    for sample, _count in result.results:
        assert sum(sample) == expected_weight


@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize(
    "block_size,hamming_weight",
    [(2, 1), (4, 1), (4, 2)],
)
def test_preserve_hamming_weight_single_basis_state(
    name, TranspilerCls, seed, p, block_size, hamming_weight
):
    """Tests that single_basis_state-initialized AOA preserves Hamming weight across seeds, p, and register sizes."""
    rng = np.random.default_rng(seed)
    n_qubits = 2 * block_size
    gammas = rng.uniform(-np.pi, np.pi, size=p).tolist()
    betas = rng.uniform(-np.pi, np.pi, size=p).tolist()

    model = BinaryModel.from_hubo(
        {(i, (i + 1) % n_qubits): 1.0 for i in range(n_qubits)}
    )
    expected_weight = hamming_weight * (n_qubits // block_size)
    converter = AOAConverter(model)
    transpiler = TranspilerCls()

    executable = converter.transpile(
        transpiler,
        p=p,
        initial_state="single_basis_state",
        hamming_weight=hamming_weight,
        mixer="ring",
        block_size=block_size,
    )

    job = executable.sample(
        transpiler.executor(),
        shots=32,
        bindings={"gammas": gammas, "betas": betas},
    )
    result = job.result()

    assert len(result.results) > 0
    for sample, _count in result.results:
        assert sum(sample) == expected_weight


# ---------------------------------------------------------------------------
# HUBO converter path tests (P1: covers _transpile_hubo*)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
@pytest.mark.parametrize(
    "initial_state",
    ["uniform", "dicke", "single_basis_state"],
)
def test_hubo_transpile_all_initial_states_smoke(name, TranspilerCls, initial_state):
    """Tests that all supported initial states for HUBO models transpile and sample successfully.

    Exercises the three HUBO converter branches: _transpile_hubo (uniform),
    _transpile_hubo_dicke (dicke), and _transpile_hubo_basis_state (single_basis_state).
    The model has a 3-body term so spin_model.higher is non-empty, triggering the HUBO path.
    """
    model = _make_3bit_hubo_model()
    converter = AOAConverter(model)
    transpiler = TranspilerCls()

    executable = converter.transpile(
        transpiler,
        p=1,
        initial_state=initial_state,
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
@pytest.mark.parametrize("seed", [0, 42])
def test_hubo_preserve_hamming_weight_dicke(name, TranspilerCls, seed):
    """Tests that HUBO AOA with Dicke initial state preserves Hamming weight in all samples.

    The model has a 3-body higher-order term triggering the HUBO path. With the Dicke
    initial state and the ring XY mixer, every sample must have Hamming weight 1 per
    block by symmetry of the XY mixer within the feasible subspace.
    """
    rng = np.random.default_rng(seed)
    block_size = 3
    n_qubits = 2 * block_size
    gammas = rng.uniform(-np.pi, np.pi, size=1).tolist()
    betas = rng.uniform(-np.pi, np.pi, size=1).tolist()

    model = BinaryModel.from_hubo(
        {(0, 1, 2): 1.0, **{(i, (i + 1) % n_qubits): 0.5 for i in range(n_qubits)}}
    )
    converter = AOAConverter(model)
    transpiler = TranspilerCls()

    executable = converter.transpile(
        transpiler,
        p=1,
        initial_state="dicke",
        hamming_weight=1,
        mixer="ring",
        block_size=block_size,
    )

    job = executable.sample(
        transpiler.executor(),
        shots=32,
        bindings={"gammas": gammas, "betas": betas},
    )
    result = job.result()

    expected_weight = n_qubits // block_size
    assert len(result.results) > 0
    for sample, _count in result.results:
        assert sum(sample) == expected_weight
