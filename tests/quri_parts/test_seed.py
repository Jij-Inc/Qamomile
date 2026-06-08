"""Tests for reproducible qulacs sampling via a seeded QURI Parts executor.

These regression tests cover the seed support added to
``QuriPartsExecutor`` / ``QuriPartsTranspiler.executor``: sampling the
qulacs vector simulator with the same seed and circuit must yield
identical shot counts, which is what unblocks reproducible tutorials and
benchmarks.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.quri_parts

pytest.importorskip("quri_parts.circuit")
pytest.importorskip("quri_parts.qulacs")

import qamomile.circuit as qmc  # noqa: E402
from qamomile.circuit import qkernel  # noqa: E402
from qamomile.circuit.frontend.constructors import qubit_array  # noqa: E402
from qamomile.quri_parts.transpiler import (  # noqa: E402
    QuriPartsExecutor,
    QuriPartsTranspiler,
)


def _uniform_superposition_kernel(n: int):
    """Build a kernel measuring ``n`` qubits in a uniform superposition.

    Sampling a uniform superposition is a strong test of seeding: every
    bitstring is equally likely, so byte-identical histograms across runs
    only happen when the seed truly controls the sampler's randomness.

    Args:
        n (int): Number of qubits to allocate, apply Hadamard to, and
            measure.

    Returns:
        QKernel: A qkernel returning the measured bits of an ``n``-qubit
            register prepared in a uniform superposition.
    """

    @qkernel
    def circuit() -> qmc.Vector[qmc.Bit]:
        qs = qubit_array(n, "qs")
        for i in qmc.range(n):
            qs[i] = qmc.h(qs[i])
        return qmc.measure(qs)

    return circuit


def _counts(executable, executor, shots: int) -> dict[tuple[int, ...], int]:
    """Sample ``executable`` with ``executor`` and return a counts dict.

    Args:
        executable (ExecutableProgram): The transpiled program to sample.
        executor (QuriPartsExecutor): The executor to run sampling with.
        shots (int): Number of measurement shots.

    Returns:
        dict[tuple[int, ...], int]: Mapping from measured bit tuples to
            their observed counts.
    """
    result = executable.sample(executor, shots=shots).result()
    return {bits: count for bits, count in result.results}


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_same_seed_gives_identical_counts(n: int, seed: int) -> None:
    """Two executors with the same seed produce identical sampling counts."""
    transpiler = QuriPartsTranspiler()
    exe = transpiler.transpile(_uniform_superposition_kernel(n))

    counts_a = _counts(exe, transpiler.executor(seed=seed), shots=2000)
    counts_b = _counts(exe, transpiler.executor(seed=seed), shots=2000)

    assert counts_a == counts_b
    assert sum(counts_a.values()) == 2000


@pytest.mark.parametrize("seed", [0, 7, 123])
def test_seed_is_reusable_on_one_executor(seed: int) -> None:
    """A single seeded executor yields identical counts on repeated samples.

    The sampler is cached on first use, so this guards against the seed
    being consumed/advanced between calls on the same executor instance.
    """
    transpiler = QuriPartsTranspiler()
    exe = transpiler.transpile(_uniform_superposition_kernel(3))
    executor = transpiler.executor(seed=seed)

    counts_a = _counts(exe, executor, shots=1500)
    counts_b = _counts(exe, executor, shots=1500)

    assert counts_a == counts_b


def test_different_seeds_can_differ() -> None:
    """Distinct seeds generally yield different sampling counts.

    The seeds are fixed (0 and 1), so this assertion is deterministic
    rather than probabilistic: with three qubits and 2000 shots these two
    specific seeds produce different histograms, proving the seed is
    genuinely threaded through rather than ignored.
    """
    transpiler = QuriPartsTranspiler()
    exe = transpiler.transpile(_uniform_superposition_kernel(3))

    counts_seed0 = _counts(exe, transpiler.executor(seed=0), shots=2000)
    counts_seed1 = _counts(exe, transpiler.executor(seed=1), shots=2000)

    assert counts_seed0 != counts_seed1


def test_unseeded_executor_still_samples() -> None:
    """An executor without a seed keeps working (non-deterministic path)."""
    transpiler = QuriPartsTranspiler()
    exe = transpiler.transpile(_uniform_superposition_kernel(2))

    counts = _counts(exe, transpiler.executor(), shots=1000)

    assert sum(counts.values()) == 1000
    assert all(len(bits) == 2 for bits in counts)


def test_executor_constructor_seed_matches_factory() -> None:
    """Constructing ``QuriPartsExecutor(seed=...)`` directly also seeds.

    This guards the lower-level constructor path independent of the
    transpiler factory, since both are part of the public surface.
    """
    transpiler = QuriPartsTranspiler()
    exe = transpiler.transpile(_uniform_superposition_kernel(3))

    counts_ctor = _counts(exe, QuriPartsExecutor(seed=99), shots=1500)
    counts_factory = _counts(exe, transpiler.executor(seed=99), shots=1500)

    assert counts_ctor == counts_factory
