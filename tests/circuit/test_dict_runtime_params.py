"""Tests for Dict kernel arguments kept as runtime parameters.

Covers declaring ``Dict[K, Float]`` in ``transpile(parameters=[...])``:
per-key backend parameter creation from constant-key subscript lookups
(``d[key]``, including tuple keys and ``qmc.range`` loop variables that
unrolling makes constant), execution-time decomposition of
``bindings={"coeffs": {...}}`` onto those parameters, re-binding the
same executable with different dict values, and the rejection paths
(non-Float value types, items() iteration, auto-detection).
"""

from __future__ import annotations

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.ir.operation.classical_ops import DictGetItemOperation

from .conftest import run_statevector

# ---------------------------------------------------------------------------
# Kernels under test
# ---------------------------------------------------------------------------


@qmc.qkernel
def rx_by_dict(
    n: qmc.UInt,
    angles: qmc.Dict[qmc.UInt, qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    """Rotate each qubit by the dict entry looked up with the loop index."""
    q = qmc.qubit_array(n, name="q")
    for i in qmc.range(n):
        q[i] = qmc.rx(q[i], angle=angles[i])
    return qmc.measure(q)


@qmc.qkernel
def ising_chain_layer(
    n: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    """One chain-coupled Ising layer with per-term dict coefficients."""
    q = qmc.qubit_array(n, name="q")
    q = qmc.h(q)
    for i in qmc.range(n):
        q[i] = qmc.rz(q[i], angle=2.0 * linear[i])
    for i in qmc.range(n - 1):
        q[i], q[i + 1] = qmc.rzz(q[i], q[i + 1], angle=2.0 * quad[(i, i + 1)])
    return qmc.measure(q)


@qmc.qkernel
def ising_chain_layer_expval(
    n: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    hamiltonian: qmc.Observable,
) -> qmc.Float:
    """Expval variant of :func:`ising_chain_layer`."""
    q = qmc.qubit_array(n, name="q")
    q = qmc.h(q)
    for i in qmc.range(n):
        q[i] = qmc.rz(q[i], angle=2.0 * linear[i])
    for i in qmc.range(n - 1):
        q[i], q[i + 1] = qmc.rzz(q[i], q[i + 1], angle=2.0 * quad[(i, i + 1)])
    return qmc.expval(q, hamiltonian)


# ---------------------------------------------------------------------------
# Random problem generation
# ---------------------------------------------------------------------------


def _random_chain_problem(rng: np.random.Generator, n: int) -> dict:
    """Draw random chain-coupled Ising coefficients.

    Args:
        rng (np.random.Generator): Seeded generator.
        n (int): Number of qubits.

    Returns:
        dict: ``{"quad": ..., "linear": ...}`` coefficient dicts for
            :func:`ising_chain_layer`.
    """
    quad = {(i, i + 1): float(rng.uniform(-1, 1)) for i in range(n - 1)}
    linear = {i: float(rng.uniform(-1, 1)) for i in range(n)}
    return {"quad": quad, "linear": linear}


def _chain_hamiltonian(coeffs: dict) -> qm_o.Hamiltonian:
    """Build the Ising cost Hamiltonian matching the problem instance.

    Args:
        coeffs (dict): Output of :func:`_random_chain_problem`.

    Returns:
        qm_o.Hamiltonian: ``sum_i h_i Z_i + sum_(i,j) J_ij Z_i Z_j``.
    """
    hamiltonian = qm_o.Hamiltonian()
    for i, hi in coeffs["linear"].items():
        hamiltonian += hi * qm_o.Z(i)
    for (i, j), jij in coeffs["quad"].items():
        hamiltonian += jij * qm_o.Z(i) * qm_o.Z(j)
    return hamiltonian


# ---------------------------------------------------------------------------
# Build / validation behavior
# ---------------------------------------------------------------------------


class TestDictParameterValidation:
    """Frontend acceptance and rejection of Dict runtime parameters."""

    def test_dict_accepted_in_parameters(self):
        """A Dict[K, Float] argument builds with parameters=[...] and traces lookups."""

        @qmc.qkernel
        def const_key(coeffs: qmc.Dict[qmc.UInt, qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, name="q")
            q[0] = qmc.rx(q[0], angle=coeffs[0])
            return qmc.measure(q)

        block = const_key.build(parameters=["coeffs"])
        assert any(isinstance(op, DictGetItemOperation) for op in block.operations)

    def test_dict_uint_value_type_rejected(self):
        """A Dict with a non-Float value type cannot be a runtime parameter."""

        @qmc.qkernel
        def uint_valued(counts: qmc.Dict[qmc.UInt, qmc.UInt]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, name="q")
            q[0] = qmc.rx(q[0], angle=0.5 * counts[0])
            return qmc.measure(q)

        with pytest.raises(TypeError, match=r"Dict\[K, Float\]"):
            uint_valued.build(parameters=["counts"])

    def test_dict_not_auto_detected_as_parameter(self):
        """An unbound Dict without parameters=[...] is not picked up as a parameter.

        Dict runtime parameters are an explicit opt-in; a Dict left out of
        both bindings and parameters keeps the legacy symbolic-dummy-input
        behavior (used for visualization) and never enters
        ``Block.parameters``.
        """

        @qmc.qkernel
        def const_key(coeffs: qmc.Dict[qmc.UInt, qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, name="q")
            q[0] = qmc.rx(q[0], angle=coeffs[0])
            return qmc.measure(q)

        block = const_key.build()
        assert "coeffs" not in block.parameters

    def test_items_iteration_rejected(self):
        """items() over a runtime-parameter dict fails at trace time."""

        @qmc.qkernel
        def items_loop(
            n: qmc.UInt,
            coeffs: qmc.Dict[qmc.UInt, qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            for i, c in coeffs.items():
                q[i] = qmc.rx(q[i], angle=c)
            return qmc.measure(q)

        with pytest.raises(TypeError, match="runtime parameter"):
            items_loop.build(parameters=["coeffs"], n=2)

    def test_string_key_rejected(self):
        """A non-int constant key has no symbolic representation and is rejected."""

        @qmc.qkernel
        def str_key(coeffs: qmc.Dict[qmc.UInt, qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, name="q")
            q[0] = qmc.rx(q[0], angle=coeffs["alpha"])
            return qmc.measure(q)

        with pytest.raises(NotImplementedError, match="non-int constant keys"):
            str_key.build(parameters=["coeffs"])

    def test_overlap_with_bindings_rejected(self, qiskit_transpiler):
        """A Dict name in both bindings and parameters hits the disjointness check."""

        @qmc.qkernel
        def const_key(coeffs: qmc.Dict[qmc.UInt, qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, name="q")
            q[0] = qmc.rx(q[0], angle=coeffs[0])
            return qmc.measure(q)

        with pytest.raises(ValueError, match="both"):
            qiskit_transpiler.transpile(
                const_key,
                bindings={"coeffs": {0: 0.5}},
                parameters=["coeffs"],
            )


# ---------------------------------------------------------------------------
# Emit-level behavior (Qiskit structural checks)
# ---------------------------------------------------------------------------


class TestDictParameterEmit:
    """Backend-parameter creation from Dict lookups on the Qiskit backend."""

    def test_parameter_names_scalar_and_tuple_keys(self, qiskit_transpiler):
        """Each looked-up key becomes one backend parameter with the shared naming."""
        exe = qiskit_transpiler.transpile(
            ising_chain_layer,
            bindings={"n": 3},
            parameters=["quad", "linear"],
        )
        assert sorted(exe.parameter_names) == [
            "linear[0]",
            "linear[1]",
            "linear[2]",
            "quad[(0, 1)]",
            "quad[(1, 2)]",
        ]

    def test_same_key_shares_one_parameter(self, qiskit_transpiler):
        """Repeated lookups of one key reuse a single backend parameter."""

        @qmc.qkernel
        def repeated(coeffs: qmc.Dict[qmc.UInt, qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, name="q")
            q[0] = qmc.rx(q[0], angle=coeffs[0])
            q[0] = qmc.rz(q[0], angle=coeffs[0])
            return qmc.measure(q)

        exe = qiskit_transpiler.transpile(repeated, parameters=["coeffs"])
        assert exe.parameter_names == ["coeffs[0]"]

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_bound_circuit_matches_compile_time_bound(self, qiskit_transpiler, seed):
        """Assigning the emitted parameters reproduces the compile-time-bound circuit."""
        n = 3
        coeffs = _random_chain_problem(np.random.default_rng(seed), n)

        exe_runtime = qiskit_transpiler.transpile(
            ising_chain_layer,
            bindings={"n": n},
            parameters=["quad", "linear"],
        )
        values = {
            f"quad[({i}, {i + 1})]": coeffs["quad"][(i, i + 1)] for i in range(n - 1)
        }
        values.update({f"linear[{i}]": coeffs["linear"][i] for i in range(n)})
        assignment = {
            param: values[param.name]
            for param in exe_runtime.quantum_circuit.parameters
        }
        bound_runtime = exe_runtime.quantum_circuit.assign_parameters(assignment)

        exe_bound = qiskit_transpiler.transpile(
            ising_chain_layer,
            bindings={"n": n, **coeffs},
        )
        np.testing.assert_allclose(
            run_statevector(bound_runtime),
            run_statevector(exe_bound.quantum_circuit),
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Cross-backend execution
# ---------------------------------------------------------------------------


class TestDictParameterExecution:
    """Sampling and expval execution with execution-time dict bindings."""

    def test_sampling_deterministic(self, sdk_transpiler):
        """Per-key angles 0 / pi produce the expected deterministic bitstring."""
        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(rx_by_dict, bindings={"n": 3}, parameters=["angles"])
        result = exe.sample(
            transpiler.executor(),
            shots=256,
            bindings={"angles": {0: np.pi, 1: 0.0, 2: np.pi}},
        ).result()

        assert result.results == [((1, 0, 1), 256)]

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_expval_matches_compile_time_bound(self, sdk_transpiler, seed, n):
        """Execution-time dict bindings reproduce the compile-time-bound <H>."""
        transpiler = sdk_transpiler.transpiler
        coeffs = _random_chain_problem(np.random.default_rng(seed), n)
        hamiltonian = _chain_hamiltonian(coeffs)

        exe_runtime = transpiler.transpile(
            ising_chain_layer_expval,
            bindings={"n": n, "hamiltonian": hamiltonian},
            parameters=["quad", "linear"],
        )
        expval_runtime = exe_runtime.run(
            transpiler.executor(), bindings=coeffs
        ).result()

        exe_bound = transpiler.transpile(
            ising_chain_layer_expval,
            bindings={"n": n, "hamiltonian": hamiltonian, **coeffs},
        )
        expval_bound = exe_bound.run(transpiler.executor(), bindings={}).result()

        assert float(expval_runtime) == pytest.approx(float(expval_bound), abs=1e-8)

    def test_rebind_same_executable(self, sdk_transpiler):
        """One transpiled executable re-binds with different dict values per run."""
        transpiler = sdk_transpiler.transpiler
        n = 3
        hamiltonian = _chain_hamiltonian(
            _random_chain_problem(np.random.default_rng(0), n)
        )
        exe_runtime = transpiler.transpile(
            ising_chain_layer_expval,
            bindings={"n": n, "hamiltonian": hamiltonian},
            parameters=["quad", "linear"],
        )

        for seed in (1, 2):
            coeffs = _random_chain_problem(np.random.default_rng(seed), n)
            expval_runtime = exe_runtime.run(
                transpiler.executor(), bindings=coeffs
            ).result()
            exe_bound = transpiler.transpile(
                ising_chain_layer_expval,
                bindings={"n": n, "hamiltonian": hamiltonian, **coeffs},
            )
            expval_bound = exe_bound.run(transpiler.executor(), bindings={}).result()
            assert float(expval_runtime) == pytest.approx(float(expval_bound), abs=1e-8)

    def test_numpy_integer_keys_normalize(self, qiskit_transpiler):
        """np.int64 keys in the execution-time dict match the emitted names."""
        exe = qiskit_transpiler.transpile(
            rx_by_dict, bindings={"n": 2}, parameters=["angles"]
        )
        result = exe.sample(
            qiskit_transpiler.executor(),
            shots=128,
            bindings={"angles": {np.int64(0): np.pi, np.int64(1): 0.0}},
        ).result()

        assert result.results == [((1, 0), 128)]

    def test_missing_key_at_execution_raises(self, qiskit_transpiler):
        """A missing per-key entry surfaces as a missing-binding error."""
        exe = qiskit_transpiler.transpile(
            rx_by_dict, bindings={"n": 2}, parameters=["angles"]
        )
        with pytest.raises(ValueError, match=r"angles\[1\]"):
            exe.sample(
                qiskit_transpiler.executor(),
                shots=16,
                bindings={"angles": {0: np.pi}},
            ).result()
