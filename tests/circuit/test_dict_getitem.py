"""Tests for Dict subscript lookup (``d[key]``) in qkernel.

Covers the frontend ``Dict.__getitem__`` tracing (symbolic
``DictGetItemOperation`` emission and constant-key eager folding), the
emit-time key resolution against bound dict data, value-type wiring
(``Dict[K, Float]`` / ``Dict[K, UInt]``), the forced unrolling of
``qmc.range`` loops whose body looks up a dict with the loop variable,
and cross-backend execution equivalence for a multi-angle QAOA-style
kernel that indexes one dict with the iteration keys of another.
"""

from __future__ import annotations

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.ir.operation.classical_ops import DictGetItemOperation
from qamomile.circuit.ir.operation.control_flow import ForItemsOperation

# ---------------------------------------------------------------------------
# Kernels under test
# ---------------------------------------------------------------------------


@qmc.qkernel
def ma_qaoa_subscript(
    n: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    quad_gamma: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear_gamma: qmc.Dict[qmc.UInt, qmc.Float],
    beta: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    """Multi-angle QAOA layer indexing per-term angle dicts by loop keys."""
    q = qmc.qubit_array(n, name="q")
    q = qmc.h(q)
    for i, hi in linear.items():
        q[i] = qmc.rz(q[i], angle=hi * linear_gamma[i])
    for (i, j), Jij in quad.items():
        q[i], q[j] = qmc.rzz(q[i], q[j], angle=Jij * quad_gamma[(i, j)])
    for i in qmc.range(n):
        q[i] = qmc.rx(q[i], angle=2.0 * beta)
    return qmc.measure(q)


@qmc.qkernel
def ma_qaoa_reference(
    n: qmc.UInt,
    quad_angle: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear_angle: qmc.Dict[qmc.UInt, qmc.Float],
    beta: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    """Same layer with the per-term angles pre-multiplied host-side."""
    q = qmc.qubit_array(n, name="q")
    q = qmc.h(q)
    for i, ai in linear_angle.items():
        q[i] = qmc.rz(q[i], angle=ai)
    for (i, j), aij in quad_angle.items():
        q[i], q[j] = qmc.rzz(q[i], q[j], angle=aij)
    for i in qmc.range(n):
        q[i] = qmc.rx(q[i], angle=2.0 * beta)
    return qmc.measure(q)


@qmc.qkernel
def ma_qaoa_subscript_expval(
    n: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    quad_gamma: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear_gamma: qmc.Dict[qmc.UInt, qmc.Float],
    beta: qmc.Float,
    hamiltonian: qmc.Observable,
) -> qmc.Float:
    """Expval variant of :func:`ma_qaoa_subscript`."""
    q = qmc.qubit_array(n, name="q")
    q = qmc.h(q)
    for i, hi in linear.items():
        q[i] = qmc.rz(q[i], angle=hi * linear_gamma[i])
    for (i, j), Jij in quad.items():
        q[i], q[j] = qmc.rzz(q[i], q[j], angle=Jij * quad_gamma[(i, j)])
    for i in qmc.range(n):
        q[i] = qmc.rx(q[i], angle=2.0 * beta)
    return qmc.expval(q, hamiltonian)


@qmc.qkernel
def ma_qaoa_reference_expval(
    n: qmc.UInt,
    quad_angle: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear_angle: qmc.Dict[qmc.UInt, qmc.Float],
    beta: qmc.Float,
    hamiltonian: qmc.Observable,
) -> qmc.Float:
    """Expval variant of :func:`ma_qaoa_reference`."""
    q = qmc.qubit_array(n, name="q")
    q = qmc.h(q)
    for i, ai in linear_angle.items():
        q[i] = qmc.rz(q[i], angle=ai)
    for (i, j), aij in quad_angle.items():
        q[i], q[j] = qmc.rzz(q[i], q[j], angle=aij)
    for i in qmc.range(n):
        q[i] = qmc.rx(q[i], angle=2.0 * beta)
    return qmc.expval(q, hamiltonian)


# ---------------------------------------------------------------------------
# Random problem generation
# ---------------------------------------------------------------------------


def _random_problem(rng: np.random.Generator, n: int) -> dict:
    """Draw a random ring-coupled Ising instance with per-term gammas.

    Args:
        rng (np.random.Generator): Seeded generator.
        n (int): Number of qubits (ring of n couplings for n >= 2).

    Returns:
        dict: Bindings for :func:`ma_qaoa_subscript` (without
            ``hamiltonian``).
    """
    pairs = [(i, (i + 1) % n) for i in range(n)] if n >= 3 else [(0, 1)]
    quad = {p: float(rng.uniform(-1, 1)) for p in pairs}
    linear = {i: float(rng.uniform(-1, 1)) for i in range(n)}
    quad_gamma = {p: float(rng.uniform(0, 2 * np.pi)) for p in pairs}
    linear_gamma = {i: float(rng.uniform(0, 2 * np.pi)) for i in range(n)}
    return dict(
        n=n,
        quad=quad,
        linear=linear,
        quad_gamma=quad_gamma,
        linear_gamma=linear_gamma,
        beta=float(rng.uniform(0, np.pi)),
    )


def _reference_bindings(bindings: dict) -> dict:
    """Fold coeff * gamma host-side into the reference kernel's bindings.

    Args:
        bindings (dict): Bindings produced by :func:`_random_problem`.

    Returns:
        dict: Bindings for :func:`ma_qaoa_reference`.
    """
    return dict(
        n=bindings["n"],
        quad_angle={
            k: bindings["quad"][k] * bindings["quad_gamma"][k] for k in bindings["quad"]
        },
        linear_angle={
            k: bindings["linear"][k] * bindings["linear_gamma"][k]
            for k in bindings["linear"]
        },
        beta=bindings["beta"],
    )


def _ising_hamiltonian(bindings: dict) -> qm_o.Hamiltonian:
    """Build the Ising cost Hamiltonian matching the problem instance.

    Args:
        bindings (dict): Bindings produced by :func:`_random_problem`.

    Returns:
        qm_o.Hamiltonian: ``sum_i h_i Z_i + sum_(i,j) J_ij Z_i Z_j``.
    """
    hamiltonian = qm_o.Hamiltonian()
    for i, hi in bindings["linear"].items():
        hamiltonian += hi * qm_o.Z(i)
    for (i, j), jij in bindings["quad"].items():
        hamiltonian += jij * qm_o.Z(i) * qm_o.Z(j)
    return hamiltonian


# ---------------------------------------------------------------------------
# Trace-level behavior
# ---------------------------------------------------------------------------


class TestDictGetItemTrace:
    """Frontend tracing behavior of ``Dict.__getitem__``."""

    def test_symbolic_key_traces_dict_getitem_op(self):
        """A loop-variable key lookup traces a DictGetItemOperation."""
        block = ma_qaoa_subscript.build(**_random_problem(np.random.default_rng(0), 3))
        for_items_ops = [
            op for op in block.operations if isinstance(op, ForItemsOperation)
        ]
        assert len(for_items_ops) == 2
        lookups = [
            op
            for loop in for_items_ops
            for op in loop.operations
            if isinstance(op, DictGetItemOperation)
        ]
        assert len(lookups) == 2
        arities = sorted(op.key_arity for op in lookups)
        assert arities == [1, 2]

    def test_constant_key_folds_eagerly(self):
        """A constant key with bound data folds to a constant at trace time."""

        @qmc.qkernel
        def const_key(
            gammas: qmc.Dict[qmc.UInt, qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, name="q")
            q[0] = qmc.rz(qmc.h(q[0]), angle=gammas[0])
            return qmc.measure(q)

        block = const_key.build(gammas={0: 0.5, 2: 0.6})
        assert not any(isinstance(op, DictGetItemOperation) for op in block.operations)

    def test_constant_string_key_folds_eagerly(self):
        """A constant non-int key works through the eager-fold path."""

        @qmc.qkernel
        def str_key(
            params: qmc.Dict[qmc.UInt, qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, name="q")
            q[0] = qmc.rx(qmc.h(q[0]), angle=params["alpha"])
            return qmc.measure(q)

        block = str_key.build(params={"alpha": 0.7})
        assert not any(isinstance(op, DictGetItemOperation) for op in block.operations)

    def test_missing_constant_key_raises_key_error(self):
        """A constant key absent from the bound data raises KeyError."""

        @qmc.qkernel
        def missing_key(
            gammas: qmc.Dict[qmc.UInt, qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, name="q")
            q[0] = qmc.rz(qmc.h(q[0]), angle=gammas[0])
            return qmc.measure(q)

        with pytest.raises(KeyError, match="Key 0 not found"):
            missing_key.build(gammas={2: 0.6})

    def test_empty_bound_dict_raises_key_error(self):
        """A constant-key lookup in a bound empty dict raises KeyError.

        A bound empty dict must behave like any other bound dict with a
        missing key, not fall through to the symbolic-lookup path.
        """

        @qmc.qkernel
        def empty_dict(
            gammas: qmc.Dict[qmc.UInt, qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, name="q")
            q[0] = qmc.rz(qmc.h(q[0]), angle=gammas[0])
            return qmc.measure(q)

        with pytest.raises(KeyError, match="Key 0 not found"):
            empty_dict.build(gammas={})

    def test_container_value_type_raises_not_implemented(self):
        """Subscripting a container-valued dict reports a clear error."""

        @qmc.qkernel
        def tuple_valued(
            d: qmc.Dict[qmc.UInt, qmc.Tuple[qmc.Float, qmc.Float]],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, name="q")
            q[0] = qmc.rx(qmc.h(q[0]), angle=d[0])
            return qmc.measure(q)

        with pytest.raises(NotImplementedError, match="scalar value types"):
            tuple_valued.build(d={0: (0.1, 0.2)})

    def test_symbolic_non_uint_key_raises_type_error(self):
        """A symbolic non-UInt key component (Float handle) is rejected."""

        @qmc.qkernel
        def float_key(
            gammas: qmc.Dict[qmc.UInt, qmc.Float],
            theta: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, name="q")
            q[0] = qmc.rz(qmc.h(q[0]), angle=gammas[theta])
            return qmc.measure(q)

        with pytest.raises(TypeError, match="must be UInt"):
            float_key.build(gammas={0: 0.5}, theta=0.3)


# ---------------------------------------------------------------------------
# Emit-level behavior (Qiskit structural checks)
# ---------------------------------------------------------------------------


class TestDictGetItemEmit:
    """Emit-time resolution checks on the Qiskit backend."""

    def test_emitted_angles_match_reference(self, qiskit_transpiler):
        """Subscript and pre-multiplied kernels emit identical statevectors."""
        from qiskit.quantum_info import Statevector

        rng = np.random.default_rng(42)
        bindings = _random_problem(rng, 3)
        exe = qiskit_transpiler.transpile(ma_qaoa_subscript, bindings=bindings)
        exe_ref = qiskit_transpiler.transpile(
            ma_qaoa_reference, bindings=_reference_bindings(bindings)
        )

        def statevector(circuit):
            return Statevector.from_instruction(
                circuit.remove_final_measurements(inplace=False)
            ).data

        np.testing.assert_allclose(
            statevector(exe.quantum_circuit),
            statevector(exe_ref.quantum_circuit),
            atol=1e-12,
        )

    def test_uint_valued_dict(self, qiskit_transpiler):
        """A Dict[UInt, UInt] value participates in angle arithmetic."""

        @qmc.qkernel
        def uint_values(
            n: qmc.UInt,
            counts: qmc.Dict[qmc.UInt, qmc.UInt],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            for i, c in counts.items():
                q[i] = qmc.rx(q[i], angle=0.5 * counts[i])
            return qmc.measure(q)

        exe = qiskit_transpiler.transpile(
            uint_values, bindings={"n": 2, "counts": {0: 2, 1: 3}}
        )
        angles = sorted(
            float(inst.operation.params[0])
            for inst in exe.quantum_circuit.data
            if inst.operation.name == "rx"
        )
        assert angles == pytest.approx([1.0, 1.5])

    def test_range_loop_with_dict_lookup_unrolls(self, qiskit_transpiler):
        """A qmc.range loop reading a dict by loop var forces unrolling."""

        @qmc.qkernel
        def range_lookup(
            n: qmc.UInt,
            gammas: qmc.Dict[qmc.UInt, qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            for i in qmc.range(n):
                q[i] = qmc.rx(q[i], angle=gammas[i])
            return qmc.measure(q)

        exe = qiskit_transpiler.transpile(
            range_lookup, bindings={"n": 2, "gammas": {0: 0.3, 1: 0.9}}
        )
        angles = sorted(
            float(inst.operation.params[0])
            for inst in exe.quantum_circuit.data
            if inst.operation.name == "rx"
        )
        assert angles == pytest.approx([0.3, 0.9])

    def test_mixed_constant_and_symbolic_tuple_key(self, qiskit_transpiler):
        """A tuple key mixing an int constant and a loop variable resolves."""

        @qmc.qkernel
        def mixed_key(
            n: qmc.UInt,
            gammas: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            for i in qmc.range(n):
                q[i] = qmc.rx(q[i], angle=gammas[(0, i)])
            return qmc.measure(q)

        exe = qiskit_transpiler.transpile(
            mixed_key, bindings={"n": 2, "gammas": {(0, 0): 0.4, (0, 1): 0.8}}
        )
        angles = sorted(
            float(inst.operation.params[0])
            for inst in exe.quantum_circuit.data
            if inst.operation.name == "rx"
        )
        assert angles == pytest.approx([0.4, 0.8])


# ---------------------------------------------------------------------------
# Cross-backend execution
# ---------------------------------------------------------------------------


class TestDictGetItemCrossBackend:
    """Sampling and expval equivalence on every supported SDK backend."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_expval_matches_reference(self, sdk_transpiler, seed, n):
        """Subscript and pre-multiplied kernels agree on <H> per backend."""
        transpiler = sdk_transpiler.transpiler
        bindings = _random_problem(np.random.default_rng(seed), n)
        hamiltonian = _ising_hamiltonian(bindings)

        exe = transpiler.transpile(
            ma_qaoa_subscript_expval,
            bindings=dict(bindings, hamiltonian=hamiltonian),
        )
        expval = exe.run(transpiler.executor(), bindings={}).result()

        exe_ref = transpiler.transpile(
            ma_qaoa_reference_expval,
            bindings=dict(_reference_bindings(bindings), hamiltonian=hamiltonian),
        )
        expval_ref = exe_ref.run(transpiler.executor(), bindings={}).result()

        assert float(expval) == pytest.approx(float(expval_ref), abs=1e-8)

    @pytest.mark.parametrize("seed", [0, 42])
    def test_sampling_runs(self, sdk_transpiler, seed):
        """The sampling path executes and returns full-length bitstrings."""
        transpiler = sdk_transpiler.transpiler
        n = 3
        bindings = _random_problem(np.random.default_rng(seed), n)
        exe = transpiler.transpile(ma_qaoa_subscript, bindings=bindings)
        result = exe.sample(transpiler.executor(), shots=256).result()

        assert sum(count for _, count in result.results) == 256
        assert all(len(bits) == n for bits, _ in result.results)
