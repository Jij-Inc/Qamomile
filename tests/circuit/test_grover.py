"""Tests for the Grover search stdlib kernel and its resource estimate."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

import qamomile.circuit as qmc
from qamomile.circuit.stdlib.grover import grover_iteration_count, grover_search


@qmc.composite_gate(name="mark_all_ones")
def mark_all_ones(reg: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Flip the phase of the all-ones basis state via a multi-controlled Z."""
    top = reg.shape[0] - 1
    reg[top] = qmc.h(reg[top])
    mcx = qmc.control(qmc.x, num_controls=top)
    reg[0:top], reg[top] = mcx(reg[0:top], reg[top])
    reg[top] = qmc.h(reg[top])
    return reg


class _QueryCost:
    """Report one opaque query of cost ``l + n`` per oracle call."""

    def __call__(self, ctx: qmc.OpaqueCallContext) -> qmc.ResourceEstimate:
        """Return an ``O(l + n)`` gate + one-query estimate.

        Args:
            ctx (qmc.OpaqueCallContext): Call-site context; the register width is
                read from the operand shape.

        Returns:
            qmc.ResourceEstimate: One-query gate/call estimate.
        """
        n = sum(ctx.operand_shapes.values()) if ctx.operand_shapes else sp.Symbol("n")
        cost = sp.Symbol("l", positive=True) + n
        return qmc.ResourceEstimate(
            gates=qmc.GateResources(total=cost, non_clifford=cost),
            calls=qmc.CallResources(
                calls_by_name={"query_oracle": sp.Integer(1)},
                queries_by_name={"query_oracle": sp.Integer(1)},
            ),
        )


_query_oracle = qmc.opaque(
    "query_oracle",
    signature=qmc.CallableSignature(
        inputs=[qmc.Vector[qmc.Qubit]],
        outputs=[qmc.Vector[qmc.Qubit]],
    ),
    cost=_QueryCost(),
)


@qmc.qkernel
def _grover_estimate_kernel(n: qmc.UInt, iterations: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Grover kernel with a symbolic query oracle for resource estimation."""
    reg = qmc.qubit_array(n, name="reg")
    reg = grover_search(reg, _query_oracle, iterations)
    return qmc.measure(reg)


def _numpy_grover_zexp(n: int, iterations: int) -> float:
    """Compute ``<Z_0>`` after Grover amplification of ``|1...1>``.

    Args:
        n (int): Number of search qubits.
        iterations (int): Number of Grover iterations.

    Returns:
        float: Analytic ``<Z_0>`` reference for the marked-all-ones circuit.
    """
    dim = 2**n
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)
    marked = dim - 1  # |1...1>
    uniform = np.ones(dim, dtype=complex) / np.sqrt(dim)
    for _ in range(iterations):
        state[marked] *= -1  # phase oracle
        # diffusion: 2|s><s| - I
        state = 2 * uniform * (uniform.conj() @ state) - state
    # <Z_0>: qubit 0 is the least-significant bit.
    probs = np.abs(state) ** 2
    p_bit0_one = sum(probs[x] for x in range(dim) if x & 1)
    return float(1 - 2 * p_bit0_one)


def test_grover_iteration_count_concrete_and_symbolic() -> None:
    """The optimal iteration count is floor((pi/4) sqrt(N/m))."""
    assert grover_iteration_count(4, 1) == 3
    assert grover_iteration_count(10, 1) == 25
    n = sp.Symbol("n", positive=True)
    symbolic = grover_iteration_count(n, 1)
    assert n in symbolic.free_symbols


def test_grover_iteration_count_accepts_numpy_integers() -> None:
    """NumPy integer scalars take the concrete, positivity-validated path."""
    assert grover_iteration_count(np.int64(4), np.int64(1)) == 3
    assert grover_iteration_count(np.int32(10), 1) == 25
    with pytest.raises(ValueError, match="must be positive"):
        grover_iteration_count(np.int64(0), 1)


def test_grover_symbolic_query_complexity() -> None:
    """Query count equals the (symbolic) iteration count: O(sqrt(N/m))."""
    est = _grover_estimate_kernel.estimate_resources()
    iterations = est.parameters["iterations"]
    assert est.calls.queries_by_name["query_oracle"] == iterations


def test_grover_optimal_query_complexity_via_inputs() -> None:
    """Substituting the optimal iteration count yields O(sqrt(N/m)) directly.

    Uses the ``inputs`` estimation UX to plug the optimal iteration
    formula straight into the estimate, so the universal Grover query complexity
    ``floor((pi/4) sqrt(2^n/m))`` comes out of one estimate call.
    """
    n = sp.Symbol("n", positive=True)
    m = sp.Symbol("m", positive=True)
    est = _grover_estimate_kernel.estimate_resources(
        inputs={"iterations": grover_iteration_count(n, m)}
    )
    queries = est.calls.queries_by_name["query_oracle"]
    # It is a floor of the optimal continuous count; compare the floor argument
    # (sympy keeps the simplified 2**(n/2-2) form, not the literal sqrt form).
    expected = grover_iteration_count(n, m)
    assert queries.func is sp.floor
    assert sp.simplify(queries.args[0] - expected.args[0]) == 0
    # And it evaluates to the concrete optimal count at sample sizes.
    for nn, mm in [(4, 1), (10, 1), (8, 4)]:
        assert int(queries.subs({n: nn, m: mm})) == grover_iteration_count(nn, mm)


def test_grover_qubit_count_is_linear() -> None:
    """Grover uses O(n) qubits (the search register width)."""
    est = _grover_estimate_kernel.estimate_resources()
    n = est.parameters["n"]
    assert sp.simplify(est.qubits - n) == 0


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("seed", [0, 3])
def test_grover_cross_backend_amplifies_marked_state(
    sdk_transpiler, n: int, seed: int, tmp_path
) -> None:
    """Grover amplifies the marked all-ones state on every SDK backend."""
    src = (
        "import qamomile.circuit as qmc\n"
        "from tests.circuit.test_grover import mark_all_ones\n"
        "from qamomile.circuit.stdlib.grover import "
        "grover_search, grover_iteration_count\n"
        "@qmc.qkernel\n"
        f"def grover_run() -> qmc.Vector[qmc.Bit]:\n"
        f"    reg = qmc.qubit_array({n}, name='reg')\n"
        f"    reg = grover_search(reg, mark_all_ones, grover_iteration_count({n}, 1))\n"
        "    return qmc.measure(reg)\n"
    )
    path = str(tmp_path / f"grover_run_{n}.py")
    with open(path, "w") as handle:
        handle.write(src)
    import importlib.util

    spec = importlib.util.spec_from_file_location("grover_run_mod", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    transpiler = sdk_transpiler.transpiler
    if sdk_transpiler.backend_name == "qiskit":
        from qiskit.providers.basic_provider import BasicSimulator

        backend = BasicSimulator()
        backend.set_options(seed_simulator=seed)
        executor = transpiler.executor(backend=backend)
    else:
        executor = transpiler.executor()

    exe = transpiler.transpile(module.grover_run)
    result = exe.sample(executor, shots=1024).result()

    marked = tuple(1 for _ in range(n))
    top_bits, top_count = result.most_common(1)[0]
    assert top_bits == marked, (
        f"{sdk_transpiler.backend_name}: marked state {marked} not dominant, "
        f"got {top_bits}"
    )
    # Amplitude amplification should give the marked state a clear majority.
    assert top_count / result.shots > 0.6


@pytest.mark.parametrize("n", [2, 3])
def test_grover_cross_backend_expval(sdk_transpiler, n: int, tmp_path) -> None:
    """Grover's amplified state matches the analytic ``<Z_0>`` on each backend."""
    import qamomile.observable as qm_o

    src = (
        "import qamomile.circuit as qmc\n"
        "from tests.circuit.test_grover import mark_all_ones\n"
        "from qamomile.circuit.stdlib.grover import "
        "grover_search, grover_iteration_count\n"
        "@qmc.qkernel\n"
        f"def grover_expval(obs: qmc.Observable) -> qmc.Float:\n"
        f"    reg = qmc.qubit_array({n}, name='reg')\n"
        f"    reg = grover_search(reg, mark_all_ones, grover_iteration_count({n}, 1))\n"
        "    return qmc.expval(reg, obs)\n"
    )
    path = str(tmp_path / f"grover_expval_{n}.py")
    with open(path, "w") as handle:
        handle.write(src)
    import importlib.util

    spec = importlib.util.spec_from_file_location("grover_expval_mod", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    transpiler = sdk_transpiler.transpiler
    exe = transpiler.transpile(module.grover_expval, bindings={"obs": qm_o.Z(0)})
    value = exe.run(transpiler.executor()).result()

    reference = _numpy_grover_zexp(n, grover_iteration_count(n, 1))
    atol = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert np.isclose(value, reference, atol=atol), (
        f"{sdk_transpiler.backend_name} n={n}: expected <Z_0>={reference}, got {value}"
    )
