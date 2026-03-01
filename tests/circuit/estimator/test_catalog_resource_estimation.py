"""Resource estimation tests driven by the qkernel catalog.

Expected resource values are defined in EXPECTED_RESOURCES as a mapping
from catalog entry ID to ResourceEstimate. Each catalog entry is tested
against its expected resource estimate for qubits, gate counts, and depth.
"""

import pytest
import sympy as sp

from qamomile.circuit.estimator import (
    CircuitDepth,
    GateCount,
    ResourceEstimate,
    estimate_resources,
)
from tests.circuit.qkernel_catalog import (
    QKERNEL_CATALOG,
    QKernelEntry,
    concrete_values_for,
    parametric_entries,
)

# ============================================================
# Helpers
# ============================================================

_0 = sp.Integer(0)
_1 = sp.Integer(1)
_2 = sp.Integer(2)


def _single_qubit_resource(clifford: int, rotation: int) -> ResourceEstimate:
    """Create a ResourceEstimate for a single-qubit gate circuit."""
    return ResourceEstimate(
        qubits=_1,
        gates=GateCount(
            total=_1,
            single_qubit=_1,
            two_qubit=_0,
            multi_qubit=_0,
            t_gates=_0,
            clifford_gates=sp.Integer(clifford),
            rotation_gates=sp.Integer(rotation),
        ),
        depth=CircuitDepth(
            total_depth=_1,
            t_depth=_0,
            two_qubit_depth=_0,
            multi_qubit_depth=_0,
            rotation_depth=sp.Integer(rotation),
        ),
    )


def _two_qubit_resource(clifford: int, rotation: int) -> ResourceEstimate:
    """Create a ResourceEstimate for a two-qubit gate circuit."""
    return ResourceEstimate(
        qubits=_2,
        gates=GateCount(
            total=_1,
            single_qubit=_0,
            two_qubit=_1,
            multi_qubit=_0,
            t_gates=_0,
            clifford_gates=sp.Integer(clifford),
            rotation_gates=sp.Integer(rotation),
        ),
        depth=CircuitDepth(
            total_depth=_1,
            t_depth=_0,
            two_qubit_depth=_1,
            multi_qubit_depth=_0,
            rotation_depth=sp.Integer(rotation),
        ),
    )


def assert_expr_equal(actual: sp.Expr, expected: sp.Expr, msg: str = "") -> None:
    """Assert two SymPy expressions are symbolically equal."""
    diff = sp.simplify(actual - expected)
    assert diff == 0, (
        f"SymPy expressions not equal{' (' + msg + ')' if msg else ''}: "
        f"actual={actual}, expected={expected}, diff={diff}"
    )


def _assert_gate_counts(actual: GateCount, expected: GateCount) -> None:
    """Assert all gate count fields match."""
    assert_expr_equal(actual.total, expected.total, "total")
    assert_expr_equal(actual.single_qubit, expected.single_qubit, "single_qubit")
    assert_expr_equal(actual.two_qubit, expected.two_qubit, "two_qubit")
    assert_expr_equal(actual.multi_qubit, expected.multi_qubit, "multi_qubit")
    assert_expr_equal(actual.t_gates, expected.t_gates, "t_gates")
    assert_expr_equal(actual.clifford_gates, expected.clifford_gates, "clifford_gates")
    assert_expr_equal(actual.rotation_gates, expected.rotation_gates, "rotation_gates")
    for name, expected_count in expected.oracle_calls.items():
        assert name in actual.oracle_calls, f"Missing oracle call: {name}"
        assert_expr_equal(
            actual.oracle_calls[name], expected_count, f"oracle_calls[{name}]"
        )
    assert set(actual.oracle_calls.keys()) == set(expected.oracle_calls.keys()), (
        f"oracle_calls keys mismatch: "
        f"actual={set(actual.oracle_calls.keys())}, "
        f"expected={set(expected.oracle_calls.keys())}"
    )


def _assert_depth(actual: CircuitDepth, expected: CircuitDepth) -> None:
    """Assert all depth fields match."""
    assert_expr_equal(actual.total_depth, expected.total_depth, "total_depth")
    assert_expr_equal(actual.t_depth, expected.t_depth, "t_depth")
    assert_expr_equal(
        actual.two_qubit_depth, expected.two_qubit_depth, "two_qubit_depth"
    )
    assert_expr_equal(
        actual.multi_qubit_depth, expected.multi_qubit_depth, "multi_qubit_depth"
    )
    assert_expr_equal(actual.rotation_depth, expected.rotation_depth, "rotation_depth")


# ============================================================
# Expected resources
# ============================================================

n = sp.Symbol("n", integer=True, positive=True)
m = sp.Symbol("m", integer=True, positive=True)
n_iters = sp.Symbol("n_iters", integer=True, positive=True)

EXPECTED_RESOURCES: dict[str, ResourceEstimate] = {
    "no_operation": ResourceEstimate(
        qubits=n,
        gates=GateCount(
            total=0,
            single_qubit=0,
            two_qubit=sp.Integer(0),
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=0,
            rotation_gates=0,
        ),
        depth=CircuitDepth(
            total_depth=0,
            t_depth=sp.Integer(0),
            two_qubit_depth=sp.Integer(0),
            multi_qubit_depth=sp.Integer(0),
            rotation_depth=0,
        ),
    ),
    "only_measurements": ResourceEstimate(
        qubits=n,
        gates=GateCount(
            total=0,
            single_qubit=0,
            two_qubit=sp.Integer(0),
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=0,
            rotation_gates=0,
        ),
        depth=CircuitDepth(
            total_depth=1,
            t_depth=sp.Integer(0),
            two_qubit_depth=sp.Integer(0),
            multi_qubit_depth=sp.Integer(0),
            rotation_depth=0,
        ),
    ),
    "full_entanglement": ResourceEstimate(
        qubits=n,
        gates=GateCount(
            total=n * (n - 1) / 2,
            single_qubit=0,
            two_qubit=n * (n - 1) / 2,
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=n * (n - 1) / 2,
            rotation_gates=0,
        ),
        depth=CircuitDepth(
            total_depth=2 * n - 3,
            t_depth=sp.Integer(0),
            two_qubit_depth=2 * n - 3,
            multi_qubit_depth=sp.Integer(0),
            rotation_depth=0,
        ),
    ),
    "linear_entanglement": ResourceEstimate(
        qubits=n,
        gates=GateCount(
            total=n - 1,
            single_qubit=0,
            two_qubit=n - 1,
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=n - 1,
            rotation_gates=0,
        ),
        depth=CircuitDepth(
            total_depth=n - 1,
            t_depth=sp.Integer(0),
            two_qubit_depth=n - 1,
            multi_qubit_depth=sp.Integer(0),
            rotation_depth=0,
        ),
    ),
    "simple_for_loop": ResourceEstimate(
        qubits=sp.Integer(1),
        gates=GateCount(
            total=m,
            single_qubit=m,
            two_qubit=sp.Integer(0),
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=m,
            rotation_gates=sp.Integer(0),
        ),
        depth=CircuitDepth(
            total_depth=m,
            t_depth=sp.Integer(0),
            two_qubit_depth=sp.Integer(0),
            multi_qubit_depth=sp.Integer(0),
            rotation_depth=sp.Integer(0),
        ),
    ),
    "all_rx": ResourceEstimate(
        qubits=n,
        gates=GateCount(
            total=n,
            single_qubit=n,
            two_qubit=sp.Integer(0),
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=0,
            rotation_gates=n,
        ),
        depth=CircuitDepth(
            total_depth=1,
            t_depth=sp.Integer(0),
            two_qubit_depth=sp.Integer(0),
            multi_qubit_depth=sp.Integer(0),
            rotation_depth=1,
        ),
    ),
    "bell_state": ResourceEstimate(
        qubits=sp.Integer(2),
        gates=GateCount(
            total=sp.Integer(2),
            single_qubit=sp.Integer(1),
            two_qubit=sp.Integer(1),
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=sp.Integer(2),
            rotation_gates=sp.Integer(0),
        ),
        depth=CircuitDepth(
            total_depth=sp.Integer(2),
            t_depth=sp.Integer(0),
            two_qubit_depth=sp.Integer(1),
            multi_qubit_depth=sp.Integer(0),
            rotation_depth=sp.Integer(0),
        ),
    ),
    "ghz_state": ResourceEstimate(
        # Ref: https://bloqade.quera.com/v0.22.3/digital/examples/ghz/?utm_source=chatgpt.com
        qubits=n,
        gates=GateCount(
            total=n,
            single_qubit=sp.Integer(1),
            two_qubit=n - 1,
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=n,
            rotation_gates=sp.Integer(0),
        ),
        depth=CircuitDepth(
            total_depth=n,
            t_depth=sp.Integer(0),
            two_qubit_depth=n - 1,
            multi_qubit_depth=sp.Integer(0),
            rotation_depth=sp.Integer(0),
        ),
    ),
    "parallel_ghz_state": ResourceEstimate(
        # Ref: https://bloqade.quera.com/v0.22.3/digital/examples/ghz/?utm_source=chatgpt.com
        qubits=2**m,
        gates=GateCount(
            total=2**m,
            single_qubit=sp.Integer(1),
            two_qubit=2**m - 1,
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=2**m,
            rotation_gates=sp.Integer(0),
        ),
        depth=CircuitDepth(
            total_depth=m + 1,
            t_depth=sp.Integer(0),
            two_qubit_depth=m,
            multi_qubit_depth=sp.Integer(0),
            rotation_depth=sp.Integer(0),
        ),
    ),
    "qft": ResourceEstimate(
        # Ref: Nielsen & Chuang
        qubits=n,
        gates=GateCount(
            total=n * (n + 1) / 2 + sp.floor(n / 2),
            single_qubit=n,
            two_qubit=n * (n - 1) / 2 + sp.floor(n / 2),
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=n + sp.floor(n / 2),
            rotation_gates=(n * (n - 1)) / 2,
        ),
        depth=CircuitDepth(
            total_depth=2 * n,
            t_depth=sp.Integer(0),
            two_qubit_depth=2 * n - 2,
            multi_qubit_depth=sp.Integer(0),
            rotation_depth=2 * n - 3,
        ),
    ),
    "iqft": ResourceEstimate(
        # Ref: Nielsen & Chuang
        qubits=n,
        gates=GateCount(
            total=n * (n + 1) / 2 + sp.floor(n / 2),
            single_qubit=n,
            two_qubit=n * (n - 1) / 2 + sp.floor(n / 2),
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=n + sp.floor(n / 2),
            rotation_gates=(n * (n - 1)) / 2,
        ),
        depth=CircuitDepth(
            total_depth=2 * n,
            t_depth=sp.Integer(0),
            two_qubit_depth=2 * n - 2,
            multi_qubit_depth=sp.Integer(0),
            rotation_depth=2 * n - 3,
        ),
    ),
    "phase_gate_qpe": ResourceEstimate(
        qubits=n + 1,
        gates=GateCount(
            total=2**n + n**2 / 2 + 3 * n / 2 + sp.floor(n / 2) - 1,
            single_qubit=2 * n,
            two_qubit=2**n + n**2 / 2 - n / 2 + sp.floor(n / 2) - 1,
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=2 * n + sp.floor(n / 2),
            rotation_gates=2**n + n * (n - 1) / 2 - 1,
        ),
        depth=CircuitDepth(
            total_depth=2**n + 2 * n,
            t_depth=sp.Integer(0),
            two_qubit_depth=2**n + 2 * n - 3,
            multi_qubit_depth=sp.Integer(0),
            rotation_depth=2**n + 2 * n - 4,
        ),
    ),
    "draper_inplace_qc_adder": ResourceEstimate(
        qubits=n,
        gates=GateCount(
            total=n**2 + 2 * n + 2 * sp.floor(n / 2),
            single_qubit=3 * n,
            two_qubit=n**2 - n + 2 * sp.floor(n / 2),
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=2 * n + 2 * sp.floor(n / 2),
            rotation_gates=n**2,
        ),
        depth=CircuitDepth(
            total_depth=4 * n + 1,
            t_depth=sp.Integer(0),
            two_qubit_depth=4 * n - 4,
            multi_qubit_depth=sp.Integer(0),
            rotation_depth=4 * n - 5,
        ),
    ),
    "simplest_oracle": ResourceEstimate(
        qubits=sp.Integer(1),
        gates=GateCount(
            total=sp.Integer(0),
            single_qubit=sp.Integer(0),
            two_qubit=sp.Integer(0),
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=sp.Integer(0),
            rotation_gates=sp.Integer(0),
            oracle_calls={"one_qubit_oracle": sp.Integer(1)},
        ),
        depth=CircuitDepth(
            total_depth=sp.Integer(1),
            t_depth=sp.Integer(0),
            two_qubit_depth=sp.Integer(0),
            multi_qubit_depth=sp.Integer(0),
            rotation_depth=sp.Integer(0),
        ),
    ),
    "stub_oracle_qpe": ResourceEstimate(
        qubits=n + 1,
        gates=GateCount(
            total=n**2 / 2 + 3 * n / 2 + sp.floor(n / 2),
            single_qubit=2 * n,
            two_qubit=n**2 / 2 - n / 2 + sp.floor(n / 2),
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=2 * n + sp.floor(n / 2),
            rotation_gates=n * (n - 1) / 2,
            oracle_calls={"controlled_u": 2**n - 1},  # type: ignore
        ),
        depth=CircuitDepth(
            total_depth=2**n + 2 * n,
            t_depth=sp.Integer(0),
            two_qubit_depth=2**n + 2 * n - 3,
            multi_qubit_depth=sp.Integer(0),
            rotation_depth=2 * n - 3,
        ),
    ),
    "deutsch": ResourceEstimate(
        qubits=2,
        gates=GateCount(
            total=sp.Integer(4),
            single_qubit=sp.Integer(4),
            two_qubit=sp.Integer(0),
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=sp.Integer(4),
            rotation_gates=sp.Integer(0),
            oracle_calls={"two_qubit_oracle": 1},  # type: ignore
        ),
        depth=CircuitDepth(
            total_depth=sp.Integer(5),
            t_depth=sp.Integer(0),
            two_qubit_depth=sp.Integer(0),
            multi_qubit_depth=sp.Integer(0),
            rotation_depth=sp.Integer(0),
        ),
    ),
    "simon": ResourceEstimate(
        qubits=2 * n,
        gates=GateCount(
            total=2 * n,
            single_qubit=2 * n,
            two_qubit=sp.Integer(0),
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=2 * n,
            rotation_gates=sp.Integer(0),
            oracle_calls={"simon_oracle": sp.Integer(1)},
        ),
        depth=CircuitDepth(
            total_depth=sp.Integer(4),
            t_depth=sp.Integer(0),
            two_qubit_depth=sp.Integer(0),
            multi_qubit_depth=sp.Integer(0),
            rotation_depth=sp.Integer(0),
        ),
    ),
    "teleportation": ResourceEstimate(
        qubits=3,
        gates=GateCount(
            # X, H, CX, CX, H, [X, Z]
            # We don't count the measurement operations as gates.
            total=sp.Integer(7),
            single_qubit=sp.Integer(5),  # X, H, H + (X, Z)
            two_qubit=sp.Integer(2),
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=sp.Integer(7),
            rotation_gates=sp.Integer(0),
            oracle_calls={},
        ),
        depth=CircuitDepth(
            # We need to wait for the measurement results before applying the final corrections, X and Z.
            # Here, we assume measurement can happen at the same step as the other gate operations.
            # Algorithmic Resource Timeline (Depth: 6)
            # =========================================================================
            # | Layer | q0 (Send)    | q1 (Ancilla) | q2 (Recv)    | Description          |
            # |-------|--------------|--------------|--------------|----------------------|
            # | L1    | X (prepare)  | H            | -            | Init & Bell Prep     |
            # | L2    | -            | CX (control) | CX (target)  | Entangle q1-q2       |
            # | L3    | CX (control) | CX (target)  | -            | Interaction q0-q1    |
            # | L4    | H            | Measure (M1) | -            | Parallel H & M1      |
            # | L5    | Measure (M0) | -            | X-Correction | Parallel M0 & Corr X |
            # | L6    | -            | -            | Z-Correction | Final Z-Correction   |
            # =========================================================================
            total_depth=sp.Integer(6),
            t_depth=sp.Integer(0),
            two_qubit_depth=sp.Integer(2),
            multi_qubit_depth=sp.Integer(0),
            rotation_depth=sp.Integer(0),
        ),
    ),
    "vchain_controlled_z": ResourceEstimate(
        qubits=2 * n - 2,
        gates=GateCount(
            total=2 * n - 1,
            single_qubit=sp.Integer(2),
            two_qubit=sp.Integer(1),
            multi_qubit=2 * n - 4,
            t_gates=sp.Integer(0),
            clifford_gates=sp.Integer(3),
            rotation_gates=sp.Integer(0),
            oracle_calls={},
        ),
        depth=CircuitDepth(
            total_depth=2 * n - 3,
            t_depth=sp.Integer(0),
            two_qubit_depth=sp.Integer(1),
            multi_qubit_depth=2 * n - 4,
            rotation_depth=sp.Integer(0),
        ),
    ),
    # "naive_multi_controlled_z": ResourceEstimate(
    #     qubits=n,
    #     gates=GateCount(
    #         total=sp.Integer(1),
    #         single_qubit=sp.Integer(0),
    #         two_qubit=sp.Piecewise((sp.Integer(1), sp.Eq(n, 2)), (sp.Integer(0), True)),
    #         multi_qubit=sp.Piecewise((sp.Integer(1), n > 2), (sp.Integer(0), True)),
    #         t_gates=sp.Integer(0),
    #         clifford_gates=sp.Integer(0),
    #         rotation_gates=sp.Integer(0),
    #         oracle_calls={},
    #     ),
    #     depth=CircuitDepth(
    #         total_depth=sp.Integer(1),
    #         t_depth=sp.Integer(0),
    #         two_qubit_depth=sp.Piecewise(
    #             (sp.Integer(1), sp.Eq(n, 2)), (sp.Integer(0), True)
    #         ),
    #         multi_qubit_depth=sp.Piecewise(
    #             (sp.Integer(1), n > 2), (sp.Integer(0), True)
    #         ),
    #         rotation_depth=sp.Integer(0),
    #     ),
    # ),
    # "grover_vchain": ResourceEstimate(
    #     qubits=n + n_iters * (n - 2) + 1,
    #     gates=GateCount(
    #         total=n + n_iters * (4 * n + 1) + 2,
    #         single_qubit=4 * n * n_iters + n + 2,
    #         two_qubit=sp.Integer(0),
    #         multi_qubit=sp.Integer(0),
    #         t_gates=sp.Integer(0),
    #         clifford_gates=4 * n * n_iters + n + 2,
    #         rotation_gates=n_iters,
    #         oracle_calls={"grover_oracle": sp.Integer(1)},
    #     ),
    #     depth=CircuitDepth(
    #         total_depth=n_iters + 3,
    #         t_depth=sp.Integer(0),
    #         two_qubit_depth=sp.Integer(0),
    #         multi_qubit_depth=n_iters,
    #         rotation_depth=sp.Integer(0),
    #     ),
    # ),
    "maj": ResourceEstimate(
        qubits=3,
        gates=GateCount(
            total=3,
            single_qubit=0,
            two_qubit=2,
            multi_qubit=1,
            t_gates=0,
            clifford_gates=2,
            rotation_gates=0,
            oracle_calls={},
        ),
        depth=CircuitDepth(
            total_depth=3,
            t_depth=0,
            two_qubit_depth=2,
            multi_qubit_depth=1,
            rotation_depth=0,
        ),
    ),
    "uma_2_cnot": ResourceEstimate(
        qubits=3,
        gates=GateCount(
            total=3,
            single_qubit=0,
            two_qubit=2,
            multi_qubit=1,
            t_gates=0,
            clifford_gates=2,
            rotation_gates=0,
            oracle_calls={},
        ),
        depth=CircuitDepth(
            total_depth=3,
            t_depth=0,
            two_qubit_depth=2,
            multi_qubit_depth=1,
            rotation_depth=0,
        ),
    ),
    "uma_3_cnot": ResourceEstimate(
        qubits=3,
        gates=GateCount(
            total=6,
            single_qubit=2,
            two_qubit=3,
            multi_qubit=1,
            t_gates=0,
            clifford_gates=5,
            rotation_gates=0,
            oracle_calls={},
        ),
        depth=CircuitDepth(
            total_depth=5,
            t_depth=0,
            two_qubit_depth=3,
            multi_qubit_depth=1,
            rotation_depth=0,
        ),
    ),
    "simple_ripple_carry_adder_2_cnot": ResourceEstimate(
        qubits=2 * n + 2,
        gates=GateCount(
            total=6 * n + 1,
            single_qubit=0,
            two_qubit=4 * n + 1,
            multi_qubit=2 * n,
            t_gates=0,
            clifford_gates=4 * n + 1,
            rotation_gates=0,
            oracle_calls={},
        ),
        depth=CircuitDepth(
            total_depth=5 * n + 2,
            t_depth=0,
            two_qubit_depth=3 * n + 2,
            multi_qubit_depth=2 * n,
            rotation_depth=0,
        ),
    ),
    "simple_ripple_carry_adder_3_cnot": ResourceEstimate(
        qubits=2 * n + 2,
        gates=GateCount(
            total=9 * n + 1,
            single_qubit=2 * n,
            two_qubit=5 * n + 1,
            multi_qubit=2 * n,
            t_gates=0,
            clifford_gates=7 * n + 1,
            rotation_gates=0,
            oracle_calls={},
        ),
        depth=CircuitDepth(
            total_depth=4 * n + 4,
            t_depth=0,
            two_qubit_depth=2 * n + 3,
            multi_qubit_depth=2 * n,
            rotation_depth=0,
        ),
    ),
    # --- Single-qubit gate entries ---
    "single_h": _single_qubit_resource(clifford=1, rotation=0),
    "single_x": _single_qubit_resource(clifford=1, rotation=0),
    "single_p": _single_qubit_resource(clifford=0, rotation=1),
    "single_rx": _single_qubit_resource(clifford=0, rotation=1),
    "single_ry": _single_qubit_resource(clifford=0, rotation=1),
    "single_rz": _single_qubit_resource(clifford=0, rotation=1),
    # --- Two-qubit gate entries ---
    "single_cx": _two_qubit_resource(clifford=1, rotation=0),
    "single_cz": _two_qubit_resource(clifford=1, rotation=0),
    "single_cp": _two_qubit_resource(clifford=0, rotation=1),
    "single_swap": _two_qubit_resource(clifford=1, rotation=0),
    "single_rzz": _two_qubit_resource(clifford=0, rotation=1),
}


# ============================================================
# Tests
# ============================================================


def _get_expected(entry: QKernelEntry) -> ResourceEstimate:
    """Look up the expected ResourceEstimate for a catalog entry."""
    assert entry.id in EXPECTED_RESOURCES, f"No expected resource for '{entry.id}'"
    return EXPECTED_RESOURCES[entry.id]


@pytest.mark.parametrize(
    "entry",
    QKERNEL_CATALOG,
    ids=[e.id for e in QKERNEL_CATALOG],
)
class TestCatalogResourceEstimation:
    """Test resource estimation against expected values for each catalog entry."""

    def test_qubits(self, entry):
        est = estimate_resources(entry.qkernel.block)
        expected = _get_expected(entry)
        assert_expr_equal(est.qubits, expected.qubits, "qubits")

    def test_gate_counts(self, entry):
        est = estimate_resources(entry.qkernel.block)
        expected = _get_expected(entry)
        _assert_gate_counts(est.gates, expected.gates)

    def test_depth(self, entry):
        est = estimate_resources(entry.qkernel.block)
        expected = _get_expected(entry)
        _assert_depth(est.depth, expected.depth)


def _parametric_cases() -> list[tuple[str, dict[str, int]]]:
    """Generate (test_id, subs) pairs for all parametric entries."""
    cases = []
    for entry in parametric_entries():
        for subs in concrete_values_for(entry):
            vals = "_".join(f"{k}={v}" for k, v in subs.items())
            cases.append(pytest.param(entry, subs, id=f"{entry.id}[{vals}]"))
    return cases


@pytest.mark.parametrize("entry,subs", _parametric_cases())
def test_parametric_substitution(entry, subs):
    est = estimate_resources(entry.qkernel.block).substitute(**subs)
    expected = _get_expected(entry).substitute(**subs)
    assert_expr_equal(est.qubits, expected.qubits, "qubits")
    _assert_gate_counts(est.gates, expected.gates)
    _assert_depth(est.depth, expected.depth)
