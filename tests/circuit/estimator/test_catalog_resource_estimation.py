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
from tests.circuit.estimator.assertions import (
    assert_depth,
    assert_expr_equal,
    assert_gate_counts,
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


def resource(
    qubits,
    *,
    total=0,
    single_qubit=0,
    two_qubit=0,
    multi_qubit=0,
    t_gates=0,
    clifford_gates=0,
    rotation_gates=0,
    oracle_calls=None,
    total_depth=0,
    t_depth=0,
    two_qubit_depth=0,
    multi_qubit_depth=0,
    rotation_depth=0,
) -> ResourceEstimate:
    """Build a ResourceEstimate with zero defaults for unspecified fields."""
    return ResourceEstimate(
        qubits=qubits,
        gates=GateCount(
            total=total,
            single_qubit=single_qubit,
            two_qubit=two_qubit,
            multi_qubit=multi_qubit,
            t_gates=t_gates,
            clifford_gates=clifford_gates,
            rotation_gates=rotation_gates,
            oracle_calls=oracle_calls or {},
        ),
        depth=CircuitDepth(
            total_depth=total_depth,
            t_depth=t_depth,
            two_qubit_depth=two_qubit_depth,
            multi_qubit_depth=multi_qubit_depth,
            rotation_depth=rotation_depth,
        ),
    )


# ============================================================
# Expected resources
# ============================================================

n = sp.Symbol("n", integer=True, positive=True)
m = sp.Symbol("m", integer=True, positive=True)
n_iters = sp.Symbol("n_iters", integer=True, positive=True)
num_layers = sp.Symbol("num_layers", integer=True, positive=True)
quad = sp.Symbol("|quad|", integer=True, positive=True)
linear = sp.Symbol("|linear|", integer=True, positive=True)

EXPECTED_RESOURCES: dict[str, ResourceEstimate] = {
    # --- Single-qubit gate entries ---
    "single_h": resource(1, total=1, single_qubit=1, clifford_gates=1, total_depth=1),
    "single_x": resource(1, total=1, single_qubit=1, clifford_gates=1, total_depth=1),
    "single_p": resource(
        1, total=1, single_qubit=1, rotation_gates=1, total_depth=1, rotation_depth=1
    ),
    "single_rx": resource(
        1, total=1, single_qubit=1, rotation_gates=1, total_depth=1, rotation_depth=1
    ),
    "single_ry": resource(
        1, total=1, single_qubit=1, rotation_gates=1, total_depth=1, rotation_depth=1
    ),
    "single_rz": resource(
        1, total=1, single_qubit=1, rotation_gates=1, total_depth=1, rotation_depth=1
    ),
    # --- Two-qubit gate entries ---
    "single_cx": resource(
        2, total=1, two_qubit=1, clifford_gates=1, total_depth=1, two_qubit_depth=1
    ),
    "single_cz": resource(
        2, total=1, two_qubit=1, clifford_gates=1, total_depth=1, two_qubit_depth=1
    ),
    "single_cp": resource(
        2,
        total=1,
        two_qubit=1,
        rotation_gates=1,
        total_depth=1,
        two_qubit_depth=1,
        rotation_depth=1,
    ),
    "single_swap": resource(
        2, total=1, two_qubit=1, clifford_gates=1, total_depth=1, two_qubit_depth=1
    ),
    "single_rzz": resource(
        2,
        total=1,
        two_qubit=1,
        rotation_gates=1,
        total_depth=1,
        two_qubit_depth=1,
        rotation_depth=1,
    ),
    # --- Basic circuits ---
    "no_operation": resource(n),
    "only_measurements": resource(n, total_depth=1),
    "simple_for_loop": resource(
        1, total=m, single_qubit=m, clifford_gates=m, total_depth=m
    ),
    "all_rx": resource(
        n, total=n, single_qubit=n, rotation_gates=n, total_depth=1, rotation_depth=1
    ),
    # --- Entanglement ---
    "bell_state": resource(
        2,
        total=2,
        single_qubit=1,
        two_qubit=1,
        clifford_gates=2,
        total_depth=2,
        two_qubit_depth=1,
    ),
    "linear_entanglement": resource(
        n,
        total=n - 1,
        two_qubit=n - 1,
        clifford_gates=n - 1,
        total_depth=n - 1,
        two_qubit_depth=n - 1,
    ),
    "full_entanglement": resource(
        n,
        total=n * (n - 1) / 2,
        two_qubit=n * (n - 1) / 2,
        clifford_gates=n * (n - 1) / 2,
        total_depth=2 * n - 3,
        two_qubit_depth=2 * n - 3,
    ),
    "ghz_state": resource(
        # Ref: https://bloqade.quera.com/v0.22.3/digital/examples/ghz/
        n,
        total=n,
        single_qubit=1,
        two_qubit=n - 1,
        clifford_gates=n,
        total_depth=n,
        two_qubit_depth=n - 1,
    ),
    "parallel_ghz_state": resource(
        # Ref: https://bloqade.quera.com/v0.22.3/digital/examples/ghz/
        2**m,
        total=2**m,
        single_qubit=1,
        two_qubit=2**m - 1,
        clifford_gates=2**m,
        total_depth=m + 1,
        two_qubit_depth=m,
    ),
    # --- QFT / IQFT ---
    "qft": resource(
        # Ref: Nielsen & Chuang
        n,
        total=n * (n + 1) / 2 + sp.floor(n / 2),
        single_qubit=n,
        two_qubit=n * (n - 1) / 2 + sp.floor(n / 2),
        clifford_gates=n + sp.floor(n / 2),
        rotation_gates=(n * (n - 1)) / 2,
        total_depth=2 * n,
        two_qubit_depth=2 * n - 2,
        rotation_depth=2 * n - 3,
    ),
    "iqft": resource(
        # Ref: Nielsen & Chuang
        n,
        total=n * (n + 1) / 2 + sp.floor(n / 2),
        single_qubit=n,
        two_qubit=n * (n - 1) / 2 + sp.floor(n / 2),
        clifford_gates=n + sp.floor(n / 2),
        rotation_gates=(n * (n - 1)) / 2,
        total_depth=2 * n,
        two_qubit_depth=2 * n - 2,
        rotation_depth=2 * n - 3,
    ),
    # --- Algorithms — quantum tests / oracle-based ---
    "hadamard_test": resource(
        2,
        total=3,
        single_qubit=2,
        two_qubit=1,
        clifford_gates=2,
        oracle_calls={"controlled_oracle": 1},
        total_depth=4,
        two_qubit_depth=1,
    ),
    "swap_test": resource(
        3,
        total=5,
        single_qubit=2,
        two_qubit=2,
        multi_qubit=1,
        clifford_gates=4,
        total_depth=4,
        two_qubit_depth=2,
        multi_qubit_depth=1,
    ),
    "simplest_oracle": resource(
        1,
        oracle_calls={"one_qubit_oracle": 1},
        total_depth=1,
    ),
    "deutsch": resource(
        2,
        total=4,
        single_qubit=4,
        clifford_gates=4,
        oracle_calls={"two_qubit_oracle": 1},
        total_depth=5,
    ),
    "deutsch_jozsa": resource(
        n + 1,
        total=2 * n + 2,
        single_qubit=2 * n + 2,
        clifford_gates=2 * n + 2,
        oracle_calls={"deutsch_jozsa_oracle": 1},
        total_depth=5,
    ),
    "simon": resource(
        2 * n,
        total=2 * n,
        single_qubit=2 * n,
        clifford_gates=2 * n,
        oracle_calls={"simon_oracle": 1},
        total_depth=4,
    ),
    "teleportation": resource(
        3,
        # X, H, CX, CX, H, [X, Z]
        # We don't count the measurement operations as gates.
        total=7,
        single_qubit=5,  # X, H, H + (X, Z)
        two_qubit=2,
        clifford_gates=7,
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
        total_depth=6,
        two_qubit_depth=2,
    ),
    # --- QPE ---
    "phase_gate_qpe": resource(
        n + 1,
        total=n**2 / 2 + 5 * n / 2 + sp.floor(n / 2),
        single_qubit=2 * n,
        two_qubit=n**2 / 2 + n / 2 + sp.floor(n / 2),
        clifford_gates=2 * n + sp.floor(n / 2),
        rotation_gates=n * (n - 1) / 2,
        total_depth=3 * n + 1,
        two_qubit_depth=3 * n - 2,
        rotation_depth=2 * n - 3,
    ),
    "stub_oracle_qpe": resource(
        n + 1,
        total=n**2 / 2 + 3 * n / 2 + sp.floor(n / 2),
        single_qubit=2 * n,
        two_qubit=n**2 / 2 - n / 2 + sp.floor(n / 2),
        clifford_gates=2 * n + sp.floor(n / 2),
        rotation_gates=n * (n - 1) / 2,
        oracle_calls={"controlled_u": 2**n - 1},
        total_depth=2**n + 2 * n,
        two_qubit_depth=2**n + 2 * n - 3,
        rotation_depth=2 * n - 3,
    ),
    # --- Variational / optimization ---
    "hardware_efficient_ansatz": resource(
        n,
        total=2 * n * num_layers + (num_layers - 1) * (n - 1),
        single_qubit=2 * n * num_layers,
        two_qubit=(num_layers - 1) * (n - 1),
        clifford_gates=(num_layers - 1) * (n - 1),
        rotation_gates=2 * n * num_layers,
        total_depth=2 * num_layers + (num_layers - 1) * (n - 1),
        two_qubit_depth=(num_layers - 1) * (n - 1),
        rotation_depth=2 * num_layers,
    ),
    "qaoa_state": resource(
        n,
        total=n + num_layers * (quad + linear + n),
        single_qubit=n + num_layers * (linear + n),
        two_qubit=num_layers * quad,
        clifford_gates=n,
        rotation_gates=num_layers * (quad + linear + n),
        total_depth=1 + num_layers * (quad + linear + 1),
        two_qubit_depth=num_layers * quad,
        rotation_depth=num_layers * (quad + linear + 1),
    ),
    # --- Multi-controlled gates ---
    "vchain_controlled_z": resource(
        2 * n - 2,
        total=2 * n - 1,
        single_qubit=2,
        two_qubit=1,
        multi_qubit=2 * n - 4,
        clifford_gates=3,
        total_depth=2 * n - 3,
        two_qubit_depth=1,
        multi_qubit_depth=2 * n - 4,
    ),
    "naive_multi_controlled_z": resource(
        n,
        total=1,
        two_qubit=sp.Piecewise((sp.Integer(1), sp.Eq(n, 2)), (sp.Integer(0), True)),
        multi_qubit=sp.Piecewise((sp.Integer(1), n > 2), (sp.Integer(0), True)),
        total_depth=1,
        two_qubit_depth=sp.Piecewise(
            (sp.Integer(1), sp.Eq(n, 2)), (sp.Integer(0), True)
        ),
        multi_qubit_depth=sp.Piecewise(
            (sp.Integer(1), n > 2), (sp.Integer(0), True)
        ),
    ),
    # --- Grover ---
    "grover_vchain": resource(
        n + n_iters * (n - 2) + 1,
        total=(n + 2) + n_iters * (6 * n - 1),
        single_qubit=(n + 2) + n_iters * (4 * n + 2),
        two_qubit=n_iters,
        multi_qubit=n_iters * (2 * n - 4),
        clifford_gates=(n + 2) + n_iters * (4 * n + 3),
        oracle_calls={"grover_oracle": n_iters},
        total_depth=3 + n_iters * (2 * n + 2),
        two_qubit_depth=n_iters,
        multi_qubit_depth=n_iters * (2 * n - 4),
    ),
    "grover_naive_multi_controlled_z": resource(
        n + 1,
        total=(n + 2) + n_iters * (4 * n + 1),
        single_qubit=(n + 2) + n_iters * 4 * n,
        two_qubit=(
            n_iters
            * sp.Piecewise((sp.Integer(1), sp.Eq(n, 2)), (sp.Integer(0), True))
        ),
        multi_qubit=(
            n_iters * sp.Piecewise((sp.Integer(1), n > 2), (sp.Integer(0), True))
        ),
        clifford_gates=(n + 2) + n_iters * 4 * n,
        oracle_calls={"grover_oracle": n_iters},
        total_depth=3 + n_iters * 6,
        two_qubit_depth=(
            n_iters
            * sp.Piecewise((sp.Integer(1), sp.Eq(n, 2)), (sp.Integer(0), True))
        ),
        multi_qubit_depth=(
            n_iters * sp.Piecewise((sp.Integer(1), n > 2), (sp.Integer(0), True))
        ),
    ),
    # --- Arithmetic ---
    "maj": resource(
        3, total=3, two_qubit=2, multi_qubit=1, clifford_gates=2,
        total_depth=3, two_qubit_depth=2, multi_qubit_depth=1,
    ),
    "uma_2_cnot": resource(
        3, total=3, two_qubit=2, multi_qubit=1, clifford_gates=2,
        total_depth=3, two_qubit_depth=2, multi_qubit_depth=1,
    ),
    "uma_3_cnot": resource(
        3, total=6, single_qubit=2, two_qubit=3, multi_qubit=1, clifford_gates=5,
        total_depth=5, two_qubit_depth=3, multi_qubit_depth=1,
    ),
    "simple_ripple_carry_adder_2_cnot": resource(
        2 * n + 2,
        total=6 * n + 1,
        two_qubit=4 * n + 1,
        multi_qubit=2 * n,
        clifford_gates=4 * n + 1,
        total_depth=5 * n + 2,
        two_qubit_depth=3 * n + 2,
        multi_qubit_depth=2 * n,
    ),
    "simple_ripple_carry_adder_3_cnot": resource(
        2 * n + 2,
        total=9 * n + 1,
        single_qubit=2 * n,
        two_qubit=5 * n + 1,
        multi_qubit=2 * n,
        clifford_gates=7 * n + 1,
        total_depth=4 * n + 4,
        two_qubit_depth=2 * n + 3,
        multi_qubit_depth=2 * n,
    ),
    "draper_inplace_qc_adder": resource(
        n,
        total=n**2 + 2 * n + 2 * sp.floor(n / 2),
        single_qubit=3 * n,
        two_qubit=n**2 - n + 2 * sp.floor(n / 2),
        clifford_gates=2 * n + 2 * sp.floor(n / 2),
        rotation_gates=n**2,
        total_depth=4 * n + 1,
        two_qubit_depth=4 * n - 4,
        rotation_depth=4 * n - 5,
    ),
    "ttk_adder": resource(
        2 * n + 1,
        total=7 * n - 6,
        two_qubit=5 * n - 5,
        multi_qubit=2 * n - 1,
        clifford_gates=5 * n - 5,
        total_depth=5 * n - 3,
        two_qubit_depth=3 * n - 2,
        multi_qubit_depth=2 * n - 1,
    ),
    "cdkm_adder": resource(
        2 * n + 2,
        total=9 * n - 8,
        single_qubit=2 * n - 4,
        two_qubit=5 * n - 3,
        multi_qubit=2 * n - 1,
        clifford_gates=7 * n - 7,
        total_depth=2 * n + 4,
        two_qubit_depth=2 * n + 2,
        multi_qubit_depth=2 * n - 1,
    ),
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
        assert_gate_counts(est.gates, expected.gates)

    def test_depth(self, entry):
        est = estimate_resources(entry.qkernel.block)
        expected = _get_expected(entry)
        assert_depth(est.depth, expected.depth)


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
    assert_gate_counts(est.gates, expected.gates)
    assert_depth(est.depth, expected.depth)
