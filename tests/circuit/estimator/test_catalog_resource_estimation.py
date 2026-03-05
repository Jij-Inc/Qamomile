"""Resource estimation tests driven by the qkernel catalog.

Expected resource values are defined in EXPECTED_RESOURCES as a mapping
from catalog entry ID to ResourceEstimate. Each catalog entry is tested
against its expected resource estimate for qubits and gate counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import sympy as sp

from qamomile.circuit.estimator import (
    GateCount,
    ResourceEstimate,
    estimate_resources,
)
from tests.circuit.estimator.assertions import (
    assert_expr_equal,
    assert_gate_counts,
)
from tests.circuit.qkernel_catalog import (
    QKERNEL_BY_ID,
    QKERNEL_CATALOG,
    QKernelEntry,
    concrete_values_for,
    parametric_entries,
)

# ============================================================
# Helpers
# ============================================================


def resource(
    qubits: int | float | sp.Expr,
    *,
    total: int | float | sp.Expr = 0,
    single_qubit: int | float | sp.Expr = 0,
    two_qubit: int | float | sp.Expr = 0,
    multi_qubit: int | float | sp.Expr = 0,
    t_gates: int | float | sp.Expr = 0,
    clifford_gates: int | float | sp.Expr = 0,
    rotation_gates: int | float | sp.Expr = 0,
    oracle_calls: dict[str, int | float | sp.Expr] | None = None,
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
    "single_h": resource(1, total=1, single_qubit=1, clifford_gates=1),
    "single_x": resource(1, total=1, single_qubit=1, clifford_gates=1),
    "single_y": resource(1, total=1, single_qubit=1, clifford_gates=1),
    "single_z": resource(1, total=1, single_qubit=1, clifford_gates=1),
    "single_t": resource(1, total=1, single_qubit=1, t_gates=1),
    "single_tdg": resource(1, total=1, single_qubit=1, t_gates=1),
    "single_s": resource(1, total=1, single_qubit=1, clifford_gates=1),
    "single_sdg": resource(1, total=1, single_qubit=1, clifford_gates=1),
    "single_p": resource(1, total=1, single_qubit=1, rotation_gates=1),
    "single_rx": resource(1, total=1, single_qubit=1, rotation_gates=1),
    "single_ry": resource(1, total=1, single_qubit=1, rotation_gates=1),
    "single_rz": resource(1, total=1, single_qubit=1, rotation_gates=1),
    # --- Two-qubit gate entries ---
    "single_cx": resource(2, total=1, two_qubit=1, clifford_gates=1),
    "single_cz": resource(2, total=1, two_qubit=1, clifford_gates=1),
    "single_cp": resource(
        2,
        total=1,
        two_qubit=1,
        rotation_gates=1,
    ),
    "single_swap": resource(2, total=1, two_qubit=1, clifford_gates=1),
    "single_rzz": resource(
        2,
        total=1,
        two_qubit=1,
        rotation_gates=1,
    ),
    # --- Basic circuits ---
    "no_operation": resource(n),
    "only_measurements": resource(n),
    "simple_for_loop": resource(1, total=m, single_qubit=m, clifford_gates=m),
    "all_rx": resource(n, total=n, single_qubit=n, rotation_gates=n),
    "naive_toffoli_decomposition": resource(
        # https://arxiv.org/pdf/1210.0974
        3,
        total=16,
        single_qubit=10,
        two_qubit=6,
        t_gates=7,
        clifford_gates=9,
    ),
    "commutated_toffoli_decomposition": resource(
        # https://arxiv.org/pdf/1210.0974
        3,
        total=16,
        single_qubit=10,
        two_qubit=6,
        t_gates=7,
        clifford_gates=9,
    ),
    "optimal_toffoli_decomposition": resource(
        # https://arxiv.org/pdf/1210.0974
        3,
        total=17,
        single_qubit=10,
        two_qubit=7,
        t_gates=7,
        clifford_gates=10,
    ),
    "optimal_toffoli_decomposition_loop": resource(
        # https://arxiv.org/pdf/1210.0974
        3,
        total=17 * (m + 1),
        single_qubit=10 * (m + 1),
        two_qubit=7 * (m + 1),
        t_gates=7 * (m + 1),
        clifford_gates=10 * (m + 1),
    ),
    # --- Entanglement ---
    "bell_state": resource(
        2,
        total=2,
        single_qubit=1,
        two_qubit=1,
        clifford_gates=2,
    ),
    "linear_entanglement": resource(
        n,
        total=n - 1,
        two_qubit=n - 1,
        clifford_gates=n - 1,
    ),
    "full_entanglement": resource(
        n,
        total=n * (n - 1) / 2,  # type:ignore
        two_qubit=n * (n - 1) / 2,  # type:ignore
        clifford_gates=n * (n - 1) / 2,  # type:ignore
    ),
    "ghz_state": resource(
        # Ref: https://bloqade.quera.com/v0.22.3/digital/examples/ghz/
        n,
        total=n,
        single_qubit=1,
        two_qubit=n - 1,
        clifford_gates=n,
    ),
    "parallel_ghz_state": resource(
        # Ref: https://bloqade.quera.com/v0.22.3/digital/examples/ghz/
        2**m,
        total=2**m,
        single_qubit=1,
        two_qubit=2**m - 1,
        clifford_gates=2**m,
    ),
    # --- QFT / IQFT ---
    "qft": resource(
        # Ref: Nielsen & Chuang (gate counts)
        n,
        total=n * (n + 1) / 2 + sp.floor(n / 2),
        single_qubit=n,
        two_qubit=n * (n - 1) / 2 + sp.floor(n / 2),
        clifford_gates=n + sp.floor(n / 2),
        rotation_gates=(n * (n - 1)) / 2,  # type:ignore
    ),
    "iqft": resource(
        # Ref: Nielsen & Chuang (gate counts)
        n,
        total=n * (n + 1) / 2 + sp.floor(n / 2),
        single_qubit=n,
        two_qubit=n * (n - 1) / 2 + sp.floor(n / 2),
        clifford_gates=n + sp.floor(n / 2),
        rotation_gates=(n * (n - 1)) / 2,  # type:ignore
    ),
    # --- Algorithms — quantum tests / oracle-based ---
    "hadamard_test": resource(
        2,
        total=3,
        single_qubit=2,
        two_qubit=1,
        clifford_gates=2,
        oracle_calls={"controlled_oracle": 1},
    ),
    "swap_test": resource(
        3,
        total=5,
        single_qubit=2,
        two_qubit=2,
        multi_qubit=1,
        clifford_gates=4,
    ),
    "simplest_oracle": resource(
        1,
        oracle_calls={"one_qubit_oracle": 1},
    ),
    "deutsch": resource(
        # Ref: Quantum algorithms for optimizers (https://arxiv.org/abs/2408.07086) (circuit)
        2,
        total=5,
        single_qubit=4,
        two_qubit=1,
        clifford_gates=4,
        oracle_calls={"two_qubit_oracle": 1},
    ),
    "deutsch_jozsa": resource(
        # Ref: Nielsen & Chuang (circuit)
        n + 1,
        total=2 * n + 2,
        single_qubit=2 * n + 2,
        clifford_gates=2 * n + 2,
        oracle_calls={"deutsch_jozsa_oracle": 1},
    ),
    "simon": resource(
        # Ref: Quantum algorithms for optimizers (https://arxiv.org/abs/2408.07086) (circuit)
        2 * n,
        total=2 * n,
        single_qubit=2 * n,
        clifford_gates=2 * n,
        oracle_calls={"simon_oracle": 1},
    ),
    "teleportation": resource(
        # Ref: Nielsen & Chuang (circuit)
        3,
        # X, H, CX, CX, H, [X, Z]
        # We don't count the measurement operations as gates.
        total=7,
        single_qubit=5,  # X, H, H + (X, Z)
        two_qubit=2,
        clifford_gates=7,
    ),
    # --- QPE ---
    "phase_gate_qpe": resource(
        # Ref: Nielsen & Chuang (circuit)
        n + 1,
        total=2 * n + (n * (n + 1)) / 2 + sp.floor(n / 2),
        single_qubit=2 * n,
        two_qubit=n + (n * (n - 1)) / 2 + sp.floor(n / 2),
        clifford_gates=2 * n + sp.floor(n / 2),
        rotation_gates=n * (n - 1) / 2,  # type:ignore
    ),
    "stub_oracle_qpe": resource(
        # Ref: Nielsen & Chuang (circuit)
        n + 1,
        total=n + (2**n - 1) + (n * (n + 1)) / 2 + sp.floor(n / 2),
        single_qubit=2 * n,
        two_qubit=(2**n - 1) + (n * (n - 1)) / 2 + sp.floor(n / 2),
        clifford_gates=2 * n + sp.floor(n / 2),
        rotation_gates=n * (n - 1) / 2,
        oracle_calls={"controlled_u": 2**n - 1},
    ),
    # --- Variational / optimization ---
    "hardware_efficient_ansatz": resource(
        # https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.EfficientSU2 (circuit)
        n,
        total=2 * n * num_layers + (num_layers - 1) * (n - 1),
        single_qubit=2 * n * num_layers,
        two_qubit=(num_layers - 1) * (n - 1),
        clifford_gates=(num_layers - 1) * (n - 1),
        rotation_gates=2 * n * num_layers,
    ),
    "qaoa_state_umbiguous": resource(
        # Ref: Quantum algorithms for optimizers (https://arxiv.org/abs/2408.07086) (circuit)
        n,
        total=n + num_layers * (quad + linear + n),  # type:ignore
        single_qubit=n + num_layers * (linear + n),  # type:ignore
        two_qubit=num_layers * quad,  # type:ignore
        clifford_gates=n,
        rotation_gates=num_layers * (quad + linear + n),  # type:ignore
    ),
    # --- Multi-controlled gates ---
    "network_decomposition_controlled_z": resource(
        # Ref: Nielsen & Chuang P. 184 (circuit and ancillas)
        2 * n,
        total=2 * n - 1,
        single_qubit=2,
        two_qubit=1,
        multi_qubit=2 * n - 4,
        clifford_gates=3,
    ),
    "naive_multi_controlled_z": resource(
        n,
        total=1,
        two_qubit=sp.Piecewise((sp.Integer(1), sp.Eq(n, 2)), (sp.Integer(0), True)),
        multi_qubit=sp.Piecewise((sp.Integer(1), n > 2), (sp.Integer(0), True)),
    ),
    # --- Grover ---
    "grover_network_decomposition": resource(
        n + 1 + n_iters * (n - 2),
        total=(n + 2) + n_iters * (6 * n - 1),
        single_qubit=(n + 2) + n_iters * (4 * n + 2),
        two_qubit=n_iters,
        multi_qubit=n_iters * (2 * n - 4),
        clifford_gates=(n + 2) + n_iters * (4 * n + 3),
        oracle_calls={"grover_oracle": n_iters},
    ),
    "grover_naive_multi_controlled_z": resource(
        n + 1,
        total=(n + 2) + n_iters * (4 * n + 1),
        single_qubit=(n + 2) + n_iters * 4 * n,
        two_qubit=(
            n_iters * sp.Piecewise((sp.Integer(1), sp.Eq(n, 2)), (sp.Integer(0), True))
        ),  # type:ignore
        multi_qubit=(
            n_iters * sp.Piecewise((sp.Integer(1), n > 2), (sp.Integer(0), True))
        ),  # type:ignore
        clifford_gates=(n + 2) + n_iters * 4 * n,
        oracle_calls={"grover_oracle": n_iters},
    ),
    "quantum_counting": resource(
        # Ref: Nielsen & Chuang (circuit)
        n + m + (m - 1) + 1,
        total=n + m + 1 + n + n * (n + 1) / 2 + sp.floor(n / 2),  # type:ignore
        single_qubit=n + m + 1 + n,  # type:ignore
        two_qubit=n * (n - 1) / 2 + sp.floor(n / 2),
        multi_qubit=n,
        clifford_gates=n + m + 1 + n + sp.floor(n / 2),  # type:ignore
        rotation_gates=(n * (n - 1)) / 2,  # type:ignore
    ),
    # --- Arithmetic ---
    "maj": resource(
        # A new quantum ripple-carry addition circuit (https://arxiv.org/abs/quant-ph/0410184) (circuit)
        3,
        total=3,
        two_qubit=2,
        multi_qubit=1,
        clifford_gates=2,
    ),
    "maj_loop": resource(
        # A new quantum ripple-carry addition circuit (https://arxiv.org/abs/quant-ph/0410184) (circuit)
        2 * n + 1,
        total=3 * n,
        two_qubit=2 * n,
        multi_qubit=n,
        clifford_gates=2 * n,
    ),
    "uma_2_cnot": resource(
        # A new quantum ripple-carry addition circuit (https://arxiv.org/abs/quant-ph/0410184) (circuit)
        3,
        total=3,
        two_qubit=2,
        multi_qubit=1,
        clifford_gates=2,
    ),
    "uma_2_cnot_loop": resource(
        # A new quantum ripple-carry addition circuit (https://arxiv.org/abs/quant-ph/0410184) (circuit)
        2 * n + 1,
        total=3 * n,
        two_qubit=2 * n,
        multi_qubit=n,
        clifford_gates=2 * n,
    ),
    "uma_3_cnot": resource(
        # A new quantum ripple-carry addition circuit (https://arxiv.org/abs/quant-ph/0410184) (circuit)
        3,
        total=6,
        single_qubit=2,
        two_qubit=3,
        multi_qubit=1,
        clifford_gates=5,
    ),
    "uma_3_cnot_loop": resource(
        # A new quantum ripple-carry addition circuit (https://arxiv.org/abs/quant-ph/0410184) (circuit)
        2 * n + 1,
        total=6 * n,
        single_qubit=2 * n,
        two_qubit=3 * n,
        multi_qubit=n,
        clifford_gates=5 * n,
    ),
    "simple_ripple_carry_adder_2_cnot": resource(
        # A new quantum ripple-carry addition circuit (https://arxiv.org/abs/quant-ph/0410184) (circuit)
        2 * n + 2,
        total=6 * n + 1,
        two_qubit=4 * n + 1,
        multi_qubit=2 * n,
        clifford_gates=4 * n + 1,
    ),
    "simple_ripple_carry_adder_3_cnot": resource(
        # A new quantum ripple-carry addition circuit (https://arxiv.org/abs/quant-ph/0410184) (circuit)
        2 * n + 2,
        total=9 * n + 1,
        single_qubit=2 * n,
        two_qubit=5 * n + 1,
        multi_qubit=2 * n,
        clifford_gates=7 * n + 1,
    ),
    "draper_inplace_qc_adder": resource(
        # Addition on a Quantum Computer (https://arxiv.org/abs/quant-ph/0008033) (circuit)
        n,
        total=n + n * (n + 1) + 2 * sp.floor(n / 2),
        single_qubit=3 * n,
        two_qubit=n * (n - 1) + 2 * sp.floor(n / 2),
        clifford_gates=2 * n + 2 * sp.floor(n / 2),
        rotation_gates=n + n * (n - 1),
    ),
    "ttk_adder": resource(
        # Quantum Addition Circuits and Unbounded Fan-Out (https://arxiv.org/abs/0910.2530) (circuit, total, two and toffoli (multi) counts, and total depth)
        2 * n + 1,
        total=7 * n - 6,
        two_qubit=5 * n - 5,
        multi_qubit=2 * n - 1,
        clifford_gates=5 * n - 5,
    ),
    "cdkm_adder": resource(
        # A new quantum ripple-carry addition circuit (https://arxiv.org/abs/quant-ph/0410184) (circuit, total, single, two and toffoli (multi), clifford counts, and total and toffoli (multi) depth)
        2 * n + 2,
        total=9 * n - 8,
        single_qubit=2 * n - 4,
        two_qubit=5 * n - 3,
        multi_qubit=2 * n - 1,
        clifford_gates=7 * n - 7,
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
        est = entry.qkernel.estimate_resources()
        expected = _get_expected(entry)
        assert_expr_equal(est.qubits, expected.qubits, "qubits")

    def test_gate_counts(self, entry):
        est = entry.qkernel.estimate_resources()
        expected = _get_expected(entry)
        assert_gate_counts(est.gates, expected.gates)


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
    est = entry.qkernel.estimate_resources().substitute(**subs)
    expected = _get_expected(entry).substitute(**subs)
    assert_expr_equal(est.qubits, expected.qubits, "qubits")
    assert_gate_counts(est.gates, expected.gates)


# ============================================================
# Concrete dict bindings tests
# ============================================================


@dataclass(frozen=True)
class BindingsCase:
    """A test case for concrete dict bindings with expected resources."""

    id: str
    entry_id: str
    bindings: dict[str, Any]
    expected: ResourceEstimate


EXPECTED_BINDINGS_RESOURCES: list[BindingsCase] = [
    # --- qaoa_state_umbiguous: non-overlapping qubits (parallel RZZ) ---
    BindingsCase(
        id="qaoa_non_overlapping",
        entry_id="qaoa_state_umbiguous",
        bindings={
            "n": 4,
            "num_layers": 1,
            "quad": {(0, 1): 0.5, (2, 3): 0.3},
            "linear": {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4},
        },
        expected=resource(
            4,
            total=4 + 2 + 4 + 4,  # H(4) + RZZ(2) + RZ(4) + RX(4)
            single_qubit=4 + 4 + 4,  # H(4) + RZ(4) + RX(4)
            two_qubit=2,  # RZZ(2)
            clifford_gates=4,  # H(4)
            rotation_gates=2 + 4 + 4,  # RZZ(2) + RZ(4) + RX(4)
        ),
    ),
    # --- qaoa_state_umbiguous: overlapping qubits (sequential RZZ) ---
    BindingsCase(
        id="qaoa_overlapping",
        entry_id="qaoa_state_umbiguous",
        bindings={
            "n": 3,
            "num_layers": 1,
            "quad": {(0, 1): 0.5, (1, 2): 0.3},
            "linear": {0: 0.1, 1: 0.2, 2: 0.3},
        },
        expected=resource(
            3,
            total=3 + 2 + 3 + 3,  # H(3) + RZZ(2) + RZ(3) + RX(3)
            single_qubit=3 + 3 + 3,  # H(3) + RZ(3) + RX(3)
            two_qubit=2,
            clifford_gates=3,
            rotation_gates=2 + 3 + 3,
        ),
    ),
    # --- qaoa_state_umbiguous: multi-layer ---
    BindingsCase(
        id="qaoa_multi_layer",
        entry_id="qaoa_state_umbiguous",
        bindings={
            "n": 4,
            "num_layers": 2,
            "quad": {(0, 1): 0.5, (2, 3): 0.3},
            "linear": {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4},
        },
        expected=resource(
            4,
            total=4 + 2 * (2 + 4 + 4),  # H(4) + 2*(RZZ(2)+RZ(4)+RX(4))
            single_qubit=4 + 2 * (4 + 4),
            two_qubit=2 * 2,
            clifford_gates=4,
            rotation_gates=2 * (2 + 4 + 4),
        ),
    ),
    # --- qaoa_state_umbiguous: empty dicts ---
    BindingsCase(
        id="qaoa_empty_dicts",
        entry_id="qaoa_state_umbiguous",
        bindings={
            "n": 2,
            "num_layers": 1,
            "quad": {},
            "linear": {},
        },
        expected=resource(
            2,
            total=2 + 0 + 0 + 2,  # H(2) + RX(2) only
            single_qubit=2 + 0 + 2,
            two_qubit=0,
            clifford_gates=2,
            rotation_gates=0 + 0 + 2,
        ),
    ),
    # --- qaoa_state_umbiguous: quad-only (linear empty) ---
    BindingsCase(
        id="qaoa_quad_only",
        entry_id="qaoa_state_umbiguous",
        bindings={
            "n": 3,
            "num_layers": 1,
            "quad": {(0, 1): 0.5, (1, 2): 0.3},
            "linear": {},
        },
        expected=resource(
            3,
            total=3 + 2 + 0 + 3,  # H(3) + RZZ(2) + RZ(0) + RX(3)
            single_qubit=3 + 0 + 3,
            two_qubit=2,
            clifford_gates=3,
            rotation_gates=2 + 0 + 3,
        ),
    ),
    # --- qaoa_state_umbiguous: linear-only (quad empty) ---
    BindingsCase(
        id="qaoa_linear_only",
        entry_id="qaoa_state_umbiguous",
        bindings={
            "n": 4,
            "num_layers": 1,
            "quad": {},
            "linear": {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4},
        },
        expected=resource(
            4,
            total=4 + 0 + 4 + 4,  # H(4) + RZZ(0) + RZ(4) + RX(4)
            single_qubit=4 + 4 + 4,
            two_qubit=0,
            clifford_gates=4,
            rotation_gates=0 + 4 + 4,
        ),
    ),
    # --- qaoa_state_umbiguous: single-element dicts ---
    BindingsCase(
        id="qaoa_single_element_dict",
        entry_id="qaoa_state_umbiguous",
        bindings={
            "n": 2,
            "num_layers": 1,
            "quad": {(0, 1): 0.5},
            "linear": {0: 0.1},
        },
        expected=resource(
            2,
            total=2 + 1 + 1 + 2,  # H(2) + RZZ(1) + RZ(1) + RX(2)
            single_qubit=2 + 1 + 2,
            two_qubit=1,
            clifford_gates=2,
            rotation_gates=1 + 1 + 2,
        ),
    ),
]


def _bindings_cases() -> list:
    """Generate pytest params for concrete bindings test cases."""
    return [pytest.param(case, id=case.id) for case in EXPECTED_BINDINGS_RESOURCES]


@pytest.mark.parametrize("case", _bindings_cases())
def test_concrete_bindings(case: BindingsCase):
    """Test resource estimation with concrete dict bindings via QKernel method."""
    entry = QKERNEL_BY_ID[case.entry_id]
    est = entry.qkernel.estimate_resources(bindings=case.bindings)
    assert_expr_equal(est.qubits, case.expected.qubits, "qubits")
    assert_gate_counts(est.gates, case.expected.gates)


def test_no_bindings_backward_compat():
    """Without bindings, behavior should be unchanged (symbolic)."""
    entry = QKERNEL_BY_ID["qaoa_state_umbiguous"]
    est = estimate_resources(entry.qkernel.block)
    quad_sym = sp.Symbol("|quad|", integer=True, positive=True)
    assert quad_sym in est.gates.total.free_symbols


def test_partial_bindings_dicts_only():
    """Partial bindings (dicts only, no scalars) should not crash."""
    entry = QKERNEL_BY_ID["qaoa_state_umbiguous"]
    est = estimate_resources(
        entry.qkernel.block,
        bindings={"quad": {(0, 1): 0.5}, "linear": {0: 0.1}},
    )
    # Scalar parameters should remain symbolic
    num_layers_sym = sp.Symbol("num_layers", integer=True, positive=True)
    assert num_layers_sym in est.gates.total.free_symbols
    # Dict cardinalities should be substituted (not symbolic)
    quad_sym = sp.Symbol("|quad|", integer=True, positive=True)
    linear_sym = sp.Symbol("|linear|", integer=True, positive=True)
    assert quad_sym not in est.gates.total.free_symbols
    assert linear_sym not in est.gates.total.free_symbols


# ============================================================
# Equivalence: QKernel method vs function API
# ============================================================


@pytest.mark.parametrize(
    "entry",
    QKERNEL_CATALOG[:3],
    ids=[e.id for e in QKERNEL_CATALOG[:3]],
)
def test_method_function_equivalence(entry):
    """QKernel.estimate_resources() must equal estimate_resources(block)."""
    method_est = entry.qkernel.estimate_resources()
    func_est = estimate_resources(entry.qkernel.block)
    assert_expr_equal(method_est.qubits, func_est.qubits, "qubits")
    assert_gate_counts(method_est.gates, func_est.gates)


def test_method_function_equivalence_with_bindings():
    """QKernel.estimate_resources(bindings=...) must equal function API with same bindings."""
    entry = QKERNEL_BY_ID["qaoa_state_umbiguous"]
    bindings = {"quad": {(0, 1): 0.5}, "linear": {0: 0.1}}
    method_est = entry.qkernel.estimate_resources(bindings=bindings)
    func_est = estimate_resources(entry.qkernel.block, bindings=bindings)
    assert_expr_equal(method_est.qubits, func_est.qubits, "qubits")
    assert_gate_counts(method_est.gates, func_est.gates)
