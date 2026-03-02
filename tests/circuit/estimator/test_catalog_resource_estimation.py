"""Resource estimation tests driven by the qkernel catalog.

Expected resource values are defined in EXPECTED_RESOURCES as a mapping
from catalog entry ID to ResourceEstimate. Each catalog entry is tested
against its expected resource estimate for qubits, gate counts, and depth.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

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
        total=n * (n - 1) / 2,  # type:ignore
        two_qubit=n * (n - 1) / 2,  # type:ignore
        clifford_gates=n * (n - 1) / 2,  # type:ignore
        # Circuit depth calculation:
        # - First CX layer: (n-1) gates (cannot be parallelised due to shared qubits)
        # - Remaining layers: (n-2) depth (all CXs except the last can be parallelised in each layer)
        # - Total depth: (n-1) + (n-2) = 2n-3
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
        # Ref: Nielsen & Chuang (gate counts)
        n,
        total=n * (n + 1) / 2 + sp.floor(n / 2),
        single_qubit=n,
        two_qubit=n * (n - 1) / 2 + sp.floor(n / 2),
        clifford_gates=n + sp.floor(n / 2),
        rotation_gates=(n * (n - 1)) / 2,  # type:ignore
        total_depth=2 * n,
        two_qubit_depth=2 * n - 2,
        rotation_depth=2 * n - 3,
    ),
    "iqft": resource(
        # Ref: Nielsen & Chuang (gate counts)
        n,
        total=n * (n + 1) / 2 + sp.floor(n / 2),
        single_qubit=n,
        two_qubit=n * (n - 1) / 2 + sp.floor(n / 2),
        clifford_gates=n + sp.floor(n / 2),
        rotation_gates=(n * (n - 1)) / 2,  # type:ignore
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
        # Ref: Quantum algorithms for optimizers (https://arxiv.org/abs/2408.07086) (circuit)
        2,
        total=5,
        single_qubit=4,
        two_qubit=1,
        clifford_gates=4,
        oracle_calls={"two_qubit_oracle": 1},
        total_depth=5,
        two_qubit_depth=1,
    ),
    "deutsch_jozsa": resource(
        # Ref: Nielsen & Chuang (circuit)
        n + 1,
        total=2 * n + 2,
        single_qubit=2 * n + 2,
        clifford_gates=2 * n + 2,
        oracle_calls={"deutsch_jozsa_oracle": 1},
        total_depth=5,
    ),
    "simon": resource(
        # Ref: Quantum algorithms for optimizers (https://arxiv.org/abs/2408.07086) (circuit)
        2 * n,
        total=2 * n,
        single_qubit=2 * n,
        clifford_gates=2 * n,
        oracle_calls={"simon_oracle": 1},
        total_depth=4,
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
        # Ref: Nielsen & Chuang (circuit)
        n + 1,
        total=2 * n + (n * (n + 1)) / 2 + sp.floor(n / 2),
        single_qubit=2 * n,
        two_qubit=n + (n * (n - 1)) / 2 + sp.floor(n / 2),
        clifford_gates=2 * n + sp.floor(n / 2),
        rotation_gates=n * (n - 1) / 2,  # type:ignore
        total_depth=3 * n + 1,
        two_qubit_depth=3 * n - 2,
        rotation_depth=2 * n - 3,
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
        total_depth=2**n + 2 * n,
        two_qubit_depth=(2 * n - 2) + (2**n - 1),
        rotation_depth=2 * n - 3,
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
        total_depth=2 * num_layers + (num_layers - 1) * (n - 1),
        two_qubit_depth=(num_layers - 1) * (n - 1),
        rotation_depth=2 * num_layers,
    ),
    "qaoa_state_umbiguous": resource(
        # Ref: Quantum algorithms for optimizers (https://arxiv.org/abs/2408.07086) (circuit)
        n,
        total=n + num_layers * (quad + linear + n),  # type:ignore
        single_qubit=n + num_layers * (linear + n),  # type:ignore
        two_qubit=num_layers * quad,  # type:ignore
        clifford_gates=n,
        rotation_gates=num_layers * (quad + linear + n),  # type:ignore
        total_depth=1 + num_layers * (quad + linear + 1),  # type:ignore
        two_qubit_depth=num_layers * quad,  # type:ignore
        rotation_depth=num_layers * (quad + linear + 1),  # type:ignore
    ),
    # --- Multi-controlled gates ---
    "network_decomposition_controlled_z": resource(
        # Ref: Nielsen & Chuang P. 184 (circuit and ancillas)
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
        multi_qubit_depth=sp.Piecewise((sp.Integer(1), n > 2), (sp.Integer(0), True)),
    ),
    # --- Grover ---
    "grover_network_decomposition": resource(
        2 * n - 1,
        total=(n + 2) + n_iters * (6 * n - 1),
        single_qubit=(n + 2) + n_iters * (4 * n + 2),
        two_qubit=n_iters,
        multi_qubit=n_iters * (2 * n - 4),
        clifford_gates=(n + 2) + n_iters * (4 * n + 3),
        oracle_calls={"grover_oracle": n_iters},
        total_depth=3 + n_iters * (2 * n + 1),
        two_qubit_depth=n_iters,
        multi_qubit_depth=n_iters * (2 * n - 4),
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
        total_depth=2 + n_iters * 5,
        two_qubit_depth=(
            n_iters * sp.Piecewise((sp.Integer(1), sp.Eq(n, 2)), (sp.Integer(0), True))
        ),  # type:ignore
        multi_qubit_depth=(
            n_iters * sp.Piecewise((sp.Integer(1), n > 2), (sp.Integer(0), True))
        ),  # type:ignore
    ),
    # --- Arithmetic ---
    "maj": resource(
        # A new quantum ripple-carry addition circuit (https://arxiv.org/abs/quant-ph/0410184) (circuit)
        3,
        total=3,
        two_qubit=2,
        multi_qubit=1,
        clifford_gates=2,
        total_depth=3,
        two_qubit_depth=2,
        multi_qubit_depth=1,
    ),
    "maj_loop": resource(
        # A new quantum ripple-carry addition circuit (https://arxiv.org/abs/quant-ph/0410184) (circuit)
        2 * n + 1,
        total=3 * n,
        two_qubit=2 * n,
        multi_qubit=n,
        clifford_gates=2 * n,
        total_depth=2 * n + 1,
        two_qubit_depth=n + 1,
        multi_qubit_depth=n,
    ),
    "uma_2_cnot": resource(
        # A new quantum ripple-carry addition circuit (https://arxiv.org/abs/quant-ph/0410184) (circuit)
        2 * n + 1,
        total=3 * n,
        two_qubit=2 * n,
        multi_qubit=n,
        clifford_gates=2 * n,
        total_depth=3 * n,
        two_qubit_depth=2 * n,
        multi_qubit_depth=n,
    ),
    "uma_2_cnot_loop": resource(
        # A new quantum ripple-carry addition circuit (https://arxiv.org/abs/quant-ph/0410184) (circuit)
        3,
        total=3,
        two_qubit=2,
        multi_qubit=1,
        clifford_gates=2,
        total_depth=3,
        two_qubit_depth=2,
        multi_qubit_depth=1,
    ),
    "uma_3_cnot_loop": resource(
        # A new quantum ripple-carry addition circuit (https://arxiv.org/abs/quant-ph/0410184) (circuit)
        2 * n + 1,
        total=6 * n,
        single_qubit=2 * n,
        two_qubit=3 * n,
        multi_qubit=n,
        clifford_gates=2,
        total_depth=2 * n + 3,
        two_qubit_depth=2 * n + 3,
        multi_qubit_depth=n,
    ),
    "uma_3_cnot": resource(
        # A new quantum ripple-carry addition circuit (https://arxiv.org/abs/quant-ph/0410184) (circuit)
        3,
        total=6,
        single_qubit=2,
        two_qubit=3,
        multi_qubit=1,
        clifford_gates=5,
        total_depth=5,
        two_qubit_depth=3,
        multi_qubit_depth=1,
    ),
    "simple_ripple_carry_adder_2_cnot": resource(
        # A new quantum ripple-carry addition circuit (https://arxiv.org/abs/quant-ph/0410184) (circuit)
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
        # A new quantum ripple-carry addition circuit (https://arxiv.org/abs/quant-ph/0410184) (circuit)
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
        # A new quantum ripple-carry addition circuit (circuit)
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
            total_depth=4,  # H:1 + RZZ(parallel):1 + RZ:1 + RX:1
            two_qubit_depth=1,
            rotation_depth=3,  # RZZ:1 + RZ:1 + RX:1
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
            total_depth=5,  # H:1 + RZZ(sequential):2 + RZ:1 + RX:1
            two_qubit_depth=2,
            rotation_depth=4,  # RZZ:2 + RZ:1 + RX:1
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
            total_depth=7,  # H:1 + 2*(RZZ:1+RZ:1+RX:1)
            two_qubit_depth=2,
            rotation_depth=6,  # 2*(RZZ:1+RZ:1+RX:1)
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
            total_depth=2,  # H:1 + RX:1
            two_qubit_depth=0,
            rotation_depth=1,  # RX:1
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
            total_depth=4,  # H:1 + RZZ(sequential, share q1):2 + RX:1
            two_qubit_depth=2,
            rotation_depth=3,  # RZZ:2 + RX:1
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
            total_depth=4,  # H:1 + RZZ:1 + RZ:1 + RX:1
            two_qubit_depth=1,
            rotation_depth=3,  # RZZ:1 + RZ:1 + RX:1
        ),
    ),
]


def _bindings_cases() -> list:
    """Generate pytest params for concrete bindings test cases."""
    return [pytest.param(case, id=case.id) for case in EXPECTED_BINDINGS_RESOURCES]


@pytest.mark.parametrize("case", _bindings_cases())
def test_concrete_bindings(case: BindingsCase):
    """Test resource estimation with concrete dict bindings."""
    entry = QKERNEL_BY_ID[case.entry_id]
    est = estimate_resources(entry.qkernel.block, bindings=case.bindings)
    assert_expr_equal(est.qubits, case.expected.qubits, "qubits")
    assert_gate_counts(est.gates, case.expected.gates)
    assert_depth(est.depth, case.expected.depth)


def test_no_bindings_backward_compat():
    """Without bindings, behavior should be unchanged (symbolic)."""
    entry = QKERNEL_BY_ID["qaoa_state_umbiguous"]
    est = estimate_resources(entry.qkernel.block)
    quad_sym = sp.Symbol("|quad|", integer=True, positive=True)
    assert quad_sym in est.depth.total_depth.free_symbols


def test_partial_bindings_dicts_only():
    """Partial bindings (dicts only, no scalars) should not crash."""
    entry = QKERNEL_BY_ID["qaoa_state_umbiguous"]
    est = estimate_resources(
        entry.qkernel.block,
        bindings={"quad": {(0, 1): 0.5}, "linear": {0: 0.1}},
    )
    # Scalar parameters should remain symbolic
    num_layers_sym = sp.Symbol("num_layers", integer=True, positive=True)
    assert num_layers_sym in est.depth.total_depth.free_symbols
    # Dict cardinalities should be substituted (not symbolic)
    quad_sym = sp.Symbol("|quad|", integer=True, positive=True)
    linear_sym = sp.Symbol("|linear|", integer=True, positive=True)
    assert quad_sym not in est.depth.total_depth.free_symbols
    assert linear_sym not in est.depth.total_depth.free_symbols


def test_substitute_warns_worst_case():
    """substitute() should re-emit worst-case warning for dict depths."""
    from tests.circuit.qkernel_catalog import qaoa_state_umbiguous

    # First, get symbolic result (no bindings)
    est = estimate_resources(qaoa_state_umbiguous.block)
    # The result should have worst-case flag for quad and linear
    assert len(est._worst_case_depth_dicts) > 0

    # Calling substitute should re-emit warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = est.substitute(n=4, num_layers=1)
    worst_case_msgs = [w for w in caught if "worst-case" in str(w.message)]
    assert len(worst_case_msgs) > 0
    # Verify warning messages mention specific dict names
    msg_texts = [str(w.message) for w in worst_case_msgs]
    assert any("quad" in m for m in msg_texts)
    assert any("linear" in m for m in msg_texts)
