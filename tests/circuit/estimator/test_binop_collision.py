"""Tests for BinOp name collision fix in depth estimation.

Verifies that the symbol_map-based qubit key resolution correctly
distinguishes different BinOp results (e.g., n+1 vs n+2) and correctly
identifies identical BinOp expressions as the same qubit.
"""

import sympy as sp

import qamomile.circuit as qm
from qamomile.circuit.estimator import estimate_resources
from qamomile.circuit.estimator.depth_estimator import estimate_depth


class TestBinOpCollisionDepth:
    """Test that BinOp name collisions are resolved correctly in depth estimation."""

    def test_distinct_binop_indices_are_parallel(self):
        """qs[n+1] and qs[n+2] are different qubits -> gates can run in parallel -> depth=1."""

        @qm.qkernel
        def circuit(n: qm.UInt) -> qm.Vector[qm.Qubit]:
            qs = qm.qubit_array(n, name="qs")
            idx1 = n + 1
            idx2 = n + 2
            qs[idx1] = qm.h(qs[idx1])
            qs[idx2] = qm.h(qs[idx2])
            return qs

        est = estimate_depth(circuit.block)
        assert sp.simplify(est.total_depth - sp.Integer(1)) == 0, (
            f"Expected depth=1 (parallel), got {est.total_depth}"
        )

    def test_inline_binop_same_qubit_is_sequential(self):
        """qs[n+1] used twice sequentially -> depth=2."""

        @qm.qkernel
        def circuit(n: qm.UInt) -> qm.Vector[qm.Qubit]:
            qs = qm.qubit_array(n, name="qs")
            qs[n + 1] = qm.h(qs[n + 1])
            qs[n + 1] = qm.x(qs[n + 1])
            return qs

        est = estimate_depth(circuit.block)
        assert sp.simplify(est.total_depth - sp.Integer(2)) == 0, (
            f"Expected depth=2 (sequential on same qubit), got {est.total_depth}"
        )

    def test_mixed_binop_and_constant_indices(self):
        """qs[0] and qs[n+1] are different qubits -> parallel -> depth=1."""

        @qm.qkernel
        def circuit(n: qm.UInt) -> qm.Vector[qm.Qubit]:
            qs = qm.qubit_array(n, name="qs")
            qs[0] = qm.h(qs[0])
            idx = n + 1
            qs[idx] = qm.x(qs[idx])
            return qs

        est = estimate_depth(circuit.block)
        assert sp.simplify(est.total_depth - sp.Integer(1)) == 0, (
            f"Expected depth=1 (parallel), got {est.total_depth}"
        )
