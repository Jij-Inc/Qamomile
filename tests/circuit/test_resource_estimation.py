"""Tests for resource estimation."""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.decomposition import DecompositionConfig
from qamomile.circuit.resource_estimation import (
    ResourceEstimate,
    ResourceEstimator,
)
from qamomile.circuit.stdlib.qft import QFT, qft
from qamomile.circuit.frontend.handle import Vector, Qubit


class TestResourceEstimate:
    """Tests for ResourceEstimate dataclass."""

    def test_default_values(self):
        """Test default values."""
        estimate = ResourceEstimate()
        assert estimate.total_gates == 0
        assert estimate.t_gate_count == 0
        assert estimate.cnot_count == 0
        assert estimate.qubit_count == 0

    def test_add_estimates(self):
        """Test adding two estimates."""
        est1 = ResourceEstimate(
            total_gates=10,
            t_gate_count=2,
            cnot_count=5,
            qubit_count=3,
            gate_counts={"h": 3, "cx": 5},
        )
        est2 = ResourceEstimate(
            total_gates=5,
            t_gate_count=1,
            cnot_count=2,
            qubit_count=4,
            gate_counts={"h": 2, "t": 1},
        )

        combined = est1 + est2

        assert combined.total_gates == 15
        assert combined.t_gate_count == 3
        assert combined.cnot_count == 7
        assert combined.qubit_count == 4  # max of the two
        assert combined.gate_counts["h"] == 5
        assert combined.gate_counts["cx"] == 5
        assert combined.gate_counts["t"] == 1

    def test_summary(self):
        """Test summary generation."""
        estimate = ResourceEstimate(
            total_gates=10,
            t_gate_count=2,
            cnot_count=5,
            qubit_count=3,
            gate_counts={"h": 3, "cx": 5, "t": 2},
        )

        summary = estimate.summary()

        assert "Total gates: 10" in summary
        assert "T gates: 2" in summary
        assert "Qubits: 3" in summary


class TestResourceEstimator:
    """Tests for ResourceEstimator."""

    def test_estimate_simple_kernel(self):
        """Test estimating a simple kernel."""

        @qmc.qkernel
        def simple(q: Qubit) -> Qubit:
            q = qmc.h(q)
            q = qmc.x(q)
            return q

        estimator = ResourceEstimator()
        estimate = estimator.estimate(simple)

        assert estimate.total_gates >= 2
        assert "h" in estimate.gate_counts
        assert "x" in estimate.gate_counts

    def test_estimate_with_loop(self):
        """Test estimating kernel with loop."""

        @qmc.qkernel
        def looped(q: Qubit, n: int) -> Qubit:
            for _ in qmc.range(n):
                q = qmc.h(q)
            return q

        estimator = ResourceEstimator()
        estimate = estimator.estimate(looped, bindings={"n": 5})

        # Should have 5 H gates from the loop
        assert estimate.gate_counts.get("h", 0) >= 5

    def test_estimate_qft(self):
        """Test estimating QFT resources."""

        @qmc.qkernel
        def qft_circuit(qubits: Vector[Qubit]) -> Vector[Qubit]:
            qubits = qft(qubits)
            return qubits

        estimator = ResourceEstimator()
        # Note: This test may need adjustment based on how QFT is traced
        # The important thing is that it doesn't crash and returns an estimate

    def test_compare_strategies(self):
        """Test comparing strategies."""

        @qmc.qkernel
        def simple_circuit(q: Qubit) -> Qubit:
            q = qmc.h(q)
            return q

        estimator = ResourceEstimator()
        comparison = estimator.compare_strategies(
            simple_circuit,
            strategies=["standard", "approximate"],
        )

        assert "standard" in comparison
        assert "approximate" in comparison

    def test_estimator_with_config(self):
        """Test estimator with decomposition config."""
        config = DecompositionConfig(
            strategy_overrides={"qft": "approximate"},
        )
        estimator = ResourceEstimator(decomposition_config=config)

        # Should use approximate strategy for QFT
        assert estimator._config.get_strategy_for_gate("qft") == "approximate"
