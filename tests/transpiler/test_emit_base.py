"""Tests for emit_base pass — UInt BinOp folding into loop bounds.

These tests verify that UInt BinOp operations (``//``, ``**``) are correctly
folded by the constant folding pass so that the resulting constant can be
used as a ``qmc.range`` argument for loop unrolling in the emit pass.
"""

from typing import Any, TYPE_CHECKING

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.qiskit.transpiler import QiskitTranspiler

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


# ---------------------------------------------------------------------------
# Module-level @qkernel definitions (required for inspect.getsource)
# ---------------------------------------------------------------------------


@qmc.qkernel
def binop_floordiv_circuit(
    n: qmc.UInt, theta: qmc.Float
) -> qmc.Vector[qmc.Bit]:
    """Apply RX(theta) to first n // 2 qubits of a 4-qubit register."""
    q = qmc.qubit_array(4, "q")
    count = n // 2
    for i in qmc.range(count):
        q[i] = qmc.rx(q[i], angle=theta)
    return qmc.measure(q)


@qmc.qkernel
def binop_pow_circuit(
    n: qmc.UInt, theta: qmc.Float
) -> qmc.Vector[qmc.Bit]:
    """Apply RX(theta) to first n ** 2 qubits of a 4-qubit register."""
    q = qmc.qubit_array(4, "q")
    count = n ** 2
    for i in qmc.range(count):
        q[i] = qmc.rx(q[i], angle=theta)
    return qmc.measure(q)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _transpile_and_get_circuit(
    kernel: Any,
    bindings: dict[str, Any] | None = None,
) -> tuple[Any, "QuantumCircuit"]:
    """Transpile a kernel and return ``(executable, circuit)``."""
    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(kernel, bindings=bindings)
    qc = exe.compiled_quantum[0].circuit
    return exe, qc


def _extract_rx_angles(qc: "QuantumCircuit") -> list[float]:
    """Extract all RX gate angles from a Qiskit circuit."""
    return [
        float(inst.operation.params[0])
        for inst in qc.data
        if inst.operation.name == "rx"
    ]


# ---------------------------------------------------------------------------
# Tests — UInt BinOp (floordiv, pow) folding into loop bounds
# ---------------------------------------------------------------------------


class TestUIntBinOpFolding:
    """Tests for UInt BinOp kinds (``//``, ``**``) that affect loop bounds.

    At tracing time ``n`` is a symbolic ``UInt`` handle, so ``n // 2``
    emits a ``BinOp(FLOORDIV)`` into the IR.  The constant folding pass
    resolves the parameter binding and evaluates the BinOp, producing
    a concrete loop bound that the emit pass can unroll.
    """

    @pytest.mark.parametrize(
        "n, theta, expected_rx_count",
        [
            (4, 0.5, 2),   # 4 // 2 = 2
            (6, 0.3, 3),   # 6 // 2 = 3
            (2, 1.0, 1),   # 2 // 2 = 1
        ],
        ids=["4//2=2", "6//2=3", "2//2=1"],
    )
    def test_floordiv_loop_bound(
        self, n: int, theta: float, expected_rx_count: int
    ) -> None:
        """``n // 2`` correctly folded as loop bound; angles verified."""
        _, qc = _transpile_and_get_circuit(
            binop_floordiv_circuit, bindings={"n": n, "theta": theta}
        )
        rx_angles = _extract_rx_angles(qc)

        assert len(rx_angles) == expected_rx_count, (
            f"Expected {expected_rx_count} RX gates (n={n}, n//2={n // 2}), "
            f"got {len(rx_angles)}"
        )
        for i, angle in enumerate(rx_angles):
            assert np.isclose(angle, theta), (
                f"RX[{i}] angle {angle} != expected {theta}"
            )

    @pytest.mark.parametrize(
        "n, theta, expected_rx_count",
        [
            (2, 0.3, 4),   # 2 ** 2 = 4
            (1, 0.5, 1),   # 1 ** 2 = 1
        ],
        ids=["2**2=4", "1**2=1"],
    )
    def test_pow_loop_bound(
        self, n: int, theta: float, expected_rx_count: int
    ) -> None:
        """``n ** 2`` correctly folded as loop bound; angles verified."""
        _, qc = _transpile_and_get_circuit(
            binop_pow_circuit, bindings={"n": n, "theta": theta}
        )
        rx_angles = _extract_rx_angles(qc)

        assert len(rx_angles) == expected_rx_count, (
            f"Expected {expected_rx_count} RX gates (n={n}, n**2={n ** 2}), "
            f"got {len(rx_angles)}"
        )
        for i, angle in enumerate(rx_angles):
            assert np.isclose(angle, theta), (
                f"RX[{i}] angle {angle} != expected {theta}"
            )
