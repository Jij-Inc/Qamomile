"""PennyLane transpiler test configuration.

This module configures the transpiler test suite for the PennyLane backend.

NOTE: PennyLane currently doesn't implement the GateEmitter protocol.
These tests are placeholders for when that support is added.
"""

from __future__ import annotations

from typing import Any

import pytest
import numpy as np

# Skip this entire module if PennyLane doesn't have emitter
pytestmark = pytest.mark.skip(
    reason="PennyLane does not yet implement GateEmitter protocol"
)


# Placeholder class for future implementation
# class TestPennylaneTranspiler(TranspilerTestSuite, HamiltonianTestMixin):
#     """Test suite for PennyLane transpiler.
#
#     PennyLane supports most standard gates with some name differences.
#     """
#
#     backend_name = "pennylane"
#     unsupported_gates: set[str] = set()
#
#     @classmethod
#     def get_emitter(cls) -> Any:
#         """Get PennyLane GateEmitter instance."""
#         # TODO: Implement when PennyLane emitter is available
#         raise NotImplementedError("PennyLane emitter not yet available")
#
#     @classmethod
#     def get_transpiler(cls) -> Any:
#         """Get PennyLane transpiler instance."""
#         from qamomile.pennylane import PennylaneTranspiler
#         return PennylaneTranspiler()
#
#     @classmethod
#     def get_simulator(cls) -> Any:
#         """Get PennyLane statevector device."""
#         import pennylane as qml
#         return qml.device("default.qubit")
#
#     @classmethod
#     def run_circuit_statevector(cls, circuit: Any) -> np.ndarray:
#         """Run circuit and extract statevector."""
#         # TODO: Implement when PennyLane emitter is available
#         raise NotImplementedError("PennyLane statevector extraction not yet available")
