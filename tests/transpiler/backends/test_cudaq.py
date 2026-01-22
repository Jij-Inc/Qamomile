"""CUDA-Q transpiler test configuration.

This module configures the transpiler test suite for the CUDA-Q backend.

NOTE: CUDA-Q currently doesn't implement the GateEmitter protocol.
These tests are placeholders for when that support is added.
"""

from __future__ import annotations

from typing import Any

import pytest
import numpy as np

# Skip this entire module if CUDA-Q doesn't have emitter
pytestmark = pytest.mark.skip(
    reason="CUDA-Q does not yet implement GateEmitter protocol"
)


# Placeholder class for future implementation
# class TestCudaqTranspiler(TranspilerTestSuite):
#     """Test suite for CUDA-Q transpiler.
#
#     CUDA-Q has specific gate set limitations.
#     """
#
#     backend_name = "cudaq"
#     unsupported_gates: set[str] = {"MEASURE"}  # CUDA-Q handles measurement differently
#
#     @classmethod
#     def get_emitter(cls) -> Any:
#         """Get CUDA-Q GateEmitter instance."""
#         # TODO: Implement when CUDA-Q emitter is available
#         raise NotImplementedError("CUDA-Q emitter not yet available")
#
#     @classmethod
#     def get_transpiler(cls) -> Any:
#         """Get CUDA-Q transpiler instance."""
#         from qamomile.cudaq import CudaqTranspiler
#         return CudaqTranspiler()
#
#     @classmethod
#     def get_simulator(cls) -> Any:
#         """Get CUDA-Q simulator."""
#         # TODO: Implement when CUDA-Q emitter is available
#         raise NotImplementedError("CUDA-Q simulator not yet available")
#
#     @classmethod
#     def run_circuit_statevector(cls, circuit: Any) -> np.ndarray:
#         """Run circuit and extract statevector."""
#         # TODO: Implement when CUDA-Q emitter is available
#         raise NotImplementedError("CUDA-Q statevector extraction not yet available")
