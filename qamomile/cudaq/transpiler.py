"""CUDA-Q backend transpiler implementation.

This module provides CudaqTranspiler for converting Qamomile QKernels
into CUDA-Q kernels, along with CudaqEmitPass and CudaqExecutor.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence, TYPE_CHECKING

from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.passes.separate import SeparatePass
from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass
from qamomile.circuit.transpiler.executable import (
    QuantumExecutor,
    ParameterMetadata,
)

from .emitter import CudaqCircuit, CudaqGateEmitter

if TYPE_CHECKING:
    import qamomile.observable as qm_o


@dataclasses.dataclass
class BoundCudaqCircuit:
    """CUDA-Q kernel with bound parameter values.

    Used as the return type of ``CudaqExecutor.bind_parameters``.
    The executor dispatches to ``cudaq.sample(kernel, param_values, ...)``
    or ``cudaq.observe(kernel, spin_op, param_values)`` when it receives
    this type.
    """

    kernel: Any
    num_qubits: int
    param_values: list[float]


class CudaqEmitPass(StandardEmitPass[CudaqCircuit]):
    """CUDA-Q-specific emission pass.

    Uses StandardEmitPass with CudaqGateEmitter for gate emission.
    CUDA-Q does not support native control flow in this implementation,
    so all for-loops are unrolled and if/while raise NotImplementedError.
    """

    def __init__(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ):
        parametric = bool(parameters)
        emitter = CudaqGateEmitter(parametric=parametric)
        composite_emitters: list[Any] = []
        super().__init__(emitter, bindings, parameters, composite_emitters)


class CudaqExecutor(QuantumExecutor[CudaqCircuit]):
    """CUDA-Q quantum executor.

    Supports sampling via ``cudaq.sample`` and expectation value estimation
    via ``cudaq.observe``.

    Args:
        target: CUDA-Q target name (e.g., ``"qpp-cpu"``). If None, uses
            the default CUDA-Q target.
    """

    def __init__(self, target: str | None = None):
        self._target = target

    def execute(self, circuit: Any, shots: int) -> dict[str, int]:
        """Execute circuit and return bitstring counts.

        For non-parametric circuits (``CudaqCircuit``), calls
        ``cudaq.sample(kernel, shots_count=shots)``.
        For bound circuits (``BoundCudaqCircuit``), passes parameter values.

        CUDA-Q ``sample`` automatically measures all qubits when no
        explicit ``mz`` calls are present in the kernel.
        """
        import cudaq

        if self._target:
            cudaq.set_target(self._target)

        if isinstance(circuit, BoundCudaqCircuit):
            result = cudaq.sample(
                circuit.kernel, circuit.param_values, shots_count=shots
            )
            num_qubits = circuit.num_qubits
        else:
            result = cudaq.sample(circuit.kernel, shots_count=shots)
            num_qubits = circuit.num_qubits

        counts: dict[str, int] = {}
        for bitstring in result:
            count = result.count(bitstring)
            # Pad bitstring to num_qubits length
            padded = bitstring.zfill(num_qubits)
            counts[padded] = counts.get(padded, 0) + count

        return counts

    def bind_parameters(
        self,
        circuit: CudaqCircuit,
        bindings: dict[str, Any],
        parameter_metadata: ParameterMetadata,
    ) -> BoundCudaqCircuit:
        """Bind parameters by wrapping the kernel with parameter values.

        CUDA-Q does not support in-place parameter binding. Instead,
        parameters are passed at execution time. This method creates a
        ``BoundCudaqCircuit`` that stores the kernel and parameter values
        together.
        """
        param_values = []
        for param_info in parameter_metadata.parameters:
            if param_info.name in bindings:
                param_values.append(float(bindings[param_info.name]))
            else:
                raise ValueError(
                    f"Missing binding for parameter '{param_info.name}'. "
                    f"Provided bindings: {list(bindings.keys())}. "
                    f"Required parameters: "
                    f"{[p.name for p in parameter_metadata.parameters]}"
                )

        return BoundCudaqCircuit(
            kernel=circuit.kernel,
            num_qubits=circuit.num_qubits,
            param_values=param_values,
        )

    def estimate(
        self,
        circuit: Any,
        hamiltonian: "qm_o.Hamiltonian",
        params: Sequence[float] | None = None,
    ) -> float:
        """Estimate expectation value using ``cudaq.observe``.

        Dispatches based on circuit type:
        - ``BoundCudaqCircuit``: uses stored param_values
        - ``CudaqCircuit`` with params: passes params to observe
        - ``CudaqCircuit`` without params: no-parameter observe
        """
        import cudaq
        import qamomile.observable as qm_o

        from qamomile.cudaq.observable import hamiltonian_to_cudaq_spin_op

        if self._target:
            cudaq.set_target(self._target)

        if isinstance(hamiltonian, qm_o.Hamiltonian):
            spin_op = hamiltonian_to_cudaq_spin_op(hamiltonian)
        else:
            spin_op = hamiltonian

        if isinstance(circuit, BoundCudaqCircuit):
            result = cudaq.observe(circuit.kernel, spin_op, circuit.param_values)
        elif isinstance(circuit, CudaqCircuit):
            if params:
                result = cudaq.observe(circuit.kernel, spin_op, list(params))
            else:
                result = cudaq.observe(circuit.kernel, spin_op)
        else:
            raise TypeError(f"Unexpected circuit type: {type(circuit)}")

        return result.expectation()


class CudaqTranspiler(Transpiler[CudaqCircuit]):
    """CUDA-Q transpiler for qamomile.circuit module.

    Converts Qamomile QKernels into CUDA-Q kernels.

    Example:
        from qamomile.cudaq import CudaqTranspiler
        import qamomile.circuit as qm

        @qm.qkernel
        def bell_state(q0: qm.Qubit, q1: qm.Qubit) -> tuple[qm.Bit, qm.Bit]:
            q0 = qm.h(q0)
            q0, q1 = qm.cx(q0, q1)
            return qm.measure(q0), qm.measure(q1)

        transpiler = CudaqTranspiler()
        executable = transpiler.transpile(bell_state)
    """

    def _create_separate_pass(self) -> SeparatePass:
        return SeparatePass()

    def _create_emit_pass(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> EmitPass[CudaqCircuit]:
        return CudaqEmitPass(bindings, parameters)

    def executor(
        self,
        target: str | None = None,
    ) -> CudaqExecutor:
        """Create a CUDA-Q executor.

        Args:
            target: CUDA-Q target name (e.g., ``"qpp-cpu"``).
                If None, uses the default CUDA-Q target.
        """
        return CudaqExecutor(target)
