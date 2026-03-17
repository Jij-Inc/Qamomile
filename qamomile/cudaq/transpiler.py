"""CUDA-Q backend transpiler implementation.

This module provides CudaqTranspiler for converting Qamomile QKernels
into CUDA-Q kernels, along with CudaqEmitPass and CudaqExecutor.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence, TYPE_CHECKING

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.transpiler.errors import EmitError
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

    Args:
        kernel (Any): The CUDA-Q kernel builder instance.
        num_qubits (int): Number of qubits in the circuit.
        param_values (list[float]): Bound parameter values in order.
    """

    kernel: Any
    num_qubits: int
    param_values: list[float]


class CudaqEmitPass(StandardEmitPass[CudaqCircuit]):
    """CUDA-Q-specific emission pass.

    Uses StandardEmitPass with CudaqGateEmitter for gate emission.
    Supports ``c_if`` (if-then, no else) for mid-circuit measurement
    feedback via ``kernel.c_if(mz_result, callable)``. For-loops are
    unrolled and while-loops raise ``EmitError``.
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

    def _emit_quantum_segment(
        self,
        operations: list[Operation],
        bindings: dict[str, Any],
    ) -> tuple[CudaqCircuit, dict[str, int], dict[str, int]]:
        """Emit quantum segment with post-processing for ``c_if`` circuits.

        After the standard emission, if any ``mz()`` calls were emitted
        (by ``_emit_if`` for ``c_if`` conditions), this method adds
        ``mz()`` for the remaining measured qubits.  This ensures
        ``cudaq.sample()`` reports full-length bitstrings covering all
        measured qubits, not just the ``c_if`` condition qubits.

        For circuits **without** ``c_if``, no ``mz()`` calls exist and
        ``cudaq.sample()`` auto-measures all qubits — so no post-processing
        is needed and ``cudaq.get_state()`` still returns a pure state.
        """
        circuit, qubit_map, clbit_map = super()._emit_quantum_segment(
            operations, bindings
        )

        # If c_if emitted any mz() calls, also emit mz() for the
        # remaining measured-but-not-yet-mz'd qubits so that
        # cudaq.sample() reports full bitstrings.
        if circuit.measurement_results:
            for clbit_idx, qubit_idx in self._measurement_qubit_map.items():
                if clbit_idx not in circuit.measurement_results:
                    mz_result = circuit.kernel.mz(circuit.qubits[qubit_idx])
                    circuit.measurement_results[clbit_idx] = mz_result

        return circuit, qubit_map, clbit_map

    def _emit_if(
        self,
        circuit: CudaqCircuit,
        op: IfOperation,
        qubit_map: dict[str, int],
        clbit_map: dict[str, int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit conditional execution using CUDA-Q ``kernel.c_if``.

        CUDA-Q's builder API supports ``c_if`` (if-then only). Else
        branches are not supported and raise ``EmitError``.

        Compile-time constant conditions are handled by the base class
        (``StandardEmitPass._emit_if``).

        For runtime conditions (measurement results), the ``QuakeValue``
        from ``kernel.mz()`` is obtained lazily: since
        ``noop_measurement`` is ``True``, no ``mz()`` call is emitted
        during measurement processing. Instead, ``mz()`` is called here
        just before ``c_if``, using the ``_measurement_qubit_map``
        populated by ``StandardEmitPass``.

        Args:
            circuit (CudaqCircuit): The CUDA-Q circuit being built.
            op (IfOperation): The if-operation from the IR.
            qubit_map (dict[str, int]): UUID-to-qubit-index mapping.
            clbit_map (dict[str, int]): UUID-to-clbit-index mapping.
            bindings (dict[str, Any]): Parameter bindings.

        Raises:
            EmitError: If the operation has else branches or if the
                condition classical bit has no associated measurement.
        """
        condition = op.condition

        # Compile-time constant conditions are handled by the base class.
        if not hasattr(condition, "uuid"):
            super()._emit_if(circuit, op, qubit_map, clbit_map, bindings)
            return

        condition_uuid = condition.uuid

        if condition_uuid not in clbit_map:
            return

        if op.false_operations:
            raise EmitError(
                "CUDA-Q c_if does not support else branches. "
                "Use if-then only (no else) for mid-circuit measurement feedback."
            )

        clbit_idx = clbit_map[condition_uuid]

        qubit_idx = self._measurement_qubit_map.get(clbit_idx)
        if qubit_idx is None:
            raise EmitError(
                f"No measurement found for classical bit {clbit_idx}. "
                "CUDA-Q c_if requires a measurement result as condition."
            )

        # Get or create the mz QuakeValue (avoid double-mz on same qubit)
        if clbit_idx not in circuit.measurement_results:
            mz_result = circuit.kernel.mz(circuit.qubits[qubit_idx])
            circuit.measurement_results[clbit_idx] = mz_result
        mz_result = circuit.measurement_results[clbit_idx]

        def true_body() -> None:
            self._emit_operations(
                circuit, op.true_operations, qubit_map, clbit_map, bindings
            )

        circuit.kernel.c_if(mz_result, true_body)

        self._register_phi_outputs(op, qubit_map, clbit_map, bindings)


class CudaqExecutor(QuantumExecutor[CudaqCircuit]):
    """CUDA-Q quantum executor.

    Supports sampling via ``cudaq.sample`` and expectation value estimation
    via ``cudaq.observe``.

    Args:
        target: CUDA-Q target name (e.g., ``"qpp-cpu"``). If None, uses
            the default CUDA-Q target.
    """

    def __init__(self, target: str | None = None):
        import cudaq

        self._target = target
        if self._target:
            cudaq.set_target(self._target)

    def execute(self, circuit: Any, shots: int) -> dict[str, int]:
        """Execute circuit and return bitstring counts.

        For non-parametric circuits (``CudaqCircuit``), calls
        ``cudaq.sample(kernel, shots_count=shots)``.
        For bound circuits (``BoundCudaqCircuit``), passes parameter values.

        CUDA-Q ``sample`` automatically measures all qubits when no
        explicit ``mz`` calls are present in the kernel.
        """
        import cudaq

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
            if len(bitstring) > num_qubits:
                raise ValueError(
                    f"Bitstring '{bitstring}' has length {len(bitstring)} > "
                    f"num_qubits={num_qubits}"
                )
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

        if isinstance(hamiltonian, qm_o.Hamiltonian):
            spin_op = hamiltonian_to_cudaq_spin_op(hamiltonian)
        else:
            spin_op = hamiltonian

        if isinstance(circuit, BoundCudaqCircuit):
            result = cudaq.observe(circuit.kernel, spin_op, circuit.param_values)
        elif isinstance(circuit, CudaqCircuit):
            if params is not None:
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
