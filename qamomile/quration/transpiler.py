"""Quration transpiler, simulator executor, and FTQC resource compilation."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from qamomile.circuit.transpiler.circuit_ir import (
    CircuitBackendEmitPass,
)
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.passes.separate import SegmentationPass
from qamomile.circuit.transpiler.quantum_executor import QuantumExecutor
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.quration.materializer import PyQretMaterializer, _require_pyqret


@dataclasses.dataclass(frozen=True)
class QurationResourceResult:
    """Package Quration FTQC compilation and resource information.

    Args:
        circuit (Any): Materialized and compiled PyQret circuit.
        compiler (Any): Owning ``pyqret.backend.Compiler``. PyQret compile
            information borrows native state from this object.
        compile_result (Any): ``pyqret.backend.CompileResult`` containing pass
            timing and ordering.
        compile_info (Any): ``pyqret.backend.ScLsFixedV0CompileInfo`` resource
            summary.
    """

    circuit: Any
    compiler: Any
    compile_result: Any
    compile_info: Any


class QurationExecutor(QuantumExecutor[Any]):
    """Execute PyQret circuits with its full-quantum simulator.

    Args:
        seed (int): Base simulator seed. Defaults to zero.
    """

    def __init__(self, seed: int = 0) -> None:
        """Initialize a deterministic-seed Quration executor.

        Args:
            seed (int): Base simulator seed. Defaults to zero.
        """
        self.seed = seed

    def execute(self, circuit: Any, shots: int) -> dict[str, int]:
        """Sample a PyQret circuit and return big-endian bitstring counts.

        Args:
            circuit (Any): ``pyqret.frontend.Circuit`` to execute.
            shots (int): Positive number of samples.

        Returns:
            dict[str, int]: Big-endian bitstring counts.

        Raises:
            ValueError: If ``shots`` is not positive.
            ImportError: If PyQret is unavailable.
        """
        if shots <= 0:
            raise ValueError("shots must be positive")
        _require_pyqret()
        from pyqret.runtime import (  # type: ignore[import-not-found]
            QuantumStateType,
            Simulator,
            SimulatorConfig,
        )

        counts: dict[str, int] = {}
        for offset in range(shots):
            config = SimulatorConfig(
                state_type=QuantumStateType.FullQuantum,
                seed=self.seed + offset,
            )
            result = Simulator(config, circuit).run()
            measured = result.get("c", [])
            bitstring = "".join("1" if bit else "0" for bit in reversed(measured))
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    def estimate(
        self,
        circuit: Any,
        hamiltonian: Any,
        params: Sequence[float] | None = None,
    ) -> float:
        """Evaluate an observable against PyQret's full state vector.

        Args:
            circuit (Any): State-preparation PyQret circuit.
            hamiltonian (Any): Qamomile ``Hamiltonian`` observable.
            params (Sequence[float] | None): Unsupported runtime parameters.
                Defaults to ``None``.

        Returns:
            float: Real expectation value.

        Raises:
            ValueError: If runtime ``params`` are supplied or the state and
                observable dimensions are inconsistent.
            ImportError: If PyQret is unavailable.
        """
        if params:
            raise ValueError("Quration parameters must be bound before materialization")
        _require_pyqret()
        import numpy as np
        from pyqret.runtime import (  # type: ignore[import-not-found]
            QuantumStateType,
            Simulator,
            SimulatorConfig,
        )

        simulator = Simulator(
            SimulatorConfig(
                state_type=QuantumStateType.FullQuantum,
                seed=self.seed,
            ),
            circuit,
        )
        simulator.run()
        state = np.asarray(
            simulator.get_full_quantum_state().get_state_vector(),
            dtype=np.complex128,
        )
        matrix = np.asarray(hamiltonian.to_numpy(), dtype=np.complex128)
        if state.size % matrix.shape[0] != 0:
            raise ValueError("Observable dimension does not divide circuit state")
        extra_dimension = state.size // matrix.shape[0]
        if extra_dimension > 1:
            matrix = np.kron(np.eye(extra_dimension, dtype=np.complex128), matrix)
        return float(np.vdot(state, matrix @ state).real)


class QurationTranspiler(Transpiler[Any]):
    """Transpile Qamomile programs to Quration through PyQret.

    Args:
        rotation_precision (float): Positive rotation synthesis precision
            forwarded to PyQret. Defaults to ``1e-10``.
    """

    def __init__(self, rotation_precision: float = 1e-10) -> None:
        """Initialize the Quration transpiler.

        Args:
            rotation_precision (float): Positive rotation synthesis precision.
                Defaults to ``1e-10``.

        Raises:
            ValueError: If ``rotation_precision`` is not positive.
        """
        if rotation_precision <= 0:
            raise ValueError("rotation_precision must be positive")
        self.rotation_precision = rotation_precision

    def _create_segmentation_pass(self) -> SegmentationPass:
        """Create the circuit-family C-to-Q-to-C planner.

        Returns:
            SegmentationPass: Standard single-quantum-segment planner.
        """
        return SegmentationPass()

    def _create_emit_pass(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> EmitPass[Any]:
        """Create the CircuitProgram-to-PyQret emission pipeline.

        Args:
            bindings (dict[str, Any] | None): Compile-time bindings. Defaults
                to ``None``.
            parameters (list[str] | None): Runtime parameter names. Defaults
                to ``None``.

        Returns:
            EmitPass[Any]: Quration emission pass.
        """
        return CircuitBackendEmitPass(
            PyQretMaterializer(self.rotation_precision),
            bindings,
            parameters,
        )

    def executor(self, **kwargs: Any) -> QurationExecutor:
        """Create a PyQret full-quantum simulator executor.

        Args:
            **kwargs (Any): Reserved executor options. Unknown values are
                rejected except for integer ``seed``.

        Returns:
            QurationExecutor: Quration simulator executor.

        Raises:
            TypeError: If unsupported executor options are supplied.
        """
        seed = kwargs.pop("seed", 0)
        if not isinstance(seed, int):
            raise TypeError("Quration executor seed must be an integer")
        if kwargs:
            raise TypeError(f"Unsupported Quration executor options: {kwargs}")
        return QurationExecutor(seed=seed)

    def compile_resources(
        self,
        kernel: Any,
        option: Any,
        bindings: dict[str, Any] | None = None,
    ) -> QurationResourceResult:
        """Compile a qkernel to a configured Quration FTQC target.

        Args:
            kernel (Any): QKernel or qkernel-like entrypoint.
            option (Any): ``pyqret.backend.CompileOption`` including the
                desired FTQC target configuration.
            bindings (dict[str, Any] | None): Compile-time bindings. Defaults
                to ``None``.

        Returns:
            QurationResourceResult: Materialized circuit, compile pass result,
                and resource information.

        Raises:
            ImportError: If PyQret is unavailable.
            QamomileCompileError: If Qamomile or Quration compilation fails.
        """
        _require_pyqret()
        from pyqret.backend import Compiler  # type: ignore[import-not-found]

        executable = self.transpile(kernel, bindings=bindings)
        circuit = executable.quantum_circuit
        compiler = Compiler(option)
        compile_result = compiler.compile(circuit)
        return QurationResourceResult(
            circuit=circuit,
            compiler=compiler,
            compile_result=compile_result,
            compile_info=compiler.get_compile_info(),
        )
