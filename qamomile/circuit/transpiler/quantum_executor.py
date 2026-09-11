"""Abstract base class for quantum backend execution.

This module provides a simple interface for implementing custom quantum executors.
Executors bridge Qamomile's compiled circuits to various quantum backends including
local simulators and cloud quantum devices.

Basic Example (Local Simulator):
    from qiskit_aer import AerSimulator

    class MyExecutor(QuantumExecutor[QuantumCircuit]):
        def __init__(self):
            self.backend = AerSimulator()

        def execute(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
            from qiskit import transpile
            if circuit.num_clbits == 0:
                circuit = circuit.copy()
                circuit.measure_all()
            transpiled = transpile(circuit, self.backend)
            job = self.backend.run(transpiled, shots=shots)
            return job.result().get_counts()

Cloud Backend Example (IBM Quantum):
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    class IBMQuantumExecutor(QuantumExecutor[QuantumCircuit]):
        def __init__(self, backend_name: str = "ibm_brisbane"):
            self.service = QiskitRuntimeService()
            self.backend_name = backend_name

        def execute(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
            from qiskit.transpiler.preset_passmanagers import (
                generate_preset_pass_manager,
            )
            backend = self.service.backend(self.backend_name)
            pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
            transpiled = pm.run(circuit)
            sampler = SamplerV2(backend)
            job = sampler.run([transpiled], shots=shots)
            return job.result()[0].data.meas.get_counts()

        def bind_parameters(self, circuit, bindings, metadata):
            # Use helper method for easy conversion
            return circuit.assign_parameters(metadata.to_binding_dict(bindings))

Bitstring Format:
    The execute() method returns dict[str, int] where keys are bitstrings
    in big-endian format (leftmost bit = highest qubit index).
    Example: {"011": 512, "100": 488} for 3 qubits
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from qamomile.circuit.transpiler.execution_capability import ExecutionCapabilities
from qamomile.circuit.transpiler.execution_handle import (
    CompletedExecutionHandle,
    CompositeExecutionHandle,
    ExecutionHandle,
    ExecutionReference,
)
from qamomile.circuit.transpiler.execution_request import (
    CircuitInvocation,
    EstimateRequest,
    SampleRequest,
)
from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata

if TYPE_CHECKING:
    import qamomile.observable as qm_o

T = TypeVar("T")  # Engine circuit type


class QuantumExecutor(ABC, Generic[T]):
    """Abstract base class for quantum backend execution.

    To implement a custom executor:

    1. **execute() [Required]**
       Execute circuit and return bitstring counts as dict[str, int].
       Keys are bitstrings in big-endian format (e.g., "011" means q2=0, q1=1, q0=1).

    2. **bind_parameters() [Optional]**
       Bind parameter values to parametric circuits. Override if your executor
       supports parametric circuits (e.g., QAOA variational circuits).
       Use ParameterMetadata.to_binding_dict() for easy conversion.

    3. **estimate() [Optional]**
       Compute expectation values <psi|H|psi>. Override if your executor
       supports estimation primitives (e.g., Qiskit Estimator, QURI Parts).

    Example (Minimal):
        class MyExecutor(QuantumExecutor[QuantumCircuit]):
            def __init__(self):
                from qiskit_aer import AerSimulator
                self.backend = AerSimulator()

            def execute(self, circuit, shots):
                from qiskit import transpile
                if circuit.num_clbits == 0:
                    circuit = circuit.copy()
                    circuit.measure_all()
                transpiled = transpile(circuit, self.backend)
                return self.backend.run(transpiled, shots=shots).result().get_counts()

    Example (With Parameter Binding):
        def bind_parameters(self, circuit, bindings, metadata):
            # metadata.to_binding_dict() converts indexed names to engine params
            return circuit.assign_parameters(metadata.to_binding_dict(bindings))
    """

    @abstractmethod
    def execute(self, circuit: T, shots: int) -> dict[str, int]:
        """Execute the circuit and return bitstring counts.

        Args:
            circuit: The quantum circuit to execute
            shots: Number of measurement shots

        Returns:
            Dictionary mapping bitstrings to counts.
            Example: {"00": 512, "11": 512}
        """
        pass

    @property
    def capabilities(self) -> ExecutionCapabilities:
        """Describe backend execution features available through this adapter.

        Synchronous compatibility executors inherit conservative lifecycle
        defaults. Estimation support is inferred from the overridden method so
        existing third-party executors do not need an immediate source change.

        Returns:
            ExecutionCapabilities: Immutable executor capability declaration.
        """
        return ExecutionCapabilities(
            supports_estimation=type(self).estimate is not QuantumExecutor.estimate
        )

    def submit_sample(
        self,
        request: SampleRequest[T],
    ) -> ExecutionHandle[dict[str, int]]:
        """Submit a sampling request through the synchronous compatibility path.

        Remote executors should override this method and return immediately
        with a provider-backed handle. Existing synchronous executors inherit
        this implementation unchanged.

        Args:
            request (SampleRequest[T]): Circuit invocation and shot count.

        Returns:
            ExecutionHandle[dict[str, int]]: Completed compatibility handle.
        """
        circuit = self.bind_invocation(request.invocation)
        return CompletedExecutionHandle(self.execute(circuit, request.shots))

    def submit_samples(
        self,
        requests: Sequence[SampleRequest[T]],
    ) -> ExecutionHandle[tuple[dict[str, int], ...]]:
        """Submit an ordered collection of sampling requests.

        Backends with native batch or parameter-sweep support should override
        this method. The default preserves ordering with individual requests.

        Args:
            requests (Sequence[SampleRequest[T]]): Sampling requests.

        Returns:
            ExecutionHandle[tuple[dict[str, int], ...]]: Composite result
                handle preserving request order.
        """
        return CompositeExecutionHandle(
            [self.submit_sample(request) for request in requests]
        )

    def submit_estimate(
        self,
        request: EstimateRequest[T],
    ) -> ExecutionHandle[float]:
        """Submit an expectation request through the synchronous path.

        Args:
            request (EstimateRequest[T]): Circuit, Hamiltonian, and optional
                accuracy policy.

        Returns:
            ExecutionHandle[float]: Completed compatibility handle.

        Raises:
            NotImplementedError: If an explicit accuracy policy is requested
                from an executor that has not implemented request submission.
        """
        if request.accuracy is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support per-request estimation "
                "accuracy; configure the executor or override submit_estimate()."
            )
        circuit = self.bind_invocation(request.invocation)
        return CompletedExecutionHandle(self.estimate(circuit, request.hamiltonian))

    def submit_estimates(
        self,
        requests: Sequence[EstimateRequest[T]],
    ) -> ExecutionHandle[tuple[float, ...]]:
        """Submit an ordered collection of expectation requests.

        Args:
            requests (Sequence[EstimateRequest[T]]): Expectation requests.

        Returns:
            ExecutionHandle[tuple[float, ...]]: Composite result handle in
                request order.
        """
        return CompositeExecutionHandle(
            [self.submit_estimate(request) for request in requests]
        )

    def bind_invocation(self, invocation: CircuitInvocation[T]) -> T:
        """Bind one invocation for a backend without native input submission.

        Args:
            invocation (CircuitInvocation[T]): Circuit, flattened bindings,
                and engine parameter metadata.

        Returns:
            T: Bound circuit, or the original circuit when it has no runtime
                parameters.
        """
        if not invocation.parameter_metadata.parameters:
            return invocation.circuit
        return self.bind_parameters(
            invocation.circuit,
            dict(invocation.bindings),
            invocation.parameter_metadata,
        )

    def restore(
        self,
        reference: ExecutionReference,
    ) -> ExecutionHandle[Any]:
        """Restore a remote execution from a secret-free reference.

        Args:
            reference (ExecutionReference): Provider execution reference.

        Returns:
            ExecutionHandle[Any]: Restored provider-backed handle.

        Raises:
            NotImplementedError: If this executor cannot restore jobs.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support execution restoration"
        )

    def bind_parameters(
        self,
        circuit: T,
        bindings: dict[str, Any],
        parameter_metadata: ParameterMetadata,
    ) -> T:
        """Bind parameter values to the circuit.

        Default implementation returns the circuit unchanged.
        Override for backends that support parametric circuits.

        Args:
            circuit: The parameterized circuit
            bindings: Dict mapping parameter names (indexed format) to values.
                     e.g., {"gammas[0]": 0.1, "gammas[1]": 0.2}
            parameter_metadata: Metadata about circuit parameters

        Returns:
            New circuit with parameters bound
        """
        return circuit

    def estimate(
        self,
        circuit: T,
        hamiltonian: "qm_o.Hamiltonian",
        params: Sequence[float] | None = None,
    ) -> float:
        """Estimate the expectation value of a Hamiltonian.

        This method computes <psi|H|psi> where psi is the quantum state
        prepared by the circuit and H is the Hamiltonian.

        Backends can override this method to provide optimized implementations
        using their native estimator primitives.

        Args:
            circuit: The quantum circuit (state preparation ansatz)
            hamiltonian: The qamomile.observable.Hamiltonian to measure
            params: Optional parameter values for parametric circuits

        Returns:
            The estimated expectation value

        Raises:
            NotImplementedError: If the executor does not support estimation
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support expectation value estimation. "
            "Use an executor with estimator support (e.g., QiskitExecutor with estimator)."
        )
