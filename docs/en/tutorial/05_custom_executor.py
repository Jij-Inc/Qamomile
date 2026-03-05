# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: qamomile
#     language: python
#     name: qamomile
# ---

# %% [markdown]
# # Implementing Custom Executors: Cloud Backend Integration
#
# This tutorial teaches you how to implement custom quantum Executors in Qamomile.
# By customizing an Executor, you can run circuits on cloud quantum devices
# such as IBM Quantum and AWS Braket.
#
# ## What You'll Learn
# - The role and structure of QuantumExecutor
# - How to create a minimal custom Executor
# - How to connect to IBM Quantum cloud
# - How to implement parameter binding
# - How to implement expectation value calculation (estimate)

# %%
from qiskit import QuantumCircuit

import qamomile.circuit as qm

# %% [markdown]
# ## 1. What is QuantumExecutor?
#
# **QuantumExecutor** is an interface for executing Qamomile's compiled circuits
# on actual quantum backends.
#
# The Qamomile pipeline looks like this:
#
# ```
# @qm.qkernel (Python function)
#     ↓ transpile()
# ExecutableProgram (compiled program)
#     ↓ sample() / run()
# QuantumExecutor executes the circuit
#     ↓
# Results (bitstring counts)
# ```
#
# The standard `QiskitTranspiler` uses `AerSimulator`, but
# you can use any backend by creating a custom Executor.
# %% [markdown]
# ## 2. Basic Structure of QuantumExecutor
#
# `QuantumExecutor` is an abstract base class with three methods:
#
# | Method | Required/Optional | Description |
# |---------|---------------|------|
# | `execute()` | **Required** | Execute circuit and return bitstring counts |
# | `bind_parameters()` | Optional | Bind parameters in parametric circuits |
# | `estimate()` | Optional | Calculate expectation values of observables |
#
# The most important method is `execute()`. It's sufficient to implement just this method.
# %% [markdown]
# ### The execute() Method Specification
#
# ```python
# def execute(self, circuit: T, shots: int) -> dict[str, int]:
#     """
#     Args:
#         circuit: Backend-specific quantum circuit
#         shots: Number of measurements
#
#     Returns:
#         Dictionary from bitstrings to counts
#         Example: {"00": 512, "11": 512}
#     """
# ```
#
# **Important**: The returned bitstrings are in big-endian format.
# - "011" means qubit[2]=0, qubit[1]=1, qubit[0]=1
# - The leftmost bit is the highest indexed qubit
# %% [markdown]
# ## 3. Creating a Minimal Custom Executor
#
# Let's create a minimal Executor implementing only `execute()`.
# %%
from qamomile.circuit.transpiler.quantum_executor import QuantumExecutor
from qamomile.qiskit import QiskitTranspiler


class MySimpleExecutor(QuantumExecutor[QuantumCircuit]):
    """Minimal custom Executor

    A simple implementation using AerSimulator.
    """

    def __init__(self):
        """Initialize with AerSimulator backend"""
        from qiskit_aer import AerSimulator

        self.backend = AerSimulator()

    def execute(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
        """Execute circuit and return bitstring counts

        Args:
            circuit: Qiskit QuantumCircuit
            shots: Number of measurements

        Returns:
            Dictionary of bitstring counts (e.g., {"00": 512, "11": 512})
        """
        from qiskit import transpile

        # Add measurements if none exist
        if circuit.num_clbits == 0:
            circuit = circuit.copy()
            circuit.measure_all()

        # Transpile for backend
        transpiled = transpile(circuit, self.backend)

        # Execute
        job = self.backend.run(transpiled, shots=shots)
        return job.result().get_counts()


# %% [markdown]
# ### Testing the Custom Executor
#
# Let's use the created Executor to generate a Bell state.


# %%
@qm.qkernel
def bell_state() -> tuple[qm.Bit, qm.Bit]:
    """Generate Bell state"""
    q0 = qm.qubit(name="q0")
    q1 = qm.qubit(name="q1")
    q0 = qm.h(q0)
    q0, q1 = qm.cx(q0, q1)
    return qm.measure(q0), qm.measure(q1)


bell_state.draw()

# %%
# Transpile
transpiler = QiskitTranspiler()
executable = transpiler.transpile(bell_state)

# Execute with custom Executor
my_executor = MySimpleExecutor()
job = executable.sample(my_executor, shots=1000)
result = job.result()

print("=== Bell State Generated with Custom Executor ===")
for value, count in result.results:
    print(f"  {value}: {count} times")

# %% [markdown]
# You can confirm that a proper Bell state (roughly equal counts of |00⟩ and |11⟩) has been generated.

# %% [markdown]
# ## 4. IBM Quantum Cloud Integration
#
# Next, we'll create an Executor that connects to IBM Quantum Platform cloud backends.
#
# ### Prerequisites
#
# 1. Create an account on [IBM Quantum](https://quantum.ibm.com/)
# 2. Obtain an API token
# 3. Install `qiskit-ibm-runtime`
#
# ```bash
# pip install qiskit-ibm-runtime
# ```
#
# 4. Configure the API token
#
# ```python
# from qiskit_ibm_runtime import QiskitRuntimeService
# QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")
# ```

# %%
# IBM Quantum Executor implementation example
# Note: Requires an IBM Quantum account to actually execute


class IBMQuantumExecutor(QuantumExecutor[QuantumCircuit]):
    """Custom Executor for IBM Quantum Platform

    Executes circuits on actual IBM quantum devices or simulators.
    """

    def __init__(
        self,
        backend_name: str = "ibm_brisbane",
        channel: str = "ibm_quantum",
    ):
        """Connect to IBM Quantum Service

        Args:
            backend_name: Backend name to use
                - "ibm_brisbane": 127-qubit device
                - "ibm_sherbrooke": 127-qubit device
                - "ibmq_qasm_simulator": Cloud simulator
            channel: Channel ("ibm_quantum" or "ibm_cloud")
        """
        # Uncomment to actually use
        # from qiskit_ibm_runtime import QiskitRuntimeService
        # self.service = QiskitRuntimeService(channel=channel)
        # self.backend_name = backend_name
        self.service = None
        self.backend_name = backend_name
        self._channel = channel

    def execute(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
        """Execute circuit on IBM Quantum

        Uses the SamplerV2 Primitive for execution.
        """
        if self.service is None:
            raise RuntimeError(
                "IBM Quantum Service is not configured.\n"
                "Please configure your token with QiskitRuntimeService.save_account()."
            )

        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import SamplerV2 as Sampler

        # Get backend
        backend = self.service.backend(self.backend_name)

        # Transpile for backend (optimization level 1)
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        transpiled = pm.run(circuit)

        # Execute with SamplerV2
        sampler = Sampler(backend)
        job = sampler.run([transpiled], shots=shots)
        result = job.result()

        # Convert result to dict[str, int] format
        pub_result = result[0]
        counts = pub_result.data.meas.get_counts()
        return counts


# %% [markdown]
# ### Usage Example (Simulation)
#
# Even without an IBM Quantum account, you can test locally as follows:

# %%
# Test with local simulator
print("=== IBM Quantum Executor (Local Mock) ===")
print("Note: Actual IBM Quantum connection requires account setup")
print()

# Use local AerSimulator as substitute for testing
local_executor = MySimpleExecutor()
job = executable.sample(local_executor, shots=1000)
result = job.result()
print("Local simulation results:")
for value, count in result.results:
    print(f"  {value}: {count} times")

# %% [markdown]
# ## 5. Implementing Parameter Binding
#
# For parametric circuits like QAOA, you need to implement the `bind_parameters()` method.
#
# ### ParameterMetadata Structure
#
# `bind_parameters()` receives three arguments:
#
# ```python
# def bind_parameters(
#     self,
#     circuit: T,                          # Parameterized circuit
#     bindings: dict[str, Any],            # Parameter name → value mapping
#     parameter_metadata: ParameterMetadata # Parameter metadata
# ) -> T:
# ```
#
# `bindings` has a format like this:
# ```python
# {
#     "gammas[0]": 0.1,
#     "gammas[1]": 0.2,
#     "betas[0]": 0.3,
#     "betas[1]": 0.4
# }
# ```

# %% [markdown]
# ### ParameterMetadata Helper Methods
#
# `ParameterMetadata` provides useful helper methods:
#
# - `to_binding_dict(bindings)`: Convert to Qiskit-format binding dictionary
# - `get_ordered_params()`: Get parameters as ordered list (for QURI Parts)

# %%
from typing import Any

from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata


class MyParametricExecutor(QuantumExecutor[QuantumCircuit]):
    """Executor with parameter binding support"""

    def __init__(self):
        from qiskit_aer import AerSimulator

        self.backend = AerSimulator()

    def execute(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
        from qiskit import transpile

        if circuit.num_clbits == 0:
            circuit = circuit.copy()
            circuit.measure_all()

        transpiled = transpile(circuit, self.backend)
        job = self.backend.run(transpiled, shots=shots)
        return job.result().get_counts()

    def bind_parameters(
        self,
        circuit: QuantumCircuit,
        bindings: dict[str, Any],
        parameter_metadata: ParameterMetadata,
    ) -> QuantumCircuit:
        """Bind parameters

        Using ParameterMetadata.to_binding_dict(), you can
        easily create mappings to backend-specific parameter objects.
        """
        # Convert to Qiskit format using helper method
        qiskit_bindings = parameter_metadata.to_binding_dict(bindings)
        return circuit.assign_parameters(qiskit_bindings)


# %% [markdown]
# ### Testing Parametric Circuits


# %%
@qm.qkernel
def parametric_circuit(theta: qm.Float) -> qm.Bit:
    """Parameterized circuit"""
    q = qm.qubit(name="q")
    q = qm.h(q)
    q = qm.rz(q, theta)
    q = qm.h(q)
    return qm.measure(q)


parametric_circuit.draw()

# %%
# Transpile while preserving parameters
executable_param = transpiler.transpile(parametric_circuit, parameters=["theta"])

# Execute with parametric Executor
param_executor = MyParametricExecutor()

print("=== Parametric Circuit Test ===")
print()

for theta_val in [0.0, 1.57, 3.14]:  # 0, π/2, π
    job = executable_param.sample(
        param_executor, shots=1000, bindings={"theta": theta_val}
    )
    result = job.result()
    print(f"theta = {theta_val:.2f}:")
    for value, count in result.results:
        print(f"  {value}: {count} times")
    print()

# %% [markdown]
# ## 6. Implementing Expectation Value Calculation (estimate)
#
# Variational algorithms like QAOA require calculating expectation values of Hamiltonians.
# By implementing the `estimate()` method, you can use it in optimization loops.
#
# ### The estimate() Method Specification
#
# ```python
# def estimate(
#     self,
#     circuit: T,              # State preparation circuit
#     hamiltonian: qm_o.Hamiltonian,  # Hamiltonian to measure
#     params: Sequence[float] | None = None  # Parameter values
# ) -> float:
#     """Calculate expectation value <ψ|H|ψ>"""
# ```

# %%
from typing import Sequence

import qamomile.observable as qm_o


class MyFullExecutor(QuantumExecutor[QuantumCircuit]):
    """Custom Executor with full functionality

    - execute(): Circuit execution
    - bind_parameters(): Parameter binding
    - estimate(): Expectation value calculation
    """

    def __init__(self):
        from qiskit_aer import AerSimulator

        self.backend = AerSimulator()
        self._estimator = None

    def execute(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
        from qiskit import transpile

        if circuit.num_clbits == 0:
            circuit = circuit.copy()
            circuit.measure_all()

        transpiled = transpile(circuit, self.backend)
        job = self.backend.run(transpiled, shots=shots)
        return job.result().get_counts()

    def bind_parameters(
        self,
        circuit: QuantumCircuit,
        bindings: dict[str, Any],
        parameter_metadata: ParameterMetadata,
    ) -> QuantumCircuit:
        qiskit_bindings = parameter_metadata.to_binding_dict(bindings)
        return circuit.assign_parameters(qiskit_bindings)

    def estimate(
        self,
        circuit: QuantumCircuit,
        hamiltonian: qm_o.Hamiltonian,
        params: Sequence[float] | None = None,
    ) -> float:
        """Calculate Hamiltonian expectation value

        Uses the Qiskit Estimator primitive.
        """
        from qiskit.primitives import Estimator

        from qamomile.qiskit.observable import hamiltonian_to_sparse_pauli_op

        if self._estimator is None:
            self._estimator = Estimator()

        # Convert Hamiltonian to Qiskit format
        sparse_pauli_op = hamiltonian_to_sparse_pauli_op(hamiltonian)

        # Set parameter values
        param_values = list(params) if params is not None else []

        # Calculate expectation value
        job = self._estimator.run([(circuit, sparse_pauli_op, param_values)])
        result = job.result()

        return float(result[0].data.evs)


# %% [markdown]
# ## 7. Summary
#
# In this tutorial, we learned how to implement custom QuantumExecutors.
#
# ### Three Levels of Implementation
#
# | Level | Methods to Implement | Use Case |
# |-------|----------------|------|
# | **Basic** | `execute()` | Simple circuit execution |
# | **Intermediate** | + `bind_parameters()` | Parametric circuits |
# | **Advanced** | + `estimate()` | Variational algorithms (QAOA, etc.) |
#
# ### Implementation Key Points
#
# 1. **execute()**: Return bitstring counts as `dict[str, int]` (big-endian format)
# 2. **bind_parameters()**: Leverage `ParameterMetadata.to_binding_dict()`
# 3. **estimate()**: Use backend's Estimator primitive
#
# ### Code Example Summary
#
# ```python
# from qamomile.circuit.transpiler.quantum_executor import QuantumExecutor
# from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata
#
# class MyExecutor(QuantumExecutor[QuantumCircuit]):
#     def __init__(self):
#         self.backend = ...  # Initialize backend
#
#     def execute(self, circuit, shots):
#         # Execute circuit and return bitstring counts
#         ...
#         return {"00": 512, "11": 512}
#
#     def bind_parameters(self, circuit, bindings, metadata):
#         # Convert to backend format with metadata.to_binding_dict()
#         return circuit.assign_parameters(metadata.to_binding_dict(bindings))
#
#     def estimate(self, circuit, observable, params):
#         # Calculate expectation value with Estimator primitive
#         ...
#         return expectation_value
# ```
#
# ### Next Steps
#
# - **qaoa.py**: Use custom Executors in QAOA algorithms
# - **qpe.py**: Quantum Phase Estimation algorithm
