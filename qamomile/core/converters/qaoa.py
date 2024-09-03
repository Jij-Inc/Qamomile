"""
This module implements the Quantum Approximate Optimization Algorithm (QAOA) converter
for the Qamomile framework. It provides functionality to convert optimization problems
into QAOA circuits, construct cost Hamiltonians, and decode quantum computation results.

The QAOAConverter class extends the QuantumConverter base class, specializing in
QAOA-specific operations such as ansatz circuit generation and result decoding.

Key Features:
- Generation of QAOA ansatz circuits
- Construction of cost Hamiltonians for QAOA
- Decoding of quantum computation results into classical optimization solutions

Usage:
    from qamomile.core.qaoa.qaoa import QAOAConverter

    # Initialize with a compiled optimization problem instance
    qaoa_converter = QAOAConverter(compiled_instance)

    # Generate QAOA circuit
    p = 2  # Number of QAOA layers
    qaoa_circuit = qaoa_converter.get_ansatz_circuit(p)

    # Get cost Hamiltonian
    cost_hamiltonian = qaoa_converter.get_cost_hamiltonian()


Note: This module requires jijmodeling and jijmodeling_transpiler for problem representation
and decoding functionalities.
"""

import typing as typ
import jijmodeling_transpiler.core as jmt
import qamomile.core.bitssample as qm_bs
import qamomile.core.circuit as qm_c
import qamomile.core.operator as qm_o
from qamomile.core.converters.converter import QuantumConverter


class QAOAConverter(QuantumConverter):
    """
    QAOA (Quantum Approximate Optimization Algorithm) converter class.

    This class provides methods to convert optimization problems into QAOA circuits,
    construct cost Hamiltonians, and decode quantum computation results.
    """

    def get_cost_ansatz(
        self, beta: qm_c.Parameter, name: str = "Cost"
    ) -> qm_c.QuantumCircuit:
        """
        Generate the cost ansatz circuit for QAOA.

        Args:
            beta (qm_c.Parameter): The beta parameter for the cost ansatz.
            name (str, optional): Name of the circuit. Defaults to "Cost".

        Returns:
            qm_c.QuantumCircuit: The cost ansatz circuit.
        """
        ising = self.get_ising()
        num_qubits = ising.num_bits()

        cost = qm_c.QuantumCircuit(num_qubits, 0, name=name)

        # Apply RZ gates for linear terms
        for i, hi in ising.linear.items():
            if hi != 0.0:
                cost.rz(2 * hi * beta, i)

        # Apply CNOT and RZ gates for quadratic terms
        for (i, j), Jij in ising.quad.items():
            if Jij != 0.0:
                cost.rzz(2 * Jij * beta, i, j)

        cost.update_qubits_label(self.int2varlabel)

        return cost

    def get_qaoa_ansatz(
        self, p: int, initial_hadamard: bool = True
    ) -> qm_c.QuantumCircuit:
        """
        Generate the complete QAOA ansatz circuit.

        Args:
            p (int): Number of QAOA layers.
            initial_hadamard (bool, optional): Whether to apply initial Hadamard gates. Defaults to True.

        Returns:
            qm_c.QuantumCircuit: The complete QAOA ansatz circuit.
        """
        ising = self.get_ising()
        num_qubits = ising.num_bits()
        qaoa_circuit = qm_c.QuantumCircuit(num_qubits, 0, name="QAOA")

        # Apply initial Hadamard gates if specified
        if initial_hadamard:
            for i in range(num_qubits):
                qaoa_circuit.h(i)

        # Construct QAOA layers
        for _p in range(p):
            beta = qm_c.Parameter(f"beta_{_p}")
            cost = self.get_cost_ansatz(beta, name=f"Cost_{_p}")

            mixer = qm_c.QuantumCircuit(num_qubits, 0, name=f"Mixer_{_p}")
            gamma = qm_c.Parameter(f"gamma_{_p}")
            for i in range(num_qubits):
                mixer.rx(2 * gamma, i)

            qaoa_circuit.append(cost)
            qaoa_circuit.append(mixer)

        qaoa_circuit.update_qubits_label(self.int2varlabel)

        return qaoa_circuit

    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        """
        Construct the cost Hamiltonian for QAOA.

        Returns:
            qm_o.Hamiltonian: The cost Hamiltonian.
        """
        hamiltonian = qm_o.Hamiltonian()
        ising = self.get_ising()

        # Add linear terms
        for i, hi in ising.linear.items():
            if hi != 0.0:
                hamiltonian.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, i),), hi)

        # Add quadratic terms
        for (i, j), Jij in ising.quad.items():
            if Jij != 0.0:
                hamiltonian.add_term(
                    (
                        qm_o.PauliOperator(qm_o.Pauli.Z, i),
                        qm_o.PauliOperator(qm_o.Pauli.Z, j),
                    ),
                    Jij,
                )

        hamiltonian.constant = ising.constant
        return hamiltonian
