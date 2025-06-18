"""
This module implements the Quantum Approximate Optimization Algorithm (QAOA) converter
for the Qamomile framework :cite:`farhi2014quantum`.
The parameterized state :math:`|\\vec{\\beta},\\vec{\gamma}\\rangle` of :math:`p`-layer QAOA is defined as:

.. math::
    |\\vec{\\beta},\\vec{\gamma}\\rangle = U(\\vec{\\beta},\\vec{\gamma})|0\\rangle^{\otimes n} = e^{-i\\beta_{p-1} H_M}e^{-i\gamma_{p-1} H_P} \cdots e^{-i\\beta_0 H_M}e^{-i\gamma_0 H_P} H^{\otimes n}|0\\rangle^{\otimes n}

where :math:`H_P` is the cost Hamiltonian, :math:`H_M` is the mixer Hamiltonian and :math:`\gamma_l` and :math:`\\beta_l` are the variational parameters.
The 2 :math:`p` variational parameters are optimized classically to minimize the expectation value :math:`\langle \\vec{\\beta},\\vec{\gamma}|H_P|\\vec{\\beta},\\vec{\gamma}\\rangle`.

This module provides functionality to convert optimization problems which written by `jijmodeling`
into QAOA circuits (:math:`U(\\vec{\\beta},\\vec{\gamma})`), construct cost Hamiltonians (:math:`H_P`), and decode quantum computation results.

The `QAOAConverter` class extends the `QuantumConverter` base class, specializing in
QAOA-specific operations such as ansatz circuit generation and result decoding.


Key Features:
- Generation of QAOA ansatz circuits
- Construction of cost Hamiltonians for QAOA
- Decoding of quantum computation results into classical optimization solutions


Note: This module requires `jijmodeling` for problem representation.

.. bibliography::
    :filter: docname in docnames

"""

import typing as typ
import qamomile.core.bitssample as qm_bs
import qamomile.core.circuit as qm_c
import qamomile.core.operator as qm_o
from qamomile.core.converters.converter import QuantumConverter
from qamomile.core.converters.utils import is_close_zero


class QAOAConverter(QuantumConverter):
    """
    QAOA (Quantum Approximate Optimization Algorithm) converter class.

    This class provides methods to convert optimization problems into QAOA circuits,
    construct cost Hamiltonians, and decode quantum computation results.

    Examples:

    .. code::

        from qamomile.core.qaoa.qaoa import QAOAConverter

        # Initialize with a compiled optimization problem instance
        qaoa_converter = QAOAConverter(compiled_instance)

        # Generate QAOA circuit and cost Hamiltonian
        p = 2  # Number of QAOA layers
        qaoa_circuit = qaoa_converter.get_ansatz_circuit(p)
        cost_hamiltonian = qaoa_converter.get_cost_hamiltonian()

    """

    def get_cost_ansatz(
        self, gamma: qm_c.Parameter, name: str = "Cost"
    ) -> qm_c.QuantumCircuit:
        """
        Generate the cost ansatz circuit (:math:`e^{-\gamma H_P}`) for QAOA.
        This function is mainly used when you have designed your own mixer Hamiltonian and only need the cost Hamiltonian.

        Args:
            gamma (qm_c.Parameter): The gamma parameter for the cost ansatz.
            name (str, optional): Name of the circuit. Defaults to "Cost".

        Returns:
            qm_c.QuantumCircuit: The cost ansatz circuit.
        """
        ising = self.get_ising()
        num_qubits = ising.num_bits()

        cost = qm_c.QuantumCircuit(num_qubits, 0, name=name)

        # Apply RZ gates for linear terms
        for i, hi in ising.linear.items():
            if not is_close_zero(hi):
                cost.rz(2 * hi * gamma, i)

        # Apply CNOT and RZ gates for quadratic terms
        for (i, j), Jij in ising.quad.items():
            if not is_close_zero(Jij):
                cost.rzz(2 * Jij * gamma, i, j)

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
            gamma = qm_c.Parameter(f"gamma_{_p}")
            cost = self.get_cost_ansatz(gamma, name=f"Cost_{_p}")

            mixer = qm_c.QuantumCircuit(num_qubits, 0, name=f"Mixer_{_p}")
            for i in range(num_qubits):
                mixer.rx(2 * beta, i)

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
            if not is_close_zero(hi):
                hamiltonian.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, i),), hi)

        # Add quadratic terms
        for (i, j), Jij in ising.quad.items():
            if not is_close_zero(Jij):
                hamiltonian.add_term(
                    (
                        qm_o.PauliOperator(qm_o.Pauli.Z, i),
                        qm_o.PauliOperator(qm_o.Pauli.Z, j),
                    ),
                    Jij,
                )

        hamiltonian.constant = ising.constant
        return hamiltonian
