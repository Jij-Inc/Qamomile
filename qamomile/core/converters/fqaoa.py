"""
This module implements the Fermionic QAOA converter
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


Note: This module requires `jijmodeling` and `jijmodeling_transpiler` for problem representation.

.. bibliography::
    :filter: docname in docnames

"""

import numpy as np
import typing as typ
import jijmodeling_transpiler.core as jmt
import qamomile.core.bitssample as qm_bs
import qamomile.core.circuit as qm_c
import qamomile.core.operator as qm_o
from qamomile.core.converters.converter import QuantumConverter
from qamomile.core.converters.utils import is_close_zero
import ommx.v1
from openfermion.linalg import givens_decomposition


class FQAOAConverter(QuantumConverter):
    """
    FQAOA (Fermionic Quantum Approximate Optimization Algorithm) converter class.

    This class provides methods to convert optimization problems into FQAOA circuits,
    construct cost Hamiltonians, and decode quantum computation results.

    Examples:

    .. code::

        from qamomile.core.qaoa.fqaoa import FQAOAConverter
        
        # Initialize with a compiled optimization problem instance 
        fqaoa_converter = FQAOAConverter(compiled_instance) 

        # Generate QAOA circuit and cost Hamiltonian
        p = 2  # Number of QAOA layers
        qaoa_circuit = fqaoa_converter.get_ansatz_circuit(p) 
        cost_hamiltonian = fqaoa_converter.get_cost_hamiltonian()

    """
    
    def get_init_state(
		self, slater_det: np.array, 
	):

        givens_angles = 1
        
    def add_snake_jw_hopping_operator(
    circuit: qm_c.QuantumCircuit,
    beta: qm_c.Parameter,
    hopping,
    index: typ.Optional[int]
	):
        circuit.

    def get_mixer_ansatz(
		self,
		beta: qm_c.Parameter, 
		lattice_structure: typ.Optional[typ.Literal["cyclic", "ladder"]] = "cyclic",
		hopping: float = 1.0,
		name: str = "Mixer"
	) -> qm_c.QuantumCircuit:
        """
    	Generate the fermionic mixer ansatz circuit (:math:`e^{-\gamma H_d}`) for FQAOA.

		Args:
			beta (qm_c.Parameter): The gamma parameter for the cost ansatz.
			lattice_structure (Literal["abs_max", "rms"] | None): The lattice structure of the driver Hamiltonian. 
                Available options:
                - "cyclic": Use an one-dimentional cyclic lattice
                - "ladder": Use a D-leg ladder lattice
                Defaults to "cyclic".
            hopping (float): The hopping integral.
				Defaults to 1.0.
			name (str, optional): Name of the circuit.
				Defaults to "Mixer".

		Returns:
			qm_c.QuantumCircuit: The fermionic driver ansatz circuit.
		"""
        ising = self.get_ising()
        num_qubits = ising.num_bits()

        mixer = qm_c.QuantumCircuit(num_qubits, 0, name=name)
  
        if isinstance(lattice_structure, str):
            if lattice_structure == "cyclic":
				# I(odd)
                for i in range(0, num_qubits-1, 2):
                    mixer.rxx(beta, i, i+1)
                    mixer.ryy(beta, i, i+1)
                # II(even)
                for i in range(1, num_qubits-1, 2):
                    mixer.rxx(beta, i, i+1)
                    mixer.ryy(beta, i, i+1)
                # BD
                for i in range(0, num_qubits, 2):
                    mixer.rxx(beta, i, 0)
                    mixer.ryy(beta, i, 0)
            
            elif lattice_structure == "ladder":
                D = 1
                N = 1
                # parallel direction 
                for d in range(D):
                    # I
                    for l in range(0, N-1, 2):
                        mixer.rxx(beta, l, l+1)
                        mixer.ryy(beta, l, l+1)
                    # II
                    for l in range(1, N-1, 2):
                        mixer.rxx(beta, l, l+1)
                        mixer.ryy(beta, l, l+1)
                    # BD
                    mixer.rxx(beta, N-1, 0)
                    mixer.ryy(beta, N-1, 0)
                
                # vertical direction
                
            else:
                raise ValueError(
                    f"Invalid value for lattice_structure: {lattice_structure}"
                )

        return mixer

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

    def get_fqaoa_ansatz(
		self,
		p: int,
		lattice_structure: typ.Optional[typ.Literal["cyclic", "ladder"]] = "cyclic",
		hopping: float = 1.0
  	) -> qm_c.QuantumCircuit:
        """
        Generate the FQAOA ansatz circuit.

        Args:
            p (int): Number of QAOA layers.
            lattice_structure (Literal["abs_max", "rms"] | None): The lattice structure of the driver Hamiltonian. 
                Available options:
                - "cyclic": Use an one-dimentional cyclic lattice
                - "ladder": Use a D-leg ladder lattice
                Defaults to "cyclic".
            hopping (float): The hopping integral.

        Returns:
            qm_c.QuantumCircuit: The FQAOA ansatz circuit.
        """
        ising = self.get_ising()
        num_qubits = ising.num_bits()
        fqaoa_circuit = qm_c.QuantumCircuit(num_qubits, 0, name="FQAOA")

		# Construct QAOA layers
        init = self.get_init_state()
        fqaoa_circuit.append(init)

        for _p in range(p):
            beta = qm_c.Parameter(f"beta_{_p}")
            gamma = qm_c.Parameter(f"gamma_{_p}")
            
            cost = self.get_cost_ansatz(gamma, name=f"Cost_{_p}")
            mixer = self.get_mixer_ansatz(beta, lattice_structure=lattice_structure, hopping=hopping, name=f"Mixer_{_p}")

            fqaoa_circuit.append(cost)
            fqaoa_circuit.append(mixer)

        fqaoa_circuit.update_qubits_label(self.int2varlabel)

        return fqaoa_circuit

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
