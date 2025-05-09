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

import copy
import numpy as np
import typing as typ
import qamomile.core.bitssample as qm_bs
import qamomile.core.circuit as qm_c
import qamomile.core.operator as qm_o
from qamomile.core.ising_qubo import IsingModel, qubo_to_ising
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
    
    def __init__(
        self,
        instance: ommx.v1.Instance,
        num_fermions: int,
        mixer_connectivity: typ.Optional[typ.Literal["ladder", "cyclic"]] = "cyclic",
        normalize_model: bool = False,
        normalize_ising: typ.Optional[typ.Literal["abs_max", "rms"]] = None
    ):
        super().__init__(instance)
        
        n, d = self.original_instance.decision_variables.iloc[-1]["subscripts"]
        self.num_integers, self.num_bits = n+1, d+1
        self.num_fermions = num_fermions
        
        if isinstance(mixer_connectivity, str):
            if mixer_connectivity == "cyclic":
                self.connectivity = mixer_connectivity
                self.var_map = self.cyclic_mapping()
            elif mixer_connectivity == "ladder":
                self.connectivity = mixer_connectivity
                self.var_map = self.fermion_mapping()
        else:
            raise ValueError(f"Invalid value for connectivity: {mixer_connectivity}")
        
        self.ising = self.get_ising()
        self.num_qubits = self.ising.num_bits()
        
    def instance_to_qubo(self) -> tuple[dict[tuple[int, int], float], float]:
        """
        Convert the instance to QUBO format.

        This method converts the optimization problem instance into a QUBO (Quadratic Unconstrained Binary Optimization)
        representation, which is suitable for quantum computation.

        Returns:
            tuple[dict[int, float], float]: A tuple containing the QUBO dictionary and the constant term.

        """
        instance_copy = copy.deepcopy(self.original_instance)
        qubo, constant = instance_copy.to_qubo(uniform_penalty_weight=0.0)
        return qubo, constant

    def get_ising(self) -> IsingModel:
        """
        Get the Ising model representation of the problem.

        Returns:
            IsingModel: The Ising model representation.

        """
        if self._ising is None:
            self._ising = self.ising_encode()
        return self._ising
        
    def cyclic_mapping(self) -> dict[int, tuple[int, int]]:
        """
        Get variable maps between decision variable indices (l,d) and qubit index i.
        
        Return:
			dict[int, tuple[int, int]] : A variable map for ring driver.
        """
        cyclic_var_map = {}
        for id, pos in self.original_instance.decision_variables.subscripts.items():
            # l = pos[0], d = pos[1]
            i = pos[0] + self.num_integers * pos[1] 
            cyclic_var_map[i] = pos
        
        return cyclic_var_map
    
    def ladder_mapping(self) -> dict[int, tuple[int, int]]:
        """
        Get variable maps between decision variable indices (l,d) and qubit index i.
        
        Return:
			dict[int, tuple[int, int]] : A variable map for ladder driver.
        """
        fermi_var_map = {}
        for id, pos in self.original_instance.decision_variables.subscripts.items():
            # l = pos[0], d = pos[1]
            i = (-1)**(pos[1]) * pos[0] + self.num_integers * pos[1] + (1-(-1)**pos[1]) * (self.num_integers-1) / 2 
            fermi_var_map[i] = pos
        
        return fermi_var_map
    
    def get_fermi_orbital(self):
        """
        Compute the single-particle wave functions of the occupied spin orbitals.
        
        Return:
			nd.array: A 2D numpy array of shape (num_fermions, num_qubits)

        """
        orbital = np.zeros((self.num_fermions, self.num_qubits))
        
        if self.connectivity == "cyclic":
            if self.num_fermions % 2 == 0: # num_fermion is even
                for i in range(self.num_qubits):
                    for k in range(int(self.num_fermions/2)):
                        angle = 2.0 * np.pi * (k+0.5) * (i+1) / self.num_qubits
                        orbital[k, i] = np.sqrt(2.0/self.num_qubits) * np.sin(angle)
                        orbital[int(self.num_fermions-1-k), i] = np.sqrt(2.0/self.num_qubits) * np.cos(angle)
            else: # num_fermion is odd
                for i in range(self.num_qubits):
                    orbital[0, i] = np.sqrt(1.0/self.num_qubits)
                    for k in range(int(self.num_fermions/2)):
                        angle = 2.0 * np.pi * (k+1) * (i+1) / self.num_qubits
                        orbital[k, i] = np.sqrt(2.0/self.num_qubits) * np.sin(angle)
                        orbital[int(self.num_fermions-1-k), i] = np.sqrt(2.0/self.num_qubits) * np.cos(angle)
                
        else:
            # 要修正
            # (k,m)
            for i, pos_i in self.var_map.items():
                if pos_i[0] == (int(self.num_qubits*0.5-1) or int(self.num_qubits-1)):
                    coef = np.sqrt(2/((self.num_bits+1)*self.num_integers))
                else:
                    coef = np.sqrt(4/((self.num_bits+1)*self.num_integers))
                
                # (l,d)
                for j, pos_j in self.var_map.items():
                    if pos_i[0] < int(self.num_qubits*0.5-1):
                        orbital[i,j] = coef * np.sin(2*np.pi*pos_i[0]*pos_j[0]/self.num_integers) * np.sin(np.pi*pos_i[1]*pos_j[1]/(self.num_bits+1))
                    else:
                        orbital[i,j] = coef * np.cos(2*np.pi*pos_i[0]*pos_j[0]/self.num_integers) * np.sin(np.pi*pos_i[1]*pos_j[1]/(self.num_bits+1))
                        
        return orbital
    
    def givens_rotation(self, givens_angles, gate_id):
        i, j, theta, varphi = givens_angles
        
        circuit = qm_c.QuantumCircuit(self.num_qubits, 0, name=f"g_{gate_id}")
        circuit.cnot(j,i)
        circuit.cry(-2.0*theta, i, j)
        circuit.cnot(j,i)
        circuit.rz(varphi, j)
        
        return circuit
        
    def fermion_swap_gate(self, i):
        circuit = qm_c.QuantumCircuit(self.num_qubits, 0, name="FSWAP")
        circuit.swap(i,i+1)
        circuit.h(i+1)
        circuit.cnot(i,i+1)
        circuit.h(i+1)
        
        return circuit
    
    def get_init_state(self):
        unitary_rows = self.get_fermi_orbital()
        givens_rotations = givens_decomposition(unitary_rows)
            
        init = qm_c.QuantumCircuit(self.num_qubits, 0, name="init")
        
        for i in range(self.num_fermions):
            init.x(i)
            
        givens_rot_list = []
        for givens_rot_paralell in reversed(givens_rotations[0]):
            for givens_rot in reversed(givens_rot_paralell):
                givens_rot_list.append(givens_rot)
        
        i = 0
        for givens_angles in givens_rot_list:
            init.append(self.givens_rotation(givens_angles, int(len(givens_rot_list)-i)))
            i += 1
            
        return init

    def get_mixer_ansatz(
		self, beta: qm_c.Parameter, hopping: float = 1.0, name: str = "Mixer"
	) -> qm_c.QuantumCircuit:
        """
    	Generate the fermionic mixer ansatz circuit (:math:`e^{-\gamma H_d}`) for FQAOA.

		Args:
			beta (qm_c.Parameter): The gamma parameter for the cost ansatz.
   			hopping (float): The hopping integral. Defaults to 1.0.
			name (str, optional): Name of the circuit. Defaults to "Mixer".

		Returns:
			qm_c.QuantumCircuit: The fermionic driver ansatz circuit.
		"""

        mixer = qm_c.QuantumCircuit(self.num_qubits, 0, name=name)
  
        # 1d-cyclic
        if self.connectivity == "cyclic":
			# layer-I(odd)
            for i in range(0, self.num_qubits-1, 2):
                mixer.rx(-1.0*np.pi/2, i)
                mixer.rx(np.pi/2, i+1)
                mixer.cnot(i, i+1)
                mixer.rx(-1.0*beta*hopping, i)
                mixer.rz(beta*hopping, i+1)
                mixer.cnot(i, i+1)
                mixer.rx(np.pi/2, i)
                mixer.rx(-1.0*np.pi/2, i+1)
                
            # layer-II(even)
            for i in range(1, self.num_qubits-1, 2):
                mixer.rx(-1.0*np.pi/2, i)
                mixer.rx(np.pi/2, i+1)
                mixer.cnot(i, i+1)
                mixer.rx(-1.0*beta*hopping, i)
                mixer.rz(beta*hopping, i+1)
                mixer.cnot(i, i+1)
                mixer.rx(np.pi/2, i)
                mixer.rx(-1.0*np.pi/2, i+1)
                
            # BD
            mixer.rx(-1.0*np.pi/2, 0)
            mixer.rx(np.pi/2, self.num_qubits-1)
            mixer.cnot(0, self.num_qubits-1)
            mixer.rx(-1.0*beta*hopping, 0)
            mixer.rz(beta*hopping, self.num_qubits-1)
            mixer.cnot(0, self.num_qubits-1)
            mixer.rx(np.pi/2, 0)
            mixer.rx(-1.0*np.pi/2, self.num_qubits-1)
            
        # 2d-ladder
        else:
            """
            # parallel
            for i in range(self.num_qubits):
                # layer-I
                if [(x % 2 == 0) for x in self.var_map[i]] in ([True, True], [False, False]):
                    if self.var_map[i][0] in (0,N-1): # without BD
                        mixer.rxx(beta*hopping, i, i+1)
                        mixer.ryy(beta*hopping, i, i+1)
                # layer-II
                else:
                    if self.var_map[i][0] in (0,N-1): #without BD
                        mixer.rxx(beta*hopping, i, i+1)
                        mixer.ryy(beta*hopping, i, i+1)
            # vertical
            for i in range(self.num_qubits): # VB
                # F_I
                if [(x % 2 == 0) for x in self.var_map[i]] in ([True, True], [False, False]):
                    if self.var_map[i][0] in (0,N-1):
                        mixer.append(self.fermion_swap_gate(i))
                # F_II
                else:
                    if self.var_map[i][0] in (0,N-1):
                        mixer.append(self.fermion_swap_gate(i))
                # hopping
                for 
            for i in range(D): #BD
            for i in range(self.num_qubits): # VB
            """
            raise ValueError(f"Not implements")

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
        cost = qm_c.QuantumCircuit(self.num_qubits, 0, name=name)

        # Apply RZ gates for linear terms
        for i, hi in self.ising.linear.items():
            if not is_close_zero(hi):
                cost.rz(2 * hi * gamma, i)

        # Apply CNOT and RZ gates for quadratic terms
        for (i, j), Jij in self.ising.quad.items():
            if not is_close_zero(Jij):
                cost.rzz(2 * Jij * gamma, i, j)

        cost.update_qubits_label(self.int2varlabel)

        return cost

    def get_fqaoa_ansatz(
		self, p: int, hopping: float = 1.0
  	) -> qm_c.QuantumCircuit:
        """
        Generate the FQAOA ansatz circuit.

        Args:
            p (int): Number of QAOA layers.
            hopping (float): The hopping integral.

        Returns:
            qm_c.QuantumCircuit: The FQAOA ansatz circuit.
        """

        fqaoa_circuit = qm_c.QuantumCircuit(self.num_qubits, 0, name="FQAOA")

		# Construct QAOA layers
        init = self.get_init_state()
        fqaoa_circuit.append(init)

        for _p in range(p):
            beta = qm_c.Parameter(f"beta_{_p}")
            gamma = qm_c.Parameter(f"gamma_{_p}")
            
            cost = self.get_cost_ansatz(gamma, name=f"Cost_{_p}")
            mixer = self.get_mixer_ansatz(beta, hopping=hopping, name=f"Mixer_{_p}")

            fqaoa_circuit.append(cost)
            fqaoa_circuit.append(mixer)

        fqaoa_circuit.update_qubits_label(self.int2varlabel)

        return fqaoa_circuit

    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        """
        Construct the cost Hamiltonian for FQAOA.

        Returns:
            qm_o.Hamiltonian: The cost Hamiltonian.
        """
        hamiltonian = qm_o.Hamiltonian()

        # Add linear terms
        for i, hi in self.ising.linear.items():
            if not is_close_zero(hi):
                hamiltonian.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, i),), hi)

        # Add quadratic terms
        for (i, j), Jij in self.ising.quad.items():
            if not is_close_zero(Jij):
                hamiltonian.add_term(
                    (
                        qm_o.PauliOperator(qm_o.Pauli.Z, i),
                        qm_o.PauliOperator(qm_o.Pauli.Z, j),
                    ),
                    Jij,
                )

        return hamiltonian
