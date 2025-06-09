"""
This module implements the Fermionic QAOA (FQAOA) converter for the Qamomile framework :cite:`yoshioka2023fermionic`. 
FQAOA translates the Hamiltonians into the representation of fermion systems, 
and the equality constraint is naturally incorporated as a constant number of particles condition.

The parameterized state :math:`|\\vec{\\beta},\\vec{\gamma}\\rangle` of :math:`p`-layer QAOA is defined as:

.. math::
    |\\vec{\\beta},\\vec{\gamma}\\rangle = U(\\vec{\\beta},\\vec{\gamma})|0\\rangle^{\otimes n} = e^{-i\\beta_{p-1} H_M}e^{-i\gamma_{p-1} H_P} \cdots e^{-i\\beta_0 H_M}e^{-i\gamma_0 H_P} U_{init}|0\\rangle^{\otimes n}

where :math:`H_P` is the cost Hamiltonian, :math:`H_M` is the mixer Hamiltonian and :math:`\gamma_l` and :math:`\\beta_l` are the variational parameters.
The :math:`2p` variational parameters are optimized classically to minimize the expectation value :math:`\langle \\vec{\\beta},\\vec{\gamma}|H_P|\\vec{\\beta},\\vec{\gamma}\\rangle`.
:math:`U_{init}` prepares the initial state using Givens rotation gates :cite:`jiang2018quantum`.

This module provides functionality to convert optimization problems which written by `jijmodeling`
into FQAOA circuits (:math:`U(\\vec{\\beta}, \\vec{\gamma})`), construct cost Hamiltonians (:math:`H_P`), and decode quantum computation results.

The `QAOAConverter` class extends the `QuantumConverter` base class, specializing in
FQAOA-specific operations such as ansatz circuit generation and result decoding.


Key Features:
	- Generation of FQAOA ansatz circuits
	- Construction of cost Hamiltonians for QAOA
	- Decoding of quantum computation results into classical optimization solutions

Note: 
	This module requires `jijmodeling` and `jijmodeling_transpiler` for problem representation and `openfermion` for preparing a Givens rotation gate.

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

# lazy import
try:
    from openfermion.linalg import givens_decomposition
except ImportError:
    raise ImportError("To use the FQAOAConverter class, 'openfermion' is required. \n Installation method: 'pip install openfermion'")

class FQAOAConverter(QuantumConverter):
    """
    FQAOA (Fermionic Quantum Approximate Optimization Algorithm) converter class.

    This class provides methods to convert optimization problems into FQAOA circuits,
    construct cost Hamiltonians, and decode quantum computation results.

    Examples:

    .. code::

        from qamomile.core.qaoa.fqaoa import FQAOAConverter
        
        # Initialize with a compiled optimization problem instance 
        fqaoa_converter = FQAOAConverter(compiled_instance, num_fermion=4, mixer_connectivity='cyclic') 

        # Generate QAOA circuit and cost Hamiltonian
        p = 2  # Number of QAOA layers
        fqaoa_circuit = fqaoa_converter.get_ansatz_circuit(p) 
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
        """
        Initialize the FQAOAConverter.

        This method initializes the converter with the compiled instance of the optimization problem.

        Args:
            compiled_instance: ommx.v1.Instance.
            num_fermions (int): Number of fermions. This means the constraint :math:`M = \\sum_{l,d} x_{l,d}`.
            mixer_connectivity (Literal["ladder", "cyclic"] | 'cyclic'): The lattice structure of mixer Hamiltonian.\
				Available options:
				- 'ladder': two-dimensional Ladder lattice
				- 'cyclic': one-dimensional cyclic lattice
				Defaults to 'cyclic'.
            normalize_model (bool): The objective function and the constraints are normalized using the maximum absolute value of the coefficients contained in each.\
                Defaults to False
            normalize_ising (Literal["abs_max", "rms"] | None): The normalization method for the Ising Hamiltonian. \
                Available options:
                - "abs_max": Normalize by absolute maximum value
                - "rms": Normalize by root mean square
                Defaults to None.

        """
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
                self.var_map = self.ladder_mapping()
        else:
            raise ValueError(f"Invalid value for connectivity: {mixer_connectivity}")
        
        self.ising = self.fqaoa_get_ising()
        self.num_qubits = self.ising.num_bits()
        
    def fqaoa_instance_to_qubo(self) -> tuple[dict[tuple[int, int], float], float]:
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

    def fqaoa_get_ising(self) -> IsingModel:
        """
        Get the Ising model representation of the problem.

        Returns:
            IsingModel: The Ising model representation.

        """
        if self._ising is None:
            self._ising = self.fqaoa_ising_encode()
        return self._ising

    def fqaoa_ising_encode(self) -> IsingModel:
        qubo, constant = self.fqaoa_instance_to_qubo()
        ising = qubo_to_ising(qubo, simplify=False)
        ising.constant += constant

		# normalize
        if isinstance(self.normalize_ising, str):
            if self.normalize_ising == "abs_max":
                ising.normalize_by_abs_max()
            elif self.normalize_ising == "rms":
                ising.normalize_by_rms()
            else:
                raise ValueError(
                    f"Invalid value for normalize_ising: {self.normalize_ising}"
                )

		# index labeling
        deci_vars = {dv.id: dv for dv in self.original_instance.raw.decision_variables}
        
        for ising_index, qubo_index in ising.index_map.items():
            deci_var = deci_vars[qubo_index]
            var_name = deci_var.name
            subscripts = tuple(deci_var.subscripts)
            
            fermionic_index = self.var_map[subscripts]
            self.int2varlabel[fermionic_index]  = var_name + "_{" + ",".join(map(str, subscripts)) + "}"

        return ising
        
    def cyclic_mapping(self) -> dict[tuple[int, int], int]:
        """
        Get variable maps between decision variable indices :math:`(l,d)` and qubit index :math:`i`.
        
        Return:
			dict[tuple[int, int], int] : A variable map for ring driver.
        """
        cyclic_var_map = {}
        for id, pos in self.original_instance.decision_variables.subscripts.items():
            # l = pos[0], d = pos[1]
            cyclic_var_map[tuple(pos)] = pos[0] + self.num_integers * pos[1]
        
        return cyclic_var_map
    
    def ladder_mapping(self) -> dict[tuple[int, int], int]:
        """
        Get variable maps between decision variable indices :math:`(l,d)` and qubit index :math:`i`.
        
        Return:
			dict[tuple[int, int], int] : A variable map for ladder driver.
        """
        fermi_var_map = {}
        for id, pos in self.original_instance.decision_variables.subscripts.items():
            # l = pos[0], d = pos[1]
            fermi_var_map[tuple(pos)] = (-1)**(pos[1]) * pos[0] + self.num_integers * pos[1] + (1-(-1)**pos[1]) * (self.num_integers-1) / 2
        
        return fermi_var_map
    
    def get_fermi_orbital(self) -> np.ndarray:
        """
        Compute the single-particle wave functions of the occupied spin orbitals.
        
        Return:
			numpy.ndarray: A 2D numpy array of shape (num_fermions, num_qubits)

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
    
    def givens_rotation(self, givens_angles: tuple[int, int, float, float], gate_id: int) -> qm_c.QuantumCircuit:
        """
        Generate givens rotation gates for the initial state preparation of FQAOA.
        
        Args:
    	    givens_angles (tuple[int, int, float, float]): Parameters which represents a givens rotation of coordinates :math:`(i,j)` by angles :math:`(\\theta, \\varphi)`.
    	    gate_id (int): The index of givens rotations gate.
        Returns:
            qm_c.QuantumCircuit: The :math:`i`-th givens rotation gate.
        """
        i, j, theta, varphi = givens_angles
        
        circuit = qm_c.QuantumCircuit(self.num_qubits, 0, name=f"g_{gate_id}")
        circuit.cnot(j,i)
        circuit.cry(-2.0*theta, i, j)
        circuit.cnot(j,i)
        circuit.rz(varphi, j)
        
        return circuit
        
    def fermion_swap_gate(self, i) -> qm_c.QuantumCircuit:
        circuit = qm_c.QuantumCircuit(self.num_qubits, 0, name="FSWAP")
        circuit.swap(i,i+1)
        circuit.h(i+1)
        circuit.cnot(i,i+1)
        circuit.h(i+1)
        
        return circuit
    
    def hopping_gate(self, i, j, beta, hopping) -> qm_c.QuantumCircuit:
        circuit = qm_c.QuantumCircuit(self.num_qubits, 0, name="hopping")
        
        circuit.rx(-0.5*np.pi, i)
        circuit.rx(0.5*np.pi, j)
        circuit.cnot(i, j)
        circuit.rx(-1.0*beta*hopping, i)
        circuit.rz(beta*hopping, j)
        circuit.cnot(i, j)
        circuit.rx(0.5*np.pi, i)
        circuit.rx(-0.5*np.pi, j)
        
        return circuit
    
    def get_init_state(self) -> qm_c.QuantumCircuit:
        """
        Generate the initial state preparation for FQAOA.
        """
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
        Generate the fermionic mixer ansatz circuit (:math:`e^{-\\beta H_d}`) for FQAOA.
        
        Args:
        	beta (qm_c.Parameter): The beta parameter for the mixer ansatz.
         	hopping (float): The hopping integral. Defaults to 1.0.
          	name (str, optional): Name of the circuit. Defaults to "Mixer".
        
        Returns:
        	qm_c.QuantumCircuit: The fermionic mixer ansatz circuit.
		"""

        mixer = qm_c.QuantumCircuit(self.num_qubits, 0, name=name)
  
        # 1d-cyclic
        if self.connectivity == "cyclic":
			# layer-I(odd)
            for i in range(0, self.num_qubits-1, 2):
                mixer.append(self.hopping_gate(i, i+1, beta, hopping))
                
            # layer-II(even)
            for i in range(1, self.num_qubits-1, 2):
                mixer.append(self.hopping_gate(i, i+1, beta, hopping))
                
            # BD
            mixer.append(self.hopping_gate(0, self.num_qubits-1, beta, hopping))
            
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
        Construct the cost Hamiltonian (:math:`H_P`) for FQAOA.

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
