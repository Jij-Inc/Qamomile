"""
This module implements the Fermionic QAOA (FQAOA) converter for the Qamomile framework :cite:`yoshioka2023fermionic`. 
FQAOA translates the Hamiltonians into the representation of fermion systems, 
and the equality constraint is naturally incorporated as a constant number of particles condition.

The parameterized state :math:`|\\vec{\\beta},\\vec{\\gamma}\\rangle` of :math:`p`-layer QAOA is defined as:

.. math::
    |\\vec{\\beta},\\vec{\\gamma}\\rangle = U(\\vec{\\beta},\\vec{\\gamma})|0\\rangle^{\\otimes n} = e^{-i\\beta_{p-1} H_M}e^{-i\\gamma_{p-1} H_P} \\cdots e^{-i\\beta_0 H_M}e^{-i\\gamma_0 H_P} U_{init}|0\\rangle^{\\otimes n}

where :math:`H_P` is the cost Hamiltonian, :math:`H_M` is the mixer Hamiltonian and :math:`\\gamma_l` and :math:`\\beta_l` are the variational parameters.
The :math:`2p` variational parameters are optimized classically to minimize the expectation value :math:`\\langle \\vec{\\beta},\\vec{\\gamma}|H_P|\\vec{\\beta},\\vec{\\gamma}\\rangle`.
:math:`U_{init}` prepares the initial state using Givens rotation gates :cite:`jiang2018quantum`.

This module provides functionality to convert optimization problems which written by `jijmodeling`
into FQAOA circuits (:math:`U(\\vec{\\beta}, \\vec{\\gamma})`), construct cost Hamiltonians (:math:`H_P`), and decode quantum computation results.

The `QAOAConverter` class extends the `QuantumConverter` base class, specializing in
FQAOA-specific operations such as ansatz circuit generation and result decoding.


Key Features:
	- Generation of FQAOA ansatz circuits
	- Construction of cost Hamiltonians for QAOA
	- Decoding of quantum computation results into classical optimization solutions

Note: 
	This module requires `jijmodeling` and `jijmodeling_transpiler` for problem representation.

.. bibliography::
    :filter: docname in docnames

"""

import copy
import numpy as np
import typing as typ
import qamomile.core.circuit as qm_c
import qamomile.core.operator as qm_o
from qamomile.core.ising_qubo import IsingModel
from qamomile.core.converters.converter import QuantumConverter
from qamomile.core.converters.utils import is_close_zero
import ommx.v1

class FQAOAConverter(QuantumConverter):
    """
    FQAOA (Fermionic Quantum Approximate Optimization Algorithm) converter class.

    This class provides methods to convert optimization problems into FQAOA circuits,
    construct cost Hamiltonians, and decode quantum computation results.

    Examples:

    .. code::

        from qamomile.core.qaoa.fqaoa import FQAOAConverter
        
        # Initialize with a compiled optimization problem instance 
        fqaoa_converter = FQAOAConverter(compiled_instance, num_fermion=4) 

        # Generate QAOA circuit and cost Hamiltonian
        p = 2  # Number of QAOA layers
        fqaoa_circuit = fqaoa_converter.get_ansatz_circuit(p) 
        cost_hamiltonian = fqaoa_converter.get_cost_hamiltonian()

    """
    
    def __init__(
        self,
        instance: ommx.v1.Instance,
        num_fermions: int,
        normalize_model: bool = False,
        normalize_ising: typ.Optional[typ.Literal["abs_max", "rms"]] = None
    ):
        """
        Initialize the FQAOAConverter.

        This method initializes the converter with the compiled instance of the optimization problem.

        Args:
            compiled_instance: ommx.v1.Instance.
            num_fermions (int): Number of fermions. This means the constraint :math:`M = \\sum_{l,d} x_{l,d}`.
            normalize_model (bool): The objective function and the constraints are normalized using the maximum absolute value of the coefficients contained in each.\
                Defaults to False
            normalize_ising (Literal["abs_max", "rms"] | None): The normalization method for the Ising Hamiltonian. \
                Available options:
                - "abs_max": Normalize by absolute maximum value
                - "rms": Normalize by root mean square
                Defaults to None.

        """
        super().__init__(instance)
        
        last_var = self.original_instance.decision_variables[-1]
        n, d = last_var.subscripts
        self.num_integers, self.num_bits = n+1, d+1
        self.num_fermions = num_fermions
        self.var_map = self.cyclic_mapping()
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
        ising = IsingModel.from_qubo(qubo, simplify=False)
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
        for ising_index, qubo_index in ising.index_map.items():
            deci_var = self.original_instance.get_decision_variable_by_id(qubo_index)
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
        for var in self.original_instance.decision_variables:
            # l = pos[0], d = pos[1]
            pos = var.subscripts
            cyclic_var_map[tuple(pos)] = pos[0] + self.num_integers * pos[1]
        
        return cyclic_var_map
    
    def get_fermi_orbital(self) -> np.ndarray:
        """
        Compute the single-particle wave functions of the occupied spin orbitals.
        
        Return:
			numpy.ndarray: A 2D numpy array of shape (num_fermions, num_qubits)

        """
        orbital = np.zeros((self.num_fermions, self.num_qubits))

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
                        
        return orbital
    
    def _givens_decomposition(self, fermi_orbital):
        m, n = fermi_orbital.shape
        matrix = fermi_orbital.copy()
  
	    # left unitary
        for j in reversed(range(n-m, n)):
            for i in range(m-n+j):
                # givens rotation matrix
                sin_ = -matrix[i,j] / np.sqrt(matrix[i,j]**2 + matrix[i+1,j]**2)
                cos_ = matrix[i+1,j] / np.sqrt(matrix[i,j]**2 + matrix[i+1,j]**2)
                
                # rotate
                row_1 = matrix[i].copy()
                row_2 = matrix[i+1].copy()
                matrix[i] = cos_ * row_1 + sin_ * row_2
                matrix[i+1] = -sin_ * row_1 + cos_ * row_2
        
        # right unitary
        givens_angles = []
        for i in range(m):
            for j in reversed(range(i, i+n-m)):
                # givens rotation matrix
                cos_ = matrix[i,j] / np.sqrt(matrix[i,j]**2 + matrix[i,j+1]**2)
                sin_ = -matrix[i,j+1] / np.sqrt(matrix[i,j]**2 + matrix[i,j+1]**2)
                if sin_ >= 0:
                    angle = np.arccos(cos_)
                else:
                    angle = -np.arccos(cos_)
                
                givens_angles.append([(j,j+1), angle])
                
                # rotate
                col_1 = matrix[:,j].copy()
                col_2 = matrix[:,j+1].copy()
                matrix[:,j] = np.cos(angle) * col_1 - np.sin(angle) * col_2
                matrix[:,j+1] = np.sin(angle) * col_1 + np.cos(angle) * col_2
    
        return givens_angles
    
    def givens_rotation(self, givens_angles) -> qm_c.QuantumCircuit:
        """
        Generate givens rotation gates for the initial state preparation of FQAOA.
        
        Args:
    	    givens_angles (list[tuple[int, int], float]): Parameters which represents a givens rotation of coordinates :math:`(i,j)` by angles :math:`theta`.
        Returns:
            qm_c.QuantumCircuit: A givens rotation gate.
        """
        (i,j), theta = givens_angles
        
        circuit = qm_c.QuantumCircuit(self.num_qubits, 0)
        circuit.cnot(j,i)
        circuit.cry(-2.0*theta, i, j)
        circuit.cnot(j,i)
        
        return circuit
    
    def _hopping_gate(self, i, j, beta, hopping) -> qm_c.QuantumCircuit:
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
        givens_rotations = self._givens_decomposition(unitary_rows)
            
        init = qm_c.QuantumCircuit(self.num_qubits, 0, name="init")
        
        for i in range(self.num_fermions):
            init.x(i)
            
        for givens_angles in givens_rotations:
            init.append(self.givens_rotation(givens_angles))
            
        return init

    def get_mixer_ansatz(self, beta: qm_c.Parameter, hopping: float = 1.0, name: str = "Mixer") -> qm_c.QuantumCircuit:
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
		# layer-I(odd)
        for i in range(0, self.num_qubits-1, 2):
            mixer.append(self._hopping_gate(i, i+1, beta, hopping))
            
        # layer-II(even)
        for i in range(1, self.num_qubits-1, 2):
            mixer.append(self._hopping_gate(i, i+1, beta, hopping))
            
        # BD
        mixer.append(self._hopping_gate(0, self.num_qubits-1, beta, hopping))

        return mixer

    def get_cost_ansatz(self, gamma: qm_c.Parameter, name: str = "Cost") -> qm_c.QuantumCircuit:
        """
        Generate the cost ansatz circuit (:math:`e^{-\\gamma H_P}`) for QAOA.
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

    def get_fqaoa_ansatz(self, p: int, hopping: float = 1.0) -> qm_c.QuantumCircuit:
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
