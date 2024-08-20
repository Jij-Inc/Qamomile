import qamomile.core.circuit as qm_c

class EfficientSU2:

    def __init__(self,num_qubits:int, rotation_blocks:list[str] = None, entanglement:str = 'linear',skip_final_rotation_layer:bool= False,reps:int = 1) -> None:
        self.circuit = None
        self._num_qubits = num_qubits
        self._reps = reps
        self._rotation_blocks = rotation_blocks
        self._entanglement = entanglement
        self._skip_final_rotation_layer = skip_final_rotation_layer
        self._num_params = 0

        if self._rotation_blocks is None:
            self._rotation_blocks = ["ry","rz"]


    def _add_rotation_blocks(self):
        for gate in self._rotation_blocks:
            for i in range(self._num_qubits):
                param = qm_c.Parameter(f"theta_{self._num_params}")
                self._num_params += 1
                if gate == "ry":
                    self.circuit.ry(param,i)
                elif gate == "rz":
                    self.circuit.rz(param,i)
                elif gate == "rx":
                    self.circuit.rx(param,i)
                else:
                    raise NotImplementedError("Gate not implemented")


    def _add_entanglement_blocks(self):
        if self._entanglement == 'linear':
            for i in range(self._num_qubits - 1):
                self.circuit.cx(i,i+1)
        else:
            raise NotImplementedError("Entanglement type not implemented")
        
        

    def build_circuit(self):
        self.circuit = qm_c.QuantumCircuit(self._num_qubits, 0, name="TwoLocal")
        
        for _ in range(self._reps):
            self._add_rotation_blocks()
            self._add_entanglement_blocks()
            
        if not self._skip_final_rotation_layer:
            self._add_rotation_blocks()
            
    def get_ansatz(self) -> qm_c.QuantumCircuit:
        if self.circuit is None:
            self.build_circuit()

        return self.circuit
            