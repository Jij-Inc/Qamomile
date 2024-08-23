import qamomile.core.circuit as qm_c


def create_efficient_su2_circuit(
    num_qubits: int,
    rotation_blocks: list[str] | None = None,
    entanglement: str = "linear",
    skip_final_rotation_layer: bool = False,
    reps: int = 1,
) -> qm_c.QuantumCircuit:
    """Creates an Efficient SU2 variational quantum circuit.

    This function generates a parameterized quantum circuit based on the Efficient SU2 ansatz.
    The circuit consists of alternating layers of rotation gates and entanglement operations.

    Args:
        num_qubits (int): The number of qubits in the circuit.
        rotation_blocks (list[str] | None): A list of rotation gates to apply in each rotation layer.
            If None, defaults to ["ry", "rz"].
        entanglement (str): The type of entanglement to apply. Options are 'linear', 'full',
            'circular', or 'reverse_linear'.
        skip_final_rotation_layer (bool): If True, skips the final rotation layer.
        reps (int): The number of repetitions of the rotation-entanglement block.

    Returns:
        qm_c.QuantumCircuit: The constructed Efficient SU2 variational quantum circuit.

    Raises:
        NotImplementedError: If an unsupported rotation gate or entanglement type is specified.

    Example:
        >>> circuit = create_efficient_su2_circuit(3, rotation_blocks=["rx", "ry"], entanglement="full", reps=2)
    """
    if rotation_blocks is None:
        rotation_blocks = ["ry", "rz"]

    circuit = qm_c.QuantumCircuit(num_qubits, 0, name="TwoLocal")

    def add_rotation_blocks(
        circuit: qm_c.QuantumCircuit, num_params: int
    ) -> tuple[qm_c.QuantumCircuit, int]:
        for gate in rotation_blocks:
            for i in range(num_qubits):
                param = qm_c.Parameter(f"theta_{num_params}")
                num_params += 1
                if gate == "ry":
                    circuit.ry(param, i)
                elif gate == "rz":
                    circuit.rz(param, i)
                elif gate == "rx":
                    circuit.rx(param, i)
                else:
                    raise NotImplementedError(f"Gate {gate} not implemented")
        return circuit, num_params

    def add_entanglement_blocks(circuit: qm_c.QuantumCircuit) -> qm_c.QuantumCircuit:
        if entanglement == "linear":
            for i in range(num_qubits - 1):
                circuit.cx(i, i + 1)

        elif entanglement == "full":
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    circuit.cx(i, j)

        elif entanglement == "circular":
            for i in range(num_qubits):
                circuit.cx(i, (i + 1) % num_qubits)

        elif entanglement == "reverse_linear":
            for i in range(num_qubits - 1, 0, -1):
                circuit.cx(i, i - 1)
        else:
            raise NotImplementedError(
                f"Entanglement type {entanglement} not implemented"
            )
        return circuit

    num_params = 0
    for _ in range(reps):
        circuit, num_params = add_rotation_blocks(circuit, num_params)
        circuit = add_entanglement_blocks(circuit)

    if not skip_final_rotation_layer:
        circuit, num_params = add_rotation_blocks(circuit, num_params)

    return circuit
