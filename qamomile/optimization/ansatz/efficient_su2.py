from __future__ import annotations

import qamomile.circuit as qm_c

_SUPPORTED_ROTATION_GATES = {"rx", "ry", "rz"}
_SUPPORTED_ENTANGLEMENT = {"linear", "full", "circular", "reverse_linear"}


@qm_c.qkernel
def rx_layer(
    q: qm_c.Vector[qm_c.Qubit],
    thetas: qm_c.Vector[qm_c.Float],
    offset: qm_c.UInt,
) -> qm_c.Vector[qm_c.Qubit]:
    n = q.shape[0]
    for i in qm_c.range(n):
        q[i] = qm_c.rx(q[i], thetas[offset + i])
    return q


@qm_c.qkernel
def ry_layer(
    q: qm_c.Vector[qm_c.Qubit],
    thetas: qm_c.Vector[qm_c.Float],
    offset: qm_c.UInt,
) -> qm_c.Vector[qm_c.Qubit]:
    n = q.shape[0]
    for i in qm_c.range(n):
        q[i] = qm_c.ry(q[i], thetas[offset + i])
    return q


@qm_c.qkernel
def rz_layer(
    q: qm_c.Vector[qm_c.Qubit],
    thetas: qm_c.Vector[qm_c.Float],
    offset: qm_c.UInt,
) -> qm_c.Vector[qm_c.Qubit]:
    n = q.shape[0]
    for i in qm_c.range(n):
        q[i] = qm_c.rz(q[i], thetas[offset + i])
    return q


@qm_c.qkernel
def ry_rz_layer(
    q: qm_c.Vector[qm_c.Qubit],
    thetas: qm_c.Vector[qm_c.Float],
    offset: qm_c.UInt,
) -> qm_c.Vector[qm_c.Qubit]:
    n = q.shape[0]
    for i in qm_c.range(n):
        q[i] = qm_c.ry(q[i], thetas[offset + 2 * i])
        q[i] = qm_c.rz(q[i], thetas[offset + 2 * i + 1])
    return q


@qm_c.qkernel
def rx_rz_layer(
    q: qm_c.Vector[qm_c.Qubit],
    thetas: qm_c.Vector[qm_c.Float],
    offset: qm_c.UInt,
) -> qm_c.Vector[qm_c.Qubit]:
    n = q.shape[0]
    for i in qm_c.range(n):
        q[i] = qm_c.rx(q[i], thetas[offset + 2 * i])
        q[i] = qm_c.rz(q[i], thetas[offset + 2 * i + 1])
    return q


@qm_c.qkernel
def rx_ry_layer(
    q: qm_c.Vector[qm_c.Qubit],
    thetas: qm_c.Vector[qm_c.Float],
    offset: qm_c.UInt,
) -> qm_c.Vector[qm_c.Qubit]:
    n = q.shape[0]
    for i in qm_c.range(n):
        q[i] = qm_c.rx(q[i], thetas[offset + 2 * i])
        q[i] = qm_c.ry(q[i], thetas[offset + 2 * i + 1])
    return q


@qm_c.qkernel
def linear_entangling_layer(
    q: qm_c.Vector[qm_c.Qubit],
) -> qm_c.Vector[qm_c.Qubit]:
    n = q.shape[0]
    for i in qm_c.range(n - 1):
        q[i], q[i + 1] = qm_c.cx(q[i], q[i + 1])
    return q


@qm_c.qkernel
def full_entangling_layer(
    q: qm_c.Vector[qm_c.Qubit],
) -> qm_c.Vector[qm_c.Qubit]:
    n = q.shape[0]
    for i in qm_c.range(n):
        for j in qm_c.range(i + 1, n):
            q[i], q[j] = qm_c.cx(q[i], q[j])
    return q


@qm_c.qkernel
def circular_entangling_layer(
    q: qm_c.Vector[qm_c.Qubit],
) -> qm_c.Vector[qm_c.Qubit]:
    n = q.shape[0]
    for i in qm_c.range(n - 1):
        q[i], q[i + 1] = qm_c.cx(q[i], q[i + 1])
    if n > 1:
        q[n - 1], q[0] = qm_c.cx(q[n - 1], q[0])
    return q


@qm_c.qkernel
def reverse_linear_entangling_layer(
    q: qm_c.Vector[qm_c.Qubit],
) -> qm_c.Vector[qm_c.Qubit]:
    n = q.shape[0]
    for i in qm_c.range(n - 1, 0, -1):
        q[i], q[i - 1] = qm_c.cx(q[i], q[i - 1])
    return q


_SINGLE_ROTATION_LAYERS = {
    "rx": rx_layer,
    "ry": ry_layer,
    "rz": rz_layer,
}

_COMBINED_ROTATION_LAYERS = {
    ("ry", "rz"): (ry_rz_layer, 2),
    ("rx", "rz"): (rx_rz_layer, 2),
    ("rx", "ry"): (rx_ry_layer, 2),
}

_ENTANGLEMENT_LAYERS = {
    "linear": linear_entangling_layer,
    "full": full_entangling_layer,
    "circular": circular_entangling_layer,
    "reverse_linear": reverse_linear_entangling_layer,
}


def num_parameters(
    num_qubits: int,
    rotation_blocks: list[str] | None = None,
    skip_final_rotation_layer: bool = False,
    reps: int = 1,
) -> int:
    """Return the number of rotation parameters for Efficient SU2.

    Args:
        num_qubits: Number of qubits.
        rotation_blocks: Rotation gates per layer. Defaults to ["ry", "rz"].
        skip_final_rotation_layer: If True, omit the final rotation layer.
        reps: Number of rotation-entanglement repetitions.

    Returns:
        Number of parameters required for thetas.
    """
    if rotation_blocks is None:
        rotation_blocks = ["ry", "rz"]
    _validate_rotation_blocks(rotation_blocks)
    layers = reps if skip_final_rotation_layer else reps + 1
    return num_qubits * len(rotation_blocks) * layers


def num_entangling_gates(num_qubits: int, entanglement: str = "linear") -> int:
    """Return the number of entangling CX gates per entanglement layer."""
    return len(_entanglement_pairs(num_qubits, entanglement))


def _validate_rotation_blocks(rotation_blocks: list[str]) -> None:
    for gate in rotation_blocks:
        if gate not in _SUPPORTED_ROTATION_GATES:
            raise NotImplementedError(f"Gate {gate} not implemented")


def _entanglement_pairs(num_qubits: int, entanglement: str) -> list[tuple[int, int]]:
    if entanglement not in _SUPPORTED_ENTANGLEMENT:
        raise NotImplementedError(
            f"Entanglement type {entanglement} not implemented"
        )

    if entanglement == "linear":
        pairs = [(i, i + 1) for i in range(num_qubits - 1)]
        return [pair for pair in pairs if pair[0] != pair[1]]

    if entanglement == "full":
        pairs = [
            (i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)
        ]
        return [pair for pair in pairs if pair[0] != pair[1]]

    if entanglement == "circular":
        pairs = [(i, (i + 1) % num_qubits) for i in range(num_qubits)]
        return [pair for pair in pairs if pair[0] != pair[1]]

    # reverse_linear
    pairs = [(i, i - 1) for i in range(num_qubits - 1, 0, -1)]
    return [pair for pair in pairs if pair[0] != pair[1]]


def create_efficient_su2_circuit(
    num_qubits: int,
    rotation_blocks: list[str] | None = None,
    entanglement: str = "linear",
    skip_final_rotation_layer: bool = False,
    reps: int = 1,
) -> qm_c.QKernel:
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
        qm_c.QKernel: A QKernel implementing the Efficient SU2 ansatz.

    Raises:
        NotImplementedError: If an unsupported rotation gate or entanglement type is specified.

    Example:
        >>> kernel = create_efficient_su2_circuit(3, rotation_blocks=["rx", "ry"], entanglement="full", reps=2)
    """
    if rotation_blocks is None:
        rotation_blocks = ["ry", "rz"]

    _validate_rotation_blocks(rotation_blocks)
    if entanglement not in _ENTANGLEMENT_LAYERS:
        raise NotImplementedError(
            f"Entanglement type {entanglement} not implemented"
        )
    entanglement_layer = _ENTANGLEMENT_LAYERS[entanglement]

    def _apply_rotation_blocks(
        q: qm_c.Vector[qm_c.Qubit],
        thetas: qm_c.Vector[qm_c.Float],
        start_idx: qm_c.UInt,
    ) -> tuple[qm_c.Vector[qm_c.Qubit], qm_c.UInt]:
        blocks_key = tuple(rotation_blocks)
        if blocks_key in _COMBINED_ROTATION_LAYERS:
            layer_fn, block_size = _COMBINED_ROTATION_LAYERS[blocks_key]
            q = layer_fn(q, thetas, start_idx)
            return q, start_idx + num_qubits * block_size

        param_idx = start_idx
        for gate in rotation_blocks:
            layer_fn = _SINGLE_ROTATION_LAYERS[gate]
            q = layer_fn(q, thetas, param_idx)
            param_idx = param_idx + num_qubits
        return q, param_idx

    if skip_final_rotation_layer:

        @qm_c.qkernel
        def efficient_su2(
            q: qm_c.Vector[qm_c.Qubit],
            thetas: qm_c.Vector[qm_c.Float],
        ) -> qm_c.Vector[qm_c.Qubit]:
            param_idx = qm_c.uint(0)
            for _ in range(reps):
                q, param_idx = _apply_rotation_blocks(q, thetas, param_idx)
                q = entanglement_layer(q)
            return q

    else:

        @qm_c.qkernel
        def efficient_su2(
            q: qm_c.Vector[qm_c.Qubit],
            thetas: qm_c.Vector[qm_c.Float],
        ) -> qm_c.Vector[qm_c.Qubit]:
            param_idx = qm_c.uint(0)
            for _ in range(reps):
                q, param_idx = _apply_rotation_blocks(q, thetas, param_idx)
                q = entanglement_layer(q)
            q, param_idx = _apply_rotation_blocks(q, thetas, param_idx)
            return q

    return efficient_su2
