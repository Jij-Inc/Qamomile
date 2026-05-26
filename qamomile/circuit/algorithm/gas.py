from __future__ import annotations

import qamomile.circuit as qmc
import numpy as np

@qmc.qkernel
def zero_degree_qft_encoding(
    q_output: qmc.Vector[qmc.Qubit],
    q_input: qmc.Vector[qmc.Qubit],
    coef: qmc.Float,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Apply a phase gate without any control qubits."""
    m = q_output.shape[0]
    for i in range(m):
        qmc.p(q_output[i], (2 ** i) * 2*np.pi*coef/(2**m))
    return q_output, q_input

@qmc.qkernel
def first_degree_qft_encoding(
    q_output: qmc.Vector[qmc.Qubit],
    q_input: qmc.Vector[qmc.Qubit],
    control_idx: qmc.UInt,
    coef: qmc.Float,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Apply a phase gate controlled by a single control qubit."""
    m = q_output.shape[0]
    mcp_phase = qmc.controlled(qmc.p, num_controls=1)
    for i in range(m):
        mcp_phase(q_input[control_idx], q_output[i], theta= (2 ** i) * 2*np.pi*coef/(2**m) )
    return q_output, q_input

@qmc.qkernel
def second_degree_qft_encoding(
    q_output: qmc.Vector[qmc.Qubit],
    q_input: qmc.Vector[qmc.Qubit],
    control_idx0: qmc.UInt,
    control_idx1: qmc.UInt,
    coef: qmc.Float,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Apply a phase gate controlled by two control qubits."""
    m = q_output.shape[0]
    mcp_phase = qmc.controlled(qmc.p, num_controls=2)
    for i in range(m):
        mcp_phase(q_input[control_idx0], q_input[control_idx1], q_output[i], theta= (2 ** i) * 2*np.pi*coef/(2**m) )
    return q_output, q_input

@qmc.qkernel
def apply_function_preparation_qubo(
    q_output: qmc.Vector[qmc.Qubit],
    q_input: qmc.Vector[qmc.Qubit],
    y: qmc.UInt,
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    
    n = q_input.shape[0]
    m = q_output.shape[0]

        # Prepare superposition
    for i in range(m):
        q_output[i] = qmc.h(q_output[i])
    for i in range(n):
        q_input[i] = qmc.h(q_input[i])

    # Encode precomputed phase angles
    q_output, q_input = zero_degree_qft_encoding(q_output, q_input, y)

    for control_idx, coef in qmc.items(linear):
        q_output, q_input = first_degree_qft_encoding(q_output, q_input, control_idx, coef)
    
    for (ctrl0, ctrl1), coef in quad.items():
        q_output, q_input = second_degree_qft_encoding(q_output, q_input, ctrl0, ctrl1, coef)
    
    q_output = qmc.iqft(q_output)

    return q_output, q_input

@qmc.qkernel
def function_preparation_qubo(
    n: qmc.UInt,
    m: qmc.UInt,
    y: qmc.UInt,
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Prepare a quantum state that encodes a function defined by phase angles."""
    q_output = qmc.qubit_array(m, name="q_output")
    q_input = qmc.qubit_array(n, name="q_input")
    
    q_output, q_input = apply_function_preparation_qubo(q_output, q_input, y, linear, quad)

    return q_output, q_input

@qmc.qkernel
def apply_function_preparation_qubo_dagger(
    q_output: qmc.Vector[qmc.Qubit],
    q_input: qmc.Vector[qmc.Qubit],
    y: qmc.UInt,
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Apply the Hermitian conjugate of function_preparation_qubo on existing registers."""
    # Reverse of final iqft in function_preparation_qubo
    q_output = qmc.qft(q_output)

    # Reverse controlled-phase encodings with opposite angles
    for (ctrl0, ctrl1), coef in qmc.items(quad):
        q_output, q_input = second_degree_qft_encoding(
            q_output, q_input, ctrl0, ctrl1, (-1.0) * coef
        )
    for control_idx, coef in qmc.items(linear):
        q_output, q_input = first_degree_qft_encoding(
            q_output, q_input, control_idx, (-1.0) * coef
        )
    q_output, q_input = zero_degree_qft_encoding(q_output, q_input, (-1.0) * y)

    # Reverse of initial Hadamards
    n = q_input.shape[0]
    m = q_output.shape[0]
    for i in range(n):
        q_input[i] = qmc.h(q_input[i])
    for i in range(m):
        q_output[i] = qmc.h(q_output[i])

    return q_output, q_input


@qmc.qkernel
def function_preparation_qubo_dagger(
    n: qmc.UInt,
    m: qmc.UInt,
    y: qmc.UInt,
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Hermitian conjugate of function_preparation_qubo."""
    q_output = qmc.qubit_array(m, name="q_output")
    q_input = qmc.qubit_array(n, name="q_input")
    q_output, q_input = apply_function_preparation_qubo_dagger(
        q_output, q_input, y, linear, quad
    )
    return q_output, q_input

@qmc.qkernel
def diffusion_op(
    q_input: qmc.Vector[qmc.Qubit]
) -> qmc.Vector[qmc.Qubit]:
    """Apply a diffusion operator on the input register."""
    
    n = q_input.shape[0]
    controlled_z = qmc.controlled(qmc.z, num_controls=n-1)

    for i in range(n):
        q_input[i] = qmc.x(q_input[i])
    controls = q_input[0:n-1]
    target = q_input[n-1]
    controlled_z(controls, target)
    for i in range(n):
        q_input[i] = qmc.x(q_input[i])
    return q_input

@qmc.qkernel
def grover_operator(
    q_output: qmc.Vector[qmc.Qubit],
    q_input: qmc.Vector[qmc.Qubit],
    y: qmc.UInt,
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Apply U then U dagger on the same registers and measure."""

    m = q_output.shape[0]

    # Oracle
    q = q_output[m - 1]
    q = qmc.z(q)
    q_output[m - 1] = q

    # A_y^dagger
    q_output, q_input = apply_function_preparation_qubo_dagger(
        q_output, q_input, y, linear, quad
    )

    # Diffusion
    q_input = diffusion_op(q_input)

    # A_y
    q_output, q_input = apply_function_preparation_qubo(
        q_output, q_input, y, linear, quad
    )

    return q_output, q_input

@qmc.qkernel
def grover_algorithm(
    n: qmc.UInt,
    m: qmc.UInt,
    y: qmc.UInt,
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    iters: qmc.UInt = 1
 ) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    # original state
    q_output, q_input = function_preparation_qubo(n, m, y, linear, quad)

    # Apply grover operator
    for _ in range(iters):
        q_output, q_input = grover_operator(
            q_output,
            q_input,
            y=y,
            linear=linear,
            quad=quad,
        )

    return q_output, q_input