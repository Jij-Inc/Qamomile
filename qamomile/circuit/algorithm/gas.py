from __future__ import annotations

import numpy as np

import qamomile.circuit as qmc


@qmc.qkernel
def qft_encoding(
    q: qmc.Vector[qmc.Qubit],
    coef: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Encode a scalar coefficient as phase rotations in the QFT basis.

    Args:
        q (qmc.Vector[qmc.Qubit]): Output register represented in the Fourier basis.
        coef (qmc.Float): The coefficient to encode as a phase.

    Returns:
        qmc.Vector[qmc.Qubit]: The output register with the QFT encoding of coef.

    """
    m = q.shape[0]
    theta = 2 * np.pi * coef / (2**m)
    for i in qmc.range(m):
        q[i] = qmc.p(q[i], theta * (2**i))
    return q


@qmc.qkernel
def zero_degree_qft_encoding(
    q_output: qmc.Vector[qmc.Qubit],
    q_input: qmc.Vector[qmc.Qubit],
    coef: qmc.Float,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Apply an unconditional phase-encoding term.

    Args:
        q_output (qmc.Vector[qmc.Qubit]): Output register in the Fourier basis.
        q_input (qmc.Vector[qmc.Qubit]): Input register carried through unchanged.
        coef (qmc.Float): Coefficient of the constant term to encode.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Updated output and input registers.

    """
    q_output = qft_encoding(q_output, coef)
    return q_output, q_input


@qmc.qkernel
def first_degree_qft_encoding(
    q_output: qmc.Vector[qmc.Qubit],
    q_input: qmc.Vector[qmc.Qubit],
    control_idx: qmc.UInt,
    coef: qmc.Float,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Apply a phase-encoding term controlled by one input qubit.

    Args:
        q_output (qmc.Vector[qmc.Qubit]): Output register in the Fourier basis.
        q_input (qmc.Vector[qmc.Qubit]): Input register containing control qubits.
        control_idx (qmc.UInt): Index of the control qubit in the input register.
        coef (qmc.Float): Coefficient to encode when the control is active.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Updated output and input registers.

    """
    ctrl_qft = qmc.control(qft_encoding)
    ctrl_qubit = q_input[control_idx]
    ctrl_qubit, q_output = ctrl_qft(ctrl_qubit, q_output, coef)
    q_input[control_idx] = ctrl_qubit
    return q_output, q_input


@qmc.qkernel
def second_degree_qft_encoding(
    q_output: qmc.Vector[qmc.Qubit],
    q_input: qmc.Vector[qmc.Qubit],
    control_idx0: qmc.UInt,
    control_idx1: qmc.UInt,
    coef: qmc.Float,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Apply a phase-encoding term controlled by two input qubits.

    Args:
        q_output (qmc.Vector[qmc.Qubit]): Output register in the Fourier basis.
        q_input (qmc.Vector[qmc.Qubit]): Input register containing control qubits.
        control_idx0 (qmc.UInt): Index of the first control qubit.
        control_idx1 (qmc.UInt): Index of the second control qubit.
        coef (qmc.Float): Coefficient to encode when both controls are active.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Updated output and input registers.

    """
    ctrl_qft = qmc.control(qft_encoding, num_controls=2)
    ctrl_qubit0 = q_input[control_idx0]
    ctrl_qubit1 = q_input[control_idx1]
    ctrl_qubit0, ctrl_qubit1, q_output = ctrl_qft(
        ctrl_qubit0, ctrl_qubit1, q_output, coef
    )
    q_input[control_idx0] = ctrl_qubit0
    q_input[control_idx1] = ctrl_qubit1
    return q_output, q_input


@qmc.qkernel
def apply_function_preparation_qubo(
    q_output: qmc.Vector[qmc.Qubit],
    q_input: qmc.Vector[qmc.Qubit],
    y: qmc.Float,
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Prepare the Quantum Dictionary Σ|x,f(x)> where f is the QUBO function on the given registers.

    Args:
        q_output (qmc.Vector[qmc.Qubit]): Output register for arithmetic encoding.
        q_input (qmc.Vector[qmc.Qubit]): Input register for decision variables.
        y (qmc.Float): Objective threshold offset encoded as a constant term.
        linear (qmc.Dict[qmc.UInt, qmc.Float]): Linear coefficients indexed by variable.
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]): Quadratic coefficients indexed by variable pairs.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Updated output and input registers.

    """
    n = q_input.shape[0]
    m = q_output.shape[0]

    # Prepare superposition
    for i in qmc.range(m):
        q_output[i] = qmc.h(q_output[i])
    for i in qmc.range(n):
        q_input[i] = qmc.h(q_input[i])

    # Encode precomputed phase angles
    q_output, q_input = zero_degree_qft_encoding(q_output, q_input, y)

    for control_idx, coef in qmc.items(linear):
        q_output, q_input = first_degree_qft_encoding(
            q_output, q_input, control_idx, coef
        )

    for (ctrl0, ctrl1), coef in qmc.items(quad):
        q_output, q_input = second_degree_qft_encoding(
            q_output, q_input, ctrl0, ctrl1, coef
        )

    q_output = qmc.iqft(q_output)

    return q_output, q_input


@qmc.qkernel
def function_preparation_qubo(
    n: qmc.UInt,
    m: qmc.UInt,
    y: qmc.Float,
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Prepare the Quantum Dictionary Σ|x,f(x)> where f is the QUBO function on new registers.

    Args:
        n (qmc.UInt): Number of input qubits.
        m (qmc.UInt): Number of output qubits.
        y (qmc.Float): Objective threshold offset encoded as a constant term.
        linear (qmc.Dict[qmc.UInt, qmc.Float]): Linear coefficients indexed by variable.
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]): Quadratic coefficients indexed by variable pairs.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Prepared output and input registers.

    """
    q_output = qmc.qubit_array(m, name="q_output")
    q_input = qmc.qubit_array(n, name="q_input")
    q_output, q_input = apply_function_preparation_qubo(
        q_output, q_input, y, linear, quad
    )
    return q_output, q_input


@qmc.qkernel
def apply_function_preparation_qubo_dagger(
    q_output: qmc.Vector[qmc.Qubit],
    q_input: qmc.Vector[qmc.Qubit],
    y: qmc.Float,
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Apply the Hermitian conjugate of the state preparation ansatz.

    Args:
        q_output (qmc.Vector[qmc.Qubit]): Output register for arithmetic encoding.
        q_input (qmc.Vector[qmc.Qubit]): Input register for decision variables.
        y (qmc.Float): Objective threshold offset encoded as a constant term.
        linear (qmc.Dict[qmc.UInt, qmc.Float]): Linear coefficients indexed by variable.
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]): Quadratic coefficients indexed by variable pairs.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Updated output and input registers.

    """
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
    for i in qmc.range(n):
        q_input[i] = qmc.h(q_input[i])
    for i in qmc.range(m):
        q_output[i] = qmc.h(q_output[i])

    return q_output, q_input


@qmc.qkernel
def function_preparation_qubo_dagger(
    n: qmc.UInt,
    m: qmc.UInt,
    y: qmc.Float,
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Prepare new registers and apply the hermitian conjugate of the state preparation ansatz.

    Args:
        n (qmc.UInt): Number of input qubits.
        m (qmc.UInt): Number of output qubits.
        y (qmc.Float): Objective threshold offset encoded as a constant term.
        linear (qmc.Dict[qmc.UInt, qmc.Float]): Linear coefficients indexed by variable.
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]): Quadratic coefficients indexed by variable pairs.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Updated output and input registers.

    """
    q_output = qmc.qubit_array(m, name="q_output")
    q_input = qmc.qubit_array(n, name="q_input")
    q_output, q_input = apply_function_preparation_qubo_dagger(
        q_output, q_input, y, linear, quad
    )
    return q_output, q_input


@qmc.qkernel
def diffusion_op(
    q_input: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Apply the Grover diffusion operator on the input register.

    Implements the reflection 2|s><s| - I about the uniform superposition
    via the X^n C^{n-1}Z X^n circuit identity.

    Args:
        q_input (qmc.Vector[qmc.Qubit]): Input register to reflect around the uniform superposition.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated input register.

    """
    n = q_input.shape[0]
    controlled_z = qmc.control(qmc.z, num_controls=n - 1)

    for i in qmc.range(n):
        q_input[i] = qmc.x(q_input[i])
    controls = q_input[0 : n - 1]
    target = q_input[n - 1]
    controls, target = controlled_z(controls, target)
    q_input[0 : n - 1] = controls  # ReleaseSliceViewOperation — releases borrow
    q_input[n - 1] = target
    for i in qmc.range(n):
        q_input[i] = qmc.x(q_input[i])
    return q_input


@qmc.qkernel
def grover_operator(
    q_output: qmc.Vector[qmc.Qubit],
    q_input: qmc.Vector[qmc.Qubit],
    y: qmc.Float,
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Apply one Grover iteration for the QUBO GAS oracle.

    Args:
        q_output (qmc.Vector[qmc.Qubit]): Output register for arithmetic encoding.
        q_input (qmc.Vector[qmc.Qubit]): Input register for decision variables.
        y (qmc.Float): Objective threshold offset encoded as a constant term.
        linear (qmc.Dict[qmc.UInt, qmc.Float]): Linear coefficients indexed by variable.
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]): Quadratic coefficients indexed by variable pairs.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Updated output and input registers.

    """
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
    y: qmc.Float,
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    iters: qmc.UInt = 1,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Run repeated Grover iterations for the QUBO GAS circuit.

    Args:
        n (qmc.UInt): Number of input qubits.
        m (qmc.UInt): Number of output qubits.
        y (qmc.Float): Objective threshold offset encoded as a constant term.
        linear (qmc.Dict[qmc.UInt, qmc.Float]): Linear coefficients indexed by variable.
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]): Quadratic coefficients indexed by variable pairs.
        iters (qmc.UInt): Number of Grover iterations.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Output and input qubit registers after all iterations.

    """
    # original state
    q_output, q_input = function_preparation_qubo(n, m, y, linear, quad)

    # Apply grover operator
    for _ in qmc.range(iters):
        q_output, q_input = grover_operator(
            q_output,
            q_input,
            y=y,
            linear=linear,
            quad=quad,
        )

    return q_output, q_input
