"""QFT-arithmetic building blocks for Grover Adaptive Search (GAS).

These kernels implement the "quantum dictionary" state Σ|x, f(x)> for a QUBO
objective ``f``: each polynomial term is encoded as a (controlled) phase
rotation in the Fourier basis, and a closing inverse QFT turns the accumulated
phase into a two's-complement integer in the output register. On top of that,
this module provides the GAS oracle reflection, the diffusion operator, and the
full fixed-threshold Grover kernel.

The degree-specific encoders here cover degrees 0, 1, and 2, which is all a
QUBO needs. Higher-degree (HUBO) terms are handled by
``qamomile.optimization.gas``, which builds arbitrary-arity encoders from a
factory over the same ``qft_encoding`` primitive.
"""

from __future__ import annotations

import numpy as np

import qamomile.circuit as qmc
from qamomile.circuit.frontend.operation.control_flow import for_loop


def _resolve_width(q_input: qmc.Vector[qmc.Qubit]) -> int | None:
    """Return a register's width as a Python int when it is known at trace time.

    ``Vector.shape[0]`` is a ``UInt`` handle even when the register was
    allocated from a compile-time binding, but it carries the concrete value.
    Mirrors ``qamomile.circuit.algorithm.trotter._resolve_hamiltonian_len``.

    Args:
        q_input (qmc.Vector[qmc.Qubit]): Register whose width is wanted.

    Returns:
        int | None: The width, or ``None`` when it is genuinely symbolic.

    """
    dim = q_input.shape[0]
    if isinstance(dim, int):
        return dim
    if isinstance(dim, qmc.UInt) and dim.value.is_constant():
        return int(dim.value.get_const())
    return None


def _apply_diffusion(q_input: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Apply the reflection about |0...0> to ``q_input``.

    Deliberately a plain Python function rather than inline kernel-body code:
    the DSL transformer rewrites an ``if`` inside a qkernel body into an
    ``IfOperation`` and traces *both* branches, so the multi-controlled branch
    would still build ``qmc.control(qmc.z, num_controls=0)`` for a one-qubit
    register and raise before the dead branch is ever pruned. The transformer
    never descends into a called helper, so the branch below is resolved at
    trace time and only the taken path is emitted.

    Args:
        q_input (qmc.Vector[qmc.Qubit]): Register to reflect. Consumed and
            returned.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated input register.

    """
    n = q_input.shape[0]
    if _resolve_width(q_input) == 1:
        # X Z X = -Z, so a single Z realizes the one-qubit reflection up to the
        # same global sign the X^n C^{n-1}Z X^n identity already carries for
        # n >= 2. The general path would ask for `num_controls=0`, which
        # `qmc.control` rejects, making a valid single-variable problem
        # untranspilable.
        q_input[0] = qmc.z(q_input[0])
        return q_input

    controlled_z = qmc.control(qmc.z, num_controls=n - 1)

    with for_loop(0, n, var_name="i") as i:
        q_input[i] = qmc.x(q_input[i])
    controls = q_input[0 : n - 1]  # type: ignore[misc]
    target = q_input[n - 1]  # type: ignore[misc]
    controls, target = controlled_z(controls, target)
    q_input[0 : n - 1] = controls  # type: ignore[misc]  # ReleaseSliceViewOperation — releases borrow
    q_input[n - 1] = target  # type: ignore[misc]
    with for_loop(0, n, var_name="i") as i:
        q_input[i] = qmc.x(q_input[i])
    return q_input


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
        q[i] = qmc.p(q[i], theta * (2**i))  # type: ignore[operator]
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
    via the X^n C^{n-1}Z X^n circuit identity. A single-qubit register uses a
    bare Z instead, since that identity would degenerate into a controlled gate
    with no controls.

    Args:
        q_input (qmc.Vector[qmc.Qubit]): Input register to reflect around the uniform superposition.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated input register.

    """
    return _apply_diffusion(q_input)


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
    iters: qmc.UInt = 1,  # type: ignore[assignment]
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
