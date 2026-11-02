"""Small shared semantic witnesses executed by every backend."""

from __future__ import annotations

from math import pi, sqrt

import qamomile as qm
from qamomile.stdlib import (
    BlockEncoding,
    Region,
    Response,
    Signal,
    add_constant,
    add_modulo,
    add_registers,
    amplify,
    amplitude_encoding,
    compare_unsigned,
    interfere,
    multiply_modulo_constant,
    response,
    sample,
    select,
    turn,
)

_add_two = add_registers(width=2)
_add_one_wrapping = add_constant(width=2, value=1)
_multiply_two_mod_three = multiply_modulo_constant(width=2, value=2, modulus=3)
_add_modulo_one_bit = add_modulo(width=1)
_compare_one_bit = compare_unsigned(width=1)


@qm.qkernel
def standard_optimization_state() -> qm.Q[qm.Bits[2]]:
    """Exercise exact zero-angle and dependency-disjoint cancellation."""

    state = qm.qubits(2)
    state = qm.rx(state, 0.0, target=0)
    state = qm.x(state, target=0)
    state = qm.h(state, target=1)
    state = qm.x(state, target=0)
    return qm.h(state, target=1)


@qm.qkernel
def standard_optimization_sample() -> qm.Bits[2]:
    """Sample the standard exact-optimization witness."""

    return qm.measure(standard_optimization_state())


@qm.qkernel
def realization_add_state() -> qm.Q[qm.Bits[4]]:
    """Add one to three, exposing wraparound and asymmetric bit order."""
    state = qm.x(qm.qubits(4), target=0)
    state = qm.x(state, target=2)
    state = qm.x(state, target=3)
    return _add_two(state)


@qm.qkernel
def realization_add_sample() -> qm.Bits[4]:
    """Sample the wrapping addition witness."""
    return qm.measure(realization_add_state())


@qm.qkernel
def _reference_increment(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    """Increment two bits with one CNOT followed by one X."""
    state = qm.cx(state, control=0, target=1)
    return qm.x(state, target=0)


@qm.qkernel
def _reference_high_flip(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    """Add two to a two-bit target."""
    return qm.x(state, target=1)


@qm.qkernel
def _reference_add(state: qm.Q[qm.Bits[4]]) -> qm.Q[qm.Bits[4]]:
    """Independent two-bit reversible oracle expressed only with basic gates."""
    addend, target = qm.split_at(state, 2)
    addend, target = qm.control(_reference_increment, addend, target, at=0)
    addend, target = qm.control(_reference_high_flip, addend, target, at=1)
    return qm.concat(addend, target)


@qm.qkernel
def _reference_subtract(state: qm.Q[qm.Bits[4]]) -> qm.Q[qm.Bits[4]]:
    """Invert the independent oracle using the ordinary callable transform."""
    return qm.adjoint(_reference_add, state)


@qm.qkernel
def _coherent_preparation(state: qm.Q[qm.Bits[5]]) -> qm.Q[qm.Bits[5]]:
    """Prepare distinct amplitudes and phases on operands and the control."""
    state = qm.h(state, target=0)
    state = qm.ry(state, 0.7, target=1)
    state = qm.ry(state, 1.1, target=2)
    state = qm.ry(state, -0.4, target=3)
    state = qm.ry(state, 0.9, target=4)
    return qm.s(state, target=2)


@qm.qkernel
def realization_add_coherent_state() -> qm.Q[qm.Bits[5]]:
    """Compare controlled addition with an independent gate-level inverse."""
    state = _coherent_preparation(qm.qubits(5))
    selector, registers = qm.split_at(state, 1)
    selector, registers = qm.control(_add_two, selector, registers)
    selector, registers = qm.control(_reference_subtract, selector, registers)
    state = qm.concat(selector, registers)
    return qm.adjoint(_coherent_preparation, state)


@qm.qkernel
def realization_add_coherent_sample() -> qm.Bits[5]:
    """Interference detects arithmetic mistakes and control-dependent phases."""
    return qm.measure(realization_add_coherent_state())


@qm.qkernel
def realization_subtract_sample() -> qm.Bits[4]:
    """Subtract one from zero through the abstract callable adjoint."""
    state = qm.x(qm.qubits(4), target=0)
    return qm.measure(qm.adjoint(_add_two, state))


@qm.qkernel
def operator_arithmetic_sample() -> qm.Bits[8]:
    """Exercise unsigned + and - successor tuples on every backend."""
    add_left = qm.as_uint(qm.prepare((1, 0)))
    add_right = qm.as_uint(qm.prepare((0, 1)))
    add_left, add_right = add_left + add_right
    sub_left = qm.as_uint(qm.prepare((1, 0)))
    sub_right = qm.as_uint(qm.prepare((0, 1)))
    sub_left, sub_right = sub_left - sub_right
    state = qm.concat(
        qm.as_bits(add_left),
        qm.as_bits(add_right),
        qm.as_bits(sub_left),
        qm.as_bits(sub_right),
    )
    return qm.measure(state)


@qm.qkernel
def stdlib_arithmetic_ux_sample() -> qm.Bits[10]:
    """Exercise wrapping and modular arithmetic APIs on every backend."""
    added = _add_one_wrapping(qm.prepare((0, 1)))
    multiplied = _multiply_two_mod_three(qm.prepare((0, 1)))
    register_sum = _add_modulo_one_bit(qm.prepare((1, 1, 0)))
    compared = _compare_one_bit(qm.prepare((0, 1)))
    return qm.measure(qm.concat(added, multiplied, register_sum, compared))


@qm.qkernel
def first_class_mod_add_sample() -> qm.Bits[6]:
    """Retain addend and modulus while wrapping the target modulo three."""

    target = qm.as_uint(qm.prepare((1, 0)))
    addend = qm.as_uint(qm.prepare((0, 1)))
    modulus = qm.as_uint(qm.prepare((1, 1)))
    target, addend, modulus = qm.mod_add(target, addend, modulus)
    return qm.measure(
        qm.concat(qm.as_bits(target), qm.as_bits(addend), qm.as_bits(modulus))
    )


@qm.qkernel
def first_class_mod_add_invalid_sample() -> qm.Bits[6]:
    """Leave an addend at the modulus outside the modular update domain."""

    target = qm.as_uint(qm.prepare((1, 0)))
    addend = qm.as_uint(qm.prepare((1, 1)))
    modulus = qm.as_uint(qm.prepare((1, 1)))
    target, addend, modulus = qm.mod_add(target, addend, modulus)
    return qm.measure(
        qm.concat(qm.as_bits(target), qm.as_bits(addend), qm.as_bits(modulus))
    )


@qm.qkernel
def first_class_mod_add_constant_sample() -> qm.Bits[2]:
    """Apply modular addition with two static operands."""

    target = qm.as_uint(qm.prepare((0, 1)))
    return qm.measure(qm.as_bits(qm.mod_add(target, 2, 3)))


@qm.qkernel
def first_class_mul_add_sample() -> qm.Bits[8]:
    """Add six to a nonzero product accumulator and retain both factors."""

    product = qm.as_uint(qm.prepare((1, 0, 0, 0)))
    left = qm.as_uint(qm.prepare((0, 1)))
    right = qm.as_uint(qm.prepare((1, 1)))
    product, left, right = qm.mul_add(product, left, right)
    return qm.measure(
        qm.concat(qm.as_bits(product), qm.as_bits(left), qm.as_bits(right))
    )


@qm.qkernel
def first_class_mod_mul_state() -> qm.Q[qm.Bits[3]]:
    """Retain the modular multiplication result for a value estimate."""

    target = qm.as_uint(qm.prepare((0, 1, 0)))
    return qm.as_bits(qm.mod_mul(target, factor=2, modulus=5))


@qm.qkernel
def first_class_mod_mul_sample() -> qm.Bits[3]:
    """Multiply two modulo five while preserving out-of-domain states."""

    return qm.measure(first_class_mod_mul_state())


@qm.qkernel
def first_class_mod_mul_invalid_sample() -> qm.Bits[3]:
    """Preserve a value above the modulus under modular multiplication."""

    target = qm.as_uint(qm.prepare((0, 1, 1)))
    return qm.measure(qm.as_bits(qm.mod_mul(target, factor=2, modulus=5)))


@qm.qkernel
def first_class_mod_pow_sample() -> qm.Bits[5]:
    """Raise two coherently to exponent two and retain the exponent."""

    target = qm.as_uint(qm.prepare((1, 0, 0)))
    exponent = qm.as_uint(qm.prepare((0, 1)))
    target, exponent = qm.mod_pow(target, exponent, base=2, modulus=5)
    return qm.measure(qm.concat(qm.as_bits(target), qm.as_bits(exponent)))


@qm.qkernel
def first_class_mod_pow_invalid_sample() -> qm.Bits[5]:
    """Preserve an out-of-domain target while retaining the exponent."""

    target = qm.as_uint(qm.prepare((0, 1, 1)))
    exponent = qm.as_uint(qm.prepare((0, 1)))
    target, exponent = qm.mod_pow(target, exponent, base=2, modulus=5)
    return qm.measure(qm.concat(qm.as_bits(target), qm.as_bits(exponent)))


@qm.qkernel
def mod_pow_identity_sample() -> qm.Bits[2]:
    """Exercise the semantic boundary with the exact identity residue."""

    target = qm.as_uint(qm.prepare((1,)))
    exponent = qm.as_uint(qm.prepare((1,)))
    target, exponent = qm.mod_pow(target, exponent, base=1, modulus=2)
    return qm.measure(qm.concat(qm.as_bits(target), qm.as_bits(exponent)))


@qm.qkernel
def first_class_comparison_sample() -> qm.Bits[5]:
    """Toggle a comparison flag and retain asymmetric unsigned operands."""

    flag = qm.as_uint(qm.prepare((0,)))
    lower = qm.as_uint(qm.prepare((1, 0)))
    upper = qm.as_uint(qm.prepare((0, 1)))
    flag, lower, upper = qm.less_than_xor(flag, lower, upper)
    return qm.measure(qm.concat(qm.as_bits(flag), qm.as_bits(lower), qm.as_bits(upper)))


@qm.qkernel
def first_class_constant_add_sample() -> qm.Bits[3]:
    """Exercise a literal added to a coherent unsigned register."""

    value = qm.as_uint(qm.prepare((1, 1, 0)))
    return qm.measure(qm.as_bits(value + 2))


@qm.qkernel
def first_class_mod_mul_owned(target: qm.Q[qm.UInt[3]]) -> qm.Q[qm.UInt[3]]:
    """Expose modular multiplication as an invertible owned transform."""

    return qm.mod_mul(target, factor=2, modulus=5)


@qm.qkernel
def first_class_mod_mul_roundtrip() -> qm.Bits[3]:
    """Undo modular multiplication with the compiler-derived adjoint."""

    target = qm.as_uint(qm.prepare((0, 1, 0)))
    target = first_class_mod_mul_owned(target)
    target = qm.adjoint(first_class_mod_mul_owned, target)
    return qm.measure(qm.as_bits(target))


@qm.qkernel
def first_class_mod_mul_controlled() -> qm.Bits[4]:
    """Control modular multiplication without exposing scratch bits."""

    control = qm.prepare((1,))
    target = qm.as_uint(qm.prepare((0, 1, 0)))
    control, target = qm.control(first_class_mod_mul_owned, control, target)
    return qm.measure(qm.concat(control, qm.as_bits(target)))


_signal_input_turn = turn(Region("00"))
_signal_output_turn = turn(Region("11"))
_signal_full_turn = turn(Region("*"))
_signal_mask_turn = turn(Region("1*0*"))
_fixed_point_expected = amplitude_encoding(
    ((3 + 3j * sqrt(3)) / 16,) * 3 + ((-1 + 7j * sqrt(3)) / 16,)
)


@qm.qkernel
def signal_prepare(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    """Prepare a uniform search input without an algorithm-specific operation."""
    return qm.h(state)


@qm.qkernel
def signal_search_process(
    state: qm.Q[qm.Bits[2]], phases: qm.Matrix[qm.Real, 1, 2]
) -> qm.Q[qm.Bits[2]]:
    """Expose one editable phase round with the same functional composition."""
    state = signal_prepare(state)
    for i in range(1):
        state = _signal_output_turn(state, phases[i, 1])
        state = qm.adjoint(signal_prepare, state)
        state = _signal_input_turn(state, phases[i, 0])
        state = signal_prepare(state)
    return state


@qm.qkernel
def signal_search_state(
    phases: qm.Param[qm.Matrix[qm.Real, 1, 2]],
) -> qm.Q[qm.Bits[2]]:
    """Retain phase coordinates in the ordinary launch ABI."""
    return signal_search_process(qm.qubits(2), phases)


@qm.qkernel
def signal_search_sample(phases: qm.Param[qm.Matrix[qm.Real, 1, 2]]) -> qm.Bits[2]:
    """Sample the same search process with different phase schedules."""
    return qm.measure(signal_search_state(phases))


@qm.qkernel
def signal_fixed_point_state() -> qm.Q[qm.Bits[2]]:
    """Compare a pi/3 round with independently derived complex amplitudes."""
    state = signal_search_process(qm.qubits(2), [[pi / 3, pi / 3]])
    return qm.adjoint(_fixed_point_expected, state)


@qm.qkernel
def signal_fixed_point_sample() -> qm.Bits[2]:
    """Expose both magnitudes and relative phases of the fixed-point witness."""
    return qm.measure(signal_fixed_point_state())


@qm.qkernel
def signal_scalar_phase_sample(angle: qm.Param[qm.Real]) -> qm.Bits[2]:
    """Make a full-space scalar phase observable through coherent control."""
    selector = qm.h(qm.qubits(1))
    target = qm.qubits(1)
    selector, target = qm.control(_signal_full_turn, selector, target, angle)
    return qm.measure(qm.concat(qm.h(selector), target))


@qm.qkernel
def signal_mask_state(angle: qm.Param[qm.Real]) -> qm.Q[qm.Bits[4]]:
    """Phase a noncontiguous region and retain asymmetric free coordinates."""
    state = qm.x(qm.qubits(4), target=0)
    state = qm.x(state, target=1)
    state = qm.h(state, target=2)
    state = qm.x(state, target=3)
    state = _signal_mask_turn(state, angle)
    return qm.h(state, target=2)


@qm.qkernel
def signal_mask_sample(angle: qm.Param[qm.Real]) -> qm.Bits[4]:
    """Measure all constrained and unconstrained lanes after phase routing."""
    return qm.measure(signal_mask_state(angle))


@qm.qkernel
def signal_identity_two(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    return state


@qm.qkernel
def signal_x_two(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    return qm.x(state, target=1)


@qm.qkernel
def signal_plus_one(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.h(state)


_algebra_x = Signal.from_block_encoding(BlockEncoding(signal_x_two, 2.0, 1, 1))
_algebra_i = Signal.from_block_encoding(BlockEncoding(signal_identity_two, 4.0, 1, 1))
signal_product_readout = sample(_algebra_x @ _algebra_x)
signal_interference_readout = sample(
    interfere(((0.5, _algebra_x), (-0.25, _algebra_i))), source=signal_plus_one
)
_algebra_rect = select(signal_identity_two, input=Region("**"), output=Region("0*"))
_algebra_even = response(
    _algebra_rect, Response.chebyshev((0.0, 0.0, 1.0)), side="input"
)
signal_response_readout = sample(
    _algebra_even, source=signal_prepare, readout=signal_prepare
)
_algebra_even_process = _algebra_even.process
signal_amplified_readout = sample(
    amplify(
        select(signal_prepare, input=Region("00"), output=Region("11")),
        phases=((pi, pi),),
    )
)


@qm.qkernel
def signal_algebra_expectation_state() -> qm.Q[qm.Bits[3]]:
    state = qm.h(qm.qubits(3), target=1)
    state = qm.h(state, target=2)
    state = _algebra_even_process(state)
    state = qm.h(state, target=1)
    return qm.h(state, target=2)
