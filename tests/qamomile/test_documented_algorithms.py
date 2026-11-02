"""Executable Qamomile counterparts for documented application algorithms."""

from __future__ import annotations

import cmath
import math

import qamomile as qm
from qamomile.compilation import ExecutionEndpoint, SampleResult
from qamomile.optimization import BinaryModel, PauliCorrelationEncoding
from qamomile.stdlib import (
    add_registers,
    amplitude_angles,
    amplitude_encoding,
    amplitude_encoding_from_angles,
    gf2_multiply_add,
    givens_rotation,
    has_duplicates_sorted,
    inverse_qft,
    less_than_registers,
    multiply_add,
    pauli_block_encoding,
    qft,
    subtract_registers,
    suzuki_trotter,
    uniformly_controlled_angles,
    uniformly_controlled_ry,
    uniformly_controlled_ry_from_angles,
    windowed_phase_estimation,
    xor_lookup,
)


def _kikuchi_entry_values(
    vertices: tuple[tuple[int, ...], ...],
    rhs_by_scope: dict[tuple[int, ...], int],
    *,
    index_width: int,
) -> tuple[int, ...]:
    """Return packed presence/sign values for one bounded Kikuchi oracle."""

    values = []
    for address in range(1 << (2 * index_width)):
        row = address & ((1 << index_width) - 1)
        column = address >> index_width
        packed = 0
        if row < len(vertices) and column < len(vertices):
            difference = tuple(sorted(set(vertices[row]) ^ set(vertices[column])))
            if difference in rhs_by_scope:
                packed = 1 | (int(rhs_by_scope[difference] < 0) << 1)
        values.append(packed)
    return tuple(values)


_CLASSIFICATION_CASES = (
    ((0.25, -0.5), (0.25, -0.5), (0, 0)),
    ((0.0, 0.0), (math.pi, 0.0), (1, 0)),
)
_HYDROGEN_ANGLE = -math.atan2(0.18093119978423156, 2 * 0.39793742484318045)
_HYDROGEN_HAMILTONIAN = (
    -1.052373245772859
    + 0.39793742484318045 * qm.Z(0)
    - 0.39793742484318045 * qm.Z(1)
    - 0.01128010425623538 * qm.Z(0) * qm.Z(1)
    + 0.18093119978423156 * qm.X(0) * qm.X(1)
)
_QSCI_ANGLE = math.pi / 2.0
_QSCI_HAMILTONIAN = -qm.Z(0) * qm.Z(1) - 0.7 * qm.X(0) - 0.7 * qm.X(1)
_RABI_OMEGA = 1.2
_RABI_DRIVE = 0.8
_RABI_TIME = 1.5
_RABI_TERMS = (
    0.5 * _RABI_OMEGA * qm.Z(0),
    0.5 * _RABI_DRIVE * qm.X(0),
)
_RABI_EVOLUTION = suzuki_trotter(term_count=2, order=4, steps=4)
_MOTTONEN_AMPLITUDES = (
    1.0 + 0.0j,
    1.0 + 1.0j,
    1.0 - 1.0j,
    0.0 + 2.0j,
)
_MOTTONEN_COORDINATES = amplitude_angles(_MOTTONEN_AMPLITUDES)
_MOTTONEN_RUNTIME = amplitude_encoding_from_angles(2)
_MOTTONEN_STATIC = amplitude_encoding(_MOTTONEN_AMPLITUDES)
_MULTIDIMENSIONAL_QFT_INPUT = tuple(
    1.0 if 4 <= index < 8 else 0.0 for index in range(16)
)
_MULTIDIMENSIONAL_QFT_COORDINATES = amplitude_angles(_MULTIDIMENSIONAL_QFT_INPUT)
_MULTIDIMENSIONAL_QFT_RUNTIME = amplitude_encoding_from_angles(4)
_MULTIDIMENSIONAL_QFT_STATIC = amplitude_encoding(_MULTIDIMENSIONAL_QFT_INPUT)
_HQNN_INPUTS = (math.pi / 4.0,) * 4
_HQNN_WEIGHTS_Z1 = (0.0, -0.2, 0.3, -0.4)
_HQNN_WEIGHTS_Y = (math.pi / 4.0, math.pi / 4.0, 0.0, -math.pi / 4.0)
_HQNN_WEIGHTS_Z2 = (-0.4, 0.3, -0.2, 0.1)
_PCE_MODEL = BinaryModel.from_ising(
    {
        (0, 1): 0.5,
        (1, 2): 0.5,
        (2, 3): 0.5,
        (0, 3): 0.5,
    },
    constant=-2.0,
    variables=range(4),
)
_PCE_ENCODING = PauliCorrelationEncoding(_PCE_MODEL, correlator_order=2)
_PCE_ANGLES = (math.pi / 2.0, -math.pi / 4.0, math.pi / 4.0, -math.pi / 4.0)
_PCE_EXPECTATIONS = (
    math.sqrt(2.0) / 4.0,
    -math.sqrt(2.0) / 4.0,
    0.5,
    -math.sqrt(2.0) / 4.0,
)
_OPTION_PROBABILITIES = (0.75, 0.25)
_OPTION_DISTRIBUTION_COORDINATES = amplitude_angles(
    tuple(math.sqrt(probability) for probability in _OPTION_PROBABILITIES)
)
_OPTION_PAYOFF_BRANCH_ANGLES = (0.0, math.pi)
_OPTION_PAYOFF_COORDINATES = uniformly_controlled_angles(_OPTION_PAYOFF_BRANCH_ANGLES)
_OPTION_DISTRIBUTION_RUNTIME = amplitude_encoding_from_angles(1)
_OPTION_DISTRIBUTION_STATIC = amplitude_encoding(
    tuple(math.sqrt(probability) for probability in _OPTION_PROBABILITIES)
)
_OPTION_PAYOFF_RUNTIME = uniformly_controlled_ry_from_angles(1)
_OPTION_PAYOFF_STATIC = uniformly_controlled_ry(_OPTION_PAYOFF_BRANCH_ANGLES)
_OPTION_RATE = 0.05
_OPTION_MATURITY = 1.0
_QEMCMC_GAMMA = 0.45
_QEMCMC_TIME = 12.0
_QEMCMC_BETA = 1.0
_QEMCMC_INPUT_BITS = (1, 0)
_QEMCMC_INPUT_ANGLES = tuple(math.pi * bit for bit in _QEMCMC_INPUT_BITS)
_QEMCMC_HAMILTONIANS = (
    (1.0 - _QEMCMC_GAMMA) * (qm.X(0) + qm.X(1)),
    -_QEMCMC_GAMMA * qm.Z(0) * qm.Z(1),
)
_QEMCMC_EVOLUTION = suzuki_trotter(term_count=2, order=2, steps=1)
_BIT_FLIP_THETA = math.pi / 3.0
_BIT_FLIP_SYNDROMES = ((1, 1), (1, 0), (0, 1))
_EXPECTATION_SHOTS = 64
_STATISTICAL_SHOTS = 128
_WINDOWED_QPE_KAISER = amplitude_encoding(
    (
        0.00479608396221936,
        0.02421703544211841,
        0.06204631649516534,
        0.12000662094308828,
        0.19421005500270414,
        0.27486050829902164,
        0.34805969895198946,
        0.399336135628929,
        0.4177797309175418,
        0.399336135628929,
        0.34805969895198946,
        0.27486050829902164,
        0.19421005500270414,
        0.12000662094308828,
        0.06204631649516534,
        0.02421703544211841,
    )
)
_DOCUMENTED_LESS_THAN = less_than_registers(width=2)
_DOCUMENTED_MULTIPLY = multiply_add(width=2)
_DOCUMENTED_MULTIPLY_BOUNDARY = multiply_add(width=1)
_DOCUMENTED_GF8_MULTIPLY = gf2_multiply_add(width=3, modulus=0b1011)
_DOCUMENTED_PHASE_GRADIENT_ADD = add_registers(width=2)
_DOCUMENTED_BATCH_LOOKUP = xor_lookup((13, 2), value_width=4)
_DOCUMENTED_ALIAS_LOOKUP = xor_lookup((10, 11, 2, 3), value_width=4)
_DOCUMENTED_ALIAS_COMPARE = less_than_registers(width=2)
_DOCUMENTED_TRIANGULAR_LOOKUP = xor_lookup((0, 1, 3, 6), value_width=4)
_DOCUMENTED_CONTIGUOUS_ADD = add_registers(width=4)
_DOCUMENTED_ARCSIN_VALUES = tuple(int(math.asin(value / 8) * 16) for value in range(8))
_DOCUMENTED_ARCTAN_VALUES = tuple(
    int(value > 0) | (int(abs(-2 * math.atan(value) / math.pi) * 16) << 1)
    for value in range(8)
)
_DOCUMENTED_ARCSIN_LOOKUP = xor_lookup(_DOCUMENTED_ARCSIN_VALUES, value_width=5)
_DOCUMENTED_ARCTAN_LOOKUP = xor_lookup(_DOCUMENTED_ARCTAN_VALUES, value_width=5)
_DOCUMENTED_ARCTAN_BOUNDARY_LOOKUP = xor_lookup(
    _DOCUMENTED_ARCTAN_VALUES[:4],
    value_width=5,
)
_DOCUMENTED_DUPLICATES = has_duplicates_sorted(length=3, item_width=2)
_DOCUMENTED_DUPLICATES_BOUNDARY = has_duplicates_sorted(length=2, item_width=1)
_DOCUMENTED_SYMMETRIC_DIFFERENCE_VALUES = (0, 6, 6, 0)
_DOCUMENTED_SYMMETRIC_DIFFERENCE = xor_lookup(
    _DOCUMENTED_SYMMETRIC_DIFFERENCE_VALUES,
    value_width=3,
)
_DOCUMENTED_KIKUCHI_VERTICES = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
)
_DOCUMENTED_KIKUCHI_RHS = {
    (0, 1): -1,
    (0, 2): 1,
    (0, 3): -1,
    (1, 2): -1,
    (1, 3): 1,
    (2, 3): -1,
}
_DOCUMENTED_KIKUCHI_VALUES = _kikuchi_entry_values(
    _DOCUMENTED_KIKUCHI_VERTICES,
    _DOCUMENTED_KIKUCHI_RHS,
    index_width=3,
)
_DOCUMENTED_KIKUCHI_BOUNDARY_VALUES = _kikuchi_entry_values(
    _DOCUMENTED_KIKUCHI_VERTICES[:4],
    _DOCUMENTED_KIKUCHI_RHS,
    index_width=2,
)
_DOCUMENTED_KIKUCHI_BOUNDARY = xor_lookup(
    _DOCUMENTED_KIKUCHI_BOUNDARY_VALUES,
    value_width=2,
)
_DOCUMENTED_KIKUCHI_MINIMAL_VALUES = _kikuchi_entry_values(
    _DOCUMENTED_KIKUCHI_VERTICES[:2],
    _DOCUMENTED_KIKUCHI_RHS,
    index_width=1,
)
_DOCUMENTED_KIKUCHI_MINIMAL = xor_lookup(
    _DOCUMENTED_KIKUCHI_MINIMAL_VALUES,
    value_width=2,
)
_DOCUMENTED_GUIDED_STATE = amplitude_encoding((0.5, math.sqrt(3.0) / 2.0))
_DOCUMENTED_EC_SECRET = 3
_DOCUMENTED_EC_POINT_LABELS = tuple(
    (coefficient_p + _DOCUMENTED_EC_SECRET * coefficient_q) % 4
    for coefficient_q in range(4)
    for coefficient_p in range(4)
)
_DOCUMENTED_EC_ADD = add_registers(width=2)
_DOCUMENTED_EC_SUBTRACT = subtract_registers(width=2)
_DOCUMENTED_SF_OPERATOR = 0.625 + 0.5 * qm.Z(0) * qm.Z(1)
_DOCUMENTED_SF_ENCODING = pauli_block_encoding(_DOCUMENTED_SF_OPERATOR)
_DOCUMENTED_SF_UNITARY = _DOCUMENTED_SF_ENCODING.unitary
_DOCUMENTED_THC_CHI = amplitude_encoding((math.sqrt(0.5), math.sqrt(0.5)))
_DOCUMENTED_THC_ZETA = xor_lookup((0, 1, 1, 0), value_width=1)
_DOCUMENTED_FIRST_QUANTIZED_EVOLUTION = suzuki_trotter(
    term_count=2,
    order=1,
    steps=1,
)
_DOCUMENTED_FIRST_QUANTIZED_TERMS = (
    qm.X(0) + qm.X(1),
    0.5 - 0.5 * qm.Z(0) * qm.Z(1),
)
_DOCUMENTED_PROJECTILE_EVOLUTION = suzuki_trotter(
    term_count=2,
    order=1,
    steps=1,
)
_DOCUMENTED_PROJECTILE_TERMS = (
    qm.X(0) + qm.X(1) + 0.5 * qm.X(2),
    0.25 * qm.Z(0) * qm.Z(1) - 0.5 * qm.Z(0) * qm.Z(2) - 0.5 * qm.Z(1) * qm.Z(2),
)


@qm.qkernel
def windowed_qpe_fifth_phase(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    """Apply the off-grid one-fifth eigenphase used by the window tutorial.

    Args:
        state (qm.Q[qm.Bits[1]]): Owned one-qubit eigenstate register.

    Returns:
        qm.Q[qm.Bits[1]]: Phase-shifted successor state.
    """

    return qm.p(state, 1.2566370614359172)


@qm.qkernel
def kaiser_windowed_qpe_sample() -> qm.Shot[qm.UFixed[0, 4]]:
    """Sample the documented four-bit Kaiser-windowed phase estimate.

    Returns:
        qm.Shot[qm.UFixed[0, 4]]: Numeric four-bit phase observation.
    """

    window = _WINDOWED_QPE_KAISER(qm.qubits(4))
    target = qm.prepare((1,))
    phase, target = windowed_phase_estimation(
        windowed_qpe_fifth_phase,
        window,
        target,
        4,
    )
    result = qm.measure(phase)
    _target_result = qm.measure(target)
    return result


@qm.qkernel
def reversible_sort_descending_sample() -> qm.Shot[qm.Bits[6]]:
    """Sort one two-word basis input while retaining the swap history.

    Returns:
        qm.Shot[qm.Bits[6]]: Descending operands, swap flag, and clean work
            component.
    """

    state = _DOCUMENTED_LESS_THAN(qm.prepare((1, 0, 1, 1, 0, 0)))
    for index in range(2):
        state = qm.cx(state, control=index, target=2 + index)
        state = qm.ccx(state, control1=4, control2=2 + index, target=index)
        state = qm.cx(state, control=index, target=2 + index)
    return qm.measure(state)


@qm.qkernel
def reversible_register_product_state() -> qm.Q[qm.Bits[8]]:
    """Multiply two retained two-bit factors into a four-bit accumulator.

    Returns:
        qm.Q[qm.Bits[8]]: Factors followed by their exact product.
    """

    return _DOCUMENTED_MULTIPLY(qm.prepare((0, 1, 1, 1, 0, 0, 0, 0)))


@qm.qkernel
def reversible_register_product_sample() -> qm.Shot[qm.Bits[8]]:
    """Measure both factors and their exact four-bit product.

    Returns:
        qm.Shot[qm.Bits[8]]: Factors followed by their exact product.
    """

    return qm.measure(reversible_register_product_state())


@qm.qkernel
def reversible_register_product_boundary_state() -> qm.Q[qm.Bits[4]]:
    """Exercise the minimal one-bit register-multiplication boundary.

    Returns:
        qm.Q[qm.Bits[4]]: Two one-bit factors followed by their two-bit product.
    """

    return _DOCUMENTED_MULTIPLY_BOUNDARY(qm.prepare((1, 1, 0, 0)))


@qm.qkernel
def gf8_polynomial_product_sample() -> qm.Shot[qm.Bits[9]]:
    """Multiply five by three modulo x^3 + x + 1.

    Returns:
        qm.Shot[qm.Bits[9]]: Retained factors and their GF(2^3) product.
    """

    state = _DOCUMENTED_GF8_MULTIPLY(qm.prepare((1, 0, 1, 1, 1, 0, 0, 0, 0)))
    return qm.measure(state)


@qm.qkernel
def prepare_two_bit_phase_gradient(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    """Prepare a two-bit Fourier state from the integer one.

    Args:
        state (qm.Q[qm.Bits[2]]): Clean phase-gradient register.

    Returns:
        qm.Q[qm.Bits[2]]: Fourier-basis eigenstate of modular addition.
    """

    state = qm.x(state, target=0)
    state = qm.h(state, target=1)
    state = qm.cp(state, 1.5707963267948966, control=0, target=1)
    state = qm.h(state, target=0)
    return qm.swap(state, left=0, right=1)


@qm.qkernel
def phase_gradient_qvr_sample() -> qm.Shot[qm.Bits[5]]:
    """Kick a half-turn phase back from a phase-gradient register.

    Returns:
        qm.Shot[qm.Bits[5]]: Probe, retained value, and clean gradient bits.
    """

    probe = qm.h(qm.qubits(1))
    value = qm.prepare((0, 1))
    gradient = prepare_two_bit_phase_gradient(qm.qubits(2))
    probe, arithmetic = qm.control(
        _DOCUMENTED_PHASE_GRADIENT_ADD,
        probe,
        qm.concat(value, gradient),
    )
    probe = qm.h(probe)
    value, gradient = qm.split_at(arithmetic, 2)
    gradient = qm.adjoint(prepare_two_bit_phase_gradient, gradient)
    return qm.measure(qm.concat(probe, value, gradient))


@qm.qkernel
def hamming_weight_phase(state: qm.Q[qm.Bits[3]]) -> qm.Q[qm.Bits[3]]:
    """Apply a quarter-turn phase for every occupied component.

    Args:
        state (qm.Q[qm.Bits[3]]): Three-bit data register.

    Returns:
        qm.Q[qm.Bits[3]]: Data phased by its Hamming weight.
    """

    for target in range(3):
        state = qm.p(state, 1.5707963267948966, target=target)
    return state


@qm.qkernel
def hamming_weight_phase_sample() -> qm.Shot[qm.Bits[4]]:
    """Kick the phase of a Hamming-weight-two basis state to a probe.

    Returns:
        qm.Shot[qm.Bits[4]]: Probe followed by unchanged data bits.
    """

    probe = qm.h(qm.qubits(1))
    data = qm.prepare((1, 1, 0))
    probe, data = qm.control(hamming_weight_phase, probe, data)
    probe = qm.h(probe)
    return qm.measure(qm.concat(probe, data))


@qm.qkernel
def swap_two_bit_words(state: qm.Q[qm.Bits[4]]) -> qm.Q[qm.Bits[4]]:
    """Swap the leading and trailing two-bit words.

    Args:
        state (qm.Q[qm.Bits[4]]): Two adjacent two-bit words.

    Returns:
        qm.Q[qm.Bits[4]]: Words in the opposite order.
    """

    state = qm.swap(state, left=0, right=2)
    return qm.swap(state, left=1, right=3)


@qm.qkernel
def select_swap_qroam_load(state: qm.Q[qm.Bits[6]]) -> qm.Q[qm.Bits[6]]:
    """Load one four-row table value with a two-row select-swap block.

    Args:
        state (qm.Q[qm.Bits[6]]): Address, clean target, and clean junk words.

    Returns:
        qm.Q[qm.Bits[6]]: Address, selected target, and retained batch junk.
    """

    address, bank = qm.split_at(state, 2)
    low, high = qm.split_at(address, 1)
    high_bank = _DOCUMENTED_BATCH_LOOKUP(qm.concat(high, bank))
    high, bank = qm.split_at(high_bank, 1)
    low, bank = qm.control(swap_two_bit_words, low, bank)
    return qm.concat(low, high, bank)


@qm.qkernel
def select_swap_qroam_unload(state: qm.Q[qm.Bits[6]]) -> qm.Q[qm.Bits[6]]:
    """Reverse the documented select-swap load in explicit order.

    Args:
        state (qm.Q[qm.Bits[6]]): Address, selected target, and batch junk.

    Returns:
        qm.Q[qm.Bits[6]]: Address and restored target plus junk words.
    """

    address, bank = qm.split_at(state, 2)
    low, high = qm.split_at(address, 1)
    low, bank = qm.control(swap_two_bit_words, low, bank)
    high_bank = _DOCUMENTED_BATCH_LOOKUP(qm.concat(high, bank))
    high, bank = qm.split_at(high_bank, 1)
    return qm.concat(low, high, bank)


@qm.qkernel
def clean_qroam_sample() -> qm.Shot[qm.Bits[8]]:
    """Copy one selected value and uncompute all QROAM workspace.

    Returns:
        qm.Shot[qm.Bits[8]]: Address, clean workspace, and selected value.
    """

    address = qm.h(qm.qubits(2))
    loaded = select_swap_qroam_load(qm.concat(address, qm.qubits(4)))
    address, bank = qm.split_at(loaded, 2)
    target, junk = qm.split_at(bank, 2)
    target_result = qm.concat(target, qm.qubits(2))
    target_result = qm.cx(target_result, control=0, target=2)
    target_result = qm.cx(target_result, control=1, target=3)
    target, result = qm.split_at(target_result, 2)
    loaded = qm.concat(address, target, junk)
    loaded = select_swap_qroam_unload(loaded)
    address, workspace = qm.split_at(loaded, 2)
    return qm.measure(qm.concat(address, workspace, result))


@qm.qkernel
def coherent_alias_sample() -> qm.Shot[qm.Bits[10]]:
    """Prepare the one-one-three-three distribution with alias sampling.

    Returns:
        qm.Shot[qm.Bits[10]]: Selection and retained alias temporary registers.
    """

    selection = qm.h(qm.qubits(2))
    sigma = qm.h(qm.qubits(2))
    lookup = _DOCUMENTED_ALIAS_LOOKUP(qm.concat(selection, qm.qubits(4)))
    selection, temporary = qm.split_at(lookup, 2)
    alternate, keep = qm.split_at(temporary, 2)
    comparison = _DOCUMENTED_ALIAS_COMPARE(qm.concat(sigma, keep, qm.qubits(2)))
    sigma, comparison_output = qm.split_at(comparison, 2)
    keep, comparison_bits = qm.split_at(comparison_output, 2)
    choose_alias, compare_work = qm.split_at(comparison_bits, 1)
    choose_alias, candidates = qm.control(
        swap_two_bit_words,
        choose_alias,
        qm.concat(selection, alternate),
    )
    selection, alternate = qm.split_at(candidates, 2)
    return qm.measure(
        qm.concat(
            selection,
            sigma,
            alternate,
            keep,
            choose_alias,
            compare_work,
        )
    )


@qm.qkernel
def coherent_alias_branch_sample() -> qm.Shot[qm.Bits[10]]:
    """Exercise one deterministic alias-table branch for backend acceptance.

    Returns:
        qm.Shot[qm.Bits[10]]: Alias-selected index and its retained temporary.
    """

    lookup = _DOCUMENTED_ALIAS_LOOKUP(qm.qubits(6))
    selection, temporary = qm.split_at(lookup, 2)
    alternate, keep = qm.split_at(temporary, 2)
    comparison = _DOCUMENTED_ALIAS_COMPARE(qm.concat(qm.qubits(2), keep, qm.qubits(2)))
    sigma, comparison_output = qm.split_at(comparison, 2)
    keep, comparison_bits = qm.split_at(comparison_output, 2)
    choose_alias, compare_work = qm.split_at(comparison_bits, 1)
    choose_alias, candidates = qm.control(
        swap_two_bit_words,
        choose_alias,
        qm.concat(selection, alternate),
    )
    selection, alternate = qm.split_at(candidates, 2)
    return qm.measure(
        qm.concat(
            selection,
            sigma,
            alternate,
            keep,
            choose_alias,
            compare_work,
        )
    )


@qm.qkernel
def basis_permutation_sample() -> qm.Shot[qm.Bits[2]]:
    """Apply the four-cycle zero-one-three-two in place.

    Returns:
        qm.Shot[qm.Bits[2]]: Permuted two-bit basis value.
    """

    state = qm.swap(qm.prepare((1, 0)), left=0, right=1)
    state = qm.x(state, target=0)
    return qm.measure(state)


@qm.qkernel
def contiguous_index_sample() -> qm.Shot[qm.Bits[10]]:
    """Compute s = nu(nu + 1)/2 + mu for mu one and nu two.

    Returns:
        qm.Shot[qm.Bits[10]]: Retained inputs, contiguous index, and clean pad.
    """

    mu = qm.prepare((1, 0))
    nu = qm.prepare((0, 1))
    nu_index = _DOCUMENTED_TRIANGULAR_LOOKUP(qm.concat(nu, qm.qubits(4)))
    nu, index = qm.split_at(nu_index, 2)
    extended_mu = qm.concat(mu, qm.qubits(2))
    addition = _DOCUMENTED_CONTIGUOUS_ADD(qm.concat(extended_mu, index))
    extended_mu, index = qm.split_at(addition, 4)
    mu, padding = qm.split_at(extended_mu, 2)
    return qm.measure(qm.concat(mu, nu, index, padding))


@qm.qkernel
def reversible_arcsin_sample() -> qm.Shot[qm.Bits[8]]:
    """Evaluate the bounded arcsine table at one half.

    Returns:
        qm.Shot[qm.Bits[8]]: Input and four-fractional-bit arcsine output.
    """

    state = _DOCUMENTED_ARCSIN_LOOKUP(qm.prepare((0, 0, 1, 0, 0, 0, 0, 0)))
    return qm.measure(state)


@qm.qkernel
def reversible_arctan_sample() -> qm.Shot[qm.Bits[8]]:
    """Evaluate signed normalized arctangent at integer three.

    Returns:
        qm.Shot[qm.Bits[8]]: Input, sign, and four-bit magnitude output.
    """

    state = _DOCUMENTED_ARCTAN_LOOKUP(qm.prepare((1, 1, 0, 0, 0, 0, 0, 0)))
    return qm.measure(state)


@qm.qkernel
def reversible_arctan_boundary_sample() -> qm.Shot[qm.Bits[7]]:
    """Evaluate the same arctangent rule at the two-bit backend boundary.

    Returns:
        qm.Shot[qm.Bits[7]]: Two-bit input, sign, and four-bit magnitude output.
    """

    state = _DOCUMENTED_ARCTAN_BOUNDARY_LOOKUP(qm.prepare((1, 1, 0, 0, 0, 0, 0)))
    return qm.measure(state)


@qm.qkernel
def sorted_duplicate_boundary_sample() -> qm.Shot[qm.Bits[5]]:
    """Detect a duplicate in the minimal sorted backend witness.

    Returns:
        qm.Shot[qm.Bits[5]]: Retained items, duplicate flag, and clean work.
    """

    state = _DOCUMENTED_DUPLICATES_BOUNDARY(qm.prepare((1, 1, 0, 0, 0)))
    return qm.measure(state)


@qm.qkernel
def symmetric_difference_sample() -> qm.Shot[qm.Bits[5]]:
    """Compute the symmetric difference of the singleton sets zero and one.

    Returns:
        qm.Shot[qm.Bits[5]]: Inputs, sorted two-item difference, and valid flag.
    """

    state = _DOCUMENTED_SYMMETRIC_DIFFERENCE(qm.prepare((0, 1, 0, 0, 0)))
    return qm.measure(state)


@qm.qkernel
def kikuchi_entry_sample() -> qm.Shot[qm.Bits[6]]:
    """Query one signed edge of the bounded Kikuchi matrix.

    Returns:
        qm.Shot[qm.Bits[6]]: Row, column, presence, and sign components.
    """

    state = _DOCUMENTED_KIKUCHI_BOUNDARY(qm.prepare((0, 0, 1, 0, 0, 0)))
    return qm.measure(state)


@qm.qkernel
def kikuchi_entry_boundary_sample() -> qm.Shot[qm.Bits[4]]:
    """Query the same signed edge in the minimal backend submatrix.

    Returns:
        qm.Shot[qm.Bits[4]]: One-bit row and column, presence, and sign.
    """

    state = _DOCUMENTED_KIKUCHI_MINIMAL(qm.prepare((0, 1, 0, 0)))
    return qm.measure(state)


@qm.qkernel
def guided_hamiltonian_phase(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    """Apply the exact three-eighths high-eigenspace phase.

    Args:
        state (qm.Q[qm.Bits[1]]): Owned one-qubit Hamiltonian state.

    Returns:
        qm.Q[qm.Bits[1]]: State after one exact evolution step.
    """

    return qm.p(state, 2.356194490192345)


@qm.qkernel
def guided_hamiltonian_high_sample() -> qm.Shot[qm.UFixed[0, 3]]:
    """Estimate the exact high-eigenspace phase for backend acceptance.

    Returns:
        qm.Shot[qm.UFixed[0, 3]]: Exact three-bit phase estimate.
    """

    phase, target = windowed_phase_estimation(
        guided_hamiltonian_phase,
        qm.h(qm.h(qm.h(qm.qubits(3), target=0), target=1), target=2),
        qm.prepare((1,)),
        3,
    )
    result = qm.measure(phase)
    _target_result = qm.measure(target)
    return result


@qm.qkernel
def guided_hamiltonian_mixture_sample() -> qm.Shot[qm.UFixed[0, 3]]:
    """Estimate phases after preparing a three-quarter-overlap guide.

    Returns:
        qm.Shot[qm.UFixed[0, 3]]: Mixture of low and high eigenphases.
    """

    target = _DOCUMENTED_GUIDED_STATE(qm.qubits(1))
    phase, target = windowed_phase_estimation(
        guided_hamiltonian_phase,
        qm.h(qm.h(qm.h(qm.qubits(3), target=0), target=1), target=2),
        target,
        3,
    )
    result = qm.measure(phase)
    _target_result = qm.measure(target)
    return result


@qm.qkernel
def elliptic_curve_dlog_sample() -> tuple[
    qm.Shot[qm.UInt[2]],
    qm.Shot[qm.UInt[2]],
    qm.Shot[qm.UInt[2]],
]:
    """Sample the hidden subgroup for a bounded elliptic-curve instance.

    Returns:
        tuple[qm.Shot[qm.UInt[2]], qm.Shot[qm.UInt[2]], qm.Shot[qm.UInt[2]]]:
            Two typed Fourier frequencies followed by the retained
            elliptic-curve point label.
    """

    coefficient_p = qm.h(qm.h(qm.qubits(2), target=0), target=1)
    coefficient_q = qm.h(qm.h(qm.qubits(2), target=0), target=1)
    point = qm.qubits(2)
    coefficient_p, point = qm.split_at(
        _DOCUMENTED_EC_ADD(qm.concat(coefficient_p, point)),
        2,
    )
    coefficient_q, point = qm.split_at(
        _DOCUMENTED_EC_SUBTRACT(qm.concat(coefficient_q, point)),
        2,
    )
    coefficient_p = inverse_qft(coefficient_p, 2)
    coefficient_q = inverse_qft(coefficient_q, 2)
    coefficient_p = qm.swap(coefficient_p, left=0, right=1)
    coefficient_q = qm.swap(coefficient_q, left=0, right=1)
    point = qm.swap(point, left=0, right=1)
    return (
        qm.measure(qm.as_uint(coefficient_p)),
        qm.measure(qm.as_uint(coefficient_q)),
        qm.measure(qm.as_uint(point)),
    )


@qm.qkernel
def single_factorization_block_sample() -> qm.Shot[qm.Bits[3]]:
    """Sample the exact LCU block encoding of one squared density factor.

    Returns:
        qm.Shot[qm.Bits[3]]: Signal followed by the two-orbital system state.
    """

    return qm.measure(_DOCUMENTED_SF_UNITARY(qm.qubits(3)))


@qm.qkernel
def double_factorization_reflection(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    """Apply one basis-rotated orbital reflection.

    Args:
        state (qm.Q[qm.Bits[2]]): One two-orbital occupation register.

    Returns:
        qm.Q[qm.Bits[2]]: State after rotation, diagonal phase, and unrotation.
    """

    state = givens_rotation(state, 0, 1, 1.5707963267948966)
    state = qm.z(state, target=0)
    return qm.adjoint(givens_rotation, state, 0, 1, 1.5707963267948966)


@qm.qkernel
def double_factorization_basis_sample() -> qm.Shot[qm.Bits[3]]:
    """Kick one double-factorized orbital reflection into a probe.

    Returns:
        qm.Shot[qm.Bits[3]]: Probe followed by the retained orbital register.
    """

    probe = qm.h(qm.qubits(1))
    state = qm.adjoint(
        givens_rotation,
        qm.prepare((1, 0)),
        0,
        1,
        1.5707963267948966,
    )
    probe, state = qm.control(
        double_factorization_reflection,
        probe,
        state,
    )
    probe = qm.h(probe)
    state = givens_rotation(state, 0, 1, 1.5707963267948966)
    return qm.measure(qm.concat(probe, state))


@qm.qkernel
def tensor_hypercontraction_sample() -> qm.Shot[qm.Bits[3]]:
    """Load one central THC coupling for two prepared auxiliary indices.

    Returns:
        qm.Shot[qm.Bits[3]]: Two auxiliary indices and their coupling label.
    """

    left = _DOCUMENTED_THC_CHI(qm.qubits(1))
    right = _DOCUMENTED_THC_CHI(qm.qubits(1))
    return qm.measure(_DOCUMENTED_THC_ZETA(qm.concat(left, right, qm.qubits(1))))


@qm.qkernel
def first_quantized_chemistry_sample(
    hamiltonians: qm.Const[qm.Vector[qm.Observable, 2]],  # type: ignore[valid-type]
) -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Evolve two position registers through kinetic and Coulomb terms.

    Args:
        hamiltonians (qm.Const[qm.Vector[qm.Observable, 2]]): Kinetic and
            position-diagonal potential terms.

    Returns:
        qm.Shot[qm.Bits[2]]: Final positions of the two particles.
    """

    state = _DOCUMENTED_FIRST_QUANTIZED_EVOLUTION(
        qm.prepare((0, 1)),
        hamiltonians,
        1.5707963267948966,
    )
    return qm.measure(state)


@qm.qkernel
def quantum_projectile_sample(
    hamiltonians: qm.Const[qm.Vector[qm.Observable, 2]],  # type: ignore[valid-type]
) -> qm.Shot[qm.Bits[3]]:  # type: ignore[valid-type]
    """Evolve two electrons and one distinguishable projectile register.

    Args:
        hamiltonians (qm.Const[qm.Vector[qm.Observable, 2]]): Mass-weighted
            kinetic and pair-potential terms.

    Returns:
        qm.Shot[qm.Bits[3]]: Electron and projectile positions after one step.
    """

    state = _DOCUMENTED_PROJECTILE_EVOLUTION(
        qm.prepare((0, 1, 0)),
        hamiltonians,
        3.141592653589793,
    )
    return qm.measure(state)


@qm.qkernel
def classification_feature_map(
    state: qm.Q[qm.Bits[2]],  # type: ignore[valid-type]
    features: qm.Vector[qm.Real, 2],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[2]]:  # type: ignore[valid-type]
    """Encode one lifted feature vector into a two-qubit state.

    Args:
        state (qm.Q[qm.Bits[2]]): Owned two-qubit input state.
        features (qm.Vector[qm.Real, 2]): Two real feature coordinates.

    Returns:
        qm.Q[qm.Bits[2]]: Encoded two-qubit state.
    """

    for index in range(2):
        state = qm.h(state, target=index)
        state = qm.rz(state, features[index], target=index)
    state = qm.cx(state, control=0, target=1)
    return qm.rz(state, features[0] * features[1], target=1)


@qm.qkernel
def classification_overlap_state(
    left: qm.Vector[qm.Real, 2],  # type: ignore[valid-type]
    right: qm.Vector[qm.Real, 2],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[2]]:  # type: ignore[valid-type]
    """Build the reversible overlap witness for two feature vectors.

    Args:
        left (qm.Vector[qm.Real, 2]): First feature vector.
        right (qm.Vector[qm.Real, 2]): Second feature vector.

    Returns:
        qm.Q[qm.Bits[2]]: State whose zero amplitude encodes the overlap.
    """

    state = classification_feature_map(qm.qubits(2), left)
    return qm.adjoint(classification_feature_map, state, right)


@qm.qkernel
def classification_overlap(
    left: qm.Param[qm.Vector[qm.Real, 2]],  # type: ignore[valid-type]
    right: qm.Param[qm.Vector[qm.Real, 2]],  # type: ignore[valid-type]
) -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Measure a runtime-parametric overlap witness.

    Args:
        left (qm.Param[qm.Vector[qm.Real, 2]]): First runtime feature vector.
        right (qm.Param[qm.Vector[qm.Real, 2]]): Second runtime feature vector.

    Returns:
        qm.Shot[qm.Bits[2]]: Measured overlap witness bits.
    """

    return qm.measure(classification_overlap_state(left, right))


@qm.qkernel
def static_classification_overlap(
    left: qm.Const[qm.Vector[qm.Real, 2]],  # type: ignore[valid-type]
    right: qm.Const[qm.Vector[qm.Real, 2]],  # type: ignore[valid-type]
) -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Measure a specialized overlap witness for static-only targets.

    Args:
        left (qm.Const[qm.Vector[qm.Real, 2]]): First static feature vector.
        right (qm.Const[qm.Vector[qm.Real, 2]]): Second static feature vector.

    Returns:
        qm.Shot[qm.Bits[2]]: Measured overlap witness bits.
    """

    return qm.measure(classification_overlap_state(left, right))


@qm.qkernel
def hydrogen_vqe_ansatz(
    state: qm.Q[qm.Bits[2]],  # type: ignore[valid-type]
    theta: qm.Real,
) -> qm.Q[qm.Bits[2]]:  # type: ignore[valid-type]
    """Prepare the compact particle-preserving hydrogen trial state.

    Args:
        state (qm.Q[qm.Bits[2]]): Owned two-qubit input state.
        theta (qm.Real): Variational rotation angle in radians.

    Returns:
        qm.Q[qm.Bits[2]]: One-particle hydrogen trial state.
    """

    state = qm.x(state, target=0)
    state = qm.ry(state, theta, target=1)
    return qm.cx(state, control=1, target=0)


@qm.qkernel
def hydrogen_vqe_state(
    theta: qm.Param[qm.Real],
) -> qm.Q[qm.Bits[2]]:  # type: ignore[valid-type]
    """Build one runtime-parametric hydrogen VQE trial state.

    Args:
        theta (qm.Param[qm.Real]): Runtime variational angle in radians.

    Returns:
        qm.Q[qm.Bits[2]]: One-particle hydrogen trial state.
    """

    return hydrogen_vqe_ansatz(qm.qubits(2), theta)


@qm.qkernel
def static_hydrogen_vqe_state(
    theta: qm.Const[qm.Real],
) -> qm.Q[qm.Bits[2]]:  # type: ignore[valid-type]
    """Build the same hydrogen trial state for static-only targets.

    Args:
        theta (qm.Const[qm.Real]): Static variational angle in radians.

    Returns:
        qm.Q[qm.Bits[2]]: One-particle hydrogen trial state.
    """

    return hydrogen_vqe_ansatz(qm.qubits(2), theta)


@qm.qkernel
def qsci_input_state(
    state: qm.Q[qm.Bits[2]],  # type: ignore[valid-type]
    theta: qm.Real,
) -> qm.Q[qm.Bits[2]]:  # type: ignore[valid-type]
    """Prepare a hardware-efficient QSCI input state.

    Args:
        state (qm.Q[qm.Bits[2]]): Owned two-qubit input state.
        theta (qm.Real): Shared variational rotation angle in radians.

    Returns:
        qm.Q[qm.Bits[2]]: Entangled trial state for computational-basis sampling.
    """

    state = qm.ry(state, theta, target=0)
    state = qm.ry(state, theta, target=1)
    return qm.cx(state, control=0, target=1)


@qm.qkernel
def qsci_samples(
    theta: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Measure a runtime-parametric QSCI input state.

    Args:
        theta (qm.Param[qm.Real]): Runtime variational angle in radians.

    Returns:
        qm.Shot[qm.Bits[2]]: Sampled computational-basis configuration.
    """

    return qm.measure(qsci_input_state(qm.qubits(2), theta))


@qm.qkernel
def static_qsci_samples(
    theta: qm.Const[qm.Real],
) -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Measure the same QSCI state on static-only targets.

    Args:
        theta (qm.Const[qm.Real]): Static variational angle in radians.

    Returns:
        qm.Shot[qm.Bits[2]]: Sampled computational-basis configuration.
    """

    return qm.measure(qsci_input_state(qm.qubits(2), theta))


@qm.qkernel
def rabi_samples(
    hamiltonians: qm.Const[qm.Vector[qm.Observable, 2]],  # type: ignore[valid-type]
    time: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[1]]:  # type: ignore[valid-type]
    """Simulate and measure one runtime-timed Rabi oscillation.

    Args:
        hamiltonians (qm.Const[qm.Vector[qm.Observable, 2]]): Ordered static
            drift and drive Hamiltonian terms.
        time (qm.Param[qm.Real]): Total runtime evolution time.

    Returns:
        qm.Shot[qm.Bits[1]]: Measured excitation-state bit.
    """

    state = _RABI_EVOLUTION(qm.qubits(1), hamiltonians, time)
    return qm.measure(state)


@qm.qkernel
def static_rabi_samples(
    hamiltonians: qm.Const[qm.Vector[qm.Observable, 2]],  # type: ignore[valid-type]
) -> qm.Shot[qm.Bits[1]]:  # type: ignore[valid-type]
    """Simulate the same Rabi oscillation on static-only targets.

    Args:
        hamiltonians (qm.Const[qm.Vector[qm.Observable, 2]]): Ordered static
            drift and drive Hamiltonian terms.

    Returns:
        qm.Shot[qm.Bits[1]]: Measured excitation-state bit.
    """

    state = _RABI_EVOLUTION(qm.qubits(1), hamiltonians, _RABI_TIME)
    return qm.measure(state)


@qm.qkernel
def mottonen_state(
    ry_angles: qm.Param[qm.Vector[qm.Real, 3]],  # type: ignore[valid-type]
    rz_angles: qm.Param[qm.Vector[qm.Real, 3]],  # type: ignore[valid-type]
    phase: qm.Param[qm.Real],
) -> qm.Q[qm.Bits[2]]:  # type: ignore[valid-type]
    """Prepare a runtime-rebindable two-qubit Möttönen state.

    Args:
        ry_angles (qm.Param[qm.Vector[qm.Real, 3]]): Magnitude-cascade angles.
        rz_angles (qm.Param[qm.Vector[qm.Real, 3]]): Relative-phase angles.
        phase (qm.Param[qm.Real]): Retained global phase.

    Returns:
        qm.Q[qm.Bits[2]]: Prepared two-qubit state.
    """

    return _MOTTONEN_RUNTIME(qm.qubits(2), ry_angles, rz_angles, phase)


@qm.qkernel
def mottonen_samples(
    ry_angles: qm.Param[qm.Vector[qm.Real, 3]],  # type: ignore[valid-type]
    rz_angles: qm.Param[qm.Vector[qm.Real, 3]],  # type: ignore[valid-type]
    phase: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Measure a runtime-rebindable two-qubit Möttönen state.

    Args:
        ry_angles (qm.Param[qm.Vector[qm.Real, 3]]): Magnitude-cascade angles.
        rz_angles (qm.Param[qm.Vector[qm.Real, 3]]): Relative-phase angles.
        phase (qm.Param[qm.Real]): Retained global phase.

    Returns:
        qm.Shot[qm.Bits[2]]: Sampled computational-basis state.
    """

    return qm.measure(mottonen_state(ry_angles, rz_angles, phase))


@qm.qkernel
def static_mottonen_state() -> qm.Q[qm.Bits[2]]:  # type: ignore[valid-type]
    """Prepare the same documented Möttönen state with static coordinates.

    Returns:
        qm.Q[qm.Bits[2]]: Prepared two-qubit state.
    """

    return _MOTTONEN_STATIC(qm.qubits(2))


@qm.qkernel
def static_mottonen_samples() -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Measure the same documented statically prepared state.

    Returns:
        qm.Shot[qm.Bits[2]]: Sampled computational-basis state.
    """

    return qm.measure(static_mottonen_state())


@qm.qkernel
def multidimensional_qft_state(
    ry_angles: qm.Param[qm.Vector[qm.Real, 15]],  # type: ignore[valid-type]
    rz_angles: qm.Param[qm.Vector[qm.Real, 15]],  # type: ignore[valid-type]
    phase: qm.Param[qm.Real],
) -> qm.Q[qm.Bits[4]]:  # type: ignore[valid-type]
    """Prepare data and apply a QFT independently along two grid axes.

    Args:
        ry_angles (qm.Param[qm.Vector[qm.Real, 15]]): Magnitude-cascade angles.
        rz_angles (qm.Param[qm.Vector[qm.Real, 15]]): Relative-phase angles.
        phase (qm.Param[qm.Real]): Retained global phase.

    Returns:
        qm.Q[qm.Bits[4]]: Two-axis reciprocal-space state.
    """

    state = _MULTIDIMENSIONAL_QFT_RUNTIME(
        qm.qubits(4),
        ry_angles,
        rz_angles,
        phase,
    )
    x_axis, y_axis = qm.split_at(state, 2)
    x_axis = qft(x_axis, 2)
    y_axis = qft(y_axis, 2)
    return qm.concat(x_axis, y_axis)


@qm.qkernel
def multidimensional_qft_samples(
    ry_angles: qm.Param[qm.Vector[qm.Real, 15]],  # type: ignore[valid-type]
    rz_angles: qm.Param[qm.Vector[qm.Real, 15]],  # type: ignore[valid-type]
    phase: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[4]]:  # type: ignore[valid-type]
    """Measure a runtime-rebindable two-axis Fourier transform.

    Args:
        ry_angles (qm.Param[qm.Vector[qm.Real, 15]]): Magnitude-cascade angles.
        rz_angles (qm.Param[qm.Vector[qm.Real, 15]]): Relative-phase angles.
        phase (qm.Param[qm.Real]): Retained global phase.

    Returns:
        qm.Shot[qm.Bits[4]]: Sampled reciprocal-grid coordinate.
    """

    return qm.measure(multidimensional_qft_state(ry_angles, rz_angles, phase))


@qm.qkernel
def static_multidimensional_qft_state() -> qm.Q[qm.Bits[4]]:  # type: ignore[valid-type]
    """Apply the same two-axis QFT with statically closed input data.

    Returns:
        qm.Q[qm.Bits[4]]: Two-axis reciprocal-space state.
    """

    state = _MULTIDIMENSIONAL_QFT_STATIC(qm.qubits(4))
    x_axis, y_axis = qm.split_at(state, 2)
    x_axis = qft(x_axis, 2)
    y_axis = qft(y_axis, 2)
    return qm.concat(x_axis, y_axis)


@qm.qkernel
def static_multidimensional_qft_samples() -> qm.Shot[qm.Bits[4]]:  # type: ignore[valid-type]
    """Measure the same statically closed two-axis transform.

    Returns:
        qm.Shot[qm.Bits[4]]: Sampled reciprocal-grid coordinate.
    """

    return qm.measure(static_multidimensional_qft_state())


@qm.qkernel
def pce_ansatz(
    state: qm.Q[qm.Bits[2]],  # type: ignore[valid-type]
    angles: qm.Vector[qm.Real, 4],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[2]]:  # type: ignore[valid-type]
    """Prepare the documented two-qubit PCE trial state.

    Args:
        state (qm.Q[qm.Bits[2]]): Owned two-qubit input state.
        angles (qm.Vector[qm.Real, 4]): Y and Z angles for both qubits.

    Returns:
        qm.Q[qm.Bits[2]]: Product-state PCE ansatz.
    """

    state = qm.ry(state, angles[0], target=0)
    state = qm.rz(state, angles[1], target=0)
    state = qm.ry(state, angles[2], target=1)
    return qm.rz(state, angles[3], target=1)


@qm.qkernel
def pce_state(
    angles: qm.Param[qm.Vector[qm.Real, 4]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[2]]:  # type: ignore[valid-type]
    """Prepare one runtime-rebindable PCE trial state.

    Args:
        angles (qm.Param[qm.Vector[qm.Real, 4]]): Runtime ansatz angles.

    Returns:
        qm.Q[qm.Bits[2]]: Runtime-parametric PCE state.
    """

    return pce_ansatz(qm.qubits(2), angles)


@qm.qkernel
def static_pce_state(
    angles: qm.Const[qm.Vector[qm.Real, 4]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[2]]:  # type: ignore[valid-type]
    """Prepare the same PCE state with statically closed angles.

    Args:
        angles (qm.Const[qm.Vector[qm.Real, 4]]): Static ansatz angles.

    Returns:
        qm.Q[qm.Bits[2]]: Statically closed PCE state.
    """

    return pce_ansatz(qm.qubits(2), angles)


@qm.qkernel
def hqnn_ansatz(
    state: qm.Q[qm.Bits[4]],  # type: ignore[valid-type]
    inputs: qm.Vector[qm.Real, 4],  # type: ignore[valid-type]
    weights_z1: qm.Vector[qm.Real, 4],  # type: ignore[valid-type]
    weights_y: qm.Vector[qm.Real, 4],  # type: ignore[valid-type]
    weights_z2: qm.Vector[qm.Real, 4],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[4]]:  # type: ignore[valid-type]
    """Apply one documented Hybrid QNN quantum layer.

    Args:
        state (qm.Q[qm.Bits[4]]): Owned four-qubit input state.
        inputs (qm.Vector[qm.Real, 4]): Classical feature angles.
        weights_z1 (qm.Vector[qm.Real, 4]): First trainable Z rotations.
        weights_y (qm.Vector[qm.Real, 4]): Trainable Y rotations.
        weights_z2 (qm.Vector[qm.Real, 4]): Final trainable Z rotations.

    Returns:
        qm.Q[qm.Bits[4]]: Variational feature state.
    """

    for index in range(4):
        state = qm.ry(state, inputs[index], target=index)
    for index in range(4):
        state = qm.rz(state, weights_z1[index], target=index)
        state = qm.ry(state, weights_y[index], target=index)
        state = qm.rz(state, weights_z2[index], target=index)
    for index in range(3):
        state = qm.cz(state, control=index, target=index + 1)
    return state


@qm.qkernel
def hqnn_samples(
    inputs: qm.Param[qm.Vector[qm.Real, 4]],  # type: ignore[valid-type]
    weights_z1: qm.Param[qm.Vector[qm.Real, 4]],  # type: ignore[valid-type]
    weights_y: qm.Param[qm.Vector[qm.Real, 4]],  # type: ignore[valid-type]
    weights_z2: qm.Param[qm.Vector[qm.Real, 4]],  # type: ignore[valid-type]
) -> qm.Shot[qm.Bits[4]]:  # type: ignore[valid-type]
    """Measure one runtime-rebindable Hybrid QNN layer.

    Args:
        inputs (qm.Param[qm.Vector[qm.Real, 4]]): Runtime feature angles.
        weights_z1 (qm.Param[qm.Vector[qm.Real, 4]]): First Z weights.
        weights_y (qm.Param[qm.Vector[qm.Real, 4]]): Y weights.
        weights_z2 (qm.Param[qm.Vector[qm.Real, 4]]): Final Z weights.

    Returns:
        qm.Shot[qm.Bits[4]]: Sampled quantum feature bits.
    """

    state = hqnn_ansatz(
        qm.qubits(4),
        inputs,
        weights_z1,
        weights_y,
        weights_z2,
    )
    return qm.measure(state)


@qm.qkernel
def static_hqnn_samples(
    inputs: qm.Const[qm.Vector[qm.Real, 4]],  # type: ignore[valid-type]
    weights_z1: qm.Const[qm.Vector[qm.Real, 4]],  # type: ignore[valid-type]
    weights_y: qm.Const[qm.Vector[qm.Real, 4]],  # type: ignore[valid-type]
    weights_z2: qm.Const[qm.Vector[qm.Real, 4]],  # type: ignore[valid-type]
) -> qm.Shot[qm.Bits[4]]:  # type: ignore[valid-type]
    """Measure the same Hybrid QNN layer with statically closed inputs.

    Args:
        inputs (qm.Const[qm.Vector[qm.Real, 4]]): Static feature angles.
        weights_z1 (qm.Const[qm.Vector[qm.Real, 4]]): First Z weights.
        weights_y (qm.Const[qm.Vector[qm.Real, 4]]): Y weights.
        weights_z2 (qm.Const[qm.Vector[qm.Real, 4]]): Final Z weights.

    Returns:
        qm.Shot[qm.Bits[4]]: Sampled quantum feature bits.
    """

    state = hqnn_ansatz(
        qm.qubits(4),
        inputs,
        weights_z1,
        weights_y,
        weights_z2,
    )
    return qm.measure(state)


@qm.qkernel
def option_pricing_prepare(
    state: qm.Q[qm.Bits[2]],  # type: ignore[valid-type]
    distribution_ry: qm.Param[qm.Vector[qm.Real, 1]],  # type: ignore[valid-type]
    distribution_rz: qm.Param[qm.Vector[qm.Real, 1]],  # type: ignore[valid-type]
    distribution_phase: qm.Param[qm.Real],
    payoff_angles: qm.Param[qm.Vector[qm.Real, 2]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[2]]:  # type: ignore[valid-type]
    """Prepare a two-point call-option payoff amplitude.

    Args:
        state (qm.Q[qm.Bits[2]]): Asset-address and payoff qubits.
        distribution_ry (qm.Param[qm.Vector[qm.Real, 1]]): Runtime magnitude
            coordinate for the discretized risk-neutral distribution.
        distribution_rz (qm.Param[qm.Vector[qm.Real, 1]]): Runtime relative
            phase coordinate for the distribution.
        distribution_phase (qm.Param[qm.Real]): Runtime global phase coordinate.
        payoff_angles (qm.Param[qm.Vector[qm.Real, 2]]): Runtime Gray-walk
            coordinates for the normalized call payoff.

    Returns:
        qm.Q[qm.Bits[2]]: State whose final qubit encodes expected payoff.
    """

    asset, payoff = qm.split_at(state, 1)
    asset = _OPTION_DISTRIBUTION_RUNTIME(
        asset,
        distribution_ry,
        distribution_rz,
        distribution_phase,
    )
    state = qm.concat(asset, payoff)
    return _OPTION_PAYOFF_RUNTIME(state, payoff_angles)


@qm.qkernel
def option_pricing_samples(
    distribution_ry: qm.Param[qm.Vector[qm.Real, 1]],  # type: ignore[valid-type]
    distribution_rz: qm.Param[qm.Vector[qm.Real, 1]],  # type: ignore[valid-type]
    distribution_phase: qm.Param[qm.Real],
    payoff_angles: qm.Param[qm.Vector[qm.Real, 2]],  # type: ignore[valid-type]
    iterations: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Sample a discretized call option after amplitude amplification.

    Args:
        distribution_ry (qm.Param[qm.Vector[qm.Real, 1]]): Runtime magnitude
            coordinate for the asset-price distribution.
        distribution_rz (qm.Param[qm.Vector[qm.Real, 1]]): Runtime relative
            phase coordinate for the distribution.
        distribution_phase (qm.Param[qm.Real]): Runtime global phase coordinate.
        payoff_angles (qm.Param[qm.Vector[qm.Real, 2]]): Runtime call-payoff
            multiplexer coordinates.
        iterations (qm.Const[qm.Int]): Number of Grover amplification steps.

    Returns:
        qm.Shot[qm.Bits[2]]: Asset-address and payoff measurement.
    """

    state = option_pricing_prepare(
        qm.qubits(2),
        distribution_ry,
        distribution_rz,
        distribution_phase,
        payoff_angles,
    )
    for _ in range(iterations):
        state = qm.z(state, target=1)
        state = qm.adjoint(
            option_pricing_prepare,
            state,
            distribution_ry,
            distribution_rz,
            distribution_phase,
            payoff_angles,
        )
        state = qm.x(state, target=0)
        state = qm.x(state, target=1)
        state = qm.cz(state, control=0, target=1)
        state = qm.x(state, target=0)
        state = qm.x(state, target=1)
        state = option_pricing_prepare(
            state,
            distribution_ry,
            distribution_rz,
            distribution_phase,
            payoff_angles,
        )
    return qm.measure(state)


@qm.qkernel
def static_option_pricing_prepare(
    state: qm.Q[qm.Bits[2]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[2]]:  # type: ignore[valid-type]
    """Prepare the same call-option amplitude from closed data.

    Args:
        state (qm.Q[qm.Bits[2]]): Asset-address and payoff qubits.

    Returns:
        qm.Q[qm.Bits[2]]: State whose final qubit encodes expected payoff.
    """

    asset, payoff = qm.split_at(state, 1)
    asset = _OPTION_DISTRIBUTION_STATIC(asset)
    return _OPTION_PAYOFF_STATIC(qm.concat(asset, payoff))


@qm.qkernel
def static_option_pricing_samples(
    iterations: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Sample the statically closed call option after amplification.

    Args:
        iterations (qm.Const[qm.Int]): Number of Grover amplification steps.

    Returns:
        qm.Shot[qm.Bits[2]]: Asset-address and payoff measurement.
    """

    state = static_option_pricing_prepare(qm.qubits(2))
    for _ in range(iterations):
        state = qm.z(state, target=1)
        state = qm.adjoint(static_option_pricing_prepare, state)
        state = qm.x(state, target=0)
        state = qm.x(state, target=1)
        state = qm.cz(state, control=0, target=1)
        state = qm.x(state, target=0)
        state = qm.x(state, target=1)
        state = static_option_pricing_prepare(state)
    return qm.measure(state)


@qm.qkernel
def qemcmc_samples(
    input_angles: qm.Param[qm.Vector[qm.Real, 2]],  # type: ignore[valid-type]
    hamiltonians: qm.Const[qm.Vector[qm.Observable, 2]],  # type: ignore[valid-type]
    time: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Sample one runtime-rebindable quantum-enhanced MCMC proposal.

    Args:
        input_angles (qm.Param[qm.Vector[qm.Real, 2]]): Zero-or-pi rotations
            encoding the current Markov-chain bitstring.
        hamiltonians (qm.Const[qm.Vector[qm.Observable, 2]]): Mixer and Ising
            cost Hamiltonian terms.
        time (qm.Param[qm.Real]): Total proposal evolution time.

    Returns:
        qm.Shot[qm.Bits[2]]: Proposed two-spin bitstring.
    """

    state = qm.qubits(2)
    for index in range(2):
        state = qm.rx(state, input_angles[index], target=index)
    state = _QEMCMC_EVOLUTION(state, hamiltonians, time)
    return qm.measure(state)


@qm.qkernel
def static_qemcmc_samples(
    hamiltonians: qm.Const[qm.Vector[qm.Observable, 2]],  # type: ignore[valid-type]
) -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Sample the same quantum-enhanced MCMC proposal from closed inputs.

    Args:
        hamiltonians (qm.Const[qm.Vector[qm.Observable, 2]]): Mixer and Ising
            cost Hamiltonian terms.

    Returns:
        qm.Shot[qm.Bits[2]]: Proposed two-spin bitstring.
    """

    state = _QEMCMC_EVOLUTION(
        qm.prepare(_QEMCMC_INPUT_BITS),
        hamiltonians,
        _QEMCMC_TIME,
    )
    return qm.measure(state)


@qm.qkernel
def bit_flip_code_state(
    state: qm.Q[qm.Bits[5]],  # type: ignore[valid-type]
    theta: qm.Real,
    error_position: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[5]]:  # type: ignore[valid-type]
    """Encode, diagnose, and coherently correct one bit-flip error.

    Args:
        state (qm.Q[qm.Bits[5]]): Three data and two clean syndrome qubits.
        theta (qm.Real): Logical-state Y rotation angle in radians.
        error_position (qm.Const[qm.Int]): Erroneous data-qubit index.

    Returns:
        qm.Q[qm.Bits[5]]: Decoded logical qubit, clean data, and syndrome bits.
    """

    state = qm.ry(state, theta, target=0)
    state = qm.cx(state, control=0, target=1)
    state = qm.cx(state, control=0, target=2)
    state = qm.x(state, target=error_position)

    state = qm.cx(state, control=0, target=3)
    state = qm.cx(state, control=1, target=3)
    state = qm.cx(state, control=0, target=4)
    state = qm.cx(state, control=2, target=4)

    state = qm.ccx(state, control1=3, control2=4, target=0)
    state = qm.x(state, target=4)
    state = qm.ccx(state, control1=3, control2=4, target=1)
    state = qm.x(state, target=4)
    state = qm.x(state, target=3)
    state = qm.ccx(state, control1=3, control2=4, target=2)
    state = qm.x(state, target=3)

    state = qm.cx(state, control=0, target=2)
    return qm.cx(state, control=0, target=1)


@qm.qkernel
def bit_flip_code_samples(
    theta: qm.Param[qm.Real],
    error_position: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[5]]:  # type: ignore[valid-type]
    """Sample the runtime-parametric three-qubit bit-flip code.

    Args:
        theta (qm.Param[qm.Real]): Runtime logical-state angle in radians.
        error_position (qm.Const[qm.Int]): Erroneous data-qubit index.

    Returns:
        qm.Shot[qm.Bits[5]]: Logical, clean-data, and syndrome bits.
    """

    return qm.measure(bit_flip_code_state(qm.qubits(5), theta, error_position))


@qm.qkernel
def static_bit_flip_code_samples(
    error_position: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[5]]:  # type: ignore[valid-type]
    """Sample the same bit-flip code with a closed logical input.

    Args:
        error_position (qm.Const[qm.Int]): Erroneous data-qubit index.

    Returns:
        qm.Shot[qm.Bits[5]]: Logical, clean-data, and syndrome bits.
    """

    state = bit_flip_code_state(qm.qubits(5), _BIT_FLIP_THETA, error_position)
    return qm.measure(state)


@qm.qkernel
def steane_encode_zero(
    state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
    """Encode the final seven components as the Steane logical zero.

    Args:
        state (qm.Q[qm.Bits[10]]): Three syndrome and seven data qubits.

    Returns:
        qm.Q[qm.Bits[10]]: State with components 3 through 9 encoded as logical zero.
    """

    state = qm.h(state, target=6)
    for target in (7, 8, 9):
        state = qm.cx(state, control=6, target=target)
    state = qm.h(state, target=4)
    for target in (5, 8, 9):
        state = qm.cx(state, control=4, target=target)
    state = qm.h(state, target=3)
    for target in (5, 7, 9):
        state = qm.cx(state, control=3, target=target)
    return state


@qm.qkernel
def steane_extract_bit_syndrome(
    state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
    """Extract one Steane bit-error syndrome coherently.

    Args:
        state (qm.Q[qm.Bits[10]]): Encoded data and three clean ancillas.

    Returns:
        qm.Q[qm.Bits[10]]: Data with the syndrome in components 0 through 2.
    """

    for data in (3, 4, 5, 6):
        state = qm.cx(state, control=data + 3, target=0)
    for data in (1, 2, 5, 6):
        state = qm.cx(state, control=data + 3, target=1)
    for data in (0, 2, 4, 6):
        state = qm.cx(state, control=data + 3, target=2)

    return state


@qm.qkernel
def steane_extract_phase_syndrome(
    state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
    """Extract one Steane phase-error syndrome coherently.

    Args:
        state (qm.Q[qm.Bits[10]]): Encoded data and three clean ancillas.

    Returns:
        qm.Q[qm.Bits[10]]: Data with the syndrome in components 0 through 2.
    """

    state = qm.h(state, target=0)
    for data in (3, 4, 5, 6):
        state = qm.cx(state, control=0, target=data + 3)
    state = qm.h(state, target=0)
    state = qm.h(state, target=1)
    for data in (1, 2, 5, 6):
        state = qm.cx(state, control=1, target=data + 3)
    state = qm.h(state, target=1)
    state = qm.h(state, target=2)
    for data in (0, 2, 4, 6):
        state = qm.cx(state, control=2, target=data + 3)
    state = qm.h(state, target=2)
    return state


def _steane_syndrome_correction(
    target: int,
    *,
    pauli: int,
) -> qm.QKernel:
    """Construct one coherent three-bit syndrome correction.

    Args:
        target (int): Data component corrected by this syndrome value.
        pauli (int): Pauli selector: one X, two Y, three Z.

    Returns:
        qm.QKernel: Unary correction with its syndrome pattern closed in.
    """

    syndrome_value = target + 1
    zero_controls = tuple(
        offset for offset, mask in enumerate((4, 2, 1)) if syndrome_value & mask == 0
    )

    if pauli == 1:

        @qm.qkernel
        def correct_data(
            data: qm.Q[qm.Bits[7]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[7]]:  # type: ignore[valid-type]
            return qm.x(data, target=target)

    elif pauli == 2:

        @qm.qkernel
        def correct_data(
            data: qm.Q[qm.Bits[7]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[7]]:  # type: ignore[valid-type]
            return qm.y(data, target=target)

    else:

        @qm.qkernel
        def correct_data(
            data: qm.Q[qm.Bits[7]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[7]]:  # type: ignore[valid-type]
            return qm.z(data, target=target)

    if not zero_controls:

        @qm.qkernel
        def active_correction(
            state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
            syndrome, data = qm.split_at(state, 3)
            syndrome, data = qm.control(correct_data, syndrome, data)
            return qm.concat(syndrome, data)

        return active_correction

    @qm.qkernel
    def correct(
        state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
        syndrome, data = qm.split_at(state, 3)
        for control in zero_controls:
            syndrome = qm.x(syndrome, target=control)
        syndrome, data = qm.control(correct_data, syndrome, data)
        for control in zero_controls:
            syndrome = qm.x(syndrome, target=control)
        return qm.concat(syndrome, data)

    return correct


def _steane_correction(pauli: int) -> qm.QKernel:
    """Compose all seven nonzero syndrome corrections.

    Args:
        pauli (int): Pauli selector: one X, two Y, three Z.

    Returns:
        qm.QKernel: Total coherent correction for one Pauli component.
    """

    @qm.qkernel
    def identity(
        state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
        return state

    current = identity
    for target in range(7):
        previous = current
        correction = _steane_syndrome_correction(target, pauli=pauli)

        @qm.qkernel
        def append_correction(
            state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
            return correction(previous(state))

        current = append_correction
    return current


_STEANE_X_CORRECTION = _steane_correction(1)
_STEANE_Y_CORRECTION = _steane_correction(2)
_STEANE_Z_CORRECTION = _steane_correction(3)


@qm.qkernel
def steane_code_samples(
    error_kind: qm.Const[qm.Int],
    error_position: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[10]]:  # type: ignore[valid-type]
    """Correct one statically selected Pauli error in the Steane code.

    Args:
        error_kind (qm.Const[qm.Int]): Pauli selector: one X, two Y, three Z.
        error_position (qm.Const[qm.Int]): Erroneous data-qubit index.

    Returns:
        qm.Shot[qm.Bits[10]]: Corrected codeword and retained syndrome.
    """

    state = steane_encode_zero(qm.qubits(10))
    if error_kind == 1:
        state = qm.x(state, target=error_position + 3)
        state = steane_extract_bit_syndrome(state)
        state = _STEANE_X_CORRECTION(state)
    if error_kind == 2:
        state = qm.y(state, target=error_position + 3)
        state = steane_extract_bit_syndrome(state)
        state = _STEANE_Y_CORRECTION(state)
    if error_kind == 3:
        state = qm.z(state, target=error_position + 3)
        state = steane_extract_phase_syndrome(state)
        state = _STEANE_Z_CORRECTION(state)
    return qm.measure(state)


def exercise_classification_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
    cases: tuple[
        tuple[tuple[float, float], tuple[float, float], tuple[int, int]], ...
    ] = (_CLASSIFICATION_CASES[1],),
) -> None:
    """Execute the documented overlap kernel through one backend endpoint.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
        cases: Feature pairs and expected bit strings to execute.
    """

    for left, right, expected in cases:
        result = qm.run(
            classification_overlap,
            on=endpoint,
            parameters={"left": left, "right": right},
            shots=4,
            seed=seed,
        ).result

        assert isinstance(result, SampleResult)
        assert result.counts == {expected: 4}


def exercise_static_classification_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
    cases: tuple[
        tuple[tuple[float, float], tuple[float, float], tuple[int, int]], ...
    ] = (_CLASSIFICATION_CASES[1],),
) -> None:
    """Execute a statically closed overlap kernel through one endpoint.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
        cases: Feature pairs and expected bit strings to execute.
    """

    for left, right, expected in cases:
        result = qm.run(
            static_classification_overlap,
            on=endpoint,
            **{"left": left, "right": right},
            shots=4,
            seed=seed,
        ).result

        assert isinstance(result, SampleResult)
        assert result.counts == {expected: 4}


def exercise_hydrogen_vqe_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Estimate the documented hydrogen energy through one backend.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    result = qm.expectation(
        hydrogen_vqe_state,
        _HYDROGEN_HAMILTONIAN,
        on=endpoint,
        parameters={"theta": _HYDROGEN_ANGLE},
        shots=_EXPECTATION_SHOTS,
        seed=seed,
    )

    assert result.backend == endpoint.name
    assert len(result.terms) == 4
    assert result.shots == _EXPECTATION_SHOTS
    assert abs(result.value - _hydrogen_energy(_HYDROGEN_ANGLE)) <= (
        6 * result.standard_error + 1e-12
    )


def exercise_static_hydrogen_vqe_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Estimate a closed hydrogen trial state on a static-only target.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    result = qm.expectation(
        static_hydrogen_vqe_state,
        _HYDROGEN_HAMILTONIAN,
        on=endpoint,
        **{"theta": _HYDROGEN_ANGLE},
        shots=_EXPECTATION_SHOTS,
        seed=seed,
    )

    assert result.backend == endpoint.name
    assert len(result.terms) == 4
    assert result.shots == _EXPECTATION_SHOTS
    assert abs(result.value - _hydrogen_energy(_HYDROGEN_ANGLE)) <= (
        6 * result.standard_error + 1e-12
    )


def exercise_qsci_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Sample the documented QSCI input through one backend.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    result = qm.run(
        qsci_samples,
        on=endpoint,
        parameters={"theta": _QSCI_ANGLE},
        shots=64,
        seed=seed,
    ).result

    assert isinstance(result, SampleResult)
    assert result.backend == endpoint.name
    assert sum(result.counts.values()) == 64
    assert set(result.counts) == {(0, 0), (1, 0), (0, 1), (1, 1)}


def exercise_static_qsci_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Sample a closed QSCI input through one static-only backend.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    result = qm.run(
        static_qsci_samples,
        on=endpoint,
        **{"theta": _QSCI_ANGLE},
        shots=64,
        seed=seed,
    ).result

    assert isinstance(result, SampleResult)
    assert result.backend == endpoint.name
    assert sum(result.counts.values()) == 64
    assert set(result.counts) == {(0, 0), (1, 0), (0, 1), (1, 1)}


def exercise_rabi_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Execute documented Rabi simulation through one backend.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    result = qm.run(
        rabi_samples,
        on=endpoint,
        **{"hamiltonians": _RABI_TERMS},
        parameters={"time": _RABI_TIME},
        shots=_STATISTICAL_SHOTS,
        seed=seed,
    ).result

    _assert_rabi_result(result, endpoint)


def exercise_static_rabi_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Execute closed documented Rabi simulation on a static-only target.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    result = qm.run(
        static_rabi_samples,
        on=endpoint,
        **{"hamiltonians": _RABI_TERMS},
        shots=_STATISTICAL_SHOTS,
        seed=seed,
    ).result

    _assert_rabi_result(result, endpoint)


def _assert_rabi_result(result: object, endpoint: ExecutionEndpoint) -> None:
    """Check sampled excitation probability against the exact Rabi solution.

    Args:
        result (object): Backend execution result to validate.
        endpoint (ExecutionEndpoint): Endpoint that produced the result.
    """

    assert isinstance(result, SampleResult)
    expected = _rabi_excitation_probability()
    sampling_error = 6.0 * math.sqrt(expected * (1.0 - expected) / result.shots)

    assert result.backend == endpoint.name
    assert result.shots == _STATISTICAL_SHOTS
    assert abs(result.probability((1,)) - expected) <= sampling_error + 0.01


def _rabi_excitation_probability() -> float:
    """Return the exact one-qubit Rabi excitation probability.

    Returns:
        float: Probability of measuring the evolved state as one.
    """

    energy = math.hypot(_RABI_OMEGA, _RABI_DRIVE)
    return (_RABI_DRIVE / energy) ** 2 * math.sin(energy * _RABI_TIME / 2.0) ** 2


def exercise_mottonen_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Execute documented runtime Möttönen preparation on one backend.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    parameters = _MOTTONEN_COORDINATES.bindings()
    samples = qm.run(
        mottonen_samples,
        on=endpoint,
        parameters=parameters,
        shots=_STATISTICAL_SHOTS,
        seed=seed,
    ).result
    phase = qm.expectation(
        mottonen_state,
        qm.Y(0) * qm.X(1),
        on=endpoint,
        parameters=parameters,
        shots=_EXPECTATION_SHOTS,
        seed=seed,
    )

    _assert_mottonen_result(samples, phase, endpoint)


def exercise_static_mottonen_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Execute documented static Möttönen preparation on one backend.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    samples = qm.run(
        static_mottonen_samples,
        on=endpoint,
        shots=_STATISTICAL_SHOTS,
        seed=seed,
    ).result
    phase = qm.expectation(
        static_mottonen_state,
        qm.Y(0) * qm.X(1),
        on=endpoint,
        shots=_EXPECTATION_SHOTS,
        seed=seed,
    )

    _assert_mottonen_result(samples, phase, endpoint)


def _assert_mottonen_result(
    samples: object,
    phase: qm.ExpectationResult,
    endpoint: ExecutionEndpoint,
) -> None:
    """Check documented magnitudes and one phase-sensitive observable.

    Args:
        samples (object): Backend sampling result to validate.
        phase (qm.ExpectationResult): Estimate of the phase-sensitive witness.
        endpoint (ExecutionEndpoint): Endpoint that produced both results.
    """

    assert isinstance(samples, SampleResult)
    probabilities = {
        (0, 0): 1.0 / 9.0,
        (1, 0): 2.0 / 9.0,
        (0, 1): 2.0 / 9.0,
        (1, 1): 4.0 / 9.0,
    }

    assert samples.backend == endpoint.name
    assert phase.backend == endpoint.name
    for bits, expected in probabilities.items():
        sampling_error = 6.0 * math.sqrt(expected * (1.0 - expected) / samples.shots)
        assert abs(samples.probability(bits) - expected) <= sampling_error
    expected_phase = 8.0 / 9.0
    sampling_error = 6.0 * math.sqrt(
        (1.0 - expected_phase * expected_phase) / _EXPECTATION_SHOTS
    )
    assert abs(phase.value - expected_phase) <= sampling_error + 1e-12


def exercise_multidimensional_qft_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Execute documented runtime two-axis QFT on one backend.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    result = qm.run(
        multidimensional_qft_samples,
        on=endpoint,
        parameters=_MULTIDIMENSIONAL_QFT_COORDINATES.bindings(),
        shots=32,
        seed=seed,
    ).result

    _assert_multidimensional_qft_result(result, endpoint)


def exercise_static_multidimensional_qft_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Execute documented static two-axis QFT on one backend.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    result = qm.run(
        static_multidimensional_qft_samples,
        on=endpoint,
        shots=32,
        seed=seed,
    ).result

    _assert_multidimensional_qft_result(result, endpoint)


def exercise_documented_windowed_qpe_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Execute the documented Kaiser-windowed QPE on one backend.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    result = qm.run(
        kaiser_windowed_qpe_sample,
        on=endpoint,
        shots=64,
        seed=seed,
    ).result

    assert result.backend == endpoint.name
    assert result.shots == 64
    assert result.most_likely_value == 3.0 / 16.0


def exercise_documented_reversible_arithmetic_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Execute the documented compare-swap and register product.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    sorted_result = qm.run(
        reversible_sort_descending_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result
    product_value = qm.expectation(
        reversible_register_product_boundary_state,
        qm.Z(2),
        on=endpoint,
        shots=8,
        seed=seed,
    ).value

    assert sorted_result.backend == endpoint.name
    assert sorted_result.counts == {(1, 1, 1, 0, 1, 0): 4}
    assert product_value == -1.0


def exercise_documented_gf2_arithmetic_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Execute the documented GF(2^3) polynomial product.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    result = qm.run(
        gf8_polynomial_product_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result

    assert result.backend == endpoint.name
    assert result.counts == {(1, 0, 1, 1, 1, 0, 0, 0, 1): 4}


def exercise_documented_phase_gradient_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Execute phase-gradient QVR and Hamming-weight phasing.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    qvr_result = qm.run(
        phase_gradient_qvr_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result
    weight_result = qm.run(
        hamming_weight_phase_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result

    assert qvr_result.backend == endpoint.name
    assert qvr_result.counts == {(1, 0, 1, 0, 0): 4}
    assert weight_result.backend == endpoint.name
    assert weight_result.counts == {(1, 1, 1, 0): 4}


def exercise_documented_fault_tolerant_data_loading_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Execute clean select-swap QROAM and one alias branch.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    qroam_result = qm.run(
        clean_qroam_sample,
        on=endpoint,
        shots=8,
        seed=seed,
    ).result
    alias_result = qm.run(
        coherent_alias_branch_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result
    table = (1, 3, 2, 0)

    assert qroam_result.backend == endpoint.name
    for bits in qroam_result.counts:
        address = bits[0] + 2 * bits[1]
        value = bits[6] + 2 * bits[7]
        assert bits[2:6] == (0, 0, 0, 0)
        assert value == table[address]
    assert alias_result.backend == endpoint.name
    assert alias_result.counts == {(0, 1, 0, 0, 0, 0, 0, 1, 1, 0): 4}


def exercise_documented_reversible_indexing_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Execute basis permutation and triangular contiguous indexing.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    permutation_result = qm.run(
        basis_permutation_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result
    index_result = qm.run(
        contiguous_index_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result

    assert permutation_result.backend == endpoint.name
    assert permutation_result.counts == {(1, 1): 4}
    assert index_result.backend == endpoint.name
    assert index_result.counts == {(1, 0, 0, 1, 0, 0, 1, 0, 0, 0): 4}


def exercise_documented_reversible_trigonometry_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Execute bounded reversible arcsine and arctangent maps.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    arcsin_result = qm.run(
        reversible_arcsin_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result
    arctan_result = qm.run(
        reversible_arctan_boundary_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result

    assert arcsin_result.backend == endpoint.name
    assert arcsin_result.counts == {(0, 0, 1, 0, 0, 0, 1, 0): 4}
    assert arctan_result.backend == endpoint.name
    assert arctan_result.counts == {(1, 1, 1, 0, 0, 1, 1): 4}


def exercise_documented_kikuchi_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Execute sorted-set and bounded Kikuchi-oracle witnesses.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    duplicate_result = qm.run(
        sorted_duplicate_boundary_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result
    difference_result = qm.run(
        symmetric_difference_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result
    entry_result = qm.run(
        kikuchi_entry_boundary_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result

    assert duplicate_result.backend == endpoint.name
    assert duplicate_result.counts == {(1, 1, 1, 0, 0): 4}
    assert difference_result.backend == endpoint.name
    assert difference_result.counts == {(0, 1, 0, 1, 1): 4}
    assert entry_result.backend == endpoint.name
    assert entry_result.counts == {(0, 1, 1, 1): 4}


def exercise_documented_guided_hamiltonian_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Execute the exact high-eigenspace guided-phase witness.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    result = qm.run(
        guided_hamiltonian_high_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result

    assert result.backend == endpoint.name
    assert result.counts == {(0, 1, 1): 4}
    assert result.most_likely_value == 0.375


def exercise_documented_elliptic_curve_dlog_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Execute the bounded elliptic-curve hidden-subgroup witness.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    result = qm.run(
        elliptic_curve_dlog_sample,
        on=endpoint,
        shots=16,
        seed=seed,
    ).result

    assert result.backend == endpoint.name
    assert result.shots == 16
    for frequency_p, frequency_q, _point_label in result.values:
        assert frequency_q == _DOCUMENTED_EC_SECRET * frequency_p % 4


def exercise_documented_factorized_chemistry_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Execute SF, DF, THC, first-quantized, and projectile witnesses.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    sf_result = qm.run(
        single_factorization_block_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result
    df_result = qm.run(
        double_factorization_basis_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result
    thc_result = qm.run(
        tensor_hypercontraction_sample,
        on=endpoint,
        shots=8,
        seed=seed,
    ).result
    first_quantized_result = qm.run(
        first_quantized_chemistry_sample,
        on=endpoint,
        hamiltonians=_DOCUMENTED_FIRST_QUANTIZED_TERMS,
        shots=4,
        seed=seed,
    ).result
    projectile_result = qm.run(
        quantum_projectile_sample,
        on=endpoint,
        hamiltonians=_DOCUMENTED_PROJECTILE_TERMS,
        shots=4,
        seed=seed,
    ).result

    assert sf_result.backend == endpoint.name
    assert sf_result.counts == {(0, 0, 0): 4}
    assert df_result.backend == endpoint.name
    assert df_result.counts == {(1, 1, 0): 4}
    assert thc_result.backend == endpoint.name
    for bits in thc_result.counts:
        assert bits[2] == bits[0] ^ bits[1]
    assert first_quantized_result.backend == endpoint.name
    assert first_quantized_result.counts == {(1, 0): 4}
    assert projectile_result.backend == endpoint.name
    assert projectile_result.counts == {(0, 1, 1): 4}


def _assert_multidimensional_qft_result(
    result: object,
    endpoint: ExecutionEndpoint,
) -> None:
    """Check the analytic support of the compact two-axis Fourier input.

    Args:
        result (object): Backend sampling result to validate.
        endpoint (ExecutionEndpoint): Endpoint that produced the result.
    """

    assert isinstance(result, SampleResult)
    observed_y = {
        sum(bit << index for index, bit in enumerate(bits[2:]))
        for bits in result.counts
    }

    assert result.backend == endpoint.name
    assert result.shots == 32
    assert all(bits[:2] == (0, 0) for bits in result.counts)
    assert observed_y == set(range(4))


def exercise_pce_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Evaluate and decode the documented runtime PCE state.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    estimates = tuple(
        qm.expectation(
            pce_state,
            observable,
            on=endpoint,
            parameters={"angles": _PCE_ANGLES},
            shots=_EXPECTATION_SHOTS,
            seed=0 if seed == 0 else seed + index,
        )
        for index, observable in enumerate(_PCE_ENCODING.observables)
    )
    _assert_pce_results(estimates, endpoint)


def exercise_static_pce_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Evaluate and decode the statically closed documented PCE state.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    estimates = tuple(
        qm.expectation(
            static_pce_state,
            observable,
            on=endpoint,
            **{"angles": _PCE_ANGLES},
            shots=_EXPECTATION_SHOTS,
            seed=0 if seed == 0 else seed + index,
        )
        for index, observable in enumerate(_PCE_ENCODING.observables)
    )
    _assert_pce_results(estimates, endpoint)


def _assert_pce_results(
    estimates: tuple[qm.ExpectationResult, ...],
    endpoint: ExecutionEndpoint,
) -> None:
    """Check PCE expectations and source-domain sign decoding.

    Args:
        estimates (tuple[qm.ExpectationResult, ...]): Correlator estimates.
        endpoint (ExecutionEndpoint): Endpoint that produced the estimates.
    """

    assert len(estimates) == len(_PCE_EXPECTATIONS)
    for estimate, expected in zip(estimates, _PCE_EXPECTATIONS, strict=True):
        assert estimate.backend == endpoint.name
        assert estimate.shots == _EXPECTATION_SHOTS
        assert abs(estimate.value - expected) <= 6.0 * estimate.standard_error + 1e-12
        assert math.copysign(1.0, estimate.value) == math.copysign(1.0, expected)

    decoded = _PCE_ENCODING.decode(tuple(estimate.value for estimate in estimates))
    assert decoded.lowest.assignment == ((0, 1), (1, -1), (2, 1), (3, -1))
    assert decoded.lowest.energy == -4.0


def exercise_hqnn_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Execute documented runtime Hybrid QNN inference on one backend.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    result = qm.run(
        hqnn_samples,
        on=endpoint,
        parameters=_hqnn_parameters(_HQNN_INPUTS),
        shots=_STATISTICAL_SHOTS,
        seed=seed,
    ).result

    _assert_hqnn_result(result, endpoint)


def exercise_static_hqnn_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Execute documented static Hybrid QNN inference on one backend.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    result = qm.run(
        static_hqnn_samples,
        on=endpoint,
        **_hqnn_parameters(_HQNN_INPUTS),
        shots=_STATISTICAL_SHOTS,
        seed=seed,
    ).result

    _assert_hqnn_result(result, endpoint)


def _hqnn_parameters(inputs: tuple[float, ...]) -> dict[str, object]:
    """Return the standard Hybrid QNN launch parameters.

    Args:
        inputs (tuple[float, ...]): Four feature angles.

    Returns:
        dict[str, object]: Runtime aggregate parameter bindings.
    """

    return {
        "inputs": inputs,
        "weights_z1": _HQNN_WEIGHTS_Z1,
        "weights_y": _HQNN_WEIGHTS_Y,
        "weights_z2": _HQNN_WEIGHTS_Z2,
    }


def _assert_hqnn_result(result: object, endpoint: ExecutionEndpoint) -> None:
    """Check the four analytic Z features of one Hybrid QNN inference.

    Args:
        result (object): Backend sampling result to validate.
        endpoint (ExecutionEndpoint): Endpoint that produced the result.
    """

    assert isinstance(result, SampleResult)
    expected = tuple(
        math.cos(feature) * math.cos(weight_y)
        - math.sin(feature) * math.sin(weight_y) * math.cos(weight_z)
        for feature, weight_y, weight_z in zip(
            _HQNN_INPUTS,
            _HQNN_WEIGHTS_Y,
            _HQNN_WEIGHTS_Z1,
            strict=True,
        )
    )

    assert result.backend == endpoint.name
    assert result.shots == _STATISTICAL_SHOTS
    for index, value in enumerate(expected):
        sampling_error = 6.0 * math.sqrt((1.0 - value * value) / result.shots)
        assert abs(result.expectation_z(index) - value) <= sampling_error + 1e-12


def exercise_option_pricing_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Execute runtime option pricing and one Grover amplification step.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    parameters = _option_pricing_parameters()
    baseline = qm.run(
        option_pricing_samples,
        on=endpoint,
        **{"iterations": 0},
        parameters=parameters,
        shots=_STATISTICAL_SHOTS,
        seed=seed,
    ).result
    amplified = qm.run(
        option_pricing_samples,
        on=endpoint,
        **{"iterations": 1},
        parameters=parameters,
        shots=4,
        seed=seed,
    ).result

    _assert_option_pricing_results(baseline, amplified, endpoint)


def exercise_static_option_pricing_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Execute the same option-pricing experiment with closed data.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    baseline = qm.run(
        static_option_pricing_samples,
        on=endpoint,
        **{"iterations": 0},
        shots=_STATISTICAL_SHOTS,
        seed=seed,
    ).result
    amplified = qm.run(
        static_option_pricing_samples,
        on=endpoint,
        **{"iterations": 1},
        shots=4,
        seed=seed,
    ).result

    _assert_option_pricing_results(baseline, amplified, endpoint)


def _option_pricing_parameters() -> dict[str, object]:
    """Return launch coordinates for the two-point call option.

    Returns:
        dict[str, object]: Runtime aggregate parameter bindings.
    """

    bindings = _OPTION_DISTRIBUTION_COORDINATES.bindings()
    return {
        "distribution_ry": bindings["ry_angles"],
        "distribution_rz": bindings["rz_angles"],
        "distribution_phase": bindings["phase"],
        "payoff_angles": _OPTION_PAYOFF_COORDINATES,
    }


def _assert_option_pricing_results(
    baseline: object,
    amplified: object,
    endpoint: ExecutionEndpoint,
) -> None:
    """Check expected payoff, discounted price, and Grover amplification.

    Args:
        baseline (object): Unamplified backend sample result.
        amplified (object): One-step amplified backend sample result.
        endpoint (ExecutionEndpoint): Endpoint that produced both results.
    """

    assert isinstance(baseline, SampleResult)
    assert isinstance(amplified, SampleResult)
    expected_payoff = 0.25
    observed_payoff = (
        sum(count for bits, count in baseline.counts.items() if bits[1] == 1)
        / baseline.shots
    )
    sampling_error = 6.0 * math.sqrt(
        expected_payoff * (1.0 - expected_payoff) / baseline.shots
    )
    discount = math.exp(-_OPTION_RATE * _OPTION_MATURITY)
    expected_price = discount * expected_payoff
    observed_price = discount * observed_payoff

    assert baseline.backend == endpoint.name
    assert amplified.backend == endpoint.name
    assert abs(observed_payoff - expected_payoff) <= sampling_error
    assert abs(observed_price - expected_price) <= discount * sampling_error
    assert amplified.counts == {(1, 1): 4}


def exercise_qemcmc_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Execute one runtime quantum-enhanced MCMC proposal on a backend.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    result = qm.run(
        qemcmc_samples,
        on=endpoint,
        **{"hamiltonians": _QEMCMC_HAMILTONIANS},
        parameters={
            "input_angles": _QEMCMC_INPUT_ANGLES,
            "time": _QEMCMC_TIME,
        },
        shots=_STATISTICAL_SHOTS,
        seed=seed,
    ).result

    _assert_qemcmc_result(result, endpoint)


def exercise_static_qemcmc_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
) -> None:
    """Execute the same closed quantum-enhanced MCMC proposal.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
    """

    result = qm.run(
        static_qemcmc_samples,
        on=endpoint,
        **{"hamiltonians": _QEMCMC_HAMILTONIANS},
        shots=_STATISTICAL_SHOTS,
        seed=seed,
    ).result

    _assert_qemcmc_result(result, endpoint)


def exercise_bit_flip_code_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
    error_positions: tuple[int, ...] = (1,),
) -> None:
    """Execute the runtime-parametric bit-flip code on one backend.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
        error_positions: Data-qubit positions to exercise.
    """

    for error_position in error_positions:
        result = qm.run(
            bit_flip_code_samples,
            on=endpoint,
            **{"error_position": error_position},
            parameters={"theta": _BIT_FLIP_THETA},
            shots=_EXPECTATION_SHOTS,
            seed=seed,
        ).result
        _assert_bit_flip_code_result(result, endpoint, error_position)


def exercise_static_bit_flip_code_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
    error_positions: tuple[int, ...] = (1,),
) -> None:
    """Execute the closed bit-flip code on one static-only backend.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
        error_positions: Data-qubit positions to exercise.
    """

    for error_position in error_positions:
        result = qm.run(
            static_bit_flip_code_samples,
            on=endpoint,
            **{"error_position": error_position},
            shots=_EXPECTATION_SHOTS,
            seed=seed,
        ).result
        _assert_bit_flip_code_result(result, endpoint, error_position)


def _assert_bit_flip_code_result(
    result: object,
    endpoint: ExecutionEndpoint,
    error_position: int,
) -> None:
    """Check logical-state recovery and the retained error syndrome.

    Args:
        result (object): Backend sampling result to validate.
        endpoint (ExecutionEndpoint): Endpoint that produced the result.
        error_position (int): Injected data-qubit error location.
    """

    assert isinstance(result, SampleResult)
    syndrome = _BIT_FLIP_SYNDROMES[error_position]
    assert result.backend == endpoint.name
    assert all(bits[1:3] == (0, 0) and bits[3:5] == syndrome for bits in result.counts)
    expected_one_probability = math.sin(_BIT_FLIP_THETA / 2.0) ** 2
    observed_one_probability = result.probability((1, 0, 0, *syndrome))
    sampling_error = 6.0 * math.sqrt(
        expected_one_probability * (1.0 - expected_one_probability) / result.shots
    )
    assert abs(observed_one_probability - expected_one_probability) <= sampling_error


def exercise_steane_code_backend(
    *,
    endpoint: ExecutionEndpoint,
    seed: int = 0,
    cases: tuple[tuple[int, int], ...] = ((2, 3),),
) -> None:
    """Execute bounded coherent Steane correction through one backend.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Deterministic sampling seed.
        cases (tuple[tuple[int, int], ...]): Pauli selectors and data positions.
    """

    for error_kind, error_position in cases:
        result = qm.run(
            steane_code_samples,
            on=endpoint,
            **{
                "error_kind": error_kind,
                "error_position": error_position,
            },
            shots=4,
            seed=seed,
        ).result
        _assert_steane_code_result(result, endpoint, error_position=error_position)


def _assert_steane_code_result(
    result: object,
    endpoint: ExecutionEndpoint,
    *,
    error_position: int,
) -> None:
    """Check the corrected codeword and retained three-bit syndrome.

    Args:
        result (object): Backend sampling result to validate.
        endpoint (ExecutionEndpoint): Endpoint that produced the result.
        error_position (int): Injected data-qubit error location.
    """

    assert isinstance(result, SampleResult)
    syndrome_value = error_position + 1
    syndrome = tuple(bool(syndrome_value & mask) for mask in (4, 2, 1))

    assert result.backend == endpoint.name
    assert result.shots == 4
    for bits in result.counts:
        assert bits[:3] == syndrome
        assert _is_steane_zero_word(bits[3:10])


def _is_steane_zero_word(bits: tuple[int, ...]) -> bool:
    """Return whether seven bits form a Steane logical-zero codeword.

    Args:
        bits (tuple[int, ...]): Seven little-endian data bits.

    Returns:
        bool: True for an even-weight word satisfying all Hamming checks.
    """

    return (
        len(bits) == 7
        and (bits[3] ^ bits[4] ^ bits[5] ^ bits[6]) == 0
        and (bits[1] ^ bits[2] ^ bits[5] ^ bits[6]) == 0
        and (bits[0] ^ bits[2] ^ bits[4] ^ bits[6]) == 0
        and sum(bits) % 2 == 0
    )


def _assert_qemcmc_result(result: object, endpoint: ExecutionEndpoint) -> None:
    """Check a sampled proposal against the exact palindromic product formula.

    Args:
        result (object): Backend sampling result to validate.
        endpoint (ExecutionEndpoint): Endpoint that produced the result.
    """

    assert isinstance(result, SampleResult)
    expected = _qemcmc_proposal_probabilities(_QEMCMC_INPUT_BITS)

    assert result.backend == endpoint.name
    assert result.shots == _STATISTICAL_SHOTS
    for basis, probability in enumerate(expected):
        bits = tuple((basis >> index) & 1 for index in range(2))
        sampling_error = 6.0 * math.sqrt(
            probability * (1.0 - probability) / result.shots
        )
        assert abs(result.probability(bits) - probability) <= sampling_error + 0.01


def _qemcmc_proposal_probabilities(
    input_bits: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Return the exact two-spin Suzuki proposal probabilities.

    Args:
        input_bits (tuple[int, int]): Little-endian current-state bitstring.

    Returns:
        tuple[float, float, float, float]: Address-ordered proposal
        probabilities.
    """

    basis_index = input_bits[0] | (input_bits[1] << 1)
    amplitudes = [0.0j] * 4
    amplitudes[basis_index] = 1.0 + 0.0j
    mixer_angle = (1.0 - _QEMCMC_GAMMA) * _QEMCMC_TIME / 2.0
    amplitudes = _apply_x_evolution(amplitudes, target=0, angle=mixer_angle)
    amplitudes = _apply_x_evolution(amplitudes, target=1, angle=mixer_angle)
    for basis in range(4):
        z0 = -1 if basis & 1 else 1
        z1 = -1 if basis & 2 else 1
        amplitudes[basis] *= cmath.exp(1.0j * _QEMCMC_GAMMA * _QEMCMC_TIME * z0 * z1)
    amplitudes = _apply_x_evolution(amplitudes, target=0, angle=mixer_angle)
    amplitudes = _apply_x_evolution(amplitudes, target=1, angle=mixer_angle)
    probabilities = [abs(amplitude) ** 2 for amplitude in amplitudes]
    return probabilities[0], probabilities[1], probabilities[2], probabilities[3]


def _apply_x_evolution(
    amplitudes: list[complex],
    *,
    target: int,
    angle: float,
) -> list[complex]:
    """Apply ``exp(-i angle X_target)`` to a two-qubit host vector.

    Args:
        amplitudes (list[complex]): Address-ordered input amplitudes.
        target (int): Little-endian target index.
        angle (float): Pauli-evolution angle.

    Returns:
        list[complex]: Evolved amplitudes.
    """

    cosine = math.cos(angle)
    sine = -1.0j * math.sin(angle)
    return [
        cosine * amplitudes[basis] + sine * amplitudes[basis ^ (1 << target)]
        for basis in range(4)
    ]


def _ising_energy(bits: tuple[int, int]) -> float:
    """Return the two-spin ferromagnetic Ising energy.

    Args:
        bits (tuple[int, int]): Little-endian binary spin encoding.

    Returns:
        float: Energy ``-s0 * s1`` for spins ``s = 1 - 2 * bit``.
    """

    return -float((1 - 2 * bits[0]) * (1 - 2 * bits[1]))


def _hydrogen_energy(theta: float) -> float:
    """Return the exact compact-H2 energy for one ansatz coordinate.

    The trial state has Pauli expectations ``<Z0> = -cos(theta)``,
    ``<Z1> = cos(theta)``, ``<Z0 Z1> = -1``, and
    ``<X0 X1> = sin(theta)``. Substitution into the Hamiltonian above gives
    the expression returned here.

    Args:
        theta (float): Variational angle in radians.

    Returns:
        float: Exact energy of the trial state.
    """

    return (
        -1.052373245772859
        + 0.01128010425623538
        - 2 * 0.39793742484318045 * math.cos(theta)
        + 0.18093119978423156 * math.sin(theta)
    )


def test_quantum_kernel_classification_is_ordinary_function_composition() -> None:
    """The documented classifier kernel needs no application-specific IR."""

    exercise_classification_backend(
        endpoint=qm.Reference(),
        seed=2,
        cases=_CLASSIFICATION_CASES,
    )
    inspection = classification_overlap.module().inspect()
    study = qm.resources.logical(classification_overlap)

    assert "qcore.adjoint_call" in inspection
    assert all(
        operation.opcode != "quantum.classification"
        for function in classification_overlap.module().functions
        for operation in function.body.operations
    )
    assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (12, 12)


def test_phase_gradient_rotations_are_ordinary_composition() -> None:
    """QVR and Hamming-weight phasing need no specialized compiler op."""

    exercise_documented_phase_gradient_backend(endpoint=qm.Reference(), seed=2)
    program = phase_gradient_qvr_sample.specialize()
    inspection = program.inspect()
    study = qm.resources.logical(phase_gradient_qvr_sample)

    assert "quantum.qvr" not in inspection
    assert "quantum.phase_gradient" not in inspection
    assert "qcore.control_call" in inspection
    assert study.metric(qm.resources.LOGICAL_ALLOCATED_QUBITS).numeric_enclosure() == (
        5,
        5,
    )


def test_fault_tolerant_data_loading_is_ordinary_composition() -> None:
    """Alias sampling and select-swap lookup need no specialized compiler op."""

    exercise_documented_fault_tolerant_data_loading_backend(
        endpoint=qm.Reference(),
        seed=7,
    )
    alias_result = qm.run(
        coherent_alias_sample,
        on=qm.Reference(),
        shots=128,
        seed=7,
    ).result
    marginal = [0, 0, 0, 0]
    for bits, count in alias_result.counts.items():
        marginal[bits[0] + 2 * bits[1]] += count

    assert all(
        abs(observed / 128 - expected) < 0.08
        for observed, expected in zip(marginal, (0.125, 0.125, 0.375, 0.375))
    )
    qroam_inspection = clean_qroam_sample.specialize().inspect()
    alias_inspection = coherent_alias_sample.specialize().inspect()
    assert "quantum.qrom" not in qroam_inspection
    assert "quantum.alias" not in alias_inspection
    assert "select_swap_qroam_unload" in qroam_inspection
    assert "qcore.control_call" in alias_inspection


def test_reversible_indexing_is_ordinary_composition() -> None:
    """Permutation and contiguous indexing reuse lookup and register addition."""

    exercise_documented_reversible_indexing_backend(
        endpoint=qm.Reference(),
        seed=2,
    )
    permutation_inspection = basis_permutation_sample.specialize().inspect()
    index_inspection = contiguous_index_sample.specialize().inspect()

    assert "quantum.permutation" not in permutation_inspection
    assert "quantum.contiguous_index" not in index_inspection
    assert "quantum.swap" in permutation_inspection
    assert "quantum.x" in permutation_inspection
    assert "qcore.call" in index_inspection


def test_reversible_trigonometry_matches_bounded_host_tables() -> None:
    """Arcsine and arctangent maps are coherent bounded approximations."""

    exercise_documented_reversible_trigonometry_backend(
        endpoint=qm.Reference(),
        seed=2,
    )
    for value in range(8):
        assert _DOCUMENTED_ARCSIN_VALUES[value] == int(math.asin(value / 8) * 16)
        packed = _DOCUMENTED_ARCTAN_VALUES[value]
        assert packed & 1 == int(value > 0)
        assert packed >> 1 == int(abs(-2 * math.atan(value) / math.pi) * 16)

    arcsin_inspection = reversible_arcsin_sample.specialize().inspect()
    arctan_inspection = reversible_arctan_sample.specialize().inspect()
    assert "quantum.arcsin" not in arcsin_inspection
    assert "quantum.arctan" not in arctan_inspection
    assert "qcore.select_call" in arcsin_inspection
    assert "qcore.select_call" in arctan_inspection


def test_kikuchi_oracles_compose_sorted_set_primitives() -> None:
    """The bounded k-XOR oracle matches symmetric-difference host data."""

    exercise_documented_kikuchi_backend(endpoint=qm.Reference(), seed=2)
    for row, left in enumerate(_DOCUMENTED_KIKUCHI_VERTICES):
        for column, right in enumerate(_DOCUMENTED_KIKUCHI_VERTICES):
            packed = _DOCUMENTED_KIKUCHI_VALUES[row + (column << 3)]
            reverse = _DOCUMENTED_KIKUCHI_VALUES[column + (row << 3)]
            difference = tuple(sorted(set(left) ^ set(right)))

            assert packed == reverse
            assert bool(packed & 1) == (difference in _DOCUMENTED_KIKUCHI_RHS)
            if packed & 1:
                assert bool(packed & 2) == (_DOCUMENTED_KIKUCHI_RHS[difference] < 0)

    duplicate_inspection = _DOCUMENTED_DUPLICATES.module().inspect()
    difference_inspection = symmetric_difference_sample.specialize().inspect()
    entry_inspection = kikuchi_entry_sample.specialize().inspect()
    assert "quantum.has_duplicates" not in duplicate_inspection
    assert "quantum.symmetric_difference" not in difference_inspection
    assert "quantum.kikuchi" not in entry_inspection
    assert "qcore.control_call" in duplicate_inspection
    assert "qcore.select_call" in difference_inspection
    assert "qcore.select_call" in entry_inspection


def test_guided_hamiltonian_phase_estimation_resolves_overlap() -> None:
    """QPE maps guide overlap to exact low/high phase probabilities."""

    exercise_documented_guided_hamiltonian_backend(
        endpoint=qm.Reference(),
        seed=5,
    )
    mixture = qm.run(
        guided_hamiltonian_mixture_sample,
        on=qm.Reference(),
        shots=128,
        seed=5,
    ).result
    high_probability = mixture.counts.get((0, 1, 1), 0) / 128

    assert set(mixture.counts) == {(0, 0, 0), (0, 1, 1)}
    assert abs(high_probability - 0.75) < 0.1
    inspection = guided_hamiltonian_mixture_sample.specialize().inspect()
    assert "quantum.guided_hamiltonian" not in inspection
    assert "qcore.control_call" in inspection
    assert "qcore.adjoint_call" in inspection


def test_elliptic_curve_dlog_is_ordinary_hidden_subgroup_composition() -> None:
    """The toy curve recovers its secret without an ECC compiler operation."""

    exercise_documented_elliptic_curve_dlog_backend(
        endpoint=qm.Reference(),
        seed=9,
    )
    result = qm.run(
        elliptic_curve_dlog_sample,
        on=qm.Reference(),
        shots=128,
        seed=9,
    ).result
    recovered = {
        frequency_q * pow(frequency_p, -1, 4) % 4
        for frequency_p, frequency_q, _point_label in result.values
        if frequency_p % 2 == 1
    }

    assert recovered == {_DOCUMENTED_EC_SECRET}
    inspection = elliptic_curve_dlog_sample.specialize().inspect()
    study = qm.resources.logical(elliptic_curve_dlog_sample)
    assert "quantum.elliptic_curve" not in inspection
    assert "quantum.add_into" in inspection
    assert "qcore.adjoint_call" in inspection
    assert study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure() == (
        6,
        6,
    )


def test_factorized_chemistry_families_are_ordinary_composition() -> None:
    """Five chemistry formulations reuse existing typed quantum primitives."""

    exercise_documented_factorized_chemistry_backend(
        endpoint=qm.Reference(),
        seed=4,
    )
    sf_inspection = single_factorization_block_sample.specialize().inspect()
    df_inspection = double_factorization_basis_sample.specialize().inspect()
    thc_inspection = tensor_hypercontraction_sample.specialize().inspect()
    first_inspection = first_quantized_chemistry_sample.specialize(
        hamiltonians=_DOCUMENTED_FIRST_QUANTIZED_TERMS,
    ).inspect()
    projectile_inspection = quantum_projectile_sample.specialize(
        hamiltonians=_DOCUMENTED_PROJECTILE_TERMS,
    ).inspect()

    assert _DOCUMENTED_SF_ENCODING.normalization == 1.125
    assert "quantum.single_factorization" not in sf_inspection
    assert "qcore.select_call" in sf_inspection
    assert "quantum.double_factorization" not in df_inspection
    assert "qcore.control_call" in df_inspection
    assert "quantum.tensor_hypercontraction" not in thc_inspection
    assert "qcore.select_call" in thc_inspection
    assert "quantum.first_quantized_chemistry" not in first_inspection
    assert "quantum.projectile" not in projectile_inspection
    assert "qcore.call" in first_inspection
    assert "qcore.call" in projectile_inspection


def test_hydrogen_vqe_is_an_ordinary_variational_expectation() -> None:
    """The documented VQE needs no chemistry-specific compiler operation."""

    exercise_hydrogen_vqe_backend(endpoint=qm.Reference(), seed=2)
    inspection = hydrogen_vqe_state.module().inspect()
    study = qm.resources.logical(hydrogen_vqe_state)

    assert "qcore.call" in inspection
    assert all(
        operation.opcode != "quantum.vqe"
        for function in hydrogen_vqe_state.module().functions
        for operation in function.body.operations
    )
    assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (3, 3)
    assert _hydrogen_energy(_HYDROGEN_ANGLE) < _hydrogen_energy(0.0)


def test_pce_is_an_ordinary_state_expectation_and_typed_decode() -> None:
    """The documented PCE needs no encoding-specific compiler operation."""

    exercise_pce_backend(endpoint=qm.Reference(), seed=2)
    inspection = pce_state.module().inspect()
    study = qm.resources.logical(pce_state)

    assert "quantum.pce" not in inspection
    assert _PCE_ENCODING.num_qubits == 2
    assert _PCE_MODEL.num_variables == 4
    assert study.metric(qm.resources.LOGICAL_ROTATION_GATES).numeric_enclosure() == (
        4,
        4,
    )


def test_qsci_samples_then_solves_nested_variational_subspaces() -> None:
    """The documented QSCI flow composes sampling and host diagonalization."""

    result = qm.run(
        qsci_samples,
        on=qm.Reference(),
        parameters={"theta": _QSCI_ANGLE},
        shots=64,
        seed=2,
    ).result
    assert isinstance(result, SampleResult)
    assert result.backend == qm.Reference().name
    assert sum(result.counts.values()) == 64
    assert set(result.counts) == {(0, 0), (1, 0), (0, 1), (1, 1)}
    samples = tuple(
        bits
        for bits, _ in sorted(
            result.counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    exact = -math.sqrt(1.0 + 4.0 * 0.7**2)
    energies = tuple(
        float(qm.linalg.solve_subspace(samples[:size], _QSCI_HAMILTONIAN)[0][0])
        for size in (1, 2, 4)
    )
    inspection = qsci_samples.module().inspect()

    assert all(left >= right - 1e-12 for left, right in zip(energies, energies[1:]))
    assert all(energy >= exact - 1e-12 for energy in energies)
    assert math.isclose(energies[-1], exact, rel_tol=0.0, abs_tol=1e-12)
    assert "qcore.call" in inspection
    assert all(
        operation.opcode != "quantum.qsci"
        for function in qsci_samples.module().functions
        for operation in function.body.operations
    )
    assert qm.resources.logical(qsci_samples).metric(
        qm.resources.LOGICAL_GATES
    ).numeric_enclosure() == (3, 3)


def test_hamiltonian_simulation_is_an_ordinary_product_formula() -> None:
    """The documented Rabi simulation uses shared product composition only."""

    exercise_rabi_backend(endpoint=qm.Reference(), seed=2)
    program = rabi_samples.specialize(hamiltonians=_RABI_TERMS)
    inspection = program.inspect()
    study = qm.resources.logical(program)

    assert qm.commutator(*_RABI_TERMS) == (0.5j * _RABI_OMEGA * _RABI_DRIVE * qm.Y(0))
    assert inspection.count("quantum.pauli_evolution") == 1
    assert "quantum.hamiltonian_simulation" not in inspection
    assert study.metric(
        qm.resources.logical_operation_count("quantum.pauli_evolution")
    ).numeric_enclosure() == (44, 44)


def test_mottonen_encoding_is_runtime_rebindable_gate_composition() -> None:
    """The documented Möttönen preparation needs no state-preparation opcode."""

    exercise_mottonen_backend(endpoint=qm.Reference(), seed=2)
    program = mottonen_samples.specialize()
    inspection = program.inspect()
    study = qm.resources.logical(program)

    assert "quantum.amplitude" not in inspection
    assert inspection.count("quantum.ry") == 2
    assert inspection.count("quantum.rz") == 2
    assert inspection.count("quantum.cx") == 2
    assert study.metric(qm.resources.LOGICAL_ROTATION_GATES).numeric_enclosure() == (
        6,
        6,
    )
    assert study.metric(
        qm.resources.logical_operation_count("quantum.cx")
    ).numeric_enclosure() == (4, 4)


def test_multidimensional_qft_is_axiswise_function_composition() -> None:
    """The documented two-axis QFT needs no multidimensional compiler op."""

    program = multidimensional_qft_samples.specialize()
    endpoint = qm.Reference()
    record = qm.run(
        program,
        on=endpoint,
        parameters=_MULTIDIMENSIONAL_QFT_COORDINATES.bindings(),
        shots=32,
        seed=2,
    )
    _assert_multidimensional_qft_result(record.result, endpoint)
    inspection = program.inspect()
    study = record.resources
    qft_symbols = {
        function.symbol
        for function in program.functions
        if function.symbol.rsplit(".", 1)[-1] == "qft"
    }

    assert len(qft_symbols) == 1
    assert inspection.count(f"callee='{next(iter(qft_symbols))}'") == 2
    assert "qcore.factor" not in inspection
    assert "qcore.tensor" not in inspection
    assert "quantum.multidimensional_qft" not in inspection
    assert study.metric(
        qm.resources.logical_operation_count("quantum.h")
    ).numeric_enclosure() == (4, 4)
    assert study.metric(
        qm.resources.logical_operation_count("quantum.cp")
    ).numeric_enclosure() == (2, 2)
    assert study.metric(
        qm.resources.logical_operation_count("quantum.swap")
    ).numeric_enclosure() == (2, 2)
    assert qm.resources.ResourceStudy.from_bytes(study.to_bytes()) == study


def test_hybrid_qnn_composes_runtime_inference_and_parameter_shift() -> None:
    """The documented Hybrid QNN keeps differentiation on the host."""

    exercise_hqnn_backend(endpoint=qm.Reference(), seed=2)
    shift = math.pi / 2.0
    plus = list(_HQNN_INPUTS)
    minus = list(_HQNN_INPUTS)
    plus[0] += shift
    minus[0] -= shift
    endpoint = qm.Reference()
    plus_result = qm.run(
        hqnn_samples,
        on=endpoint,
        parameters=_hqnn_parameters(tuple(plus)),
        shots=8,
        seed=2,
    ).result
    minus_result = qm.run(
        hqnn_samples,
        on=endpoint,
        parameters=_hqnn_parameters(tuple(minus)),
        shots=8,
        seed=2,
    ).result
    assert isinstance(plus_result, SampleResult)
    assert isinstance(minus_result, SampleResult)
    gradient = (plus_result.expectation_z(0) - minus_result.expectation_z(0)) / 2.0
    inspection = hqnn_samples.module().inspect()
    study = qm.resources.logical(hqnn_samples)

    assert math.isclose(gradient, -1.0, rel_tol=0.0, abs_tol=1e-12)
    assert "quantum.neural_network" not in inspection
    assert study.metric(qm.resources.LOGICAL_ROTATION_GATES).numeric_enclosure() == (
        16,
        16,
    )
    assert study.metric(
        qm.resources.logical_operation_count("quantum.cz")
    ).numeric_enclosure() == (3, 3)


def test_option_pricing_composes_state_loading_payoff_and_amplification() -> None:
    """The documented option pricer needs no finance-specific compiler op."""

    exercise_option_pricing_backend(endpoint=qm.Reference(), seed=2)
    program = option_pricing_samples.specialize(iterations=1)
    inspection = program.inspect()
    study = qm.resources.logical(program)

    assert "qcore.adjoint_call" in inspection
    assert "quantum.option_pricing" not in inspection
    assert study.metric(qm.resources.LOGICAL_ROTATION_GATES).numeric_enclosure() == (
        12,
        12,
    )
    assert study.metric(
        qm.resources.logical_operation_count("quantum.cx")
    ).numeric_enclosure() == (4, 4)
    assert study.metric(
        qm.resources.logical_operation_count("quantum.cz")
    ).numeric_enclosure() == (1, 1)
    assert study.metric(qm.resources.LOGICAL_TWO_QUBIT_GATES).numeric_enclosure() == (
        7,
        7,
    )


def test_qemcmc_composes_quantum_proposals_with_host_metropolis_updates() -> None:
    """The documented QeMCMC flow keeps Metropolis updates on the host."""

    exercise_qemcmc_backend(endpoint=qm.Reference(), seed=2)
    program = qemcmc_samples.specialize(hamiltonians=_QEMCMC_HAMILTONIANS)
    inspection = program.inspect()
    study = qm.resources.logical(program)
    states = tuple(
        tuple((basis >> index) & 1 for index in range(2)) for basis in range(4)
    )
    proposal = tuple(_qemcmc_proposal_probabilities(bits) for bits in states)
    energies = tuple(_ising_energy(bits) for bits in states)
    boltzmann_weights = tuple(math.exp(-_QEMCMC_BETA * energy) for energy in energies)
    partition = sum(boltzmann_weights)
    stationary = tuple(weight / partition for weight in boltzmann_weights)
    transition: list[list[float]] = [[0.0] * 4 for _ in range(4)]
    for source in range(4):
        for target in range(4):
            if source == target:
                continue
            acceptance = min(
                1.0,
                math.exp(-_QEMCMC_BETA * (energies[target] - energies[source])),
            )
            transition[source][target] = proposal[source][target] * acceptance
        transition[source][source] = 1.0 - sum(transition[source])

    assert "quantum.qemcmc" not in inspection
    assert inspection.count("quantum.pauli_evolution") == 1
    assert study.metric(
        qm.resources.logical_operation_count("quantum.pauli_evolution")
    ).numeric_enclosure() == (3, 3)
    for source in range(4):
        assert math.isclose(sum(transition[source]), 1.0, abs_tol=1e-12)
        for target in range(4):
            assert math.isclose(
                proposal[source][target],
                proposal[target][source],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            assert math.isclose(
                stationary[source] * transition[source][target],
                stationary[target] * transition[target][source],
                rel_tol=0.0,
                abs_tol=1e-12,
            )


def test_bit_flip_code_is_coherent_gate_composition() -> None:
    """The documented repetition code needs no QEC-specific compiler op."""

    exercise_bit_flip_code_backend(
        endpoint=qm.Reference(),
        seed=2,
        error_positions=(0, 1, 2),
    )
    program = bit_flip_code_samples.specialize(error_position=1)
    inspection = program.inspect()
    study = qm.resources.logical(program)

    assert "quantum.error_correction" not in inspection
    assert study.metric(
        qm.resources.logical_operation_count("quantum.cx")
    ).numeric_enclosure() == (8, 8)
    assert study.metric(
        qm.resources.logical_operation_count("quantum.ccx")
    ).numeric_enclosure() == (3, 3)
    assert study.metric(qm.resources.LOGICAL_ROTATION_GATES).numeric_enclosure() == (
        1,
        1,
    )


def test_steane_code_corrects_documented_pauli_errors_coherently() -> None:
    """Execute one combined bit-and-phase error within the ten-qubit budget."""

    program = steane_code_samples.specialize(**{"error_kind": 2, "error_position": 3})
    endpoint = qm.Reference()
    record = qm.run(program, on=endpoint, shots=4, seed=2)
    _assert_steane_code_result(record.result, endpoint, error_position=3)
    inspection = program.inspect()
    study = record.resources

    assert "quantum.steane" not in inspection
    assert "quantum.error_correction" not in inspection
    assert study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure() == (
        10,
        10,
    )
    assert study.metric(
        qm.resources.logical_operation_count("quantum.cx")
    ).numeric_enclosure() == (21, 21)
    assert study.metric(
        qm.resources.logical_operation_count("qcore.control_call")
    ).numeric_enclosure() == (7, 7)
