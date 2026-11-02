"""Functional standard-library and closed source-linking tests."""

from __future__ import annotations

import math
import random
from itertools import combinations_with_replacement
from types import SimpleNamespace

import pytest

import qamomile as qm
from qamomile.circuit import scalar_parameter_bindings, scalar_parameters
from qamomile.effects import CallableKind, Requirement
from qamomile.persistence import serialize
from qamomile.stdlib import (
    BlockEncoding,
    CertifiedQSVTPolynomial,
    CertifiedReciprocalSVT,
    QSVTPhases,
    QSVTSuccess,
    add_constant,
    add_modulo,
    add_register_modulo,
    add_registers,
    amplitude_angles,
    amplitude_encoding,
    amplitude_encoding_from_angles,
    boolean_isometry,
    compare_unsigned,
    divide_constant_modulo,
    ekera_hastad_factoring,
    ekera_hastad_factoring_state,
    fqaoa_state,
    gf2_multiply_add,
    grover_iteration_count,
    grover_search,
    has_duplicates_sorted,
    inverse_qft,
    lcu_block_encoding,
    less_than_registers,
    modular_decrement,
    modular_increment,
    modular_multi_exponentiation,
    multiply_add,
    multiply_constant_modulo,
    multiply_modulo_constant,
    pauli_block_encoding,
    periodic_shift_block_encoding,
    periodic_shift_block_encoding_from_matrix,
    phase_estimation,
    product_tree_multi_exponentiation,
    projected_block_encoding,
    qaoa_state,
    qft,
    qsvt,
    qsvt_application,
    qsvt_from_phases,
    reciprocal_svt,
    regev_factoring,
    regev_factoring_state,
    shor_order_finding,
    shor_order_finding_state,
    sparse_matrix_lookup,
    subtract_constant,
    subtract_register_modulo,
    subtract_registers,
    suzuki_trotter,
    synthesize_qsvt_polynomial,
    uniformly_controlled_angles,
    uniformly_controlled_ry,
    uniformly_controlled_ry_from_angles,
    windowed_phase_estimation,
    xor_lookup,
)
from qamomile.stdlib.arithmetic import (
    _add_constant_modulo_with_work as internal_add_constant_modulo,
    _subtract_constant_modulo_with_work as internal_subtract_constant_modulo,
)

from .conftest import lower_adjoint_calls

_SORTED_DUPLICATES = has_duplicates_sorted(length=3, item_width=2)


@qm.qkernel
def sorted_duplicates_sample() -> qm.Shot[qm.Bits[10]]:
    """Measure a sorted list containing one repeated item.

    Returns:
        qm.Shot[qm.Bits[10]]: Items, duplicate output, and clean workspace.
    """

    state = _SORTED_DUPLICATES(qm.prepare((1, 0, 1, 0, 1, 1, 0, 0, 0, 0)))
    return qm.measure(state)


def test_grover_iteration_count_is_validated_and_arbitrary_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grover host configuration avoids binary64 overflow and invalid spaces."""

    import importlib

    grover_module = importlib.import_module("qamomile.stdlib.grover")

    assert grover_iteration_count(4) == 3
    assert grover_iteration_count(10) == 25
    assert grover_iteration_count(1024) == int(
        "105304677233626590548617053711398470263139993283723136513986712720259514"
        "455690247299484713430619315866109428242290833713318232291563997903855884"
        "43550958149"
    )
    with pytest.raises(TypeError, match="num_qubits must be an integer"):
        grover_iteration_count(True)
    with pytest.raises(ValueError, match="num_marked must be positive"):
        grover_iteration_count(4, 0)
    with pytest.raises(ValueError, match="cannot exceed the search space"):
        grover_iteration_count(2, 5)
    with pytest.raises(ValueError, match="16384-qubit budget"):
        grover_iteration_count(16_385)
    with pytest.raises(ValueError, match="16384-qubit budget"):
        grover_iteration_count(10**100)

    monkeypatch.setattr(grover_module.native, "grover_iteration_count", None)
    with pytest.raises(TypeError, match="Q7594"):
        grover_iteration_count(4)
    monkeypatch.setattr(
        grover_module.native,
        "grover_iteration_count",
        lambda _qubits, _marked: 3,
    )
    with pytest.raises(TypeError, match="Q7595"):
        grover_iteration_count(4)


@qm.qkernel
def qft_roundtrip() -> qm.Bits[3]:
    """Apply imported Fourier functions without a registry or compiler hook."""

    state = qm.prepare((1, 0, 1))
    state = qft(state, 3)
    state = inverse_qft(state, 3)
    return qm.measure(state)


@qm.qkernel
def qft_frequency_three() -> qm.Shot[qm.Bits[3]]:
    """Transform one asymmetric Fourier mode into little-endian integer three."""

    state = qm.qubits(3)
    state = qm.h(state, target=0)
    state = qm.rz(state, -2.356194490192345, target=0)
    state = qm.h(state, target=1)
    state = qm.rz(state, -4.71238898038469, target=1)
    state = qm.h(state, target=2)
    state = qm.rz(state, -9.42477796076938, target=2)
    state = qft(state, 3)
    return qm.measure(state)


@qm.qkernel
def qft_temporary() -> qm.Unit:
    """Create a Fourier-transformed temporary for automatic restoration."""

    state = qm.qubits(3)
    state = qft(state, 3)
    return None


@qm.qkernel
def nested_qft(state: qm.Q[qm.Bits[n]], size: qm.Const[qm.Int]) -> qm.Q[qm.Bits[n]]:  # type: ignore[valid-type]  # noqa: F821
    """Exercise adjoint composition through a polymorphic helper call."""

    return qft(state, size)


@qm.qkernel
def nested_qft_roundtrip() -> qm.Bits[3]:
    """Apply a nested helper and its compiler-derived inverse."""

    state = qm.prepare((1, 1, 0))
    state = nested_qft(state, 3)
    state = qm.adjoint(nested_qft, state, 3)
    return qm.measure(state)


@qm.qkernel
def sign_phase(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    """Expose the exactly representable eigenphase one half."""

    return qm.z(state)


@qm.qkernel
def quarter_phase(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    """Expose the exactly representable eigenphase one quarter."""

    return qm.p(state, 1.5707963267948966)


@qm.qkernel
def eighth_phase(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    """Expose the exactly representable eigenphase one eighth."""

    return qm.p(state, 0.7853981633974483)


@qm.qkernel
def nary_tensor_sample() -> qm.Shot[qm.Bits[3]]:
    """Exercise Python-tuple tensoring across three coherent leaves."""

    first = qm.prepare((1,))
    second = qm.prepare((0,))
    third = qm.prepare((1,))
    return qm.measure(qm.concat(first, second, third))


def exercise_nary_tensor_backend(*, endpoint: object, seed: int = 0) -> None:
    """Verify n-ary coherent tuple reassociation on one execution backend."""

    assert qm.run(
        nary_tensor_sample,
        on=endpoint,  # type: ignore[arg-type]
        shots=4,
        seed=seed,
    ).result.counts == {(1, 0, 1): 4}


@qm.qkernel
def mark_eleven(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    """Mark the basis state eleven with an exact phase flip."""

    return qm.cz(state, control=0, target=1)


amplify_eleven = grover_search(mark_eleven, width=2)
increment_three = modular_increment(width=3)
decrement_three = modular_decrement(width=3)
add_three = add_constant(width=3, value=3)
subtract_three = subtract_constant(width=3, value=3)
add_register_pair = add_registers(width=3)
subtract_register_pair = subtract_registers(width=3)
add_register_mod_three = add_register_modulo(width=2)
subtract_register_mod_three = subtract_register_modulo(width=2)
add_three_mod_five = internal_add_constant_modulo(width=3, value=3, modulus=5)
subtract_three_mod_five = internal_subtract_constant_modulo(width=3, value=3, modulus=5)
multiply_two_mod_five = multiply_constant_modulo(width=3, value=2, modulus=5)
divide_two_mod_five = divide_constant_modulo(width=3, value=2, modulus=5)
multiply_two_mod_three = multiply_constant_modulo(width=2, value=2, modulus=3)
less_than_two = less_than_registers(width=2)
multiply_pair_two = multiply_add(width=2)
gf8_multiply_add = gf2_multiply_add(width=3, modulus=0b1011)
shor_three_sample = qm.specialize(
    shor_order_finding,
    base=2,
    modulus=3,
    width=2,
    precision=2,
)


shor_backend_sample = qm.specialize(
    shor_order_finding,
    base=1,
    modulus=2,
    width=1,
    precision=1,
)


@qm.qkernel
def shor_backend_state() -> qm.Q[qm.Bits[2]]:
    """Prepare the bounded specialization used by SDK backend witnesses.

    Returns:
        qm.Q[qm.Bits[2]]: Counting qubit followed by the modular qubit.
    """

    counting = qm.qubits(1)
    target = qm.x(qm.qubits(1), target=0)
    counting, target = shor_order_finding_state(counting, target, 1, 2, 1)
    return qm.concat(counting, target)


@qm.qkernel
def shor_three_state() -> qm.Q[qm.Bits[4]]:
    """Prepare the four-qubit specialization used by backend witnesses.

    Returns:
        qm.Q[qm.Bits[4]]: Counting register followed by the modular register.
    """

    counting = qm.qubits(2)
    target = qm.x(qm.qubits(2), target=0)
    counting, target = shor_order_finding_state(counting, target, 2, 3, 2)
    return qm.concat(counting, target)


ekera_three_sample = ekera_hastad_factoring(
    generator=2,
    modulus=3,
    parameter_width=1,
)
regev_three_sample = regev_factoring(
    bases=(2,),
    modulus=3,
    coordinate_width=1,
    radius=0.75,
)
multi_exponentiation_baseline = modular_multi_exponentiation(
    bases=(1,), modulus=3, coordinate_width=1
)
multi_exponentiation_product_tree = product_tree_multi_exponentiation(
    bases=(1,), modulus=3, coordinate_width=1
)


@qm.qkernel
def multi_exponentiation_sample() -> qm.Bits[3]:
    """Measure the shared one-coordinate callable-realization witness.

    Returns:
        qm.Bits[3]: Retained coordinate followed by ``1**coordinate mod 3``.
    """

    coordinates = qm.prepare((1,))
    return qm.measure(multi_exponentiation_baseline(coordinates))


first_class_multi_exponentiation_sample = qm.build_source(
    """
import qamomile as qm

@qm.qkernel
def sample(
    bases: qm.Const[qm.Vector[qm.Int, 1]],
) -> qm.Shot[qm.Bits[2]]:
    coordinates = qm.x(qm.qubits(1), target=0)
    target = qm.as_uint(qm.x(qm.qubits(1), target=0))
    coordinates, target = qm.mod_multi_pow(
        coordinates,
        target,
        bases=bases,
        base_width=1,
        modulus=2,
        coordinate_width=1,
    )
    return qm.measure(qm.concat(coordinates, qm.as_bits(target)))
""",
    module_name="first_class_multi_exponentiation_sample",
).specialize("sample", bases=[1])


uniform_flip = uniformly_controlled_ry((math.pi,) * 4)
runtime_uniform_flip = uniformly_controlled_ry_from_angles(2)
xor_table = xor_lookup((0, 3, 1, 2), value_width=2)
sparse_table = sparse_matrix_lookup(
    ((0, 3), (2, 1)),
    ((1, 2), (3, 0)),
    column_width=2,
    value_width=2,
)


@qm.qkernel
def matrix_rotations(
    state: qm.Q[qm.Bits[1]],
    angles: qm.Matrix[qm.Real, 2, 2],
) -> qm.Q[qm.Bits[1]]:
    """Apply a rectangular runtime table through typed row iteration."""

    for row in angles:
        for angle in row:
            state = qm.rx(state, angle)
    return state


@qm.qkernel
def runtime_matrix_rotation_sample(
    angles: qm.Param[qm.Matrix[qm.Real, 2, 2]],
) -> qm.Shot[qm.Bits[1]]:
    """Measure rotations loaded from one runtime Matrix ABI."""

    return qm.measure(matrix_rotations(qm.qubits(1), angles))


@qm.qkernel
def static_matrix_rotation_sample(
    angles: qm.Const[qm.Matrix[qm.Real, 2, 2]],
) -> qm.Shot[qm.Bits[1]]:
    """Measure rotations loaded from one specialized Matrix ABI."""

    return qm.measure(matrix_rotations(qm.qubits(1), angles))


fourth_order_evolution = suzuki_trotter(term_count=2, order=4, steps=2)


@qm.qkernel
def qaoa_sample(
    hamiltonian: qm.Const[qm.Observable],
    gammas: qm.Param[qm.Vector[qm.Real, 1]],
    betas: qm.Param[qm.Vector[qm.Real, 1]],
) -> qm.Shot[qm.Bits[1]]:
    """Measure a one-layer QAOA state built from the public composition."""

    return qm.measure(qaoa_state(1, 1, hamiltonian, gammas, betas))


@qm.qkernel
def fqaoa_sample(
    hamiltonian: qm.Const[qm.Observable],
    givens_angles: qm.Param[qm.Vector[qm.Real, 1]],
    hopping: qm.Param[qm.Real],
    gammas: qm.Param[qm.Vector[qm.Real, 1]],
    betas: qm.Param[qm.Vector[qm.Real, 1]],
) -> qm.Shot[qm.Bits[2]]:
    """Measure a minimal particle-preserving FQAOA composition."""

    state = fqaoa_state(
        2,
        1,
        1,
        [(0, 1)],
        givens_angles,
        1,
        hamiltonian,
        hopping,
        gammas,
        betas,
    )
    return qm.measure(state)


@qm.qkernel
def product_formula_sample(
    hamiltonians: qm.Const[qm.Vector[qm.Observable, 2]],
    time: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[1]]:
    """Measure a fourth-order product formula over two abstract terms."""

    state = fourth_order_evolution(qm.qubits(1), hamiltonians, time)
    return qm.measure(state)


@qm.qkernel
def static_variational_sample(
    hamiltonian: qm.Const[qm.Observable],
) -> tuple[qm.Shot[qm.Bits[1]], qm.Shot[qm.Bits[2]]]:
    """Close variational coordinates for targets without a parameter ABI."""

    qaoa = qaoa_state(
        1,
        1,
        hamiltonian,
        [0.7853981633974483],
        [0.7853981633974483],
    )
    fqaoa = fqaoa_state(
        2,
        1,
        1,
        [(0, 1)],
        [0.0],
        1,
        hamiltonian,
        0.0,
        [0.0],
        [0.0],
    )
    return qm.measure(qaoa), qm.measure(fqaoa)


@qm.qkernel
def static_product_formula_sample(
    hamiltonians: qm.Const[qm.Vector[qm.Observable, 2]],
) -> qm.Shot[qm.Bits[1]]:
    """Close product-formula time for targets without a parameter ABI."""

    state = fourth_order_evolution(
        qm.qubits(1),
        hamiltonians,
        1.5707963267948966,
    )
    return qm.measure(state)


def exercise_variational_backend(*, endpoint: object, seed: int = 0) -> None:
    """Execute QAOA and FQAOA through one backend without target branches."""

    qaoa = qm.run(  # type: ignore[arg-type]
        qaoa_sample,
        on=endpoint,
        **{"hamiltonian": qm.Z(0)},
        parameters={"gammas": (math.pi / 4.0,), "betas": (math.pi / 4.0,)},
        shots=4,
        seed=seed,
    ).result
    fqaoa = qm.run(  # type: ignore[arg-type]
        fqaoa_sample,
        on=endpoint,
        **{"hamiltonian": qm.Z(0) * qm.Z(1)},
        parameters={
            "givens_angles": (0.0,),
            "hopping": 0.0,
            "gammas": (0.0,),
            "betas": (0.0,),
        },
        shots=4,
        seed=seed,
    ).result

    assert qaoa.counts == {(0,): 4}
    assert fqaoa.counts == {(1, 0): 4}


def exercise_static_variational_backend(*, endpoint: object, seed: int = 0) -> None:
    """Execute statically closed QAOA and FQAOA through one backend."""

    result = qm.run(  # type: ignore[arg-type]
        static_variational_sample,
        on=endpoint,
        **{"hamiltonian": qm.Z(0)},
        shots=4,
        seed=seed,
    ).result

    assert result.values == {((0,), (1, 0)): 4}


def exercise_product_formula_backend(*, endpoint: object, seed: int = 0) -> None:
    """Execute one product formula through an unchanged backend adapter."""

    result = qm.run(  # type: ignore[arg-type]
        product_formula_sample,
        on=endpoint,
        **{
            "hamiltonians": (
                qm.X(0),
                qm.Hamiltonian.zero(num_qubits=1),
            )
        },
        parameters={"time": math.pi / 2.0},
        shots=4,
        seed=seed,
    ).result

    assert result.counts == {(1,): 4}


def exercise_static_product_formula_backend(*, endpoint: object, seed: int = 0) -> None:
    """Execute statically timed product evolution through one backend."""

    result = qm.run(  # type: ignore[arg-type]
        static_product_formula_sample,
        on=endpoint,
        **{
            "hamiltonians": (
                qm.X(0),
                qm.Hamiltonian.zero(num_qubits=1),
            )
        },
        shots=4,
        seed=seed,
    ).result

    assert result.counts == {(1,): 4}


def exercise_modular_arithmetic_backend(*, endpoint: object, seed: int = 0) -> None:
    """Execute one sample and expectation witness per arithmetic family."""

    assert qm.run(  # type: ignore[arg-type]
        subtract_constant_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result.counts == {(0, 1, 1): 4}
    assert (
        qm.expectation(  # type: ignore[arg-type]
            add_constant_state,
            qm.Z(0),
            on=endpoint,
            shots=4,
            seed=seed,
        ).value
        == -1.0
    )
    assert qm.run(  # type: ignore[arg-type]
        controlled_add_modulo_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result.counts == {(1, 0, 0, 1, 0, 0): 4}
    assert (
        qm.expectation(  # type: ignore[arg-type]
            add_modulo_state,
            qm.Z(2),
            on=endpoint,
            shots=4,
            seed=seed,
        ).value
        == -1.0
    )
    assert qm.run(  # type: ignore[arg-type]
        multiply_modulo_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result.counts == {(1, 0, 0, 0, 0, 0, 0, 0): 4}
    assert (
        qm.expectation(  # type: ignore[arg-type]
            multiply_modulo_state,
            qm.Z(0),
            on=endpoint,
            shots=4,
            seed=seed,
        ).value
        == -1.0
    )
    assert qm.run(  # type: ignore[arg-type]
        less_than_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result.counts == {(0, 1, 1, 1, 1, 0): 4}
    assert qm.run(  # type: ignore[arg-type]
        multiply_register_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result.counts == {(0, 1, 1, 1, 0, 1, 1, 0): 4}
    assert (
        qm.expectation(  # type: ignore[arg-type]
            multiply_register_state,
            qm.Z(5),
            on=endpoint,
            shots=4,
            seed=seed,
        ).value
        == -1.0
    )


def exercise_shor_order_finding_backend(*, endpoint: object, seed: int = 0) -> None:
    """Execute the shared Shor sampler and coherent-state witness.

    Args:
        endpoint (object): Backend execution endpoint.
        seed (int): Deterministic sampling seed.
    """

    result = qm.run(  # type: ignore[arg-type]
        shor_backend_sample,
        on=endpoint,
        shots=8,
        seed=seed,
    ).result
    expectation = qm.expectation(  # type: ignore[arg-type]
        shor_backend_state,
        qm.Z(1),
        on=endpoint,
        shots=4,
        seed=seed,
    )
    assert result.counts == {(0,): 8}
    assert expectation.value == -1.0


def exercise_regev_factoring_backend(*, endpoint: object, seed: int = 0) -> None:
    """Execute the shared bounded Regev sampling witness.

    Args:
        endpoint (object): Backend execution endpoint.
        seed (int): Deterministic sampling seed.
    """

    regev_result = qm.run(  # type: ignore[arg-type]
        regev_three_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result

    assert sum(regev_result.counts.values()) == 4
    assert all(len(bits) == 1 for bits in regev_result.counts)


def exercise_multi_exponentiation_realization_backend(
    *, endpoint: object, seed: int = 0
) -> None:
    """Execute the shared whole-callable realization witness.

    The caller supplies an endpoint configured with
    ``multi_exponentiation_baseline`` mapped to
    ``multi_exponentiation_product_tree``.

    Args:
        endpoint (object): Configured backend execution endpoint.
        seed (int): Deterministic sampling seed.
    """

    result = qm.run(  # type: ignore[arg-type]
        multi_exponentiation_sample,
        on=endpoint,
        shots=1,
        seed=seed,
    ).result
    assert result.counts == {(1, 1, 0): 1}


def exercise_first_class_multi_exponentiation_backend(
    *, endpoint: object, seed: int = 0
) -> None:
    """Execute the shared semantic multi-exponentiation primitive.

    Args:
        endpoint (object): Backend execution endpoint.
        seed (int): Deterministic sampling seed.
    """

    result = qm.run(  # type: ignore[arg-type]
        first_class_multi_exponentiation_sample,
        on=endpoint,
        shots=1,
        seed=seed,
    ).result
    assert result.counts == {(1, 1): 1}


def test_first_class_multi_exponentiation_executes_on_reference() -> None:
    """Execute the independent primitive realization through Reference."""

    result = qm.run(
        first_class_multi_exponentiation_sample,
        on=qm.Reference(),
        shots=1,
        seed=0,
    ).result
    assert result.counts == {(1, 1): 1}


def exercise_fixed_qpe_backend(*, endpoint: object, seed: int = 0) -> None:
    """Execute typed fixed-point QPE sampling and state expectation."""

    record = qm.run(  # type: ignore[arg-type]
        qpe_eighth_fixed_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    )
    expectation = qm.expectation(  # type: ignore[arg-type]
        qpe_eighth_state,
        qm.Z(2),
        on=endpoint,
        shots=4,
        seed=seed,
    )

    assert record.result.counts == {(0, 0, 1): 4}
    assert record.result.values == {0.125: 4}
    assert record.result.most_likely_value == 0.125
    assert record.artifact.result_abi.source_order == (0, 1, 2)
    assert record.artifact.result_abi.native_width == 4
    assert expectation.value == -1.0


def exercise_fixed_qpe_hugr_backend(*, endpoint: object, seed: int = 0) -> None:
    """Execute the rotation-free typed phase profile through direct HUGR."""

    result = qm.run(  # type: ignore[arg-type]
        qpe_one_bit_fixed_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result
    expectation = qm.expectation(  # type: ignore[arg-type]
        qpe_one_bit_state,
        qm.Z(0),
        on=endpoint,
        shots=4,
        seed=seed,
    )

    assert result.counts == {(1,): 4}
    assert result.values == {0.5: 4}
    assert result.most_likely_value == 0.5
    assert expectation.value == -1.0


def exercise_windowed_qpe_backend(*, endpoint: object, seed: int = 0) -> None:
    """Execute one explicit-window QPE source through a backend endpoint."""

    result = qm.run(  # type: ignore[arg-type]
        windowed_qpe_one_bit_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result
    expectation = qm.expectation(  # type: ignore[arg-type]
        windowed_qpe_one_bit_state,
        qm.Z(0),
        on=endpoint,
        shots=4,
        seed=seed,
    )

    assert result.counts == {(1,): 4}
    assert result.values == {0.5: 4}
    assert expectation.value == -1.0


@qm.qkernel
def uniformly_controlled_sample() -> qm.Shot[qm.Bits[3]]:
    """Apply the same branch rotation for every two-bit address."""

    return qm.measure(uniform_flip(qm.prepare((1, 0, 0))))


@qm.qkernel
def runtime_uniformly_controlled_sample(
    angles: qm.Param[qm.Vector[qm.Real, 4]],
) -> qm.Shot[qm.Bits[3]]:
    """Apply one runtime-parametric Gray-walk multiplexer."""

    state = runtime_uniform_flip(qm.prepare((0, 1, 0)), angles)
    return qm.measure(state)


@qm.qkernel
def xor_lookup_sample() -> qm.Shot[qm.Bits[4]]:
    """Look up address two into a zero two-bit XOR target."""

    return qm.measure(xor_table(qm.prepare((0, 1, 0, 0))))


@qm.qkernel
def sparse_lookup_state() -> qm.Q[qm.Bits[6]]:
    """Return row-one/slot-zero sparse lookup output before measurement."""

    address, column, value = sparse_table(
        qm.prepare((1, 0)),
        qm.qubits(2),
        qm.qubits(2),
    )
    return qm.concat(address, column, value)


@qm.qkernel
def sparse_lookup_sample() -> qm.Shot[qm.Bits[6]]:
    """Read row one, slot zero into zeroed column and value targets."""

    return qm.measure(sparse_lookup_state())


def exercise_uniformly_controlled_backend(
    *,
    endpoint: object,
    seed: int = 0,
) -> None:
    """Execute the public multiplexer through one backend unchanged."""

    assert qm.run(  # type: ignore[arg-type]
        uniformly_controlled_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result.counts == {(1, 0, 1): 4}


def exercise_xor_lookup_backend(*, endpoint: object, seed: int = 0) -> None:
    """Execute coherent lookup through one backend unchanged."""

    assert qm.run(  # type: ignore[arg-type]
        xor_lookup_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result.counts == {(0, 1, 1, 0): 4}


def exercise_sparse_lookup_backend(*, endpoint: object, seed: int = 0) -> None:
    """Execute type-safe sparse row/slot access through one backend unchanged."""

    assert qm.run(  # type: ignore[arg-type]
        sparse_lookup_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result.counts == {(1, 0, 0, 1, 1, 1): 4}
    assert (  # type: ignore[arg-type]
        qm.expectation(
            sparse_lookup_state,
            qm.Z(0),
            on=endpoint,
            shots=4,
            seed=seed,
        ).value
        == -1.0
    )


def exercise_runtime_matrix_backend(*, endpoint: object, seed: int = 0) -> None:
    """Execute nested Matrix access through one runtime backend ABI."""

    assert qm.run(  # type: ignore[arg-type]
        runtime_matrix_rotation_sample,
        on=endpoint,
        parameters={"angles": ((math.pi, 0.0), (0.0, 0.0))},
        shots=4,
        seed=seed,
    ).result.counts == {(1,): 4}


def exercise_static_matrix_backend(*, endpoint: object, seed: int = 0) -> None:
    """Execute nested Matrix access after ordinary Const specialization."""

    assert qm.run(  # type: ignore[arg-type]
        static_matrix_rotation_sample,
        on=endpoint,
        **{"angles": ((math.pi, 0.0), (0.0, 0.0))},
        shots=4,
        seed=seed,
    ).result.counts == {(1,): 4}


def exercise_runtime_qsvt_backend(*, endpoint: object, seed: int = 0) -> None:
    """Execute one compiled QSVT body with a runtime phase Vector."""

    assert qm.run(  # type: ignore[arg-type]
        runtime_qsvt_sample,
        on=endpoint,
        parameters={"phases": (0.0, 0.0)},
        shots=4,
        seed=seed,
    ).result.counts == {(0, 1): 4}


def exercise_deferred_block_backend(*, endpoint: object, seed: int = 0) -> None:
    """Compile one source with two later-selected blocks on each target."""

    for encoding, expected in (
        (identity_block_encoding, (0, 0)),
        (x_block_encoding, (0, 1)),
    ):
        artifact = qm.compile(  # type: ignore[arg-type]
            deferred_block_sample,
            to=endpoint.target,
            block=encoding,
        )
        assert serialize(artifact)
        assert qm.run(  # type: ignore[arg-type]
            artifact,
            on=endpoint,
            shots=4,
            seed=seed,
        ).result.counts == {expected: 4}
        qsvt_artifact = qm.compile(  # type: ignore[arg-type]
            deferred_qsvt_sample,
            to=endpoint.target,
            block=encoding,
        )
        assert qm.run(  # type: ignore[arg-type]
            qsvt_artifact,
            on=endpoint,
            parameters={"phases": (0.0, 0.0)},
            shots=4,
            seed=seed,
        ).result.counts == {expected: 4}
    inverse_artifact = qm.compile(  # type: ignore[arg-type]
        deferred_qsvt_roundtrip_sample,
        to=endpoint.target,
        block=phase_block_encoding,
    )
    assert qm.run(  # type: ignore[arg-type]
        inverse_artifact,
        on=endpoint,
        parameters={"phases": (0.0, 0.0, 0.0)},
        shots=4,
        seed=seed,
    ).result.counts == {(0, 0): 4}


@qm.qkernel
def encoded_x(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    """Encode Pauli X while leaving the leading signal component unchanged."""

    return qm.x(state, target=1)


@qm.qkernel
def identity_encoding(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    """Provide an exact identity block encoding for projector-phase tests."""

    return state


@qm.qkernel
def identity_three_encoding(state: qm.Q[qm.Bits[3]]) -> qm.Q[qm.Bits[3]]:
    """Provide a wider identity for block-binding shape rejection.

    Args:
        state (qm.Q[qm.Bits[3]]): Joint signal/system register.

    Returns:
        qm.Q[qm.Bits[3]]: The unchanged register.
    """

    return state


@qm.qkernel
def phase_encoding(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    """Phase the system component without changing the signal.

    Args:
        state (qm.Q[qm.Bits[2]]): Joint signal/system register.

    Returns:
        qm.Q[qm.Bits[2]]: Joint register with an S gate on the system.
    """

    return qm.s(state, target=1)


x_block_encoding = BlockEncoding(encoded_x, 1.0, 1, 1)
identity_block_encoding = BlockEncoding(identity_encoding, 1.0, 1, 1)
phase_block_encoding = BlockEncoding(phase_encoding, 1.0, 1, 1)


@qm.qkernel
def deferred_block_sample(block: BlockEncoding[1, 1]) -> qm.Shot[qm.Bits[2]]:
    """Measure the block selected when the algorithm is compiled.

    Args:
        block (BlockEncoding[1, 1]): Later-selected signal/system unitary.

    Returns:
        qm.Shot[qm.Bits[2]]: Measured joint register.
    """

    return qm.measure(block(qm.qubits(2)))


@qm.qkernel
def deferred_keyword_block_sample(
    *,
    block: BlockEncoding[1, 1],
    flip: qm.Const[qm.Bool] = False,
) -> qm.Shot[qm.Bits[2]]:
    """Keep a keyword-only Const default while binding one block.

    Args:
        block (BlockEncoding[1, 1]): Later-selected encoded unitary.
        flip (qm.Const[qm.Bool]): Whether to flip the system before the block.

    Returns:
        qm.Shot[qm.Bits[2]]: Joint register measurement.
    """

    state = qm.qubits(2)
    if flip:
        state = qm.x(state, target=1)
    return qm.measure(block(state))


deferred_qsvt_step = qsvt_application(signal_width=1, system_width=1, degree=1)
deferred_qsvt_roundtrip = qsvt_application(signal_width=1, system_width=1, degree=2)


@qm.qkernel
def deferred_qsvt_sample(
    block: BlockEncoding[1, 1],
    phases: qm.Param[qm.Vector[qm.Real, 2]],
) -> qm.Shot[qm.Bits[2]]:
    """Apply a QSVT step to a later-bound block with runtime phases.

    Args:
        block (BlockEncoding[1, 1]): Later-selected signal/system unitary.
        phases (qm.Param[qm.Vector[qm.Real, 2]]): Runtime projector phases.

    Returns:
        qm.Shot[qm.Bits[2]]: Measured transformed register.
    """

    state = deferred_qsvt_step(block, qm.qubits(2), phases)
    return qm.measure(state)


@qm.qkernel
def deferred_qsvt_roundtrip_sample(
    block: BlockEncoding[1, 1],
    phases: qm.Param[qm.Vector[qm.Real, 3]],
) -> qm.Shot[qm.Bits[2]]:
    """Interfere a block and its adjoint around a prepared superposition.

    Args:
        block (BlockEncoding[1, 1]): Later-selected block encoding.
        phases (qm.Param[qm.Vector[qm.Real, 3]]): Runtime phases.

    Returns:
        qm.Shot[qm.Bits[2]]: Measured interference register.
    """

    state = qm.h(qm.qubits(2), target=1)
    state = deferred_qsvt_roundtrip(block, state, phases)
    return qm.measure(qm.h(state, target=1))


certified_x_polynomial = synthesize_qsvt_polynomial(
    (0.0, 1.0),
    max_error=1e-15,
    max_iterations=1,
)
reciprocal_x = reciprocal_svt(
    x_block_encoding,
    minimum_singular_value=1.0,
    max_error=1e-15,
    max_degree=1,
    max_iterations=1,
)
reciprocal_x_transform = reciprocal_x.transform
qsvt_x = qsvt(x_block_encoding, polynomial=certified_x_polynomial)
runtime_qsvt_x = qsvt_from_phases(x_block_encoding, degree=1)
qsvt_x_roundtrip = qsvt(x_block_encoding, phases=(0.0, 0.0, 0.0))
qsvt_half_turn = qsvt(
    identity_block_encoding,
    phases=QSVTPhases((math.pi / 2.0,), odd=False, domain=(-1.0, 1.0)),
)

linear_system_operator = 0.75 + 0.25 * qm.Z(0)
linear_system_source_encoding = pauli_block_encoding(linear_system_operator)
linear_system_reciprocal = reciprocal_svt(
    linear_system_source_encoding,
    minimum_singular_value=0.5,
    max_error=0.005,
    max_degree=31,
)
linear_system_transform = linear_system_reciprocal.transform


@qm.qkernel
def parity_bit(bits: qm.Bits[2]) -> qm.Bool:
    """Return one asymmetric parity witness for coherent embedding."""

    return bool(bits[0]) ^ bool(bits[1])


@qm.qkernel
def ordered_pair(bits: qm.Bits[2]) -> tuple[qm.Bool, qm.Bool]:
    """Return two ordered Boolean coordinates for coherent embedding."""

    return (not bool(bits[1]), bool(bits[0]) ^ bool(bits[1]))


parity_isometry = boolean_isometry(parity_bit)
pair_isometry = boolean_isometry(ordered_pair)


@qm.qkernel
def boolean_isometry_sample() -> qm.Shot[qm.Bits[4]]:
    """Prepare an asymmetric input and sample its Boolean graph embedding."""

    state = qm.x(qm.qubits(2), target=0)
    return qm.measure(pair_isometry(state))


@qm.qkernel
def system_identity(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    """Provide the identity term of an exact projector LCU."""

    return state


@qm.qkernel
def system_z(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    """Provide the signed term of an exact projector LCU."""

    return qm.z(state)


y_eigenstate = amplitude_encoding((1.0, 1.0j))
basis_eleven = amplitude_encoding((0.0, 0.0, 0.0, 1.0))
runtime_amplitude_two = amplitude_encoding_from_angles(2)
projector_lcu = lcu_block_encoding(
    ((1.0, system_identity), (-1.0, system_z)),
    system_width=1,
)
projector_reciprocal = reciprocal_svt(
    projector_lcu,
    minimum_singular_value=projector_lcu.normalization,
    max_error=1e-15,
    max_degree=1,
    max_iterations=1,
)
projector_qsvt = qsvt(projector_lcu, polynomial=projector_reciprocal.polynomial)
pauli_x_encoding = pauli_block_encoding(1.0j * qm.X(0))
pauli_x_unitary = pauli_x_encoding.unitary


@qm.qkernel
def prepare_projected_signal(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    """Prepare a nontrivial signal basis for a projected block encoding."""

    return qm.h(state)


@qm.qkernel
def projected_x_oracle(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    """Apply X to the system within one joint signal/system state."""

    return qm.x(state, target=1)


projected_x_encoding = projected_block_encoding(
    prepare_projected_signal,
    projected_x_oracle,
    signal_width=1,
    system_width=1,
    normalization=1.0,
)
projected_x_unitary = projected_x_encoding.unitary
periodic_translation_encoding = periodic_shift_block_encoding(
    {(1, -1): 1.0j},
    register_sizes=(1, 2),
)
periodic_translation_unitary = periodic_translation_encoding.unitary


@qm.qkernel
def amplitude_state() -> qm.Q[qm.Bits[1]]:
    """Prepare the positive-Y eigenstate from complex amplitudes."""

    return y_eigenstate(qm.qubits(1))


@qm.qkernel
def amplitude_sample() -> qm.Shot[qm.Bits[1]]:
    """Sample the positive-Y eigenstate in the computational basis."""

    return qm.measure(amplitude_state())


@qm.qkernel
def amplitude_two_state() -> qm.Q[qm.Bits[2]]:
    """Exercise the multilevel Gray walk on a deterministic boundary state."""

    return basis_eleven(qm.qubits(2))


@qm.qkernel
def runtime_amplitude_two_state(
    ry_angles: qm.Param[qm.Vector[qm.Real, 3]],
    rz_angles: qm.Param[qm.Vector[qm.Real, 3]],
    phase: qm.Param[qm.Real],
) -> qm.Q[qm.Bits[2]]:
    """Prepare a two-qubit state from one reusable aggregate Param ABI."""

    return runtime_amplitude_two(qm.qubits(2), ry_angles, rz_angles, phase)


@qm.qkernel
def runtime_amplitude_two_sample(
    ry_angles: qm.Param[qm.Vector[qm.Real, 3]],
    rz_angles: qm.Param[qm.Vector[qm.Real, 3]],
    phase: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[2]]:
    """Measure the runtime-parametric two-qubit preparation."""

    return qm.measure(runtime_amplitude_two_state(ry_angles, rz_angles, phase))


def runtime_amplitude_parameters(
    amplitudes: tuple[complex, complex, complex, complex],
) -> dict[str, object]:
    """Return launch bindings for a normalized two-qubit amplitude vector."""

    return amplitude_angles(amplitudes).bindings()


@qm.qkernel
def pauli_block_state() -> qm.Q[qm.Bits[2]]:
    """Apply a complex-coefficient Pauli block encoding to zero input."""

    return pauli_x_unitary(qm.qubits(2))


@qm.qkernel
def pauli_block_sample() -> qm.Shot[qm.Bits[2]]:
    """Measure the deterministic signal/system Pauli block result."""

    return qm.measure(pauli_block_state())


@qm.qkernel
def projected_block_state() -> qm.Q[qm.Bits[2]]:
    """Apply the generic projected-oracle construction to zero input."""

    return projected_x_unitary(qm.qubits(2))


@qm.qkernel
def projected_block_sample() -> qm.Shot[qm.Bits[2]]:
    """Measure the deterministic generic projected block result."""

    return qm.measure(projected_block_state())


@qm.qkernel
def periodic_block_state() -> qm.Q[qm.Bits[4]]:
    """Apply a two-axis periodic translation to zero input."""

    return periodic_translation_unitary(qm.qubits(4))


@qm.qkernel
def periodic_block_sample() -> qm.Shot[qm.Bits[4]]:
    """Measure the deterministic periodic signal/system result."""

    return qm.measure(periodic_block_state())


def exercise_pauli_block_backend(*, endpoint: object, seed: int = 0) -> None:
    """Sample and estimate the functional Pauli producer on one backend."""

    assert qm.run(  # type: ignore[arg-type]
        pauli_block_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result.counts == {(0, 1): 4}
    value = qm.expectation(  # type: ignore[arg-type]
        pauli_block_state,
        qm.Z(1),
        on=endpoint,
        shots=4,
        seed=seed,
    ).value
    assert value == pytest.approx(-1.0, abs=1e-12, rel=0.0)


def exercise_functional_block_producers_backend(
    *,
    endpoint: object,
    seed: int = 0,
) -> None:
    """Execute generic projected and periodic producers on one backend."""

    assert qm.run(  # type: ignore[arg-type]
        projected_block_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result.counts == {(0, 1): 4}
    assert qm.run(  # type: ignore[arg-type]
        periodic_block_sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result.counts == {(0, 1, 1, 1): 4}
    value = qm.expectation(  # type: ignore[arg-type]
        periodic_block_state,
        qm.Z(3),
        on=endpoint,
        shots=4,
        seed=seed,
    ).value
    assert value == pytest.approx(-1.0, abs=1e-12, rel=0.0)


def exercise_runtime_amplitude_backend(
    *,
    endpoint: object,
    target: object,
    seed: int = 0,
) -> None:
    """Compile once and exercise aggregate runtime coordinates on one backend."""

    zero = runtime_amplitude_parameters((1.0, 0.0, 0.0, 0.0))
    eleven = runtime_amplitude_parameters((0.0, 0.0, 0.0, 1.0))
    artifact = qm.compile(runtime_amplitude_two_sample.specialize(), to=target)  # type: ignore[arg-type]
    assert qm.execute(  # type: ignore[arg-type]
        artifact,
        on=endpoint,
        parameters=zero,
        shots=4,
        seed=seed,
    ).counts == {(0, 0): 4}
    assert qm.execute(  # type: ignore[arg-type]
        artifact,
        on=endpoint,
        parameters=eleven,
        shots=4,
        seed=seed,
    ).counts == {(1, 1): 4}
    value = qm.expectation(  # type: ignore[arg-type]
        runtime_amplitude_two_state,
        qm.Z(0),
        on=endpoint,
        parameters=eleven,
        shots=4,
        seed=seed,
    ).value
    assert value == pytest.approx(-1.0, abs=1e-12, rel=0.0)


@qm.qkernel
def amplitude_two_sample() -> qm.Shot[qm.Bits[2]]:
    """Sample the two-qubit amplitude-preparation boundary state."""

    return qm.measure(amplitude_two_state())


@qm.qkernel
def lcu_qsvt_state() -> qm.Q[qm.Bits[2]]:
    """Apply degree-one QSVT to the LCU encoding of ``(I - Z) / 2``."""

    return projector_qsvt(qm.qubits(2))


@qm.qkernel
def lcu_qsvt_sample() -> qm.Shot[qm.Bits[2]]:
    """Observe the LCU selector rejecting the minus-one Z eigenspace."""

    return qm.measure(lcu_qsvt_state())


@qm.qkernel
def qsvt_state() -> qm.Q[qm.Bits[2]]:
    """Apply a degree-one QSVT sequence for the encoded Pauli X block."""

    return qsvt_x(qm.qubits(2))


@qm.qkernel
def qsvt_sample() -> qm.Shot[qm.Bits[2]]:
    """Measure the deterministic degree-one QSVT result."""

    return qm.measure(qsvt_state())


@qm.qkernel
def reciprocal_x_state() -> qm.Q[qm.Bits[3]]:
    """Apply the realified scaled reciprocal of the Pauli-X encoding."""

    return reciprocal_x_transform(qm.qubits(3))


@qm.qkernel
def reciprocal_x_sample() -> qm.Shot[qm.Bits[3]]:
    """Measure the complete reciprocal signal and transformed system."""

    return qm.measure(reciprocal_x_state())


@qm.qkernel
def linear_system_reciprocal_state() -> qm.Q[qm.Bits[3]]:
    """Apply a nontrivial scaled reciprocal to an equal superposition."""

    state = qm.h(qm.qubits(3), target=2)
    return linear_system_transform(state)


@qm.qkernel
def runtime_qsvt_sample(
    phases: qm.Param[qm.Vector[qm.Real, 2]],
) -> qm.Shot[qm.Bits[2]]:
    """Measure degree-one QSVT with a reusable runtime phase vector."""

    return qm.measure(runtime_qsvt_x(qm.qubits(2), phases))


@qm.qkernel
def static_vector_qsvt_sample() -> qm.Shot[qm.Bits[2]]:
    """Call the vector-phase QSVT body with a literal phase Vector."""

    return qm.measure(runtime_qsvt_x(qm.qubits(2), [0.0, 0.0]))


@qm.qkernel
def qsvt_roundtrip_sample() -> qm.Shot[qm.Bits[2]]:
    """Observe the alternating U/U-adjoint sequence returning to identity."""

    return qm.measure(qsvt_x_roundtrip(qm.qubits(2)))


@qm.qkernel
def qsvt_phase_state() -> qm.Q[qm.Bits[2]]:
    """Expose a nontrivial projector phase as a deterministic interference test."""

    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qsvt_half_turn(state)
    return qm.h(state, target=0)


@qm.qkernel
def qsvt_phase_sample() -> qm.Shot[qm.Bits[2]]:
    """Measure the all-zero-projector phase interference result."""

    return qm.measure(qsvt_phase_state())


@qm.qkernel
def qpe_state() -> qm.Q[qm.Bits[3]]:
    """Prepare a joint counting/eigenstate register and estimate its phase."""

    target = qm.prepare((1,))
    phase, target = phase_estimation(sign_phase, target, 2)
    return qm.concat(phase, target)


@qm.qkernel
def qpe_quarter_state() -> qm.Q[qm.Bits[4]]:
    """Prepare a three-bit estimate of the eigenphase one quarter."""

    target = qm.prepare((1,))
    phase, target = phase_estimation(quarter_phase, target, 3)
    return qm.concat(phase, target)


@qm.qkernel
def qpe_quarter_fixed_sample() -> qm.Shot[qm.UFixed[0, 3]]:
    """Return a numeric phase while consuming the retained eigenstate."""

    target = qm.prepare((1,))
    phase, target = phase_estimation(quarter_phase, target, 3)
    phase_value = qm.measure(phase)
    _target_value = qm.measure(target)
    return phase_value


@qm.qkernel
def qpe_eighth_state() -> qm.Q[qm.Bits[4]]:
    """Prepare a three-bit estimate of the eigenphase one eighth."""

    target = qm.prepare((1,))
    phase, target = phase_estimation(eighth_phase, target, 3)
    return qm.concat(phase, target)


@qm.qkernel
def qpe_eighth_fixed_sample() -> qm.Shot[qm.UFixed[0, 3]]:
    """Expose a non-palindromic numeric phase for bit-order regression tests."""

    target = qm.prepare((1,))
    phase, target = phase_estimation(eighth_phase, target, 3)
    phase_value = qm.measure(phase)
    _target_value = qm.measure(target)
    return phase_value


@qm.qkernel
def qpe_sample() -> qm.Shot[qm.Bits[3]]:
    """Measure the exact estimate together with its retained target."""

    target = qm.prepare((1,))
    phase, target = phase_estimation(sign_phase, target, 2)
    return qm.measure(qm.concat(phase, target))


@qm.qkernel
def qpe_one_bit_state() -> qm.Q[qm.Bits[2]]:
    """Build the rotation-free one-bit profile used by direct HUGR."""

    target = qm.prepare((1,))
    phase, target = phase_estimation(sign_phase, target, 1)
    return qm.concat(phase, target)


@qm.qkernel
def qpe_one_bit_fixed_sample() -> qm.Shot[qm.UFixed[0, 1]]:
    """Return the rotation-free half phase as one numeric sampled value."""

    target = qm.prepare((1,))
    phase, target = phase_estimation(sign_phase, target, 1)
    phase_value = qm.measure(phase)
    _target_value = qm.measure(target)
    return phase_value


@qm.qkernel
def qpe_one_bit_sample() -> qm.Shot[qm.Bits[2]]:
    """Measure the exact one-bit estimate and its retained eigenstate."""

    target = qm.prepare((1,))
    phase, target = phase_estimation(sign_phase, target, 1)
    return qm.measure(qm.concat(phase, target))


@qm.qkernel
def windowed_qpe_one_bit_state() -> qm.Q[qm.Bits[2]]:
    """Estimate the half phase from an explicitly prepared rectangular window."""

    window = qm.h(qm.qubits(1))
    target = qm.prepare((1,))
    phase, target = windowed_phase_estimation(sign_phase, window, target, 1)
    return qm.concat(phase, target)


@qm.qkernel
def windowed_qpe_one_bit_sample() -> qm.Shot[qm.UFixed[0, 1]]:
    """Sample the half phase from the shared explicit-window QPE body."""

    window = qm.h(qm.qubits(1))
    target = qm.prepare((1,))
    phase, target = windowed_phase_estimation(sign_phase, window, target, 1)
    phase_value = qm.measure(phase)
    _target_value = qm.measure(target)
    return phase_value


@qm.qkernel
def grover_state() -> qm.Q[qm.Bits[2]]:
    """Prepare the marked two-bit state with one Grover iteration."""

    return amplify_eleven(qm.qubits(2), 1)


@qm.qkernel
def grover_sample() -> qm.Shot[qm.Bits[2]]:
    """Measure the marked result of functional amplitude amplification."""

    return qm.measure(amplify_eleven(qm.qubits(2), 1))


@qm.qkernel
def increment_state() -> qm.Q[qm.Bits[3]]:
    """Increment little-endian three to four."""

    return increment_three(qm.prepare((1, 1, 0)))


@qm.qkernel
def increment_sample() -> qm.Shot[qm.Bits[3]]:
    """Measure little-endian four after incrementing three."""

    return qm.measure(increment_three(qm.prepare((1, 1, 0))))


@qm.qkernel
def decrement_sample() -> qm.Shot[qm.Bits[3]]:
    """Decrement little-endian four to three and measure it."""

    return qm.measure(decrement_three(qm.prepare((0, 0, 1))))


@qm.qkernel
def add_constant_state() -> qm.Q[qm.Bits[3]]:
    """Add three to little-endian six modulo eight, producing one."""

    return add_three(qm.prepare((0, 1, 1)))


@qm.qkernel
def add_constant_sample() -> qm.Shot[qm.Bits[3]]:
    """Measure constant addition across the modulo-eight boundary."""

    return qm.measure(add_constant_state())


@qm.qkernel
def subtract_constant_sample() -> qm.Shot[qm.Bits[3]]:
    """Apply the compiler-derived inverse, mapping one back to six."""

    return qm.measure(subtract_three(qm.prepare((1, 0, 0))))


@qm.qkernel
def add_register_state() -> qm.Q[qm.Bits[6]]:
    """Retain addend three while adding it to target six modulo eight."""

    state = qm.prepare((1, 1, 0, 0, 1, 1))
    return add_register_pair(state)


@qm.qkernel
def add_register_sample() -> qm.Shot[qm.Bits[6]]:
    """Measure register addition across the modulo-eight boundary."""

    return qm.measure(add_register_state())


@qm.qkernel
def subtract_register_sample() -> qm.Shot[qm.Bits[6]]:
    """Retain addend three while subtracting it from target one."""

    state = qm.prepare((1, 1, 0, 1, 0, 0))
    return qm.measure(subtract_register_pair(state))


@qm.qkernel
def add_modulo_state() -> qm.Q[qm.Bits[5]]:
    """Add three to four modulo five and restore both work components."""

    return add_three_mod_five(qm.prepare((0, 0, 0, 1, 0)))


@qm.qkernel
def add_modulo_sample() -> qm.Shot[qm.Bits[5]]:
    """Measure modular constant addition and its restored workspace."""

    return qm.measure(add_modulo_state())


@qm.qkernel
def subtract_modulo_sample() -> qm.Shot[qm.Bits[5]]:
    """Subtract three from two modulo five and restore both work components."""

    return qm.measure(subtract_three_mod_five(qm.prepare((0, 0, 1, 0, 0))))


@qm.qkernel
def controlled_add_modulo_sample() -> qm.Shot[qm.Bits[6]]:
    """Coherently enable modular addition through the ordinary control API."""

    state = qm.prepare((1, 0, 0, 0, 1, 0))
    control, modular = qm.split_at(state, 1)
    control, modular = qm.control(add_three_mod_five, control, modular)
    return qm.measure(qm.concat(control, modular))


@qm.qkernel
def add_register_modulo_state() -> qm.Q[qm.Bits[9]]:
    """Add two to two modulo three and restore all retained state."""

    state = qm.prepare((0, 1, 1, 1, 0, 1, 0, 0, 0))
    return add_register_mod_three(state)


@qm.qkernel
def add_register_modulo_sample() -> qm.Shot[qm.Bits[9]]:
    """Measure register-modulus addition and its restored workspace."""

    return qm.measure(add_register_modulo_state())


@qm.qkernel
def subtract_register_modulo_sample() -> qm.Shot[qm.Bits[9]]:
    """Apply the common inverse to restore the register-modulus input."""

    state = qm.prepare((0, 1, 1, 1, 1, 0, 0, 0, 0))
    return qm.measure(subtract_register_mod_three(state))


@qm.qkernel
def controlled_add_register_modulo_sample() -> qm.Shot[qm.Bits[10]]:
    """Control register-modulus addition through the ordinary transform."""

    state = qm.prepare((1, 0, 1, 1, 1, 0, 1, 0, 0, 0))
    control_state, arithmetic = qm.split_at(state, 1)
    control_state, arithmetic = qm.control(
        add_register_mod_three,
        control_state,
        arithmetic,
    )
    return qm.measure(qm.concat(control_state, arithmetic))


@qm.qkernel
def multiply_modulo_state() -> qm.Q[qm.Bits[8]]:
    """Multiply three by two modulo five and restore the accumulator."""

    return multiply_two_mod_five(qm.prepare((1, 1, 0, 0, 0, 0, 0, 0)))


@qm.qkernel
def multiply_modulo_sample() -> qm.Shot[qm.Bits[8]]:
    """Measure in-place constant modular multiplication."""

    return qm.measure(multiply_modulo_state())


@qm.qkernel
def divide_modulo_sample() -> qm.Shot[qm.Bits[8]]:
    """Apply the derived inverse, mapping one back to three modulo five."""

    return qm.measure(divide_two_mod_five(qm.prepare((1, 0, 0, 0, 0, 0, 0, 0))))


@qm.qkernel
def controlled_multiply_modulo_sample() -> qm.Shot[qm.Bits[7]]:
    """Coherently enable modular multiplication through ordinary control."""

    state = qm.prepare((1, 1, 0, 0, 0, 0, 0))
    control, multiplication = qm.split_at(state, 1)
    control, multiplication = qm.control(
        multiply_two_mod_three,
        control,
        multiplication,
    )
    return qm.measure(qm.concat(control, multiplication))


@qm.qkernel
def less_than_state() -> qm.Q[qm.Bits[6]]:
    """Compare unsigned two and three while restoring the work component."""

    return less_than_two(qm.prepare((0, 1, 1, 1, 0, 0)))


@qm.qkernel
def less_than_sample() -> qm.Shot[qm.Bits[6]]:
    """Measure the coherent unsigned less-than result."""

    return qm.measure(less_than_state())


@qm.qkernel
def multiply_register_state() -> qm.Q[qm.Bits[8]]:
    """Multiply unsigned two by three into a four-bit accumulator."""

    return multiply_pair_two(qm.prepare((0, 1, 1, 1, 0, 0, 0, 0)))


@qm.qkernel
def multiply_register_sample() -> qm.Shot[qm.Bits[8]]:
    """Measure the coherent two-register product."""

    return qm.measure(multiply_register_state())


@qm.qkernel
def gf8_product_state() -> qm.Q[qm.Bits[9]]:
    """Multiply five by three in GF(2^3) with modulus x^3 + x + 1."""

    return gf8_multiply_add(qm.prepare((1, 0, 1, 1, 1, 0, 0, 0, 0)))


@qm.qkernel
def gf8_product_sample() -> qm.Shot[qm.Bits[9]]:
    """Measure retained GF(2^3) factors and their product."""

    return qm.measure(gf8_product_state())


def _measure_basis_transform(
    transform: qm.QKernel,
    bits: tuple[int, ...],
) -> qm.QKernel:
    """Close one transform and basis input into a measured function."""

    total = len(bits)

    @qm.qkernel
    def applied() -> qm.Shot[qm.Bits[total]]:  # type: ignore[valid-type]
        return qm.measure(transform(qm.prepare(bits)))

    return applied


def _little_endian(value: int, width: int) -> tuple[int, ...]:
    """Return one fixed-width little-endian classical bit tuple."""

    return tuple((value >> index) & 1 for index in range(width))


def _gf2_product(left: int, right: int, *, width: int, modulus: int) -> int:
    """Return one classical binary-polynomial product remainder."""

    product = 0
    for index in range(width):
        if right & (1 << index):
            product ^= left << index
    while product.bit_length() - 1 >= width:
        product ^= modulus << (product.bit_length() - width - 1)
    return product


def register_modulus_case(
    seed: int,
) -> tuple[qm.QKernel, tuple[int, ...]]:
    """Build one seeded register-modulus sample case."""

    generator = random.Random(seed)
    width = generator.choice((1, 2, 3))
    modulus = generator.randrange(1, 1 << width)
    addend = generator.randrange(modulus)
    target = generator.randrange(modulus)
    transform = add_register_modulo(width=width)
    total = 3 * width + 3
    input_bits = (
        *_little_endian(addend, width),
        *_little_endian(modulus, width),
        *_little_endian(target, width),
        0,
        0,
        0,
    )
    output_bits = (
        *_little_endian(addend, width),
        *_little_endian(modulus, width),
        *_little_endian((target + addend) % modulus, width),
        0,
        0,
        0,
    )

    @qm.qkernel
    def state() -> qm.Q[qm.Bits[total]]:  # type: ignore[valid-type]
        return transform(qm.prepare(input_bits))

    @qm.qkernel
    def sample() -> qm.Shot[qm.Bits[total]]:  # type: ignore[valid-type]
        return qm.measure(state())

    return sample, output_bits


def exercise_register_modulus_backend(*, endpoint: object, seed: int = 0) -> None:
    """Execute one retained-modulus sample witness."""

    sample, expected = register_modulus_case(seed)
    assert qm.run(  # type: ignore[arg-type]
        sample,
        on=endpoint,
        shots=4,
        seed=seed,
    ).result.counts == {expected: 4}


def test_imported_stdlib_functions_are_statically_linked() -> None:
    """A QKernel closes imported source functions into its portable module."""

    program = qft_roundtrip.specialize()
    study = qm.resources.logical(qft_roundtrip)

    assert program.entry.kind is CallableKind.INSTRUMENT
    assert qm.evaluate(program) == (1, 0, 1)
    assert qm.run(qft_roundtrip, on=qm.Reference(), shots=4).result.counts == {
        (1, 0, 1): 4
    }
    assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (14, 14)
    assert "qcore.adjoint_call" in inverse_qft.module().inspect()
    assert {function.symbol.rsplit(".", 1)[-1] for function in program.functions} >= {
        "qft",
        "inverse_qft",
        "qft_roundtrip",
    }


def test_qft_preserves_little_endian_frequency_order() -> None:
    """An asymmetric Fourier mode decodes to its little-endian integer."""

    assert qm.run(
        qft_frequency_three,
        on=qm.Reference(),
        shots=4,
        seed=2,
    ).result.counts == {(1, 1, 0): 4}


def test_structured_adjoint_lowering_cannot_reenter_semantic_analysis() -> None:
    """Compact target lowering cannot masquerade as verified semantic QCore."""

    program = qft_roundtrip.specialize()
    lowered = lower_adjoint_calls(program)

    assert "qcore.adjoint_call" in program.inspect()
    assert "reverse=True" in lowered.inspect()
    assert qm.evaluate(program) == (1, 0, 1)
    assert qm.resources.logical(program).metric(
        qm.resources.LOGICAL_GATES
    ).numeric_enclosure() == (14, 14)
    assert not hasattr(lowered, "to_bytes")
    with pytest.raises(TypeError, match="logical expects"):
        qm.resources.logical(lowered)


def test_automatic_uncomputation_reuses_the_structured_adjoint() -> None:
    """Cleanup of a library call uses the same checked inverse transform."""

    program = qft_temporary.specialize()

    assert "qcore.adjoint_call" in program.inspect()
    assert qm.evaluate(program) is None
    assert qm.resources.logical(program).metric(
        qm.resources.LOGICAL_CLEANUP_WORK
    ).numeric_enclosure() == (8, 8)


def test_structured_adjoint_composes_through_polymorphic_calls() -> None:
    """Nested callable inversion reuses call-site shape and stage substitution."""

    assert qm.evaluate(nested_qft_roundtrip.specialize()) == (1, 1, 0)


def test_phase_estimation_is_composed_source_not_a_compiler_primitive() -> None:
    """QPE derives execution and resources from ordinary linked function bodies."""

    program = qpe_sample.specialize()
    restored = qm.Program.from_bytes(program.to_bytes())

    assert qm.evaluate(restored, seed=0) == (1, 0, 1)
    assert "qcore.control_call" in restored.inspect()
    assert "quantum.qpe" not in restored.inspect()
    assert (
        Requirement.ROTATION
        in lower_adjoint_calls(qpe_state.specialize()).entry.requirements
    )
    qpe_gates = qm.resources.logical(qpe_state).metric(qm.resources.LOGICAL_GATES)
    assert qpe_gates.relation is qm.resources.Relation.PARTIAL_ENCLOSURE
    assert qpe_gates.numeric_enclosure() == (10, None)


def test_phase_estimation_fixed_value_uses_generic_types_and_result_abi() -> None:
    """QPE exposes a numeric phase without dedicated IR or backend handling."""

    program = qpe_quarter_fixed_sample.specialize()
    restored = qm.Program.from_bytes(program.to_bytes())
    record = qm.run(restored, on=qm.Reference(), shots=4, seed=2)

    assert restored.entry.result_type == qm.Shot[qm.UFixed[0, 3]]
    assert "Q[UFixed[0, precision]] = qcore.reshape" in restored.inspect()
    assert "fixed.measure" not in restored.inspect()
    assert "fixed.decode" not in restored.inspect()
    assert record.result.counts == {(0, 1, 0): 4}
    assert record.result.values == {0.25: 4}


def test_phase_estimation_is_not_a_host_factory() -> None:
    """QPE is one source declaration and cannot manufacture kernels at runtime."""

    assert isinstance(phase_estimation, qm.QKernel)
    with pytest.raises(TypeError, match="source declarations"):
        phase_estimation(sign_phase, object(), 2)  # type: ignore[arg-type]


def test_windowed_phase_estimation_reuses_the_textbook_qpe_body() -> None:
    """An explicit rectangular window preserves QPE values and persistence."""

    program = windowed_qpe_one_bit_sample.specialize()
    restored = qm.Program.from_bytes(program.to_bytes())
    record = qm.run(restored, on=qm.Reference(), shots=4, seed=2)

    assert restored.entry.result_type == qm.Shot[qm.UFixed[0, 1]]
    assert "qcore.control_call" in restored.inspect()
    assert "quantum.qpe" not in restored.inspect()
    assert record.result.counts == {(1,): 4}
    assert record.result.values == {0.5: 4}


def test_grover_search_is_body_composition_with_symbolic_iterations() -> None:
    """Grover execution and resources follow its linked oracle and source body."""

    program = grover_sample.specialize()
    symbolic = qm.resources.logical(amplify_eleven)

    assert qm.evaluate(program, seed=0) == (1, 1)
    assert "qcore.control_call" in program.inspect()
    assert "quantum.grover" not in program.inspect()
    grover_gates = qm.resources.logical(grover_state).metric(qm.resources.LOGICAL_GATES)
    assert grover_gates.relation is qm.resources.Relation.PARTIAL_ENCLOSURE
    assert grover_gates.numeric_enclosure() == (14, None)
    specialized_gates = symbolic.evaluate(iterations=3).metric(
        qm.resources.LOGICAL_GATES
    )
    assert specialized_gates.relation is qm.resources.Relation.PARTIAL_ENCLOSURE
    assert specialized_gates.numeric_enclosure() == (38, None)


def test_grover_search_validates_its_host_configuration() -> None:
    """The Grover factory rejects ambiguous callables and dimensions."""

    import pytest

    with pytest.raises(TypeError, match="QKernel"):
        grover_search(object(), width=2)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="width"):
        grover_search(mark_eleven, width=True)
    with pytest.raises(ValueError, match="width"):
        grover_search(mark_eleven, width=0)


def test_modular_increment_and_derived_decrement_are_function_composition() -> None:
    """Arithmetic and its inverse use existing calls, control, and adjoint only."""

    decrement_program = decrement_sample.specialize()
    forward = qm.resources.logical(increment_three)
    reverse = qm.resources.logical(decrement_three)

    assert qm.evaluate(increment_sample.specialize()) == (0, 0, 1)
    assert qm.evaluate(decrement_program) == (1, 1, 0)
    assert qm.run(increment_sample, on=qm.Reference(), shots=4).result.counts == {
        (0, 0, 1): 4
    }
    assert "qcore.control_call" in decrement_program.inspect()
    assert "quantum.increment" not in decrement_program.inspect()
    assert "qcore.adjoint_call" in decrement_three.module().inspect()
    assert forward.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (
        reverse.metric(qm.resources.LOGICAL_GATES).numeric_enclosure()
    )


def test_modular_arithmetic_validates_width_and_one_bit_boundary() -> None:
    """Host factories reject ambiguous widths and preserve the one-bit case."""

    import pytest

    assert modular_increment(width=1).symbol.endswith(".flip")
    with pytest.raises(TypeError, match="width"):
        modular_increment(width=True)
    with pytest.raises(ValueError, match="width"):
        modular_decrement(width=0)


def test_constant_addition_scales_by_width_and_derives_subtraction() -> None:
    """Set-bit suffix increments implement signed constants without new IR."""

    addition = add_constant_sample.specialize()
    subtraction = subtract_constant_sample.specialize()

    assert qm.evaluate(addition) == (1, 0, 0)
    assert qm.evaluate(subtraction) == (0, 1, 1)
    assert (
        qm.resources.logical(add_three)
        .metric(qm.resources.LOGICAL_GATES)
        .numeric_enclosure()
        == qm.resources.logical(subtract_three)
        .metric(qm.resources.LOGICAL_GATES)
        .numeric_enclosure()
    )
    assert "quantum.add_constant" not in addition.inspect()
    assert "qcore.adjoint_call" in subtract_three.module().inspect()
    assert isinstance(add_constant(width=3, value=-1), qm.QKernel)
    assert isinstance(add_constant(width=3, value=-(1 << 10_000) + 5), qm.QKernel)
    with pytest.raises(TypeError, match="value"):
        add_constant(width=3, value=True)
    with pytest.raises(ValueError, match="4096-qubit budget"):
        add_constant(width=4097, value=1)


def test_arithmetic_factories_require_well_formed_native_plans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generated arithmetic topology has one fail-closed native authority."""

    import importlib

    arithmetic = importlib.import_module("qamomile.stdlib.arithmetic")

    monkeypatch.setattr(arithmetic.native, "wrapping_addition_offsets", None)
    with pytest.raises(TypeError, match="Q7636"):
        add_constant(width=3, value=1)
    monkeypatch.setattr(
        arithmetic.native,
        "wrapping_addition_offsets",
        lambda _width, _value: [True],
    )
    with pytest.raises(TypeError, match="Q7637"):
        add_constant(width=3, value=1)

    monkeypatch.setattr(arithmetic.native, "modular_addition_residue", None)
    with pytest.raises(TypeError, match="Q7636"):
        internal_add_constant_modulo(width=3, value=1, modulus=5)
    monkeypatch.setattr(
        arithmetic.native,
        "modular_addition_residue",
        lambda _width, _value, _modulus: "5",
    )
    with pytest.raises(TypeError, match="Q7637"):
        internal_add_constant_modulo(width=3, value=1, modulus=5)

    monkeypatch.setattr(arithmetic.native, "plan_modular_multiplication", None)
    with pytest.raises(TypeError, match="Q7636"):
        multiply_constant_modulo(width=3, value=2, modulus=5)
    monkeypatch.setattr(
        arithmetic.native,
        "plan_modular_multiplication",
        lambda _width, _value, _modulus: SimpleNamespace(
            width=3,
            modulus="5",
            multiplier="2",
            inverse="3",
            accumulate=["2"],
            erase=["3"],
        ),
    )
    with pytest.raises(TypeError, match="Q7637"):
        multiply_constant_modulo(width=3, value=2, modulus=5)


def test_register_arithmetic_retains_semantic_addition_and_common_adjoint() -> None:
    """Register addition remains abstract and subtraction uses the shared adjoint."""

    addition = add_register_sample.specialize()
    subtraction = subtract_register_sample.specialize()

    assert qm.evaluate(addition) == (1, 1, 0, 1, 0, 0)
    assert qm.evaluate(subtraction) == (1, 1, 0, 0, 1, 1)
    assert (
        qm.resources.logical(add_register_pair)
        .metric(qm.resources.LOGICAL_GATES)
        .numeric_enclosure()
        == qm.resources.logical(subtract_register_pair)
        .metric(qm.resources.LOGICAL_GATES)
        .numeric_enclosure()
    )
    assert "quantum.add_into" in addition.inspect()
    assert "qcore.adjoint_call" in subtract_register_pair.module().inspect()


def test_modular_constant_arithmetic_matches_classical_permutations() -> None:
    """Nontrivial signed residues add in place and restore workspace."""
    width, modulus, value, target = (2, 3, -2, 1)
    addition = internal_add_constant_modulo(width=width, value=value, modulus=modulus)
    subtraction = internal_subtract_constant_modulo(
        width=width,
        value=value,
        modulus=modulus,
    )
    input_bits = (0, *_little_endian(target, width), 0)
    added = qm.evaluate(_measure_basis_transform(addition, input_bits).specialize())
    restored = qm.evaluate(_measure_basis_transform(subtraction, added).specialize())

    assert added == (0, *_little_endian((target + value) % modulus, width), 0)
    assert restored == input_bits
    assert "quantum.modular" not in addition.module().inspect()


def test_total_modular_multiplication_fixes_out_of_range_states() -> None:
    """Constant modular multiplication is total over its value register."""

    multiplication = multiply_modulo_constant(width=3, value=2, modulus=5)
    for basis in range(8):
        bits = _little_endian(basis, 3)
        multiplied = qm.evaluate(
            _measure_basis_transform(multiplication, bits).specialize()
        )
        expected_multiply = (basis * 2) % 5 if basis < 5 else basis
        assert multiplied == _little_endian(expected_multiply, 3)


def test_total_register_modular_ux_is_identity_outside_its_domain() -> None:
    """Register modulus arithmetic defines every computational basis state."""

    addition = add_modulo(width=2)
    valid = (*_little_endian(2, 2), *_little_endian(3, 2), *_little_endian(2, 2))
    invalid = (*_little_endian(3, 2), *_little_endian(3, 2), *_little_endian(1, 2))

    assert qm.evaluate(_measure_basis_transform(addition, valid).specialize()) == (
        *_little_endian(2, 2),
        *_little_endian(3, 2),
        *_little_endian(1, 2),
    )
    assert (
        qm.evaluate(_measure_basis_transform(addition, invalid).specialize()) == invalid
    )


def test_comparison_ux_returns_flag_but_not_clean_workspace() -> None:
    """Comparison retains operands and exposes only its semantic output flag."""

    comparison = compare_unsigned(width=2)
    bits = (*_little_endian(1, 2), *_little_endian(2, 2))

    @qm.qkernel
    def sample() -> qm.Shot[qm.Bits[5]]:
        return qm.measure(comparison(qm.prepare(bits)))

    result = qm.evaluate(sample.specialize())

    assert result == (*bits, 1)


def test_modular_constant_arithmetic_derives_control_and_inverse_generically() -> None:
    """The fixed conformance body uses shared control, adjoint, and resources."""

    assert qm.evaluate(add_modulo_sample.specialize()) == (0, 0, 1, 0, 0)
    assert qm.evaluate(subtract_modulo_sample.specialize()) == (0, 0, 0, 1, 0)
    assert qm.evaluate(controlled_add_modulo_sample.specialize()) == (
        1,
        0,
        0,
        1,
        0,
        0,
    )
    assert (
        qm.resources.logical(add_three_mod_five)
        .metric(qm.resources.LOGICAL_GATES)
        .numeric_enclosure()
        == qm.resources.logical(subtract_three_mod_five)
        .metric(qm.resources.LOGICAL_GATES)
        .numeric_enclosure()
    )
    assert "qcore.adjoint_call" in subtract_three_mod_five.module().inspect()
    with pytest.raises(TypeError, match="modulus"):
        internal_add_constant_modulo(width=3, value=1, modulus=True)
    with pytest.raises(ValueError, match="modulus"):
        internal_add_constant_modulo(width=3, value=1, modulus=8)


def test_register_modulus_arithmetic_matches_classical_permutations() -> None:
    """A nontrivial bounded triple adds modulo three and restores all work."""

    width = 2
    modulus = 3
    addend = 2
    target = 2
    addition = add_register_modulo(width=width)
    subtraction = subtract_register_modulo(width=width)
    input_bits = (
        *_little_endian(addend, width),
        *_little_endian(modulus, width),
        *_little_endian(target, width),
        0,
        0,
        0,
    )
    added = qm.evaluate(_measure_basis_transform(addition, input_bits).specialize())
    restored = qm.evaluate(_measure_basis_transform(subtraction, added).specialize())

    assert added == (
        *_little_endian(addend, width),
        *_little_endian(modulus, width),
        *_little_endian((target + addend) % modulus, width),
        0,
        0,
        0,
    )
    assert restored == input_bits
    assert "quantum.register_modulus" not in addition.module().inspect()


def test_register_modulus_arithmetic_reuses_tensor_symmetry_and_adjoint() -> None:
    """Register-modulus variants need no new semantic or resource branch."""

    assert qm.evaluate(controlled_add_register_modulo_sample.specialize()) == (
        1,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
    )
    assert (
        qm.resources.logical(add_register_mod_three)
        .metric(qm.resources.LOGICAL_GATES)
        .numeric_enclosure()
        == qm.resources.logical(subtract_register_mod_three)
        .metric(qm.resources.LOGICAL_GATES)
        .numeric_enclosure()
    )
    assert "qcore.adjoint_call" in subtract_register_mod_three.module().inspect()
    with pytest.raises(TypeError, match="width"):
        add_register_modulo(width=True)
    with pytest.raises(ValueError, match="width"):
        add_register_modulo(width=0)


def test_modular_constant_multiplication_matches_classical_permutations() -> None:
    """Nonidentity units multiply in place and the common adjoint restores input."""
    width, modulus, value, target = (2, 3, 2, 1)
    multiplication = multiply_constant_modulo(
        width=width,
        value=value,
        modulus=modulus,
    )
    division = divide_constant_modulo(
        width=width,
        value=value,
        modulus=modulus,
    )
    input_bits = (*_little_endian(target, width), *((0,) * (width + 2)))
    multiplied = qm.evaluate(
        _measure_basis_transform(multiplication, input_bits).specialize()
    )
    restored = qm.evaluate(_measure_basis_transform(division, multiplied).specialize())

    assert multiplied == (
        *_little_endian((target * value) % modulus, width),
        *((0,) * (width + 2)),
    )
    assert restored == input_bits
    assert "quantum.multiply" not in multiplication.module().inspect()


def test_modular_multiplication_derives_control_and_inverse_generically() -> None:
    """The fixed multiplication body needs no inverse or controlled variant."""

    assert qm.evaluate(controlled_multiply_modulo_sample.specialize()) == (
        1,
        0,
        1,
        0,
        0,
        0,
        0,
    )
    assert (
        qm.resources.logical(multiply_two_mod_five)
        .metric(qm.resources.LOGICAL_GATES)
        .numeric_enclosure()
        == qm.resources.logical(divide_two_mod_five)
        .metric(qm.resources.LOGICAL_GATES)
        .numeric_enclosure()
    )
    assert "qcore.adjoint_call" in divide_two_mod_five.module().inspect()
    with pytest.raises(ValueError, match="coprime"):
        multiply_constant_modulo(width=3, value=2, modulus=4)
    with pytest.raises(ValueError, match="512-qubit budget"):
        multiply_constant_modulo(width=513, value=2, modulus=3)
    with pytest.raises(ValueError, match="below the 4096-qubit budget"):
        internal_add_constant_modulo(width=4096, value=1, modulus=3)


def test_unsigned_comparison_matches_every_two_bit_basis_pair() -> None:
    """The coherent flag matches classical order and work is restored."""

    for left in range(4):
        for right in range(4):
            input_bits = (
                *_little_endian(left, 2),
                *_little_endian(right, 2),
                0,
                0,
            )
            compared = qm.evaluate(
                _measure_basis_transform(less_than_two, input_bits).specialize()
            )
            expected = (
                *_little_endian(left, 2),
                *_little_endian(right, 2),
                int(left < right),
                0,
            )

            assert compared == expected
            assert (
                qm.evaluate(
                    _measure_basis_transform(less_than_two, compared).specialize()
                )
                == input_bits
            )

    program = less_than_sample.specialize()
    assert "qcore.adjoint_call" in program.inspect()
    assert "quantum.compare" not in program.inspect()


def test_sorted_duplicate_detection_matches_every_three_item_multiset() -> None:
    """Adjacent comparison detects duplicates and restores every work bit."""

    for items in combinations_with_replacement(range(4), 3):
        input_bits = (
            *(bit for item in items for bit in _little_endian(item, 2)),
            0,
            0,
            0,
            0,
        )
        detected = qm.evaluate(
            _measure_basis_transform(_SORTED_DUPLICATES, input_bits).specialize()
        )
        expected = (*input_bits[:6], int(len(set(items)) < len(items)), 0, 0, 0)

        assert detected == expected
        assert (
            qm.evaluate(
                _measure_basis_transform(_SORTED_DUPLICATES, detected).specialize()
            )
            == input_bits
        )

    program = sorted_duplicates_sample.specialize()
    assert "qcore.adjoint_call" in program.inspect()
    assert "qcore.control_call" in program.inspect()
    assert "quantum.has_duplicates" not in program.inspect()
    with pytest.raises(TypeError, match="length"):
        has_duplicates_sorted(length=True, item_width=2)
    with pytest.raises(ValueError, match="at least two"):
        has_duplicates_sorted(length=1, item_width=2)
    with pytest.raises(ValueError, match="width"):
        has_duplicates_sorted(length=3, item_width=0)


def test_register_multiplication_matches_every_two_bit_basis_pair() -> None:
    """The double-width accumulator receives exact products reversibly."""

    for left in range(4):
        for right in range(4):
            input_bits = (
                *_little_endian(left, 2),
                *_little_endian(right, 2),
                0,
                0,
                0,
                0,
            )
            product = qm.evaluate(
                _measure_basis_transform(multiply_pair_two, input_bits).specialize()
            )

            assert product == (
                *_little_endian(left, 2),
                *_little_endian(right, 2),
                *_little_endian(left * right, 4),
            )

    program = multiply_register_sample.specialize()
    assert "quantum.mul_add" in program.inspect()
    with pytest.raises(TypeError, match="width"):
        multiply_add(width=True)
    with pytest.raises(ValueError, match="width"):
        less_than_registers(width=0)


def test_one_bit_register_multiplication_elides_identity_permutation() -> None:
    """A complete in-order selection keeps no dependent identity callable."""

    multiplication = multiply_add(width=1)
    program = _measure_basis_transform(
        multiplication,
        (1, 1, 0, 0),
    ).specialize()

    assert qm.evaluate(program) == (1, 1, 1, 0)
    assert "qamomile.stdlib._structure.identity" not in program.inspect()


def test_gf2_multiply_add_matches_all_gf8_basis_products() -> None:
    """The coherent polynomial product matches all GF(2^3) basis pairs."""

    for left in range(8):
        for right in range(8):
            input_bits = (
                *_little_endian(left, 3),
                *_little_endian(right, 3),
                0,
                0,
                0,
            )
            product = qm.evaluate(
                _measure_basis_transform(gf8_multiply_add, input_bits).specialize()
            )
            expected = _gf2_product(left, right, width=3, modulus=0b1011)

            assert product == (
                *_little_endian(left, 3),
                *_little_endian(right, 3),
                *_little_endian(expected, 3),
            )
            assert (
                qm.evaluate(
                    _measure_basis_transform(gf8_multiply_add, product).specialize()
                )
                == input_bits
            )

    program = gf8_product_sample.specialize()
    assert "quantum.ccx" in program.inspect()
    assert "quantum.gf" not in program.inspect()
    with pytest.raises(TypeError, match="modulus"):
        gf2_multiply_add(width=3, modulus=True)
    with pytest.raises(ValueError, match="monic"):
        gf2_multiply_add(width=3, modulus=0b0011)
    with pytest.raises(ValueError, match="nonzero constant"):
        gf2_multiply_add(width=3, modulus=0b1010)
    with pytest.raises(ValueError, match="512-qubit budget"):
        gf2_multiply_add(width=171, modulus=(1 << 171) | 1)


@pytest.mark.resource_slow
def test_shor_order_finding_is_symbolic_and_executable() -> None:
    """One symbolic body supports algebraic analysis and specialization."""

    pytest.importorskip("qiskit_aer")
    from qamomile.qiskit import Aer

    record = qm.run(shor_three_sample, on=Aer(), shots=16, seed=2)
    result = record.result
    study = qm.resources.logical(shor_order_finding)
    bindings = {"base": 2, "modulus": 15, "width": 4, "precision": 8}

    assert set(result.counts) == {(0, 0), (1, 0)}
    assert sum(result.counts.values()) == 16
    assert study.metric(qm.resources.LOGICAL_PEAK_QUBITS).evaluate(
        bindings
    ).numeric_enclosure() == (12, 12)
    assert study.metric(
        qm.resources.logical_operation_count("quantum.mod_pow")
    ).evaluate(bindings).numeric_enclosure() == (1, 1)
    assert qm.resources.ResourceStudy.from_bytes(study.to_bytes()) == study
    assert "quantum.shor" not in shor_three_sample.inspect()
    with pytest.raises(qm.SpecializationError, match="Q3007"):
        qm.specialize(
            shor_order_finding,
            base=2,
            modulus=3,
            width=0,
            precision=2,
        )


@pytest.mark.resource_slow
def test_ekera_hastad_is_two_ordinary_phase_estimation_compositions() -> None:
    """Short-DLP construction has executable bodies and no algorithm hook."""

    record = qm.run(ekera_three_sample, on=qm.Reference(), shots=8, seed=2)
    result = record.result
    study = record.resources

    assert sum(result.counts.values()) == 8
    assert all(bits[2:] == (0,) for bits in result.counts)
    assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure()[0] > 0
    assert "quantum.ekera" not in ekera_three_sample.module().inspect()
    with pytest.raises(TypeError, match="parameter_width"):
        ekera_hastad_factoring(generator=2, modulus=3, parameter_width=True)
    with pytest.raises(ValueError, match="parameter_width"):
        ekera_hastad_factoring_state(generator=2, modulus=3, parameter_width=0)


@pytest.mark.resource_slow
def test_regev_factoring_is_composed_and_resource_projectable() -> None:
    """Analyze two Gaussian coordinates without an algorithm-specific rule."""

    sample = regev_factoring(
        bases=(2, 5),
        modulus=33,
        coordinate_width=2,
        radius=1.0,
    )
    logical = qm.resources.logical(sample)
    assert logical.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure() == (
        10,
        10,
    )
    assert logical.metric(
        qm.resources.logical_operation_count("quantum.mod_pow")
    ).numeric_enclosure() == (2, 2)
    unresolved = logical.metric(qm.resources.LOGICAL_GATES)
    assert unresolved.numeric_enclosure() == (19, None)
    assert unresolved.unknown_contributors == (
        "quantum.mod_pow: gate cost requires realization",
    )

    options = qm.RealizationOptions({qm.mod_pow: "mod_pow.square_and_multiply"})
    projected = qm.resources.project(
        logical,
        qm.resources.GateCostModel(realizations=options),
    )
    assert projected.metric(
        qm.resources.portable_metric("gates")
    ).numeric_enclosure() == (68_023, 68_023)
    assert projected.metric(
        qm.resources.portable_metric("gate_depth")
    ).numeric_enclosure() == (67_990, 67_990)
    workspace = projected.metric(qm.resources.portable_metric("clean_ancilla_qubits"))
    # Identity factors can suppress the multiplier workspace.
    assert workspace.relation is qm.resources.Relation.ENCLOSURE
    assert workspace.numeric_enclosure() == (0, 21)
    assert projected.metric(
        qm.resources.CIRCUIT_OUTPUT_TRACE_DISTANCE_BOUND
    ).numeric_enclosure() == (0, 0)
    assert qm.resources.ResourceStudy.from_bytes(projected.to_bytes()) == projected
    assert "quantum.regev" not in sample.module().inspect()


def test_regev_factoring_validates_the_finite_gaussian_domain() -> None:
    """Reject malformed units and configurations outside the paper relation."""

    with pytest.raises(TypeError, match="bases"):
        regev_factoring(  # type: ignore[arg-type]
            bases=[2],
            modulus=3,
            coordinate_width=2,
            radius=1.1,
        )
    with pytest.raises(ValueError, match="coprime"):
        regev_factoring_state(
            bases=(3,),
            modulus=9,
            coordinate_width=2,
            radius=1.1,
        )
    with pytest.raises(ValueError, match="sqrt"):
        regev_factoring(
            bases=(2,),
            modulus=3,
            coordinate_width=2,
            radius=10.0,
        )
    with pytest.raises(ValueError, match="amplitude budget"):
        regev_factoring(
            bases=(2,),
            modulus=3,
            coordinate_width=21,
            radius=float(1 << 19),
        )


def test_shor_order_finding_retains_semantic_modular_power_in_symbolic_ir() -> None:
    """Modular exponentiation stays semantic until realization selection."""

    symbolic = shor_order_finding.module().inspect()
    specialized = shor_three_sample.inspect()

    assert "qcore.for" in symbolic
    assert symbolic.count("quantum.mod_pow") == 1
    assert specialized.count("quantum.mod_pow") == 1


def test_short_dlp_factory_requires_a_well_formed_native_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Short-DLP coordinates fail closed without their native authority."""

    import importlib

    shor = importlib.import_module("qamomile.stdlib.shor")

    monkeypatch.setattr(shor.native, "plan_ekera_hastad", None)
    with pytest.raises(TypeError, match="Q7644"):
        ekera_hastad_factoring(generator=2, modulus=3, parameter_width=1)


def test_qsvt_is_linked_function_composition_without_a_new_semantic_op() -> None:
    """Block encoding, projector rotation, and inverse share existing QCore."""

    program = qsvt_phase_sample.specialize()
    restored = qm.Program.from_bytes(program.to_bytes())

    assert qm.evaluate(qsvt_sample.specialize()) == (0, 1)
    assert qm.evaluate(qsvt_roundtrip_sample.specialize()) == (0, 0)
    assert "qcore.adjoint_call" in qsvt_roundtrip_sample.specialize().inspect()
    assert qm.evaluate(restored) == (1, 0)
    assert "qcore.control_call" in restored.inspect()
    assert "quantum.global_phase" in restored.inspect()
    assert "quantum.qsvt" not in restored.inspect()
    qsvt_gates = qm.resources.logical(qsvt_x).metric(qm.resources.LOGICAL_GATES)
    assert qsvt_gates.relation is qm.resources.Relation.PARTIAL_ENCLOSURE
    assert qsvt_gates.numeric_enclosure() == (1, None)
    short = qsvt(identity_block_encoding, phases=(0.1, 0.2)).module()
    long = qsvt(identity_block_encoding, phases=(0.1,) * 17).module()
    assert len(short.functions) == len(long.functions) == 4
    runtime_program = runtime_qsvt_sample.specialize()
    assert (
        runtime_program.entry.parameters[0].value.type
        == qm.Param[qm.Vector[qm.Real, 2]]
    )
    exercise_runtime_qsvt_backend(endpoint=qm.Reference(), seed=2)


def test_block_encoding_is_bound_after_algorithm_declaration() -> None:
    """One source closes with alternate blocks and persists without Python values."""

    with pytest.raises(qm.CapabilityError, match="without a closed call site"):
        deferred_block_sample.module()
    bound = deferred_block_sample.bind(block=x_block_encoding)
    assert isinstance(bound, qm.QKernel)
    program = bound.specialize()
    restored = qm.Program.from_bytes(program.to_bytes())
    assert qm.evaluate(restored) == (0, 1)
    assert "function_ref" not in restored.inspect()
    assert qm.evaluate(
        deferred_block_sample.specialize(block=identity_block_encoding)
    ) == (
        0,
        0,
    )
    inverse = deferred_qsvt_roundtrip_sample.specialize(block=phase_block_encoding)
    assert "qcore.adjoint_call" in inverse.inspect()
    assert qm.evaluate(inverse, phases=(0.0, 0.0, 0.0)) == (0, 0)
    exercise_deferred_block_backend(endpoint=qm.Reference(), seed=2)
    artifact = qm.compile(
        deferred_block_sample,
        to=qm.Reference().target,
        block=x_block_encoding,
    )
    with pytest.raises(
        TypeError, match="CompiledProgram does not accept Const bindings"
    ):
        qm.run(
            artifact,
            on=qm.Reference(),
            block=identity_block_encoding,
        )


def test_block_binding_leaves_compilation_until_later(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binding an encoded object constructs source without invoking the compiler."""

    import importlib

    source_module = importlib.import_module("qamomile.source")

    def reject_compile(*_args: object, **_kwargs: object) -> None:
        pytest.fail("binding must not compile source")

    monkeypatch.setattr(source_module, "build_source", reject_compile)
    assert isinstance(deferred_block_sample.bind(block=x_block_encoding), qm.QKernel)


def test_block_encoding_binding_rejects_wrong_shape_or_parameter() -> None:
    """Host binding preserves signal/system shape before source compilation."""

    with pytest.raises(ValueError, match="BlockEncoding\\[1, 1\\]"):
        deferred_block_sample.bind(
            block=BlockEncoding(identity_three_encoding, 1.0, 2, 1)
        )
    with pytest.raises(TypeError, match="unknown block parameter"):
        deferred_block_sample.bind(unknown=x_block_encoding)
    with pytest.raises(TypeError, match="must be a BlockEncoding"):
        deferred_block_sample.bind(block=identity_encoding)


def test_block_binding_preserves_keyword_defaults() -> None:
    """A later-bound block retains other arguments and their Const defaults."""

    assert qm.evaluate(
        deferred_keyword_block_sample.specialize(block=x_block_encoding)
    ) == (0, 1)
    assert qm.evaluate(
        deferred_keyword_block_sample.specialize(
            block=x_block_encoding,
            flip=True,
        )
    ) == (0, 0)


def test_boolean_isometry_is_typed_closed_composition() -> None:
    """A clean Boolean graph embedding derives one ordinary isometry."""

    parity_entry = parity_isometry.module().function(parity_isometry.function_name)
    pair_entry = pair_isometry.module().function(pair_isometry.function_name)
    program = boolean_isometry_sample.specialize()
    restored = qm.Program.from_bytes(program.to_bytes())

    assert qm.evaluate(parity_bit.specialize(), bits=(1, 0)) is True
    assert qm.evaluate(ordered_pair.specialize(), bits=(1, 0)) == (True, True)
    assert parity_entry.kind is CallableKind.ISOMETRY
    assert pair_entry.kind is CallableKind.ISOMETRY
    assert parity_entry.result_type == qm.Q[qm.Bits[3]]
    assert pair_entry.result_type == qm.Q[qm.Bits[4]]
    assert qm.evaluate(restored) == (1, 0, 1, 1)
    assert "qcore.xor_by" in restored.inspect()
    assert "qcore.boolean_isometry" not in restored.inspect()

    study = qm.resources.logical(pair_isometry, strict=True)
    assert study.metric(
        qm.resources.logical_operation_count("qcore.xor_by")
    ).numeric_enclosure() == (1, 1)
    assert study.metric(qm.resources.LOGICAL_ALLOCATED_QUBITS).numeric_enclosure() == (
        2,
        2,
    )


def test_boolean_isometry_rejects_ambiguous_host_contracts() -> None:
    """The constructor accepts only fixed body-backed Boolean signatures."""

    @qm.qkernel
    def wrong_input(value: qm.Int) -> qm.Bool:
        return value == 0

    @qm.qkernel
    def wrong_output(bits: qm.Bits[1]) -> qm.Int:
        return len(bits)

    with pytest.raises(TypeError, match="Qamomile QKernel"):
        boolean_isometry(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"fixed-width Bits\[N\]"):
        boolean_isometry(wrong_input)
    with pytest.raises(TypeError, match="return Bool"):
        boolean_isometry(wrong_output)


def test_qsvt_polynomial_synthesis_is_checked_and_consumed_without_new_ir() -> None:
    """Rust certifies one typed polynomial that ordinary QSVT source consumes."""

    certificate = certified_x_polynomial

    assert isinstance(certificate, CertifiedQSVTPolynomial)
    assert certificate.chebyshev_coefficients == (0.0, 1.0)
    assert certificate.degree == certificate.queries == 1
    assert certificate.odd is True
    assert certificate.domain == (-1.0, 1.0)
    assert certificate.uniform_error_bound == 0.0
    assert certificate.target_magnitude_bound == 1.0
    assert certificate.realized_magnitude_bound == 1.0
    assert certificate.projector_rotations == 2
    assert certificate.success is QSVTSuccess.HERALDED
    assert certificate.phases.odd is True
    assert certificate.phases.domain == certificate.domain
    assert qm.evaluate(qsvt_sample.specialize()) == (0, 1)
    assert "quantum.qsvt" not in qsvt_sample.specialize().inspect()

    nontrivial = synthesize_qsvt_polynomial(
        (0.002, 0.0, 0.003),
        max_error=1e-7,
        max_iterations=512,
    )
    assert nontrivial.degree == 2
    assert nontrivial.odd is False
    assert nontrivial.uniform_error_bound <= 1e-7
    assert nontrivial.target_magnitude_bound == pytest.approx(0.005)


def test_reciprocal_svt_certifies_scale_error_cost_and_shared_source() -> None:
    """The high-level reciprocal producer closes into the ordinary QSVT path."""

    exact = reciprocal_x
    assert isinstance(exact, CertifiedReciprocalSVT)
    assert exact.polynomial == certified_x_polynomial
    assert exact.normalization == exact.minimum_singular_value == 1.0
    assert exact.normalized_gap == exact.reciprocal_scale == 1.0
    assert exact.operator_scale == 1.0
    assert exact.approximation_error_bound == exact.total_error_bound == 0.0
    assert exact.spectral_domain == ((-1.0, -1.0), (1.0, 1.0))
    assert exact.queries == 1
    assert exact.projector_rotations == 2
    assert exact.success is QSVTSuccess.HERALDED
    assert exact.block_encoding.signal_width == 2
    assert exact.block_encoding.system_width == 1
    assert exact.block_encoding.error_bound == 0.0
    assert exact.transform == exact.block_encoding.unitary
    assert qm.evaluate(reciprocal_x_sample.specialize()) == (0, 0, 1)
    assert "quantum.reciprocal" not in reciprocal_x_sample.specialize().inspect()

    nontrivial = reciprocal_svt(
        x_block_encoding,
        minimum_singular_value=0.8,
        max_error=0.01,
        max_degree=3,
        max_iterations=32,
    )
    assert nontrivial.polynomial.degree == nontrivial.queries == 3
    assert nontrivial.normalized_gap == 0.8
    assert nontrivial.reciprocal_scale == pytest.approx(0.143362831858407)
    assert nontrivial.approximation_error_bound < 0.005
    assert nontrivial.total_error_bound <= 0.01
    assert nontrivial.success is QSVTSuccess.HERALDED


def test_qsvt_polynomial_synthesis_fails_closed_at_native_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthesis and consumption reject missing or malformed Rust evidence."""

    import importlib

    qsvt_module = importlib.import_module("qamomile.stdlib.qsvt")

    with pytest.raises(TypeError, match="mint a certificate"):
        CertifiedQSVTPolynomial()
    with pytest.raises(TypeError, match="exactly one"):
        qsvt(x_block_encoding)
    with pytest.raises(TypeError, match="exactly one"):
        qsvt(
            x_block_encoding,
            phases=(0.0, 0.0),
            polynomial=certified_x_polynomial,
        )
    with pytest.raises(ValueError, match="Q7702"):
        synthesize_qsvt_polynomial((0.1, 0.1))
    with pytest.raises(ValueError, match="Q7703"):
        synthesize_qsvt_polynomial((0.2, 0.0, 0.2))
    with pytest.raises(ValueError, match="Q7706"):
        synthesize_qsvt_polynomial((0.0, 1.0), max_iterations=-1)

    monkeypatch.setattr(qsvt_module.native, "synthesize_qsvt_polynomial", None)
    with pytest.raises(TypeError, match="Q7700"):
        synthesize_qsvt_polynomial((0.0, 1.0))

    monkeypatch.undo()
    monkeypatch.setattr(
        qsvt_module.native,
        "synthesize_qsvt_polynomial",
        lambda *_arguments: SimpleNamespace(
            chebyshev_coefficients=[0.0, 1.0],
            qsp_phases=[math.pi, math.pi],
            projector_phases=[0.0, 0.0],
            degree=1,
            odd=True,
            uniform_error_bound=0.0,
            target_magnitude_bound=1.0,
            realized_magnitude_bound=1.0,
            queries=1,
            projector_rotations=2,
            success="heralded",
        ),
    )
    with pytest.raises(TypeError, match="Q7711"):
        synthesize_qsvt_polynomial((0.0, 1.0))

    monkeypatch.undo()
    monkeypatch.setattr(qsvt_module.native, "verify_qsvt_polynomial", None)
    with pytest.raises(TypeError, match="Q7700"):
        qsvt(x_block_encoding, polynomial=certified_x_polynomial)


def test_reciprocal_svt_fails_closed_at_native_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reciprocal synthesis rejects invalid requests and missing Rust evidence."""

    import importlib

    qsvt_module = importlib.import_module("qamomile.stdlib.qsvt")

    with pytest.raises(TypeError, match="mint a certificate"):
        CertifiedReciprocalSVT()
    with pytest.raises(ValueError, match="no larger than normalization"):
        reciprocal_svt(x_block_encoding, minimum_singular_value=2.0)
    with pytest.raises(ValueError, match="Q7723"):
        reciprocal_svt(
            x_block_encoding,
            minimum_singular_value=1.0,
            max_degree=0,
        )

    monkeypatch.setattr(qsvt_module.native, "synthesize_reciprocal_qsvt", None)
    with pytest.raises(TypeError, match="Q7720"):
        reciprocal_svt(x_block_encoding, minimum_singular_value=1.0)

    monkeypatch.undo()
    monkeypatch.setattr(qsvt_module.native, "verify_reciprocal_qsvt", None)
    with pytest.raises(TypeError, match="Q7720"):
        reciprocal_svt(x_block_encoding, minimum_singular_value=1.0)


def test_qsvt_projector_phase_matches_random_analytic_interference() -> None:
    """Random phases implement ``cos(2 phi)`` signal-Z interference."""

    for seed in (0, 42):
        phase = random.Random(seed).uniform(-math.pi, math.pi)
        transform = qsvt(identity_block_encoding, phases=(phase,))

        @qm.qkernel
        def interferometer() -> qm.Q[qm.Bits[2]]:
            state = qm.qubits(2)
            state = qm.h(state, target=0)
            state = transform(state)
            return qm.h(state, target=0)

        estimate = qm.expectation(
            interferometer,
            qm.Z(0),
            on=qm.Reference(),
            shots=128,
            seed=seed,
        )
        expected = math.cos(2.0 * phase)
        tolerance = 6.0 * math.sqrt((1.0 - expected * expected) / estimate.shots)
        assert abs(estimate.value - expected) <= tolerance + 1e-12


def test_qsvt_phase_coordinates_canonicalize_qsp_wx_without_an_enum() -> None:
    """One canonical value replaces convention enum plus conversion function."""

    source = (0.1, 0.2, 0.3, 0.4)
    phases = QSVTPhases.from_qsp_wx(
        source,
        odd=True,
        domain=(-0.8, 0.8),
        scale=0.5,
    )
    expected = (
        source[0] + 5.0 * math.pi / 4.0,
        source[1] - math.pi / 2.0,
        source[2] - math.pi / 2.0,
        source[3] - math.pi / 4.0,
    )
    expected = tuple(
        (value + math.pi) % (2.0 * math.pi) - math.pi for value in expected
    )

    assert phases.phases == pytest.approx(expected, abs=1e-12, rel=0.0)
    assert phases.odd is True
    assert phases.domain == (-0.8, 0.8)
    assert phases.scale == 0.5
    assert isinstance(qsvt(identity_block_encoding, phases=phases), qm.QKernel)


def test_qsvt_phase_conversion_requires_well_formed_native_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QSP conversion has no Python formula fallback or unchecked projection."""

    import importlib

    qsvt_module = importlib.import_module("qamomile.stdlib.qsvt")

    scalar = QSVTPhases.from_qsp_wx((0.25,))
    assert scalar.phases == pytest.approx((0.25 - math.pi / 2.0,), abs=1e-15)
    converted_transform = qsvt(
        identity_block_encoding,
        phases=QSVTPhases.from_qsp_wx((math.pi,)),
    )

    @qm.qkernel
    def converted_sample() -> qm.Bits[2]:
        """Interfere one native-converted degree-zero projector phase."""

        state = qm.h(qm.qubits(2), target=0)
        state = converted_transform(state)
        return qm.measure(qm.h(state, target=0))

    assert qm.evaluate(converted_sample.specialize()) == (1, 0)

    monkeypatch.setattr(qsvt_module.native, "convert_qsp_wx_phases", None)
    with pytest.raises(TypeError, match="Q7620"):
        QSVTPhases.from_qsp_wx((0.1, 0.2))

    monkeypatch.setattr(
        qsvt_module.native,
        "convert_qsp_wx_phases",
        lambda _phases: [math.pi, 0.0],
    )
    with pytest.raises(TypeError, match="Q7625"):
        QSVTPhases.from_qsp_wx((0.1, 0.2))


def test_amplitude_encoding_matches_random_single_qubit_states() -> None:
    """Random complex vectors preserve probabilities and relative phases."""

    for seed in (0, 42):
        generator = random.Random(seed)
        raw = (
            complex(generator.uniform(-1.0, 1.0), generator.uniform(-1.0, 1.0)),
            complex(generator.uniform(-1.0, 1.0), generator.uniform(-1.0, 1.0)),
        )
        norm = math.sqrt(sum(abs(value) ** 2 for value in raw))
        left, right = (value / norm for value in raw)
        transform = amplitude_encoding(raw)

        @qm.qkernel
        def state() -> qm.Q[qm.Bits[1]]:
            return transform(qm.qubits(1))

        @qm.qkernel
        def sample() -> qm.Shot[qm.Bits[1]]:
            return qm.measure(state())

        shots = 128
        result = qm.run(sample, on=qm.Reference(), shots=shots, seed=seed).result
        probability_one = result.counts.get((1,), 0) / shots
        expected_probability = abs(right) ** 2
        sampling_error = 6.0 * math.sqrt(
            expected_probability * (1.0 - expected_probability) / shots
        )
        assert abs(probability_one - expected_probability) <= sampling_error
        expected = (
            2.0 * (left.conjugate() * right).real,
            2.0 * (left.conjugate() * right).imag,
            abs(left) ** 2 - abs(right) ** 2,
        )
        for observable, value in zip(
            (qm.X(0), qm.Y(0), qm.Z(0)),
            expected,
            strict=True,
        ):
            estimate = qm.expectation(
                state,
                observable,
                on=qm.Reference(),
                shots=shots,
                seed=seed,
            )
            tolerance = 6.0 * math.sqrt((1.0 - value * value) / estimate.shots)
            assert abs(estimate.value - value) <= tolerance + 1e-12


def test_amplitude_angles_cross_required_native_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coordinate synthesis has no shadow Python numerical implementation."""

    from qamomile.stdlib import state_preparation

    native_derive = state_preparation.native.mottonen_amplitude_coordinates
    calls = 0

    def record_native_derivation(
        amplitudes: list[tuple[float, float]],
    ) -> object:
        """Record and delegate one native coordinate request."""

        nonlocal calls
        calls += 1
        return native_derive(amplitudes)

    monkeypatch.setattr(
        state_preparation.native,
        "mottonen_amplitude_coordinates",
        record_native_derivation,
    )
    coordinates = amplitude_angles((1.0, 1.0j))

    assert calls == 1
    assert coordinates.ry == pytest.approx((math.pi / 2.0,), abs=1e-15)
    assert coordinates.rz == pytest.approx((math.pi / 2.0,), abs=1e-15)
    assert coordinates.phase == pytest.approx(math.pi / 4.0, abs=1e-15)

    monkeypatch.setattr(
        state_preparation.native, "mottonen_amplitude_coordinates", None
    )
    with pytest.raises(TypeError, match="Q7500"):
        amplitude_angles((1.0, 0.0))

    monkeypatch.setattr(
        state_preparation.native,
        "mottonen_amplitude_coordinates",
        lambda _amplitudes: (1.5, [0.0], [0.0], 0.0),
    )
    with pytest.raises(TypeError, match="Q7506"):
        amplitude_angles((1.0, 0.0))


def test_amplitude_angles_scale_before_normalizing() -> None:
    """Large finite vectors do not overflow during native normalization."""

    import sys

    coordinates = amplitude_angles((sys.float_info.max, sys.float_info.max))

    assert coordinates.ry == pytest.approx((math.pi / 2.0,), abs=1e-15)
    assert coordinates.rz == (0.0,)
    assert coordinates.phase == 0.0


def test_amplitude_encoding_preserves_multiqubit_relative_phase() -> None:
    """The complex Gray walk prepares a product of Y-plus and X-minus states."""

    transform = amplitude_encoding((1.0, 1.0j, -1.0, -1.0j))

    @qm.qkernel
    def state() -> qm.Q[qm.Bits[2]]:
        return transform(qm.qubits(2))

    endpoint = qm.Reference()
    for observable, expected in (
        (qm.Y(0), 1.0),
        (qm.X(1), -1.0),
        (qm.Y(0) * qm.X(1), -1.0),
    ):
        value = qm.expectation(
            state,
            observable,
            on=endpoint,
            shots=16,
            seed=2,
        ).value
        assert value == pytest.approx(expected, abs=1e-12, rel=0.0)


def test_runtime_amplitude_encoding_reuses_one_structural_artifact() -> None:
    """Aggregate Param values change prepared data without recompilation."""

    from qamomile.circuit import lower_to_circuit, scalar_parameters

    program = runtime_amplitude_two_sample.specialize()
    artifact = qm.compile(program, to=qm.Reference())
    endpoint = qm.Reference()
    zero = runtime_amplitude_parameters((1.0, 0.0, 0.0, 0.0))
    eleven = runtime_amplitude_parameters((0.0, 0.0, 0.0, 1.0))

    assert qm.execute(
        artifact,
        on=endpoint,
        parameters=zero,
        shots=8,
        seed=2,
    ).counts == {(0, 0): 8}
    assert qm.execute(
        artifact,
        on=endpoint,
        parameters=eleven,
        shots=8,
        seed=2,
    ).counts == {(1, 1): 8}

    circuit = lower_to_circuit(program)
    assert tuple(slot.name for slot in circuit.parameter_abi.slots) == (
        "ry_angles",
        "rz_angles",
        "phase",
    )
    assert len(scalar_parameters(circuit.parameter_abi)) == 7
    assert b'"parameter":"ry_angles","path":[2]' in circuit.to_bytes()
    assert "quantum.amplitude" not in program.inspect()


def test_runtime_amplitude_encoding_preserves_complex_phase() -> None:
    """Runtime coordinates retain the same exact phase convention as static data."""

    parameters = runtime_amplitude_parameters((1.0, 1.0j, -1.0, -1.0j))
    endpoint = qm.Reference()
    for observable, expected in (
        (qm.Y(0), 1.0),
        (qm.X(1), -1.0),
        (qm.Y(0) * qm.X(1), -1.0),
    ):
        value = qm.expectation(
            runtime_amplitude_two_state,
            observable,
            on=endpoint,
            parameters=parameters,
            shots=16,
            seed=2,
        ).value
        assert value == pytest.approx(expected, abs=1e-12, rel=0.0)


def test_uniformly_controlled_rotation_is_one_shared_gray_walk() -> None:
    """Static and runtime coordinates use the same ordinary RY/CX body."""

    angles = uniformly_controlled_angles((math.pi,) * 4)
    static_program = uniformly_controlled_sample.specialize()
    runtime_program = runtime_uniformly_controlled_sample.specialize()

    assert angles == pytest.approx((math.pi, 0.0, 0.0, 0.0), abs=1e-12)
    assert qm.evaluate(static_program) == (1, 0, 1)
    assert qm.run(
        runtime_program,
        on=qm.Reference(),
        parameters={"angles": angles},
        shots=4,
        seed=2,
    ).result.counts == {(0, 1, 1): 4}
    assert "quantum.uniformly_controlled" not in static_program.inspect()
    assert qm.resources.logical(static_program).metric(
        qm.resources.LOGICAL_ROTATION_GATES
    ).numeric_enclosure() == (4, 4)


def test_uniformly_controlled_rotation_coordinates_are_rust_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public transform delegates once and rejects a missing native path."""

    from qamomile.stdlib import uniformly_controlled

    native_convert = (
        uniformly_controlled.native.uniformly_controlled_rotation_coordinates
    )
    calls = 0

    def record_native_conversion(branch_angles: list[float]) -> object:
        """Record and delegate one native Walsh/Gray transform."""

        nonlocal calls
        calls += 1
        return native_convert(branch_angles)

    monkeypatch.setattr(
        uniformly_controlled.native,
        "uniformly_controlled_rotation_coordinates",
        record_native_conversion,
    )

    assert uniformly_controlled_angles((1.0, 2.0, 4.0, 8.0)) == pytest.approx(
        (3.75, -1.25, 0.75, -2.25),
        abs=1e-15,
    )
    assert calls == 1

    monkeypatch.setattr(
        uniformly_controlled.native,
        "uniformly_controlled_rotation_coordinates",
        None,
    )
    with pytest.raises(TypeError, match="Q7510"):
        uniformly_controlled_angles((0.0,))

    monkeypatch.setattr(
        uniformly_controlled.native,
        "uniformly_controlled_rotation_coordinates",
        lambda _angles: (0.0,),
    )
    with pytest.raises(TypeError, match="Q7515"):
        uniformly_controlled_angles((0.0,))


def test_gray_walk_topology_crosses_required_native_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both public factories consume their Rust-owned topology exactly once."""

    from qamomile.stdlib import state_preparation, uniformly_controlled

    native_mottonen = state_preparation.native.mottonen_rotation_schedule
    native_gray = uniformly_controlled.native.cyclic_gray_control_schedule
    calls = {"mottonen": 0, "gray": 0}

    def record_mottonen(width: int) -> object:
        """Record and delegate one native Möttönen topology request."""

        calls["mottonen"] += 1
        return native_mottonen(width)

    def record_gray(width: int) -> object:
        """Record and delegate one native cyclic Gray topology request."""

        calls["gray"] += 1
        return native_gray(width)

    monkeypatch.setattr(
        state_preparation.native,
        "mottonen_rotation_schedule",
        record_mottonen,
    )
    monkeypatch.setattr(
        uniformly_controlled.native,
        "cyclic_gray_control_schedule",
        record_gray,
    )

    amplitude_encoding_from_angles(2)
    uniformly_controlled_ry_from_angles(2)

    assert calls == {"mottonen": 1, "gray": 1}


def test_gray_walk_topology_fails_closed_at_native_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing or malformed native schedules never fall back to Python."""

    from qamomile.stdlib import state_preparation, uniformly_controlled

    monkeypatch.setattr(
        state_preparation.native,
        "mottonen_rotation_schedule",
        None,
    )
    with pytest.raises(TypeError, match="Q7500"):
        amplitude_encoding_from_angles(2)

    monkeypatch.setattr(
        uniformly_controlled.native,
        "cyclic_gray_control_schedule",
        None,
    )
    with pytest.raises(TypeError, match="Q7510"):
        uniformly_controlled_ry_from_angles(2)

    monkeypatch.setattr(
        state_preparation.native,
        "mottonen_rotation_schedule",
        lambda _width: [(1, 0, 0), (2, 0, 1)],
    )
    with pytest.raises(TypeError, match="Q7506"):
        amplitude_encoding_from_angles(2)

    monkeypatch.setattr(
        uniformly_controlled.native,
        "cyclic_gray_control_schedule",
        lambda _width: (0, 1, 0, 1),
    )
    with pytest.raises(TypeError, match="Q7515"):
        uniformly_controlled_ry_from_angles(2)


def test_gray_walk_topology_budgets_fail_at_public_factories() -> None:
    """Oversized topology fails before dependent source construction."""

    with pytest.raises(ValueError, match="Q7502"):
        amplitude_encoding_from_angles(21)
    with pytest.raises(ValueError, match="Q7512"):
        uniformly_controlled_ry_from_angles(21)


def test_uniformly_controlled_rotation_validates_host_coordinates() -> None:
    """Invalid host tables fail before source or QCore construction."""

    with pytest.raises(ValueError, match="power of two"):
        uniformly_controlled_angles((0.0, 1.0, 2.0))
    with pytest.raises(ValueError, match="finite"):
        uniformly_controlled_angles((0.0, math.inf))
    with pytest.raises(ValueError, match="nonnegative"):
        uniformly_controlled_ry_from_angles(-1)


def test_xor_lookup_is_total_select_not_a_qrom_operation() -> None:
    """An immutable lookup table reuses the callable selection algebra."""

    program = xor_lookup_sample.specialize()

    assert qm.evaluate(program) == (0, 1, 1, 0)
    assert "qcore.select_call" in program.inspect()
    assert "quantum.lookup" not in program.inspect()
    assert (
        qm.resources.logical(program)
        .metric(qm.resources.LOGICAL_GATES)
        .numeric_enclosure()[0]
        > 0
    )
    with pytest.raises(ValueError, match="power of two"):
        xor_lookup((0, 1, 2))
    with pytest.raises(ValueError, match="fit"):
        xor_lookup((0, 4), value_width=2)


def test_sparse_matrix_lookup_is_typed_total_function_composition() -> None:
    """Sparse access preserves rectangular host syntax and ordinary QCore."""

    program = sparse_lookup_sample.specialize()

    assert qm.evaluate(program) == (1, 0, 0, 1, 1, 1)
    assert "qcore.select_call" in program.inspect()
    assert "quantum.sparse" not in program.inspect()
    with pytest.raises(ValueError, match="same shape"):
        sparse_matrix_lookup(
            ((0, 1), (1, 0)),
            ((1, 2),),
            column_width=1,
        )
    with pytest.raises(ValueError, match="powers of two"):
        sparse_matrix_lookup(
            ((0, 1, 2), (1, 2, 0)),
            ((1, 2, 3), (3, 2, 1)),
            column_width=2,
        )
    with pytest.raises(ValueError, match="column index"):
        sparse_matrix_lookup(
            ((0,), (2,)),
            ((0,), (0,)),
            column_width=1,
        )


def test_lookup_factories_require_native_plans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lookup factories delegate schedules and fail closed on stale projections."""

    import importlib

    lookup = importlib.import_module("qamomile.stdlib.lookup")
    xor_calls: list[tuple[list[str], str | None]] = []
    sparse_calls: list[tuple[object, ...]] = []
    original_xor = lookup.native.plan_xor_lookup
    original_sparse = lookup.native.plan_sparse_lookup

    def tracked_xor(values: list[str], width: str | None) -> object:
        """Record and forward one XOR planning request."""

        xor_calls.append((values, width))
        return original_xor(values, width)

    def tracked_sparse(*args: object) -> object:
        """Record and forward one sparse planning request."""

        sparse_calls.append(args)
        return original_sparse(*args)

    monkeypatch.setattr(lookup.native, "plan_xor_lookup", tracked_xor)
    monkeypatch.setattr(lookup.native, "plan_sparse_lookup", tracked_sparse)
    xor_lookup((0, 3, 1, 2), value_width=2)
    sparse_matrix_lookup(((0,), (1,)), ((1,), (0,)), column_width=1)
    assert xor_calls == [(["0", "3", "1", "2"], "2")]
    assert len(sparse_calls) == 1

    monkeypatch.setattr(lookup.native, "plan_xor_lookup", None)
    with pytest.raises(TypeError, match="Q7661"):
        xor_lookup((0, 1))

    monkeypatch.setattr(
        lookup.native,
        "plan_sparse_lookup",
        lambda *_args: SimpleNamespace(
            address_width=1,
            column_width=1,
            value_width=1,
            flip_targets=[[0], [2]],
        ),
    )
    with pytest.raises(TypeError, match="Q7660"):
        sparse_matrix_lookup(((0,), (1,)), ((1,), (0,)), column_width=1)


def test_runtime_matrix_coordinates_lower_to_scalar_parameter_leaves() -> None:
    """Nested Matrix loops compile once and retain one shaped launch argument."""

    program = runtime_matrix_rotation_sample.specialize()
    artifact = qm.compile(program, to=qm.Reference())

    assert program.entry.parameters[0].value.type == qm.Param[qm.Matrix[qm.Real, 2, 2]]
    assert tuple(
        leaf.value.path for leaf in scalar_parameters(artifact.parameter_abi)
    ) == (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    )
    exercise_runtime_matrix_backend(endpoint=qm.Reference(), seed=2)


def test_scalar_parameter_projection_is_a_required_native_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SDK scalar paths come only from the checked Rust parameter ABI."""

    import importlib

    from qamomile.compilation import ParameterABI, ParameterSlot

    circuit = importlib.import_module("qamomile.circuit")
    abi = ParameterABI((ParameterSlot("matrix", qm.Param[qm.Matrix[qm.Real, 2, 2]]),))
    original = circuit.native.project_scalar_parameters
    calls = 0

    def track_projection(payload: bytes) -> object:
        """Record and forward one native scalar parameter projection."""

        nonlocal calls
        calls += 1
        return original(payload)

    monkeypatch.setattr(
        circuit.native,
        "project_scalar_parameters",
        track_projection,
    )
    assert tuple(leaf.value.path for leaf in circuit.scalar_parameters(abi)) == (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    )
    assert calls == 1

    monkeypatch.setattr(circuit.native, "project_scalar_parameters", None)
    with pytest.raises(qm.CapabilityError, match="Q7245"):
        circuit.scalar_parameters(abi)

    monkeypatch.setattr(
        circuit.native,
        "project_scalar_parameters",
        lambda _payload: [("matrix", [(0, 0)], "Real")],
    )
    with pytest.raises(TypeError, match="violates the Python ABI"):
        circuit.scalar_parameters(abi)


def test_scalar_parameter_values_are_bound_and_flattened_only_in_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SDK values retain native row-major order and default materialization."""

    import importlib

    from qamomile.compilation import ParameterABI, ParameterSlot

    circuit = importlib.import_module("qamomile.circuit")
    abi = ParameterABI(
        (
            ParameterSlot("matrix", qm.Param[qm.Matrix[qm.Real, 2, 2]]),
            ParameterSlot(
                "offset",
                qm.Param[qm.Vector[qm.Int, 2]],
                default=(7, 8),
                has_default=True,
            ),
        )
    )
    original = circuit.native.bind_scalar_parameters
    calls = 0

    def track_binding(parameter_abi: bytes, supplied_values: bytes) -> object:
        """Record and forward one native scalar parameter binding."""

        nonlocal calls
        calls += 1
        return original(parameter_abi, supplied_values)

    monkeypatch.setattr(circuit.native, "bind_scalar_parameters", track_binding)
    monkeypatch.setattr(circuit.native, "project_scalar_parameters", None)
    bindings = scalar_parameter_bindings(
        abi,
        {"matrix": ((1.0, 2.0), (3.0, 4.0))},
    )

    assert tuple((parameter.path, value) for parameter, value in bindings) == (
        ((0, 0), 1.0),
        ((0, 1), 2.0),
        ((1, 0), 3.0),
        ((1, 1), 4.0),
        ((0,), 7),
        ((1,), 8),
    )
    assert calls == 1

    monkeypatch.setattr(circuit.native, "bind_scalar_parameters", None)
    with pytest.raises(qm.CapabilityError, match="Q7245"):
        scalar_parameter_bindings(abi, {"matrix": ((1.0, 2.0), (3.0, 4.0))})

    monkeypatch.setattr(
        circuit.native,
        "bind_scalar_parameters",
        lambda _abi, _values: [("matrix", [(0, None)], "Real", b"invalid")],
    )
    with pytest.raises(TypeError, match="violates the Python ABI"):
        scalar_parameter_bindings(abi, {"matrix": ((1.0, 2.0), (3.0, 4.0))})


@pytest.mark.parametrize("width", (2, 3))
@pytest.mark.parametrize("seed", (0, 42))
def test_amplitude_encoding_matches_random_basis_probabilities(
    width: int,
    seed: int,
) -> None:
    """Random sizes exercise the complete Gray walk against analytic weights."""

    generator = random.Random(seed)
    raw = tuple(
        complex(generator.uniform(-1.0, 1.0), generator.uniform(-1.0, 1.0))
        for _ in range(1 << width)
    )
    norm = math.sqrt(sum(abs(value) ** 2 for value in raw))
    expected = tuple(abs(value / norm) ** 2 for value in raw)
    transform = amplitude_encoding(raw)

    @qm.qkernel
    def sample() -> qm.Shot[qm.Bits[width]]:  # type: ignore[valid-type]
        return qm.measure(transform(qm.qubits(width)))

    shots = 128
    result = qm.run(sample, on=qm.Reference(), shots=shots, seed=seed).result
    for basis_state in range(1 << width):
        bits = tuple((basis_state >> index) & 1 for index in range(width))
        observed = result.counts.get(bits, 0) / shots
        probability = expected[basis_state]
        sampling_error = 6.0 * math.sqrt(probability * (1.0 - probability) / shots)
        assert abs(observed - probability) <= sampling_error


def test_lcu_block_encoding_and_qsvt_are_ordinary_closed_composition() -> None:
    """PREPARE/SELECT/adjoint close without an LCU or QSVT semantic opcode."""

    program = lcu_qsvt_sample.specialize()
    restored = qm.Program.from_bytes(program.to_bytes())

    assert qm.evaluate(restored) == (1, 0)
    assert projector_lcu.normalization == pytest.approx(
        2.0,
        abs=1e-12,
        rel=0.0,
    )
    assert projector_lcu.signal_width == 1
    assert "qcore.select_call" in restored.inspect()
    assert "qcore.adjoint_call" in restored.inspect()
    assert "quantum.lcu" not in restored.inspect()
    assert "quantum.qsvt" not in restored.inspect()
    assert (
        qm.resources.logical(lcu_qsvt_state)
        .metric(qm.resources.LOGICAL_GATES)
        .numeric_enclosure()[0]
        > 0
    )


def test_pauli_block_encoding_is_a_thin_exact_lcu_adapter() -> None:
    """Canonical Pauli sums reuse LCU, adjoint, and body-derived resources."""

    operator = 0.5 + qm.X(0) + qm.Y(1)
    encoding = pauli_block_encoding(operator)
    transformed = qsvt(encoding, phases=(0.1, 0.2, 0.3))
    program = transformed.specialize()

    assert encoding.normalization == pytest.approx(2.5, abs=1e-12, rel=0.0)
    assert encoding.signal_width == 2
    assert encoding.system_width == 2
    assert "quantum.pauli_block_encoding" not in program.inspect()
    assert (
        qm.resources.logical(transformed)
        .metric(qm.resources.LOGICAL_GATES)
        .numeric_enclosure()[0]
        > 0
    )


def test_projected_block_encoding_replaces_sparse_producer_types() -> None:
    """A prepared-oracle sandwich is ordinary functional composition."""

    program = projected_block_sample.specialize()

    assert qm.evaluate(program) == (0, 1)
    assert projected_x_encoding.normalization == pytest.approx(
        1.0,
        abs=1e-12,
        rel=0.0,
    )
    assert projected_x_encoding.signal_width == 1
    assert projected_x_encoding.system_width == 1
    assert "qcore.adjoint_call" in program.inspect()
    assert "quantum.projected_block_encoding" not in program.inspect()
    assert "sparse" not in program.inspect()


def test_periodic_shift_block_encoding_is_arithmetic_plus_lcu() -> None:
    """Signed modular aliases close to constant additions and existing LCU."""

    encoding = periodic_shift_block_encoding(
        {(1, -1): 0.25j, (3, 3): 0.75j},
        register_sizes=(1, 2),
    )
    transform = encoding.unitary

    @qm.qkernel
    def sample() -> qm.Shot[qm.Bits[4]]:
        return qm.measure(transform(qm.qubits(4)))

    program = sample.specialize()
    assert qm.evaluate(program) == (0, 1, 1, 1)
    assert encoding.normalization == pytest.approx(1.0, abs=1e-12, rel=0.0)
    assert encoding.signal_width == 1
    assert encoding.system_width == 3
    assert "qcore.select_call" in program.inspect()
    assert "quantum.periodic_shift" not in program.inspect()


def test_dense_periodic_matrix_recognition_returns_the_same_functional_body() -> None:
    """Exact dense recognition ends at the coefficient-map constructor."""

    matrix = tuple(
        tuple(1.0 if row == (column + 1) % 4 else 0.0 for column in range(4))
        for row in range(4)
    )
    encoding = periodic_shift_block_encoding_from_matrix(
        matrix,
        register_sizes=(2,),
    )
    transform = encoding.unitary

    @qm.qkernel
    def sample() -> qm.Shot[qm.Bits[3]]:
        return qm.measure(transform(qm.qubits(3)))

    assert qm.evaluate(sample.specialize()) == (0, 1, 0)
    assert "qcore.select_call" in sample.specialize().inspect()
    with pytest.raises(ValueError, match="constant-coefficient"):
        periodic_shift_block_encoding_from_matrix(
            ((1.0, 0.0), (0.0, 2.0)),
            register_sizes=(1,),
        )


def test_periodic_shift_coefficients_use_native_exact_canonicalization() -> None:
    """Signed aliases and cancellation retain one subnormal native residue."""

    import sys

    from qamomile.stdlib import periodic_shift

    residue = float.fromhex("0x0.0000000000001p-1022")
    terms = periodic_shift._native_periodic_terms(
        {
            -1: sys.float_info.max,
            3: -sys.float_info.max,
            (1 << 20_000) + 3: residue,
        },
        (2,),
    )

    assert terms == (((3,), complex(residue, 0.0)),)


def test_periodic_shift_requires_well_formed_native_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Periodic semantics cannot fall back to a Python normalizer."""

    from qamomile.stdlib import periodic_shift

    monkeypatch.setattr(periodic_shift.native, "normalize_periodic_shift_terms", None)
    with pytest.raises(TypeError, match="Q7530"):
        periodic_shift_block_encoding({0: 1.0}, register_sizes=(1,))

    monkeypatch.setattr(
        periodic_shift.native,
        "normalize_periodic_shift_terms",
        lambda _sizes, _offsets, _coefficients: [(["2"], 1.0, 0.0)],
    )
    with pytest.raises(TypeError, match="Q7541"):
        periodic_shift_block_encoding({0: 1.0}, register_sizes=(1,))


def test_functional_block_producers_validate_host_contracts() -> None:
    """Malformed producer inputs fail before any compiler specialization."""

    with pytest.raises(TypeError, match="signal preparation"):
        projected_block_encoding(
            object(),  # type: ignore[arg-type]
            projected_x_oracle,
            signal_width=1,
            system_width=1,
            normalization=1.0,
        )
    with pytest.raises(ValueError, match="at least one axis"):
        periodic_shift_block_encoding({}, register_sizes=())
    with pytest.raises(ValueError, match="one component per register axis"):
        periodic_shift_block_encoding({(1,): 1.0}, register_sizes=(1, 1))
    with pytest.raises(ValueError, match="finite"):
        periodic_shift_block_encoding({0: complex("nan")}, register_sizes=(1,))


def test_pauli_block_encoding_preserves_zero_and_complex_terms() -> None:
    """Zero and complex coefficient paths retain exact LCU conventions."""

    zero = pauli_block_encoding(qm.Hamiltonian.zero(num_qubits=2))
    zero_unitary = zero.unitary

    @qm.qkernel
    def zero_sample() -> qm.Shot[qm.Bits[3]]:
        return qm.measure(zero_unitary(qm.qubits(3)))

    assert qm.evaluate(pauli_block_sample.specialize()) == (0, 1)
    assert qm.evaluate(zero_sample.specialize()) == (1, 0, 0)
    assert zero.normalization == pytest.approx(1.0, abs=1e-12, rel=0.0)
    assert zero.signal_width == 1


def test_zero_lcu_has_an_exact_zero_projected_block() -> None:
    """Cancellation uses one signal flip rather than a fake zero resource rule."""

    zero = lcu_block_encoding(
        ((1.0, system_z), (-1.0, system_z)),
        system_width=1,
    )
    zero_unitary = zero.unitary

    @qm.qkernel
    def sample() -> qm.Shot[qm.Bits[2]]:
        return qm.measure(zero_unitary(qm.qubits(2)))

    assert zero.normalization == pytest.approx(1.0, abs=1e-12, rel=0.0)
    assert qm.evaluate(sample.specialize()) == (1, 0)


def test_lcu_coefficients_use_one_exact_bounded_native_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LCU cancellation, normalization, and projection cannot diverge in Python."""

    from qamomile.stdlib import lcu

    residue = float.fromhex("0x0.0000000000001p-1022")
    encoding = lcu_block_encoding(
        (
            (float.fromhex("0x1.fffffffffffffp+1023"), system_z),
            (-float.fromhex("0x1.fffffffffffffp+1023"), system_z),
            (residue, system_z),
        ),
        system_width=1,
    )

    assert encoding.normalization == residue
    assert encoding.signal_width == 1
    with pytest.raises(ValueError, match="Q7614"):
        lcu_block_encoding(
            (
                (float.fromhex("0x1.fffffffffffffp+1023"), system_z),
                (float.fromhex("0x1.fffffffffffffp+1023"), system_z),
            ),
            system_width=1,
        )

    monkeypatch.setattr(lcu.native, "plan_lcu_coefficients", None)
    with pytest.raises(TypeError, match="Q7610"):
        lcu_block_encoding(((1.0, system_z),), system_width=1)

    monkeypatch.setattr(
        lcu.native,
        "plan_lcu_coefficients",
        lambda _groups, _rows: SimpleNamespace(
            terms=[(0, 1.0, 0.0, 0.0, 0.0)],
            normalization=1.0,
            signal_width=1,
            case_count=2,
        ),
    )
    with pytest.raises(TypeError, match="Q7618"):
        lcu_block_encoding(((1.0, system_z),), system_width=1)


def test_block_encoding_and_qsvt_reject_ambiguous_host_contracts() -> None:
    """Descriptor and phase factories reject invalid semantics before QCore."""

    import pytest

    with pytest.raises(ValueError, match="width"):
        BlockEncoding(identity_encoding, 1.0, 1, 2)
    with pytest.raises(ValueError, match="normalization"):
        BlockEncoding(identity_encoding, 0.0, 1, 1)
    with pytest.raises(ValueError, match="at least one phase"):
        qsvt(identity_block_encoding, phases=())
    with pytest.raises(ValueError, match="finite"):
        qsvt(identity_block_encoding, phases=(math.inf,))
    with pytest.raises(TypeError, match="degree"):
        qsvt_from_phases(identity_block_encoding, degree=True)
    with pytest.raises(ValueError, match="nonnegative"):
        qsvt_from_phases(identity_block_encoding, degree=-1)
    with pytest.raises(ValueError, match="power of two"):
        amplitude_encoding((1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="all be zero"):
        amplitude_encoding((0.0, 0.0))
    with pytest.raises(TypeError, match="integer"):
        amplitude_encoding_from_angles(True)
    with pytest.raises(ValueError, match="positive"):
        amplitude_encoding_from_angles(0)
    with pytest.raises(ValueError, match="width"):
        lcu_block_encoding(((1.0, system_z),), system_width=2)
    with pytest.raises(TypeError, match="Hamiltonian"):
        pauli_block_encoding(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        pauli_block_encoding(qm.Hamiltonian.zero())


def test_variational_algorithms_are_plain_function_composition() -> None:
    """QAOA and FQAOA need no algorithm opcode or resource dispatch table."""

    exercise_variational_backend(endpoint=qm.Reference(), seed=2)
    program = qaoa_sample.specialize(hamiltonian=qm.Z(0))

    assert "quantum.qaoa" not in program.inspect()
    assert "quantum.fqaoa" not in fqaoa_sample.inspect()
    assert qm.resources.logical(program).metric(
        qm.resources.logical_operation_count("quantum.pauli_evolution")
    ).numeric_enclosure() == (1, 1)


def test_product_formula_has_one_compact_body_derived_resource_path() -> None:
    """A large Suzuki schedule stays data, while one loop body defines meaning."""

    program = product_formula_sample.specialize(
        **{
            "hamiltonians": (
                qm.X(0),
                qm.Hamiltonian.zero(num_qubits=1),
            )
        }
    )
    restored = qm.Program.from_bytes(program.to_bytes())
    study = qm.resources.logical(restored)

    exercise_product_formula_backend(endpoint=qm.Reference(), seed=2)
    assert restored.inspect().count("quantum.pauli_evolution") == 1
    assert "quantum.trotter" not in restored.inspect()
    assert study.metric(
        qm.resources.logical_operation_count("quantum.pauli_evolution")
    ).numeric_enclosure() == (22, 22)


@pytest.mark.parametrize(
    ("options", "error", "message"),
    (
        ({"term_count": 1}, ValueError, "at least two"),
        ({"term_count": 2, "order": 3}, ValueError, "positive even"),
        ({"term_count": 2, "steps": 0}, ValueError, "positive"),
        ({"term_count": True}, TypeError, "integer"),
    ),
)
def test_product_formula_rejects_invalid_static_topology(
    options: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    """Malformed formula topology fails once at the host composition boundary."""

    with pytest.raises(error, match=message):
        suzuki_trotter(**options)  # type: ignore[arg-type]


def test_product_formula_schedule_is_rust_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The factory delegates schedule semantics and fails closed without Rust."""

    from qamomile.stdlib import product_formula

    native_derive = product_formula.native.suzuki_product_formula_schedule
    calls = 0

    def record_native_schedule(term_count: int, order: int) -> object:
        """Record and delegate one native Suzuki request."""

        nonlocal calls
        calls += 1
        return native_derive(term_count, order)

    monkeypatch.setattr(
        product_formula.native,
        "suzuki_product_formula_schedule",
        record_native_schedule,
    )
    schedule = product_formula._native_suzuki_schedule(2, 4)

    assert calls == 1
    assert len(schedule) == 11
    assert schedule == tuple(reversed(schedule))

    monkeypatch.setattr(
        product_formula.native,
        "suzuki_product_formula_schedule",
        None,
    )
    with pytest.raises(TypeError, match="Q7520"):
        suzuki_trotter(term_count=2, order=2)

    monkeypatch.setattr(
        product_formula.native,
        "suzuki_product_formula_schedule",
        lambda _term_count, _order: [(2, 1.0)],
    )
    with pytest.raises(TypeError, match="Q7525"):
        suzuki_trotter(term_count=2, order=2)


def test_product_formula_rejects_unbounded_native_schedule() -> None:
    """An excessive base formula fails before allocating its output row."""

    with pytest.raises(ValueError, match="Q7523"):
        suzuki_trotter(term_count=500_001, order=2)
