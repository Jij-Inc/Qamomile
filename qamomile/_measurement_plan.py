"""Checked projections of Rust-owned Pauli measurement plans and transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._native import compiler as native
from .errors import CapabilityError, ExecutionError, TypeCheckError
from .expectation import PauliReadout
from .ir import Program
from .observable import Hamiltonian, Pauli, PauliFactor, PauliWord


@dataclass(frozen=True, slots=True)
class _MeasurementGroupPlan:
    """One defensively decoded transient native measurement group."""

    basis: PauliWord | None
    term_indices: tuple[int, ...]
    shots: int
    seed: int
    readouts: tuple[PauliReadout, ...] = ()
    diagonalizing_gates: int = 0
    strategy: Literal["independent", "qwc", "commuting"] = "qwc"


def _native_qwc_expectation_plan(
    observable: Hamiltonian,
    *,
    shots: int,
    seed: int,
    independent: bool = False,
) -> tuple[_MeasurementGroupPlan, ...]:
    """Project one canonical Rust local-basis grouping and total-shot allocation.

    Args:
        observable (Hamiltonian): Canonical Hermitian Pauli sum.
        shots (int): Total device-shot budget.
        seed (int): Unsigned base seed, with zero preserving endpoint randomness.
        independent (bool): Use one group per term instead of shared local bases.

    Returns:
        tuple[_MeasurementGroupPlan, ...]: Checked groups in canonical order.

    Raises:
        CapabilityError: If the selected native planner is unavailable.
        ExecutionError: If the budget or seed is invalid, or its output is malformed.
    """

    planner = getattr(
        native,
        "plan_independent_pauli_expectation"
        if independent
        else "plan_qwc_pauli_expectation",
        None,
    )
    if not callable(planner):
        raise CapabilityError("Q7300: native QWC expectation planning is unavailable")
    try:
        projection = planner(observable._native_literal, str(shots), seed)
    except (OverflowError, TypeError) as error:
        raise ExecutionError(
            "Q8138: seed must be an unsigned 64-bit integer"
        ) from error
    except ValueError as error:
        raise ExecutionError(str(error)) from error
    try:
        if not isinstance(projection, list):
            raise TypeError
        groups: list[_MeasurementGroupPlan] = []
        observed_terms: list[int] = []
        observed_shots = 0
        observed_seeds: list[int] = []
        for row in projection:
            if not isinstance(row, tuple) or len(row) != 4:
                raise TypeError
            raw_basis, raw_terms, raw_shots, raw_seed = row
            if (
                not isinstance(raw_basis, list)
                or not raw_basis
                or not isinstance(raw_terms, list)
                or not raw_terms
                or (independent and len(raw_terms) != 1)
                or not isinstance(raw_shots, str)
                or not raw_shots.isdecimal()
                or type(raw_seed) is not int
                or not 0 <= raw_seed < 1 << 32
            ):
                raise TypeError
            factors = tuple(
                PauliFactor(index, Pauli(pauli))
                for index, pauli in raw_basis
                if type(index) is int
            )
            if len(factors) != len(raw_basis):
                raise TypeError
            term_indices = tuple(raw_terms)
            if any(type(index) is not int or index < 0 for index in term_indices):
                raise TypeError
            if any(
                left >= right for left, right in zip(term_indices, term_indices[1:])
            ):
                raise TypeError
            group_shots = int(raw_shots)
            if group_shots <= 0:
                raise TypeError
            basis = PauliWord(factors)
            if tuple(
                (factor.index, factor.pauli.value) for factor in basis.factors
            ) != tuple(raw_basis):
                raise TypeError
            groups.append(
                _MeasurementGroupPlan(
                    basis,
                    term_indices,
                    group_shots,
                    raw_seed,
                    strategy="independent" if independent else "qwc",
                )
            )
            observed_terms.extend(term_indices)
            observed_shots += group_shots
            observed_seeds.append(raw_seed)
        if sorted(observed_terms) != list(range(len(observable.terms))):
            raise TypeError
        if observable.terms and observed_shots != shots:
            raise TypeError
        if seed == 0 and any(observed_seeds):
            raise TypeError
        if seed != 0 and len(set(observed_seeds)) != len(observed_seeds):
            raise TypeError
        return tuple(groups)
    except (TypeError, ValueError) as error:
        raise ExecutionError("Q7310: malformed native QWC expectation plan") from error


def _native_commuting_expectation_plan(
    observable: Hamiltonian,
    *,
    shots: int,
    seed: int,
) -> tuple[_MeasurementGroupPlan, ...]:
    """Project one canonical Rust general-commuting measurement plan.

    Args:
        observable (Hamiltonian): Canonical Hermitian Pauli sum.
        shots (int): Total device-shot budget.
        seed (int): Unsigned base seed; zero preserves endpoint randomness.

    Returns:
        tuple[_MeasurementGroupPlan, ...]: Checked groups and signed readouts.

    Raises:
        CapabilityError: If native commuting planning is unavailable.
        ExecutionError: If planning fails or its projection is malformed.
    """

    planner = getattr(native, "plan_commuting_pauli_expectation", None)
    gate_limit = getattr(native, "MAX_COMMUTING_MEASUREMENT_GATES", None)
    if not callable(planner) or type(gate_limit) is not int or gate_limit <= 0:
        raise CapabilityError(
            "Q7300: native commuting expectation planning is unavailable"
        )
    try:
        projection = planner(observable._native_literal, str(shots), seed)
    except (OverflowError, TypeError) as error:
        raise ExecutionError(
            "Q8138: seed must be an unsigned 64-bit integer"
        ) from error
    except ValueError as error:
        raise ExecutionError(str(error)) from error
    try:
        if not isinstance(projection, list):
            raise TypeError
        groups: list[_MeasurementGroupPlan] = []
        observed_terms: list[int] = []
        observed_shots = 0
        observed_seeds: list[int] = []
        for row in projection:
            if not isinstance(row, tuple) or len(row) != 5:
                raise TypeError
            raw_terms, raw_readouts, gate_count, raw_shots, raw_seed = row
            if (
                not isinstance(raw_terms, list)
                or not raw_terms
                or not isinstance(raw_readouts, list)
                or len(raw_readouts) != len(raw_terms)
                or type(gate_count) is not int
                or not 0 <= gate_count <= gate_limit
                or not isinstance(raw_shots, str)
                or not raw_shots.isdecimal()
                or type(raw_seed) is not int
                or not 0 <= raw_seed < 1 << 32
            ):
                raise TypeError
            term_indices = tuple(raw_terms)
            if any(type(index) is not int or index < 0 for index in term_indices):
                raise TypeError
            if any(
                left >= right for left, right in zip(term_indices, term_indices[1:])
            ):
                raise TypeError
            readouts: list[PauliReadout] = []
            for raw_readout in raw_readouts:
                if not isinstance(raw_readout, tuple) or len(raw_readout) != 3:
                    raise TypeError
                term_index, raw_factors, negative = raw_readout
                if (
                    type(term_index) is not int
                    or not isinstance(raw_factors, list)
                    or not raw_factors
                    or any(type(index) is not int or index < 0 for index in raw_factors)
                    or any(
                        left >= right
                        for left, right in zip(raw_factors, raw_factors[1:])
                    )
                    or type(negative) is not bool
                ):
                    raise TypeError
                word = PauliWord(
                    tuple(PauliFactor(index, Pauli.Z) for index in raw_factors)
                )
                readouts.append(PauliReadout(term_index, word, negative))
            if tuple(readout.term_index for readout in readouts) != term_indices:
                raise TypeError
            group_shots = int(raw_shots)
            if group_shots <= 0:
                raise TypeError
            groups.append(
                _MeasurementGroupPlan(
                    None,
                    term_indices,
                    group_shots,
                    raw_seed,
                    tuple(readouts),
                    gate_count,
                    "commuting",
                )
            )
            observed_terms.extend(term_indices)
            observed_shots += group_shots
            observed_seeds.append(raw_seed)
        if sorted(observed_terms) != list(range(len(observable.terms))):
            raise TypeError
        if observable.terms and observed_shots != shots:
            raise TypeError
        if seed == 0 and any(observed_seeds):
            raise TypeError
        if seed != 0 and len(set(observed_seeds)) != len(observed_seeds):
            raise TypeError
        return tuple(groups)
    except (TypeError, ValueError) as error:
        raise ExecutionError(
            "Q7310: malformed native commuting expectation plan"
        ) from error


def _measurement_program(program: Program, word: PauliWord) -> Program:
    """Derive exact Pauli-basis measurement through the Rust compiler.

    Args:
        program (Program): Verified state-preparation definition.
        word (PauliWord): Measurement basis in logical output order.

    Returns:
        Program: Verified program returning measured classical bits.

    Raises:
        CapabilityError: If native measurement composition is unavailable.
        TypeCheckError: If the compiler rejects measurement composition.
    """

    prepare = getattr(native, "prepare_pauli_measurement", None)
    if not callable(prepare):
        raise CapabilityError("Q7300: native Pauli measurement is unavailable")
    factors = tuple((factor.index, factor.pauli.value) for factor in word.factors)
    try:
        payload = bytes(prepare(program.to_bytes(), factors))
    except ValueError as error:
        raise TypeCheckError(str(error)) from error
    return Program.from_bytes(payload)


def _group_measurement_program(
    program: Program,
    observable: Hamiltonian,
    group: _MeasurementGroupPlan,
) -> Program:
    """Derive one basis from Rust-owned observable term positions.

    Args:
        program (Program): Verified state-preparation definition.
        observable (Hamiltonian): Canonical Pauli sum.
        group (_MeasurementGroupPlan): Checked local or Clifford measurement group.

    Returns:
        Program: Ordinary program returning classical measured bits.

    Raises:
        CapabilityError: If the required native transform is unavailable.
        TypeCheckError: If the compiler rejects measurement composition.
    """

    function_name = (
        "prepare_qwc_pauli_measurement"
        if group.strategy == "qwc"
        else "prepare_commuting_pauli_measurement"
    )
    if group.strategy == "independent":
        return _measurement_program(program, observable.terms[group.term_indices[0]][0])
    prepare = getattr(native, function_name, None)
    if not callable(prepare):
        raise CapabilityError("Q7300: native grouped Pauli measurement is unavailable")
    try:
        payload = bytes(
            prepare(
                program.to_bytes(),
                observable._native_literal,
                list(group.term_indices),
            )
        )
    except ValueError as error:
        raise TypeCheckError(str(error)) from error
    return Program.from_bytes(payload)
