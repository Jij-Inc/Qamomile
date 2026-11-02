"""HUGR module preparation and bounded package finalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qamomile._compiler_optimization import _preserve_preparation
from qamomile._native import compiler as native
from qamomile.effects import CallableKind
from qamomile.errors import CapabilityError, SerializationError, VerificationError
from qamomile.ir import Program
from qamomile.lowering import RealizationOptions
from qamomile.serialization import decode_target_adjoint

from ._compiler import (
    MAX_HUGR_BOOLEAN_LIFT_WIDTH,
    MAX_HUGR_CONTROLLED_PHASE_TERMS,
    MAX_HUGR_GENERAL_PERMUTATION_BITS,
    _BooleanLiftPlan,
    _DirectHugrLowerer,
    _EmissionBudget,
    _EmissionProgram,
    _hugr_type_row,
    _hugr_type_row_for_parameters,
    _reachable_functions,
    _reconstruct_parameters,
)

MAX_HUGR_LINK_NAME_BYTES = 4 * 1024
MAX_HUGR_PACKAGE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _PreparedHugrModule:
    """Shared callable definitions prepared for one HUGR package."""

    program: _EmissionProgram
    module: Any
    definitions: dict[str, Any]
    adjoint_helpers: dict[str, str]
    boolean_plans: dict[str, _BooleanLiftPlan]
    emission_budget: _EmissionBudget
    helper_logical_operations: int
    helper_native_operations: int


def _prepare_hugr_module(
    program: Program, options: RealizationOptions = RealizationOptions()
) -> _PreparedHugrModule:
    """Prepare shared private callables and checked realizations for HUGR emission.

    Args:
        program (Program): Verified and specialized QCore program.
        options (RealizationOptions): Exact realization choices.

    Returns:
        Shared HUGR module, target-lowered program, definitions, metrics, and
        one package-wide emission budget.

    Raises:
        CapabilityError: If native adjoint lowering or a retained callable
            cannot be emitted through the supported HUGR profile.
    """

    options.require_capabilities(())
    realized_program = (
        _preserve_preparation(
            program,
            Program.from_bytes(options._apply(program.to_bytes(), boundary="program")),
        )
        if options._custom
        else program
    )
    boolean_plans = _native_boolean_plans(realized_program)
    lower_adjoint = getattr(native, "lower_optimized_adjoint_calls_for_target", None)
    if not callable(lower_adjoint):
        raise CapabilityError("Q6200: native target adjoint lowering is unavailable")
    try:
        target_payload = bytes(lower_adjoint(realized_program._native_program()))
        emission_program = decode_target_adjoint(target_payload)
    except (ValueError, SerializationError, VerificationError) as error:
        raise CapabilityError(str(error)) from error

    import hugr.build as build

    module = build.Module()
    reachable = _reachable_functions(emission_program)
    retained = tuple(
        function
        for function in reachable
        if function.symbol != emission_program.entrypoint
        and function.kind is CallableKind.UNITARY
        and all(parameter.value.type.is_quantum for parameter in function.parameters)
    )
    definitions: dict[str, Any] = {}
    helper_builders: dict[str, Any] = {}
    for function in retained:
        helper = module.define_function(
            function.symbol.replace(".", "__"),
            _hugr_type_row_for_parameters(function),
            _hugr_type_row(function.result_type),
            visibility="Private",
        )
        definitions[function.symbol] = helper
        helper_builders[function.symbol] = helper

    emission_budget = _EmissionBudget(options=options)
    adjoint_helpers: dict[str, str] = {}
    helper_logical_operations = 0
    helper_native_operations = 0
    for function in retained:
        helper = helper_builders[function.symbol]
        lowerer = _DirectHugrLowerer(
            emission_program,
            helper,
            definitions=definitions,
            adjoint_helpers=adjoint_helpers,
            boolean_plans=boolean_plans,
            emission_budget=emission_budget,
        )
        inputs = _reconstruct_parameters(function, tuple(helper.inputs()))
        kind, returned = lowerer.lower_function(function, inputs)
        if kind != "return" or len(returned) != 1:
            raise CapabilityError(
                f"Q8636: HUGR helper {function.symbol!r} did not return one value"
            )
        helper.set_outputs(*lowerer._flatten_value(returned[0], function.result_type))
        helper_logical_operations += lowerer.logical_operations
        helper_native_operations += lowerer.native_operations

    return _PreparedHugrModule(
        emission_program,
        module,
        definitions,
        adjoint_helpers,
        boolean_plans,
        emission_budget,
        helper_logical_operations,
        helper_native_operations,
    )


def _native_boolean_plans(
    program: Program,
) -> dict[str, _BooleanLiftPlan]:
    """Collect exact Boolean ANF plans from the native semantic authority.

    Args:
        program (Program): Verified source-independent program before target
            rewriting.

    Returns:
        dict[str, _BooleanLiftPlan]: Predicate symbols mapped to input widths,
            output-ordered canonical ANFs, and optional permutation certificates.

    Raises:
        CapabilityError: If the native planner is absent or rejects a retained
            Boolean predicate.
    """

    planner = getattr(native, "plan_boolean_lifts", None)
    if not callable(planner):
        raise CapabilityError("Q8685: native Boolean planner is unavailable")
    try:
        rows = planner(program._native_program())
    except ValueError as error:
        raise CapabilityError(str(error)) from error
    plans: dict[str, _BooleanLiftPlan] = {}
    previous_predicate: str | None = None
    try:
        for predicate, width, outputs, permutation_order, basis_transpositions in rows:
            if (
                not isinstance(predicate, str)
                or not isinstance(width, int)
                or isinstance(width, bool)
                or width < 0
                or width > MAX_HUGR_BOOLEAN_LIFT_WIDTH
                or not isinstance(outputs, list)
                or not 0 < len(outputs) <= MAX_HUGR_BOOLEAN_LIFT_WIDTH
                or (previous_predicate is not None and predicate <= previous_predicate)
            ):
                raise ValueError("noncanonical native Boolean plan header")
            if any(not isinstance(monomials, list) for monomials in outputs):
                raise ValueError("invalid native Boolean output row")
            if (
                sum(len(monomials) for monomials in outputs)
                > MAX_HUGR_CONTROLLED_PHASE_TERMS
            ):
                raise ValueError("native Boolean plan exceeds its term budget")
            decoded_outputs = tuple(
                tuple(tuple(monomial) for monomial in monomials)
                for monomials in outputs
            )
            if not decoded_outputs:
                raise ValueError("empty native Boolean output row")
            for decoded_monomials in decoded_outputs:
                if (
                    any(
                        not isinstance(index, int)
                        or isinstance(index, bool)
                        or index < 0
                        or index >= width
                        for monomial in decoded_monomials
                        for index in monomial
                    )
                    or any(
                        tuple(sorted(set(monomial))) != monomial
                        for monomial in decoded_monomials
                    )
                    or len(set(decoded_monomials)) != len(decoded_monomials)
                ):
                    raise ValueError("invalid native Boolean monomial")
                if (
                    tuple(
                        sorted(
                            decoded_monomials,
                            key=lambda monomial: (len(monomial), monomial),
                        )
                    )
                    != decoded_monomials
                ):
                    raise ValueError("noncanonical native Boolean monomial order")
            decoded_permutation_order = (
                None if permutation_order is None else tuple(permutation_order)
            )
            if decoded_permutation_order is not None and (
                len(decoded_permutation_order) != len(decoded_outputs)
                or any(
                    not isinstance(index, int) or isinstance(index, bool)
                    for index in decoded_permutation_order
                )
                or tuple(sorted(decoded_permutation_order))
                != tuple(range(len(decoded_outputs)))
            ):
                raise ValueError("invalid native Boolean permutation order")
            decoded_basis_transpositions = (
                None
                if basis_transpositions is None
                else tuple(
                    tuple(transposition) for transposition in basis_transpositions
                )
            )
            if decoded_basis_transpositions is not None and (
                width > MAX_HUGR_GENERAL_PERMUTATION_BITS
                or not decoded_basis_transpositions
                or len(decoded_basis_transpositions) > (1 << width) - 1
                or any(
                    len(transposition) != 2
                    or any(
                        not isinstance(index, int)
                        or isinstance(index, bool)
                        or index < 0
                        or index >= 1 << width
                        for index in transposition
                    )
                    or transposition[0] == transposition[1]
                    for transposition in decoded_basis_transpositions
                )
            ):
                raise ValueError("invalid native Boolean basis transpositions")
            if (
                decoded_permutation_order is not None
                and decoded_basis_transpositions is not None
            ):
                raise ValueError("ambiguous native Boolean permutation proof")
            plans[predicate] = (
                width,
                decoded_outputs,
                decoded_permutation_order,
                decoded_basis_transpositions,
            )
            previous_predicate = predicate
    except (TypeError, ValueError, OverflowError) as error:
        raise CapabilityError("Q8685: malformed native Boolean plans") from error
    return plans


def _finish_hugr_package(
    module: Any,
    *,
    record_results: bool = False,
) -> tuple[Any, bytes]:
    """Validate and bound one fully emitted HUGR package.

    Args:
        module: Completed HUGR module builder.
        record_results: Whether the module uses ordered QSystem result records.

    Returns:
        The HUGR package and its deterministic serialized bytes.

    Raises:
        CapabilityError: If the encoded package exceeds the transfer budget.
    """

    import tket_exts
    from hugr.cli import validate
    from hugr.package import Package

    extensions = [
        tket_exts.quantum(),
        tket_exts.rotation(),
        tket_exts.bool(),
        tket_exts.global_phase(),
    ]
    if record_results:
        extensions.append(tket_exts.result())
    package = Package([module.hugr], extensions)
    payload = package.to_bytes()
    if len(payload) > MAX_HUGR_PACKAGE_BYTES:
        raise CapabilityError("Q8683: HUGR package exceeds the byte budget")
    validate(payload)
    return package, payload


def _validate_link_name(link_name: object) -> None:
    """Require one bounded public HUGR link symbol.

    Args:
        link_name: Candidate symbol supplied by the user or QCore entrypoint.

    Raises:
        CapabilityError: If the symbol is not printable, nonempty UTF-8 text
            within the explicit link-name budget.
    """

    if (
        not isinstance(link_name, str)
        or not link_name
        or not link_name.isprintable()
        or len(link_name.encode()) > MAX_HUGR_LINK_NAME_BYTES
    ):
        raise CapabilityError("Q8680: invalid HUGR library link name")
