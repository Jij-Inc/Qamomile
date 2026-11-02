"""HUGR target lifecycle over the structured compiler emitter."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

from qamomile.compilation import (
    CapabilitySnapshot,
    LoweredProgram,
    ParameterABI,
    ResultABI,
    _target_optimization_stages,
)
from qamomile.effects import Requirement
from qamomile.errors import CapabilityError
from qamomile.ir import FrozenMapping, Program, thaw_value
from qamomile.lowering import RealizationOptions

from ._compiler import (
    _DirectHugrLowerer,
    _hugr_type_row,
    _hugr_type_row_for_parameter_row,
    _parameter_qubit_width,
    _reconstruct_parameter_row,
)
from ._module import _finish_hugr_package, _prepare_hugr_module, _validate_link_name
from ._profile import CAPABILITIES as _CAPABILITIES, FORMAT as FORMAT


@dataclass(frozen=True, slots=True)
class Hugr:
    """Lower restored QCore directly into a validated HUGR package.

    Unlike circuit-family targets, this path walks structured QCore itself. It
    can therefore grow HUGR-native control and callable lowering without first
    flattening the program into a circuit data model.

    Args:
        record_results (bool): Include result-recording operations in HUGR.
        realizations (RealizationOptions): Exact operation implementations.
    """

    record_results: bool = True
    realizations: RealizationOptions = RealizationOptions()

    @property
    def name(self) -> str:
        """Return the stable artifact format identifier."""

        return FORMAT

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return the direct HUGR package capability contract.

        Returns:
            CapabilitySnapshot: Supported format and requirements.
        """

        return CapabilitySnapshot(FORMAT, _CAPABILITIES, realizations=self.realizations)

    @property
    def launch_materialized_requirements(self) -> frozenset[Requirement]:
        """Return requirements closed before HUGR package emission.

        Returns:
            frozenset[Requirement]: Requirements discharged at launch.
        """

        return frozenset(
            {
                Requirement.DYNAMIC_CLASSICAL_INDEX,
                Requirement.DYNAMIC_QUBIT_SELECTION,
                Requirement.HOST_DISPATCH,
                Requirement.PARAMETERIZED_EXECUTION,
            }
        )

    @property
    def fingerprint(self) -> str:
        """Return the pinned HUGR and TKET extension identity."""

        return (
            "hugr-tket-"
            + hashlib.sha256(
                (
                    "hugr-tket-v2:"
                    + str(self.record_results)
                    + repr(self.realizations.overrides)
                    + ":"
                    + version("hugr")
                    + ":"
                    + version("tket-exts")
                ).encode()
            ).hexdigest()[:20]
        )

    def lower(self, program: Program) -> LoweredProgram:
        """Retain a validated HUGR package and ABI facts.

        Args:
            program (Program): Verified QCore program for direct emission.

        Returns:
            LoweredProgram: Package owner and native-derived ABI facts.

        Raises:
            CapabilityError: If unresolved parameters or unsupported operations remain.
        """

        return self._lower(program, entry_bindings={})

    def lower_at(
        self,
        program: Program,
        parameters: Mapping[str, object],
    ) -> LoweredProgram:
        """Emit a closed HUGR package for one exact invocation.

        Args:
            program: Verified QCore program retained by the sealed launch
                artifact.
            parameters: Exact launch values already selected by the public
                artifact ABI.

        Returns:
            LoweredProgram: Argument-free HUGR package for this profile.

        Raises:
            CapabilityError: If native launch materialization rejects the
                invocation or the structured HUGR profile cannot emit it.
            ExecutionError: If the direct target method receives an invalid
                launch ABI mapping.
        """

        bound = dict(ParameterABI.from_program(program).bind(parameters))
        self._verify_native_launch(program, bound)
        return self._lower(program, entry_bindings=bound)

    def export_library(
        self,
        program: Program,
        *,
        link_name: str | None = None,
    ) -> Any:
        """Export one typed Qamomile function as a public HUGR library.

        Args:
            program: Verified and Const-specialized Qamomile program.
            link_name: Public symbol matched by HUGR package linking. The
                Qamomile entrypoint symbol is used by default.

        Returns:
            A validator-accepted ``hugr.package.Package`` whose public function
            retains the entrypoint's flattened typed input and output rows.

        Raises:
            CapabilityError: If the name, ABI, QCore body, or generated package
                is unsupported or exceeds an explicit emission budget.
        """

        name = program.entrypoint if link_name is None else link_name
        _validate_link_name(name)
        prepared = _prepare_hugr_module(program, self.realizations)
        function = prepared.program.entry
        bindings = dict(prepared.program.const_bindings)
        parameters = tuple(
            parameter
            for parameter in function.parameters
            if parameter.name not in bindings
        )
        definition = prepared.module.define_function(
            name,
            _hugr_type_row_for_parameter_row(parameters),
            _hugr_type_row(function.result_type),
            visibility="Public",
        )
        lowerer = _DirectHugrLowerer(
            prepared.program,
            definition,
            definitions=prepared.definitions,
            adjoint_helpers=prepared.adjoint_helpers,
            boolean_plans=prepared.boolean_plans,
            emission_budget=prepared.emission_budget,
        )
        inputs = _reconstruct_parameter_row(parameters, tuple(definition.inputs()))
        supplied = iter(inputs)
        environment: dict[int, object] = {}
        for parameter in function.parameters:
            if parameter.name in bindings:
                environment[parameter.value.id] = thaw_value(bindings[parameter.name])
            else:
                environment[parameter.value.id] = next(supplied)
        lowerer.live_qubits = _parameter_qubit_width(parameters)
        lowerer.allocated_qubits = lowerer.live_qubits
        kind, returned = lowerer._emit_function_body(function, environment)
        if kind != "return" or len(returned) != 1:
            raise CapabilityError(
                "Q8684: exported HUGR function did not return one typed value"
            )
        definition.set_outputs(
            *lowerer._flatten_value(returned[0], function.result_type)
        )
        package, _ = _finish_hugr_package(prepared.module)
        return package

    def _verify_native_launch(
        self,
        program: Program,
        parameters: Mapping[str, object],
    ) -> None:
        """Require Rust to validate the exact launch parameter ABI.

        Args:
            program (Program): Verified source program.
            parameters (Mapping[str, object]): Complete, typed invocation mapping.

        Raises:
            CapabilityError: If native launch validation rejects the parameter ABI.
        """

        from qamomile.circuit import validate_launch_parameters

        validate_launch_parameters(program, parameters)

    def _lower(
        self,
        program: Program,
        *,
        entry_bindings: Mapping[str, object],
    ) -> LoweredProgram:
        """Emit one direct HUGR package with optional exact entry values.

        Args:
            program (Program): Verified QCore program.
            entry_bindings (Mapping[str, object]): Exact launch values used
                only for the transient launch-materialized package.

        Returns:
            LoweredProgram: Validated HUGR package and execution ABI facts.

        Raises:
            CapabilityError: If unresolved parameters or unsupported QCore
                remain at direct emission.
        """

        source_parameter_abi = ParameterABI.from_program(program)
        declared_result = ResultABI.from_program(program)

        if source_parameter_abi.slots and not entry_bindings:
            raise CapabilityError(
                "Q8601: the HUGR/Selene profile requires launch materialization "
                "before direct emission of Param entrypoints"
            )
        prepared = _prepare_hugr_module(program, self.realizations)
        main = prepared.module.define_function("main", [], [], visibility="Public")
        lowerer = _DirectHugrLowerer(
            prepared.program,
            main,
            definitions=prepared.definitions,
            adjoint_helpers=prepared.adjoint_helpers,
            boolean_plans=prepared.boolean_plans,
            entry_bindings=entry_bindings,
            emission_budget=prepared.emission_budget,
            record_results=self.record_results,
        )
        lowerer.lower(declared_result)
        main.set_outputs()
        if self.record_results and lowerer.previous_measurement is not None:
            main.add_state_order(lowerer.previous_measurement, main.output_node)
        package, _ = _finish_hugr_package(
            prepared.module,
            record_results=self.record_results,
        )
        native_result_width = (
            declared_result.width
            if lowerer.uses_tail_result_measurements
            else lowerer.next_measurement
        )
        result_abi = ResultABI(
            declared_result.type,
            lowerer.result_order,
            native_result_width,
        )
        from ._payload import HugrExecutable

        return LoweredProgram(
            payload=HugrExecutable(package),
            parameter_abi=(ParameterABI() if entry_bindings else source_parameter_abi),
            result_abi=result_abi,
            stages=(
                "qcore-restored",
                *_target_optimization_stages(program),
                *(("launch-bound-entry",) if entry_bindings else ()),
                "callable-adjoint-expansion",
                "direct-hugr-emission",
                *(
                    f"realization:{identity}"
                    for identity in sorted(prepared.emission_budget.realizations)
                ),
                "hugr-validation",
            ),
            circuit_qubits=lowerer.allocated_qubits,
            circuit_gates=(
                lowerer.logical_operations + prepared.helper_logical_operations
            ),
            emitted_instructions=(
                lowerer.native_operations + prepared.helper_native_operations
            ),
            assumptions=(
                "The Selene profile closes entrypoint parameters before emission.",
            )
            if entry_bindings
            else (),
            metadata=FrozenMapping(
                (
                    ("retained_functions", len(prepared.definitions)),
                    ("hugr_version", version("hugr")),
                    ("record_results", self.record_results),
                    (
                        "result_measurements",
                        "trailing"
                        if lowerer.uses_tail_result_measurements
                        else "absolute",
                    ),
                    ("tket_exts_version", version("tket-exts")),
                )
            ),
        )
