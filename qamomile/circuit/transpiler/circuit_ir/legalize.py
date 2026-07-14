"""Target legalization and legality verification for circuit programs.

Legalization is an IR-to-IR pass: it consumes one verified
:class:`CircuitProgram`, selects target-native realizations without erasing
callable boundaries, and returns a new immutable program. Verification then
proves the result against the target's declared capabilities before any
materializer runs.

The pass rebuilds the program with freshly numbered wires instead of
patching instruction tuples in place. Fallback bodies stay attached and are
legalized recursively, allowing each materializer to lower them only at its
own SDK boundary.
"""

from __future__ import annotations

import dataclasses
import enum

from qamomile.circuit.transpiler.circuit_ir.capability import (
    CallControlMode,
    CallPhaseMode,
    CallTransformCapabilities,
    CircuitCapabilities,
    CompilationPolicy,
    ScalarAtom,
    ScalarCapabilities,
    ScalarExpressionForm,
    StandalonePhaseMode,
)
from qamomile.circuit.transpiler.circuit_ir.model import (
    BarrierInstruction,
    BinaryExpr,
    BinaryOperator,
    CallableIdentity,
    CallInstruction,
    CircuitInstruction,
    CircuitProgram,
    ClassicalBitExpr,
    ForInstruction,
    GateInstruction,
    IfInstruction,
    LiteralExpr,
    LoopVariableExpr,
    MeasureInstruction,
    MeasureVectorInstruction,
    ParameterExpr,
    PauliEvolutionInstruction,
    PauliEvolutionRealization,
    ResetInstruction,
    ScalarExpr,
    UnaryExpr,
    UnaryOperator,
    WhileInstruction,
    WireId,
)
from qamomile.circuit.transpiler.errors import TargetCapabilityError


def legalize_program(
    program: CircuitProgram,
    capabilities: CircuitCapabilities,
    policy: CompilationPolicy,
) -> CircuitProgram:
    """Rewrite one circuit program until it is legal for a target.

    Calls whose semantic key the target implements natively receive a
    target-owned realization identifier. Every other call retains its
    semantic identity and recursively legalized fallback body.

    Args:
        program (CircuitProgram): Verified backend-neutral circuit program.
        capabilities (CircuitCapabilities): Declared target capabilities.
        policy (CompilationPolicy): User realization preferences.

    Returns:
        CircuitProgram: Rebuilt program with freshly numbered wires.
    """
    return _Rewriter(capabilities, policy).rewrite_program(program)


def verify_target_legal(
    program: CircuitProgram,
    capabilities: CircuitCapabilities,
) -> None:
    """Prove a legalized program against declared target capabilities.

    Args:
        program (CircuitProgram): Legalized circuit program, including every
            nested reusable-call body.
        capabilities (CircuitCapabilities): Declared target capabilities.

    Raises:
        TargetCapabilityError: If any instruction requires a gate kind,
            semantic realization, control-flow construct, reset, Pauli evolution, or
            scalar-expression shape the target does not declare.
    """
    _verify_legal_program(program, capabilities)


class _Rewriter:
    """Rebuild a circuit program while applying legalization decisions.

    Args:
        capabilities (CircuitCapabilities): Declared target capabilities.
        policy (CompilationPolicy): User realization preferences.
    """

    def __init__(
        self,
        capabilities: CircuitCapabilities,
        policy: CompilationPolicy,
    ) -> None:
        """Initialize one single-program rewrite.

        Args:
            capabilities (CircuitCapabilities): Declared target capabilities.
            policy (CompilationPolicy): User realization preferences.
        """
        self._capabilities = capabilities
        self._policy = policy
        self._next_wire = 0

    def rewrite_program(
        self,
        program: CircuitProgram,
        inherited_coherent_controls: int = 0,
        inherited_distributed_controls: int = 0,
    ) -> CircuitProgram:
        """Rebuild one program with fresh wires and applied decisions.

        Args:
            program (CircuitProgram): Program to rewrite.
            inherited_coherent_controls (int): Enclosing controls that make
                program phase observable. Defaults to zero.
            inherited_distributed_controls (int): Enclosing controls that are
                physically distributed into this program. Defaults to zero.

        Returns:
            CircuitProgram: Rebuilt immutable program.
        """
        environment: dict[WireId, WireId] = {}
        input_wires = tuple(self._fresh() for _ in program.input_wires)
        for old, new in zip(program.input_wires, input_wires, strict=True):
            environment[old] = new
        operations = self._rewrite_region(
            program.operations,
            environment,
            inherited_coherent_controls,
            inherited_distributed_controls,
        )
        return CircuitProgram(
            name=program.name,
            num_qubits=program.num_qubits,
            num_clbits=program.num_clbits,
            input_wires=input_wires,
            output_wires=tuple(environment[wire] for wire in program.output_wires),
            operations=tuple(operations),
            global_phase=program.global_phase,
        )

    def _fresh(self) -> WireId:
        """Allocate one fresh wire in the rebuilt program's numbering.

        Returns:
            WireId: Newly allocated wire identifier.
        """
        wire = WireId(self._next_wire)
        self._next_wire += 1
        return wire

    def _define(
        self,
        outputs: tuple[WireId, ...],
        environment: dict[WireId, WireId],
    ) -> tuple[WireId, ...]:
        """Mint fresh wires for instruction outputs and record them.

        Args:
            outputs (tuple[WireId, ...]): Original output wires.
            environment (dict[WireId, WireId]): Original-to-rebuilt mapping.

        Returns:
            tuple[WireId, ...]: Fresh rebuilt output wires.
        """
        fresh = tuple(self._fresh() for _ in outputs)
        for old, new in zip(outputs, fresh, strict=True):
            environment[old] = new
        return fresh

    def _rewrite_region(
        self,
        operations: tuple[CircuitInstruction, ...],
        environment: dict[WireId, WireId],
        inherited_coherent_controls: int,
        inherited_distributed_controls: int,
    ) -> list[CircuitInstruction]:
        """Rewrite one structured region.

        Args:
            operations (tuple[CircuitInstruction, ...]): Region instructions.
            environment (dict[WireId, WireId]): Original-to-rebuilt mapping,
                shared across the whole program because original wires are
                SSA-unique.
            inherited_coherent_controls (int): Enclosing coherent controls.
            inherited_distributed_controls (int): Enclosing controls being
                distributed into operations in this region.

        Returns:
            list[CircuitInstruction]: Rebuilt region instructions.

        Raises:
            TypeError: If the region contains an unknown instruction type.
        """
        result: list[CircuitInstruction] = []
        for operation in operations:
            if isinstance(operation, GateInstruction):
                result.append(
                    GateInstruction(
                        kind=operation.kind,
                        inputs=self._map(operation.inputs, environment),
                        outputs=self._define(operation.outputs, environment),
                        parameters=operation.parameters,
                    )
                )
            elif isinstance(operation, MeasureInstruction):
                result.append(
                    MeasureInstruction(
                        input=environment[operation.input],
                        output=self._define((operation.output,), environment)[0],
                        clbit=operation.clbit,
                    )
                )
            elif isinstance(operation, MeasureVectorInstruction):
                result.append(
                    MeasureVectorInstruction(
                        inputs=self._map(operation.inputs, environment),
                        outputs=self._define(operation.outputs, environment),
                        clbits=operation.clbits,
                    )
                )
            elif isinstance(operation, ResetInstruction):
                result.append(
                    ResetInstruction(
                        input=environment[operation.input],
                        output=self._define((operation.output,), environment)[0],
                    )
                )
            elif isinstance(operation, BarrierInstruction):
                result.append(
                    BarrierInstruction(self._map(operation.wires, environment))
                )
            elif isinstance(operation, PauliEvolutionInstruction):
                realization = self._select_pauli_realization()
                result.append(
                    PauliEvolutionInstruction(
                        hamiltonian=operation.hamiltonian,
                        time=operation.time,
                        inputs=self._map(operation.inputs, environment),
                        outputs=self._define(operation.outputs, environment),
                        realization=realization,
                    )
                )
            elif isinstance(operation, CallInstruction):
                self._rewrite_call(
                    operation,
                    environment,
                    result,
                    inherited_coherent_controls,
                    inherited_distributed_controls,
                )
            elif isinstance(operation, ForInstruction):
                inputs = self._map(operation.inputs, environment)
                body = self._rewrite_region(
                    operation.body,
                    environment,
                    inherited_coherent_controls,
                    inherited_distributed_controls,
                )
                result.append(
                    ForInstruction(
                        indexset=operation.indexset,
                        loop_variable=operation.loop_variable,
                        inputs=inputs,
                        body=tuple(body),
                        body_outputs=self._map(operation.body_outputs, environment),
                        outputs=self._define(operation.outputs, environment),
                    )
                )
            elif isinstance(operation, IfInstruction):
                inputs = self._map(operation.inputs, environment)
                true_body = self._rewrite_region(
                    operation.true_body,
                    environment,
                    inherited_coherent_controls,
                    inherited_distributed_controls,
                )
                true_outputs = self._map(operation.true_outputs, environment)
                false_body = self._rewrite_region(
                    operation.false_body,
                    environment,
                    inherited_coherent_controls,
                    inherited_distributed_controls,
                )
                false_outputs = self._map(operation.false_outputs, environment)
                result.append(
                    IfInstruction(
                        condition=operation.condition,
                        inputs=inputs,
                        true_body=tuple(true_body),
                        false_body=tuple(false_body),
                        true_outputs=true_outputs,
                        false_outputs=false_outputs,
                        outputs=self._define(operation.outputs, environment),
                    )
                )
            elif isinstance(operation, WhileInstruction):
                inputs = self._map(operation.inputs, environment)
                body = self._rewrite_region(
                    operation.body,
                    environment,
                    inherited_coherent_controls,
                    inherited_distributed_controls,
                )
                result.append(
                    WhileInstruction(
                        condition=operation.condition,
                        inputs=inputs,
                        body=tuple(body),
                        body_outputs=self._map(operation.body_outputs, environment),
                        outputs=self._define(operation.outputs, environment),
                    )
                )
            else:  # pragma: no cover - closed union defensive guard
                raise TypeError(
                    f"Unknown circuit instruction: {type(operation).__name__}"
                )
        return result

    def _select_pauli_realization(self) -> PauliEvolutionRealization:
        """Choose one target-supported Pauli-evolution realization.

        Returns:
            PauliEvolutionRealization: Native or gadget realization selected
            from target capabilities and compilation policy.

        Raises:
            TargetCapabilityError: If the target accepts no concrete Pauli
                evolution realization.
        """
        supported = self._capabilities.pauli_realizations
        if (
            self._policy.prefer_native_pauli_evolution
            and PauliEvolutionRealization.NATIVE in supported
        ):
            return PauliEvolutionRealization.NATIVE
        if PauliEvolutionRealization.GADGET in supported:
            return PauliEvolutionRealization.GADGET
        if PauliEvolutionRealization.NATIVE in supported:
            return PauliEvolutionRealization.NATIVE
        raise TargetCapabilityError(
            f"Target '{self._capabilities.name}' cannot realize Pauli evolution",
            target=self._capabilities.name,
            operation="PauliEvolutionInstruction",
        )

    @staticmethod
    def _map(
        wires: tuple[WireId, ...],
        environment: dict[WireId, WireId],
    ) -> tuple[WireId, ...]:
        """Map original wire references into the rebuilt numbering.

        Args:
            wires (tuple[WireId, ...]): Original wire references.
            environment (dict[WireId, WireId]): Original-to-rebuilt mapping.

        Returns:
            tuple[WireId, ...]: Rebuilt wire references.
        """
        return tuple(environment[wire] for wire in wires)

    def _rewrite_call(
        self,
        operation: CallInstruction,
        environment: dict[WireId, WireId],
        result: list[CircuitInstruction],
        inherited_coherent_controls: int,
        inherited_distributed_controls: int,
    ) -> None:
        """Select a native realization while preserving the call boundary.

        Args:
            operation (CallInstruction): Original call instruction.
            environment (dict[WireId, WireId]): Original-to-rebuilt mapping.
            result (list[CircuitInstruction]): Region being rebuilt.
            inherited_coherent_controls (int): Enclosing coherent controls.
            inherited_distributed_controls (int): Enclosing controls being
                distributed into this call.
        """
        callee = operation.callee
        identity = callee.identity
        native_declaration = (
            self._capabilities.native_semantic_op(identity.key)
            if identity is not None
            else None
        )
        native = (
            self._policy.prefer_native_semantic_ops
            and native_declaration is not None
            and native_declaration.accepts(callee, inherited_distributed_controls)
            and (
                _is_zero_literal(callee.body.global_phase)
                or native_declaration.call_transforms.phase_mode
                is CallPhaseMode.NATIVE_BODY
            )
        )
        if native:
            assert native_declaration is not None
            self._keep_call(
                operation,
                environment,
                result,
                identity,
                legalize_body=False,
                native_realization=native_declaration.realization,
                inherited_coherent_controls=0,
                inherited_distributed_controls=0,
            )
            return
        body_coherent_controls = inherited_coherent_controls + callee.controls
        body_distributed_controls = 0
        if self._capabilities.generic_calls.control_mode is CallControlMode.DISTRIBUTE:
            body_distributed_controls = inherited_distributed_controls + callee.controls
        self._keep_call(
            operation,
            environment,
            result,
            identity,
            legalize_body=True,
            native_realization=None,
            inherited_coherent_controls=body_coherent_controls,
            inherited_distributed_controls=body_distributed_controls,
        )

    def _keep_call(
        self,
        operation: CallInstruction,
        environment: dict[WireId, WireId],
        result: list[CircuitInstruction],
        identity: CallableIdentity | None,
        legalize_body: bool,
        native_realization: str | None,
        inherited_coherent_controls: int,
        inherited_distributed_controls: int,
    ) -> None:
        """Retain a call while legalizing its body recursively.

        Args:
            operation (CallInstruction): Original call instruction.
            environment (dict[WireId, WireId]): Original-to-rebuilt mapping.
            result (list[CircuitInstruction]): Region being rebuilt.
            identity (CallableIdentity | None): Identity for the rebuilt
                callee, possibly demoted from an semantic identity.
            legalize_body (bool): Whether the fallback body will execute on
                this target and therefore requires recursive legalization.
            native_realization (str | None): Selected target realization, or
                ``None`` to retain the reusable fallback body.
            inherited_coherent_controls (int): Controls inherited by the
                fallback body for phase semantics.
            inherited_distributed_controls (int): Controls physically
                distributed into the fallback body.
        """
        callee = operation.callee
        rebuilt_body = (
            _Rewriter(
                self._capabilities,
                self._policy,
            ).rewrite_program(
                callee.body,
                inherited_coherent_controls,
                inherited_distributed_controls,
            )
            if legalize_body
            else callee.body
        )
        result.append(
            CallInstruction(
                callee=dataclasses.replace(
                    callee,
                    body=rebuilt_body,
                    identity=identity,
                    native_realization=native_realization,
                ),
                inputs=self._map(operation.inputs, environment),
                outputs=self._define(operation.outputs, environment),
            )
        )


def _is_zero_literal(expression: ScalarExpr) -> bool:
    """Return whether an expression is the literal zero.

    Args:
        expression (ScalarExpr): Expression to inspect.

    Returns:
        bool: True when the expression is ``LiteralExpr(0)`` up to type.
    """
    return isinstance(expression, LiteralExpr) and not float(expression.value)


def _verify_legal_program(
    program: CircuitProgram,
    capabilities: CircuitCapabilities,
    inherited_coherent_controls: int = 0,
    inherited_distributed_controls: int = 0,
    phase_support: CallTransformCapabilities | None = None,
) -> None:
    """Verify one complete program against target capabilities.

    Args:
        program (CircuitProgram): Program whose global phase and operations are
            target-legalized.
        capabilities (CircuitCapabilities): Declared target capabilities.
        inherited_coherent_controls (int): Coherent controls accumulated from
            enclosing reusable calls. Defaults to zero.
        inherited_distributed_controls (int): Enclosing controls physically
            distributed into this program. Defaults to zero.
        phase_support (CallTransformCapabilities | None): Transform contract
            responsible for a phase made observable by inherited controls.
            Defaults to ``None`` at the entrypoint.

    Raises:
        TargetCapabilityError: If global phase or any nested instruction lies
            outside the declared target language.
    """
    _verify_program_phase(
        program,
        capabilities,
        inherited_coherent_controls,
        phase_support,
    )
    _verify_legal_region(
        program.operations,
        capabilities,
        inherited_coherent_controls,
        inherited_distributed_controls,
    )


def _verify_program_phase(
    program: CircuitProgram,
    capabilities: CircuitCapabilities,
    inherited_controls: int,
    phase_support: CallTransformCapabilities | None,
) -> None:
    """Verify one program phase in standalone or coherent-control context.

    Args:
        program (CircuitProgram): Program whose phase is inspected.
        capabilities (CircuitCapabilities): Target capability declaration.
        inherited_controls (int): Number of active coherent controls.
        phase_support (CallTransformCapabilities | None): Call realization
            responsible for an observable controlled phase.

    Raises:
        TargetCapabilityError: If the phase or its scalar expression is not
            supported in the active context.
    """
    if _is_zero_literal(program.global_phase):
        return
    if inherited_controls:
        if (
            phase_support is None
            or phase_support.phase_mode is CallPhaseMode.UNSUPPORTED
            or phase_support.controlled_phase_scalars is None
        ):
            raise TargetCapabilityError(
                f"Target '{capabilities.name}' cannot preserve a reusable "
                "global phase under coherent controls",
                target=capabilities.name,
                operation="controlled global phase",
            )
        scalar_capabilities = phase_support.controlled_phase_scalars
        context = "controlled reusable global phase"
    else:
        if capabilities.global_phase is None:
            raise TargetCapabilityError(
                f"Target '{capabilities.name}' cannot accept a nonzero "
                "standalone global phase",
                target=capabilities.name,
                operation="global phase",
            )
        if capabilities.global_phase.standalone_mode is StandalonePhaseMode.DISCARD:
            return
        if program.num_qubits < capabilities.global_phase.min_qubits:
            raise TargetCapabilityError(
                f"Target '{capabilities.name}' requires at least "
                f"{capabilities.global_phase.min_qubits} qubit to preserve a "
                "nonzero standalone global phase",
                target=capabilities.name,
                operation="global phase",
            )
        scalar_capabilities = capabilities.global_phase.scalars
        context = "global phase"
    _check_scalar(
        program.global_phase,
        scalar_capabilities,
        capabilities.name,
        context=context,
    )


def _verify_legal_region(
    operations: tuple[CircuitInstruction, ...],
    capabilities: CircuitCapabilities,
    inherited_coherent_controls: int = 0,
    inherited_distributed_controls: int = 0,
) -> None:
    """Verify one structured region against target capabilities.

    Args:
        operations (tuple[CircuitInstruction, ...]): Region instructions.
        capabilities (CircuitCapabilities): Declared target capabilities.
        inherited_coherent_controls (int): Coherent controls inherited by this
            region. Defaults to zero.
        inherited_distributed_controls (int): Controls physically distributed
            into this region. Defaults to zero.

    Raises:
        TargetCapabilityError: If any nested instruction is illegal for the
            target.
    """
    name = capabilities.name
    for operation in operations:
        if isinstance(operation, GateInstruction):
            if operation.kind not in capabilities.primitive_gates:
                raise TargetCapabilityError(
                    f"Target '{name}' does not support the {operation.kind.name} gate",
                    target=name,
                    operation=operation.kind.name,
                )
            for parameter in operation.parameters:
                _check_scalar(
                    parameter,
                    capabilities.gate_parameters,
                    capabilities.name,
                    context=f"{operation.kind.name} angle",
                )
        elif isinstance(operation, ResetInstruction):
            if not capabilities.supports_reset:
                raise TargetCapabilityError(
                    f"Target '{name}' cannot represent a mid-circuit reset",
                    target=name,
                    operation="ResetInstruction",
                )
        elif isinstance(operation, PauliEvolutionInstruction):
            if operation.realization is PauliEvolutionRealization.ABSTRACT:
                raise TargetCapabilityError(
                    "Pauli evolution survived legalization without a concrete "
                    f"realization for target '{name}'",
                    target=name,
                    operation="PauliEvolutionInstruction",
                )
            if operation.realization not in capabilities.pauli_realizations:
                raise TargetCapabilityError(
                    f"Target '{name}' does not accept the selected Pauli "
                    f"evolution realization {operation.realization.name}",
                    target=name,
                    operation="PauliEvolutionInstruction",
                )
            _check_scalar(
                operation.time,
                capabilities.pauli_time,
                capabilities.name,
                context="Pauli evolution time",
            )
        elif isinstance(operation, CallInstruction):
            identity = operation.callee.identity
            realization = operation.callee.native_realization
            if realization is not None:
                declaration = (
                    capabilities.native_semantic_op(identity.key)
                    if identity is not None
                    else None
                )
                if (
                    declaration is None
                    or declaration.realization != realization
                    or not declaration.accepts(
                        operation.callee,
                        inherited_distributed_controls,
                    )
                ):
                    raise TargetCapabilityError(
                        f"Native realization {realization!r} survived legalization "
                        f"without a matching capability on target '{name}'; "
                        "this is a legalization invariant violation",
                        target=name,
                        operation=identity.symbol if identity is not None else "call",
                    )
                # The fallback body is retained for other targets but is not
                # executed by this target's native realization. Structural
                # verification already recursed into it via verify_circuit().
                assert declaration is not None
                effective_controls = (
                    inherited_coherent_controls + operation.callee.controls
                )
                _verify_program_phase(
                    operation.callee.body,
                    capabilities,
                    effective_controls,
                    declaration.call_transforms,
                )
                continue
            _verify_call(
                operation,
                capabilities,
                inherited_distributed_controls,
            )
            effective_controls = inherited_coherent_controls + operation.callee.controls
            distributed_controls = 0
            if capabilities.generic_calls.control_mode is CallControlMode.DISTRIBUTE:
                distributed_controls = (
                    inherited_distributed_controls + operation.callee.controls
                )
            _verify_legal_program(
                operation.callee.body,
                capabilities,
                inherited_coherent_controls=effective_controls,
                inherited_distributed_controls=distributed_controls,
                phase_support=capabilities.generic_calls,
            )
        elif isinstance(operation, IfInstruction):
            if not capabilities.supports_dynamic_if:
                raise TargetCapabilityError(
                    f"Target '{name}' cannot represent measurement-"
                    "conditioned branching inside a circuit artifact",
                    target=name,
                    operation="IfInstruction",
                )
            _check_scalar(
                operation.condition,
                capabilities.predicates,
                capabilities.name,
                context="if predicate",
            )
            _verify_legal_region(
                operation.true_body,
                capabilities,
                inherited_coherent_controls,
                inherited_distributed_controls,
            )
            _verify_legal_region(
                operation.false_body,
                capabilities,
                inherited_coherent_controls,
                inherited_distributed_controls,
            )
        elif isinstance(operation, WhileInstruction):
            if not capabilities.supports_dynamic_while:
                raise TargetCapabilityError(
                    f"Target '{name}' cannot represent measurement-"
                    "conditioned loops inside a circuit artifact",
                    target=name,
                    operation="WhileInstruction",
                )
            _check_scalar(
                operation.condition,
                capabilities.predicates,
                capabilities.name,
                context="while predicate",
            )
            _verify_legal_region(
                operation.body,
                capabilities,
                inherited_coherent_controls,
                inherited_distributed_controls,
            )
        elif isinstance(operation, ForInstruction):
            _verify_legal_region(
                operation.body,
                capabilities,
                inherited_coherent_controls,
                inherited_distributed_controls,
            )
        # Measure and barrier instructions are legal on every circuit target.


def _verify_call(
    operation: CallInstruction,
    capabilities: CircuitCapabilities,
    inherited_distributed_controls: int,
) -> None:
    """Verify a generic reusable call against target call capabilities.

    Args:
        operation (CallInstruction): Generic call to inspect.
        capabilities (CircuitCapabilities): Declared target capabilities.
        inherited_distributed_controls (int): Controls physically distributed
            from enclosing calls.

    Raises:
        TargetCapabilityError: If transforms or a nonunitary body are not
            accepted by the target.
    """
    callee = operation.callee
    support = capabilities.generic_calls
    if not support.accepts(callee, inherited_distributed_controls):
        raise TargetCapabilityError(
            f"Target '{capabilities.name}' cannot realize reusable call "
            f"transforms (power={callee.power}, inverse={callee.inverse}, "
            f"controls={callee.controls})",
            target=capabilities.name,
            operation="CallInstruction",
        )
    if not support.supports_nonunitary_body and (
        callee.body.num_clbits > 0 or not _is_unitary_region(callee.body.operations)
    ):
        raise TargetCapabilityError(
            f"Target '{capabilities.name}' cannot realize a reusable call "
            "whose body contains measurement, reset, or dynamic control",
            target=capabilities.name,
            operation="CallInstruction",
        )
    if not support.supports_barrier_body and _contains_barrier(callee.body.operations):
        raise TargetCapabilityError(
            f"Target '{capabilities.name}' cannot preserve a barrier inside a "
            "reusable call body",
            target=capabilities.name,
            operation="CallInstruction",
        )
    effective_controls = inherited_distributed_controls + callee.controls
    if effective_controls and support.control_mode is CallControlMode.DISTRIBUTE:
        _verify_distributed_control_region(
            callee.body.operations,
            support,
            capabilities.name,
        )


def _is_unitary_region(operations: tuple[CircuitInstruction, ...]) -> bool:
    """Return whether a reusable region contains only unitary constructs.

    Args:
        operations (tuple[CircuitInstruction, ...]): Region to inspect.

    Returns:
        bool: Whether the region can be treated as a reusable unitary body.
    """
    for operation in operations:
        if isinstance(
            operation,
            (MeasureInstruction, MeasureVectorInstruction, ResetInstruction),
        ):
            return False
        if isinstance(operation, (IfInstruction, WhileInstruction)):
            return False
        if isinstance(operation, ForInstruction):
            if not _is_unitary_region(operation.body):
                return False
        elif isinstance(operation, CallInstruction):
            if not _is_unitary_region(operation.callee.body.operations):
                return False
    return True


def _contains_barrier(operations: tuple[CircuitInstruction, ...]) -> bool:
    """Return whether a reusable region contains a barrier recursively.

    Args:
        operations (tuple[CircuitInstruction, ...]): Region to inspect.

    Returns:
        bool: Whether any nested region or call body contains a barrier.
    """
    for operation in operations:
        if isinstance(operation, BarrierInstruction):
            return True
        if isinstance(operation, ForInstruction) and _contains_barrier(operation.body):
            return True
        if isinstance(operation, CallInstruction) and _contains_barrier(
            operation.callee.body.operations
        ):
            return True
    return False


def _verify_distributed_control_region(
    operations: tuple[CircuitInstruction, ...],
    support: CallTransformCapabilities,
    target_name: str,
) -> None:
    """Verify a body whose outer controls are distributed to instructions.

    Args:
        operations (tuple[CircuitInstruction, ...]): Reusable body operations.
        support (CallTransformCapabilities): Distributed-control declaration.
        target_name (str): Stable target name used in diagnostics.

    Raises:
        TargetCapabilityError: If a body instruction has no declared
            controlled realization.
    """
    for operation in operations:
        if isinstance(operation, GateInstruction):
            if operation.kind not in support.controlled_gate_kinds:
                raise TargetCapabilityError(
                    f"Target '{target_name}' cannot distribute call controls "
                    f"onto {operation.kind.name}",
                    target=target_name,
                    operation=operation.kind.name,
                )
        elif isinstance(operation, PauliEvolutionInstruction):
            if support.controlled_pauli_time is None:
                raise TargetCapabilityError(
                    f"Target '{target_name}' cannot distribute call controls "
                    "onto Pauli evolution",
                    target=target_name,
                    operation="PauliEvolutionInstruction",
                )
            _check_scalar(
                operation.time,
                support.controlled_pauli_time,
                target_name,
                context="controlled Pauli evolution time",
            )
        elif isinstance(operation, ForInstruction):
            _verify_distributed_control_region(operation.body, support, target_name)


class _ParameterForm(enum.Enum):
    """Classify how runtime parameters occur inside a scalar expression."""

    CONST = "const"
    LINEAR = "linear"
    NONLINEAR = "nonlinear"


def _check_scalar(
    expression: ScalarExpr,
    capabilities: ScalarCapabilities,
    target_name: str,
    context: str,
) -> None:
    """Check one scalar expression against a context-specific declaration.

    Args:
        expression (ScalarExpr): Expression to check.
        capabilities (ScalarCapabilities): Scalar language accepted in this
            instruction context.
        target_name (str): Stable target name used in diagnostics.
        context (str): Human-readable location used in diagnostics.

    Raises:
        TargetCapabilityError: If the expression shape exceeds the declared
            :class:`ScalarExpressionForm`.
    """
    _check_scalar_vocabulary(expression, capabilities, target_name, context)
    form = capabilities.parameter_form
    if form is ScalarExpressionForm.ARBITRARY:
        return
    classified = _classify(expression)
    if classified is _ParameterForm.CONST:
        return
    names = ", ".join(sorted(_parameter_names(expression)))
    if form is ScalarExpressionForm.CONCRETE_ONLY:
        raise TargetCapabilityError(
            f"Target '{target_name}' accepts only concrete values in "
            f"{context}; runtime parameters ({names}) must be supplied "
            "through bindings",
            target=target_name,
            operation=context,
        )
    if classified is _ParameterForm.NONLINEAR:
        raise TargetCapabilityError(
            f"Target '{target_name}' accepts only linear parameter "
            f"combinations in {context}, but the expression is non-linear "
            f"in runtime parameters ({names}); bind one side to a concrete "
            "value first",
            target=target_name,
            operation=context,
        )


def _check_scalar_vocabulary(
    expression: ScalarExpr,
    capabilities: ScalarCapabilities,
    target_name: str,
    context: str,
) -> None:
    """Verify scalar atoms and operators for one target context.

    Args:
        expression (ScalarExpr): Expression to inspect recursively.
        capabilities (ScalarCapabilities): Accepted atoms and operators.
        target_name (str): Stable target name used in diagnostics.
        context (str): Human-readable expression location.

    Raises:
        TargetCapabilityError: If an atom or operator is not accepted.
    """
    atom: ScalarAtom | None = None
    if isinstance(expression, LiteralExpr):
        atom = ScalarAtom.LITERAL
    elif isinstance(expression, ParameterExpr):
        atom = ScalarAtom.PARAMETER
    elif isinstance(expression, ClassicalBitExpr):
        atom = ScalarAtom.CLASSICAL_BIT
    elif isinstance(expression, LoopVariableExpr):
        atom = ScalarAtom.LOOP_VARIABLE
    if atom is not None:
        if atom not in capabilities.atoms:
            suggestion = (
                "; runtime parameters must be supplied through bindings"
                if atom is ScalarAtom.PARAMETER
                else ""
            )
            raise TargetCapabilityError(
                f"Target '{target_name}' does not accept {atom.value} values "
                f"in {context}{suggestion}",
                target=target_name,
                operation=context,
            )
        return
    if isinstance(expression, UnaryExpr):
        if expression.operator not in capabilities.unary_operators:
            raise TargetCapabilityError(
                f"Target '{target_name}' does not accept unary operator "
                f"{expression.operator.name} in {context}",
                target=target_name,
                operation=context,
            )
        _check_scalar_vocabulary(
            expression.operand,
            capabilities,
            target_name,
            context,
        )
        return
    if isinstance(expression, BinaryExpr):
        if expression.operator not in capabilities.binary_operators:
            raise TargetCapabilityError(
                f"Target '{target_name}' does not accept binary operator "
                f"{expression.operator.name} in {context}",
                target=target_name,
                operation=context,
            )
        _check_scalar_vocabulary(
            expression.left,
            capabilities,
            target_name,
            context,
        )
        _check_scalar_vocabulary(
            expression.right,
            capabilities,
            target_name,
            context,
        )


def _classify(expression: ScalarExpr) -> _ParameterForm:
    """Classify an expression's shape with respect to runtime parameters.

    Classical bits and loop variables count as concrete: they carry no
    runtime parameter and materialize to per-target concrete values.

    Args:
        expression (ScalarExpr): Expression to classify.

    Returns:
        _ParameterForm: CONST when parameter-free, LINEAR when parameters
            occur only in linear combinations, NONLINEAR otherwise.
    """
    if isinstance(expression, (LiteralExpr, ClassicalBitExpr, LoopVariableExpr)):
        return _ParameterForm.CONST
    if isinstance(expression, ParameterExpr):
        return _ParameterForm.LINEAR
    if isinstance(expression, UnaryExpr):
        operand = _classify(expression.operand)
        if expression.operator is UnaryOperator.NEG:
            return operand
        return (
            _ParameterForm.CONST
            if operand is _ParameterForm.CONST
            else _ParameterForm.NONLINEAR
        )
    # The scalar union is closed, so the remaining case is BinaryExpr.
    left = _classify(expression.left)
    right = _classify(expression.right)
    if _ParameterForm.NONLINEAR in (left, right):
        return _ParameterForm.NONLINEAR
    operator = expression.operator
    if operator in (BinaryOperator.ADD, BinaryOperator.SUB):
        if left is _ParameterForm.CONST and right is _ParameterForm.CONST:
            return _ParameterForm.CONST
        return _ParameterForm.LINEAR
    if operator is BinaryOperator.MUL:
        if left is _ParameterForm.CONST and right is _ParameterForm.CONST:
            return _ParameterForm.CONST
        if left is _ParameterForm.CONST or right is _ParameterForm.CONST:
            return _ParameterForm.LINEAR
        return _ParameterForm.NONLINEAR
    if operator is BinaryOperator.DIV:
        if right is _ParameterForm.CONST:
            return left
        return _ParameterForm.NONLINEAR
    if left is _ParameterForm.CONST and right is _ParameterForm.CONST:
        return _ParameterForm.CONST
    return _ParameterForm.NONLINEAR


def _parameter_names(expression: ScalarExpr) -> set[str]:
    """Collect the runtime parameter names referenced by an expression.

    Args:
        expression (ScalarExpr): Expression to inspect.

    Returns:
        set[str]: Referenced runtime parameter names, possibly empty.
    """
    if isinstance(expression, ParameterExpr):
        return {expression.name}
    if isinstance(expression, BinaryExpr):
        return _parameter_names(expression.left) | _parameter_names(expression.right)
    if isinstance(expression, UnaryExpr):
        return _parameter_names(expression.operand)
    return set()
