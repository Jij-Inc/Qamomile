"""Target-neutral compiler entrypoint for Qamomile programs."""

from __future__ import annotations

import dataclasses
from typing import Any, TypeVar

from qamomile.circuit.frontend.param_validation import (
    validate_bindings_parameters_disjoint,
)
from qamomile.circuit.frontend.qkernel_like import QKernelLike
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.transpiler.artifact import CompiledProgram
from qamomile.circuit.transpiler.config import CompilerConfig
from qamomile.circuit.transpiler.passes.entrypoint_validation import (
    EntrypointValidationPass,
)
from qamomile.circuit.transpiler.passes.parameter_shape_resolution import (
    ParameterShapeResolutionPass,
)
from qamomile.circuit.transpiler.passes.substitution import SubstitutionPass
from qamomile.circuit.transpiler.prepared import (
    EntrypointMode,
    PreparedModule,
    _normalize_entrypoint_mode,
    prepare_module,
)
from qamomile.circuit.transpiler.target import CompilationTarget

PlanT = TypeVar("PlanT")
ArtifactT = TypeVar("ArtifactT")


class QamomileCompiler:
    """Prepare Qamomile semantics and dispatch explicit target compilation.

    Args:
        config (CompilerConfig | None): Shared frontend and substitution
            configuration. Defaults to :class:`CompilerConfig`.
    """

    def __init__(self, config: CompilerConfig | None = None) -> None:
        """Initialize the target-neutral compiler.

        Args:
            config (CompilerConfig | None): Shared frontend configuration.
                Defaults to :class:`CompilerConfig`.
        """
        self.config = config or CompilerConfig()

    def to_block(
        self,
        kernel: QKernelLike,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> Block:
        """Trace a qkernel-like object into a hierarchical semantic block.

        Args:
            kernel (QKernelLike): Frontend object to trace.
            bindings (dict[str, Any] | None): Compile-time argument values.
                Defaults to ``None``.
            parameters (list[str] | None): Argument names retained as runtime
                parameters. Defaults to ``None``.

        Returns:
            Block: Hierarchical Qamomile semantic block.

        Raises:
            ValueError: If ``bindings`` and ``parameters`` overlap or frontend
                argument construction fails.
        """
        validate_bindings_parameters_disjoint(bindings, parameters)
        if bindings or parameters:
            traced = kernel.build(parameters=parameters, **(bindings or {}))
            return Block(
                name=traced.name,
                label_args=traced.label_args,
                input_values=traced.input_values,
                output_values=traced.output_values,
                output_names=traced.output_names,
                operations=traced.operations,
                kind=BlockKind.HIERARCHICAL,
                parameters=traced.parameters,
                param_slots=traced.param_slots,
            )
        return kernel.block

    def prepare(
        self,
        kernel: QKernelLike,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> PreparedModule:
        """Prepare a hierarchical semantic module without destroying calls.

        Args:
            kernel (QKernelLike): Top-level qkernel-like entrypoint.
            bindings (dict[str, Any] | None): Compile-time bindings used for
                tracing and shape resolution. Defaults to ``None``.
            parameters (list[str] | None): Runtime parameter names. Defaults
                to ``None``.

        Returns:
            PreparedModule: Program-level semantic input for target planning.

        Raises:
            ValueError: If bindings overlap runtime parameters.
            EntrypointValidationError: If the top-level qkernel has quantum
                inputs or outputs.
        """
        block = self.to_block(kernel, bindings, parameters)
        return self.prepare_block(
            block,
            bindings,
            mode=EntrypointMode.PROGRAM,
        )

    def prepare_block(
        self,
        block: Block,
        bindings: dict[str, Any] | None = None,
        *,
        mode: EntrypointMode | str = EntrypointMode.PROGRAM,
    ) -> PreparedModule:
        """Prepare an already-traced semantic block for target planning.

        This entrypoint lets non-execution consumers, such as circuit
        visualization, supply a trace whose quantum parameters represent
        external wires. Program mode retains the existing classical-only
        top-level validation; circuit-fragment mode skips only that ABI check.
        Preparation applies only transforms valid at the supplied block stage.

        Args:
            block (Block): Semantic entrypoint at the ``TRACED``,
                ``HIERARCHICAL``, ``AFFINE``, or ``ANALYZED`` stage.
            bindings (dict[str, Any] | None): Compile-time bindings used for
                parameter-shape resolution and retained by the prepared
                module. Defaults to ``None``.
            mode (EntrypointMode | str): Public entrypoint contract. Defaults
                to :attr:`EntrypointMode.PROGRAM`. String enum values are
                accepted.

        Returns:
            PreparedModule: Prepared semantic module preserving callable
            boundaries and the selected entrypoint mode.

        Raises:
            EntrypointValidationError: If ``mode`` is ``PROGRAM`` and the
                block has quantum inputs or outputs.
            ValueError: If ``mode`` or the block kind is unsupported, or if
                substitutions are configured after the hierarchical stage.
        """
        normalized_mode = _normalize_entrypoint_mode(mode)
        if block.kind is BlockKind.TRACED:
            block = dataclasses.replace(block, kind=BlockKind.HIERARCHICAL)
        if normalized_mode is EntrypointMode.PROGRAM:
            EntrypointValidationPass().run(block)
        if block.kind is BlockKind.HIERARCHICAL:
            if self.config.substitutions.rules:
                block = SubstitutionPass(self.config.substitutions).run(block)
            block = ParameterShapeResolutionPass(bindings).run(block)
        elif block.kind not in (BlockKind.AFFINE, BlockKind.ANALYZED):
            raise ValueError(f"Cannot prepare block kind {block.kind}")
        elif self.config.substitutions.rules:
            raise ValueError(
                "Substitutions must be applied before a block reaches "
                f"the {block.kind.name} stage"
            )
        return prepare_module(block, bindings, mode=normalized_mode)

    def compile(
        self,
        kernel: QKernelLike,
        target: CompilationTarget[PlanT, ArtifactT],
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> CompiledProgram[ArtifactT]:
        """Compile a qkernel with an explicit target implementation.

        Args:
            kernel (QKernelLike): Top-level qkernel-like entrypoint.
            target (CompilationTarget[PlanT, ArtifactT]): Target planner,
                lowerer, materializer, and validator.
            bindings (dict[str, Any] | None): Compile-time bindings. Defaults
                to ``None``.
            parameters (list[str] | None): Runtime parameter names. Defaults
                to ``None``.

        Returns:
            CompiledProgram[ArtifactT]: Validated target-native artifact.

        Raises:
            Exception: If semantic preparation, target compilation, or
                target-native validation fails.
        """
        program = self.prepare(kernel, bindings, parameters)
        return self.compile_prepared(program, target)

    def compile_prepared(
        self,
        program: PreparedModule,
        target: CompilationTarget[PlanT, ArtifactT],
    ) -> CompiledProgram[ArtifactT]:
        """Compile a prepared module through an explicit target pipeline.

        Args:
            program (PreparedModule): Prepared semantics to snapshot for the
                target.
            target (CompilationTarget[PlanT, ArtifactT]): Target planner,
                compiler, and validator.

        Returns:
            CompiledProgram[ArtifactT]: Validated target-native artifact.

        Raises:
            Exception: If target planning, compilation, or validation fails.
        """
        owned_program = program.owned_snapshot()
        plan = target.plan(owned_program)
        compiled = target.compile(owned_program, plan)
        target.validate(compiled.artifact)
        return compiled
