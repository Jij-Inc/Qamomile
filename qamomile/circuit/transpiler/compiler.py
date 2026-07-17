"""Target-neutral compiler entrypoint for Qamomile programs."""

from __future__ import annotations

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
from qamomile.circuit.transpiler.prepared import PreparedModule, prepare_module
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
            TypeError: If specialization is requested for a block-only
                qkernel-like object that has no ``build`` method.
        """
        validate_bindings_parameters_disjoint(bindings, parameters)
        build = getattr(kernel, "build", None)
        if not callable(build):
            if bindings or parameters:
                raise TypeError(
                    "Cannot specialize a block-only qkernel-like object; "
                    "provide an object with build(parameters=..., **bindings)."
                )
            block = getattr(kernel, "block", None)
            if not isinstance(block, Block):
                raise TypeError(
                    "Expected a qkernel-like object with build() or a Block "
                    "in its .block attribute."
                )
            return block
        traced = build(parameters=parameters, **(bindings or {}))
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
            EntrypointValidationError: If the top-level kernel has quantum
                inputs or outputs.
        """
        block = self.to_block(kernel, bindings, parameters)
        EntrypointValidationPass().run(block)
        if self.config.substitutions.rules:
            block = SubstitutionPass(self.config.substitutions).run(block)
        block = ParameterShapeResolutionPass(bindings).run(block)
        return prepare_module(block, bindings)

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
        owned_program = program.owned_snapshot()
        plan = target.plan(owned_program)
        compiled = target.compile(owned_program, plan)
        target.validate(compiled.artifact)
        return compiled
