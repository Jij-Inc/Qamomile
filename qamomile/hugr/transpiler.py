"""Public Qamomile-to-HUGR transpiler facade."""

from __future__ import annotations

from typing import Any

from qamomile.circuit.transpiler import (
    CompiledProgram,
    CompilerConfig,
    QamomileCompiler,
    QKernelLike,
)
from qamomile.hugr._nexus import NexusExecutionOptions
from qamomile.hugr.executable import HugrExecutable
from qamomile.hugr.execution import (
    HugrExecutionTarget,
    HugrExecutor,
    SeleneExecutionOptions,
)
from qamomile.hugr.lowerer import HugrTarget


class HugrTranspiler:
    """Build HUGR executables and Guppy-compatible compiled packages.

    Use ``transpile()`` for a program with ``run()`` and ``sample()`` methods,
    or ``compile()`` for a compiled package with its ABI and metadata.

    Args:
        config (CompilerConfig | None): Shared semantic preparation
            configuration. Defaults to :class:`CompilerConfig`.
    """

    def __init__(self, config: CompilerConfig | None = None) -> None:
        """Initialize the direct program-graph transpiler.

        Args:
            config (CompilerConfig | None): Semantic preparation
                configuration. Defaults to :class:`CompilerConfig`.
        """
        self.compiler = QamomileCompiler(config)
        self.target = HugrTarget()

    def compile(
        self,
        kernel: QKernelLike,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> CompiledProgram[Any]:
        """Compile a qkernel directly to a validated HUGR package.

        Return the compilation result without execution methods. Use
        ``transpile()`` to prepare an executable, including runtime array
        shapes and expectation-value measurement programs.

        Args:
            kernel (QKernelLike): Top-level qkernel-like entrypoint.
            bindings (dict[str, Any] | None): Compile-time bindings. Defaults
                to ``None``.
            parameters (list[str] | None): Runtime parameter names retained as
                HUGR function inputs. Defaults to ``None``.

        Returns:
            CompiledProgram[Any]: Validated ``hugr.package.Package`` artifact.

        Raises:
            ImportError: If HUGR dependencies are unavailable.
            ValueError: If compile-time bindings overlap runtime parameters.
            QamomileCompileError: If semantic preparation or HUGR lowering
                rejects the program.
            HugrCliError: If target-native validation rejects the package.
        """
        return self.compiler.compile(
            kernel,
            self.target,
            bindings=bindings,
            parameters=parameters,
        )

    def to_hugr(
        self,
        kernel: QKernelLike,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> Any:
        """Return only the validated HUGR package artifact.

        Args:
            kernel (QKernelLike): Top-level qkernel-like entrypoint.
            bindings (dict[str, Any] | None): Compile-time bindings. Defaults
                to ``None``.
            parameters (list[str] | None): Runtime parameter names. Defaults
                to ``None``.

        Returns:
            Any: ``hugr.package.Package`` artifact.

        Raises:
            ImportError: If HUGR dependencies are unavailable.
            ValueError: If compile-time bindings overlap runtime parameters.
            QamomileCompileError: If semantic preparation or HUGR lowering
                rejects the program.
            HugrCliError: If target-native validation rejects the package.
        """
        return self.compile(kernel, bindings, parameters).artifact

    def transpile(
        self,
        kernel: QKernelLike,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
        *,
        parameter_shapes: dict[str, tuple[int, ...]] | None = None,
    ) -> HugrExecutable:
        """Transpile a qkernel into an executable for Selene and Nexus Helios.

        Args:
            kernel (QKernelLike): Public qkernel entrypoint.
            bindings (dict[str, Any] | None): Compile-time structural values.
                Defaults to ``None``.
            parameters (list[str] | None): Names retained as runtime arguments.
                Defaults to ``None``.
            parameter_shapes (dict[str, tuple[int, ...]] | None): Static shapes
                of runtime arrays. Shape values define structure; array contents
                remain function arguments and are supplied to run or sample.
                Defaults to ``None``.

        Returns:
            HugrExecutable: Native packages with ``run()`` and ``sample()``
                methods accepting a HUGR executor and runtime bindings.

        Raises:
            ValueError: If compile-time bindings overlap runtime parameters.
            QamomileCompileError: If semantics or target capabilities are invalid.
            Exception: If native HUGR validation fails.
        """
        from qamomile.circuit.transpiler import CompilationMetadata
        from qamomile.hugr._expval import prepare_expval
        from qamomile.hugr._shapes import resolve_parameter_shapes

        program = self.compiler.prepare(kernel, bindings, parameters).owned_snapshot()
        program = resolve_parameter_shapes(program, parameter_shapes)
        expectation = prepare_expval(program)
        if expectation is None:
            compiled = self.target.compile(program, self.target.plan(program))
            self.target.validate(compiled.artifact)
            return HugrExecutable(compiled)
        terms = []
        for circuit in expectation.circuits:
            compiled = self.target.compile(circuit, self.target.plan(circuit))
            self.target.validate(compiled.artifact)
            terms.append(compiled)
        compiled = CompiledProgram(
            artifact=tuple(term.artifact for term in terms),
            abi=program.abi,
            metadata=CompilationMetadata(
                target="hugr",
                pipeline="program_graph",
                properties={
                    "estimation": "shots",
                    "runtime_inputs": tuple(
                        name
                        for name in program.abi.public_inputs
                        if name not in program.bindings
                    ),
                },
            ),
        )
        return HugrExecutable._for_expectation(compiled, expectation, tuple(terms))

    def executor(
        self,
        target: str | HugrExecutionTarget = HugrExecutionTarget.SELENE,
        *,
        options: SeleneExecutionOptions | NexusExecutionOptions | None = None,
    ) -> HugrExecutor:
        """Create an executor for a selected HUGR destination.

        Args:
            target (str | HugrExecutionTarget): Selene or Helios destination.
            options (SeleneExecutionOptions | NexusExecutionOptions | None):
                Destination-specific execution options.

        Returns:
            HugrExecutor: Destination-independent public executor.

        Raises:
            ValueError: If target is unknown.
            TypeError: If options do not match the destination.
        """
        return HugrExecutor(target, options=options)
