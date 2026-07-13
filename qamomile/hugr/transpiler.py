"""Public Qamomile-to-HUGR transpiler facade."""

from __future__ import annotations

from typing import Any

from qamomile.circuit.frontend.qkernel_like import QKernelLike
from qamomile.circuit.transpiler.artifact import CompiledProgram
from qamomile.circuit.transpiler.compiler import QamomileCompiler
from qamomile.circuit.transpiler.config import CompilerConfig
from qamomile.hugr.lowerer import HugrTarget


class HugrTranspiler:
    """Compile Qamomile programs to Guppy-compatible HUGR packages.

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

    def transpile(
        self,
        kernel: QKernelLike,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> CompiledProgram[Any]:
        """Transpile a qkernel directly to a validated HUGR package.

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
        """
        return self.transpile(kernel, bindings, parameters).artifact
