"""User-facing QKernel convenience method mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from qamomile.circuit.frontend.qkernel_build import build_qkernel
from qamomile.circuit.frontend.qkernel_metadata import (
    estimate_qkernel_resources,
)
from qamomile.circuit.frontend.qkernel_visualization import (
    build_graph_for_visualization,
    build_graph_with_qubit_arrays,
    draw_qkernel,
    has_qubit_array_params,
)
from qamomile.circuit.ir.block import Block

if TYPE_CHECKING:
    from qamomile.circuit.estimator.resource_estimator import ResourceEstimate
    from qamomile.circuit.frontend.qkernel import QKernel


class QKernelBuildMixin:
    """Provide build and resource-estimation helpers for QKernel."""

    def estimate_resources(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        strategies: dict[str, str] | None = None,
        trace: bool = False,
        unknown_policy: Any = None,
        basis: Any = None,
        precision: float = 1e-10,
    ) -> ResourceEstimate:
        """Estimate all resources for this kernel's circuit.

        Convenience wrapper around ``ResourceEstimator().estimate(...)``.

        Args:
            inputs (dict[str, Any] | None): QKernel input values used to
                specialize the symbolic estimate without constructing a
                problem-sized circuit. Defaults to ``None``.
            strategies (dict[str, str] | None): Callable strategy overrides.
                Defaults to ``None``.
            trace (bool): Whether to retain the explanation tree. Defaults to
                ``False``.
            unknown_policy (Any): Optional ``UnknownResourcePolicy`` override.
                Defaults to ``None``.
            basis (Any): Optional ``GateBasis`` override. Defaults to logical
                gates when ``None``.
            precision (float): Rotation-synthesis precision for a lowered
                basis. Defaults to ``1e-10``.

        Returns:
            ResourceEstimate: Logical symbolic resource estimate.

        Example:
            >>> @qm.qkernel
            ... def bell() -> qm.Vector[qm.Qubit]:
            ...     q = qm.qubit_array(2)
            ...     q[0] = qm.h(q[0])
            ...     q[0], q[1] = qm.cx(q[0], q[1])
            ...     return q
            >>> est = bell.estimate_resources()
            >>> print(est.qubits)  # 2
        """
        kernel = cast("QKernel[..., Any]", self)
        return estimate_qkernel_resources(
            kernel,
            inputs=inputs,
            strategies=strategies,
            trace=trace,
            unknown_policy=unknown_policy,
            basis=basis,
            precision=precision,
        )

    def build(
        self,
        parameters: list[str] | None = None,
        **kwargs: Any,
    ) -> Block:
        """Build a traced Block by tracing this kernel.

        Args:
            parameters (list[str] | None): List of argument names to keep as
                unbound parameters. ``None`` auto-detects required non-quantum
                arguments without values/defaults. ``[]`` requires every
                non-quantum argument to have a value/default. Defaults to
                ``None``.
            **kwargs (Any): Concrete values for non-parameter arguments.

        Returns:
            Block: The traced block ready for transpilation, estimation,
            or visualization.

        Raises:
            TypeError: If a non-parameterizable type is specified as a
                parameter.
            ValueError: If required arguments are missing, or if a name appears
                in both ``parameters`` and ``kwargs``, violating the
                bindings/parameters disjointness rule.

        Example:
            >>> @qm.qkernel
            ... def circuit(q: qm.Qubit, theta: float) -> qm.Qubit:
            ...     q = qm.rx(q, theta)
            ...     return q
            >>> block = circuit.build(parameters=["theta"])
        """
        return build_qkernel(self, parameters=parameters, **kwargs)


class QKernelVisualizationMixin:
    """Provide visualization helpers for QKernel."""

    def _has_qubit_array_params(self) -> bool:
        """Check if kernel has any Qubit array parameters.

        Returns:
            bool: ``True`` when at least one parameter is a quantum array.
        """
        return has_qubit_array_params(self)

    def _build_graph_for_visualization(self, **kwargs: Any) -> Block:
        """Build a traced Block suitable for visualization.

        Args:
            **kwargs (Any): Concrete values for kernel arguments. For
                ``Vector[Qubit]`` parameters, pass an integer size.

        Returns:
            Block: Traced block with visualization metadata.
        """
        return build_graph_for_visualization(self, **kwargs)

    def _build_graph_with_qubit_arrays(self, kwargs: dict[str, Any]) -> Block:
        """Build traced block with Vector[Qubit] support for visualization.

        Args:
            kwargs (dict[str, Any]): Concrete values for kernel arguments.
                Integer values for ``Vector[Qubit]`` parameters are interpreted
                as register sizes.

        Returns:
            Block: Traced block with quantum-array parameters realized as
            concrete 1-D registers.

        Raises:
            NotImplementedError: If the kernel declares a rank>1 quantum array
                parameter.
            ValueError: If a ``Vector[Qubit]`` parameter is missing its integer
                size in ``kwargs``.
        """
        return build_graph_with_qubit_arrays(self, kwargs)

    def draw(
        self,
        inline: bool = False,
        fold_loops: bool = True,
        expand_composite: bool = False,
        inline_depth: int | None = None,
        fold_ifs: bool = False,
        fold_whiles: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Visualize the circuit using Matplotlib.

        Drawing uses the same verified target-neutral ``CircuitProgram``
        lowering as transpilation. Scalar ``Qubit`` and sized
        ``Vector[Qubit]`` inputs remain external wires, and quantum outputs are
        preserved. Structural quantum indices, slice bounds, register sizes,
        and loop ranges must be concrete rather than approximated.

        Args:
            inline (bool): If ``True``, expand retained source qkernel regions
                and safe direct reusable-circuit calls. If ``False``, show
                those boundaries as boxes. Defaults to ``False``.
            fold_loops (bool): If ``True``, display ``ForOperation`` as folded
                blocks instead of unrolling. Defaults to ``True``.
            expand_composite (bool): Compatibility alias for ``inline``.
                Transformed or opaque calls remain boxed. Defaults to
                ``False``.
            inline_depth (int | None): Maximum nested source/reusable-call
                expansion depth. ``None`` means unlimited. Defaults to
                ``None``.
            fold_ifs (bool): If ``True``, display ``IfOperation`` as folded
                summary blocks. Defaults to ``False``.
            fold_whiles (bool): If ``True``, display ``WhileOperation`` as a
                folded summary block. If ``False``, show its body inside a
                ``while <condition>:`` region. Defaults to ``False``.
            **kwargs (Any): Concrete values for arguments. Arguments not
                provided here and without defaults are shown as symbolic
                parameters.

        Returns:
            Any: Matplotlib figure object.

        Raises:
            CircuitDrawingError: If the qkernel cannot lower to one exact,
                verified ``CircuitProgram``.
            ImportError: If matplotlib is not installed.
            TypeError: If a draw-time binding has an invalid frontend type.
            ValueError: If a ``Vector[Qubit]`` parameter requires a concrete
                size for visualization and no size is provided, or if another
                structural value cannot be resolved to an exact circuit.
        """
        return draw_qkernel(
            self,
            inline=inline,
            fold_loops=fold_loops,
            fold_ifs=fold_ifs,
            fold_whiles=fold_whiles,
            expand_composite=expand_composite,
            inline_depth=inline_depth,
            **kwargs,
        )
