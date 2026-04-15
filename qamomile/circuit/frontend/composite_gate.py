"""Frontend interface for composite gates."""

from __future__ import annotations

import abc
import dataclasses
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Sequence, overload

from qamomile.circuit.frontend.handle.primitives import Qubit
from qamomile.circuit.frontend.tracer import Tracer, get_current_tracer, trace
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    CompositeGateType,
    ResourceMetadata,
)
from qamomile.circuit.ir.value import Value

if TYPE_CHECKING:
    from qamomile.circuit.frontend.decomposition import DecompositionStrategy
    from qamomile.circuit.frontend.handle.array import Vector


class CompositeGate(abc.ABC):
    """Base class for user-facing composite gate definitions.

    Subclasses can define composite gates in two ways:

    1. **Using _decompose() (recommended for users)**:
       Define the gate decomposition using frontend syntax (same as @qkernel).

       ```python
       class QFT(CompositeGate):
           def __init__(self, num_qubits: int):
               self._num_qubits = num_qubits

           @property
           def num_target_qubits(self) -> int:
               return self._num_qubits

           def _decompose(self, qubits: Vector[Qubit]) -> Vector[Qubit]:
               # Use frontend syntax: qm.h(), qm.cp(), qm.range(), etc.
               n = self._num_qubits
               for j in qmc.range(n - 1, -1, -1):
                   qubits[j] = qmc.h(qubits[j])
                   for k in qmc.range(j - 1, -1, -1):
                       angle = math.pi / (2 ** (j - k))
                       qubits[j], qubits[k] = qmc.cp(qubits[j], qubits[k], angle)
               return qubits

           def _resources(self) -> ResourceMetadata:
               return ResourceMetadata(t_gates=0)
       ```

    2. **Using get_implementation() (advanced)**:
       Return a pre-built Block directly.

    Example usage:
        ```python
        # Factory function pattern
        def qft(qubits: Vector[Qubit]) -> Vector[Qubit]:
            n = _get_size(qubits)
            return QFT(n)(qubits)

        # Direct class usage
        result = QFT(3)(*qubit_list)
        ```
    """

    gate_type: CompositeGateType = CompositeGateType.CUSTOM
    custom_name: str = ""

    # Strategy registry: maps strategy name to DecompositionStrategy
    _strategies: ClassVar[dict[str, "DecompositionStrategy"]] = {}
    _default_strategy: ClassVar[str] = "standard"

    @classmethod
    def register_strategy(
        cls,
        name: str,
        strategy: "DecompositionStrategy",
    ) -> None:
        """Register a decomposition strategy for this gate type.

        Args:
            name: Strategy identifier (e.g., "standard", "approximate")
            strategy: DecompositionStrategy instance

        Example:
            QFT.register_strategy("approximate", ApproximateQFTStrategy(k=3))
        """
        # Create a new dict if inheriting from parent class
        if "_strategies" not in cls.__dict__:
            cls._strategies = {}
        cls._strategies[name] = strategy

    @classmethod
    def get_strategy(
        cls,
        name: str | None = None,
    ) -> "DecompositionStrategy | None":
        """Get a registered decomposition strategy.

        Args:
            name: Strategy name, or None for default strategy

        Returns:
            DecompositionStrategy instance, or None if not found
        """
        target_name = name or cls._default_strategy
        return cls._strategies.get(target_name)

    @classmethod
    def list_strategies(cls) -> list[str]:
        """List all registered strategy names.

        Returns:
            List of strategy names
        """
        return list(cls._strategies.keys())

    @classmethod
    def set_default_strategy(cls, name: str) -> None:
        """Set the default decomposition strategy.

        Args:
            name: Strategy name to use as default

        Raises:
            ValueError: If strategy is not registered
        """
        if name not in cls._strategies:
            raise ValueError(
                f"Strategy '{name}' not registered. "
                f"Available: {list(cls._strategies.keys())}"
            )
        cls._default_strategy = name

    @property
    @abc.abstractmethod
    def num_target_qubits(self) -> int:
        """Number of target qubits this gate operates on."""
        pass

    @property
    def num_control_qubits(self) -> int:
        """Number of control qubits (default: 0)."""
        return 0

    def _decompose(
        self,
        qubits: "Vector[Qubit] | tuple[Qubit, ...]",
    ) -> "Vector[Qubit] | tuple[Qubit, ...] | None":
        """Define the gate decomposition using frontend syntax.

        Override this method to provide a decomposition. The decomposition
        is traced and converted to a Block automatically.

        Args:
            qubits: Input qubits as Vector[Qubit] or tuple of Qubits

        Returns:
            Output qubits (same type as input), or None if no decomposition.

        The method body can use any frontend operations:
        - qmc.h(), qmc.x(), qmc.cx(), etc. for gates
        - qmc.cp(), qmc.rz(), etc. for parameterized gates
        - qmc.range() for loops
        - Standard Python control flow

        Example:
            def _decompose(self, qubits: Vector[Qubit]) -> Vector[Qubit]:
                n = self._num_qubits
                for j in qmc.range(n):
                    qubits[j] = qmc.h(qubits[j])
                return qubits
        """
        return None

    def _resources(self) -> ResourceMetadata | None:
        """Return resource estimation metadata.

        Override to provide resource hints for the composite gate.
        This is an alternative to get_resource_metadata() with a simpler name.

        Returns:
            ResourceMetadata with query_complexity, t_gates, etc.

        Example:
            def _resources(self) -> ResourceMetadata:
                n = self._num_qubits
                return ResourceMetadata(
                    t_gates=0,
                    total_gates=n,
                )
        """
        return None

    def _decompose_with_strategy(
        self,
        qubits: "tuple[Qubit, ...] | Vector[Qubit]",
        strategy_name: str | None = None,
    ) -> "tuple[Qubit, ...] | Vector[Qubit] | None":
        """Decompose using a specific strategy.

        Args:
            qubits: Input qubits
            strategy_name: Strategy to use, or None for default

        Returns:
            Output qubits, or None if no strategy/decomposition available
        """
        strategy = self.get_strategy(strategy_name)
        if strategy is not None:
            return strategy.decompose(qubits)  # type: ignore
        # Fall back to _decompose if no strategy registered
        return self._decompose(qubits)

    def get_resources_for_strategy(
        self,
        strategy_name: str | None = None,
    ) -> ResourceMetadata | None:
        """Get resource metadata for a specific strategy.

        Args:
            strategy_name: Strategy to query, or None for default

        Returns:
            ResourceMetadata for the strategy, or None if not available
        """
        strategy = self.get_strategy(strategy_name)
        if strategy is not None:
            return strategy.resources(self.num_target_qubits)
        return self._resources()

    def get_implementation(self) -> Block | None:
        """Get the implementation Block, if any.

        Return None for stub gates (used in resource estimation).
        Override in subclasses to provide implementation.

        Note: If _decompose() is defined, it takes precedence over this method.
        """
        return None

    def get_resource_metadata(self) -> ResourceMetadata | None:
        """Get resource estimation metadata.

        Returns _resources() if defined, otherwise None.
        Override _resources() to provide resource hints.
        """
        return self._resources()

    def build_decomposition(
        self,
        *qubits: Qubit,
        **params: Any,
    ) -> Block | None:
        """Build the decomposition circuit dynamically.

        Override this method to provide a decomposition that depends on
        runtime arguments (e.g., QPE needs the unitary Block).

        This method is called by InlinePass when inlining composite gates
        that have dynamic implementations.

        Args:
            *qubits: The qubits passed to the gate
            **params: Additional parameters (e.g., unitary for QPE)

        Returns:
            Block containing the decomposition, or None if not available.

        Example:
            class QPE(CompositeGate):
                def build_decomposition(self, *qubits, **params):
                    unitary = params.get("unitary")
                    # Build QPE circuit using the unitary
                    return self._build_qpe_impl(qubits, unitary)
        """
        # Default: return static implementation if available
        return self.get_implementation()

    def _build_decomposition_block(
        self,
        target_qubits: "tuple[Qubit, ...] | Vector[Qubit]",
        strategy_name: str | None = None,
    ) -> Block | None:
        """Build a Block by tracing _decompose() or a strategy.

        This method creates a Block from the decomposition implementation
        by running it in a tracing context. If a strategy is specified and
        registered, it uses the strategy's decompose method.

        Args:
            target_qubits: The target qubits to decompose
            strategy_name: Optional strategy name to use for decomposition

        Returns:
            Block containing the traced operations, or None
        """
        from qamomile.circuit.frontend.handle.array import Vector
        from qamomile.circuit.ir.types.primitives import QubitType
        from qamomile.circuit.ir.value import Value

        # Determine which decomposition method to use
        strategy = self.get_strategy(strategy_name) if strategy_name else None
        has_strategy = strategy is not None
        has_decompose = self._decompose.__func__ is not CompositeGate._decompose  # type: ignore

        if not has_strategy and not has_decompose:
            return None

        # Create a fresh tracer for decomposition
        decomp_tracer = Tracer()

        # Determine if input is a Vector
        is_vector_input = isinstance(target_qubits, Vector)

        # Create fresh qubit handles for decomposition WITHOUT QInitOperation
        # These are parameter qubits passed to the gate, not created inside it
        input_values: list[Value] = []

        if is_vector_input:
            # For Vector input, we don't support Vector-based _decompose yet
            # Fall back to tuple-based decomposition
            n = self.num_target_qubits
            fresh_qubits = []
            for i in range(n):
                # Create Value directly without QInitOperation
                q_value = Value(
                    type=QubitType(),
                    name=f"_decomp_q{i}",
                )
                # Create Qubit handle pointing to this value
                q = Qubit(value=q_value)
                fresh_qubits.append(q)
                input_values.append(q_value)

            with trace(decomp_tracer):
                if has_strategy:
                    result = strategy.decompose(tuple(fresh_qubits))  # type: ignore
                else:
                    result = self._decompose(tuple(fresh_qubits))
        else:
            # For tuple input, create fresh Qubit handles
            fresh_qubits = []
            for i in range(len(target_qubits)):  # type: ignore
                # Create Value directly without QInitOperation
                q_value = Value(
                    type=QubitType(),
                    name=f"_decomp_q{i}",
                )
                # Create Qubit handle pointing to this value
                q = Qubit(value=q_value)
                fresh_qubits.append(q)
                input_values.append(q_value)

            with trace(decomp_tracer):
                if has_strategy:
                    result = strategy.decompose(tuple(fresh_qubits))  # type: ignore
                else:
                    result = self._decompose(tuple(fresh_qubits))

        if result is None:
            return None

        # Collect return values
        return_values: list[Value]
        if isinstance(result, Vector):
            return_values = [result.value]
        elif isinstance(result, tuple):
            return_values = [q.value for q in result]
        else:
            return_values = [result.value]

        # Create Block from traced operations (no QInitOperations)
        block = Block(
            operations=decomp_tracer.operations,
            input_values=input_values,
            output_values=return_values,
            name=self.custom_name or self.gate_type.value,
            kind=BlockKind.HIERARCHICAL,
        )

        return block

    def __call__(
        self,
        *target_qubits: Qubit,
        controls: Sequence[Qubit] = (),
        strategy: str | None = None,
    ) -> tuple[Qubit, ...]:
        """Apply this composite gate to qubits.

        Args:
            *target_qubits: Target qubits for the gate
            controls: Optional control qubits
            strategy: Optional strategy name for decomposition.
                If None, uses the default strategy (or _decompose if no strategies).

        Returns:
            Tuple of output qubits (controls + targets)
        """
        if len(target_qubits) != self.num_target_qubits:
            raise ValueError(
                f"{self.custom_name or self.gate_type.value} requires "
                f"{self.num_target_qubits} target qubits, got {len(target_qubits)}"
            )

        if len(controls) != self.num_control_qubits:
            raise ValueError(
                f"{self.custom_name or self.gate_type.value} requires "
                f"{self.num_control_qubits} control qubits, got {len(controls)}"
            )

        # Try to get implementation from _decompose() first, then get_implementation()
        impl = self._build_decomposition_block(target_qubits, strategy)
        if impl is None:
            impl = self.get_implementation()
        has_impl = impl is not None

        # Consume all qubit handles (enforces affine type)
        gate_name = self.custom_name or self.gate_type.value
        consumed_controls = [
            c.consume(operation_name=f"{gate_name}[control]") for c in controls
        ]
        consumed_targets = [
            t.consume(operation_name=f"{gate_name}[target]") for t in target_qubits
        ]

        # Build operands
        operands: list[Any] = []

        for c in consumed_controls:
            operands.append(c.value)

        for t in consumed_targets:
            operands.append(t.value)

        # Build results (new versions of qubits)
        results: list[Value] = []
        for c in consumed_controls:
            results.append(c.value.next_version())
        for t in consumed_targets:
            results.append(t.value.next_version())

        # Get resource metadata for the selected strategy
        resource_meta = self.get_resources_for_strategy(strategy)

        # Create operation
        op = CompositeGateOperation(
            operands=operands,
            results=results,
            gate_type=self.gate_type,
            num_control_qubits=len(controls),
            num_target_qubits=len(target_qubits),
            custom_name=self.custom_name,
            resource_metadata=resource_meta,
            has_implementation=has_impl,
            implementation_block=impl,
            composite_gate_instance=self,  # Store reference for emit pass
            strategy_name=strategy,  # Pass strategy name to operation
        )

        # Emit to tracer
        tracer = get_current_tracer()
        tracer.add_operation(op)

        # Return output handles
        output_qubits: list[Qubit] = []
        for i, c in enumerate(consumed_controls):
            output_qubits.append(
                Qubit(
                    value=results[i],
                    parent=c.parent,
                    indices=c.indices,
                )
            )

        for i, t in enumerate(consumed_targets):
            output_qubits.append(
                Qubit(
                    value=results[len(consumed_controls) + i],
                    parent=t.parent,
                    indices=t.indices,
                )
            )

        return tuple(output_qubits)


@dataclasses.dataclass
class _WrappedCompositeGate(CompositeGate):
    """Internal class for composite gates created from qkernel."""

    _gate_type: CompositeGateType = CompositeGateType.CUSTOM
    _custom_name: str = ""
    _num_targets: int = 0
    _num_controls: int = 0
    _qkernel: Any = None  # QKernel instance
    _resource_metadata: ResourceMetadata | None = None

    @property
    def gate_type(self) -> CompositeGateType:  # type: ignore
        return self._gate_type

    @property
    def custom_name(self) -> str:  # type: ignore
        return self._custom_name

    @property
    def num_target_qubits(self) -> int:
        return self._num_targets

    @property
    def num_control_qubits(self) -> int:
        return self._num_controls

    def get_implementation(self) -> Block | None:
        if self._qkernel is None:
            return None
        return self._qkernel.block

    def _resources(self) -> ResourceMetadata | None:
        return self._resource_metadata


@dataclasses.dataclass
class _StubCompositeGate(CompositeGate):
    """Internal class for stub composite gates (no implementation)."""

    _gate_type: CompositeGateType = CompositeGateType.CUSTOM
    _custom_name: str = ""
    _num_targets: int = 0
    _num_controls: int = 0
    _resource_metadata: ResourceMetadata | None = None

    @property
    def gate_type(self) -> CompositeGateType:  # type: ignore
        return self._gate_type

    @property
    def custom_name(self) -> str:  # type: ignore
        return self._custom_name

    @property
    def num_target_qubits(self) -> int:
        return self._num_targets

    @property
    def num_control_qubits(self) -> int:
        return self._num_controls

    def get_implementation(self) -> None:
        return None

    def _resources(self) -> ResourceMetadata | None:
        return self._resource_metadata


@overload
def composite_gate(
    func: Callable,
) -> _WrappedCompositeGate:
    """Decorator form: @composite_gate applied directly to a qkernel."""
    ...


@overload
def composite_gate(
    *,
    name: str = "",
    num_controls: int = 0,
    gate_type: CompositeGateType = CompositeGateType.CUSTOM,
) -> Callable[[Callable], _WrappedCompositeGate]:
    """Decorator form with arguments: @composite_gate(name="my_gate")."""
    ...


@overload
def composite_gate(
    *,
    stub: bool,
    name: str,
    num_qubits: int,
    num_controls: int = 0,
    resource_metadata: ResourceMetadata | None = None,
    gate_type: CompositeGateType = CompositeGateType.CUSTOM,
) -> Callable[[Callable], _StubCompositeGate]:
    """Decorator form for stub: @composite_gate(stub=True, name="oracle", num_qubits=5)."""
    ...


def composite_gate(
    func: Callable | None = None,
    *,
    stub: bool = False,
    name: str = "",
    num_qubits: int | None = None,
    num_controls: int = 0,
    resource_metadata: ResourceMetadata | None = None,
    gate_type: CompositeGateType = CompositeGateType.CUSTOM,
) -> (
    _WrappedCompositeGate
    | _StubCompositeGate
    | Callable[[Callable], _WrappedCompositeGate | _StubCompositeGate]
):
    """Decorator to create a CompositeGate from a qkernel function or as a stub.

    Usage with qkernel (implementation provided):
        @composite_gate(name="my_qft")
        @qkernel
        def my_qft(q0: Qubit, q1: Qubit) -> tuple[Qubit, Qubit]:
            q0 = h(q0)
            q0, q1 = cp(q0, q1, pi/2)
            q1 = h(q1)
            return q0, q1

        # Usage:
        q0, q1 = my_qft(q0, q1)

    Usage as stub (no implementation, for resource estimation):
        @composite_gate(
            stub=True, name="oracle", num_qubits=5,
            resource_metadata=ResourceMetadata(query_complexity=100, t_gates=10),
        )
        def oracle():
            pass

        # Usage:
        results = oracle(*qubits)
        metadata = oracle.get_resource_metadata()

    Args:
        func: The qkernel function (when used without arguments)
        stub: If True, create a stub gate with no implementation
        name: Name for the composite gate
        num_qubits: Number of target qubits (required for stub)
        num_controls: Number of control qubits (default: 0)
        resource_metadata: ResourceMetadata for resource estimation (stub mode)
        gate_type: The type of composite gate (default: CUSTOM)

    Returns:
        A CompositeGate instance that can be called like a gate function.
    """
    from qamomile.circuit.frontend.handle import Qubit
    from qamomile.circuit.frontend.qkernel import QKernel

    def decorator(
        kernel_or_func: Callable,
    ) -> _WrappedCompositeGate | _StubCompositeGate:
        if stub:
            # Stub mode - no implementation required
            if num_qubits is None:
                raise ValueError("num_qubits is required for stub composite gates")

            resource_meta = resource_metadata

            gate_name = name or getattr(kernel_or_func, "__name__", "stub")

            return _StubCompositeGate(
                _gate_type=gate_type,
                _custom_name=gate_name,
                _num_targets=num_qubits,
                _num_controls=num_controls,
                _resource_metadata=resource_meta,
            )

        # Implementation mode - requires a QKernel
        if not isinstance(kernel_or_func, QKernel):
            raise TypeError(
                "composite_gate decorator must be applied to a @qkernel function. "
                "Use @composite_gate @qkernel def func(...) or "
                "@composite_gate(stub=True, ...) for stubs."
            )

        qkernel_instance = kernel_or_func

        # Infer num_target_qubits from qkernel signature
        num_targets = sum(
            1
            for t in qkernel_instance.input_types.values()
            if t is Qubit or (hasattr(t, "__origin__") and t.__origin__ is tuple)
        )

        # Count Qubit inputs more precisely
        num_targets = 0
        for param_type in qkernel_instance.input_types.values():
            if param_type is Qubit:
                num_targets += 1
            # Handle tuple types like tuple[Qubit, Qubit]
            elif hasattr(param_type, "__origin__") and param_type.__origin__ is tuple:
                args = getattr(param_type, "__args__", ())
                num_targets += sum(1 for arg in args if arg is Qubit)

        gate_name = name or qkernel_instance.name

        return _WrappedCompositeGate(
            _gate_type=gate_type,
            _custom_name=gate_name,
            _num_targets=num_targets,
            _num_controls=num_controls,
            _qkernel=qkernel_instance,
        )

    # Handle direct decoration: @composite_gate
    if func is not None:
        return decorator(func)

    # Handle decoration with arguments: @composite_gate(name="foo")
    return decorator
