"""Frontend interface for composite gates."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, ClassVar, Sequence

from qamomile.circuit.frontend.handle.primitives import Qubit
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import CompositeGateType

if TYPE_CHECKING:
    from qamomile.circuit.frontend.decomposition import DecompositionStrategy
    from qamomile.circuit.frontend.handle.array import Vector


class CompositeGate(abc.ABC):
    """Provide the advanced base class for boxed composite operations.

    User-defined named operations should normally be written with the
    :func:`composite_gate` decorator. The decorator keeps the frontend
    interface close to ordinary qkernel code while the compiler records a
    named callable with body, resource metadata, and implementation candidates.

    This base class remains public for advanced use cases: stdlib
    implementations such as QFT/IQFT, strategy registries, backend migration
    tests, and compatibility with older class-based custom gates. Subclasses
    may either trace frontend operations from :meth:`_decompose` or provide a
    pre-built :class:`~qamomile.circuit.ir.block.Block` through
    :meth:`get_implementation`.

    Example:
        ```python
        import qamomile.circuit as qmc

        @qmc.composite_gate(name="h_layer")
        def h_layer(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            for i in qmc.range(q.shape[0]):
                q[i] = qmc.h(q[i])
            return q
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
            name (str): Strategy identifier such as ``"standard"`` or
                ``"approximate"``.
            strategy (DecompositionStrategy): Decomposition strategy instance.

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
            name (str | None): Strategy name, or ``None`` for the default
                strategy.

        Returns:
            DecompositionStrategy | None: Strategy instance, or ``None`` if not
            found.
        """
        target_name = name or cls._default_strategy
        return cls._strategies.get(target_name)

    @classmethod
    def list_strategies(cls) -> list[str]:
        """List all registered strategy names.

        Returns:
            list[str]: Strategy names registered on this class.
        """
        return list(cls._strategies.keys())

    @classmethod
    def set_default_strategy(cls, name: str) -> None:
        """Set the default decomposition strategy.

        Args:
            name (str): Strategy name to use as default.

        Raises:
            ValueError: If the strategy is not registered.
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
        """Define an advanced class-based decomposition using frontend syntax.

        Override this method in stdlib or migration subclasses to provide a
        decomposition. New user-defined named operations should prefer the
        ``composite_gate`` decorator, which records the same kind of boxed
        callable without requiring a subclass.

        Args:
            qubits (Vector[Qubit] | tuple[Qubit, ...]): Input qubits as a
                vector or tuple.

        Returns:
            Vector[Qubit] | tuple[Qubit, ...] | None: Output qubits with the
            same shape as input, or ``None`` if no decomposition is available.

        Example:
            def _decompose(self, qubits: Vector[Qubit]) -> Vector[Qubit]:
                n = self._num_qubits
                for j in qmc.range(n):
                    qubits[j] = qmc.h(qubits[j])
                return qubits
        """
        return None

    def _decompose_with_strategy(
        self,
        qubits: "tuple[Qubit, ...] | Vector[Qubit]",
        strategy_name: str | None = None,
    ) -> "tuple[Qubit, ...] | Vector[Qubit] | None":
        """Decompose using a specific strategy.

        Args:
            qubits (tuple[Qubit, ...] | Vector[Qubit]): Input qubits.
            strategy_name (str | None): Strategy to use, or ``None`` for the
                default strategy.

        Returns:
            tuple[Qubit, ...] | Vector[Qubit] | None: Output qubits, or
            ``None`` if no strategy/decomposition is available.
        """
        strategy = self.get_strategy(strategy_name)
        if strategy is not None:
            return strategy.decompose(qubits)  # type: ignore
        # Fall back to _decompose if no strategy registered
        return self._decompose(qubits)

    def get_implementation(self) -> Block | None:
        """Get the implementation Block, if any.

        Override in subclasses to provide a pre-built implementation block.

        Returns:
            Block | None: Implementation block, or ``None`` when the
            composite must be built from ``_decompose`` or is unavailable.

        Note:
            If ``_decompose()`` is defined, it takes precedence over this
            method.
        """
        return None

    def build_decomposition(
        self,
        *qubits: Qubit,
        **params: Any,
    ) -> Block | None:
        """Build the decomposition circuit dynamically.

        Override this method to provide a decomposition that depends on
        runtime arguments. Backend emit fallback can use this body when no
        native implementation is selected.

        Args:
            *qubits (Qubit): Qubits passed to the gate.
            **params (Any): Additional decomposition parameters.

        Returns:
            Block | None: Decomposition block, or ``None`` if not available.

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
            target_qubits (tuple[Qubit, ...] | Vector[Qubit]): Target qubits
                used only to determine decomposition arity and shape.
            strategy_name (str | None): Optional strategy name to use for
                decomposition.

        Returns:
            Block | None: Block containing traced operations, or ``None`` if no
            decomposition path is available.
        """
        from qamomile.circuit.frontend.composite_gate_decomposition import (
            build_decomposition_block,
        )

        has_decompose = self._decompose.__func__ is not CompositeGate._decompose  # type: ignore[attr-defined]
        return build_decomposition_block(
            self,
            target_qubits,
            strategy_name=strategy_name,
            has_decompose=has_decompose,
        )

    def __call__(
        self,
        *target_qubits: Qubit,
        controls: Sequence[Qubit] = (),
        strategy: str | None = None,
    ) -> tuple[Qubit, ...]:
        """Apply this composite gate to qubits.

        Args:
            *target_qubits (Qubit): Target qubits for the gate.
            controls (Sequence[Qubit]): Optional control qubits.
            strategy (str | None): Optional strategy name for decomposition.
                If None, uses the default strategy (or _decompose if no strategies).

        Returns:
            tuple[Qubit, ...]: Output qubits, with controls first followed by
            targets.

        Raises:
            ValueError: If the supplied arity is invalid or the composite has
            no implementation body.
        """
        from qamomile.circuit.frontend.composite_gate_invocation import (
            invoke_composite_gate,
        )

        return invoke_composite_gate(
            self,
            *target_qubits,
            controls=controls,
            strategy=strategy,
        )


from qamomile.circuit.frontend.composite_gate_wrapped import (  # noqa: E402
    _WrappedCompositeGate as _WrappedCompositeGate,
    composite as composite,
    composite_gate as composite_gate,
)
