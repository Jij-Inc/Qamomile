"""Structural protocol for qkernel-like frontend objects."""

from __future__ import annotations

import inspect
from typing import Any, Protocol

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.effect import KernelEffect


class QKernelLike(Protocol):
    """Describe the frontend surface required by compiler entrypoints.

    This protocol is intentionally structural. It lets decorator-created
    composites reuse the qkernel inspection and build interface without making
    them inherit from ``QKernel`` or exposing the compiler-facing callable
    descriptor model as a frontend concept.

    Attributes:
        name (str): User-facing callable name.
        signature (inspect.Signature): Python call signature.
        input_types (dict[str, Any]): Frontend input type annotations by name.
        output_types (list[Any]): Frontend output type annotations.
        block (Block): Cached hierarchical body block.
        effects (KernelEffect): Cached semantic effect set.
    """

    @property
    def name(self) -> str:
        """Return the user-facing callable name.

        Returns:
            str: Callable name.
        """
        ...

    @property
    def signature(self) -> inspect.Signature:
        """Return the Python call signature.

        Returns:
            inspect.Signature: Signature used to bind frontend arguments.
        """
        ...

    @property
    def input_types(self) -> dict[str, Any]:
        """Return frontend input annotations by parameter name.

        Returns:
            dict[str, Any]: Mapping from input names to frontend handle types.
        """
        ...

    @property
    def output_types(self) -> list[Any]:
        """Return frontend output annotations.

        Returns:
            list[Any]: Frontend return handle types.
        """
        ...

    @property
    def block(self) -> Block:
        """Return the cached hierarchical body block.

        Returns:
            Block: Body block for compiler passes.
        """
        ...

    @property
    def effects(self) -> KernelEffect:
        """Return cached semantic effects of the qkernel body.

        Returns:
            KernelEffect: Aggregated non-unitary effects.
        """
        ...

    def build(
        self,
        parameters: list[str] | None = None,
        **kwargs: Any,
    ) -> Block:
        """Build a traced body block.

        Args:
            parameters (list[str] | None): Runtime parameter names to
                preserve. Defaults to ``None``.
            **kwargs (Any): Compile-time bindings for non-parameter
                arguments.

        Returns:
            Block: Traced hierarchical body block.
        """
        ...
