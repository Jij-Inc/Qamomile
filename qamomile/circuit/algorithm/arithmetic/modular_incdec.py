"""Modular increment and decrement operators for basis-state registers."""

from __future__ import annotations

from typing import Any

import qamomile.circuit as qmc


def _is_zero_length_register(q: qmc.Vector[qmc.Qubit]) -> bool:
    """Return whether ``q`` has a concrete zero length.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register handle to inspect.

    Returns:
        bool: ``True`` when the register shape is concretely known to be
        zero; otherwise ``False``.
    """
    if not q.shape:
        return False
    dim = q.shape[0]
    if isinstance(dim, int):
        return dim == 0
    value = getattr(dim, "value", None)
    return bool(
        value is not None and value.is_constant() and int(value.get_const()) == 0
    )


class _CheckedModularQKernel:
    """Reject empty registers before delegating to a modular qkernel.

    Args:
        kernel (Any): Underlying ``QKernel`` object that implements the
            non-empty modular arithmetic primitive.
        op_name (str): User-facing operation name used in error messages
            and callable metadata.
    """

    def __init__(self, kernel: Any, op_name: str) -> None:
        """Initialize the checked qkernel wrapper.

        Args:
            kernel (Any): Underlying qkernel to delegate to.
            op_name (str): User-facing operation name.
        """
        self._kernel = kernel
        self._op_name = op_name
        self.name = op_name
        if hasattr(kernel, "name"):
            kernel.name = op_name
        self.__name__ = op_name
        self.__doc__ = getattr(kernel, "__doc__", None)

    def __getattr__(self, name: str) -> Any:
        """Forward qkernel attributes to the wrapped kernel.

        Args:
            name (str): Attribute name to retrieve.

        Returns:
            Any: Attribute value from the wrapped qkernel.

        Raises:
            AttributeError: If the wrapped qkernel does not expose
                ``name``.
        """
        return getattr(self._kernel, name)

    def __call__(self, q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply the wrapped primitive after validating ``q``.

        Args:
            q (qmc.Vector[qmc.Qubit]): Qubit register to update.

        Returns:
            qmc.Vector[qmc.Qubit]: Updated qubit register.

        Raises:
            ValueError: If ``q`` is a concrete zero-length register.
        """
        if _is_zero_length_register(q):
            raise ValueError(
                f"{self._op_name} requires at least one qubit; "
                f"got a zero-length register."
            )
        return self._kernel(q)


@qmc.qkernel
def _modular_increment(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Apply the modular increment ``|j> -> |j + 1 mod 2^n>``.

    The qubit vector is interpreted in little-endian order: ``q[0]`` is
    the least-significant bit of the encoded integer.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register to increment in place.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register.
    """
    n = q.shape[0]
    for k in qmc.range(1, n):
        target_index = n - k
        mcx = qmc.control(qmc.x, num_controls=target_index)
        q[0:target_index], q[target_index] = mcx(q[0:target_index], q[target_index])
    q[0] = qmc.x(q[0])
    return q


@qmc.qkernel
def _modular_decrement(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Apply the modular decrement ``|j> -> |j - 1 mod 2^n>``.

    The qubit vector is interpreted in little-endian order: ``q[0]`` is
    the least-significant bit of the encoded integer.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register to decrement in place.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register.
    """
    n = q.shape[0]
    q[0] = qmc.x(q[0])
    for target_index in qmc.range(1, n):
        mcx = qmc.control(qmc.x, num_controls=target_index)
        q[0:target_index], q[target_index] = mcx(q[0:target_index], q[target_index])
    return q


def _apply_fixed_window_periodic_shift(
    system: qmc.Vector[qmc.Qubit],
    start: int,
    width: int,
    shift: int,
) -> qmc.Vector[qmc.Qubit]:
    """Apply a constant modular shift to one fixed window of a flat register.

    The helper emits the same little-endian increment/decrement ladders as the
    public modular primitives, but addresses the parent vector directly. This
    avoids constructing a nested symbolic slice while a concrete axis view is
    borrowed by a multidimensional caller.

    Args:
        system (qmc.Vector[qmc.Qubit]): Flattened parent register to update.
        start (int): First qubit index of the target window.
        width (int): Positive number of qubits in the target window.
        shift (int): Signed number of unit modular shifts. Positive values
            increment and negative values decrement.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated flattened parent register.
    """
    increment = shift > 0
    for _ in range(abs(shift)):
        if not increment:
            system[start] = qmc.x(system[start])

        target_indices = range(width - 1, 0, -1) if increment else range(1, width)
        for target_index in target_indices:
            controls = system[start : start + target_index]
            target = system[start + target_index]
            controls, target = qmc.mcx(controls, target)
            system[start : start + target_index] = controls
            system[start + target_index] = target

        if increment:
            system[start] = qmc.x(system[start])
    return system


modular_increment: Any = _CheckedModularQKernel(
    _modular_increment,
    "modular_increment",
)
modular_decrement: Any = _CheckedModularQKernel(
    _modular_decrement,
    "modular_decrement",
)


__all__ = [
    "modular_decrement",
    "modular_increment",
]
