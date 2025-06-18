import jijmodeling as jm


class PauliExpr:
    """Pauli expression class."""

    def x(shape: int | jm.Placeholder | tuple[int, ...]) -> jm.BinaryVar:
        """Create a PauliX expression.

        Args:
            shape (int | jm.Placeholder | tuple[int, ...]): Shape of the PauliX expression.

        Raises:
            ValueError: If the shape is invalid.

        Returns:
            jm.BinaryVar: PauliX expression.
        """
        if isinstance(shape, int):
            return jm.BinaryVar(
                "_PauliX", shape=(shape,), latex=r"\hat{X}", description="PauliX"
            )
        elif isinstance(shape, tuple):
            return jm.BinaryVar(
                "_PauliX", shape=shape, latex=r"\hat{X}", description="PauliX"
            )
        else:
            raise ValueError("The shape is invalid.")

    def y(shape: int | jm.Placeholder | tuple[int, ...]) -> jm.BinaryVar:
        """Create a PauliY expression.

        Args:
            shape (int | jm.Placeholder | tuple[int, ...]): Shape of the PauliY expression.

        Raises:
            ValueError: If the shape is invalid.

        Returns:
            jm.BinaryVar: PauliY expression.
        """

        if isinstance(shape, int):
            return jm.BinaryVar(
                "_PauliY", shape=(shape,), latex=r"\hat{Y}", description="PauliY"
            )
        elif isinstance(shape, tuple):
            return jm.BinaryVar(
                "_PauliY", shape=shape, latex=r"\hat{Y}", description="PauliY"
            )
        else:
            raise ValueError("The shape is invalid.")

    def z(shape: int | jm.Placeholder | tuple[int, ...]) -> jm.BinaryVar:
        """Create a PauliZ expression.

        Args:
            shape (int | jm.Placeholder | tuple[int, ...]): Shape of the PauliZ expression.

        Raises:
            ValueError: If the shape is invalid.

        Returns:
            jm.BinaryVar: PauliZ expression.
        """

        if isinstance(shape, int):
            return jm.BinaryVar(
                "_PauliZ", shape=(shape,), latex=r"\hat{Z}", description="PauliZ"
            )
        elif isinstance(shape, tuple):
            return jm.BinaryVar(
                "_PauliZ", shape=shape, latex=r"\hat{Z}", description="PauliZ"
            )
        else:
            raise ValueError("The shape is invalid.")
