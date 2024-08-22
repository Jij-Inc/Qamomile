import jijmodeling as jm


class PauliExpr:
    def x(shape: int | tuple[int, ...]) -> jm.BinaryVar:
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

    def y(shape: int | tuple[int, ...]) -> jm.BinaryVar:
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

    def z(shape: int | tuple[int, ...]) -> jm.BinaryVar:
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
