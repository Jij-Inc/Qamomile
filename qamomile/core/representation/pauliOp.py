from __future__ import annotations
import jijmodeling as jm

def pauli_x(shape: int | tuple[int, ...]) -> jm.BinaryVar:
    if isinstance(shape, int):
        return jm.BinaryVar("X",shape = (shape,), latex=r"\hat{X}", description="PauliX")
    elif isinstance(shape, tuple):
        return jm.BinaryVar("X",shape = shape, latex=r"\hat{X}", description="PauliX")
    else:
        raise ValueError("Invalid shape")
    
def pauli_y(shape: int | tuple[int, ...]) -> jm.BinaryVar:
    if isinstance(shape, int):
        return jm.BinaryVar("Y",shape = (shape,), latex=r"\hat{Y}", description="PauliY")
    elif isinstance(shape, tuple):
        return jm.BinaryVar("Y",shape = shape, latex=r"\hat{Y}", description="PauliY")
    else:
        raise ValueError("Invalid shape")
    
def pauli_z(shape: int | tuple[int, ...]) -> jm.BinaryVar:
    if isinstance(shape, int):
        return jm.BinaryVar("Z",shape = (shape,), latex=r"\hat{Z}", description="PauliZ")
    elif isinstance(shape, tuple):
        return jm.BinaryVar("Z",shape = shape, latex=r"\hat{Z}", description="PauliZ")
    else:
        raise ValueError("Invalid shape")