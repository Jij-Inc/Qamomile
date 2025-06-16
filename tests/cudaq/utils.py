from typing import Callable, Final

import pyqir


QIR_PARAMETRIC_OPERATIONS_PARAM_MAPS: Final[dict[str, int]] = {
    "__quantum__qis__exp_pauli": 1,
    "__quantum__qis__rz": 1,
    "__quantum__qis__ry": 1,
    "__quantum__qis__rx": 1,
    "__quantum__qis__u3": 3,
    "__quantum__qis__u2": 2,
    "__quantum__qis__u1": 1,
}
QIR_NON_PARAMETRIC_OPERATIONS: Final[list[str]] = [
    "__quantum__qis__h",
    "__quantum__qis__x",
    "__quantum__qis__y",
    "__quantum__qis__z",
    "__quantum__qis__s",
    "__quantum__qis__t",
    "__quantum__qis__cnot",
    "__quantum__qis__ccnot",
    "__quantum__qis__cz",
    "__quantum__qis__swap",
    "__quantum__qis__cswap",
    "__quantum__qis__mz",
]
QIR_OPERATIONS: Final[list[str]] = (
    list(QIR_PARAMETRIC_OPERATIONS_PARAM_MAPS.keys()) + QIR_NON_PARAMETRIC_OPERATIONS
)


def count_qir_parameters(qir_str: str) -> int:
    """Count the number of parameters including values used in a QIR (Quantum Intermediate Representation) string.
    This function is not completely accurate. Check the following points.
    1. this function counts the number of parameters including values, not only the number of symbolic parameters.
       Thus, if the QIR string contains RX(0) and RX(theta), then this function will return 2, not 1.
    2. Even if you use the same parameter multiple times, it will be counted multiple times.
    3. If you use the exponential of a Pauli operator with a hamiltonian that has multiple terms,
       it will count the number of terms in the hamiltonian as the number of parameters.
       For example, if you have `exp(i * (XZ + ZX))`, it will count 2 parameters, not 1.


    Args:
        qir_str (str): a Quantum Intermediate Representation (QIR) str to analyse

    Returns:
        int: the number of unique parameter including values found in the QIR string
    """
    # Create a context and module from the QIR string to analyse it.
    context = pyqir.Context()
    module = pyqir.Module.from_ir(context=context, ir=qir_str)

    # Count the total number of parameters including values in the QIR module.
    total_params = 0
    for func in module.functions:
        for block in func.basic_blocks:
            for inst in block.instructions:
                if isinstance(inst, pyqir.Call):
                    callee_name = inst.callee.name
                    if callee_name in QIR_PARAMETRIC_OPERATIONS_PARAM_MAPS:
                        total_params += QIR_PARAMETRIC_OPERATIONS_PARAM_MAPS[
                            callee_name
                        ]
    return total_params


def count_qir_operations(qir_str: str) -> dict[str, int]:
    """Count the number of operations in a Quantum Intermediate Representation (QIR) string.
    This function is not completely accurate. Check the following point.
    1. If you use the exponential of a Pauli operator with a hamiltonian that has multiple terms,
       it will count the number of terms in the hamiltonian as the number of parameters.
       For example, if you have `exp(i * (XZ + ZX))`, it will count 2 parameters, not 1.

    Args:
        qir_str (str): a Quantum Intermediate Representation (QIR) str to analyse

    Returns:
        dict[str, int]: the number of operations found in the QIR string
    """
    # Create a context and module from the QIR string to analyse it.
    context = pyqir.Context()
    module = pyqir.Module.from_ir(context, qir_str)

    # Count the number of operations in the QIR module.
    gate_counts = {}
    for func in module.functions:
        for block in func.basic_blocks:
            for inst in block.instructions:
                if isinstance(inst, pyqir.Call):
                    callee_name = inst.callee.name
                    if callee_name in QIR_OPERATIONS:
                        try:
                            # Found the gate before, increment the count.
                            gate_counts[callee_name] += 1
                        except KeyError:
                            # Found the gate for the first time, initialise the count.
                            gate_counts[callee_name] = 1
    return gate_counts
