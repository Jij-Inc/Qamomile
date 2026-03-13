"""Observable type for Hamiltonian parameter representation.

This module defines the ObservableType for the Qamomile IR, which represents
a reference to a Hamiltonian observable provided via bindings during transpilation.

Unlike the previous HamiltonianExprType, this is purely a reference type -
the actual qamomile.observable.Hamiltonian is provided from Python code.
"""

import dataclasses

from .primitives import ObjectTypeMixin, ValueType


@dataclasses.dataclass
class ObservableType(ObjectTypeMixin, ValueType):
    """Type representing a Hamiltonian observable parameter.

    This is a reference type - the actual qamomile.observable.Hamiltonian
    is provided via bindings during transpilation. It cannot be constructed
    or manipulated within qkernels.

    Example usage:
        ```python
        import qamomile.circuit as qm
        import qamomile.observable as qm_o

        # Build Hamiltonian in Python
        H = qm_o.Z(0) * qm_o.Z(1)

        @qm.qkernel
        def vqe(q: qm.Vector[qm.Qubit], H: qm.Observable) -> qm.Float:
            return qm.expval(q, H)

        # H is passed as binding
        executable = transpiler.transpile(vqe, bindings={"H": H})
        ```
    """

    pass
