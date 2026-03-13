"""Observable handle for Hamiltonian parameters.

This module provides the Observable handle class that represents a reference
to a Hamiltonian observable provided via bindings during transpilation.
Unlike HamiltonianExpr in previous versions, this is a pure parameter handle
with no arithmetic operations.
"""

from __future__ import annotations

import dataclasses

from .handle import Handle


@dataclasses.dataclass
class Observable(Handle):
    """Handle representing a Hamiltonian observable parameter.

    This is a reference type - the actual qamomile.observable.Hamiltonian
    is provided via bindings during transpilation. It cannot be constructed
    or manipulated within qkernels.

    Example:
        ```python
        import qamomile.circuit as qm
        import qamomile.observable as qm_o

        # Build Hamiltonian in Python
        H = qm_o.Z(0) * qm_o.Z(1) + 0.5 * qm_o.X(0)

        @qm.qkernel
        def vqe(q: qm.Vector[qm.Qubit], H: qm.Observable) -> qm.Float:
            # Use Hamiltonian from bindings
            return qm.expval(q, H)

        # Pass via bindings
        executable = transpiler.transpile(vqe, bindings={"H": H})
        ```
    """

    pass
