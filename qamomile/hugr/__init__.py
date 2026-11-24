"""Expose HUGR compilation and local Selene or Helios execution.

``Hugr()`` selects the local Selene profile and its zero-origin counter
convention. This convention is recorded in the target fingerprint and metadata;
it is not a format-wide HUGR guarantee. Logical current-shot execution accepts
the Selene 0.2.17 and 0.3.2 worker schedules. Direct Boolean result recording is
verified with the bundled Selene 0.2.17. Execution with Selene 0.3.2 and this
result format has not been reverified. Helios leaves logical current-shot
execution unavailable until its endpoint contract is independently verified.
Runtime Int operations use a checked signed32 domain carried on int64 wires.
Every HUGR profile records measured bits and computed Boolean outputs directly in
source order, without allocating qubits solely to report results.
The HUGR extra includes the pinned Selene 0.2.17 profile and the Nexus client
used for Helios execution.
"""

from ._execution import load_package as load_package
from ._nexus import Helios as Helios, HeliosOptions as HeliosOptions
from ._profile import FORMAT as FORMAT
from ._selene import Selene as Selene
from ._target import Hugr as Hugr

__all__ = [
    "FORMAT",
    "Helios",
    "HeliosOptions",
    "Hugr",
    "Selene",
    "load_package",
]
