"""HUGR program-graph target for Qamomile.

The engine lowers prepared hierarchical Qamomile semantics directly to a
HUGR package. It deliberately bypasses circuit segmentation and CircuitProgram
so function boundaries, typed dataflow, and hybrid control can be preserved.
Generated quantum operations use the same ``tket.*`` extensions as Guppy,
making the result interoperable with the Guppy/HUGR ecosystem.
"""

from qamomile.hugr._nexus import NexusExecutionOptions, NexusRegion
from qamomile.hugr.executable import HugrExecutable
from qamomile.hugr.execution import (
    HugrExecutionTarget,
    HugrExecutor,
    SeleneExecutionOptions,
)
from qamomile.hugr.lowerer import HugrCompilationPlan, HugrTarget
from qamomile.hugr.transpiler import HugrTranspiler

__all__ = [
    "HugrCompilationPlan",
    "HugrTarget",
    "HugrTranspiler",
    "HugrExecutable",
    "HugrExecutor",
    "HugrExecutionTarget",
    "SeleneExecutionOptions",
    "NexusExecutionOptions",
    "NexusRegion",
]
