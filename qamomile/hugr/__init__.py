"""HUGR program-graph target for Qamomile.

The engine lowers prepared hierarchical Qamomile semantics directly to a
HUGR package. It deliberately bypasses circuit segmentation and CircuitProgram
so function boundaries, typed dataflow, and hybrid control can be preserved.
Generated quantum operations use the same ``tket.*`` extensions as Guppy,
making the result interoperable with the Guppy/HUGR ecosystem.

Destructive measurements consume qubits with ``MeasureFree``. Projections
retain their measured state with ``Measure``. Direct measurement/reset pairs
use ``MeasureReset`` when their quantum wire and ordering permit fusion.
Opaque measurement bits are converted to the existing native Boolean carrier.

``prepare_h2_submission`` exposes final QIR and detached recovery metadata
before any Nexus client or network operation. ``HugrExecutor.submit_prepared``
sends those bytes unchanged with a caller-owned job name. The saved context
restores tagged rows through ``retrieve``; typed executable restoration retains
its existing ABI and snapshot requirements.
"""

from qamomile.hugr._nexus import (
    NexusExecutionOptions,
    NexusRegion,
    PreparedH2Submission,
    prepare_h2_submission,
)
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
    "PreparedH2Submission",
    "prepare_h2_submission",
]
