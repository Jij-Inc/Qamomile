"""Persist one unbound qkernel and reconstruct it without Python source.

The protobuf format preserves the highest target-neutral static semantics:
the qkernel interface, hierarchical IR body, shared callable definitions, and
all value-identity relationships needed by later compiler passes. Random
process-local UUID and logical-ID spellings are replaced by canonical
graph-local identities. Invocation bindings, runtime values, prepared compiler
modules, backend artifacts, resource estimates, and Python evaluation performed
during tracing are outside the format.

After :func:`deserialize`, pass the returned :class:`SerializedQKernel` to an
ordinary Qamomile transpiler with fresh ``bindings`` and ``parameters``.

Registered compile-time object arguments are stored only as typed binding
slots. Their concrete values are deliberately absent from the payload and must
be supplied through ``bindings`` after deserialization.
"""

from qamomile.circuit.serialization.kernel import SerializedQKernel
from qamomile.circuit.serialization.protobuf import deserialize, serialize
from qamomile.circuit.serialization.schema import QAMOMILE_VERSION

__all__ = [
    "QAMOMILE_VERSION",
    "SerializedQKernel",
    "deserialize",
    "serialize",
]
