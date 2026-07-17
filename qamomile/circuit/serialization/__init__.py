"""Persist static quantum artifacts and reconstruct them without user source.

The protobuf format preserves the highest target-neutral static semantics:
the qkernel interface, hierarchical IR body, shared callable definitions, and
all value-identity relationships needed by later compiler passes. Random
process-local UUID and logical-ID spellings are replaced by canonical
graph-local identities. Invocation bindings, runtime values, prepared compiler
modules, backend artifacts, resource estimates, and Python evaluation performed
during tracing are outside the format.

After :func:`deserialize`, pass the returned :class:`SerializedQKernel` to an
ordinary Qamomile transpiler with fresh ``bindings`` and ``parameters``.

Factory-produced Pauli LCU block-encoding descriptors use the dedicated
:func:`serialize_pauli_lcu_block_encoding` and
:func:`deserialize_pauli_lcu_block_encoding` adapters. Their typed recipe
stores the retained Pauli decomposition, while deserialization rebuilds the
descriptor's concrete unitary, normalization, and register widths.
"""

from qamomile.circuit.serialization.kernel import SerializedQKernel
from qamomile.circuit.serialization.pauli_lcu_block_encoding import (
    deserialize_pauli_lcu_block_encoding,
    serialize_pauli_lcu_block_encoding,
)
from qamomile.circuit.serialization.protobuf import deserialize, serialize
from qamomile.circuit.serialization.schema import QAMOMILE_VERSION

__all__ = [
    "QAMOMILE_VERSION",
    "SerializedQKernel",
    "deserialize",
    "deserialize_pauli_lcu_block_encoding",
    "serialize",
    "serialize_pauli_lcu_block_encoding",
]
