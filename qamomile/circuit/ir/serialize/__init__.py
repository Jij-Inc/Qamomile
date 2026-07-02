"""Public surface for the Qamomile IR serialization package.

Two wire formats are supported on top of a shared intermediate dict
schema (see :mod:`qamomile.circuit.ir.serialize.schema`):

- ``dump_json`` / ``load_json`` — UTF-8 JSON, easier to debug.
- ``dump_msgpack`` / ``load_msgpack`` — binary, more compact (numpy
  payloads pass through as native ``bin``).

``to_dict`` / ``from_dict`` are exposed for tests and tooling that
want to operate on the intermediate Python dict directly.

``hamiltonian_to_dict`` / ``dict_to_hamiltonian`` expose the
``$hamiltonian`` payload wrapper as a standalone wire form so an
embedding host (e.g. a workflow runner carrying a Qamomile quantum
node) can ship observables next to — rather than inside — a
serialized Block and re-supply them as ``bindings`` at
``transpile_block`` time.

Scope: ``BlockKind.AFFINE`` and ``BlockKind.ANALYZED`` only.
``HIERARCHICAL`` blocks still hold ``CallBlockOperation`` references
by Python identity and are deferred.
"""

from qamomile.circuit.ir.serialize.decode import from_dict
from qamomile.circuit.ir.serialize.encode import to_dict
from qamomile.circuit.ir.serialize.hamiltonian_io import (
    dict_to_hamiltonian,
    hamiltonian_to_dict,
)
from qamomile.circuit.ir.serialize.json_io import dump_json, load_json
from qamomile.circuit.ir.serialize.msgpack_io import dump_msgpack, load_msgpack
from qamomile.circuit.ir.serialize.schema import SCHEMA_VERSION

__all__ = [
    "SCHEMA_VERSION",
    "dict_to_hamiltonian",
    "dump_json",
    "dump_msgpack",
    "from_dict",
    "hamiltonian_to_dict",
    "load_json",
    "load_msgpack",
    "to_dict",
]
