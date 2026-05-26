"""Intermediate dict schema for IR serialization.

This module defines ``SCHEMA_VERSION`` and the shape of the
intermediate Python ``dict`` that ``encode`` produces and ``decode``
consumes. The wire encoders (``json_io``, ``msgpack_io``) work on top
of this dict.

Top-level envelope
------------------

::

    {
        "schema_version": <int>,
        "block": <block dict>
    }

``SCHEMA_VERSION`` increments whenever the schema changes in a way
that breaks forward / backward compatibility (added required fields,
removed fields, semantic changes). The current version is 1.

Block dict
----------

::

    {
        "$type": "Block",
        "kind": "AFFINE" | "ANALYZED",
        "name": <str>,
        "label_args": [<str>, ...],
        "input_value_refs": [<uuid>, ...],
        "output_value_refs": [<uuid>, ...],
        "output_names": [<str>, ...],
        "parameters": {<name>: <uuid>, ...},
        "param_slots": [<param_slot dict>, ...],
        "value_table": [<value dict>, ...],   # every Value appears once
        "operations": [<op dict>, ...]
    }

Only ``BlockKind.AFFINE`` and ``BlockKind.ANALYZED`` are accepted.
``HIERARCHICAL`` and ``TRACED`` raise on encode because they may
embed ``CallBlockOperation`` references to sibling Blocks by Python
identity; cross-process module references will be addressed
separately.

Type tags
---------

Every polymorphic structure carries a ``"$type"`` string used by the
decoder's closed dispatch table. The decoder NEVER imports modules
or looks up classes from these strings; it dispatches through a
hard-coded ``$type -> factory`` map. Unknown tags raise
``ValueError``. This is the load-bearing security invariant.

Value references
----------------

Within a Block dict, every Value appears in exactly one place:
``value_table``. Everything else (operand lists, output lists,
parameter slots, parent_array pointers, etc.) references Values by
their canonical UUID string. The dict shape is therefore a tree
(no Python-level cycles) plus a flat Value table.

Numpy arrays
------------

``ParamSlot.bound_value`` and metadata fields like
``ArrayRuntimeMetadata.const_array`` may carry ``numpy.ndarray``
payloads. They are encoded with the tagged-dict wrapper defined in
:mod:`qamomile.circuit.ir.serialize.numpy_io`::

    {
        "$np_array": True,
        "dtype": <str>,         # one of the allow-list
        "shape": [<int>, ...],
        "data": <bytes>         # ndarray.tobytes()
    }

JSON I/O converts ``data`` bytes to a base64 string at the JSON
boundary and back on read; msgpack passes the bytes through as a
``bin`` native type.

Forward compatibility
---------------------

When the decoder encounters a Block whose ``schema_version``
exceeds the loader's ``SCHEMA_VERSION``, it raises ``ValueError``
rather than guess at unknown fields. Earlier versions are likewise
rejected today; a future PR may add a migration table.
"""

from __future__ import annotations

# The current schema version. Bump on every breaking change.
#
# Version history:
#   1 — initial wire format.
#   2 — ``SymbolicControlledU`` gains a ``controlled_index_refs`` slot
#       (per the controlled-API redesign, step 3).  The field is always
#       written; a ``None`` value reproduces the v1 semantics ("use the
#       entire control vector as the pool").
SCHEMA_VERSION: int = 2
