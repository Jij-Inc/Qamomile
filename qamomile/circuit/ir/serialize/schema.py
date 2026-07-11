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
that breaks **backward** compatibility — i.e. the current decoder can
no longer read previously-written payloads (required fields added or
removed, field semantics changed).  Purely additive changes do not
bump the version: new optional fields are read with
``d.get(...)``-with-default semantics (old payloads decode to the
default), and unknown extra fields in newer payloads are ignored by
older readers.  New operation ``$type`` tags are likewise additive —
an older reader rejects them loudly with the closed-dispatch
``ValueError`` rather than mis-decoding.  The current value and the
per-bump history are recorded next to the constant at the bottom of
this module.

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
embed inline-policy ``InvokeOperation`` references to nested Blocks before
inlining; cross-process module references will be addressed separately.
Box-preserved ``InvokeOperation`` definitions may embed a concrete
``body`` Block or an optional ``body_ref`` for deferred stdlib bodies
whose width is resolved later.

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

Hamiltonian payloads
--------------------

``ParamSlot.bound_value`` and ``ArrayRuntimeMetadata.const_array``
may carry ``qamomile.observable.Hamiltonian`` objects — the bound
values of ``Observable`` kernel parameters (e.g. a Trotter kernel
built with ``bindings={"Hs": [1.2 * Z(0), 0.8 * X(0)]}``). They are
encoded with the tagged-dict wrapper defined in
:mod:`qamomile.circuit.ir.serialize.hamiltonian_io`::

    {
        "$hamiltonian": True,
        "terms": [
            [[["Z", 0], ["X", 1]], <coeff>],   # one Pauli product per entry
            ...
        ],
        "constant": <coeff>,
        "num_qubits": <int | None>             # declared register width
    }

``<coeff>`` is a plain int / float, or — for complex coefficients —
``{"$complex": True, "real": <float>, "imag": <float>}``. Term order
follows the Hamiltonian's own term-dict iteration order and the
float-vs-complex distinction is preserved, so the reconstructed
object is ``repr``-identical to the original. (``content_hash`` hashes
Hamiltonian payloads structurally and order-independently via this
same wrapper — see ``canonical._hamiltonian_token`` — so wire fidelity
and hash identity are decoupled.) The wrapper contains no raw bytes,
so both wire formats carry it unchanged.

Forward compatibility
---------------------

When the decoder encounters a Block whose ``schema_version``
exceeds the loader's ``SCHEMA_VERSION``, it raises ``ValueError``
rather than guess at unknown fields. Earlier versions are likewise
rejected today; a future PR may add a migration table.
"""

from __future__ import annotations

# The current schema version.
#
# Project policy (2026-07): the wire format is NOT yet stable and no
# cross-version compatibility is promised. Format changes — including
# breaking ones that drop or reshape fields — land under version 1
# without a bump; producers and consumers are expected to run the same
# qamomile revision. Version negotiation / migration is deferred until
# the format is declared stable.
#
# Note: ``SymbolicControlledU`` gained an optional
# ``control_index_refs`` slot during the controlled-API redesign,
# but the field is kept additive-only (encoder writes it, decoder reads
# it with ``d.get(...)``-with-default semantics so a v1 payload without
# the field decodes to ``control_indices=None``).  This is a
# deliberately *backward-compatible* addition (the new decoder can
# still read old, field-less v1 payloads) rather than a schema bump;
# the version stays at 1.  The same pattern now also applies to the
# ``num_control_args`` field added alongside the multi-arg control
# prefix. New operation tags such as ``InverseBlockOperation`` are also
# additive: old loaders fail loudly on the unknown tag, while v1 readers
# that know the tag can still read older payloads.  The ``ArrayValue``
# slice-view refs (``slice_of_ref`` / ``slice_start_ref`` /
# ``slice_step_ref``) follow the same additive pattern: payloads
# written before they existed decode to a non-sliced array.  The
# ``$hamiltonian`` payload wrapper (with its ``$complex`` coefficient
# sub-wrapper) is additive too: pre-existing payloads contain no such
# wrapper (encoding a Hamiltonian used to raise ``TypeError``), so v1
# readers that know the tag can still read every older payload.  The
# ``loop_carried_rebinds`` records on loop operations follow the same
# additive encoding (safety caveat: a reader that predates them drops
# the records and with them the loop-carried rebind rejection — accepted
# under the same-revision policy above).  The ``region_args`` records on
# loop operations (explicit loop-carried values: ``var_name`` plus
# ``init_ref`` / ``block_arg_ref`` / ``yielded_ref`` / ``result_ref``)
# are additive in the same way; a reader that predates them would drop
# the loop-carried threading entirely, which is accepted only under the
# same-revision policy above.  ``IfOperation`` branch merges
# moved from embedded ``phi_ops`` operation dicts to
# ``true_yield_refs`` / ``false_yield_refs`` UUID lists parallel to
# ``result_refs`` — the first genuinely breaking reshape under the
# same-revision policy: pre-change payloads fail loudly (unknown
# ``PhiOp`` tag or merge-inconsistency ``ValueError``) instead of being
# migrated.
SCHEMA_VERSION: int = 1
