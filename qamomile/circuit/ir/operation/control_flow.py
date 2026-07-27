from __future__ import annotations

import dataclasses
import typing

from qamomile.circuit.ir.types.primitives import BitType, BlockType, UIntType
from qamomile.circuit.ir.value import Value, ValueBase

from .operation import Operation, OperationKind, ParamHint, Signature


class IfMerge(typing.NamedTuple):
    """One branch-merge slot of an :class:`IfOperation`.

    An ``IfOperation`` merges each variable touched by its branches back
    into a single SSA value. ``IfMerge`` is the read-side view of one such
    merge slot, decoupling every consumer from how the merge is stored in
    the IR (today the parallel ``IfOperation.true_yields`` /
    ``false_yields`` lists; the storage may change without touching
    consumers).

    Attributes:
        index (int): Position of this merge among the if's results.
        true_value (Value): Value the merge selects when the condition is
            true.
        false_value (Value): Value the merge selects when the condition is
            false.
        result (Value): The merged SSA output (``IfOperation.results[index]``).
    """

    # The field intentionally shadows ``tuple.index`` (allowed for
    # NamedTuple fields); nothing calls the method on merges.
    index: int  # type: ignore[assignment]
    true_value: Value
    false_value: Value
    result: Value

    def select(self, taken: bool) -> Value:
        """Return the branch source selected by a resolved condition.

        Args:
            taken (bool): The condition's truth value (``True`` selects the
                true branch).

        Returns:
            Value: ``true_value`` when ``taken`` is true, else
                ``false_value``.
        """
        return self.true_value if taken else self.false_value

    @property
    def is_identity(self) -> bool:
        """Whether both branches merge the same underlying value.

        The frontend creates a merge slot for every variable referenced in
        a branch, including read-only ones; those slots carry the same IR
        value on both sides and resolve deterministically regardless of the
        condition.

        Returns:
            bool: ``True`` when ``true_value`` and ``false_value`` share a
                UUID.
        """
        return self.true_value.uuid == self.false_value.uuid


@dataclasses.dataclass(frozen=True)
class Region:
    """Expose one structured-control region through a uniform interface.

    This is a view over the existing semantic IR rather than a parallel value
    system: every entry is an ordinary Qamomile ``ValueBase`` carrying the
    current UUID identity. Control-flow operations remain the owners of the
    stored fields and construct ``Region`` views through ``nested_regions``.

    Args:
        operations (tuple[Operation, ...]): Operations evaluated inside the
            region in program order.
        block_args (tuple[ValueBase, ...]): Values defined at region entry,
            such as a loop induction variable or carried-value formal.
        captures (tuple[ValueBase, ...]): Explicit outer-scope values read by
            the region, ordered by first use.
        yields (tuple[ValueBase, ...]): Values yielded at the region boundary
            in result-slot order.
    """

    operations: tuple[Operation, ...]
    block_args: tuple[ValueBase, ...] = ()
    captures: tuple[ValueBase, ...] = ()
    yields: tuple[ValueBase, ...] = ()


class HasNestedOps:
    """Mixin for operations that contain nested operation lists.

    ``nested_regions()`` is the canonical traversal API because it exposes
    operations together with block arguments, captures, and yields.
    ``nested_op_lists()`` / ``rebuild_nested()`` remain compatibility helpers
    for specialized consumers while they migrate to the region interface.
    """

    def nested_op_lists(self) -> list[list[Operation]]:
        """Return all nested operation lists in this control flow op."""
        raise NotImplementedError

    def rebuild_nested(self, new_lists: list[list[Operation]]) -> Operation:
        """Return a copy with nested operation lists replaced.

        ``new_lists`` must have the same length/order as ``nested_op_lists()``.
        """
        raise NotImplementedError

    def nested_regions(self) -> tuple[Region, ...]:
        """Return uniform views of every nested operation region.

        Subclasses with explicit block arguments, captures, or yields override
        this method. The fallback keeps legacy operation-owned blocks visible
        while consumers migrate from ``nested_op_lists``.

        Returns:
            tuple[Region, ...]: Region views in ``nested_op_lists`` order.
        """
        return tuple(
            Region(operations=tuple(operations))
            for operations in self.nested_op_lists()
        )

    def rebuild_regions(self, regions: typing.Sequence[Region]) -> Operation:
        """Return a copy with replacement region operation sequences.

        Concrete control-flow operations override this method to rebuild both
        their body operations and boundary values. The fallback supports
        legacy region owners whose boundary remains operation-specific.

        Args:
            regions (Sequence[Region]): Replacement regions in
                ``nested_regions`` order.

        Returns:
            Operation: Rebuilt control-flow operation.

        Raises:
            ValueError: If the replacement region count differs from the
                operation's current region count.
        """
        expected = len(self.nested_regions())
        if len(regions) != expected:
            raise ValueError(
                f"Expected {expected} replacement regions, got {len(regions)}"
            )
        return self.rebuild_nested([list(region.operations) for region in regions])


@dataclasses.dataclass(frozen=True)
class RegionArg:
    """Explicit loop-carried value on a loop operation (MLIR-style iter_arg).

    A ``RegionArg`` makes a loop-carried dependency explicit in the IR,
    the way MLIR's ``scf.for`` models ``iter_args`` / ``scf.yield``:

    - On iteration 0 the body reads ``block_arg`` bound to ``init``.
    - After each iteration, ``block_arg`` is rebound to that iteration's
      ``yielded`` value.
    - After the loop, ``result`` holds the final carried value (``init``
      when the loop ran zero iterations).

    The loop body's operations reference ``block_arg`` (the frontend
    substitutes the traced pre-loop reads), and post-loop operations
    reference ``result`` (the frontend rebinds the Python handle when it
    closes the loop). ``result`` is also appended to the loop operation's
    ``results`` list so dependency analysis sees the loop as its
    producer.

    This subsumes the trace-once staleness that ``LoopCarriedRebind``
    records exist to reject: a rebind represented as a ``RegionArg`` is
    a supported loop-carried value, not a miscompilation hazard.

    Attributes:
        var_name (str): Display name of the carried Python variable.
            Used for printers and error messages only.
        init (Value): The value entering iteration 0 (the pre-loop
            value). A genuine input of the loop operation.
        block_arg (Value): The region argument the body reads each
            iteration. A definition owned by the loop operation (like
            ``ForOperation.loop_var_value``), not an outer-scope read.
        yielded (Value): The value carried into the next iteration —
            the body's final binding of the variable. Usually produced
            by an operation inside ``operations``, but NOT always: a
            loop-invariant overwrite (``x = c`` with ``c`` defined
            before the loop) yields the outer-scope value directly, and
            an identity carry yields ``block_arg`` itself. Consumers
            must not assume a body-internal producer; in particular,
            liveness must treat the yield as a read that keeps an
            outer-scope producer alive (see
            ``CompileTimeIfLoweringPass._collect_used_uuids``).
        result (Value): The value visible after the loop. A definition
            owned by the loop operation; also present in ``results``.
    """

    var_name: str
    init: Value
    block_arg: Value
    yielded: Value
    result: Value


def _region_arg_values(
    region_args: tuple[RegionArg, ...],
) -> list[ValueBase]:
    """Collect the Value fields of region-argument records.

    Args:
        region_args (tuple[RegionArg, ...]): Records attached to a loop
            operation.

    Returns:
        list[ValueBase]: ``init``, ``block_arg``, ``yielded``, and
            ``result`` of every record, in order.
    """
    values: list[ValueBase] = []
    for arg in region_args:
        values.extend((arg.init, arg.block_arg, arg.yielded, arg.result))
    return values


def _replace_region_arg_values(
    region_args: tuple[RegionArg, ...],
    mapping: dict[str, ValueBase],
) -> tuple[RegionArg, ...] | None:
    """Substitute region-argument values through a UUID mapping.

    Args:
        region_args (tuple[RegionArg, ...]): Records to rewrite.
        mapping (dict[str, ValueBase]): UUID-keyed substitution map.

    Returns:
        tuple[RegionArg, ...] | None: Rewritten records, or ``None``
            when nothing changed.
    """
    new_args: list[RegionArg] = []
    changed = False
    for arg in region_args:
        replacements: dict[str, Value] = {}
        for field_name in ("init", "block_arg", "yielded", "result"):
            current = getattr(arg, field_name)
            mapped = mapping.get(current.uuid)
            if isinstance(mapped, Value) and mapped is not current:
                replacements[field_name] = mapped
        if replacements:
            arg = RegionArg(
                var_name=arg.var_name,
                init=replacements.get("init", arg.init),
                block_arg=replacements.get("block_arg", arg.block_arg),
                yielded=replacements.get("yielded", arg.yielded),
                result=replacements.get("result", arg.result),
            )
            changed = True
        new_args.append(arg)
    return tuple(new_args) if changed else None


def _replace_value_tuple(
    values: tuple[ValueBase, ...],
    mapping: dict[str, ValueBase],
) -> tuple[ValueBase, ...] | None:
    """Substitute a tuple of boundary values through a UUID mapping.

    Args:
        values (tuple[ValueBase, ...]): Region boundary values to rewrite.
        mapping (dict[str, ValueBase]): UUID-keyed substitution map.

    Returns:
        tuple[ValueBase, ...] | None: Rewritten values, or ``None`` when no
            entry changes.
    """
    rewritten = tuple(mapping.get(value.uuid, value) for value in values)
    if all(before is after for before, after in zip(values, rewritten, strict=True)):
        return None
    return rewritten


def _require_region_values(
    values: typing.Sequence[ValueBase],
    *,
    label: str,
) -> tuple[Value, ...]:
    """Require scalar/array ``Value`` objects at a region boundary.

    Args:
        values (Sequence[ValueBase]): Boundary values to validate.
        label (str): Boundary label used in diagnostics.

    Returns:
        tuple[Value, ...]: Values narrowed to the concrete IR value family.

    Raises:
        ValueError: If a tuple/dict aggregate appears where a region slot
            requires a single SSA value.
    """
    if not all(isinstance(value, Value) for value in values):
        raise ValueError(f"{label} must contain only scalar or array Values")
    return typing.cast(tuple[Value, ...], tuple(values))


@dataclasses.dataclass(frozen=True)
class LoopCarriedRebind:
    """Trace-time record of a variable rebound inside a loop body.

    Two rebind families share this record type, distinguished by the
    type of ``before``:

    - **Classical scalar** (``before`` classical): the frontend traces a
      loop body exactly once, so a Python-level reassignment like
      ``total = total + i`` produces IR whose right-hand side reads the
      fixed pre-loop value instead of the previous iteration's value.
      Most such carries are now represented as explicit ``RegionArg``s
      (see above) and are fully supported; a classical record is only
      created for the shapes region binding declines — ``while``-body
      carries (a runtime while loop cannot be unrolled) and
      measurement-backed ``Bit`` carries — and the transpiler's
      classical loop-carried check rejects those with a targeted error
      instead of silently miscompiling.
    - **Quantum** (``before`` quantum): the loop body left the variable
      bound to a different quantum resource (``logical_id`` change — a
      fresh allocation or another register, not a gate self-update).
      The transpiler's control-flow discard check
      (``reject_control_flow_quantum_discard``) rejects the ones whose
      incoming state the body never consumes.

    Attributes:
        var_name (str): Display name of the rebound Python variable.
            Used only for error messages.
        before (ValueBase): The variable's IR value before the loop body ran
            (for classical records, the stale value the traced body
            reads; for quantum records, the incoming state a rebinding
            iteration would discard).
        after (ValueBase): The variable's IR value after the loop body ran
            (typically a ``BinOp`` result, an ``IfOperation`` merge
            output, or a fresh quantum allocation).
        before_synthesized (bool): True when the pre-loop value was a
            plain Python number with no IR identity (e.g. ``total = 0``),
            in which case ``before`` is a synthesized constant ``Value``
            that does not appear in the body's dataflow. Always False
            for quantum records.
    """

    var_name: str
    before: ValueBase
    after: ValueBase
    before_synthesized: bool = False


def _rebind_input_values(
    rebinds: tuple[LoopCarriedRebind, ...],
) -> list[ValueBase]:
    """Collect the Value fields of loop-carried rebind records.

    Args:
        rebinds (tuple[LoopCarriedRebind, ...]): Records attached to a
            loop operation.

    Returns:
        list[ValueBase]: ``before`` and ``after`` values of every record,
            in order.
    """
    values: list[ValueBase] = []
    for rebind in rebinds:
        values.append(rebind.before)
        values.append(rebind.after)
    return values


def _replace_rebind_values(
    rebinds: tuple[LoopCarriedRebind, ...],
    mapping: dict[str, ValueBase],
) -> tuple[LoopCarriedRebind, ...] | None:
    """Substitute rebind record values through a UUID mapping.

    Args:
        rebinds (tuple[LoopCarriedRebind, ...]): Records to rewrite.
        mapping (dict[str, ValueBase]): UUID-keyed substitution map.

    Returns:
        tuple[LoopCarriedRebind, ...] | None: Rewritten records, or
            ``None`` when nothing changed.
    """
    new_rebinds: list[LoopCarriedRebind] = []
    changed = False
    for rebind in rebinds:
        before = rebind.before
        after = rebind.after
        mapped_before = mapping.get(before.uuid)
        if isinstance(mapped_before, ValueBase):
            before = mapped_before
        mapped_after = mapping.get(after.uuid)
        if isinstance(mapped_after, ValueBase):
            after = mapped_after
        if before is not rebind.before or after is not rebind.after:
            rebind = dataclasses.replace(rebind, before=before, after=after)
            changed = True
        new_rebinds.append(rebind)
    return tuple(new_rebinds) if changed else None


def _replace_yield_values(
    values: list[Value],
    mapping: dict[str, ValueBase],
) -> list[Value] | None:
    """Substitute branch-yield values through a UUID mapping.

    Args:
        values (list[Value]): Yield values to rewrite.
        mapping (dict[str, ValueBase]): UUID-keyed substitution map.

    Returns:
        list[Value] | None: Rewritten list, or ``None`` when nothing
            changed.
    """
    new_values: list[Value] = []
    changed = False
    for value in values:
        mapped = mapping.get(value.uuid)
        if isinstance(mapped, Value) and mapped is not value:
            new_values.append(mapped)
            changed = True
        else:
            new_values.append(value)
    return new_values if changed else None


@dataclasses.dataclass
class WhileOperation(HasNestedOps, Operation):
    """Represents a while loop operation.

    Only measurement-backed conditions are supported: the condition must
    be a ``Bit`` value produced by ``qmc.measure()``.  Non-measurement
    conditions (classical variables, constants, comparisons) are rejected
    by ``ValidateWhileContractPass`` before reaching backend emit.

    Example::

        bit = qmc.measure(q)
        while bit:
            q = qmc.h(q)
            bit = qmc.measure(q)

    Attributes:
        operations: List of operations in the loop body.
        region_args: Explicit loop-carried values (see ``RegionArg``).
            Each entry's ``result`` also appears in ``results``.
        operands[0]: Initial condition (required). Must be a measurement
            result (``Bit`` from ``qmc.measure()``).
        operands[1]: Loop-carried condition (optional). When the loop body
            reassigns the condition variable (e.g., ``bit = qmc.measure(q)``),
            this captures the updated handle so that the transpiler can alias
            both UUIDs to the same classical bit during resource allocation.
            Without this alias, SSA-like value versioning would allocate a
            separate clbit for the body measurement, causing the while
            condition to never update (infinite loop on Qiskit) or read from
            the wrong qubit (wrong return value on QURI Parts).
    """

    operations: list[Operation] = dataclasses.field(default_factory=list)
    max_iterations: int | None = None
    loop_carried_rebinds: tuple[LoopCarriedRebind, ...] = ()
    region_args: tuple[RegionArg, ...] = ()
    captures: tuple[ValueBase, ...] = ()

    def nested_op_lists(self) -> list[list[Operation]]:
        return [self.operations]

    def rebuild_nested(self, new_lists: list[list[Operation]]) -> Operation:
        return dataclasses.replace(self, operations=new_lists[0])

    def nested_regions(self) -> tuple[Region, ...]:
        """Return the while body with explicit boundary values.

        Returns:
            tuple[Region, ...]: One body region whose block arguments and
                yields are aligned with ``region_args``. The updated
                condition, when present, is appended as the final yield.
        """
        condition_yields: tuple[ValueBase, ...] = (
            (self.operands[1],) if len(self.operands) > 1 else ()
        )
        return (
            Region(
                operations=tuple(self.operations),
                block_args=tuple(arg.block_arg for arg in self.region_args),
                captures=self.captures,
                yields=(
                    *(arg.yielded for arg in self.region_args),
                    *condition_yields,
                ),
            ),
        )

    def rebuild_regions(self, regions: typing.Sequence[Region]) -> Operation:
        """Rebuild the while body and its complete boundary interface.

        Args:
            regions (Sequence[Region]): Exactly one replacement body region.

        Returns:
            Operation: Rebuilt while operation.

        Raises:
            ValueError: If arity or boundary value kinds are inconsistent.
        """
        if len(regions) != 1:
            raise ValueError(f"WhileOperation expects 1 region, got {len(regions)}")
        region = regions[0]
        block_args = _require_region_values(
            region.block_args, label="WhileOperation block arguments"
        )
        if len(block_args) != len(self.region_args):
            raise ValueError(
                "WhileOperation block-argument count must match region_args"
            )
        if len(region.yields) not in {
            len(self.region_args),
            len(self.region_args) + 1,
        }:
            raise ValueError(
                "WhileOperation yields must contain carried values and an "
                "optional updated condition"
            )
        carried_yields = _require_region_values(
            region.yields[: len(self.region_args)],
            label="WhileOperation carried yields",
        )
        region_args = tuple(
            dataclasses.replace(arg, block_arg=block_arg, yielded=yielded)
            for arg, block_arg, yielded in zip(
                self.region_args,
                block_args,
                carried_yields,
                strict=True,
            )
        )
        operands = list(self.operands[:1])
        if len(region.yields) > len(self.region_args):
            condition_yield = _require_region_values(
                region.yields[len(self.region_args) :],
                label="WhileOperation condition yield",
            )[0]
            operands.append(condition_yield)
        return dataclasses.replace(
            self,
            operations=list(region.operations),
            operands=operands,
            region_args=region_args,
            captures=tuple(region.captures),
        )

    def all_input_values(self) -> list[ValueBase]:
        """Include rebind records and region args for cloning/substitution.

        Same rationale as ``ForOperation.all_input_values``: rebind
        records and region arguments reference body/pre-loop values by
        identity, so inline cloning must remap them in lockstep with
        body operands.

        Returns:
            list[ValueBase]: Base input values plus rebind-record and
                region-argument values.
        """
        values = super().all_input_values()
        values.extend(self.captures)
        values.extend(_rebind_input_values(self.loop_carried_rebinds))
        values.extend(_region_arg_values(self.region_args))
        return values

    def replace_values(self, mapping: dict[str, ValueBase]) -> Operation:
        """Substitute operand, rebind-record, and region-arg values.

        Args:
            mapping (dict[str, ValueBase]): UUID-keyed substitution map.

        Returns:
            Operation: The rewritten operation.
        """
        result = super().replace_values(mapping)
        assert isinstance(result, WhileOperation)
        new_rebinds = _replace_rebind_values(result.loop_carried_rebinds, mapping)
        if new_rebinds is not None:
            result = dataclasses.replace(result, loop_carried_rebinds=new_rebinds)
        new_region_args = _replace_region_arg_values(result.region_args, mapping)
        if new_region_args is not None:
            result = dataclasses.replace(result, region_args=new_region_args)
        new_captures = _replace_value_tuple(result.captures, mapping)
        if new_captures is not None:
            result = dataclasses.replace(result, captures=new_captures)
        return result

    @property
    def signature(self) -> Signature:
        result_hints = [
            ParamHint(name=f"result_{i}", type=r.type)
            for i, r in enumerate(self.results)
        ]
        return Signature(
            operands=[
                ParamHint("condition", BlockType()),
                ParamHint("loop_carried", BlockType()),
            ],
            results=result_hints,
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CONTROL


@dataclasses.dataclass
class ForOperation(HasNestedOps, Operation):
    """Represents a for loop operation.

    Example:
        for i in range(start, stop, step):
            body

    Attributes:
        loop_var: Display name of the loop variable (e.g., "i"). After
            the UUID-keyed migration this is **display-only**: it appears
            in IR printers, error messages, and the legacy name-fallback
            ``bindings`` view, but never participates in identity or
            lookup. Identity is carried by ``loop_var_value.uuid``.
        loop_var_value: The loop variable's IR ``Value`` (typically a
            ``UInt``). Carries the UUID that identifies this loop
            variable across the IR — the binding written by the emit
            pass and the binding read inside the loop body both key on
            ``loop_var_value.uuid``. ``None`` only for legacy IR
            constructed before the migration; new construction must
            always provide it.
        operations: List of operations in the loop body
        region_args: Explicit loop-carried values (see ``RegionArg``).
            Each entry's ``result`` also appears in ``results``.
        operands[0]: start (UInt type)
        operands[1]: stop (UInt type)
        operands[2]: step (UInt type)
    """

    loop_var: str = ""
    loop_var_value: Value | None = None
    operations: list[Operation] = dataclasses.field(default_factory=list)
    loop_carried_rebinds: tuple[LoopCarriedRebind, ...] = ()
    region_args: tuple[RegionArg, ...] = ()
    captures: tuple[ValueBase, ...] = ()

    def nested_op_lists(self) -> list[list[Operation]]:
        return [self.operations]

    def rebuild_nested(self, new_lists: list[list[Operation]]) -> Operation:
        return dataclasses.replace(self, operations=new_lists[0])

    def nested_regions(self) -> tuple[Region, ...]:
        """Return the range-loop body with its explicit interface.

        Returns:
            tuple[Region, ...]: One body region containing the induction
                value, carried-value formals, captures, and carried yields.
        """
        block_args: list[ValueBase] = []
        if self.loop_var_value is not None:
            block_args.append(self.loop_var_value)
        block_args.extend(arg.block_arg for arg in self.region_args)
        return (
            Region(
                operations=tuple(self.operations),
                block_args=tuple(block_args),
                captures=self.captures,
                yields=tuple(arg.yielded for arg in self.region_args),
            ),
        )

    def rebuild_regions(self, regions: typing.Sequence[Region]) -> Operation:
        """Rebuild the range-loop body and complete boundary interface.

        Args:
            regions (Sequence[Region]): Exactly one replacement body region.

        Returns:
            Operation: Rebuilt range-loop operation.

        Raises:
            ValueError: If arity or boundary value kinds are inconsistent.
        """
        if len(regions) != 1:
            raise ValueError(f"ForOperation expects 1 region, got {len(regions)}")
        region = regions[0]
        formals = _require_region_values(
            region.block_args, label="ForOperation block arguments"
        )
        offset = 1 if self.loop_var_value is not None else 0
        if len(formals) != offset + len(self.region_args):
            raise ValueError(
                "ForOperation block arguments must contain its induction "
                "value followed by region_args"
            )
        yields = _require_region_values(region.yields, label="ForOperation yields")
        if len(yields) != len(self.region_args):
            raise ValueError("ForOperation yield count must match region_args")
        region_args = tuple(
            dataclasses.replace(arg, block_arg=block_arg, yielded=yielded)
            for arg, block_arg, yielded in zip(
                self.region_args,
                formals[offset:],
                yields,
                strict=True,
            )
        )
        return dataclasses.replace(
            self,
            loop_var_value=formals[0] if offset else None,
            operations=list(region.operations),
            region_args=region_args,
            captures=tuple(region.captures),
        )

    def all_input_values(self) -> list[ValueBase]:
        """Include ``loop_var_value`` so cloning/substitution stays consistent.

        Without this override, ``UUIDRemapper`` would clone every body
        reference to the loop variable to a fresh UUID, but leave
        ``loop_var_value`` pointing at the un-cloned original — emit-time
        UUID-keyed lookups for the loop variable would then miss.
        Loop-carried rebind records and region arguments are included
        for the same reason.
        """
        values = super().all_input_values()
        if self.loop_var_value is not None:
            values.append(self.loop_var_value)
        values.extend(self.captures)
        values.extend(_rebind_input_values(self.loop_carried_rebinds))
        values.extend(_region_arg_values(self.region_args))
        return values

    def replace_values(self, mapping: dict[str, ValueBase]) -> Operation:
        result = super().replace_values(mapping)
        assert isinstance(result, ForOperation)
        if result.loop_var_value is not None and result.loop_var_value.uuid in mapping:
            mapped = mapping[result.loop_var_value.uuid]
            if isinstance(mapped, Value):
                result = dataclasses.replace(result, loop_var_value=mapped)
        new_rebinds = _replace_rebind_values(result.loop_carried_rebinds, mapping)
        if new_rebinds is not None:
            result = dataclasses.replace(result, loop_carried_rebinds=new_rebinds)
        new_region_args = _replace_region_arg_values(result.region_args, mapping)
        if new_region_args is not None:
            result = dataclasses.replace(result, region_args=new_region_args)
        new_captures = _replace_value_tuple(result.captures, mapping)
        if new_captures is not None:
            result = dataclasses.replace(result, captures=new_captures)
        return result

    @property
    def signature(self) -> Signature:
        result_hints = [
            ParamHint(name=f"result_{i}", type=r.type)
            for i, r in enumerate(self.results)
        ]
        return Signature(
            operands=[
                ParamHint("start", UIntType()),
                ParamHint("stop", UIntType()),
                ParamHint("step", UIntType()),
            ],
            results=result_hints,
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CONTROL


@dataclasses.dataclass
class ForItemsOperation(HasNestedOps, Operation):
    """Represents iteration over dict/iterable items.

    Example:
        for (i, j), Jij in qmc.items(ising):
            body

    Attributes:
        key_vars: Display names of key unpacking variables (e.g.,
            ``["i", "j"]``). Display-only; identity comes from
            ``key_var_values``.
        value_var: Display name of value variable. Display-only.
        key_var_values: IR ``Value`` objects for the key variables. For
            scalar keys, one entry per ``key_vars`` entry. For
            ``key_is_vector=True`` (Vector keys), a single
            ``ArrayValue``-typed entry. Carries the UUIDs used to bind
            keys at emit time. ``None`` only for legacy IR.
        value_var_value: IR ``Value`` for the value variable. ``None``
            only for legacy IR.
        operations: List of operations in the loop body.
        region_args: Explicit loop-carried values (see ``RegionArg``).
            Each entry's ``result`` also appears in ``results``.
        operands[0]: The dict/iterable value (DictValue type).

    Note:
        This operation is always unrolled at transpile time since quantum
        backends cannot natively iterate over classical data structures.
    """

    key_vars: list[str] = dataclasses.field(default_factory=list)
    value_var: str = ""
    key_is_vector: bool = False
    key_var_values: tuple[Value, ...] | None = None
    value_var_value: Value | None = None
    operations: list[Operation] = dataclasses.field(default_factory=list)
    loop_carried_rebinds: tuple[LoopCarriedRebind, ...] = ()
    region_args: tuple[RegionArg, ...] = ()
    captures: tuple[ValueBase, ...] = ()

    def nested_op_lists(self) -> list[list[Operation]]:
        return [self.operations]

    def rebuild_nested(self, new_lists: list[list[Operation]]) -> Operation:
        return dataclasses.replace(self, operations=new_lists[0])

    def nested_regions(self) -> tuple[Region, ...]:
        """Return the items-loop body with its explicit interface.

        Returns:
            tuple[Region, ...]: One body region containing key/value formals,
                carried-value formals, captures, and carried yields.
        """
        block_args: list[ValueBase] = []
        if self.key_var_values is not None:
            block_args.extend(self.key_var_values)
        if self.value_var_value is not None:
            block_args.append(self.value_var_value)
        block_args.extend(arg.block_arg for arg in self.region_args)
        return (
            Region(
                operations=tuple(self.operations),
                block_args=tuple(block_args),
                captures=self.captures,
                yields=tuple(arg.yielded for arg in self.region_args),
            ),
        )

    def rebuild_regions(self, regions: typing.Sequence[Region]) -> Operation:
        """Rebuild the items-loop body and complete boundary interface.

        Args:
            regions (Sequence[Region]): Exactly one replacement body region.

        Returns:
            Operation: Rebuilt items-loop operation.

        Raises:
            ValueError: If arity or boundary value kinds are inconsistent.
        """
        if len(regions) != 1:
            raise ValueError(f"ForItemsOperation expects 1 region, got {len(regions)}")
        region = regions[0]
        formals = _require_region_values(
            region.block_args, label="ForItemsOperation block arguments"
        )
        key_count = len(self.key_var_values or ())
        value_count = 1 if self.value_var_value is not None else 0
        offset = key_count + value_count
        if len(formals) != offset + len(self.region_args):
            raise ValueError(
                "ForItemsOperation block arguments must contain item formals "
                "followed by region_args"
            )
        yields = _require_region_values(region.yields, label="ForItemsOperation yields")
        if len(yields) != len(self.region_args):
            raise ValueError("ForItemsOperation yield count must match region_args")
        region_args = tuple(
            dataclasses.replace(arg, block_arg=block_arg, yielded=yielded)
            for arg, block_arg, yielded in zip(
                self.region_args,
                formals[offset:],
                yields,
                strict=True,
            )
        )
        keys = tuple(formals[:key_count]) if self.key_var_values is not None else None
        value_formal = formals[key_count] if value_count else None
        return dataclasses.replace(
            self,
            key_var_values=keys,
            value_var_value=value_formal,
            operations=list(region.operations),
            region_args=region_args,
            captures=tuple(region.captures),
        )

    def all_input_values(self) -> list[ValueBase]:
        """Include the per-key/value ``Value`` fields for cloning/substitution.

        Same rationale as ``ForOperation.all_input_values``: keep the IR
        identity fields in lockstep with body references so UUID-keyed
        lookups stay valid after inline cloning. Loop-carried rebind
        records and region arguments are included for the same reason.
        """
        values = super().all_input_values()
        if self.key_var_values is not None:
            values.extend(self.key_var_values)
        if self.value_var_value is not None:
            values.append(self.value_var_value)
        values.extend(self.captures)
        values.extend(_rebind_input_values(self.loop_carried_rebinds))
        values.extend(_region_arg_values(self.region_args))
        return values

    def replace_values(self, mapping: dict[str, ValueBase]) -> Operation:
        result = super().replace_values(mapping)
        assert isinstance(result, ForItemsOperation)
        # Substitute key_var_values element-wise.
        if result.key_var_values is not None:
            new_keys: list[Value] = []
            keys_changed = False
            for kv in result.key_var_values:
                if kv.uuid in mapping:
                    mapped = mapping[kv.uuid]
                    if isinstance(mapped, Value):
                        new_keys.append(mapped)
                        keys_changed = True
                        continue
                new_keys.append(kv)
            if keys_changed:
                result = dataclasses.replace(result, key_var_values=tuple(new_keys))
        # Substitute value_var_value.
        if (
            result.value_var_value is not None
            and result.value_var_value.uuid in mapping
        ):
            mapped = mapping[result.value_var_value.uuid]
            if isinstance(mapped, Value):
                result = dataclasses.replace(result, value_var_value=mapped)
        new_rebinds = _replace_rebind_values(result.loop_carried_rebinds, mapping)
        if new_rebinds is not None:
            result = dataclasses.replace(result, loop_carried_rebinds=new_rebinds)
        new_region_args = _replace_region_arg_values(result.region_args, mapping)
        if new_region_args is not None:
            result = dataclasses.replace(result, region_args=new_region_args)
        new_captures = _replace_value_tuple(result.captures, mapping)
        if new_captures is not None:
            result = dataclasses.replace(result, captures=new_captures)
        return result

    @property
    def signature(self) -> Signature:
        # Signature is flexible - operand is the dict/iterable being iterated
        result_hints = [
            ParamHint(name=f"result_{i}", type=r.type)
            for i, r in enumerate(self.results)
        ]
        return Signature(
            operands=[ParamHint("iterable", BlockType())],  # BlockType as placeholder
            results=result_hints,
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CONTROL


def validate_region_args(
    op: ForOperation | ForItemsOperation | WhileOperation,
) -> tuple[RegionArg, ...]:
    """Validate the SSA identities owned by a loop's region arguments.

    A loop owns several definition namespaces: its iteration variables,
    every ``RegionArg.block_arg``, and every ``RegionArg.result``.  Those
    identities must be pairwise disjoint.  Otherwise different stages can
    assign incompatible meanings to one UUID: a UUID-keyed environment has
    only one slot, so binding either the iteration variable or the carried
    value overwrites the other and makes both reads observe the same value.

    Args:
        op (ForOperation | ForItemsOperation | WhileOperation): Loop
            operation whose region arguments should be validated.

    Returns:
        tuple[RegionArg, ...]: The validated ``op.region_args`` tuple.

    Raises:
        ValueError: If result counts or positions disagree, slot types differ,
            or any loop-owned definition identity collides with another
            definition or with a region initializer/body yield.
    """
    region_args = op.region_args
    if len(region_args) != len(op.results):
        raise ValueError(
            "Loop RegionArg data is inconsistent: "
            f"{len(region_args)} region args for {len(op.results)} results."
        )

    block_arg_uuids: set[str] = set()
    result_uuids: set[str] = set()
    init_uuids = {arg.init.uuid for arg in region_args}
    yielded_uuids = {arg.yielded.uuid for arg in region_args}
    for index, (arg, result) in enumerate(zip(region_args, op.results, strict=True)):
        if arg.result.uuid != result.uuid:
            raise ValueError(
                "Loop RegionArg data is inconsistent: "
                f"region_args[{index}].result does not match results[{index}]."
            )
        if not (
            arg.init.type == arg.block_arg.type == arg.yielded.type == arg.result.type
        ):
            raise ValueError(
                f"Loop RegionArg '{arg.var_name}' has mismatched value types: "
                f"init={arg.init.type}, block_arg={arg.block_arg.type}, "
                f"yielded={arg.yielded.type}, result={arg.result.type}."
            )
        if arg.block_arg.uuid in block_arg_uuids:
            raise ValueError(
                "Loop RegionArg data contains duplicate block-argument identities."
            )
        if arg.result.uuid in result_uuids:
            raise ValueError(
                "Loop RegionArg data contains duplicate result identities."
            )
        block_arg_uuids.add(arg.block_arg.uuid)
        result_uuids.add(arg.result.uuid)

    if block_arg_uuids & result_uuids:
        raise ValueError(
            "Loop RegionArg block-argument and result identities must be disjoint."
        )
    if block_arg_uuids & init_uuids:
        raise ValueError(
            "Loop RegionArg block arguments must not reuse initializer identities."
        )
    if result_uuids & init_uuids:
        raise ValueError(
            "Loop RegionArg results must not reuse initializer identities."
        )
    if result_uuids & yielded_uuids:
        raise ValueError("Loop RegionArg results must not reuse body-yield identities.")

    loop_formal_uuids: set[str] = set()
    if isinstance(op, ForOperation):
        if op.loop_var_value is not None:
            loop_formal_uuids.add(op.loop_var_value.uuid)
    elif isinstance(op, ForItemsOperation):
        if op.key_var_values is not None:
            for key_value in op.key_var_values:
                loop_formal_uuids.add(key_value.uuid)
                if hasattr(key_value, "shape"):
                    loop_formal_uuids.update(dim.uuid for dim in key_value.shape)
        if op.value_var_value is not None:
            loop_formal_uuids.add(op.value_var_value.uuid)

    collisions = loop_formal_uuids & (block_arg_uuids | result_uuids)
    if collisions:
        raise ValueError(
            "Loop RegionArg block/result identities must be disjoint from "
            "loop-variable, item-key, item-value, and key-shape identities: "
            f"{sorted(collisions)}."
        )
    if loop_formal_uuids & init_uuids:
        raise ValueError(
            "Loop RegionArg initializers must not reuse a loop-formal "
            "identity (loop variable, item key/value, or key shape); an "
            "initializer is a pre-loop value the loop reads, never a "
            "definition the loop owns."
        )
    return region_args


@dataclasses.dataclass(frozen=True)
class BranchRebind:
    """Trace-time record of a quantum variable rebound inside an if branch.

    The frontend's branch tracing merges only the *new* branch values
    through merge operations; when both branches rebind a variable, the
    value the variable held before the branch no longer appears anywhere
    in the ``IfOperation``. These records preserve that pre-branch
    binding so the transpiler's control-flow discard check
    (``reject_control_flow_quantum_discard`` in
    ``qamomile.circuit.transpiler.passes.analyze``) can verify that the
    pre-branch quantum state is consumed or carried on every runtime
    execution path instead of being silently dropped.

    Attributes:
        var_name (str): Display name of the rebound Python variable.
            Used only for error messages.
        before (Value): The variable's IR value at branch entry (the
            state that is dropped on a rebinding path unless that branch
            consumes it or merges it out through a merge).
        rebound_in_true (bool): True when the true branch left the
            variable bound to a different IR value.
        rebound_in_false (bool): True when the false branch left the
            variable bound to a different IR value.
    """

    var_name: str
    before: Value
    rebound_in_true: bool
    rebound_in_false: bool


def _branch_rebind_input_values(
    rebinds: tuple[BranchRebind, ...],
) -> list[ValueBase]:
    """Collect the Value fields of branch rebind records.

    Args:
        rebinds (tuple[BranchRebind, ...]): Records attached to an
            ``IfOperation``.

    Returns:
        list[ValueBase]: The ``before`` value of every record, in order.
    """
    return [rebind.before for rebind in rebinds]


def _replace_branch_rebind_values(
    rebinds: tuple[BranchRebind, ...],
    mapping: dict[str, ValueBase],
) -> tuple[BranchRebind, ...] | None:
    """Substitute branch rebind record values through a UUID mapping.

    Args:
        rebinds (tuple[BranchRebind, ...]): Records to rewrite.
        mapping (dict[str, ValueBase]): UUID-keyed substitution map.

    Returns:
        tuple[BranchRebind, ...] | None: Rewritten records, or ``None``
            when nothing changed.
    """
    new_rebinds: list[BranchRebind] = []
    changed = False
    for rebind in rebinds:
        mapped_before = mapping.get(rebind.before.uuid)
        if isinstance(mapped_before, Value):
            new_rebinds.append(dataclasses.replace(rebind, before=mapped_before))
            changed = True
        else:
            new_rebinds.append(rebind)
    if not changed:
        return None
    return tuple(new_rebinds)


@dataclasses.dataclass
class IfOperation(HasNestedOps, Operation):
    """Represents an if-else conditional operation.

    Example:
        if condition:
            true_body
        else:
            false_body

    Attributes:
        true_operations: List of operations in the true branch
        false_operations: List of operations in the false branch (may be empty)
        true_yields: Values the true branch yields into the merges,
            index-aligned with ``results`` (``true_yields[i]`` merges
            into ``results[i]``)
        false_yields: Values the false branch yields into the merges,
            index-aligned with ``results``
        branch_rebinds: Trace-time records of quantum variables whose
            binding changed in a branch (see ``BranchRebind``); consumed
            by the transpiler's branch-discard check
        operands[0]: condition (Bit type from measurement or comparison)
        results: Merged output values, one per yield pair

    Note:
        Yields are deliberately NOT operands: the affine-type walk counts
        quantum operands as consumes, and an identity merge legitimately
        carries the same quantum Value on both sides, which would
        double-count as a second consume. Generic passes reach the yields
        through the ``all_input_values`` / ``replace_values`` overrides
        instead.
    """

    true_operations: list[Operation] = dataclasses.field(default_factory=list)
    false_operations: list[Operation] = dataclasses.field(default_factory=list)
    true_yields: list[Value] = dataclasses.field(default_factory=list)
    false_yields: list[Value] = dataclasses.field(default_factory=list)
    branch_rebinds: tuple[BranchRebind, ...] = ()
    true_captures: tuple[ValueBase, ...] = ()
    false_captures: tuple[ValueBase, ...] = ()

    def nested_op_lists(self) -> list[list[Operation]]:
        """Return the two branch bodies (merge yields are not operations).

        Returns:
            list[list[Operation]]: ``[true_operations, false_operations]``.
                The branch-merge yields are values, not operations, so
                they are intentionally absent here.
        """
        return [self.true_operations, self.false_operations]

    def rebuild_nested(self, new_lists: list[list[Operation]]) -> Operation:
        """Return a copy with the true and false branch bodies replaced.

        Args:
            new_lists (list[list[Operation]]): The replacement branch
                bodies in ``nested_op_lists`` order
                (``[true_operations, false_operations]``).

        Returns:
            Operation: A copy of this if-else with the branch bodies
                swapped and all other fields (yields, rebinds) preserved.
        """
        return dataclasses.replace(
            self,
            true_operations=new_lists[0],
            false_operations=new_lists[1],
        )

    def nested_regions(self) -> tuple[Region, ...]:
        """Return the true and false branch interfaces.

        Returns:
            tuple[Region, ...]: True and false regions in that order, with
                branch-local captures and merge yields.
        """
        return (
            Region(
                operations=tuple(self.true_operations),
                captures=self.true_captures,
                yields=tuple(self.true_yields),
            ),
            Region(
                operations=tuple(self.false_operations),
                captures=self.false_captures,
                yields=tuple(self.false_yields),
            ),
        )

    def rebuild_regions(self, regions: typing.Sequence[Region]) -> Operation:
        """Rebuild both branches and their complete boundary interfaces.

        Args:
            regions (Sequence[Region]): True and false replacement regions.

        Returns:
            Operation: Rebuilt conditional operation.

        Raises:
            ValueError: If region count, block arguments, or yield signatures
                are inconsistent.
        """
        if len(regions) != 2:
            raise ValueError(f"IfOperation expects 2 regions, got {len(regions)}")
        true_region, false_region = regions
        if true_region.block_args or false_region.block_args:
            raise ValueError("IfOperation regions do not define block arguments")
        if len(true_region.yields) != len(self.results) or len(
            false_region.yields
        ) != len(self.results):
            raise ValueError("IfOperation branch yield counts must match results")
        true_yields = _require_region_values(
            true_region.yields, label="IfOperation true yields"
        )
        false_yields = _require_region_values(
            false_region.yields, label="IfOperation false yields"
        )
        return dataclasses.replace(
            self,
            true_operations=list(true_region.operations),
            false_operations=list(false_region.operations),
            true_captures=tuple(true_region.captures),
            false_captures=tuple(false_region.captures),
            true_yields=list(true_yields),
            false_yields=list(false_yields),
        )

    def all_input_values(self) -> list[ValueBase]:
        """Include branch-yield values and rebind records for cloning/substitution.

        The yields are subclass-specific Value fields (not operands —
        see the class docstring), so generic passes reach them through
        this override, mirroring ``ForItemsOperation.key_var_values``.
        Branch rebind records follow the loop operations' rationale: the
        recorded pre-branch values reference program values by identity,
        so inline cloning must remap them in lockstep with operands.
        Read-based checks must not treat the records as reads (see
        ``_op_read_uuids`` in the analyze pass module).

        Returns:
            list[ValueBase]: Base input values plus the true/false
                yields and rebind-record values.
        """
        values = super().all_input_values()
        values.extend(self.true_captures)
        values.extend(self.false_captures)
        values.extend(self.true_yields)
        values.extend(self.false_yields)
        values.extend(_branch_rebind_input_values(self.branch_rebinds))
        return values

    def replace_values(self, mapping: dict[str, ValueBase]) -> Operation:
        """Substitute operand, result, branch-yield, and rebind-record values.

        Args:
            mapping (dict[str, ValueBase]): UUID-keyed substitution map.

        Returns:
            Operation: The rewritten operation.
        """
        result = super().replace_values(mapping)
        assert isinstance(result, IfOperation)
        new_true = _replace_yield_values(result.true_yields, mapping)
        if new_true is not None:
            result = dataclasses.replace(result, true_yields=new_true)
        new_false = _replace_yield_values(result.false_yields, mapping)
        if new_false is not None:
            result = dataclasses.replace(result, false_yields=new_false)
        new_rebinds = _replace_branch_rebind_values(result.branch_rebinds, mapping)
        if new_rebinds is not None:
            result = dataclasses.replace(result, branch_rebinds=new_rebinds)
        new_true_captures = _replace_value_tuple(result.true_captures, mapping)
        if new_true_captures is not None:
            result = dataclasses.replace(result, true_captures=new_true_captures)
        new_false_captures = _replace_value_tuple(result.false_captures, mapping)
        if new_false_captures is not None:
            result = dataclasses.replace(result, false_captures=new_false_captures)
        return result

    @property
    def condition(self) -> Value:
        return self.operands[0]

    def iter_merges(self) -> typing.Iterator[IfMerge]:
        """Iterate the branch-merge slots of this if-else.

        This is the single read API for merge semantics: passes must
        consume merges through it (never through the yield lists
        directly) so the underlying storage can change without touching
        consumers.

        Yields:
            IfMerge: One entry per merged output, in result order.

        Raises:
            RuntimeError: If the stored merge data is internally
                inconsistent (the yield-list lengths differ from the
                result count, or the condition operand is missing while
                merges are attached). This indicates IR corruption, not
                a user error. The per-merge corruption modes of the old
                embedded-operation storage (a foreign entry, a malformed
                or mismatched condition, a result copy diverging from
                ``results[i]``) cannot be represented in the yield-list
                storage and need no checks.
        """
        if not (len(self.true_yields) == len(self.false_yields) == len(self.results)):
            raise RuntimeError(
                "[FOR DEVELOPER] IfOperation merge data is inconsistent: "
                f"{len(self.true_yields)} true_yields / "
                f"{len(self.false_yields)} false_yields for "
                f"{len(self.results)} results. Merges must be created "
                "through add_merge()."
            )
        if self.results and not self.operands:
            raise RuntimeError(
                "[FOR DEVELOPER] IfOperation merge data is inconsistent: "
                "merges are attached but the condition operand is missing."
            )
        for i, (true_value, false_value, result) in enumerate(
            zip(self.true_yields, self.false_yields, self.results, strict=True)
        ):
            yield IfMerge(
                index=i,
                true_value=true_value,
                false_value=false_value,
                result=result,
            )

    def add_merge(self, true_value: Value, false_value: Value, result: Value) -> None:
        """Append a branch-merge slot to this if-else.

        The only sanctioned construction path for merges: it keeps the
        yield lists and ``results`` index-aligned so ``iter_merges`` can
        rely on the invariants it checks.

        Args:
            true_value (Value): Value selected when the condition is true.
            false_value (Value): Value selected when the condition is
                false. Must have the same type as ``true_value``.
            result (Value): Fresh SSA value representing the merged output.
                Must have the same type as the branch values.

        Raises:
            RuntimeError: If the condition operand has not been attached to
                this operation yet (``operands[0]`` must exist before
                merges are added), or the branch / result types do not
                match.
        """
        if not self.operands:
            raise RuntimeError(
                "[FOR DEVELOPER] IfOperation.add_merge requires the "
                "condition operand to be attached first."
            )
        if false_value.type != true_value.type or result.type != true_value.type:
            raise RuntimeError(
                "[FOR DEVELOPER] IfOperation.add_merge requires matching "
                "branch and result types; got "
                f"true={true_value.type.label()}, "
                f"false={false_value.type.label()}, "
                f"result={result.type.label()}."
            )
        self.true_yields.append(true_value)
        self.false_yields.append(false_value)
        self.results.append(result)

    @property
    def signature(self) -> Signature:
        result_hints = [
            ParamHint(name=f"result_{i}", type=r.type)
            for i, r in enumerate(self.results)
        ]
        return Signature(
            operands=[ParamHint("condition", BitType())], results=result_hints
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CONTROL


def genuine_input_values(op: Operation) -> list[ValueBase]:
    """Return an operation's input values that count as genuine reads.

    Structured operations derive reads from their explicit region interface:
    enclosing operands, captures, loop initializers, and region yields.
    Block arguments and operation results are definitions, while legacy
    rebind records are diagnostics rather than dataflow. Leaf operations keep
    their ordinary ``all_input_values`` contract.

    Args:
        op (Operation): Operation to inspect.

    Returns:
        list[ValueBase]: Semantic reads in interface order.
    """
    if isinstance(op, WhileOperation):
        while_values: list[ValueBase] = list(op.operands[:1])
        while_values.extend(op.captures)
        while_values.extend(region_arg.init for region_arg in op.region_args)
        while_values.extend(op.nested_regions()[0].yields)
        return while_values
    if isinstance(op, (ForOperation, ForItemsOperation)):
        loop_values: list[ValueBase] = list(op.operands)
        loop_values.extend(op.captures)
        loop_values.extend(region_arg.init for region_arg in op.region_args)
        loop_values.extend(op.nested_regions()[0].yields)
        return loop_values
    if isinstance(op, IfOperation):
        branch_values: list[ValueBase] = list(op.operands)
        for region in op.nested_regions():
            branch_values.extend(region.captures)
            branch_values.extend(region.yields)
        return branch_values
    return list(op.all_input_values())
