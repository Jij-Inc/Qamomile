from __future__ import annotations

import dataclasses
import typing
from typing import cast

from qamomile.circuit.ir.types.primitives import BitType, BlockType, UIntType
from qamomile.circuit.ir.value import Value, ValueBase

from .operation import Operation, OperationKind, ParamHint, Signature

if typing.TYPE_CHECKING:
    from .arithmetic_operations import PhiOp


class HasNestedOps:
    """Mixin for operations that contain nested operation lists.

    Subclasses implement ``nested_op_lists()`` and ``rebuild_nested()``
    so that generic passes can recurse into control flow without
    isinstance chains.
    """

    def nested_op_lists(self) -> list[list[Operation]]:
        """Return all nested operation lists in this control flow op."""
        raise NotImplementedError

    def rebuild_nested(self, new_lists: list[list[Operation]]) -> Operation:
        """Return a copy with nested operation lists replaced.

        ``new_lists`` must have the same length/order as ``nested_op_lists()``.
        """
        raise NotImplementedError


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
        yielded (Value): The body-produced value carried into the next
            iteration (produced by an operation inside ``operations``).
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
        before (Value): The variable's IR value before the loop body ran
            (for classical records, the stale value the traced body
            reads; for quantum records, the incoming state a rebinding
            iteration would discard).
        after (Value): The variable's IR value after the loop body ran
            (typically a ``BinOp`` result, an ``IfOperation`` phi
            output, or a fresh quantum allocation).
        before_synthesized (bool): True when the pre-loop value was a
            plain Python number with no IR identity (e.g. ``total = 0``),
            in which case ``before`` is a synthesized constant ``Value``
            that does not appear in the body's dataflow. Always False
            for quantum records.
    """

    var_name: str
    before: Value
    after: Value
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
        if isinstance(mapped_before, Value):
            before = mapped_before
        mapped_after = mapping.get(after.uuid)
        if isinstance(mapped_after, Value):
            after = mapped_after
        if before is not rebind.before or after is not rebind.after:
            rebind = dataclasses.replace(rebind, before=before, after=after)
            changed = True
        new_rebinds.append(rebind)
    return tuple(new_rebinds) if changed else None


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

    def nested_op_lists(self) -> list[list[Operation]]:
        return [self.operations]

    def rebuild_nested(self, new_lists: list[list[Operation]]) -> Operation:
        return dataclasses.replace(self, operations=new_lists[0])

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

    def nested_op_lists(self) -> list[list[Operation]]:
        return [self.operations]

    def rebuild_nested(self, new_lists: list[list[Operation]]) -> Operation:
        return dataclasses.replace(self, operations=new_lists[0])

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

    def nested_op_lists(self) -> list[list[Operation]]:
        return [self.operations]

    def rebuild_nested(self, new_lists: list[list[Operation]]) -> Operation:
        return dataclasses.replace(self, operations=new_lists[0])

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


@dataclasses.dataclass(frozen=True)
class BranchRebind:
    """Trace-time record of a quantum variable rebound inside an if branch.

    The frontend's branch tracing merges only the *new* branch values
    through phi operations; when both branches rebind a variable, the
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
            consumes it or merges it out through a phi).
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
        phi_ops: List of PhiOp instances merging values from both branches
        branch_rebinds: Trace-time records of quantum variables whose
            binding changed in a branch (see ``BranchRebind``); consumed
            by the transpiler's branch-discard check
        operands[0]: condition (Bit type from measurement or comparison)
        results: Phi-merged output values from both branches
    """

    true_operations: list[Operation] = dataclasses.field(default_factory=list)
    false_operations: list[Operation] = dataclasses.field(default_factory=list)
    phi_ops: list[PhiOp] = dataclasses.field(default_factory=list)
    branch_rebinds: tuple[BranchRebind, ...] = ()

    def nested_op_lists(self) -> list[list[Operation]]:
        return [
            self.true_operations,
            self.false_operations,
            cast(list[Operation], self.phi_ops),
        ]

    def rebuild_nested(self, new_lists: list[list[Operation]]) -> Operation:
        return dataclasses.replace(
            self,
            true_operations=new_lists[0],
            false_operations=new_lists[1],
            phi_ops=cast("list[PhiOp]", new_lists[2]),
        )

    def all_input_values(self) -> list[ValueBase]:
        """Include branch rebind records for cloning/substitution.

        Same rationale as the loop operations' rebind records: the
        recorded pre-branch values reference program values by identity,
        so inline cloning must remap them in lockstep with operands.
        Read-based checks must not treat them as reads (see
        ``_op_read_uuids`` in the analyze pass module).

        Returns:
            list[ValueBase]: Base input values plus rebind-record values.
        """
        values = super().all_input_values()
        values.extend(_branch_rebind_input_values(self.branch_rebinds))
        return values

    def replace_values(self, mapping: dict[str, ValueBase]) -> Operation:
        """Substitute operand and rebind-record values.

        Args:
            mapping (dict[str, ValueBase]): UUID-keyed substitution map.

        Returns:
            Operation: The rewritten operation.
        """
        result = super().replace_values(mapping)
        assert isinstance(result, IfOperation)
        new_rebinds = _replace_branch_rebind_values(result.branch_rebinds, mapping)
        if new_rebinds is not None:
            result = dataclasses.replace(result, branch_rebinds=new_rebinds)
        return result

    @property
    def condition(self) -> Value:
        return self.operands[0]

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
