from __future__ import annotations

import dataclasses
import typing
from typing import cast

from qamomile.circuit.ir.types.primitives import BitType, BlockType, UIntType
from qamomile.circuit.ir.value import Value, ValueBase

from .operation import Operation, OperationKind, ParamHint, Signature

if typing.TYPE_CHECKING:
    from .arithmetic_operations import PhiOp


class IfMerge(typing.NamedTuple):
    """One branch-merge slot of an :class:`IfOperation`.

    An ``IfOperation`` merges each variable touched by its branches back
    into a single SSA value. ``IfMerge`` is the read-side view of one such
    merge slot, decoupling every consumer from how the merge is stored in
    the IR (today a ``PhiOp`` in ``IfOperation.phi_ops``; the storage may
    change without touching consumers).

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
class LoopCarriedRebind:
    """Trace-time record of a variable rebound inside a loop body.

    Two rebind families share this record type, distinguished by the
    type of ``before``:

    - **Classical scalar** (``before`` classical): the frontend traces a
      loop body exactly once, so a Python-level reassignment like
      ``total = total + i`` produces IR whose right-hand side reads the
      fixed pre-loop value instead of the previous iteration's value — a
      loop-carried dependency the IR cannot represent. The transpiler's
      classical loop-carried check rejects these with a targeted error
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

    def nested_op_lists(self) -> list[list[Operation]]:
        return [self.operations]

    def rebuild_nested(self, new_lists: list[list[Operation]]) -> Operation:
        return dataclasses.replace(self, operations=new_lists[0])

    def all_input_values(self) -> list[ValueBase]:
        """Include loop-carried rebind records for cloning/substitution.

        Same rationale as ``ForOperation.all_input_values``: rebind
        records reference body/pre-loop values by identity, so inline
        cloning must remap them in lockstep with body operands.

        Returns:
            list[ValueBase]: Base input values plus rebind-record values.
        """
        values = super().all_input_values()
        values.extend(_rebind_input_values(self.loop_carried_rebinds))
        return values

    def replace_values(self, mapping: dict[str, ValueBase]) -> Operation:
        """Substitute operand and rebind-record values.

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
        return result

    @property
    def signature(self) -> Signature:
        return Signature(
            operands=[
                ParamHint("condition", BlockType()),
                ParamHint("loop_carried", BlockType()),
            ],
            results=[],
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
        operands[0]: start (UInt type)
        operands[1]: stop (UInt type)
        operands[2]: step (UInt type)
    """

    loop_var: str = ""
    loop_var_value: Value | None = None
    operations: list[Operation] = dataclasses.field(default_factory=list)
    loop_carried_rebinds: tuple[LoopCarriedRebind, ...] = ()

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
        Loop-carried rebind records are included for the same reason.
        """
        values = super().all_input_values()
        if self.loop_var_value is not None:
            values.append(self.loop_var_value)
        values.extend(_rebind_input_values(self.loop_carried_rebinds))
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
        return result

    @property
    def signature(self) -> Signature:
        return Signature(
            operands=[
                ParamHint("start", UIntType()),
                ParamHint("stop", UIntType()),
                ParamHint("step", UIntType()),
            ],
            results=[],
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

    def nested_op_lists(self) -> list[list[Operation]]:
        return [self.operations]

    def rebuild_nested(self, new_lists: list[list[Operation]]) -> Operation:
        return dataclasses.replace(self, operations=new_lists[0])

    def all_input_values(self) -> list[ValueBase]:
        """Include the per-key/value ``Value`` fields for cloning/substitution.

        Same rationale as ``ForOperation.all_input_values``: keep the IR
        identity fields in lockstep with body references so UUID-keyed
        lookups stay valid after inline cloning. Loop-carried rebind
        records are included for the same reason.
        """
        values = super().all_input_values()
        if self.key_var_values is not None:
            values.extend(self.key_var_values)
        if self.value_var_value is not None:
            values.append(self.value_var_value)
        values.extend(_rebind_input_values(self.loop_carried_rebinds))
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
        return result

    @property
    def signature(self) -> Signature:
        # Signature is flexible - operand is the dict/iterable being iterated
        return Signature(
            operands=[ParamHint("iterable", BlockType())],  # BlockType as placeholder
            results=[],
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

    def iter_merges(self) -> typing.Iterator[IfMerge]:
        """Iterate the branch-merge slots of this if-else.

        This is the single read API for phi semantics: passes must consume
        merges through it (never through ``phi_ops`` internals) so the
        underlying storage can change without touching consumers.

        Yields:
            IfMerge: One entry per merged output, in result order.

        Raises:
            RuntimeError: If the stored merge data is internally
                inconsistent (merge count differs from result count, a
                stored merge is not a ``PhiOp``, a merge does not have
                exactly condition/true/false operands and one result, a
                merge result does not match the if-operation result at
                the same position, or a merge's condition operand does
                not match the if-operation's condition — including a
                missing condition while merges are attached). Any of
                these indicates IR corruption, not a user error.
        """
        # Runtime import mirrors the TYPE_CHECKING guard above: PhiOp lives
        # in a sibling module that also imports from .operation.
        from .arithmetic_operations import PhiOp

        if len(self.phi_ops) != len(self.results):
            raise RuntimeError(
                "[FOR DEVELOPER] IfOperation merge data is inconsistent: "
                f"{len(self.phi_ops)} phi_ops for {len(self.results)} "
                "results. Merges must be created through add_merge()."
            )
        if self.phi_ops and not self.operands:
            raise RuntimeError(
                "[FOR DEVELOPER] IfOperation merge data is inconsistent: "
                "merges are attached but the condition operand is missing."
            )
        condition = self.operands[0] if self.operands else None
        for i, phi in enumerate(self.phi_ops):
            if not isinstance(phi, PhiOp):
                raise RuntimeError(
                    "[FOR DEVELOPER] IfOperation merge data is inconsistent: "
                    f"phi_ops[{i}] is {type(phi).__name__}, expected PhiOp."
                )
            if len(phi.operands) != 3 or len(phi.results) != 1:
                raise RuntimeError(
                    "[FOR DEVELOPER] IfOperation merge data is inconsistent: "
                    f"phi_ops[{i}] has {len(phi.operands)} operands and "
                    f"{len(phi.results)} results; expected "
                    "[condition, true_value, false_value] -> [output]."
                )
            phi_condition = phi.operands[0]
            if isinstance(phi_condition, ValueBase) and isinstance(
                condition, ValueBase
            ):
                condition_matches = phi_condition.uuid == condition.uuid
            else:
                # Compile-time-constant conditions may be raw Python bools
                # with no IR identity; identity comparison covers them.
                condition_matches = phi_condition is condition
            if not condition_matches:
                raise RuntimeError(
                    "[FOR DEVELOPER] IfOperation merge data is inconsistent: "
                    f"phi_ops[{i}] condition does not match the "
                    "if-operation's condition operand."
                )
            if phi.results[0].uuid != self.results[i].uuid:
                raise RuntimeError(
                    "[FOR DEVELOPER] IfOperation merge data is inconsistent: "
                    f"phi_ops[{i}] output '{phi.results[0].uuid}' does not "
                    f"match results[{i}] '{self.results[i].uuid}'."
                )
            yield IfMerge(
                index=i,
                true_value=phi.operands[1],
                false_value=phi.operands[2],
                result=self.results[i],
            )

    def add_merge(self, true_value: Value, false_value: Value, result: Value) -> None:
        """Append a branch-merge slot to this if-else.

        The only sanctioned construction path for merges: it keeps the
        merge storage (currently a ``PhiOp`` plus the mirrored entry in
        ``results``) consistent so ``iter_merges`` can rely on the
        invariants it checks.

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
        # Runtime import mirrors the TYPE_CHECKING guard above: PhiOp lives
        # in a sibling module that also imports from .operation.
        from .arithmetic_operations import PhiOp

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
        self.phi_ops.append(
            PhiOp(
                operands=[self.condition, true_value, false_value],
                results=[result],
            )
        )
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
