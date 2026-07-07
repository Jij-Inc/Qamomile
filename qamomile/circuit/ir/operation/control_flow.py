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


class LoopCarry(typing.NamedTuple):
    """One loop-carried classical value slot of a loop operation.

    A loop body that updates a classical scalar per iteration
    (``total = total + i``) needs the previous iteration's value, which
    a traced-once body cannot reference directly. A ``LoopCarry`` slot
    formalizes that back-edge the same way MLIR's ``scf.for``
    ``iter_args`` do: the loop enters with ``iter_arg``, each iteration
    reads the previous iteration's value through the ``body_arg``
    formal, computes ``body_yield`` for the next iteration, and the
    loop's ``result`` holds the final value (``iter_arg`` when the loop
    runs zero times). Post-loop code reads ``result``.

    Attributes:
        index (int): Position of this carry among the loop's results.
        var_name (str): Display name of the carried Python variable.
            Used by the frontend's post-loop handle rebinding and by
            error messages.
        iter_arg (Value): The value entering the first iteration — a
            reference to pre-loop dataflow (or an embedded constant for
            plain-Python initializations like ``total = 0``).
        body_arg (Value): Fresh formal the body reads as "the previous
            iteration's value". The frontend substitutes every body
            reference to the pre-loop value with this formal.
        body_yield (Value): The value the body computes for the next
            iteration (the variable's post-body binding).
        result (Value): The loop operation's merged output
            (``op.results[index]``), holding the final carried value.
    """

    # The field intentionally shadows ``tuple.index`` (allowed for
    # NamedTuple fields); nothing calls the method on carry slots.
    index: int  # type: ignore[assignment]
    var_name: str
    iter_arg: Value
    body_arg: Value
    body_yield: Value
    result: Value


class LoopCarryMixin:
    """Shared loop-carried-value storage for loop operations.

    Mixed into ``ForOperation`` / ``ForItemsOperation`` /
    ``WhileOperation``. Subclasses must be dataclasses defining
    ``carried_names: list[str]``, ``iter_args: list[Value]``,
    ``body_args: list[Value]``, ``body_yields: list[Value]`` fields and
    inherit ``results`` from ``Operation``; the mixin keeps those lists
    and ``results`` index-aligned and exposes the read/write API
    (``iter_carries`` / ``add_carry``).
    """

    carried_names: list[str]
    iter_args: list[Value]
    body_args: list[Value]
    body_yields: list[Value]
    results: list[Value]

    def iter_carries(self) -> typing.Iterator[LoopCarry]:
        """Iterate the loop-carried value slots of this loop.

        The single read API for carry semantics: passes must consume
        carries through it (never through the parallel lists directly)
        so the underlying storage can change without touching consumers.

        Yields:
            LoopCarry: One entry per carried value, in result order.

        Raises:
            RuntimeError: If the stored carry data is internally
                inconsistent (list lengths differ from the result
                count). This indicates IR corruption, not a user error.
        """
        counts = {
            len(self.carried_names),
            len(self.iter_args),
            len(self.body_args),
            len(self.body_yields),
            len(self.results),
        }
        if counts != {len(self.results)}:
            raise RuntimeError(
                "[FOR DEVELOPER] Loop carry data is inconsistent: "
                f"{len(self.carried_names)} names / {len(self.iter_args)} "
                f"iter_args / {len(self.body_args)} body_args / "
                f"{len(self.body_yields)} body_yields for "
                f"{len(self.results)} results. Carries must be created "
                "through add_carry()."
            )
        for i, (name, iter_arg, body_arg, body_yield, result) in enumerate(
            zip(
                self.carried_names,
                self.iter_args,
                self.body_args,
                self.body_yields,
                self.results,
                strict=True,
            )
        ):
            yield LoopCarry(
                index=i,
                var_name=name,
                iter_arg=iter_arg,
                body_arg=body_arg,
                body_yield=body_yield,
                result=result,
            )

    def add_carry(
        self,
        var_name: str,
        iter_arg: Value,
        body_arg: Value,
        body_yield: Value,
        result: Value,
    ) -> None:
        """Append a loop-carried value slot to this loop.

        The only sanctioned construction path for carries: it keeps the
        parallel lists and ``results`` index-aligned so ``iter_carries``
        can rely on the invariants it checks.

        Args:
            var_name (str): Display name of the carried variable.
            iter_arg (Value): Value entering the first iteration.
            body_arg (Value): Fresh formal the body reads as the
                previous iteration's value.
            body_yield (Value): Value the body computes for the next
                iteration. Must have the same type as ``body_arg``.
            result (Value): Fresh SSA value representing the final
                carried value.
        """
        self.carried_names.append(var_name)
        self.iter_args.append(iter_arg)
        self.body_args.append(body_arg)
        self.body_yields.append(body_yield)
        self.results.append(result)

    def _carry_input_values(self) -> list[ValueBase]:
        """Collect the input-side Value fields of every carry slot.

        ``results`` entries are outputs and are reached through the base
        ``Operation.results`` protocol instead.

        Returns:
            list[ValueBase]: ``iter_arg``, ``body_arg``, and
                ``body_yield`` of every slot, in order.
        """
        values: list[ValueBase] = []
        for iter_arg, body_arg, body_yield in zip(
            self.iter_args, self.body_args, self.body_yields, strict=True
        ):
            values.append(iter_arg)
            values.append(body_arg)
            values.append(body_yield)
        return values

    def _replace_carry_fields(self, mapping: dict[str, ValueBase]) -> typing.Any:
        """Substitute carry-slot values through a UUID mapping.

        Args:
            mapping (dict[str, ValueBase]): UUID-keyed substitution map.

        Returns:
            typing.Any: ``self`` when nothing changed, otherwise a copy
                with substituted carry lists. ``results`` entries are
                rewritten by the base ``Operation.replace_values`` and
                are not touched here.
        """
        new_iter_args = _replace_yield_values(self.iter_args, mapping)
        new_body_args = _replace_yield_values(self.body_args, mapping)
        new_body_yields = _replace_yield_values(self.body_yields, mapping)
        if new_iter_args is None and new_body_args is None and new_body_yields is None:
            return self
        return dataclasses.replace(
            typing.cast(typing.Any, self),
            iter_args=new_iter_args if new_iter_args is not None else self.iter_args,
            body_args=new_body_args if new_body_args is not None else self.body_args,
            body_yields=(
                new_body_yields if new_body_yields is not None else self.body_yields
            ),
        )


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
class WhileOperation(LoopCarryMixin, HasNestedOps, Operation):
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
        carried_names: Display names of the loop-carried classical
            values, index-aligned with ``iter_args`` / ``body_args`` /
            ``body_yields`` / ``results`` (see ``LoopCarry``). A while
            loop's trip count is a runtime measurement outcome, so no
            backend can realize a classical carry today — carries on a
            while are rejected with a targeted error before emit.
        iter_args: Values entering the first iteration, one per carry.
        body_args: Fresh formals the body reads as the previous
            iteration's values, one per carry.
        body_yields: Values the body computes for the next iteration,
            one per carry.
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
    carried_names: list[str] = dataclasses.field(default_factory=list)
    iter_args: list[Value] = dataclasses.field(default_factory=list)
    body_args: list[Value] = dataclasses.field(default_factory=list)
    body_yields: list[Value] = dataclasses.field(default_factory=list)

    def nested_op_lists(self) -> list[list[Operation]]:
        return [self.operations]

    def rebuild_nested(self, new_lists: list[list[Operation]]) -> Operation:
        return dataclasses.replace(self, operations=new_lists[0])

    def all_input_values(self) -> list[ValueBase]:
        """Include rebind records and carry slots for cloning/substitution.

        Same rationale as ``ForOperation.all_input_values``: rebind
        records and carry slots reference body/pre-loop values by
        identity, so inline cloning must remap them in lockstep with
        body operands.

        Returns:
            list[ValueBase]: Base input values plus rebind-record and
                carry-slot values.
        """
        values = super().all_input_values()
        values.extend(_rebind_input_values(self.loop_carried_rebinds))
        values.extend(self._carry_input_values())
        return values

    def replace_values(self, mapping: dict[str, ValueBase]) -> Operation:
        """Substitute operand, rebind-record, and carry-slot values.

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
        replaced = result._replace_carry_fields(mapping)
        assert isinstance(replaced, WhileOperation)
        return replaced

    @property
    def signature(self) -> Signature:
        return Signature(
            operands=[
                ParamHint("condition", BlockType()),
                ParamHint("loop_carried", BlockType()),
            ],
            results=[
                ParamHint(name=f"carry_{i}", type=r.type)
                for i, r in enumerate(self.results)
            ],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CONTROL


@dataclasses.dataclass
class ForOperation(LoopCarryMixin, HasNestedOps, Operation):
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
        carried_names: Display names of the loop-carried classical
            values, index-aligned with ``iter_args`` / ``body_args`` /
            ``body_yields`` / ``results`` (see ``LoopCarry``).
        iter_args: Values entering the first iteration, one per carry.
        body_args: Fresh formals the body reads as the previous
            iteration's values, one per carry.
        body_yields: Values the body computes for the next iteration,
            one per carry.
        operands[0]: start (UInt type)
        operands[1]: stop (UInt type)
        operands[2]: step (UInt type)
    """

    loop_var: str = ""
    loop_var_value: Value | None = None
    operations: list[Operation] = dataclasses.field(default_factory=list)
    loop_carried_rebinds: tuple[LoopCarriedRebind, ...] = ()
    carried_names: list[str] = dataclasses.field(default_factory=list)
    iter_args: list[Value] = dataclasses.field(default_factory=list)
    body_args: list[Value] = dataclasses.field(default_factory=list)
    body_yields: list[Value] = dataclasses.field(default_factory=list)

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
        Loop-carried rebind records and carry slots are included for the
        same reason.
        """
        values = super().all_input_values()
        if self.loop_var_value is not None:
            values.append(self.loop_var_value)
        values.extend(_rebind_input_values(self.loop_carried_rebinds))
        values.extend(self._carry_input_values())
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
        replaced = result._replace_carry_fields(mapping)
        assert isinstance(replaced, ForOperation)
        return replaced

    @property
    def signature(self) -> Signature:
        return Signature(
            operands=[
                ParamHint("start", UIntType()),
                ParamHint("stop", UIntType()),
                ParamHint("step", UIntType()),
            ],
            results=[
                ParamHint(name=f"carry_{i}", type=r.type)
                for i, r in enumerate(self.results)
            ],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CONTROL


@dataclasses.dataclass
class ForItemsOperation(LoopCarryMixin, HasNestedOps, Operation):
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
        carried_names: Display names of the loop-carried classical
            values, index-aligned with ``iter_args`` / ``body_args`` /
            ``body_yields`` / ``results`` (see ``LoopCarry``).
        iter_args: Values entering the first iteration, one per carry.
        body_args: Fresh formals the body reads as the previous
            iteration's values, one per carry.
        body_yields: Values the body computes for the next iteration,
            one per carry.
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
    carried_names: list[str] = dataclasses.field(default_factory=list)
    iter_args: list[Value] = dataclasses.field(default_factory=list)
    body_args: list[Value] = dataclasses.field(default_factory=list)
    body_yields: list[Value] = dataclasses.field(default_factory=list)

    def nested_op_lists(self) -> list[list[Operation]]:
        return [self.operations]

    def rebuild_nested(self, new_lists: list[list[Operation]]) -> Operation:
        return dataclasses.replace(self, operations=new_lists[0])

    def all_input_values(self) -> list[ValueBase]:
        """Include the per-key/value ``Value`` fields for cloning/substitution.

        Same rationale as ``ForOperation.all_input_values``: keep the IR
        identity fields in lockstep with body references so UUID-keyed
        lookups stay valid after inline cloning. Loop-carried rebind
        records and carry slots are included for the same reason.
        """
        values = super().all_input_values()
        if self.key_var_values is not None:
            values.extend(self.key_var_values)
        if self.value_var_value is not None:
            values.append(self.value_var_value)
        values.extend(_rebind_input_values(self.loop_carried_rebinds))
        values.extend(self._carry_input_values())
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
        replaced = result._replace_carry_fields(mapping)
        assert isinstance(replaced, ForItemsOperation)
        return replaced

    @property
    def signature(self) -> Signature:
        # Signature is flexible - operand is the dict/iterable being iterated
        return Signature(
            operands=[ParamHint("iterable", BlockType())],  # BlockType as placeholder
            results=[
                ParamHint(name=f"carry_{i}", type=r.type)
                for i, r in enumerate(self.results)
            ],
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

    def nested_op_lists(self) -> list[list[Operation]]:
        """Return the two branch bodies (merge yields are not operations)."""
        return [self.true_operations, self.false_operations]

    def rebuild_nested(self, new_lists: list[list[Operation]]) -> Operation:
        """Return a copy with the true and false branch bodies replaced."""
        return dataclasses.replace(
            self,
            true_operations=new_lists[0],
            false_operations=new_lists[1],
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
                result count). This indicates IR corruption, not a user
                error.
        """
        if not (len(self.true_yields) == len(self.false_yields) == len(self.results)):
            raise RuntimeError(
                "[FOR DEVELOPER] IfOperation merge data is inconsistent: "
                f"{len(self.true_yields)} true_yields / "
                f"{len(self.false_yields)} false_yields for "
                f"{len(self.results)} results. Merges must be created "
                "through add_merge()."
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

        Raises:
            RuntimeError: If the condition operand has not been attached to
                this operation yet (``operands[0]`` must exist before
                merges are added).
        """
        if not self.operands:
            raise RuntimeError(
                "[FOR DEVELOPER] IfOperation.add_merge requires the "
                "condition operand to be attached first."
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
