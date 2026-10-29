"""Normalize H2 shot indices and prove table bounds before QIR conversion."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any

from qamomile.circuit.transpiler import ExecutionError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hugr import Hugr, ops
    from hugr.hugr.node_port import Node, OutPort
    from hugr.package import Package

SHOT_SUCCESS_TAG = "__qamomile_h2_shot_success_v1"
_SELECTION_KEY = "qamomile.classical_selection.v1"
_GUARD_KEY = "qamomile.h2_shot_range.v1"
_UINT_MAX = 2**64 - 1
_SHOT_OP = "tket.qsystem.utils.GetCurrentShot"


@dataclass(frozen=True)
class _Interval:
    low: int
    high: int

    def combine(self, other: _Interval, multiply: bool = False) -> _Interval:
        if multiply:
            low, high = self.low * other.low, self.high * other.high
        else:
            low, high = self.low + other.low, self.high + other.high
        if high > _UINT_MAX:
            raise ExecutionError("H2 shot-index arithmetic may overflow UInt64")
        return _Interval(low, high)


def _name(operation: Any) -> str:
    method = getattr(operation, "name", None)
    return method().partition("<")[0] if callable(method) else ""


def _source(graph: Any, node: Any, offset: int) -> Any:
    sources = list(graph.linked_ports(node.inp(offset)))
    if len(sources) != 1:
        raise ExecutionError("H2 shot-range proof requires one source per input")
    return sources[0]


class _Proof:
    def __init__(self, graph: Any, shots: int) -> None:
        self.graph = graph
        self.shots = shots
        self.cache: dict[Any, Any] = {}
        self.active: set[Any] = set()

    def value(self, wire: Any) -> Any:
        if wire in self.cache:
            return self.cache[wire]
        if wire in self.active:
            raise ExecutionError(
                "H2 shot-range proof does not support recursive values"
            )
        self.active.add(wire)
        try:
            value = self._value(wire)
            self.cache[wire] = value
            return value
        finally:
            self.active.remove(wire)

    def interval(self, wire: Any) -> _Interval:
        value = self.value(wire)
        if not isinstance(value, _Interval):
            raise ExecutionError("H2 table indices must be scalar UInt values")
        return value

    def _constant(self, value: Any) -> Any:
        from hugr import val
        from hugr.std.int import IntVal

        if isinstance(value, IntVal) and value.width == 6:
            unsigned = value.to_value().val["value"]
            return _Interval(unsigned, unsigned)
        if isinstance(value, val.UnitSum) and value.typ == val.TRUE.typ:
            return _Interval(value.tag, value.tag)
        if isinstance(value, val.Tuple):
            return tuple(self._constant(item) for item in value.vals)
        raise ExecutionError("H2 shot-range proof supports fixed UInt and Bit tables")

    def _join(self, values: list[Any]) -> Any:
        if values and all(isinstance(value, _Interval) for value in values):
            return _Interval(
                min(value.low for value in values), max(value.high for value in values)
            )
        if values and all(
            isinstance(value, tuple) and len(value) == len(values[0])
            for value in values
        ):
            return tuple(
                self._join([value[i] for value in values])
                for i in range(len(values[0]))
            )
        raise ExecutionError("H2 shot-range proof cannot resolve function arguments")

    def _value(self, wire: Any) -> Any:
        from hugr import ops

        graph = self.graph
        operation = graph[wire.node].op
        name = _name(operation)
        if name == _SHOT_OP:
            return _Interval(0, self.shots - 1)
        if isinstance(operation, ops.LoadConst):
            constant = graph[_source(graph, wire.node, 0).node].op
            if isinstance(constant, ops.Const):
                return self._constant(constant.val)
        if name in ("arithmetic.int.iadd", "arithmetic.int.imul"):
            lhs = self.interval(_source(graph, wire.node, 0))
            rhs = self.interval(_source(graph, wire.node, 1))
            if name.endswith("imul") and lhs.low != lhs.high and rhs.low != rhs.high:
                raise ExecutionError(
                    "H2 shot-range proof supports multiplication by constants"
                )
            return lhs.combine(rhs, name.endswith("imul"))
        if isinstance(operation, ops.MakeTuple):
            return tuple(
                self.value(_source(graph, wire.node, i))
                for i in range(len(operation.types))
            )
        if isinstance(operation, ops.UnpackTuple):
            value = self.value(_source(graph, wire.node, 0))
            if isinstance(value, tuple):
                return value[wire.offset]
        if isinstance(operation, ops.Input):
            parent = graph[wire.node].parent
            owner = graph[parent].op
            if isinstance(owner, ops.FuncDefn):
                arguments = []
                for consumer in graph.linked_ports(parent.out(0)):
                    caller = graph[consumer.node].op
                    if not isinstance(caller, ops.Call) or consumer.offset != len(
                        caller.signature.body.input
                    ):
                        raise ExecutionError(
                            "H2 shot-range proof requires direct function calls; "
                            "function values and indirect calls are unsupported"
                        )
                    arguments.append(
                        self.value(_source(graph, consumer.node, wire.offset))
                    )
                return self._join(arguments)
            if isinstance(owner, ops.Case):
                conditional = graph[parent].parent
                conditional_op = graph[conditional].op
                if isinstance(
                    conditional_op, ops.Conditional
                ) and conditional_op.sum_ty.variant_rows == [[], []]:
                    return self.value(_source(graph, conditional, wire.offset + 1))
            if isinstance(owner, ops.DFG):
                return self.value(_source(graph, parent, wire.offset))
        if isinstance(operation, ops.Call):
            target = _source(graph, wire.node, len(operation.signature.body.input)).node
            output = graph.children(target)[1]
            return self.value(_source(graph, output, wire.offset))
        raise ExecutionError(
            "Cannot prove H2 table index or immutable table source; "
            "use get_current_shot() with nonnegative constant addition/multiplication "
            "and a fixed UInt or Bit table"
        )


def _selection(graph: Any, node: Any, proof: _Proof) -> Any:
    from hugr import ops, tys
    from qamomile.hugr.lowerer import _hugr_uint_type

    operation = graph[node].op
    uint = _hugr_uint_type()
    if (
        not isinstance(operation, ops.Conditional)
        or operation.sum_ty != tys.Bool
        or len(operation.other_inputs) < 2
        or operation.other_inputs[0] != uint
        or len(operation.outputs) != 1
        or operation.outputs[0] not in (uint, tys.Bool)
        or any(
            element != operation.outputs[0] for element in operation.other_inputs[1:]
        )
    ):
        raise ExecutionError("Unsupported H2 shot-indexed selection shape")
    index = _source(graph, node, 1)
    predicate = _source(graph, node, 0)
    compare = graph[predicate.node].op
    if (
        _name(compare) != "arithmetic.int.ilt_u"
        or _source(graph, predicate.node, 0) != index
    ):
        raise ExecutionError("H2 selection metadata does not match its bounds check")
    size = proof.interval(_source(graph, predicate.node, 1))
    length = len(operation.other_inputs) - 1
    if size != _Interval(length, length):
        raise ExecutionError("H2 selection bounds do not match its table length")
    bound = proof.interval(index)
    if bound.low < 0 or bound.high >= length:
        raise ExecutionError(
            f"H2 table index range [{bound.low}, {bound.high}] exceeds table length {length}"
        )
    for offset in range(2, length + 2):
        element = proof.interval(_source(graph, node, offset))
        if element.low != element.high:
            raise ExecutionError(
                "H2 shot-indexed tables must have immutable bound elements"
            )
    cases = graph.children(node)
    if len(cases) != 2 or any(
        not isinstance(graph[case].op, ops.Case) for case in cases
    ):
        raise ExecutionError("Unsupported H2 selection cases")
    invalid = cases[0]
    children = graph.children(invalid)
    if (
        len(children) != 5
        or not isinstance(graph[children[0]].op, ops.Input)
        or not isinstance(graph[children[1]].op, ops.Output)
    ):
        raise ExecutionError("Unsupported H2 selection failure region")
    panic = _source(graph, children[1], 0).node
    load = _source(graph, panic, 0).node
    constant = _source(graph, load, 0).node
    if (
        _name(graph[panic].op) != "prelude.panic"
        or not isinstance(graph[load].op, ops.LoadConst)
        or not isinstance(graph[constant].op, ops.Const)
        or {panic, load, constant} != set(children[2:])
        or any(
            list(graph.incoming_order_links(child))
            or list(graph.outgoing_order_links(child))
            for child in children
        )
    ):
        raise ExecutionError("Unsupported H2 selection failure instructions")
    return panic


def _integer(builder: Any, value: int) -> Any:
    from qamomile.hugr.lowerer import _hugr_uint_value

    return builder.load(_hugr_uint_value(value))[0]


def _binary(builder: Any, name: str, lhs: Any, rhs: Any, output: Any = None) -> Any:
    from hugr import tys
    from hugr.std.int import INT_OPS_EXTENSION
    from qamomile.hugr.lowerer import _hugr_uint_type

    uint = _hugr_uint_type()
    operation = INT_OPS_EXTENSION.get_op(name).instantiate(
        list(uint.args),
        concrete_signature=tys.FunctionType([uint, uint], [output or uint]),
    )
    return builder.add_op(operation, lhs, rhs)[0]


def _take_bool_outputs(graph: Hugr, main: Node) -> list[tuple[ops.DataflowOp, OutPort]]:
    """Replace the instrumented entry's Boolean records with returned values.

    Args:
        graph (Hugr): Independent graph with a closed, instrumented entry.
        main (Node): Entry function containing the public result records.

    Returns:
        list[tuple[ops.DataflowOp, OutPort]]: Recording operations and their Boolean
        input wires in public output order.
    """
    from hugr import ops
    from hugr.build.dfg import Dfg

    records = {
        node: operation
        for node in graph.children(main)
        for operation in (graph[node].op,)
        if isinstance(operation, ops.DataflowOp)
        and _name(operation) == "tket.result.result_bool"
    }
    outputs = [
        (operation, _source(graph, node, 0)) for node, operation in records.items()
    ]
    # Compact shared order ports before deletion. The returned wires refill
    # the data fanout slots vacated by the records.
    links = [
        (source, list(targets))
        for node in graph.children(main)
        for source, targets in graph.outgoing_links(node)
        if source.offset == -1
    ]
    for source, targets in links:
        for target in reversed(targets):
            graph.delete_link(source, target)
    for node in records:
        graph.delete_node(node)
    for source, targets in links:
        for target in targets:
            if source.node not in records and target.node not in records:
                graph.add_link(source, target)
    Dfg._new_existing(graph, main).set_outputs(*(wire for _, wire in outputs))
    return outputs


def _normalize_shot_reads(graph: Hugr, origin: int) -> None:
    """Subtract the device origin from every existing shot read in place.

    Args:
        graph (Hugr): Independent graph whose shot reads are not yet normalized.
        origin (int): Internal policy for the first raw device shot number.
    """
    from hugr.build.dfg import Dfg

    for node, data in list(graph.nodes()):
        if _name(data.op) != _SHOT_OP:
            continue
        consumers = list(graph.linked_ports(node.out(0)))
        # The SDK requires removing fanout links from the last subport first.
        for consumer in reversed(consumers):
            graph.delete_link(node.out(0), consumer)
        builder = Dfg._new_existing(graph, data.parent)
        normalized = _binary(builder, "isub", node.out(0), _integer(builder, origin))
        for consumer in consumers:
            graph.add_link(normalized, consumer)


def _guard(graph: Hugr, origin: int, shots: int) -> None:
    """Normalize existing shot reads and guard entry before user operations.

    Args:
        graph (Hugr): Independent HUGR graph with a closed, instrumented main.
        origin (int): First valid raw device shot number.
        shots (int): Positive number of valid consecutive shots.

    Raises:
        ExecutionError: If the graph lacks a unique closed main entry point.
    """
    import tket_exts

    from hugr import ops, tys, val
    from hugr.build import Module
    from qamomile.hugr.lowerer import _hugr_uint_type

    _normalize_shot_reads(graph, origin)
    functions = [
        (node, data.op)
        for node, data in graph.nodes()
        if isinstance(data.op, ops.FuncDefn)
    ]
    candidates = [
        (node, operation) for node, operation in functions if operation.f_name == "main"
    ]
    if len(candidates) != 1:
        raise ExecutionError("H2 shot-range preparation requires one main function")
    original, signature = candidates[0]
    if graph.entrypoint != original:
        raise ExecutionError(
            "H2 shot-range preparation requires main as the entrypoint"
        )
    if signature.inputs or signature.outputs or signature.params:
        raise ExecutionError("H2 shot-range preparation requires a closed void main")
    outputs = _take_bool_outputs(graph, original)
    name = "_qamomile_h2_shot_body"
    names = {operation.f_name for _, operation in functions}
    while name in names:
        name += "_"
    signature.f_name = name
    signature.visibility = "Private"
    module = Module(graph)
    uint = _hugr_uint_type()
    entry = module.define_function("main", [], [uint], visibility="Public")
    raw = entry.add_op(
        tket_exts.qsystem_utils()
        .get_op("GetCurrentShot")
        .instantiate(concrete_signature=tys.FunctionType([], [uint]))
    )[0]
    lower = _binary(entry, "ige_u", raw, _integer(entry, origin), tys.Bool)
    predicate = entry.add_if(lower, raw)
    predicate.set_outputs(
        _binary(
            predicate,
            "ilt_u",
            predicate.inputs()[0],
            _integer(predicate, origin + shots),
            tys.Bool,
        )
    )
    below = predicate.add_else()
    below.set_outputs(below.load(val.FALSE)[0])
    body = entry.add_if(predicate.conditional_node[0])
    call = body.call(original)
    body.add_state_order(body.input_node, call)
    body.add_state_order(call, body.output_node)
    body.set_outputs(
        _integer(body, 0),
        body.load(val.TRUE)[0],
        *(call[index] for index in range(len(outputs))),
    )
    failed = body.add_else()
    failed.set_outputs(
        _integer(failed, 1),
        failed.load(val.FALSE)[0],
        *(failed.load(val.FALSE)[0] for _ in outputs),
    )
    previous = body.conditional_node
    for index, (operation, _) in enumerate(outputs):
        recorded = entry.add_op(operation, body.conditional_node[index + 2])
        entry.add_state_order(previous, recorded)
        previous = recorded
    marker = entry.add_op(
        tket_exts.result.result_bool(SHOT_SUCCESS_TAG), body.conditional_node[1]
    )
    entry.add_state_order(previous, marker)
    entry.add_state_order(marker, entry.output_node)
    entry.set_outputs(body.conditional_node[0])
    graph.entrypoint = entry.parent_node
    graph[graph.module_root].metadata[_GUARD_KEY] = {"origin": origin, "shots": shots}


def normalize_h2_shots(
    package: Package, shots: int, origin: int | None
) -> tuple[Package, bool]:
    """Normalize every H2 shot read on an independent guarded package.

    Args:
        package (Package): Bound HUGR package with a closed, instrumented main.
        shots (int): Requested positive shot count.
        origin (int | None): Internal raw device shot origin, required for reads.

    Returns:
        tuple[Package, bool]: Normalized package and whether private success
        checks are required. A package without shot reads is returned unchanged.

    Raises:
        ExecutionError: If the device origin or guarded graph is unsupported.
    """
    return _prepare_h2_shots(package, shots, origin, prove_tables=False)


def prepare_h2_shot_range(
    package: Package, shots: int, origin: int | None
) -> tuple[Package, bool]:
    """Prepare an independent H2 package for proven fixed table reads.

    Args:
        package (Package): Original bound HUGR package, before normalization.
        shots (int): Requested positive shot count.
        origin (int | None): Internal raw device shot origin, required for tables.

    Returns:
        tuple[Package, bool]: Prepared package and whether private success checks
        are required. A package without shot-indexed selections is unchanged.

    Raises:
        ExecutionError: If an index, table, device origin, or graph is unsupported.
    """
    return _prepare_h2_shots(package, shots, origin, prove_tables=True)


def _prepare_h2_shots(
    package: Package, shots: int, origin: int | None, *, prove_tables: bool
) -> tuple[Package, bool]:
    """Copy, validate, and normalize H2 reads, optionally proving table bounds.

    Args:
        package (Package): Original bound HUGR package with unnormalized reads.
        shots (int): Requested positive shot count.
        origin (int | None): Internal raw device shot origin.
        prove_tables (bool): Require selections and remove only their proven
            unreachable failure instructions before normalizing shot reads.

    Returns:
        tuple[Package, bool]: Independent prepared package and required private
        success checks, or the unchanged package when preparation is inapplicable.

    Raises:
        ExecutionError: If shot bounds, graph structure, or requested table
            proofs are invalid, or the package has already been guarded.
    """
    has_selections = any(
        _SELECTION_KEY in data.metadata
        for graph in package.modules
        for _, data in graph.nodes()
    )
    has_shot = any(
        _name(data.op) == _SHOT_OP
        for graph in package.modules
        for _, data in graph.nodes()
    )
    if not has_shot or (prove_tables and not has_selections):
        return package, False
    if len(package.modules) != 1:
        raise ExecutionError("H2 shot-range preparation requires one module")
    if (
        isinstance(shots, bool)
        or not isinstance(shots, (int, Integral))
        or not 0 < shots <= _UINT_MAX
    ):
        raise ExecutionError("H2 shot count must fit positive UInt64")
    if (
        isinstance(origin, bool)
        or not isinstance(origin, (int, Integral))
        or not 0 <= origin <= _UINT_MAX
    ):
        raise ExecutionError("H2 shot reads require a known UInt64 shot_origin")
    shots, origin = int(shots), int(origin)
    if origin + shots > _UINT_MAX:
        raise ExecutionError(
            "H2 shot origin plus shots overflows the UInt64 exclusive bound"
        )
    from hugr.cli import validate
    from hugr.package import Package

    graph = deepcopy(package.modules[0])
    if _GUARD_KEY in graph[graph.module_root].metadata:
        raise ExecutionError("H2 shot-range preparation cannot be applied twice")
    for _, data in graph.nodes():
        args: Sequence[object] = getattr(data.op, "args", ())
        if (
            _name(data.op) == "tket.result.result_bool"
            and args
            and getattr(args[0], "value", None) == SHOT_SUCCESS_TAG
        ):
            raise ExecutionError("H2 shot-success output label is reserved")
    if prove_tables:
        proof = _Proof(graph, shots)
        proven_unreachable_panics = []
        for node, data in list(graph.nodes()):
            if _SELECTION_KEY not in data.metadata:
                continue
            if (
                type(data.metadata[_SELECTION_KEY]) is not int
                or data.metadata[_SELECTION_KEY] != 1
            ):
                raise ExecutionError("Unsupported H2 table selection metadata version")
            proven_unreachable_panics.append(_selection(graph, node, proof))
        known = set(proven_unreachable_panics)
        if any(
            _name(data.op) == "prelude.panic" and node not in known
            for node, data in graph.nodes()
        ):
            raise ExecutionError(
                "H2 shot-range preparation cannot remove unproved panic instructions"
            )
        for panic in proven_unreachable_panics:
            case = graph[panic].parent
            children = graph.children(case)
            inputs, output = children[:2]
            graph.delete_node(panic)
            graph.add_link(inputs.out(1), output.inp(0))
    _guard(graph, origin, shots)
    extensions = list(package.extensions)
    import tket_exts

    if not any(extension.name == "tket.result" for extension in extensions):
        extensions.append(tket_exts.result())
    result = Package([graph], extensions)
    validate(result.to_bytes())
    return result, True
