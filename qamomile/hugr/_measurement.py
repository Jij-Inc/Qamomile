"""Emit destructive measurements and fuse direct measurement/reset pairs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hugr.build.dfg import DfBase
    from hugr.hugr import Hugr
    from hugr.hugr.node_port import OutPort, Wire


def measure_free(builder: DfBase, qubit: Wire) -> OutPort:
    """Consume a qubit and return its measurement as a native HUGR Boolean.

    Args:
        builder (DfBase): Dataflow builder receiving the measurement.
        qubit (Wire): Live qubit consumed exactly once.

    Returns:
        OutPort: Native Boolean measurement result.
    """
    from tket_exts import bool as opaque_bool, quantum

    [bit] = builder.add_op(quantum.measure_free, qubit)
    [result] = builder.add_op(opaque_bool.read, bit)
    return result


def _redirect_output(graph: Hugr, source: OutPort, replacement: OutPort) -> None:
    """Move every consumer of an output to its replacement.

    Args:
        graph (Hugr): Graph containing both output ports.
        source (OutPort): Output whose existing links are removed.
        replacement (OutPort): Output supplying the same typed value.
    """
    consumers = list(graph.linked_ports(source))
    # Removing the last link first keeps HUGR's fanout subports contiguous.
    for consumer in reversed(consumers):
        graph.delete_link(source, consumer)
    for consumer in consumers:
        graph.add_link(replacement, consumer)


def fuse_measure_reset(graph: Hugr) -> None:
    """Fuse direct same-region measurement/reset pairs after all wires exist.

    The pass visits a fixed snapshot of nodes once and rewires only matching
    pairs and their consumers. It preserves the original measurement bit and
    the reset qubit, including Boolean fanout. Explicit state-order edges
    exclude a pair because merging their endpoints could create a cycle.

    Args:
        graph (Hugr): Completed target graph to update in place.
    """
    from tket_exts import bool as opaque_bool, qsystem, quantum

    from hugr import ops

    for measure, data in list(graph.nodes()):
        if (
            not isinstance(data.op, ops.ExtOp)
            or data.op.name() != quantum.measure.name()
        ):
            continue
        consumers = list(graph.linked_ports(measure.out(0)))
        if len(consumers) != 1:
            continue
        [reset_input] = consumers
        reset = reset_input.node
        reset_data = graph[reset]
        if (
            reset_input.offset != 0
            or reset_data.parent != data.parent
            or not isinstance(reset_data.op, ops.ExtOp)
            or reset_data.op.name() != quantum.reset.name()
        ):
            continue
        if any(
            list(graph.linked_ports(port))
            for node in (measure, reset)
            for port in (node.inp(-1), node.out(-1))
        ):
            continue

        read = graph.add_node(opaque_bool.read, data.parent, num_outs=1)
        _redirect_output(graph, measure.out(1), read.out(0))
        reset_consumers = list(graph.linked_ports(reset.out(0)))
        # delete_node removes the first link at each port and expects it to exist.
        for consumer in reversed(reset_consumers[1:]):
            graph.delete_link(reset.out(0), consumer)
        graph.delete_node(reset)
        for consumer in reset_consumers:
            graph.add_link(measure.out(0), consumer)
        data.op = qsystem.measure_reset
        graph.add_link(measure.out(1), read.inp(0))
