"""Exact coherent embeddings of body-backed Boolean functions."""

from __future__ import annotations

import qamomile as qm
from qamomile.types import Bool, ProductType, WordType, strip_stage


def boolean_isometry(function: qm.QKernel) -> qm.QKernel:
    r"""Embed one classical Boolean function into a clean quantum target.

    For a body-backed function ``f: Bits[N] -> Bool`` or
    ``f: Bits[N] -> tuple[Bool, ...]``, return an ordinary Qamomile function
    with denotation ``|x> -> |x>|f(x)>``. The returned callable allocates its
    output row in the all-zero state and applies the Rust-proved exact
    ``xor_by`` embedding. Its callable kind is therefore derived as an
    isometry rather than asserted by this helper.

    Args:
        function: Body-backed Boolean Qamomile source function to embed.

    Returns:
        A typed ``Q[Bits[N]] -> Q[Bits[N + M]]`` Qamomile function reference.

    Raises:
        TypeError: If ``function`` is not a supported body-backed Qamomile
            Boolean function.
    """

    checked_function, input_width, output_width = _boolean_signature(function)
    total_width = input_width + output_width

    @qm.qkernel
    def embedded(
        state: qm.Q[qm.Bits[input_width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
        """Embed the configured Boolean function into a reversible transform.

        Args:
            state (qm.Q[qm.Bits[input_width]]): Owned Boolean input register.

        Returns:
            qm.Q[qm.Bits[total_width]]: Input register followed by the function
            output register.
        """

        target = qm.qubits(output_width)
        state, target = qm.xor_by(checked_function, state, target)
        return qm.concat(state, target)

    return embedded


def _boolean_signature(function: object) -> tuple[qm.QKernel, int, int]:
    """Return the fixed input and output widths of one Boolean function."""

    if not isinstance(function, qm.QKernel):
        raise TypeError("boolean isometry requires a Qamomile QKernel")
    definition = function.module().function(function.function_name)
    if definition.external:
        raise TypeError("boolean isometry requires a body-backed function")
    if len(definition.parameters) != 1:
        raise TypeError("boolean isometry function must accept exactly one Bits[N]")
    parameter_type = strip_stage(definition.parameters[0].value.type)
    if (
        not isinstance(parameter_type, WordType)
        or parameter_type.kind != "Bits"
        or not isinstance(parameter_type.width, int)
    ):
        raise TypeError("boolean isometry function input must be fixed-width Bits[N]")

    result_type = strip_stage(definition.result_type)
    if result_type == Bool:
        output_width = 1
    elif (
        isinstance(result_type, ProductType)
        and result_type.elements
        and all(element == Bool for element in result_type.elements)
    ):
        output_width = len(result_type.elements)
    else:
        raise TypeError(
            "boolean isometry function must return Bool or a nonempty Product of Bool"
        )
    return function, parameter_type.width, output_width


__all__ = ["boolean_isometry"]
