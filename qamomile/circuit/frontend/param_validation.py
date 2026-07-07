"""Validate bound qkernel argument handles against declared parameter types.

These predicates and validators classify a kernel parameter's declared
annotation (scalar ``Qubit`` vs. ``Vector[Qubit]``, quantum vs. classical)
and check that the handle bound to it at a call site matches that
declaration. They are shared by the plain qkernel call path
(``QKernel.__call__`` in :mod:`qamomile.circuit.frontend.qkernel`) and the
controlled-gate call path (``ControlledGate`` in
:mod:`qamomile.circuit.frontend.operation.control`). Keeping them in this
neutral module means neither of those modules has to import the other just
to reach these checks.
"""

from __future__ import annotations

import types as _types
from typing import Any, Union, get_args, get_origin

from qamomile.circuit.frontend.func_to_block import is_array_type
from qamomile.circuit.frontend.handle import Handle, Observable
from qamomile.circuit.frontend.handle.primitives import Bit, Float, Qubit, UInt
from qamomile.circuit.ir.types.hamiltonian import ObservableType
from qamomile.circuit.ir.types.primitives import BitType, FloatType, UIntType

# Maps a classical scalar element annotation to the IR ``ValueType`` an
# array of that element carries on its ``ArrayValue.type``. Used to check
# that a bound array handle's element type matches the declared element
# type (e.g. a ``Vector[Bit]`` argument really carries ``BitType``
# elements). Covers every classical scalar that ``handle_type_map`` admits
# as a parameter element: ``Float`` / ``UInt`` / ``Bit`` / ``Observable``.
_CLASSICAL_ELEMENT_IR_TYPE: dict[Any, type] = {
    UInt: UIntType,
    int: UIntType,
    Float: FloatType,
    float: FloatType,
    Bit: BitType,
    bool: BitType,
    Observable: ObservableType,
}


def _union_members(param_type: Any) -> tuple[Any, ...]:
    """Return members when a parameter annotation is a union.

    Args:
        param_type (Any): A resolved qkernel input annotation.

    Returns:
        tuple[Any, ...]: Union member annotations, or an empty tuple
            when *param_type* is not a union annotation.
    """
    origin = get_origin(param_type)
    if origin in (Union, _types.UnionType):
        return get_args(param_type)
    return ()


def _array_element_type(param_type: Any) -> Any | None:
    """Return the element handle type for an array annotation.

    Args:
        param_type (Any): A frontend annotation such as
            ``Vector[Float]`` or ``Vector[Qubit]``.

    Returns:
        Any | None: The element handle type when *param_type* is an
            array annotation, or ``None`` for scalar annotations.
    """
    if not is_array_type(param_type):
        return None
    args = get_args(param_type)
    if args:
        return args[0]
    return getattr(param_type, "element_type", None)


def _is_scalar_classical_param_decl(param_type: Any) -> bool:
    """Return whether a parameter declaration is scalar classical.

    Args:
        param_type (Any): A resolved qkernel input annotation.

    Returns:
        bool: ``True`` for scalar ``Float`` / ``UInt`` declarations,
            including scalar union forms such as ``float | Float``.
    """
    scalar_types = (Float, float, UInt, int)
    if param_type in scalar_types:
        return True
    members = _union_members(param_type)
    return bool(members) and all(member in scalar_types for member in members)


def _is_uint_param_decl(param_type: Any) -> bool:
    """Return whether a parameter declaration is integer-like.

    Args:
        param_type (Any): A resolved qkernel input annotation.

    Returns:
        bool: ``True`` for scalar ``UInt`` / ``int`` declarations,
            including union forms whose members are all integer-like.
    """
    if param_type in (UInt, int):
        return True
    members = _union_members(param_type)
    return bool(members) and all(member in (UInt, int) for member in members)


def _is_quantum_param_decl(param_type: Any) -> bool:
    """Return whether a qkernel parameter declaration is quantum.

    Args:
        param_type (Any): A resolved qkernel input annotation.

    Returns:
        bool: ``True`` for ``Qubit`` and ``Vector[Qubit]``-like
            declarations, otherwise ``False``.
    """
    if param_type is Qubit:
        return True
    return _array_element_type(param_type) is Qubit


def _is_classical_param_decl(param_type: Any) -> bool:
    """Return whether a qkernel parameter declaration is classical.

    Args:
        param_type (Any): A resolved qkernel input annotation.

    Returns:
        bool: ``True`` for every classical value declaration: scalar
            ``Float`` / ``UInt`` / ``Bit`` / ``Observable`` and their array
            forms (``Vector[Float]``, ``Vector[Bit]``,
            ``Vector[Observable]``, ...). These are exactly the classical
            element types ``handle_type_map`` admits as parameters.
    """
    if param_type is Observable:
        return True
    if param_type is Bit or param_type is bool:
        return True
    if _is_scalar_classical_param_decl(param_type):
        return True
    return _array_element_type(param_type) in _CLASSICAL_ELEMENT_IR_TYPE


def _is_quantum_handle(value: Any) -> bool:
    """Return whether a runtime handle carries quantum data.

    Args:
        value (Any): A caller argument or bound qkernel argument.

    Returns:
        bool: ``True`` for scalar ``Qubit`` handles and arrays whose
            element IR type is quantum.
    """
    from qamomile.circuit.frontend.handle.array import ArrayBase

    if isinstance(value, Qubit):
        return True
    if isinstance(value, ArrayBase):
        return value.value.type.is_quantum()
    return False


def _validate_classical_param_handle(
    param_name: str,
    declared: Any,
    param_value: Handle,
    context: str = "control()",
) -> None:
    """Validate that a handle matches a classical parameter declaration.

    Args:
        param_name (str): Kernel parameter name, used in error messages.
        declared (Any): Resolved kernel input annotation.
        param_value (Handle): Caller-supplied handle value.
        context (str): Call-site label used to prefix the error message,
            e.g. ``"control()"`` or ``"my_kernel()"``. Defaults to
            ``"control()"`` so existing callers keep their wording.

    Raises:
        TypeError: If the supplied handle is quantum, or does not match the
            declared scalar (``Float`` / ``UInt`` / ``Bit`` / ``Observable``)
            or array (``Vector[Float]`` / ``Vector[Bit]`` / ...) parameter
            kind.
    """
    from qamomile.circuit.frontend.handle.array import ArrayBase

    if _is_quantum_handle(param_value):
        raise TypeError(
            f"{context}: parameter {param_name!r} is declared as a "
            f"classical parameter but received quantum handle "
            f"{type(param_value).__name__}. Pass a classical handle instead."
        )

    if declared is Observable:
        if not isinstance(param_value, Observable):
            raise TypeError(
                f"{context}: parameter {param_name!r} is declared as "
                f"Observable but received {type(param_value).__name__}. "
                f"Pass an Observable handle from the enclosing qkernel."
            )
        return

    element_type = _array_element_type(declared)
    if element_type is not None:
        if not isinstance(param_value, ArrayBase):
            raise TypeError(
                f"{context}: parameter {param_name!r} is declared as "
                f"an array parameter but received "
                f"{type(param_value).__name__}. Pass the corresponding "
                f"Vector handle from the caller kernel instead."
            )
        expected_cls = _CLASSICAL_ELEMENT_IR_TYPE.get(element_type)
        if expected_cls is not None:
            expected_type = expected_cls()
            if param_value.value.type != expected_type:
                raise TypeError(
                    f"{context}: parameter {param_name!r} expects "
                    f"{expected_type.label()} elements but received "
                    f"{param_value.value.type.label()} elements."
                )
        return

    if declared is Bit or declared is bool:
        if not isinstance(param_value, Bit):
            raise TypeError(
                f"{context}: parameter {param_name!r} is declared as "
                f"Bit but received {type(param_value).__name__}. "
                f"Pass a Bit handle instead."
            )
        return

    if _is_uint_param_decl(declared):
        if not isinstance(param_value, UInt):
            raise TypeError(
                f"{context}: parameter {param_name!r} is declared as "
                f"UInt/int but received {type(param_value).__name__}. "
                f"Pass a UInt handle instead."
            )
        return

    if not isinstance(param_value, Float):
        raise TypeError(
            f"{context}: parameter {param_name!r} is declared as "
            f"Float/float but received {type(param_value).__name__}. "
            f"Pass a Float handle instead."
        )


def _validate_quantum_param_handle(
    param_name: str,
    declared: Any,
    param_value: Handle,
    context: str = "control()",
    allow_broadcast: bool = False,
) -> None:
    """Validate that a handle matches a quantum parameter declaration.

    Catches the arity mismatches that previously leaked an opaque
    ``AttributeError`` (``'Qubit' object has no attribute 'shape'``) or
    silently miscompiled. A classical handle bound to a quantum parameter
    is rejected outright. The two arity directions are not symmetric:

    * **An array declaration receiving a scalar ``Qubit``** (e.g. passing
      a single qubit to a ``Vector[Qubit]`` target) is always rejected --
      the callee indexes the register internally, so a lone qubit cannot
      satisfy it. The concrete array kind must also match the declared
      rank: a ``Vector[Qubit]`` declaration accepts ``Vector`` /
      ``VectorView`` but not a higher-rank ``Matrix`` / ``Tensor``, and
      vice versa.
    * **A scalar ``Qubit`` declaration receiving a quantum array** is a
      *broadcast* in the control path (the controlled gate is applied
      once per target qubit -- see ``controlled_native_broadcast_target``)
      but a silent miscompile in a plain qkernel callable invocation
      (the whole register collapses onto one scalar dummy input).
      ``allow_broadcast`` selects which contract applies. Only a 1-D
      ``Vector`` / ``VectorView`` is a valid broadcast source; a
      higher-rank ``Matrix`` / ``Tensor`` is rejected even when
      ``allow_broadcast`` is set.

    Args:
        param_name (str): Kernel parameter name, used in error messages.
        declared (Any): Resolved kernel input annotation. Either scalar
            ``Qubit`` or an array form such as ``Vector[Qubit]``.
        param_value (Handle): Caller-supplied handle value.
        context (str): Call-site label used to prefix the error message,
            e.g. ``"control()"`` or ``"my_kernel()"``. Defaults to
            ``"control()"``.
        allow_broadcast (bool): When ``True``, a quantum array bound to a
            scalar ``Qubit`` declaration is accepted as a per-element
            broadcast (the control path). When ``False`` (a plain
            qkernel call, which does not broadcast), the same shape is
            rejected. Defaults to ``False``.

    Raises:
        TypeError: If the supplied handle is classical, or its arity
            (scalar vs. array) does not match the declared quantum
            parameter kind under the active broadcast contract.
    """
    from qamomile.circuit.frontend.handle.array import ArrayBase, Vector

    if not _is_quantum_handle(param_value):
        raise TypeError(
            f"{context}: parameter {param_name!r} is declared as a "
            f"quantum parameter but received non-quantum handle "
            f"{type(param_value).__name__}. Pass a Qubit or Vector[Qubit] "
            f"handle instead."
        )

    if declared is Qubit:
        if isinstance(param_value, Qubit):
            return
        # A quantum array bound to a scalar ``Qubit`` parameter broadcasts
        # in the control path but silently miscompiles in a plain call.
        # Only 1-D ``Vector`` / ``VectorView`` (``VectorView`` subclasses
        # ``Vector``) broadcasts are supported: the control-path expansion
        # assumes 1-D register semantics, so a higher-rank ``Matrix`` /
        # ``Tensor`` is not a valid broadcast target and is rejected.
        if allow_broadcast and isinstance(param_value, Vector):
            return
        raise TypeError(
            f"{context}: parameter {param_name!r} is declared as a "
            f"scalar Qubit but received {type(param_value).__name__}. "
            f"Pass a single Qubit handle (e.g. index the register with "
            f"qs[0]) instead."
        )

    # Array quantum declaration (``Vector[Qubit]`` / higher-rank register).
    if not isinstance(param_value, ArrayBase):
        raise TypeError(
            f"{context}: parameter {param_name!r} is declared as a quantum "
            f"array (e.g. Vector[Qubit]) but received scalar "
            f"{type(param_value).__name__}. Pass a Vector[Qubit] register "
            f"instead -- allocate one with qmc.qubit_array(N, ...) at the "
            f"call site, or pass an existing Vector handle."
        )

    # The concrete array kind must match the declared rank. A
    # ``Vector[Qubit]`` declaration accepts ``Vector`` / ``VectorView``
    # (``VectorView`` subclasses ``Vector``) but not a higher-rank
    # ``Matrix`` / ``Tensor``, and a ``Matrix[Qubit]`` / ``Tensor[Qubit]``
    # declaration rejects a 1-D ``Vector``. Many call sites assume 1-D
    # indexing for ``Vector[Qubit]``, so a rank mismatch left unchecked
    # would surface as a later failure or miscompile.
    declared_origin = getattr(declared, "__origin__", declared)
    if isinstance(declared_origin, type) and not isinstance(
        param_value, declared_origin
    ):
        raise TypeError(
            f"{context}: parameter {param_name!r} is declared as "
            f"{declared_origin.__name__}[Qubit] but received a "
            f"{type(param_value).__name__} handle of a different rank. "
            f"Pass a {declared_origin.__name__}[Qubit] handle instead."
        )


def _validate_param_handle(
    param_name: str,
    declared: Any,
    param_value: Handle,
    context: str = "control()",
    allow_broadcast: bool = False,
) -> None:
    """Validate a bound handle against its qkernel parameter declaration.

    Routes to the quantum or classical validator based on the declared
    parameter kind. Structural-container declarations (``Tuple`` /
    ``Dict``) and unannotated parameters are passed through unchecked --
    their shape is enforced by the inline / segmentation passes instead.

    Args:
        param_name (str): Kernel parameter name, used in error messages.
        declared (Any): Resolved kernel input annotation.
        param_value (Handle): Caller-supplied handle bound to the
            parameter.
        context (str): Call-site label used to prefix the error message,
            e.g. ``"control()"`` or ``"my_kernel()"``. Defaults to
            ``"control()"``.
        allow_broadcast (bool): Forwarded to
            :func:`_validate_quantum_param_handle`; ``True`` lets a
            quantum array bind to a scalar ``Qubit`` parameter as a
            per-element broadcast (the control path). Has no effect on
            classical parameters. Defaults to ``False``.

    Raises:
        TypeError: If *param_value* does not match the quantum or
            classical declaration of *param_name*.
    """
    if _is_quantum_param_decl(declared):
        _validate_quantum_param_handle(
            param_name, declared, param_value, context, allow_broadcast=allow_broadcast
        )
    elif _is_classical_param_decl(declared):
        _validate_classical_param_handle(param_name, declared, param_value, context)
    # else: a structural container (``Tuple`` / ``Dict``) or an
    # unannotated parameter -- there is no scalar/array arity contract to
    # enforce here, so the handle passes through unchecked.


def _validate_bound_handles(
    input_types: dict[str, Any],
    arguments: dict[str, Any],
    *,
    context: str = "control()",
    allow_broadcast: bool = False,
) -> None:
    """Validate every bound argument handle against its declared parameter.

    Iterates the bound-argument mapping and runs :func:`_validate_param_handle`
    on each value that is a frontend ``Handle``. This is the single shared
    entry point for every call site that binds caller handles to a kernel's
    declared parameters -- the plain qkernel call (``QKernel.__call__``), the
    controlled-gate call (``ControlledGate``), and the inverse-gate call
    (``InverseGate``) -- so the same arity / quantum-vs-classical / rank checks
    fire wherever arguments are bound, before any handle is consumed.

    Non-``Handle`` values (raw Python literals such as a ``theta=0.5`` default)
    and parameters absent from *input_types* are skipped: their promotion and
    coercion are handled by each call site's own logic.

    Args:
        input_types (dict[str, Any]): Kernel parameter name to resolved
            declared annotation (as recorded on ``QKernel.input_types``).
        arguments (dict[str, Any]): Bound argument mapping (parameter name to
            caller value), e.g. ``signature.bind(...).arguments``.
        context (str): Call-site label used to prefix error messages, e.g.
            ``"control()"`` or ``"my_kernel()"``. Defaults to ``"control()"``.
        allow_broadcast (bool): Forwarded to :func:`_validate_param_handle`;
            ``True`` lets a quantum 1-D ``Vector`` / ``VectorView`` bind to a
            scalar ``Qubit`` parameter as a per-element broadcast (the control
            path). Defaults to ``False``.

    Raises:
        TypeError: If any bound handle does not match its declared parameter
            kind.
    """
    for name, value in arguments.items():
        declared = input_types.get(name)
        if declared is not None and isinstance(value, Handle):
            _validate_param_handle(
                name, declared, value, context, allow_broadcast=allow_broadcast
            )
