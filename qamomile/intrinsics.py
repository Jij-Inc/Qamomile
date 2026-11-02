"""Source-only intrinsic names recognized by the Qamomile frontend."""

from __future__ import annotations

from typing import Any, NoReturn, TypeVar

_Owned = TypeVar("_Owned")
_TargetOwned = TypeVar("_TargetOwned")


def _source_only(name: str) -> NoReturn:
    """Reject host execution of a source-language intrinsic."""

    raise RuntimeError(
        f"qm.{name} is a Qamomile source intrinsic. It is interpreted only "
        "inside a function compiled by qm.build or qm.build_source."
    )


def log2(_value: object) -> Any:
    """Compute a source-language base-two logarithm."""

    return _source_only("log2")


def ceil(_value: object) -> Any:
    """Round a source-language numeric value upward to an integer."""

    return _source_only("ceil")


def sqrt(_value: object) -> Any:
    """Compute the nonnegative source-language square root as Real.

    Int operands use the language's finite binary64 conversion. The result
    preserves the operand's Const, Param, or Shot stage. Rust checks the
    finite, nonnegative domain when the operand becomes available.

    Args:
        _value (object): Source-language Int or Real operand.

    Returns:
        Any: Source-language Real with the operand's stage.

    Raises:
        RuntimeError: If called directly from host Python.
    """

    return _source_only("sqrt")


def qubits(_size: object) -> Any:
    """Allocate a clean logical quantum state in Qamomile source."""

    return _source_only("qubits")


def prepare(_value: object) -> Any:
    """Prepare classical logical data as an owned quantum value."""

    return _source_only("prepare")


def as_fixed(_state: _Owned, *, integer_bits: object = 0) -> Any:
    """View coherent bits as one unsigned fixed-point value without gates."""

    del integer_bits
    return _source_only("as_fixed")


def as_uint(_state: _Owned) -> Any:
    """View coherent bits as one unsigned integer without applying gates.

    Args:
        _state (_Owned): Owned coherent bit word to reinterpret.

    Returns:
        Any: Source-only ``Q[UInt[N]]`` value with the same physical carriers.

    Raises:
        RuntimeError: If called from host Python instead of Qamomile source.
    """

    return _source_only("as_uint")


def as_bits(_state: _Owned) -> Any:
    """View one coherent word as uninterpreted bits without applying gates.

    Args:
        _state (_Owned): Owned coherent ``UInt`` or ``UFixed`` word to
            reinterpret in its canonical logical digit order.

    Returns:
        Any: Source-only ``Q[Bits[N]]`` value with the same physical carriers.

    Raises:
        RuntimeError: If called from host Python instead of Qamomile source.
    """

    return _source_only("as_bits")


def split_at(_state: _Owned, _at: object) -> Any:
    """Split one coherent bit owner at a static logical boundary.

    Args:
        _state (_Owned): Owned coherent bit state to partition.
        _at (object): Compile-time boundary between the nonempty left and
            right results.

    Returns:
        Any: Source-only pair of disjoint coherent owners in logical order.

    Raises:
        RuntimeError: If called from host Python instead of Qamomile source.
    """

    return _source_only("split_at")


def concat(_first: _Owned, _second: _Owned, *_rest: _Owned) -> Any:
    """Concatenate coherent owners into one logical bit state.

    Args:
        _first (_Owned): First coherent owner in the resulting logical order.
        _second (_Owned): Second coherent owner in the resulting logical order.
        *_rest (_Owned): Additional coherent owners in logical order.

    Returns:
        Any: Source-only coherent owner spanning every input component.

    Raises:
        RuntimeError: If called from host Python instead of Qamomile source.
    """

    return _source_only("concat")


def h(_state: _Owned, **_selection: object) -> _Owned:
    """Apply a logical Hadamard transformation."""

    return _source_only("h")


def x(_state: _Owned, **_selection: object) -> _Owned:
    """Apply a logical Pauli-X transformation."""

    return _source_only("x")


def y(_state: _Owned, **_selection: object) -> _Owned:
    """Apply a logical Pauli-Y transformation."""

    return _source_only("y")


def z(_state: _Owned, **_selection: object) -> _Owned:
    """Apply a logical Pauli-Z transformation."""

    return _source_only("z")


def s(_state: _Owned, **_selection: object) -> _Owned:
    """Apply a logical phase-S transformation."""

    return _source_only("s")


def sdg(_state: _Owned, **_selection: object) -> _Owned:
    """Apply the adjoint logical phase-S transformation."""

    return _source_only("sdg")


def t(_state: _Owned, **_selection: object) -> _Owned:
    """Apply a logical phase-T transformation."""

    return _source_only("t")


def tdg(_state: _Owned, **_selection: object) -> _Owned:
    """Apply the adjoint logical phase-T transformation."""

    return _source_only("tdg")


def p(_state: _Owned, _angle: object, **_selection: object) -> _Owned:
    """Apply an exact logical phase transformation."""

    return _source_only("p")


def rx(_state: _Owned, _angle: object, **_selection: object) -> _Owned:
    """Apply a logical X rotation."""

    return _source_only("rx")


def ry(_state: _Owned, _angle: object, **_selection: object) -> _Owned:
    """Apply a logical Y rotation."""

    return _source_only("ry")


def rz(_state: _Owned, _angle: object, **_selection: object) -> _Owned:
    """Apply a logical Z rotation."""

    return _source_only("rz")


def cx(_state: _Owned, **_selection: object) -> _Owned:
    """Apply a logical controlled-X transformation."""

    return _source_only("cx")


def cz(_state: _Owned, **_selection: object) -> _Owned:
    """Apply a logical controlled-Z transformation."""

    return _source_only("cz")


def ccx(_state: _Owned, **_selection: object) -> _Owned:
    """Apply a logical doubly controlled X transformation."""

    return _source_only("ccx")


def cp(_state: _Owned, _angle: object, **_selection: object) -> _Owned:
    """Apply an exact logical controlled phase transformation."""

    return _source_only("cp")


def rzz(_state: _Owned, _angle: object, **_selection: object) -> _Owned:
    """Apply a logical two-qubit ZZ rotation."""

    return _source_only("rzz")


def swap(_state: _Owned, **_selection: object) -> _Owned:
    """Swap two logical components."""

    return _source_only("swap")


def measure(_state: object) -> Any:
    """Consume a quantum value and produce a shot-time logical value."""

    return _source_only("measure")


def measure_z(_state: object, **_selection: object) -> Any:
    """Measure one component in the Z basis and retain the resulting state.

    Args:
        _state (object): Owned quantum state containing the selected component.
        **_selection (object): Optional ``target`` selector, required unless the
            state has exactly one coherent lane.

    Returns:
        Any: Pair containing ``Shot[Bool]`` and the successor quantum state.

    Raises:
        RuntimeError: If called from host Python instead of Qamomile source.
    """

    return _source_only("measure_z")


def measure_x(_state: object, **_selection: object) -> Any:
    """Measure one component in the X basis and retain the resulting state.

    Args:
        _state (object): Owned quantum state containing the selected component.
        **_selection (object): Optional ``target`` selector, required unless the
            state has exactly one coherent lane.

    Returns:
        Any: Pair containing ``Shot[Bool]`` and the successor quantum state.

    Raises:
        RuntimeError: If called from host Python instead of Qamomile source.
    """

    return _source_only("measure_x")


def measure_y(_state: object, **_selection: object) -> Any:
    """Measure one component in the Y basis and retain the resulting state.

    Args:
        _state (object): Owned quantum state containing the selected component.
        **_selection (object): Optional ``target`` selector, required unless the
            state has exactly one coherent lane.

    Returns:
        Any: Pair containing ``Shot[Bool]`` and the successor quantum state.

    Raises:
        RuntimeError: If called from host Python instead of Qamomile source.
    """

    return _source_only("measure_y")


def measure_reset(_state: object) -> Any:
    """Measure an owned state and return both the result and a clean state."""

    return _source_only("measure_reset")


def reset(_state: _Owned) -> _Owned:
    """Explicitly reset an owned quantum value."""

    return _source_only("reset")


def discard(_state: object) -> None:
    """Explicitly trace out an owned quantum value."""

    return _source_only("discard")


def adjoint(_callable: object, _state: _Owned, *_arguments: object) -> _Owned:
    """Apply the exact adjoint of a statically known unitary function."""

    return _source_only("adjoint")


def phase_by(_predicate: object, _state: _Owned) -> _Owned:
    """Apply a pi phase exactly where a pure Boolean predicate is true."""

    return _source_only("phase_by")


def xor_by(
    _predicate: object,
    _source: _Owned,
    _target: _TargetOwned,
) -> tuple[_Owned, _TargetOwned]:
    """XOR one proved Boolean function into a matching owned target word."""

    return _source_only("xor_by")


def permute_by(_function: object, _state: _Owned) -> _Owned:
    """Apply one Rust-proved in-place Boolean permutation."""

    return _source_only("permute_by")


def control(
    _callable: object,
    _controls: object,
    _state: _Owned,
    *_arguments: object,
    **_keywords: object,
) -> Any:
    """Apply a static unitary coherently and return controls with its result.

    The optional source keyword ``at=index`` selects one logical component of
    an aggregate control value without exposing or borrowing a qubit handle.
    """

    return _source_only("control")


def multiplex(
    _index: object,
    _cases: object,
    _state: _Owned,
    *_arguments: object,
    **_keywords: object,
) -> Any:
    """Apply one static unitary selected by a coherent logical index.

    Args:
        _index (object): Owned coherent selector retained in the result.
        _cases (object): Static tuple or list containing exactly one compatible
            unitary qkernel for every logical index value.
        _state (_Owned): Owned quantum state consumed by the selected case.
        *_arguments (object): Additional positional arguments shared by every
            case after the leading owned state.
        **_keywords (object): Additional named arguments shared by every case.

    Returns:
        Any: Source-only pair containing the retained index and successor
            state.

    Raises:
        RuntimeError: If called from host Python instead of Qamomile source.
    """

    return _source_only("multiplex")


def global_phase(_state: _Owned, _angle: object) -> _Owned:
    """Multiply an owned quantum value by an exact scalar phase."""

    return _source_only("global_phase")


def add_into(_state: _Owned) -> _Owned:
    """Add the first little-endian half into the second half modulo its domain.

    Args:
        _state (_Owned): Owned state with ``2*n`` coherent bits, ``n > 0``.
            Bits ``0:n`` encode the retained addend; bits ``n:2*n`` encode the
            target. Both halves are unsigned little-endian integers.

    Returns:
        _Owned: The same owned shape with target ``(target + addend) mod 2**n``.

    Raises:
        RuntimeError: If called directly instead of inside a source qkernel.
    """

    return _source_only("add_into")


def mod_add(_target: _Owned, _addend: object, _modulus: object) -> Any:
    """Add into an unsigned register modulo a classical or coherent modulus.

    Args:
        _target (_Owned): Owned unsigned target register.
        _addend (object): Equally wide owned unsigned register or Const integer.
        _modulus (object): Equally wide owned unsigned register or Const integer.

    Returns:
        Any: Updated target alone for classical inputs, or updated target,
            addend, and modulus for quantum inputs.

    Raises:
        RuntimeError: If called outside compiled Qamomile source.
    """

    return _source_only("mod_add")


def mul_add(_product: _Owned, _left: _Owned, _right: _Owned) -> Any:
    """Add the exact product of two retained unsigned words into an accumulator.

    Args:
        _product (_Owned): Owned accumulator with twice the factor width.
        _left (_Owned): Owned first unsigned factor.
        _right (_Owned): Owned second unsigned factor.

    Returns:
        Any: Updated accumulator and both retained factors in source order.

    Raises:
        RuntimeError: If called outside compiled Qamomile source.
    """

    return _source_only("mul_add")


def mod_mul_add(
    _accumulator: _Owned,
    _left: _Owned,
    _right: _Owned,
    *,
    modulus: object,
) -> Any:
    """Add a retained-register product modulo a classical modulus.

    The complete-basis operation is the identity unless all three equally
    wide registers encode values below ``modulus``. On the valid domain it
    updates only the accumulator.

    Args:
        _accumulator (_Owned): Owned unsigned modular accumulator.
        _left (_Owned): Owned first unsigned factor, retained unchanged.
        _right (_Owned): Owned second unsigned factor, retained unchanged.
        modulus (object): Const positive modulus fitting the register width.

    Returns:
        Any: Updated accumulator followed by both retained factors in source
            order.

    Raises:
        RuntimeError: If called outside compiled Qamomile source.
    """

    del modulus
    return _source_only("mod_mul_add")


def mod_mul(_target: _Owned, *, factor: object, modulus: object) -> _Owned:
    """Multiply an unsigned register by an invertible classical residue.

    Args:
        _target (_Owned): Owned unsigned target register.
        factor (object): Const integer coprime to the modulus.
        modulus (object): Const positive modulus fitting the register width.

    Returns:
        _Owned: Updated target, with out-of-domain values left unchanged.

    Raises:
        RuntimeError: If called outside compiled Qamomile source.
    """

    del factor, modulus
    return _source_only("mod_mul")


def mod_pow(
    _target: _TargetOwned,
    _exponent: _Owned,
    *,
    base: object,
    modulus: object,
) -> tuple[_TargetOwned, _Owned]:
    """Multiply a residue by a coherently exponentiated classical base.

    Args:
        _target (_TargetOwned): Owned unsigned modular-value register.
        _exponent (_Owned): Owned unsigned exponent register, retained unchanged.
        base (object): Const integer coprime to the modulus.
        modulus (object): Const positive modulus fitting the target width.

    Returns:
        tuple[_TargetOwned, _Owned]: Updated target and retained exponent in
            source order. Target values outside the standard residue range are
            left unchanged.

    Raises:
        RuntimeError: If called outside compiled Qamomile source.
    """

    del base, modulus
    return _source_only("mod_pow")


def mod_multi_pow(
    _coordinates: _Owned,
    _target: _Owned,
    *,
    bases: object,
    base_width: object,
    modulus: object,
    coordinate_width: object,
) -> Any:
    """Apply packed modular multi-exponentiation without choosing a circuit.

    For ``d = len(bases)``, this maps a packed ``d * m``-bit coordinate value
    and an ``n``-bit modular target to the same coordinates and
    ``target * product(bases[i] ** coordinates[i]) mod modulus``.  The
    implementation is selected later through :class:`RealizationOptions`.

    Args:
        _coordinates (_Owned): Packed coherent coordinate words in source
            order, each with ``coordinate_width`` bits.
        _target (_Owned): Coherent unsigned modular target.
        bases (object): Specialization-time vector of modular units.
        base_width (object): Positive specialization-time upper bound on each
            base's binary width, used by product-tree realizations.
        modulus (object): Specialization-time positive modulus.
        coordinate_width (object): Positive specialization-time width of each
            coordinate word.

    Returns:
        Any: Source-only pair containing retained coordinates and the updated
            target.

    Raises:
        RuntimeError: If called from host Python instead of Qamomile source.
    """

    del bases, base_width, modulus, coordinate_width
    return _source_only("mod_multi_pow")


def less_than_xor(_flag: _Owned, _left: _Owned, _right: _Owned) -> Any:
    """XOR an unsigned comparison into one owned flag bit.

    Args:
        _flag (_Owned): Owned one-bit flag.
        _left (_Owned): Owned first unsigned register.
        _right (_Owned): Owned equally wide second unsigned register.

    Returns:
        Any: Updated flag and both retained registers in source order.

    Raises:
        RuntimeError: If called outside compiled Qamomile source.
    """

    return _source_only("less_than_xor")


def pauli_evolve(
    _state: _Owned,
    _hamiltonian: object,
    _time: object,
) -> _Owned:
    """Apply exact evolution for a static commuting Pauli Hamiltonian."""

    return _source_only("pauli_evolve")
