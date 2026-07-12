"""Qiskit backend transpiler implementation.

This module provides QiskitTranspiler for converting Qamomile QKernels
into Qiskit QuantumCircuits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, cast

if TYPE_CHECKING:
    import qamomile.observable as qm_o
    from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
    from qiskit import QuantumCircuit

from qamomile.circuit.ir.operation.arithmetic_operations import (
    CompOp,
    CompOpKind,
    CondOp,
    CondOpKind,
    NotOp,
    RuntimeClassicalExpr,
    RuntimeOpKind,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.executable import (
    ParameterMetadata,
    QuantumExecutor,
)
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.passes.emit_support import (
    ClbitMap,
    QubitAddress,
    QubitMap,
    resolve_condition_address,
    resolve_if_condition,
)
from qamomile.circuit.transpiler.passes.separate import SegmentationPass
from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.qiskit.emitter import QiskitGateEmitter


class QiskitEmitPass(StandardEmitPass["QuantumCircuit"]):
    """Qiskit-specific emission pass.

    Extends StandardEmitPass with Qiskit-specific control flow handling
    using context managers.
    """

    def __init__(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
        use_native_composite: bool = True,
    ):
        """Initialize the Qiskit emit pass.

        Args:
            bindings: Parameter bindings for the circuit
            parameters: List of parameter names to preserve as backend parameters
            use_native_composite: If True, use native Qiskit implementations
                                  for QFT/IQFT. If False, use manual decomposition.
        """
        emitter = QiskitGateEmitter()
        composite_emitters = self._init_emitters() if use_native_composite else []
        super().__init__(
            emitter,
            bindings,
            parameters,
            composite_emitters,
            backend_name="qiskit",
        )
        self._use_native_composite = use_native_composite

    def _init_emitters(self) -> list:
        """Initialize native CompositeGate emitters."""
        from qamomile.qiskit.emitters import QiskitQFTEmitter

        return [QiskitQFTEmitter()]

    def _emit_for(
        self,
        circuit: "QuantumCircuit",
        op: ForOperation,
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
        force_unroll: bool = False,
    ) -> None:
        """Emit a range loop with Qiskit's native or unrolled representation.

        Args:
            circuit (QuantumCircuit): Qiskit circuit being mutated.
            op (ForOperation): Range loop to emit.
            qubit_map (QubitMap): Current logical-to-physical qubit map.
            clbit_map (ClbitMap): Current logical-to-physical clbit map.
            bindings (dict[str, Any]): Active compile-time and loop-local
                bindings.
            force_unroll (bool): Whether to bypass Qiskit's native loop.
                Defaults to False.

        Raises:
            EmitError: If loop identities or carried values are malformed.
            ValueError: If a loop requiring unrolling has unresolved bounds.
        """
        from qamomile.circuit.transpiler.passes.emit_support.control_flow_emission import (
            _publish_region_results,
            _seed_region_args,
            resolve_loop_bounds,
            validated_loop_indexset,
        )

        start, stop, step = resolve_loop_bounds(self._resolver, op, bindings)

        if start is None or stop is None or step is None:
            self._emit_for_unrolled(circuit, op, qubit_map, clbit_map, bindings)
            return

        indexset = validated_loop_indexset(start, stop, step)
        if len(indexset) == 0:
            # Zero-trip passthrough: publish each RegionArg initializer
            # as the loop result, mirroring the shared emit path.
            if op.region_args:
                carried = _seed_region_args(self, op, bindings)
                _publish_region_results(op, carried, bindings)
            return

        if force_unroll:
            self._emit_for_unrolled(circuit, op, qubit_map, clbit_map, bindings)
            return

        if self._loop_analyzer.should_unroll(op, bindings):
            self._emit_for_unrolled(circuit, op, qubit_map, clbit_map, bindings)
            return

        # Use Qiskit's native for_loop context manager
        with circuit.for_loop(indexset) as loop_param:  # type: ignore[call-overload]
            loop_bindings = bindings.copy()
            from qamomile.circuit.transpiler.passes.emit_support.control_flow_emission import (
                _bind_loop_var,
            )

            _bind_loop_var(loop_bindings, op, loop_param)
            # ``emit_qinit_reset=True`` mirrors the base ``emit_for`` native
            # branch (control_flow_emission.py) and the if/else branches
            # below: a fresh ``qmc.qubit(...)`` allocated inside the loop body
            # must be reset to |0> at the start of every iteration. Without it
            # the second and later iterations silently reuse the ancilla in its
            # post-measurement state, computing a wrong quantum state.
            self._emit_operations(
                circuit,
                op.operations,
                qubit_map,
                clbit_map,
                loop_bindings,
                emit_qinit_reset=True,
            )

    def _emit_if(
        self,
        circuit: "QuantumCircuit",
        op: IfOperation,
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Emit an if/else with Qiskit's dynamic-control context manager.

        Args:
            circuit (QuantumCircuit): Qiskit circuit being mutated.
            op (IfOperation): Conditional operation to emit.
            qubit_map (QubitMap): Current logical-to-physical qubit map.
            clbit_map (ClbitMap): Current logical-to-physical clbit map.
            bindings (dict[str, Any]): Active compile-time and loop-local
                bindings.

        Raises:
            EmitError: If the runtime condition or a branch merge cannot be
                represented safely.
        """
        condition = op.condition

        # Compile-time constant conditions are handled by the base class.
        if resolve_if_condition(condition, bindings) is not None:
            super()._emit_if(circuit, op, qubit_map, clbit_map, bindings)
            return

        if_test_condition = self._resolve_runtime_condition(
            circuit, condition, clbit_map, bindings
        )

        entry_overwrites = set(self._overwritten_runtime_condition_sources)
        with circuit.if_test(if_test_condition) as else_:
            self._emit_operations(
                circuit,
                op.true_operations,
                qubit_map,
                clbit_map,
                bindings,
                emit_qinit_reset=True,
            )
        true_overwrites = set(self._overwritten_runtime_condition_sources)
        self._overwritten_runtime_condition_sources.clear()
        self._overwritten_runtime_condition_sources.update(entry_overwrites)
        with else_:
            self._emit_operations(
                circuit,
                op.false_operations,
                qubit_map,
                clbit_map,
                bindings,
                emit_qinit_reset=True,
            )
        false_overwrites = set(self._overwritten_runtime_condition_sources)
        self._overwritten_runtime_condition_sources.clear()
        self._overwritten_runtime_condition_sources.update(
            true_overwrites | false_overwrites
        )

        # Register merge outputs after emitting both branches, matching the
        # base-class runtime contract. ResourceAllocator pre-registers the
        # qubit / clbit merges on this backend (so the physical mapping is
        # mostly a no-op here), but classical identity merges bind into
        # ``bindings`` only through register_classical_merge_aliases.
        from qamomile.circuit.transpiler.passes.emit_support.control_flow_emission import (  # noqa: E501
            register_classical_merge_aliases,
            register_merge_outputs,
        )

        register_merge_outputs(self, op, qubit_map, clbit_map, bindings)
        register_classical_merge_aliases(self, op, bindings, None)

    def _emit_while(
        self,
        circuit: "QuantumCircuit",
        op: WhileOperation,
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Emit while loop using Qiskit's while_loop context manager.

        Args:
            circuit (QuantumCircuit): The circuit being emitted into.
            op (WhileOperation): The while loop operation to emit.
            qubit_map (QubitMap): Map from qubit addresses to physical qubits.
            clbit_map (ClbitMap): Map from qubit addresses to clbit indices.
            bindings (dict[str, Any]): Current value bindings.

        Raises:
            EmitError: If the loop carries classical values (no runtime
                loop can thread them between iterations) or has no
                condition operand.
        """
        if op.region_args:
            # Mirror the base emit_while backstop: this override replaces
            # the base function entirely, and the transpile-time rejection
            # is the only earlier line of defense.
            carried_names = ", ".join(arg.var_name for arg in op.region_args)
            raise EmitError(
                "Loop-carried classical values in a while loop cannot be "
                f"emitted ({carried_names}): a runtime loop "
                "re-executes one static body and cannot thread a classical "
                "value between iterations.",
                operation="WhileOperation",
            )
        if not op.operands:
            raise EmitError(
                "WhileOperation requires a condition operand.",
                operation="WhileOperation",
            )

        condition = op.operands[0]
        condition_value = condition.value if hasattr(condition, "value") else condition
        while_condition = self._resolve_runtime_condition(
            circuit, condition_value, clbit_map, bindings
        )

        with circuit.while_loop(while_condition):  # type: ignore[call-overload]
            self._emit_operations(
                circuit,
                op.operations,
                qubit_map,
                clbit_map,
                bindings,
                emit_qinit_reset=True,
            )

    def _resolve_runtime_condition(
        self,
        circuit: "QuantumCircuit",
        condition: Any,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> Any:
        """Resolve a runtime if/while condition to a Qiskit ``if_test`` argument.

        The returned value is suitable for passing to
        ``QuantumCircuit.if_test`` / ``while_loop``: either a
        ``(clbit, value)`` tuple for a single-bit measurement condition, or
        a ``qiskit.circuit.classical.expr.Expr`` for a compound predicate
        (``CondOp`` / ``NotOp`` / ``CompOp`` over measurement bits)
        previously stored in ``bindings`` by ``_build_runtime_predicate_expr``.

        Args:
            circuit: The Qiskit circuit being emitted (for clbit lookup).
            condition: The IR condition Value.
            clbit_map: Map from ``QubitAddress`` to physical clbit index.
            bindings: Current bindings; may hold a backend ``Expr`` keyed by
                the condition's UUID.

        Returns:
            A condition object accepted by ``if_test`` / ``while_loop``.

        Raises:
            EmitError: If the condition is neither a measurement clbit nor a
                stored runtime expression.
        """
        condition_uuid = (
            condition.uuid if hasattr(condition, "uuid") else str(condition)
        )

        # Compound predicate built earlier by _build_runtime_predicate_expr.
        stored = bindings.get(condition_uuid)
        if stored is not None and not isinstance(stored, (bool, int, float)):
            return stored

        if isinstance(condition, Value):
            condition_addr = resolve_condition_address(
                condition, bindings, self._resolver
            )
        else:
            condition_addr = QubitAddress(condition_uuid)
        if condition_addr in clbit_map:
            clbit_idx = clbit_map[condition_addr]
            return (circuit.clbits[clbit_idx], 1)

        raise EmitError(
            "Runtime control-flow conditions (if / while) must come from "
            "measurement results or be bound before transpilation. The "
            "condition value was neither resolved at compile time nor "
            "backed by a measurement result."
        )

    def _emit_runtime_classical_expr(
        self,
        circuit: "QuantumCircuit",
        op: RuntimeClassicalExpr,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Lower ``RuntimeClassicalExpr`` to a Qiskit ``expr.Expr``.

        Counterpart of (and supersedes for the runtime path) the legacy
        ``_build_runtime_predicate_expr``. The IR has already declared
        this op runtime via ``ClassicalLoweringPass``, so we go straight
        to expression construction without a fold attempt.

        Args:
            circuit: The Qiskit circuit being emitted.
            op: The runtime classical expression to lower.
            clbit_map: Map from ``QubitAddress`` → physical clbit index.
            bindings: Current bindings; result is stored here.

        Raises:
            EmitError: If an operand cannot be resolved to a clbit /
                sub-expression / constant — meaning the IR is malformed
                or the lowering pass missed a case.
        """
        from qamomile.circuit.transpiler.errors import EmitError
        from qiskit.circuit.classical import expr

        # Typed-slot lookups when bindings is an EmitContext; fall back to
        # flat-dict access for legacy plain-dict callers (compat shim).
        get_runtime_expr = getattr(bindings, "get_runtime_expr", None)
        get_loop_var = getattr(bindings, "get_loop_var", None)
        params_slot = getattr(bindings, "_params", None)

        def resolve_operand(v: Any) -> Any:
            """Resolve an operand to a Qiskit ``Expr`` / ``Clbit`` / Python literal.

            Identity policy:
              1. Backend ``expr.Expr`` from a prior ``RuntimeClassicalExpr``:
                 read from the ``runtime_exprs`` typed slot via UUID.
              2. Constants: ``v.get_const()`` returns the IR-typed value
                 (Bit→bool, UInt→int, Float→float). No ``bool(...)``
                 coercion — that would clobber numeric arithmetic.
              3. Clbit references: ``resolve_condition_address(v, ...)``
                 keyed lookup in ``clbit_map`` — a scalar bit resolves to
                 ``QubitAddress(v.uuid)`` while a measured ``Vector[Bit]``
                 element resolves to ``QubitAddress(root_array.uuid,
                 root_index)`` (walking ``parent_array`` / ``slice_of``).
              4. User parameters: read from the ``parameters`` typed slot
                 by parameter name (the only legitimate name path).
              5. Loop variables: read from the ``loop_vars`` typed slot
                 via UUID (preserved through inline by the all_input_values
                 / replace_values protocol).

            No name fallback — every step keys on UUID or an explicit
            ``is_parameter()`` flag, never on display names.
            """
            if not hasattr(v, "uuid"):
                return None

            # 1. Backend expr already built for this Value (UUID-keyed).
            if callable(get_runtime_expr):
                expr_obj = get_runtime_expr(v.uuid)
                if expr_obj is not None:
                    return expr_obj
            else:
                stored = bindings.get(v.uuid) if hasattr(bindings, "get") else None
                if stored is not None and not isinstance(stored, (bool, int, float)):
                    return stored

            # 2. Constant — preserve IR type.
            if hasattr(v, "is_constant") and v.is_constant():
                return v.get_const()

            # 3. Clbit reference. ``Vector[Bit]`` element accesses route
            #    through ``QubitAddress(parent_array.uuid, index)``; scalar
            #    bits use ``QubitAddress(v.uuid)``.
            addr = resolve_condition_address(v, bindings, self._resolver)
            if addr in clbit_map:
                return circuit.clbits[clbit_map[addr]]

            # 4. User parameter (name-keyed by definition; surface from
            #    typed slot when available).
            if hasattr(v, "is_parameter") and v.is_parameter():
                pname = v.parameter_name()
                if pname:
                    if params_slot is not None and pname in params_slot:
                        return params_slot[pname]
                    if hasattr(bindings, "get"):
                        bound = bindings.get(pname)
                        if bound is not None:
                            return bound

            # 5. Loop variable (UUID-keyed).
            if callable(get_loop_var):
                lv = get_loop_var(v.uuid)
                if lv is not None:
                    return lv

            # 6. Legacy compat shim: plain-dict caller, scalar binding.
            if hasattr(bindings, "get"):
                stored = bindings.get(v.uuid)
                if isinstance(stored, (bool, int, float)):
                    return stored

            return None

        kind = op.kind
        if kind is RuntimeOpKind.NOT:
            inner = resolve_operand(op.operands[0])
            if inner is None:
                raise EmitError(
                    f"Cannot resolve operand for RuntimeClassicalExpr(NOT): "
                    f"{op.operands[0]!r}"
                )
            result = expr.logic_not(inner)
        else:
            assert kind is not None
            lhs = resolve_operand(op.operands[0])
            rhs = resolve_operand(op.operands[1])
            if lhs is None or rhs is None:
                raise EmitError(
                    f"Cannot resolve operands for RuntimeClassicalExpr({kind!r})"
                )
            result = self._build_qiskit_binary_expr(kind, lhs, rhs)
            if result is None:
                raise EmitError(
                    f"Unsupported RuntimeClassicalExpr kind for Qiskit backend: {kind}"
                )

        # Store result so downstream ``_emit_if`` / ``_emit_while`` /
        # nested ``RuntimeClassicalExpr`` can consume it.
        set_runtime_expr = getattr(bindings, "set_runtime_expr", None)
        if callable(set_runtime_expr):
            set_runtime_expr(op.results[0].uuid, result)
        else:
            bindings[op.results[0].uuid] = result

    @staticmethod
    def _build_qiskit_binary_expr(kind: RuntimeOpKind, lhs: Any, rhs: Any) -> Any:
        """Map a binary ``RuntimeOpKind`` to its Qiskit ``expr`` constructor.

        Every arm of ``RuntimeOpKind`` (except ``NOT``, which is handled
        upstream) is dispatched here. Kinds without a Qiskit ``expr``
        equivalent — currently ``FLOORDIV``, ``MOD`` and ``POW`` — raise
        ``NotImplementedError`` rather than silently returning ``None``,
        so the contract gap is loud at emit time instead of producing a
        misleading "Unsupported kind" error.
        """
        from qiskit.circuit.classical import expr

        match kind:
            # Logical
            case RuntimeOpKind.AND:
                return expr.logic_and(lhs, rhs)
            case RuntimeOpKind.OR:
                return expr.logic_or(lhs, rhs)
            # Comparison
            case RuntimeOpKind.EQ:
                return expr.equal(lhs, rhs)
            case RuntimeOpKind.NEQ:
                return expr.not_equal(lhs, rhs)
            case RuntimeOpKind.LT:
                return expr.less(lhs, rhs)
            case RuntimeOpKind.LE:
                return expr.less_equal(lhs, rhs)
            case RuntimeOpKind.GT:
                return expr.greater(lhs, rhs)
            case RuntimeOpKind.GE:
                return expr.greater_equal(lhs, rhs)
            # Arithmetic
            case RuntimeOpKind.ADD:
                return expr.add(lhs, rhs)
            case RuntimeOpKind.SUB:
                return expr.sub(lhs, rhs)
            case RuntimeOpKind.MUL:
                return expr.mul(lhs, rhs)
            case RuntimeOpKind.DIV:
                return expr.div(lhs, rhs)
            case RuntimeOpKind.FLOORDIV | RuntimeOpKind.MOD | RuntimeOpKind.POW:
                raise NotImplementedError(
                    f"RuntimeOpKind.{kind.name} is not supported by the Qiskit "
                    f"backend (Qiskit's classical expr has no equivalent). "
                    f"If you need this kind, fold it at compile time before "
                    f"the runtime classical-lowering pass."
                )
            case _:
                raise NotImplementedError(
                    f"Unhandled RuntimeOpKind in _build_qiskit_binary_expr: {kind!r}"
                )

    def _build_runtime_predicate_expr(
        self,
        circuit: "QuantumCircuit",
        op: "CompOp | CondOp | NotOp",
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> Any:
        """Build a Qiskit ``expr.Expr`` for an unlowered runtime predicate.

        Fallback path used by ``StandardEmitPass`` only when a
        ``CompOp``/``CondOp``/``NotOp`` reaches emit without having been
        rewritten to ``RuntimeClassicalExpr`` by ``ClassicalLoweringPass``
        — i.e., predicates that depend on emit-time-bound values not
        visible to the pre-emit lowering pass (e.g. computed from a loop
        variable that wraps a measurement). The primary path is
        ``_emit_runtime_classical_expr``; prefer extending the lowering
        pass over adding new cases here.

        Recursively resolves operands to either Qiskit ``Clbit`` references
        (via ``clbit_map``), constants, or already-built sub-expressions
        stored in ``bindings``. Returns ``None`` when any operand cannot be
        resolved — the caller leaves the predicate unbound, and the
        downstream ``if`` / ``while`` falls back to its (failing) clbit
        lookup, which raises a clear ``EmitError``.

        Supported ops:
            * ``CondOp(AND/OR)`` → ``expr.logic_and`` / ``expr.logic_or``
            * ``NotOp`` → ``expr.logic_not``
            * ``CompOp(EQ/NEQ)`` over Bit operands → ``expr.equal`` /
              ``expr.not_equal``

        Args:
            circuit: The Qiskit circuit being emitted (for clbit lookup).
            op: The unresolved classical predicate.
            clbit_map: Map from ``QubitAddress`` → physical clbit index.
            bindings: Current bindings; may hold sub-expression results
                keyed by Value UUIDs.

        Returns:
            A ``qiskit.circuit.classical.expr.Expr`` object, or ``None`` if
            the predicate cannot be expressed.
        """
        from qiskit.circuit.classical import expr

        def resolve_operand(v: Any) -> Any:
            """Operand → Qiskit ``Expr`` / ``Clbit`` / Python literal, or None."""
            if hasattr(v, "uuid"):
                stored_v = bindings.get(v.uuid)
                if stored_v is not None and not isinstance(stored_v, (bool, int)):
                    return stored_v
                if hasattr(v, "is_constant") and v.is_constant():
                    return bool(v.get_const())
                addr = (
                    resolve_condition_address(v, bindings, self._resolver)
                    if isinstance(v, Value)
                    else QubitAddress(v.uuid)
                )
                if addr in clbit_map:
                    return circuit.clbits[clbit_map[addr]]
                if isinstance(stored_v, (bool, int)):
                    return bool(stored_v)
            return None

        if isinstance(op, NotOp):
            inner = resolve_operand(op.operands[0])
            if inner is None:
                return None
            return expr.logic_not(inner)

        lhs = resolve_operand(op.operands[0])
        rhs = resolve_operand(op.operands[1])
        if lhs is None or rhs is None:
            return None

        if isinstance(op, CondOp):
            if op.kind is CondOpKind.AND:
                return expr.logic_and(lhs, rhs)
            if op.kind is CondOpKind.OR:
                return expr.logic_or(lhs, rhs)
            return None

        if isinstance(op, CompOp):
            if op.kind is CompOpKind.EQ:
                return expr.equal(lhs, rhs)
            if op.kind is CompOpKind.NEQ:
                return expr.not_equal(lhs, rhs)
            return None

        return None  # type: ignore[unreachable]

    def _emit_pauli_evolve(
        self,
        circuit: "QuantumCircuit",
        op: "PauliEvolveOp",
        qubit_map: QubitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Emit Pauli evolution using Qiskit's PauliEvolutionGate.

        Falls back to the shared gadget decomposition — always via
        ``_emit_gadget_pauli_evolve``, which completes the Hamiltonian's
        constant term as a circuit global phase — when native composite
        emission is disabled or the native path cannot resolve the
        Hamiltonian, gamma, or qubit indices.

        Args:
            circuit (QuantumCircuit): Qiskit circuit being emitted into.
            op (PauliEvolveOp): The Pauli evolution operation to emit.
            qubit_map (QubitMap): Physical qubit map the operation's
                quantum operands resolve against; updated in place with
                the evolved-register result addresses.
            bindings (dict[str, Any]): Active emit bindings used to
                resolve the Hamiltonian, gamma, and register shape.

        Raises:
            EmitError: If the Hamiltonian is non-Hermitian (a term or the
                constant has a non-real coefficient), the Hamiltonian is
                larger than the register, or the gadget fallback cannot
                resolve the Hamiltonian, gamma, or a term qubit.
        """
        import qamomile.observable as qm_o

        if not self._use_native_composite:
            self._emit_gadget_pauli_evolve(circuit, op, qubit_map, bindings)
            return

        # Resolve Hamiltonian from bindings
        obs_value = op.observable
        hamiltonian = None
        if hasattr(obs_value, "name") and obs_value.name in bindings:
            hamiltonian = bindings[obs_value.name]
        if hamiltonian is None and hasattr(obs_value, "uuid"):
            hamiltonian = bindings.get(obs_value.uuid)
        if not isinstance(hamiltonian, qm_o.Hamiltonian):
            self._emit_gadget_pauli_evolve(circuit, op, qubit_map, bindings)
            return

        # Resolve gamma: concrete float OR backend Parameter for parametric
        # gamma (scalar / array element). Qiskit's PauliEvolutionGate accepts
        # a Parameter (or ParameterExpression) as ``time``.
        from qamomile.circuit.transpiler.passes.emit_support.pauli_evolve_emission import (
            _resolve_gamma,
            validate_hamiltonian_within_register,
        )

        gamma = _resolve_gamma(self, op, bindings)
        if gamma is None:
            self._emit_gadget_pauli_evolve(circuit, op, qubit_map, bindings)
            return

        # Validate qubit count: logical array size vs Hamiltonian. A
        # Hamiltonian smaller than the register is embedded into the
        # register (identity on the untouched qubits) by appending the
        # evolution gate onto only its declared qubits below.
        input_array = op.qubits
        num_h_qubits = hamiltonian.num_qubits
        from qamomile.circuit.ir.value import ArrayValue

        if isinstance(input_array, ArrayValue) and input_array.shape:
            n_resolved = self._resolver.resolve_int_value(
                input_array.shape[0], bindings
            )
            if n_resolved is not None:
                validate_hamiltonian_within_register(num_h_qubits, n_resolved)

        # Validate Hermitian (real coefficients)
        from qamomile.observable.hamiltonian import (
            HERMITIAN_IMAG_ATOL,
            PAULI_TERM_ZERO_ATOL,
        )

        for operators, coeff in hamiltonian:
            if abs(coeff.imag) > HERMITIAN_IMAG_ATOL:
                raise EmitError(
                    f"PauliEvolveOp requires a Hermitian Hamiltonian "
                    f"(real coefficients), but found complex coefficient "
                    f"{coeff} on term {operators}.",
                    operation="PauliEvolveOp",
                )

        # Hamiltonian.__iter__ yields only the Pauli terms, so the loop
        # above never sees the constant (identity) term. Validate it here
        # for ALL Hamiltonians so the native path rejects a non-real
        # constant with the same clean EmitError as the gadget fallback,
        # instead of letting it slip through to a raw Qiskit ValueError
        # inside PauliEvolutionGate (for num_h_qubits > 0).
        if abs(hamiltonian.constant.imag) > HERMITIAN_IMAG_ATOL:
            raise EmitError(
                f"PauliEvolveOp requires a Hermitian Hamiltonian "
                f"(real coefficients), but found a complex constant "
                f"{hamiltonian.constant}.",
                operation="PauliEvolveOp",
            )

        if num_h_qubits == 0:
            # An empty / constant-only Hamiltonian evolves the register by
            # the global phase exp(-i*gamma*constant) only. Record it as
            # the circuit's global phase instead of skipping it: standalone
            # it stays unobservable, but when this circuit is converted to
            # a gate and placed under controls (``_blockvalue_to_gate`` +
            # ``Gate.control``) Qiskit promotes the definition's global
            # phase to the observable relative phase on the all-controls-on
            # subspace — matching how the shared controlled path and CUDA-Q
            # re-apply the constant term under ``qmc.control``. Building
            # the evolution gate instead would fail:
            # ``hamiltonian_to_sparse_pauli_op`` widens a 0-qubit
            # Hamiltonian to a 1-qubit ``SparsePauliOp(["I"])`` whose
            # ``PauliEvolutionGate`` cannot be appended onto the empty
            # qubit list. (For ``num_h_qubits > 0`` the constant is already
            # carried inside the ``SparsePauliOp``, so this branch must not
            # apply.) The result register stays resolvable through the
            # allocator's element aliases. The constant's Hermiticity was
            # already validated above for all Hamiltonians, so only the
            # magnitude check remains here.
            constant = hamiltonian.constant
            if abs(constant) > PAULI_TERM_ZERO_ATOL:
                # Works for both concrete float gamma and a backend
                # Parameter (ParameterExpression global phase is bound
                # together with the rest of the circuit parameters).
                circuit.global_phase += -float(constant.real) * gamma
            return

        qubit_indices: list[int] = []
        for i in range(num_h_qubits):
            addr = QubitAddress(input_array.uuid, i)
            if addr in qubit_map:
                qubit_indices.append(qubit_map[addr])
            else:
                self._emit_gadget_pauli_evolve(circuit, op, qubit_map, bindings)
                return

        try:
            from qamomile.qiskit.observable import hamiltonian_to_sparse_pauli_op
            from qiskit.circuit.library import PauliEvolutionGate

            sparse_op = hamiltonian_to_sparse_pauli_op(hamiltonian)
            # ``time`` accepts both ``float`` and Qiskit ``Parameter`` /
            # ``ParameterExpression``. For parametric gamma we pass the
            # backend parameter through so it can be bound at run-time.
            time_arg = float(gamma) if isinstance(gamma, (int, float)) else gamma
            evo_gate = PauliEvolutionGate(sparse_op, time=time_arg)
            circuit.append(evo_gate, qubit_indices)
        except ImportError:
            # Fallback to manual decomposition when Qiskit library unavailable
            self._emit_gadget_pauli_evolve(circuit, op, qubit_map, bindings)
            return

        # Map result array to same physical qubits
        result_array = op.evolved_qubits
        for i, phys_idx in enumerate(qubit_indices):
            result_addr = QubitAddress(result_array.uuid, i)
            if result_addr not in qubit_map:
                qubit_map[result_addr] = phys_idx

    def _emit_gadget_pauli_evolve(
        self,
        circuit: "QuantumCircuit",
        op: "PauliEvolveOp",
        qubit_map: QubitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Delegate to the shared gadget emitter and complete the constant term.

        The shared gadget decomposition emits only the Hamiltonian's
        Pauli terms — a constant (identity) term contributes no gates
        there, so it is silently dropped. Every gadget delegation from
        ``_emit_pauli_evolve`` goes through this wrapper so the dropped
        constant is re-applied as a Qiskit global phase exactly once,
        keeping the fallback's phase behavior aligned with the native
        ``PauliEvolutionGate`` path (whose ``SparsePauliOp`` carries the
        constant).

        Args:
            circuit (QuantumCircuit): Qiskit circuit being emitted into.
            op (PauliEvolveOp): The Pauli evolution operation to emit.
            qubit_map (QubitMap): Physical qubit map the operation's
                quantum operands resolve against; updated in place with
                the evolved-register result addresses.
            bindings (dict[str, Any]): Active emit bindings used to
                resolve the Hamiltonian, gamma, and register shape.

        Raises:
            EmitError: Propagated from the shared gadget emitter
                (unresolvable Hamiltonian / gamma / term qubit,
                non-Hermitian term coefficient, or a Hamiltonian larger
                than the register), or raised by the constant-phase
                completion when the constant term has a non-real
                coefficient.
        """
        super()._emit_pauli_evolve(circuit, op, qubit_map, bindings)
        self._emit_gadget_constant_phase(circuit, op, bindings)

    def _emit_gadget_constant_phase(
        self,
        circuit: "QuantumCircuit",
        op: "PauliEvolveOp",
        bindings: dict[str, Any],
    ) -> None:
        """Re-apply the constant term dropped by the gadget decomposition.

        ``exp(-i*gamma*H)`` with ``H = ... + c`` carries the phase
        ``exp(-i*gamma*c)``. Standalone this is an unobservable global
        phase, but when the emitted circuit is converted to a gate and
        placed under controls (``_blockvalue_to_gate`` + ``Gate.control``)
        Qiskit promotes the definition's global phase to the observable
        relative phase on the all-controls-on subspace — matching the
        native ``PauliEvolutionGate`` path, the shared controlled path's
        explicit ``P(-gamma*c)`` on a control, and CUDA-Q's controlled
        constant re-application. Hamiltonian and gamma resolution mirrors
        the shared gadget emitter (same resolver and ``_resolve_gamma``),
        so whenever the gadget emission succeeded the constant resolves
        identically; if either is unresolvable this method is a no-op
        (every such reachable case has already raised inside the gadget
        emission).

        Args:
            circuit (QuantumCircuit): Circuit the gadget fallback just
                emitted into; its ``global_phase`` is updated in place.
            op (PauliEvolveOp): The Pauli evolution operation being
                emitted.
            bindings (dict[str, Any]): Active emit bindings used to
                resolve the Hamiltonian and gamma.

        Raises:
            EmitError: If the constant term has a non-real coefficient
                (non-Hermitian Hamiltonian).
        """
        import qamomile.observable as qm_o
        from qamomile.circuit.transpiler.passes.emit_support.pauli_evolve_emission import (
            _resolve_gamma,
        )
        from qamomile.observable.hamiltonian import (
            HERMITIAN_IMAG_ATOL,
            PAULI_TERM_ZERO_ATOL,
        )

        hamiltonian = self._resolver.resolve_bound_value(op.observable, bindings)
        if not isinstance(hamiltonian, qm_o.Hamiltonian):
            return
        constant = hamiltonian.constant
        if abs(constant) <= PAULI_TERM_ZERO_ATOL:
            return
        if abs(constant.imag) > HERMITIAN_IMAG_ATOL:
            raise EmitError(
                f"PauliEvolveOp requires a Hermitian Hamiltonian "
                f"(real coefficients), but found a complex constant "
                f"{constant}.",
                operation="PauliEvolveOp",
            )
        gamma = _resolve_gamma(self, op, bindings)
        if gamma is None:
            return
        # Works for both concrete float gamma and a backend Parameter
        # (ParameterExpression global phase is bound together with the
        # rest of the circuit parameters).
        circuit.global_phase += -float(constant.real) * gamma


class QiskitExecutor(QuantumExecutor["QuantumCircuit"]):
    """Qiskit quantum executor using a safe local simulator or other backends.

    Example:
        executor = QiskitExecutor()  # Uses AerSimulator when available
        counts = executor.execute(circuit, shots=1000)
        # counts: {"00": 512, "11": 512}

        # With expectation value estimation
        from qamomile.qiskit.observable import QiskitExpectationEstimator
        executor = QiskitExecutor(estimator=QiskitExpectationEstimator())
        exp_val = executor.estimate(circuit, observable)
    """

    def __init__(self, backend=None, estimator=None):
        """Initialize executor with backend and optional estimator.

        Args:
            backend: Qiskit backend (defaults to AerSimulator if available)
            estimator: Optional QiskitExpectationEstimator for expectation values
        """
        self.backend = backend
        self._estimator = estimator

        if self.backend is None:
            try:
                from qiskit_aer import AerSimulator

                self.backend = AerSimulator(max_parallel_threads=1)
            except ImportError:
                try:
                    from qiskit.providers.basic_provider import BasicSimulator

                    self.backend = BasicSimulator()
                except ImportError:
                    pass

    def execute(self, circuit: "QuantumCircuit", shots: int) -> dict[str, int]:
        """Execute circuit and return bitstring counts.

        Args:
            circuit (QuantumCircuit): The quantum circuit to execute.
            shots (int): Number of measurement shots.

        Returns:
            dict[str, int]: Mapping from bitstrings to counts (for example,
                ``{"00": 512, "11": 512}``). A zero-qubit circuit returns
                ``{"": shots}`` without invoking the backend.

        Raises:
            RuntimeError: If no Qiskit backend is configured.
        """
        from qiskit import transpile

        if self.backend is None:
            raise RuntimeError("No backend available for execution")
        if circuit.num_qubits == 0 and circuit.num_clbits == 0:
            return {"": shots}

        circuit_with_meas = self._ensure_measurements(circuit)
        transpiled = transpile(circuit_with_meas, self.backend)
        job = self.backend.run(transpiled, shots=shots)
        return job.result().get_counts()

    def bind_parameters(
        self,
        circuit: "QuantumCircuit",
        bindings: dict[str, Any],
        parameter_metadata: ParameterMetadata,
    ) -> "QuantumCircuit":
        """Bind parameter values to the Qiskit circuit.

        Args:
            circuit: The parameterized circuit
            bindings: Dict mapping parameter names (indexed format) to values
            parameter_metadata: Metadata about circuit parameters

        Returns:
            New circuit with parameters bound
        """
        qiskit_bindings = {}
        for param_info in parameter_metadata.parameters:
            if param_info.name in bindings:
                qiskit_bindings[param_info.backend_param] = bindings[param_info.name]

        return circuit.assign_parameters(qiskit_bindings)

    def estimate(
        self,
        circuit: "QuantumCircuit",
        hamiltonian: "qm_o.Hamiltonian",
        params: Sequence[float] | None = None,
    ) -> float:
        """Estimate the expectation value of a Hamiltonian.

        Args:
            circuit: Qiskit QuantumCircuit (state preparation ansatz)
            hamiltonian: The qamomile.observable.Hamiltonian to measure
            params: Optional parameter values for parametric circuits

        Returns:
            The estimated expectation value

        Raises:
            RuntimeError: If no estimator is configured
        """
        if self._estimator is None:
            # Create default Qiskit Estimator
            try:
                from qiskit.primitives import StatevectorEstimator

                self._estimator = StatevectorEstimator()
            except ImportError:
                try:
                    # Fallback for older Qiskit versions
                    from qiskit.primitives import (
                        Estimator,  # type: ignore[attr-defined]
                    )

                    self._estimator = Estimator()
                except ImportError:
                    from qiskit_aer.primitives import Estimator

                    self._estimator = Estimator()

        # Convert Hamiltonian to SparsePauliOp
        from qamomile.qiskit.observable import hamiltonian_to_sparse_pauli_op

        sparse_pauli_op = hamiltonian_to_sparse_pauli_op(hamiltonian)

        # Run estimation
        if params is not None:
            param_values = list(params)
        else:
            param_values = []

        # Check if this is V1 or V2 interface
        # V2 interface (new): run([(circuit, observable, params)])
        # V1 interface (old): run(circuits, observables, parameter_values)
        estimator_run = cast(Any, self._estimator).run
        try:
            # Try V2 interface first
            job = estimator_run([(circuit, sparse_pauli_op, param_values)])
            result = job.result()
            return float(result[0].data.evs)
        except (TypeError, AttributeError):
            # Fall back to V1 interface
            job = estimator_run(
                [circuit],
                [sparse_pauli_op],
                [param_values] if param_values else None,
            )
            result = job.result()
            return float(result.values[0])

    def _ensure_measurements(self, circuit: "QuantumCircuit") -> "QuantumCircuit":
        """Ensure circuit has measurements, adding measure_all if needed."""
        if circuit.num_clbits > 0:
            return circuit

        circuit_copy = circuit.copy()
        circuit_copy.measure_all()
        return circuit_copy


class QiskitTranspiler(Transpiler["QuantumCircuit"]):
    """Qiskit backend transpiler.

    Converts Qamomile QKernels into Qiskit QuantumCircuits.

    Args:
        use_native_composite: If True (default), use native Qiskit library
                              implementations for QFT/IQFT. If False, use
                              manual decomposition for all composite gates.

    Example:
        from qamomile.qiskit import QiskitTranspiler
        import qamomile as qm

        @qm.qkernel
        def bell_state(q0: qm.Qubit, q1: qm.Qubit) -> tuple[qm.Bit, qm.Bit]:
            q0 = qm.h(q0)
            q0, q1 = qm.cx(q0, q1)
            return qm.measure(q0), qm.measure(q1)

        transpiler = QiskitTranspiler()
        circuit = transpiler.to_circuit(bell_state)
        print(circuit.draw())
    """

    def __init__(self, use_native_composite: bool = True):
        """Initialize the Qiskit transpiler.

        Args:
            use_native_composite: If True, use native Qiskit implementations
                                  for QFT/IQFT. If False, use manual decomposition.
        """
        self._use_native_composite = use_native_composite

    def _create_segmentation_pass(self) -> SegmentationPass:
        return SegmentationPass()

    def _create_emit_pass(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> EmitPass["QuantumCircuit"]:
        return QiskitEmitPass(
            bindings, parameters, use_native_composite=self._use_native_composite
        )

    def executor(  # type: ignore[override]
        self,
        backend=None,
    ) -> QiskitExecutor:
        """Create a Qiskit executor.

        Args:
            backend: Qiskit backend (defaults to AerSimulator)

        Returns:
            QiskitExecutor configured with the backend
        """
        return QiskitExecutor(backend)
