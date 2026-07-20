"""Regression: ``Transpiler.transpile`` rejects ``parameters``/``bindings`` overlap.

A parameter name appearing in both ``bindings`` and ``parameters`` is
fundamentally ambiguous (placeholder value vs runtime symbol) and used to
silently miscompile control-flow predicates that depended on the parameter
array's elements (Issue #354 B-series). After Phase 1, the overlap raises
``ValueError`` at the public API entry, eliminating the silent miscompilation
class entirely.
"""

from __future__ import annotations

import pytest

import qamomile.circuit as qmc


@pytest.fixture
def qiskit_transpiler():
    """Return a QiskitTranspiler, skipping the test if qiskit is unavailable."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    return QiskitTranspiler()


@qmc.qkernel
def _identity_kernel(theta: qmc.Float) -> qmc.Bit:
    """Trivial kernel used to exercise the disjointness check.

    Returns:
        Measurement of a Hadamard-prepared qubit.
    """
    q = qmc.qubit(name="q")
    q = qmc.rx(q, theta)
    return qmc.measure(q)


class TestParamBindingsDisjoint:
    """Validates the API-level disjointness contract."""

    def test_overlap_single_name_rejected(self, qiskit_transpiler) -> None:
        """A single shared name between bindings and parameters raises ValueError."""
        with pytest.raises(ValueError, match=r"appear in both"):
            qiskit_transpiler.transpile(
                _identity_kernel,
                bindings={"theta": 0.5},
                parameters=["theta"],
            )

    def test_overlap_multiple_names_listed_in_message(self, qiskit_transpiler) -> None:
        """Multiple shared names are surfaced in the error message for debuggability."""

        @qmc.qkernel
        def two_param(a: qmc.Float, b: qmc.Float) -> qmc.Bit:
            """Two-parameter kernel for the multi-name overlap test."""
            q = qmc.qubit(name="q")
            q = qmc.rx(q, a)
            q = qmc.ry(q, b)
            return qmc.measure(q)

        with pytest.raises(ValueError) as excinfo:
            qiskit_transpiler.transpile(
                two_param,
                bindings={"a": 0.1, "b": 0.2},
                parameters=["a", "b"],
            )
        message = str(excinfo.value)
        assert "'a'" in message and "'b'" in message

    def test_disjoint_bindings_and_parameters_compile(self, qiskit_transpiler) -> None:
        """The QAOA-style pattern (compile-time scalar + runtime array) compiles cleanly."""

        @qmc.qkernel
        def qaoa_like(p: qmc.UInt, gamma: qmc.Float) -> qmc.Bit:
            """Compile-time depth ``p`` plus runtime rotation angle ``gamma``."""
            q = qmc.qubit(name="q")
            for _ in qmc.range(p):
                q = qmc.rx(q, gamma)
            return qmc.measure(q)

        # No overlap: ``p`` is compile-time bound, ``gamma`` is runtime symbolic.
        exe = qiskit_transpiler.transpile(
            qaoa_like,
            bindings={"p": 2},
            parameters=["gamma"],
        )
        # Sanity-check that the runtime parameter actually flows through.
        result = exe.sample(
            qiskit_transpiler.executor(),
            shots=8,
            bindings={"gamma": 0.5},
        ).result()
        assert sum(count for _, count in result.results) == 8

    def test_only_bindings_no_overlap_check(self, qiskit_transpiler) -> None:
        """``bindings`` alone (no ``parameters``) is unaffected by the disjointness check."""
        exe = qiskit_transpiler.transpile(_identity_kernel, bindings={"theta": 0.0})
        result = exe.sample(qiskit_transpiler.executor(), shots=4).result()
        assert sum(count for _, count in result.results) == 4

    def test_only_parameters_no_overlap_check(self, qiskit_transpiler) -> None:
        """``parameters`` alone (no ``bindings``) is unaffected by the disjointness check."""
        exe = qiskit_transpiler.transpile(_identity_kernel, parameters=["theta"])
        result = exe.sample(
            qiskit_transpiler.executor(),
            shots=4,
            bindings={"theta": 0.3},
        ).result()
        assert sum(count for _, count in result.results) == 4

    def test_required_scalar_is_auto_detected_as_runtime_parameter(
        self, qiskit_transpiler
    ) -> None:
        """The build manifest reaches emit when ``parameters`` is omitted."""
        executable = qiskit_transpiler.transpile(_identity_kernel)
        assert executable.parameter_names == ["theta"]

        result = executable.sample(
            qiskit_transpiler.executor(),
            shots=4,
            bindings={"theta": 0.3},
        ).result()
        assert sum(count for _, count in result.results) == 4

    def test_unbound_scalar_next_to_binding_is_auto_detected(
        self, qiskit_transpiler
    ) -> None:
        """A partially bound kernel preserves every remaining scalar symbol."""

        @qmc.qkernel
        def partially_bound(a: qmc.Float, b: qmc.Float) -> qmc.Bit:
            """Use one compile-time and one runtime rotation angle."""
            q = qmc.qubit("q")
            q = qmc.rx(q, a)
            q = qmc.rz(q, b)
            return qmc.measure(q)

        executable = qiskit_transpiler.transpile(
            partially_bound,
            bindings={"a": 0.7},
        )
        assert executable.parameter_names == ["b"]

        result = executable.sample(
            qiskit_transpiler.executor(),
            shots=4,
            bindings={"b": 0.2},
        ).result()
        assert sum(count for _, count in result.results) == 4


class TestStepByStepDisjointness:
    """The disjointness rule is enforced on every entry point, not just transpile.

    Before this fix the overlap check lived only inside ``transpile``. The
    documented step-by-step API (``to_block`` → … → ``emit``) and the frontend
    ``build`` bypassed it, so a name in both ``bindings`` and ``parameters``
    was silently baked in (its runtime parameter dropped) — the exact #354
    class the check exists to prevent. The check now runs in the shared helper
    called from ``build`` / ``to_block`` / ``emit`` / ``transpile``.
    """

    def test_build_overlap_rejected(self) -> None:
        """``QKernel.build`` rejects a name in both parameters and kwargs."""
        with pytest.raises(ValueError, match=r"appear in both"):
            _identity_kernel.build(parameters=["theta"], theta=0.5)

    def test_to_block_overlap_rejected(self, qiskit_transpiler) -> None:
        """``Transpiler.to_block`` rejects the overlap before tracing."""
        with pytest.raises(ValueError, match=r"appear in both"):
            qiskit_transpiler.to_block(
                _identity_kernel,
                bindings={"theta": 0.5},
                parameters=["theta"],
            )

    def test_to_block_without_arguments_applies_python_defaults(
        self, qiskit_transpiler
    ) -> None:
        """``to_block`` uses the validated build path even with no options."""

        @qmc.qkernel
        def with_default(theta: qmc.Float = 0.75) -> qmc.Bit:
            """Rotate by a Python-defaulted compile-time angle."""
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        block = qiskit_transpiler.to_block(with_default)
        theta_slot = next(slot for slot in block.param_slots if slot.name == "theta")
        assert theta_slot.bound_value == 0.75

    def test_to_block_without_arguments_validates_required_bindings(
        self, qiskit_transpiler
    ) -> None:
        """A required non-parameterizable argument cannot bypass validation."""

        @qmc.qkernel
        def requires_bit(flag: qmc.Bit) -> qmc.Bit:
            """Return a required compile-time Bit input."""
            return flag

        with pytest.raises(ValueError, match="Argument 'flag' must be provided"):
            qiskit_transpiler.to_block(requires_bit)

    def test_emit_overlap_rejected(self, qiskit_transpiler) -> None:
        """``Transpiler.emit`` rejects the overlap on the step-by-step path.

        This is the exact bypass the fix closes: a plan built without overlap,
        then emitted with a name in both ``bindings`` and ``parameters``, used
        to silently bake the binding and drop the runtime parameter.
        """
        block = qiskit_transpiler.to_block(_identity_kernel, parameters=["theta"])
        block = qiskit_transpiler.resolve_parameter_shapes(
            qiskit_transpiler.substitute(block), {}
        )
        block = qiskit_transpiler.inline(block)
        block = qiskit_transpiler.affine_validate(block)
        block = qiskit_transpiler.partial_eval(block, {})
        block = qiskit_transpiler.analyze(block)
        plan = qiskit_transpiler.plan(block)

        with pytest.raises(ValueError, match=r"appear in both"):
            qiskit_transpiler.emit(plan, bindings={"theta": 0.5}, parameters=["theta"])

    def test_emit_disjoint_still_works(self, qiskit_transpiler) -> None:
        """A disjoint step-by-step emit is unaffected (no false positive)."""
        block = qiskit_transpiler.to_block(_identity_kernel, parameters=["theta"])
        block = qiskit_transpiler.resolve_parameter_shapes(
            qiskit_transpiler.substitute(block), {}
        )
        block = qiskit_transpiler.inline(block)
        block = qiskit_transpiler.affine_validate(block)
        block = qiskit_transpiler.partial_eval(block, {})
        block = qiskit_transpiler.analyze(block)
        plan = qiskit_transpiler.plan(block)

        exe = qiskit_transpiler.emit(plan, parameters=["theta"])
        assert exe.parameter_names == ["theta"]

    def test_direct_emit_pass_construction_rejected(self, qiskit_transpiler) -> None:
        """Constructing an emit pass directly with an overlap raises.

        ``EmitPass.__init__`` is the innermost emit-side choke point: it guards
        even the advanced path that builds a pass directly via
        ``Transpiler._create_emit_pass``, skipping the ``transpile`` / ``emit``
        wrappers. The concrete backend pass reaches the base ``__init__`` via
        ``super().__init__``, so the check fires here too.
        """
        with pytest.raises(ValueError, match=r"appear in both"):
            qiskit_transpiler._create_emit_pass(
                bindings={"theta": 0.5}, parameters=["theta"]
            )

    def test_helper_unit(self) -> None:
        """The shared helper raises exactly on overlap and is None-safe."""
        from qamomile.circuit.frontend.param_validation import (
            validate_bindings_parameters_disjoint,
        )

        # None / empty inputs never raise.
        validate_bindings_parameters_disjoint(None, None)
        validate_bindings_parameters_disjoint({"a": 1}, None)
        validate_bindings_parameters_disjoint(None, ["a"])
        validate_bindings_parameters_disjoint({"a": 1}, ["b"])

        # Overlap raises with both names listed.
        with pytest.raises(ValueError, match=r"appear in both"):
            validate_bindings_parameters_disjoint({"a": 1, "b": 2}, ["a", "b"])


class TestIssue354BSeriesRepro:
    """The exact repro from Issue #354 B-series no longer silently miscompiles.

    Pre-fix: ``transpile(basis_only, bindings={'input': [0]*4}, parameters=['input'])``
    silently dropped the X gates (sample returned ``[((0,0,0,0), shots)]`` regardless
    of the runtime ``input`` binding). Post-fix: the same call raises ``ValueError``
    at the API boundary, making the silent-wrong-answer impossible.
    """

    def test_repro_now_raises_value_error(self, qiskit_transpiler) -> None:
        """The B-series pattern is rejected with a clear ValueError."""

        @qmc.qkernel
        def computational_basis_state(
            q: qmc.Vector[qmc.Qubit],
            input: qmc.Vector[qmc.UInt],
        ) -> qmc.Vector[qmc.Qubit]:
            """Apply X to qubits where the corresponding input bit is 1."""
            n = q.shape[0]
            for i in qmc.range(n):
                if input[i] == 1:
                    q[i] = qmc.x(q[i])
            return q

        @qmc.qkernel
        def basis_only(input: qmc.Vector[qmc.UInt]) -> qmc.Vector[qmc.Bit]:
            """Wrap ``computational_basis_state`` in a measure-all entrypoint."""
            q = qmc.qubit_array(input.shape[0], name="q")
            q = computational_basis_state(q, input)
            return qmc.measure(q)

        with pytest.raises(ValueError, match=r"appear in both"):
            qiskit_transpiler.transpile(
                basis_only,
                bindings={"input": [0] * 4},
                parameters=["input"],
            )
