# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# tags: [tutorial]
# ---
#
# # Controlling Gates and Sub-Kernels with `qmc.control`
#
# [Tutorial 03](03_vector_slicing.ipynb) showed how `VectorView`
# slices let one helper kernel operate on a contiguous chunk of a
# larger register. This chapter is about a related but distinct
# building block: turning *any* gate (built-in or a custom
# `@qmc.qkernel`) into a *controlled* version of itself with
# `qmc.control`.
#
# `qmc.control(fn)` returns a `ControlledGate` wrapper that, when
# called, emits the controlled-U form of `fn`. The wrapper has two
# modes — *concrete* (Python integer control count) and *symbolic*
# (the count is a `qmc.UInt` kernel parameter). Most features
# (`power=`, default args, sub-kernels that take `Vector[Qubit]`,
# ...) work in both modes; a few features are specific to one or
# the other. This chapter is organised around exactly that split:
#
# - Section 3 — patterns that work in **both** modes.
# - Section 4 — patterns that work only in **concrete** mode.
# - Section 5 — patterns that work only in **symbolic** mode.
# - Section 6 — patterns that **do not work**, each labelled with
#   the exception type it raises and the mode that triggers it.

# %%
# Install the latest Qamomile from pip.
# # !pip install qamomile

# %%
import math

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.errors import QubitConsumedError
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()


# %% [markdown]
# ## 1. The minimal example: controlled-RX
#
# The smallest interesting use of `qmc.control` is wrapping a single
# built-in rotation. `qmc.rx(q, angle)` is a one-qubit gate; passing
# it to `qmc.control` produces a two-qubit controlled-RX.


# %%
@qmc.qkernel
def crx_demo() -> qmc.Bit:
    c = qmc.qubit(name="c")
    t = qmc.qubit(name="t")
    # Drive the control to |1> so the controlled rotation fires.
    c = qmc.x(c)
    crx = qmc.control(qmc.rx)
    c, t = crx(c, t, angle=math.pi)
    return qmc.measure(t)


crx_demo.draw()


# %% [markdown]
# A few observations on the call site:
#
# - `qmc.control(qmc.rx)` is evaluated at *decoration time*. The
#   returned `ControlledGate` (here bound to `crx`) is reusable.
# - When you call `crx(c, t, angle=...)`, the control qubits come
#   first as positional arguments, then the targets, then any
#   classical keyword arguments — the same order as the underlying
#   `qmc.rx(q, angle)` signature, with one extra control prefix.
# - The keyword name for the classical parameter is whatever the
#   wrapped function uses (`angle` for `qmc.rx`, `theta` for
#   `qmc.p`, etc.).

# %% [markdown]
# ## 2. Two modes at a glance
#
# Almost every shape `qmc.control` accepts falls into one of two
# modes, decided by what you pass for `num_controls`:
#
# | Mode | `num_controls` | Control argument(s) | `controlled_indices=` |
# | --- | --- | --- | --- |
# | **Concrete** | `int` (default `1`) | one or more `Qubit` / `VectorView` / `Vector[Qubit]` positional args whose qubit counts sum to `num_controls` | not allowed |
# | **Symbolic** | a `qmc.UInt` handle | a single `Vector[Qubit]` *pool* (the actual controls are selected from it at compile time) | optional — picks which slots of the pool are active |
#
# Use *concrete* mode when you know the control count at
# decoration time and you want to wire specific qubits in by name.
# Use *symbolic* mode when the count is a kernel parameter or an
# expression over one (`num_controls=n - 1` is the textbook
# multi-controlled form). The next three sections enumerate which
# feature lives in which mode.

# %% [markdown]
# ## 3. Patterns that work in BOTH modes
#
# Everything in this section is mode-agnostic — the same call site
# (modulo `num_controls`) compiles in either concrete or symbolic
# mode. Where a concrete example is shown the same shape works
# verbatim with `num_controls=<UInt handle>` if you want to defer
# the count to `bindings`.

# %% [markdown]
# ### 3.1 Wrapping any callable
#
# `qmc.control` accepts either a built-in gate function (`qmc.rx`,
# `qmc.h`, `qmc.p`, ...) or a user-defined `@qmc.qkernel`. The
# wrapper does not care which:


# %%
@qmc.qkernel
def _h_then_rx(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    q = qmc.h(q)
    q = qmc.rx(q, theta)
    return q


@qmc.qkernel
def wrap_any_callable_demo() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(3, "q")  # q[0] = control, q[1] / q[2] = targets
    q[0] = qmc.x(q[0])
    # Built-in gate function — no wrapper kernel required.
    ch = qmc.control(qmc.h)
    q[0], q[1] = ch(q[0], q[1])
    # User @qkernel with classical parameter.
    cg = qmc.control(_h_then_rx)
    q[0], q[2] = cg(q[0], q[2], theta=math.pi / 4)
    return qmc.measure(q)


# (The same applies in symbolic mode: replace num_controls=1 with
# num_controls=<UInt>.  The body of the wrapped callable is
# unchanged.)
wrap_any_callable_demo.draw()


# %% [markdown]
# ### 3.2 Sub-kernels taking `Vector[Qubit]`
#
# A wrapped kernel may take a `Vector[Qubit]` argument. The caller
# passes a `Vector` or `VectorView` of the matching length, and
# the controlled-U emit pass expands the body per-element. This is
# how you write a controlled register-wide operation without
# spelling out each qubit by hand.


# %%
@qmc.qkernel
def _vec_h(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    qs[0] = qmc.h(qs[0])
    qs[1] = qmc.h(qs[1])
    return qs


@qmc.qkernel
def vec_target_demo() -> qmc.Vector[qmc.Bit]:
    qs = qmc.qubit_array(3, "qs")
    qs[0] = qmc.x(qs[0])
    cg = qmc.control(_vec_h, num_controls=1)
    qs[0], qs[1:3] = cg(qs[0], qs[1:3])
    return qmc.measure(qs)


vec_target_demo.draw()


# %% [markdown]
# ### 3.3 Default values from `@qmc.qkernel` signatures
#
# When the wrapped `@qmc.qkernel` declares a Python default for a
# classical parameter, callers may omit that keyword. The wrapper
# fills it in via `inspect.Signature.bind + apply_defaults`, so the
# default value reaches the controlled-U just like a normal direct
# call. This works identically in either mode.


# %%
@qmc.qkernel
def _phase(q: qmc.Qubit, theta: qmc.Float = math.pi / 2) -> qmc.Qubit:
    return qmc.rx(q, theta)


@qmc.qkernel
def default_arg_demo() -> qmc.Bit:
    c = qmc.qubit(name="c")
    t = qmc.qubit(name="t")
    c = qmc.x(c)
    cg = qmc.control(_phase)
    c, t = cg(c, t)  # theta defaults to math.pi / 2
    return qmc.measure(t)


default_arg_demo.draw()


# %% [markdown]
# ### 3.4 Classical keyword arguments in any order
#
# Classical kwargs at the call site are matched by name and
# reordered to follow the wrapped kernel's signature, so both
# `cg(c, t, alpha=..., beta=...)` and
# `cg(c, t, beta=..., alpha=...)` compile to the same circuit.
# Mode-independent.


# %%
@qmc.qkernel
def _two_param(q: qmc.Qubit, alpha: qmc.Float, beta: qmc.Float) -> qmc.Qubit:
    q = qmc.rx(q, alpha)
    q = qmc.rz(q, beta)
    return q


@qmc.qkernel
def kwargs_in_order() -> qmc.Bit:
    c = qmc.qubit(name="c")
    t = qmc.qubit(name="t")
    c = qmc.x(c)
    cg = qmc.control(_two_param)
    c, t = cg(c, t, alpha=0.7, beta=1.3)
    return qmc.measure(t)


@qmc.qkernel
def kwargs_reversed() -> qmc.Bit:
    c = qmc.qubit(name="c")
    t = qmc.qubit(name="t")
    c = qmc.x(c)
    cg = qmc.control(_two_param)
    c, t = cg(c, t, beta=1.3, alpha=0.7)
    return qmc.measure(t)


exe_a = transpiler.transpile(kwargs_in_order)
exe_b = transpiler.transpile(kwargs_reversed)
assert str(exe_a.compiled_quantum[0].circuit) == str(exe_b.compiled_quantum[0].circuit)

kwargs_in_order.draw()


# %% [markdown]
# ### 3.5 Controlling `U^k` with `power=`
#
# Passing `power=k` controls the *k-th power* of the wrapped
# unitary instead of `U` itself — the standard pattern in QPE,
# where the j-th register applies a controlled-`U^(2**j)`. `power`
# accepts a Python `int` (resolved at compile time) **or** a
# `qmc.UInt` handle (resolved at transpile time from `bindings`),
# and this is independent of whether `num_controls` is concrete or
# symbolic.


# %%
@qmc.qkernel
def power_demo_concrete() -> qmc.Bit:
    c = qmc.qubit(name="c")
    t = qmc.qubit(name="t")
    c = qmc.x(c)
    cg = qmc.control(qmc.rx)  # num_controls = 1 (concrete)
    c, t = cg(c, t, angle=math.pi / 4, power=3)
    return qmc.measure(t)


@qmc.qkernel
def power_demo_symbolic(k: qmc.UInt) -> qmc.Bit:
    c = qmc.qubit(name="c")
    t = qmc.qubit(name="t")
    c = qmc.x(c)
    cg = qmc.control(qmc.rx)
    c, t = cg(c, t, angle=math.pi / 4, power=k)  # power is also symbolic
    return qmc.measure(t)


power_demo_concrete.draw()
# Symbolic example needs bindings to draw / transpile:
power_demo_symbolic.draw(k=3)


# %% [markdown]
# ## 4. Concrete-mode-only patterns
#
# These call shapes require a Python-`int` `num_controls`. In
# symbolic mode the wrapper accepts only a *single* `Vector[Qubit]`
# pool as the control argument, which rules out the patterns
# below.

# %% [markdown]
# ### 4.1 Multiple separate positional control args (CCX style)
#
# With `num_controls=2`, you write two distinct positional control
# arguments (`c0`, `c1`) followed by the target. This is the
# direct CCX / Toffoli form — symbolic mode does not have an
# equivalent shape because it takes one pool, not N separate
# arguments.


# %%
@qmc.qkernel
def toffoli_demo() -> qmc.Bit:
    c0 = qmc.qubit(name="c0")
    c1 = qmc.qubit(name="c1")
    t = qmc.qubit(name="t")
    c0 = qmc.x(c0)
    c1 = qmc.x(c1)
    ccx = qmc.control(qmc.x, num_controls=2)
    c0, c1, t = ccx(c0, c1, t)
    return qmc.measure(t)


toffoli_demo.draw()


# %% [markdown]
# ### 4.2 Mixing scalar Qubit and `VectorView` controls
#
# The positional control prefix in concrete mode may freely mix
# scalar `Qubit` handles, `VectorView` slices, and whole
# `Vector[Qubit]`s, as long as the total qubit count adds up to
# `num_controls`. Here `qs[0]` (1 qubit) plus `qs[1:3]` (2 qubits)
# supplies the three controls for a `num_controls=3` controlled-H.
# Symbolic mode has no equivalent — it takes exactly one control
# argument.


# %%
@qmc.qkernel
def mixed_controls_demo() -> qmc.Vector[qmc.Bit]:
    qs = qmc.qubit_array(5, "qs")
    qs[0] = qmc.x(qs[0])
    qs[1] = qmc.x(qs[1])
    qs[2] = qmc.x(qs[2])
    cg = qmc.control(qmc.h, num_controls=3)
    qs[0], qs[1:3], qs[3] = cg(qs[0], qs[1:3], qs[3])
    return qmc.measure(qs)


mixed_controls_demo.draw()


# %% [markdown]
# ## 5. Symbolic-mode-only patterns
#
# These call shapes require `num_controls` to be a `qmc.UInt`
# handle. They share two properties: the control argument is a
# single pool (`Vector` or `VectorView`), and the number of active
# controls is determined at transpile time from `bindings`.

# %% [markdown]
# ### 5.1 `num_controls = n` over a whole pool
#
# The simplest symbolic shape: the entire pool becomes the active
# controls. `n` is a kernel parameter resolved from `bindings`.


# %%
@qmc.qkernel
def symbolic_pool(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    ctrls = qmc.qubit_array(n, "ctrls")
    tgt = qmc.qubit(name="tgt")
    ctrls[0] = qmc.x(ctrls[0])
    ctrls[1] = qmc.x(ctrls[1])
    ctrls[2] = qmc.x(ctrls[2])
    cg = qmc.control(qmc.x, num_controls=n)
    ctrls, tgt = cg(ctrls, tgt)
    return qmc.measure(ctrls)


symbolic_pool.draw(n=3)


# %% [markdown]
# ### 5.2 Canonical `n - 1` multi-controlled form
#
# A common shape: use the first `n - 1` qubits of a register as
# controls and the last one as the target. `num_controls=n - 1`
# threads the symbolic expression through; the slice `qs[0:n-1]`
# is the control pool.


# %%
@qmc.qkernel
def mcx_demo(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    qs = qmc.qubit_array(n, "qs")
    qs[0] = qmc.x(qs[0])
    qs[1] = qmc.x(qs[1])
    qs[2] = qmc.x(qs[2])
    mcx = qmc.control(qmc.x, num_controls=n - 1)
    qs[0 : n - 1], qs[n - 1] = mcx(qs[0 : n - 1], qs[n - 1])
    return qmc.measure(qs)


mcx_demo.draw(n=4)


# %% [markdown]
# ### 5.3 Selecting a subset with `controlled_indices=`
#
# Sometimes the control pool is wider than the number of active
# controls you want. `controlled_indices=` selects exactly which
# slots of the pool participate; the rest pass through untouched.
# The pool here has 4 qubits but only the first three are wired in
# as active controls.


# %%
@qmc.qkernel
def subset_pool(n: qmc.UInt, k_ctrls: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    pool = qmc.qubit_array(n, "pool")
    tgt = qmc.qubit(name="tgt")
    pool[0] = qmc.x(pool[0])
    pool[1] = qmc.x(pool[1])
    pool[2] = qmc.x(pool[2])
    cg = qmc.control(qmc.x, num_controls=k_ctrls)
    pool, tgt = cg(pool, tgt, controlled_indices=[0, 1, 2])
    return qmc.measure(pool)


subset_pool.draw(n=4, k_ctrls=3)


# %% [markdown]
# ### 5.4 `controlled_indices` with `UInt` entries
#
# Entries inside `controlled_indices` may be Python `int`
# literals, `qmc.UInt` handles, or arithmetic expressions over
# `UInt` values (`k - 1`). Literal-`int` entries are validated at
# compose time; symbolic entries are validated at transpile time
# once `bindings` make them concrete.


# %%
@qmc.qkernel
def subset_pool_with_uint(n: qmc.UInt, k_ctrls: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    pool = qmc.qubit_array(n, "pool")
    tgt = qmc.qubit(name="tgt")
    pool[0] = qmc.x(pool[0])
    pool[1] = qmc.x(pool[1])
    pool[2] = qmc.x(pool[2])
    cg = qmc.control(qmc.x, num_controls=k_ctrls)
    pool, tgt = cg(pool, tgt, controlled_indices=[0, 1, k_ctrls - 1])
    return qmc.measure(pool)


subset_pool_with_uint.draw(n=4, k_ctrls=3)


# %% [markdown]
# ## 6. Patterns that don't work
#
# Each cell below tries one rejected shape and asserts the
# expected exception. The `expect_error` helper raises an
# `AssertionError` if **no** exception is raised; any other
# unexpected exception propagates as-is and surfaces as a normal
# cell error. The mode column tells you which mode each rejection
# applies to.
#
# | Case | Mode | Exception |
# | --- | --- | --- |
# | 6.1 control qubit count mismatch | concrete | `ValueError` |
# | 6.2 `controlled_indices=` in concrete mode | concrete | `ValueError` |
# | 6.3 same qubit used twice | both | `QubitConsumedError` |
# | 6.4 symbolic-length `VectorView` in concrete | concrete | `NotImplementedError` |
# | 6.5 typo in classical kwarg | both | `TypeError` |
# | 6.6 `bool` / negative / duplicate in `controlled_indices` | symbolic | `TypeError` / `ValueError` |
# | 6.7 invalid `power` (zero or `bool`) | both | `ValueError` / `TypeError` |
# | 6.8 `num_controls=0` | both | `ValueError` |
# | 6.9 plain function with a Python default | both | `TypeError` |


# %%
def expect_error(label: str, exc_type: type, body) -> None:
    """Assert that ``body`` raises an exception of ``exc_type``.

    The helper only catches the *expected* exception class.  Any
    other exception propagates out untouched so a regression that
    swaps the exception type surfaces with a normal traceback in
    the cell.  Missing the exception entirely raises an
    ``AssertionError``.
    """
    try:
        body()
    except exc_type as exc:
        print(f"[{type(exc).__name__}] {label}: {exc}")
        return
    raise AssertionError(
        f"{label}: expected {exc_type.__name__}, but no exception was raised"
    )


# %% [markdown]
# ### 6.1 Control qubit count does not match `num_controls` (concrete)
#
# Concrete mode counts positional control qubits. Passing a slice
# that is too wide (or too narrow) for the declared `num_controls`
# is rejected at compose time as `ValueError`.


# %%
def case_count_mismatch() -> None:
    @qmc.qkernel
    def kernel() -> qmc.Bit:
        qs = qmc.qubit_array(6, "qs")
        cg = qmc.control(qmc.x, num_controls=3)
        view, t = cg(qs[0:5], qs[5])  # 5 qubits supplied, 3 expected
        qs[0:5] = view
        return qmc.measure(qs[5])

    _ = kernel.block


expect_error("control count mismatch", ValueError, case_count_mismatch)


# %% [markdown]
# ### 6.2 `controlled_indices=` in concrete mode (concrete)
#
# `controlled_indices` makes sense only when there is a control
# *pool* to select from, which is a symbolic-mode concept.
# Supplying it alongside a concrete `num_controls` raises
# `ValueError` at compose time.


# %%
def case_controlled_indices_in_concrete() -> None:
    @qmc.qkernel
    def kernel() -> qmc.Bit:
        c = qmc.qubit(name="c")
        t = qmc.qubit(name="t")
        cg = qmc.control(qmc.x)  # num_controls defaults to 1 (concrete)
        c, t = cg(c, t, controlled_indices=[0])
        return qmc.measure(t)

    _ = kernel.block


expect_error(
    "controlled_indices in concrete mode",
    ValueError,
    case_controlled_indices_in_concrete,
)


# %% [markdown]
# ### 6.3 Using the same qubit twice (both modes)
#
# Each `Qubit` handle can be consumed once. Passing the same
# scalar `Qubit` to both a control and the target — or to two
# control positions — is caught by the linear-type machinery as a
# `QubitConsumedError`. The same restriction applies in symbolic
# mode if a pool is constructed in a way that aliases a previously
# consumed qubit.


# %%
def case_alias() -> None:
    @qmc.qkernel
    def kernel() -> qmc.Bit:
        q = qmc.qubit(name="q")
        cg = qmc.control(qmc.x)
        a, b = cg(q, q)  # control and target both reference q
        return qmc.measure(b)

    _ = kernel.block


expect_error("alias (q used twice)", QubitConsumedError, case_alias)


# %% [markdown]
# ### 6.4 Symbolic-length `VectorView` in concrete mode (concrete)
#
# Concrete mode must compute the qubit count of every control
# argument at compile time. A slice whose length depends on a
# `UInt` (here `qs[0:m]` for symbolic `m`) is not yet supported in
# concrete mode and raises `NotImplementedError`. The workaround
# is to switch to symbolic mode — `num_controls=m` with `cg(qs, t)`
# accepts exactly this shape (see Section 5.1).


# %%
def case_symbolic_view_in_concrete() -> None:
    @qmc.qkernel
    def kernel(m: qmc.UInt) -> qmc.Bit:
        qs = qmc.qubit_array(m, "qs")
        cg = qmc.control(qmc.x, num_controls=3)
        view, q_out = cg(qs[0:m], qs[m - 1])
        qs[0:m] = view
        qs[m - 1] = q_out
        return qmc.measure(qs[m - 1])

    _ = kernel.block


expect_error(
    "symbolic-length VectorView in concrete mode",
    NotImplementedError,
    case_symbolic_view_in_concrete,
)


# %% [markdown]
# ### 6.5 Typo in a classical keyword argument (both modes)
#
# `qmc.control` inspects the wrapped kernel's signature, so an
# unknown keyword name is caught at compose time. The error
# message lists the parameters the wrapper actually understands.


# %%
def case_kwarg_typo() -> None:
    @qmc.qkernel
    def _gate(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
        return qmc.rx(q, angle)

    @qmc.qkernel
    def kernel() -> qmc.Bit:
        c = qmc.qubit(name="c")
        t = qmc.qubit(name="t")
        cg = qmc.control(_gate)
        c, t = cg(c, t, agnle=0.5)  # typo: agnle -> angle
        return qmc.measure(t)

    _ = kernel.block


expect_error("classical kwarg typo", TypeError, case_kwarg_typo)


# %% [markdown]
# ### 6.6 Invalid entries in `controlled_indices` (symbolic)
#
# `controlled_indices` is symbolic-mode-only (see 6.2), and its
# literal entries are validated at compose time:
#
# - `bool` values (`True` / `False`) are rejected even though
#   Python treats them as ints, to prevent the silent
#   ``True == 1`` / ``False == 0`` confusion. Use an explicit
#   `int(...)` cast if you really mean that.
# - Negative literals are rejected — pool indices are unsigned.
# - Duplicate literals are rejected because each pool slot can
#   wire in at most one active control.


# %%
def case_bool_entry() -> None:
    @qmc.qkernel
    def kernel(k_ctrls: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        pool = qmc.qubit_array(3, "pool")
        tgt = qmc.qubit(name="tgt")
        cg = qmc.control(qmc.z, num_controls=k_ctrls)
        pool, tgt = cg(pool, tgt, controlled_indices=[True, 1])
        return qmc.measure(pool)

    _ = kernel.block


def case_negative_entry() -> None:
    @qmc.qkernel
    def kernel(k_ctrls: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        pool = qmc.qubit_array(3, "pool")
        tgt = qmc.qubit(name="tgt")
        cg = qmc.control(qmc.z, num_controls=k_ctrls)
        pool, tgt = cg(pool, tgt, controlled_indices=[-1, 0, 1])
        return qmc.measure(pool)

    _ = kernel.block


def case_duplicate_entry() -> None:
    @qmc.qkernel
    def kernel(k_ctrls: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        pool = qmc.qubit_array(3, "pool")
        tgt = qmc.qubit(name="tgt")
        cg = qmc.control(qmc.z, num_controls=k_ctrls)
        pool, tgt = cg(pool, tgt, controlled_indices=[0, 0, 1])
        return qmc.measure(pool)

    _ = kernel.block


expect_error("controlled_indices: bool entry", TypeError, case_bool_entry)
expect_error("controlled_indices: negative entry", ValueError, case_negative_entry)
expect_error("controlled_indices: duplicate entry", ValueError, case_duplicate_entry)


# %% [markdown]
# ### 6.7 Invalid `power` (both modes)
#
# `power` must be a strictly positive integer (`int` or
# `qmc.UInt`). Zero and negative values raise `ValueError`. A
# Python `bool` is rejected as `TypeError` so that `power=True`
# does not silently mean `power=1`. The same restriction applies
# in both concrete and symbolic mode.


# %%
def case_power_zero() -> None:
    @qmc.qkernel
    def kernel() -> qmc.Bit:
        c = qmc.qubit(name="c")
        t = qmc.qubit(name="t")
        cg = qmc.control(qmc.x)
        c, t = cg(c, t, power=0)
        return qmc.measure(t)

    _ = kernel.block


def case_power_bool() -> None:
    @qmc.qkernel
    def kernel() -> qmc.Bit:
        c = qmc.qubit(name="c")
        t = qmc.qubit(name="t")
        cg = qmc.control(qmc.x)
        c, t = cg(c, t, power=True)
        return qmc.measure(t)

    _ = kernel.block


expect_error("power=0", ValueError, case_power_zero)
expect_error("power=True (bool)", TypeError, case_power_bool)


# %% [markdown]
# ### 6.8 `num_controls=0` (both modes)
#
# A controlled gate with zero controls would just be the
# underlying gate, which makes the wrapper meaningless.
# `qmc.control` rejects this at decoration time as `ValueError`.
# The same goes for negative `num_controls`.


# %%
def case_num_controls_zero() -> None:
    qmc.control(qmc.x, num_controls=0)


expect_error("num_controls=0", ValueError, case_num_controls_zero)


# %% [markdown]
# ### 6.9 Plain function with a Python default (both modes)
#
# When the callable passed to `qmc.control` is not a `@qmc.qkernel`
# (just a plain Python function), the wrapper auto-synthesises a
# kernel around it. The synthesiser cannot turn Python-side
# default values into IR-level defaults, so plain functions with
# defaults are rejected at decoration time. The fix is to mark the
# function as a `@qmc.qkernel` (where defaults are tracked
# end-to-end) or to drop the default and pass the value
# explicitly at the call site.


# %%
def case_plain_fn_with_default() -> None:
    def _bad_sub(q: qmc.Qubit, theta: qmc.Float = 0.5) -> qmc.Qubit:
        return qmc.rx(q, theta)

    qmc.control(_bad_sub)


expect_error("plain function with default value", TypeError, case_plain_fn_with_default)


# %% [markdown]
# ## 7. Summary
#
# `qmc.control(fn, num_controls=...)` returns a reusable
# `ControlledGate`. The right mental model is a two-axis matrix
# rather than two separate APIs:
#
# - **Mode is `num_controls`'s type.** Python `int` -> concrete,
#   `qmc.UInt` (or an expression over one) -> symbolic.
# - **Most features are mode-agnostic.** `power=`, default
#   values, classical-kwarg reordering, sub-kernels that take
#   `Vector[Qubit]`, and wrapping built-in / `@qkernel` callables
#   all behave identically in either mode (Section 3).
# - **A few features are mode-specific.** Multiple separate
#   positional control args and scalar-plus-VectorView mixing are
#   concrete-only (Section 4); the single-pool call shape,
#   `num_controls = <UInt expression>`, and
#   `controlled_indices=` are symbolic-only (Section 5).
#
# Practical decision rule: reach for *symbolic* mode whenever the
# control count is a kernel parameter or an expression over one
# (including the very common "all but one" form
# `num_controls=n - 1`). Reach for *concrete* mode when the count
# is a literal and the controls live on specific qubits you can
# name.
#
# Section 6 doubles as a regression test for the validation rules
# of both modes: every rejected shape asserts the expected
# exception type, so a future change that loses (or changes) a
# check would surface immediately in the docs build.
