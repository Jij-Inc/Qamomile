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
# # Controlled Operations with `qm.control`
#
# `qm.control` applies a quantum kernel to target qubits conditionally on the
# state of one or more control qubits. By default, the kernel's operation is
# applied when all controls are in the $|1\rangle$ state. The controlled kernel
# must implement a unitary operation.
#
# The control qubits are not measured, so the operation acts on a superposition
# as a whole. For a single control qubit,
#
# $$
# (\alpha|0\rangle+\beta|1\rangle)|t\rangle
# \longrightarrow
# \alpha|0\rangle|t\rangle+\beta|1\rangle U|t\rangle,
# $$
#
# where $|t\rangle$ is the target state and $U$ is the operation defined by the
# kernel. The target is unchanged in the control-$0$ component and receives $U$
# in the control-$1$ component. These components remain part of one coherent
# joint quantum state, which may be entangled.
#
# This chapter starts with a controlled RX rotation, then extends the same
# approach to multiple controls and kernels that act on several target qubits.
# We will also see how to choose the control state that activates an operation,
# select control qubits from a larger register, and use a `Const` parameter to
# set the number of controls.

# %%
# Install the latest Qamomile through pip.
# # !pip install qamomile

# %%
from __future__ import annotations

from math import pi

import qamomile as qm

# %% [markdown]
# ## 1. A First Example: Controlled RX
#
# In Qamomile, `qm.rx(state, theta)` applies an RX gate to a single qubit, with
# `theta` specified in radians. In the computational basis
# $\{|0\rangle, |1\rangle\}$, the gate is represented by
#
# $$
# R_X(\theta)=
# \begin{pmatrix}
# \cos(\theta/2) & -i\sin(\theta/2)\\
# -i\sin(\theta/2) & \cos(\theta/2)
# \end{pmatrix}.
# $$
#
# In this example, the target starts in $|0\rangle$, and we choose $\theta=\pi$.
# The rotation therefore gives
#
# $$
# R_X(\pi)|0\rangle
# =
# \begin{pmatrix}
# 0 & -i\\
# -i & 0
# \end{pmatrix}
# \begin{pmatrix}1\\0\end{pmatrix}
# =
# \begin{pmatrix}0\\-i\end{pmatrix}
# =
# -i|1\rangle.
# $$
#
# Measuring this state gives 1 with certainty. The factor $-i$ is a phase factor
# and does not change that measurement probability.
#
# We define this rotation in a quantum kernel named `rotate`, then pass it to
# `qm.control` to make the operation conditional on a control qubit.
#
# The two examples below differ only in how they prepare the control.
# `qm.prepare(False)` prepares $|0\rangle$, while `qm.prepare(True)` prepares
# $|1\rangle$. Both examples initialize the target in $|0\rangle$ and use the
# same rotation angle, $\pi$.


# %%
@qm.qkernel
def rotate(state: qm.Q[qm.Bits[1]], theta: qm.Real) -> qm.Q[qm.Bits[1]]:
    return qm.rx(state, theta)


@qm.qkernel
def crx_control_off() -> qm.Shot[qm.Bits[2]]:
    control = qm.prepare(False)
    target = qm.qubits(1)
    control, target = qm.control(
        rotate,
        control,
        target,
        pi,
    )
    return qm.measure(qm.concat(control, target))


@qm.qkernel
def crx_control_on() -> qm.Shot[qm.Bits[2]]:
    control = qm.prepare(True)
    target = qm.qubits(1)
    control, target = qm.control(
        rotate,
        control,
        target,
        pi,
    )
    return qm.measure(qm.concat(control, target))


# %%
assert qm.evaluate(qm.specialize(crx_control_off)) == (0, 0)
assert qm.evaluate(qm.specialize(crx_control_on)) == (1, 1)

# %% [markdown]
# The rotation is applied only in `crx_control_on`, where the control is
# $|1\rangle$, so measuring the target gives 1. In `crx_control_off`, the
# rotation is not applied, and the target remains in $|0\rangle$.
#
# The assertions use `qm.evaluate(qm.specialize(...))` to execute one shot and
# check the results. Since `qm.concat(control, target)` places the control
# before the target, the returned tuples are `(0, 0)` and `(1, 1)`, respectively.
#
# > **Note:** Qamomile includes a built-in simulator. `qm.evaluate(...)` uses
# > it to execute one shot, so no engine needs to be specified. To use the
# > same simulator with `qm.run(...)`, specify `on=qm.Reference()`.
#
# **Key things to notice:**
#
# - Pass the quantum kernel first, followed by the control, the target, and any
#   additional arguments to the kernel—in this case, the rotation angle.
# - `qm.control` returns both quantum values after the operation. Receive them
#   as `control, target` and use these returned values in subsequent operations.

# %% [markdown]
# ## 2. Controlling a Quantum Kernel
#
# In the previous example, `rotate` contained a single RX gate. A quantum
# kernel can also contain a sequence of gates acting on several qubits.
# Passing that kernel to `qm.control` makes the entire operation conditional
# on the control state.
#
# Here, `x_all_two` applies an X gate to each of two qubits. An X gate
# exchanges $|0\rangle$ and $|1\rangle$, so this kernel transforms
# $|00\rangle$ into $|11\rangle$.


# %%
@qm.qkernel
def x_all_two(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    for index in range(2):
        state = qm.x(state, target=index)
    return state


@qm.qkernel
def vector_target_demo() -> qm.Shot[qm.Bits[3]]:
    control = qm.prepare(True)
    target = qm.qubits(2)
    control, target = qm.control(x_all_two, control, target)
    return qm.measure(qm.concat(control, target))


# %%
assert qm.evaluate(qm.specialize(vector_target_demo)) == (1, 1, 1)

# %% [markdown]
# The control starts in $|1\rangle$, and the two target qubits start in
# $|00\rangle$. The controlled call applies both X gates, changing the target
# to $|11\rangle$. The measured result is therefore `(1, 1, 1)`, with the
# control bit first. If the control were prepared in $|0\rangle$, neither
# X gate would be applied, and the result would be `(0, 0, 0)`.
#
# The definition of `x_all_two` does not need a control argument. The caller
# supplies the control separately through `qm.control`, so the same kernel
# can be reused with or without control.
#
# A kernel passed to `qm.control` must take one quantum state as its first
# argument and return an updated quantum state with a matching type. Here,
# both are `qm.Q[qm.Bits[2]]`. It may also accept classical arguments, such as
# `theta` in the earlier `rotate` example.
#
# The controlled kernel must implement a unitary operation. Operations such as
# measurement and reset cannot be included in its body. In this example,
# measurement takes place in `vector_target_demo`, after the controlled call.
#
# > **Note:** Classical arguments can also be passed by name, such as `theta`
# > in a call to `rotate`. If the kernel defines a default value for a
# > classical parameter, that argument can be omitted.

# %% [markdown]
# ## 3. Using Multiple Controls and Choosing the Control State
#
# To use several control qubits, pass their quantum state as the control
# argument to `qm.control`. By default, the kernel's operation is applied
# only when all control qubits are in $|1\rangle$.
#
# The following example uses two control qubits with the `rotate` kernel
# defined earlier.


# %%
@qm.qkernel
def multi_control_rx() -> qm.Shot[qm.Bits[3]]:
    controls = qm.prepare((1, 1))
    target = qm.qubits(1)
    controls, target = qm.control(
        rotate,
        controls,
        target,
        pi,
    )
    return qm.measure(qm.concat(controls, target))


# %%
assert qm.evaluate(qm.specialize(multi_control_rx)) == (1, 1, 1)

# %% [markdown]
# Both controls are prepared in $|1\rangle$, so the RX rotation is applied.
# As in the first example, $R_X(\pi)$ transforms the target from $|0\rangle$
# to $-i|1\rangle$, and measuring it gives 1. If either control were prepared
# in $|0\rangle$, the rotation would not be applied.
#
# The same call form works with more control qubits.

# %% [markdown]
# ### Choosing a different control state
#
# Use `control_state` to specify a different combination of control-qubit
# values that activates the operation. This argument selects the condition;
# it does not prepare or measure the controls.
#
# Qamomile interprets the control qubit at index 0 as the **least significant
# bit**. For two controls with values `(b0, b1)`, the corresponding integer is
# `b0 + 2*b1`. Thus, `control_state=1` activates the operation for `(1, 0)`:
# the first control is 1 and the second is 0.
#
# The example below applies X gates to two separate targets. The first call
# uses `control_state=1`; the second uses the default condition that both
# controls must be 1.


# %%
@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)


@qm.qkernel
def activation_patterns() -> qm.Shot[qm.Bits[4]]:
    controls = qm.prepare((1, 0))
    first = qm.qubits(1)
    second = qm.qubits(1)

    controls, first = qm.control(flip, controls, first, control_state=1)
    controls, second = qm.control(flip, controls, second)

    return qm.measure(qm.concat(controls, first, second))


# %%
assert qm.evaluate(qm.specialize(activation_patterns)) == (1, 0, 1, 0)

# %% [markdown]
# The controls are prepared as `(1, 0)`, so the first target flips to
# $|1\rangle$. The second target remains in $|0\rangle$ because the default
# condition requires `(1, 1)`. The returned tuple contains the two control
# bits, followed by the first and second target bits.
#
# > **Note:** `control_state` must be a non-negative integer literal that
# > fits the number of control qubits. For two controls, the valid values
# > are `0`, `1`, `2`, and `3`.

# %% [markdown]
# ## 4. Selecting Control Qubits from a Register
#
# Sometimes only some of the qubits in a register should act as controls.
# Use `at` to select one qubit by index, or `qm.split_at` and `qm.concat`
# to group several control qubits.

# %% [markdown]
# ### Selecting one control qubit
#
# The `at` argument selects which qubit in the control register determines
# whether the operation is applied. Indices start at zero.


# %%
@qm.qkernel
def selected_control() -> qm.Shot[qm.Bits[3]]:
    controls = qm.prepare((1, 0))
    target = qm.qubits(1)
    controls, target = qm.control(flip, controls, target, at=0)
    return qm.measure(qm.concat(controls, target))


# %%
assert qm.evaluate(qm.specialize(selected_control)) == (1, 0, 1)

# %% [markdown]
# Here, `at=0` selects the first control qubit, which is prepared in
# $|1\rangle$, so the target flips. The second control qubit is not part of
# the activation condition. Changing the argument to `at=1` would select
# the second qubit, which is in $|0\rangle$, and the result would be
# `(1, 0, 0)`.
#
# `qm.control` returns the entire control register together with the target,
# even when `at` selects only one qubit. In this example, the returned values
# are assigned back to `controls` and `target`, then combined and measured.
#
# When `at` is supplied, `control_state` refers only to the selected qubit.
# Its valid values are therefore `0` and `1`.

# %% [markdown]
# ### Selecting several control qubits
#
# For several controls, separate the required qubits with `qm.split_at` and
# combine them with `qm.concat`.
#
# `qm.split_at(state, k)` separates the first `k` qubits from the remaining
# qubits. `qm.concat` combines its arguments in the order supplied. These
# operations change how the qubits are grouped and indexed; they add no
# quantum gates and preserve any existing entanglement.
#
# The following example uses original indices 0, 1, and 3 as controls, with
# index 2 as the target.


# %%
@qm.qkernel
def subset_control() -> qm.Shot[qm.Bits[4]]:
    state = qm.prepare((1, 1, 0, 1))
    prefix, remainder = qm.split_at(state, 2)  # Original [0:2], [2:4]
    target, last_control = qm.split_at(remainder, 1)  # Original [2:3], [3:4]

    controls = qm.concat(prefix, last_control)  # Original indices 0, 1, 3
    controls, target = qm.control(flip, controls, target)

    prefix, last_control = qm.split_at(controls, 2)  # Original [0:2], [3:4]
    state = qm.concat(prefix, target, last_control)  # Original order: 0, 1, 2, 3
    return qm.measure(state)


# %%
assert qm.evaluate(qm.specialize(subset_control)) == (1, 1, 1, 1)

# %% [markdown]
# All three selected controls are in $|1\rangle$, so the target at original
# index 2 flips from $|0\rangle$ to $|1\rangle$. The final concatenation
# restores the original index order before measurement.
#
# The control and target qubits must be distinct. Splitting the register lets
# us pass them separately. As with gate operations, splitting and combining
# consume their input values: continue with the returned values rather than
# reusing the inputs.
#
# > **Note:** These splitting and combining operations currently require
# > fixed register widths. Split positions must also be concrete integers;
# > symbolic widths and split positions expressed as `Const` arguments are
# > not supported here.

# %% [markdown]
# ### Choosing an index for each run
#
# To choose a single control qubit for each run, add a kernel parameter such
# as `index: qm.Param[qm.Int]` and pass it as `at=index`. Supply its value
# through the `parameters` argument to `qm.run`.
#
# This requires engine support for selecting a qubit at run time.
# `qm.Reference()` supports this use of `at`.

# %% [markdown]
# ## 5. Parameterizing Controlled Operations
#
# Controlled operations can use the `Const` and `Param` annotations
# introduced in Tutorial 02. The following examples parameterize the number
# of control qubits, the number of repetitions, and the rotation angle.

# %% [markdown]
# ### Parameterizing the number of control qubits
#
# Declare the number of control qubits as a `Const` parameter and supply its
# value when calling `qm.specialize`. The call to `qm.control` uses all
# qubits in the resulting control register.
#
# The loop applies an X gate to each qubit in `controls`, preparing all
# controls in $|1\rangle$. Here, `target=index` selects the qubit within
# `controls` that receives the X gate. The separate target register for the
# controlled operation is created afterward.


# %%
@qm.qkernel
def variable_controls(
    width: qm.Const[qm.Int],
) -> tuple[qm.Shot[qm.Bits[width]], qm.Shot[qm.Bits[1]]]:  # noqa: F821
    controls = qm.qubits(width)
    for index in range(width):
        controls = qm.x(controls, target=index)
    target = qm.qubits(1)
    controls, target = qm.control(flip, controls, target)
    return qm.measure(controls), qm.measure(target)


# %%
variable_program = qm.specialize(variable_controls, width=3)
assert qm.evaluate(variable_program) == ((1, 1, 1), (1,))

# %% [markdown]
# With `width=3`, the kernel creates three control qubits and one target
# qubit. The loop prepares all controls in $|1\rangle$, so the controlled
# X gate flips the target to $|1\rangle$.
#
# The two registers are measured separately. The result therefore contains
# two tuples: the three control bits and the single target bit.
#
# The same kernel can be specialized with a different positive value of
# `width`. Omitting `control_state` continues to mean that all control
# qubits must be in $|1\rangle$, regardless of their number.

# %% [markdown]
# ### Repeating a controlled operation
#
# A loop can apply the same controlled operation several times. In this
# example, `repetitions` is declared as `Const`, fixing the number of
# iterations during specialization. The rotation angle `theta` is declared
# as `Param`, so its value can be supplied separately for each run.


# %%
@qm.qkernel
def repeated_control(
    repetitions: qm.Const[qm.Int],
    theta: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[2]]:
    state = qm.prepare((1, 0))
    control, target = qm.split_at(state, 1)
    for _ in range(repetitions):
        control, target = qm.control(rotate, control, target, theta)
    return qm.measure(qm.concat(control, target))


# %% [markdown]
# For an operation $U$, repeating its controlled form $k$ times implements
# a controlled $U^k$. Here, the loop repeats the RX rotation under the same
# control.

# %%
program = qm.specialize(repeated_control, repetitions=3)
artifact = qm.compile(program, to=qm.Reference())

record = qm.run(
    artifact,
    on=qm.Reference(),
    parameters={"theta": pi},
    shots=8,
    seed=7,
)

print("controlled rotations:", record.result.counts)
assert record.result.counts == {(1, 1): 8}

# %% [markdown]
# The control is prepared in $|1\rangle$, so all three rotations are applied.
# With $\theta=\pi$, the target is measured as 1, giving `(1, 1)` in every
# shot.
#
# The same `artifact` can be reused with another value of `theta`. For
# example, setting `theta` to `0.0` leaves the target in $|0\rangle$, giving
# `(1, 0)`. Because `repetitions` is declared as `Const` in this example,
# changing its value requires a new specialization and compilation.

# %% [markdown]
# ## 6. Phase in Controlled Operations (Optional)
#
# `qm.global_phase(state, angle)` multiplies the entire state by
# $e^{i\,\mathrm{angle}}$, with the angle specified in radians. On its own,
# this **global phase** does not change measurement probabilities.
#
# When the operation is applied through `qm.control`, however, the phase
# factor applies only to the component that satisfies the control condition.
# It can therefore become a **relative phase** between components of the
# joint state and affect later measurement results.
#
# The following example uses an angle of $\pi$, giving a phase factor of
# $-1$. Hadamard gates before and after the controlled operation make this
# phase difference observable.


# %%
@qm.qkernel
def scalar_phase(state: qm.Q[qm.Bits[1]], angle: qm.Real) -> qm.Q[qm.Bits[1]]:
    return qm.global_phase(state, angle)


@qm.qkernel
def phase_kickback() -> qm.Shot[qm.Bits[2]]:
    control = qm.prepare(False)
    control = qm.h(control)
    target = qm.qubits(1)
    control, target = qm.control(
        scalar_phase,
        control,
        target,
        pi,
    )
    control = qm.h(control)
    return qm.measure(qm.concat(control, target))


# %%
assert qm.evaluate(qm.specialize(phase_kickback)) == (1, 0)

# %% [markdown]
# The first Hadamard gate prepares the control in
# $(|0\rangle+|1\rangle)/\sqrt{2}$. The controlled operation then changes
# the joint state as follows:
#
# $$
# \frac{|0\rangle+|1\rangle}{\sqrt{2}}|0\rangle
# \;\longrightarrow\;
# \frac{|0\rangle-|1\rangle}{\sqrt{2}}|0\rangle.
# $$
#
# At this angle, `scalar_phase` applies $-I$. With one control qubit,
# its controlled form is
# $C(-I)=Z_{\mathrm{control}}\otimes I_{\mathrm{target}}$: it acts as a
# Z gate on the control and leaves the target unchanged. The phase factor
# becomes a relative phase of the control qubit. This is **phase kickback**.
#
# The second Hadamard gate maps $(|0\rangle-|1\rangle)/\sqrt{2}$ to
# $|1\rangle$, while the target remains in $|0\rangle$. The measured result
# is therefore `(1, 0)`.
#
# With an angle of `0.0`, there would be no phase difference, and the second
# Hadamard would return the control to $|0\rangle$, giving `(0, 0)`.
#
# The same principle explains why the factor $-i$ in $R_X(\pi)=-iX$
# cannot generally be ignored when constructing a controlled operation.

# %% [markdown]
# ## 7. Summary
#
# `qm.control` makes a unitary quantum kernel conditional on control qubits
# without measuring them. The kernel can contain a single gate or a sequence
# of operations acting on several target qubits.
#
# - **Choose the activation condition.** By default, the operation is applied
#   when all controls are in $|1\rangle$. Use `control_state` to specify a
#   different computational-basis pattern.
# - **Select the control qubits.** Use `at` to select one qubit from a control
#   register. For fixed register widths, use `qm.split_at` and `qm.concat` to
#   group several controls.
# - **Keep controls and targets separate.** They must refer to distinct
#   qubits. `qm.control` returns both the control register and the updated
#   target for subsequent operations.
# - **Parameterize the operation.** In this chapter's examples, `Const` fixes
#   the number of controls or repetitions during specialization, while `Param`
#   allows the rotation angle to vary between runs.
# - **Repeat controlled operations with a loop.** Applying a controlled $U$
#   operation $k$ times implements a controlled $U^k$.
# - **Account for phase.** A global phase of the original operation can become
#   a relative phase under control and affect later measurement results, as
#   illustrated by phase kickback.
#
# **Next:** [Resource Estimation](04_resource_estimation.ipynb) explains how to
# estimate logical resources, such as qubit and gate counts, without executing
# a quantum kernel.
