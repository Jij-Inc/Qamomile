# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: qamomile
#     language: python
#     name: qamomile
# ---

# %% [markdown]
# # Parametric Circuits and Variational Quantum Algorithms
#
# In this tutorial, we will learn how to use **parametric circuits** in Qamomile
# and apply them to build a **variational quantum classifier** from scratch.
# Parametric circuits are the foundation of all variational quantum algorithms (VQAs),
# which combine quantum circuits with classical optimization.
#
# ## What We Will Learn
# - Why parametric circuits matter for variational quantum algorithms
# - The distinction between `bindings=` and `parameters=` in transpilation
# - How to use `Observable` and `expval()` for expectation values
# - Data encoding with rotation gates
# - Building a variational quantum classifier step by step
# - Running a hybrid quantum-classical optimization loop

# %%
import math

import numpy as np

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 1. Parametric Circuits in Qamomile
#
# ### Why Parametric Circuits?
#
# Many quantum algorithms follow a **variational** approach:
# prepare a parameterized quantum state, measure a cost function, and use a
# classical optimizer to adjust the parameters. This is the core idea behind
# VQAs such as **VQE**, **QAOA**, and **Quantum Machine Learning**.
#
# For these algorithms, we need circuits where parameters can be changed
# efficiently between executions **without rebuilding the entire circuit**.

# %% [markdown]
# ### `bindings=` vs. `parameters=` in Transpilation
#
# `qmc.Float` parameters can be resolved at two different times:
#
# | Mechanism | When resolved | Use case |
# |-----------|---------------|----------|
# | `bindings=` | At **transpile time** | Values the circuit structure depends on (array sizes, loop counts, Hamiltonians) |
# | `parameters=` | At **execution time** | Values that can change between runs (rotation angles for optimization) |
#
# When we list a parameter name in `parameters=`, Qamomile keeps it as a symbolic
# variable in the transpiled circuit. We can supply different values at execution
# time without re-transpiling.

# %% [markdown]
# ### Example: A Simple Parameterized Rotation
#
# Let's see this in action with a single-qubit circuit.


# %%
@qmc.qkernel
def param_rotation(theta: qmc.Float) -> qmc.Bit:
    """A single RY rotation with a tunable angle."""
    q = qmc.qubit(name="q")
    q = qmc.ry(q, theta)
    return qmc.measure(q)


param_rotation.draw()

# %%
# Transpile with theta as a free parameter (not a fixed binding)
executable_rot = transpiler.transpile(param_rotation, parameters=["theta"])

# Execute with different parameter values — no retranspilation needed
print("=== Parameterized Rotation: theta sweep ===\n")

for theta_val in [0.0, math.pi / 4, math.pi / 2, math.pi]:
    result = executable_rot.sample(
        transpiler.executor(), bindings={"theta": theta_val}, shots=1000
    ).result()

    counts = {str(v): c for v, c in result.results}
    p1 = counts.get("1", 0) / 1000
    print(f"  theta = {theta_val:.4f}  ->  P(1) = {p1:.3f}")

# %% [markdown]
# As `theta` increases from 0 to $\pi$, the probability of measuring `1`
# goes from 0 to 1, following $P(1) = \sin^2(\theta/2)$.
#
# The important point is that **we only called `transpile()` once**.
# Each call to `executable_rot.sample(...)` with different `bindings`
# simply plugs in a new value for the symbolic parameter.

# %% [markdown]
# ### When to Use Which
#
# - **`bindings=`**: values that affect circuit **structure** (qubit counts, loop counts,
#   edge lists, Hamiltonians). Changing requires a new `transpile()` call.
# - **`parameters=`**: values that are **optimized** or **swept** (rotation angles).
#   Can be changed freely at execution time.

# %% [markdown]
# ## 2. Observables and Expectation Values
#
# Variational algorithms compute **expectation values** of quantum observables.
# Qamomile supports this through the `Observable` type and `expval()` operation.
#
# ### Building Hamiltonians

# %%
import qamomile.observable as qmo

# Single Pauli operators on specific qubits
Z0 = qmo.Z(0)  # Z operator on qubit 0
Z1 = qmo.Z(1)  # Z operator on qubit 1

# Combine with arithmetic
hamiltonian_simple = Z0 + 0.5 * Z0 * Z1

print("Hamiltonian:", hamiltonian_simple)

# %% [markdown]
# ### Using `qmc.expval()` in a QKernel
#
# The `qmc.expval()` function takes a qubit array and an `Observable` parameter,
# and returns a `Float` representing $\langle \psi | H | \psi \rangle$.
#
# The `Observable` type is a special parameter type — it is always provided
# via `bindings` (because the Hamiltonian determines the measurement structure).


# %%
@qmc.qkernel
def simple_vqe(theta: qmc.Float, H: qmc.Observable) -> qmc.Float:
    """Prepare a parameterized state and compute <psi|H|psi>."""
    q = qmc.qubit_array(1, name="q")
    q[0] = qmc.ry(q[0], theta)
    return qmc.expval(q, H)


simple_vqe.draw()

# %% [markdown]
# Let's transpile and sweep theta to see how the expectation value changes.
#
# **`run()` vs `sample()`**: Because this kernel returns a `Float` via
# `qmc.expval()` (not a measurement result), we use `executable.run()`
# instead of `executable.sample()`. `run()` computes the expectation
# value and returns a `Float` directly, while `sample()` performs
# shot-based measurement and returns a `SampleResult` with bitstring
# counts. See [01_introduction](01_introduction.ipynb) for an overview.

# %%
# Build Hamiltonian: H = Z_0
H_z = qmo.Z(0)

# Transpile: H is bound (structure), theta is a free parameter
executable_vqe = transpiler.transpile(
    simple_vqe,
    bindings={"H": H_z},
    parameters=["theta"],
)

# Sweep theta and compute expectation values
thetas = np.linspace(0, 2 * np.pi, 21)
energies = []

for theta_val in thetas:
    job = executable_vqe.run(
        transpiler.executor(),
        bindings={"theta": float(theta_val)},
    )
    energy = job.result()
    energies.append(energy)

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(thetas, energies, "o-", markersize=4)
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\langle Z \rangle$")
plt.title(r"Expectation value $\langle \psi(\theta) | Z | \psi(\theta) \rangle$")
plt.axhline(y=-1, color="r", linestyle="--", alpha=0.5, label="Ground state energy")
plt.legend()
plt.grid(True)
# plt.show()

# %% [markdown]
# The expectation value $\langle Z \rangle = \cos(\theta)$ for $RY(\theta)|0\rangle$.
# The minimum $-1$ at $\theta = \pi$ corresponds to the ground state $|1\rangle$.
# This variational approach is the foundation of both quantum optimization (QAOA)
# and quantum machine learning.

# %% [markdown]
# ## 3. A Simple Variational Quantum Classifier
#
# Let's apply what we've learned to build a **variational quantum classifier** — a
# quantum circuit that learns to classify data points into two categories.
#
# The idea is straightforward:
# 1. **Encode** classical data as rotation angles on qubits
# 2. **Apply** trainable variational layers, **re-encoding** the data in each layer
# 3. **Measure** an observable — its expectation value becomes the prediction
# 4. **Optimize** the circuit parameters to minimize classification error
#
# No deep machine learning knowledge is needed — it's just function fitting
# with a quantum circuit.

# %% [markdown]
# ### The Dataset
#
# We generate a simple 2D binary classification problem: two clusters of points.

# %%
np.random.seed(42)
n_samples = 15

# Class 0: cluster centered at (-0.5, 0)
X0 = np.random.randn(n_samples, 2) * 0.3 + np.array([-0.5, 0.0])
# Class 1: cluster centered at (+0.5, 0)
X1 = np.random.randn(n_samples, 2) * 0.3 + np.array([+0.5, 0.0])

X_data = np.vstack([X0, X1]) * np.pi  # Scale for angle encoding
y_data = np.array([0] * n_samples + [1] * n_samples)

plt.figure(figsize=(6, 4))
plt.scatter(
    X_data[:n_samples, 0], X_data[:n_samples, 1], c="blue", label="Class 0", marker="o"
)
plt.scatter(
    X_data[n_samples:, 0], X_data[n_samples:, 1], c="red", label="Class 1", marker="x"
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Training Data")
plt.legend()
plt.grid(True)
# plt.show()

# %% [markdown]
# ### Data Encoding
#
# **Angle encoding** maps each feature to a rotation angle on a qubit.
# With 2 features and 2 qubits, we apply $RY(x_i)$ to qubit $i$.
#
# The `ry_layer` function from the algorithm module does exactly this — it applies
# $RY$ rotations from a parameter vector to each qubit starting at a given offset.
# By passing the data vector `x` with offset 0, `ry_layer` serves as our angle encoder.
#
# We also scale the raw features by $\pi$ so that the cluster centres at $\pm 0.5$
# map to rotation angles $\pm \pi/2$, giving near-orthogonal quantum states for the
# two classes.


# %% [markdown]
# ### The Classifier Circuit
#
# We build the classifier using `ry_layer` and `cz_entangling_layer`
# from Qamomile's algorithm module (see [05_stdlib](05_stdlib.ipynb)).
#
# A key technique is **data re-uploading**: we encode the input data in
# *every* variational layer, not just at the beginning. This interleaves
# data with trainable rotations and entanglement, giving the circuit much
# richer function-approximation power (similar to how a neural network
# applies weights at every layer).
#
# The circuit structure (repeated for each layer):
# 1. $RY$ data encoding — `ry_layer(qubits, x, 0)`
# 2. $RY$ variational rotations — `ry_layer(qubits, params, offset)`
# 3. $CZ$ entanglement — `cz_entangling_layer(qubits)`
#
# With 2 qubits and 2 layers, we have **4 trainable parameters**.

# %%
from qamomile.circuit.algorithm import cz_entangling_layer, ry_layer


@qmc.qkernel
def classifier(
    x: qmc.Vector[qmc.Float],
    params: qmc.Vector[qmc.Float],
    H: qmc.Observable,
) -> qmc.Float:
    """Variational quantum classifier with data re-uploading.

    Args:
        x: Input features (2D data point, pre-scaled)
        params: Trainable parameters (4 values for 2 layers × 2 qubits)
        H: Observable to measure (Z on qubit 0)
    """
    qubits = qmc.qubit_array(2, name="q")
    n = qubits.shape[0]

    for layer in qmc.range(2):
        # Data encoding (re-uploaded each layer)
        qubits = ry_layer(qubits, x, 0)
        # Variational rotations + entanglement
        qubits = ry_layer(qubits, params, layer * n)
        qubits = cz_entangling_layer(qubits)

    return qmc.expval(qubits, H)


classifier.draw(fold_loops=False, inline=True)

# %% [markdown]
# ## 4. Transpilation: Bindings vs. Parameters in Action
#
# This is a great example of the `bindings=` vs. `parameters=` distinction:
#
# - **`H` (Observable)** is a `binding` — it determines the measurement structure
# - **`x` (data) and `params` (trainable weights)** are `parameters` — they change
#   per execution without retranspiling
#
# We transpile once and can evaluate any data point with any parameter values.

# %%
# Build the measurement observable: Z on qubit 0.
# The num_qubits parameter ensures the observable matches the 2-qubit circuit.
H_label = qmo.Hamiltonian(num_qubits=2)
H_label += qmo.Z(0)

executable = transpiler.transpile(
    classifier,
    bindings={"H": H_label},
    parameters=["x", "params"],
)

# Quick test: evaluate with a sample data point
test_expval = executable.run(
    transpiler.executor(),
    bindings={"x": [0.5, -0.3], "params": [0.1, 0.2, 0.3, 0.4]},
).result()
print(f"Test expectation value: {test_expval:.4f}")

# %% [markdown]
# ## 5. Training the Classifier
#
# The training loop is the same hybrid quantum-classical pattern used in all
# variational algorithms: the quantum circuit computes predictions, and a
# classical optimizer adjusts the parameters to minimize the loss.
#
# ### Label Convention
#
# We map class labels to $Z$ eigenvalues:
# - Class 0 → target $\langle Z \rangle = +1$
# - Class 1 → target $\langle Z \rangle = -1$
#
# The loss is the mean squared error between the predicted and target values.

# %%
loss_history = []


def classification_loss(params_flat, executable, transpiler, X_train, y_train):
    """Compute MSE between predicted <Z> and target labels."""
    total_loss = 0.0
    for xi, yi in zip(X_train, y_train):
        target = 1.0 - 2.0 * yi  # 0 → +1, 1 → −1
        pred = executable.run(
            transpiler.executor(),
            bindings={"x": xi.tolist(), "params": params_flat.tolist()},
        ).result()
        total_loss += (pred - target) ** 2
    mse = total_loss / len(X_train)
    loss_history.append(mse)
    return mse


# %% [markdown]
# ### Running the Optimization
#
# We use `scipy.optimize.minimize` with the COBYLA method — a
# gradient-free optimizer well-suited for noisy quantum objective functions.

# %%
from scipy.optimize import minimize

# Initial random parameters (4 values for 2 layers × 2 qubits)
np.random.seed(42)
n_params = 4
init_params = np.random.uniform(-np.pi, np.pi, size=n_params)

print(f"Initial parameters: {init_params}")

# Clear history
loss_history = []

# Run COBYLA optimization
result_opt = minimize(
    classification_loss,
    init_params,
    args=(executable, transpiler, X_data, y_data),
    method="COBYLA",
    options={"maxiter": 80, "disp": True},
)

print(f"\nOptimized parameters: {result_opt.x}")
print(f"Final loss: {result_opt.fun:.4f}")

# %% [markdown]
# ### Visualizing Convergence

# %%
plt.figure(figsize=(8, 4))
plt.plot(loss_history, marker="o", markersize=3)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Classifier Training Convergence")
plt.grid(True)
# plt.show()

# %% [markdown]
# ## 6. Evaluating the Classifier
#
# Let's check how well the trained classifier performs on our dataset.
# We predict class 0 if $\langle Z \rangle > 0$, and class 1 otherwise.

# %%
optimal_params = result_opt.x

# Predict on training data
predictions = []
for xi in X_data:
    pred = executable.run(
        transpiler.executor(),
        bindings={"x": xi.tolist(), "params": optimal_params.tolist()},
    ).result()
    predictions.append(pred)

predictions = np.array(predictions)
predicted_labels = (predictions < 0).astype(int)  # <Z> < 0 → class 1

accuracy = np.mean(predicted_labels == y_data)
print(f"Classification accuracy: {accuracy:.1%}")
print("\nPer-sample predictions:")
for i, (xi, yi, pred, label) in enumerate(
    zip(X_data, y_data, predictions, predicted_labels)
):
    status = "correct" if label == yi else "WRONG"
    print(f"  [{i:2d}] true={yi}, pred_label={label}, <Z>={pred:+.3f}  {status}")

# %% [markdown]
# ### Visualizing the Decision Boundary
#
# We evaluate $\langle Z \rangle$ on a grid of points to see the decision boundary.

# %%
# Create a grid (matching the π-scaled feature range)
grid_range = np.linspace(-3.0, 3.0, 25)
xx, yy = np.meshgrid(grid_range, grid_range)
grid_points = np.column_stack([xx.ravel(), yy.ravel()])

# Evaluate classifier on grid
grid_predictions = []
for point in grid_points:
    pred = executable.run(
        transpiler.executor(),
        bindings={"x": point.tolist(), "params": optimal_params.tolist()},
    ).result()
    grid_predictions.append(pred)

grid_predictions = np.array(grid_predictions).reshape(xx.shape)

# Plot
plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, grid_predictions, levels=20, cmap="RdBu", alpha=0.7)
plt.colorbar(label=r"$\langle Z \rangle$")
plt.contour(xx, yy, grid_predictions, levels=[0.0], colors="black", linewidths=2)
plt.scatter(
    X_data[:n_samples, 0],
    X_data[:n_samples, 1],
    c="blue",
    edgecolors="k",
    label="Class 0",
    marker="o",
)
plt.scatter(
    X_data[n_samples:, 0], X_data[n_samples:, 1], c="red", label="Class 1", marker="x"
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Quantum Classifier Decision Boundary")
plt.legend()
# plt.show()

# %% [markdown]
# The black line shows the decision boundary ($\langle Z \rangle = 0$).
# Blue regions predict class 0, red regions predict class 1.

# %% [markdown]
# ## 7. Summary
#
# In this tutorial, we learned the key concepts behind parametric quantum circuits
# and applied them to build a variational quantum classifier.
#
# ### Key Takeaways
#
# 1. **Parametric circuits** are the foundation of variational quantum algorithms.
#    Qamomile lets us declare free parameters with `parameters=` so they can be
#    changed at execution time without retranspiling.
#
# 2. **`bindings=` vs. `parameters=`**:
#    - `bindings`: Fixed at transpile time (problem structure, Hamiltonians)
#    - `parameters`: Free until execution time (rotation angles, input data)
#
# 3. **Observable and `expval()`**: Qamomile supports computing expectation values
#    $\langle \psi | H | \psi \rangle$ directly in qkernels using the `Observable` type
#    and `qmc.expval()`.
#
# 4. **Variational quantum classifier**:
#    - Data encoding: scale features and map them to qubit rotations
#    - Data re-uploading: encode data in every layer for richer expressiveness
#    - Variational ansatz: trainable rotation + entanglement layers
#    - Prediction: expectation value of a Pauli observable
#    - Training: classical optimizer minimizes a loss function
#
# 5. **Hybrid optimization loop**: The same quantum-classical pattern applies to
#    both quantum optimization (QAOA) and quantum machine learning.
#
# ### Next Steps
#
# - [QAOA](../optimization/qaoa.ipynb): See how Qamomile's built-in converters handle
#   combinatorial optimization problems
# - [Resource Estimation](09_resource_estimation.ipynb): Estimate circuit depth and
#   gate counts for our circuits
# - [Custom Executor](11_custom_executor.ipynb): Run circuits on cloud quantum hardware

# %% [markdown]
# ## What We Learned
#
# - **Why parametric circuits matter for variational quantum algorithms** — They allow the same circuit structure to be reused with different parameter values, enabling classical-quantum optimization loops.
# - **The distinction between `bindings=` and `parameters=` in transpilation** — `bindings` fixes values at transpile time (problem structure); `parameters` keeps them free until execution (trainable angles).
# - **How to use `Observable` and `expval()` for expectation values** — `qmc.expval(qubits, H)` computes $\langle\psi|H|\psi\rangle$ directly, returning a `Float` via `run()`.
# - **Data encoding with rotation gates** — Classical features are scaled and mapped to qubit rotations via `ry_layer`. Re-uploading data in every variational layer (data re-uploading) greatly enhances the circuit's expressiveness.
# - **Building a variational quantum classifier step by step** — Combines data encoding layers, trainable rotation + entanglement ansatz layers, and a Pauli-Z observable for binary classification.
# - **Running a hybrid quantum-classical optimization loop** — A classical optimizer (e.g. scipy) updates circuit parameters to minimize a loss function computed from quantum expectation values.
