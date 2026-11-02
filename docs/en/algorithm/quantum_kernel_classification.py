# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# tags: [algorithm, machine-learning]
# ---
# # Quantum Kernel Classification
#
# This tutorial demonstrates **quantum kernel methods** for binary
# classification using Qamomile. The idea is to use a parameterized quantum
# circuit to compute a kernel (similarity) matrix, which is then fed to a
# classical Support Vector Machine (SVM).
#
# We will:
#
# 1. Prepare a toy dataset (`make_circles`) that is not linearly separable.
# 2. Build a **feature map** circuit $U(x)$ as a `@qkernel`.
# 3. Construct the **overlap kernel** $k(x, x') = |\langle 0^n | U(x')^\dagger U(x) | 0^n \rangle|^2$.
# 4. Compute the Gram matrix and train an SVM with `kernel="precomputed"`.
# 5. Compare against classical linear and RBF kernels.

# %% [markdown]
# ## Installation
#
# ```bash
# pip install qamomile scikit-learn
# ```

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile scikit-learn

# %%
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import qamomile as qm

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ## Background: Quantum Kernel Methods
#
# A quantum kernel estimates the similarity between two data points by
# measuring the overlap of their quantum feature states:
#
# $$
# k(x, x') = |\langle 0^n | U(x')^\dagger \, U(x) | 0^n \rangle|^2
# $$
#
# This overlap is estimated by running $U(x')^\dagger U(x)$ on $|0^n\rangle$
# and measuring the probability of obtaining the all-zero bitstring.
# The resulting kernel matrix can be plugged into any kernel-based
# classifier such as an SVM.


# %% [markdown]
# ## Hyperparameters


# %% [markdown]
# ## Dataset
#
# We use scikit-learn's `make_circles` to generate a 2D dataset with two
# concentric circles — a classic example where a linear classifier fails.


# %% [markdown]
# ## Classical Preprocessing (Feature Lifting)
#
# Before encoding data into the quantum circuit, we apply a classical **feature lifting** step that maps the original 2D input into a higher-dimensional space. By adding nonlinear combinations of the raw features, we give the quantum feature map richer structure to work with.
#
# The two-stage pipeline is:
#
# 1. Scale the raw 2D coordinates into $[0, \pi]$.
# 2. Construct nonlinear features $(x_0 - x_1)^2$ and $(x_0 + x_1)^2$
#    and scale those into $[0, \pi]$ as well.
#
# This yields a 4-dimensional feature vector per sample:
# $f(x) = [x_0, x_1, (x_0 - x_1)^2, (x_0 + x_1)^2]$.
#
# The quantum kernel is therefore defined on the preprocessed features:
#
# $$
# k(x, x') = |\langle 0^n | U(f(x'))^\dagger \, U(f(x)) | 0^n \rangle|^2
# $$


# %% [markdown]
# ## Feature Map Circuit
#
# The feature block $U(x)$ encodes data via single-qubit rotations and
# entangling gates. `qm.adjoint` derives its exact inverse $U(x)^\dagger$
# from the same feature-map body.


# %% [markdown]
# ## Overlap Kernel Circuit
#
# The full kernel circuit applies $U(x)$ followed by $U(x')^\dagger$.
# The kernel value $k(x, x')$ is the probability of measuring $|00\rangle$.
#
# We repeat the feature block `layers` times for increased expressiveness.
# Since `layers` is a structure parameter, it is bound during specialization.


# %%
@qm.qkernel
def classification_feature_map(
    state: qm.Q[qm.Bits[2]],  # type: ignore[valid-type]
    features: qm.Vector[qm.Real, 4],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[2]]:  # type: ignore[valid-type]
    """Encode one feature vector into an owned two-qubit state.

    Args:
        state (qm.Q[qm.Bits[2]]): Owned two-qubit input state.
        features (qm.Vector[qm.Real, 4]): Two scaled coordinates and two
            nonlinear lifted coordinates.

    Returns:
        qm.Q[qm.Bits[2]]: Encoded two-qubit state.
    """

    for _ in range(2):
        for index in range(2):
            state = qm.h(state, target=index)
            state = qm.rz(state, features[index], target=index)
            state = qm.ry(state, features[index], target=index)
        state = qm.cz(state, control=0, target=1)
        state = qm.rzz(state, features[2], left=0, right=1)
        state = qm.rz(state, features[3], target=0)
        state = qm.rz(state, features[3], target=1)
    return state


@qm.qkernel
def classification_overlap_state(
    left: qm.Vector[qm.Real, 4],  # type: ignore[valid-type]
    right: qm.Vector[qm.Real, 4],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[2]]:  # type: ignore[valid-type]
    """Build the reversible overlap witness for two feature vectors.

    Args:
        left (qm.Vector[qm.Real, 2]): First feature vector.
        right (qm.Vector[qm.Real, 2]): Second feature vector.

    Returns:
        qm.Q[qm.Bits[2]]: State whose zero probability is the kernel value.
    """

    state = classification_feature_map(qm.qubits(2), left)
    return qm.adjoint(classification_feature_map, state, right)


@qm.qkernel
def classification_overlap(
    left: qm.Param[qm.Vector[qm.Real, 4]],  # type: ignore[valid-type]
    right: qm.Param[qm.Vector[qm.Real, 4]],  # type: ignore[valid-type]
) -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Measure one runtime-rebindable overlap witness.

    Args:
        left (qm.Param[qm.Vector[qm.Real, 2]]): First runtime feature vector.
        right (qm.Param[qm.Vector[qm.Real, 2]]): Second runtime feature vector.

    Returns:
        qm.Shot[qm.Bits[2]]: Two logical overlap-witness bits per shot.
    """

    return qm.measure(classification_overlap_state(left, right))


# %% [markdown]
# `qm.adjoint` derives the inverse from the feature-map body. There is no copied
# reverse gate list to keep synchronized with the forward map, and linear state
# ownership is checked across both calls.

# %% [markdown]
# ## Inspect and estimate the authored body
#
# Specialization produces the target-independent program used by both
# inspection and resource analysis. The body contains an adjoint call but no
# quantum-classification opcode. Rust derives forty logical gates from the
# visible two-layer forward and inverse feature maps.

# %%
program = qm.specialize(classification_overlap)
inspection = program.inspect()
study = qm.resources.logical(program)

assert "qcore.adjoint_call" in inspection
assert "quantum.classification" not in inspection
assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (40, 40)
assert qm.resources.ResourceStudy.from_bytes(study.to_bytes()) == study

# %% [markdown]
# ## Dataset
#
# We use scikit-learn's `make_circles` to generate a 2D dataset with two
# concentric circles — a classic example where a linear classifier fails.

# %%
N_SAMPLES = 8 if docs_test_mode else 40
X_raw, labels = make_circles(
    n_samples=N_SAMPLES,
    noise=0.04,
    factor=0.4,
    random_state=7,
)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw,
    labels,
    test_size=0.25,
    random_state=7,
    stratify=labels,
)

# %% [markdown]
# ## Classical Preprocessing (Feature Lifting)
#
# Before encoding data into the quantum circuit, we apply a classical **feature lifting** step that maps the original 2D input into a higher-dimensional space. By adding nonlinear combinations of the raw features, we give the quantum feature map richer structure to work with.
#
# The two-stage pipeline is:
#
# 1. Scale the raw 2D coordinates into $[0, \pi]$.
# 2. Construct nonlinear features $(x_0 - x_1)^2$ and $(x_0 + x_1)^2$
#    and scale those into $[0, \pi]$ as well.
#
# This yields a 4-dimensional feature vector per sample:
# $f(x) = [x_0, x_1, (x_0 - x_1)^2, (x_0 + x_1)^2]$.
#
# The quantum kernel is therefore defined on the preprocessed features:
#
# $$
# k(x, x') = |\langle 0^n | U(f(x'))^\dagger \, U(f(x)) | 0^n \rangle|^2
# $$

# %%
angle_scaler = MinMaxScaler(feature_range=(0.0, math.pi), clip=True)
pair_scaler = MinMaxScaler(feature_range=(0.0, math.pi), clip=True)
X_train_angles = angle_scaler.fit_transform(X_train_raw)
X_test_angles = angle_scaler.transform(X_test_raw)


def lift_features(values: np.ndarray) -> np.ndarray:
    """Lift two scaled coordinates into four nonlinear circuit features.

    Args:
        values (np.ndarray): Two-column scaled input matrix.

    Returns:
        np.ndarray: Four-column matrix with two quadratic coordinates.
    """

    x0 = values[:, 0]
    x1 = values[:, 1]
    return np.column_stack((x0, x1, (x0 - x1) ** 2, (x0 + x1) ** 2))


F_train = lift_features(X_train_angles)
F_test = lift_features(X_test_angles)
F_train[:, 2:4] = pair_scaler.fit_transform(F_train[:, 2:4])
F_test[:, 2:4] = pair_scaler.transform(F_test[:, 2:4])

TRAIN_FEATURES = tuple(tuple(float(value) for value in row) for row in F_train)
TEST_FEATURES = tuple(tuple(float(value) for value in row) for row in F_test)
TRAIN_LABELS = tuple(int(label) for label in y_train)
EXPECTED_LABELS = tuple(int(label) for label in y_test)
SHOTS = 16 if docs_test_mode else 64

assert F_train.shape == (len(X_train_raw), 4)
assert F_test.shape == (len(X_test_raw), 4)

# %% [markdown]
# ## Compile Once, Run Many Times
#
# The overlap program is compiled once. Every Gram-matrix entry reuses the same
# artifact with different runtime vectors.

# %%
endpoint = qm.Reference()
artifact = qm.compile(program, to=endpoint.target)
kernel_runs: list[qm.RunRecord] = []

assert tuple(slot.name for slot in artifact.parameter_abi.slots) == (
    "left",
    "right",
)


def kernel_value(
    left: tuple[float, ...],
    right: tuple[float, ...],
    *,
    seed: int,
) -> float:
    """Estimate one all-zero overlap through the shared artifact.

    Args:
        left: First four-coordinate feature vector.
        right: Second four-coordinate feature vector.
        seed: Deterministic sampling seed.

    Returns:
        Estimated squared overlap in the closed interval from zero to one.
    """

    record = qm.run(
        artifact,
        on=endpoint,
        parameters={"left": left, "right": right},
        shots=SHOTS,
        seed=seed,
    )
    kernel_runs.append(record)
    return record.result.probability((0, 0))


train_kernel_rows = [[0.0] * len(TRAIN_FEATURES) for _ in TRAIN_FEATURES]
seed = 1
for row in range(len(TRAIN_FEATURES)):
    train_kernel_rows[row][row] = 1.0
    for column in range(row):
        value = kernel_value(
            TRAIN_FEATURES[row],
            TRAIN_FEATURES[column],
            seed=seed,
        )
        seed += 1
        train_kernel_rows[row][column] = value
        train_kernel_rows[column][row] = value
train_kernel_raw = np.asarray(train_kernel_rows, dtype=float)
test_kernel = np.asarray(
    [
        [
            kernel_value(point, training, seed=100 + row * 10 + column)
            for column, training in enumerate(TRAIN_FEATURES)
        ]
        for row, point in enumerate(TEST_FEATURES)
    ]
)

expected_kernel_runs = len(TRAIN_FEATURES) * (len(TRAIN_FEATURES) - 1) // 2
expected_kernel_runs += len(TEST_FEATURES) * len(TRAIN_FEATURES)
assert len(kernel_runs) == expected_kernel_runs
assert all(record.artifact is artifact for record in kernel_runs)
assert sum(record.result.shots for record in kernel_runs) == (
    expected_kernel_runs * SHOTS
)


def project_to_psd_correlation(matrix: np.ndarray) -> np.ndarray:
    """Project a shot-noisy symmetric Gram matrix to a PSD correlation matrix."""

    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    positive = np.maximum(eigenvalues, 1e-12)
    projected = (eigenvectors * positive) @ eigenvectors.T
    diagonal = np.sqrt(np.maximum(np.diag(projected), 1e-12))
    projected = projected / np.outer(diagonal, diagonal)
    projected = 0.5 * (projected + projected.T)
    np.fill_diagonal(projected, 1.0)
    return projected


train_kernel = project_to_psd_correlation(train_kernel_raw)
assert np.allclose(train_kernel, train_kernel.T)
assert np.allclose(np.diag(train_kernel), 1.0)
assert float(np.linalg.eigvalsh(train_kernel).min()) >= -1e-9
assert test_kernel.min() >= 0.0 and test_kernel.max() <= 1.0

# %% [markdown]
# Symmetry supplies the other triangular training entries, while self-overlap
# fixes the diagonal to one. A production training set of
# size $n$ still needs $O(n^2)$ kernel values; batching and measurement planning
# are execution concerns rather than new source-language semantics.

# %% [markdown]
# ## Train Classifiers
#
# We train three SVMs for comparison:
# - **Quantum kernel SVC** using the precomputed quantum kernel matrix.
# - **Linear SVC** using the raw 2D features (expected to fail on circles).
# - **RBF SVC** using a classical Gaussian kernel (strong baseline).

# %%
classifier = SVC(kernel="precomputed", C=1.0)
classifier.fit(train_kernel, TRAIN_LABELS)
predictions = classifier.predict(test_kernel)

linear_classifier = SVC(kernel="linear", C=1.0)
linear_classifier.fit(X_train_raw, y_train)
linear_predictions = linear_classifier.predict(X_test_raw)

rbf_classifier = SVC(kernel="rbf", C=1.0, gamma="scale")
rbf_classifier.fit(X_train_raw, y_train)
rbf_predictions = rbf_classifier.predict(X_test_raw)

print("test kernel:", test_kernel)
print("predicted labels:", predictions)

quantum_accuracy = accuracy_score(EXPECTED_LABELS, predictions)
linear_accuracy = accuracy_score(EXPECTED_LABELS, linear_predictions)
rbf_accuracy = accuracy_score(EXPECTED_LABELS, rbf_predictions)
print("Quantum kernel SVC:", quantum_accuracy)
print("Linear SVC:", linear_accuracy)
print("RBF SVC:", rbf_accuracy)
assert predictions.shape == y_test.shape
assert linear_predictions.shape == y_test.shape
assert rbf_predictions.shape == y_test.shape

# %% [markdown]
# ## Visualization
#
# We plot four panels:
# 1. The training data.
# 2. The quantum kernel matrix (Gram matrix).
# 3. The quantum kernel SVC decision boundary.
# 4. The classical RBF SVC decision boundary for comparison.


# %% [markdown]
# ### Decision Boundary Helpers


# %% [markdown]
# ### Plot Results

# %%
GRID_SIZE = 2 if docs_test_mode else 10


def preprocess_for_kernel(raw_points: np.ndarray) -> np.ndarray:
    """Apply the fitted scaling and nonlinear lift to new raw points."""

    angles = angle_scaler.transform(raw_points)
    features = lift_features(angles)
    features[:, 2:4] = pair_scaler.transform(features[:, 2:4])
    return features


margin = 0.35
x_min = X_raw[:, 0].min() - margin
x_max = X_raw[:, 0].max() + margin
y_min = X_raw[:, 1].min() - margin
y_max = X_raw[:, 1].max() + margin
grid_x, grid_y = np.meshgrid(
    np.linspace(x_min, x_max, GRID_SIZE),
    np.linspace(y_min, y_max, GRID_SIZE),
)
raw_grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
feature_grid = preprocess_for_kernel(raw_grid)
grid_kernel = np.asarray(
    [
        [
            kernel_value(
                tuple(float(value) for value in point),
                training,
                seed=1_000 + row * len(TRAIN_FEATURES) + column,
            )
            for column, training in enumerate(TRAIN_FEATURES)
        ]
        for row, point in enumerate(feature_grid)
    ]
)
quantum_boundary = classifier.predict(grid_kernel).reshape(grid_x.shape)
rbf_boundary = rbf_classifier.predict(raw_grid).reshape(grid_x.shape)

figure, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].scatter(X_train_raw[:, 0], X_train_raw[:, 1], c=y_train)
axes[0, 0].set_title("Training data")
image = axes[0, 1].imshow(train_kernel, cmap="viridis", vmin=0.0, vmax=1.0)
axes[0, 1].set_title("Quantum kernel matrix")
figure.colorbar(image, ax=axes[0, 1])
axes[1, 0].contourf(grid_x, grid_y, quantum_boundary, alpha=0.35)
axes[1, 0].scatter(X_test_raw[:, 0], X_test_raw[:, 1], c=y_test)
axes[1, 0].set_title("Quantum kernel SVC boundary")
axes[1, 1].contourf(grid_x, grid_y, rbf_boundary, alpha=0.35)
axes[1, 1].scatter(X_test_raw[:, 0], X_test_raw[:, 1], c=y_test)
axes[1, 1].set_title("RBF SVC boundary")
figure.tight_layout()
plt.show()

# %% [markdown]
# ## Interpreting the Results
#
# The visualization above shows how well the quantum kernel classifier captures the nonlinear class structure of `make_circles`.
#
# The **upper-left panel** (training data) shows that the two classes have a concentric-circle structure that cannot be separated by a straight line.
# In fact, the linear SVC achieved a test accuracy of 0.5 in this run, which is close to random guessing.
# This is because a linear decision boundary in the raw 2D coordinates cannot represent the concentric-circle class structure.
#
# The **upper-right panel** (quantum kernel matrix) displays the Gram matrix of pairwise similarities
#
# $$
# k(x, x') =
# \left|
# \langle 0^n |
# U(f(x'))^\dagger U(f(x))
# |0^n\rangle
# \right|^2
# $$
#
# computed over the training set.
# Diagonal entries are always 1 (each point is identical to itself).
# Off-diagonal entries indicate how close two data points are in the quantum feature space.
# Since the matrix rows are not sorted by class label, we cannot read a clear block structure directly.
# Nevertheless, when this matrix is passed to the SVM, it enables classification of data that is not linearly separable in the original 2D space.
#
# The **lower-left panel** (quantum kernel SVC boundary) shows a nonlinear decision boundary in input space.
# In this run, the quantum kernel SVC achieved a test accuracy of 1.0.
# This indicates that, for this small `make_circles` dataset, the quantum feature map we designed provides an effective similarity measure.
#
# The **lower-right panel** (RBF SVC) also forms a nonlinear decision boundary.
# In this run, the RBF SVC likewise achieved a test accuracy of 1.0.
# Thus, on this dataset both the quantum kernel SVC and the RBF SVC correctly classified all test points.
# However, the RBF SVC boundary is relatively smooth, whereas the quantum kernel SVC boundary appears somewhat rougher.
# This may be due to the small visualization grid (`GRID_SIZE = 15`) and the estimation of kernel values from `SHOTS = 1024` samples.
#
# In summary, the results of this experiment are:
#
# - Linear SVC: 0.5
# - Quantum kernel SVC: 1.0
# - RBF SVC: 1.0
#
# The quantum kernel SVC clearly outperformed the linear SVC and matched the test accuracy of the RBF SVC.
#
# However, this result alone does not demonstrate quantum advantage.
# The dataset is small-scale for visualization purposes, and classical feature lifting is also applied.
# What this notebook demonstrates is:
# "We can construct a quantum kernel matrix with Qamomile and feed it to an SVM for nonlinear binary classification."
# Discussing whether a quantum kernel is inherently superior to classical kernels would require larger datasets, multiple random seeds, shot-count dependence, layer-depth dependence, and detailed comparisons with classical kernels.
#
# Also note that the quantum kernel evaluation executes a quantum circuit for each pair of data points.
# For a training set of size $n$, computing the training kernel matrix requires roughly $O(n^2)$ kernel evaluations.
# This is not an issue for small datasets like ours, but the computational cost grows rapidly as the dataset size increases.
