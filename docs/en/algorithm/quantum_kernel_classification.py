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
#
# # Quantum Kernel Classification
#
# This tutorial implements a binary classifier using a quantum kernel with Qamomile. A two-qubit feature-map circuit estimates the similarity between data points. We use these similarities to train scikit-learn's `SVC`, a classifier based on a support vector machine.
#
# We proceed as follows:
#
# 1. Prepare a two-class dataset of concentric circles (`make_circles`).
# 2. Use classical preprocessing to turn each two-dimensional coordinate into four circuit inputs.
# 3. Define the feature-map and overlap-measurement circuits with `@qkernel`.
# 4. Compute the kernel matrices and classify the data with an SVC using `kernel="precomputed"`.
# 5. Compare the results with linear and RBF SVCs applied to the raw two-dimensional coordinates.
#
# This is a small demonstration for learning how to use a quantum kernel. Because the preprocessing adds classical nonlinear features, the classification results alone do not demonstrate an advantage for the quantum kernel.

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
# ## Background
#
# ### Quantum Kernel Methods
#
# Let $x$ be a raw data point and $f(x)$ its four-dimensional feature vector after preprocessing. The two-qubit quantum feature state is
#
# $$
# |\phi(x)\rangle = U(f(x))|00\rangle.
# $$
#
# We define the fidelity kernel (overlap kernel) used here as
#
# $$
# k(x,x') = |\langle\phi(x')|\phi(x)\rangle|^2
# = \left|\langle00|U(f(x'))^{\dagger}U(f(x))|00\rangle\right|^2.
# $$
#
# To estimate $k(x,x')$, apply $U(f(x))$ followed by $U(f(x'))^{\dagger}$ and estimate the probability of measuring $|00\rangle$ {cite:p}`10.1038/s41586-019-0980-2`.
#
# The ideal kernel satisfies $0\leq k(x,x')\leq1$ and $k(x,x)=1$. Its kernel matrix is positive semidefinite (PSD). Indeed, writing $\rho(x)=|\phi(x)\rangle\langle\phi(x)|$ gives $k(x,x')=\operatorname{Tr}[\rho(x)\rho(x')]$.
#
# However, a matrix whose entries are estimated separately with a finite number of shots need not be PSD because of statistical error. This motivates the correction to the training matrix later in the tutorial.

# %% [markdown]
# ## Problem Settings
#
# ### Dataset
#
# We use scikit-learn's `make_circles` to generate two classes arranged in concentric circles. This is an example that is difficult to separate with a straight line in the raw two-dimensional coordinate space.
#
# This example generates 40 samples, split into 30 training samples and 10 test samples.

# %%
N_SAMPLES = 4 if docs_test_mode else 40
X_raw, labels = make_circles(
    n_samples=N_SAMPLES,
    noise=0.04,
    factor=0.4,
    random_state=7,
)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw,
    labels,
    test_size=0.5 if docs_test_mode else 0.25,
    random_state=7,
    stratify=labels,
)

# %% [markdown]
# ### Classical Preprocessing: Feature Lifting
#
# The quantum circuit takes four real-valued inputs. For an original coordinate $x=(x_0,x_1)$, first apply a scaler $s_1$ fitted to the training data:
#
# $$
# a=s_1(x)=(a_0,a_1), \qquad a_i\in[0,\pi].
# $$
#
# Next, define two nonlinear features:
#
# $$
# q(a)=\big((a_0-a_1)^2,\ (a_0+a_1)^2\big).
# $$
#
# Apply a second scaler $s_2$ to these two columns. The final circuit input is
#
# $$
# f(x)=\big(a_0,a_1,s_2(q(a))_0,s_2(q(a))_1\big).
# $$
#
# Thus, we not only add squared terms but also rescale the two added columns to $[0,\pi]$.
#
# Both `MinMaxScaler` instances are fitted only to the training data; the test data is processed with `transform` only.
#
# Because `clip=True` clips coordinates outside the training range to its endpoints, different inputs in the outer region of the visualization may map to the same circuit features.

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
SHOTS = 1 if docs_test_mode else 64

assert F_train.shape == (len(X_train_raw), 4)
assert F_test.shape == (len(X_test_raw), 4)

# %% [markdown]
# ## Algorithm
#
# ### Feature-Map Circuit
#
# We encode four real-valued features into two qubits. The `classification_feature_map` circuit repeats a fixed block twice, using H, RZ, and RY gates on each qubit together with gates such as CZ and RZZ.
#
# The circuit inputs come from the data, so we do not optimize quantum circuit parameters here. We train the `SVC` classifier.
#
# ### Overlap-Measurement Circuit
#
# First apply $U(f(x))$, then use `qm.adjoint` to apply $U(f(x'))^{\dagger}$. The probability of measuring `(0, 0)` is the kernel value.
#
# The `left` and `right` arguments of `classification_overlap` each receive a real-valued vector of length four at runtime. The feature block is repeated twice in the code; there is no specialization parameter named `layers`.


# %% [markdown]
# ## Implementation

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
        left (qm.Vector[qm.Real, 4]): First feature vector.
        right (qm.Vector[qm.Real, 4]): Second feature vector.

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
        left (qm.Param[qm.Vector[qm.Real, 4]]): First runtime feature vector.
        right (qm.Param[qm.Vector[qm.Real, 4]]): Second runtime feature vector.

    Returns:
        qm.Shot[qm.Bits[2]]: Two logical overlap-witness bits per shot.
    """

    return qm.measure(classification_overlap_state(left, right))


# %% [markdown]
# `qm.adjoint` expresses the inverse operation using the same feature-map circuit. There is no need to define the reversed gate sequence by hand, making it easier to keep the forward and inverse operations consistent when the feature map changes.
#
# ### Specialize the Circuit and Inspect Logical Resources
#
# Use `qm.specialize` to finalize the circuit as a program with runtime parameters, then inspect its logical gate estimate with `qm.resources.logical`.
#
# In the current circuit, one feature block contains 10 gates, and the feature map repeats it twice for a total of 20 gates. Combining the forward and inverse feature maps gives 40 logical gates before measurement in the overlap circuit. Resource counts can change with the circuit definition and counting convention.

# %%
program = qm.specialize(classification_overlap)
study = qm.resources.logical(program)
logical_gate_range = study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure()
print("Logical gates (lower, upper):", logical_gate_range)

# %% [markdown]
# ## Result
#
# ### Compile Once and Evaluate the Kernel for Different Inputs
#
# Compile the overlap circuit once, then bind `left` and `right` for each pair of data points and execute the same artifact repeatedly. The execution endpoint is Qamomile's `Reference()`.
#
# For the training kernel matrix, use the symmetry $k(x_i,x_j)=k(x_j,x_i)$ to measure only entries below the diagonal and reuse the same values for the corresponding entries above it. Since the ideal self-overlap is $k(x_i,x_i)=1$, set each diagonal entry to one.
#
# This example uses 64 shots per pair. The observed frequency of `(0, 0)` is an estimate in increments of $1/64$, with a maximum standard error of $1/(2\sqrt{64})=0.0625$. This is finite-shot statistical error, not a simulation of hardware noise.

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
next_train_seed = 1
for row in range(len(TRAIN_FEATURES)):
    train_kernel_rows[row][row] = 1.0
    for column in range(row):
        value = kernel_value(
            TRAIN_FEATURES[row],
            TRAIN_FEATURES[column],
            seed=next_train_seed,
        )
        next_train_seed += 1
        train_kernel_rows[row][column] = value
        train_kernel_rows[column][row] = value
train_kernel_raw = np.asarray(train_kernel_rows, dtype=float)
test_seed_start = next_train_seed
test_kernel = np.asarray(
    [
        [
            kernel_value(point, training, seed=test_seed_start + row * len(TRAIN_FEATURES) + column)
            for column, training in enumerate(TRAIN_FEATURES)
        ]
        for row, point in enumerate(TEST_FEATURES)
    ]
)

grid_seed_start = test_seed_start + len(TEST_FEATURES) * len(TRAIN_FEATURES)

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
print("Raw train Gram minimum eigenvalue:", float(np.linalg.eigvalsh(train_kernel_raw)[0]))
print("Maximum absolute PSD correction:", float(np.max(np.abs(train_kernel - train_kernel_raw))))
assert np.allclose(train_kernel, train_kernel.T)
assert np.allclose(np.diag(train_kernel), 1.0)
assert float(np.linalg.eigvalsh(train_kernel).min()) >= -1e-9
assert np.all(np.isfinite(train_kernel)) and np.all(np.isfinite(test_kernel))
assert test_kernel.min() >= 0.0 and test_kernel.max() <= 1.0

# %% [markdown]
# ### Finite-Shot Kernel Matrices and PSD Correction
#
# The ideal fidelity kernel matrix is PSD, but `train_kernel_raw`, whose off-diagonal entries are each estimated with 64 shots, may have negative eigenvalues. This notebook therefore replaces eigenvalues below a small positive threshold with that threshold, reconstructs the matrix, and normalizes its diagonal to one before passing the resulting `train_kernel` to the SVC. This is a numerical correction to make the training matrix PSD, not an exact optimization for the nearest such matrix.
#
# The correction also changes individual matrix entries. After correction, entries are no longer guaranteed to equal the original measured probabilities and can even be slightly negative.
#
# The test matrix, `test_kernel`, retains the uncorrected measurement estimates, so this procedure does not guarantee that both matrices come from a single exact quantum kernel. A rigorous assessment of this approximation would require comparisons across shot counts or correction methods.
#
# With 30 training samples, this example requires 435 off-diagonal training kernel evaluations and 300 comparisons between the 10 test samples and the 30 training samples. The 10×10 grid used later for the decision boundary adds another 3000 pairs, for a total of 3735 circuit evaluations.
#
# ### Train the Classifiers
#
# We compare three SVC models:
#
# - Quantum kernel SVC: uses a matrix estimated from the quantum feature map, including its preprocessing.
# - Linear SVC: uses the original two-dimensional coordinates directly.
# - RBF SVC: applies a Gaussian kernel to the original two-dimensional coordinates.
#
# The quantum kernel includes classical nonlinear feature lifting, which is not applied to the classical SVC baselines. Consequently, differences in accuracy among these three models do not isolate the effect of quantum computation. A fair performance comparison would require additional baselines, such as classical models using the same preprocessing.

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

print("Test kernel shape:", test_kernel.shape)
print("First test kernel row (up to 8 values):", np.round(test_kernel[0, :8], 3))
print("True labels:", EXPECTED_LABELS)
print("Predicted labels:", predictions)

quantum_accuracy = accuracy_score(EXPECTED_LABELS, predictions)
linear_accuracy = accuracy_score(EXPECTED_LABELS, linear_predictions)
rbf_accuracy = accuracy_score(EXPECTED_LABELS, rbf_predictions)
print("Test sample count:", len(EXPECTED_LABELS))
print("Quantum kernel SVC:", quantum_accuracy)
print("Linear SVC:", linear_accuracy)
print("RBF SVC:", rbf_accuracy)
assert predictions.shape == y_test.shape
assert linear_predictions.shape == y_test.shape
assert rbf_predictions.shape == y_test.shape

# %% [markdown]
# ### Visualization
#
# We show four plots:
#
# 1. The training data.
# 2. The PSD-corrected training kernel matrix passed to the SVC.
# 3. The quantum kernel SVC decision boundary.
# 4. The RBF SVC decision boundary.
#
# The grid contains only 10×10 points, so the displayed boundary is a coarse approximation. Grid points outside the training range are also clipped by `MinMaxScaler(clip=True)`.


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
                seed=grid_seed_start + row * len(TRAIN_FEATURES) + column,
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
image = axes[0, 1].imshow(
    train_kernel, cmap="viridis", vmin=min(0.0, float(train_kernel.min())), vmax=1.0
)
axes[0, 1].set_title("PSD-adjusted training kernel")
figure.colorbar(image, ax=axes[0, 1])
axes[1, 0].contourf(grid_x, grid_y, quantum_boundary, levels=[-0.5, 0.5, 1.5], alpha=0.35)
axes[1, 0].scatter(X_test_raw[:, 0], X_test_raw[:, 1], c=y_test)
axes[1, 0].set_title("Quantum kernel SVC boundary")
axes[1, 1].contourf(grid_x, grid_y, rbf_boundary, levels=[-0.5, 0.5, 1.5], alpha=0.35)
axes[1, 1].scatter(X_test_raw[:, 0], X_test_raw[:, 1], c=y_test)
axes[1, 1].set_title("RBF SVC boundary")
figure.tight_layout()
plt.show()

# %% [markdown]
# ### Interpreting the Results and Their Limitations
#
# The training data forms concentric circles, a structure that is difficult to separate with a straight line in the raw two-dimensional coordinate space. Both the quantum kernel SVC and the RBF SVC can produce nonlinear decision boundaries; use the displayed test accuracies to assess which performs better in this run. Accuracy depends on the shot count and random seeds.
#
# The training kernel matrix has ones on its diagonal. Off-diagonal entries represent similarities between data points, but the displayed matrix has undergone PSD correction. Because of this correction, do not interpret its entries as the uncorrected measured probabilities. The training data is also not sorted by class, so clusters of colors cannot simply be interpreted as within-class or between-class similarities.
#
# This example uses 30 training samples and 10 test samples. A test accuracy of 0.9 means that nine of the ten test samples were classified correctly. At this scale, changing the outcome of a single prediction changes accuracy by 0.1, so avoid generalizing from a single data split.
#
# This comparison demonstrates that Qamomile can estimate a quantum kernel and pass it to scikit-learn's SVC for binary classification. It does not demonstrate quantum advantage or an inherent performance advantage over classical kernels. Performance comparisons require repeated experiments with different data splits and seeds, an analysis of shot-count dependence, and control experiments using the same classical preprocessing.
#
# For $n$ training samples, constructing the training kernel requires roughly $O(n^2)$ pair evaluations. Finer decision-boundary plots also increase the number of kernel evaluations, since each grid point must be compared with every training sample.

# %% [markdown]
# ## Summary
#
# - Quantum kernel methods encode classical data into quantum states and use the fidelity between those states as a measure of similarity between data points. This tutorial constructs a kernel matrix using a two-qubit feature map.
# - Qamomile's `@qkernel` and `qm.adjoint` let us implement the feature map and its inverse concisely. We can also reuse a compiled quantum circuit and change its input parameters to evaluate the kernel for different pairs of data points.
# - Finite-shot measurement error can cause an estimated kernel matrix to lose its theoretical PSD property. This tutorial applies a numerical correction to the training kernel matrix before using it for classification with a classical support vector classifier (SVC).
# - Combining a quantum kernel with classical machine learning lets us implement nonlinear binary classification. These results alone do not establish an advantage for the quantum kernel; performance evaluation requires varying data splits and shot counts and making fair comparisons with classical models.
