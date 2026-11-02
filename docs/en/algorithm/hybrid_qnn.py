# ---
# jupyter:
#   jupytext:
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
# tags: [algorithm, machine-learning, variational]
# ---
#
# # Hybrid Quantum Neural Network (HQNN)
#
# This tutorial demonstrates how to build a **Hybrid Quantum Neural Network** (HQNN) on Fashion-MNIST that combines classical neural network layers with a quantum variational circuit layer. The quantum layer is defined using Qamomile's `@qkernel` API and integrated into a PyTorch training pipeline via the parameter shift rule.
#
# ## Architecture
#
# The entire model — CNN feature extractor, quantum layer, and fusion classifier — is trained **end-to-end from scratch**:
#
# ```
# Input Image → CNN → π·σ(·) → Quantum Layer → [CNN feats, Q output] → Classifier → Output
# ```
#
# Here, $\pi \cdot \sigma(\cdot)$ denotes element-wise sigmoid scaling: the CNN output is passed through the sigmoid function $\sigma$ and multiplied by $\pi$, mapping each feature to the interval $(0, \pi)$. This ensures that the rotation angles fed into the quantum layer remain bounded, avoiding wrap-around artifacts that can hinder training.
#
# The fusion classifier takes both classical CNN features and quantum expectation values, allowing the CNN and quantum layer to learn cooperatively from the start.

# %% [markdown]
# ## Installation
#
# Install Qamomile and PyTorch in the same environment:
#
# ```bash
# pip install qamomile torch
# ```

# %%
# Install the latest packages through pip!
# # !pip install qamomile torch

# %%
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

import qamomile as qm

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
N_QUBITS = 4
N_LAYERS = 1 if docs_test_mode else 2
N_WEIGHTS_PER_LAYER = N_QUBITS * 3
N_WEIGHTS = N_LAYERS * N_WEIGHTS_PER_LAYER

print(f"Qubits: {N_QUBITS}, Layers: {N_LAYERS}, Trainable weights: {N_WEIGHTS}")
assert N_WEIGHTS == (12 if docs_test_mode else 24)

# %% [markdown]
# ## Define the Variational Ansatz with `qkernel`
#
# Four input features are angle-encoded with $R_y$. Each variational layer then
# applies $R_zR_yR_z$ on every qubit and a nearest-neighbour `CZ` ladder.
# Inputs and weights are runtime `Param` vectors, so their values can change for
# every image without recompiling the circuit. Their lengths and the four-qubit
# state width are nevertheless checked statically.


# %%
@qm.qkernel
def hqnn_state(
    state: qm.Q[qm.Bits[4]],  # type: ignore[valid-type]
    inputs: qm.Vector[qm.Real, 4],  # type: ignore[valid-type]
    weights: qm.Vector[qm.Real, N_WEIGHTS],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[4]]:  # type: ignore[valid-type]
    """Apply the Hybrid QNN variational feature layers.

    Args:
        state (qm.Q[qm.Bits[4]]): Owned four-qubit input state.
        inputs (qm.Vector[qm.Real, 4]): Classical feature angles.
        weights (qm.Vector[qm.Real, N_WEIGHTS]): Trainable Z-Y-Z rotations.

    Returns:
        qm.Q[qm.Bits[4]]: Variational quantum feature state.
    """

    for index in range(4):
        state = qm.ry(state, inputs[index], target=index)
    for layer in range(N_LAYERS):
        base = layer * N_WEIGHTS_PER_LAYER
        for index in range(N_QUBITS):
            state = qm.rz(state, weights[base + index], target=index)
            state = qm.ry(state, weights[base + N_QUBITS + index], target=index)
            state = qm.rz(
                state,
                weights[base + 2 * N_QUBITS + index],
                target=index,
            )
        for index in range(N_QUBITS - 1):
            state = qm.cz(state, control=index, target=index + 1)
    return state


@qm.qkernel
def hqnn_samples(
    inputs: qm.Param[qm.Vector[qm.Real, 4]],  # type: ignore[valid-type]
    weights: qm.Param[qm.Vector[qm.Real, N_WEIGHTS]],  # type: ignore[valid-type]
) -> qm.Shot[qm.Bits[4]]:  # type: ignore[valid-type]
    """Measure one runtime-rebindable Hybrid QNN layer.

    Args:
        inputs (qm.Param[qm.Vector[qm.Real, 4]]): Runtime feature angles.
        weights (qm.Param[qm.Vector[qm.Real, N_WEIGHTS]]): Z-Y-Z weights.

    Returns:
        qm.Shot[qm.Bits[4]]: Four measured quantum features.
    """

    state = hqnn_state(
        qm.qubits(4),
        inputs,
        weights,
    )
    return qm.measure(state)


# %% [markdown]
# Every quantum operation consumes the previous state and returns its successor.
# Accidentally reusing an old state, passing a vector of the wrong length, or
# moving a shot-only value into a runtime parameter is rejected before backend
# lowering.

# %% [markdown]
# ## Transpile and Estimate Resources
#
# The logical estimator walks the same function body that will execute. It sees
# four input rotations, twelve rotations and three entanglers per layer. No
# neural-network operation or hand-authored resource formula is involved.

# %%
program = qm.specialize(hqnn_samples)
inspection = program.inspect()
study = qm.resources.logical(program)

assert "quantum.neural_network" not in inspection
assert study.metric(qm.resources.LOGICAL_ROTATION_GATES).numeric_enclosure() == (
    16 if docs_test_mode else 28,
    16 if docs_test_mode else 28,
)
assert study.metric(
    qm.resources.logical_operation_count("quantum.cz")
).numeric_enclosure() == (
    3 if docs_test_mode else 6,
    3 if docs_test_mode else 6,
)
assert qm.resources.ResourceStudy.from_bytes(study.to_bytes()) == study

reference = qm.Reference()
artifact = qm.compile(program, to=reference)

# %% [markdown]
# `artifact` is compiled once. The forward and backward passes below only bind
# new values to its two runtime vector slots. A backend change replaces
# `Reference` with another execution endpoint; it does not change the model
# source, parameter-shift rule, result order, or resource meaning.

# %% [markdown]
# ## Quantum Forward Pass
#
# One sampled result contains all four logical bits. `expectation_z(index)`
# derives each empirical $\langle Z_i\rangle$ from that shared histogram, so the
# quantum layer does not compile or execute four copies of the same circuit.

# %%
QUANTUM_SHOTS = 4 if docs_test_mode else 512


def quantum_features_batch(
    requests: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    seed: int,
) -> torch.Tensor:
    """Evaluate an ordered batch of sampled Z features through one artifact.

    Args:
        requests (list[tuple[torch.Tensor, torch.Tensor]]): Input and weight
            tensors for independent circuit invocations.
        seed (int): Deterministic backend seed for the first invocation.

    Returns:
        torch.Tensor: One row of four empirical Z expectations per request.
    """

    parameters = [
        {
            "inputs": tuple(float(value) for value in inputs.detach()),
            "weights": tuple(float(value) for value in weights.detach()),
        }
        for inputs, weights in requests
    ]
    results = qm.execute_batch(
        artifact,
        on=reference,
        parameters=parameters,
        shots=QUANTUM_SHOTS,
        seed=seed,
    )
    rows = []
    for result in results:
        assert isinstance(result, qm.SampleResult)
        rows.append(tuple(result.expectation_z(index) for index in range(N_QUBITS)))
    dtype = requests[0][0].dtype if requests else torch.float32
    return torch.tensor(rows, dtype=dtype)


# %% [markdown]
# ## Parameter Shift Rule for Gradients
#
# To train the quantum layer with PyTorch's autograd, we need gradients of the expectation values with respect to every parameter. The **parameter shift rule** gives us an exact gradient formula for gates of the form $e^{-i\theta G/2}$:
#
# $$
# \frac{\partial}{\partial \theta_k} \langle Z_i \rangle
# = \frac{1}{2} \Big[
#     \langle Z_i \rangle\big|_{\theta_k + \pi/2}
#   - \langle Z_i \rangle\big|_{\theta_k - \pi/2}
# \Big]
# $$
#
# We implement this as a custom `torch.autograd.Function` so that PyTorch can backpropagate through the quantum layer. All shifted parameter sets enter one ordered `execute_batch` call, so the compiled Reference artifact is restored only once per backward pass.

# %%
SHIFT = math.pi / 2.0


class QuantumFeatures(torch.autograd.Function):
    """Bridge one Qamomile artifact to PyTorch automatic differentiation."""

    @staticmethod
    def forward(
        ctx: object,
        inputs: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate a batch and retain values needed by parameter shift.

        Args:
            ctx (object): PyTorch autograd context.
            inputs (torch.Tensor): Batch of four feature angles.
            weights (torch.Tensor): Shared vector of all quantum weights.

        Returns:
            torch.Tensor: Batch of four quantum features.
        """

        ctx.save_for_backward(inputs, weights)  # type: ignore[attr-defined]
        return quantum_features_batch(
            [(row, weights) for row in inputs],
            seed=10,
        )

    @staticmethod
    def backward(
        ctx: object,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return gradients for input features and quantum weights.

        Args:
            ctx (object): PyTorch autograd context containing saved tensors.
            grad_output (torch.Tensor): Upstream feature gradient.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Input and weight gradients.
        """

        inputs, weights = ctx.saved_tensors  # type: ignore[attr-defined]
        grad_inputs = torch.zeros_like(inputs)
        grad_weights = torch.zeros_like(weights)
        requests: list[tuple[torch.Tensor, torch.Tensor]] = []
        for row in inputs:
            for input_index in range(row.numel()):
                plus = row.detach().clone()
                minus = row.detach().clone()
                plus[input_index] += SHIFT
                minus[input_index] -= SHIFT
                requests.extend(((plus, weights), (minus, weights)))

            for weight_index in range(weights.numel()):
                plus = weights.detach().clone()
                minus = weights.detach().clone()
                plus[weight_index] += SHIFT
                minus[weight_index] -= SHIFT
                requests.extend(((row, plus), (row, minus)))

        shifted = quantum_features_batch(requests, seed=100)
        cursor = 0
        for batch_index, row in enumerate(inputs):
            for input_index in range(row.numel()):
                derivative = (shifted[cursor] - shifted[cursor + 1]) / 2.0
                grad_inputs[batch_index, input_index] = torch.dot(
                    grad_output[batch_index], derivative
                )
                cursor += 2

            for weight_index in range(weights.numel()):
                derivative = (shifted[cursor] - shifted[cursor + 1]) / 2.0
                grad_weights[weight_index] += torch.dot(
                    grad_output[batch_index], derivative
                )
                cursor += 2

        return grad_inputs, grad_weights


# %% [markdown]
# ## Quantum Layer as a PyTorch Module
#
# We wrap the autograd function into a standard `nn.Module` so it can be composed with other layers.


# %%
class QuantumLayer(nn.Module):
    """A trainable PyTorch module backed by one Qamomile artifact."""

    def __init__(self) -> None:
        """Initialize all small trainable rotation angles."""

        super().__init__()
        self.weights = nn.Parameter(torch.randn(N_WEIGHTS) * 0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate quantum features for one batch.

        Args:
            inputs (torch.Tensor): Batch of four bounded feature angles.

        Returns:
            torch.Tensor: Batch of four sampled quantum features.
        """

        return QuantumFeatures.apply(inputs, self.weights)


# %% [markdown]
# ## Prepare Data (Fashion-MNIST)
#
# We use a subset of [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) with 4 visually distinct classes. Fashion-MNIST consists of 28x28 grayscale clothing images.
#
# To keep computation time reasonable, we select 4 classes (T-shirt, Trouser, Sandal, Bag) with 60 training and 30 test samples per class.

# %%
N_CLASSES = 4
SELECTED_CLASSES = [0, 1, 5, 8]
CLASS_NAMES = ["T-shirt", "Trouser", "Sandal", "Bag"]
N_TRAIN_PER_CLASS = 1 if docs_test_mode else 60
N_TEST_PER_CLASS = 1 if docs_test_mode else 30


def subset_dataset(
    dataset: object,
    classes: list[int],
    n_per_class: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract a reproducible balanced subset of Fashion-MNIST.

    Args:
        dataset: Torchvision dataset with ``targets`` and indexed samples.
        classes: Original Fashion-MNIST class IDs to retain.
        n_per_class: Number of images selected from each class.

    Returns:
        Images and consecutive labels for the selected classes.
    """

    images: list[torch.Tensor] = []
    labels: list[int] = []
    generator = np.random.default_rng(0)
    targets = dataset.targets  # type: ignore[attr-defined]
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    else:
        targets = np.asarray(targets)
    for new_label, original_label in enumerate(classes):
        class_indices = np.where(targets == original_label)[0]
        chosen = generator.choice(class_indices, size=n_per_class, replace=False)
        for index in chosen:
            image, _ = dataset[int(index)]  # type: ignore[index]
            images.append(image)
            labels.append(new_label)
    return torch.stack(images), torch.tensor(labels)


if docs_test_mode:
    data_generator = torch.Generator().manual_seed(0)
    X_train = torch.rand((N_CLASSES, 1, 28, 28), generator=data_generator)
    X_test = torch.rand((N_CLASSES, 1, 28, 28), generator=data_generator)
    y_train = torch.arange(N_CLASSES)
    y_test = torch.arange(N_CLASSES)
else:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    data_root = os.path.join(os.path.expanduser("~"), ".cache", "fashion_mnist")
    full_train = datasets.FashionMNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )
    full_test = datasets.FashionMNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )
    X_train, y_train = subset_dataset(
        full_train,
        SELECTED_CLASSES,
        N_TRAIN_PER_CLASS,
    )
    X_test, y_test = subset_dataset(
        full_test,
        SELECTED_CLASSES,
        N_TEST_PER_CLASS,
    )

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Classes: {CLASS_NAMES}")
assert X_train.shape == (N_CLASSES * N_TRAIN_PER_CLASS, 1, 28, 28)
assert X_test.shape == (N_CLASSES * N_TEST_PER_CLASS, 1, 28, 28)

# %% [markdown]
# ## End-to-End Hybrid Model
#
# We define the full hybrid model and train it **end-to-end from scratch**. The CNN feature extractor and quantum layer learn cooperatively — gradients flow from the loss through the fusion classifier, through the quantum layer (via parameter shift rule), and back into the CNN.
#
# ```
# Input Image → [CNN Feature Extractor] → π·σ(·) → [Quantum Layer] → [CNN feats, Q output] → [Classifier] → Output
# ```
#
# - **Sigmoid scaling**: $\pi \cdot \sigma(\cdot)$ maps CNN outputs to $(0, \pi)$, keeping angles bounded and avoiding wrap-around.
# - **Fusion classifier**: Concatenation of `[CNN features, quantum outputs]` gives the classifier access to both classical and quantum representations.


# %%
class HybridClassifier(nn.Module):
    """End-to-end classical-quantum Fashion-MNIST classifier."""

    def __init__(self) -> None:
        """Construct the classical extractor, quantum layer, and classifier."""

        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(4, 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(7 * 7 * 4, 4),
        )
        self.quantum_layer = QuantumLayer()
        self.classifier = nn.Linear(8, N_CLASSES)

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logits and both feature representations.

        Args:
            images (torch.Tensor): Batch of one-channel 28-by-28 images.

        Returns:
            Logits, classical features, and quantum features.
        """

        classical = math.pi * torch.sigmoid(self.feature_extractor(images))
        quantum = self.quantum_layer(classical)
        logits = self.classifier(torch.cat((classical, quantum), dim=1))
        return logits, classical, quantum


EPOCHS = 1 if docs_test_mode else 10
BATCH_SIZE = 4

torch.manual_seed(42)
model = HybridClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
train_losses: list[float] = []
test_accuracies: list[float] = []

print(f"=== End-to-End Training ({EPOCHS} epochs) ===")
for epoch in range(EPOCHS):
    permutation = torch.randperm(len(X_train))
    shuffled_images = X_train[permutation]
    shuffled_labels = y_train[permutation]
    epoch_loss = 0.0
    batches = 0

    for start in range(0, len(shuffled_images), BATCH_SIZE):
        images = shuffled_images[start : start + BATCH_SIZE]
        labels = shuffled_labels[start : start + BATCH_SIZE]
        optimizer.zero_grad()
        logits, _, _ = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        batches += 1

    train_losses.append(epoch_loss / batches)
    with torch.no_grad():
        test_logits, _, _ = model(X_test)
        test_accuracy = (test_logits.argmax(1) == y_test).float().mean().item()
        test_accuracies.append(test_accuracy)
    print(
        f"Epoch {epoch + 1}/{EPOCHS}: "
        f"loss={train_losses[-1]:.4f}, test_acc={test_accuracy:.2%}"
    )

optimizer.zero_grad()
weights_before = model.quantum_layer.weights.detach().clone()
logits, classical_features, quantum_outputs = model(X_train[:BATCH_SIZE])
loss = criterion(logits, y_train[:BATCH_SIZE])
loss.backward()

first_convolution = model.feature_extractor[0]
assert isinstance(first_convolution, nn.Conv2d)
assert first_convolution.weight.grad is not None
assert model.quantum_layer.weights.grad is not None
assert model.classifier.weight.grad is not None
assert model.quantum_layer.weights.grad.abs().sum().item() > 0.0

optimizer.step()

assert logits.shape == (BATCH_SIZE, N_CLASSES)
assert classical_features.shape == (BATCH_SIZE, 4)
assert quantum_outputs.shape == (BATCH_SIZE, 4)
assert loss.item() > 0.0
assert not torch.equal(weights_before, model.quantum_layer.weights.detach())

print("logits:", logits.detach().tolist())
print("loss:", loss.item())
print(
    "quantum gradient norm:",
    model.quantum_layer.weights.grad.norm().item(),
)

# %% [markdown]
# The assertions establish the complete end-to-end path: the loss reaches the
# first CNN convolution, all quantum weights through parameter shift,
# and the fusion classifier; the optimizer then changes the quantum parameters.

# %% [markdown]
# ## Evaluation

# %%
model.eval()
with torch.no_grad():
    evaluation_logits, _, evaluation_quantum = model(X_test)
    predictions = evaluation_logits.argmax(1)
    accuracy = (predictions == y_test).float().mean().item()

print(f"End-to-end quantum model accuracy: {accuracy:.2%}")
for class_index, class_name in enumerate(CLASS_NAMES):
    mask = y_test == class_index
    class_accuracy = (predictions[mask] == y_test[mask]).float().mean().item()
    print(f"  {class_name}: {class_accuracy:.2%}")

assert evaluation_logits.shape == (len(X_test), N_CLASSES)
assert evaluation_quantum.shape == (len(X_test), 4)

figure, axes = plt.subplots(1, 3, figsize=(15, 4))
epoch_numbers = range(1, EPOCHS + 1)
axes[0].plot(epoch_numbers, train_losses, "o-", label="Train loss")
axes[0].plot(epoch_numbers, test_accuracies, "s-", label="Test accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss / Accuracy")
axes[0].set_title("Learning Curves")
axes[0].legend()

confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
for expected, predicted in zip(y_test.numpy(), predictions.numpy(), strict=True):
    confusion[expected, predicted] += 1
axes[1].imshow(confusion, cmap="Blues")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")
axes[1].set_title("Confusion Matrix")
axes[1].set_xticks(range(N_CLASSES), CLASS_NAMES, rotation=45)
axes[1].set_yticks(range(N_CLASSES), CLASS_NAMES)

shown = min(8, len(X_test))
combined = np.concatenate(X_test[:shown, 0].numpy(), axis=1)
axes[2].imshow(combined, cmap="gray", aspect="auto")
for index in range(shown):
    color = "lime" if predictions[index] == y_test[index] else "red"
    axes[2].text(
        28 * index + 14,
        -1.5,
        CLASS_NAMES[int(predictions[index].item())],
        ha="center",
        va="bottom",
        fontsize=7,
        color=color,
        clip_on=False,
    )
axes[2].set_ylim(33, -8)
axes[2].set_title("Sample Predictions")
axes[2].axis("off")
figure.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# - One typed four-qubit Qamomile function defines the quantum layer.
# - One compiled artifact is rebound for inference and every shifted gradient
#   evaluation; four observables are derived from one shared histogram.
# - QCore checks parameter shapes, staging, owned-state flow, result order, and
#   exact logical resources without learning PyTorch semantics.
# - PyTorch owns the parameter-shift bridge, end-to-end gradient flow, and
#   optimizer update.
# - Normal execution trains and evaluates the four-class Fashion-MNIST fusion
#   model and renders the same three diagnostic views as the original tutorial.
# - Documentation tests retain that entire path with one synthetic image per
#   class, one epoch, and four shots per quantum launch.
#
# Next, [Möttönen Amplitude Encoding](mottonen_amplitude_encoding.ipynb) shows
# how Rust-derived coordinates prepare arbitrary complex amplitude data.
