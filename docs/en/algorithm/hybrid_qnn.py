# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# This tutorial presents an example of implementing a **Hybrid Quantum Neural Network** (HQNN), which combines a classical neural network with a variational quantum circuit, using Qamomile. For an image-recognition task on four classes from the Fashion-MNIST dataset, we combine a classical convolutional neural network with a four-qubit quantum circuit. We estimate gradients for the quantum circuit using the parameter-shift rule and pass them to PyTorch to train the classical and quantum parameters together.

# %%
# Install the dependencies in the Python environment used by this notebook.
# # !python -m pip install qamomile torch torchvision numpy matplotlib

# %%
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

import qamomile as qm

# %% [markdown]
# ## Background: Image Recognition and Machine Learning
#
# ### Convolutional Neural Networks (CNNs)
#
# A neural network is a machine-learning model that produces a prediction by transforming an input through a sequence of layers. If $\mathbf{h}^{(0)}=\mathbf{x}$ denotes the input, layer $l$ computes
#
# $$
# \mathbf{h}^{(l)} = \phi^{(l)}\!\left(
#     W^{(l)}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}
# \right), \qquad l=1,\ldots,L,
# $$
#
# where $W^{(l)}$ and $\mathbf{b}^{(l)}$ are trainable weights and biases, and $\phi^{(l)}$ is a nonlinear activation function. For classification, the final representation gives the predicted class probabilities through
#
# $$
# \hat{\mathbf{y}}
# = \operatorname{softmax}\!\left(
#     W_{\mathrm{out}}\mathbf{h}^{(L)} + \mathbf{b}_{\mathrm{out}}
# \right).
# $$
#
# Training computes a loss from the predictions and target labels, propagates its derivatives backward through the network, and updates the weights and biases in every layer.
#
# Image recognition is a machine-learning task that assigns a class to an input image. Processing an image with fully connected layers alone does not directly exploit spatial relationships between pixels and can require many parameters. A convolutional neural network (CNN) instead shares small filters across the image to extract local patterns such as edges and textures as features. The local region of the image examined by a filter at one time is called its local receptive field. Pooling layers then progressively reduce the spatial resolution. A classifier uses the resulting features to distinguish image classes {cite:p}`10.1038/nature14539`.
#
# ### Quantum Neural Networks (QNNs)
#
# A quantum neural network (QNN) uses trainable quantum operations as part of a learning model. In the classifier proposed by Farhi and Neven, a parameterized quantum circuit transforms the input state, and a measurement on a readout qubit provides the classification output. A classical optimizer adjusts the circuit parameters using labeled examples. Classical inputs first require an encoding into quantum states; the model can also classify quantum states supplied directly {cite:p}`10.48550/arXiv.1802.06002`.
#
# The term QNN also covers other architectures. For example, Beer et al. study feedforward quantum networks trained on pairs of input and target quantum states, using their state fidelity, a measure of how closely two quantum states agree, as the training objective {cite:p}`10.1038/s41467-020-14454-2`.
#
# The quantum convolutional neural network (QCNN) proposed by Cong et al. applies convolution and pooling to quantum-state inputs. Local unitary operations share parameters, while pooling measures some qubits and uses the results to control operations on the remaining qubits, progressively reducing the number of active qubits {cite:p}`10.1038/s41567-019-0648-8`.
#
# A related construction for classical images is quanvolution: small image patches are encoded into quantum states, the same quantum circuit is repeatedly applied as a filter, and measured values form feature maps. Senokosov et al. study hybrid image classifiers that combine classical processing with quantum circuits, including an architecture with a quanvolutional layer {cite:p}`10.1088/2632-2153/ad2aef`. In this tutorial, the image convolution is performed by a classical CNN, and a variational quantum circuit transforms the resulting compact feature vector.

# %% [markdown]
# ## Problem Setting: Fashion-MNIST
#
# [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) is an image-recognition dataset of $28 \times 28$ grayscale clothing images organized into 10 classes. It contains 60,000 training images and 10,000 test images and is commonly used as a benchmark for comparing image-classification models.
#
# In this tutorial, we select four classes: T-shirt/top (0), Trouser (1), Sandal (5), and Bag (8). We use 60 training and 30 test images from each class, giving 240 training images and 120 test images in total. The original class labels are mapped to consecutive labels from 0 to 3. We fix the random seed used for selection so that the same subsets can be reproduced.
#
# The following code selects the training and test subsets, converting the selected images to tensors and normalizing their pixel values. Training uses only the training subset; the test subset is used to evaluate the classifier.

# %%
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
N_CLASSES = 4
SELECTED_CLASSES = [0, 1, 5, 8]
CLASS_NAMES = ["T-shirt/top", "Trouser", "Sandal", "Bag"]
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
# ## Algorithm
#
# To keep the quantum circuit small, we use a classical convolutional block as a front-end feature extractor. It compresses local image features into a four-dimensional vector, which is then processed by a compact parameterized quantum circuit. Senokosov et al. provide a precedent for hybrid image classification that combines classical dimensionality reduction with quantum feature transformations {cite:p}`10.1088/2632-2153/ad2aef`. We use this work as architectural context; the circuit, feature fusion, and training settings below form a separate example.
#
# The HQNN has three components: a classical convolutional layer, a quantum circuit layer, and a classical fully connected layer. The classifier receives both the CNN features and the quantum features:
#
# $$
# \begin{aligned}
# \text{Input image}
# &\longrightarrow
# \underbrace{\mathrm{CNN}\longrightarrow\pi\times\operatorname{sigmoid}}
# _{\text{Classical convolutional layer}}
# \longrightarrow \text{Quantum circuit layer} \\
# &\longrightarrow
# \underbrace{
#     \text{Concatenate outputs}(\text{CNN features},\,\text{quantum output})
#     \longrightarrow \text{Classifier}
# }_{\text{Classical fully connected layer}}
# \longrightarrow \text{Output}
# \end{aligned}
# $$
#
# In this workflow, a classical convolutional layer extracts $n$ features from an image, a quantum circuit layer transforms them into $m$ estimated expectation values, and a classical fully connected layer predicts probabilities over $K$ classes. Here, $n=m=K=4$. The same classification loss trains every component together.
#
# ### Classical Convolutional Layer
#
# The classical convolutional layer extracts local patterns from the input image as features and compresses them into an $n$-dimensional feature vector that can be supplied to the quantum circuit.
#
# Let $C$, $H$, and $W$ denote the number of input channels, height, and width, respectively, so that a normalized image is $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$. Each convolution block applies a learned $k_h \times k_w$ filter, an activation function $\phi$, and $p_h \times p_w$ pooling. If $\mathbf{h}^{(0)} = \mathbf{X}$, the $L_{\mathrm{CNN}}$ blocks can be written schematically as
#
# $$
# \mathbf{h}^{(l)} = \operatorname{Pool}\!\left(
#     \phi\!\left(\mathbf{K}^{(l)} * \mathbf{h}^{(l-1)}
#     + \mathbf{b}^{(l)}\right)
# \right), \qquad l \in \mathcal{L}_{\mathrm{CNN}},
# $$
#
# where $\operatorname{Pool}$ denotes pooling over a $p_h \times p_w$ window, $*$ denotes convolution, and $\mathcal{L}_{\mathrm{CNN}}$ is the index set of convolution blocks. The resulting tensor is flattened and projected to an $n$-dimensional feature vector. These features are converted into quantum rotation angles by
#
# $$
# \mathbf{x} = \alpha\,\sigma\!\left(
#     W_{\mathrm{CNN}}\operatorname{vec}(\mathbf{h}^{(L_{\mathrm{CNN}})})
#     + \mathbf{b}_{\mathrm{CNN}}
# \right) \in \mathbb{R}^{n}.
# $$
#
# Here, $\sigma$ is the sigmoid function and $\alpha=\pi$, so each component of $\mathbf{x}$ lies in $(0,\pi)$. Bounding the angles avoids large inputs that span multiple rotation periods, but this alone does not guarantee stable training.
#
# The implementation uses two convolution blocks, each with four output channels, a $3\times3$ filter, ReLU activation, and $2\times2$ max pooling. The spatial size decreases from $28\times28$ to $14\times14$ and then $7\times7$. A fully connected layer projects the resulting $4\times7\times7=196$ values to the four features used as quantum rotation angles.
#
# ### Quantum Circuit Layer
#
# The quantum circuit layer encodes the features extracted by the classical convolutional layer into a quantum state, transforms them with a variational quantum circuit, and outputs $m=4$ estimated expectation values.
#
# The $n=4$ CNN features are encoded into four qubits with RY rotations. If $\mathcal{Q}=\{0,1,2,3\}$ denotes the qubit index set, the encoding unitary is
#
# $$
# U_{\mathrm{enc}}(\mathbf{x})
# = \prod_{i \in \mathcal{Q}} R_{Y,i}(x_i).
# $$
#
# A variational circuit $V(\boldsymbol{\theta})$ with $L_{\mathrm{Q}}=2$ layers then applies trainable RZ-RY-RZ rotations to every qubit followed by a linear chain of CZ gates on pairs $(0,1)$, $(1,2)$, and $(2,3)$. The resulting state is
#
# $$
# |\psi(\mathbf{x},\boldsymbol{\theta})\rangle
# = V(\boldsymbol{\theta})U_{\mathrm{enc}}(\mathbf{x})
# |\mathbf{0}\rangle_{\mathcal{Q}}.
# $$
#
# Each layer has $d=3$ trainable rotation parameters per qubit, so the circuit has $L_{\mathrm{Q}}\times n\times d=2\times4\times3=24$ quantum parameters.
#
# Before measurement, we apply a Hadamard gate to every qubit. Writing the state after these gates as $|\psi_{\mathrm{read}}\rangle=H^{\otimes4}|\psi\rangle$, the ideal readout expectation values are
#
# $$
# q_j
# = \langle\psi_{\mathrm{read}}|Z_j|\psi_{\mathrm{read}}\rangle
# = \langle\psi|X_j|\psi\rangle,
# \qquad j\in\mathcal{Q}.
# $$
#
# This equality follows from $HZH=X$: measuring Z after the Hadamard gates reads the X expectation values of the state before those gates. All four observables can be measured together.
#
# The implementation estimates these expectation values from $S=512$ shots, where each shot prepares the circuit state and measures all four qubits. If $b_j^{(s)}\in\{0,1\}$ is the result for qubit $j$ in shot $s$, the estimated feature is
#
# $$
# \hat q_j = \frac{1}{S}\sum_{s=1}^{S}(-1)^{b_j^{(s)}},
# \qquad S=512,\qquad \hat{\mathbf{q}}\in[-1,1]^4.
# $$
#
# The same four-bit measurement records supply all four features. Because the number of shots is finite, $\hat{\mathbf{q}}$ has statistical fluctuations around the ideal expectation values.
#
# ### Classical Fully Connected Layer
#
# The classical fully connected layer combines the features extracted by the classical convolutional layer with the output of the quantum circuit layer and uses both to compute a classification score for each class.
#
# The classifier concatenates the four CNN features with the four quantum features, preserving both representations. Its input is
#
# $$
# \mathbf{f} = [\mathbf{x};\hat{\mathbf{q}}] \in \mathbb{R}^{n+m}=\mathbb{R}^{8},
# $$
#
# so it can learn how much information to use from the classical representation and how much to use from the quantum transformation. A classical fully connected layer maps this fused vector to $K$ logits, the class scores before conversion to probabilities,
#
# $$
# \mathbf{s}=W_{\mathrm{out}}\mathbf{f}+\mathbf{b}_{\mathrm{out}}
# \in \mathbb{R}^{K}, \qquad
# \mathbf{p}=\operatorname{softmax}(\mathbf{s}).
# $$
#
# If $\mathcal{C}$ denotes the class index set, training minimizes the cross-entropy loss
#
# $$
# \mathcal{L} = -\sum_{c \in \mathcal{C}} y_c \log p_c.
# $$
#
# Here, $\mathbf{y}$ is a one-hot label, with 1 for the correct class and 0 elsewhere, and $K=4$. In the implementation, `nn.CrossEntropyLoss` receives the raw logits $\mathbf{s}$ and integer class labels directly; it includes the log-softmax calculation internally. This loss is the objective for jointly learning the CNN, quantum circuit, and classifier parameters. The classification accuracy of this example alone does not establish that the quantum layer outperforms a classical model.

# %% [markdown]
# ## Implementation with Qamomile
#
# We now implement the HQNN workflow with [PyTorch](https://pytorch.org/) and Qamomile. PyTorch describes the classical layers and updates the model parameters. Qamomile defines the quantum circuit and evaluates its measurement-derived features.
#
# ### Quantum Circuit Layer with Qamomile
#
# Connecting the quantum layer to PyTorch requires two operations. The forward pass computes quantum features from the CNN outputs and circuit weights. During training, the backward pass receives the loss gradient from the classifier and returns gradients for both the CNN outputs and quantum weights. PyTorch cannot trace the quantum simulator's internal computation, so we provide this backward calculation explicitly.
#
# #### Variational Quantum Circuit
#
# The circuit has four qubits and two variational layers, with 24 trainable rotation angles. It applies the following operations:
#
# 1. **Angle encoding:** apply an $R_y$ rotation to each qubit using one of the four CNN features as its angle.
# 2. **Variational layers:** apply $R_z$, $R_y$, and $R_z$ to each qubit, then connect neighboring qubits with a chain of `CZ` gates. Each layer has $4\times3=12$ trainable angles.
# 3. **Readout rotation:** apply a Hadamard gate to every qubit before measuring in the Z basis.
#
# The final Hadamard gates make the measurement sensitive to X-basis information. Without them, the final $R_z$ and `CZ` gates would not change the Z-basis measurement probabilities.
#
# `hqnn_state(...)` receives a quantum state, applies these operations, and returns the updated state. `hqnn_samples(...)` allocates the four qubits, calls this helper, and measures the result. Runtime `inputs` and `weights` belong to the measurement entry point. The array lengths and layer count are fixed when the kernels are defined; changing the number of layers requires redefining and recompiling them.

# %%
N_QUBITS = 4
N_LAYERS = 1 if docs_test_mode else 2
N_WEIGHTS_PER_LAYER = N_QUBITS * 3
N_WEIGHTS = N_LAYERS * N_WEIGHTS_PER_LAYER

print(f"Qubits: {N_QUBITS}, Layers: {N_LAYERS}, Trainable weights: {N_WEIGHTS}")
assert N_WEIGHTS == (12 if docs_test_mode else 24)


# %%
@qm.qkernel
def hqnn_state(
    state: qm.Q[qm.Bits[4]],  # type: ignore[valid-type]
    inputs: qm.Vector[qm.Real, 4],  # type: ignore[valid-type]
    weights: qm.Vector[qm.Real, N_WEIGHTS],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[4]]:  # type: ignore[valid-type]
    """Apply variational layers and Hadamard readout rotations.

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

    # Applying H before Z measurement reads the original state's X expectation.
    # This also makes phase information from the final RZ / CZ gates observable.
    for index in range(N_QUBITS):
        state = qm.h(state, target=index)
    return state


@qm.qkernel
def hqnn_samples(
    inputs: qm.Param[qm.Vector[qm.Real, 4]],  # type: ignore[valid-type]
    weights: qm.Param[qm.Vector[qm.Real, N_WEIGHTS]],  # type: ignore[valid-type]
) -> qm.Shot[qm.Bits[4]]:  # type: ignore[valid-type]
    """Sample Z bits after the variational circuit and H readout.

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
# #### Compile and Inspect the Circuit
#
# We specialize the measurement kernel into a Qamomile program. Logical resource analysis uses this program to count the operations before any hardware-specific compilation.
#
# For $L$ variational layers, the circuit contains $4+12L$ rotation gates, $3L$ `CZ` gates, and four final Hadamard gates. With two layers, these are 28 rotations, six `CZ` gates, and four Hadamard gates. These are logical counts, not hardware execution times or gate counts after device-specific transformations.
#
# We compile the program once for `qm.Reference()` and retain the resulting `artifact`. Subsequent forward and backward passes reuse it while binding different values to `inputs` and `weights`. Another supported execution backend requires its own compilation target and execution settings.

# %%
program = qm.specialize(hqnn_samples)
study = qm.resources.logical(program)

rotation_count = study.metric(qm.resources.LOGICAL_ROTATION_GATES).numeric_enclosure()
cz_count = study.metric(
    qm.resources.logical_operation_count("quantum.cz")
).numeric_enclosure()
print("Logical rotations:", rotation_count)
print("Logical CZ gates:", cz_count)
print("Final Hadamard readout gates: 4 (source-level count)")

assert rotation_count == (
    16 if docs_test_mode else 28,
    16 if docs_test_mode else 28,
)
assert cz_count == (
    3 if docs_test_mode else 6,
    3 if docs_test_mode else 6,
)

reference = qm.Reference()
artifact = qm.compile(program, to=reference)

# %% [markdown]
# #### Quantum Feature Forward Pass
#
# Each request supplies four input angles and the vector of quantum weights. `qm.execute_batch(...)` evaluates an ordered list of these requests using the same compiled artifact. In this notebook, each request uses 512 shots: repeated circuit executions that produce four-bit measurement outcomes.
#
# A single measurement histogram supplies all four quantum features. If $n_{\mathbf b}$ is the count for bit string $\mathbf b$ and $S$ is the number of shots, the feature for qubit $i$ is
#
# $$
# \hat q_i=\frac{1}{S}\sum_{\mathbf b}n_{\mathbf b}(1-2b_i).
# $$
#
# `result.expectation_z(i)` computes this empirical expectation value. Because the circuit applies H before measuring, it estimates the X expectation of the state before that final rotation: $HZH=X$. There is no need to execute four separate circuits to obtain the four features. The features lie in $[-1,1]$, but finite sampling introduces statistical fluctuations around the ideal expectation values.

# %%
QUANTUM_SHOTS = 1 if docs_test_mode else 512


def quantum_features_batch(
    requests: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    seed: int,
) -> torch.Tensor:
    """Evaluate sampled features from H-rotated Z measurements.

    Args:
        requests (list[tuple[torch.Tensor, torch.Tensor]]): Input and weight
            tensors for independent circuit invocations.
        seed (int): Deterministic backend seed for the first invocation.

    Returns:
        torch.Tensor: Four readout features per request (X before H).
    """

    rows = []
    max_requests = 4096  # Maximum parameter rows per execute_batch call.
    for start in range(0, len(requests), max_requests):
        parameters = [
            {
                "inputs": tuple(float(value) for value in inputs.detach()),
                "weights": tuple(float(value) for value in weights.detach()),
            }
            for inputs, weights in requests[start : start + max_requests]
        ]
        results = qm.execute_batch(
            artifact,
            on=reference,
            parameters=parameters,
            shots=QUANTUM_SHOTS,
            seed=seed + start,
        )
        for result in results:
            assert isinstance(result, qm.SampleResult)
            rows.append(tuple(result.expectation_z(index) for index in range(N_QUBITS)))
    dtype = requests[0][0].dtype if requests else torch.float32
    return torch.tensor(rows, dtype=dtype).reshape(-1, N_QUBITS)


# %% [markdown]
# #### Gradient Computation: Parameter Shift
#
# Training needs to know how changing an input angle or quantum weight changes the loss. The parameter-shift rule provides an exact derivative formula for ideal expectation values when a gate has the form $e^{-i\theta G/2}$ and the generator $G$ has eigenvalues $+1$ and $-1$ {cite:p}`10.1103/PhysRevA.98.032309,10.1103/PhysRevA.99.032331`.
#
# Each input angle and trainable weight in this circuit appears in exactly one $R_y$ or $R_z$ gate, so for quantum feature $f_i$ and parameter $\theta_k$,
#
# $$
# \frac{\partial f_i}{\partial\theta_k}
# =\frac{1}{2}\left[
# f_i\!\left(\boldsymbol\theta+\frac{\pi}{2}\mathbf e_k\right)
# -f_i\!\left(\boldsymbol\theta-\frac{\pi}{2}\mathbf e_k\right)
# \right].
# $$
#
# Here, $\mathbf e_k$ selects the parameter to shift. We evaluate the circuit with that angle increased and decreased by $\pi/2$, then take half the difference. The identity is exact for ideal expectations. Using sampled features instead gives a gradient estimate with measurement noise.
#
# The custom `QuantumFeatures` function saves the inputs and weights in `forward(...)`. Its `backward(...)` evaluates the shifted requests, combines each feature derivative with the gradient supplied by the classifier, and returns input and weight gradients. Input gradients propagate through the CNN by the chain rule. Since every image shares the same quantum weights, the weight gradients accumulate across the minibatch.
#
# There are four input angles and $W$ quantum weights, so a backward pass requires $2(4+W)$ circuit evaluations per image. With two layers, $W=24$: this is 56 evaluations per image, or 224 for a minibatch of four images. For this default minibatch, `qm.execute_batch(...)` groups all 224 requests into one API call; it does not turn them into one circuit evaluation. For larger request lists, `quantum_features_batch(...)` splits them into groups of at most 4096 requests and combines the results in the original order, preserving the seed assigned to each request. This lets the training minibatch keep its original size.

# %%
SHIFT = math.pi / 2.0


class QuantumFeatures(torch.autograd.Function):
    """Bridge sampled Qamomile features to PyTorch autograd."""

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
        """Estimate gradients from the parameter-shift rule.

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
# ### HQNN Workflow with PyTorch
#
# We wrap `QuantumFeatures` in an ordinary PyTorch `nn.Module` called `QuantumLayer`. The quantum weights are stored as an `nn.Parameter`, so PyTorch's optimizer can update them together with the CNN and classifier weights.
#
# Calling `QuantumFeatures.apply(...)` evaluates the quantum features during forward propagation. When training calls `loss.backward()`, PyTorch invokes the backward implementation above and continues propagating gradients through the classical layers. This implementation calls the quantum simulator and creates its output tensors on the CPU; running the entire model on a GPU would require changes to the data transfers.


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
# ### Construct the Complete Hybrid Model
#
# `HybridClassifier` combines the classical feature extractor, quantum layer, and final classifier. Two convolution-and-pooling blocks extract image features, and a linear layer reduces them to four values. Multiplying their sigmoid outputs by $\pi$ gives the four input rotation angles.
#
# The classifier receives both these CNN-derived features and the four quantum features. Their concatenation is an eight-dimensional vector, which the final linear layer maps to four logits: scores for the four classes. `CrossEntropyLoss` uses these logits and the correct class labels to compute the training objective.
#
# Keeping both feature vectors lets the classifier use information from the classical representation and its quantum transformation. It does not by itself show that the quantum features improve classification; that would require a controlled comparison with an appropriate classical model.


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


# %% [markdown]
# ## Results
#
# ### Training
#
# We train the model for 10 epochs using 240 images, minibatches of four images, cross-entropy loss, and the Adam optimizer with learning rate 0.001. An epoch is one pass through the training images. After each epoch, we evaluate classification accuracy on the separate set of 120 test images.
#
# Each update trains the CNN, quantum layer, and classifier together. We record the gradient norms from the final training minibatch and compare the quantum weights before and after training. Finite gradients and changed quantum weights confirm that the quantum layer participates in the update. A nonzero gradient norm does not mean that every individual weight has a nonzero gradient for every input.

# %%
EPOCHS = 1 if docs_test_mode else 10
BATCH_SIZE = 4

torch.manual_seed(42)
model = HybridClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
train_losses: list[float] = []
test_accuracies: list[float] = []
initial_quantum_weights = model.quantum_layer.weights.detach().clone()

# Record gradients from the final minibatch without an extra optimizer step.
last_gradient_norms: dict[str, float] = {}
last_batch_loss = float("nan")

print(f"=== End-to-End Training ({EPOCHS} epochs) ===")
for epoch in range(EPOCHS):
    model.train()
    permutation = torch.randperm(len(X_train))
    shuffled_images = X_train[permutation]
    shuffled_labels = y_train[permutation]
    epoch_loss = 0.0
    batches = 0

    for start in range(0, len(shuffled_images), BATCH_SIZE):
        images = shuffled_images[start : start + BATCH_SIZE]
        labels = shuffled_labels[start : start + BATCH_SIZE]
        optimizer.zero_grad()
        logits, classical_features, quantum_outputs = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        if epoch == EPOCHS - 1 and start + BATCH_SIZE >= len(shuffled_images):
            first_conv = model.feature_extractor[0]
            assert isinstance(first_conv, nn.Conv2d)
            assert first_conv.weight.grad is not None
            assert model.quantum_layer.weights.grad is not None
            assert model.classifier.weight.grad is not None
            last_gradient_norms = {
                "cnn": first_conv.weight.grad.norm().item(),
                "quantum": model.quantum_layer.weights.grad.norm().item(),
                "classifier": model.classifier.weight.grad.norm().item(),
            }
            last_batch_loss = loss.item()
            assert classical_features.shape == (len(images), 4)
            assert quantum_outputs.shape == (len(images), 4)
            assert logits.shape == (len(images), N_CLASSES)

        optimizer.step()
        epoch_loss += loss.item()
        batches += 1

    train_losses.append(epoch_loss / batches)
    model.eval()
    with torch.no_grad():
        test_logits, _, _ = model(X_test)
        test_accuracy = (test_logits.argmax(1) == y_test).float().mean().item()
        test_accuracies.append(test_accuracy)
    print(
        f"Epoch {epoch + 1}/{EPOCHS}: "
        f"loss={train_losses[-1]:.4f}, test_acc={test_accuracy:.2%}"
    )

# Check gradient flow through all layers, not a nonzero gradient for every weight.
assert last_gradient_norms
assert all(np.isfinite(value) for value in last_gradient_norms.values())
assert last_gradient_norms["quantum"] > 0.0
assert not torch.equal(
    initial_quantum_weights,
    model.quantum_layer.weights.detach(),
)

print("Last minibatch loss:", last_batch_loss)
print("Last minibatch gradient norms:", last_gradient_norms)

# %%
assert len(train_losses) == len(test_accuracies) == EPOCHS
assert all(np.isfinite(value) for value in train_losses)
assert all(0.0 <= value <= 1.0 for value in test_accuracies)

epoch_numbers = np.arange(1, EPOCHS + 1)
learning_figure, learning_axes = plt.subplots(
    1, 2, figsize=(11, 4), layout="constrained"
)
learning_axes[0].plot(epoch_numbers, train_losses, "o-", color="tab:red")
learning_axes[0].set_xlabel("Epoch")
learning_axes[0].set_ylabel("Cross-entropy loss")
learning_axes[0].set_title("Training Loss")
learning_axes[1].plot(
    epoch_numbers, 100.0 * np.asarray(test_accuracies), "s-", color="tab:blue"
)
learning_axes[1].set_xlabel("Epoch")
learning_axes[1].set_ylabel("Test accuracy (%)")
learning_axes[1].set_ylim(0.0, 100.0)
learning_axes[1].set_title("Test Accuracy")
for axis in learning_axes:
    axis.set_xticks(epoch_numbers)
    axis.grid(alpha=0.25)
plt.show()
plt.close(learning_figure)

# %% [markdown]
# ### Evaluation
#
# The recorded run classifies 114 of the 120 test images correctly, giving 95% accuracy. The per-class results are 26 of 30 T-shirt/top images, 29 of 30 Trouser images, all 30 Sandal images, and 29 of 30 Bag images. These values describe the particular dataset subset, initialization, and sampling settings used here.
#
# The learning curves show how training loss and test accuracy change across epochs. The confusion matrix provides a more detailed view: rows are true classes, columns are predicted classes, and each entry is the number of images with that combination. Diagonal entries count correct predictions, while off-diagonal entries show which classes are confused.
#
# The sample-image plot displays both the true and predicted labels so that individual decisions can be inspected. Green prediction labels indicate correct classifications; red labels indicate mistakes. These examples complement the aggregate metrics, but a few selected images are not a substitute for evaluating the full test set.
#
# The quantum features and gradients are estimated from finite samples, so changing the measurement settings or random initialization can change the learning curves and final accuracy. This small four-class experiment demonstrates a working training workflow; it does not establish an advantage over classical classifiers.

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

# %%
assert len(CLASS_NAMES) == N_CLASSES
assert predictions.shape == y_test.shape
assert len(y_test) > 0
true_labels = y_test.detach().cpu().numpy()
predicted_labels = predictions.detach().cpu().numpy()
assert np.all((0 <= true_labels) & (true_labels < N_CLASSES))
assert np.all((0 <= predicted_labels) & (predicted_labels < N_CLASSES))

confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
for expected, predicted in zip(true_labels, predicted_labels, strict=True):
    confusion[expected, predicted] += 1
assert confusion.sum() == len(y_test)
assert np.array_equal(
    confusion.sum(axis=1), np.bincount(true_labels, minlength=N_CLASSES)
)
assert np.isclose(np.trace(confusion) / len(y_test), accuracy)

confusion_figure, confusion_axis = plt.subplots(figsize=(6, 5), layout="constrained")
maximum_count = int(confusion.max())
confusion_image = confusion_axis.imshow(
    confusion, cmap="Blues", vmin=0, vmax=max(1, maximum_count)
)
colorbar = confusion_figure.colorbar(
    confusion_image, ax=confusion_axis, label="Number of images"
)
colorbar.set_ticks(np.linspace(0, maximum_count, min(maximum_count + 1, 6), dtype=int))
confusion_axis.set_xlabel("Predicted class")
confusion_axis.set_ylabel("True class")
confusion_axis.set_title("Confusion Matrix")
confusion_axis.set_xticks(range(N_CLASSES), CLASS_NAMES, rotation=45, ha="right")
confusion_axis.set_yticks(range(N_CLASSES), CLASS_NAMES)
for true_class in range(N_CLASSES):
    for predicted_class in range(N_CLASSES):
        count = int(confusion[true_class, predicted_class])
        confusion_axis.text(
            predicted_class,
            true_class,
            str(count),
            ha="center",
            va="center",
            color="white" if count > maximum_count / 2 else "black",
        )
plt.show()
plt.close(confusion_figure)

# %%
# Interleave the balanced test classes so the grid includes each class.
examples_by_class = [
    np.flatnonzero(true_labels == label)[:8] for label in range(N_CLASSES)
]
sample_indices = np.stack(examples_by_class, axis=1).ravel()[:8]
shown = len(sample_indices)

columns = min(4, shown)
rows = math.ceil(shown / columns)
sample_figure, sample_axes = plt.subplots(
    rows,
    columns,
    figsize=(3.1 * columns, 3.2 * rows),
    squeeze=False,
    layout="constrained",
)
sample_figure.suptitle("Sample Predictions (green: correct; red: incorrect)")
image_minimum = float(X_test.min())
image_maximum = float(X_test.max())
for axis in sample_axes.flat:
    axis.axis("off")
for axis, index in zip(sample_axes.flat, sample_indices):
    expected = int(y_test[index])
    predicted = int(predictions[index])
    axis.imshow(
        X_test[index, 0].detach().cpu().numpy(),
        cmap="gray",
        vmin=image_minimum,
        vmax=image_maximum,
    )
    axis.set_title(
        f"True: {CLASS_NAMES[expected]}\nPred: {CLASS_NAMES[predicted]}",
        color="tab:green" if predicted == expected else "tab:red",
        fontsize=10,
    )
plt.show()
plt.close(sample_figure)

# %% [markdown]
# ## Summary
#
# - **HQNN workflow:** a CNN reduces each image to four input angles, a four-qubit circuit produces four quantum features, and a classifier combines both representations to predict the class.
# - **Qamomile and PyTorch:** the quantum kernels define the circuit, while a custom forward/backward function connects sampled features to PyTorch's training process. The same compiled artifact is reused with different input angles and weights.
# - **Readout and gradients:** Hadamard rotations before measurement expose X expectations. The parameter-shift relation is exact for ideal expectations; finite-shot features give noisy gradient estimates.
# - **Cost and results:** the number of circuit evaluations grows with the number of trainable angles and images in a minibatch. This run reaches 95% accuracy on its 120-image test set, but judging the contribution of the quantum layer requires further comparisons.
#
# Next, [Möttönen Amplitude Encoding](mottonen_amplitude_encoding.ipynb) introduces a different way to encode classical data into a quantum state.
