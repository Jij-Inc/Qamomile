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
# # ハイブリッド量子ニューラルネットワーク (HQNN)
#
# このチュートリアルでは、古典ニューラルネットワークと変分量子回路を組み合わせた**ハイブリッド量子ニューラルネットワーク**（HQNN）をQamomileで実装する例を紹介します。Fashion-MNISTデータセットから選んだ4クラスの画像認識タスクについて、古典畳み込みニューラルネットワークと4量子ビットの量子回路を組み合わせます。量子回路の勾配をパラメータシフトルールで推定し、PyTorchへ渡すことで古典パラメータと量子パラメータを一体として学習します。

# %%
# 依存パッケージは、使用する Python 環境へ事前にインストールしてください。
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
# ## 背景: 画像認識と機械学習
#
# ### 畳み込みニューラルネットワーク (CNN)
#
# ニューラルネットワークは、入力を複数の層で順番に変換して予測を出力する機械学習モデルです。入力を$\mathbf{h}^{(0)}=\mathbf{x}$とすると、第$l$層の計算は
#
# $$
# \mathbf{h}^{(l)} = \phi^{(l)}\!\left(
#     W^{(l)}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}
# \right), \qquad l=1,\ldots,L
# $$
#
# と表せます。ここで$W^{(l)}$と$\mathbf{b}^{(l)}$は学習可能な重みとバイアス、$\phi^{(l)}$は非線形活性化関数です。分類問題では、最終層の表現から
#
# $$
# \hat{\mathbf{y}}
# = \operatorname{softmax}\!\left(
#     W_{\mathrm{out}}\mathbf{h}^{(L)} + \mathbf{b}_{\mathrm{out}}
# \right)
# $$
#
# によって各クラスの予測確率を得ます。予測と正解ラベルから計算した損失を誤差逆伝播し、各層の重みとバイアスを更新することで学習を行います。
#
# 画像認識は、入力画像をクラスへ分類する機械学習タスクです。通常の全結合層だけで画像を処理すると、画素間の空間的な関係を直接利用しにくく、多数のパラメータが必要になります。畳み込みニューラルネットワーク（CNN）は、小さなフィルタを画像全体で共有しながら適用することで、エッジやテクスチャなどの局所的なパターンを特徴量として抽出します。このとき、フィルタが一度に参照する画像内の局所領域を局所受容野と呼びます。さらに、プーリング層によって空間方向の解像度を段階的に圧縮します。こうして得られた特徴量を分類器へ入力することで、画像のクラスを識別します{cite:p}`10.1038/nature14539`。
#
# ### 量子ニューラルネットワーク (QNN)
#
# 量子ニューラルネットワーク（QNN）は、学習可能な量子演算を学習モデルの一部として用います。FarhiとNevenによる分類器では、パラメータ付き量子回路が入力状態を変換し、読み出し用量子ビットの測定から分類の出力を得ます。古典最適化アルゴリズムが、ラベル付きの訓練例を使って回路のパラメータを更新します。古典的な入力は最初に量子状態へ符号化する必要がありますが、直接与えられた量子状態を分類することもできます{cite:p}`10.48550/arXiv.1802.06002`。
#
# QNNには、このほかにもさまざまな構成があります。たとえばBeerらは、入力量子状態と目標量子状態の組を使ってフィードフォワード型量子ネットワークを学習する方法を研究しています。学習の目的関数には、2つの量子状態の一致度を表す状態忠実度を用います{cite:p}`10.1038/s41467-020-14454-2`。
#
# Congらが提案した量子畳み込みニューラルネットワーク（QCNN）は、量子状態の入力に対して畳み込みとプーリングを行います。局所的なユニタリ演算でパラメータを共有し、プーリングでは一部の量子ビットを測定して、その結果に応じて残りの量子ビットを操作します。これにより、処理に使う量子ビット数を段階的に減らします{cite:p}`10.1038/s41567-019-0648-8`。
#
# 古典的な画像を扱う関連手法には、quanvolution（画像の局所領域に量子フィルタを適用する手法）があります。小さな画像領域を量子状態へ符号化し、同じ量子回路をフィルタとして繰り返し適用して、測定値から特徴量マップを構成します。Senokosovらは、quanvolution層を含む構成など、古典処理と量子回路を組み合わせたハイブリッド画像分類器を研究しています{cite:p}`10.1088/2632-2153/ad2aef`。本チュートリアルでは、画像の畳み込みは古典CNNで行い、得られた少数の特徴量を変分量子回路で変換します。

# %% [markdown]
# ## 問題設定: Fashion-MNIST
#
# [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)は、衣料品を表す$28 \times 28$のグレースケール画像を10クラスに分類した画像認識データセットです。60,000枚の訓練画像と10,000枚のテスト画像で構成され、画像分類モデルの性能を比較するためのベンチマークとして利用されています。
#
# 本チュートリアルでは、T-shirt/top（0）、Trouser（1）、Sandal（5）、Bag（8）の4クラスを選びます。各クラス60枚の訓練画像と30枚のテスト画像を使用するため、合計で訓練画像240枚、テスト画像120枚になります。元のクラスラベルは、0から3までの連続したラベルへ対応付けます。抽出に用いる乱数のシードを固定し、同じ部分集合を再現できるようにします。
#
# 以下のコードでは、訓練用とテスト用の部分集合を抽出し、選んだ画像をテンソルへ変換して画素値を正規化します。学習には訓練用の部分集合だけを用い、テスト用の部分集合で分類器を評価します。

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
# ## アルゴリズム
#
# 量子回路を小規模に保つため、古典畳み込みブロックを前段の特徴量抽出器として用います。画像の局所的な特徴を4次元のベクトルへ圧縮し、小規模なパラメータ付き量子回路で処理します。古典的な次元削減と量子回路による特徴量変換を組み合わせたハイブリッド画像分類には、Senokosovらの先行研究があります{cite:p}`10.1088/2632-2153/ad2aef`。本チュートリアルでは構成の考え方を参考にし、回路、特徴量の結合方法、学習設定は独立した実装例として用意します。
#
# HQNNは、古典畳み込み層、量子回路層、古典全結合層の3つから構成されます。分類器は、CNN特徴量と量子特徴量の両方を受け取ります。
#
# $$
# \begin{aligned}
# \text{入力画像}
# &\longrightarrow
# \underbrace{\mathrm{CNN}\longrightarrow\pi\times\operatorname{sigmoid}}
# _{\text{古典畳み込み層}}
# \longrightarrow \text{量子回路層} \\
# &\longrightarrow
# \underbrace{
#     \text{出力の結合}(\text{CNN特徴量},\,\text{量子出力})
#     \longrightarrow \text{分類器}
# }_{\text{古典全結合層}}
# \longrightarrow \text{出力}
# \end{aligned}
# $$
#
# この処理では、古典畳み込み層が画像から$n$個の特徴量を抽出し、量子回路層がその特徴量を$m$個の期待値の推定値へ変換し、古典全結合層が$K$個のクラスに対する予測確率を出力します。ここでは$n=m=K=4$です。各構成要素は、同じ分類損失から一体として学習されます。
#
# ### 古典畳み込み層
#
# 古典畳み込み層は、入力画像から局所的なパターンを特徴量として抽出し、量子回路へ入力できる$n$次元の特徴量ベクトルへ圧縮します。
#
# 入力チャネル数を$C$、高さを$H$、幅を$W$とし、正規化された画像を$\mathbf{X} \in \mathbb{R}^{C \times H \times W}$とします。各畳み込みブロックでは、学習可能な$k_h \times k_w$フィルタ、活性化関数$\phi$、$p_h \times p_w$のプーリングを順に適用します。$\mathbf{h}^{(0)}=\mathbf{X}$とすると、$L_{\mathrm{CNN}}$個のブロックは概略的に
#
# $$
# \mathbf{h}^{(l)} = \operatorname{Pool}\!\left(
#     \phi\!\left(\mathbf{K}^{(l)} * \mathbf{h}^{(l-1)}
#     + \mathbf{b}^{(l)}\right)
# \right), \qquad l \in \mathcal{L}_{\mathrm{CNN}}
# $$
#
# と書けます。ここで$\operatorname{Pool}$は$p_h \times p_w$のウィンドウに対するプーリング演算を表し、$*$は畳み込みを表します。また、$\mathcal{L}_{\mathrm{CNN}}$は畳み込みブロックの添字集合です。得られたテンソルの要素を一列に並べ直し、$n$次元の特徴量ベクトルへ射影します。その後、
#
# $$
# \mathbf{x} = \alpha\,\sigma\!\left(
#     W_{\mathrm{CNN}}\operatorname{vec}(\mathbf{h}^{(L_{\mathrm{CNN}})})
#     + \mathbf{b}_{\mathrm{CNN}}
# \right) \in \mathbb{R}^{n}
# $$
#
# によって量子回路の回転角へ変換します。ここで$\sigma$はシグモイド関数、$\alpha=\pi$であり、$\mathbf{x}$の各成分は$(0,\pi)$の範囲に収まります。角度の範囲を制限すると、回転の周期を何度もまたぐ大きな入力を避けられます。ただし、この操作だけで学習の安定性が保証されるわけではありません。
#
# 実装では、出力チャネル数4、$3\times3$のフィルタ、ReLU活性化関数、$2\times2$の最大値プーリングからなる畳み込みブロックを2つ使います。空間方向のサイズは$28\times28$から$14\times14$、さらに$7\times7$へと小さくなります。得られた$4\times7\times7=196$個の値を全結合層で4個の特徴量へ変換し、量子回路の回転角に用います。
#
# ### 量子回路層
#
# 量子回路層は、古典畳み込み層が抽出した特徴量を量子状態へ符号化し、変分量子回路によって変換した結果を$m=4$個の期待値の推定値として出力します。
#
# $n=4$個のCNN特徴量をRY回転によって4量子ビットへ符号化します。量子ビットの添字集合を$\mathcal{Q}=\{0,1,2,3\}$とすると、符号化を行うユニタリ演算は
#
# $$
# U_{\mathrm{enc}}(\mathbf{x})
# = \prod_{i \in \mathcal{Q}} R_{Y,i}(x_i).
# $$
#
# です。続いて、$L_{\mathrm{Q}}=2$層の変分量子回路$V(\boldsymbol{\theta})$が、各量子ビットへ学習可能なRZ-RY-RZ回転を適用し、その後にCZゲートを量子ビットの組$(0,1)$、$(1,2)$、$(2,3)$へ順番に作用させます。得られる量子状態は
#
# $$
# |\psi(\mathbf{x},\boldsymbol{\theta})\rangle
# = V(\boldsymbol{\theta})U_{\mathrm{enc}}(\mathbf{x})
# |\mathbf{0}\rangle_{\mathcal{Q}}.
# $$
#
# です。各層・各量子ビットあたりの学習可能な回転パラメータ数は$d=3$なので、量子パラメータの総数は$L_{\mathrm{Q}}\times n\times d=2\times4\times3=24$です。
#
# 測定の前に、すべての量子ビットへHadamardゲートを適用します。これらのゲートを適用した後の状態を$|\psi_{\mathrm{read}}\rangle=H^{\otimes4}|\psi\rangle$とすると、読み出しの理想的な期待値は
#
# $$
# q_j
# = \langle\psi_{\mathrm{read}}|Z_j|\psi_{\mathrm{read}}\rangle
# = \langle\psi|X_j|\psi\rangle,
# \qquad j\in\mathcal{Q}.
# $$
#
# となります。$HZH=X$なので、Hadamardゲートの後でZを測定すると、ゲートの前の状態に対するX期待値を読み出せます。4つの観測量は一緒に測定できます。
#
# 実装では、$S=512$ショット（回路状態の準備と全4量子ビットの測定を繰り返す回数）から、これらの期待値を推定します。ショット$s$での量子ビット$j$の測定結果を$b_j^{(s)}\in\{0,1\}$とすると、特徴量の推定値は
#
# $$
# \hat q_j = \frac{1}{S}\sum_{s=1}^{S}(-1)^{b_j^{(s)}},
# \qquad S=512,\qquad \hat{\mathbf{q}}\in[-1,1]^4.
# $$
#
# です。同じ4ビットの測定記録から、4個すべての特徴量を求めます。ショット数が有限なので、$\hat{\mathbf{q}}$には理想的な期待値のまわりで統計的な変動が生じます。
#
# ### 古典全結合層
#
# 古典全結合層は、古典畳み込み層が抽出した特徴量と量子回路層の出力を結合し、その両方を用いて各クラスの分類スコアを計算します。
#
# 分類器は、4個のCNN特徴量と4個の量子特徴量を結合し、両方の表現を保持します。分類器への入力は
#
# $$
# \mathbf{f} = [\mathbf{x};\hat{\mathbf{q}}] \in \mathbb{R}^{n+m}=\mathbb{R}^{8}
# $$
#
# です。これにより、分類器は古典表現と量子変換後の表現をどの程度利用するかを学習できます。古典全結合層は、この結合ベクトルを各クラスに対応する$K$個の分類スコアへ写します。
#
# $$
# \mathbf{s}=W_{\mathrm{out}}\mathbf{f}+\mathbf{b}_{\mathrm{out}}
# \in \mathbb{R}^{K}, \qquad
# \mathbf{p}=\operatorname{softmax}(\mathbf{s}).
# $$
#
# クラスの添字集合を$\mathcal{C}$とします。正解クラスの成分だけが$1$で、それ以外が$0$となるラベル$\mathbf{y}$に対して、次の交差エントロピー損失を最小化します。
#
# $$
# \mathcal{L} = -\sum_{c \in \mathcal{C}} y_c \log p_c
# $$
#
# ここでは$K=4$です。実装では、`nn.CrossEntropyLoss`へロジット（確率へ変換する前の分類スコア）$\mathbf{s}$と整数のクラスラベルを直接渡します。対数ソフトマックスの計算は`nn.CrossEntropyLoss`の内部で行われます。この損失を目的関数として、CNN、量子回路、分類器のパラメータを一体として学習します。この例の分類精度だけでは、量子層が古典モデルより優れているとは判断できません。

# %% [markdown]
# ## Qamomileでの実装
#
# ここからは、[PyTorch](https://pytorch.org/)とQamomileを使ってHQNNを実装します。PyTorchで古典的なニューラルネットワークを記述し、モデルのパラメータを更新します。Qamomileでは量子回路を定義し、測定結果から得られる特徴量を計算します。
#
# ### Qamomileによる量子回路層
#
# 量子回路層をPyTorchにつなぐには、順伝播（入力から出力を計算する処理）と逆伝播（出力側から渡された勾配を入力側へ伝える処理）の両方が必要です。順伝播では、CNNの出力と量子回路の重みから量子特徴量を求めます。学習時の逆伝播では、分類器から受け取った損失の勾配を使って、CNNの出力と量子回路の重みに対する勾配を返します。PyTorchは量子シミュレーター内部の計算をそのまま追跡できないため、この逆伝播の計算を明示的に実装します。
#
# #### 変分量子回路
#
# 回路は4量子ビットと2層の変分層（学習によって回転角を調整する層）からなり、学習する回転角は24個です。次の操作を順に適用します。
#
# 1. **角度エンコーディング（特徴量を回転角として入力する方法）:** CNNが出力する4つの特徴量を、それぞれ対応する量子ビットの$R_y$回転角に使います。
# 2. **変分層:** 各量子ビットに$R_z$、$R_y$、$R_z$を適用し、その後で隣り合う量子ビットを`CZ`ゲートでつなぎます。1層あたりの学習パラメータは$4\times3=12$個です。
# 3. **測定前の回転:** 全量子ビットにHadamardゲートを適用してから、Z基底で測定します。
#
# 最後のHadamardゲートにより、測定前の状態が持つX基底の情報を読み取れます。この操作がない場合、最後の$R_z$と`CZ`ゲートはZ基底の測定確率を変えません。
#
# `hqnn_state(...)`は量子状態を受け取り、これらの操作を適用した状態を返します。`hqnn_samples(...)`は4量子ビットを確保してこの関数を呼び出し、結果を測定します。実行ごとに変える`inputs`と`weights`は、この測定用の入口に宣言します。配列の長さと層数は量子カーネルの定義時に固定されるため、層数を変える場合は定義とコンパイル（実行用の回路への変換）をやり直します。

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

    # Z 測定の直前に H を適用すると、元の状態の X 期待値を読み出せる。
    # これにより最後の RZ / CZ の位相情報も観測可能になる。
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
# #### コンパイルと回路の確認
#
# 測定用の量子カーネルからQamomileのプログラムを作ります。論理リソース解析（特定の実機に合わせる前の回路の操作数を調べる処理）では、このプログラムを使ってゲート数を求めます。
#
# 変分層が$L$層なら、回転ゲートは$4+12L$個、`CZ`ゲートは$3L$個、最後のHadamardゲートは4個です。2層の場合は、それぞれ28個、6個、4個になります。これらは論理回路の操作数であり、実機に合わせた変換後のゲート数や実行時間を直接表すものではありません。
#
# このプログラムを`qm.Reference()`向けに一度コンパイルし、生成された`artifact`（実行用に変換済みのプログラム）を保持します。以後の順伝播と逆伝播では、`inputs`と`weights`の値だけを変えて再利用します。別の対応バックエンド（回路を実行するソフトウェア）を使う場合は、その変換先と実行設定に合わせて準備します。

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
# #### 量子特徴量の順伝播
#
# 1件の実行要求には、4つの入力角度と量子回路の重みベクトルを渡します。`qm.execute_batch(...)`は、同じコンパイル済みプログラムで複数の要求を評価し、要求と同じ順序で結果を返します。このノートブックでは、1件につき512ショットを使い、4ビットの測定結果を集めます。
#
# 4つの量子特徴量は、同じ測定結果の集計から求められます。ビット列$\mathbf b$が得られた回数を$n_{\mathbf b}$、測定回数を$S$とすると、量子ビット$i$の特徴量は
#
# $$
# \hat q_i=\frac{1}{S}\sum_{\mathbf b}n_{\mathbf b}(1-2b_i)
# $$
#
# です。`result.expectation_z(i)`が、この測定結果に基づく期待値を計算します。今回は測定直前にHゲートを適用しているため、$HZH=X$の関係から、Hゲートを適用する前の状態のX期待値を推定しています。4つの特徴量を得るために別々の回路を4回実行する必要はありません。特徴量は$[-1,1]$に収まりますが、測定回数が有限なので、理想的な期待値のまわりに統計的なばらつきが生じます。

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
# #### 勾配計算: パラメータシフト
#
# 学習では、入力角度や量子回路の重みを動かしたときに、損失がどのように変わるかを調べます。パラメータシフト法は、ゲートが$e^{-i\theta G/2}$の形で、その生成子$G$（回転の軸を定める行列）の固有値が$+1$と$-1$の場合に、理想的な期待値の微分を厳密に求める関係式を与えます {cite:p}`10.1103/PhysRevA.98.032309,10.1103/PhysRevA.99.032331`。
#
# この回路では、各入力角度と学習用の重みが、それぞれ1つの$R_y$または$R_z$ゲートに一度だけ現れます。そのため、量子特徴量$f_i$とパラメータ$\theta_k$について、
#
# $$
# \frac{\partial f_i}{\partial\theta_k}
# =\frac{1}{2}\left[
# f_i\!\left(\boldsymbol\theta+\frac{\pi}{2}\mathbf e_k\right)
# -f_i\!\left(\boldsymbol\theta-\frac{\pi}{2}\mathbf e_k\right)
# \right]
# $$
#
# が成り立ちます。$\mathbf e_k$は、ずらすパラメータを指定するベクトルです。その角度を$\pi/2$だけ増やした回路と減らした回路を評価し、結果の差を半分にします。この関係式は理想的な期待値に対して厳密ですが、測定から求めた特徴量を使う場合、得られる勾配にも測定のばらつきが含まれます。
#
# 独自の`QuantumFeatures`は、`forward(...)`で入力と重みを保存します。`backward(...)`では角度をずらした要求を評価し、各特徴量の微分と分類器から渡された勾配を組み合わせて、入力と重みに対する勾配を返します。入力の勾配は、連鎖律によってCNNにも伝わります。量子回路の重みは全画像で共有されるため、ミニバッチ内の各画像から得た重みの勾配を足し合わせます。
#
# 入力角度が4個、量子回路の重みが$W$個なら、逆伝播には画像1枚あたり$2(4+W)$回の回路評価が必要です。2層では$W=24$なので1枚あたり56回、4枚のミニバッチでは224回になります。この標準設定では、`qm.execute_batch(...)`が224件を1回のAPI呼び出しにまとめますが、回路の評価そのものが1回になるわけではありません。要求が多い場合、`quantum_features_batch(...)`は1回あたり4096件以下に分けて実行し、元の順序で結果をつなぎます。各要求に割り当てる乱数シードも保つため、学習のミニバッチ自体を小さくする必要はありません。

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
# ### PyTorchによるHQNNワークフロー
#
# `QuantumFeatures`を、PyTorchの通常の`nn.Module`である`QuantumLayer`で包みます。量子回路の重みは`nn.Parameter`として保持するため、CNNや分類器の重みと同じ最適化手法で更新できます。
#
# 順伝播で`QuantumFeatures.apply(...)`を呼ぶと、量子特徴量を計算します。学習時に`loss.backward()`を呼ぶと、PyTorchは先ほど実装した逆伝播を実行し、古典的な層へも勾配を伝えます。この実装は量子シミュレーターを呼び出し、出力のテンソルをCPU上で作ります。モデル全体をGPUで実行する場合は、データの転送処理を変更する必要があります。


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
# ### ハイブリッドモデル全体の構築
#
# `HybridClassifier`で、古典的な特徴抽出器、量子回路層、最後の分類器を組み合わせます。畳み込みとプーリングを2回適用し、線形層で特徴を4つの値に減らします。その値にシグモイド関数を適用して$\pi$倍し、量子回路に渡す4つの回転角を作ります。
#
# 分類器には、このCNN由来の4つの特徴量と、量子回路から得た4つの特徴量の両方を渡します。連結した8次元ベクトルを最後の線形層に入力し、4クラスに対応するロジット（各クラスに対する正規化前のスコア）を出力します。`CrossEntropyLoss`は、このスコアと正解のクラスを使って学習の目標となる損失を計算します。
#
# 両方の特徴量を残すことで、分類器は古典的な表現と、それを量子回路で変換した表現を利用できます。ただし、この構成だけで量子特徴量が分類性能を改善すると示せるわけではありません。それを調べるには、適切な古典モデルと条件をそろえて比較する必要があります。


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
# ## 結果
#
# ### 学習
#
# 学習には240枚の画像を使い、4枚ずつのミニバッチで10エポック実行します。損失には交差エントロピー、更新には学習率0.001のAdamを使います。各エポックの終了後に、学習用とは別の120枚の評価画像で分類精度を求めます。
#
# 1回の更新で、CNN、量子回路層、分類器をまとめて学習します。最後の学習ミニバッチで得た勾配のノルムを記録し、量子回路の重みが学習前後で変わったことも確認します。勾配が有限で、重みが実際に変わっていれば、量子回路層も更新に参加していると分かります。ただし、勾配全体の大きさが0でないことは、全ての重みがどの入力に対しても0でない勾配を持つことを意味しません。

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

# 最終ミニバッチの勾配を記録する。学習後に余分な optimizer.step() は行わない。
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

# すべての層に勾配が渡る経路を確認する。個々の重みの非ゼロ勾配は保証しない。
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
# ### 評価
#
# 保存された実行結果では、評価画像120枚のうち114枚を正しく分類し、精度は95%になりました。クラスごとの正解数は、T-shirt/topが30枚中26枚、Trouserが30枚中29枚、Sandalが30枚全て、Bagが30枚中29枚です。これらは、ここで使った画像の選び方、初期値、測定設定に対する結果です。
#
# 学習曲線では、エポックごとの学習損失と評価精度の変化を確認します。混同行列では、行が正解クラス、列が予測クラスを表します。対角成分は正しく分類した画像の件数で、それ以外の成分から、どのクラス同士を取り違えたかが分かります。
#
# サンプル画像の図には正解ラベルと予測ラベルの両方を表示し、個々の予測を確認できるようにします。予測が正しければ緑、誤っていれば赤で表示します。これらの例は全体の指標を理解する助けになりますが、少数の画像を見るだけで評価データ全体の性能を判断することはできません。
#
# 量子特徴量と勾配は有限回の測定から推定しているため、測定設定や初期値を変えると、学習曲線や最終精度も変わることがあります。この小規模な4クラスの実験は、学習の一連の処理が動くことを示す例であり、古典的な分類器に対する優位性を示すものではありません。

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
# ## まとめ
#
# - **HQNNの流れ:** CNNで画像を4つの入力角度に減らし、4量子ビットの回路から4つの量子特徴量を求めます。分類器は両方の表現を組み合わせてクラスを予測します。
# - **QamomileとPyTorchの連携:** 量子カーネルで回路を定義し、独自の順伝播・逆伝播処理を通じて、測定から得た特徴量をPyTorchの学習につなぎます。入力角度と重みを変えながら、同じコンパイル済みプログラムを再利用します。
# - **読出しと勾配:** 測定前のHadamardゲートによってX期待値を読み取ります。パラメータシフトの関係式は理想的な期待値に対して厳密ですが、有限回の測定を使う勾配の推定にはばらつきがあります。
# - **計算量と結果:** 必要な回路評価の回数は、学習する角度の数とミニバッチ内の画像数に応じて増えます。今回の120枚の評価データでは95%の精度になりましたが、量子回路層が性能にどの程度寄与したかを判断するには、さらに比較が必要です。
