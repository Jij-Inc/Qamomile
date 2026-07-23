# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: qamomile
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
# このチュートリアルでは、古典ニューラルネットワークと変分量子回路を組み合わせた **ハイブリッド量子ニューラルネットワーク** (HQNN) をQamomileで実装する例を紹介します。Fashion-MNISTデータセットによる画像認識タスクについて、4量子ビットの量子回路をパラメータシフトルールによる勾配計算を利用して学習し、学習されたHQNNがデータセットの画像を分類できることを示します。

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install "qamomile[qiskit,visualization]" torch torchvision

# %%
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from torchvision import datasets, transforms

import qamomile.circuit as qmc
import qamomile.observable as qmo
from qamomile.circuit.algorithm import cz_entangling_layer, ry_layer, rz_layer
from qamomile.qiskit import QiskitTranspiler, hamiltonian_to_sparse_pauli_op

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

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
# 画像認識は、入力画像をクラスへ分類する機械学習タスクです。通常の全結合層だけで画像を処理すると、画素間の空間的な関係を直接利用しにくく、多数のパラメータが必要になります。畳み込みニューラルネットワーク (CNN) は、小さなフィルタを画像全体で共有しながら適用することで、エッジやテクスチャなどの局所的なパターンを特徴量として抽出します。このとき、フィルタが一度に参照する画像内の局所領域を局所受容野と呼びます。さらに、プーリング層によって空間方向の解像度を段階的に圧縮します。こうして得られた特徴量を分類器へ入力することで、画像のクラスを識別します{cite:p}`10.1038/nature14539`。
#
# ### 量子ニューラルネットワーク(QNN)
#
# 量子ニューラルネットワーク (QNN) は、入力された量子状態にパラメータ依存のユニタリ変換を順番に作用させる教師あり学習モデルとして捉えられます。二値分類では、読み出し用量子ビット上の Pauli オブザーバブルを測定し、その出力を予測値とします。古典最適化アルゴリズムは、この予測値が訓練ラベルへ近づくようにユニタリ変換のパラメータを更新します。古典サンプルは最初に量子状態へ符号化する必要がありますが、同じ枠組みで量子入力状態に直接付与されたラベルを学習することもできます{cite:p}`10.48550/arXiv.1802.06002,10.1038/s41467-020-14454-2`。
#
# 量子畳み込みニューラルネットワーク (QCNN){cite:p}`10.1038/s41567-019-0648-8` は、CNNの局所受容野と重み共有の考え方を量子回路へ取り入れた画像向けのQNNです。小さな画像領域を量子状態へ符号化し、同じパラメータ付き量子回路を量子フィルタとして繰り返し適用して、測定された期待値から特徴量マップを構成します。こうした量子畳み込み層を古典層と組み合わせることで、現在の量子ハードウェアでも扱えるハイブリッドな画像分類器を構成できます{cite:p}`10.1088/2632-2153/ad2aef`。

# %% [markdown]
# ## 問題設定: Fashion-MNIST
#
# [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)は、衣料品を表す$28 \times 28$のグレースケール画像を10クラスに分類した画像認識データセットです。60,000枚の訓練画像と10,000枚のテスト画像で構成され、画像分類モデルの性能を比較するためのベンチマークとして利用されています。
#
# 本チュートリアルでは、視覚的に異なる4クラス（T-shirt、Trouser、Sandal、Bag）を選択し、各クラス60枚の訓練データ、30枚のテストデータを使用します。

# %%
N_CLASSES = 4
SELECTED_CLASSES = [0, 1, 5, 8]  # T-shirt, Trouser, Sandal, Bag
CLASS_NAMES = ["T-shirt", "Trouser", "Sandal", "Bag"]
N_TRAIN_PER_CLASS = 2 if docs_test_mode else 60
N_TEST_PER_CLASS = 2 if docs_test_mode else 30

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ]
)

# 永続キャッシュディレクトリを使用（再ダウンロードを回避）
_data_root = os.path.join(os.path.expanduser("~"), ".cache", "fashion_mnist")
full_train = datasets.FashionMNIST(
    root=_data_root, train=True, download=True, transform=transform
)
full_test = datasets.FashionMNIST(
    root=_data_root, train=False, download=True, transform=transform
)


def subset_dataset(dataset, classes, n_per_class):
    """指定クラスからランダムにサブセットを抽出する。"""
    images, labels = [], []
    rng_data = np.random.default_rng(0)
    # dataset.targets でインデックスを取得（全画像のロード・変換を回避）
    targets = dataset.targets
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    else:
        targets = np.array(targets)
    for new_label, orig_label in enumerate(classes):
        class_indices = np.where(targets == orig_label)[0]
        chosen = rng_data.choice(class_indices, size=n_per_class, replace=False)
        for idx in chosen:
            img, _ = dataset[int(idx)]
            images.append(img)
            labels.append(new_label)
    return torch.stack(images), torch.tensor(labels)


X_train, y_train = subset_dataset(full_train, SELECTED_CLASSES, N_TRAIN_PER_CLASS)
X_test, y_test = subset_dataset(full_test, SELECTED_CLASSES, N_TEST_PER_CLASS)

print(f"訓練データ: {X_train.shape}, テストデータ: {X_test.shape}")
# 4 クラス × N_PER_CLASS サンプル; Fashion-MNIST は 1x28x28 グレースケール。
assert X_train.shape == (N_CLASSES * N_TRAIN_PER_CLASS, 1, 28, 28)
assert X_test.shape == (N_CLASSES * N_TEST_PER_CLASS, 1, 28, 28)
print(f"クラス: {CLASS_NAMES}")


# %% [markdown]
# ## アルゴリズム: ハイブリッド量子古典畳み込みニューラルネットワーク
#
# 高次元の画像をそのまま量子状態へ符号化し、十分な表現力を持つQCNNを実行するには、多数の量子ビットと深い回路が必要になります。しかし、NISQデバイスでは大規模な量子回路の実行が難しいため、画像の次元を量子回路へ入力できる大きさまで古典的に縮約する必要があります。そこで、古典畳み込みブロックを前段の特徴量抽出器として用い、画像の局所的な特徴量を少数の値へ圧縮した後、小規模なパラメータ付き量子回路で処理する **ハイブリッド量子ニューラルネットワーク(HQNN)** の処理の流れを採用します。この構成は、NISQ時代の画像分類において、古典CNNによる次元削減と量子層による特徴量変換を組み合わせる考え方に基づいています{cite:p}`10.1088/2632-2153/ad2aef`。
#
# 本チュートリアルでは、HQNNの処理の流れにいくつかの工夫を施したハイブリッドアルゴリズムを構築します。
#
# $$
# \begin{aligned}
# \text{入力画像}
# &\longrightarrow
# \underbrace{\mathrm{CNN}\longrightarrow\text{Sigmoidフィルタ}}
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
# この処理では、古典畳み込み層が画像から$n$個の特徴量を抽出し、量子回路層がその特徴量を$m$個の期待値へ変換し、古典全結合層が$K$個のクラスに対する予測を出力します。各構成要素は、同じ分類損失から一体として学習されます。
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
# と書けます。ここで$*$は畳み込みを表し、$\mathcal{L}_{\mathrm{CNN}}$は畳み込みブロックの添字集合です。得られたテンソルの要素を一列に並べ直し、$n$次元の特徴量ベクトルへ射影します。その後、
#
# $$
# \mathbf{x} = \alpha\,\sigma\!\left(
#     W_{\mathrm{CNN}}\operatorname{vec}(\mathbf{h}^{(L_{\mathrm{CNN}})})
#     + \mathbf{b}_{\mathrm{CNN}}
# \right) \in \mathbb{R}^{n}
# $$
#
# によって量子回路の回転角へ変換します。ここで$\sigma$は有界な活性化関数、$\alpha$は角度の尺度です。$\mathbf{x}$の各成分を有界にすることで、過度に大きい回転角や、回転角の周期性によって同じ状態を表す角度へ戻る現象を抑えます。
#
# ### 量子回路層
#
# 量子回路層は、古典畳み込み層が抽出した特徴量を量子状態へ符号化し、変分量子回路によって変換した結果を$m$個の期待値として出力します。
#
# $n$個のCNN特徴量をRY回転によって$n$量子ビットへ符号化します。量子ビットの添字集合を$\mathcal{Q}$とすると、符号化を行うユニタリ演算は
#
# $$
# U_{\mathrm{enc}}(\mathbf{x})
# = \prod_{i \in \mathcal{Q}} R_{Y,i}(x_i)
# $$
#
# です。続いて、$L_{\mathrm{Q}}$層の変分量子回路$V(\boldsymbol{\theta})$が、各量子ビットへ学習可能なRZ-RY-RZ回転を適用し、その後にCZゲートを隣り合う量子ビットへ順番に作用させます。得られる量子状態は
#
# $$
# |\psi(\mathbf{x},\boldsymbol{\theta})\rangle
# = V(\boldsymbol{\theta})U_{\mathrm{enc}}(\mathbf{x})
# |\mathbf{0}\rangle_{\mathcal{Q}}
# $$
#
# です。各層・各量子ビットあたりの学習可能な回転パラメータ数を$d$とすると、この回路ではRZ-RY-RZ回転の3つに対応するため$d=3$です。量子パラメータの総数は$L_{\mathrm{Q}} \times n \times d$です。観測量の添字集合を$\mathcal{O}$とし、量子回路層の出力にはPauli-$Z$期待値のベクトル
#
# $$
# q_j = \langle\psi(\mathbf{x},\boldsymbol{\theta})|Z_j|
# \psi(\mathbf{x},\boldsymbol{\theta})\rangle,
# \qquad j \in \mathcal{O},\quad \mathbf{q} \in \mathbb{R}^{m}
# $$
#
# を使用します。$\mathcal{O}$に含まれる$Z_j$は互いに可換なので、これらの期待値は一括して評価できます。
#
# ### 古典全結合層
#
# 古典全結合層は、古典畳み込み層が抽出した特徴量と量子回路層の出力を結合し、その両方を用いて各クラスの分類スコアを計算します。
#
# 本チュートリアルの分類器は、量子出力だけを分類に用いるのではなく、古典CNN特徴量を特徴量レベル融合によって意図的に残すよう工夫しています。分類器への入力は
#
# $$
# \mathbf{f} = [\mathbf{x};\mathbf{q}] \in \mathbb{R}^{n+m}
# $$
#
# です。これにより、分類器は古典表現と量子変換後の表現をどの程度利用するかを学習できます。古典全結合層は、この結合ベクトルを各クラスに対応する$K$個の分類スコアへ写します:
#
# $$
# \mathbf{s}=W_{\mathrm{out}}\mathbf{f}+\mathbf{b}_{\mathrm{out}}
# \in \mathbb{R}^{K}, \qquad
# \mathbf{p}=\operatorname{softmax}(\mathbf{s}).
# $$
#
# クラスの添字集合を$\mathcal{C}$とします。正解クラスの成分だけが$1$で、それ以外が$0$となるラベル$\mathbf{y}$に対して、交差エントロピー損失$\mathcal{L}=-\sum_{c\in\mathcal{C}} y_c\log p_c$を最小化します。この損失を目的関数として、古典パラメータと量子パラメータを協調的に学習します。

# %% [markdown]
# ## Qamomileでの実装
#
# それでは、上で見たHQNNの処理の流れをQamomileを用いて実装してみましょう。ニューラルネットワークを用いた機械学習の実装には、[PyTorch](https://pytorch.org/)という強力なライブラリを利用することができます。ここでは、HQNNの処理全体をPyTorchで記述し、量子回路層の処理をQamomileで実装していきます。
#
# ### Qamomileによる量子回路層
#
# PyTorchの計算グラフに量子回路層を組み込むには、順伝播を担う`forward`と逆伝播を担う`backward`の両方をPyTorchから利用できるようにする必要があります。順伝播の`forward`は学習時と推論時のどちらでも呼び出され、CNN特徴量と量子回路の重みから期待値を計算します。一方、逆伝播の`backward`は学習時に損失に対して`backward()`が呼び出されたときに必要となり、後段の層から渡された勾配を使って、CNN特徴量と量子回路の重みに関する勾配を返します。
#
# 量子回路の実行環境で行われる計算を、PyTorchの自動微分機能`autograd`が直接追跡することはできません。そのため、量子回路を実行して期待値を返す処理と、その期待値の勾配を計算する処理をそれぞれ用意します。以下では、まず「変分量子回路」でQamomileによる回路定義と順伝播に必要な回路実行を構成し、続く「勾配計算: パラメータシフト」でパラメータシフトルールによる微分とPyTorchの`backward`への接続を実装します。


# %%
N_QUBITS = 4
N_LAYERS = 2
N_WEIGHTS_PER_LAYER = N_QUBITS * 3  # 各量子ビットに RZ, RY, RZ
N_WEIGHTS = N_LAYERS * N_WEIGHTS_PER_LAYER

print(f"量子ビット数: {N_QUBITS}, 層数: {N_LAYERS}, 学習パラメータ数: {N_WEIGHTS}")
assert N_QUBITS == 4
assert N_LAYERS == 2
# N_WEIGHTS = N_LAYERS * N_QUBITS * 3(1層あたり量子ビットごとに RZ + RY + RZ)。
assert N_WEIGHTS == 24


# %% [markdown]
# #### 変分量子回路
#
# Qamomile の `@qkernel` デコレータを使ってパラメータ付き量子回路を定義します。回路は以下の構成です：
#
# 1. **回転角による符号化**: 各入力特徴量を対応する量子ビットのRY回転として埋め込みます。
# 2. **変分層**: Qamomileの標準ライブラリに含まれる層関数を使い、各層でRZ-RY-RZ回転を適用した後、CZゲートを直線状に並べて量子もつれを生成します。
#
# 回路は与えられた観測量の期待値$\langle H \rangle$を計算します。


# %%
@qmc.qkernel
def variational_ansatz(
    n_qubits: qmc.UInt,
    n_layers: qmc.UInt,
    inputs: qmc.Vector[qmc.Float],
    weights: qmc.Vector[qmc.Float],
    hamiltonian: qmc.Observable,
) -> qmc.Float:
    q = qmc.qubit_array(n_qubits, name="q")

    # 回転角による符号化: 古典特徴量をRY回転で埋め込む
    for i in qmc.range(n_qubits):
        q[i] = qmc.ry(q[i], inputs[i])

    # Qamomileの標準ライブラリに含まれる層関数を使った変分層
    for layer_idx in qmc.range(n_layers):
        base = layer_idx * n_qubits * 3
        q = rz_layer(q, weights, base)
        q = ry_layer(q, weights, base + n_qubits)
        q = rz_layer(q, weights, base + n_qubits * 2)
        q = cz_entangling_layer(q)

    return qmc.expval(q, hamiltonian)


# %% [markdown]
# 回路構造を可視化してみましょう。`inline=False`を指定すると、標準ライブラリの各層関数がひとまとまりの箱として表示されます：

# %%
variational_ansatz.draw(
    n_qubits=N_QUBITS,
    n_layers=N_LAYERS,
    hamiltonian=qmo.Z(0),
    fold_loops=False,
    inline=False,
)

# %% [markdown]
# 各量子ビットについて、$Z_i$の期待値$\langle Z_i \rangle$を測定するための観測量を定義します。
# すべての$Z_i$は互いに可換なので、Aerでは回路を一度実行するだけでまとめて評価できます。
# Qiskitの期待値計算器は、回路と同じ量子ビット数を持つ観測量を要求します。
# そこで、`Hamiltonian(num_qubits=...)`を使って、観測量が作用しない量子ビットを補います。
# 次に、パラメータ付き変分量子回路を実行可能な形式へ一度だけ変換し、観測量も
# Qiskitの表現へ変換します。最後に、回路、観測量、パラメータを1つの実行単位
# （PUB）にまとめてAerへ渡します。`estimate_resources()`を使うと、変換後の回路を
# 直接参照せずにゲート数を確認できます。

# %%
observables = []
for i in range(N_QUBITS):
    obs = qmo.Hamiltonian(num_qubits=N_QUBITS)
    obs.add_term((qmo.PauliOperator(qmo.Pauli.Z, i),), 1.0)
    observables.append(obs)

transpiler = QiskitTranspiler()

executable = transpiler.transpile(
    variational_ansatz,
    bindings={
        "n_qubits": N_QUBITS,
        "n_layers": N_LAYERS,
        "hamiltonian": observables[0],
    },
    parameters=["inputs", "weights"],
)
qiskit_circuit = executable.get_first_circuit()
assert qiskit_circuit is not None
parameter_metadata = executable.compiled_quantum[0].parameter_metadata
qiskit_observables = [
    hamiltonian_to_sparse_pauli_op(observable) for observable in observables
]
aer_estimator = AerEstimator(
    options={"backend_options": {"method": "statevector"}},
)

est = variational_ansatz.estimate_resources(
    inputs={"n_qubits": N_QUBITS, "n_layers": N_LAYERS},
)
print(est)
assert est.qubits == 4
# 入力RY 4個 + 2層 * (RZ 4個 + RY 4個 + RZ 4個) = 単一量子ビット回転28個。
assert est.gates.single_qubit == 28
# 2層 * 3個のCZ（4量子ビットを直線状に接続）= 二量子ビットのクリフォードゲート6個。
assert est.gates.two_qubit == 6
assert est.gates.total == 34
assert est.gates.rotation_gates == 28

# %% [markdown]
# 順伝播では、具体的な値をQamomileが生成したパラメータへ対応付け、回路、パラメータ、
# および可換なすべての$Z_i$を1つの実行単位としてAerへ渡します。Aerでは、一度の
# 回路シミュレーションからすべての期待値を計算できるため、量子ビットごとに個別に
# 回路を実行する必要がありません。


# %%
def quantum_forward(input_vals: np.ndarray, weight_vals: np.ndarray) -> np.ndarray:
    """各量子ビットの <Z_i> を評価する。

    Args:
        input_vals: 特徴量の値。配列の形は (N_QUBITS,)。
        weight_vals: 学習可能な重み。配列の形は (N_WEIGHTS,)。

    Returns:
        期待値。配列の形は (N_QUBITS,)。
    """
    indexed_bindings = {
        **{f"inputs[{i}]": float(value) for i, value in enumerate(input_vals)},
        **{f"weights[{i}]": float(value) for i, value in enumerate(weight_vals)},
    }
    qiskit_bindings = parameter_metadata.to_binding_dict(indexed_bindings)
    result = aer_estimator.run(
        [
            (qiskit_circuit, qiskit_observables, qiskit_bindings),
        ]
    ).result()
    return np.asarray(result[0].data.evs, dtype=float)


# %%
# 動作テスト: 乱数で生成した入力と重み
rng = np.random.default_rng(42)
test_inputs = rng.uniform(-np.pi, np.pi, N_QUBITS)
test_weights = rng.uniform(-np.pi, np.pi, N_WEIGHTS)

expvals = quantum_forward(test_inputs, test_weights)
print("期待値:", expvals)
# Z の期待値は [-1, 1]、量子ビット 1 個に 1 値。
assert expvals.shape == (N_QUBITS,)
assert all(-1.0 <= float(e) <= 1.0 for e in expvals)

# %% [markdown]
# #### 勾配計算: パラメータシフト
#
# ニューラルネットワークの学習では、損失関数の勾配に沿ってパラメータを更新する勾配降下法が広く使われています。量子回路層を含む場合も同様に、誤差逆伝播によって損失関数の勾配を計算するには、量子回路の各パラメータに関するハミルトニアン$H$の期待値$\langle H \rangle$の勾配が必要です。
#
# 量子デバイスでは、期待値を有限回の測定結果から推定します。そのため学習には、推定値の平均が真の勾配と一致する勾配の不偏推定量が必要になります。生成子$G$の固有値が$\pm 1$である$e^{-i\theta G/2}$形式のゲートに対して、**パラメータシフトルール**{cite:p}`10.1103/PhysRevA.98.032309,10.1103/PhysRevA.99.032331` は、パラメータを正負にずらした2つの期待値から勾配を求める厳密な関係式を与えます：
#
# $$
# \frac{\partial}{\partial \theta_k} \langle H \rangle
# = \frac{1}{2} \Big[
#     \langle H \rangle\big|_{\theta_k + \pi/2}
#   - \langle H \rangle\big|_{\theta_k - \pi/2}
# \Big]
# $$
#
# それぞれの期待値を不偏に推定し、その差を取ることで、期待値の勾配に対する不偏推定量が得られます。本チュートリアルでは$H=Z_i$として各量子ビットの期待値を計算し、この規則を独自の`torch.autograd.Function`として実装することで、PyTorchが量子層を通じて誤差逆伝播できるようにします。不要な配列の複製を避けるため、パラメータをずらす際は元の配列を一時的に書き換え、計算後に元の値へ戻します。

# %%
SHIFT = math.pi / 2


class QuantumFunction(torch.autograd.Function):
    """PyTorchと量子回路を橋渡しする独自の自動微分関数。"""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(inputs, weights)
        # ひとまとまりの入力データ（バッチ）に含まれる各標本の回路を評価する
        batch_results = []
        weights_np = weights.detach().cpu().numpy()
        for inp in inputs:
            expvals = quantum_forward(inp.detach().cpu().numpy(), weights_np)
            batch_results.append(expvals)
        return torch.tensor(
            np.array(batch_results), dtype=inputs.dtype, device=inputs.device
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, weights = ctx.saved_tensors
        weights_np = weights.detach().cpu().numpy().copy()

        grad_inputs = torch.zeros_like(inputs)
        grad_weights = torch.zeros(
            weights.shape[0], dtype=weights.dtype, device=weights.device
        )

        for b, inp in enumerate(inputs):
            inp_np = inp.detach().cpu().numpy().copy()
            g_out = grad_output[b].detach().cpu().numpy()  # 配列の形: (N_QUBITS,)

            # 重みに関する勾配（元の配列を一時的に書き換えて複製を避ける）
            for k in range(len(weights_np)):
                weights_np[k] += SHIFT
                fwd_plus = quantum_forward(inp_np, weights_np)
                weights_np[k] -= 2 * SHIFT
                fwd_minus = quantum_forward(inp_np, weights_np)
                weights_np[k] += SHIFT  # 元の値に復元

                param_grad = (fwd_plus - fwd_minus) / 2.0  # 配列の形: (N_QUBITS,)
                grad_weights[k] += np.dot(g_out, param_grad)

            # 入力に関する勾配（元の配列を一時的に書き換えて複製を避ける）
            for k in range(len(inp_np)):
                inp_np[k] += SHIFT
                fwd_plus = quantum_forward(inp_np, weights_np)
                inp_np[k] -= 2 * SHIFT
                fwd_minus = quantum_forward(inp_np, weights_np)
                inp_np[k] += SHIFT  # 元の値に復元

                input_grad = (fwd_plus - fwd_minus) / 2.0
                grad_inputs[b, k] = np.dot(g_out, input_grad)

        return grad_inputs, grad_weights


# %% [markdown]
# ### PyTorchによるHQNNワークフロー
#
# これで、量子回路層の処理をPyTorchの計算の流れに組み込むための準備が整いました。
#
# 量子回路の順伝播と逆伝播を定義した`QuantumFunction`を、標準的な`nn.Module`として扱える`QLayer`に組み込みます。`QLayer.forward`が`QuantumFunction.apply`を呼び出すことで、通常の順伝播では量子回路の期待値が計算され、学習時の誤差逆伝播では対応する`backward`が自動的に呼び出されます。


# %%
class QLayer(nn.Module):
    """Qamomileの変分量子回路をPyTorchから利用するためのモジュール。"""

    def __init__(self, n_weights: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_weights) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return QuantumFunction.apply(x, self.weights)


# %% [markdown]
# 最後に、HQNNの処理全体をPyTorchで記述します。入力画像からCNNで特徴量を抽出し、次元を削減した特徴量に対して量子回路層を実行します。その後、量子回路の出力を分類器へ渡し、画像のクラスを識別します。基本的な処理は以上ですが、ここでは学習精度を高めるために、次の処理を追加しています。
#
# - **sigmoid関数フィルタ**: $\pi \cdot \sigma(\cdot)$によってCNNの出力を$(0, \pi)$へ写し、量子回路へ入力する回転角を有界にします。これにより、回転角の周期性によって同じ状態を表す角度へ戻る現象を抑えます。
# - **特徴量レベル融合分類器**: CNN特徴量と量子回路の出力を結合し、古典表現と量子変換後の表現の両方を分類器へ入力します。このように分類前の特徴量を連結する方法は、特徴量レベル融合または特徴量連結と呼ばれます。


# %%
class EndToEndHybridHQNN(nn.Module):
    """
    モデル全体を最初から一貫して学習するハイブリッドモデル。
    画像 -> CNN特徴量 -> 量子層 -> 分類器

    分類器には [古典特徴量, 量子出力] を両方入れる特徴量レベル融合構造。
    CNN と量子層が最初から協調して学習する。
    """

    def __init__(self, n_qubits: int, n_weights: int, n_classes: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            # 畳み込みブロック1: 局所的な特徴量を抽出し、空間サイズを縮小する。
            nn.Conv2d(1, 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 畳み込みブロック2: 特徴量をさらに変換し、空間サイズを縮小する。
            nn.Conv2d(4, 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 特徴量削減: 特徴量マップを量子ビット数と同じ次元へ射影する。
            nn.Flatten(),
            nn.Linear(7 * 7 * 4, n_qubits),
        )
        # 量子回路層: 削減された特徴量を期待値へ変換する。
        self.qlayer = QLayer(n_weights)
        # 特徴量レベル融合分類器: CNN特徴量と量子出力からクラスを予測する。
        self.classifier = nn.Linear(n_qubits * 2, n_classes)

    def forward(self, x: torch.Tensor):
        # CNN特徴量を抽出し、sigmoid関数フィルタで量子回路の入力角度へ写す。
        feats = math.pi * torch.sigmoid(
            self.feature_extractor(x)
        )  # 配列の形: (B, N_QUBITS)
        # 量子回路層を実行して、各量子ビットの期待値を得る。
        q_out = self.qlayer(feats)  # 配列の形: (B, N_QUBITS)

        # CNN特徴量と量子出力を連結して、特徴量レベル融合を行う。
        fused = torch.cat([feats, q_out], dim=1)  # 配列の形: (B, 2*N_QUBITS)
        # 融合した特徴量から各クラスの分類スコアを計算する。
        logits = self.classifier(fused)
        return logits, feats, q_out


# %% [markdown]
# ## 結果
#
# ここでは、構築したHQNNをFashion-MNISTの訓練データで学習し、テストデータを用いて分類性能を評価します。学習中の損失とテスト精度を記録するとともに、学習後の勾配、クラスごとの精度、混同行列、および実際の予測例を確認します。
#
# ### 学習
#
# 交差エントロピー損失と Adam optimizer を使い、ハイブリッドモデルを10エポック学習します。各更新では、古典ネットワークと量子回路の両方を微分します。学習後には、最初の CNN 畳み込み層と学習可能な量子重みへ勾配が到達していることを明示的に確認します。

# %%
EPOCHS = 2 if docs_test_mode else 10
BATCH_SIZE = 4

torch.manual_seed(42)
hybrid_model = EndToEndHybridHQNN(N_QUBITS, N_WEIGHTS, N_CLASSES)

optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_losses = []
test_accs = []

print(f"=== End-to-End 学習 ({EPOCHS} エポック) ===")
for epoch in range(EPOCHS):
    perm = torch.randperm(len(X_train))
    X_shuf, y_shuf = X_train[perm], y_train[perm]

    epoch_loss = 0.0
    total = 0

    for i in range(0, len(X_shuf), BATCH_SIZE):
        xb = X_shuf[i : i + BATCH_SIZE]
        yb = y_shuf[i : i + BATCH_SIZE]

        optimizer.zero_grad()
        logits, feats, q_out = hybrid_model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        total += len(yb)

    avg_loss = epoch_loss / max(1, math.ceil(total / BATCH_SIZE))
    train_losses.append(avg_loss)

    with torch.no_grad():
        test_logits, _, _ = hybrid_model(X_test)
        test_acc = (test_logits.argmax(1) == y_test).float().mean().item()
        test_accs.append(test_acc)

    print(
        f"  エポック {epoch + 1}/{EPOCHS}  損失={avg_loss:.4f}  テスト精度={test_acc:.2%}"
    )

# データセットとモデルの乱数シードを固定しているため、通常実行の結果は再現できる。
if not docs_test_mode:
    assert train_losses[-1] < train_losses[0]
    assert math.isclose(test_accs[-1], 0.975, rel_tol=0.0, abs_tol=1e-6)

# %%
# 学習曲線
_, axes = plt.subplots(1, 2, figsize=(10, 4))
epochs_range = range(1, EPOCHS + 1)

axes[0].plot(epochs_range, train_losses, "o-", color="#FF6B6B")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss")

axes[1].plot(epochs_range, test_accs, "s-", color="#2696EB")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Test Accuracy")

plt.tight_layout()
plt.show()

# %%
# 勾配が CNN と量子層の両方に流れているか確認
xb = X_train[:BATCH_SIZE]
yb = y_train[:BATCH_SIZE]

optimizer.zero_grad()
logits, feats, q_out = hybrid_model(xb)
loss = criterion(logits, yb)
loss.backward()

# ``loss.backward()`` で ``.grad`` が populate されるので、torch の stub で
# ``Tensor | None`` と宣言された ``.grad`` を ``assert`` で narrow する。
# ``nn.Sequential.__getitem__`` は torch の stub 上 ``Tensor | None | Module``
# を返す (整数 index と slice の両方を受けるため) ので、まず ``nn.Module`` に
# narrow してから ``.weight`` を取り出す。
first_conv = hybrid_model.feature_extractor[0]
assert isinstance(first_conv, torch.nn.Module)
first_conv_weight = first_conv.weight  # type: ignore[union-attr]
assert first_conv_weight.grad is not None
quantum_grad = hybrid_model.qlayer.weights.grad
assert quantum_grad is not None
print(
    "feature_extractor first conv grad mean:",
    # zuban は stub 上 Tensor の ``.grad.abs()`` チェーンを "Tensor not
    # callable" と判定するが、runtime 側は通常の ``Tensor.abs`` メソッド呼び出し。
    first_conv_weight.grad.abs().mean().item(),  # type: ignore[operator]
)
print("quantum weights grad mean:", quantum_grad.abs().mean().item())

optimizer.zero_grad()

# %% [markdown]
# ### 評価
#
# 学習済みモデルをテストデータへ適用し、データセット全体の精度とクラスごとの精度を確認します。さらに、混同行列からクラス間の誤分類傾向を、予測例から個々の画像に対する予測結果をそれぞれ確認します。

# %%
with torch.no_grad():
    q_logits, F_test_e2e, q_outputs = hybrid_model(X_test)
    preds = q_logits.argmax(1)
    quantum_acc = (preds == y_test).float().mean().item()

print(f"end-to-end 量子モデルの精度: {quantum_acc:.2%}")
print()

for c in range(N_CLASSES):
    mask = y_test == c
    class_acc = (preds[mask] == y_test[mask]).float().mean().item()
    print(f"  {CLASS_NAMES[c]}: {class_acc:.2%}")

# %% [markdown]
# 混同行列は、行に正解クラス、列にモデルが予測したクラスを取り、それぞれの組み合わせに該当する画像数を示します。対角成分は正しく分類された画像数を、対角成分以外は誤分類された画像数を表すため、モデルがどのクラスを混同しやすいかを確認できます。

# %%
# 混同行列
conf = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
for t, p in zip(y_test.numpy(), preds.numpy()):
    conf[t, p] += 1
if not docs_test_mode:
    expected_conf = np.array(
        [
            [29, 1, 0, 0],
            [0, 30, 0, 0],
            [0, 0, 30, 0],
            [1, 0, 1, 28],
        ]
    )
    np.testing.assert_array_equal(conf, expected_conf)
_, ax = plt.subplots(figsize=(5, 4))
ax.imshow(conf, cmap="Blues")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix")
ax.set_xticks(range(N_CLASSES))
ax.set_yticks(range(N_CLASSES))
ax.set_xticklabels(CLASS_NAMES, fontsize=7, rotation=45)
ax.set_yticklabels(CLASS_NAMES, fontsize=7)
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        ax.text(
            j,
            i,
            str(conf[i, j]),
            ha="center",
            va="center",
            color="white" if conf[i, j] > conf.max() / 2 else "black",
        )
plt.tight_layout()
plt.show()

# %% [markdown]
# 混同行列では、120件のテストデータのうち117件が正しく分類されています。TrouserとSandalはすべて正解しており、T-shirtは1件がTrouserへ、Bagは1件がT-shirtへ、もう1件がSandalへ誤分類されています。予測が対角成分へ集中していることから、モデルは選択した4クラスを高い精度で識別できています。


# %%
# 予測例
n_show = min(8, len(X_test))
sample_imgs = X_test[:n_show, 0].numpy()
if not docs_test_mode:
    assert torch.equal(y_test[:n_show], torch.zeros(n_show, dtype=y_test.dtype))
    assert torch.equal(preds[:n_show], y_test[:n_show])
combined = np.concatenate(sample_imgs, axis=1)
_, ax = plt.subplots(figsize=(12, 3))
ax.imshow(combined, cmap="gray", aspect="auto")
for i in range(n_show):
    color = "#4ECDC4" if preds[i] == y_test[i] else "#FF6B6B"
    ax.text(
        28 * i + 14,
        -1.5,
        CLASS_NAMES[preds[i].item()],
        ha="center",
        va="bottom",
        fontsize=7,
        color=color,
        clip_on=False,
    )
    # ``Tensor.item()`` は torch の stub 上は ``int | float | bool`` を返す
    # ことになっていて、整数型 tensor でも狭い型に narrow されない。実行時の
    # 値は常に ``int`` のクラスラベルなので、リスト index 用に明示 cast する。
    ax.text(
        28 * i + 14,
        29,
        CLASS_NAMES[int(y_test[i].item())],
        ha="center",
        va="top",
        fontsize=7,
        clip_on=False,
    )
ax.set_ylim(33, -8)
ax.set_title("Sample Predictions (green=correct, red=wrong)", pad=12)
ax.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# 表示した8件の予測例は、いずれも正解ラベルと予測ラベルがT-shirtで一致しています。画像の上側に示した予測ラベルがすべて緑色で表示されており、これらのサンプルを正しく識別できていることが確認できます。

# %% [markdown]
# ## まとめ
#
# このチュートリアルでは、以下の内容を学びました:
#
# - **HQNNのワークフロー**: 古典CNNで画像から特徴量を抽出し、その特徴量を変分量子回路で変換した後、特徴量レベル融合分類器によって画像のクラスを予測する流れを確認しました。
# - **QamomileとPyTorchによる実装**: QamomileでAnsatzと期待値計算を記述し、パラメータシフトルールを用いた`forward`と`backward`をPyTorchのautogradへ接続することで、量子回路層をend-to-end学習へ組み込みました。
# - **学習と評価**: Fashion-MNISTの4クラス分類では、学習損失がエポックとともに低下し、テストデータに対して97.5%の分類精度を得ました。混同行列と予測例からも、各クラスを高い精度で識別できることを確認しました。
