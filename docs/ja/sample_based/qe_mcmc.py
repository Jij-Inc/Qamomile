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
# # Quantum-enhanced Markov chain Monte Carlo
#
# このチュートリアルでは、Qamomileを使ってQuantum-enhanced Markov chain Monte Carlo (QeMCMC) {https://doi.org/10.1038/s41586-023-06095-4} を実装する例を示します。

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install qamomile

# %% [markdown]
# ---
# ## 背景

# %% [markdown]
# ### ボルツマン分布のサンプリング

# %% [markdown]
# 多くの物理・工学的な問題では、ある確率分布$\pi(\bm{x})$からサンプル$\bm{x}$を得ることが重要な計算タスクになります。代表的な例が、統計力学における**ボルツマン分布**です:
# $$
# \pi(\bm{x}) = \frac{1}{Z} \exp\bigl(-\beta E(\bm{x})\bigr), \quad Z = \sum_{\bm{x}} \exp\bigl(-\beta E(\bm{x})\bigr).
# $$
# ここで、$E(\bm{x})$は状態$\bm{x}$のエネルギー、$\beta = 1/T$は逆温度、$Z$は分配関数と呼ばれる規格化定数です。ボルツマン分布は熱平衡状態のスピン配置を記述するだけでなく、組合せ最適化問題に対するサンプリング手法としても広く利用されています。
#
# ボルツマン分布の具体例として、**イジング模型**を考えてみましょう。イジング模型は、各サイト$i$にスピン変数$x_i \in \{-1, +1\}$が置かれた系で、そのエネルギーは次式で与えられます:
# $$
# E(\bm{x}) = -\sum_{\langle i, j \rangle} J_{ij} \, x_i x_j - \sum_i h_i \, x_i.
# $$
# ここで、$J_{ij}$はスピン間の相互作用、$h_i$はサイト$i$にかかる外部磁場です。状態の総数は$2^n$と指数的に増えるため、$n$が大きい場合には分配関数$Z$を厳密に計算することは困難です。そこで、$\pi(\bm{x})$から直接サンプルを得る手法として、後述するMCMCが利用されます。
#
# まずは、小さなイジング模型について実際にボルツマン分布を可視化してみましょう。ここでは、1次元強磁性イジング鎖（$J_{ij} = 1$、$h_i = 0$）を考え、逆温度$\beta$を変えながら、エネルギー$E(\bm{x})$ごとの確率を集計したヒストグラムをプロットします。

# %%
import numpy as np
import matplotlib.pyplot as plt

n_spins = 6
J = 1.0

# 1次元強磁性イジング鎖のエネルギー (外部磁場なし)
def ising_energy(state):
    return -J * np.sum(state[:-1] * state[1:])

# 2^n すべての状態を列挙
all_states = np.array(
    [[1 - 2 * ((k >> i) & 1) for i in range(n_spins)] for k in range(2**n_spins)]
)
energies = np.array([ising_energy(s) for s in all_states])

# エネルギーごとにボルツマン分布を集計してヒストグラムとしてプロット
unique_energies = np.unique(energies)
betas = [0.5, 1.0, 2.0]

fig, axes = plt.subplots(1, len(betas), figsize=(12, 3.5), sharey=True)
for ax, beta in zip(axes, betas):
    weights = np.exp(-beta * energies)
    probs = weights / weights.sum()
    e_probs = np.array([probs[energies == e].sum() for e in unique_energies])
    ax.bar(unique_energies, e_probs, width=0.8)
    ax.set_xlabel(r"Energy $E(\mathbf{x})$")
    ax.set_title(rf"$\beta = {beta}$")
axes[0].set_ylabel(r"Probability $\pi(E)$")
fig.suptitle(f"Boltzmann distribution of {n_spins}-spin Ising chain")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### マルコフ連鎖モンテカルロ法 (MCMC)
#
# マルコフ連鎖モンテカルロ法 (MCMC) は、確率分布からのサンプリングに使用される一般的な手法です。MCMCはマルコフ連鎖と呼ばれる確率過程を利用することで、所望の確率分布からのサンプリングを実現します。ここでは、MCMCの一般的な実装である、Metropolis-Hastings (MH) アルゴリズム {https://doi.org/10.1093/biomet/57.1.97} を紹介します。
#
# MHアルゴリズムは、ある確率$Q(\bm{y}|\bm{x})$に従ってマルコフ連鎖の新たな遷移$\bm{x} \rightarrow \bm{x}'$を生成し、この遷移を採択確率
# $$
# A(\bm{y} | \bm{x}) = \min \left(1, \frac{\pi(\bm{y})}{\pi(\bm{x})} \cdot \frac{Q(\bm{x} | \bm{y})}{Q(\bm{y} | \bm{x})} \right).
# $$
# に従って採用または棄却するという2つのステップからなります。この一連の手順により、マルコフ連鎖の時間$t$の状態から、次の時間$t+1$の状態を生成します。このマルコフ連鎖は、十分な時間を経たのちに、状態はサンプリングしたい分布$\pi(\bm{x})$に従います。よって、十分に遷移させることで、得られるマルコフ連鎖の状態$\{\bm{x}^{(t)}\}$を目的のサンプルとして得ることが可能です。
#
# 実際に確認してみましょう。先ほど用意したボルツマン分布をMH法を使ってサンプリングしてみます。提案分布にはさまざまなものが使えますが、最もシンプルなものとして、ランダムなスピンを1つだけ選んで反転させるというものを利用します。

# %%
rng = np.random.default_rng(seed=0)

def local_update(state):
    n = len(state)
    flip_index = rng.integers(0, n)
    new_state = state.copy()
    new_state[flip_index] = int(-1 * state[flip_index])
    return new_state

# %% [markdown]
# 次に、採択確率に従って、提案された遷移を確率的に処理するステップを実装します。先ほどの提案分布は、ボルツマン分布の場合、以下のように簡略化されます。

# %%
def metropolis_hastings(state, new_state, energy_func, beta):

    delta_energy = energy_func(new_state) - energy_func(state)

    if delta_energy < 0 or rng.random() < np.exp(-beta * delta_energy):
        return new_state
    else:
        return state

# %% [markdown]
# これでMCMCが実装できました。それでは、MCMCを使ってサンプリングしてみましょう。

# %%
T = 10000 # MCMCのステップ数
beta = 1.0 # 逆温度

sample = np.zeros((T, n_spins))
state = np.ones(n_spins) # 初期状態

for t in range(T):
    new_state = local_update(state)
    state = metropolis_hastings(state, new_state, ising_energy, beta)
    sample[t] = state

# %% [markdown]
# 得られたサンプル列がボルツマン分布に従っているかを確認しましょう。得られたサンプルを用いて、ボルツマン分布に関する物理量を推定してみます。ここでは、スピンの平均磁化:
# $$
# \langle \mu \rangle = \sum_{\bm{x}} \mu(\bm{x}) m(\bm{x})
# $$
# を推定します。磁化は、
# $$
# m(\bm{x}) = \frac{1}{n} \sum_{i=1}^n x_i
# $$
# です。平均磁化は、磁化のボルツマン分布に関する期待値なので、得られたサンプルがボルツマン分布に近いほど良い推定量が得られるはずです。MCMCの$t$番目までのサンプルによる推定量$\bar{\mu_t}$をプロットしてみましょう。

# %%
def average_magnetization(sample):
    magnetization = np.mean(sample, axis=1)
    return np.mean(magnetization)

sample_magnetization = np.array([average_magnetization(sample[:i]) for i in range(1, T + 1)])

# 現在の逆温度 beta におけるボルツマン分布から、平均磁化の理論値を計算
weights = np.exp(-beta * energies)
probs = weights / weights.sum()
magnetization_per_state = all_states.mean(axis=1)
theoretical_magnetization = np.sum(probs * magnetization_per_state)

plt.plot(sample_magnetization, label="MCMC estimate")
plt.axhline(
    theoretical_magnetization,
    color="red",
    linestyle="--",
    label=f"Theoretical ({theoretical_magnetization:.3f})",
)
plt.xlabel("Step")
plt.ylabel("Magnetization")
plt.title("Magnetization vs. Step")
plt.legend()
plt.show()

# %% [markdown]
# ---
# ## アルゴリズム

# %% [markdown]
# Quantum-enhanced MCMCアルゴリズムは、量子回路からのサンプリングを提案分布として利用するMCMCです {https://doi.org/10.1038/s41586-023-06095-4}。現在の状態$\bm{x}$に対して、量子回路$U$を作用させ、計算基底で測定することで、新たな状態$\bm{y}$を得ます。このとき、提案分布$Q(\bm{y}|\bm{x})$は以下のようになります:
# $$
# Q(\bm{y}|\bm{x}) = \| \langle \bm{y} | U | \bm{x} \rangle \|^2
# $$
# この確率を直接計算するのは困難ですが、量子回路が$U = U^\top$を満たすとき、$Q$を計算する必要がなくなります。例えば、イジング模型に対するボルツマン分布のサンプリングのために、QAOAのようなミキサーハミルトニアン$H_M$とイジングハミルトニアン$H_C$に対するTrotter分解の時間発展の量子回路を用いることができます:
# $$
# U(\gamma, t) = \exp(-i H t) \quad \quad H = (1-\gamma) H_M + \gamma H_C.
# $$

# %% [markdown]
# ---
# ## アルゴリズムの実装

# %% [markdown]
# ### 1. ハミルトニアンの準備
# いよいよ、アルゴリズムを実装していきましょう。まず、サンプリングしたいモデルのイジングハミルトニアン$H_C$と、提案回路$U$のためのミキサーハミルトニアン$H_M$を準備します。

# %%
from qamomile.observable.hamiltonian import Hamiltonian, X, Z

mixer_hamiltonian = Hamiltonian()
for i in range(n_spins):
    mixer_hamiltonian += X(i)

cost_hamiltonian = Hamiltonian()
for i in range(n_spins - 1):
    cost_hamiltonian += -J * Z(i) * Z(i + 1)

# %% [markdown]
# ### 2. 量子回路の構築
#
# 次に、量子回路部分を実装していきましょう。量子回路は、Trotter分解による時間発展シミュレーションを利用します。ここでは、`qamomile.circuit.algorithm.trotter`を使い、準備したハミルトニアンに関する量子回路を実装します。

# %%
import qamomile.circuit as qmc
from qamomile.circuit.algorithm.basic import computational_basis_state
from qamomile.circuit.algorithm.trotter import trotterized_time_evolution

@qmc.qkernel
def qemcmc_circuit(
    n: qmc.UInt,
    input: qmc.Vector[qmc.UInt],
    Hs: qmc.Vector[qmc.Observable],
    order: qmc.UInt,
    time: qmc.Float,
    step: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    # step 1: prepare the initial state
    q = computational_basis_state(n, q, input)

    # step 2: apply the trotterized evolution under the mixer and cost Hamiltonians
    q = trotterized_time_evolution(q, Hs, order, time, step)

    return qmc.measure(q)

# %% [markdown]
# ### 3. トランスパイル
#
# カーネルをトランスパイルします。量子回路を実行するには、ハミルトニアンの混合係数$\gamma$とシミュレーション時間$t$を固定する必要があります。{https://doi.org/10.1103/PhysRevA.111.042615} に従い、$\gamma=0.45$、$t=12$、$\Delta t = 0.8$と設定します。トランスパイル時に`n`、`order`、`time`、`step`をバインドし、`input`はランタイムパラメータとして残します。
# %%
from qamomile.qiskit import QiskitTranspiler

gamma = 0.45  # 混合係数
time = 12.0   # 総発展時間
step = 15    # Trotter ステップ数
order = 2    # Suzuki-Trotter 近似次数

Hs = [
    (1 - gamma) * mixer_hamiltonian,
    gamma * cost_hamiltonian,
]

transpiler = QiskitTranspiler()

executable = transpiler.transpile(
    qemcmc_circuit,
    bindings={
        "n": n_spins,
        "Hs": Hs,
        "order": order,
        "time": time,
        "step": step,
    },
    parameters=["input"],
)

# %% [markdown]
# ### 4. 量子回路をMCMCに組み込む
#
# これで量子回路シミュレーションの準備が整いました。最後に、量子計算のブロックをMCMCに組み込みましょう。回路の入力および出力はビット列$\bm{x} \in \{0,1\}^n$であるため、スピン変数$\bm{s} \in \{1, -1\}^n$との変換も準備しておきます。

# %%
def spin_binary_convert(x: np.ndarray, *, input_kind: str = "auto") -> np.ndarray:
    """スピン変数 {-1, +1} ↔ バイナリ変数 {0, 1} の相互変換。

    ``input_kind`` で入力の表現を指定します。``"spin"`` は ``x`` を {-1, +1}
    として扱いバイナリを返し、``"binary"`` は {0, 1} として扱いスピンを返
    します。デフォルトの ``"auto"`` は値から推論しますが、全要素が 1 で
    曖昧な場合は ``ValueError`` を投げます。
    """
    x = np.asarray(x, dtype=int)
    values = np.unique(x)

    if input_kind == "spin":
        if not np.all(np.isin(values, [-1, 1])):
            raise ValueError(f"input_kind='spin' は {{-1, 1}} の要素を要求します: {values.tolist()}")
        return (1 - x) // 2
    if input_kind == "binary":
        if not np.all(np.isin(values, [0, 1])):
            raise ValueError(f"input_kind='binary' は {{0, 1}} の要素を要求します: {values.tolist()}")
        return 1 - 2 * x
    if input_kind != "auto":
        raise ValueError(f"input_kind は 'spin', 'binary', 'auto' のいずれか: {input_kind!r}")

    if np.any(values == -1) and np.all(np.isin(values, [-1, 1])):
        return (1 - x) // 2
    if np.any(values == 0) and np.all(np.isin(values, [0, 1])):
        return 1 - 2 * x
    if np.array_equal(values, [1]):
        raise ValueError(
            "入力が全て 1 のためスピン/バイナリを判別できません。"
            "input_kind='spin' または input_kind='binary' を明示してください。"
        )
    raise ValueError(
        f"要素は {{-1, 1}} または {{0, 1}} のみ許容されます: {values.tolist()}"
    )


def quantum_proposal(state: np.ndarray, executable, executor) -> np.ndarray:
    """現在のスピン状態を入力として量子回路から提案状態を得る。"""
    binary_state = spin_binary_convert(state, input_kind="spin").tolist()
    result = executable.sample(
        executor,
        shots=1,
        bindings={"input": binary_state},
    ).result()
    (proposed_bits, _count), = result.results
    return spin_binary_convert(np.array(proposed_bits, dtype=int), input_kind="binary")

# %% [markdown]
# ---
# ## 実行例

# %% [markdown]
# 実装したQeMCMCアルゴリズムを実行してみましょう。比較のために、先ほどの古典的な提案分布も同時に実行します。

# %%
from qiskit_aer import AerSimulator

T_quantum = 1000  # 量子回路シミュレーションのコストが高いため古典より小さめに設定

executor = transpiler.executor(backend=AerSimulator(seed_simulator=7))

quantum_sample = np.zeros((T_quantum, n_spins), dtype=int)
state = np.ones(n_spins, dtype=int)  # 初期状態

for t in range(T_quantum):
    proposed_state = quantum_proposal(state, executable, executor)
    state = metropolis_hastings(state, proposed_state, ising_energy, beta)
    quantum_sample[t] = state

# %%
quantum_sample_magnetization = np.array([average_magnetization(quantum_sample[:i]) for i in range(1, T_quantum + 1)])

plt.plot(sample_magnetization[:T_quantum], label="MCMC estimate")
plt.plot(quantum_sample_magnetization, label="QeMCMC estimate")
plt.axhline(
    theoretical_magnetization,
    color="red",
    linestyle="--",
    label=f"Theoretical ({theoretical_magnetization:.3f})",
)
plt.xlabel("Step")
plt.ylabel("Magnetization")
plt.title("Magnetization vs. Step")
plt.legend()
plt.show()

# %% [markdown]
# ---
# ## まとめ
#
# 本チュートリアルでは、古典的なMetropolis-Hastings法によるMCMCの復習からはじめ、量子回路$U(\gamma, t) = \exp(-i t H)$（$H = (1-\gamma) H_M + \gamma H_C$）を提案分布として利用するQuantum-enhanced MCMC (QeMCMC) をQamomile上で実装しました。具体的には、`qamomile.observable`でミキサー/コストハミルトニアンを準備したうえで、`@qkernel`の中で`trotterized_time_evolution`を用いたSuzuki-Trotter時間発展による提案回路を構築しました。最後に、トランスパイルされたexecutorを介して量子提案分布を既存のMHループに組み込み、古典MCMCとQeMCMCを同一のイジング鎖上で走らせて平均磁化の収束を比較することで、一連の量子古典ハイブリッドループが意図どおり動作することを確認しました。
