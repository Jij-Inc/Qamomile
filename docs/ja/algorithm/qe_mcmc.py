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
# tags: [algorithm, sample-based]
# ---
#
# # Quantum-enhanced Markov chain Monte Carlo (QeMCMC)
#
# このチュートリアルでは、Qamomileを使ってQuantum-enhanced Markov chain Monte Carlo (QeMCMC) {cite:p}`10.1038/s41586-023-06095-4`を実装する例を示します。

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install "qamomile[qiskit]"

# %%
import os
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from qiskit_aer import AerSimulator

import qamomile.circuit as qmc
from qamomile.circuit.algorithm import trotterized_time_evolution
from qamomile.circuit.stdlib import computational_basis_state
from qamomile.observable.hamiltonian import Hamiltonian, X, Z
from qamomile.qiskit import QiskitTranspiler

# %% [markdown]
# ---
# ## 背景

# %% [markdown]
# ### ボルツマン分布のサンプリング

# %% [markdown]
# 多くの物理・工学的な問題では、ある確率分布$\mu(\bm{x})$からサンプル$\bm{x}$を得ることが重要な計算タスクになります。代表的な例が、統計力学における**ボルツマン分布**です:
# $$
# \mu(\bm{x}) = \frac{1}{Z} \exp\bigl(-\beta E(\bm{x})\bigr), \quad Z = \sum_{\bm{x}} \exp\bigl(-\beta E(\bm{x})\bigr).
# $$
# ここで、$E(\bm{x})$は状態$\bm{x}$のエネルギー、$\beta = 1/T$は逆温度、$Z$は分配関数と呼ばれる規格化定数です。
# ボルツマン分布は熱平衡状態における状態$\bm{x}$の確率分布を与えるだけでなく、これを目標分布とするサンプリングは組合せ最適化問題の解法としても広く利用されています。
#
# ボルツマン分布のエネルギー関数の具体例として、**イジング模型**を考えてみましょう。イジング模型は、各サイト$i$にスピン変数$x_i \in \{-1, +1\}$が置かれた系で、そのエネルギーは次式で与えられます:
# $$
# E(\bm{x}) = -\sum_{\langle i, j \rangle} J_{ij} \, x_i x_j - \sum_i h_i \, x_i.
# $$
# ここで、$J_{ij}$はスピン間の相互作用、$h_i$はサイト$i$にかかる外部磁場です。状態の総数は$2^n$と指数的に増えるため、$n$が大きい場合には分配関数$Z$を厳密に計算することは困難です。そこで、$\mu(\bm{x})$から直接サンプルを得る手法として、後述するMCMCが利用されます。
#
# まずは、小さなイジング模型について実際にボルツマン分布を可視化してみましょう。ここでは、1次元強磁性イジング鎖（$J_{i,i+1} = 1$、$h_i = 0$）を考え、逆温度$\beta$を変えながら、エネルギー$E(\bm{x})$ごとの確率を集計したヒストグラムをプロットします。

# %%
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
RANDOM_SEED = 42
QAOA_BLUE = "#2696EB"
QAOA_RED = "#FF6B6B"

n_spins = 6
J = 1.0


def ising_energy(state: np.ndarray) -> float:
    """スピン配置``state``の1次元強磁性イジング鎖のエネルギーを返す（外部磁場なし）。"""
    return -J * np.sum(state[:-1] * state[1:])


# 2^n すべての状態を列挙
all_states = np.array(
    [[1 - 2 * ((k >> i) & 1) for i in range(n_spins)] for k in range(2**n_spins)]
)
energies = np.array([ising_energy(s) for s in all_states])
# n_spins サイト上の 2^n 通りのスピン配置、エネルギーは強磁性 / 反強磁性の
# 両極端 +/- (n - 1) * J に挟まれる。
assert all_states.shape == (2**n_spins, n_spins)
assert energies.shape == (2**n_spins,)
assert energies.min() == -(n_spins - 1) * J
assert energies.max() == (n_spins - 1) * J

# エネルギーごとにボルツマン分布を集計してヒストグラムとしてプロット
unique_energies = np.unique(energies)
betas = [0.5, 1.0, 2.0]

fig, axes = plt.subplots(1, len(betas), figsize=(12, 3.5), sharey=True)
for ax, beta in zip(axes, betas):
    weights = np.exp(-beta * energies)
    probs = weights / weights.sum()
    e_probs = np.array([probs[energies == e].sum() for e in unique_energies])
    ax.bar(unique_energies, e_probs, width=0.8, color=QAOA_BLUE)
    ax.set_xlabel(r"Energy $E(\mathbf{x})$")
    ax.set_title(rf"$\beta = {beta}$")
axes[0].set_ylabel(r"Probability $\mu(E)$")
fig.suptitle(f"Boltzmann distribution of {n_spins}-spin Ising chain")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### マルコフ連鎖モンテカルロ法 (MCMC)
#
# マルコフ連鎖モンテカルロ法 (MCMC) は、確率分布からのサンプリングに使用される一般的な手法です。MCMCはマルコフ連鎖と呼ばれる確率過程を利用することで、目的の分布$\mu(\bm{x})$からのサンプリングを実現します。ここでは、MCMCの代表的な実装であるMetropolis-Hastings (MH) アルゴリズム{cite:p}`10.1093/biomet/57.1.97`を紹介します。
#
# MHアルゴリズムは、ある提案確率$Q(\bm{y}|\bm{x})$に従ってマルコフ連鎖の新たな遷移$\bm{x} \rightarrow \bm{y}$を生成し、次の採択確率
# $$
# A(\bm{y} | \bm{x}) = \min \left(1, \frac{\mu(\bm{y})}{\mu(\bm{x})} \cdot \frac{Q(\bm{x} | \bm{y})}{Q(\bm{y} | \bm{x})} \right).
# $$
# に従って採用または棄却するという2つのステップからなります。この一連の手順により、時刻$t$の状態から時刻$t+1$の状態を生成します。十分な時間が経過したのちには、このマルコフ連鎖の状態はサンプリングしたい分布$\mu(\bm{x})$に従います。よって、十分に遷移を重ねることで、得られるマルコフ連鎖の状態列$\{\bm{x}^{(t)}\}$を目的のサンプルとして利用できます。
#
# 実際に確認してみましょう。先ほど用意したボルツマン分布をMH法でサンプリングしてみます。提案分布にはさまざまなものが使えますが、ここでは最もシンプルな、ランダムなスピンを1つだけ選んで反転させるものを利用します。

# %%
rng = np.random.default_rng(seed=RANDOM_SEED)


def local_update(state: np.ndarray) -> np.ndarray:
    """``state``からランダムに選んだ1スピンを反転させた新しいスピン配置を提案する。"""
    n = len(state)
    flip_index = rng.integers(0, n)
    new_state = state.copy()
    new_state[flip_index] = int(-1 * state[flip_index])
    return new_state


# %% [markdown]
# 次に、採択確率に従って、提案された遷移を確率的に処理するステップを実装します。この提案分布は$Q(\bm{x} \mid \bm{y}) = Q(\bm{y} \mid \bm{x})$を満たすため、採択確率において$Q$の比は打ち消されます。したがって、目的分布をボルツマン分布とすると、採択確率は以下の簡潔な形になります:
# $$
# A(\bm{y} | \bm{x}) = \min \left(1, \frac{\mu(\bm{y})}{\mu(\bm{x})}\right) = \min \left(1, \exp(-\beta (E(\bm{y}) - E(\bm{x})))\right).
# $$


# %%
def metropolis_hastings(
    state: np.ndarray,
    new_state: np.ndarray,
    energy_func: Callable[[np.ndarray], float],
    beta: float,
) -> np.ndarray:
    """逆温度``beta``のボルツマン分布に対するMetropolis-Hastingsルールで
    ``new_state``を``state``に対して採択または棄却する。"""

    delta_energy = energy_func(new_state) - energy_func(state)

    if delta_energy < 0 or rng.random() < np.exp(
        -beta * delta_energy
    ):  # delta_energy < 0 のとき、新状態は常により低エネルギーで採択される。
        return new_state
    else:
        return state


# %% [markdown]
# これでMCMCが実装できました。それでは、MCMCを使ってサンプリングしてみましょう。

# %%
T = 100 if docs_test_mode else 1000  # MCMCのステップ数
beta = 0.5  # 逆温度

sample = np.zeros((T, n_spins))
state = np.ones(n_spins)  # 初期状態

for t in range(T):
    new_state = local_update(state)
    state = metropolis_hastings(state, new_state, ising_energy, beta)
    sample[t] = state

# %% [markdown]
# 得られたサンプル列がボルツマン分布に従っているかを確認しましょう。得られたサンプルを用いて、ボルツマン分布$\mu(\bm{x})$に関する物理量を推定してみます。ここでは、スピンの平均磁化:
# $$
# \langle m \rangle = \sum_{\bm{x}} \mu(\bm{x}) m(\bm{x})
# $$
# を推定します。磁化は、
# $$
# m(\bm{x}) = \frac{1}{n} \sum_{i=1}^n x_i
# $$
# です。平均磁化は磁化のボルツマン分布に関する期待値なので、サンプル数を増やすとともに、得られたサンプル列がボルツマン分布に近いほど推定精度はよくなるはずです。MCMCの$t$番目までのサンプルによる推定量$\bar{m}_t$をプロットしてみましょう。


# %%
def average_magnetization(sample: np.ndarray) -> float:
    """形状``(T, n_spins)``のMCMCサンプルから平均磁化の推定値を返す。"""
    magnetization = np.mean(sample, axis=1)
    return np.mean(magnetization)


sample_magnetization = np.array(
    [average_magnetization(sample[:i]) for i in range(1, T + 1)]
)

# 現在の逆温度 beta におけるボルツマン分布から、平均磁化の理論値を計算
weights = np.exp(-beta * energies)
probs = weights / weights.sum()
magnetization_per_state = all_states.mean(axis=1)
theoretical_magnetization = np.sum(probs * magnetization_per_state)
# Z2 対称性: 全スピン反転は (h_i = 0 の) エネルギーを変えず磁化だけ反転させるので、
# ボルツマン重み付き平均はぴったり 0 になる — しかも {+spins, -spins} の対は同じ
# `probs` を共有するため浮動小数演算上も誤差なくキャンセルする。
assert theoretical_magnetization == 0.0
assert np.isclose(probs.sum(), 1.0)

plt.plot(sample_magnetization, color=QAOA_BLUE, label="MCMC estimate")
plt.axhline(
    theoretical_magnetization,
    color="black",
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
# Quantum-enhanced MCMCアルゴリズムは、量子回路からのサンプリングを提案分布として利用するMCMCです{cite:p}`10.1038/s41586-023-06095-4`。現在の状態$\bm{x}$から始め、量子回路$U$を作用させて計算基底で測定することで、新たな状態$\bm{y}$を得ます。このとき、提案分布$Q(\bm{y}|\bm{x})$は以下のようになります:
# $$
# Q(\bm{y}|\bm{x}) = \| \langle \bm{y} | U | \bm{x} \rangle \|^2
# $$
# この確率を直接計算するのは困難ですが、量子回路が$U = U^\top$を満たすとき、提案分布は$Q(\bm{x} \mid \bm{y}) = Q(\bm{y} \mid \bm{x})$を満たし、採択確率の中で$Q$の項は打ち消されるため、$Q$を明示的に計算する必要がなくなります。例えば、イジング模型のボルツマン分布をサンプリングするためには、時間に依存しないハミルトニアンによるTrotter化された時間発展を利用できます:
# $$
# U(\gamma, t) = \exp(-i H t), \quad \quad
# H = (1-\gamma) \alpha H_C + \gamma H_M.
# $$
# ここで、$H_M$はミキサーハミルトニアンと呼ばれ、状態間の量子遷移を生み出します。一方、$H_C$はイジングハミルトニアンです。$\gamma \in [0,1]$はミキサーハミルトニアンの重みを制御します。正規化係数$\alpha = \lVert H_M \rVert_F / \lVert H_C \rVert_F$は、ミキサーハミルトニアンとコストハミルトニアンのFrobeniusノルムのスケールを揃えます。$(\gamma, t)$はMCMCの効率を決める調整可能なパラメータです。

# %% [markdown]
# ---
# ## Qamomileによる実装

# %% [markdown]
# ### 1. ハミルトニアンの準備
# いよいよ、アルゴリズムを実装していきましょう。まず、サンプリングしたいモデルのイジングハミルトニアン$H_C$と、提案回路$U$のためのミキサーハミルトニアン$H_M$を準備します。

# %%
mixer_hamiltonian = Hamiltonian()
for i in range(n_spins):
    mixer_hamiltonian += X(i)

cost_hamiltonian = Hamiltonian()
for i in range(n_spins - 1):
    cost_hamiltonian += -J * Z(i) * Z(i + 1)

# 両方のFrobeniusノルムに共通する2**(n_spins / 2)の因子は相殺される
alpha = np.sqrt(
    sum(abs(coefficient) ** 2 for coefficient in mixer_hamiltonian.terms.values())
    / sum(abs(coefficient) ** 2 for coefficient in cost_hamiltonian.terms.values())
)
assert np.isclose(alpha, np.sqrt(n_spins / ((n_spins - 1) * J**2)))

# %% [markdown]
# ### 2. 量子回路の構築
#
# 次に、量子回路を実装していきましょう。まず、現在の状態$\bm{x}$を入力状態として符号化するため、`computational_basis_state`を用いて量子状態$\ket{\bm{x}}$を準備します。提案遷移にはTrotter分解に基づく時間発展シミュレーションを利用します。先ほど準備したハミルトニアンに対する回路を、`trotterized_time_evolution`を使って構築します。


# %%
@qmc.qkernel
def qemcmc_circuit(
    n: qmc.UInt,
    input_bits: qmc.Vector[qmc.UInt],
    Hs: qmc.Vector[qmc.Observable],
    order: qmc.UInt,
    time: qmc.Float,
    step: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    """QeMCMC用の提案回路: ``n``量子ビット上に``|input_bits>``を準備し、
    指定した``order``の鈴木-Trotter分解で全時間``time``、``step``ステップに
    わたり``sum_k Hs[k]``の時間発展を作用させ、最後に全量子ビットを測定する。"""
    q = qmc.qubit_array(n, name="q")

    # step 1: 初期状態を準備
    q = computational_basis_state(q, input_bits)

    # step 2: ミキサー/コストハミルトニアンによるTrotter化された時間発展を適用
    q = trotterized_time_evolution(q, Hs, order, time, step)

    return qmc.measure(q)


# %% [markdown]
# ### 3. トランスパイル
#
# 量子カーネルをトランスパイルします。{cite:t}`10.1103/PhysRevA.111.042615`に従い、ミキサー項の重み$\gamma=0.45$、シミュレーション時間$t=12$、$\Delta t = 0.8$と設定します。トランスパイル時に`n`、`order`、`time`、`step`をバインドし、`input_bits`はランタイムパラメータとして残します。
# %%
gamma = 0.45  # ミキサー項の重み
time = 12.0  # 総発展時間
delta_t = 0.8  # Trotterステップの時間幅
step = int(time / delta_t)  # Trotterステップ数
order = 2  # Suzuki-Trotter近似次数
assert step == 15  # 12.0 / 0.8

Hs = [
    gamma * mixer_hamiltonian,
    (1 - gamma) * alpha * cost_hamiltonian,
]
assert len(Hs) == 2

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
    parameters=["input_bits"],
)

# %% [markdown]
# 以下では、代表例として0と1が交互に並ぶ入力状態に対するQamomileレベルの提案回路を描画します。繰り返されるSuzuki-Trotter構造を簡潔に示すため、ループは折りたたんで表示します。

# %%
qemcmc_circuit.draw(
    n=n_spins,
    input_bits=[i % 2 for i in range(n_spins)],
    Hs=Hs,
    order=order,
    time=time,
    step=step,
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
            raise ValueError(
                f"input_kind='spin' は {{-1, 1}} の要素を要求します: {values.tolist()}"
            )
        return (1 - x) // 2
    if input_kind == "binary":
        if not np.all(np.isin(values, [0, 1])):
            raise ValueError(
                f"input_kind='binary' は {{0, 1}} の要素を要求します: {values.tolist()}"
            )
        return 1 - 2 * x
    if input_kind != "auto":
        raise ValueError(
            f"input_kind は 'spin', 'binary', 'auto' のいずれか: {input_kind!r}"
        )

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


def quantum_proposal(state: np.ndarray, executable: Any, executor: Any) -> np.ndarray:
    """現在のスピン状態を入力として量子回路から提案状態を得る。"""
    binary_state = spin_binary_convert(state, input_kind="spin").tolist()
    result = executable.sample(
        executor,
        shots=1,
        bindings={"input_bits": binary_state},
    ).result()
    ((proposed_bits, _count),) = result.results
    return spin_binary_convert(np.array(proposed_bits, dtype=int), input_kind="binary")


# %% [markdown]
# ---
# ## 結果

# %% [markdown]
# 実装したQeMCMCアルゴリズムを実行してみましょう。先ほどよりも低温の$\beta = 2.0$に設定し、古典MCMCの局所更新では混合が遅くなる条件で量子提案分布の挙動を観察します。公平な比較のため、同じ$\beta = 2.0$で古典MCMCも併走させます。

# %%
beta = 2.0  # 局所更新では混合が遅くなる低温に切り替える
T_quantum = (
    20 if docs_test_mode else 1000
)  # 量子回路シミュレーションのコストが高いため小さめに設定

# 各サンプリングジョブを同じ疑似乱数列にリセットせず、再現可能かつ異なるseed列を生成
quantum_job_seeds = np.random.default_rng(seed=RANDOM_SEED).integers(
    0,
    np.iinfo(np.uint32).max,
    size=T_quantum,
    dtype=np.uint32,
)
assert np.unique(quantum_job_seeds).size == T_quantum

# beta=2.0におけるボルツマン分布から平均磁化の理論値を再計算
weights = np.exp(-beta * energies)
probs = weights / weights.sum()
theoretical_magnetization = np.sum(probs * magnetization_per_state)
# Z2対称性により依然として0であり、betaは対の重みを変えるだけで
# {+spins, -spins}の縮退を破らない
assert theoretical_magnetization == 0.0

# 比較用に同じbeta、同じステップ数で古典MCMCも実行
classical_compare_sample = np.zeros((T_quantum, n_spins))
state = np.ones(n_spins)  # 初期状態
for t in range(T_quantum):
    new_state = local_update(state)
    state = metropolis_hastings(state, new_state, ising_energy, beta)
    classical_compare_sample[t] = state

# QeMCMC
quantum_backend = AerSimulator()
executor = transpiler.executor(backend=quantum_backend)
quantum_sample = np.zeros((T_quantum, n_spins), dtype=int)
state = np.ones(n_spins, dtype=int)  # 初期状態
for t, job_seed in enumerate(quantum_job_seeds):
    quantum_backend.set_options(seed_simulator=int(job_seed))
    proposed_state = quantum_proposal(state, executable, executor)
    state = metropolis_hastings(state, proposed_state, ising_energy, beta)
    quantum_sample[t] = state
# 両chainともn_spinsサイト・T_quantumサンプル分の+/- 1スピン列を生成済み
assert quantum_sample.shape == (T_quantum, n_spins)
assert classical_compare_sample.shape == (T_quantum, n_spins)
assert set(np.unique(quantum_sample).tolist()).issubset({-1, 1})


# %% [markdown]
# 平均磁化の推定量を計算し、同じ$\beta = 2.0$で実行した古典MCMCの結果と比較します。

# %%
quantum_sample_magnetization = np.array(
    [average_magnetization(quantum_sample[:i]) for i in range(1, T_quantum + 1)]
)
classical_compare_magnetization = np.array(
    [
        average_magnetization(classical_compare_sample[:i])
        for i in range(1, T_quantum + 1)
    ]
)

plt.plot(classical_compare_magnetization, color=QAOA_BLUE, label="MCMC estimate")
plt.plot(quantum_sample_magnetization, color=QAOA_RED, label="QeMCMC estimate")
plt.axhline(
    theoretical_magnetization,
    color="black",
    linestyle="--",
    label=f"Theoretical ({theoretical_magnetization:.3f})",
)
plt.xlabel("Step")
plt.ylabel("Magnetization")
plt.title("Magnetization vs. Step")
plt.legend()
plt.show()

# %% [markdown]
# 低温では、局所更新は異なる配置へ遷移する確率が低下するため、混合が非常に遅くなります。対して、QeMCMCは量子回路による遷移を利用することで、低温においても頻繁に状態が遷移し、厳密な値へ収束していく様子が確認できます。

# %% [markdown]
# ---
# ## まとめ
#
# このノートブックでは、以下を行いました。
#
# - 古典的なMetropolis-Hastings MCMCによるボルツマン分布のサンプリングと、低温で局所更新の混合が遅くなる理由を確認しました。
# - ミキサーハミルトニアンとコストハミルトニアンによる時間発展を`trotterized_time_evolution`で実装し、QamomileでQeMCMCの提案回路を構築しました。
# - 量子提案をMetropolis-Hastings法へ組み込み、QeMCMCと古典MCMCを比較しました。
