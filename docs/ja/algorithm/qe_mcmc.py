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
# tags: [algorithm, sample-based]
# ---
#
# # Quantum-enhanced Markov chain Monte Carlo
#
# このチュートリアルでは、Qamomileを使ってQuantum-enhanced Markov chain Monte Carlo (QeMCMC) [](https://doi.org/10.1038/s41586-023-06095-4) を実装する例を示します。

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install qamomile

# %%
import cmath
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np

import qamomile as qm
from qamomile.stdlib import suzuki_trotter

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

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
N_SPINS = 6
ISING_COUPLING = 1.0


def spin_chain_energy(state: np.ndarray) -> float:
    """Return the energy of a 1D ferromagnetic Ising chain."""

    return -ISING_COUPLING * float(np.sum(state[:-1] * state[1:]))


all_spin_states = np.array(
    [
        [1 - 2 * ((basis >> index) & 1) for index in range(N_SPINS)]
        for basis in range(1 << N_SPINS)
    ]
)
spin_energies = np.array([spin_chain_energy(state) for state in all_spin_states])
assert all_spin_states.shape == (1 << N_SPINS, N_SPINS)
assert spin_energies.min() == -(N_SPINS - 1) * ISING_COUPLING
assert spin_energies.max() == (N_SPINS - 1) * ISING_COUPLING

unique_energies = np.unique(spin_energies)
betas = [0.5, 1.0, 2.0]
figure, axes = plt.subplots(1, len(betas), figsize=(12, 3.5), sharey=True)
for axis, inverse_temperature in zip(axes, betas, strict=True):
    boltzmann_weights = np.exp(-inverse_temperature * spin_energies)
    probabilities = boltzmann_weights / boltzmann_weights.sum()
    energy_probabilities = np.array(
        [probabilities[spin_energies == energy].sum() for energy in unique_energies]
    )
    axis.bar(unique_energies, energy_probabilities, width=0.8)
    axis.set_xlabel(r"Energy $E(\mathbf{x})$")
    axis.set_title(rf"$\beta = {inverse_temperature}$")
axes[0].set_ylabel(r"Probability $\mu(E)$")
figure.suptitle(f"Boltzmann distribution of {N_SPINS}-spin Ising chain")
figure.tight_layout()
plt.show()

# %% [markdown]
# ### マルコフ連鎖モンテカルロ法 (MCMC)
#
# マルコフ連鎖モンテカルロ法 (MCMC) は、確率分布からのサンプリングに使用される一般的な手法です。MCMCはマルコフ連鎖と呼ばれる確率過程を利用することで、目的の分布$\mu(\bm{x})$からのサンプリングを実現します。ここでは、MCMCの代表的な実装であるMetropolis-Hastings (MH) アルゴリズム [](https://doi.org/10.1093/biomet/57.1.97) を紹介します。
#
# MHアルゴリズムは、ある提案確率$Q(\bm{y}|\bm{x})$に従ってマルコフ連鎖の新たな遷移$\bm{x} \rightarrow \bm{y}$を生成し、次の採択確率
# $$
# A(\bm{y} | \bm{x}) = \min \left(1, \frac{\mu(\bm{y})}{\mu(\bm{x})} \cdot \frac{Q(\bm{x} | \bm{y})}{Q(\bm{y} | \bm{x})} \right).
# $$
# に従って採用または棄却するという2つのステップからなります。この一連の手順により、時刻$t$の状態から時刻$t+1$の状態を生成します。十分な時間が経過したのちには、このマルコフ連鎖の状態はサンプリングしたい分布$\mu(\bm{x})$に従います。よって、十分に遷移を重ねることで、得られるマルコフ連鎖の状態列$\{\bm{x}^{(t)}\}$を目的のサンプルとして利用できます。
#
# 実際に確認してみましょう。先ほど用意したボルツマン分布をMH法でサンプリングしてみます。提案分布にはさまざまなものが使えますが、ここでは最もシンプルな、ランダムなスピンを1つだけ選んで反転させるものを利用します。

# %% [markdown]
# 次に、採択確率に従って、提案された遷移を確率的に処理するステップを実装します。この提案分布は$Q(\bm{x} \mid \bm{y}) = Q(\bm{y} \mid \bm{x})$を満たすため、採択確率において$Q$の比は打ち消されます。したがって、目的分布をボルツマン分布とすると、採択確率は以下の簡潔な形になります:
# $$
# A(\bm{y} | \bm{x}) = \min \left(1, \frac{\mu(\bm{y})}{\mu(\bm{x})}\right) = \min \left(1, \exp(-\beta (E(\bm{y}) - E(\bm{x})))\right).
# $$

# %% [markdown]
# これでMCMCが実装できました。それでは、MCMCを使ってサンプリングしてみましょう。

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
classical_rng = random.Random(0)


def local_spin_update(state: np.ndarray) -> np.ndarray:
    """Propose a spin configuration by flipping one selected spin."""

    proposal = state.copy()
    index = classical_rng.randrange(len(state))
    proposal[index] *= -1
    return proposal


def metropolis_hastings_spins(
    state: np.ndarray,
    candidate: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Accept or reject a spin proposal under the Boltzmann target."""

    delta_energy = spin_chain_energy(candidate) - spin_chain_energy(state)
    if delta_energy < 0.0 or classical_rng.random() < math.exp(-beta * delta_energy):
        return candidate
    return state


CLASSICAL_STEPS = 12 if docs_test_mode else 1_000
CLASSICAL_BETA = 0.5
classical_samples = np.zeros((CLASSICAL_STEPS, N_SPINS))
classical_state = np.ones(N_SPINS)
for step in range(CLASSICAL_STEPS):
    classical_candidate = local_spin_update(classical_state)
    classical_state = metropolis_hastings_spins(
        classical_state,
        classical_candidate,
        CLASSICAL_BETA,
    )
    classical_samples[step] = classical_state


def average_magnetization(samples: np.ndarray) -> float:
    """Return the average magnetization of a two-dimensional sample array."""

    return float(np.mean(np.mean(samples, axis=1)))


classical_magnetization = np.array(
    [
        average_magnetization(classical_samples[:step])
        for step in range(1, CLASSICAL_STEPS + 1)
    ]
)
boltzmann_weights = np.exp(-CLASSICAL_BETA * spin_energies)
boltzmann_probabilities = boltzmann_weights / boltzmann_weights.sum()
theoretical_magnetization = float(
    np.sum(boltzmann_probabilities * all_spin_states.mean(axis=1))
)
assert math.isclose(theoretical_magnetization, 0.0, abs_tol=1e-15)

plt.plot(classical_magnetization, label="MCMC estimate")
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
# Quantum-enhanced MCMCアルゴリズムは、量子回路からのサンプリングを提案分布として利用するMCMCです [](https://doi.org/10.1038/s41586-023-06095-4)。現在の状態$\bm{x}$から始め、量子回路$U$を作用させて計算基底で測定することで、新たな状態$\bm{y}$を得ます。このとき、提案分布$Q(\bm{y}|\bm{x})$は以下のようになります:
# $$
# Q(\bm{y}|\bm{x}) = \| \langle \bm{y} | U | \bm{x} \rangle \|^2
# $$
# この確率を直接計算するのは困難ですが、量子回路が$U = U^\top$を満たすとき、提案分布は$Q(\bm{x} \mid \bm{y}) = Q(\bm{y} \mid \bm{x})$を満たし、採択確率の中で$Q$の項は打ち消されるため、$Q$を明示的に計算する必要がなくなります。例えば、イジング模型のボルツマン分布をサンプリングするためには、時間に依存しないハミルトニアンによるTrotter化された時間発展を利用できます:
# $$
# U(\gamma, t) = \exp(-i H t), \quad \quad
# H = (1-\gamma) \alpha H_M + \gamma H_C.
# $$
# ここで、$H_M$はミキサーハミルトニアンと呼ばれ、状態間の量子遷移を生み出します。一方、$H_C$はイジングハミルトニアンです。$\gamma \in [0,1]$は2つの項の重みを制御するパラメータです。$\alpha$は、ミキサーハミルトニアンとコストハミルトニアンの固有値のスケールを揃えるための規格化因子です。$(\gamma, t)$はMCMCの効率を決める調整可能なパラメータです。

# %%
GAMMA = 0.45
EVOLUTION_TIME = 12.0
BETA = 1.0
N_SPINS = 6
TROTTER_STEPS = 1 if docs_test_mode else 15
INPUT_BITS = (1, 0, 1, 0, 1, 0)
INPUT_ANGLES = tuple(math.pi * bit for bit in INPUT_BITS)
HAMILTONIANS = (
    (1.0 - GAMMA)
    * sum(
        (qm.X(index) for index in range(N_SPINS)),
        qm.Hamiltonian.zero(num_qubits=N_SPINS),
    ),
    -GAMMA
    * sum(
        (qm.Z(index) * qm.Z(index + 1) for index in range(N_SPINS - 1)),
        qm.Hamiltonian.zero(num_qubits=N_SPINS),
    ),
)
EVOLUTION = suzuki_trotter(term_count=2, order=2, steps=TROTTER_STEPS)

assert len(HAMILTONIANS) == 2
assert TROTTER_STEPS == (1 if docs_test_mode else 15)

# %% [markdown]
# ## 1つの型付き提案を書く
#
# 0または$\pi$の`RX`回転で現在状態の各bitを準備します。固定長Vectorにより、入力と2つのHamiltonian項はどちらも検査済みの関数contractに含まれます。affineな`Q[Bits[6]]`値は共有の積公式を通り、1つの型付きshot resultになります。


# %%
@qm.qkernel
def qemcmc_samples(
    input_angles: qm.Param[qm.Vector[qm.Real, 6]],  # type: ignore[valid-type]
    hamiltonians: qm.Const[qm.Vector[qm.Observable, 2]],  # type: ignore[valid-type]
    time: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[6]]:  # type: ignore[valid-type]
    """Sample one runtime-rebindable quantum-enhanced MCMC proposal.

    Args:
        input_angles (qm.Param[qm.Vector[qm.Real, 6]]): Zero-or-pi rotations
            encoding the current Markov-chain bitstring.
        hamiltonians (qm.Const[qm.Vector[qm.Observable, 2]]): Mixer and Ising
            cost Hamiltonian terms.
        time (qm.Param[qm.Real]): Total proposal evolution time.

    Returns:
        qm.Shot[qm.Bits[6]]: Proposed six-spin bitstring.
    """

    state = qm.qubits(6)
    for index in range(6):
        state = qm.rx(state, input_angles[index], target=index)
    state = EVOLUTION(state, hamiltonians, time)
    return qm.measure(state)


program = qemcmc_samples.specialize(hamiltonians=HAMILTONIANS)

# %% [markdown]
# 特殊化は、記述したsignatureを消去せずに固定Ising modelを記録します。最終的なartifact ABIには`input_angles`と`time`だけが残ります。inspectからも、QeMCMCは特権的なcompiler operationではなく合成であることが分かります。

# %%
assert tuple(parameter.name for parameter in program.entry.parameters) == (
    "input_angles",
    "hamiltonians",
    "time",
)
assert tuple(name for name, _ in program.const_bindings) == ("hamiltonians",)
assert "quantum.qemcmc" not in program.inspect()
assert program.inspect().count("quantum.pauli_evolution") == 1

# %% [markdown]
# ## 同じbodyのresourceを見積もる
#
# 2項の2次公式では、各stepが$H_0/2,H_1,H_0/2$という回文構造のscheduleになります。そのため、logical analysisはTrotter stepごとに3つの抽象Pauli evolutionを厳密に報告します。backend固有のgate synthesisは後段で行われ、このsource-level countを変更できません。

# %%
study = qm.resources.logical(program)
pauli_evolutions = qm.resources.logical_operation_count("quantum.pauli_evolution")

expected_evolutions = 3 * TROTTER_STEPS
assert study.metric(pauli_evolutions).numeric_enclosure() == (
    expected_evolutions,
    expected_evolutions,
)
assert qm.resources.ResourceStudy.from_bytes(study.to_bytes()) == study

# %% [markdown]
# ## 1回コンパイルして1つの提案をsampleする
#
# 実行例は6量子ビットに対して128 shotsのlaunchを1回行います。実際のMarkov chainでは、現在のbitを0または$\pi$のangleに変換し、同じartifactへ繰り返しbindingします。

# %%
reference = qm.Reference()
artifact = qm.compile(program, to=reference.target)
execution = qm.run(
    artifact,
    on=reference,
    parameters={"input_angles": INPUT_ANGLES, "time": EVOLUTION_TIME},
    shots=128,
    seed=2,
)

assert tuple(slot.name for slot in artifact.parameter_abi.slots) == (
    "input_angles",
    "time",
)
assert execution.artifact is artifact
assert execution.invocation.values() == {
    "input_angles": INPUT_ANGLES,
    "time": EVOLUTION_TIME,
}
assert execution.result.shots == 128

# %% [markdown]
# ## hybrid Markov semanticsを厳密に検証する
#
# samplingは実行経路を示しますが、可逆性や定常性の証明には適しません。この6-spin例では、64個すべての提案行をhost上で厳密に構成できます。以下のhelperは、記述したprogramと同じSuzuki factorを辿ります。tutorialのassertion用oracleであり、別のcompiler実装ではありません。


# %%
def apply_x_evolution(
    amplitudes: list[complex],
    *,
    target: int,
    angle: float,
) -> list[complex]:
    """Apply exp(-i angle X_target) to a six-qubit host vector."""

    cosine = math.cos(angle)
    sine = -1.0j * math.sin(angle)
    return [
        cosine * amplitudes[basis] + sine * amplitudes[basis ^ (1 << target)]
        for basis in range(1 << N_SPINS)
    ]


def proposal_probabilities(
    input_bits: tuple[int, ...],
) -> tuple[float, ...]:
    """Return exact address-ordered proposal probabilities."""

    basis_index = sum(bit << index for index, bit in enumerate(input_bits))
    amplitudes = [0.0j] * (1 << N_SPINS)
    amplitudes[basis_index] = 1.0 + 0.0j
    step_time = EVOLUTION_TIME / TROTTER_STEPS
    mixer_angle = (1.0 - GAMMA) * step_time / 2.0
    for _ in range(TROTTER_STEPS):
        for target in range(N_SPINS):
            amplitudes = apply_x_evolution(
                amplitudes,
                target=target,
                angle=mixer_angle,
            )
        for basis in range(1 << N_SPINS):
            interaction = sum(
                (-1 if basis & (1 << index) else 1)
                * (-1 if basis & (1 << (index + 1)) else 1)
                for index in range(N_SPINS - 1)
            )
            amplitudes[basis] *= cmath.exp(1.0j * GAMMA * step_time * interaction)
        for target in range(N_SPINS):
            amplitudes = apply_x_evolution(
                amplitudes,
                target=target,
                angle=mixer_angle,
            )
    return tuple(abs(amplitude) ** 2 for amplitude in amplitudes)


def ising_energy(bits: tuple[int, ...]) -> float:
    """Return the six-spin ferromagnetic Ising-chain energy."""

    spins = tuple(1 - 2 * bit for bit in bits)
    return -float(sum(spins[index] * spins[index + 1] for index in range(N_SPINS - 1)))


states = tuple(
    tuple((basis >> index) & 1 for index in range(N_SPINS))
    for basis in range(1 << N_SPINS)
)
proposal = tuple(proposal_probabilities(bits) for bits in states)
energies = tuple(ising_energy(bits) for bits in states)
weights = tuple(math.exp(-BETA * energy) for energy in energies)
partition = sum(weights)
stationary = tuple(weight / partition for weight in weights)
state_count = 1 << N_SPINS
transition: list[list[float]] = [[0.0] * state_count for _ in range(state_count)]

for source in range(state_count):
    for target in range(state_count):
        if source == target:
            continue
        acceptance = min(
            1.0,
            math.exp(-BETA * (energies[target] - energies[source])),
        )
        transition[source][target] = proposal[source][target] * acceptance
    transition[source][source] = 1.0 - sum(transition[source])

for source in range(state_count):
    assert math.isclose(sum(proposal[source]), 1.0, abs_tol=1e-12)
    assert math.isclose(sum(transition[source]), 1.0, abs_tol=1e-12)
    for target in range(state_count):
        assert math.isclose(
            proposal[source][target],
            proposal[target][source],
            abs_tol=1e-12,
        )
        assert math.isclose(
            stationary[source] * transition[source][target],
            stationary[target] * transition[target][source],
            abs_tol=1e-12,
        )

input_basis = sum(bit << index for index, bit in enumerate(INPUT_BITS))
for basis, expected in enumerate(proposal[input_basis]):
    bits = tuple((basis >> index) & 1 for index in range(N_SPINS))
    sampling_error = 6.0 * math.sqrt(expected * (1.0 - expected) / 128)
    assert abs(execution.result.probability(bits) - expected) <= sampling_error + 0.01

# %% [markdown]
# 最初の等式は各提案行と遷移行が正規化されていることを示します。提案の対称性によりMetropolisの式を簡略化でき、最後の等式はBoltzmann分布に対する詳細釣り合いを示します。これにより、乱数、採択、Markov chain policyをQCoreへ移さずに量子・古典部分を合成できます。

# %% [markdown]
# ---
# ## 実行例

# %% [markdown]
# 実装したQeMCMCアルゴリズムを実行してみましょう。先ほどよりも低温の $\beta = 1.0$ に設定し、古典MCMCの局所更新では混合が遅くなる条件で量子提案分布の挙動を観察します。公平な比較のため、同じ $\beta = 1.0$ で古典MCMCも併走させます。

# %%
rng = random.Random(7)
current = (0,) * N_SPINS
chain = [current]
accepted = 0
CHAIN_STEPS = 12 if docs_test_mode else 1_000

for step in range(CHAIN_STEPS):
    proposal_result = qm.run(
        artifact,
        on=reference,
        parameters={
            "input_angles": tuple(math.pi * bit for bit in current),
            "time": EVOLUTION_TIME,
        },
        shots=1,
        seed=100 + step,
    ).result
    candidate = next(iter(proposal_result.counts))
    acceptance = min(
        1.0,
        math.exp(-BETA * (ising_energy(candidate) - ising_energy(current))),
    )
    if rng.random() < acceptance:
        current = candidate
        accepted += 1
    chain.append(current)

print("chain endpoints:", chain[:5], "...", chain[-5:])
print("accepted transitions:", accepted)
assert len(chain) == CHAIN_STEPS + 1
assert all(state in states for state in chain)

# %% [markdown]
# ### 古典local-update MCMCとの比較
#
# 元のチュートリアルと同じく、同じ逆温度で古典MCMCとQeMCMCのmagnetization収束を比較します。6-spin modelなら上で量子遷移matrix全体を検証しながら、元の比較を保持できます。spin-flip対称性により厳密な平衡magnetizationは0です。


# %%
def local_update(bits: tuple[int, ...], generator: random.Random) -> tuple[int, ...]:
    """Return a proposal with one uniformly selected spin flipped."""

    index = generator.randrange(N_SPINS)
    proposal_bits = list(bits)
    proposal_bits[index] = 1 - proposal_bits[index]
    return tuple(proposal_bits)


def accept_metropolis(
    source: tuple[int, ...],
    candidate: tuple[int, ...],
    generator: random.Random,
) -> tuple[int, ...]:
    """Accept or reject one proposal using the shared Ising objective."""

    probability = min(
        1.0,
        math.exp(-BETA * (ising_energy(candidate) - ising_energy(source))),
    )
    return candidate if generator.random() < probability else source


classical_rng = random.Random(7)
classical_current = (0,) * N_SPINS
classical_chain = [classical_current]
for _ in range(CHAIN_STEPS):
    classical_candidate = local_update(classical_current, classical_rng)
    classical_current = accept_metropolis(
        classical_current,
        classical_candidate,
        classical_rng,
    )
    classical_chain.append(classical_current)


def cumulative_magnetization(samples: list[tuple[int, ...]]) -> np.ndarray:
    """Return the running mean magnetization of a bitstring chain."""

    per_sample = np.asarray(
        [sum(1 - 2 * bit for bit in bits) / N_SPINS for bits in samples]
    )
    return np.cumsum(per_sample) / np.arange(1, len(per_sample) + 1)


quantum_magnetization = cumulative_magnetization(chain)
classical_magnetization = cumulative_magnetization(classical_chain)
theoretical_magnetization = sum(
    probability * sum(1 - 2 * bit for bit in bits) / N_SPINS
    for probability, bits in zip(stationary, states, strict=True)
)

assert math.isclose(theoretical_magnetization, 0.0, abs_tol=1e-12)
assert len(classical_chain) == len(chain) == CHAIN_STEPS + 1

plt.plot(classical_magnetization, label="MCMC estimate")
plt.plot(quantum_magnetization, label="QeMCMC estimate")
plt.axhline(theoretical_magnetization, linestyle="--", label="Theoretical")
plt.xlabel("Markov step")
plt.ylabel("Average magnetization")
plt.legend()
plt.show()

# %% [markdown]
# ---
# ## まとめ
#
# 本チュートリアルでは、古典的なMetropolis-Hastings法によるMCMCの復習からはじめ、量子回路$U(\gamma, t) = \exp(-i t H)$（$H = (1-\gamma) H_M + \gamma H_C$）を提案分布として利用するQuantum-enhanced MCMC (QeMCMC) をQamomile上で実装しました。具体的には、Qamomileのobservableでミキサー/コストHamiltonianを準備したうえで、`qkernel`の中で`suzuki_trotter`を用いたSuzuki-Trotter時間発展によるproposal circuitを構築しました。最後に、proposalを`Reference` target向けに1回compileして既存のMH loopに組み込み、古典MCMCとQeMCMCを同一のIsing chain上で走らせて平均磁化の収束を比較することで、一連の量子古典hybrid loopが意図どおり動作することを確認しました。
