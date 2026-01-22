# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# # 量子位相推定 (QPE) チュートリアル
# 
# このチュートリアルでは、Qamomileを使用して量子位相推定（QPE）アルゴリズムを実装する方法を説明します。
# 
# ## 自ら実装する量子位相推定
# まずは、Qamomileの基本的な量子ゲートを使用してQPEを実装してみましょう。
# 
# ### 逆量子フーリエ変換 (IQFT)
#
# 逆量子フーリエ変換は、QPEアルゴリズムの重要な部分です。以下にIQFTを実装します。
# 

# %%
import math
import qamomile.circuit as qmc


@qmc.qkernel
def iqft(qubits: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Inverse Quantum Fourier Transform (IQFT) on a vector of qubits."""
    n = qubits.shape[0]
    for j in qmc.range(n // 2):
        qubits[j], qubits[n - j - 1] = qmc.swap(qubits[j], qubits[n - j - 1])
    for j in qmc.range(n):
        for k in qmc.range(j):
            angle = -math.pi / (2 ** (j - k))
            qubits[j], qubits[k] = qmc.cp(qubits[j], qubits[k], angle)
        qubits[j] = qmc.h(qubits[j])
    return qubits



# %% [markdown]
# ### Phase Gate の定義
# 今回はQPEのターゲットとしてPhase Gateを使用します。Phase Gateは以下のように定義されます。
# $$P(\theta)|1\rangle = e^{i\theta}|1\rangle$$
# ここで、$|1\rangle$は固有状態であり、$e^{i\theta}$は対応する固有値です。
# この固有値をQPEで推定します。

# %%
@qmc.qkernel
def phase_gate(q: qmc.Qubit, theta: float, iter: int) -> qmc.Qubit:
    """Phase gate: P(θ)|1⟩ = e^{iθ}|1⟩"""
    for _ in qmc.range(iter):
        q = qmc.p(q, theta)
    return q

# %%
# QPEの実装
@qmc.qkernel
def qpe(phase: float) -> qmc.Vector[qmc.Bit]:
    phase_register = qmc.qubit_array(3, name="phase_reg")
    target = qmc.qubit(name="target")

    target = qmc.x(target)  # |0⟩ → |1⟩

    controlled_phase_gate = qmc.controlled(phase_gate)

    # Superposition preparation
    n = phase_register.shape[0]
    for i in qmc.range(n):
        phase_register[i] = qmc.h(phase_register[i])

    # QPEアルゴリズムの適用
    # controlled() API: (control, target, **params) -> (control_out, target_out)
    # 制御qubit i に対して 2^i 回のphase gateを適用
    for i in qmc.range(3):
        phase_register[i], target = controlled_phase_gate(phase_register[i], target, theta=phase, iter=2**i)
    iqft(phase_register)

    bits = qmc.measure(phase_register)

    return bits

# %% [markdown]
# ### 異なる量子SDKでのQPE実行
#
# Qamomileは複数の量子SDKをサポートしています。お好みのバックエンドを選択してください:
#
# ::::{tab-set}
# :::{tab-item} Qiskit
# :sync: sdk
#
# ```python
# from qamomile.qiskit import QiskitTranspiler
#
# transpiler = QiskitTranspiler()
# executable = transpiler.transpile(qpe, bindings={"phase": math.pi / 2})
#
# job = executable.sample(transpiler.executor(), shots=1024)
# sample_result = job.result()
# ```
#
# :::
# :::{tab-item} Quri-Parts
# :sync: sdk
#
# ```python
# from qamomile.quri_parts import QuriPartsCircuitTranspiler
#
# transpiler = QuriPartsCircuitTranspiler()
# executable = transpiler.transpile(qpe, bindings={"phase": math.pi / 2})
#
# # シミュレーションにはquri-parts-qulacsが必要
# job = executable.sample(transpiler.executor(), shots=1024)
# sample_result = job.result()
# ```
#
# :::
# :::{tab-item} PennyLane
# :sync: sdk
#
# ```python
# from qamomile.pennylane import PennylaneTranspiler
#
# transpiler = PennylaneTranspiler()
# executable = transpiler.transpile(qpe, bindings={"phase": math.pi / 2})
#
# job = executable.sample(transpiler.executor(), shots=1024)
# sample_result = job.result()
# ```
#
# :::
# :::{tab-item} CUDA-Q
# :sync: sdk
#
# ```{note}
# CUDA-QはNVIDIA GPUを搭載したLinuxシステムでのみ利用可能です。
# ```
#
# ```python
# from qamomile.cudaq import CudaqTranspiler
#
# transpiler = CudaqTranspiler()
# executable = transpiler.transpile(qpe, bindings={"phase": math.pi / 2})
#
# job = executable.sample(transpiler.executor(), shots=1024)
# sample_result = job.result()
# ```
#
# :::
# ::::
#
# 以下のコードはQiskitを使用してQPEを実行します（メインの例）:

# %%
from qamomile.qiskit import QiskitTranspiler


transpiler = QiskitTranspiler()
executable = transpiler.transpile(qpe, bindings={"phase": math.pi / 2})

job = executable.sample(transpiler.executor(), shots=1024)
sample_result = job.result()

# Decode results
num_bits = 3
for bits, count in sample_result.results:
    phase_estimate = sum(bit * (1 / (2 ** (i + 1))) for i, bit in enumerate(reversed(bits)))
    print(f"Measured bits: {bits}, Count: {count}, Estimated phase: {phase_estimate:.4f}")


# %% [markdown]
# QPEを実装して実際に動かすことができることがわかりました。QiskitのTranspilerに設定されているExecutorはデフォルトはQiskit-Aerのシミュレータですが、自らExecutorを実装してQamomileのTranspilerに渡すことで、他のバックエンドでも実行可能です。
# 実際どういうQiskitの量子回路が生成されているかを確認しましょう。

# %%
qiskit_circuit = executable.get_first_circuit()
print(qiskit_circuit.draw(output="text"))

# %% [markdown]
# このように実装した回路がQiskitの量子回路として生成されていることがわかります。
# 次にQamomileで提供されているqpe()関数を使用して、同様のQPEを実装してみましょう。

# %% [markdown]
# ## Qamomileのqpe()関数を使用した量子位相推定
# 定義済みのqpe()関数を使用すると、より簡潔にQPEを実装できます。
#
# **重要**: `qmc.qpe()`は内部で自動的に`U^(2^k)`の繰り返しを行うため、
# ユニタリは**1回の適用のみ**を定義する必要があります。

# %%
# qmc.qpe()用のシンプルなphase gate（1回適用のみ）
@qmc.qkernel
def p_gate(q: qmc.Qubit, theta: float) -> qmc.Qubit:
    """Simple phase gate: P(θ)|1⟩ = e^{iθ}|1⟩"""
    return qmc.p(q, theta)

@qmc.qkernel
def qpe_3bit(phase: float) -> qmc.Float:
    q_phase = qmc.qubit_array(3, name="phase_reg")
    target = qmc.qubit(name="target")
    target = qmc.x(target)  # |0⟩ → |1⟩
    # p_gateを使用（qmc.qpe()が内部で2^k回繰り返す）
    phase_q: qmc.QFixed = qmc.qpe(target, q_phase, p_gate, theta=phase)
    return qmc.measure(phase_q)

# %% [markdown]
# phaseを格納するregisterを用意し、ターゲット状態を初期化した後、qpe()関数を呼び出すだけでQPEが実装できます。
# 測定結果はQFixed型で返されるため、measure()関数で測定してFloat型に変換します。measureは渡される型に応じて自動的にデコードを行います。
# 
# ### Qiskitを用いたシミュレーション実行
# 先ほどと同様にQiskitシミュレータで実行し、結果を確認します。

# %%
transpiler = QiskitTranspiler()
test_phase = math.pi / 2  # θ = π/2, expected output ≈ 0.25 (since θ/(2π) = 0.25)
executable = transpiler.transpile(qpe_3bit, bindings={"phase": test_phase})

executor = transpiler.executor()
job = executable.sample(executor)
result = job.result()
for value, count in result.results:
    print(f"Measured value: {value}, Count: {count}")

# %% [markdown]
# Qamomileのqpe()関数を使用しても、同様にQPEを実装して実行できることがわかりました。このようにqamomileではQFixed型を使用して量子固定小数点数を扱うことで、量子アルゴリズムの実装が簡素化されます。
# またその場合、measure()関数を使用してQFixedをFloatに変換するだけで、デコードも自動的に行われます。
# qpe関数を利用した場合の生成される量子回路も確認してみましょう。

# %%
qiskit_circuit = executable.get_first_circuit()
print(qiskit_circuit.draw(output="text"))

# %% [markdown]
# こちらも同様にQiskitの量子回路として生成されていることがわかります。
# Qamomileではbackend側でサポートされている演算があれば可能な限り、直接その演算を使用するように量子回路が生成されます。
# 例えば、QiskitではIQFTがネイティブにサポートされているため、QPEの中のIQFT部分も直接IQFTゲートとして生成されています。

