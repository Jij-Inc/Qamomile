# ---
# jupyter:
#   jupytext:
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
# # 重ね合わせとエンタングルメント
#
# これまでのチュートリアルでは、型システムと量子ゲートを紹介しました。
# 本チュートリアルでは、*ゲートの仕組み*から視点を移し、
# 量子コンピューティングを古典コンピューティングと根本的に異なるものにする*概念*に焦点を当てます。
# それが**重ね合わせ**、**干渉**、そして**エンタングルメント**です。
#
# ## このチュートリアルで学ぶこと
# - 重ね合わせ：量子コイン投げ
# - 位相と $|+\rangle$ / $|-\rangle$ 状態
# - 量子干渉：位相が結果を制御する仕組み
# - CNOTによるエンタングルメントとベル状態
# - GHZ状態：多量子ビットのエンタングルメント

# %%
import math

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 1. 重ね合わせ：量子コイン投げ
#
# 古典ビットは常に0か1のどちらかです。
# 量子ビット（qubit）は、両方の状態の**重ね合わせ**として
# 同時に存在することができます。
#
# アダマールゲート（H）は、確定した $|0\rangle$ 状態を次のように変換します：
#
# $$H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}} \equiv |+\rangle$$
#
# この状態を測定すると、等しい確率で0または1が得られます。
# しかし重要なのは、測定前の量子ビットは密かに0か1のどちらかであるわけでは*ない*ということです。
# 両方の可能性を真に同時に保持しているのです。
# これが「量子コイン投げ」を古典的なランダムコインと区別するものです。


# %%
@qmc.qkernel
def superposition() -> qmc.Bit:
    """Create a superposition state with the H gate."""
    q = qmc.qubit(name="q")
    q = qmc.h(q)
    return qmc.measure(q)


superposition.draw()

# %%
exec_super = transpiler.transpile(superposition)
result_super = exec_super.sample(transpiler.executor(), shots=1000).result()

print("=== Superposition Measurement Results ===")
for value, count in result_super.results:
    percentage = count / 1000 * 100
    print(f"  {value}: {count} ({percentage:.1f}%)")

# %% [markdown]
# ### 結果の解釈
#
# 測定結果はおおよそ50%が `0`、50%が `1` になります。
# 各回の測定結果はランダムですが、統計的な分布は予測可能です。
#
# 重要なポイント：測定されるまで、量子ビットは両方の状態を「同時に」保持しています。
# 測定の瞬間に、1つの確定した値に「崩壊」します。
# どの結果が現れるかは確率振幅によって決まります。
# 確率振幅は、確率だけでなく*位相*も表す数値であり、
# 次のセクションで重要になります。

# %% [markdown]
# ## 2. 位相と $|+\rangle$ / $|-\rangle$ 状態
#
# 先ほど作成した重ね合わせ状態は $|+\rangle$ と呼ばれます。
# もう1つの重ね合わせ状態 $|-\rangle$ があり、符号が異なります：
#
# - $|+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$ -- $H|0\rangle$ から得られる
# - $|-\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}$ -- $H|1\rangle$ から得られる
#
# どちらの状態も同じ測定統計を示します：50%の確率で0、50%の確率で1です。
# しかし、これらは*物理的に異なる*量子状態です。
#
# 違いは**相対位相**にあります -- $|-\rangle$ のマイナス符号です。
# この符号は単一の測定では見えませんが、
# これから見るように干渉を通じて検出可能になります。
#
# $|-\rangle$ を作成するには、まずXゲートで $|0\rangle$ を $|1\rangle$ に反転させ、
# 次にHを適用します。


# %%
@qmc.qkernel
def minus_state() -> qmc.Bit:
    """Create |-> = H|1>."""
    q = qmc.qubit(name="q")
    q = qmc.x(q)  # |0> -> |1>
    q = qmc.h(q)  # |1> -> |->
    return qmc.measure(q)


minus_state.draw()

# %%
exec_plus = transpiler.transpile(superposition)
exec_minus = transpiler.transpile(minus_state)

result_plus = exec_plus.sample(transpiler.executor(), shots=1000).result()
result_minus = exec_minus.sample(transpiler.executor(), shots=1000).result()

print("=== |+> State (from section 1) ===")
for value, count in result_plus.results:
    print(f"  {value}: {count}")

print("\n=== |-> State ===")
for value, count in result_minus.results:
    print(f"  {value}: {count}")

print("\nBoth are approximately 50/50, but they are different quantum states!")

# %% [markdown]
# ### 違いを明らかにする
#
# $|+\rangle$ と $|-\rangle$ が同じ測定結果を与えるなら、どうやって区別できるのでしょうか？
# アダマールゲートを**もう一度**適用します。$H^2 = I$ なので：
#
# - $H|+\rangle = H \cdot H|0\rangle = |0\rangle$ -- 常に **0** が測定される
# - $H|-\rangle = H \cdot H|1\rangle = |1\rangle$ -- 常に **1** が測定される
#
# 直接測定では「同じに見える」2つの状態に対して、同じゲート（H）を適用すると、
# 全く逆の決定論的な結果が得られます。


# %%
@qmc.qkernel
def reveal_plus() -> qmc.Bit:
    """H|+> = |0>: always measures 0."""
    q = qmc.qubit(name="q")
    q = qmc.h(q)  # |0> -> |+>
    q = qmc.h(q)  # |+> -> |0>
    return qmc.measure(q)


reveal_plus.draw()


# %%
@qmc.qkernel
def reveal_minus() -> qmc.Bit:
    """H|-> = |1>: always measures 1."""
    q = qmc.qubit(name="q")
    q = qmc.x(q)  # |0> -> |1>
    q = qmc.h(q)  # |1> -> |->
    q = qmc.h(q)  # |-> -> |1>
    return qmc.measure(q)


reveal_minus.draw()

# %%
exec_rp = transpiler.transpile(reveal_plus)
exec_rm = transpiler.transpile(reveal_minus)

result_rp = exec_rp.sample(transpiler.executor(), shots=1000).result()
result_rm = exec_rm.sample(transpiler.executor(), shots=1000).result()

print("=== Reveal |+> with H ===")
for value, count in result_rp.results:
    print(f"  {value}: {count}")

print("\n=== Reveal |-> with H ===")
for value, count in result_rm.results:
    print(f"  {value}: {count}")

# %% [markdown]
# $|+\rangle$ と $|-\rangle$ は直接測定すると同じ50/50の結果を示しますが、
# *同じ*ゲート（H）に対して異なる応答をします。
# 相対位相 -- あのマイナス符号 -- は、量子状態の実在する物理的な性質です。
#
# 2つ目のHゲートにより、振幅が（$|+\rangle$ では）足し合わされたり、
# （$|-\rangle$ では）打ち消し合ったりします。これが**干渉**です。
# 次のセクションで詳しく見ていきます。

# %% [markdown]
# ## 3. 量子干渉
#
# 干渉は、量子アルゴリズムに威力を与えるメカニズムです。
# 確率振幅は複素数であり、強め合ったり
# （**建設的干渉**）、打ち消し合ったり（**破壊的干渉**）します。
#
# 上の実験はすでに干渉の一例です。二重アダマール回路
# $H \cdot H = I$ は、振幅が干渉し合うことで機能しています。
# ステップごとに追ってみましょう：
#
# 1. $|0\rangle$ から開始。
# 2. 最初のHが $\frac{|0\rangle + |1\rangle}{\sqrt{2}}$ を作る。
# 3. 2つ目のHが各成分に作用する：
#    - $H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$
#    - $H|1\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}$
# 4. これらを足し合わせると：$|1\rangle$ の振幅が打ち消し合い（$+1/2$ と $-1/2$）、
#    $|0\rangle$ の振幅が強め合います（$+1/2$ と $+1/2$）。
# 5. 結果：確実に $|0\rangle$ が得られる。
#
# $|1\rangle$ 成分の打ち消しが破壊的干渉であり、
# $|0\rangle$ の強め合いが建設的干渉です。
# これが量子アルゴリズムの核心的なメカニズムです：
# 位相をうまく配置して、間違った答えが打ち消し合い、
# 正しい答えが強め合うようにするのです。

# %% [markdown]
# ### 位相で制御する干渉
#
# 二重アダマールは、より一般的なパターンの特殊なケースです。
# 2つのHゲートの間に**位相回転**（RZ）を挿入することで、
# 完全な建設的干渉から完全な破壊的干渉まで、滑らかに調整できます。
#
# 回路 $H \to RZ(\theta) \to H$ は以下の結果を生み出します：
#
# - $\theta = 0$ : $H \cdot H = I$ と同じ -- 常に **0**
# - $\theta = \pi$ : 完全な位相反転 -- 常に **1**
# - $\theta = \pi/2$ : 部分的な位相 -- 再び50/50


# %%
@qmc.qkernel
def phase_interference(theta: qmc.Float) -> qmc.Bit:
    """H -> RZ(theta) -> H: phase controls the outcome."""
    q = qmc.qubit(name="q")
    q = qmc.h(q)
    q = qmc.rz(q, theta)
    q = qmc.h(q)
    return qmc.measure(q)


phase_interference.draw()

# %%
for label, angle in [("0", 0.0), ("pi", math.pi), ("pi/2", math.pi / 2)]:
    exec_pi = transpiler.transpile(phase_interference, bindings={"theta": angle})
    result_pi = exec_pi.sample(transpiler.executor(), shots=1000).result()

    print(f"theta = {label}:")
    for value, count in result_pi.results:
        percentage = count / 1000 * 100
        print(f"  {value}: {count} ({percentage:.1f}%)")
    print()

# %% [markdown]
# RZゲートは、単一量子ビットの測定確率を変えることなく位相を回転させます。
# しかし、2つのHゲートに挟まれると、
# 位相が干渉の結果としてどちらの出力が有利になるかを直接制御します。
#
# これが量子アルゴリズムのテンプレートです：**情報を位相にエンコードし、
# 干渉を使ってそれを取り出す**。

# %% [markdown]
# ## 4. CNOTによるエンタングルメント
#
# 重ね合わせは、単一量子ビットを古典ビットより豊かなものにします。
# **エンタングルメント**は、古典的には存在し得ない
# 複数量子ビット間の相関を生み出します。
#
# 手順はシンプルです：1つの量子ビットを重ね合わせ状態にし、
# CNOTゲートを適用します。
#
# - **CNOT**（制御NOT）：制御量子ビットが $|1\rangle$ のとき、ターゲット量子ビットを反転させる。
#
# 制御量子ビットが重ね合わせ状態にあるとき、CNOTはその重ね合わせを
# 両方の量子ビットに広げ、エンタングルメントを生成します。
#
# 得られる状態は**ベル状態** $|\Phi^+\rangle$ です：
#
# $$|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$$
#
# この状態には顕著な性質があります：一方の量子ビットを測定すると、
# もう一方が即座に決まります。最初の量子ビットで0が得られれば、
# 2番目も必ず0です。1が得られれば、2番目も必ず1です。
# にもかかわらず、測定前にはどちらの量子ビットも確定した値を持っていません。


# %%
@qmc.qkernel
def bell_state() -> tuple[qmc.Bit, qmc.Bit]:
    """Create Bell state |Phi+> = (|00> + |11>)/sqrt(2)."""
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")

    # Step 1: Put q0 in superposition
    q0 = qmc.h(q0)

    # Step 2: Entangle with CNOT
    q0, q1 = qmc.cx(q0, q1)

    return qmc.measure(q0), qmc.measure(q1)


bell_state.draw()

# %%
exec_bell = transpiler.transpile(bell_state)
result_bell = exec_bell.sample(transpiler.executor(), shots=1000).result()

print("=== Bell State Measurement Results ===")
for value, count in result_bell.results:
    percentage = count / 1000 * 100
    print(f"  Result: {value}, Count: {count} ({percentage:.1f}%)")

# %% [markdown]
# ### エンタングルメントの特徴
#
# `(0, 0)` と `(1, 1)` のみが現れ、`(0, 1)` や `(1, 0)` は決して現れません。
#
# これがエンタングルメントの特徴です：
#
# - 2つの量子ビットは**完全に相関**しています。
# - 一方が0と測定されれば、もう一方も必ず0です。
# - 一方が1と測定されれば、もう一方も必ず1です。
# - ただし、どちらのペアが現れるかは完全にランダムです（50/50）。
#
# この相関は、各量子ビットが秘密裏に予め決められた値を持っている
# という古典的なメカニズムでは説明できません。
# これは真に量子力学的な現象です。

# %% [markdown]
# ## 5. 4つのベル状態
#
# ベル状態 $|\Phi^+\rangle$ は、**ベル基底**と呼ばれる
# 4つの最大エンタングルメント2量子ビット状態の1つです。
# これらは合わせて、2量子ビットエンタングルメントの完全基底を形成します。
#
# | 名前 | 回路 | 式 | 測定結果 |
# |------|---------|---------|----------|
# | Phi+ | H, CX | (\|00> + \|11>) / sqrt(2) | 同値: (0,0) or (1,1) |
# | Phi- | H, CX, P(pi) | (\|00> - \|11>) / sqrt(2) | 同値: (0,0) or (1,1) |
# | Psi+ | H, CX, X | (\|01> + \|10>) / sqrt(2) | 反対: (0,1) or (1,0) |
# | Psi- | H, CX, X, P(pi) | (\|01> - \|10>) / sqrt(2) | 反対: (0,1) or (1,0) |
#
# Phi状態は同じ値のペアを生成し、Psi状態は逆の値のペアを生成します。
# +/- の符号は位相の違いであり、測定結果の確率に直接影響しませんが、
# 干渉や量子情報プロトコルにおいて重要になります。
#
# 位相反転には $\theta = \pi$ のPゲートを使います。
# これはまさにZゲートです：$P(\pi) = \begin{pmatrix}1&0\\0&-1\end{pmatrix}$。


# %%
@qmc.qkernel
def bell_phi_plus() -> tuple[qmc.Bit, qmc.Bit]:
    """Bell state |Phi+> = (|00> + |11>)/sqrt(2)."""
    q0, q1 = qmc.qubit(name="q0"), qmc.qubit(name="q1")
    q0 = qmc.h(q0)
    q0, q1 = qmc.cx(q0, q1)
    return qmc.measure(q0), qmc.measure(q1)


bell_phi_plus.draw()


# %%
@qmc.qkernel
def bell_phi_minus() -> tuple[qmc.Bit, qmc.Bit]:
    """Bell state |Phi-> = (|00> - |11>)/sqrt(2)."""
    q0, q1 = qmc.qubit(name="q0"), qmc.qubit(name="q1")
    q0 = qmc.h(q0)
    q0, q1 = qmc.cx(q0, q1)
    q0 = qmc.p(q0, math.pi)  # Z gate: phase flip
    return qmc.measure(q0), qmc.measure(q1)


bell_phi_minus.draw()


# %%
@qmc.qkernel
def bell_psi_plus() -> tuple[qmc.Bit, qmc.Bit]:
    """Bell state |Psi+> = (|01> + |10>)/sqrt(2)."""
    q0, q1 = qmc.qubit(name="q0"), qmc.qubit(name="q1")
    q0 = qmc.h(q0)
    q0, q1 = qmc.cx(q0, q1)
    q1 = qmc.x(q1)  # Flip target
    return qmc.measure(q0), qmc.measure(q1)


bell_psi_plus.draw()


# %%
@qmc.qkernel
def bell_psi_minus() -> tuple[qmc.Bit, qmc.Bit]:
    """Bell state |Psi-> = (|01> - |10>)/sqrt(2)."""
    q0, q1 = qmc.qubit(name="q0"), qmc.qubit(name="q1")
    q0 = qmc.h(q0)
    q0, q1 = qmc.cx(q0, q1)
    q1 = qmc.x(q1)
    q0 = qmc.p(q0, math.pi)  # Z gate: phase flip
    return qmc.measure(q0), qmc.measure(q1)


bell_psi_minus.draw()

# %%
bell_states = [
    ("Phi+", bell_phi_plus),
    ("Phi-", bell_phi_minus),
    ("Psi+", bell_psi_plus),
    ("Psi-", bell_psi_minus),
]

print("=== The Four Bell States ===\n")

for name, circuit in bell_states:
    exec_b = transpiler.transpile(circuit)
    result_b = exec_b.sample(transpiler.executor(), shots=1000).result()

    print(f"|{name}>:")
    for value, count in result_b.results:
        percentage = count / 1000 * 100
        print(f"  {value}: {count} ({percentage:.1f}%)")
    print()

# %% [markdown]
# ### 観察結果
#
# - **Phi+** と **Phi-** はどちらも (0,0) と (1,1) を生成します。
#   両者の違いは相対位相のみであり、これらの測定では見えません。
# - **Psi+** と **Psi-** はどちらも (0,1) と (1,0) を生成します。
#   こちらも位相の違いはここでは見えません。
#
# 位相の違いは、量子テレポーテーションや超密度符号化などの
# より高度なプロトコルで重要になります。これらのプロトコルでは、
# ベル状態間の干渉が利用されます。

# %% [markdown]
# ## 6. GHZ状態：多量子ビットのエンタングルメント
#
# エンタングルメントは2量子ビットに限られません。
# **GHZ状態**（Greenberger-Horne-Zeilinger状態）は、
# ベル状態をN量子ビットに一般化したものです：
#
# $$|GHZ_N\rangle = \frac{|00\ldots0\rangle + |11\ldots1\rangle}{\sqrt{2}}$$
#
# N個全ての量子ビットがエンタングルしています：いずれか1つを0と測定すると、
# 残り全ても0になり、1の場合も同様です。
#
# 構成方法はベル状態と同じパターンに従います：
# 最初の量子ビットを重ね合わせ状態にし、次にCNOTゲートを連鎖させて
# エンタングルメントを他の全ての量子ビットに広げます。
#
# ここでは、`qmc.qubit_array()` と `qmc.range()`（チュートリアル02で紹介）を
# 使って、一般的なN量子ビットGHZ回路を記述します。


# %%
@qmc.qkernel
def ghz_state(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Create an N-qubit GHZ state."""
    qubits = qmc.qubit_array(n, name="q")

    # Put the first qubit in superposition
    qubits[0] = qmc.h(qubits[0])

    # Chain CNOT gates to spread entanglement
    for i in qmc.range(n - 1):
        qubits[i], qubits[i + 1] = qmc.cx(qubits[i], qubits[i + 1])

    return qmc.measure(qubits)


ghz_state.draw(n=4, fold_loops=False)

# %%
# 3-qubit GHZ state
exec_ghz3 = transpiler.transpile(ghz_state, bindings={"n": 3})
result_ghz3 = exec_ghz3.sample(transpiler.executor(), shots=1000).result()

print("=== 3-Qubit GHZ State ===")
print("|GHZ> = (|000> + |111>)/sqrt(2)\n")
for value, count in result_ghz3.results:
    percentage = count / 1000 * 100
    print(f"  {value}: {count} ({percentage:.1f}%)")

# %%
# 5-qubit GHZ state
exec_ghz5 = transpiler.transpile(ghz_state, bindings={"n": 5})
result_ghz5 = exec_ghz5.sample(transpiler.executor(), shots=1000).result()

print("=== 5-Qubit GHZ State ===")
print("|GHZ> = (|00000> + |11111>)/sqrt(2)\n")
for value, count in result_ghz5.results:
    percentage = count / 1000 * 100
    print(f"  {value}: {count} ({percentage:.1f}%)")

# %% [markdown]
# ### GHZ状態の特徴
#
# 量子ビットの数に関係なく、現れる結果は全て0か全て1の2通りだけです。
# 全ての量子ビットが他の全ての量子ビットと完全に相関しています。
#
# GHZ状態は以下の分野で利用されます：
# - 量子非局所性の検証（N > 2 でのベル不等式の破れ）
# - 量子誤り訂正
# - 量子秘密分散プロトコル

# %% [markdown]
# ## 7. まとめ
#
# 本チュートリアルでは、量子コンピューティングの基盤となる3つの概念を探求しました。
#
# ### 重ね合わせ
# - Hゲートは量子ビットを $|0\rangle$ と $|1\rangle$ の
#   同時的な状態にします。
# - 測定により状態は1つの結果に崩壊し、その確率は
#   振幅によって決まります。
#
# ### 干渉
# - 確率振幅は強め合ったり（建設的）、打ち消し合ったり
#   （破壊的）します。
# - 二重アダマール実験（$H \cdot H = I$）はこれを実証します：
#   ランダム性が導入され、完全に元に戻されます。
# - 量子アルゴリズムは干渉を利用して、正しい答えを増幅し、
#   間違った答えを抑制します。
#
# ### エンタングルメント
# - H + CNOTの組み合わせにより、量子ビットが完全に相関した
#   エンタングル状態が生成されます。
# - 4つの**ベル状態**は、基本的な2量子ビットエンタングル状態です。
# - **GHZ状態**はエンタングルメントをN量子ビットに拡張します。
#
# これら3つの現象 -- 重ね合わせ、干渉、エンタングルメント -- が、
# 量子アルゴリズムにその力を与える要素です。
# 次のチュートリアルでは、Qamomileの
# [標準ライブラリ](05_stdlib.ipynb) -- QFTやQPEなど、
# まさにこれらの原理に基づいた既製の構成要素を探求します。

# %% [markdown]
# ## このチュートリアルで学んだこと
#
# - **重ね合わせ：量子コイン投げ** -- Hゲートは量子ビットを $|0\rangle$ と $|1\rangle$ の均等な混合にし、50/50の測定結果を与えます。
# - **位相と $|+\rangle$ / $|-\rangle$ 状態** -- 計算基底で測定すると同一に見えますが、逆の位相により異なる干渉の振る舞いをします。
# - **量子干渉：位相が結果を制御する仕組み** -- 振幅は強め合ったり打ち消し合ったりします。二重アダマール実験（$H \cdot H = I$）は、ランダム性が導入され完全に元に戻されることを示します。
# - **CNOTによるエンタングルメントとベル状態** -- H + CNOTは古典的には存在し得ない相関を生み出します。4つのベル状態は2量子ビットエンタングルメントの基本基底を形成します。
# - **GHZ状態：多量子ビットのエンタングルメント** -- GHZはCNOTゲートの連鎖を使って、全か無かの完全な相関を $N$ 量子ビットに拡張します。
