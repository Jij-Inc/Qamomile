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
# # 量子誤り訂正入門
#
# 本チュートリアルは、量子誤り訂正(Quantum Error Correction; QEC)を**実装しながら学ぶ教科書スタイル**の解説です。古典の繰り返し符号からはじめ、3量子ビットbit-flip符号、phase-flip符号、そしてShorの9量子ビット符号までを段階的に構築し、各符号をQamomileの`@qkernel`で実装してQiskitバックエンド上で**実際に動かして**訂正能力を検証します。最後にスタビライザー形式を導入し、表面符号など発展的なトピックへの橋渡しを行います。
#
# このチュートリアルで身につくこと：
#
# - なぜ量子誤り訂正が必要なのか、古典の繰り返し符号と何が違うのか
# - 3量子ビットbit-flip符号によるエンコード/デコードの実装
# - phase-flip符号がbit-flip符号とアダマール変換で結びつくこと
# - Shorの9量子ビット符号で任意の単一量子ビット誤り(X, Y, Z)が訂正できる仕組み
# - スタビライザー形式による符号の統一的記述
#
# 参考文献：本チュートリアルの内容はNielsen & Chuang「Quantum Computation and Quantum Information」第10章、およびGottesman「Stabilizer Codes and Quantum Error Correction」(arXiv:quant-ph/9705052)に基づいています。

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install qamomile

# %%
import math

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 1. なぜ量子誤り訂正が必要か
#
# 量子計算の最大の敵は**ノイズ**です。実機の量子ビットは環境との相互作用によりデコヒーレンスを起こし、状態が乱れていきます。古典計算機ではトランジスタ自体の誤り率は十分低く、さらに ECC 付きメモリやチェックサム等の補助的な誤り訂正で実効ビット誤り率は $10^{-15}$ 以下にまで抑えられているため、誤り訂正は通常意識されません。一方、現在の超伝導量子ビットの単一ゲート誤り率は$10^{-3} \sim 10^{-4}$程度であり、有用な量子アルゴリズムを実行するには誤り訂正が**必須**です。
#
# ### 古典の繰り返し符号
#
# 古典の通信では、ビット$b$を3つに複製して$bbb$として送り、受信側では多数決で復元する**繰り返し符号**(repetition code)が最も単純な誤り訂正です。1ビットの誤りであれば、3ビット中2ビットが正しいので多数決で正しい値を取り戻せます。
#
# ### 量子に持ち込むときの困難
#
# この方法を量子に持ち込もうとすると、すぐに二つの壁にぶつかります：
#
# 1. **複製禁止定理(no-cloning theorem)**：未知の量子状態$\lvert\psi\rangle$を$\lvert\psi\rangle\lvert\psi\rangle\lvert\psi\rangle$のように単純にコピーすることはできません。
# 2. **測定が状態を壊す**：誤りを検出するために量子ビットを測定すると、重ね合わせが崩壊し、肝心の論理状態が失われてしまいます。
#
# 量子誤り訂正の核心は、この二つの困難を**エンタングルメント**と**シンドローム測定**(状態そのものを測らずに誤りの種類だけを取り出す測定)で回避することにあります。エンコードによって論理情報を複数の物理量子ビットの相関に分散させ、補助量子ビットを介して誤りの「種類」だけを観測します。

# %% [markdown]
# ## 2. 3量子ビットbit-flip符号
#
# 最もシンプルな量子符号として、**bit-flipエラー**($X$演算子による$\lvert 0\rangle \leftrightarrow \lvert 1\rangle$の入れ替わり)だけを訂正する3量子ビット符号を構築します。これはあくまで教育用の符号で、phaseエラーは訂正できません。
#
# ### エンコード
#
# 論理状態$\alpha\lvert 0\rangle + \beta\lvert 1\rangle$を以下のように3量子ビットに分散させます：
#
# $$\alpha\lvert 0\rangle + \beta\lvert 1\rangle \quad\longrightarrow\quad \alpha\lvert 000\rangle + \beta\lvert 111\rangle$$
#
# これは**コピー**ではないことに注意してください。情報量子ビット$q_0$と補助量子ビット$q_1, q_2$をCNOTでエンタングルさせることで作っています。複製禁止定理には抵触しません。

# %% [markdown]
# ### Qamomileによる実装
#
# Qamomileの`@qkernel`でエンコーダとデコーダをそれぞれヘルパーとして定義し、後で本体カーネルから呼び出します。`@qkernel`ヘルパーは呼び出し元にインライン展開されます。


# %%
@qmc.qkernel
def encode_3qubit_bitflip(
    q0: qmc.Qubit, q1: qmc.Qubit, q2: qmc.Qubit
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    # q0 が論理量子ビット。q1, q2 は |0⟩ で初期化されている前提。
    q0, q1 = qmc.cx(q0, q1)
    q0, q2 = qmc.cx(q0, q2)
    return q0, q1, q2


# %% [markdown]
# ### デコーダ：エンコードの逆 + Toffoli
#
# 3量子ビット符号の最も簡潔な訂正回路は、エンコーダの逆(2つのCNOT)に**Toffoli(CCX)ゲート**を1つ加える形です。これは「測定を介さない決定論的訂正回路」と呼ばれます：
#
# $$\text{Decode} = \text{CCX}(q_1, q_2; q_0) \cdot \text{CNOT}(q_0, q_2) \cdot \text{CNOT}(q_0, q_1)$$
#
# どのような単一bit-flipエラー(エラーなし、$X_0$, $X_1$, $X_2$のいずれか)が起きていても、デコーダ通過後は$q_0$に元の論理状態が復元され、$q_1, q_2$には誤り位置を示す**シンドローム**が積状態として残ります。


# %%
@qmc.qkernel
def decode_3qubit_bitflip(
    q0: qmc.Qubit, q1: qmc.Qubit, q2: qmc.Qubit
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    q0, q1 = qmc.cx(q0, q1)
    q0, q2 = qmc.cx(q0, q2)
    q1, q2, q0 = qmc.ccx(q1, q2, q0)
    return q0, q1, q2


# %% [markdown]
# ### 動作検証:bit-flipエラーをすべての位置に注入
#
# `errors`は「X を適用すべき量子ビットインデックスの集合」を表します。Qamomile の `Dict[UInt, Bit]` で「キーが量子ビット index、value は単なるフラグ」と表現し、`for idx, _ in qmc.items(errors):` で各キーを走査します(value は使わないので `_` で受ける)。Set 相当のセマンティクスを Dict で実現する形です。
#
# §4 の Shor 符号では同じ `errors` 引数を `Dict[UInt, UInt]`(value は X/Y/Z の type code)として再利用し、value を使い分けます — 本節では使わないだけ。


# %%
@qmc.qkernel
def bitflip_code_run(
    errors: qmc.Dict[qmc.UInt, qmc.Bit],
    theta: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(3, name="q")

    # 論理状態 cos(theta/2)|0⟩ + sin(theta/2)|1⟩ を q[0] に準備
    q[0] = qmc.ry(q[0], theta)

    # エンコード
    q[0], q[1], q[2] = encode_3qubit_bitflip(q[0], q[1], q[2])

    # bit-flipエラーを指定位置に注入(`errors` のキーに含まれる位置だけに X を作用)
    for idx, _ in qmc.items(errors):
        q[idx] = qmc.x(q[idx])

    # デコード
    q[0], q[1], q[2] = decode_3qubit_bitflip(q[0], q[1], q[2])

    return qmc.measure(q)


# %% [markdown]
# 続いて、論理状態$\lvert 1\rangle$($\theta = \pi$, つまり$RY(\pi)\lvert 0\rangle = \lvert 1\rangle$)を準備し、bit-flipエラーを各位置に注入してデコード後の$q_0$を読みます。$q_0$は**常に1**を返すはずです。

# %%
scenarios: list[tuple[str, dict[int, bool]]] = [
    ("エラーなし", {}),
    ("X on q[0]", {0: True}),
    ("X on q[1]", {1: True}),
    ("X on q[2]", {2: True}),
]


def _first_bit_distribution(result):
    """測定結果のリスト [(outcome, count), ...] から最初のビットの分布を取り出す。

    本チュートリアルでは `qubit_array` の最初の要素(`q[0]` または `data[0]`)を
    論理量子ビットとして扱い、その値が訂正後に保たれているかを確認する場面で使う。
    """
    counts = {0: 0, 1: 0}
    for outcome, count in result.results:
        if isinstance(outcome, (list, tuple)):
            q0_bit = outcome[0]
        else:
            q0_bit = outcome & 1
        counts[q0_bit] += count
    return counts


print("論理状態 |1⟩ をエンコード後、bit-flipエラーを注入してデコード:")
for label, err in scenarios:
    exe_err = transpiler.transpile(
        bitflip_code_run, bindings={"errors": err}, parameters=["theta"]
    )
    job = exe_err.sample(transpiler.executor(), shots=256, bindings={"theta": math.pi})
    counts = _first_bit_distribution(job.result())
    print(f"  {label:12s}: q[0]=0が{counts[0]}回, q[0]=1が{counts[1]}回")

# %% [markdown]
# どのエラー位置でも$q_0$は**100%の確率で1**を返します。つまり、単一bit-flipエラーは完璧に訂正されています。
#
# ### 重ね合わせ状態でも動くことの確認
#
# 計算基底だけでなく、$\theta = \pi/3$で準備した重ね合わせ状態$\cos(\pi/6)\lvert 0\rangle + \sin(\pi/6)\lvert 1\rangle$でも訂正が効くことを確認します。理論上、$q_0 = 1$が出る確率は$\sin^2(\pi/6) = 0.25$で、これはエラーの有無によらず一定のはずです。

# %%
print("論理状態 cos(π/6)|0⟩ + sin(π/6)|1⟩ (P(q[0]=1) = 0.25 が理論値):")
for label, err in scenarios:
    exe_err = transpiler.transpile(
        bitflip_code_run, bindings={"errors": err}, parameters=["theta"]
    )
    job = exe_err.sample(
        transpiler.executor(), shots=4000, bindings={"theta": math.pi / 3}
    )
    counts = _first_bit_distribution(job.result())
    total = counts[0] + counts[1]
    print(f"  {label:12s}: P(q[0]=1) ≈ {counts[1] / total:.3f}")

# %% [markdown]
# どのエラー位置でも$P(q_0 = 1) \approx 0.25$となっており、論理状態の振幅情報が誤り訂正後も正しく保たれていることがわかります。
#
# ### この符号で**訂正できないもの**
#
# 3量子ビットbit-flip符号は名前のとおりbit-flipエラーしか訂正できません。phaseエラー($Z$演算子)が起きると、$\alpha\lvert 000\rangle + \beta\lvert 111\rangle$が$\alpha\lvert 000\rangle - \beta\lvert 111\rangle$に変わり、デコード後は$\alpha\lvert 0\rangle - \beta\lvert 1\rangle$、つまり**位相反転した状態が`q[0]`に残ります**。これを訂正するには次節のphase-flip符号が必要です。

# %% [markdown]
# ## 3. 3量子ビットphase-flip符号
#
# bit-flipエラーが$X$、phase-flipエラーが$Z$で記述されることを思い出すと、両者は**アダマール変換**$H$で結ばれていることがわかります：
#
# $$H Z H = X, \qquad H X H = Z$$
#
# つまり、phase-flipエラーは**$H$基底で見ればbit-flipエラー**にすぎません。これを利用すると、phase-flip符号はbit-flip符号の各量子ビットを$H$でくるむだけで作れます。
#
# ### 論理基底
#
# - 論理$\lvert 0_L\rangle = \lvert {+}{+}{+}\rangle$
# - 論理$\lvert 1_L\rangle = \lvert {-}{-}{-}\rangle$
#
# エンコーダはbit-flip符号のエンコードの後、各量子ビットに$H$を作用させて作れます。デコーダはその逆順です。


# %%
@qmc.qkernel
def encode_3qubit_phaseflip(
    q0: qmc.Qubit, q1: qmc.Qubit, q2: qmc.Qubit
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    q0, q1, q2 = encode_3qubit_bitflip(q0, q1, q2)
    q0 = qmc.h(q0)
    q1 = qmc.h(q1)
    q2 = qmc.h(q2)
    return q0, q1, q2


@qmc.qkernel
def decode_3qubit_phaseflip(
    q0: qmc.Qubit, q1: qmc.Qubit, q2: qmc.Qubit
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    q0 = qmc.h(q0)
    q1 = qmc.h(q1)
    q2 = qmc.h(q2)
    q0, q1, q2 = decode_3qubit_bitflip(q0, q1, q2)
    return q0, q1, q2


# %% [markdown]
# ### 動作検証：phaseエラーを注入
#
# 単独のphase-flipエラー$Z$は、論理基底$\lvert 0\rangle$, $\lvert 1\rangle$には影響しません(これらは$Z$の固有状態だからです)。検証には**重ね合わせ状態**が必要です。ここでは$\lvert +\rangle = (\lvert 0\rangle + \lvert 1\rangle)/\sqrt{2}$をエンコードし、$Z$エラー注入後に最後に$H$をかけてから測定します。エラーが正しく訂正されていれば、結果は**常に0**になるはずです。


# %%
@qmc.qkernel
def phaseflip_code_run(
    errors: qmc.Dict[qmc.UInt, qmc.Bit],
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(3, name="q")

    # 論理 |+⟩ = H|0⟩ を準備
    q[0] = qmc.h(q[0])

    # エンコード
    q[0], q[1], q[2] = encode_3qubit_phaseflip(q[0], q[1], q[2])

    # phase-flipエラーを指定位置に注入
    for idx, _ in qmc.items(errors):
        q[idx] = qmc.z(q[idx])

    # デコード
    q[0], q[1], q[2] = decode_3qubit_phaseflip(q[0], q[1], q[2])

    # 訂正後 q[0] は論理 |+⟩、q[1]/q[2] は計算基底のシンドロームビットとして残る。
    # q[0] を計算基底で測定すると 50/50 でランダムなので、X 基底で読むため H を当てる
    # (q[1]/q[2] は計算基底でそのまま測れば良いので H 不要)。
    q[0] = qmc.h(q[0])

    return qmc.measure(q)


# %%
print("論理状態 |+⟩ にphase-flipエラーを注入してデコード(q[0]は常に0が期待値):")
for label, err in [
    ("エラーなし", {}),
    ("Z on q[0]", {0: True}),
    ("Z on q[1]", {1: True}),
    ("Z on q[2]", {2: True}),
]:
    exe_pz = transpiler.transpile(phaseflip_code_run, bindings={"errors": err})
    job = exe_pz.sample(transpiler.executor(), shots=256)
    counts = _first_bit_distribution(job.result())
    print(f"  {label:12s}: q[0]=0が{counts[0]}回, q[0]=1が{counts[1]}回")

# %% [markdown]
# どの位置に$Z$エラーが入っても$q_0 = 0$が100%返ります。phase-flip符号がbit-flip符号と完全に双対になっていることが確認できました。
#
# ### 残された問題
#
# bit-flip符号は$X$しか訂正できず、phase-flip符号は$Z$しか訂正できません。実機の量子ビットでは$X$, $Z$の両方、さらに$Y = iXZ$も同時に起こり得ます。**任意の単一量子ビット誤り**を訂正するには、両方の符号を組み合わせた**Shorの9量子ビット符号**が必要です。

# %% [markdown]
# ## 4. Shorの9量子ビット符号
#
# Shor(1995)が提案した9量子ビット符号は、**phase-flip符号(外側) ∘ bit-flip符号(内側)**という階層構造(連結符号)を持ちます：
#
# 1. まず1量子ビットを3量子ビットphase-flip符号でエンコード(3ブロック、各ブロック1量子ビット)。
# 2. 続いて各ブロックの1量子ビットを3量子ビットbit-flip符号でさらにエンコード(3量子ビット × 3ブロック = 9量子ビット)。
#
# これにより：
#
# - 各ブロック内のbit-flip($X$)エラーは内側のbit-flip符号で訂正される
# - ブロック間のphase-flip($Z$)エラーは外側のphase-flip符号で訂正される
# - $Y = iXZ$は$X$と$Z$が同時に起きたものとして同時に訂正される
#
# 結果として、**9つの物理量子ビットのうち任意の1つに任意の単一量子ビット誤りが起きても訂正できる**ようになります。

# %% [markdown]
# ### Shor符号の論理基底
#
# $$
# \lvert 0_L\rangle = \frac{1}{2\sqrt{2}}(\lvert 000\rangle + \lvert 111\rangle)(\lvert 000\rangle + \lvert 111\rangle)(\lvert 000\rangle + \lvert 111\rangle)
# $$
#
# $$
# \lvert 1_L\rangle = \frac{1}{2\sqrt{2}}(\lvert 000\rangle - \lvert 111\rangle)(\lvert 000\rangle - \lvert 111\rangle)(\lvert 000\rangle - \lvert 111\rangle)
# $$
#
# 9量子ビットを3つのブロック`{q0,q1,q2}, {q3,q4,q5}, {q6,q7,q8}`に分け、各ブロック先頭(`q0`, `q3`, `q6`)を**ブロックリーダー**と呼びます。エンコーダはブロックリーダーに対してphase-flip符号を、各ブロック内でbit-flip符号を適用します。連結構造をそのまま`@qkernel`ヘルパーに分離して書きます。


# %%
@qmc.qkernel
def encode_shor(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    # 外側: q[0] を q[3], q[6] と phase-flipエンコード
    q[0], q[3], q[6] = encode_3qubit_phaseflip(q[0], q[3], q[6])
    # 内側: 各ブロックリーダーをそのブロック内で bit-flipエンコード
    q[0], q[1], q[2] = encode_3qubit_bitflip(q[0], q[1], q[2])
    q[3], q[4], q[5] = encode_3qubit_bitflip(q[3], q[4], q[5])
    q[6], q[7], q[8] = encode_3qubit_bitflip(q[6], q[7], q[8])
    return q


@qmc.qkernel
def decode_shor(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    # 内側: 各ブロックを bit-flipデコード
    q[0], q[1], q[2] = decode_3qubit_bitflip(q[0], q[1], q[2])
    q[3], q[4], q[5] = decode_3qubit_bitflip(q[3], q[4], q[5])
    q[6], q[7], q[8] = decode_3qubit_bitflip(q[6], q[7], q[8])
    # 外側: ブロックリーダーを phase-flipデコード
    q[0], q[3], q[6] = decode_3qubit_phaseflip(q[0], q[3], q[6])
    return q


# %% [markdown]
# ### 任意の単一量子ビット誤りを訂正
#
# Shor符号の真価は$X$, $Y$, $Z$のいずれの単一量子ビット誤りに対しても訂正できる点にあります。エラーチャネルを「量子ビットインデックス → エラー種類」の辞書で表現し、エンコード後にコンパイル時アンロールで指定位置にPauli演算子を作用させます。

# %%
NO_ERROR = 0
X_ERROR = 1
Y_ERROR = 2
Z_ERROR = 3


@qmc.qkernel
def shor_code_run(
    errors: qmc.Dict[qmc.UInt, qmc.UInt],
    theta: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(9, name="q")

    # 論理状態 cos(theta/2)|0⟩ + sin(theta/2)|1⟩ を q[0] に準備
    q[0] = qmc.ry(q[0], theta)

    q = encode_shor(q)

    # 各量子ビットへ指定された種類のPauliエラーを注入。
    # `else` 分岐の `q[idx] = q[idx]` は no-op だが必要: Qamomile の affine 型システムは
    # 「if/else どちらの分岐でも qubit ハンドルが同じシェイプで再代入される」ことを
    # 要求するため、「何もしない」を明示的に書く。コンパイル時に etype が NO_ERROR と
    # 確定すればこの no-op はそのまま消える。
    for idx, etype in qmc.items(errors):
        if etype == X_ERROR:
            q[idx] = qmc.x(q[idx])
        elif etype == Y_ERROR:
            q[idx] = qmc.y(q[idx])
        elif etype == Z_ERROR:
            q[idx] = qmc.z(q[idx])
        else:
            q[idx] = q[idx]  # NO_ERROR: identity placeholder for affine-type completeness

    q = decode_shor(q)

    return qmc.measure(q)


# %% [markdown]
# 論理$\lvert 1\rangle$($\theta = \pi$)を準備し、9量子ビット中の代表的な位置に$X$, $Y$, $Z$誤りを順に注入してデコード後の$q_0$を確認します。理想的にはすべてのケースで$q_0 = 1$になります。

# %%
print("論理状態 |1⟩ をShor符号でエンコード後、各量子ビットへPauliエラーを注入:")
print(f"  {'エラー種類':10s} | {'位置':4s} | P(q[0]=1)")
print(f"  {'-' * 10}-+-{'-' * 4}-+-{'-' * 10}")

for etype_name, etype_code in [("X", X_ERROR), ("Y", Y_ERROR), ("Z", Z_ERROR)]:
    for idx in [0, 4, 8]:  # 3つの異なるブロックから1つずつ代表
        exe_s = transpiler.transpile(
            shor_code_run,
            bindings={"errors": {idx: etype_code}},
            parameters=["theta"],
        )
        job = exe_s.sample(
            transpiler.executor(), shots=256, bindings={"theta": math.pi}
        )
        counts = _first_bit_distribution(job.result())
        total = counts[0] + counts[1]
        print(f"  {etype_name:10s} | q[{idx}] | {counts[1] / total:.3f}")

# %% [markdown]
# すべての$X$, $Y$, $Z$エラーがどの位置に入っても$P(q_0 = 1) = 1.0$、つまり論理状態が完全に保持されていることがわかります。これがShor符号の威力です。
#
# ### なぜ$Y$エラーまで訂正できるのか
#
# $Y = iXZ$なので、$Y$エラーは「同じ量子ビットに$X$と$Z$が同時に起きた」と等価です。Shor符号は**$X$は内側のbit-flip符号で訂正**し、**$Z$は外側のphase-flip符号で訂正**するので、両者を独立に処理することで$Y$も訂正できます。さらに一般に、デコヒーレンスによる任意のチャネルは$\{I, X, Y, Z\}$の線型結合で表せるため、Shor符号は**任意の単一量子ビット誤り**を訂正します。これを**離散化定理**(discretization theorem)と呼び、量子誤り訂正がそもそも可能であることの根拠になっています。

# %% [markdown]
# ## 5. シンドローム測定による訂正(実機ハードウェアの方法)
#
# §2 から §4 で採用してきた**エンコーダの逆 + Toffoli**は、測定を介さずに訂正を行う数学的に洗練された回路でしたが、実機の量子ハードウェアで標準的に行われている QEC とは異なります。実機 QEC は次の手順を踏みます:
#
# 1. **補助量子ビット(ancilla)** を準備する
# 2. データ量子ビットと ancilla の間に CNOT を入れて**スタビライザー演算子をパリティ測定**する
# 3. ancilla を測定して**古典シンドローム**を得る
# 4. シンドローム値に応じて**訂正 Pauli を条件付き適用**する
#
# このパターンは「mid-circuit 測定」と「測定値に依存する古典フィードバック」を必要とします。Qamomile は `qmc.measure()` で得た `Bit` ハンドルに対する `&` / `|` / `~` オペレータと `if bit:` 構文で、このパターンをそのまま記述できます。
#
# 教科書的にはこちらが**正統な**訂正回路で、本節は実装をその形に書き直したデモです。

# %% [markdown]
# ### 3量子ビットbit-flip符号のスタビライザー
#
# bit-flip 符号のスタビライザー生成子は 2 つです:
#
# - $S_0 = Z_0 Z_1$ — 量子ビット 0 と 1 のパリティ
# - $S_1 = Z_1 Z_2$ — 量子ビット 1 と 2 のパリティ
#
# 符号空間 $\{\lvert 000\rangle, \lvert 111\rangle\}$ では両者とも+1固有値を持ちます。bit-flipエラー $X_i$ が起きると、$X_i$と非可換なスタビライザーは-1固有値を持つようになります:
#
# | エラー | $S_0 = Z_0Z_1$ | $S_1 = Z_1Z_2$ | シンドローム$(s_0, s_1)$ |
# |--------|----------------|----------------|--------------------------|
# | なし   | +1 | +1 | (0, 0) |
# | $X_0$  | -1 | +1 | (1, 0) |
# | $X_1$  | -1 | -1 | (1, 1) |
# | $X_2$  | +1 | -1 | (0, 1) |
#
# シンドローム$(s_0, s_1)$から誤り位置が**一意に特定できる**(decoder lookup) — これがスタビライザー符号の中核的な仕組みです。

# %% [markdown]
# ### Qamomileによる実装
#
# 5量子ビット使います:
#
# - `data[0..2]` — 論理状態を保持する 3 つのデータ量子ビット
# - `anc[0]` — $S_0 = Z_0 Z_1$ のパリティを抽出する ancilla
# - `anc[1]` — $S_1 = Z_1 Z_2$ のパリティを抽出する ancilla
#
# パリティ測定の回路は $S_0 = Z_0 Z_1$ について `CX(data[0], anc[0]); CX(data[1], anc[0]); measure(anc[0])`。これで `anc[0] = data[0] ⊕ data[1]` のパリティを得ます。$S_1$ も同様。


# %%
@qmc.qkernel
def syndrome_decode_bitflip(
    errors: qmc.Dict[qmc.UInt, qmc.Bit],
) -> qmc.Vector[qmc.Bit]:
    data = qmc.qubit_array(3, name="data")
    anc = qmc.qubit_array(2, name="anc")

    # 論理 |1⟩ を準備
    data[0] = qmc.x(data[0])

    # エンコード: |1⟩ → |111⟩
    data[0], data[1], data[2] = encode_3qubit_bitflip(data[0], data[1], data[2])

    # bit-flipエラーを指定位置に注入
    for idx, _ in qmc.items(errors):
        data[idx] = qmc.x(data[idx])

    # スタビライザー S0 = Z0 Z1 のパリティ測定
    data[0], anc[0] = qmc.cx(data[0], anc[0])
    data[1], anc[0] = qmc.cx(data[1], anc[0])
    s0 = qmc.measure(anc[0])

    # スタビライザー S1 = Z1 Z2 のパリティ測定
    data[1], anc[1] = qmc.cx(data[1], anc[1])
    data[2], anc[1] = qmc.cx(data[2], anc[1])
    s1 = qmc.measure(anc[1])

    # シンドローム表に基づく古典フィードバック訂正
    if s0 & ~s1:
        # (1, 0) → X0 エラー
        data[0] = qmc.x(data[0])
    if s0 & s1:
        # (1, 1) → X1 エラー
        data[1] = qmc.x(data[1])
    if ~s0 & s1:
        # (0, 1) → X2 エラー
        data[2] = qmc.x(data[2])

    return qmc.measure(data)


# %% [markdown]
# ### 動作検証
#
# 各エラー位置で訂正後の `data[0]` が **常に 1** を返すことを確認します(論理 |1⟩ をエンコードしたので)。

# %%
print("シンドローム測定による訂正(論理 |1⟩):")
for label, err in [
    ("エラーなし", {}),
    ("X on data[0]", {0: True}),
    ("X on data[1]", {1: True}),
    ("X on data[2]", {2: True}),
]:
    exe_syn = transpiler.transpile(
        syndrome_decode_bitflip, bindings={"errors": err}
    )
    job = exe_syn.sample(transpiler.executor(), shots=200)
    counts = _first_bit_distribution(job.result())
    total = counts[0] + counts[1]
    print(f"  {label:14s}: P(data[0]=1) = {counts[1] / total:.3f}")

# %% [markdown]
# どのエラー位置でも `data[0]` は 100% の確率で 1 を返します。決定論的訂正回路 (§2) と同じ訂正能力を、補助量子ビットによるスタビライザー測定 + 古典フィードバックという**実機ハードウェアと同じ枠組み**で実現できました。

# %% [markdown]
# ### 2 つのアプローチの比較
#
# | 観点                         | §2 決定論的(unitary)訂正  | §5 シンドローム測定訂正    |
# |------------------------------|---------------------------|----------------------------|
# | 補助量子ビット               | 不要                      | 必要(各スタビライザーに 1) |
# | 中間測定                     | 不要                      | 必須                        |
# | 古典フィードバック           | 不要                      | 必須                        |
# | 訂正回路の深さ               | 浅い(CNOT×2 + Toffoli×1)  | 中(CNOT×4 + 測定×2 + 条件付きX) |
# | スタビライザー形式との対応   | 間接的                    | 直接的(教科書通り)         |
# | 表面符号などへの一般化       | 困難                      | 自然                        |
# | 実機ハードウェアでの採用     | 極稀(教育用)              | 標準                        |
#
# Qamomile の `Bit` 型に対する `&` / `|` / `~` オペレータはこの動的回路パターンを宣言的に書くために設計されています。Qiskit バックエンドは内部的に `qiskit.circuit.classical.expr` API を使ってクラシカル式を `if_test` の条件として埋め込んでいます。

# %% [markdown]
# ## 6. スタビライザー形式
#
# §5 で 3量子ビット bit-flip 符号のスタビライザーを具体的に見ましたが、量子誤り訂正の現代的な記述言語が**スタビライザー形式**(stabilizer formalism)です。これにより、より大きな符号(表面符号など)も簡潔に記述できます。
#
# ### 定義
#
# Pauli群$\mathcal{P}_n$の可換な部分群$\mathcal{S}$を**スタビライザー群**と呼びます。$\mathcal{S}$によって安定化される(つまり$S\lvert\psi\rangle = \lvert\psi\rangle$をすべての$S \in \mathcal{S}$について満たす)状態の集合が**符号空間**です。
#
# 各符号のスタビライザー生成子は以下のとおりです:
#
# | 符号 | 生成子 | 訂正能力 |
# |------|------|---------|
# | 3量子ビットbit-flip | $Z_0 Z_1$, $Z_1 Z_2$ | 単一$X$誤り |
# | 3量子ビットphase-flip | $X_0 X_1$, $X_1 X_2$ | 単一$Z$誤り |
# | Shor 9量子ビット | $Z_0 Z_1$, $Z_1 Z_2$, $Z_3 Z_4$, $Z_4 Z_5$, $Z_6 Z_7$, $Z_7 Z_8$, $X_0 X_1 X_2 X_3 X_4 X_5$, $X_3 X_4 X_5 X_6 X_7 X_8$ | 任意の単一量子ビット誤り |
#
# ### より大きな符号への展開
#
# §5 で 3量子ビット bit-flip 符号の syndrome decoding を実装しましたが、同じ枠組みで phase-flip 符号(スタビライザーが $X_iX_j$ になる)、Shor 符号(8 つのスタビライザー)、そして表面符号(2D 格子上のローカルなスタビライザー)も記述できます。各スタビライザーごとに ancilla を 1 つ用意し、CNOT(または CNOT + H) でパリティ抽出 → 測定 → ルックアップで訂正 Pauli を決定 → 適用、という流れは共通です。
#
# 表面符号では数十〜数千の物理量子ビットからわずか 1 つの論理量子ビットを保護しますが、各サイクルでスタビライザー測定と古典訂正を繰り返すループ自体は §5 のデモと構造的に同じです。

# %% [markdown]
# ## 7. まとめと次への展望
#
# 本チュートリアルでは：
#
# - **3量子ビットbit-flip符号**で単一$X$エラーが訂正可能なことを実装・実機相当のシミュレーションで検証
# - **3量子ビットphase-flip符号**がbit-flip符号と$H$変換で双対であることを示し、$Z$エラーが訂正できることを確認
# - **Shor 9量子ビット符号**が連結構造により任意の単一量子ビット誤り($X$, $Y$, $Z$)を訂正できることを実証
# - **シンドローム測定 + 古典フィードバック**による訂正(実機ハードウェアの方法)を補助量子ビットと `&` / `~` オペレータで実装
# - **スタビライザー形式**による統一的な記述を導入
#
# ### この先に広がる世界
#
# Shor符号は教科書的にきれいですが、9量子ビットを使って1論理量子ビットしか保護できず、エラー率も高い符号です。実用的な量子誤り訂正符号としては以下の発展があります：
#
# - **Steane符号(7量子ビット)**：CSS符号(Calderbank-Shor-Steane)の代表例。横断的なClifford演算が可能。
# - **表面符号(Surface code)**：2次元格子上のスタビライザー符号。局所的な相互作用のみで実装でき、しきい値定理($\sim 1\%$のエラー率まで)が成立する。現在の超伝導量子コンピュータ実機(Google, IBM等)で主流。
# - **カラー符号(Color code)**：表面符号と同じ局所性を持ち、横断的なClifford群全体が実装可能。
# - **フォールトトレラント計算**：エンコード後の論理量子ビットに対する計算を、エラーが伝播・増幅しないように設計する技法。Magic state distillationで非Clifford演算($T$ゲート)を補強する。
#
# ### 次へ
#
# 量子誤り訂正は2020年代後半の量子コンピュータ実用化の最重要課題です。さらに学ぶには以下が出発点として有用です：
#
# - Nielsen & Chuang, "Quantum Computation and Quantum Information" 第10章
# - Gottesman, "Stabilizer Codes and Quantum Error Correction" (arXiv:quant-ph/9705052)
# - Fowler et al., "Surface codes: Towards practical large-scale quantum computation" (arXiv:1208.0928)
#
# Qamomileのチュートリアルとしては：
#
# - [古典制御フローパターン](05_classical_flow_patterns.ipynb) — 測定結果に基づく$X$/$Z$補正(`if bit:`)を使った測定ベースのシンドローム訂正の実装に役立ちます
# - [再利用パターン](06_reuse_patterns.ipynb) — 本チュートリアルで使った`@qkernel`ヘルパーと`@composite_gate`による再利用パターン
