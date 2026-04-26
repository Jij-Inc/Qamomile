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
# # 量子誤り訂正(2): Steane [[7,1,3]] 符号と CSS 構成
#
# [前章](10_quantum_error_correction.ipynb)で 3量子ビット符号と Shor 9量子ビット符号を扱いました。本章では次のステップとして **Steane の [[7,1,3]] 符号** を取り上げます。Steane 符号は:
#
# - **CSS 構成** (Calderbank-Shor-Steane) の最も有名な例 — 古典の Hamming [7,4,3] 符号を 2 つ重ねて作る
# - **横断的 Clifford ゲート**(transversal Clifford gates)が物理ゲートを各量子ビットに同時適用するだけで実装できる — フォールトトレラント計算の入口
# - 7 物理量子ビットで 1 論理量子ビットを保護(Shor の 9 より少ない)
# - スタビライザーの重みは 4 — 3量子ビット符号(重み 2)から表面符号(重み 4)への自然なステップ
#
# このチュートリアルで身につくこと:
#
# - CSS 符号の構成原理(古典符号のパリティ検査行列から量子符号を作る)
# - 6 つのスタビライザー(3 つの X 型と 3 つの Z 型)を独立に測定する syndrome extraction
# - X / Y / Z すべての単一量子ビット誤りを訂正する Steane 符号の動作
# - **横断的 Hadamard** によって logical H が物理 H 7 つで実装できること

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install qamomile

# %%
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()


# %% [markdown]
# ## 1. CSS 構成: 古典 Hamming 符号から量子符号へ
#
# ### 古典 Hamming [7,4,3] 符号の復習
#
# 古典の Hamming [7,4,3] 符号は 7 ビットの符号語で 4 ビットの情報を保護し、最小距離 3(= 単一ビット誤りを訂正可能)を持ちます。パリティ検査行列を:
#
# $$
# H = \begin{pmatrix}
# 0 & 0 & 0 & 1 & 1 & 1 & 1 \\
# 0 & 1 & 1 & 0 & 0 & 1 & 1 \\
# 1 & 0 & 1 & 0 & 1 & 0 & 1
# \end{pmatrix}
# $$
#
# と取ります。$H$ の列 $j$ は 2 進数表現の $j+1$ で、これにより**シンドロームから誤り位置が一意に逆算できる**(列 $j$ のシンドロームが現れたら誤りはビット $j$)というエレガントな性質があります。
#
# ### CSS 構成の基本アイデア
#
# CSS 符号は古典の線形符号 $C_X, C_Z$ を用意して:
#
# - $C_Z^\perp$ の生成元から $Z$ 型スタビライザーを作る(ビット反転 $X$ 誤りを検出)
# - $C_X^\perp$ の生成元から $X$ 型スタビライザーを作る(位相反転 $Z$ 誤りを検出)
#
# Steane 符号は $C_X = C_Z = $ Hamming [7,4,3] という同じ符号を使う**自己双対 CSS 符号**で、6 つのスタビライザーが得られます。
#
# ### Steane 符号のスタビライザー
#
# Hamming のパリティ検査行列の各行から 2 つのスタビライザー($Z$ 型と $X$ 型)を作ります:
#
# | 番号 | スタビライザー | 検出する誤り |
# |------|----------------|--------------|
# | $S_1$ | $X_3 X_4 X_5 X_6$ | 位相反転 $Z$ |
# | $S_2$ | $X_1 X_2 X_5 X_6$ | 位相反転 $Z$ |
# | $S_3$ | $X_0 X_2 X_4 X_6$ | 位相反転 $Z$ |
# | $S_4$ | $Z_3 Z_4 Z_5 Z_6$ | ビット反転 $X$ |
# | $S_5$ | $Z_1 Z_2 Z_5 Z_6$ | ビット反転 $X$ |
# | $S_6$ | $Z_0 Z_2 Z_4 Z_6$ | ビット反転 $X$ |
#
# 論理演算子は:
#
# - $\bar{X} = X_0 X_1 X_2$
# - $\bar{Z} = Z_0 Z_1 Z_2$
#
# (どちらも重み 3 = 符号の距離 $d=3$。重みがちょうど距離に等しい論理演算子の存在は CSS 符号の典型的特徴。)

# %% [markdown]
# ## 2. $\lvert 0_L\rangle$ エンコーダ
#
# Steane 符号の論理 $\lvert 0_L\rangle$ は **Hamming 符号の偶重み符号語の重ね合わせ**:
#
# $$
# \lvert 0_L\rangle = \frac{1}{2\sqrt{2}} \sum_{c \in C, w(c) \text{ even}} \lvert c\rangle
# $$
#
# 8 つの偶重み Hamming 符号語があります(全 16 符号語の半分)。
#
# エンコーダは初期状態 $\lvert 0\rangle^{\otimes 7}$ から各 $X$ 型スタビライザー $S_i$ について **$(\mathbb{1} + S_i)/\sqrt{2}$ 投影**を順次作用させて構成できます。各投影は「スタビライザーの 1 量子ビットに $H$ を当て、残りに CNOT を当てる」というシンプルなパターン。


# %%
@qmc.qkernel
def encode_steane_zero(data: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    # S1 = X3 X4 X5 X6 投影: H q[3], CNOT q[3] -> q[4,5,6]
    data[3] = qmc.h(data[3])
    data[3], data[4] = qmc.cx(data[3], data[4])
    data[3], data[5] = qmc.cx(data[3], data[5])
    data[3], data[6] = qmc.cx(data[3], data[6])

    # S2 = X1 X2 X5 X6 投影: H q[1], CNOT q[1] -> q[2,5,6]
    data[1] = qmc.h(data[1])
    data[1], data[2] = qmc.cx(data[1], data[2])
    data[1], data[5] = qmc.cx(data[1], data[5])
    data[1], data[6] = qmc.cx(data[1], data[6])

    # S3 = X0 X2 X4 X6 投影: H q[0], CNOT q[0] -> q[2,4,6]
    data[0] = qmc.h(data[0])
    data[0], data[2] = qmc.cx(data[0], data[2])
    data[0], data[4] = qmc.cx(data[0], data[4])
    data[0], data[6] = qmc.cx(data[0], data[6])

    return data


# %% [markdown]
# このエンコーダ単体の動作を確認します。$\lvert 0\rangle^{\otimes 7}$ をエンコードして測定すると、8 通りの偶重み Hamming 符号語が等確率(=各 1/8)で得られるはずです。


# %%
@qmc.qkernel
def encode_zero_and_measure() -> qmc.Vector[qmc.Bit]:
    data = qmc.qubit_array(7, name="data")
    data = encode_steane_zero(data)
    return qmc.measure(data)


# %%
print("|0_L⟩ をエンコードして測定 (期待: 8 つの偶重み Hamming 符号語が等確率):")
exe = transpiler.transpile(encode_zero_and_measure)
result = exe.sample(transpiler.executor(), shots=4000).result()
for outcome, count in sorted(result.results, key=lambda x: -x[1]):
    bits = outcome if isinstance(outcome, (list, tuple)) else [int(b) for b in f"{outcome:07b}"]
    weight = sum(bits)
    print(f"  {tuple(bits)}: {count:4d}回 (重み={weight})")

# %% [markdown]
# 重みが偶数(0 または 4)の 8 つのビット列のみが等確率で出現していることが確認できます。

# %% [markdown]
# ## 3. シンドローム抽出: CSS 復号の独立性
#
# CSS 符号の最大の特徴は、**$X$ 誤りと $Z$ 誤りを独立に復号できる**ことです:
#
# - $X$ 誤りは **$Z$ 型スタビライザー** $S_4, S_5, S_6$ で検出される(3 ビットのシンドローム)
# - $Z$ 誤りは **$X$ 型スタビライザー** $S_1, S_2, S_3$ で検出される(3 ビットのシンドローム)
#
# シンドロームのビット列は Hamming パリティ検査行列の列(= 誤り位置の 2 進表現)になり、ルックアップで誤り量子ビットを一意に特定できます。
#
# | $X$ 誤り位置 | $X$-syndrome $(s^X_2, s^X_1, s^X_0)$ |
# |:-:|:-:|
# | なし   | (0, 0, 0) |
# | $q_0$ | (0, 0, 1) |
# | $q_1$ | (0, 1, 0) |
# | $q_2$ | (0, 1, 1) |
# | $q_3$ | (1, 0, 0) |
# | $q_4$ | (1, 0, 1) |
# | $q_5$ | (1, 1, 0) |
# | $q_6$ | (1, 1, 1) |
#
# ($Z$ 誤りも同じ表で、X-stabilizer による syndrome を読む。)

# %% [markdown]
# ### 測定パターン
#
# - $Z$ 型スタビライザー $S_i = Z_a Z_b Z_c Z_d$ の測定: ancilla を $\lvert 0\rangle$ で初期化、`CNOT(data[a], anc); CNOT(data[b], anc); CNOT(data[c], anc); CNOT(data[d], anc); measure(anc)`
# - $X$ 型スタビライザー $S_i = X_a X_b X_c X_d$ の測定: ancilla を $\lvert + \rangle$ ($H\lvert 0\rangle$) に初期化、`CNOT(anc, data[a..d])` を 4 回、最後に $H$ をかけて測定
#
# 6 個の ancilla を使って 6 つのスタビロームを並列に測定します(直列でもよいが並列の方が浅い)。

# %% [markdown]
# ## 4. シンドロームによる訂正回路
#
# 上のルックアップ表をそのまま `if (条件) & (条件):` のチェーンに翻訳します。`Bit` ハンドルの `&` / `~` オペレータがそのまま使えます。シンドロームの 7 通りの非ゼロパターンに対し、対応する量子ビットに $X$(または $Z$)を作用。


# %%
NO_ERROR = 0
X_ERROR = 1
Y_ERROR = 2
Z_ERROR = 3


@qmc.qkernel
def steane_run(
    error_type: qmc.UInt, error_pos: qmc.UInt
) -> qmc.Vector[qmc.Bit]:
    """`error_type` ∈ {1=X, 2=Y, 3=Z}, `error_pos` ∈ {0..6} で 1 つの誤りを指定。

    `for j in qmc.range(7)` で位置を走査し、`(error_type == X) & (error_pos == j)`
    で該当位置にだけパウリゲートを当てる。両パラメータをコンパイル時に束縛する
    ので、ループ・if は折り畳まれて 1 つのゲートに簡約される。
    """
    data = qmc.qubit_array(7, name="data")
    anc = qmc.qubit_array(6, name="anc")  # 0..2: X-error syndrome, 3..5: Z-error syndrome

    # ---- エンコード: |0_L⟩ ----
    data = encode_steane_zero(data)

    # ---- 単一量子ビット誤り注入 ----
    for j in qmc.range(7):
        if (error_type == X_ERROR) & (error_pos == j):
            data[j] = qmc.x(data[j])
        if (error_type == Y_ERROR) & (error_pos == j):
            data[j] = qmc.y(data[j])
        if (error_type == Z_ERROR) & (error_pos == j):
            data[j] = qmc.z(data[j])

    # ---- Z 型スタビライザー測定 (X 誤りのシンドローム) ----
    # S4 = Z3 Z4 Z5 Z6
    data[3], anc[0] = qmc.cx(data[3], anc[0])
    data[4], anc[0] = qmc.cx(data[4], anc[0])
    data[5], anc[0] = qmc.cx(data[5], anc[0])
    data[6], anc[0] = qmc.cx(data[6], anc[0])
    sx_2 = qmc.measure(anc[0])

    # S5 = Z1 Z2 Z5 Z6
    data[1], anc[1] = qmc.cx(data[1], anc[1])
    data[2], anc[1] = qmc.cx(data[2], anc[1])
    data[5], anc[1] = qmc.cx(data[5], anc[1])
    data[6], anc[1] = qmc.cx(data[6], anc[1])
    sx_1 = qmc.measure(anc[1])

    # S6 = Z0 Z2 Z4 Z6
    data[0], anc[2] = qmc.cx(data[0], anc[2])
    data[2], anc[2] = qmc.cx(data[2], anc[2])
    data[4], anc[2] = qmc.cx(data[4], anc[2])
    data[6], anc[2] = qmc.cx(data[6], anc[2])
    sx_0 = qmc.measure(anc[2])

    # ---- X 型スタビライザー測定 (Z 誤りのシンドローム) ----
    # S1 = X3 X4 X5 X6: ancilla を |+⟩ に、anc 制御の CNOT 4 個、最後に H で X 基底測定
    anc[3] = qmc.h(anc[3])
    anc[3], data[3] = qmc.cx(anc[3], data[3])
    anc[3], data[4] = qmc.cx(anc[3], data[4])
    anc[3], data[5] = qmc.cx(anc[3], data[5])
    anc[3], data[6] = qmc.cx(anc[3], data[6])
    anc[3] = qmc.h(anc[3])
    sz_2 = qmc.measure(anc[3])

    # S2 = X1 X2 X5 X6
    anc[4] = qmc.h(anc[4])
    anc[4], data[1] = qmc.cx(anc[4], data[1])
    anc[4], data[2] = qmc.cx(anc[4], data[2])
    anc[4], data[5] = qmc.cx(anc[4], data[5])
    anc[4], data[6] = qmc.cx(anc[4], data[6])
    anc[4] = qmc.h(anc[4])
    sz_1 = qmc.measure(anc[4])

    # S3 = X0 X2 X4 X6
    anc[5] = qmc.h(anc[5])
    anc[5], data[0] = qmc.cx(anc[5], data[0])
    anc[5], data[2] = qmc.cx(anc[5], data[2])
    anc[5], data[4] = qmc.cx(anc[5], data[4])
    anc[5], data[6] = qmc.cx(anc[5], data[6])
    anc[5] = qmc.h(anc[5])
    sz_0 = qmc.measure(anc[5])

    # ---- X 誤り訂正 (X-syndrome → 列インデックスを X 訂正) ----
    if (~sx_2) & (~sx_1) & sx_0:
        data[0] = qmc.x(data[0])  # 列 (0,0,1) → q[0]
    if (~sx_2) & sx_1 & (~sx_0):
        data[1] = qmc.x(data[1])  # 列 (0,1,0) → q[1]
    if (~sx_2) & sx_1 & sx_0:
        data[2] = qmc.x(data[2])  # 列 (0,1,1) → q[2]
    if sx_2 & (~sx_1) & (~sx_0):
        data[3] = qmc.x(data[3])  # 列 (1,0,0) → q[3]
    if sx_2 & (~sx_1) & sx_0:
        data[4] = qmc.x(data[4])  # 列 (1,0,1) → q[4]
    if sx_2 & sx_1 & (~sx_0):
        data[5] = qmc.x(data[5])  # 列 (1,1,0) → q[5]
    if sx_2 & sx_1 & sx_0:
        data[6] = qmc.x(data[6])  # 列 (1,1,1) → q[6]

    # ---- Z 誤り訂正 (Z-syndrome → 列インデックスを Z 訂正) ----
    if (~sz_2) & (~sz_1) & sz_0:
        data[0] = qmc.z(data[0])
    if (~sz_2) & sz_1 & (~sz_0):
        data[1] = qmc.z(data[1])
    if (~sz_2) & sz_1 & sz_0:
        data[2] = qmc.z(data[2])
    if sz_2 & (~sz_1) & (~sz_0):
        data[3] = qmc.z(data[3])
    if sz_2 & (~sz_1) & sz_0:
        data[4] = qmc.z(data[4])
    if sz_2 & sz_1 & (~sz_0):
        data[5] = qmc.z(data[5])
    if sz_2 & sz_1 & sz_0:
        data[6] = qmc.z(data[6])

    return qmc.measure(data)


# %% [markdown]
# ## 5. 動作検証: $X$, $Y$, $Z$ すべての単一量子ビット誤り
#
# Steane 符号は $X$, $Y$, $Z$ いずれの単一量子ビット誤りでも訂正できます。各 (誤り種類, 位置) について 1000 ショット実行し、訂正後の論理状態が **$\lvert 0_L\rangle$ のままで保たれている**ことを確認します。
#
# 検証指標: $\lvert 0_L\rangle$ は偶重み Hamming 符号語のみで構成されるので、訂正後に全 7 量子ビットを測定したビット列の **重み(1 の個数)が偶数**であれば論理状態は保たれています。

# %%
def _is_even_weight(outcome) -> bool:
    """Steane 符号語(7 ビット)の重みが偶数なら True。"""
    if isinstance(outcome, (list, tuple)):
        bits = outcome
    else:
        bits = [(outcome >> i) & 1 for i in range(7)]
    return sum(bits) % 2 == 0


print("Steane 符号: 各単一量子ビット誤り(X/Y/Z × 7 位置 = 21 通り)に対して訂正後の論理状態を確認")
print(f"  {'誤り':4s} | {'位置':5s} | 偶重み(=|0_L⟩保持)")
print(f"  {'-' * 4}-+-{'-' * 5}-+-{'-' * 22}")

for ename, ecode in [("X", X_ERROR), ("Y", Y_ERROR), ("Z", Z_ERROR)]:
    for pos in range(7):
        exe_e = transpiler.transpile(
            steane_run, bindings={"error_type": ecode, "error_pos": pos}
        )
        result = exe_e.sample(transpiler.executor(), shots=400).result()
        even = sum(c for outcome, c in result.results if _is_even_weight(outcome))
        total = sum(c for _, c in result.results)
        ratio = even / total
        print(f"  {ename:4s} | q[{pos}]  | {ratio:.3f}")

# %% [markdown]
# 21 通りすべてで偶重みの割合が 1.000 (= 全ショットで $\lvert 0_L\rangle$ が保持されている) になることが確認できます。これが Steane 符号の訂正能力です。

# %% [markdown]
# ## 6. 横断的 Hadamard: logical $\bar{H}$ が物理 $H$ 7 つで実装できる
#
# Steane 符号の最も美しい特徴の一つが**横断的 Clifford ゲート**です。論理 Hadamard $\bar{H}$ は 7 つの物理量子ビットすべてに同時に $H$ を作用させるだけで実装できます:
#
# $$
# \bar{H} = H \otimes H \otimes H \otimes H \otimes H \otimes H \otimes H
# $$
#
# これは Steane 符号の自己双対性 ($X$ と $Z$ スタビライザーが同じ Hamming パターン)から来ます。フォールトトレラント計算で重要なのは、横断的ゲートは**誤りを増幅しない** — 1 つの物理ゲートに誤りが乗っても、それは 1 つの物理量子ビットにしか影響せず、符号全体としては単一誤りのまま訂正できるからです。
#
# ### 実装と検証
#
# 横断的 $\bar{H}$ を $\lvert 0_L\rangle$ に作用させると $\lvert +_L\rangle = (\lvert 0_L\rangle + \lvert 1_L\rangle)/\sqrt{2}$ になります。さらにもう一度 $\bar{H}$ を作用させると $\lvert 0_L\rangle$ に戻ります。途中で測定して X 基底で $\lvert +_L\rangle$ を確認しましょう: 全量子ビットに $H$ を作用させてから測定すれば、結果は再び偶重み符号語になります。


# %%
@qmc.qkernel
def transversal_hadamard_demo() -> qmc.Vector[qmc.Bit]:
    data = qmc.qubit_array(7, name="data")

    # |0_L⟩ をエンコード
    data = encode_steane_zero(data)

    # 横断的 Hadamard: 全 7 量子ビットに H → |0_L⟩ から |+_L⟩ へ
    for i in qmc.range(7):
        data[i] = qmc.h(data[i])

    # X 基底で測定するため、もう一度 H をかけてから Z 基底で測定
    # (= H 2 回で |0_L⟩ に戻ってから測定)
    for i in qmc.range(7):
        data[i] = qmc.h(data[i])

    return qmc.measure(data)


# %%
print("横断的 H を 2 回適用 → |0_L⟩ に戻ることを確認 (偶重み符号語のみが出現):")
exe_th = transpiler.transpile(transversal_hadamard_demo)
result = exe_th.sample(transpiler.executor(), shots=2000).result()
even_count = sum(c for outcome, c in result.results if _is_even_weight(outcome))
total = sum(c for _, c in result.results)
print(f"  偶重み(= |0_L⟩)の割合: {even_count / total:.3f}")
print(f"  総ショット数: {total}, 異なる結果: {len(result.results)} 通り")

# %% [markdown]
# 全ショットで偶重みになります(= $\lvert 0_L\rangle$ が再現)。これは横断的 $\bar{H}$ が確かに論理 Hadamard として作用していることを示します。
#
# ### 横断的にできない演算
#
# Clifford 群は横断的にすべて実装できますが、**$T$ ゲート(非 Clifford)は横断的に実装できない**(Eastin-Knill の定理)。これがフォールトトレラント量子計算の中心的困難で、解決策として **magic state distillation** (魔法状態蒸留) などが必要になります。

# %% [markdown]
# ## 7. Shor 符号 / 表面符号との比較
#
# | 観点 | Shor [[9,1,3]] | Steane [[7,1,3]] | 表面符号 (距離 $d$) |
# |---|---|---|---|
# | 物理量子ビット数 | 9 | 7 | $d^2$ (たとえば $d=3$ なら 9) |
# | スタビローム数 | 8 | 6 | $d^2 - 1$ |
# | スタビライザー重み | 2 と 6 | 4 | 4 (バルク) |
# | 構成 | 連結符号(phase ∘ bit) | CSS(自己双対) | CSS(2D 格子) |
# | 横断的 Clifford | 不完全 | $H, S, \text{CNOT}$ 全て可 | $\text{CNOT}$ のみ(横断的でない手法も) |
# | スタビライザーの局所性 | 非局所 | 非局所 | **局所**(2D 格子上で隣接) |
# | しきい値定理 | なし(教育用) | なし(限定的) | **約 1%**(実機の現実的目標) |
# | 実機での主流度 | × | × | ○ (Google, IBM 等) |
#
# Steane 符号は Shor から自然な進歩で、CSS 構成と横断的 Clifford ゲートの教科書的なデモンストレーションになっています。次の大きな飛躍は**表面符号**で、これは CSS 構成を 2 次元格子上のローカルなスタビライザーに展開したものです。
#
# ### 次へ: 表面符号
#
# 表面符号は次のチュートリアルで扱います。重要な追加概念:
#
# - 2D 格子トポロジー(plaquette と vertex のスタビロームを区別)
# - 距離 $d$ のスケール則(物理量子ビットを増やすと指数的に論理誤り率が下がる)
# - 繰り返しシンドローム測定(誤りが時間方向にも伝播する)
# - 復号アルゴリズム(最小重み完全マッチングなど)

# %% [markdown]
# ## まとめ
#
# 本チュートリアルでは:
#
# - 古典 **Hamming [7,4,3]** 符号から **CSS 構成** で Steane [[7,1,3]] 符号を構築
# - 6 つのスタビライザー(3 つの $X$ 型、3 つの $Z$ 型)を独立に測定する syndrome extraction を実装
# - **$X$ / $Y$ / $Z$ 任意の単一量子ビット誤り**(21 通り)を訂正できることを実証
# - **横断的 Hadamard** が物理 $H$ 7 つで論理 $\bar{H}$ として動作することを確認
# - Shor との比較 — 同じ訂正能力でより少ない量子ビット、より体系的な構造
#
# Qamomile の `Bit` ハンドルに対する `&` / `~` オペレータは Steane の 7 通りシンドロームルックアップを宣言的に書くために自然に活躍しました。Qiskit バックエンドは内部で `expr.logic_and` / `logic_not` を使ったクラシカル式を `if_test` 条件として埋め込んでいます。
#
# ### 次へ
#
# - [量子誤り訂正(1)](10_quantum_error_correction.ipynb) — 3量子ビット bit-flip / phase-flip / Shor 符号
# - 量子誤り訂正(3): 表面符号 — 2D 格子上のローカルなスタビライザー、しきい値定理(次のチュートリアル予定)
#
# ### 参考文献
#
# - Steane, A. M. (1996). "Multiple particle interference and quantum error correction." *Proc. R. Soc. A* 452, 2551
# - Calderbank, A. R. & Shor, P. W. (1996). "Good quantum error-correcting codes exist." *Phys. Rev. A* 54, 1098
# - Devitt, S. J., Munro, W. J., Nemoto, K. (2013). "Quantum error correction for beginners." *Reports on Progress in Physics* 76, 076001 (arXiv:0905.2794)
