# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# tags: [algorithm, optimization, oracle-based]
# ---
#
# # QSVT Filteringによるグラフ分割問題の求解
#
# 本チュートリアルでは、Quantum Singular Value Transform(QSVT)を使って、与えられたハミルトニアンの基底エネルギーを求め、グラフ分割問題を解く方法を説明します。
#
# Lin and Tongの基底エネルギーアルゴリズム{cite:p}`10.48550/arXiv.2002.12508`は、*フィルタリング*によってコストハミルトニアンの最小固有値を求めます。閾値パラメータ$\mu$を1つ決めると、Linear Combination of Unitaries(LCU)によって$H - \mu I$のブロックエンコーディングを構築できます。さらにsign関数のQSVT近似を用いて、閾値パラメータ$\mu$の片側にある固有空間へのprojectorを作ります。post-selectionを通過したショットの割合が、$\mu$をどちらへ動かせばよいかを古典的な二分探索に伝えます。
#
# 本チュートリアルでは、この手順をグラフ分割問題に適用し、ブロックエンコーディングの構築、単一の閾値での評価、二分探索による基底エネルギーの推定までを示します。

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install "qamomile[qiskit,qsvt,visualization]" networkx

# %%
import jijmodeling as jm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit_aer import AerSimulator

import qamomile.circuit as qmc
from qamomile.circuit.algorithm.qsvt_eigenstate_filter import qsvt_filter_projector
from qamomile.optimization.qsvt_eigenstate_filter import QSVTEigenstateFilterConverter
from qamomile.qiskit import QiskitTranspiler

SEED = 42

# %% [markdown]
# ## 問題設定
#
# グラフ分割は、頂点を同じ大きさの2つのグループへ分けながら、切断される辺の数をできるだけ少なくする問題です。ここでは等サイズ制約のもとで切断辺数を最小化します。この制約は`to_hubo`がペナルティ項として目的関数に取り込みます。
#
# グラフ分割問題そのものの詳細は[QAOAによるグラフ分割](qaoa_graph_partition)を参照してください。

# %%
problem = jm.Problem("Graph Partitioning")


@problem.update
def _(problem: jm.DecoratedProblem):
    V = problem.Dim()
    E = problem.Natural(ndim=2)  # 辺のリスト: [[u1,v1], [u2,v2], ...]
    x = problem.BinaryVar(shape=(V,))

    # 目的関数: 分割間で切断される辺の数を最小化する
    problem += (
        E.rows().map(lambda e: x[e[0]] * (1 - x[e[1]]) + x[e[1]] * (1 - x[e[0]])).sum()
    )

    # 制約: 2つの分割のサイズを等しくする
    problem += problem.Constraint("Equal Partition", x.sum() == V / 2)


problem

# %%
num_nodes = 6
# 三角柱グラフ: 三角形{0, 3, 4}と{1, 2, 5}を完全マッチング(0-5, 1-4, 2-3)で
# つないだものです。2つの三角形を分ければ切れるのはマッチングの3辺だけで、
# それ以外の等分割はどれも三角形を壊してより多くの辺を切るため、最適な分割は
# 一意に定まります。
edge_list = [
    [0, 3],
    [0, 4],
    [0, 5],
    [1, 2],
    [1, 4],
    [1, 5],
    [2, 3],
    [2, 5],
    [3, 4],
]

G = nx.Graph()
G.add_nodes_from(range(num_nodes))
G.add_edges_from(edge_list)
assert G.number_of_nodes() == num_nodes
assert G.number_of_edges() == len(edge_list)

pos = nx.spring_layout(G, seed=1)
plt.figure(figsize=(5, 5))
nx.draw(G, pos, with_labels=True, node_color="white", node_size=700, edgecolors="black")
plt.title(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
plt.show()

# %%
instance_data = {"V": num_nodes, "E": edge_list}
instance = problem.eval(instance_data)

# %% [markdown]
# ## アルゴリズム
#
# イジングコストハミルトニアンを$H$と書き、閾値$\mu$を1つ選びます。アルゴリズムが扱うのはシフトした演算子$H - \mu I$で、その固有値は$E_i - \mu$です。残したい固有状態でちょうど負に、それ以外で正になります。したがって両者を分けるのはsign関数の役目です。
#
# ブロックエンコーディングは、ユニタリでない演算子をより大きなユニタリ$U$の1つのブロックとして埋め込みます。LCUは$H - \mu I$をPauli列の重み付き和として書くことでこれを構築します。その代償が正規化定数$\alpha'$であり、実際にエンコードされるのは正規化された演算子$(H - \mu I) / \alpha'$です。そのスペクトルは$[-1, 1]$に収まります。このブロックは、ancillaレジスタ、すなわち*signal*レジスタがすべて0と測定されることで選択されます。
#
# 続いてQSVTは、エンコードされた演算子をその多項式で置き換えます。位相$\Phi$が与えられると、次の交替列を適用し、
#
# $$R(\phi_0),\; U,\; R(\phi_1),\; U^\dagger,\; \dots$$
#
# 同じブロックに$p\!\left((H - \mu I)/\alpha'\right)$を残します。ここで$p$としてsign関数の近似を選べば、
#
# $$p\!\left(\frac{H - \mu I}{\alpha'}\right) \approx
# \operatorname{sign}\!\left(\frac{H - \mu I}{\alpha'}\right),$$
#
# このブロックは*鏡映*$R$になります。すなわち$\mu$より下の固有空間には$-1$として、上の固有空間には$+1$として作用します。
#
# 鏡映はまだprojectorではありませんが、$P_\mu = (I - R)/2$はprojectorであり、しかも2つのユニタリの和なので、量子ビットを1つ追加したHadamard testによって確率的に実現できます。一様重ね合わせ$|\varphi_0\rangle = H^{\otimes n}|0\rangle$から出発したとき、すべてのancillaが0と測定されるショットの割合は次の量を推定します。
#
# $$P(\mu)=\lVert P_\mu |\varphi_0\rangle \rVert^2 = \frac{\#\{i : E_i < \mu\}}{2^n},$$
#
# これは$\mu$より下にあるスペクトルの割合です。この値を、古典的な二分探索で探索方向を決める判定条件として用います。$|\varphi_0\rangle$のなかで$\mu$より下の固有状態が重みを持つときに限り、この値は0から離れた値をとります。
#
# 探索そのものはLin and TongのAlgorithm 1{cite:p}`10.48550/arXiv.2002.12508`です。固有値は$[c-\alpha, c+\alpha)$に存在します。ここで$\alpha$は$H$単体の正規化定数、$c$はその定数項です。ある閾値$\mu$を基底エネルギー$\lambda_0$と比較して考えます。$\mu$が$[c-\alpha, \lambda_0)$にあるときは固有値がすべて除かれ、$P(\mu)=0$となります。区間$[\lambda_0,\lambda_1)$では基底の固有空間だけが残り、縮退度を$d$として$\gamma^2 = d/2^n$を与えます。$\lambda_1$より上では確率は増える一方です。したがって探索は$[c-\alpha,c+\alpha)$を間隔$h$のグリッドに分割し、$P(\mu)$が$\tau = \gamma^2/2$を超えるかどうかで進む方向を決めます。
#
# 成否を分ける入力が2つあります。$\gamma$は初期状態の重なりの下界ですが事前には分からないため、仮定して与えるしかありません。また$h$は(これも未知の)スペクトルギャップより小さくとる必要があります。最後に、近似誤差によって推定した$P(\mu)$が単調でなくなる場合があります。そのため探索は1点だけを信用せず、連続する2つのグリッド点$(B_k, B_{k+1})$を*組*にして評価します。

# %% [markdown]
# ## Qamomileによる実装
#
# `QSVTEigenstateFilterConverter`は問題インスタンスに対して直接呼び出します。ただし内部のパラメータの多くが解の品質を左右するため、以下ではそれぞれのパラメータと選び方を説明します。
#
# イジングハミルトニアンの定数項はブロックエンコーディングから除いてあります。この手法の性能は正規化定数$\alpha$に依存しており、恒等演算子の項はこの値を大きくしてしまうためです。一方で定数のオフセットはすべての固有値を等しくずらすだけなので、どの状態が最良かは変わりません。`normalization`($\alpha$)と`energy_offset`($c$)の2つが揃ってスペクトルの範囲を与え、閾値はこの区間のなかから選ぶことになります。

# %%
converter = QSVTEigenstateFilterConverter(instance)
transpiler = QiskitTranspiler()
# 本ページのすべての回路で同じseed付きexecutorを使い、実行ごとに表示される
# 数値が再現するようにします。
executor = transpiler.executor(
    backend=AerSimulator(seed_simulator=SEED, max_parallel_threads=1)
)

low = converter.energy_offset - converter.normalization
high = converter.energy_offset + converter.normalization
print(f"alpha={converter.normalization}  offset={converter.energy_offset}")
print(f"every eigenvalue lies in [{low}, {high}]")

# 参照用スペクトル: イジング演算子は対角なので、正確なエネルギーを直接読み取って
# あとから探索結果を検証できます。
# !!! これは検証のためだけの計算であり、一般には指数時間かかります。
spin = converter.spin_model
energies = np.full(1 << spin.num_bits, float(spin.constant))
for state in range(energies.size):
    for word, coeff in spin.coefficients.items():
        parity = sum((state >> i) & 1 for i in word) & 1
        energies[state] += coeff * (-1.0 if parity else 1.0)

ground_energy = float(energies.min())
degeneracy = int((energies == ground_energy).sum())
print(f"levels={np.unique(energies)}  ground={ground_energy} ({degeneracy}-fold)")

assert low <= energies.min() and energies.max() <= high

# %% [markdown]
# ### 閾値を1つ試す
#
# `transpile`は$\mu$を回路に埋め込むため、閾値ごとに個別のコンパイルが必要です。フィルタはデフォルトで$\mu$より下の固有空間を残します。これは最小固有値(基底エネルギー)を探索したい場合に必要な向きです。
#
# `degree`と`delta`は対で決まります。`delta`は多項式の遷移領域の幅を定め、`degree`はChebyshev展開の次数で、その遷移を表現できるだけ大きくとる必要があります。
#
# :::{note}
# 指定しない場合、`degree`と`delta`にはConverterがデフォルト値を設定します。
# :::
#
# 分解能は次の式で与えられます。
#
# $$
# \Delta = \frac{\min_i |E_i-\mu|}{\alpha + |\mu - c|}
# $$
#
# ここで$E_i$は固有値で、その他のパラメータはこれまでに定義したものです。分解能を知るには基底エネルギーを事前に知っている必要がある、という状況がしばしば起こります。実際には分解能を上回るように、値を推測するか上界を見積もることになります。`degree`はおおよそ$O(1/\Delta)$で増えます。
#
# 以下では$\mu=4$としてアルゴリズムを実演します。このとき上回るべき分解能は$\Delta = \frac{|4-3|}{3 + |4 - 6|} = 1/5$であり、`delta=8`とすれば遷移領域の幅がこのギャップに合い、`degree=21`でそれを表現できます。対の取り方を誤っても精度が静かに落ちるわけではなく、$|p| \le 1$のチェックによってその場で棄却されます。

# %%
DEGREE, DELTA = 21, 8
SHOTS = 2000
print(f"probe filter: degree={DEGREE}, delta={DELTA} -> {DEGREE + 1} phases")


def probe(mu, shots=SHOTS):
    """`mu`より下にあるスペクトルの割合を推定します。"""
    executable = converter.transpile(
        transpiler,
        mu=mu,
        degree=DEGREE,
        delta=DELTA,
    )
    result = executable.sample(executor, shots=shots).result()
    return converter.success_probability(result)


# 64状態のうち2つが基底エネルギーにあるため、muがそこを超えると判定条件の値は
# 2 / 64で頭打ちになります。
sampled_p4 = probe(4.0)
exact_p4 = float((energies < 4.0).mean())
print(f"p(mu=4.0) = {sampled_p4:.4f}   (exact: {exact_p4:.4f})")

# ショットノイズ6標準偏差ぶんに、有限次数のsign近似のぼやけに対する小さな余裕を
# 加えた許容幅です。
shot_noise = 6.0 * np.sqrt(exact_p4 * (1.0 - exact_p4) / SHOTS)
assert abs(sampled_p4 - exact_p4) < shot_noise + 0.01

# %% [markdown]
# パラメータから定まる多項式近似に対応する位相因子の計算には、外部ライブラリの`pyqsp`を使っています。
#
# Converterはprojectorの量子回路を構築します。その回路を実行し、サンプルのカウントから`success_probability`メソッドで成功確率を求めます。(上の厳密計算は精度を確認するための参照値です)

# %% [markdown]
# ### 構成要素
#
# `transpile`は3つの回路を入れ子に合成します。その順に描画すると、ブロックエンコーディングがどのようにprojectorへ変わっていくかが分かります。
#
# 図を読みやすく保つため、位相ベクトルは4要素に短くしてあります(上で使った22要素ではありません)。

# %%
DRAW_MU = 4.0

# `_shifted_encoding`は内部APIであり、Converterの公開APIには含まれません。
# ただし以下の回路はすべてこのdescriptorから構築されるため、この閾値で実際に
# 動く回路と図が一致します。
shifted = converter._shifted_encoding(DRAW_MU)
n_signal = shifted.num_signal_qubits
n_system = shifted.num_system_qubits

# degree=21のフィルタが使う22個ではなく4個の位相を使います。回路の形に効くのは
# 位相の個数だけで、値そのものは影響しません。
phi_demo = [0.1, 0.2, 0.3, 0.4]

print(f"alpha' = {shifted.normalization}   signal={n_signal}  system={n_system}")


# %% [markdown]
# #### 1. ブロックエンコーディング $U$
#
# イジングハミルトニアンはユニタリの線形結合として書かれ、$\operatorname{PREPARE}$、$\operatorname{SELECT}$、$\operatorname{PREPARE}^\dagger$として実現されます。$\operatorname{PREPARE}$は係数の振幅をsignalレジスタに載せ、$\operatorname{SELECT}$はsignalの状態に応じて対応するPauli-$Z$項をsystemに作用させ、$\operatorname{PREPARE}^\dagger$がそれをuncomputeします。
#
# 得られるのは、左上のブロック、つまりsignalレジスタがすべて0と測定される部分が$(H - \mu I)/\alpha'$となるようなユニタリです。そのブロックの外側は利用できないため、すべてのショットをそこでpost-selectする必要があります。

# %%
@qmc.qkernel
def block_encoding() -> qmc.Vector[qmc.Bit]:
    """(H - mu*I) / alpha'のブロックエンコーディングを|0>に作用させます。"""
    signal = qmc.qubit_array(n_signal, "signal")
    system = qmc.qubit_array(n_system, "system")
    signal, system = shifted.unitary(signal, system)
    return qmc.measure(signal)


block_encoding.draw(inline=True, inline_depth=1, fold_loops=False)


# %% [markdown]
# #### 2. QSVTの交替列
#
# ブロックエンコーディングが与えられると、`qmc.qsvt`は次を適用します。
#
# $$R(\phi_0),\; U,\; R(\phi_1),\; U^\dagger,\; R(\phi_2),\; U,\; \dots$$
#
# ここで$U$は上で構築したブロックエンコーディング、$R(\phi) = e^{i\phi(2\Pi - I)}$であり、$\Pi$はsignalレジスタをすべて0の状態へ射影します。この一連の列はすべてprimitiveが担当します。位相回転用の補助量子ビットを自前で確保し、$U$と$U^\dagger$を交互に並べ、位相を適用します。
#
# その効果は、encodeされた演算子をその多項式で置き換えることです。`pyqsp`が生成した位相を使うと、この多項式は$\mathrm{sign}$を近似するため、良いブロックは$\mu$の片側の固有空間に関する鏡映になります。

# %%
@qmc.qkernel
def qsvt_reflector(
    phi: qmc.Vector[qmc.Float],
) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    """sign((H - mu*I) / alpha')を実現するQSVTの交替列です。"""
    signal = qmc.qubit_array(n_signal, "signal")
    system = qmc.qubit_array(n_system, "system")
    signal, system = qmc.qsvt(signal, system, phi, shifted)
    return qmc.measure(signal), qmc.measure(system)


qsvt_reflector.draw(phi=phi_demo, inline=True, inline_depth=0, fold_loops=False)

# %% [markdown]
# #### 3. Lin & Tongのprojector
#
# 目標は、この鏡映を射影$P_\mu = (I \pm R)/2$に変えることです。しかしこのprojectorは**ユニタリではない**ため、回路でそのまま適用することはできません。
#
# 鍵になるのは、これが$I$と$R$という2つのユニタリの和である点です。そのためHadamard testを使えば確率的に実現できます。このテストでは*probe量子ビット*と呼ぶancillaを1つ使い、恒等演算子の分岐と鏡映の分岐をコヒーレントに選択させます。probe量子ビットを重ね合わせにし、その制御のもとでQSVTの鏡映を作用させ、2つ目のHadamardで2つの分岐を干渉させます。probe量子ビットが0と測定されれば、$P_\mu$が作用した分岐であることが分かります。このアルゴリズムに成功確率という概念があるのはそのためであり、post-selectionをprobeとsignalレジスタの**両方**に対して行う理由もここにあります。

# %%
qsvt_filter_projector(shifted).draw(
    proj=1,
    signal=n_signal,
    system=n_system,
    phi=phi_demo,
    inline=True,
    inline_depth=0,
    fold_loops=False,
)


# %% [markdown]
# ### 二分探索
#
# ここまでで、古典的な探索に必要なものは判定条件だけになりました。`B(k)`は「グリッド点$x_k$が残すべき固有空間を捉えている」という1ビットで、ループはアルゴリズムの節で述べたとおり、このビットを連続する2つずつ組にして消費します。

# %%
def binary_search_ground_energy(success_prob, low, high, gamma, h=1.0):
    """Lin & TongのAlgorithm 1: (B_k, B_k+1)を組にした評価。

    グリッドx_k = low + (k + 0.5) * h: 半セル分ずらすことで閾値が固有値の
    あいだに来ます。判定条件の比較は狭義なので、固有値の上にちょうど乗った閾値は
    0と判定されてしまうためです。B_k = 1となるのは、推定確率が
    tau = gamma**2 / 2を超えるときです。

    組で評価するループは最大4セルまで範囲を狭めます。そこから線形に走査して
    仕上げるので、返される範囲は1セル幅になります。
    戻り値は(x_L, x_U)で、x_L <= lambda_0 <= x_Uを満たします。左端を含むのは、
    基底エネルギーがちょうど`low`にある場合でも範囲に収めるためです。
    """
    n_grid = int(round((high - low) / h))
    grid = [low + (k + 0.5) * h for k in range(n_grid)]
    tau = 0.5 * gamma**2
    cache = {}

    def B(k):
        # 番兵: B_-1 = 0 (下に何もない)、B_G = 1 (すべてが下にある)。
        if not 0 <= k < n_grid:
            return 0 if k < 0 else 1
        if k not in cache:
            cache[k] = success_prob(grid[k])
        return int(cache[k] > tau)

    L, U = 0, n_grid - 1
    print(f"grid {grid[0]:+.2f}..{grid[-1]:+.2f}  G={n_grid}  h={h}  tau={tau:.4f}")

    while U - L > 3:
        k = (L + U) // 2
        bk, bk1 = B(k), B(k + 1)
        print(f"  k={k}: B({grid[k]:+.2f})={bk}  B({grid[k + 1]:+.2f})={bk1}", end="  ->  ")
        if bk == 1 and bk1 == 1:  # lambda_0 < x_k+1
            U = k + 1
            print(f"(1,1) U<-{U}  [{grid[L]:+.2f}, {grid[U]:+.2f}]")
        elif bk == 0 and bk1 == 0:  # lambda_0 > x_k
            L = k
            print(f"(0,0) L<-{L}  [{grid[L]:+.2f}, {grid[U]:+.2f}]")
        elif bk == 0 and bk1 == 1:  # x_k-1 < lambda_0 < x_k+2
            L, U = max(k - 1, 0), min(k + 2, n_grid - 1)
            print(f"(0,1) -> [{grid[L]:+.2f}, {grid[U]:+.2f}]")
            break
        else:  # x_k < lambda_0 < x_k+1
            L, U = k, k + 1
            print(f"(1,0) -> [{grid[L]:+.2f}, {grid[U]:+.2f}]")
            break

    # 1セルまで絞り込みます。判定条件が最初に立つグリッド点が、基底の固有空間を
    # まだ捉えている最小の閾値です。スペクトルの下界でクリップすることで範囲が
    # 物理的に意味のあるものになります。`low`より下に固有値は存在しないので、
    # そこを評価する必要はありません。
    for k in range(L, U + 1):
        if B(k):
            print(f"  refine: first B=1 at k={k} ({grid[k]:+.2f})")
            return max(grid[k] - h, low), grid[k]
    return grid[U], min(grid[U] + h, high)


# %% [markdown]
# ## 結果
#
# 探索を実行すると、基底エネルギーの位置を一切教えないままその範囲を絞り込めます。得られた閾値をフィルタに戻して与えれば基底状態そのものが準備でき、そこからサンプリングすることで分割が復元されます。

# %%
gamma = np.sqrt(degeneracy / energies.size)
x_lower, x_upper = binary_search_ground_energy(probe, low, high, gamma=gamma, h=1.0)

print(f"\nlambda_0 in [{x_lower:+.2f}, {x_upper:+.2f}]   true lambda_0 = {ground_energy:+.2f}")
assert x_lower <= ground_energy <= x_upper

# %% [markdown]
# :::{note}
# 二分探索で$P(\mu)>\tau$を判定するだけならフィルタがぼやけていても許容できるため、どの評価でも`degree=21`で十分でした。一方、基底の固有空間*だけ*を残そうとすると、$\mu = 3.5$では最適な分解能が探索時の$0.20$ではなく$\Delta = 0.09$になるため、最後のデコードではより鋭い多項式を使います。`degree`だけを上げても改善しません。`degree=61`に噛み合わない`delta=20`を組み合わせると、`degree=41`と`delta=11`の組(91%)より悪い85%になります。
# :::

# %%
# 判定条件の評価にはdegree 21で足りましたが、状態そのものを取り出すには同じ閾値でも
# より鋭いフィルタが必要です。
DECODE_DEGREE, DECODE_DELTA = 41, 11
print(
    f"decode filter: degree={DECODE_DEGREE}, delta={DECODE_DELTA} -> {DECODE_DEGREE + 1} phases"
)

executable = converter.transpile(
    transpiler,
    mu=x_upper,
    degree=DECODE_DEGREE,
    delta=DECODE_DELTA,
)
result = executable.sample(executor, shots=20000).result()
sampleset = converter.decode_to_binary_sampleset(result)

kept = sum(sampleset.num_occurrences)
optimal = sum(
    n
    for e, n in zip(sampleset.energy, sampleset.num_occurrences)
    if np.isclose(e, ground_energy)
)
print(f"kept {kept} of {result.shots} shots, {optimal / kept:.1%} at the ground energy")

best = max(zip(sampleset.samples, sampleset.num_occurrences), key=lambda z: z[1])[0]
side = sorted(v for v, bit in best.items() if bit == 1)
cut = sum(1 for u, v in edge_list if best[u] != best[v])
print(f"most sampled partition: {side} | cut={cut}")

assert optimal / kept > 0.85
assert cut == 3 and len(side) == num_nodes // 2

# %%
# 復元された分割を入力グラフと同じレイアウトで描画します。ノードの色がどちら側かを
# 表し、破線が最小化の対象である切断された辺です。
cut_edges = [(u, v) for u, v in G.edges() if best[u] != best[v]]
kept_edges = [(u, v) for u, v in G.edges() if best[u] == best[v]]

plt.figure(figsize=(5, 5))
nx.draw_networkx_edges(G, pos, edgelist=kept_edges, edge_color="black", width=1.0)
nx.draw_networkx_edges(
    G, pos, edgelist=cut_edges, edge_color="crimson", width=2.5, style="dashed"
)
nx.draw_networkx_nodes(
    G,
    pos,
    node_color=["#a6cee3" if best[v] == 1 else "#fdbf6f" for v in G.nodes()],
    node_size=700,
    edgecolors="black",
)
nx.draw_networkx_labels(G, pos)
plt.title(f"Recovered partition: {side} vs the rest, cut = {cut} edges")
plt.axis("off")
plt.show()

# %% [markdown]
# 2つの三角形$\{0,3,4\}$と$\{1,2,5\}$は同じ分割を表すため、実行ごとにどちらが最上位に来ることもあります。対応する目的関数値(最小カット)は$3$です。

# %% [markdown]
# ## まとめ
#
# 本notebookでは次のことを行いました。
#
# - `QSVTEigenstateFilterConverter`を使ってグラフ分割問題のインスタンスをend-to-endで解き、三角柱グラフの一意な最適分割を復元しました。
# - このアルゴリズムが確率的である理由を確認しました。$P_\mu = (I - R)/2$はユニタリではないためHadamard testで実現され、すべてのショットをprobeとsignalレジスタの両方でpost-selectする必要があります。
# - すべてのパラメータが、正規化されたギャップ$\Delta = \min_i |E_i - \mu| / \alpha'$という1つの数値に帰着することを見ました。イジングの定数項をブロックエンコーディングから除くことが重要なのも、`normalization`と`energy_offset`が対で報告されるのも、このためです。
