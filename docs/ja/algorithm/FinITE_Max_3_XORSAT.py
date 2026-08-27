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
# tags: [algorithm, optimization, sample-based]
# ---
#
# # PUBOのための有限虚時間発展(FinITE)
#
# 虚時間発展$e^{-\beta\hat H}$は励起状態を指数的に抑制します。そのためコストハミルトニアンに適用すると、状態を最適解へ集中させることができます。ただしこの操作はユニタリではないため、量子コンピュータ上でそのまま実行することはできません。
#
# **FinITE**は、post-selectionを伴うlinear combination of unitaries(LCU)を使って、虚時間発展の*スケールされた*版を実現する手法です。
#
# > J. Kim, J. Kim, G. Lee, K. Baek, D. K. Park, J. Bang, J. Huh, "Finite Imaginary-Time Evolution for Polynomial Unconstrained Binary Optimization" [arXiv:2604.27482](https://arxiv.org/abs/2604.27482))
#
#
# 本チュートリアルでは、Qamomileの`FinITEConverter`を使って**Max-3-XORSAT**のインスタンスを解きます。この問題は全ての項が3体相互作用であり、FinITEと相性が良い問題です。FinITEはPauli項1つにつきちょうど1つのancillaを使うため、ここでは制約1つにつきancilla1つとなります。また高次項は二次形式に落とすことなく、そのままblock encodingされます。
#
# 本チュートリアルは次の構成です。
#
# 1. Max-3-XORSAT問題の導入と、`jijmodeling`によるモデリング。
# 2. `FinITEConverter`の使い方とパラメータの選び方。
# 3. インスタンスを解く。デコードと結果。

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install qamomile
import os
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ## 1. Max-3-XORSAT問題

# %% [markdown]
# ### 1.1 問題設定
#
# **Max-3-XORSAT**では、バイナリ変数$x_0,\dots,x_{N-1}$と*パリティチェック*のリストが与えられます。各チェックは3つの変数と目標パリティを指定し、次を満たすときに充足されます。
#
# $$
# x_i \oplus x_j \oplus x_k \;=\; t_c , \qquad t_c \in \{0, 1\},
# $$
#
# つまり3つの変数のうち1であるものの個数が、要求されたパリティを持つときです。目的はできるだけ多くのチェックを充足することです。
#
# XORを多項式として書くのは扱いにくいのですが、**スピン**変数を使うと単なる積になります。次の置き換えを行います。
#
# $$
# s_i \;=\; 1 - 2 x_i \;\in\; \{+1, -1\},
# $$
#
# ビットが0ならスピンは$+1$、1ならスピンは$-1$になるため、積$s_i s_j s_k$は3つのビットのうち1であるものが偶数個のときにちょうど$+1$となります。目標パリティを符号$\tau_c = (-1)^{t_c}$と書くと、チェックが充足されるのは$s_i s_j s_k = \tau_c$のときであり、充足されないチェックの個数は次のようになります。
#
# $$
# C(s) \;=\; \sum_c \frac{1 - \tau_c\, s_i s_j s_k}{2},
# $$
#
# この値は充足されたチェックごとに$0$、充足されないチェックごとに$1$となります。

# %% [markdown]
# ### 1.2 JijModelingによる問題の定義

# %%
import jijmodeling as jm

problem = jm.Problem("Max-3-XORSAT")


@problem.update
def _(problem: jm.DecoratedProblem):
    N = problem.Dim()
    C = problem.Natural(ndim=2)  # 各行が1つのチェック。3つの変数を表す
    tau = problem.Float(ndim=1, latex=r"\tau")  # 目標の符号。偶数なら+1、奇数なら-1
    x = problem.BinaryVar(shape=(N,))

    # ビットではなくスピンで扱います。s = 1 - 2x は0を+1に、1を-1に写すため、
    # パリティチェックは3つのスピンの積になります。
    s = problem.NamedExpr("s", 1 - 2 * x, latex="s")

    # 充足されないチェックの個数。s_i s_j s_k が目標の符号と一致すれば0、それ以外は1です。
    problem += (
        C.rows()
        .enumerate()
        .map(lambda c, v: (1 - tau[c] * s[v[0]] * s[v[1]] * s[v[2]]) / 2)
        .sum()
    )


problem

# %% [markdown]
# ### 1.3 インスタンス
#
# 変数6個、チェック8個です。全てのチェックを充足する割り当てがちょうど1つだけ存在するように選んでいます。

# %%
NUM_VARS = 6
# 各チェックは、制約する3つの変数と、その和に要求されるパリティで表します。
CHECKS = [
    ([2, 3, 5], 0),
    ([0, 1, 3], 1),
    ([0, 3, 5], 1),
    ([1, 2, 5], 0),
    ([0, 2, 5], 0),
    ([0, 2, 3], 0),
    ([1, 4, 5], 1),
    ([3, 4, 5], 1),
]
CLAUSES = [variables for variables, _parity in CHECKS]
TARGETS = [1.0 if parity == 0 else -1.0 for _variables, parity in CHECKS]

instance = problem.eval({"N": NUM_VARS, "C": CLAUSES, "tau": TARGETS})

# %% [markdown]
# 下の二部グラフは、各チェック(四角)とそれが制約する3つの変数(丸)を結んだものです。

# %%
import math
import matplotlib.pyplot as plt
import networkx as nx

variable_nodes = [f"x{v}" for v in range(NUM_VARS)]
check_nodes = [f"c{c}" for c in range(len(CHECKS))]

graph = nx.Graph()
graph.add_nodes_from(variable_nodes)
graph.add_nodes_from(check_nodes)
for index, variables in enumerate(CLAUSES):
    graph.add_edges_from([(f"c{index}", f"x{v}") for v in variables])

# 座標を明示的に指定します。変数は左側にインデックス順、チェックは右側に並べます。
layout = {name: (0.0, -pos) for pos, name in enumerate(variable_nodes)}
scale = (len(variable_nodes) - 1) / (len(check_nodes) - 1)
layout |= {name: (1.0, -pos * scale) for pos, name in enumerate(check_nodes)}

plt.figure(figsize=(6.0, 4.5))
# 目標パリティが見えるように、偶数のチェックは実線、奇数のチェックは破線で描きます。
for index, (variables, parity) in enumerate(CHECKS):
    nx.draw_networkx_edges(
        graph,
        layout,
        edgelist=[(f"c{index}", f"x{v}") for v in variables],
        edge_color="tab:blue" if parity == 0 else "tab:orange",
        style="solid" if parity == 0 else "dashed",
        alpha=0.6,
    )
nx.draw_networkx_nodes(
    graph, layout, nodelist=variable_nodes, node_color="white",
    edgecolors="black", node_size=650,
)
nx.draw_networkx_nodes(
    graph, layout, nodelist=check_nodes, node_color="lightgray",
    edgecolors="black", node_shape="s", node_size=450,
)
nx.draw_networkx_labels(graph, layout, font_size=9)
plt.title("Parity-check factor graph\nsolid = even target, dashed = odd target")
plt.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. FinITEConverter

# %% [markdown]
# ### 2.1 Converterのセットアップ
#
# `FinITEConverter`はOMMXインスタンスを受け取ってスピンモデルに変換し、block encodingの対象となる恒等項を除いたPauli項を保持します。

# %%
from qamomile.optimization.finite_ite import FinITEConverter

converter = FinITEConverter(instance)


print(f"variables (system qubits) = {converter.spin_model.num_bits}")
print(f"Pauli terms  (ancillas)   = {converter.num_ancilla_bits}")
print(f"W = sum |x_mu|            = {converter.weight_norm} \n")

print("The problem results in a 3-body spin model :")
for term, coef in converter.spin_model.coefficients.items():
    print(f"{term} : {coef}")

# %% [markdown]
# ここで2点注目してください。
#
# 1. コストは制約1つにつきancilla1つです。FinITEはPauli項1つにつきLCUブロックを1つ、つまりancillaを1つ使います。各パリティチェックがそのまま1つのPauli項であるため、ancilla数はチェック数と等しく8になります。制約が多数のPauli項に展開される問題(3-SATの節は7項に展開されます)では、その分だけコストが増えます。
#
# 2. 3体の項はそのまま使われます。二次形式のモデルしか扱えない手法では、まず各3次項を*二次化*する必要があり、項ごとに補助変数とペナルティ重みが増えます。FinITEは3次項を直接block encodingするため、探索空間は$2^6$のままで、調整すべきペナルティ重みもありません。

# %% [markdown]
# ### 2.2 $\beta$の選び方
#
# 論文では、目標fidelity$\bar F$に対する*十分な*$\beta$が導かれています(Corollary 2)。これは基底状態fidelity$F_g$のギャップに基づく下界を反転させたものです。
#
# $$
# \beta^\star(\bar F) \;=\; \max\left\{0,\ \frac{1}{2\Delta}\log\frac{\bar F(1-\gamma_0)}{\gamma_0(1-\bar F)}\right\}.
# $$
#
# したがって必要なのはスペクトルギャップ$\Delta$と初期状態の基底状態オーバーラップ$\gamma_0$の評価だけで、スペクトル全体を知る必要はありません。実際の問題では、これらは古典的な前処理やwarm startから得ることになります。ヘルパー関数`finite_ite_beta_threshold`は、論文の計算に基づいて適切な$\beta$を求めます。

# %%
from qamomile.optimization.finite_ite import finite_ite_beta_threshold

TARGET_FIDELITY = 0.95
GAP_BOUND = 2.0
INITIAL_OVERLAP_BOUND = 0.015625
W = converter.weight_norm

BETA = finite_ite_beta_threshold(
    TARGET_FIDELITY,
    spectral_gap=GAP_BOUND,
    ground_overlap=INITIAL_OVERLAP_BOUND,
)
print(f"beta* for F_g >= {TARGET_FIDELITY} : {BETA:.4f}")

# %% [markdown]
# ### 2.3 量子回路へのトランスパイル
#
# **手法の概要** Pauli-Z文字列は互いに可換であるため、虚時間発展の演算子は厳密に因子分解でき、$\sigma_\mu^2 = I$により各因子は2項に収まります。
#
# $$
# e^{-\beta\hat H} \;=\; \prod_\mu \Big[\cosh(\beta|x_\mu|)\,I \;-\; \mathrm{sgn}(x_\mu)\sinh(\beta|x_\mu|)\,\sigma_\mu\Big].
# $$
#
# 各因子はLCUによってblock encodingでき、その積は`stdlib/block_encoding/product.py`によってencodingされます。block encoding全体の規格化定数は$\alpha = e^{\beta W}$です。
#
# encodingには`converter.block_encoding`からアクセスできます。

# %%
encoding = converter.block_encoding(BETA)
print(f"alpha   = {encoding.normalization:.6f}")
print(f"exp(bW) = {math.exp(BETA * W):.6f}")

# %% [markdown]
# 問題を解くには、`transpile`メソッドを使うだけです。

# %%
from qamomile.qiskit import QiskitTranspiler
transpiler = QiskitTranspiler()

program = converter.transpile(transpiler, beta=BETA)
circuit = program.quantum_circuit
print(
    f"qubits = {circuit.num_qubits} "
    f"({NUM_VARS} system + {converter.num_ancilla_bits} ancillas)"
)
print(f"depth (fully decomposed) = {circuit.decompose(reps=6).depth()}")

# %% [markdown]
# ## 3. インスタンスを解く。ショットベースの結果

# %% [markdown]
# ### 3.1 Block encodingの適用
#
# あとは回路からサンプリングし、8個のancillaが全て0であったショットだけを残します。これがblock encodingに成功した場合です。

# %%
shots = 20_000 if docs_test_mode else 200_000
result = program.sample(transpiler.executor(), shots=shots).result()

acceptance = converter.success_probability(result)
sample_set = converter.decode(result)

print(f"shots = {shots}, acceptance rate = {acceptance:.4f}")

# %% [markdown]
# ### 3.2 結果のデコード
#
# 最後に、残ったショットをデコードします。`decode`はpost-selectionを適用し、ConverterがOMMXインスタンスから作られている場合は、元の目的関数(充足されないチェックの個数)で評価された`ommx.v1.SampleSet`を返します。

# %%
best = sample_set.best_feasible
values = best.decision_variables_df["value"]
assignment = [int(values.loc[v]) for v in range(NUM_VARS)]
unsatisfied = sum(
    1
    for variables, parity in CHECKS
    if sum(assignment[v] for v in variables) % 2 != parity
)

print(f"\nbest assignment      = {assignment}")
print(f"unsatisfied checks   = {unsatisfied}  (objective = {best.objective:.0f})")

# %% [markdown]
# 全てのチェックが充足されており、FinITEが唯一の解を復元できたことがわかります。

# %% [markdown]
# ## まとめ
#
# 本チュートリアルでは次のことを行いました。
#
# - Max-3-XORSATをJijModelingで定式化し、各パリティチェックを3つのスピン変数$s_i = 1 - 2x_i$の積として書き直しました。
# - OMMXインスタンスを`FinITEConverter`に渡し、コスト演算子が純粋な3体スピンモデルになることを確認しました。チェック8個に対してPauli項は8個であり、制約1つにつきancilla1つで、二次化もペナルティ重みも不要です。
# - 論文のギャップに基づく閾値$\beta^\star(\bar F)$を計算する`finite_ite_beta_threshold`で$\beta$を選びました。必要なのはスペクトルギャップ$\Delta$と初期オーバーラップ$\gamma_0$の評価だけです。$F_g \ge 0.95$を目標とすると$\beta^\star \approx 1.77$が得られました。
# - $m(\beta)e^{-\beta\hat H}$の項ごとのLCUによるblock encodingを構築し、規格化定数が$\alpha = e^{\beta W} \approx 1197$であることを確認して、14量子ビットの回路(system6個+ancilla8個)にトランスパイルしました。
# - 回路をサンプリングし、ancillaが全て0であったショット(全体の約1.6%)をpost-selectionで残し、それらをデコードして`ommx.v1.SampleSet`を得ました。全てのチェックを充足する唯一の割り当て`[1, 1, 0, 1, 1, 1]`が復元されました。
#
# **本チュートリアルで扱わなかったこと** Qamomileが実装しているのは論文のstage 1、すなわちblock encodingです。post-selectionの成功確率を増幅するstage 2のfixed-point amplitude amplificationは含まれていないため、上記の実行では$O(1/P_{\mathrm{LCU}})$のショット数をそのまま必要とします。
