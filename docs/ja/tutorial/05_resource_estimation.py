# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# tags: [tutorial, resource-estimation]
# ---
#
# # リソース推定
#
# 量子カーネルを実機で実行する前に、必要な量子ビット幅、ゲート数、測定・reset回数、depthを把握したい場合があります。Qamomileの`estimate_resources()`を使うと、**量子カーネルを実行せずに**リソースを推定できます。ハードウェアのnative gateではなくalgorithmic levelのリソースを扱います。既定では、Qamomileのlogical gate表現とclean ancillaを使うToffoli制御分解モデルを組み合わせ、target固有の最適化は適用しません。concreteな量子カーネルとsymbolicな（パラメータ付き）量子カーネルの両方に対応しています。
#
# この章では以下を扱います：
#
# - 固定量子カーネルの基本的なリソース推定
# - gate basisとcoherent controlの分解方法を個別に選択
# - 制御、inverse call、SELECT、Pauli evolution、control flowの組み合わせ
# - ゲート、測定、resetリソースの分離
# - パラメータ付き量子カーネルのsymbolicなリソース推定
# - 構造上の要件、opaque boundary、trace、JSON向け出力
# - `.substitute()`によるスケーリング分析
# - ショアのアルゴリズムの位数探索への応用

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install "qamomile[qiskit,visualization]"

# %%
import qamomile.circuit as qmc

# %% [markdown]
# ## 固定量子カーネルのリソース推定
#
# パラメータを持たない量子カーネルに対しては、`estimate_resources()`は具体的な数値を返します。


# %%
@qmc.qkernel
def fixed_circuit() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(3, name="q")

    q[0] = qmc.h(q[0])
    q[0], q[1] = qmc.cx(q[0], q[1])
    q[1], q[2] = qmc.cx(q[1], q[2])

    return qmc.measure(q)


# %%
fixed_circuit.draw()

# %%
est = fixed_circuit.estimate_resources()
print("qubits:", est.qubits)
assert est.qubits == 3
print("total gates:", est.gates.total)
assert est.gates.total == 3
print("single-qubit gates:", est.gates.single_qubit)
assert est.gates.single_qubit == 1
print("two-qubit gates:", est.gates.two_qubit)
assert est.gates.two_qubit == 2
print("measurements:", est.measurements.total)
assert est.measurements.total == 3
assert est.resets.total == 0
assert est.depth.depth == 4
assert est.depth.gate_depth == 3
assert est.depth.measurement_depth == 1
assert est.depth.reset_depth == 0

# %% [markdown]
# ## Gate basisとcontrol decompositionの選択
#
# `basis`は結果に表示するgateの表現を選び、`control_decomposition`はcoherent controlの表現方法を選びます。既定値は`GateBasis.LOGICAL`と`ControlDecomposition.CLEAN_ANCILLA_TOFFOLI`です。どちらのcontrol modelでも、呼び出された量子カーネルの内部を再帰的に確認します。例えば既定モデルでは、3制御のHを4個のToffoli gate、1個のcontrolled-H gate、2個のclean ancillaとして推定します。


# %%
@qmc.qkernel
def one_h(target: qmc.Qubit) -> qmc.Qubit:
    return qmc.h(target)


@qmc.qkernel
def controlled_h() -> qmc.Qubit:
    controls = qmc.qubit_array(3, name="controls")
    target = qmc.qubit("target")
    *_, target = qmc.control(one_h, num_controls=3)(controls, target)
    return target


# %%
decomposed_controls = controlled_h.estimate_resources()
assert decomposed_controls.basis is qmc.GateBasis.LOGICAL
assert (
    decomposed_controls.control_decomposition
    is qmc.ControlDecomposition.CLEAN_ANCILLA_TOFFOLI
)
assert decomposed_controls.gates.total == 5
assert decomposed_controls.gates.toffoli == 4
assert decomposed_controls.width.clean_ancilla_qubits == 2
assert decomposed_controls.qubits == 6
assert decomposed_controls.derivation is qmc.EstimateDerivation.STRUCTURAL
assert decomposed_controls.guarantee is qmc.EstimateGuarantee.UPPER_BOUND

abstract_controls = controlled_h.estimate_resources(
    control_decomposition=qmc.ControlDecomposition.ABSTRACT,
)
assert abstract_controls.basis is qmc.GateBasis.LOGICAL
assert abstract_controls.control_decomposition is qmc.ControlDecomposition.ABSTRACT
assert abstract_controls.gates.total == 1
assert abstract_controls.width.clean_ancilla_qubits == 0
assert abstract_controls.qubits == 4

# %% [markdown]
# `CLEAN_ANCILLA_TOFFOLI`は、resource estimatorがcoherent controlを数えるための固定algorithmic modelです。通常のgateは種類によらず1単位のactive workとして数えます。2個以上のcontrolが2単位以上のactiveなworkを囲む場合、このmodelはcontrolの論理積を1回だけ計算してbody全体で共有します。operationが1個だけ、またはcontrolが1個だけの場合は、primitiveごとの保守的な分解を維持します。workの数は具体化済みのqkernel call、inverse block、branch、loopの内側まで再帰的に追うため、同じbodyを別qkernelへ切り出しただけでは結果が変わりません。nested control boundary、activeなSELECT case、Pauli evolutionは、構造上1つのleafでも共有に十分な内部controlled workを表す場合があります。このresource modelはengineのemission policyから意図的に独立しています。engine側が直接経路やnativeな多制御gateを使っても、推定式は自動的には変わりません。制御数にかかわらずソース上の各controlled primitiveを1個の抽象operationとして扱いたい場合は、`ABSTRACT`を選択します。これとは独立に、対応しているoperationをClifford+Tの合計リソースへ変換する場合は`CLIFFORD_T`を選び、任意回転の合成精度を`precision`で指定します。抽象的なcontrolled primitive自体はClifford+Tの内訳として報告できません。対応するClifford+T loweringがない場合も、架空のコストを算出せずエラーになります。これらの設定はrouting、ハードウェアのnative gateに合わせた最適化、誤り訂正のコスト算出を行いません。


# %%
import math


@qmc.qkernel
def identity_body(target: qmc.Qubit) -> qmc.Qubit:
    return target


@qmc.qkernel
def controlled_phase(theta: qmc.Float) -> qmc.Qubit:
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    _, target = qmc.control(identity_body)(
        control,
        target,
        global_phase=theta,
    )
    return target


# %%
phase_est = controlled_phase.estimate_resources(
    basis=qmc.GateBasis.CLIFFORD_T,
    precision=1e-3,
)
assert phase_est.substitute(theta=0).gates.total == 0
assert phase_est.substitute(theta=math.pi / 4).gates.t == 1
arbitrary_phase = phase_est.substitute(theta=0.3)
assert arbitrary_phase.guarantee is qmc.EstimateGuarantee.UPPER_BOUND
assert arbitrary_phase.approximation is qmc.ApproximationStatus.APPROXIMATE

# %% [markdown]
# 単独のglobal phaseは観測できないため、このtarget非依存モデルではコストを0とします。上の例のように量子制御下ではrelative phaseになります。代入後の角度が0、Z、S、Tなどのcanonicalな値なら厳密に分類し、任意の角度には指定した合成モデルを使います。
#
# 高水準の演算はopaqueな箱1個として数えず、実行上の意味に基づいて推定します。
#
# | 構成要素 | 推定時の挙動 |
# |---|---|
# | 本体を持つ量子カーネルのcallと`qmc.inverse(...)` | 選択された実装本体を再帰的にたどります。inverseはunitaryの意味を反転しますがgate数を保ち、大きな本体をcall1個には縮約しません。 |
# | `qmc.control(...)` | 入れ子になった本体へ外側の制御をすべて伝播します。`CLEAN_ANCILLA_TOFFOLI`では多制御の分解gateと再利用できるclean ancillaを含めます。`ABSTRACT`では各controlled primitiveを1個の抽象operationとして維持します。open controlには前後のXも含めます。 |
# | `qmc.select(...)` | indexの各量子ビットによる制御、open controlの前後のX、Vectorへのbroadcast、外側の制御を含め、出力されるすべての制御付きcase本体を合計します。宣言したindex幅はoperandと一致し、全caseを指定できる必要があります。 |
# | `qmc.pauli_evolve(...)` | Hermitian Hamiltonianを与えると、Pauli basis変換、parity ladder、軸回転、定数項による制御付きphaseを数えます。非可換なPauli sumはHamiltonianの項の順序による1-stepのfirst-order Lie–Trotterとして扱い、`approximate`と記録します。targetレジスタはHamiltonianのsupportを収容できる幅が必要です。 |
# | `if`、`for`、`for_items`、`while` | 可能な場合はコンパイル時のconditionとloop boundをsymbolicに保ちます。パラメータ分岐のgate数と幅は`Piecewise`で表し、測定結果に基づく分岐は保守的な最大値を使います。symbolicなaliasや複数wireの境界により厳密なscheduleを構成できない場合、dependency depthは`upper_bound`のままになります。loopのworkは累積し、再利用できる幅はlivenessに従います。扱えないcarry recurrenceは明示的にエラーにするか、仮定として表示します。 |
# | `qmc.expval(...)` | 抽象的なexpectation query 1回とmeasurement layer 1段を記録します。observable grouping、basis rotation、shot数、executorのsampling policyはengine/executorに依存するため、架空のgate数を出さず、仮定を明示して`derivation=MODELED`、`guarantee=UNKNOWN`とします。 |
# | LCU block encoding | 直接、制御、inverse、descriptor経由、serialization後のいずれでも、signalレジスタとsystemレジスタの厳密な幅要件を維持します。 |

# %% [markdown]
# ### リソースcountの算出方法、保証、数学的なapproximation
#
# 三つの独立したfieldが別々の問いに答えます。`estimate.derivation`は、見えているIRと選択した推定器の分解規則から数えたか（`STRUCTURAL`）、宣言済みまたはfallbackのmodelを使ったか（`MODELED`）を表します。`estimate.guarantee`は、そのcountが`EXACT`、安全側の`UPPER_BOUND`、どちらも保証できない`UNKNOWN`のいずれかを表します。`estimate.approximation`は、選択した回路について推定器が認識した数学的近似を記録します。ここで`EXACT`は「既知の近似を記録していない」という意味で、opaque boundaryの内側や外部helperがすでに低水準化した処理の数学的意味まで厳密だと証明するものではありません。そのため、`UPPER_BOUND`だから`APPROXIMATE`になるわけではなく、modelを使った推定が同時に上界保証を持つ場合もあります。


# %%
import qamomile.observable as qmo


@qmc.qkernel
def pauli_sum_evolution(
    hamiltonian: qmc.Observable,
    time: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    target = qmc.qubit_array(2, name="target")
    return qmc.pauli_evolve(target, hamiltonian, time)


noncommuting_hamiltonian = qmo.X(0) + qmo.Z(0)
noncommuting_est = pauli_sum_evolution.estimate_resources(
    inputs={"hamiltonian": noncommuting_hamiltonian, "time": 0.25},
)
zero_time_est = pauli_sum_evolution.estimate_resources(
    inputs={"hamiltonian": noncommuting_hamiltonian, "time": 0.0},
)
commuting_est = pauli_sum_evolution.estimate_resources(
    inputs={"hamiltonian": qmo.Z(0) + qmo.Z(1), "time": 0.25},
)

assert noncommuting_est.derivation is qmc.EstimateDerivation.STRUCTURAL
assert noncommuting_est.guarantee is qmc.EstimateGuarantee.EXACT
assert noncommuting_est.approximation is qmc.ApproximationStatus.APPROXIMATE
assert zero_time_est.approximation is qmc.ApproximationStatus.EXACT
assert commuting_est.approximation is qmc.ApproximationStatus.EXACT

# %% [markdown]
# 非可換な場合は、選択した1-step回路のリソースcountがexactであっても、理想的な指数関数を近似します。このときassumptionにはfirst-order Lie–Trotter formulaを使用したことを記録します。evolution timeが0ならoperation自体がなくなり、可換なPauli項にはproduct-formula approximationが不要なため、どちらも`ApproximationStatus.EXACT`を報告します。

# %% [markdown]
# ## symbolicなリソース推定
#
# 量子カーネルに未バインドのパラメータ（例：`n: qmc.UInt`）がある場合、`estimate_resources()`は**SymPy式**を返します。特定の値を選ばなくてもコストのスケーリングが分かります。


# %%
@qmc.qkernel
def scalable_circuit(n: qmc.UInt, theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    q = qmc.h(q)
    q = qmc.ry(q, theta)

    for i in qmc.range(n - 1):
        q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])

    return qmc.measure(q)


# %%
scalable_circuit.draw(n=4, fold_loops=False)

# %%
est = scalable_circuit.estimate_resources()
print("qubits:", est.qubits)
assert str(est.qubits) == "n"
print("total gates:", est.gates.total)
assert str(est.gates.total) == "2*n + Max(0, n - 1)"
print("single-qubit gates:", est.gates.single_qubit)
assert str(est.gates.single_qubit) == "2*n"
print("two-qubit gates:", est.gates.two_qubit)
assert str(est.gates.two_qubit) == "Max(0, n - 1)"
print("rotation gates:", est.gates.rotation_gates)
assert str(est.gates.rotation_gates) == "n"
print("parameters:", est.parameters)
assert set(est.parameters.keys()) == {"n"}

# %% [markdown]
# 出力には、量子ビット数を表す`n`や総ゲート数を表す`2*n + Max(0, n - 1)`のようなSymPy式が含まれます。明示的またはfallbackのmodelを使ったかは`est.derivation`、countが厳密か、上界か、どちらも保証できないかは`est.guarantee`で確認できます。
#
# `Max(0, ...)`は`qmc.range(n - 1)`のループ回数に由来します。`n`が未束縛のため`n >= 1`を仮定できず、`n = 0`のときに回数が`-1`になってしまわないよう0で下限を取っています。具体的な`n >= 1`を代入すればこのガードは外れるので、後述の合計値はそのまま整数になります。

# %% [markdown]
# ### 入力shapeの指定と要件の検証
#
# concreteな量子カーネル入力によってリソースが変わる場合は`inputs={...}`を使います。classical Vectorには通常のarray-likeな値を渡すと、そのshapeで推定を具体化できます。1次元の`Vector[Qubit]`入力には、dummyの量子ビットオブジェクトを作らず、幅を整数で直接指定できます。


# %%
@qmc.qkernel
def vector_input(register: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    return qmc.h(register)


# %%
vector_est = vector_input.estimate_resources(inputs={"register": 8})
assert vector_est.width.input_qubits == 8
assert vector_est.width.allocated_qubits == 0
assert vector_est.gates.total == 8
assert vector_est.parameters == {}

# %% [markdown]
# 幅、index、arityは単なる目安ではなく、リソース推定の要件です。確保数が非負整数であること、配列のaccessとview、control index、SELECTのindex幅と1個以上のtarget幅、Pauli support、block encodingのsignal/systemレジスタについて、推定中も要件を維持します。descriptorから作成したblock encodingは1次元の量子port幅を正確に保持するため、`encoding.unitary.estimate_resources()`だけで自動的に具体化します。wrapper量子カーネルを推定する場合は`inputs={"signal": encoding.num_signal_qubits, "system": encoding.num_system_qubits}`を指定でき、同じ契約を明示的に検証できます。`inputs`を渡した時点と`.substitute()`後の両方で、制御形やinverse形も含めて検証します。不正な値はもっともらしい推定値へ補正せず`ValueError`にします。serializationした出力では、これらを`requirements`リストで確認できます。
#
# 構造を決める具体値は、可能な限り最初の`estimate_resources()`呼び出しで`inputs`として渡します。こうするとdepth式を構築する前に、dependency schedulerが配列の物理index、view、loopで引き継がれるwireの同一性を解決できます。後から`.substitute()`してもsymbolic推定の安全な評価と維持された要件の検証はできますが、すでに保守的に構築されたalias scheduleは再構築できません。そのため、すべてのsymbolがconcreteになってもdepthは`upper_bound`のままになる場合があります。

# %% [markdown]
# ## `ResourceEstimate`フィールドリファレンス
#
# | フィールド | 説明 |
# |-------|------------|
# | `est.qubits` | 分解用ancillaを含むpeak時のliveな論理量子ビット数 |
# | `est.circuit_qubits` | 保守的なstatic circuit width |
# | `est.width.input_qubits` | 呼び出し側から渡される量子ビット数 |
# | `est.width.allocated_qubits` | 量子カーネル本体が確保する量子ビット数 |
# | `est.width.clean_ancilla_qubits` | 再利用できる分解用clean ancilla数 |
# | `est.gates.total` | 総ゲート数 |
# | `est.gates.single_qubit` | 単一量子ビットゲート数 |
# | `est.gates.two_qubit` | 2量子ビットゲート数 |
# | `est.gates.multi_qubit` | 多量子ビットゲート数（3量子ビット以上） |
# | `est.gates.t_gates` | Tゲート数 |
# | `est.gates.clifford_gates` | Cliffordゲート数 |
# | `est.gates.rotation_gates` | 回転ゲート数 |
# | `est.measurements.total` | 1回のlogical executionで量子ビットを測定する回数 |
# | `est.resets.total` | 1回のlogical executionで明示的に量子ビットをresetする回数 |
# | `est.depth.depth` | dependencyを考慮した全操作のalgorithmic depth |
# | `est.depth.gate_depth` | ゲートだけのalgorithmic depth |
# | `est.depth.measurement_depth` | 測定だけのalgorithmic depth |
# | `est.depth.reset_depth` | resetだけのalgorithmic depth |
# | `est.calls.calls_by_name` | 展開せずに残した名前付きsemantic boundaryのcall回数 |
# | `est.calls.queries_by_name` | 名前別のalgorithmic query complexity |
# | `est.parameters` | シンボル名からSymPyシンボルへの辞書 |
# | `est.basis` | 選択したgate basis（`logical`または`clifford_t`） |
# | `est.control_decomposition` | 選択したcoherent controlモデル（`abstract`または`clean_ancilla_toffoli`） |
# | `est.precision` | `clifford_t`のrotation synthesis精度 |
# | `est.derivation` | `structural`または`modeled`：countをどのように求めたか |
# | `est.guarantee` | `exact`、`upper_bound`、`unknown`のいずれか：選択した回路costに対するcountの保証 |
# | `est.approximation` | `exact`または`approximate`：推定器が数学的近似を認識したか |
# | `est.assumptions` | 推定時に有効なモデル上の仮定 |
# | `est.trace` / `est.explain()` | 任意で保持する説明treeとそのテキスト表示 |
#
# 数値リソースのfieldはSymPy式です。固定量子カーネルの場合は通常の整数に評価されます。`derivation`はresource countの求め方、`guarantee`は選択した回路costに対するcountの関係、`approximation`は推定器が認識した数学的近似を、それぞれ独立に示します。`approximation=EXACT`は「既知の近似を記録していない」という意味で、opaqueな意味まで厳密だと証明するものではありません。測定とresetはgateとして数えません。`N`量子ビットのvectorを測定すると`measurements.total`は`N`増えますが、並列に読み出せる場合の`measurement_depth`は1 layerです。countは量子カーネルをlogicalに1回実行した場合の値で、shots倍しません。`qmc.expval`はobservable grouping、basis rotation、shotsがexecutorに依存するため、`measurements.total`を0のままにします。ここでの0は「測定不要」ではなく「この推定には含めていない」という意味で、その不確実性は`MODELED`という算出方法、`UNKNOWN`という保証、assumption、abstract query、measurement layerで明示します。`resets.total`が数えるのは明示的な`qmc.reset`だけで、`|0>`状態の新規確保やengineがtargetに合わせて挿入するresetは含みません。各種類のdepthは独立してscheduleされるため、加減算から`depth.depth`を復元することはできません。
#
# `calls_by_name`は通常の本体を持つ量子カーネルのcallを意図的に数えません。その本体はすでに展開され、gate、幅、depth、測定、resetに反映されているためです。ここに記録するのは、選択したmodelで展開せずに残した名前付きboundaryです。明示的またはunknownなopaque callに加え、具体的なsampling実装をexecutorが決める`qmc.expval`のようなmodeled semantic boundaryも含みます。そのため、1回の`expval`はmodeledなmeasurement layerに加えて、`calls_by_name={"expval": 1}`と`queries_by_name={"expval": 1}`を記録します。

# %% [markdown]
# ## Opaque boundary、算出条件、trace
#
# 本体を持たないcallableは、コストを指定しない限り正確なgate数が分かりません。そのため既定の`UnknownResourcePolicy.ERROR`ではエラーになります。コストが分かっている場合は、`qmc.opaque(...)`に明示的な`ResourceEstimate`を指定してください。探索的な用途では、`OPAQUE_CALL`は名前付きのcallとqueryを記録し、`derivation=MODELED`、`guarantee=UNKNOWN`とします。`ZERO_WITH_WARNING`も同じmetadataでコスト0という仮定を記録します。どちらも未知の本体を分解したかのようには扱いません。
#
# 固定`ResourceEstimate`とcontext-dependentな`cost(ctx)` callbackは、どちらもOracle定義を1回適用するbase costを記述します。このbase costには`qmc.opaque(..., num_control_qubits=...)`で宣言したcontrol（D）がすでに含まれますが、後から`qmc.control(oracle, ...)`で追加したcontrol（A）と、外側のcontrolled qkernelから継承したcontrol（K）は含まれません。推定器はどちらのbase costを受け取った後も、inverseとこれらの外部controlを適用します。
#
# callbackが受け取るのは完全なcall siteではなく、`OpaqueCostContext`です。`target_qubits`、targetごとの`target_shapes`、`definition_control_qubits`、`basis`、`control_decomposition`、`precision`、任意のbase `strategy`を参照できます。追加control、継承control、inverse、open-control値は意図的に含めていません。callbackは定義levelのbase costだけを記述し、これらのcall-site transformは推定器だけが反映します。ただし、ほかの利用者指定costと同様に、callbackの作成者にはこの契約に沿ったbase costを返す責任があります。
#
# gate modelに依存するfieldを含む固定costは、その数値を求めたときのmodel設定も固定します。異なる`basis`、`control_decomposition`、またはClifford+Tの`precision`でそのOracleを推定すると、同じgate数へ別の意味を黙って付け直すのではなくエラーになります。callsとqueriesだけを含むcostのようにgate modelへ依存しないcostは、解釈し直すgate内訳がないため複数のmodelで利用できます。複数のgate modelへ対応するcallbackは、返り値を`basis=ctx.basis`、`control_decomposition=ctx.control_decomposition`、`precision=ctx.precision`で構築します。
#
# | controlの由来 | callbackから見えるか | costを含める側 |
# |---|---|---|
# | Oracle定義時に`num_control_qubits`で宣言したcontrol（D） | `ctx.definition_control_qubits` | 固定costまたはcallbackがbase costへ含めます。 |
# | 後から`qmc.control(oracle, ...)`で追加したcontrol（A） | いいえ | 推定器がbase costを受け取った後で適用します。 |
# | 外側のcontrolled qkernelから継承したcontrol（K） | いいえ | 推定器がbase costを受け取った後で適用します。 |
#
# どちらの明示costでも、そのcostに指定した`calls_by_name`と`queries_by_name`だけを記録し、推定器はOracleのcallやqueryを追加しません。測定またはresetを含むbase costは、controlされておらずinverseでもない定義でのみ使えます。D、A、Kのいずれかのcoherent controlまたはinverseがある場合、countだけでは変換後の実装を定義できないため、そのcostを拒否します。
#
# `ABSTRACT`では、外部controlを追加しても`total`を変えず、arityが分かっているbucketだけを移します。1量子ビットgateは1制御で2量子ビットgateになり、2制御以上ではmulti-qubitになります。arity不明のgateは不明のままです。`CLEAN_ANCILLA_TOFFOLI`では、`single_qubit`または`two_qubit`が分かるlogical profileを分解して推定できます。2個以上のモデル化されたoperationへ2個以上の外部controlを付ける場合、この推定modelはcontrolのANDを1回だけ計算し、既知の各primitiveをその1個の実効controlの下へ投影して、本体の後で共有ladderを逆計算します。operationが1個だけ、または外部controlが1個だけの場合は、primitiveごとの分解を使います。`CLEAN_ANCILLA_TOFFOLI`は固定されたresource-estimation modelの名前です。その式は、engine側のemission policyを変更しても自動的には変化しません。
#
# aggregate profileにはgate名や元のscheduleがありません。そのためarity fieldとgate-family fieldはfieldごとに独立したboundであり、その和が`total`と一致しない場合があります。arity不明のgateを`multi_qubit`へ誤分類することはありません。結果は`derivation=MODELED`です。与えたprofileが安全側へ投影するのに十分なら`guarantee=UPPER_BOUND`、不足する場合は`guarantee=UNKNOWN`とし、その制限を`assumptions`へ記録します。aggregateなClifford+T gate profileへ外部controlを付ける場合は、arity countだけではClifford+T loweringを特定できないため拒否します。calls/queryだけのcostには変換対象のgate profileがないので、それらのcounterは見える仮定とともに変更せず保持します。gate固有の変換後costが必要なら本体を持つcallableを使うか、宣言controlとbase costにその実装を含めた別Oracleを定義します。


# %%
costed_oracle = qmc.opaque(
    "costed_oracle",
    num_qubits=2,
    cost=qmc.ResourceEstimate(
        gates=qmc.GateResources(
            total=5,
            single_qubit=2,
            two_qubit=1,
        ),
        calls=qmc.CallResources(
            queries_by_name={"costed_oracle": 1},
        ),
    ),
)


@qmc.qkernel
def controlled_costed_oracle() -> tuple[qmc.Qubit, qmc.Qubit]:
    control_0 = qmc.qubit("control_0")
    control_1 = qmc.qubit("control_1")
    target_0 = qmc.qubit("target_0")
    target_1 = qmc.qubit("target_1")
    *_, target_0, target_1 = qmc.control(
        costed_oracle,
        num_controls=2,
    )(control_0, control_1, target_0, target_1)
    return target_0, target_1


# %%
costed_est = controlled_costed_oracle.estimate_resources()
assert costed_est.gates.total == 9
assert costed_est.depth.depth == 9
assert costed_est.width.clean_ancilla_qubits == 2
assert costed_est.calls.queries_by_name == {"costed_oracle": 1}
assert costed_est.derivation is qmc.EstimateDerivation.MODELED
assert costed_est.guarantee is qmc.EstimateGuarantee.UNKNOWN
assert any(
    "2 gate(s) with unclassified arity" in assumption.message
    for assumption in costed_est.assumptions
)


# %% [markdown]
# このprofileは5個のmodel workへ2個のcontrolを付けるため、固定recipeは1本のshared ladderを使います。controlのANDを計算して元に戻す部分は2個のToffoliです。得られた1個の実効controlの下では、1量子ビットentry 2個が2gate、保守的な2量子ビットenvelopeが3gate、未解決のplaceholder 2個が2gateなので、totalは`2 + 2 + 3 + 2 = 9`です。clean ancilla 2個は、ladder用の1個と保守的な2量子ビットenvelope内部で必要な最大1個を同時に保持するためです。


# %%
oracle = qmc.opaque("conditional_oracle", num_qubits=1)


@qmc.qkernel
def conditional_resource_branch(flag: qmc.UInt) -> qmc.Qubit:
    target = qmc.qubit("target")
    if flag:
        target = qmc.h(target)
    else:
        (target,) = oracle(target)
    return target


# %%
conditional_est = conditional_resource_branch.estimate_resources(
    unknown_policy=qmc.UnknownResourcePolicy.OPAQUE_CALL,
    trace=True,
)
exact_branch = conditional_est.substitute(flag=1)
opaque_branch = conditional_est.substitute(flag=0)

assert exact_branch.calls.calls_by_name == {}
assert exact_branch.derivation is qmc.EstimateDerivation.STRUCTURAL
assert exact_branch.guarantee is qmc.EstimateGuarantee.EXACT
assert "conditional_oracle" not in exact_branch.explain()
assert opaque_branch.calls.calls_by_name == {"conditional_oracle": 1}
assert opaque_branch.calls.queries_by_name == {"conditional_oracle": 1}
assert opaque_branch.derivation is qmc.EstimateDerivation.MODELED
assert opaque_branch.guarantee is qmc.EstimateGuarantee.UNKNOWN
assert "conditional_oracle" in opaque_branch.explain()

# %% [markdown]
# 仮定、derivation/guaranteeの情報、callの集計、trace nodeには、数値リソースと同じsymbolicな分岐guardが付きます。上の例のように`inputs`または`.substitute()`で分岐を選ぶと、実行されない側の警告やopaque callは消えます。説明treeが必要な場合だけ`trace=True`を指定します。既定では保持しないため、推定結果は小さいままです。`est.explain()`を使うと、再帰的な本体、primitive、opaqueの説明を表示できます。

# %% [markdown]
# ### JSON向け出力
#
# `to_dict()`はJSONへ変換しやすいレポート用snapshotを生成します。symbolic式と構造上の要件は文字列で格納し、basis、precision、derivation、guarantee、approximation status、仮定も含めます。compactなpayloadにopt-inのtraceは埋め込まず、`explain()`で別に表示します。このsnapshotは`ResourceEstimate`を復元するserialization形式ではありません。文字列にはQamomile固有のsymbolic nodeが含まれることがあるため、`sympy.sympify()`で評価しないでください。具体値のレポートを出力する場合は、元のestimateを先に`.substitute(...)`で具体化します。未bindの量子カーネルと対応する固定opaque costを保存する場合は、代わりに`qamomile.circuit.serialization.serialize()`を使います。


# %%
import json

payload = json.loads(json.dumps(conditional_est.to_dict()))
assert payload["basis"] == "logical"
assert payload["control_decomposition"] == "clean_ancilla_toffoli"
assert payload["derivation"] == "modeled"
assert payload["guarantee"] == "unknown"
assert payload["approximation"] == "exact"
assert "requirements" in payload

# %% [markdown]
# ## `.substitute()`によるスケーリング分析
#
# symbolic式は*数式*を示してくれますが、特定のサイズでの具体的な数値も確認したい場合`.substitute()`で既存の推定を評価できます。具体値がindex、view、loop構造などのscheduling判断に影響する場合は、前述のように推定を構築する時点で`inputs`として渡してください。

# %%
for n_val in [4, 8, 16, 32]:
    c = est.substitute(n=n_val)
    print(
        f"n={n_val:2d}: {int(c.gates.total):>3} gates total, {int(c.gates.two_qubit):>2} two-qubit"
    )
    assert int(c.gates.total) == 3 * n_val - 1
    assert int(c.gates.two_qubit) == n_val - 1

# %% [markdown]
# ## ショアのアルゴリズムを回路本体から推定する
#
# 実用的な例として、ショアのアルゴリズムの量子部分である位数探索を調べます。
# `qmc.shor_order_finding()`は、底`base`と法`modulus`を受け取り、実行にもリソース推定にも使える1つの量子カーネルを返します。レジスタ幅は`modulus.bit_length()`から決まるため、返されたカーネルに人工的な`n`引数はありません。

# %%
order_finding = qmc.shor_order_finding(base=2, modulus=15)
shor_est = order_finding.estimate_resources()

print("default peak qubits:", shor_est.qubits)
print("default total gates:", shor_est.gates.total)
print("measurements:", shor_est.measurements.total)
print("resets:", shor_est.resets.total)
print("estimate derivation:", shor_est.derivation)
print("estimate guarantee:", shor_est.guarantee)

assert shor_est.parameters == {}
assert shor_est.width.allocated_qubits == 21
assert shor_est.width.clean_ancilla_qubits == 2
assert shor_est.qubits == 23
assert shor_est.gates.total == 4585
assert shor_est.measurements.total == 80
assert shor_est.resets.total == 80
assert shor_est.derivation is qmc.EstimateDerivation.STRUCTURAL
assert shor_est.guarantee is qmc.EstimateGuarantee.UPPER_BOUND

# %% [markdown]
# この実装は`2*n`量子ビットのcountingレジスタを同時に保持しません。1つの位相量子ビットを測定・resetして再利用し、それまでに得たビットで半古典的inverse QFTの位相補正を行います。上の80回の測定・resetのうち、8回はこの位相読み出しと再利用、72回は算術内部のmeasurement-assisted carry ventingによるものです。明示的なresetは`gates.total`ではなく`resets.total`に反映されます。
#
# 固定window幅を`w`とすると、制御分解前の回路本体のpeak-live allocationは`3*n + w + 7`論理量子ビットになります。
#
# | 用途 | 幅 |
# |---|---:|
# | 再利用する位相量子ビット | `1` |
# | モジュラ値を保持するworkレジスタ | `n` |
# | モジュラ乗算のaccumulator | `n` |
# | window lookupの出力レジスタ | `n` |
# | lookupのaddress | `w` |
# | carry、vent、overflow、reduction、domain、enable | `6` |
#
# 既定値は`w=2`なので、回路本体のallocationは`3*n + 9`です。法15では`n=4`となり、本体は21量子ビットです。既定の`CLEAN_ANCILLA_TOFFOLI`モデルは再利用可能なclean ancillaも2個報告するため、peak幅とstatic circuit widthは23になります。`ABSTRACT`を選ぶと、分解前の幅21を維持します。これらは外部のコスト式を`estimate_resources()`へ登録した値ではなく、実行される量子カーネルの本体とtransformから得られます。

# %%
import sympy as sp
from IPython.display import Math, display

symbolic_n, symbolic_w = sp.symbols("n w", integer=True, positive=True)
shor_body_width = 3 * symbolic_n + symbolic_w + 7
shor_decomposed_width = shor_body_width + 2
display(Math(rf"N_\mathrm{{body}} = {sp.latex(shor_body_width)}"))
display(Math(rf"N_\mathrm{{decomposed}} = {sp.latex(shor_decomposed_width)}"))
assert (
    shor_body_width.subs({symbolic_n: 4, symbolic_w: 2})
    == shor_est.width.allocated_qubits
)
assert shor_decomposed_width.subs({symbolic_n: 4, symbolic_w: 2}) == shor_est.qubits

# %% [markdown]
# 法15に対するgate内訳も、同じ実行bodyを最後までたどって得られます。

# %%
print("single-qubit gates:", shor_est.gates.single_qubit)
print("two-qubit gates:", shor_est.gates.two_qubit)
print("multi-qubit gates:", shor_est.gates.multi_qubit)
print("Toffoli gates:", shor_est.gates.toffoli)

assert shor_est.gates.total == (
    shor_est.gates.single_qubit + shor_est.gates.two_qubit + shor_est.gates.multi_qubit
)

# %% [markdown]
# `guarantee`が`upper_bound`なのは、途中測定に基づくclassical feed-forwardの分岐を安全側に数え、`CLEAN_ANCILLA_TOFFOLI`が保守的な多制御モデルであるためです。算出方法は`structural`のままであり、この上界自体が回路を数学的な近似にするわけではありません。ここで得られるのはalgorithmic circuitのリソースであり、特定deviceのnative gateへの分解、routing、誤り訂正、magic state生成などは含みません。

# %% [markdown]
# ### なぜgate数は`O(n^3)`なのか
#
# `qmc.modmul_const()`は、`w`ビットずつsourceを読み、古典的な倍数をlookupしてaccumulatorへ足します。1回のモジュラ乗算には約`n / w`個のwindowがあります。各windowで使うripple-carry加算、定数減算、比較、条件付き復元はいずれも`O(n)`gateです。定数加減算には[Gidneyのcarry-venting adder](https://arxiv.org/abs/2507.23079)を使い、workレジスタをdirty workspaceとして借りるため、別の`n`量子ビットレジスタは増えません。
#
# したがって固定`w`では、モジュラ乗算1回が`O(n^2)`、既定の位数探索で行う`2*n`回のcontrolled multiplicationが`O(n^3)`です。半古典的inverse QFTのfeed-forwardは`O(n^2)`なので、全体のleading orderを変えません。lookupの`2^w`依存まで書けば、おおよそ`O(2^w n^3 / w)`です。

# %% [markdown]
# ### モジュラ乗算単体も同じbodyから推定する
#
# 公開primitiveは`qmc.modmul_const()`です。FTQC算術は具体的な問題インスタンスへspecializeしてから構築するため、ここでも幅4の量子カーネルをそのまま推定します。


# %%
@qmc.qkernel
def modular_multiplier() -> qmc.Vector[qmc.Qubit]:
    reg = qmc.qubit_array(4, name="reg")
    return qmc.modmul_const(
        reg,
        multiplier=2,
        modulus=15,
        window_size=2,
    )


# %%
window_est = modular_multiplier.estimate_resources()
print("windowed arithmetic qubits:", window_est.qubits)
print("windowed arithmetic gates:", window_est.gates.total)
assert window_est.width.allocated_qubits == 3 * 4 + 2 + 7
assert window_est.width.clean_ancilla_qubits == 2
assert window_est.qubits == 3 * 4 + 2 + 9
assert window_est.gates.total == 2272
assert window_est.measurements.total == 36
assert window_est.resets.total == 36

# %% [markdown]
# standalone版では、無条件実行を表す内部controlが位相量子ビットの代わりになるため、回路本体は位数探索全体と同じ`3*n + w + 7`量子ビットを確保します。既定のpeak幅は、制御分解用のclean ancillaにより2量子ビット大きくなります。`modmul_const()`は`x < modulus`の領域では`|x> -> |a*x mod modulus>`を実行し、領域外の基底状態はunitaryを保つため変更しません。

# %% [markdown]
# ### Ekerå–Håstadの短い指数スケジュール
#
# `qmc.ekera_hastad_factoring()`は、同程度のビット長を持つ2素数の積を対象とする[Ekerå–Håstad法](https://arxiv.org/abs/1702.00249)のshort discrete logarithm量子段を構築します。Qamomileでは`m = ceil(n / 2) + 1`とし、長さ`2*m`と`m`の位相スケジュールを順番に測定します。
#
# 2つの指数レジスタをcoherentに保持するのではなく、同じ位相量子ビットと算術workspaceを再利用します。そのため回路本体のallocationはShorと同じ`3*n + w + 7`で、`CLEAN_ANCILLA_TOFFOLI`では同じclean ancillaが加わります。異なるのはcontrolled modular multiplicationの回数です。返される`Vector[Bit]`は、先頭`2*m`ビットが長いスケジュール、残り`m`ビットが短いスケジュールで、それぞれlittle-endianです。
#
# 以下の法5のインスタンスは、リソース推定を小規模に確認するためのfixtureであり、完全な因数分解例ではありません。実際の因数分解では、2つの素因数を求める対象の合成数を法として与えます。

# %%
short_dlp = qmc.ekera_hastad_factoring(
    generator=2,
    modulus=5,
    window_size=2,
)
short_dlp_est = short_dlp.estimate_resources()

print("Ekerå–Håstad default qubits:", short_dlp_est.qubits)
print("Ekerå–Håstad default gates:", short_dlp_est.gates.total)
assert short_dlp_est.width.allocated_qubits == 3 * 3 + 2 + 7
assert short_dlp_est.width.clean_ancilla_qubits == 2
assert short_dlp_est.qubits == 3 * 3 + 2 + 9
assert short_dlp_est.gates.total == 4950
assert short_dlp_est.measurements.total == 81
assert short_dlp_est.resets.total == 81
assert short_dlp.output_types == [qmc.Vector[qmc.Bit]]

# %% [markdown]
# ## まとめ
#
# - `estimate_resources()`は実行せずにalgorithmic levelの量子ビット幅、ゲート、測定・reset回数、depthを算出します。
# - gate vocabularyとcoherent controlの分解は別々に選択します。既定値は`logical`と`clean_ancilla_toffoli`です。`abstract`を選ぶと制御されたソースprimitiveを分解せず維持し、`clifford_t`を選ぶと対応しているClifford+T loweringを適用します。
# - call、inverse call、SELECT、global phase、Pauli evolution、control flowは1gateに縮約せず、本体と実行上の意味を組み合わせて推定します。
# - `expval`は抽象的なqueryとmeasurement layerとして`modeled`推定にし、grouping、basis変換、shotのコストは選択したexecutorに委ねます。
# - パラメータ付き量子カーネルでは、選択したモデルにおけるスケーリングがSymPy式になります。
# - `inputs`にはclassicalな値と配列shapeに加え、1次元の量子Vectorの幅を整数で指定できます。維持された要件により、不正な幅やindexは拒否されます。
# - 結果を解釈する前に、`basis`、`derivation`、`guarantee`、`approximation`、`assumptions`、opt-inのtraceを確認します。`derivation`は構造から数えたかmodelを使ったか、`guarantee`は選択した回路costに対する関係、`approximation`は推定器が認識した数学的な近似を独立に示します。分岐を選ぶと実行されない側のmetadataは消えます。
# - `calls_by_name`が表すのは、opaque callや`expval`のようなmodeled semantic operationを含む、展開せずに残した名前付きboundaryです。本体を持つcallは再帰的に展開します。固定costとcallbackによるopaque costは、どちらも1回のbase Oracle適用を記述します。推定器は後から追加または継承したcontrolを適用し、既知の1量子ビット部分と2量子ビット部分を選択したcontrol decompositionで投影し、残りのgateは目に見えるmodeledなplaceholderとして維持します。
# - `to_dict()`は表示・レポート用snapshotを出力します。具体値を保存する場合は、元のestimateを先に`.substitute(...)`で具体化します。
# - `.substitute(n=...)`で既存の推定を特定サイズに評価して実行可能性を確認し、具体的な構造によりdependency schedulingを精密化したい場合は最初から`inputs`を使います。
# - FTQC版のShorとEkerå–Håstadは、同じ`O(n^2)`のwindowed modular multiplication bodyと、1つの再利用可能な位相量子ビットを共有します。
# - 固定window幅では、回路本体は`3*n + w + 7`量子ビットを確保します。既定のclean-ancilla Toffoli分解は最大2個の再利用可能なclean ancillaを加え、Shorの既定精度でgate数は`O(n^3)`です。
# - 問題インスタンスにspecializeされたFTQC factoryは、実行可能なbodyから具体的な幅とgate推定値を返します。
#
# **次へ**：[実行モデル](06_execution_models.ipynb)では、`sample()`と`run()`、オブザーバブル、ビット順序を扱います。
