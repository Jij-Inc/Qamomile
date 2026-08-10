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
# このチュートリアルでは、量子カーネルを実行せずにlogical resourceを推定し、その結果を読み取る方法を学びます。具体的には、次の内容を扱います。
#
# - パラメータなし・パラメータ付きの量子カーネルを推定する
# - 量子ビット幅、gate数、measurement/reset数、depthを確認する
# - 本体を持たないOracleへ固定costまたはcallbackを設定する
# - controlの分解方法と、costが不明なOracleの扱いを選ぶ
# - `derivation`、`quality`、`approximation`、`assumptions`から推定結果の意味を確認する
#
# 最後のAppendixには、`ResourceEstimate`で確認できる全フィールドをまとめます。

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install "qamomile[qiskit,visualization]"

# %% [markdown]
# このチュートリアルで使うライブラリをimportします。

# %%
import sympy as sp

import qamomile.circuit as qmc
import qamomile.observable as qmo

# %% [markdown]
# ## 1. リソース推定を実行する
#
# `estimate_resources()`を使うと、量子カーネルを実行したり、特定のengine向けにtranspileしたりせずに、量子アルゴリズム上のlogical resourceを見積もれます。パラメータ付きの量子カーネルも、パラメータを固定せずに推定できます。

# %% [markdown]
# ### 1.1 パラメータなしの量子カーネルを推定する
#
# 最初に、Bell状態を準備して2個の量子ビットを測定する量子カーネルを定義します。


# %%
@qmc.qkernel
def bell_pair() -> tuple[qmc.Bit, qmc.Bit]:
    """Prepare and measure one Bell pair."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")

    control = qmc.h(control)
    control, target = qmc.cx(control, target)

    return qmc.measure(control), qmc.measure(target)


# %% [markdown]
# 量子カーネルの`estimate_resources()`を呼び出すと、`ResourceEstimate`が返ります。これは量子カーネルを実行して得る測定結果ではありません。Qamomileが量子カーネルから作成したIRを解析し、量子ビット幅、logical gate数、測定event数、depthなどを数えた結果です。ここでは代表的な値だけを表示します。

# %%
bell_estimate = bell_pair.estimate_resources()

print("qubits:", bell_estimate.qubits)
print("total gates:", bell_estimate.gates.total)
print("single qubit gates:", bell_estimate.gates.single_qubit)
print("two qubit gates:", bell_estimate.gates.two_qubit)
print("measurements:", bell_estimate.measurements.total)
print("depth:", bell_estimate.depth.depth)
print("gate depth:", bell_estimate.depth.gate_depth)

assert bell_estimate.qubits - 2 == 0
assert bell_estimate.gates.total - 2 == 0
assert bell_estimate.gates.single_qubit - 1 == 0
assert bell_estimate.gates.two_qubit - 1 == 0
assert bell_estimate.measurements.total - 2 == 0
assert bell_estimate.depth.depth - 3 == 0
assert bell_estimate.depth.gate_depth - 2 == 0

# %% [markdown]
# Bell回路では、2個の量子ビット、Hadamard gateとCX gateの合計2個のgate、2回のmeasurementが必要です。gate列の後にmeasurement layerがあるため、全体のdepthは3です。

# %% [markdown]
# ### 1.2 パラメータ付き量子カーネルを推定する
#
# Bell状態は、2個の量子ビットからなるGHZ（Greenberger–Horne–Zeilinger）状態と考えられます。これを`n`個の量子ビットからなるGHZ状態へ一般化しましょう。最初の量子ビットへHadamard gateを適用し、隣り合う量子ビットをCX gateで順番に接続します。


# %%
@qmc.qkernel
def ghz_state(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Prepare and measure a GHZ state with a symbolic width."""
    qubits = qmc.qubit_array(n, "qubits")
    qubits[0] = qmc.h(qubits[0])
    for index in qmc.range(n - 1):
        qubits[index], qubits[index + 1] = qmc.cx(
            qubits[index],
            qubits[index + 1],
        )
    return qmc.measure(qubits)


# %% [markdown]
# `n`の値を渡さなくても、symbolicにリソースを推定できます。

# %%
ghz_symbolic_estimate = ghz_state.estimate_resources()

print("parameters:", ghz_symbolic_estimate.parameters)
print("qubits:", ghz_symbolic_estimate.qubits)
print("total gates:", ghz_symbolic_estimate.gates.total)
print("measurements:", ghz_symbolic_estimate.measurements.total)
print("depth:", ghz_symbolic_estimate.depth.depth)
print("gate depth:", ghz_symbolic_estimate.depth.gate_depth)

n = ghz_symbolic_estimate.parameters["n"]
cx_count = sp.Max(0, n - 1)
expected_depth = sp.Piecewise(
    (cx_count + 2, n > 0),
    (cx_count + 1, True),
)

assert sp.simplify(ghz_symbolic_estimate.qubits - n) == 0
assert sp.simplify(ghz_symbolic_estimate.gates.total - (cx_count + 1)) == 0
assert sp.simplify(ghz_symbolic_estimate.measurements.total - n) == 0
assert sp.simplify(ghz_symbolic_estimate.depth.depth - expected_depth) == 0
assert sp.simplify(ghz_symbolic_estimate.depth.gate_depth - (cx_count + 1)) == 0

# %% [markdown]
# GHZ回路では、`n >= 1`のとき、1個のHadamard gateと`n - 1`個のCX gateを使います。`qmc.UInt`は0も表せるため、まだ値が決まっていない段階では、ループ回数が`Max(0, n - 1)`で表されます。したがって、`gates.total`には`Max(0, n - 1) + 1`が現れます。GHZ状態を構成できる`n >= 1`の範囲では、これは`n`と同じです。
#
# 確保する量子ビット数と測定数も`n`になり、問題サイズ`n`に応じて複数のリソース項目がどのように増えるかを確認できます。`ghz_state`は`qubits[0]`へアクセスするため、有効な入力は`n >= 1`です。symbolicな式では、この入力要件を使って式を簡約しないため、`Piecewise`の条件が残ります。有効な範囲では、gate depthは`n`、測定を含む全体のdepthは`n + 1`です。

# %% [markdown]
# ### 1.3 特定の入力で具体化する
#
# 特定の入力に対する具体的な値が必要な場合は、推定時に`inputs`を渡す方法と、すでに得たsymbolicな推定結果へ`.substitute()`を適用する方法があります。
#
# | 方法 | 値を使うタイミング | 適した用途 |
# |---|---|---|
# | `estimate_resources(inputs={...})` | リソース推定を行うとき | 配列shape、index、branch、loop構造などを最初から具体化したい場合 |
# | `estimate.substitute(...)` | すでに得た推定式を評価するとき | 同じsymbolicな推定結果を複数の問題サイズで比較したい場合 |

# %% [markdown]
# まず、`inputs`を使って、推定を始める時点で`n=4`を具体化します。

# %%
ghz_input_estimate = ghz_state.estimate_resources(
    inputs={"n": 4},
)

print("inputs -> qubits:", ghz_input_estimate.qubits)
print("inputs -> total gates:", ghz_input_estimate.gates.total)

assert ghz_input_estimate.qubits - 4 == 0
assert ghz_input_estimate.gates.total - 4 == 0
assert ghz_input_estimate.measurements.total - 4 == 0
assert ghz_input_estimate.depth.depth - 5 == 0
assert ghz_input_estimate.depth.gate_depth - 4 == 0

# %% [markdown]
# 次に、すでに得た`ghz_symbolic_estimate`へ`.substitute(n=4)`を適用します。この方法では、量子カーネルを再解析せず、完成している推定式の`n`を4へ置き換えます。

# %%
ghz_substituted_estimate = ghz_symbolic_estimate.substitute(n=4)

print("substitute -> qubits:", ghz_substituted_estimate.qubits)
print("substitute -> total gates:", ghz_substituted_estimate.gates.total)

assert ghz_substituted_estimate.qubits - ghz_input_estimate.qubits == 0
assert ghz_substituted_estimate.gates.total - ghz_input_estimate.gates.total == 0
assert (
    ghz_substituted_estimate.measurements.total - ghz_input_estimate.measurements.total
    == 0
)
assert ghz_substituted_estimate.depth.depth - ghz_input_estimate.depth.depth == 0
assert (
    ghz_substituted_estimate.depth.gate_depth - ghz_input_estimate.depth.gate_depth == 0
)

# %% [markdown]
# このGHZ回路では、どちらの方法でも量子ビット数、gate数、measurement数、depthは同じです。ただし、値を適用するタイミングが異なるため、どの回路でも同じ結果になるとは限りません。

# %% [markdown]
# `.substitute()`は、すでに得た推定式を置換するだけで、量子ビット間の依存関係やdepthのscheduleを再計算しません。二つの方法で違いが現れる例として、`qubits[0]`へHadamard gateを適用した後、symbolicなindexを使って`qubits[i]`へX gateを適用する場合を考えます。


# %%
@qmc.qkernel
def symbolic_access(i: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Access qubits[0] and qubits[i] in a symbolic way."""
    qubits = qmc.qubit_array(i + 1, "qubits")
    qubits[0] = qmc.h(qubits[0])
    qubits[i] = qmc.x(qubits[i])
    return qubits


# %% [markdown]
# `i`を指定せずに推定すると、二つのgateが同じ量子ビットへ作用する可能性があるため、安全側に直列化してdepth 2と推定します。

# %%
access_symbolic_estimate = symbolic_access.estimate_resources()

print("symbolic -> depth:", access_symbolic_estimate.depth.depth)
assert access_symbolic_estimate.depth.depth - 2 == 0

# %% [markdown]
# `inputs={"i": 0}`を推定時に渡すと、二つのgateが同じ量子ビットへ作用すると分かるため、depthは2です。

# %%
access_i0_estimate = symbolic_access.estimate_resources(inputs={"i": 0})

print("inputs, i=0 -> depth:", access_i0_estimate.depth.depth)
assert access_i0_estimate.depth.depth - 2 == 0

# %% [markdown]
# `inputs={"i": 1}`を推定時に渡すと、二つのgateが異なる量子ビットへ作用すると分かり、並列に配置できるためdepthは1です。

# %%
access_i1_estimate = symbolic_access.estimate_resources(inputs={"i": 1})

print("inputs, i=1 -> depth:", access_i1_estimate.depth.depth)
assert access_i1_estimate.depth.depth - 1 == 0

# %% [markdown]
# 一方、symbolicな推定結果へ`.substitute(i=1)`を適用しても、すでに決まったscheduleは再計算されません。そのため、式の値を置き換えた後もdepthは2のままです。

# %%
access_i1_substituted_estimate = access_symbolic_estimate.substitute(i=1)

print("substitute, i=1 -> depth:", access_i1_substituted_estimate.depth.depth)
assert access_i1_substituted_estimate.depth.depth - 2 == 0

# %% [markdown]
# ## 2. よく使う推定結果を読む

# %% [markdown]
# `estimate_resources()`が返す`ResourceEstimate`では、量子ビット幅、gate、measurement/reset、depthが項目別に整理されています。ここでは、普段よく確認する項目に絞って読み方を説明します。全フィールドはAppendixにまとめます。

# %% [markdown]
# ### 2.1 量子ビット幅

# %% [markdown]
# 量子ビット幅には、次の項目があります。
#
# - `width.input_qubits`：呼び出し側から量子カーネルへ渡される量子ビット数
# - `width.allocated_qubits`：量子カーネルの本体で確保する量子ビット数
# - `width.clean_ancilla_qubits`：初期状態が`|0>`である必要がある補助量子ビット数
# - `width.dirty_ancilla_qubits`：任意の初期状態を借り、処理後に元の状態へ戻す補助量子ビット数
# - `width.peak_qubits`：実行中に同時に必要になる論理量子ビットの最大数
# - `width.circuit_qubits`：入力、すべての確保場所、clean/dirty ancillaを合わせた静的な回路幅
#
# `ResourceEstimate.qubits`は`width.peak_qubits`のalias、`ResourceEstimate.circuit_qubits`は`width.circuit_qubits`のaliasです。第1章の4個の量子ビットからなるGHZ回路を使って、これらの値を確認します。


# %%
ghz_estimate = ghz_state.estimate_resources(inputs={"n": 4})

print("peak qubits:", ghz_estimate.qubits)
print("input qubits:", ghz_estimate.width.input_qubits)
print("allocated qubits:", ghz_estimate.width.allocated_qubits)
print("clean ancilla qubits:", ghz_estimate.width.clean_ancilla_qubits)
print("dirty ancilla qubits:", ghz_estimate.width.dirty_ancilla_qubits)
print("circuit qubits:", ghz_estimate.circuit_qubits)

assert ghz_estimate.qubits - 4 == 0
assert ghz_estimate.width.input_qubits - 0 == 0
assert ghz_estimate.width.allocated_qubits - 4 == 0
assert ghz_estimate.width.clean_ancilla_qubits - 0 == 0
assert ghz_estimate.width.dirty_ancilla_qubits - 0 == 0
assert ghz_estimate.circuit_qubits - 4 == 0

# %% [markdown]
# `ghz_state`は量子ビットを引数として受け取らず、本体で4個確保します。この例では4個すべてを同時に使い、追加のancillaも必要ないため、peak幅と静的な回路幅はどちらも4です。

# %% [markdown]
# `qubits`と`circuit_qubits`は異なる場合があります。次の量子カーネルでは、最初の量子ビットを測定してその生存期間を終えた後に、別の量子ビットを確保します。


# %%
@qmc.qkernel
def released_qubit_example() -> qmc.Qubit:
    """Measure one qubit before allocating its replacement."""
    first = qmc.qubit("first")
    _measured = qmc.measure(first)
    return qmc.qubit("second")


released_qubit_estimate = released_qubit_example.estimate_resources()

print("peak qubits:", released_qubit_estimate.qubits)
print("allocated qubits:", released_qubit_estimate.width.allocated_qubits)
print("circuit qubits:", released_qubit_estimate.circuit_qubits)

assert released_qubit_estimate.qubits - 1 == 0
assert released_qubit_estimate.width.allocated_qubits - 2 == 0
assert released_qubit_estimate.circuit_qubits - 2 == 0

# %% [markdown]
# 二つの量子ビットを同時には使わないため、peak幅は1です。一方、静的な回路では二つの確保場所を保持するため、`circuit_qubits`は2です。

# %% [markdown]
# ### 2.2 gate・measurement・reset

# %% [markdown]
# gateは、総数に加えて、作用する量子ビット数ごとの内訳を確認できます。`multi_qubit`は、3個以上の量子ビットへ作用するlogical gateを数えます。
#
# 本体を解析できる通常の量子カーネルでは、作用する量子ビット数ごとの内訳を足すと`total`と一致します。第3章でOracleへ指定するcostでは、内訳を部分的にしか記述しない場合があるため、常に一致するとは限りません。


# %%
print("total gates:", ghz_estimate.gates.total)
print("single qubit gates:", ghz_estimate.gates.single_qubit)
print("two qubit gates:", ghz_estimate.gates.two_qubit)
print("multi qubit gates:", ghz_estimate.gates.multi_qubit)

assert ghz_estimate.gates.total - 4 == 0
assert ghz_estimate.gates.single_qubit - 1 == 0
assert ghz_estimate.gates.two_qubit - 3 == 0
assert ghz_estimate.gates.multi_qubit - 0 == 0
assert (
    ghz_estimate.gates.total
    - (
        ghz_estimate.gates.single_qubit
        + ghz_estimate.gates.two_qubit
        + ghz_estimate.gates.multi_qubit
    )
    == 0
)

# %% [markdown]
# 4個の量子ビットからなるGHZ回路には、1個のHadamard gateと3個のCX gateがあります。そのため、`single_qubit`は1、`two_qubit`は3、総gate数の`total`は4です。

# %% [markdown]
# measurementとresetはgateに含めず、独立したeventとして数えます。


# %%
print("measurements:", ghz_estimate.measurements.total)
print("resets:", ghz_estimate.resets.total)

assert ghz_estimate.measurements.total - 4 == 0
assert ghz_estimate.resets.total - 0 == 0

# %% [markdown]
# 4個の量子ビットを測定するため、`measurements.total`は4です。この量子カーネルにはresetを書いていないため、`resets.total`は0です。これらはlogical circuitを1回実行する場合のevent数であり、shots数は掛けません。

# %% [markdown]
# ### 2.3 depth

# %% [markdown]
# gate数がlogical gateの実行回数を表すのに対し、depthは依存関係を守りながら何層に配置できるかを表します。
#
# `depth.depth`は回路全体のcritical path、`gate_depth`、`measurement_depth`、`reset_depth`はそれぞれの種類だけを独立にscheduleした値です。一般には、種類ごとのdepthを足して`depth.depth`を復元することはできません。


# %%
print("total depth:", ghz_estimate.depth.depth)
print("gate depth:", ghz_estimate.depth.gate_depth)
print("measurement depth:", ghz_estimate.depth.measurement_depth)
print("reset depth:", ghz_estimate.depth.reset_depth)

assert ghz_estimate.depth.depth - 5 == 0
assert ghz_estimate.depth.gate_depth - 4 == 0
assert ghz_estimate.depth.measurement_depth - 1 == 0
assert ghz_estimate.depth.reset_depth - 0 == 0

# %% [markdown]
# GHZ回路では、同じ量子ビットの状態を順に伝えるため、Hadamard gateと3個のCX gateは4層になります。その後にmeasurement layerが1つ必要なので、全体のdepthは5です。

# %% [markdown]
# 一方、異なる量子ビットに作用して依存関係を持たないgateは、同じlayerへ配置できます。次の量子カーネルではHadamard gateを2個使いますが、並列に実行できるためgate depthは1です。


# %%
@qmc.qkernel
def parallel_hadamards() -> qmc.Vector[qmc.Bit]:
    """Apply two independent Hadamard gates and measure both qubits."""
    qubits = qmc.qubit_array(2, "qubits")
    qubits[0] = qmc.h(qubits[0])
    qubits[1] = qmc.h(qubits[1])
    return qmc.measure(qubits)


parallel_estimate = parallel_hadamards.estimate_resources()

print("total gates:", parallel_estimate.gates.total)
print("gate depth:", parallel_estimate.depth.gate_depth)
print("measurements:", parallel_estimate.measurements.total)
print("measurement depth:", parallel_estimate.depth.measurement_depth)
print("total depth:", parallel_estimate.depth.depth)

assert parallel_estimate.gates.total - 2 == 0
assert parallel_estimate.depth.gate_depth - 1 == 0
assert parallel_estimate.measurements.total - 2 == 0
assert parallel_estimate.depth.measurement_depth - 1 == 0
assert parallel_estimate.depth.depth - 2 == 0

# %% [markdown]
# 2個のHadamard gateは1つのgate layerに、2回のmeasurementも1つのmeasurement layerに並べられます。そのため、gate数とmeasurement数はそれぞれ2ですが、全体のdepthは2です。

# %% [markdown]
# ## 3. Oracleにcostを設定する
#
# 通常の量子カーネルでは、Qamomileが本体のoperationを再帰的にたどってリソースを推定します。一方、トップダウンで設計するときは、入出力だけを決めて、問題固有の処理をまだ実装しない場合があります。
#
# このような処理は、Qamomile IR上で本体を持たない`qmc.Oracle`として表せます。推定に利用できる内部のgate列がないため、既知のcostを固定値またはcallbackで設定します。

# %% [markdown]
# ### 3.1 固定costを与える
#
# 1回の呼び出しに必要なcostが決まっている場合は、`qmc.Oracle`の`cost`へ固定の`ResourceEstimate`を渡します。次のOracleは2個の量子ビットへ作用し、2個のsingle-qubit gateと3個のtwo-qubit gateを使うものとします。内部には一部のgateを並列に実行できる箇所があり、gate depthは4であることも分かっているとします。
#
# 固定costは、Oracleを通常どおり1回適用する場合のbase costです。`cost`で省略したリソース項目は0として扱われます。gate数から内部の依存関係や並列性は復元できないため、depthも必要な場合は`DepthResources`として明示します。


# %%
fixed_lookup = qmc.Oracle(
    "fixed_lookup",
    num_qubits=2,
    cost=qmc.ResourceEstimate(
        gates=qmc.GateResources(
            total=5,
            single_qubit=2,
            two_qubit=3,
        ),
        depth=qmc.DepthResources(
            depth=4,
            gate_depth=4,
        ),
    ),
)

# %% [markdown]
# Oracleは、量子カーネル内で通常の量子operationと同じように呼び出せます。


# %%
@qmc.qkernel
def fixed_oracle_circuit() -> qmc.Vector[qmc.Qubit]:
    """Apply one fixed-cost Oracle."""
    register = qmc.qubit_array(2, "register")
    return fixed_lookup(register)


# %% [markdown]
# この量子カーネルを推定すると、Oracleへ設定した固定costに、呼び出し側で確保した2個の量子ビットが組み合わされます。

# %%
fixed_oracle_estimate = fixed_oracle_circuit.estimate_resources()

print("qubits:", fixed_oracle_estimate.qubits)
print("total gates:", fixed_oracle_estimate.gates.total)
print("single qubit gates:", fixed_oracle_estimate.gates.single_qubit)
print("two qubit gates:", fixed_oracle_estimate.gates.two_qubit)
print("gate depth:", fixed_oracle_estimate.depth.gate_depth)

assert fixed_oracle_estimate.qubits - 2 == 0
assert fixed_oracle_estimate.gates.total - 5 == 0
assert fixed_oracle_estimate.gates.single_qubit - 2 == 0
assert fixed_oracle_estimate.gates.two_qubit - 3 == 0
assert fixed_oracle_estimate.depth.depth - 4 == 0
assert fixed_oracle_estimate.depth.gate_depth - 4 == 0

# %% [markdown]
# 推定結果には、Oracleへ指定した5個のgateとdepth 4がそのまま反映されています。2個のtargetは呼び出し側の量子カーネルが確保するため、peak幅は2です。

# %% [markdown]
# ### 3.2 callbackでcostを計算する
#
# 受け取る量子ビット数が可変長の場合は、Oracleの`signature`に`CallableSignature`を指定します。また、受け取る量子ビット数などによってcostが変わる場合は、`OpaqueCostContext`を受け取るPython callbackを使ってcostを定義します。次の`CallableSignature`は、可変長量子ビットvectorを1つ受け取り、vectorを1つ返す型を表します。このOracleのvector呼び出しでは、入力と同じshapeが出力にも引き継がれます。
#
# callbackはリソース推定時に呼び出されます。`context.target_qubits`からtargetとなる量子ビット数を参照でき、`context.control_decomposition`には現在のリソース推定で選択されたcontrol分解方法が入ります。callbackが返す`ResourceEstimate`は、固定costと同様に、Oracleそのものを通常どおり1回適用する場合のbase costです。外側のcontrolによるcostを先回りして含める必要はありません。


# %%
def serial_sweep_cost(
    context: qmc.OpaqueCostContext,
) -> qmc.ResourceEstimate:
    """Return the cost of one serial sweep over all target qubits."""
    target_qubits = context.target_qubits
    return qmc.ResourceEstimate(
        gates=qmc.GateResources(
            total=target_qubits,
            single_qubit=target_qubits,
        ),
        depth=qmc.DepthResources(
            depth=target_qubits,
            gate_depth=target_qubits,
        ),
        control_decomposition=context.control_decomposition,
    )


serial_sweep = qmc.Oracle(
    "serial_sweep",
    signature=qmc.CallableSignature(
        inputs=[qmc.Vector[qmc.Qubit]],
        outputs=[qmc.Vector[qmc.Qubit]],
    ),
    cost=serial_sweep_cost,
)

# %% [markdown]
# このcallbackは、targetとなる量子ビットごとに1個のsingle-qubit gateを直列に適用するcostを返します。返り値の`control_decomposition`には、contextから受け取った設定を引き継いでいます。次に、このOracleを可変長registerへ1回適用する量子カーネルを定義します。


# %%
@qmc.qkernel
def callback_oracle_circuit(width: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Apply an Oracle whose cost follows the register width."""
    register = qmc.qubit_array(width, "register")
    return serial_sweep(register)


# %% [markdown]
# `width`を指定せずに推定すると、callbackへsymbolicなtarget幅が渡され、gate数とdepthもsymbolicな式として返ります。

# %%
callback_oracle_estimate = callback_oracle_circuit.estimate_resources()

print("parameters:", callback_oracle_estimate.parameters)
print("total gates:", callback_oracle_estimate.gates.total)
print("gate depth:", callback_oracle_estimate.depth.gate_depth)

width = callback_oracle_estimate.parameters["width"]
assert sp.simplify(callback_oracle_estimate.qubits - width) == 0
assert sp.simplify(callback_oracle_estimate.gates.total - width) == 0
assert sp.simplify(callback_oracle_estimate.gates.single_qubit - width) == 0
assert sp.simplify(callback_oracle_estimate.depth.depth - width) == 0
assert sp.simplify(callback_oracle_estimate.depth.gate_depth - width) == 0

# %% [markdown]
# gate数とdepthには、symbolicなtarget幅がそのまま現れます。現在、Oracleへcontrolを追加できるのは固定幅のscalar signatureだけなので、この例の可変長vector Oracleは直接呼び出して使います。

# %% [markdown]
# ## 4. 推定方法を設定する
#
# `estimate_resources()`には、推定対象の構造をどのようなmodelで数えるかを選ぶ設定があります。ここでは、controlの分解方法と、costが分からない処理の扱いを説明します。どちらも、engineのnative gateやhardwareを選ぶ設定ではありません。

# %% [markdown]
# ### 4.1 control decomposition
#
# `control_decomposition`は、coherent controlをどのように数えるかを選びます。既定値の`CLEAN_ANCILLA_TOFFOLI`は、Toffoli gateと初期状態が`|0>`のclean ancillaを使う固定の分解modelです。`ABSTRACT`は、制御された各primitiveを、分解せずに1個の多制御operationとして数えます。
#
# 次の例では、3個の制御量子ビットを使って1個のHadamard gateを制御します。


# %%
@qmc.qkernel
def three_controlled_hadamard() -> qmc.Qubit:
    """Apply one Hadamard gate under three coherent controls."""
    controls = qmc.qubit_array(3, "controls")
    target = qmc.qubit("target")
    controls, target = qmc.control(qmc.h, num_controls=3)(controls, target)
    return target


# %% [markdown]
# 最初に、既定値の`CLEAN_ANCILLA_TOFFOLI`で推定します。このmodelでは、3個の制御条件の論理積を2個のclean ancillaへ計算し、controlled-Hadamardの適用後に元へ戻します。

# %%
decomposed_control_estimate = three_controlled_hadamard.estimate_resources()

print("decomposed total gates:", decomposed_control_estimate.gates.total)
print("decomposed Toffoli gates:", decomposed_control_estimate.gates.toffoli)
print(
    "decomposed clean ancillas:",
    decomposed_control_estimate.width.clean_ancilla_qubits,
)

assert decomposed_control_estimate.gates.total - 5 == 0
assert decomposed_control_estimate.gates.toffoli - 4 == 0
assert decomposed_control_estimate.width.clean_ancilla_qubits - 2 == 0
assert decomposed_control_estimate.qubits - 6 == 0

# %% [markdown]
# 4個のToffoli gateで論理積の計算とuncomputationを行うため、総gate数はcontrolled-Hadamardと合わせて5です。

# %% [markdown]
# 次に、`ABSTRACT`を指定します。このmodelでは、controlled-Hadamardを分解せずに1個の多制御operationとして数え、分解用ancillaを追加しません。

# %%
abstract_control_estimate = three_controlled_hadamard.estimate_resources(
    control_decomposition=qmc.ControlDecomposition.ABSTRACT,
)

print("abstract total gates:", abstract_control_estimate.gates.total)
print("abstract clean ancillas:", abstract_control_estimate.width.clean_ancilla_qubits)

assert abstract_control_estimate.gates.total - 1 == 0
assert abstract_control_estimate.width.clean_ancilla_qubits - 0 == 0
assert abstract_control_estimate.qubits - 4 == 0

# %% [markdown]
# `CLEAN_ANCILLA_TOFFOLI`と`ABSTRACT`は、同じcoherent controlを異なる粒度で数えます。以下では、Toffoli gateとclean ancillaを使った分解を手書きし、総gate数、Toffoli数、depth、peak幅が`CLEAN_ANCILLA_TOFFOLI`の推定結果と一致することを確認します。


# %%
@qmc.qkernel
def three_controlled_hadamard_manually() -> qmc.Qubit:
    """Apply a manually decomposed three-controlled Hadamard gate."""
    controls = qmc.qubit_array(3, "controls")
    target = qmc.qubit("target")
    ancillae = qmc.qubit_array(2, "ancillae")

    controls[0], controls[1], ancillae[0] = qmc.ccx(
        controls[0],
        controls[1],
        ancillae[0],
    )
    controls[2], ancillae[0], ancillae[1] = qmc.ccx(
        controls[2],
        ancillae[0],
        ancillae[1],
    )
    ancillae[1], target = qmc.control(qmc.h)(ancillae[1], target)
    controls[2], ancillae[0], ancillae[1] = qmc.ccx(
        controls[2],
        ancillae[0],
        ancillae[1],
    )
    controls[0], controls[1], ancillae[0] = qmc.ccx(
        controls[0],
        controls[1],
        ancillae[0],
    )
    return target


# %% [markdown]
# 手書きした回路を通常どおり推定し、`CLEAN_ANCILLA_TOFFOLI`の結果と比較します。

# %%
manual_control_estimate = three_controlled_hadamard_manually.estimate_resources()

print("manual total gates:", manual_control_estimate.gates.total)
print("manual Toffoli gates:", manual_control_estimate.gates.toffoli)
print("manual allocated qubits:", manual_control_estimate.width.allocated_qubits)
print(
    "manual clean_ancilla_qubits:",
    manual_control_estimate.width.clean_ancilla_qubits,
)

assert (
    manual_control_estimate.gates.total - decomposed_control_estimate.gates.total == 0
)
assert (
    manual_control_estimate.gates.toffoli - decomposed_control_estimate.gates.toffoli
    == 0
)
assert manual_control_estimate.qubits - decomposed_control_estimate.qubits == 0
assert (
    manual_control_estimate.depth.depth - decomposed_control_estimate.depth.depth == 0
)
assert manual_control_estimate.width.allocated_qubits - 6 == 0
assert manual_control_estimate.width.clean_ancilla_qubits - 0 == 0

# %% [markdown]
# `CLEAN_ANCILLA_TOFFOLI`の推定結果と手書きの回路は、どちらも4個のToffoli gateと1個のcontrolled-Hadamardを使い、peak幅は6です。手書きの回路では、2個のancillaを量子カーネル内で明示的に確保したため、`allocated_qubits`に含まれます。一方、`CLEAN_ANCILLA_TOFFOLI`では、元の回路が確保する4個とは別に、推定modelが追加する2個を`clean_ancilla_qubits`として報告します。
#
# `CLEAN_ANCILLA_TOFFOLI`は、このような固定のalgorithmicな分解を数える設定です。特定のengineが実際に選ぶnative gate列やroutingを表すものではありません。

# %% [markdown]
# ### 4.2 unknown resource policy
#
# costを指定していないOracleを含む量子カーネルを推定する場合は、`unknown_policy`で未知部分の扱いを選びます。既定値は`ERROR`で、未知のcostを暗黙に0とせず、`ValueError`を送出します。


# %%
unpriced_step = qmc.Oracle(
    "unpriced_step",
    num_qubits=1,
)


@qmc.qkernel
def unpriced_oracle_circuit() -> qmc.Qubit:
    """Invoke an Oracle without a resource cost."""
    target = qmc.qubit("target")
    (target,) = unpriced_step(target)
    return target


# %% [markdown]
# まず、`unknown_policy`を指定せず、既定値の`ERROR`で推定します。costのないOracleを暗黙に無料とは扱わず、`ValueError`を送出します。

# %%
try:
    unpriced_oracle_circuit.estimate_resources()
except ValueError as error:
    print(error)
    assert "no body or opaque cost" in str(error)
else:
    raise AssertionError("An unpriced Oracle must fail by default.")

# %% [markdown]
# 未知部分があってもエラーにせずリソース推定を続けたい場合は、`OPAQUE_CALL`または`ZERO_WITH_WARNING`を明示的に選べます。
#
# `OPAQUE_CALL`は、未知部分のgate数や幅を推測せず、名前付きcallとqueryを1回ずつ記録します。


# %%
opaque_call_estimate = unpriced_oracle_circuit.estimate_resources(
    unknown_policy=qmc.UnknownResourcePolicy.OPAQUE_CALL,
)

print("opaque call gates:", opaque_call_estimate.gates.total)
print("opaque calls:", opaque_call_estimate.calls.calls_by_name)
print("opaque queries:", opaque_call_estimate.calls.queries_by_name)

assert opaque_call_estimate.gates.total - 0 == 0
assert opaque_call_estimate.calls.calls_by_name == {"unpriced_step": 1}
assert opaque_call_estimate.calls.queries_by_name == {"unpriced_step": 1}
assert opaque_call_estimate.quality is qmc.EstimateQuality.UNKNOWN

# %% [markdown]
# `ZERO_WITH_WARNING`は、未知部分を0と仮定して推定を続け、その仮定を結果の`assumptions`へ記録します。Pythonのwarningを送出する設定ではないため、ここでは記録された注意書きを表示します。


# %%
zero_warning_estimate = unpriced_oracle_circuit.estimate_resources(
    unknown_policy=qmc.UnknownResourcePolicy.ZERO_WITH_WARNING,
)

print("zero-with-warning gates:", zero_warning_estimate.gates.total)
print("zero-with-warning assumptions:")
for assumption in zero_warning_estimate.assumptions:
    print(f"- {assumption.message} (source: {assumption.source})")

assert zero_warning_estimate.gates.total - 0 == 0
assert zero_warning_estimate.calls.calls_by_name == {}
assert zero_warning_estimate.calls.queries_by_name == {}
assert any(
    assumption.message == "unknown callable counted as zero resources"
    and assumption.source == "unpriced_step"
    for assumption in zero_warning_estimate.assumptions
)
assert zero_warning_estimate.quality is qmc.EstimateQuality.UNKNOWN

# %% [markdown]
# | policy | 未知部分の扱い |
# |---|---|
# | `ERROR` | 既定値。costのないOracleに到達すると`ValueError`を送出する |
# | `OPAQUE_CALL` | gate costを推測せず、名前付きcallとqueryを1回ずつ記録する |
# | `ZERO_WITH_WARNING` | 未知部分を0と仮定し、その仮定を`assumptions`へ記録する |
#
# `OPAQUE_CALL`と`ZERO_WITH_WARNING`でgate数が0でも、未知部分にgateがないという意味ではありません。どちらも`quality`は`UNKNOWN`となり、推定できなかった部分があることを示します。`ZERO_WITH_WARNING`はPythonのwarningを送出する設定ではなく、注意書きを推定結果の`assumptions`に記録する設定です。
#
# `unknown_policy`が影響するのはcostを指定していないOracleです。第3章のように固定costまたはcallbackを指定したOracleでは、そのcostが使われます。

# %% [markdown]
# ## 5. 推定結果の確かさを確認する
#
# リソースの数値だけでなく、推定結果の`derivation`、`quality`、`approximation`も確認すると、その数値をどのように解釈すべきかが分かります。この三つは、それぞれ異なる問いに答える独立した項目です。

# %% [markdown]
# ### 5.1 三つの独立した軸
#
# | 項目 | 確認できること | 値 |
# |---|---|---|
# | `derivation` | 推定値をどのように求めたか | `STRUCTURAL` / `MODELED` |
# | `quality` | 選択した回路modelに対して、推定値がどのような関係にあるか | `EXACT` / `CONSERVATIVE` / `UNKNOWN` |
# | `approximation` | 推定器が認識している数学的な近似が含まれるか | `EXACT` / `APPROXIMATE` |
#
# `derivation=STRUCTURAL`は、見えているIRと選択した分解規則を再帰的に数えたことを表します。`MODELED`は、Oracleへ指定したcostや、costのないOracleに対して選択した`unknown_policy`を使って数値を求めたことを表します。
#
# `quality=EXACT`は、選択した回路modelのcostと一致することを表します。`CONSERVATIVE`は過小評価しない安全側の値、`UNKNOWN`は`EXACT`とも`CONSERVATIVE`とも確認できない値です。
#
# `approximation`は、数値の数え方ではなく、選択した回路が理想的な数学的operationを近似しているかを表します。

# %% [markdown]
# #### Bell回路：`STRUCTURAL / EXACT / EXACT`
#
# Bell回路では、推定器が量子カーネルのIRを直接たどり、選択した回路modelに対するcostを正確に数えています。また、推定器が認識する数学的な近似も含まれません。

# %%
print("derivation:", bell_estimate.derivation.value)
print("quality:", bell_estimate.quality.value)
print("approximation:", bell_estimate.approximation.value)

assert bell_estimate.derivation is qmc.EstimateDerivation.STRUCTURAL
assert bell_estimate.quality is qmc.EstimateQuality.EXACT
assert bell_estimate.approximation is qmc.ApproximationStatus.EXACT

# %% [markdown]
# #### 固定costのOracle：`MODELED / EXACT / EXACT`
#
# `fixed_lookup`の内部構造は見えませんが、利用者が1回分のcostを指定しています。そのため`derivation`は`MODELED`で、指定したbase costに対する`quality`は`EXACT`です。

# %%
print("derivation:", fixed_oracle_estimate.derivation.value)
print("quality:", fixed_oracle_estimate.quality.value)
print("approximation:", fixed_oracle_estimate.approximation.value)

assert fixed_oracle_estimate.derivation is qmc.EstimateDerivation.MODELED
assert fixed_oracle_estimate.quality is qmc.EstimateQuality.EXACT
assert fixed_oracle_estimate.approximation is qmc.ApproximationStatus.EXACT

# %% [markdown]
# #### clean ancillaを使うcontrol分解：`STRUCTURAL / CONSERVATIVE / EXACT`
#
# `CLEAN_ANCILLA_TOFFOLI`では、見えているIRを固定のalgorithmic modelで分解し、安全側のcostを求めます。これは数学的な近似ではないため、`approximation`は`EXACT`です。

# %%
print("derivation:", decomposed_control_estimate.derivation.value)
print("quality:", decomposed_control_estimate.quality.value)
print("approximation:", decomposed_control_estimate.approximation.value)

assert decomposed_control_estimate.derivation is qmc.EstimateDerivation.STRUCTURAL
assert decomposed_control_estimate.quality is qmc.EstimateQuality.CONSERVATIVE
assert decomposed_control_estimate.approximation is qmc.ApproximationStatus.EXACT

# %% [markdown]
# #### costのないOracle：`MODELED / UNKNOWN / EXACT`
#
# `OPAQUE_CALL`は、costのないOracleを名前付きcall/queryとして記録します。具体的なgate costとの関係は分からないため`quality`は`UNKNOWN`ですが、推定器が認識する数学的な近似が追加されるわけではありません。

# %%
print("derivation:", opaque_call_estimate.derivation.value)
print("quality:", opaque_call_estimate.quality.value)
print("approximation:", opaque_call_estimate.approximation.value)

assert opaque_call_estimate.derivation is qmc.EstimateDerivation.MODELED
assert opaque_call_estimate.quality is qmc.EstimateQuality.UNKNOWN
assert opaque_call_estimate.approximation is qmc.ApproximationStatus.EXACT

# %% [markdown]
# #### 非可換なPauli項の時間発展：`STRUCTURAL / EXACT / APPROXIMATE`
#
# 最後に、1個の量子ビットへ、互いに可換ではないX項とZ項の和による時間発展を適用します。推定器は選択された1 stepのgate列を正確に数えますが、その積公式は理想的なHamiltonianの時間発展に対する近似です。


# %%
@qmc.qkernel
def one_qubit_pauli_evolution(
    hamiltonian: qmc.Observable,
    time: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply one Pauli-sum evolution to a single qubit."""
    target = qmc.qubit_array(1, "target")
    return qmc.pauli_evolve(target, hamiltonian, time)


# %% [markdown]
# Hamiltonianと時間を`inputs`で具体化して推定し、三つのmetadataを確認します。

# %%
noncommuting_evolution_estimate = one_qubit_pauli_evolution.estimate_resources(
    inputs={
        "hamiltonian": qmo.X(0) + qmo.Z(0),
        "time": 0.25,
    },
)

print("derivation:", noncommuting_evolution_estimate.derivation.value)
print("quality:", noncommuting_evolution_estimate.quality.value)
print("approximation:", noncommuting_evolution_estimate.approximation.value)

assert noncommuting_evolution_estimate.derivation is qmc.EstimateDerivation.STRUCTURAL
assert noncommuting_evolution_estimate.quality is qmc.EstimateQuality.EXACT
assert (
    noncommuting_evolution_estimate.approximation is qmc.ApproximationStatus.APPROXIMATE
)

# %% [markdown]
# この例では、選択された1 stepの回路に含まれるgateを正確に数えられるため`quality=EXACT`です。一方、その回路は理想的なHamiltonianの時間発展をfirst-order Lie–Trotter product formulaで近似するため、`approximation=APPROXIMATE`になります。
#
# `approximation=EXACT`は、推定器が認識している数学的な近似がないことを表します。Oracleの内部など、推定器から見えない処理の数学的な正確さまで証明するものではありません。

# %% [markdown]
# ### 5.2 assumptions
#
# `assumptions`には、三つの項目だけでは表せない具体的な前提や理由が入ります。各要素は、説明文の`message`と、原因となったoperationなどを示す`source`を持ちます。

# %% [markdown]
# `ZERO_WITH_WARNING`を使った推定結果には、costのないOracleを0として数えたことが記録されています。


# %%
print("unknown-cost assumptions:")
for assumption in zero_warning_estimate.assumptions:
    print(f"- {assumption.message} (source: {assumption.source})")

assert any(
    assumption.message == "unknown callable counted as zero resources"
    for assumption in zero_warning_estimate.assumptions
)

# %% [markdown]
# 非可換なPauli項の時間発展には、どの積公式として数えたかが記録されています。

# %%
print("Pauli-evolution assumptions:")
for assumption in noncommuting_evolution_estimate.assumptions:
    print(f"- {assumption.message} (source: {assumption.source})")

assert any(
    "first-order Lie-Trotter" in assumption.message
    for assumption in noncommuting_evolution_estimate.assumptions
)

# %% [markdown]
# `quality=UNKNOWN`となった理由や、`approximation=APPROXIMATE`となった積公式が、具体的な文章として残ります。推定結果を利用するときは、数値と三つのmetadataに加え、`assumptions`も確認してください。

# %% [markdown]
# ## Appendix: `ResourceEstimate`の全field一覧
#
# リソース項目には、具体的な整数だけでなく、問題サイズに依存するSymPyの式が入る場合があります。通常は第2章で扱った代表的な項目を確認すれば十分ですが、ここでは公開されている全フィールドを参照用にまとめます。

# %% [markdown]
# ### `ResourceEstimate`
#
# | field | 内容 |
# |---|---|
# | `width` | 論理量子ビット幅とancillaの推定値 |
# | `gates` | logical gate数の推定値 |
# | `measurements` | measurement event数の推定値 |
# | `resets` | reset event数の推定値 |
# | `depth` | logical depthの推定値 |
# | `calls` | 名前別に記録したopaque call/queryの回数 |
# | `parameters` | 未具体化のsymbolic parameter名とSymPy symbolの対応 |
# | `assumptions` | 推定時に置いた具体的な仮定。各要素は`message`と`source`を持つ |
# | `derivation` | IRから数えた`STRUCTURAL`か、cost/policyを使った`MODELED`か |
# | `quality` | 選択した回路modelに対して`EXACT`、`CONSERVATIVE`、`UNKNOWN`のどれか |
# | `approximation` | 認識されている数学的な近似が`EXACT`か`APPROXIMATE`か |
# | `control_decomposition` | 推定に使用したcoherent controlの分解方法 |
# | `trace` | `trace=True`を指定した場合だけ保持する診断用tree。通常は`None` |

# %% [markdown]
# ### Width
#
# `estimate.width`は`WidthResources`です。
#
# | field | 内容 |
# |---|---|
# | `input_qubits` | 呼び出し側から渡される量子ビット数 |
# | `allocated_qubits` | 本体にある静的な確保場所の量子ビット数 |
# | `clean_ancilla_qubits` | 初期状態`|0>`を必要とし、使用後に`|0>`へ戻すancillaのpeak需要 |
# | `dirty_ancilla_qubits` | 任意の初期状態を借り、使用後に元へ戻すancillaのpeak需要 |
# | `peak_qubits` | 実行中に同時にliveとなる論理量子ビットの最大数 |
#
# `allocated_qubits`、`clean_ancilla_qubits`、`dirty_ancilla_qubits`は別々の分類です。clean/dirty ancillaが`allocated_qubits`の内訳という意味ではありません。

# %% [markdown]
# ### Gates
#
# `estimate.gates`は`GateResources`です。
#
# | field | 内容 |
# |---|---|
# | `total` | measurement/resetを含まないlogical gateの総数 |
# | `single_qubit` | 1個の量子ビットへ作用するgate数 |
# | `two_qubit` | 2個の量子ビットへ作用するgate数 |
# | `multi_qubit` | 3個以上の量子ビットへ作用するgate数 |
# | `clifford` | Clifford gate数 |
# | `rotation` | RX、RY、RZ、P、CP、RZZなどのrotation gate数 |
# | `t` | T/T-dagger gate数 |
# | `toffoli` | Toffoli gate数 |
# | `non_clifford` | non-Clifford gate数 |
#
# `single_qubit`、`two_qubit`、`multi_qubit`は作用する量子ビット数による分類です。`clifford`以降はgate familyによる分類であり、二つの分類は重なります。すべてのfieldを足し合わせるものではありません。また、Oracleへ部分的なcostだけを指定した場合は、arity fieldの和が`total`と一致しないことがあります。

# %% [markdown]
# ### Measurements / resets
#
# | field | 内容 |
# |---|---|
# | `estimate.measurements.total` | 量子ビットごとのmeasurement event数。`N`個の量子ビットを測定すると`N`増える |
# | `estimate.resets.total` | 量子ビットごとの明示的なreset event数 |
#
# measurementとresetは`gates.total`には含まれません。

# %% [markdown]
# ### Depth
#
# `estimate.depth`は`DepthResources`です。
#
# | field | 内容 |
# |---|---|
# | `depth` | gate、measurement、resetの依存関係を含む全体のcritical-path depth |
# | `gate_depth` | gateだけをscheduleしたdepth |
# | `measurement_depth` | measurementだけをscheduleしたdepth |
# | `reset_depth` | resetだけをscheduleしたdepth |
# | `clifford_depth` | Clifford gateだけをscheduleしたdepth |
# | `rotation_depth` | rotation gateだけをscheduleしたdepth |
# | `t_depth` | T/T-dagger gateだけをscheduleしたdepth |
# | `toffoli_depth` | Toffoli gateだけをscheduleしたdepth |
# | `non_clifford_depth` | non-Clifford gateだけをscheduleしたdepth |
#
# 種類ごとのdepthは、それぞれのoperationだけを独立にscheduleした値です。これらを足しても`depth.depth`にはなりません。

# %% [markdown]
# ### Calls / queries
#
# `estimate.calls`は`CallResources`です。
#
# | field | 内容 |
# |---|---|
# | `calls_by_name` | 本体を展開せずに残した、またはcostで明示した名前付きcallの回数 |
# | `queries_by_name` | 名前ごとに明示またはmodel化したalgorithmicなOracle query回数 |
#
# 本体を解析できる通常の量子カーネルcallは再帰的に展開されるため、`calls_by_name`には残りません。`OPAQUE_CALL`では情報がない場合の規則として両方へ1を記録しますが、callとqueryは異なる意味を持つfieldです。

# %% [markdown]
# ### Convenience aliases
#
# 次のpropertyは、よく使うfieldへ短くアクセスするためのaliasまたは計算値です。
#
# | property | 対応する値 |
# |---|---|
# | `estimate.qubits` | `estimate.width.peak_qubits` |
# | `estimate.width.circuit_qubits` | `input_qubits + allocated_qubits + clean_ancilla_qubits + dirty_ancilla_qubits` |
# | `estimate.circuit_qubits` | `estimate.width.circuit_qubits` |
# | `estimate.gates.t_gates` | `estimate.gates.t` |
# | `estimate.gates.clifford_gates` | `estimate.gates.clifford` |
# | `estimate.gates.rotation_gates` | `estimate.gates.rotation` |
# | `estimate.calls.oracle_calls` | `estimate.calls.calls_by_name` |
# | `estimate.calls.oracle_queries` | `estimate.calls.queries_by_name` |

# %% [markdown]
# ## まとめ
#
# - `estimate_resources()`を使うと、特定のengine向けにtranspileする前のlogical resourceを推定できます。
# - パラメータ付きの量子カーネルはsymbolicに推定でき、推定時に構造を具体化する場合は`inputs`、得られた式を評価する場合は`.substitute()`を使います。
# - 普段は量子ビット幅、gate数、measurement/reset数、全体のdepthを確認します。
# - 本体を持たないOracleには、固定costまたはcallbackで1回分のbase costを設定できます。
# - controlの分解方法や未知costのpolicyを選んだ場合は、数値だけでなく`derivation`、`quality`、`approximation`、`assumptions`も確認します。
