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
# tags: [tutorial]
# ---
#
# # 量子テレポーテーション
#
# このノートブックでは、量子テレポーテーションを題材に、Qamomileの基本的な言語機能を学びます。
#
# - `@qm.qkernel`で量子プログラムを定義する
# - qkernelから別のqkernelを呼び出して量子操作を再利用する
# - `Q[Bit]`型の1量子ビット状態を測定し、各ショットで古典的な`Bit`を得る
# - Pythonの`if`文で測定結果を使い、動的量子回路を書く
# - Qiskit向けにコンパイルし、`qm.run()`で回路を実行する
#
# 量子テレポーテーションは、1量子ビットの状態を転送するプロトコルです。AliceとBobがベル対$\frac{|00\rangle+|11\rangle}{\sqrt2}$を共有し、Aliceが未知の状態$|\psi\rangle=\alpha|0\rangle+\beta|1\rangle$を転送したいとします。
#
# Aliceは転送する状態と、ベル対のうち自分が持つ量子ビットを操作します。次に2つの量子ビットを測定し、得られた古典ビットをBobに送ります。BobはAliceから受け取った結果$(m_0,m_1)$に応じて$Z^{m_0}X^{m_1}$を適用し、$|\psi\rangle$を復元します。
#
# Qamomileを使って、量子テレポーテーションを量子回路で再現しましょう。

# %%
# Qamomileとこのノートブックで使うパッケージをインストールします。
# # !pip install "qamomile[qiskit]"

# %%
import qamomile as qm

# %% [markdown]
# ## 1. Qamomileプログラムの基本構造
#
# Qamomileでは、量子回路を構成する処理と、その回路を量子SDKで実行する処理を分けます。
#
# - `qamomile`は、量子ビット、ゲート、測定などの操作を使って、量子SDKに依存しない量子プログラムを書くためのAPIです。このノートブックでは`qm`という短い名前で参照します。
# - `@qm.qkernel`でデコレートしたPython関数は、量子計算を表す**量子カーネル**（qkernel）になります。Qamomileは関数の本体を解析し、量子SDKに依存しない中間表現に変換します。
# - `qm.compile`は、量子カーネルを選択したtargetで実行できる形式に変換します。
#
# 量子カーネルの引数と戻り値のアノテーションには、`qm.Q[qm.Bit]`や`qm.Shot[qm.Bit]`などのQamomileの型を使います。量子カーネルは普通のPython関数と同じように見えますが、本体は量子回路の構成を記述します。`Q[Bit]`は1量子ビットの状態を表します。これを測定すると`Shot[Bit]`が得られ、各ショットに0か1の古典ビットが対応します。

# %% [markdown]
# ## 2. ベル対を作る
#
# 最初の量子ビットに$H|0\rangle=|+\rangle$を準備し、次にCXゲートを適用します。
#
# $$
# |00\rangle\longmapsto
# \frac{|00\rangle+|11\rangle}{\sqrt2}.
# $$
#
# 片方の量子ビットだけを測定すると、0と1がランダムに現れます。2つを測定すると、結果は常に一致します。このベル対の最初の量子ビットをAliceが、2番目をBobが持つとします。
#
# 次のコードでは、ベル対を作る小さな例を示します。Qamomileの量子値は、回路操作の対象となる量子状態を識別する**ハンドル**です。ゲートは入力のハンドルを消費し、同じ状態に対する更新後のハンドルを返します。消費したハンドルは後続の操作で使えないため、`alice = qm.h(alice)`のように戻り値を変数に代入します。これは、各量子ハンドルを1回ずつ消費するQamomileの**線形所有権**のルールに従っています。
#
# `qm.prepare(0)`で各量子ビットを$|0\rangle$に初期化します。Aliceの量子ビットに`qm.h`を適用した後、`qm.concat(alice, bob)`で2つをまとめて`Q[Bits[2]]`型の状態にします。`qm.cx(pair, control=0, target=1)`では、Aliceの量子ビットを制御、Bobの量子ビットを標的にします。
#
# `bell_pair`は、測定前の2量子ビット状態を返す状態準備用のqkernelです。測定を別のqkernelである`measure_bell_pair`に分けると、後でテレポーテーション回路にベル対の準備を再利用できます。`qm.measure(state)`は量子ビットを計算基底で測定し、ショットごとにビットのタプルを返します。測定した状態には、`H`や`CX`などの量子ゲートを適用できません。`measure_bell_pair`の戻り値のアノテーション`qm.Shot[qm.Bits[2]]`は、各ショットで測定ビットが2つ得られることを示します。


# %%
@qm.qkernel
def bell_pair() -> qm.Q[qm.Bits[2]]:
    """Prepare and return a Bell pair without measuring it."""
    alice = qm.prepare(0)
    bob = qm.prepare(0)
    alice = qm.h(alice)
    pair = qm.concat(alice, bob)
    return qm.cx(pair, control=0, target=1)


@qm.qkernel
def measure_bell_pair() -> qm.Shot[qm.Bits[2]]:
    """Prepare a Bell pair and measure both qubits."""
    return qm.measure(bell_pair())


# %% [markdown]
# ### Qiskitで量子カーネルを実行する
#
# `measure_bell_pair`は量子SDKに依存しない量子プログラムです。定義した時点では実行されません。実行には次の3つの手順があります。
#
# 1. `qm.qiskit.Aer()`でQiskit Aerの実行先を作ります。
# 2. `qm.compile(qm.specialize(measure_bell_pair), to=endpoint.target)`で、その実行先に合わせてqkernelをコンパイルします。
# 3. `qm.run(artifact, on=endpoint, shots=...)`で、指定した回数だけ回路を実行します。
#
# コンパイルと実行が分かれているので、同じ量子カーネルを異なる量子SDK向けにコンパイルできます。一度コンパイルしたパラメータ付き回路を、パラメータの値を変えながら繰り返し実行することもできます。残りの回路にも同じ実行先を使います。
#
# `shots=1024`で回路の実行回数を指定し、`.result`で実行結果を取得します。`bell_result.counts`は、`(bit0, bit1)`の形の測定結果を、その結果が得られた回数に対応付けます。この例では`00`と`11`だけが現れ、回数の合計は`1024`になります。

# %%
shots = 1024
endpoint = qm.qiskit.Aer()

bell_executable = qm.compile(qm.specialize(measure_bell_pair), to=endpoint.target)
bell_result = qm.run(bell_executable, on=endpoint, shots=shots, seed=42).result

print(dict(bell_result.counts))
assert sum(bell_result.counts.values()) == shots
assert all(a == b for a, b in bell_result.counts)


# %% [markdown]
# ## 3. 測定結果を`if`文で使う
#
# ここでは$|+\rangle=H|0\rangle=(|0\rangle+|1\rangle)/\sqrt2$をBobにテレポートすることを考えます。
#
# Aliceは自分が持つ2つの量子ビット`message`と`alice`に`CX(message, alice)`と`H(message)`を適用します。これらの操作はベル基底を計算基底に写します。その後の2つの測定は、ベル測定と等価です。`m_message`と`m_alice`は、測定結果を保持する古典的な`Bit`値です。
#
# これらの`if`文はホスト側で一度だけ評価される通常のPythonの分岐とは異なります。各ショットの測定結果に応じて、targetの量子SDK上で動作する動的制御にコンパイルされます。たとえば`if m_alice:`はPythonで事前に分岐先を決めず、回路の実行中に得た測定結果が1のショットだけで`X`を適用します。このように回路の途中で分岐する回路を**動的量子回路**と呼びます。実機で実行するには、対象の装置が回路途中の測定と古典ビットで制御するゲートをサポートしている必要があります。
#
# `teleport_plus_state`では、前の節の`bell_pair()`を呼び出してベル対の準備を再利用します。qkernelから別のqkernelを呼び出す方法は、通常のPython関数と同じです。最後に`bob`にアダマールゲートを適用し、X基底からZ基底に写します。テレポートされた状態が$|+\rangle$なら、測定結果は常に0になります。


# %%
@qm.qkernel
def teleport_plus_state() -> tuple[qm.Shot[qm.Bit], qm.Shot[qm.Bit], qm.Shot[qm.Bit]]:
    """Teleport a plus state and measure Bob in the X basis."""
    message = qm.h(qm.prepare(0))
    state = qm.concat(message, bell_pair())

    state = qm.cx(state, control=0, target=1)
    state = qm.h(state, target=0)
    alice_qubits, bob = qm.split_at(state, 2)
    alice_bits = qm.measure(alice_qubits)
    m_message, m_alice = alice_bits[0], alice_bits[1]

    if m_alice:
        bob = qm.x(bob)
    if m_message:
        bob = qm.z(bob)

    bob_bits = qm.measure(qm.h(bob))
    return m_message, m_alice, bob_bits[0]


# %% [markdown]
# `teleport_plus_state`は`bell_pair()`を呼び出し、qkernelによるベル対の準備を再利用します。測定ビットを3つ返すため、`qm.run()`で得られる各測定結果にも`(m_message, m_alice, bob)`の3つの要素が含まれます。

# %%
teleport_executable = qm.compile(qm.specialize(teleport_plus_state), to=endpoint.target)
teleport_result = qm.run(teleport_executable, on=endpoint, shots=shots, seed=42).result

for outcome, count in teleport_result.counts.items():
    print(f"Bell measurement={outcome[:2]}, Bob(X basis)={outcome[2]}: {count}")

assert sum(teleport_result.counts.values()) == shots
assert all(bob == 0 for _, _, bob in teleport_result.counts)


# %% [markdown]
# Bobの測定結果は常に0になります。一方、Aliceの測定結果はランダムです。テレポートされた量子状態の情報は、2つの古典ビットだけには現れません。Bobは共有したベル対と条件付きの補正を組み合わせることで状態を復元できます。

# %% [markdown]
# ## まとめ
#
# - `@qm.qkernel`を使って量子操作を記述します。
# - ゲートは更新された量子ビットのハンドルを返すため、戻り値を変数に再代入します。
# - `measure()`で`Q[Bit]`型の状態を測定すると、各ショットに1つの古典ビットが対応する`Shot[Bit]`が得られます。
# - 測定した`Bit`を条件にするPythonの`if`文は動的制御に変換されます。
# - qkernelから別のqkernelを呼び出して量子操作を再利用できます。
# - `qm.run()`で、測定値を返す量子カーネルを実行します。
