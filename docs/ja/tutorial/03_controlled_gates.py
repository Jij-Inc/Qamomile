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
# # `qm.control`による制御演算
#
# `qm.control`は、1つ以上の制御量子ビットの状態に応じて、ターゲット量子ビットに量子カーネルの演算を適用します。デフォルトでは、すべての制御量子ビットが$|1\rangle$のときに演算が適用されます。`qm.control`に渡す量子カーネルは、ユニタリ演算を実装している必要があります。
#
# この操作では制御量子ビットを測定せず、重ね合わせの状態全体に演算を適用します。制御量子ビットが1つの場合は、次のように表せます。
#
# $$
# (\alpha|0\rangle+\beta|1\rangle)|t\rangle
# \longrightarrow
# \alpha|0\rangle|t\rangle+\beta|1\rangle U|t\rangle.
# $$
#
# ここで、$|t\rangle$はターゲットの状態、$U$は量子カーネルが定義する演算です。制御量子ビットが$|0\rangle$の成分ではターゲットは変化せず、$|1\rangle$の成分では$U$が適用されます。制御量子ビットとターゲットを合わせた系では、両成分間のコヒーレンスが保たれます。この操作によって、両者の間にエンタングルメントが生じることもあります。
#
# 本章では制御RXゲートの例から始め、複数量子ビットに作用するカーネルや、複数の制御量子ビットを使う例へ進みます。さらに、演算を適用する条件の指定、量子レジスタからの制御量子ビットの選択、`Const`を使った制御量子ビット数の指定を扱います。

# %%
# Install the latest Qamomile through pip.
# # !pip install qamomile

# %%
from __future__ import annotations

from math import pi

import qamomile as qm

# %% [markdown]
# ## 1. 最初の例：制御RXゲート
#
# Qamomileでは、`qm.rx(state, theta)`で1つの量子ビットにRXゲートを適用します。`theta`には回転角をラジアンで指定します。計算基底$\{|0\rangle, |1\rangle\}$では、RXゲートは次の行列で表されます。
#
# $$
# R_X(\theta)=
# \begin{pmatrix}
# \cos(\theta/2) & -i\sin(\theta/2)\\
# -i\sin(\theta/2) & \cos(\theta/2)
# \end{pmatrix}.
# $$
#
# この例では、ターゲットの初期状態を$|0\rangle$とし、$\theta=\pi$を指定します。この回転によって、状態は次のように変化します。
#
# $$
# R_X(\pi)|0\rangle
# =
# \begin{pmatrix}
# 0 & -i\\
# -i & 0
# \end{pmatrix}
# \begin{pmatrix}1\\0\end{pmatrix}
# =
# \begin{pmatrix}0\\-i\end{pmatrix}
# =
# -i|1\rangle.
# $$
#
# この状態を測定すると、確率1で1が得られます。係数$-i$は位相因子であり、この測定の確率を変えません。
#
# この回転を`rotate`という量子カーネルとして定義し、`qm.control`に渡すことで、制御量子ビットの状態に応じて適用します。
#
# 以下の二つの例は、制御量子ビットの初期状態だけが異なります。`qm.prepare(False)`は$|0\rangle$を、`qm.prepare(True)`は$|1\rangle$を用意します。どちらの例でも、ターゲットの初期状態は$|0\rangle$、回転角は$\pi$です。


# %%
@qm.qkernel
def rotate(state: qm.Q[qm.Bits[1]], theta: qm.Real) -> qm.Q[qm.Bits[1]]:
    return qm.rx(state, theta)


@qm.qkernel
def crx_control_off() -> qm.Shot[qm.Bits[2]]:
    control = qm.prepare(False)
    target = qm.qubits(1)
    control, target = qm.control(
        rotate,
        control,
        target,
        pi,
    )
    return qm.measure(qm.concat(control, target))


@qm.qkernel
def crx_control_on() -> qm.Shot[qm.Bits[2]]:
    control = qm.prepare(True)
    target = qm.qubits(1)
    control, target = qm.control(
        rotate,
        control,
        target,
        pi,
    )
    return qm.measure(qm.concat(control, target))


# %%
assert qm.evaluate(qm.specialize(crx_control_off)) == (0, 0)
assert qm.evaluate(qm.specialize(crx_control_on)) == (1, 1)

# %% [markdown]
# 回転が適用されるのは、制御量子ビットが$|1\rangle$である`crx_control_on`だけです。この場合、ターゲットを測定すると1が得られます。`crx_control_off`では回転は適用されず、ターゲットは$|0\rangle$のままです。
#
# 上の`assert`文では、`qm.evaluate(qm.specialize(...))`を使い、1ショット実行して結果を確認しています。`qm.concat(control, target)`は制御量子ビット、ターゲットの順に結合するため、結果のタプルもこの順に並び、それぞれ`(0, 0)`と`(1, 1)`になります。
#
# > **補足：** Qamomileには独自のシミュレーターが組み込まれています。`qm.evaluate(...)`はこれを使って1ショット実行するため、エンジンを指定する必要はありません。`qm.run(...)`で同じシミュレーターを使う場合は、`on=qm.Reference()`を指定します。
#
# **注目するポイント：**
#
# - `qm.control`には、量子カーネル、制御量子ビットの状態、ターゲットの状態、そのカーネルに渡す追加の引数の順に渡します。この例の追加引数は回転角です。
# - `qm.control`は、演算後の制御量子ビットとターゲットの状態を返します。`control, target`で受け取り、以降の演算にはこれらの戻り値を使います。

# %% [markdown]
# ## 2. 量子カーネル全体を制御する
#
# 前の例では、`rotate`に含まれるゲートはRXゲート1つでした。量子カーネルには、複数の量子ビットに作用する一連のゲート操作を記述することもできます。そのカーネルを`qm.control`に渡すと、操作全体を制御量子ビットの状態に応じて適用できます。
#
# ここで使う`x_all_two`は、2つの量子ビットそれぞれにXゲートを適用します。Xゲートは$|0\rangle$と$|1\rangle$を入れ替えるため、このカーネルは$|00\rangle$を$|11\rangle$に変えます。


# %%
@qm.qkernel
def x_all_two(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    for index in range(2):
        state = qm.x(state, target=index)
    return state


@qm.qkernel
def vector_target_demo() -> qm.Shot[qm.Bits[3]]:
    control = qm.prepare(True)
    target = qm.qubits(2)
    control, target = qm.control(x_all_two, control, target)
    return qm.measure(qm.concat(control, target))


# %%
assert qm.evaluate(qm.specialize(vector_target_demo)) == (1, 1, 1)

# %% [markdown]
# 制御量子ビットの初期状態は$|1\rangle$、2つのターゲットの初期状態は$|00\rangle$です。制御付きの呼び出しによって両方のXゲートが適用され、ターゲットは$|11\rangle$になります。したがって、測定結果は制御量子ビットを先頭として`(1, 1, 1)`になります。制御量子ビットを$|0\rangle$で準備した場合は、どちらのXゲートも適用されず、結果は`(0, 0, 0)`になります。
#
# `x_all_two`自体に、制御量子ビットを受け取る引数を追加する必要はありません。呼び出す側が`qm.control`に制御量子ビットの状態を別途渡すため、同じカーネルを制御付きでも制御なしでも再利用できます。
#
# `qm.control`に渡すカーネルは、先頭の引数で量子状態を1つ受け取り、それと同じ型の更新後の量子状態を返す必要があります。この例では、どちらも`qm.Q[qm.Bits[2]]`です。量子状態に加えて、前の`rotate`の回転角`theta`のような古典的な値も引数として受け取れます。
#
# 制御対象のカーネルはユニタリ演算を実装する必要があります。その内部には、測定や、量子ビットを$|0\rangle$に初期化するリセット操作を含めることはできません。この例では、`qm.control`の呼び出しの後に、`vector_target_demo`内で測定しています。
#
# > **補足：** 回転角`theta`のような古典的な値は、キーワード引数として渡すこともできます。カーネル定義でその引数にデフォルト値が設定されていれば、呼び出し時に省略できます。

# %% [markdown]
# ## 3. 複数の制御量子ビットと制御条件の指定
#
# 複数の量子ビットを制御に使うには、それらをまとめた量子状態を`qm.control`の制御用の引数として渡します。デフォルトでは、すべての制御量子ビットが$|1\rangle$の場合に、カーネルの演算が適用されます。
#
# 次の例では、前に定義した`rotate`を2つの制御量子ビットで制御します。


# %%
@qm.qkernel
def multi_control_rx() -> qm.Shot[qm.Bits[3]]:
    controls = qm.prepare((1, 1))
    target = qm.qubits(1)
    controls, target = qm.control(
        rotate,
        controls,
        target,
        pi,
    )
    return qm.measure(qm.concat(controls, target))


# %%
assert qm.evaluate(qm.specialize(multi_control_rx)) == (1, 1, 1)

# %% [markdown]
# 両方の制御量子ビットを$|1\rangle$で準備しているため、RX回転が適用されます。最初の例と同様に、$R_X(\pi)$によってターゲットは$|0\rangle$から$-i|1\rangle$に変わり、測定すると1が得られます。どちらかの制御量子ビットを$|0\rangle$で準備した場合は、回転は適用されません。
#
# 同じ呼び出し方で、3つ以上の制御量子ビットも使えます。

# %% [markdown]
# ### 制御条件を変更する
#
# `control_state`で、どの制御状態のときに演算を適用するかを指定できます。この引数は適用条件を指定するもので、制御量子ビットを準備したり測定したりするものではありません。
#
# Qamomileでは、インデックス0の制御量子ビットを**最下位ビット**として扱います。2つの制御量子ビットの値が`(b0, b1)`なら、対応する整数は`b0 + 2*b1`です。そのため、`control_state=1`は`(1, 0)`、つまり最初の制御量子ビットが1、2つ目が0の場合に演算を適用する指定になります。
#
# 次の例では、2つの独立したターゲットにXゲートを適用します。最初の呼び出しでは`control_state=1`を指定し、2つ目では両方の制御量子ビットが1というデフォルトの条件を使います。


# %%
@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)


@qm.qkernel
def activation_patterns() -> qm.Shot[qm.Bits[4]]:
    controls = qm.prepare((1, 0))
    first = qm.qubits(1)
    second = qm.qubits(1)

    controls, first = qm.control(flip, controls, first, control_state=1)
    controls, second = qm.control(flip, controls, second)

    return qm.measure(qm.concat(controls, first, second))


# %%
assert qm.evaluate(qm.specialize(activation_patterns)) == (1, 0, 1, 0)

# %% [markdown]
# 制御量子ビットを`(1, 0)`で準備しているため、最初のターゲットは$|1\rangle$に反転します。2つ目のターゲットは、デフォルトの条件である`(1, 1)`を満たさないため、$|0\rangle$のままです。戻り値のタプルには、2つの制御量子ビットの測定結果、その後に最初と2つ目のターゲットの測定結果が並びます。
#
# > **補足：** `control_state`には、制御量子ビット数で表せる範囲の非負の整数リテラルを指定します。制御量子ビットが2つの場合、指定できる値は`0`、`1`、`2`、`3`です。

# %% [markdown]
# ## 4. レジスタから制御量子ビットを選ぶ
#
# レジスタ（量子ビットのまとまり）のうち、一部だけを制御に使う場合があります。1つの量子ビットをインデックスで選ぶには`at`を、複数の量子ビットをまとめて使うには`qm.split_at`と`qm.concat`を使います。

# %% [markdown]
# ### 1つの制御量子ビットを選ぶ
#
# `at`引数で、制御レジスタ内のどの量子ビットを演算の適用条件に使うかを指定します。インデックスは0から始まります。


# %%
@qm.qkernel
def selected_control() -> qm.Shot[qm.Bits[3]]:
    controls = qm.prepare((1, 0))
    target = qm.qubits(1)
    controls, target = qm.control(flip, controls, target, at=0)
    return qm.measure(qm.concat(controls, target))


# %%
assert qm.evaluate(qm.specialize(selected_control)) == (1, 0, 1)

# %% [markdown]
# ここでは、`at=0`で最初の制御量子ビットを選んでいます。この量子ビットは$|1\rangle$で準備されているため、ターゲットが反転します。2つ目の量子ビットは適用条件に含まれません。`at=1`に変更すると、$|0\rangle$で準備された2つ目の量子ビットが選ばれ、結果は`(1, 0, 0)`になります。
#
# `qm.control`は、`at`で1つの量子ビットだけを選んだ場合も、制御レジスタ全体の状態とターゲットの状態を返します。この例では、それらの戻り値を`controls`と`target`という名前で受け取り、結合して測定しています。
#
# `at`を指定した場合、`control_state`は選択した量子ビットだけに対する条件になります。そのため、指定できる値は`0`と`1`です。

# %% [markdown]
# ### 複数の制御量子ビットを選ぶ
#
# 複数の量子ビットを制御に使う場合は、`qm.split_at`で必要な量子ビットを分け、`qm.concat`でまとめます。
#
# `qm.split_at(state, k)`は、先頭の`k`個の量子ビットと残りの量子ビットに分割します。`qm.concat`は、引数に渡した順に結合します。これらは量子ビットのまとめ方やインデックスを変える操作です。量子ゲートは追加されず、既存のエンタングルメントも保たれます。
#
# 次の例では、元のインデックス0、1、3の量子ビットを制御に使い、インデックス2の量子ビットをターゲットにします。


# %%
@qm.qkernel
def subset_control() -> qm.Shot[qm.Bits[4]]:
    state = qm.prepare((1, 1, 0, 1))
    prefix, remainder = qm.split_at(state, 2)  # 元の[0:2]と[2:4]
    target, last_control = qm.split_at(remainder, 1)  # 元の[2:3]と[3:4]

    controls = qm.concat(prefix, last_control)  # 元のインデックス0、1、3
    controls, target = qm.control(flip, controls, target)

    prefix, last_control = qm.split_at(controls, 2)  # 元の[0:2]と[3:4]
    state = qm.concat(prefix, target, last_control)  # 元の順序：0、1、2、3
    return qm.measure(state)


# %%
assert qm.evaluate(qm.specialize(subset_control)) == (1, 1, 1, 1)

# %% [markdown]
# 選択した3つの制御量子ビットはすべて$|1\rangle$なので、元のインデックス2にあるターゲットが$|0\rangle$から$|1\rangle$に反転します。最後の結合で元のインデックス順に戻してから測定しています。
#
# 制御量子ビットとターゲットには、別々の量子ビットを使う必要があります。レジスタを分割することで、それぞれを別の引数として渡せます。ゲート操作と同様に、分割や結合でも入力値は消費されるため、以降の操作には返された値を使います。
#
# > **補足：** 現在、これらの分割・結合には、量子ビット数が固定されたレジスタが必要です。分割位置も具体的な整数で指定します。`Const`引数で表したシンボリックな量子ビット数や分割位置には対応していません。

# %% [markdown]
# ### 実行ごとにインデックスを指定する
#
# 実行ごとに制御量子ビットを1つ選ぶには、`index: qm.Param[qm.Int]`のようなカーネル引数を追加し、`at=index`として渡します。`index`の値は、`qm.run`の`parameters`引数で指定します。
#
# この使い方には、実行時に量子ビットを選択する機能をエンジンがサポートしている必要があります。`qm.Reference()`は、この`at`の使い方に対応しています。

# %% [markdown]
# ## 5. 制御演算にパラメータを使う
#
# チュートリアル02で紹介した`Const`と`Param`は、制御演算にも使えます。次の例では、制御量子ビット数、演算の繰り返し回数、回転角をパラメータとして指定します。

# %% [markdown]
# ### 制御量子ビット数をパラメータで指定する
#
# 制御量子ビット数を`Const`パラメータとして宣言し、`qm.specialize`を呼び出すときに値を指定します。この例の`qm.control`は、作成した制御レジスタ内のすべての量子ビットを制御に使います。
#
# ループでは、`controls`の各量子ビットにXゲートを適用し、すべてを$|1\rangle$にしています。ここでの`target=index`は、`controls`内のどの量子ビットにXゲートを適用するかを指定しています。制御演算のターゲットとなる別のレジスタは、このループの後で作成しています。


# %%
@qm.qkernel
def variable_controls(
    width: qm.Const[qm.Int],
) -> tuple[qm.Shot[qm.Bits[width]], qm.Shot[qm.Bits[1]]]:  # noqa: F821
    controls = qm.qubits(width)
    for index in range(width):
        controls = qm.x(controls, target=index)
    target = qm.qubits(1)
    controls, target = qm.control(flip, controls, target)
    return qm.measure(controls), qm.measure(target)


# %%
variable_program = qm.specialize(variable_controls, width=3)
assert qm.evaluate(variable_program) == ((1, 1, 1), (1,))

# %% [markdown]
# `width=3`では、3つの制御量子ビットと1つのターゲット量子ビットが作成されます。ループですべての制御量子ビットを$|1\rangle$にしているため、制御付きXゲートによってターゲットが$|1\rangle$に反転します。
#
# 2つのレジスタを別々に測定しているため、結果には、3つの制御量子ビットの測定結果と1つのターゲットの測定結果を収めた、2つのタプルが含まれます。
#
# 同じカーネルに対して、`width`に別の正の整数を指定して`qm.specialize`を呼ぶこともできます。制御量子ビット数が変わっても、`control_state`の省略は「すべての制御量子ビットが$|1\rangle$」という条件を意味します。

# %% [markdown]
# ### 制御演算を繰り返す
#
# ループを使うと、同じ制御演算を複数回適用できます。この例では、繰り返し回数`repetitions`を`Const`として宣言し、`qm.specialize`を呼び出すときに値を指定します。一方、回転角`theta`は`Param`として宣言しているため、実行ごとに値を指定できます。


# %%
@qm.qkernel
def repeated_control(
    repetitions: qm.Const[qm.Int],
    theta: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[2]]:
    state = qm.prepare((1, 0))
    control, target = qm.split_at(state, 1)
    for _ in range(repetitions):
        control, target = qm.control(rotate, control, target, theta)
    return qm.measure(qm.concat(control, target))


# %% [markdown]
# 演算$U$を同じ制御条件で$k$回繰り返すと、制御付きの$U^k$に相当します。この例では、同じ制御量子ビットを使ってRX回転を繰り返しています。

# %%
program = qm.specialize(repeated_control, repetitions=3)
artifact = qm.compile(program, to=qm.Reference())

record = qm.run(
    artifact,
    on=qm.Reference(),
    parameters={"theta": pi},
    shots=8,
    seed=7,
)

print("controlled rotations:", record.result.counts)
assert record.result.counts == {(1, 1): 8}

# %% [markdown]
# 制御量子ビットを$|1\rangle$で準備しているため、3回すべての回転が適用されます。$\theta=\pi$ではターゲットの測定結果が1になり、各ショットで`(1, 1)`が得られます。
#
# 同じ`artifact`を、別の`theta`の値でも再利用できます。例えば、`theta`を`0.0`にするとターゲットは$|0\rangle$のままなので、結果は`(1, 0)`になります。一方、この例では`repetitions`を`Const`として宣言しているため、その値を変更するには、新しい値でのスペシャライズとコンパイルが必要です。

# %% [markdown]
# ## 6. 制御演算における位相（補足）
#
# `qm.global_phase(state, angle)`は、状態全体に$e^{i\,\mathrm{angle}}$を掛けます。`angle`には角度をラジアンで指定します。このような**グローバル位相**は、それだけでは測定確率を変えません。
#
# 一方、この演算を`qm.control`で適用すると、位相因子が掛かるのは制御条件を満たす成分だけです。そのため、制御量子ビットとターゲットを合わせた状態の成分間の**相対位相**となり、その後の測定結果に影響することがあります。
#
# 次の例では、角度を$\pi$に指定し、位相因子を$-1$にしています。制御演算の前後にアダマール（H）ゲートを適用し、この位相差が測定結果に現れるようにします。


# %%
@qm.qkernel
def scalar_phase(state: qm.Q[qm.Bits[1]], angle: qm.Real) -> qm.Q[qm.Bits[1]]:
    return qm.global_phase(state, angle)


@qm.qkernel
def phase_kickback() -> qm.Shot[qm.Bits[2]]:
    control = qm.prepare(False)
    control = qm.h(control)
    target = qm.qubits(1)
    control, target = qm.control(
        scalar_phase,
        control,
        target,
        pi,
    )
    control = qm.h(control)
    return qm.measure(qm.concat(control, target))


# %%
assert qm.evaluate(qm.specialize(phase_kickback)) == (1, 0)

# %% [markdown]
# 最初のアダマールゲートによって、制御量子ビットは$(|0\rangle+|1\rangle)/\sqrt{2}$になります。続く制御演算によって、制御量子ビットとターゲットを合わせた状態は次のように変わります。
#
# $$
# \frac{|0\rangle+|1\rangle}{\sqrt{2}}|0\rangle
# \;\longrightarrow\;
# \frac{|0\rangle-|1\rangle}{\sqrt{2}}|0\rangle.
# $$
#
# この角度では、`scalar_phase`の演算は$-I$です。制御量子ビットが1つの場合、その制御付き演算は
#
# $$
# C(-I)=Z_{\mathrm{control}}\otimes I_{\mathrm{target}}
# $$
#
# となります。つまり、ターゲットの状態を変えずに、制御量子ビットにZゲートを適用することに相当します。ターゲット側の演算に含まれる位相因子が、制御量子ビットの相対位相として現れています。これが**位相キックバック**（phase kickback）です。
#
# 2つ目のアダマールゲートは、$(|0\rangle-|1\rangle)/\sqrt{2}$を$|1\rangle$に変えます。ターゲットは$|0\rangle$のままなので、測定結果は`(1, 0)`になります。
#
# 角度を`0.0`にした場合は位相差が生じず、2つ目のアダマールゲートによって制御量子ビットが$|0\rangle$に戻るため、結果は`(0, 0)`になります。
#
# 同じ理由で、制御付き演算を作るときには、$R_X(\pi)=-iX$の位相因子$-i$も一般には無視できません。

# %% [markdown]
# ## 7. まとめ
#
# `qm.control`は、制御量子ビットを測定せずに、その状態に応じて量子カーネルのユニタリ演算を適用します。カーネルには、単一のゲートだけでなく、複数のターゲット量子ビットに作用する一連の演算も記述できます。
#
# - **制御条件の指定**：デフォルトでは、すべての制御量子ビットが$|1\rangle$の場合に演算が適用されます。適用条件となる0と1の組み合わせを変更するには、`control_state`を指定します。
# - **制御量子ビットの選択**：制御レジスタから1つの量子ビットを選ぶには`at`を使います。量子ビット数が固定されたレジスタでは、`qm.split_at`と`qm.concat`で複数の制御量子ビットをまとめられます。
# - **制御量子ビットとターゲット**：両者には別々の量子ビットを使います。`qm.control`から返される制御レジスタとターゲットの状態を、以降の演算に使います。
# - **パラメータの指定**：この章の例では、`Const`で宣言した制御量子ビット数や繰り返し回数を、`qm.specialize`を呼び出すときに確定します。`Param`で宣言した回転角は、実行ごとに変更できます。
# - **ループによる繰り返し**：制御付きの演算$U$を$k$回繰り返すと、制御付きの$U^k$に相当します。
# - **位相の扱い**：位相キックバックの例で見たように、元の演算のグローバル位相は、制御を付けると相対位相として現れ、その後の測定結果に影響することがあります。
#
# **次へ**：[リソース推定](04_resource_estimation.ipynb)では、量子カーネルを実行せずに、量子ビット数やゲート数などの論理リソースを見積もる方法を説明します。
