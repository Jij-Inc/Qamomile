---
slug: integration
---

# インテグレーション

Qamomileを外部ライブラリや量子プラットフォームと組み合わせて使うときのノートです。

::::{grid} 1 1 1 1

:::{card}
:header: **Amazon Braketサポート**
:link: braket_support
型付きmeasurement feedbackをportable OpenQASM 3へコンパイルしてlocalで実行し、managed deviceをendpoint境界に保ちます。
:::

:::{card}
:header: **serialized kernel artifactによるCUDA-Qサポート**
:link: cudaq_support
1つの型付きprogramをserialized CUDA-Q kernel dataへコンパイルし、inspectionして分離された`qpp-cpu` workerで実行します。
:::

:::{card}
:header: **HUGRサポート**
:link: hugr_support
QamomileプログラムをHUGRへコンパイルし、Seleneでローカル実行します。
:::

:::{card}
:header: **Qamomileで使うOMMX Quantum Benchmarks**
:link: ommx_quantum_benchmarks_qaoa
9量子ビットのLABSインスタンスを不変に正規化し、型付きQAOAを1回実行して同梱の参照解と比較します。
:::

:::{card}
:header: **qBraid providerでの実行**
:link: qbraid_executor
型付きQamomileプログラムから決定的なQPYを生成し、live qBraid endpointから送信します。
:::

:::{card}
:header: **Qiskit サポート**
:link: qiskit_support
1つの型付きprogramをportable QPYへコンパイルし、Aerで再利用して、同じsourceをlive BackendV2へrouteします。
:::

:::{card}
:header: **QURI Parts サポート**
:link: quri_parts_support
1つの型付きprogramをportable logical circuitへコンパイルし、inspection、Qulacs sampling、期待値評価に再利用します。
:::

::::
