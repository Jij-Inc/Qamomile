# Qamomile ドキュメントへようこそ

Qamomileは、量子最適化アルゴリズムのための強力なSDKです。数学的モデルを量子回路に変換することに特化しています。

## チュートリアル

### 基礎

- [Qamomile入門](tutorial/01_introduction) - 最初の量子回路、線形型、QiskitTranspilerでの実行
- [型システム](tutorial/02_type_system) - 型の全カタログ：Qubit、Float、UInt、Bit、Vector、Dict
- [量子ゲート一覧](tutorial/03_gates) - 完全なゲートリファレンス（全11ゲート）
- [重ね合わせとエンタングルメント](tutorial/04_superposition_entanglement) - 重ね合わせ、干渉、Bell/GHZ状態

### 標準ライブラリとアルゴリズム

- [標準ライブラリ](tutorial/05_stdlib) - QFT、IQFT、QPE、アルゴリズムモジュール
- [コンポジットゲート](tutorial/06_composite_gate) - CompositeGate、`@composite_gate`、スタブゲート
- [初めての量子アルゴリズム：Deutsch-Jozsa](tutorial/07_first_algorithm) - オラクルパターン、量子並列性と干渉
- [パラメトリック回路と変分量子アルゴリズム](tutorial/08_parametric_circuits) - bindingsとparameters、Observable、expval、変分量子分類器

### 発展トピック

- [代数的リソース推定](tutorial/09_resource_estimation) - SymPy式によるゲート数と回路深さの推定
- [トランスパイラの内部](tutorial/10_transpile) - @qkernelから実行可能プログラムへのパイプライン全体
- [カスタムExecutor](tutorial/11_custom_executor) - クラウド量子バックエンド（IBM Quantumなど）への接続

## 最適化

- [QAOA](optimization/qaoa) - QAOAによるMax-Cut問題の解法
- [FQAOA](optimization/fqaoa) - 制約付き最適化のためのFQAOA
- [QRAO](optimization/qrao31) - QRAOによる量子ビット効率の良い最適化
- [カスタムコンバータ](optimization/custom_converter) - 独自の最適化コンバータの作成

## インストール

```bash
pip install qamomile
```

## クイック例

```python
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

@qmc.qkernel
def bell_state(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Bit, qmc.Bit]:
    q0 = qmc.h(q0)
    q0, q1 = qmc.cx(q0, q1)
    return qmc.measure(q0), qmc.measure(q1)

transpiler = QiskitTranspiler()
executor = transpiler.executor()
executable = transpiler.compile(bell_state)
job = executable.sample(executor, shots=1000)
result = job.result()
print(f"Counts: {result.counts}")
```

## エラーハンドリング

`@qkernel` のエラーは2段階で発生します：

- **AST変換段階**: サポートされない制御フローパターン（直接シーケンス反復、`while` 条件内の量子操作など）は `SyntaxError` として拒否されます。
- **トランスパイラ / バックエンド段階**: 型違反、線形型エラー、バックエンド固有の問題は `QamomileCompileError` ファミリーを使用します。

`@qkernel` のエラーをキャッチする際は、`SyntaxError` と `QamomileCompileError` の両方を例外処理に含めてください。

## リンク

- [GitHub リポジトリ](https://github.com/Jij-Inc/Qamomile)
- [API リファレンス](api/index.md)
