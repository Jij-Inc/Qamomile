# Qamomile ドキュメントへようこそ

Qamomileは、量子最適化アルゴリズムのための強力なSDKです。数学的モデルを量子回路に変換することに特化しています。

## はじめに

チュートリアルでQamomileの使い方を学びましょう：

- [トランスパイルと実行](transpile/transpile_flow) - 量子カーネルの書き方とシミュレータ/実機での実行方法

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
- [API リファレンス](https://jij-inc.github.io/Qamomile/ja/)
