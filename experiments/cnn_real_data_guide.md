# CNNApprox on Real Data

Этот гайд описывает workflow для новой CNN-модели над реальными комплексными импедансными матрицами.

## Что именно реализовано

`ConvolutionalInverseApproximator` не предсказывает всю плотную обратную матрицу напрямую. Вместо этого схема такая:

1. Строится аналитический baseline `Jacobi`.
2. CNN получает матрицу как многоканальное изображение:
   `Re(A)`, `Im(A)`, каналы residual-а `P_jacobi A - I`, координаты и маску диагонали.
3. На выходе сеть предсказывает коэффициенты low-rank базиса correction-операторов.
4. Итоговый оператор:
   `P = P_jacobi + Delta_cnn`.
5. При включённом `use_correction_line_search` correction дополнительно масштабируется по критерию `||PA - I||_F`.

Это хорошо работает на семействе матриц, где inverse меняется гладко по частоте.

## Зависимости

Минимально нужны:

```bash
pip install numpy scipy matplotlib torch
```

Если `torch` не установлен, `CNNApprox` в benchmark будет автоматически пропущен.

## Быстрый CLI workflow

Smoke-test:

```bash
python tests/benchmark_real_data.py \
  --dataset large \
  --freq-limit 12 \
  --cnn \
  --cnn-train-limit 8 \
  --cnn-epochs 4 \
  --cnn-basis-rank 4 \
  --cnn-hidden-channels 10 \
  --no-ridge
```

Более близко к demo-графикам:

```bash
python experiments/cnn_real_data_demo.py \
  --dataset large \
  --freq-limit 24 \
  --train-count 16 \
  --test-count 8 \
  --epochs 10 \
  --basis-rank 6 \
  --hidden-channels 12
```

Скрипт сохраняет результаты в `experiments/cnn_real_data_demo_output/`.

## Что означают основные флаги

Для `tests/benchmark_real_data.py`:
- `--cnn` включает `CNNApprox` как ещё один `PreconditionerFactory`.
- `--cnn-train-limit` задаёт число частот для обучения CNN внутри benchmark.
- `--cnn-hidden-channels` управляет шириной сети.
- `--cnn-basis-rank` задаёт размер low-rank базиса correction-операторов.
- `--cnn-epochs`, `--cnn-batch-size`, `--cnn-learning-rate` управляют обучением.
- `--cnn-consistency-weight` меняет вес штрафа за `||PA - I||`.
- `--cnn-max-n` ограничивает размер матрицы, для которого benchmark вообще будет пытаться учить CNN.

Для `experiments/cnn_real_data_demo.py`:
- `--freq-limit` ограничивает окно частот перед split-ом.
- `--train-count` и `--test-count` задают равномерный split по частотам.
- `--output-dir` позволяет не перезаписывать стандартную папку с графиками.

## Как читать графики

`training_loss.png`
- Показывает total loss, target loss и consistency loss по эпохам.
- Если target loss падает, а consistency loss стоит высоко, модель хорошо аппроксимирует inverse в базисе, но хуже контролирует `PA ≈ I`.

`gmres_iterations_vs_frequency.png`
- Сравнивает `None`, `Diagonal` и `CNNApprox` на отложенных частотах.
- Это главный график практической пользы: насколько предобуславливатель сокращает число итераций.

`preconditioner_consistency_vs_frequency.png`
- Показывает `||PA - I||_F / n` для `Diagonal` и `CNNApprox`.
- Чем ниже кривая, тем ближе предобуславливатель к точному inverse-оператору.

`inverse_relative_error_vs_frequency.png`
- Показывает относительную ошибку `||P_cnn - A^{-1}||_F / ||A^{-1}||_F`.
- Это полезно для диагностики качества модели, но в solver-задаче более важен график по итерациям.

## Программное использование

```python
import numpy as np

from preconditioner.cnn import ConvolutionalInverseApproximator

model = ConvolutionalInverseApproximator(
    matrix_size=A_train[0].shape[0],
    hidden_channels=12,
    basis_rank=6,
    epochs=10,
    batch_size=2,
    learning_rate=2e-3,
    consistency_weight=0.3,
)
model.fit(A_train, A_inv_train, verbose=True)

P = model.predict_inverse(A_test)
preconditioner = model.as_preconditioner(A_test, name="CNNApprox")
linear_operator = preconditioner.as_linear_operator()
```

Если нужен factory для общего фреймворка:

```python
from preconditioner.cnn import build_cnn_preconditioner_factory

factory = build_cnn_preconditioner_factory(model, name="CNNApprox")
```

## Notebook

Для пошагового сценария см. cnn_real_data_walkthrough.ipynb

В ноутбуке есть:
- подготовка путей и конфигурации;
- train/test split по частотам;
- обучение `ConvolutionalInverseApproximator`;
- воспроизведение графиков inline;
- сохранение `records.csv` и `summary.json`;
- ячейка с optional full benchmark по всем solver-ам.

## Практические замечания

- Offline training time не включается в per-matrix timings benchmark-а.
- На малом числе train frequencies лучше начинать с `basis_rank=4..6`, а не сразу с больших значений.
- Для интерактивного notebook-прогона полезно сначала уменьшать `freq_limit`, `train_count` и `epochs`.
- Если нужно воспроизвести уже сохранённые артефакты ближе к один-в-один, используйте `dataset=large`, `freq_limit=24`, `train_count=16`, `test_count=8`, `epochs=10`.
