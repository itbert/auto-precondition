# Auto Preconditioner

Репозиторий для экспериментов с предобуславливателями линейных систем, в том числе для плотных комплексных импедансных матриц из задачи о метаструктурах кольцевых резонаторов.

Реальные данные лежат в `tests/data/compressed_data` и `tests/data/compressed_data555`.

## Идея CNNApprox

Новая CNN в `preconditioner/cnn.py` не пытается предсказывать полную плотную обратную матрицу напрямую. Вместо этого используется гибридная схема:

1. Базовый предобуславливатель `Jacobi` строится аналитически по диагонали матрицы.
2. CNN получает матрицу как изображение:
   исходные real/imag каналы,
   Jacobi-residual каналы `P_jacobi A - I`,
   координатные каналы и маску диагонали.
3. Сеть предсказывает не всю матрицу, а коэффициенты low-rank базиса correction-операторов, извлечённого из train inverses через SVD.
4. Итоговый предобуславливатель равен `P = P_jacobi + Delta_cnn`, а для устойчивости correction автоматически масштабируется по критерию `||PA - I||_F`.

Такой вариант хорошо подходит для матриц проекта вида `Z(omega) = z_diag(omega) I + offdiag(omega)`, где inverse меняется гладко по частоте, и одновременно остаётся пригодным для других семейств плотных матриц.

## Быстрый запуск

Установка зависимостей:

```bash
pip install numpy scipy matplotlib torch
```

Запуск обычного real-data benchmark без CNN:

```bash
python tests/benchmark_real_data.py --dataset large --freq-limit 10 --no-ridge
```

Запуск benchmark с CNNApprox:

```bash
python tests/benchmark_real_data.py \
  --dataset large \
  --freq-limit 24 \
  --cnn \
  --cnn-train-limit 16 \
  --cnn-epochs 10 \
  --cnn-basis-rank 6 \
  --cnn-hidden-channels 12
```

## Демонстрационный эксперимент с графиками

Готовый воспроизводимый пример:

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

Скрипт создаёт папку `experiments/cnn_real_data_demo_output/` и сохраняет:

- `records.csv` с per-frequency метриками;
- `summary.json` с агрегатами;
- `training_loss.png`;
- `gmres_iterations_vs_frequency.png`;
- `preconditioner_consistency_vs_frequency.png`;
- `inverse_relative_error_vs_frequency.png`.

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
)
model.fit(A_train, A_inv_train, verbose=True)

P = model.predict_inverse(A_test)
M = model.as_preconditioner(A_test).as_linear_operator()
```

Где `A_train` это список квадратных матриц, а `A_inv_train` это список их inverse-операторов, обычно `np.linalg.inv(A)`.
