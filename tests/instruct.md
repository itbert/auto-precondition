# Tests For Real Data

Все команды ниже запускать из корня репозитория:

```bash
cd ./auto-preconditioner
```

## Что есть

- `small` -> `tests/data/compressed_data`
- `large` -> `tests/data/compressed_data555`
- основной скрипт -> `tests/benchmark_real_data.py`

Скрипт строит матрицу импеданса так же, как `tests/solvercompressed.py`, но прогоняет все методы из фреймворка:

- preconditioners: `None`, `Diagonal`, `LU`, `Circulant`, `SVD`
- дополнительно: `RidgeApprox` из `preconditioner/ml.py`
- solvers: `GMRES`, `LGMRES`, `BiCGSTAB`

`RidgeApprox` включён по умолчанию.

## Быстрые проверки

Маленький датасет, короткий прогон:

```bash
python tests/benchmark_real_data.py --dataset small --freq-limit 10
```

Большой датасет, короткий прогон:

```bash
python tests/benchmark_real_data.py --dataset large --freq-limit 10
```

Большой датасет, короткий прогон со всеми аналитическими методами, включая `SVD`:

```bash
python tests/benchmark_real_data.py --dataset large --freq-limit 3 --ignore-size-limits
```

Маленький датасет без `RidgeApprox`:

```bash
python tests/benchmark_real_data.py --dataset small --freq-limit 10 --no-ridge
```

## Полные прогоны

Маленький датасет со всеми доступными для него методами, включая `RidgeApprox`:

```bash
python tests/benchmark_real_data.py --dataset small
```

Большой датасет:

```bash
python tests/benchmark_real_data.py --dataset large
```

Большой датасет со всеми аналитическими методами, включая `SVD`:

```bash
python tests/benchmark_real_data.py --dataset large --ignore-size-limits
```

Маленький датасет без ridge:

```bash
python tests/benchmark_real_data.py --dataset small --no-ridge
```

Примечания:

- без `--ignore-size-limits` `SVD` на большом датасете пропускается автоматически, потому что во фреймворке для него стоит ограничение по размеру;
- с `--ignore-size-limits` `SVD` запускается принудительно, но это значительно медленнее;
- `RidgeApprox` на маленьком датасете запускается автоматически;
- `RidgeApprox` на большом датасете пропускается автоматически, потому что исходный ridge из воркбенча строит очень большую dense-модель по признакам размера `N^2`, и для `N=450` это непрактично по памяти и времени.

## Вывод сырых значений

Если нужны списки `iterations`, `info` и `residual` по каждой частоте:

```bash
python tests/benchmark_real_data.py --dataset small --print-raw
python tests/benchmark_real_data.py --dataset large --print-raw
```

Для большого датасета со всеми аналитическими методами:

```bash
python tests/benchmark_real_data.py --dataset large --print-raw --ignore-size-limits
```

Для маленького датасета без ridge:

```bash
python tests/benchmark_real_data.py --dataset small --print-raw --no-ridge
```

## Сохранение результатов в CSV

Маленький датасет:

```bash
python tests/benchmark_real_data.py --dataset small --csv-dir tests/output_small
```

Большой датасет:

```bash
python tests/benchmark_real_data.py --dataset large --csv-dir tests/output_large
```

Большой датасет со всеми аналитическими методами:

```bash
python tests/benchmark_real_data.py --dataset large --ignore-size-limits --csv-dir tests/output_large_all
```

Маленький датасет без ridge:

```bash
python tests/benchmark_real_data.py --dataset small --no-ridge --csv-dir tests/output_small_no_ridge
```

Скрипт создаёт два файла:

- `benchmark_real_data_summary.csv`
- `benchmark_real_data_raw.csv`

## Полезные параметры

Отключить ridge:

```bash
--no-ridge
```

Изменить коэффициент регуляризации ridge:

```bash
--ridge-reg 1e-3
```

Изменить target для ridge:

```bash
--ridge-target pinv
```

Ограничить число частот, на которых обучается ridge:

```bash
--ridge-train-limit 64
```

Пример:

```bash
python tests/benchmark_real_data.py --dataset small --ridge-reg 1e-4 --ridge-train-limit 64
```

## Если нужен запуск через notebook

Открыть:

- `tests/benchmark_real_data.ipynb`