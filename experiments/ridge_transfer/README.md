# Ridge Transfer Experiment

Эта директория добавляет воспроизводимый эксперимент для текущего обучаемого baseline из `preconditioner/ml.py`.

Что внутри:
- [ridge_transfer_experiment.ipynb]
  ноутбук с обучением, base-validation, zero-shot переносом на слегка сдвинутые матрицы и микро-дообучением;
- [ridge_transfer_helpers.py]
  вся вычислительная логика, чтобы ноутбук оставался коротким и чтобы эксперимент можно было запускать как обычный Python-скрипт.

Сценарий:
- ridge обучается на `Base Local SPD`;
- затем проверяется на том же распределении;
- потом сравнивается на `Shifted Local SPD` без дообучения и после очень малого refit на нескольких shifted-матрицах;
- дополнительно показано, как обученную модель зарегистрировать в `PreconditionerWorkbenchV2`.

Проверка без Jupyter:

```bash
python experiments/ridge_transfer/ridge_transfer_helpers.py
```
