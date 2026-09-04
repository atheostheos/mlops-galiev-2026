"""Стадия train.

TODO (занятие 1): перенести сюда логику из notebooks/baseline_notebook.py,
исправив всё, что вы в ней нашли.

Обязательно:
  * никаких абсолютных путей — только src.config.resolve();
  * никаких магических чисел — только params.yaml;
  * зафиксированный seed;
  * модель сохраняется в models/model.joblib вместе с препроцессором;
  * метрики пишутся в reports/train_metrics.json.

Запуск: python -m src.train
"""
from __future__ import annotations

from src.config import load_params
from src.logging_setup import setup_logging

log = setup_logging()


def main() -> None:
    params = load_params()  # noqa: F841
    raise NotImplementedError("занятие 1: реализуйте обучение")


if __name__ == "__main__":
    main()
