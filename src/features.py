"""Построение препроцессора признаков.

TODO (занятие 1): собрать здесь ColumnTransformer.

Требования:
  * числовые признаки: заполнение пропусков + масштабирование;
  * категориальные: заполнение пропусков + OneHotEncoder;
  * бинарные: без изменений;
  * списки колонок берутся из params.yaml, а не пишутся в коде.

Подсказка: почему препроцессор обязан ехать в одном Pipeline с моделью,
разбирается на паре. Если сделать иначе — сервис на занятии 10 сломается.
"""
from __future__ import annotations

from typing import Any


def build_preprocessor(params: dict[str, Any]):
    # Подсказка по импортам, которые вам понадобятся:
    #   from sklearn.compose import ColumnTransformer
    #   from sklearn.impute import SimpleImputer
    #   from sklearn.pipeline import Pipeline
    #   from sklearn.preprocessing import OneHotEncoder, StandardScaler
    raise NotImplementedError("занятие 1: реализуйте препроцессор")
