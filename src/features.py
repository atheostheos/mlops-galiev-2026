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

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import load_params


def build_preprocessor(params: dict[str, Any]):
    params = load_params()
    f = params["features"]

    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])

    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    bin_pipe = "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, f["numeric"]),
            ("cat", cat_pipe, f["categorical"]),
            ("bin", bin_pipe, f["binary"]),
        ],
        remainder="drop",
    )
    return preprocessor
