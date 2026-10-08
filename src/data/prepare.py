"""Стадия prepare: сырой CSV -> train/val/test.

TODO (занятие 1):
  1. прочитать data/raw/churn.csv;
  2. обработать пропуски в total_charges осмысленно (не dropna!);
  3. разбить на train/val/test со stratify по churn и random_state из params;
  4. сохранить три CSV в data/processed/.

Проверка: два запуска подряд должны дать одинаковые файлы.
"""

from __future__ import annotations

import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import TARGET, load_params, resolve
from src.logging_setup import setup_logging

log = setup_logging()


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess data
    """
    null_charges = raw["total_charges"].isna()
    raw.loc[null_charges, "total_charges"] = (
        raw.loc[null_charges, "monthly_charges"] * raw.loc[null_charges, "tenure_months"]
    )

    raw = raw.drop_duplicates(subset=["customer_id"])

    return raw


def save_csv(df: pd.DataFrame, path: str, name: str) -> None:
    """
    Save DataFrame as csv in specific directory
    """
    out_path = os.path.join(path, f"{name}.csv")
    df.to_csv(out_path, index=False)

    log.info(f"{name} saved to {out_path}")


def main() -> None:
    params = load_params()
    d = params["data"]
    raw = pd.read_csv(resolve(d["raw_path"]))
    log.info("data loaded")

    df = clean(raw)
    log.info("data preprocessed")

    train_val, test = train_test_split(
        df, test_size=d["test_size"], random_state=params["seed"], stratify=df[TARGET]
    )

    val_ratio = d["val_size"] / (1.0 - d["test_size"])
    train, val = train_test_split(
        train_val, test_size=val_ratio, random_state=params["seed"], stratify=train_val[TARGET]
    )

    splits = {"train": train, "val": val, "test": test}

    processed_dir = resolve(d["processed_dir"])
    os.makedirs(processed_dir, exist_ok=True)

    for name, df in splits.items():
        save_csv(df, processed_dir, name)

    report = {
        name: {"size": len(split), "churn_mean": split[TARGET].mean().round(3)}
        for name, split in splits.items()
    }

    reports_dir = resolve(d["reports_dir"])
    with open(os.path.join(reports_dir, "prepare_report.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
