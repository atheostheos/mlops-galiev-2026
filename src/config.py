"""Загрузка params.yaml и доступ к настройкам.

Один модуль-точка входа для конфигурации: скрипты не читают YAML сами
и не хранят пути внутри себя. Это то, ради чего на занятии 3 всё выносилось
из кода — здесь видно результат.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = Path(os.getenv("PARAMS_PATH", PROJECT_ROOT / "params.yaml"))


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    """Читает params.yaml. Кэш не нужен: файл маленький, а неявный кэш путает."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(relative: str) -> Path:
    """Путь из конфига -> абсолютный путь от корня проекта.

    Без этого проект работает только если запускать его из корня —
    классическая причина 'у меня локально работало'.
    """
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def feature_columns(params: dict[str, Any]) -> list:
    f = params["features"]
    return list(f["numeric"]) + list(f["categorical"]) + list(f["binary"])


def git_sha() -> str:
    """Короткий хеш текущего коммита. В копии без .git обучение падать не должно."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


TARGET = "churn"


def data_md5(path: str) -> str:
    with open(resolve("dvc.lock"), encoding="utf-8") as f:
        lock = yaml.safe_load(f)
    for out in lock["stages"]["generate"]["outs"]:
        if out["path"] == path:
            return out["md5"]
    return "unknown"


def file_md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
