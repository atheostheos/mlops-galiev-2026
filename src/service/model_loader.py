import json
import os
from datetime import datetime, timezone

import joblib

from src.config import feature_columns, load_params, resolve
from src.logging_setup import setup_logging

log = setup_logging()


class ModelHolder:
    META_KEYS = ("source", "name", "version", "alias", "registered_at", "run_id", "git_sha")

    def __init__(self) -> None:
        self.model = None
        self.meta = None
        self.version = "not-loaded"

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def version_info(self) -> dict | None:
        return dict(self.meta) if self.meta else None

    def load(self) -> None:
        params = load_params()
        if params["mlflow"]["enabled"] and self._try_registry(params):
            return
        self._load_local(params)

    def _try_registry(self, params) -> bool:
        try:
            import mlflow
            from mlflow import MlflowClient

            mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
            name = params["mlflow"]["registered_model_name"]
            alias = params["service"]["model_alias"]

            model_uri = f"models:/{name}@{alias}"
            self.check_signature(model_uri)
            self.model = mlflow.sklearn.load_model(model_uri)
            self.version = f"registry:{name}@{alias}"
            self.meta = self._registry_meta(MlflowClient(), name, alias)
            return True
        except RuntimeError as exc:
            raise RuntimeError("Ошибка валидации сигнатуры") from exc
        except Exception as exc:
            log.warning("MLflow Registry недоступен (%s), берём локальный файл", exc)
            return False

    def check_signature(self, model_uri: str) -> None:
        import mlflow

        signature = mlflow.models.get_model_info(model_uri).signature
        if signature is None:
            raise RuntimeError("у модели нет сигнатуры — обучите с input_example")

        expected = set(feature_columns(load_params()))
        actual = {c.name for c in signature.inputs.inputs}
        if actual != expected:
            raise RuntimeError(
                f"сигнатура разошлась с конфигом: лишние {actual - expected}, недостающие {expected - actual}"
            )

    def _registry_meta(self, client, name: str, alias: str) -> dict:
        mv = client.get_model_version_by_alias(name, alias)
        run = client.get_run(mv.run_id)
        return {
            "source": "registry",
            "name": name,
            "alias": alias,
            "version": mv.version,
            "registered_at": datetime.fromtimestamp(
                mv.creation_timestamp / 1000, tz=timezone.utc
            ).isoformat(),
            "git_sha": run.data.tags.get("git_sha", "unknown"),
            "run_id": mv.run_id,
        }

    def _local_meta(self, meta_path: str) -> dict:
        with open(meta_path) as f:
            file_meta = json.load(f)
        return {
            **{key: file_meta.get(key) for key in self.META_KEYS},
            "source": "local",
        }

    def _load_local(self, params) -> None:
        model_path = resolve(params["service"]["model_path"])
        meta_path = os.path.join(os.path.dirname(model_path), "model_meta.json")

        self.model = joblib.load(model_path)

        self.meta = self._local_meta(meta_path)
        self.version = f"local:{self.meta['git_sha']}"


holder = ModelHolder()
