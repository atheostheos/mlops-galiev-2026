import matplotlib
import numpy as np
import pandas as pd

from src.config import data_md5, file_md5, git_sha, resolve

matplotlib.use("Agg")
import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def log_roc_curve(y_true: pd.Series, y_proba: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    mlflow.log_figure(fig, "roc_curve.png")
    plt.close(fig)


def log_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)


def log_to_mlflow(params, pipe, metrics, y_proba, y_true, input_example) -> None:
    import mlflow
    import mlflow.sklearn

    cfg = params["mlflow"]
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    mlflow.set_experiment(cfg["experiment_name"])

    name = params["train"]["model"]
    with mlflow.start_run():
        mlflow.log_params({"model": name, "seed": params["seed"]})
        mlflow.log_params({f"{name}.{k}": v for k, v in params["train"][name].items()})
        mlflow.log_metrics(metrics)
        mlflow.set_tag("git_sha", git_sha())
        mlflow.log_artifact(resolve("params.yaml"))
        mlflow.sklearn.log_model(pipe, artifact_path="model", input_example=input_example)

        # log graphs
        log_roc_curve(y_true, y_proba)

        thresh = params["evaluate"]["threshold"]
        y_pred = (y_proba >= thresh).astype(int)
        log_confusion_matrix(y_true, y_pred)

        # log md5 hashes
        mlflow.set_tag("data_md5", data_md5(params["data"]["raw_path"]))

        model_dir = resolve(params["train"]["save_dir"])
        model_path = os.path.join(model_dir, "model.joblib")
        mlflow.set_tag("model_md5", file_md5(model_path))
