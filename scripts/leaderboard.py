import os

import mlflow
import pandas as pd
import yaml

from src.config import load_params, resolve


def model_md5(params) -> str | None:
    mode_dir = params["train"]["save_dir"]
    model_path = os.path.join(mode_dir, "model.joblib")

    with open(resolve("dvc.lock"), encoding="utf-8") as f:
        lock = yaml.safe_load(f)
    for out in lock["stages"]["train"]["outs"]:
        if out["path"] == model_path:
            return out["md5"]
    return None


def render_cell(row: pd.Series, column: str, fmt: str = "") -> str:
    value = row.get(column)
    return "-" if (value is None or pd.isna(value)) else format(value, fmt)


def get_model_params(row: pd.Series) -> str:
    prefix = f"params.{row.get('params.model')}."
    pairs = [
        f"{col.removeprefix(prefix)}={val}"
        for col, val in row.items()
        if col.startswith(prefix) and pd.notna(val)
    ]
    return ", ".join(pairs) or "-"


def render_leaderboard_md(params, runs: pd.DataFrame) -> str:
    report = [
        "# Топ 5 моделей из MLFlow по ROC-AUC",
        "",
        "| | model | params | ROC-AUC | PR-AUC | git_sha | date | run_id |",
        "|---|---|---|---|---|---|---|---|",
    ]

    current_model = model_md5(params)

    has_current = False
    for _, row in runs.iterrows():
        is_current = current_model is not None and row.get("tags.model_md5") == current_model
        has_current = has_current or is_current

        report.append(
            f"| {'текущая' if is_current else ''} "
            f"| {render_cell(row, 'params.model')} "
            f"| {get_model_params(row)} "
            f"| {render_cell(row, 'metrics.roc_auc', '.4f')} "
            f"| {render_cell(row, 'metrics.pr_auc', '.4f')} "
            f"| {render_cell(row, 'tags.git_sha')} "
            f"| {row['start_time']:%Y-%m-%d %H:%M} "
            f"| {row['run_id'][:8]} |"
        )
    if not has_current:
        report += ["", "Текущая модель не входит в топ 5 или не записана в MLFLow."]
    return "\n".join(report) + "\n"


def main():
    params = load_params()

    cfg = params["mlflow"]
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    runs: pd.DataFrame = mlflow.search_runs(
        experiment_names=[cfg["experiment_name"]],
        order_by=["metrics.roc_auc DESC"],
        max_results=5,
        output_format="pandas",
    )

    if runs.empty:
        raise SystemExit(f"No runs found in mlflow's {cfg['experiment_name']} experiment")

    report_path = os.path.join(resolve(params["reports"]["dir"]), "LEADERBOARD.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(render_leaderboard_md(params, runs))
    print(f"Leaderboard report saved to: {report_path}")


if __name__ == "__main__":
    main()
