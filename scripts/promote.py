import argparse
import os
import sys

import mlflow
import pandas as pd
from mlflow import MlflowClient

from src.config import load_params, resolve
from src.evaluate import compute_metrics
from src.logging_setup import setup_logging
from src.service.model_loader import ModelHolder

log = setup_logging()


CHAMPION_ALIAS = None  # fetched from params


def score(params, model_uri: str, test: pd.DataFrame) -> float:
    pipe = mlflow.sklearn.load_model(model_uri)
    metrics, _, _ = compute_metrics(params, pipe, test)
    return metrics["roc_auc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, help="candidate model's version to check for promotion")
    args = parser.parse_args()

    params = load_params()
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    name = params["mlflow"]["registered_model_name"]
    min_gain = params["promote"]["min_gain"]

    CHAMPION_ALIAS = params["service"]["model_alias"]

    client = MlflowClient()
    champion = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
    candidate = client.get_model_version(name, args.candidate)

    if candidate.version == champion.version:
        log.error("Candidate version is production model already")
        sys.exit(0)

    candidate_uri = f"models:/{name}/{candidate.version}"
    try:
        ModelHolder().check_signature(candidate_uri)
    except RuntimeError as exc:
        log.error(f"Candidate v{candidate.version} didn't pass signature: {exc}")
        sys.exit(1)

    test_data_path = os.path.join(resolve(params["data"]["processed_dir"]), "test.csv")
    test = pd.read_csv(test_data_path)

    champ_auc = score(params, f"models:/{name}/{champion.version}", test)
    cand_auc = score(params, f"models:/{name}/{candidate.version}", test)
    gain = cand_auc - champ_auc
    log.info(f"production v{champion.version}: {champ_auc:.4f} \n")
    log.info(f"candidate v{candidate.version}: {cand_auc:.4f}, gain: {gain:.4f}")

    if gain < min_gain:
        log.warning(
            f"Rejected promotion, change of target metric is lower that min gain:  {gain:.4f} < {min_gain:.4f}"
        )
        sys.exit(1)

    client.transition_model_version_stage(
        name, candidate.version, "Production", archive_existing_versions=True
    )
    client.set_registered_model_alias(name, CHAMPION_ALIAS, candidate.version)
    log.info("Candidate promoted to production")


if __name__ == "__main__":
    main()
