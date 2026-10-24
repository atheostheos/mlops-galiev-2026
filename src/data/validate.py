import json
import os

import pandas as pd

from src.config import TARGET, load_params, resolve
from src.logging_setup import setup_logging

CHECKS = {
    "rows not less than a set minimum": lambda df, p: len(df) >= p["min_rows"],
    "missing share within limits": lambda df, p: df.isna().mean().max() <= p["max_missing_share"],
    "churn rate within reasonable bounds": lambda df, p: (
        p["target_rate"][0] < df[TARGET].mean() < p["target_rate"][1]
    ),
    "no duplicate customers": lambda df, p: not df["customer_id"].duplicated().any(),
    "tenure within acceptable range": lambda df, p: df["tenure_months"].between(0, 200).all(),
}

log = setup_logging()


def main():
    params = load_params()

    train_data_path = os.path.join(resolve(params["data"]["processed_dir"]), "train.csv")
    df = pd.read_csv(train_data_path)

    check_results = {
        check_name: check_func(df, params["validate"]) for check_name, check_func in CHECKS.items()
    }

    report_path = os.path.join(resolve(params["reports"]["dir"]), "validation.json")
    with open(report_path, "w", encoding="utf-8") as f:
        results_str = {k: int(v) for k, v in check_results.items()}
        json.dump(results_str, f, indent=2)
        f.write("\n")

    for check_name, result in check_results.items():
        log.info(f"{check_name}: {'PASSED' if result else 'FAILED'}")

    log.info(f"Validation report saved to: {report_path}")

    if not all(check_results.values()):
        raise ValueError("Data validation failed, check report")


if __name__ == "__main__":
    main()
