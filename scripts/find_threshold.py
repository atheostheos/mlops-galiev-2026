import os

import joblib
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.config import TARGET, feature_columns, load_params, resolve


def main() -> None:
    params = load_params()
    pipe = joblib.load(os.path.join(resolve(params["train"]["save_dir"]), "model.joblib"))
    val = pd.read_csv(os.path.join(resolve(params["data"]["processed_dir"]), "val.csv"))

    proba = pipe.predict_proba(val[feature_columns(params)])[:, 1]
    prec, rec, thr = precision_recall_curve(val[TARGET], proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    i = f1[:-1].argmax()
    print(f"лучший порог: {thr[i]:.3f}  (F1={f1[i]:.3f}, precision={prec[i]:.3f}, recall={rec[i]:.3f})")


if __name__ == "__main__":
    main()
