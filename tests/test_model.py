from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import TARGET
from src.data.prepare import clean
from src.train import build_pipeline


def test_more_support_calls_increase_churn_risk(trained_pipeline, raw_df, cols):
    """
    Направленный тест: больше обращений в поддержку -> выше риск ухода.
    Нарушение означает перепутанные признаки или таргет.
    """
    base = clean(raw_df.copy())[cols].head(200).copy()
    calm, angry = base.copy(), base.copy()
    calm.loc[:, "num_support_calls"] = 0
    angry.loc[:, "num_support_calls"] = 8
    assert (
        trained_pipeline.predict_proba(angry)[:, 1].mean() > trained_pipeline.predict_proba(calm)[:, 1].mean()
    )


def test_probabilities_are_valid(trained_pipeline, raw_df, cols):
    proba = trained_pipeline.predict_proba(clean(raw_df.copy()).tail(400)[cols])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    assert proba.std() > 0.01, "модель выдаёт почти константу — признак сломанного обучения"


def test_prediction_is_row_independent(trained_pipeline, raw_df, cols):
    """
    Предсказание для строки не зависит от соседей в батче.
    Иначе /predict и /predict/batch дадут разные ответы для одного клиента.
    """
    X = clean(raw_df.copy()).tail(400)[cols]
    one = trained_pipeline.predict_proba(X.head(1))[:, 1][0]
    batch = trained_pipeline.predict_proba(X.head(50))[:, 1][0]
    assert abs(one - batch) < 1e-9


def test_prediction_accepts_single_row_dataframe(trained_pipeline, raw_df, cols):
    """Catches batch-shape assumptions that can break single-customer requests."""
    one = clean(raw_df.copy()).tail(1)[cols]
    proba = trained_pipeline.predict_proba(one)
    assert proba.shape == (1, 2)
    assert np.isfinite(proba).all()


def test_longer_contract_reduces_churn_risk(trained_pipeline, raw_df, cols):
    base = clean(raw_df.copy())[cols].head(200).copy()
    monthly, annual = base.copy(), base.copy()
    monthly.loc[:, "contract_type"] = "month-to-month"
    annual.loc[:, "contract_type"] = "two_year"
    assert (
        trained_pipeline.predict_proba(annual)[:, 1].mean()
        < trained_pipeline.predict_proba(monthly)[:, 1].mean()
    )


def test_model_quality_exceeds_random(raw_df, params, cols):
    """A low floor catches broken labels/features while tolerating library drift."""
    df = clean(raw_df.copy())
    train, test = train_test_split(
        df,
        test_size=params["data"]["test_size"],
        random_state=params["seed"],
        stratify=df[TARGET],
    )
    pipe = build_pipeline(params)
    pipe.fit(train[cols], train[TARGET])
    assert roc_auc_score(test[TARGET], pipe.predict_proba(test[cols])[:, 1]) > 0.70


def test_retraining_is_reproducible(raw_df, params, cols):
    df = clean(raw_df.copy())
    a, b = build_pipeline(params), build_pipeline(params)
    a.fit(df[cols], df[TARGET])
    b.fit(df[cols], df[TARGET])
    np.testing.assert_allclose(a.predict_proba(df[cols].head(50)), b.predict_proba(df[cols].head(50)))


def test_target_and_customer_id_are_not_features(params, cols):
    assert TARGET not in cols
    assert "customer_id" not in cols


def test_duplicate_row_does_not_change_prediction(trained_pipeline, raw_df, cols):
    """
    Один и тот же клиент, поданный дважды, получает одинаковый ответ.
    Нарушение означает, что модель смотрит на позицию строки в батче.
    """
    row = clean(raw_df.copy())[cols].head(1)
    doubled = pd.concat([row, row], ignore_index=True)
    proba = trained_pipeline.predict_proba(doubled)[:, 1]
    assert abs(proba[0] - proba[1]) < 1e-12


@pytest.mark.slow
def test_batch_inference_is_fast_enough(trained_pipeline, raw_df, cols):
    """
    1000 объектов должны считаться быстрее секунды.
    Порог с запасом: ловим катастрофу (кто-то вызвал fit в predict),
    а не колебания на разных машинах.
    """
    batch = clean(raw_df.copy())[cols].sample(1000, replace=True, random_state=0)
    start = time.perf_counter()
    trained_pipeline.predict_proba(batch)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"инференс 1000 строк занял {elapsed:.2f} с"


def test_technical_support_lowers_churn_risk(trained_pipeline, raw_df, cols):
    """Catches a reversed support feature or a broken feature-to-column mapping."""
    without_support = clean(raw_df.copy())[cols].head(200).copy()
    with_support = without_support.copy()
    without_support.loc[:, "has_tech_support"] = 0
    with_support.loc[:, "has_tech_support"] = 1
    assert (
        trained_pipeline.predict_proba(with_support)[:, 1].mean()
        < trained_pipeline.predict_proba(without_support)[:, 1].mean()
    )
