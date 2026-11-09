from __future__ import annotations

import copy

import pandas as pd

import src.data.prepare as prepare_module
from src.config import TARGET, feature_columns
from src.data.generate import generate
from src.data.prepare import clean


def _configure_prepare(raw_df, params, tmp_path, monkeypatch):
    raw_path = tmp_path / "raw.csv"
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    raw_df.to_csv(raw_path, index=False)

    test_params = copy.deepcopy(params)
    test_params["data"]["raw_path"] = "raw.csv"
    test_params["data"]["processed_dir"] = "processed"
    test_params["reports"]["dir"] = "reports"
    resolved_paths = {"raw.csv": raw_path, "processed": processed_dir, "reports": reports_dir}
    monkeypatch.setattr(prepare_module, "load_params", lambda: test_params)
    monkeypatch.setattr(prepare_module, "resolve", lambda path: resolved_paths[path])
    return processed_dir


def test_schema_is_complete(raw_df, params):
    """
    Все колонки из params.yaml есть в данных.
    """
    missing = set(feature_columns(params) + [TARGET]) - set(raw_df.columns)
    assert not missing, f"в датасете нет колонок: {missing}"


def test_target_is_not_degenerate(raw_df):
    """
    Защита от 'все нули': такой датасет даст accuracy 0.9 и бесполезную модель.
    """
    rate = raw_df[TARGET].mean()
    assert 0.05 < rate < 0.60, f"подозрительный churn rate: {rate:.3f}"


def test_generation_is_reproducible():
    a, b = generate(n=500, seed=123), generate(n=500, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_feature_values_are_in_expected_ranges(raw_df):
    assert raw_df["tenure_months"].between(1, 72).all()
    assert raw_df["monthly_charges"].between(15, 180).all()
    assert raw_df["num_support_calls"].ge(0).all()


def test_clean_fills_missing_values(raw_df):
    cleaned = clean(raw_df.copy())
    assert cleaned["total_charges"].notna().all()


def test_clean_removes_duplicate_customers(raw_df):
    duplicated = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)
    cleaned = clean(duplicated)
    assert not cleaned["customer_id"].duplicated().any()


def test_split_is_deterministic(raw_df, params, tmp_path, monkeypatch):
    processed_dir = _configure_prepare(raw_df, params, tmp_path, monkeypatch)
    names = ("train", "val", "test")

    prepare_module.main()
    first = {name: pd.read_csv(processed_dir / f"{name}.csv") for name in names}
    prepare_module.main()
    second = {name: pd.read_csv(processed_dir / f"{name}.csv") for name in names}

    for name in names:
        pd.testing.assert_frame_equal(first[name], second[name])


def test_prepare_splits_preserve_customer_labels(raw_df, params, tmp_path, monkeypatch):
    """Catches target corruption in prepare.main before the split is saved."""
    processed_dir = _configure_prepare(raw_df, params, tmp_path, monkeypatch)

    prepare_module.main()

    actual = (
        pd.concat(
            [pd.read_csv(processed_dir / f"{name}.csv") for name in ("train", "val", "test")],
            ignore_index=True,
        )
        .set_index("customer_id")[TARGET]
        .sort_index()
    )
    expected = clean(raw_df.copy()).set_index("customer_id")[TARGET].sort_index()
    pd.testing.assert_series_equal(actual, expected)
