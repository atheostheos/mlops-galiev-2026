from __future__ import annotations

import pytest

from src.config import TARGET, feature_columns, load_params
from src.data.generate import generate
from src.data.prepare import clean
from src.train import build_pipeline


@pytest.fixture(scope="session")
def params():
    return load_params()


@pytest.fixture(scope="session")
def cols(params):
    return feature_columns(params)


@pytest.fixture(scope="session")
def raw_df():
    return generate(n=2000, seed=7)


@pytest.fixture(scope="session")
def trained_pipeline(params, raw_df, cols):
    """Train once per test session; tests never depend on saved pipeline artifacts."""
    df = clean(raw_df.copy())
    pipe = build_pipeline(params)
    pipe.fit(df[cols], df[TARGET])
    return pipe
