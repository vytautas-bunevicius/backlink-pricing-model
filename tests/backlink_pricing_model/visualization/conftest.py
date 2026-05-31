"""Shared fixtures for visualization tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def eda_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 60
    final_price = rng.uniform(50, 2000, n)
    # Inject some missing countries without feeding None into rng.choice.
    country = np.array(rng.choice(["US", "GB", "DE"], n), dtype=object)
    country[::5] = None
    return pd.DataFrame({
        "final_price": final_price,
        "log_price": np.log1p(final_price),
        "dr": rng.integers(0, 100, n).astype(float),
        "cf": rng.integers(0, 100, n).astype(float),
        "tf": rng.integers(0, 100, n).astype(float),
        "domain_traffic": rng.integers(100, 500_000, n).astype(float),
        "country": country,
        "tld": rng.choice(["com", "io", "co.uk", "org"], n),
    })


@pytest.fixture
def corr_matrix() -> pd.DataFrame:
    cols = ["dr", "cf", "tf"]
    data = np.array([
        [1.0, 0.9, 0.2],
        [0.9, 1.0, 0.1],
        [0.2, 0.1, 1.0],
    ])
    return pd.DataFrame(data, index=cols, columns=cols)
