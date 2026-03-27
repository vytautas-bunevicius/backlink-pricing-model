"""Shared test fixtures for the backlink_pricing_model test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_backlink_dataset() -> pd.DataFrame:
    """Create a sample backlink dataset for testing."""
    np.random.seed(42)
    n = 20
    return pd.DataFrame(
        {
            "final_price": np.random.uniform(50, 2000, n),
            "dr": np.random.randint(0, 100, n),
            "cf": np.random.randint(0, 100, n),
            "tf": np.random.randint(0, 100, n),
            "domain_traffic": np.random.randint(100, 500_000, n),
            "country": np.random.choice(
                ["US", "GB", "DE", "FR", "ES"], n
            ),
            "date_received": pd.date_range(
                "2024-01-01", periods=n, freq="W"
            ),
            "domain": [f"example{i}.com" for i in range(n)],
        }
    )
