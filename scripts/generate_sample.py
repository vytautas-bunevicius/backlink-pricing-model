"""Generate a small synthetic backlink dataset for public reproduction.

The real dataset lives in a private Supabase table. This script produces a
deterministic Parquet file with the same schema and plausible distributions so
that a reviewer can clone the repo and run `make preprocess && make train`
end-to-end without any credentials.

The synthetic prices are generated from a known log-linear function of the
quality metrics plus heteroskedastic noise. A model trained on this sample
should recover a reasonable R^2 but the absolute numbers are not meaningful —
they exist only to demonstrate that the pipeline runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COUNTRIES = [
    "US", "GB", "DE", "FR", "ES", "IT", "NL", "PL", "BR", "IN",
    "CA", "AU", "SE", "NO", "FI", "DK", "JP", "KR", "MX", "AR",
]
TLDS = ["com", "co.uk", "io", "de", "net", "org", "co", "es", "fr", "nl"]


def _domain(rng: np.random.Generator, idx: int) -> str:
    """Return a plausible domain name with a random TLD."""
    tld = rng.choice(TLDS)
    return f"site{idx:06d}.{tld}"


def generate(n_rows: int, seed: int) -> pd.DataFrame:
    """Generate ``n_rows`` synthetic backlink placement records."""
    rng = np.random.default_rng(seed)

    dr = np.clip(rng.beta(2.2, 3.0, size=n_rows) * 100, 1, 100)
    tf = np.clip(dr + rng.normal(0, 8, size=n_rows), 1, 100)
    cf = np.clip(tf + rng.normal(2, 6, size=n_rows), 1, 100)
    traffic = np.exp(rng.normal(8.0 + 0.04 * dr, 1.2, size=n_rows))

    # log-linear price model with heteroskedastic noise
    log_price = (
        2.5
        + 0.035 * dr
        + 0.012 * tf
        + 0.18 * np.log1p(traffic / 1000.0)
        + rng.normal(0, 0.45, size=n_rows)
    )
    final_price = np.round(np.exp(log_price), 2)

    # ~3% missing in cf, ~6% missing in tf, ~10% missing in country
    cf = np.where(rng.random(n_rows) < 0.03, np.nan, cf)
    tf = np.where(rng.random(n_rows) < 0.06, np.nan, tf)
    country_mask = rng.random(n_rows) < 0.10
    countries = np.where(
        country_mask, None, rng.choice(COUNTRIES, size=n_rows)
    )

    dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 730, size=n_rows), unit="D"
    )

    return pd.DataFrame({
        "domain": [_domain(rng, i) for i in range(n_rows)],
        "final_price": final_price,
        "dr": dr.round(1),
        "tf": np.where(np.isnan(tf), np.nan, np.round(tf, 1)),
        "cf": np.where(np.isnan(cf), np.nan, np.round(cf, 1)),
        "domain_traffic": traffic.round(0),
        "country": countries,
        "date_received": dates.date,
    })


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/sample.parquet"),
    )
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
