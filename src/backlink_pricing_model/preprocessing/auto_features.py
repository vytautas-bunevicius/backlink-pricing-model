"""Automated feature generation using OpenFE.

OpenFE discovers interaction features (multiplications, divisions, logs,
aggregations) and evaluates each candidate's incremental lift using a
boosting-based method. Only features that improve held-out performance
are retained.

Usage:
    discovered = fit_openfe(train_x, train_y)
    train_x, test_x = apply_openfe(train_x, test_x, discovered)
"""

import json
import logging
import multiprocessing
from pathlib import Path

import pandas as pd
from sklearn.metrics import root_mean_squared_error

# Patch OpenFE for sklearn >=1.7 where mean_squared_error(squared=False)
# was removed. OpenFE internally calls this deprecated API.
import openfe.openfe as _openfe_module

_openfe_module.mean_squared_error = lambda y, p, **_kw: root_mean_squared_error(y, p)

from openfe import OpenFE, transform

# macOS uses 'spawn' by default which breaks OpenFE's multiprocessing.
if multiprocessing.get_start_method(allow_none=True) != "fork":
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass

logger = logging.getLogger(__name__)


def fit_openfe(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    top_k: int = 15,
    n_jobs: int = 1,
    task: str = "regression",
) -> list:
    """Discover high-value interaction features with OpenFE.

    Fits OpenFE on training data only. The returned feature list can be
    applied to validation and test sets via apply_openfe().

    Args:
        train_x: Training features (numeric columns only).
        train_y: Training target.
        top_k: Number of top features to keep from the ranked list.
        n_jobs: Parallelism.
        task: 'regression' or 'classification'.

    Returns:
        List of top_k OpenFE feature objects.
    """
    ofe = OpenFE()
    all_features = ofe.fit(
        data=train_x,
        label=train_y,
        task=task,
        n_jobs=n_jobs,
        verbose=True,
    )

    # OpenFE returns all features ranked by importance; keep top_k.
    features = all_features[:top_k]
    logger.info(
        "OpenFE discovered %d features (kept top %d):",
        len(all_features),
        len(features),
    )
    for i, f in enumerate(features):
        logger.info("  [%d] %s", i + 1, f)
    return features


def apply_openfe(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    features: list,
    n_jobs: int = 1,
    val_x: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Apply discovered OpenFE features to data splits.

    Args:
        train_x: Training features.
        test_x: Test features.
        features: OpenFE feature objects from fit_openfe().
        n_jobs: Parallelism.
        val_x: Optional validation features.

    Returns:
        Tuple of transformed DataFrames (train, test) or (train, val, test).
    """
    if val_x is not None:
        train_transformed, val_transformed = transform(
            train_x, val_x, features, n_jobs=n_jobs
        )
        _, test_transformed = transform(
            train_x, test_x, features, n_jobs=n_jobs
        )
        n_new = len(train_transformed.columns) - len(train_x.columns)
        logger.info(
            "OpenFE added %d features: %d -> %d columns",
            n_new,
            len(train_x.columns),
            len(train_transformed.columns),
        )
        return train_transformed, val_transformed, test_transformed

    train_transformed, test_transformed = transform(
        train_x, test_x, features, n_jobs=n_jobs
    )
    n_new = len(train_transformed.columns) - len(train_x.columns)
    logger.info(
        "OpenFE added %d features: %d -> %d columns",
        n_new,
        len(train_x.columns),
        len(train_transformed.columns),
    )
    return train_transformed, test_transformed


def save_feature_descriptions(
    features: list, output_path: str | Path
) -> None:
    """Save discovered feature descriptions to JSON for reproducibility.

    Args:
        features: OpenFE feature objects.
        output_path: Path to write JSON file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptions = [str(f) for f in features]
    with path.open("w") as f:
        json.dump(descriptions, f, indent=2)
    logger.info("Saved %d feature descriptions to %s", len(descriptions), path)
