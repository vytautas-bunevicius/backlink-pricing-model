"""CLI script: run inference on new domains.

Usage:
    python -m scripts.predict --input data/raw/new_domains.csv
    python -m scripts.predict --input data/raw/new_domains.csv --output predictions.csv
"""

import logging

import click
import joblib
import numpy as np
import pandas as pd

from backlink_pricing_model.core.config import load_config, resolve_path
from backlink_pricing_model.preprocessing.data_imputation import (
    apply_domain_metric_imputer,
    impute_metrics_by_domain,
)
from backlink_pricing_model.preprocessing.feature_engineering import (
    add_log_traffic,
    add_temporal_features,
    add_tld_feature,
    normalize_country,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def prepare_input(
    df: pd.DataFrame,
    encoders: dict,
    metric_imputer: dict | None = None,
) -> pd.DataFrame:
    """Apply the same feature pipeline used during training.

    Args:
        df: Raw input DataFrame with domain, dr, cf, tf, etc.
        encoders: Label encoders from training.
        metric_imputer: Optional train-fitted metric imputer artifact.

    Returns:
        Feature-engineered DataFrame ready for prediction.
    """
    if metric_imputer:
        df = apply_domain_metric_imputer(df, metric_imputer)
    else:
        # Backward-compatibility when artifact is missing.
        df = impute_metrics_by_domain(df)
    df = normalize_country(df)
    df = add_tld_feature(df)
    df = add_log_traffic(df)
    df = add_temporal_features(df)

    # Encode categoricals using training encoders.
    for col, le in encoders.items():
        known_classes = set(le.classes_)
        col_values = df[col].fillna("unknown").astype(str)
        # Map unseen labels to "unknown" (class is guaranteed by training).
        col_values = col_values.apply(
            lambda v, kc=known_classes: v if v in kc else "unknown"
        )
        df[f"{col}_encoded"] = le.transform(col_values)

    return df


@click.command()
@click.option(
    "--config",
    "config_path",
    default="configs/training.yaml",
    help="Path to training config YAML.",
)
@click.option(
    "--input",
    "input_path",
    required=True,
    help="Path to input CSV or Parquet file.",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    help="Path to save predictions. Defaults to <input>_predictions.csv.",
)
@click.option(
    "--model",
    "model_path",
    default=None,
    help="Path to model .joblib file.",
)
def main(
    config_path: str,
    input_path: str,
    output_path: str | None,
    model_path: str | None,
) -> None:
    """Predict backlink prices for new domains."""
    cfg = load_config(config_path)
    logger.info("Loaded config: %s", config_path)

    # Resolve model path.
    if model_path is None:
        model_dir = cfg.get("model_dir", "models")
        model_filename = cfg.get("model_filename", "xgb_backlink_pricing.joblib")
        model_path = str(resolve_path(f"{model_dir}/{model_filename}"))

    # Load model and encoders.
    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    encoders_path = resolve_path(f"{cfg.get('model_dir', 'models')}/label_encoders.joblib")
    logger.info("Loading encoders from %s", encoders_path)
    encoders = joblib.load(encoders_path)

    imputer_path = resolve_path(f"{cfg.get('model_dir', 'models')}/metric_imputer.joblib")
    metric_imputer = None
    if imputer_path.exists():
        logger.info("Loading metric imputer from %s", imputer_path)
        metric_imputer = joblib.load(imputer_path)
    else:
        logger.warning(
            "Metric imputer artifact not found at %s; "
            "falling back to per-batch domain imputation",
            imputer_path,
        )

    # Load input.
    input_file = resolve_path(input_path)
    logger.info("Loading input from %s", input_file)
    if str(input_file).endswith(".parquet"):
        df = pd.read_parquet(input_file, engine="pyarrow")
    else:
        df = pd.read_csv(input_file)
    logger.info("Loaded %d rows", len(df))

    # Prepare features.
    df = prepare_input(df, encoders, metric_imputer=metric_imputer)
    feature_cols = cfg["feature_columns"]

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.warning("Missing feature columns: %s", missing_cols)

    x = df[feature_cols]

    raw_pred = model.predict(x)
    target = cfg.get("target", "log_price")
    if target == "log_price":
        df["predicted_log_price"] = raw_pred
        df["predicted_price_usd"] = np.expm1(raw_pred)
    else:
        df["predicted_price_usd"] = raw_pred

    # Summary.
    logger.info(
        "Predictions: mean=$%.2f, median=$%.2f, min=$%.2f, max=$%.2f",
        df["predicted_price_usd"].mean(),
        df["predicted_price_usd"].median(),
        df["predicted_price_usd"].min(),
        df["predicted_price_usd"].max(),
    )

    # Save.
    if output_path is None:
        stem = input_file.stem
        output_path = str(input_file.parent / f"{stem}_predictions.csv")

    output_file = resolve_path(output_path)
    df.to_csv(output_file, index=False)
    logger.info("Saved predictions to %s", output_file)


if __name__ == "__main__":
    main()
