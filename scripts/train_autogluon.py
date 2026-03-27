"""CLI script: train with AutoGluon TabularPredictor.

AutoGluon auto-stacks XGBoost, LightGBM, CatBoost, Random Forest, Neural Net,
and weighted ensembles. It handles categorical encoding natively and performs
internal hyperparameter optimization.

Usage:
    python -m scripts.train_autogluon
    python -m scripts.train_autogluon --config configs/training.yaml
    python -m scripts.train_autogluon --time-limit 600
    python -m scripts.train_autogluon --preset best_quality
"""

import json
import logging
from datetime import datetime, timezone

import click
import mlflow
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)

from backlink_pricing_model.core.config import load_config, resolve_path
from backlink_pricing_model.preprocessing.data_imputation import (
    impute_metrics_by_domain,
)
from backlink_pricing_model.preprocessing.data_loading import (
    domain_grouped_split,
    save_processed,
)
from backlink_pricing_model.preprocessing.auto_features import (
    apply_openfe,
    fit_openfe,
    save_feature_descriptions,
)
from backlink_pricing_model.preprocessing.feature_engineering import (
    add_price_ratio,
    add_temporal_features,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Compute regression metrics.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        Dict of metric name to value.
    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def prepare_autogluon_data(
    df: pd.DataFrame, ag_cfg: dict, target: str
) -> pd.DataFrame:
    """Prepare data for AutoGluon with raw categoricals.

    AutoGluon handles categorical encoding natively and often achieves
    better results with raw string categories than label-encoded integers.

    Args:
        df: Cleaned DataFrame.
        ag_cfg: AutoGluon config section.
        target: Target column name.

    Returns:
        DataFrame with feature columns + target, NaN rows dropped.
    """
    feature_cols = ag_cfg["feature_columns"]
    cols = [*feature_cols, target]
    available = [c for c in cols if c in df.columns]
    missing = set(cols) - set(available)
    if missing:
        logger.warning("Missing columns: %s", missing)

    return df[available].dropna()


def setup_mlflow(cfg: dict) -> None:
    """Configure MLflow tracking.

    Args:
        cfg: Training configuration dict.
    """
    tracking_uri = cfg.get("mlflow_tracking_uri", "mlruns")
    tracking_path = resolve_path(tracking_uri)
    tracking_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{tracking_path}")

    experiment_name = cfg.get("mlflow_experiment_name", "backlink-pricing")
    mlflow.set_experiment(experiment_name)


@click.command()
@click.option(
    "--config",
    "config_path",
    default="configs/training.yaml",
    help="Path to training config YAML.",
)
@click.option(
    "--time-limit",
    "time_limit_override",
    default=None,
    type=int,
    help="Override time_limit in seconds.",
)
@click.option(
    "--preset",
    "preset_override",
    default=None,
    type=str,
    help="Override preset (best_quality, high_quality, medium_quality).",
)
def main(
    config_path: str,
    time_limit_override: int | None,
    preset_override: str | None,
) -> None:
    """Train with AutoGluon TabularPredictor."""
    from autogluon.tabular import TabularPredictor

    cfg = load_config(config_path)
    ag_cfg = cfg.get("autogluon", {})
    logger.info("Loaded config: %s", config_path)

    target = cfg["target"]
    preset = preset_override or ag_cfg.get("preset", "best_quality")
    time_limit = time_limit_override or ag_cfg.get("time_limit", 3600)
    eval_metric = ag_cfg.get("eval_metric", "root_mean_squared_error")
    save_dir = str(resolve_path(ag_cfg.get("save_dir", "models/autogluon")))
    do_refit = ag_cfg.get("refit_full", True)

    # Setup MLflow.
    setup_mlflow(cfg)

    # Load and prepare data.
    processed_path = resolve_path(cfg["processed_data"])
    logger.info("Loading data from %s", processed_path)
    df = pd.read_parquet(processed_path, engine="pyarrow")
    logger.info("Loaded %d rows", len(df))

    # Feature engineering (same as manual pipeline).
    df = impute_metrics_by_domain(df)
    df = add_price_ratio(df)
    df = add_temporal_features(df)

    # Prepare AutoGluon data with raw categoricals (keep domain for split).
    feature_cols = ag_cfg["feature_columns"]
    keep_cols = [*feature_cols, target, "domain"]
    available = [c for c in keep_cols if c in df.columns]
    df_model = df[available].dropna(subset=[target])
    logger.info("Modeling dataset: %d rows", len(df_model))

    # Domain-grouped 3-way split.
    random_state = cfg.get("random_state", 42)
    train_df, val_df, test_df = domain_grouped_split(
        df_model,
        test_size=cfg.get("test_size", 0.2),
        val_size=cfg.get("val_size", 0.1),
        random_state=random_state,
        domain_col="domain",
    )

    # Drop domain column before modeling.
    for split in (train_df, val_df, test_df):
        if "domain" in split.columns:
            split.drop(columns=["domain"], inplace=True)

    logger.info(
        "Train: %d | Val: %d | Test: %d",
        len(train_df), len(val_df), len(test_df),
    )

    # OpenFE: discover interaction features on numeric columns.
    numeric_cols = [
        c for c in feature_cols
        if c not in ("tld", "country") and c in train_df.columns
    ]
    cat_cols = [c for c in ("tld", "country") if c in train_df.columns]
    openfe_top_k = ag_cfg.get("openfe_top_k", 15)

    if numeric_cols and openfe_top_k > 0:
        logger.info(
            "Running OpenFE on %d numeric features (top_k=%d)...",
            len(numeric_cols),
            openfe_top_k,
        )
        discovered = fit_openfe(
            train_df[numeric_cols],
            train_df[target],
            top_k=openfe_top_k,
            n_jobs=1,
            task="regression",
        )
        save_feature_descriptions(
            discovered,
            resolve_path(f"{cfg.get('model_dir', 'models')}/openfe_features.json"),
        )

        # Apply to all splits.
        train_numeric, val_numeric, test_numeric = apply_openfe(
            train_df[numeric_cols],
            test_df[numeric_cols],
            discovered,
            n_jobs=1,
            val_x=val_df[numeric_cols],
        )

        # Reassemble with categoricals and target.
        for col in cat_cols:
            train_numeric[col] = train_df[col].values
            val_numeric[col] = val_df[col].values
            test_numeric[col] = test_df[col].values

        train_numeric[target] = train_df[target].values
        val_numeric[target] = val_df[target].values
        test_numeric[target] = test_df[target].values

        train_df = train_numeric
        val_df = val_numeric
        test_df = test_numeric

        logger.info(
            "Features after OpenFE: %d", len(train_df.columns) - 1
        )

    # Save splits for reproducibility.
    save_processed(train_df, "train_df_autogluon", subdir="engineered")
    save_processed(val_df, "val_df_autogluon", subdir="engineered")
    save_processed(test_df, "test_df_autogluon", subdir="engineered")

    # Train AutoGluon.
    logger.info(
        "Starting AutoGluon: preset=%s, time_limit=%ds, metric=%s",
        preset,
        time_limit,
        eval_metric,
    )
    predictor = TabularPredictor(
        label=target,
        eval_metric=eval_metric,
        problem_type="regression",
        path=save_dir,
        verbosity=2,
    )
    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,
        time_limit=time_limit,
        presets=preset,
        use_bag_holdout=True,
    )

    # Leaderboard.
    leaderboard = predictor.leaderboard(test_df, silent=True)
    logger.info("Leaderboard:\n%s", leaderboard.to_string())

    # Best model evaluation.
    y_test = test_df[target]
    y_pred = predictor.predict(test_df)
    metrics = compute_metrics(y_test, y_pred)
    best_model = predictor.model_best
    logger.info("Best model: %s", best_model)
    for name, value in metrics.items():
        logger.info("  %s: %.4f", name.upper(), value)

    # Refit on full data for production.
    if do_refit:
        logger.info("Refitting best model on 100%% of data...")
        refit_map = predictor.refit_full()
        refit_model = refit_map.get(best_model, best_model)
        logger.info("Refit model: %s", refit_model)

    # Feature importance.
    try:
        importance = predictor.feature_importance(test_df, silent=True)
        logger.info("Feature importance:\n%s", importance.to_string())
    except Exception:
        logger.warning("Could not compute feature importance")
        importance = None

    # Save metadata.
    model_dir = resolve_path(cfg.get("model_dir", "models"))
    model_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "config": config_path,
        "method": "autogluon",
        "preset": preset,
        "time_limit": time_limit,
        "eval_metric": eval_metric,
        "best_model": best_model,
        "metrics": metrics,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "features": ag_cfg["feature_columns"],
        "leaderboard_top5": leaderboard.head().to_dict(orient="records"),
    }
    metadata_path = model_dir / "autogluon_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Save leaderboard CSV.
    leaderboard_path = model_dir / "autogluon_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    # Log to MLflow.
    with mlflow.start_run(run_name=f"autogluon_{preset}_{time_limit}s"):
        mlflow.log_param("method", "autogluon")
        mlflow.log_param("preset", preset)
        mlflow.log_param("time_limit", time_limit)
        mlflow.log_param("eval_metric", eval_metric)
        mlflow.log_param("best_model", best_model)
        mlflow.log_param("n_features", len(ag_cfg["feature_columns"]))
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))

        mlflow.log_metrics(metrics)

        mlflow.log_artifact(str(metadata_path))
        mlflow.log_artifact(str(leaderboard_path))
        mlflow.log_artifact(config_path)

        logger.info(
            "Logged run to MLflow: %s", mlflow.active_run().info.run_id
        )

    logger.info("AutoGluon training complete")
    logger.info("Model saved to: %s", save_dir)
    logger.info(
        "Load with: TabularPredictor.load('%s')", save_dir
    )


if __name__ == "__main__":
    main()
