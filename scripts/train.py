"""CLI script: train the backlink pricing model.

Usage:
    python -m scripts.train
    python -m scripts.train --config configs/training.yaml
    python -m scripts.train --config configs/experiment_v2.yaml --trials 50
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import click
import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from backlink_pricing_model.core.config import load_config, resolve_path
from backlink_pricing_model.preprocessing.data_imputation import (
    impute_metrics_by_domain,
)
from backlink_pricing_model.preprocessing.data_loading import save_processed
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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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


def prepare_features(
    df: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Impute, engineer, and encode features.

    Args:
        df: Cleaned DataFrame from preprocessing.
        cfg: Training configuration.

    Returns:
        Tuple of (feature-engineered DataFrame, label encoders dict).
    """
    df = impute_metrics_by_domain(df)
    df = add_price_ratio(df)
    df = add_temporal_features(df)

    encoders: dict[str, LabelEncoder] = {}
    for col in cfg.get("categorical_columns", []):
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col].fillna("unknown"))
        encoders[col] = le
        logger.info("Encoded '%s': %d classes", col, len(le.classes_))

    return df, encoders


def create_xgb_objective(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    search_space: dict,
    random_state: int,
) -> callable:
    """Create an Optuna objective function for XGBoost.

    Args:
        x_train: Training features.
        y_train: Training target.
        x_val: Validation features.
        y_val: Validation target.
        search_space: Hyperparameter search ranges.
        random_state: Random seed.

    Returns:
        Optuna objective callable.
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", *search_space["n_estimators"]
            ),
            "max_depth": trial.suggest_int(
                "max_depth", *search_space["max_depth"]
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", *search_space["learning_rate"], log=True
            ),
            "subsample": trial.suggest_float(
                "subsample", *search_space["subsample"]
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *search_space["colsample_bytree"]
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", *search_space["reg_alpha"], log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", *search_space["reg_lambda"], log=True
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", *search_space["min_child_weight"]
            ),
            "random_state": random_state,
            "n_jobs": -1,
        }
        model = XGBRegressor(**params)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )
        pred = model.predict(x_val)
        return float(root_mean_squared_error(y_val, pred))

    return objective


def setup_mlflow(cfg: dict) -> None:
    """Configure MLflow tracking from config.

    Args:
        cfg: Training configuration dict.
    """
    tracking_uri = cfg.get("mlflow_tracking_uri", "mlruns")
    tracking_path = resolve_path(tracking_uri)
    tracking_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{tracking_path}")

    experiment_name = cfg.get("mlflow_experiment_name", "backlink-pricing")
    mlflow.set_experiment(experiment_name)
    logger.info(
        "MLflow tracking: %s (experiment: %s)", tracking_path, experiment_name
    )


@click.command()
@click.option(
    "--config",
    "config_path",
    default="configs/training.yaml",
    help="Path to training config YAML.",
)
@click.option(
    "--trials",
    default=None,
    type=int,
    help="Override number of Optuna trials.",
)
def main(config_path: str, trials: int | None) -> None:
    """Train the backlink pricing model end-to-end."""
    cfg = load_config(config_path)
    logger.info("Loaded config: %s", config_path)

    random_state = cfg.get("random_state", 42)
    n_trials = trials or cfg.get("n_optuna_trials", 100)

    # Setup MLflow.
    setup_mlflow(cfg)

    # Load cleaned data.
    processed_path = resolve_path(cfg["processed_data"])
    logger.info("Loading processed data from %s", processed_path)
    df = pd.read_parquet(processed_path, engine="pyarrow")
    logger.info("Loaded %d rows", len(df))

    # Feature engineering.
    df, encoders = prepare_features(df, cfg)

    # Build feature matrix.
    feature_cols = cfg["feature_columns"]
    target = cfg["target"]
    df_model = df.dropna(subset=[*feature_cols, target])
    logger.info(
        "Modeling dataset: %d rows, %d features",
        len(df_model),
        len(feature_cols),
    )

    x = df_model[feature_cols]
    y = df_model[target]

    # Train/test split.
    test_size = cfg.get("test_size", 0.2)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    logger.info("Train: %d | Test: %d", len(x_train), len(x_test))

    # Save splits for later evaluation.
    save_processed(
        pd.concat([x_train, y_train], axis=1), "train_df", subdir="engineered"
    )
    save_processed(
        pd.concat([x_test, y_test], axis=1), "test_df", subdir="engineered"
    )

    # Optuna study with persistent SQLite storage.
    storage = cfg.get("optuna_storage")
    if storage:
        storage_path = resolve_path(storage.replace("sqlite:///", ""))
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{storage_path}"

    study_name = cfg.get("optuna_study_name", "xgb_backlink_pricing")
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage,
        load_if_exists=True,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    objective = create_xgb_objective(
        x_train,
        y_train,
        x_test,
        y_test,
        cfg["xgb_search_space"],
        random_state,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("Best RMSE: %.4f", study.best_value)
    logger.info("Best params: %s", study.best_params)

    # Train final model with best params.
    best_model = XGBRegressor(
        **study.best_params, random_state=random_state, n_jobs=-1
    )
    best_model.fit(
        x_train, y_train, eval_set=[(x_test, y_test)], verbose=False
    )

    # Evaluate.
    y_pred = best_model.predict(x_test)
    metrics = compute_metrics(y_test.values, y_pred)
    for name, value in metrics.items():
        logger.info("  %s: %.4f", name.upper(), value)

    # Save artifacts.
    model_dir = resolve_path(cfg.get("model_dir", "models"))
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / cfg.get(
        "model_filename", "xgb_backlink_pricing.joblib"
    )
    joblib.dump(best_model, model_path)
    logger.info("Saved model to %s", model_path)

    encoders_path = model_dir / "label_encoders.joblib"
    joblib.dump(encoders, encoders_path)
    logger.info("Saved encoders to %s", encoders_path)

    # Save metrics + metadata.
    metadata = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "config": config_path,
        "n_trials": n_trials,
        "best_params": study.best_params,
        "metrics": metrics,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "features": feature_cols,
    }
    metadata_path = model_dir / "training_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2, default=str)

    metrics_path = model_dir / cfg.get(
        "metrics_filename", "metrics_summary.csv"
    )
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    # Log everything to MLflow.
    with mlflow.start_run(run_name=f"xgb_{n_trials}trials"):
        mlflow.log_params(study.best_params)
        mlflow.log_param("n_optuna_trials", n_trials)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_rows", len(x_train))
        mlflow.log_param("test_rows", len(x_test))

        mlflow.log_metrics(metrics)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(encoders_path))
        mlflow.log_artifact(str(metadata_path))
        mlflow.log_artifact(config_path)

        mlflow.xgboost.log_model(best_model, artifact_path="model")

        logger.info("Logged run to MLflow: %s", mlflow.active_run().info.run_id)

    logger.info("Training complete")


if __name__ == "__main__":
    main()
