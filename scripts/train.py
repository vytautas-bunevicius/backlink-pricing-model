"""CLI script: train the backlink pricing model.

Usage:
    python -m scripts.train
    python -m scripts.train --config configs/training.yaml
    python -m scripts.train --config configs/experiment_v2.yaml --trials 50
"""

import json
import logging
import warnings
from datetime import UTC, datetime

import click
import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from backlink_pricing_model.core.config import load_config, resolve_path
from backlink_pricing_model.preprocessing.data_imputation import (
    apply_domain_metric_imputer,
    fit_domain_metric_imputer,
)
from backlink_pricing_model.preprocessing.data_loading import (
    domain_grouped_split,
    save_processed,
)
from backlink_pricing_model.preprocessing.feature_engineering import (
    add_missingness_flags,
    add_temporal_features,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow.*")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAPE safely by skipping zero-denominator targets."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    mask = y_true_arr != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": safe_mape(y_true, y_pred),
    }


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive non-leaky features before split."""
    df = add_temporal_features(df)
    return add_missingness_flags(df)


def fit_label_encoders(
    train_df: pd.DataFrame,
    categorical_columns: list[str],
) -> dict[str, LabelEncoder]:
    """Fit encoders using train split only (prevents leakage)."""
    encoders: dict[str, LabelEncoder] = {}
    for col in categorical_columns:
        if col not in train_df.columns:
            continue
        le = LabelEncoder()
        values = train_df[col].fillna("unknown").astype(str)
        # Force unknown class to exist so inference can map unseen labels safely.
        fit_values = pd.concat([values, pd.Series(["unknown"])], ignore_index=True)
        le.fit(fit_values)
        encoders[col] = le
        logger.info("Encoded '%s': %d classes", col, len(le.classes_))
    return encoders


def apply_label_encoders(
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder],
) -> pd.DataFrame:
    """Apply train-fitted label encoders with unseen->unknown mapping."""
    result = df.copy()
    for col, le in encoders.items():
        if col not in result.columns:
            continue
        values = result[col].fillna("unknown").astype(str)
        known = set(le.classes_)
        values = values.where(values.isin(known), "unknown")
        result[f"{col}_encoded"] = le.transform(values)
    return result


def create_xgb_objective(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    search_space: dict,
    random_state: int,
    early_stopping_rounds: int,
) -> callable:
    """Create an Optuna objective function for XGBoost."""

    def _fit_with_early_stopping(model: XGBRegressor) -> None:
        """Fit with backwards/forwards-compatible early stopping handling."""
        try:
            # XGBoost <=2.x supports early_stopping_rounds in fit().
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                verbose=False,
                early_stopping_rounds=early_stopping_rounds,
            )
        except TypeError:
            # XGBoost >=3.x moved this to constructor/params.
            model.set_params(early_stopping_rounds=early_stopping_rounds)
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                verbose=False,
            )

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
            "eval_metric": "rmse",
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": -1,
        }
        model = XGBRegressor(**params)
        _fit_with_early_stopping(model)
        pred = model.predict(x_val)
        return float(root_mean_squared_error(y_val, pred))

    return objective


def setup_mlflow(cfg: dict) -> None:
    """Configure MLflow tracking from config."""
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
    target_is_log = cfg.get("target", "final_price") == "log_price"
    early_stopping_rounds = int(cfg.get("early_stopping_rounds", 50))

    # Setup MLflow.
    setup_mlflow(cfg)

    # Load cleaned data.
    processed_path = resolve_path(cfg["processed_data"])
    logger.info("Loading processed data from %s", processed_path)
    df = pd.read_parquet(processed_path, engine="pyarrow")
    logger.info("Loaded %d rows", len(df))

    # Feature engineering.
    df = prepare_features(df)

    # Build modeling frame (keep domain for grouped split).
    feature_cols = cfg["feature_columns"]
    target = cfg["target"]
    raw_cat_cols = cfg.get("categorical_columns", [])

    required_for_split = [*raw_cat_cols, *feature_cols, target, "domain"]
    available = [c for c in required_for_split if c in df.columns]
    df_model = df[available].dropna(subset=[target])
    logger.info("Modeling dataset: %d rows", len(df_model))

    # Domain-grouped 3-way split: train / val (for HPO) / test (held out).
    train_split, val_split, test_split = domain_grouped_split(
        df_model,
        test_size=cfg.get("test_size", 0.2),
        val_size=cfg.get("val_size", 0.1),
        random_state=random_state,
        domain_col="domain",
    )

    # Fit metric imputer on train split only, then apply everywhere.
    metric_imputer = fit_domain_metric_imputer(train_split)
    train_split = apply_domain_metric_imputer(train_split, metric_imputer)
    val_split = apply_domain_metric_imputer(val_split, metric_imputer)
    test_split = apply_domain_metric_imputer(test_split, metric_imputer)

    # Fit encoders on train split only, then apply everywhere.
    encoders = fit_label_encoders(train_split, raw_cat_cols)
    train_split = apply_label_encoders(train_split, encoders)
    val_split = apply_label_encoders(val_split, encoders)
    test_split = apply_label_encoders(test_split, encoders)

    # Final feature matrix.
    train_split = train_split.dropna(subset=[*feature_cols, target])
    val_split = val_split.dropna(subset=[*feature_cols, target])
    test_split = test_split.dropna(subset=[*feature_cols, target])

    x_train = train_split[feature_cols]
    y_train = train_split[target]
    x_val = val_split[feature_cols]
    y_val = val_split[target]
    x_test = test_split[feature_cols]
    y_test = test_split[target]
    logger.info(
        "Train: %d | Val: %d | Test: %d", len(x_train), len(x_val), len(x_test)
    )

    # Save splits for later evaluation.
    save_processed(
        pd.concat([x_train, y_train], axis=1), "train_df", subdir="engineered"
    )
    save_processed(
        pd.concat([x_val, y_val], axis=1), "val_df", subdir="engineered"
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
        x_val,
        y_val,
        cfg["xgb_search_space"],
        random_state,
        early_stopping_rounds,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("Best RMSE (val): %.4f", study.best_value)
    logger.info("Best params: %s", study.best_params)

    # Retrain final model on train+val only (test remains untouched).
    x_train_full = pd.concat([x_train, x_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    best_model = XGBRegressor(
        **study.best_params, random_state=random_state, n_jobs=-1
    )
    best_model.fit(x_train_full, y_train_full, verbose=False)

    # Evaluate on held-out test set (never seen during training or HPO).
    y_pred = best_model.predict(x_test)
    metrics = {"log_scale": compute_metrics(y_test.values, y_pred)}

    if target_is_log:
        y_test_usd = np.expm1(y_test.values)
        y_pred_usd = np.expm1(y_pred)
        metrics["usd_scale"] = compute_metrics(y_test_usd, y_pred_usd)

    logger.info("Evaluation results on test set:")
    for metric_group, values in metrics.items():
        logger.info("  %s", metric_group)
        for name, value in values.items():
            logger.info("    %s: %.4f", name.upper(), value)

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

    imputer_path = model_dir / "metric_imputer.joblib"
    joblib.dump(metric_imputer, imputer_path)
    logger.info("Saved metric imputer to %s", imputer_path)

    # Save metrics + metadata.
    metadata = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "config": config_path,
        "n_trials": n_trials,
        "best_params": study.best_params,
        "metrics": metrics,
        "train_rows": len(x_train),
        "val_rows": len(x_val),
        "test_rows": len(x_test),
        "features": feature_cols,
    }
    metadata_path = model_dir / "training_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2, default=str)

    metrics_path = model_dir / cfg.get(
        "metrics_filename", "metrics_summary.csv"
    )
    pd.DataFrame([
        {
            "split": "test",
            **{f"log_{k}": v for k, v in metrics["log_scale"].items()},
            **({f"usd_{k}": v for k, v in metrics.get("usd_scale", {}).items()}),
        }
    ]).to_csv(metrics_path, index=False)

    # Log everything to MLflow.
    with mlflow.start_run(run_name=f"xgb_{n_trials}trials"):
        mlflow.log_params(study.best_params)
        mlflow.log_param("n_optuna_trials", n_trials)
        mlflow.log_param("split_method", "domain_grouped")
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_rows", len(x_train))
        mlflow.log_param("val_rows", len(x_val))
        mlflow.log_param("test_rows", len(x_test))
        mlflow.log_param("early_stopping_rounds", early_stopping_rounds)
        mlflow.log_param("final_fit", "train_plus_val")

        mlflow.log_metrics({f"log_{k}": v for k, v in metrics["log_scale"].items()})
        if "usd_scale" in metrics:
            mlflow.log_metrics({f"usd_{k}": v for k, v in metrics["usd_scale"].items()})

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(encoders_path))
        mlflow.log_artifact(str(imputer_path))
        mlflow.log_artifact(str(metadata_path))
        mlflow.log_artifact(config_path)

        mlflow.xgboost.log_model(best_model, artifact_path="model")

        logger.info("Logged run to MLflow: %s", mlflow.active_run().info.run_id)

    logger.info("Training complete")


if __name__ == "__main__":
    main()
