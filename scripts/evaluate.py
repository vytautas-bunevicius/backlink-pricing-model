"""CLI script: evaluate a trained model on the test set.

Usage:
    python -m scripts.evaluate
    python -m scripts.evaluate --model models/xgb_backlink_pricing.joblib
    python -m scripts.evaluate --test-data data/engineered/test_df.parquet
"""

import json
import logging

import click
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from backlink_pricing_model.core.config import load_config, resolve_path
from backlink_pricing_model.visualization.models_plots import (
    plot_predictions_vs_actuals,
    plot_residuals,
)
from backlink_pricing_model.visualization.importance_plots import (
    plot_feature_importance,
)
from backlink_pricing_model.visualization.plots_style import save_plot


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAPE safely by skipping zero-denominator targets."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    mask = y_true_arr != 0
    if not mask.any():
        return float("nan")
    return float(
        np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask]))
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": safe_mape(y_true, y_pred),
    }


@click.command()
@click.option(
    "--config",
    "config_path",
    default="configs/training.yaml",
    help="Path to training config YAML.",
)
@click.option(
    "--model",
    "model_path",
    default=None,
    help="Path to model .joblib file. Defaults to config value.",
)
@click.option(
    "--test-data",
    "test_data_path",
    default=None,
    help="Path to test Parquet file. Defaults to config value.",
)
@click.option(
    "--save-plots/--no-save-plots",
    default=True,
    help="Save evaluation plots to images/modeling/.",
)
def main(
    config_path: str,
    model_path: str | None,
    test_data_path: str | None,
    save_plots: bool,
) -> None:
    """Evaluate a trained model and generate metrics and plots."""
    cfg = load_config(config_path)
    logger.info("Loaded config: %s", config_path)

    # Resolve paths.
    if model_path is None:
        model_dir = cfg.get("model_dir", "models")
        model_filename = cfg.get("model_filename", "xgb_backlink_pricing.joblib")
        model_path = str(resolve_path(f"{model_dir}/{model_filename}"))

    if test_data_path is None:
        test_data_path = str(resolve_path(cfg["test_data"]))

    # Load model and data.
    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    logger.info("Loading test data from %s", test_data_path)
    test_df = pd.read_parquet(test_data_path, engine="pyarrow")

    target = cfg["target"]
    feature_cols = cfg["feature_columns"]
    x_test = test_df[feature_cols]
    y_test = test_df[target]

    # Predict.
    y_pred = model.predict(x_test)

    metrics = {"log_scale": compute_metrics(y_test.values, y_pred)}
    target_is_log = target == "log_price"
    if target_is_log:
        y_test_plot = np.expm1(y_test.values)
        y_pred_plot = np.expm1(y_pred)
        metrics["usd_scale"] = compute_metrics(y_test_plot, y_pred_plot)
    else:
        y_test_plot = y_test.values
        y_pred_plot = y_pred

    logger.info("Evaluation results:")
    for metric_group, values in metrics.items():
        logger.info("  %s", metric_group)
        for name, value in values.items():
            logger.info("    %s: %.4f", name.upper(), value)

    # Save metrics.
    metrics_path = resolve_path(
        f"{cfg.get('model_dir', 'models')}/eval_metrics.json"
    )
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)

    # Plots.
    fig_pva = plot_predictions_vs_actuals(
        y_test_plot, y_pred_plot, title="XGBoost: predictions vs actuals (USD)"
    )
    fig_res = plot_residuals(
        y_test_plot, y_pred_plot, title="XGBoost: residual analysis (USD)"
    )

    if hasattr(model, "feature_importances_"):
        fig_imp = plot_feature_importance(
            feature_cols,
            model.feature_importances_,
            title="XGBoost feature importance",
        )

    if save_plots:
        save_plot(fig_pva, "predictions_vs_actuals", "images/modeling")
        save_plot(fig_res, "residual_analysis", "images/modeling")
        if hasattr(model, "feature_importances_"):
            save_plot(fig_imp, "feature_importance", "images/modeling")
        logger.info("Saved plots to images/modeling/")
    else:
        fig_pva.show()
        fig_res.show()
        if hasattr(model, "feature_importances_"):
            fig_imp.show()

    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
