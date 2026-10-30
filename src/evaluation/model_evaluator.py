"""
Model Evaluation and Benchmark Orchestrator (Weeks 5-6).
Executes spatial cross-validation and held-out station testing across
Linear Baseline, Random Forest, and XGBoost models. Generates comparative
performance reports, bias analysis, and parity diagnostic plots.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from .metrics import compute_evaluation_metrics, calculate_station_metrics

try:
    from ..models.linear_baseline import LinearBaselineModel
    from ..models.random_forest_model import RandomForestModel
    from ..models.xgboost_model import XGBoostModel
    from ..models.spatial_temporal_cv import SpatialTemporalCV
    from ..features.feature_engineering import FeatureEngineer
except (ImportError, ValueError):
    from src.models.linear_baseline import LinearBaselineModel
    from src.models.random_forest_model import RandomForestModel
    from src.models.xgboost_model import XGBoostModel
    from src.models.spatial_temporal_cv import SpatialTemporalCV
    from src.features.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Manages training, cross-validation, and performance evaluation for PM2.5 models."""

    def __init__(
        self,
        dataset_path: Union[str, Path] = "data/processed/matched_training_dataset.csv",
        output_dir: Union[str, Path] = "outputs/evaluation",
        models_dir: Union[str, Path] = "outputs/models",
        config_path: Union[str, Path] = "config/config.yaml",
        prefix: str = "",
    ):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.models_dir = Path(models_dir)
        self.prefix = prefix
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = Path(config_path)
        self.config = self._load_yaml(self.config_path)

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
        """Load, engineer features, and separate predictors X and target y."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        raw_df = pd.read_csv(self.dataset_path)
        raw_df["date"] = pd.to_datetime(raw_df["date"])

        fe = FeatureEngineer(config_path=self.config_path)
        df_feat, feature_names = fe.fit_transform(raw_df)

        valid_mask = df_feat["pm25"].notna()
        df_valid = df_feat[valid_mask].copy().reset_index(drop=True)

        X = df_valid[feature_names]
        y = df_valid["pm25"]

        return X, y, df_valid, feature_names

    def run_spatial_cross_validation(self, n_splits: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Run Spatial GroupKFold CV across Linear Baseline, Random Forest, and XGBoost."""
        X, y, df_meta, feature_names = self.prepare_data()
        cv = SpatialTemporalCV(station_col="station_id", date_col="date")
        if n_splits is None:
            n_splits = self.config.get("cross_validation", {}).get("n_splits", 5)

        models = {
            "Linear_Baseline": LinearBaselineModel(),
            "Random_Forest": RandomForestModel(n_estimators=150, max_depth=14),
            "XGBoost": XGBoostModel(n_estimators=200, learning_rate=0.05, max_depth=6),
        }

        cv_predictions = {name: [] for name in models}
        fold_summaries = []

        logger.info(f"Initiating {n_splits}-fold Spatial Group CV across {df_meta['station_id'].nunique()} stations...")

        for fold, (train_idx, val_idx, held_out) in enumerate(cv.get_spatial_group_kfold_splits(df_meta, n_splits=n_splits)):
            logger.info(f"--- Fold {fold + 1}/{n_splits} | Held-out stations: {held_out} ---")
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            meta_val = df_meta.iloc[val_idx].copy()

            for name, model_cls in models.items():
                # Reinstantiate per fold
                if name == "Linear_Baseline":
                    m = LinearBaselineModel()
                elif name == "Random_Forest":
                    m = RandomForestModel(n_estimators=150, max_depth=14)
                else:
                    m = XGBoostModel(n_estimators=200, learning_rate=0.05, max_depth=6)

                m.fit(X_train, y_train)
                preds = m.predict(X_val)

                val_records = meta_val.copy()
                val_records["model"] = name
                val_records["fold"] = fold + 1
                val_records["actual"] = y_val.values
                val_records["pred"] = preds
                cv_predictions[name].append(val_records)

                metrics = compute_evaluation_metrics(y_val, preds)
                metrics["model"] = name
                metrics["fold"] = fold + 1
                metrics["held_out_stations"] = ", ".join(held_out)
                fold_summaries.append(metrics)

        # Combine CV predictions
        combined_preds = {name: pd.concat(cv_predictions[name], ignore_index=True) for name in models}

        # Overall summary per model
        overall_metrics = []
        for name, pred_df in combined_preds.items():
            met = compute_evaluation_metrics(pred_df["actual"], pred_df["pred"])
            met["model"] = name
            overall_metrics.append(met)
            # Save predictions
            pred_df.to_csv(self.output_dir / f"{self.prefix}{name.lower()}_spatial_cv_predictions.csv", index=False)

        summary_df = pd.DataFrame(overall_metrics).set_index("model")
        summary_path = self.output_dir / f"{self.prefix}model_comparison_benchmark.csv"
        summary_df.to_csv(summary_path)
        logger.info(f"Saved benchmark summary table to: {summary_path}")

        # Station-wise held-out performance for the best model
        best_model_name = summary_df["rmse"].idxmin()
        logger.info(f"Top-performing model by Spatial CV RMSE: {best_model_name}")

        station_perf = calculate_station_metrics(combined_preds[best_model_name])
        station_perf_path = self.output_dir / f"{self.prefix}held_out_station_performance_{best_model_name.lower()}.csv"
        station_perf.to_csv(station_perf_path, index=False)
        logger.info(f"Saved held-out station performance to: {station_perf_path}")

        # Train final models on all data and save
        logger.info("Training final full-dataset models for gridded spatial estimation...")
        for name in models:
            if name == "Linear_Baseline":
                final_m = LinearBaselineModel()
            elif name == "Random_Forest":
                final_m = RandomForestModel(n_estimators=200, max_depth=16)
            else:
                final_m = XGBoostModel(n_estimators=300, learning_rate=0.05, max_depth=6)

            final_m.fit(X, y)
            final_m.save(self.models_dir / f"{self.prefix}{name.lower()}_final.joblib")
            logger.info(f"Saved final trained model: {self.prefix}{name.lower()}_final.joblib")

        # Generate diagnostic parity plots
        self.plot_model_comparison_parity(combined_preds)
        self.plot_feature_importance(self.models_dir / f"{self.prefix}{best_model_name.lower()}_final.joblib")

        return {
            "summary": summary_df,
            "station_performance": station_perf,
            "best_model": best_model_name,
        }

    def plot_model_comparison_parity(self, preds_dict: Dict[str, pd.DataFrame]) -> Path:
        """Plot observed vs predicted scatter with 1:1 reference line for all models."""
        fig, axes = plt.subplots(1, len(preds_dict), figsize=(6 * len(preds_dict), 5.5), sharey=True, sharex=True)
        if len(preds_dict) == 1:
            axes = [axes]

        for ax, (name, df_p) in zip(axes, preds_dict.items()):
            actual = df_p["actual"].values
            pred = df_p["pred"].values
            metrics = compute_evaluation_metrics(actual, pred)

            ax.scatter(actual, pred, alpha=0.35, s=16, color="royalblue", edgecolors="none")
            max_val = max(float(np.nanpercentile(actual, 99.5)), float(np.nanpercentile(pred, 99.5))) * 1.05
            ax.plot([0, max_val], [0, max_val], color="crimson", linestyle="--", lw=1.8, label="1:1 Line")
            ax.plot([0, max_val], [0, max_val * 0.5], color="gray", linestyle=":", lw=1.0, label="FAC2 bounds")
            ax.plot([0, max_val * 0.5], [0, max_val], color="gray", linestyle=":", lw=1.0)

            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)
            ax.set_title(
                f"{name.replace('_', ' ')}\n"
                f"R²={metrics['r2']:.2f} | RMSE={metrics['rmse']:.1f} ug/m³ | Bias={metrics['mbe']:+.1f}",
                fontsize=11,
            )
            ax.set_xlabel("Observed PM2.5 (ug/m³)", fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=9)

        axes[0].set_ylabel("Spatial-CV Predicted PM2.5 (ug/m³)", fontsize=10)
        plt.suptitle("Spatial Cross-Validation Diagnostic: Held-Out Stations", fontsize=13, y=1.02)
        plt.tight_layout()

        parity_plot_path = self.output_dir / f"{self.prefix}spatial_cv_parity_plots.png"
        fig.savefig(parity_plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Parity diagnostic plots saved to: {parity_plot_path}")
        return parity_plot_path

    def plot_feature_importance(self, model_joblib_path: Path) -> Optional[Path]:
        """Plot top 15 predictor feature importances."""
        import joblib

        if not model_joblib_path.exists():
            return None
        data = joblib.load(model_joblib_path)
        model = data.get("model")
        feat_names = data.get("feature_names", [])

        if not hasattr(model, "feature_importances_"):
            return None

        fi_df = pd.DataFrame({"feature": feat_names, "importance": model.feature_importances_}).sort_values(
            "importance", ascending=True
        )

        plt.figure(figsize=(9, 7))
        top_fi = fi_df.tail(15)
        plt.barh(top_fi["feature"], top_fi["importance"], color="teal", alpha=0.85)
        plt.title(f"Predictor Feature Importance ({data.get('model_name', 'Model')})", fontsize=12, pad=12)
        plt.xlabel("Relative Importance / Gini Gain", fontsize=10)
        plt.grid(alpha=0.3, axis="x")
        plt.tight_layout()

        fi_path = self.output_dir / f"{self.prefix}feature_importance_top15.png"
        plt.savefig(fi_path, dpi=200)
        plt.close()
        logger.info(f"Feature importance plot saved to: {fi_path}")
        return fi_path


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    results = evaluator.run_spatial_cross_validation()
    print("\nModel Benchmark Summary:")
    print(results["summary"])
