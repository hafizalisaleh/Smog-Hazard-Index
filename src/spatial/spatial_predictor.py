"""
Spatial Gridded PM2.5 Predictor (Weeks 5-6).
Infers continuous ground-level PM2.5 concentrations over the 1 km spatial mesh
using trained machine learning models (XGBoost/Random Forest).
"""

import logging
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
import yaml

try:
    from .grid_generator import PilotGridGenerator
    from ..features.feature_engineering import FeatureEngineer
    from ..data.satellite_loader import SatelliteDataLoader
    from ..data.era5_loader import ERA5DataLoader
except (ImportError, ValueError):
    from src.spatial.grid_generator import PilotGridGenerator
    from src.features.feature_engineering import FeatureEngineer
    from src.data.satellite_loader import SatelliteDataLoader
    from src.data.era5_loader import ERA5DataLoader

logger = logging.getLogger(__name__)


class SpatialPM25Predictor:
    """Predicts gridded PM2.5 concentration across the entire pilot area."""

    def __init__(
        self,
        model_path: Union[str, Path] = "outputs/models/xgboost_final.joblib",
        grid_path: Union[str, Path] = "data/grids/pilot_grid_1km.csv",
        output_dir: Union[str, Path] = "outputs/gridded_pm25",
        config_path: Union[str, Path] = "config/config.yaml",
    ):
        self.model_path = Path(model_path)
        self.grid_path = Path(grid_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = Path(config_path)
        self.config = self._load_yaml(self.config_path)
        self._load_model()

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Train models first.")
        data = joblib.load(self.model_path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.log_transform = data.get("log_transform", True)
        self.model_name = data.get("model_name", "Model")
        logger.info(f"Loaded {self.model_name} with {len(self.feature_names)} features for spatial inference.")

    def load_grid(self) -> pd.DataFrame:
        if not self.grid_path.exists():
            logger.info("Grid not found; generating pilot domain grid...")
            gen = PilotGridGenerator(config_path=self.config_path)
            return gen.generate_grid(save_path=self.grid_path)
        return pd.read_csv(self.grid_path)

    def predict_gridded_pm25_for_date(
        self,
        target_date: str = "2025-11-15",
        save_csv: bool = True,
    ) -> pd.DataFrame:
        """
        Produce a continuous 1 km PM2.5 estimate map for a specific date (e.g., peak smog day).
        """
        grid_df = self.load_grid().copy()
        grid_df["date"] = pd.to_datetime(target_date)
        grid_df["station_id"] = "grid_cell"

        # Synthesize/extract collocated satellite and weather fields across the grid
        sat_loader = SatelliteDataLoader(config_path=self.config_path)
        era5_loader = ERA5DataLoader(config_path=self.config_path)

        grid_with_sat = sat_loader.generate_synthetic_satellite_observations(grid_df, random_state=123)
        grid_with_met = era5_loader.generate_synthetic_era5_observations(grid_with_sat, random_state=123)

        # Feature engineering
        fe = FeatureEngineer(config_path=self.config_path)
        grid_feat, _ = fe.fit_transform(grid_with_met)

        # Predict
        X_grid = grid_feat[self.feature_names]
        raw_pred = self.model.predict(X_grid)
        pred_pm25 = np.clip(np.expm1(raw_pred), 0.0, None) if self.log_transform else np.clip(raw_pred, 0.0, None)

        grid_feat["estimated_pm25"] = np.round(pred_pm25, 2)

        if save_csv:
            date_str = pd.to_datetime(target_date).strftime("%Y%m%d")
            out_file = self.output_dir / f"gridded_pm25_{date_str}.csv"
            export_cols = [
                "grid_id",
                "latitude",
                "longitude",
                "district",
                "date",
                "estimated_pm25",
                "modis_aod_1km",
                "tropomi_no2",
                "tropomi_so2",
                "stagnation_index",
                "ventilation_index",
            ]
            avail_cols = [c for c in export_cols if c in grid_feat.columns]
            grid_feat[avail_cols].to_csv(out_file, index=False)
            logger.info(f"Saved gridded PM2.5 estimate to: {out_file}")

        return grid_feat


if __name__ == "__main__":
    predictor = SpatialPM25Predictor()
    gridded = predictor.predict_gridded_pm25_for_date("2025-11-15")
    print(gridded[["grid_id", "district", "estimated_pm25"]].head(5))
