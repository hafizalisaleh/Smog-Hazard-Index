"""
Spatio-Temporal Dataset Collocation Module.
Matches ground-truth PM2.5 observations with satellite products (MODIS AOD, Sentinel-5P, CAMS)
and ERA5 reanalysis meteorology to build a matched multi-sensor dataset for ML modeling.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import yaml

try:
    from .ground_loader import GroundDataLoader
    from .satellite_loader import SatelliteDataLoader
    from .era5_loader import ERA5DataLoader
except (ImportError, ValueError):
    from ground_loader import GroundDataLoader
    from satellite_loader import SatelliteDataLoader
    from era5_loader import ERA5DataLoader

logger = logging.getLogger(__name__)


class DatasetCollocator:
    """Assembles the collocated ground-satellite-meteorology dataset."""

    def __init__(self, config_path: Union[str, Path] = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_yaml(self.config_path)
        self.ground_loader = GroundDataLoader(config_path=self.config_path)
        self.satellite_loader = SatelliteDataLoader(config_path=self.config_path)
        self.era5_loader = ERA5DataLoader(config_path=self.config_path)

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def assemble_matched_dataset(
        self,
        use_daily: bool = True,
        save_output: bool = True,
        output_filename: str = "matched_training_dataset.csv",
    ) -> pd.DataFrame:
        """
        Assemble the unified dataset. Daily is standard for MODIS AOD satellite-ground collocation
        because MODIS has 1 daily overpass (~10:30 AM local time for Terra, ~13:30 for Aqua).
        """
        proc_dir = Path(self.config.get("paths", {}).get("processed_dir", "data/processed"))
        ground_daily_path = proc_dir / "ground_pm25_daily.csv"

        if ground_daily_path.exists():
            logger.info(f"Loading pre-cleaned ground dataset from: {ground_daily_path}")
            ground_df = pd.read_csv(ground_daily_path)
        else:
            logger.info("Ground data not pre-cleaned yet; running GroundDataLoader...")
            _, ground_df = self.ground_loader.process_and_save()

        # Filter out rows where daily PM2.5 is NaN (insufficient valid hours)
        valid_ground = ground_df.dropna(subset=["pm25"]).copy()
        logger.info(f"Assembling collocated dataset from {len(valid_ground):,} valid station-days...")

        # 1. Merge satellite observations
        with_satellite = self.satellite_loader.load_or_generate_satellite_dataset(valid_ground)

        # 2. Merge ERA5 meteorology
        matched_df = self.era5_loader.load_or_generate_era5_dataset(with_satellite)

        # Ensure datetime format
        matched_df["date"] = pd.to_datetime(matched_df["date"])

        # Quality check & summary
        logger.info(
            f"Successfully assembled matched dataset with {len(matched_df):,} records and {matched_df.shape[1]} columns."
        )

        if save_output:
            out_path = proc_dir / output_filename
            matched_df.to_csv(out_path, index=False)
            logger.info(f"Saved matched dataset to: {out_path}")

        return matched_df


if __name__ == "__main__":
    collocator = DatasetCollocator()
    df = collocator.assemble_matched_dataset()
    print("Collocated dataset preview:")
    print(df.head(5))
