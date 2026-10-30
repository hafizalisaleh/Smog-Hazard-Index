"""
Pilot Area Spatial Grid Generator.
Constructs a regular ~1 km (0.01 deg) coordinate mesh covering
the Punjab pilot region (Lahore, Gujranwala, and Sheikhupura districts).
"""

import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class PilotGridGenerator:
    """Generates a high-resolution 1 km regular spatial lattice for pilot domain mapping."""

    def __init__(self, config_path: Union[str, Path] = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_yaml(self.config_path)
        self.domain_cfg = self.config.get("pilot_domain", {})

        self.min_lat = self.domain_cfg.get("min_lat", 31.10)
        self.max_lat = self.domain_cfg.get("max_lat", 32.35)
        self.min_lon = self.domain_cfg.get("min_lon", 73.80)
        self.max_lon = self.domain_cfg.get("max_lon", 74.65)
        self.res = self.domain_cfg.get("grid_resolution_deg", 0.01)

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def generate_grid(self, save_path: Union[str, Path] = "data/grids/pilot_grid_1km.csv") -> pd.DataFrame:
        """Create 2D coordinate mesh and attribute district centroids and spatial features."""
        lats = np.arange(self.min_lat, self.max_lat + self.res, self.res)
        lons = np.arange(self.min_lon, self.max_lon + self.res, self.res)

        grid_lon, grid_lat = np.meshgrid(lons, lats)
        flat_lon = grid_lon.ravel()
        flat_lat = grid_lat.ravel()

        logger.info(
            f"Building regular pilot grid: {len(lats)} lat x {len(lons)} lon = {len(flat_lat):,} pixels (~1 km resolution)"
        )

        grid_df = pd.DataFrame(
            {
                "grid_id": [f"pix_{i:06d}" for i in range(len(flat_lat))],
                "latitude": np.round(flat_lat, 4),
                "longitude": np.round(flat_lon, 4),
                "elevation_m": np.round(215.0 + 15.0 * (flat_lat - 31.5), 1),
            }
        )

        # Assign closest district based on administrative center coordinates
        districts = {
            "Lahore": (31.5796, 74.3562),
            "Gujranwala": (32.1617, 74.1883),
            "Sheikhupura": (31.7150, 73.9850),
        }

        closest_district = []
        c_lat, c_lon = districts["Lahore"]
        dist_to_lhr = []

        for lat, lon in zip(flat_lat, flat_lon):
            # Distance to Lahore center
            d_lhr = np.sqrt(((lat - c_lat) * 111.0) ** 2 + ((lon - c_lon) * 111.0 * np.cos(np.deg2rad(c_lat))) ** 2)
            dist_to_lhr.append(round(d_lhr, 2))

            # Nearest district centroid
            min_dist = float("inf")
            best_d = "Lahore"
            for d_name, (d_lat, d_lon) in districts.items():
                dist_km = np.sqrt(((lat - d_lat) * 111.0) ** 2 + ((lon - d_lon) * 111.0 * np.cos(np.deg2rad(d_lat))) ** 2)
                if dist_km < min_dist:
                    min_dist = dist_km
                    best_d = d_name
            closest_district.append(best_d)

        grid_df["district"] = closest_district
        grid_df["dist_to_lahore_center_km"] = dist_to_lhr

        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        grid_df.to_csv(path, index=False)
        logger.info(f"Saved pilot 1 km grid to: {path}")

        return grid_df


if __name__ == "__main__":
    generator = PilotGridGenerator()
    grid = generator.generate_grid()
    print("Grid summary:")
    print(grid["district"].value_counts())
    print(grid.head(3))
