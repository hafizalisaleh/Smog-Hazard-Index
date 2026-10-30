"""
Smog Hazard Index (SHI) Formulation Module (Weeks 5-6).
Combines standardized PM2.5 estimates, MODIS AOD, Sentinel-5P NO2/SO2,
and meteorological atmospheric stagnation indicators into a unified 0-100 hazard index.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class SmogHazardIndexCalculator:
    """Calculates the multi-pollutant and meteorological Smog Hazard Index (SHI)."""

    def __init__(self, config_path: Union[str, Path] = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_yaml(self.config_path)
        self.shi_cfg = self.config.get("smog_hazard_index", {})
        self.weights = self.shi_cfg.get(
            "weights",
            {
                "pm25_standardized": 0.50,
                "aod_standardized": 0.20,
                "no2_standardized": 0.10,
                "so2_standardized": 0.05,
                "stagnation_index": 0.15,
            },
        )

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def standardize_subindex(
        val: Union[float, np.ndarray, pd.Series],
        min_val: float,
        target_val: float,
        extreme_val: float,
    ) -> Union[float, np.ndarray, pd.Series]:
        """
        Piecewise linear standardization mapping physical concentration/index to 0.0 - 1.0.
          - 0.0 at min_val (clean background)
          - 0.5 at target_val (regulatory air quality threshold)
          - 1.0 at extreme_val (severe hazardous emergency)
        """
        v = np.asarray(val, dtype=float)
        # Piecewise mapping:
        # Below target: 0.0 to 0.5
        # Above target: 0.5 to 1.0
        norm = np.where(
            v <= target_val,
            0.5 * np.clip((v - min_val) / (target_val - min_val), 0.0, 1.0),
            0.5 + 0.5 * np.clip((v - target_val) / (extreme_val - target_val), 0.0, 1.0),
        )
        return norm

    def compute_shi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute continuous SHI (scale 0 to 100) and sub-indices for each observation / pixel.
        """
        out = df.copy()

        # 1. PM2.5 subindex (target: Punjab standard 35 ug/m3, extreme: 350 ug/m3)
        pm_col = "estimated_pm25" if "estimated_pm25" in out.columns else "pm25"
        pm_val = out[pm_col].fillna(out[pm_col].median())
        out["subindex_pm25"] = self.standardize_subindex(pm_val, min_val=5.0, target_val=35.0, extreme_val=350.0)

        # 2. AOD subindex (target: 0.40, extreme: 2.50)
        aod_val = out.get("modis_aod_1km", pd.Series(0.4, index=out.index)).fillna(0.4)
        out["subindex_aod"] = self.standardize_subindex(aod_val, min_val=0.10, target_val=0.50, extreme_val=2.50)

        # 3. NO2 subindex (Sentinel-5P mol/m2; clean: 3e-5, target: 1.5e-4, extreme: 4.5e-4)
        no2_val = out.get("tropomi_no2", pd.Series(1e-4, index=out.index)).fillna(1e-4)
        out["subindex_no2"] = self.standardize_subindex(no2_val, min_val=2e-5, target_val=1.5e-4, extreme_val=4.5e-4)

        # 4. SO2 subindex (Sentinel-5P mol/m2; clean: 2e-5, target: 8e-5, extreme: 2.5e-4)
        so2_val = out.get("tropomi_so2", pd.Series(5e-5, index=out.index)).fillna(5e-5)
        out["subindex_so2"] = self.standardize_subindex(so2_val, min_val=1e-5, target_val=8e-5, extreme_val=2.5e-4)

        # 5. Stagnation subindex (already on 0-1 scale)
        stag_val = out.get("stagnation_index", pd.Series(0.4, index=out.index)).fillna(0.4)
        out["subindex_stagnation"] = np.clip(stag_val, 0.0, 1.0)

        # Multi-criteria weighted sum
        w_pm = self.weights.get("pm25_standardized", 0.50)
        w_aod = self.weights.get("aod_standardized", 0.20)
        w_no2 = self.weights.get("no2_standardized", 0.10)
        w_so2 = self.weights.get("so2_standardized", 0.05)
        w_stag = self.weights.get("stagnation_index", 0.15)

        raw_shi = (
            w_pm * out["subindex_pm25"]
            + w_aod * out["subindex_aod"]
            + w_no2 * out["subindex_no2"]
            + w_so2 * out["subindex_so2"]
            + w_stag * out["subindex_stagnation"]
        )

        # Scale to 0 - 100
        out["shi_score"] = np.round(np.clip(raw_shi * 100.0, 0.0, 100.0), 2)

        return out
