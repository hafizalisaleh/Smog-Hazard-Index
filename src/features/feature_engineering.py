"""
Feature Engineering Module (Weeks 3-4).
Constructs domain-specific features for satellite-to-ground PM2.5 modeling and SHI formulation:
  - Meteorological stagnation indicators (Ventilation Index, Stagnation Score, Thermal Inversion proxies)
  - Atmospheric vertical column interaction features (AOD/PBLH extinction ratio, NO2/SO2 source ratio)
  - Cyclical temporal embeddings (sin/cos DOY, month, seasonality indicators)
  - Geospatial coordinates and spatial proximity indicators
"""

import logging
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Transforms raw matched ground-satellite-meteorology records into high-dimensional ML feature matrices."""

    def __init__(self, config_path: Union[str, Path] = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_yaml(self.config_path)
        self.feat_cfg = self.config.get("features", {})
        self.ws_stagnation_thresh = self.feat_cfg.get("stagnation_wind_threshold_ms", 2.0)
        self.pblh_stagnation_thresh = self.feat_cfg.get("stagnation_pblh_threshold_m", 500.0)
        self.rh_stagnation_thresh = self.feat_cfg.get("stagnation_rh_threshold_pct", 70.0)

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def compute_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical temporal encodings and seasonal indicators."""
        df_out = df.copy()
        if "date" in df_out.columns:
            date_col = pd.to_datetime(df_out["date"])
        elif "timestamp" in df_out.columns:
            date_col = pd.to_datetime(df_out["timestamp"])
        else:
            return df_out

        doy = date_col.dt.dayofyear
        month = date_col.dt.month
        dow = date_col.dt.dayofweek

        df_out["day_of_year"] = doy
        df_out["month"] = month
        df_out["day_of_week"] = dow
        df_out["is_weekend"] = dow.isin([5, 6]).astype(int)

        # Cyclical sinusoidal transformations
        df_out["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
        df_out["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
        df_out["sin_month"] = np.sin(2 * np.pi * month / 12.0)
        df_out["cos_month"] = np.cos(2 * np.pi * month / 12.0)

        # Smog season indicator: Punjab smog peak typically spans mid-October through January
        df_out["is_smog_season"] = ((month >= 10) | (month <= 1)).astype(int)

        return df_out

    def compute_meteorological_stagnation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate physical atmospheric stagnation indices:
          1. Ventilation Index (VI = Wind Speed * PBLH in m^2/s)
          2. Multi-criteria Air Stagnation Score (0 to 1 continuous scale)
          3. Stagnation binary flag (WS < 2 m/s & PBLH < 500m & RH > 70%)
        """
        df_out = df.copy()

        # Support both ERA5 and real MERRA-2 weather fields
        ws = df_out.get("era5_wind_speed", df_out.get("merra2_wind_speed", pd.Series(2.0, index=df_out.index)))
        pblh = df_out.get("era5_pblh", df_out.get("merra2_pblh", pd.Series(800.0, index=df_out.index)))
        rh = df_out.get("era5_relative_humidity", df_out.get("merra2_rh", pd.Series(60.0, index=df_out.index)))

        # 1. Ventilation Index (VI) - lower values represent severe dispersion restriction
        df_out["ventilation_index"] = np.maximum(ws * pblh, 1.0)
        df_out["log_ventilation_index"] = np.log1p(df_out["ventilation_index"])

        # 2. Binary Stagnation Criteria (US EPA / WMO atmospheric stagnation standard adapted for Punjab)
        is_calm_wind = ws <= self.ws_stagnation_thresh
        is_shallow_pblh = pblh <= self.pblh_stagnation_thresh
        is_humid = rh >= self.rh_stagnation_thresh

        df_out["stagnation_flag"] = (is_calm_wind & is_shallow_pblh).astype(int)
        df_out["severe_stagnation_flag"] = (is_calm_wind & is_shallow_pblh & is_humid).astype(int)

        # 3. Continuous Stagnation Severity Index (0 to 1):
        # Higher score = stronger atmospheric trapping
        wind_stagnation = np.clip(1.0 - (ws / 5.0), 0.0, 1.0)
        pblh_stagnation = np.clip(1.0 - (pblh / 1500.0), 0.0, 1.0)
        rh_condensation = np.clip(rh / 100.0, 0.0, 1.0)

        df_out["stagnation_index"] = (
            0.45 * wind_stagnation + 0.35 * pblh_stagnation + 0.20 * rh_condensation
        ).round(4)

        return df_out

    def compute_atmospheric_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive multi-pollutant vertical column ratios and aerosol extinction proxies.
        AOD / PBLH serves as a physically meaningful proxy for ground-level extinction coefficient.
        """
        df_out = df.copy()

        # Support MODIS 1km, MODIS MCD19A2 aod_055, composite_aod, or MERRA-2 total extinction AOD
        aod = df_out.get(
            "composite_aod",
            df_out.get("modis_aod_1km", df_out.get("modis_aod_055", df_out.get("merra2_aod_totexttau", pd.Series(0.5, index=df_out.index))))
        ).fillna(0.4)
        pblh = df_out.get("era5_pblh", df_out.get("merra2_pblh", pd.Series(600.0, index=df_out.index))).clip(lower=50.0)

        # Extinction coefficient proxy (km^-1)
        df_out["aod_pblh_ratio"] = (aod / (pblh / 1000.0)).round(4)

        # Black carbon ratio to total AOD (if MERRA-2 BC is present)
        if "merra2_bc_aod" in df_out.columns:
            df_out["bc_ratio_to_aod"] = (
                df_out["merra2_bc_aod"] / np.maximum(aod, 0.05)
            ).round(4)

        # Trace gas ratios
        if "tropomi_no2" in df_out.columns and "tropomi_so2" in df_out.columns:
            no2 = df_out["tropomi_no2"].fillna(1e-4).clip(lower=1e-6)
            so2 = df_out["tropomi_so2"].fillna(5e-5).clip(lower=1e-6)
            df_out["no2_so2_ratio"] = (no2 / so2).round(3)
        else:
            df_out["no2_so2_ratio"] = 1.0

        return df_out

    def compute_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate spatial distance to major urban centroid (Central Lahore: 31.5796, 74.3562)."""
        df_out = df.copy()
        c_lat, c_lon = 31.5796, 74.3562

        lat = df_out.get("latitude", pd.Series(c_lat, index=df_out.index))
        lon = df_out.get("longitude", pd.Series(c_lon, index=df_out.index))

        # Approx Euclidean distance in km (1 deg ~ 111 km)
        d_lat_km = (lat - c_lat) * 111.0
        d_lon_km = (lon - c_lon) * 111.0 * np.cos(np.deg2rad(c_lat))
        df_out["dist_to_lahore_center_km"] = np.sqrt(d_lat_km ** 2 + d_lon_km ** 2).round(2)

        return df_out

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Apply all feature transformations and return enriched DataFrame and list of feature names."""
        logger.info(f"Engineering features for {len(df):,} input rows...")
        d1 = self.compute_temporal_features(df)
        d2 = self.compute_meteorological_stagnation_features(d1)
        d3 = self.compute_atmospheric_interaction_features(d2)
        d4 = self.compute_spatial_features(d3)

        # Standard model predictor features
        feature_cols = [
            # Satellite & Reanalysis predictors
            "modis_aod_1km",
            "modis_aod_055",
            "modis_aod_047",
            "composite_aod",
            "aod_is_modis",
            "tropomi_no2",
            "tropomi_so2",
            "cams_pm25",
            "merra2_pm25",
            # NASA MERRA-2 Real Aerosol & Meteorology predictors
            "merra2_temp_c",
            "merra2_pblh",
            "merra2_wind_speed",
            "merra2_rh",
            "merra2_ps_hpa",
            "merra2_bc_surface",
            "merra2_aod_totexttau",
            "merra2_bc_aod",
            "merra2_bc_column",
            "bc_ratio_to_aod",
            # Meteorological predictors (ERA5)
            "era5_temperature_2m",
            "era5_wind_speed",
            "era5_u10",
            "era5_v10",
            "era5_relative_humidity",
            "era5_pblh",
            "era5_surface_pressure",
            # Engineered stagnation & physics features
            "ventilation_index",
            "log_ventilation_index",
            "stagnation_index",
            "stagnation_flag",
            "severe_stagnation_flag",
            "aod_pblh_ratio",
            "no2_so2_ratio",
            # Spatial features
            "latitude",
            "longitude",
            "elevation_m",
            "dist_to_lahore_center_km",
            # Temporal features
            "sin_doy",
            "cos_doy",
            "sin_month",
            "cos_month",
            "is_smog_season",
            "is_weekend",
            "day_of_week",
            "month",
        ]

        # Keep only features actually present in d4
        active_features = [col for col in feature_cols if col in d4.columns]
        logger.info(f"Engineered {len(active_features)} active predictor features.")

        return d4, active_features


if __name__ == "__main__":
    from src.data.collocator import DatasetCollocator

    collocator = DatasetCollocator()
    matched_df = collocator.assemble_matched_dataset(save_output=False)
    fe = FeatureEngineer()
    enriched_df, features = fe.fit_transform(matched_df)
    print("Features list:", features)
    print("Enriched sample columns:", enriched_df[features].head(3))
