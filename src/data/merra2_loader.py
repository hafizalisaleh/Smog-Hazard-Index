"""
NASA MERRA-2 Aerosol & Meteorology and MODIS Site Data Ingestion Module.
Ingests and harmonizes:
  1. MERRA-2 Reanalysis Aerosols: Black Carbon Column Mass (BCCMASS),
     Surface Mass Concentration (BCSMASS), Black Carbon Extinction AOD (BCEXTTAU),
     and Total Aerosol Extinction Optical Depth at 550nm (TOTEXTTAU).
  2. MERRA-2 Reanalysis Meteorology: 2m Temperature (T2M), Surface Pressure (PS),
     Planetary Boundary Layer Top Pressure (PBLTOP), 10m Winds (U10M, V10M, WIND_SPEED),
     and 2m Specific Humidity (QV2M).
  3. MODIS MCD19A2 MAIAC Land AOD (aod_055, aod_047) and QA/cloud flags.

Derives real physical parameters:
  - Temperature in Celsius
  - Planetary Boundary Layer Height (PBLH in meters via barometric hypsometric formula)
  - Relative Humidity (% via Tetens formulation from specific humidity & pressure)
  - Surface Black Carbon concentration (ug/m3)
  - Composite multi-sensor AOD (MODIS AOD 550nm + MERRA-2 fallback)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

try:
    from .ground_loader import GroundDataLoader
except (ImportError, ValueError):
    from ground_loader import GroundDataLoader

logger = logging.getLogger(__name__)


class MERRA2DataLoader:
    """Ingests and preprocesses NASA MERRA-2 aerosol, meteorology, and MODIS observations."""

    def __init__(self, config_path: Union[str, Path] = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_yaml(self.config_path)
        self.root_dir = Path(".")
        self.proc_dir = Path(self.config.get("paths", {}).get("processed_dir", "data/processed"))
        self.proc_dir.mkdir(parents=True, exist_ok=True)
        self.ground_loader = GroundDataLoader(config_path=self.config_path)

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _locate_file(self, filename: str, fallback_patterns: List[str]) -> Optional[Path]:
        """Search root dir and data/raw/ subdirectories for a target file."""
        candidates = [
            self.root_dir / filename,
            Path("data/raw/satellite") / filename,
            Path("data/raw/meteorology") / filename,
            Path("data/raw") / filename,
        ]
        for p in candidates:
            if p.exists():
                return p

        # Check by pattern
        for pat in fallback_patterns:
            matches = list(self.root_dir.glob(pat)) + list(Path("data/raw").rglob(pat))
            if matches:
                return matches[0]

        return None

    def load_merra2_hourly(self) -> pd.DataFrame:
        """Load and merge MERRA-2 aerosol and meteorology CSV files."""
        aero_path = self._locate_file("merra2_aerosol_site.csv", ["*merra2*aerosol*.csv"])
        met_path = self._locate_file("merra2_meteorology_site.csv", ["*merra2*met*.csv"])

        if aero_path is None or met_path is None:
            raise FileNotFoundError(
                f"MERRA-2 files not found. aero_path={aero_path}, met_path={met_path}. "
                "Ensure merra2_aerosol_site.csv and merra2_meteorology_site.csv are in the project folder."
            )

        logger.info(f"Loading MERRA-2 Aerosol from: {aero_path}")
        df_aero = pd.read_csv(aero_path)
        df_aero["time"] = pd.to_datetime(df_aero["time"])

        logger.info(f"Loading MERRA-2 Meteorology from: {met_path}")
        df_met = pd.read_csv(met_path)
        df_met["time"] = pd.to_datetime(df_met["time"])

        # Merge on timestamp
        merge_keys = ["time"]
        for k in ["lat", "lon"]:
            if k in df_aero.columns and k in df_met.columns:
                merge_keys.append(k)

        merra_hourly = pd.merge(df_aero, df_met, on=merge_keys, how="inner")
        logger.info(f"Merged {len(merra_hourly):,} hourly MERRA-2 records.")

        # Derive Physical Atmospheric Variables:
        # 1. Temperature in Celsius (from Kelvin)
        merra_hourly["merra2_temp_c"] = (merra_hourly["T2M"] - 273.15).round(2)

        # 2. Surface Pressure in hPa
        merra_hourly["merra2_ps_hpa"] = (merra_hourly["PS"] / 100.0).round(2)

        # 3. Planetary Boundary Layer Height (PBLH in meters)
        # Using the barometric hypsometric equation: h = (R_d * T) / g * ln(PS / PBLTOP)
        Rd = 287.05  # Gas constant for dry air (J/(kg*K))
        g = 9.80665  # Gravitational acceleration (m/s^2)
        ratio = np.clip(merra_hourly["PS"] / np.maximum(merra_hourly["PBLTOP"], 100.0), 1.0001, 3.0)
        merra_hourly["merra2_pblh"] = ((Rd * merra_hourly["T2M"] / g) * np.log(ratio)).round(1)

        # 4. Relative Humidity (% via Tetens formulation)
        es = 610.78 * np.exp(17.27 * (merra_hourly["T2M"] - 273.15) / (merra_hourly["T2M"] - 35.85))
        q = merra_hourly["QV2M"]
        e = (q * merra_hourly["PS"]) / (0.622 + 0.378 * q)
        merra_hourly["merra2_rh"] = np.clip((e / es) * 100.0, 1.0, 100.0).round(1)

        # 5. Black Carbon Surface Concentration (converted from kg/m^3 to ug/m^3)
        # 1 kg/m^3 = 10^9 ug/m^3
        merra_hourly["merra2_bc_surface"] = (merra_hourly["BCSMASS"] * 1e9).round(3)

        # 6. Wind Speed and Direction
        if "WIND_SPEED" in merra_hourly.columns:
            merra_hourly["merra2_wind_speed"] = merra_hourly["WIND_SPEED"].round(2)
        else:
            merra_hourly["merra2_wind_speed"] = np.sqrt(merra_hourly["U10M"]**2 + merra_hourly["V10M"]**2).round(2)

        # 7. Aerosol Optical Depth fields
        merra_hourly["merra2_aod_totexttau"] = merra_hourly["TOTEXTTAU"].round(4)
        merra_hourly["merra2_bc_aod"] = merra_hourly["BCEXTTAU"].round(4)
        merra_hourly["merra2_bc_column"] = merra_hourly["BCCMASS"]

        return merra_hourly

    def load_merra2_daily(self) -> pd.DataFrame:
        """Aggregate hourly MERRA-2 parameters to daily means."""
        hourly = self.load_merra2_hourly()
        hourly["date"] = pd.to_datetime(hourly["time"].dt.date)

        daily = hourly.groupby("date").agg({
            "merra2_temp_c": "mean",
            "merra2_pblh": "mean",
            "merra2_wind_speed": "mean",
            "merra2_rh": "mean",
            "merra2_ps_hpa": "mean",
            "merra2_bc_surface": "mean",
            "merra2_aod_totexttau": "mean",
            "merra2_bc_aod": "mean",
            "merra2_bc_column": "mean",
        }).reset_index()

        daily["date"] = pd.to_datetime(daily["date"])
        logger.info(f"Aggregated MERRA-2 daily dataset: {len(daily)} days ({daily['date'].min().date()} to {daily['date'].max().date()}).")
        return daily

    def load_modis_daily(self) -> pd.DataFrame:
        """Load and aggregate MODIS MCD19A2 AOD observations to daily resolution."""
        modis_path = self._locate_file("modis_site.csv", ["*modis*.csv"])
        if modis_path is None:
            logger.warning("MODIS site CSV not found. Returning empty table.")
            return pd.DataFrame(columns=["date", "modis_aod_055", "modis_aod_047", "cloud_mask", "aod_quality"])

        logger.info(f"Loading MODIS observations from: {modis_path}")
        modis = pd.read_csv(modis_path)
        modis["date"] = pd.to_datetime(modis["date"])

        daily_modis = modis.groupby("date").agg({
            "aod_055": "mean",
            "aod_047": "mean",
            "cloud_mask": "min",
            "aod_quality": "max",
        }).reset_index().rename(columns={
            "aod_055": "modis_aod_055",
            "aod_047": "modis_aod_047",
        })

        daily_modis["date"] = pd.to_datetime(daily_modis["date"])
        logger.info(f"MODIS daily records: {len(daily_modis)} days ({daily_modis['modis_aod_055'].notna().sum()} cloud-free AOD days).")
        return daily_modis

    def assemble_collocated_merra2_dataset(self, save_output: bool = True) -> pd.DataFrame:
        """
        Collocate ground EPA monitoring records with actual MERRA-2 & MODIS observations.
        Constructs composite AOD and domain stagnation indicators.
        """
        # 1. Load Ground Truth observations
        ground_daily_path = self.proc_dir / "ground_pm25_daily.csv"
        if ground_daily_path.exists():
            ground_df = pd.read_csv(ground_daily_path)
            ground_df["date"] = pd.to_datetime(ground_df["date"])
        else:
            logger.info("Ground data not pre-cleaned yet; running GroundDataLoader...")
            _, ground_df = self.ground_loader.process_and_save()
            ground_df["date"] = pd.to_datetime(ground_df["date"])

        valid_ground = ground_df.dropna(subset=["pm25"]).copy()

        # 2. Load MERRA-2 and MODIS
        merra_daily = self.load_merra2_daily()
        modis_daily = self.load_modis_daily()

        # 3. Temporal filter to overlapping period
        min_date = merra_daily["date"].min()
        max_date = merra_daily["date"].max()
        logger.info(f"Filtering ground observations to MERRA-2 window: {min_date.date()} to {max_date.date()}")
        sub_ground = valid_ground[(valid_ground["date"] >= min_date) & (valid_ground["date"] <= max_date)].copy()

        # 4. Merge datasets
        merged = pd.merge(sub_ground, merra_daily, on="date", how="left")
        merged = pd.merge(merged, modis_daily, on="date", how="left")

        # 5. Composite AOD:
        # Use real MODIS 550nm AOD when cloud-free; seamlessly fall back to MERRA-2 Total AOD (TOTEXTTAU)
        merged["composite_aod"] = merged["modis_aod_055"].fillna(merged["merra2_aod_totexttau"])
        merged["aod_is_modis"] = merged["modis_aod_055"].notna().astype(int)

        # 6. Physical feature ratios
        # Black Carbon fraction of total aerosol
        merged["bc_ratio_to_aod"] = (
            merged["merra2_bc_aod"] / np.maximum(merged["composite_aod"], 0.05)
        ).round(4)

        logger.info(
            f"Successfully assembled real MERRA-2 + MODIS matched dataset: "
            f"{len(merged):,} station-days across {merged['station_id'].nunique()} monitoring stations."
        )

        if save_output:
            out_path = self.proc_dir / "matched_merra2_training_dataset.csv"
            merged.to_csv(out_path, index=False)
            logger.info(f"Saved real collocated dataset to: {out_path}")

        return merged


if __name__ == "__main__":
    loader = MERRA2DataLoader()
    df = loader.assemble_collocated_merra2_dataset()
    print("\nDataset Preview:")
    print(df.head(3))

