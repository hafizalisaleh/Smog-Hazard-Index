"""
Universal Ground Truth PM2.5 Data Ingestion and Cleaning Module.
Loads hourly PM2.5 records from multiple monitoring stations across
Lahore, Gujranwala, and Sheikhupura, cleans anomalies, handles gaps,
and enriches with geospatial station coordinates.
"""

import os
import re
import json
import logging
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXCEL_EPOCH = pd.Timestamp("1899-12-30")
TIME_HEADER_KEYWORDS = {"time", "timestamp", "date & time", "date and time", "datetime", "date"}


class GroundDataLoader:
    """Ingests, cleans, and standardizes multi-station ground PM2.5 observations."""

    def __init__(
        self,
        config_path: Union[str, Path] = "config/config.yaml",
        stations_path: Union[str, Path] = "config/stations.json",
    ):
        self.config_path = Path(config_path)
        self.stations_path = Path(stations_path)
        self.config = self._load_yaml(self.config_path)
        self.stations_meta = self._load_json(self.stations_path)
        self.cleaning_cfg = self.config.get("cleaning", {})
        self.max_interp_gap_h = self.cleaning_cfg.get("max_interp_gap_hours", 6)
        self.spike_min = self.cleaning_cfg.get("spike_min_ugm3", 700.0)
        self.spike_ratio = self.cleaning_cfg.get("spike_ratio_to_median", 5.0)
        self.spike_window_h = self.cleaning_cfg.get("spike_window_hours", 336)

        # Build alias lookup mapping
        self.alias_to_station = {}
        for st in self.stations_meta["stations"]:
            st_id = st["id"]
            self.alias_to_station[st_id.lower()] = st
            for alias in st.get("aliases", []):
                self.alias_to_station[alias.lower().strip()] = st

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _load_json(path: Path) -> dict:
        if not path.exists():
            return {"stations": []}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def parse_timestamp(val) -> Tuple[pd.Timestamp, str]:
        """Convert heterogeneous timestamps (Excel serials, strings, datetimes) to pandas Timestamp."""
        if val is None or (not isinstance(val, str) and pd.isna(val)):
            return pd.NaT, "blank"
        if isinstance(val, (dt.datetime, np.datetime64)):
            return pd.Timestamp(val), "datetime"
        if isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, bool):
            if np.isfinite(val) and 20000 <= float(val) <= 80000:
                return EXCEL_EPOCH + pd.to_timedelta(float(val), unit="D"), "excel_serial"
            return pd.NaT, "unparseable"
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return pd.NaT, "blank"
            try:
                f = float(s)
                if 20000 <= f <= 80000:
                    return EXCEL_EPOCH + pd.to_timedelta(f, unit="D"), "excel_serial"
            except ValueError:
                pass
            try:
                ts = pd.to_datetime(s, errors="coerce")
                if not pd.isna(ts):
                    return ts, "text"
            except Exception:
                pass
        return pd.NaT, "unparseable"

    @staticmethod
    def parse_value(val) -> Tuple[float, str]:
        """Parse float PM2.5 readings, sanitizing string representations."""
        if val is None or (not isinstance(val, str) and pd.isna(val)):
            return np.nan, "blank"
        if isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, bool):
            return (float(val), "number") if np.isfinite(val) else (np.nan, "non_finite")
        if isinstance(val, str):
            s = val.strip().replace(",", "")
            if not s:
                return np.nan, "blank"
            try:
                f = float(s)
                return (f, "numeric_text") if np.isfinite(f) else (np.nan, "non_finite")
            except ValueError:
                return np.nan, "non_numeric_text"
        return np.nan, "non_numeric_text"

    def match_station_identifier(self, identifier: str) -> Optional[dict]:
        """Match string to recognized monitoring station metadata."""
        clean_id = re.sub(r"[_\-\s\(\)\.]+", " ", identifier.lower()).strip()
        # Direct lookup
        if clean_id in self.alias_to_station:
            return self.alias_to_station[clean_id]

        # Fuzzy regex search against aliases
        for alias, st_info in self.alias_to_station.items():
            if alias in clean_id or clean_id in alias:
                return st_info
            # Match tokens
            alias_tokens = set(re.findall(r"\w+", alias))
            id_tokens = set(re.findall(r"\w+", clean_id))
            if alias_tokens and alias_tokens.issubset(id_tokens):
                return st_info
        return None

    def read_multistation_sheet(self, filepath: Path, sheet_name: str = "Sheet1") -> pd.DataFrame:
        """Read wide-format sheet (Time in col 0, station columns afterwards)."""
        logger.info(f"Reading multi-station sheet from: {filepath.name} [{sheet_name}]")
        xl = pd.ExcelFile(filepath)
        df_raw = pd.read_excel(xl, sheet_name=sheet_name, header=None, dtype=object)

        # Locate time header row
        hdr_row = None
        for i in range(min(15, len(df_raw))):
            for j, cell in enumerate(df_raw.iloc[i]):
                if isinstance(cell, str) and cell.strip().lower() in TIME_HEADER_KEYWORDS:
                    hdr_row = i
                    time_col = j
                    break
            if hdr_row is not None:
                break

        if hdr_row is None:
            logger.warning(f"No time header detected in {filepath.name} [{sheet_name}]")
            return pd.DataFrame()

        headers = df_raw.iloc[hdr_row]
        records = []

        for col_idx in range(time_col + 1, df_raw.shape[1]):
            col_name = str(headers.iloc[col_idx])
            st_info = self.match_station_identifier(col_name)
            if not st_info:
                continue

            sub = df_raw.iloc[hdr_row + 1 :, [time_col, col_idx]].dropna(how="all")
            for _, r in sub.iterrows():
                ts, ts_kind = self.parse_timestamp(r.iloc[0])
                val, val_kind = self.parse_value(r.iloc[1])
                if pd.notna(ts):
                    records.append(
                        {
                            "station_id": st_info["id"],
                            "station_name": st_info["display_name"],
                            "district": st_info["district"],
                            "latitude": st_info["latitude"],
                            "longitude": st_info["longitude"],
                            "elevation_m": st_info.get("elevation_m", 215.0),
                            "raw_timestamp": ts,
                            "pm25_raw": val,
                            "source_file": filepath.name,
                        }
                    )

        return pd.DataFrame(records)

    def read_per_station_sheets(self, filepath: Path) -> pd.DataFrame:
        """Read workbook with one sheet per station (e.g. 2025 dataset)."""
        logger.info(f"Reading per-station workbook: {filepath.name}")
        xl = pd.ExcelFile(filepath)
        all_records = []

        for sheet in xl.sheet_names:
            st_info = self.match_station_identifier(sheet)
            if not st_info:
                continue

            raw = pd.read_excel(xl, sheet_name=sheet, header=None, dtype=object)
            # Find time header row
            loc = None
            for i in range(min(15, len(raw))):
                for j, cell in enumerate(raw.iloc[i]):
                    if isinstance(cell, str) and cell.strip().lower() in TIME_HEADER_KEYWORDS:
                        loc = (i, j)
                        break
                if loc:
                    break

            if not loc:
                continue

            hdr_row, time_col = loc
            # Value column is usually adjacent or has 'pm' in header
            val_col = None
            hdr_line = raw.iloc[hdr_row]
            for j in range(len(hdr_line)):
                if j != time_col and isinstance(hdr_line.iloc[j], str) and "pm" in hdr_line.iloc[j].lower():
                    val_col = j
                    break
            if val_col is None:
                val_col = time_col + 1 if (time_col + 1) < raw.shape[1] else None

            if val_col is None:
                continue

            sub = raw.iloc[hdr_row + 1 :, [time_col, val_col]].dropna(how="all")
            for _, r in sub.iterrows():
                ts, ts_kind = self.parse_timestamp(r.iloc[0])
                val, val_kind = self.parse_value(r.iloc[1])
                if pd.notna(ts):
                    all_records.append(
                        {
                            "station_id": st_info["id"],
                            "station_name": st_info["display_name"],
                            "district": st_info["district"],
                            "latitude": st_info["latitude"],
                            "longitude": st_info["longitude"],
                            "elevation_m": st_info.get("elevation_m", 215.0),
                            "raw_timestamp": ts,
                            "pm25_raw": val,
                            "source_file": filepath.name,
                        }
                    )

        return pd.DataFrame(all_records)

    def load_all_raw(self, base_dir: Path = Path(".")) -> pd.DataFrame:
        """Scan and ingest all configured ground data Excel files."""
        frames = []
        raw_files = self.config.get("paths", {}).get("raw_ground_data", [])

        for f_name in raw_files:
            target = base_dir / f_name
            if not target.exists():
                logger.warning(f"File not found: {target}")
                continue

            xl = pd.ExcelFile(target)
            if len(xl.sheet_names) == 1:
                df = self.read_multistation_sheet(target, xl.sheet_names[0])
            else:
                df = self.read_per_station_sheets(target)

            if not df.empty:
                frames.append(df)

        if not frames:
            raise FileNotFoundError("No valid ground PM2.5 data could be ingested from the workbooks.")

        combined = pd.concat(frames, ignore_index=True)
        logger.info(f"Loaded {len(combined):,} total raw observation rows across {combined['station_id'].nunique()} stations.")
        return combined

    def clean_station_series(self, st_df: pd.DataFrame) -> pd.DataFrame:
        """Apply full scientific QC pipeline to an individual station series."""
        df = st_df.copy()
        # Round timestamp to closest hour
        df["timestamp"] = df["raw_timestamp"].dt.round("h")

        # Deduplicate per hour (average duplicate timestamps)
        dedup = df.groupby("timestamp")["pm25_raw"].mean().sort_index()
        if dedup.empty:
            return pd.DataFrame()

        # Build continuous hourly index
        full_idx = pd.date_range(dedup.index.min(), dedup.index.max(), freq="h", name="timestamp")
        series = dedup.reindex(full_idx)

        # Flag 1: Non-positive (<= 0)
        nonpos = series <= 0.0

        # Flag 2: Extreme spike (> spike_min and > spike_ratio * 14-day rolling median)
        baseline = series.where(~nonpos).rolling(self.spike_window_h, center=True, min_periods=self.spike_window_h // 4).median()
        spike = (series >= self.spike_min) & (series > (self.spike_ratio * baseline))

        # Flag 3: Unphysically high values (> max_valid_value)
        unphysical = series > self.cleaning_cfg.get("max_valid_value", 1200.0)

        invalid_mask = nonpos | spike | unphysical
        valid = series.where(~invalid_mask)

        # Short gap interpolation (runs of <= max_interp_gap_h)
        isna = valid.isna()
        grp = (isna != isna.shift(fill_value=False)).cumsum()
        run_len = isna.groupby(grp).transform("sum")
        fill_mask = isna & (run_len <= self.max_interp_gap_h)

        interpolated = valid.interpolate(method="linear", limit_area="inside")
        clean_pm25 = valid.where(~fill_mask, interpolated)

        # Status tracking
        status = pd.Series(
            np.where(
                nonpos,
                "invalid_nonpos",
                np.where(
                    spike,
                    "invalid_spike",
                    np.where(
                        unphysical,
                        "invalid_unphysical",
                        np.where(fill_mask, "interpolated", np.where(series.isna(), "missing", "observed")),
                    ),
                ),
            ),
            index=full_idx,
        )

        out = pd.DataFrame(
            {
                "timestamp": full_idx,
                "pm25_raw": series.values,
                "pm25": clean_pm25.values,
                "status": status.values,
            }
        )

        # Carry metadata
        first_row = st_df.iloc[0]
        out["station_id"] = first_row["station_id"]
        out["station_name"] = first_row["station_name"]
        out["district"] = first_row["district"]
        out["latitude"] = first_row["latitude"]
        out["longitude"] = first_row["longitude"]
        out["elevation_m"] = first_row["elevation_m"]

        return out

    def process_and_save(self, base_dir: Path = Path(".")) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run complete extraction, cleaning, and export hourly and daily tables."""
        raw_df = self.load_all_raw(base_dir=base_dir)

        cleaned_station_dfs = []
        for st_id, group in raw_df.groupby("station_id"):
            st_clean = self.clean_station_series(group)
            cleaned_station_dfs.append(st_clean)

        hourly_df = pd.concat(cleaned_station_dfs, ignore_index=True)

        # Daily aggregate (EPA 75% completeness rule: at least 18 valid hours/day)
        hourly_df["date"] = pd.to_datetime(hourly_df["timestamp"].dt.date)
        daily_records = []

        for (st_id, date), grp in hourly_df.groupby(["station_id", "date"]):
            valid_readings = grp["pm25"].dropna()
            n_valid = len(valid_readings)
            meta = grp.iloc[0]
            if n_valid >= 18:
                daily_pm25 = float(valid_readings.mean())
            else:
                daily_pm25 = np.nan

            daily_records.append(
                {
                    "date": date,
                    "station_id": st_id,
                    "station_name": meta["station_name"],
                    "district": meta["district"],
                    "latitude": meta["latitude"],
                    "longitude": meta["longitude"],
                    "elevation_m": meta["elevation_m"],
                    "pm25": daily_pm25,
                    "valid_hours": n_valid,
                }
            )

        daily_df = pd.DataFrame(daily_records)

        # Save to processed directory
        proc_dir = base_dir / self.config.get("paths", {}).get("processed_dir", "data/processed")
        proc_dir.mkdir(parents=True, exist_ok=True)

        hourly_path = proc_dir / "ground_pm25_hourly.csv"
        daily_path = proc_dir / "ground_pm25_daily.csv"

        hourly_df.to_csv(hourly_path, index=False)
        daily_df.to_csv(daily_path, index=False)
        logger.info(f"Saved cleaned hourly dataset: {hourly_path} ({len(hourly_df):,} rows)")
        logger.info(f"Saved daily aggregated dataset: {daily_path} ({len(daily_df):,} rows)")

        return hourly_df, daily_df


if __name__ == "__main__":
    loader = GroundDataLoader()
    hourly, daily = loader.process_and_save()
    print("Sample daily records:")
    print(daily.dropna().head(10))
