"""
Spatial-Temporal Cross-Validation Harness.
Implements:
  1. Spatial Group K-Fold (grouped by monitoring station) to test generalization to unmonitored sites.
  2. Spatial Leave-One-Station-Out (LOSO) cross-validation.
  3. Temporal Block CV (chronological splitting) to test seasonal generalization.
Prevents spatial and temporal data leakage in air quality estimation.
"""

import logging
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, TimeSeriesSplit

logger = logging.getLogger(__name__)


class SpatialTemporalCV:
    """Manages spatial, station-grouped, and temporal cross-validation splits."""

    def __init__(self, station_col: str = "station_id", date_col: str = "date"):
        self.station_col = station_col
        self.date_col = date_col

    def get_spatial_group_kfold_splits(
        self, df: pd.DataFrame, n_splits: int = 5
    ) -> Generator[Tuple[np.ndarray, np.ndarray, List[str]], None, None]:
        """
        Group observations by station so no monitoring station appears in both train and validation.
        Yields: (train_indices, val_indices, held_out_station_ids)
        """
        gkf = GroupKFold(n_splits=n_splits)
        groups = df[self.station_col].values

        for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
            held_out_stations = list(df.iloc[val_idx][self.station_col].unique())
            yield train_idx, val_idx, held_out_stations

    def get_leave_one_station_out_splits(
        self, df: pd.DataFrame
    ) -> Generator[Tuple[np.ndarray, np.ndarray, str], None, None]:
        """
        Leave each monitoring station out in turn as an untouched spatial test set.
        Yields: (train_indices, val_indices, held_out_station_id)
        """
        logo = LeaveOneGroupOut()
        groups = df[self.station_col].values

        for train_idx, val_idx in logo.split(df, groups=groups):
            held_out_st = str(df.iloc[val_idx][self.station_col].iloc[0])
            yield train_idx, val_idx, held_out_st

    def get_temporal_block_splits(
        self, df: pd.DataFrame, n_splits: int = 4
    ) -> Generator[Tuple[np.ndarray, np.ndarray, Tuple[pd.Timestamp, pd.Timestamp]], None, None]:
        """
        Chronological forward-chaining splits.
        Yields: (train_indices, val_indices, (val_start_date, val_end_date))
        """
        df_sorted = df.sort_values(self.date_col).reset_index(drop=True)
        tscv = TimeSeriesSplit(n_splits=n_splits)

        for train_idx, val_idx in tscv.split(df_sorted):
            val_dates = pd.to_datetime(df_sorted.iloc[val_idx][self.date_col])
            yield train_idx, val_idx, (val_dates.min(), val_dates.max())
