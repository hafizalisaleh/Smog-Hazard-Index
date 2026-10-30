"""
Smog Hotspot Detection Module (Weeks 7-8).
Identifies spatial clusters of acute smog hazard across the pilot study area.
Extracts top vulnerability hotspots and flags critical emission zones.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HotspotDetector:
    """Detects and isolates acute smog hazard hotspots across the gridded domain."""

    def __init__(self, shi_threshold: float = 70.0, percentile_threshold: float = 90.0):
        self.shi_threshold = shi_threshold
        self.percentile_threshold = percentile_threshold

    def identify_hotspots(
        self,
        gridded_shi_df: pd.DataFrame,
        shi_col: str = "shi_score",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Flag acute smog hotspot pixels and produce a district-level hotspot summary.
        """
        df = gridded_shi_df.copy()
        cutoff_val = max(self.shi_threshold, float(np.percentile(df[shi_col], self.percentile_threshold)))

        df["is_hotspot"] = (df[shi_col] >= cutoff_val).astype(int)

        hotspot_pixels = df[df["is_hotspot"] == 1].copy()
        logger.info(f"Identified {len(hotspot_pixels):,} hotspot pixels (SHI >= {cutoff_val:.1f})")

        # District-level aggregation
        dist_summary = []
        for dist, grp in df.groupby("district"):
            total_pix = len(grp)
            hs_pix = int(grp["is_hotspot"].sum())
            mean_shi = float(grp[shi_col].mean())
            max_shi = float(grp[shi_col].max())
            severe_pct = round(100.0 * hs_pix / total_pix, 2)

            dist_summary.append(
                {
                    "district": dist,
                    "total_grid_area_km2": total_pix,
                    "hotspot_area_km2": hs_pix,
                    "hotspot_area_percent": severe_pct,
                    "mean_shi": round(mean_shi, 1),
                    "max_shi": round(max_shi, 1),
                }
            )

        summary_df = pd.DataFrame(dist_summary).sort_values("hotspot_area_percent", ascending=False)
        return df, summary_df
