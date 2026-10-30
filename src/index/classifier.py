"""
Smog Hazard Index (SHI) Classification and Advisory Module.
Categorizes SHI scores into 5 standardized severity tiers:
  - Low (0 - 25): Minimal health risk, routine activity
  - Moderate (25 - 50): Acceptable air quality; sensitive groups monitor symptoms
  - High (50 - 70): Unhealthy; active avoidance of outdoor exertion
  - Very High (70 - 85): Very unhealthy; industrial emission curtailments & school advisories
  - Extreme (85 - 100): Hazardous emergency; traffic odd-even, shutdown of non-essential brick kilns
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd


class SHIClassifier:
    """Classifies continuous SHI values into standardized smog hazard categories."""

    TIERS = [
        ("Low", 0.0, 25.0, "#2ca02c", "Good / Acceptable; no restriction on outdoor activity"),
        ("Moderate", 25.0, 50.0, "#ff7f0e", "Moderate risk; sensitive individuals should reduce prolonged exertion"),
        ("High", 50.0, 70.0, "#d62728", "Unhealthy; general public wear N95 masks; avoid outdoor exercise"),
        ("Very High", 70.0, 85.0, "#9467bd", "Very Unhealthy; school activity suspensions, commercial hours curtailed"),
        ("Extreme", 85.0, 100.0, "#8c564b", "Hazardous Smog Emergency; full industrial curtailment, odd-even traffic"),
    ]

    @classmethod
    def classify_score(cls, score: float) -> Tuple[str, str, str]:
        """Return (category, color_hex, advisory) for a single numerical SHI score."""
        for cat, low, high, color, adv in cls.TIERS:
            if low <= score <= high:
                return cat, color, adv
        return "Extreme", "#8c564b", cls.TIERS[-1][4]

    @classmethod
    def classify_dataframe(cls, df: pd.DataFrame, score_col: str = "shi_score") -> pd.DataFrame:
        """Add categorical and color-coded hazard columns to DataFrame."""
        out = df.copy()
        scores = out[score_col].values

        categories = []
        colors = []
        advisories = []

        for s in scores:
            cat, col, adv = cls.classify_score(float(s) if np.isfinite(s) else 0.0)
            categories.append(cat)
            colors.append(col)
            advisories.append(adv)

        out["hazard_category"] = categories
        out["hazard_color"] = colors
        out["public_health_advisory"] = advisories

        return out
