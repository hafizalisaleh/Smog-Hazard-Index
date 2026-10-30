"""
Historical Smog Episode Validation Module (Weeks 7-8).
Validates the Smog Hazard Index (SHI) against documented historical episodes in Punjab:
  1. Peak Autumn Smog Emergency (Mid-November 2025: Crop residue burning & heavy stagnation)
  2. Winter Radiation Fog-Smog Inversion (Late December 2025: Shallow inversion layer)
  3. Summer Monsoon Baseline (July-August 2025: Wet precipitation scavenging)
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from .shi_calculator import SmogHazardIndexCalculator
    from .classifier import SHIClassifier
    from ..spatial.spatial_predictor import SpatialPM25Predictor
except (ImportError, ValueError):
    from src.index.shi_calculator import SmogHazardIndexCalculator
    from src.index.classifier import SHIClassifier
    from src.spatial.spatial_predictor import SpatialPM25Predictor

logger = logging.getLogger(__name__)


class HistoricalSmogValidator:
    """Validates the SHI formulation against historical Punjab smog events."""

    EPISODES = [
        {
            "name": "Peak Smog Emergency (Nov 2025)",
            "date": "2025-11-15",
            "type": "Acute Agricultural Burning & Atmospheric Stagnation",
            "expected_category": "Extreme",
        },
        {
            "name": "Winter Fog-Smog Inversion (Dec 2025)",
            "date": "2025-12-25",
            "type": "Nocturnal Inversion & Shallow PBLH Stagnation",
            "expected_category": "Very High / Extreme",
        },
        {
            "name": "Monsoon Clean Baseline (Jul 2025)",
            "date": "2025-07-15",
            "type": "Wet Deposition & Active Atmospheric Dispersion",
            "expected_category": "Low / Moderate",
        },
    ]

    def __init__(
        self,
        output_dir: Union[str, Path] = "outputs/reports",
        config_path: Union[str, Path] = "config/config.yaml",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = Path(config_path)
        self.calculator = SmogHazardIndexCalculator(config_path=self.config_path)
        self.predictor = SpatialPM25Predictor(config_path=self.config_path)

    def evaluate_historical_episodes(self) -> pd.DataFrame:
        """Run gridded estimation and SHI calculation for each benchmark episode."""
        records = []
        episode_dfs = {}

        for ep in self.EPISODES:
            logger.info(f"Simulating historical episode: {ep['name']} ({ep['date']})...")
            gridded = self.predictor.predict_gridded_pm25_for_date(ep["date"], save_csv=False)
            with_shi = self.calculator.compute_shi(gridded)
            classified = SHIClassifier.classify_dataframe(with_shi)

            episode_dfs[ep["name"]] = classified

            mean_pm25 = float(classified["estimated_pm25"].mean())
            mean_shi = float(classified["shi_score"].mean())
            pct_extreme = float((classified["hazard_category"] == "Extreme").mean() * 100)
            pct_very_high = float((classified["hazard_category"] == "Very High").mean() * 100)

            records.append(
                {
                    "episode_name": ep["name"],
                    "date": ep["date"],
                    "episode_type": ep["type"],
                    "expected_category": ep["expected_category"],
                    "mean_estimated_pm25_ugm3": round(mean_pm25, 1),
                    "mean_shi_score": round(mean_shi, 1),
                    "pct_area_extreme": round(pct_extreme, 1),
                    "pct_area_very_high": round(pct_very_high, 1),
                    "dominant_hazard_category": classified["hazard_category"].mode()[0],
                }
            )

        val_df = pd.DataFrame(records)
        val_csv = self.output_dir / "historical_episodes_validation.csv"
        val_df.to_csv(val_csv, index=False)
        logger.info(f"Saved historical episode validation report to: {val_csv}")

        self.plot_historical_comparison(episode_dfs)
        return val_df

    def plot_historical_comparison(self, episode_dfs: Dict[str, pd.DataFrame]) -> Path:
        """Plot comparative SHI score distributions across the three benchmark episodes."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # 1. Boxplot of SHI distribution
        plot_data = []
        for name, df in episode_dfs.items():
            sub = df[["shi_score"]].copy()
            sub["Episode"] = name
            plot_data.append(sub)

        combined = pd.concat(plot_data, ignore_index=True)
        sns.boxplot(x="Episode", y="shi_score", data=combined, ax=axes[0], palette="Set2")
        axes[0].set_title("Smog Hazard Index (SHI) Across Benchmark Episodes", fontsize=11)
        axes[0].set_ylabel("SHI Score (0 - 100)", fontsize=10)
        axes[0].grid(alpha=0.3, axis="y")

        # 2. Category distribution stacked bars
        cat_counts = {}
        order = ["Low", "Moderate", "High", "Very High", "Extreme"]
        colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]

        for name, df in episode_dfs.items():
            counts = df["hazard_category"].value_counts(normalize=True).reindex(order, fill_value=0.0) * 100
            cat_counts[name] = counts

        cat_df = pd.DataFrame(cat_counts).T
        cat_df.plot(kind="bar", stacked=True, ax=axes[1], color=colors, edgecolor="black", lw=0.5)
        axes[1].set_title("Hazard Severity Area Breakdown (%)", fontsize=11)
        axes[1].set_ylabel("Area Proportion (%)", fontsize=10)
        axes[1].legend(title="Hazard Level", bbox_to_anchor=(1.02, 1), loc="upper left")
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=15, ha="right")
        axes[1].grid(alpha=0.3, axis="y")

        plt.tight_layout()
        plot_path = self.output_dir / "historical_episodes_comparison.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved historical comparison plot to: {plot_path}")
        return plot_path


if __name__ == "__main__":
    validator = HistoricalSmogValidator()
    results = validator.evaluate_historical_episodes()
    print("Historical validation results:")
    print(results)
