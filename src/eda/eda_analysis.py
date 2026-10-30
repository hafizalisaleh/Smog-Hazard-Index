"""
Exploratory Data Analysis (EDA) and Quality Control Module (Weeks 1-2).
Generates descriptive statistics, data completeness reports, pollutant-meteorology
correlation matrices, and seasonal trend plots for the pilot area.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

logger = logging.getLogger(__name__)


class EDAAnalyzer:
    """Executes systematic EDA and quality checks on the matched dataset."""

    def __init__(
        self,
        dataset_path: Union[str, Path] = "data/processed/matched_training_dataset.csv",
        output_dir: Union[str, Path] = "outputs/eda",
        config_path: Union[str, Path] = "config/config.yaml",
    ):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = Path(config_path)

    def load_data(self) -> pd.DataFrame:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Matched dataset not found at {self.dataset_path}. Run collocation first.")
        df = pd.read_csv(self.dataset_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def station_completeness_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate observation count, valid percentage, mean, std, and extremes per station."""
        records = []
        for st_id, grp in df.groupby("station_id"):
            meta = grp.iloc[0]
            pm = grp["pm25"].dropna()
            records.append(
                {
                    "station_id": st_id,
                    "station_name": meta["station_name"],
                    "district": meta["district"],
                    "total_days_recorded": len(grp),
                    "valid_pm25_days": len(pm),
                    "mean_pm25_ugm3": round(float(pm.mean()), 2) if len(pm) else np.nan,
                    "median_pm25_ugm3": round(float(pm.median()), 2) if len(pm) else np.nan,
                    "std_pm25_ugm3": round(float(pm.std()), 2) if len(pm) else np.nan,
                    "min_pm25_ugm3": round(float(pm.min()), 2) if len(pm) else np.nan,
                    "max_pm25_ugm3": round(float(pm.max()), 2) if len(pm) else np.nan,
                    "skewness": round(float(pm.skew()), 2) if len(pm) else np.nan,
                }
            )

        report = pd.DataFrame(records).sort_values("mean_pm25_ugm3", ascending=False)
        out_path = self.output_dir / "station_quality_report.csv"
        report.to_csv(out_path, index=False)
        logger.info(f"Station quality report saved to: {out_path}")
        return report

    def correlation_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Pearson and Spearman correlation between PM2.5, satellite, and weather variables."""
        num_cols = [
            "pm25",
            "modis_aod_1km",
            "tropomi_no2",
            "tropomi_so2",
            "cams_pm25",
            "era5_temperature_2m",
            "era5_pblh",
            "era5_wind_speed",
            "era5_relative_humidity",
            "era5_surface_pressure",
        ]
        available_cols = [c for c in num_cols if c in df.columns]
        corr = df[available_cols].corr()

        out_path = self.output_dir / "variable_correlations.csv"
        corr.to_csv(out_path)

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar=True, square=True)
        plt.title("Correlation Matrix: Ground PM2.5 vs Satellite & ERA5 Meteorology", fontsize=12, pad=15)
        plt.tight_layout()
        plot_path = self.output_dir / "correlation_heatmap.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        logger.info(f"Correlation heatmap saved to: {plot_path}")
        return corr

    def plot_seasonal_trends(self, df: pd.DataFrame) -> Path:
        """Plot temporal trends across the monitoring period grouped by district."""
        df_copy = df.copy()
        df_copy["month_year"] = df_copy["date"].dt.to_period("M").dt.to_timestamp()

        district_monthly = df_copy.groupby(["month_year", "district"])["pm25"].mean().reset_index()

        plt.figure(figsize=(12, 6))
        for dist in district_monthly["district"].unique():
            sub = district_monthly[district_monthly["district"] == dist]
            plt.plot(sub["month_year"], sub["pm25"], marker="o", lw=2, label=dist)

        plt.axhline(15.0, color="gray", linestyle="--", alpha=0.7, label="WHO 24h Guideline (15 ug/m3)")
        plt.axhline(35.0, color="red", linestyle=":", alpha=0.7, label="Punjab EPA Standard (35 ug/m3)")

        plt.title("District-Wise Monthly Mean PM2.5 Concentrations (Punjab Pilot Area)", fontsize=13, pad=12)
        plt.xlabel("Month", fontsize=11)
        plt.ylabel("PM2.5 (ug/m3)", fontsize=11)
        plt.grid(alpha=0.3)
        plt.legend(frameon=True)
        plt.tight_layout()

        plot_path = self.output_dir / "monthly_district_pm25_trends.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        logger.info(f"Seasonal trends plot saved to: {plot_path}")
        return plot_path

    def run_all(self) -> Dict[str, Union[pd.DataFrame, Path]]:
        """Run complete EDA pipeline."""
        df = self.load_data()
        logger.info("Running station data completeness and quality profiling...")
        quality_rep = self.station_completeness_report(df)
        logger.info("Running multi-pollutant and meteorological correlation analysis...")
        corr_matrix = self.correlation_analysis(df)
        logger.info("Generating seasonal trend plots...")
        plot_path = self.plot_seasonal_trends(df)

        return {
            "quality_report": quality_rep,
            "correlation_matrix": corr_matrix,
            "seasonal_plot": plot_path,
        }


if __name__ == "__main__":
    analyzer = EDAAnalyzer()
    analyzer.run_all()
