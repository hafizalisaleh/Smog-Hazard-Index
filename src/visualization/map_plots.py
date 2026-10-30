"""
Geospatial Map Plotting Module.
Generates 2D high-resolution spatial maps:
  1. Gridded PM2.5 concentration maps (ug/m3)
  2. Prototype Smog Hazard Index (SHI) maps (Low -> Extreme)
  3. Spatial Smog Hotspot maps
Overlays monitoring station locations and administrative district centers.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SpatialMapPlotter:
    """Renders 2D spatial cartographic maps of PM2.5 and SHI."""

    def __init__(
        self,
        output_dir: Union[str, Path] = "outputs/shi_maps",
        stations_path: Union[str, Path] = "config/stations.json",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stations_path = Path(stations_path)
        self.stations = self._load_stations()

    def _load_stations(self) -> pd.DataFrame:
        import json

        if not self.stations_path.exists():
            return pd.DataFrame()
        with open(self.stations_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data.get("stations", []))

    def plot_gridded_pm25_map(
        self,
        gridded_df: pd.DataFrame,
        date_str: str = "2025-11-15",
        pm25_col: str = "estimated_pm25",
    ) -> Path:
        """Create 2D pseudo-color contour map of estimated ground PM2.5."""
        piv = gridded_df.pivot(index="latitude", columns="longitude", values=pm25_col)
        lons = piv.columns.values
        lats = piv.index.values
        grid_z = piv.values

        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.cm.YlOrRd
        im = ax.pcolormesh(lons, lats, grid_z, cmap=cmap, shading="auto", vmin=0, vmax=max(350, np.nanmax(grid_z)))
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Estimated PM2.5 (ug/m³)", fontsize=11)

        # Overlay stations
        if not self.stations.empty:
            ax.scatter(
                self.stations["longitude"],
                self.stations["latitude"],
                c="cyan",
                edgecolors="black",
                s=55,
                zorder=5,
                label="EPA Monitoring Stations",
            )
            for _, st in self.stations.iterrows():
                ax.text(
                    st["longitude"] + 0.01,
                    st["latitude"] + 0.005,
                    st["id"].upper(),
                    fontsize=8,
                    fontweight="bold",
                    color="black",
                    zorder=6,
                )

        ax.set_title(f"Gridded Ground-Level PM2.5 Estimation ({date_str})\nPunjab Pilot Domain (1 km Resolution)", fontsize=12)
        ax.set_xlabel("Longitude (°E)", fontsize=10)
        ax.set_ylabel("Latitude (°N)", fontsize=10)
        ax.grid(alpha=0.3, linestyle=":")
        ax.legend(loc="lower left", fontsize=9)
        plt.tight_layout()

        out_path = self.output_dir / f"gridded_pm25_map_{date_str.replace('-', '')}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved PM2.5 map to: {out_path}")
        return out_path

    def plot_shi_map(
        self,
        gridded_shi_df: pd.DataFrame,
        date_str: str = "2025-11-15",
        shi_col: str = "shi_score",
    ) -> Path:
        """Create 2D thematic map of prototype Smog Hazard Index (SHI)."""
        piv = gridded_shi_df.pivot(index="latitude", columns="longitude", values=shi_col)
        lons = piv.columns.values
        lats = piv.index.values
        grid_z = piv.values

        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.cm.get_cmap("turbo")
        im = ax.pcolormesh(lons, lats, grid_z, cmap=cmap, shading="auto", vmin=0, vmax=100)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Smog Hazard Index (SHI, 0 - 100)", fontsize=11)

        # Overlay stations
        if not self.stations.empty:
            ax.scatter(
                self.stations["longitude"],
                self.stations["latitude"],
                c="white",
                edgecolors="black",
                s=60,
                zorder=5,
                label="Monitoring Stations",
            )

        ax.set_title(
            f"Prototype Smog Hazard Index (SHI) Map ({date_str})\nClassified Severity: Low to Extreme", fontsize=12
        )
        ax.set_xlabel("Longitude (°E)", fontsize=10)
        ax.set_ylabel("Latitude (°N)", fontsize=10)
        ax.grid(alpha=0.3, linestyle=":")
        ax.legend(loc="lower left", fontsize=9)
        plt.tight_layout()

        out_path = self.output_dir / f"shi_hazard_map_{date_str.replace('-', '')}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved SHI map to: {out_path}")
        return out_path

    def plot_hotspot_map(
        self,
        gridded_shi_df: pd.DataFrame,
        date_str: str = "2025-11-15",
    ) -> Path:
        """Create hotspot cluster map highlighting top acute hazard areas."""
        df = gridded_shi_df.copy()
        if "is_hotspot" not in df.columns:
            from ..index.hotspot_detector import HotspotDetector
            detector = HotspotDetector()
            df, _ = detector.identify_hotspots(df)

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            df["longitude"],
            df["latitude"],
            c=df["shi_score"],
            cmap="viridis",
            s=12,
            alpha=0.4,
            label="Background Grid",
        )
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("SHI Score", fontsize=11)

        # Hotspot pixels
        hs = df[df["is_hotspot"] == 1]
        ax.scatter(
            hs["longitude"],
            hs["latitude"],
            c="red",
            edgecolors="darkred",
            s=28,
            alpha=0.85,
            label=f"Acute Smog Hotspots (n={len(hs)})",
        )

        # Overlay stations
        if not self.stations.empty:
            ax.scatter(
                self.stations["longitude"],
                self.stations["latitude"],
                c="yellow",
                edgecolors="black",
                s=65,
                zorder=6,
                label="Monitoring Stations",
            )

        ax.set_title(f"Identified Acute Smog Hazard Hotspots ({date_str})\nPunjab Pilot Area", fontsize=12)
        ax.set_xlabel("Longitude (°E)", fontsize=10)
        ax.set_ylabel("Latitude (°N)", fontsize=10)
        ax.grid(alpha=0.3, linestyle=":")
        ax.legend(loc="lower left", fontsize=9)
        plt.tight_layout()

        out_path = self.output_dir / f"smog_hotspots_map_{date_str.replace('-', '')}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved hotspot map to: {out_path}")
        return out_path
