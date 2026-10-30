"""
Script 05: Generate Gridded PM2.5 Maps for the Pilot Area (Weeks 5-6).
Infers continuous ~1 km ground-level PM2.5 concentrations across
Lahore, Gujranwala, and Sheikhupura districts using the validated ML model.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.spatial.spatial_predictor import SpatialPM25Predictor
from src.visualization.map_plots import SpatialMapPlotter


def main():
    print("=" * 70)
    print("STEP 05: 1 KM GRIDDED PM2.5 ESTIMATION & MAPPING")
    print("=" * 70)

    # Use peak historical smog date as exemplar
    target_date = "2025-11-15"
    predictor = SpatialPM25Predictor()
    gridded_df = predictor.predict_gridded_pm25_for_date(target_date=target_date, save_csv=True)

    plotter = SpatialMapPlotter(output_dir="outputs/gridded_pm25")
    map_path = plotter.plot_gridded_pm25_map(gridded_df, date_str=target_date)

    print("\n[SUCCESS] Gridded PM2.5 mapping completed:")
    print(f"  - Target date              : {target_date}")
    print(f"  - Total grid pixels        : {len(gridded_df):,} (~1 km2 each)")
    print(f"  - Mean predicted PM2.5     : {gridded_df['estimated_pm25'].mean():.1f} ug/m³")
    print(f"  - Max predicted PM2.5      : {gridded_df['estimated_pm25'].max():.1f} ug/m³")
    print(f"  - Spatial map plot saved   : {map_path}")


if __name__ == "__main__":
    main()
