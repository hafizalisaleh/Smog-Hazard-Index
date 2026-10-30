"""
Script 06: Compute Prototype Smog Hazard Index (SHI) & Hotspot Maps (Weeks 5-8).
Combines standardized PM2.5, AOD, NO2, SO2, and meteorological stagnation indicators
into a unified 0-100 hazard index, classifies into Low -> Extreme, identifies
hotspots, validates against historical smog events, and plots thematic maps.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.spatial.spatial_predictor import SpatialPM25Predictor
from src.index.shi_calculator import SmogHazardIndexCalculator
from src.index.classifier import SHIClassifier
from src.index.hotspot_detector import HotspotDetector
from src.index.historical_validation import HistoricalSmogValidator
from src.visualization.map_plots import SpatialMapPlotter


def main():
    print("=" * 70)
    print("STEP 06: SMOG HAZARD INDEX (SHI) FORMULATION & HOTSPOT MAPPING")
    print("=" * 70)

    target_date = "2025-11-15"
    print(f"\n1. Ingesting gridded fields for exemplar date: {target_date}...")
    predictor = SpatialPM25Predictor()
    gridded_df = predictor.predict_gridded_pm25_for_date(target_date=target_date, save_csv=False)

    print("2. Calculating multi-pollutant & meteorological SHI score...")
    calculator = SmogHazardIndexCalculator()
    gridded_shi = calculator.compute_shi(gridded_df)

    print("3. Classifying hazard tiers (Low -> Extreme) & public health guidance...")
    classified_df = SHIClassifier.classify_dataframe(gridded_shi)

    print("4. Detecting acute spatial smog hotspots...")
    detector = HotspotDetector(shi_threshold=70.0)
    final_df, district_summary = detector.identify_hotspots(classified_df)

    # Save gridded SHI table
    out_dir = Path("outputs/shi_maps")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"gridded_shi_{target_date.replace('-', '')}.csv"
    final_df.to_csv(out_csv, index=False)
    print(f"  - Saved gridded SHI dataset : {out_csv}")

    # Plot maps
    print("5. Rendering spatial maps...")
    plotter = SpatialMapPlotter(output_dir="outputs/shi_maps")
    shi_map_path = plotter.plot_shi_map(final_df, date_str=target_date)
    hotspot_map_path = plotter.plot_hotspot_map(final_df, date_str=target_date)

    print("\nDISTRICT SMOG HAZARD & HOTSPOT SUMMARY:")
    print("-" * 75)
    print(district_summary.to_string(index=False))
    print("-" * 75)

    print("\n6. Running historical smog episodes validation (Nov peak vs Dec vs Jul monsoon)...")
    validator = HistoricalSmogValidator()
    hist_results = validator.evaluate_historical_episodes()

    print("\n[SUCCESS] Smog Hazard Index pipeline executed successfully:")
    print(f"  - SHI thematic map   : {shi_map_path}")
    print(f"  - Hotspot map        : {hotspot_map_path}")
    print(f"  - Historical report  : outputs/reports/historical_episodes_validation.csv")


if __name__ == "__main__":
    main()
