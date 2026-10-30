"""
Script 02: Assemble Collocated Ground-Satellite-Meteorology Dataset.
Matches ground PM2.5 readings with MODIS AOD (1 km daily), Sentinel-5P TROPOMI
(NO2, SO2), CAMS reanalysis, and ERA5 weather variables (T2M, Wind, RH, PBLH, SP).
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.collocator import DatasetCollocator


def main():
    print("=" * 70)
    print("STEP 02: MULTI-SENSOR SPATIO-TEMPORAL DATASET COLLOCATION")
    print("=" * 70)

    collocator = DatasetCollocator()
    df = collocator.assemble_matched_dataset(use_daily=True, save_output=True)

    print("\n[SUCCESS] Spatio-temporal dataset collocation completed:")
    print(f"  - Matched training samples : {len(df):,} station-days")
    print(f"  - Total features & columns : {df.shape[1]}")
    print(f"  - Date range               : {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  - Output file              : data/processed/matched_training_dataset.csv")


if __name__ == "__main__":
    main()
