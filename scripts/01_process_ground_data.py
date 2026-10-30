"""
Script 01: Ingest and Clean Multi-Station Ground PM2.5 Data.
Loads all 12 monitoring stations across Lahore, Gujranwala, and Sheikhupura,
removes non-positive sensor dropouts and spikes, interpolates short gaps (<=6h),
and generates harmonized hourly and daily datasets with station coordinates.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ground_loader import GroundDataLoader


def main():
    print("=" * 70)
    print("STEP 01: GROUND PM2.5 DATA INGESTION & QUALITY CONTROL")
    print("=" * 70)

    loader = GroundDataLoader()
    hourly_df, daily_df = loader.process_and_save()

    print("\n[SUCCESS] Ground data ingestion completed:")
    print(f"  - Cleaned hourly records : {len(hourly_df):,} rows")
    print(f"  - Daily averaged records : {len(daily_df):,} rows (>= 18 valid hours/day)")
    print(f"  - Monitored stations     : {daily_df['station_id'].nunique()} stations")
    print(f"  - Districts covered      : {sorted(daily_df['district'].unique())}")


if __name__ == "__main__":
    main()
