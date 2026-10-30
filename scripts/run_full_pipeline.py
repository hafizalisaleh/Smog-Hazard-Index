"""
Master End-to-End Pipeline Runner.
Executes the full research and development workflow:
  1. Ground PM2.5 Ingestion & QC (12 stations, 3 districts)
  2. Multi-Sensor Collocation (Ground + MODIS AOD + Sentinel-5P + ERA5)
  3. Exploratory Data Analysis & Correlation Diagnostics
  4. Feature Engineering & Spatial-Temporal Cross-Validation (Linear, RF, XGBoost)
  5. 1 km Gridded Continuous PM2.5 Estimation
  6. Prototype Smog Hazard Index (SHI) Formulation & Hotspot Mapping
  7. Historical Smog Episode Validation
"""

import importlib
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    start_time = time.time()
    print("=" * 80)
    print("SMOG HAZARD INDEX (SHI) & MACHINE LEARNING PM2.5 ESTIMATION PIPELINE")
    print("EHDS Lab / SUPARCO Projects — Full End-to-End Execution")
    print("=" * 80)

    # Load step modules dynamically
    step01 = importlib.import_module("scripts.01_process_ground_data")
    step02 = importlib.import_module("scripts.02_assemble_dataset")
    step03 = importlib.import_module("scripts.03_run_eda")
    step04 = importlib.import_module("scripts.04_train_evaluate_models")
    step05 = importlib.import_module("scripts.05_generate_gridded_pm25")
    step06 = importlib.import_module("scripts.06_compute_shi_and_hotspots")
    step07 = importlib.import_module("scripts.07_train_on_merra2_data")

    steps = [
        ("Step 01: Ground PM2.5 Ingestion & QC", step01.main),
        ("Step 02: Multi-Sensor Collocation", step02.main),
        ("Step 03: Exploratory Data Analysis", step03.main),
        ("Step 04: Model Training & Spatial CV (Simulated Prototype)", step04.main),
        ("Step 05: Gridded PM2.5 Estimation", step05.main),
        ("Step 06: SHI Formulation & Hotspot Mapping", step06.main),
        ("Step 07: Real NASA MERRA-2 & MODIS Multi-Sensor Modeling", step07.main),
    ]

    for title, func in steps:
        t0 = time.time()
        print(f"\n>>> Starting {title}...")
        func()
        print(f">>> {title} completed in {time.time() - t0:.1f}s")

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"ALL PIPELINE STEPS COMPLETED SUCCESSFULLY IN {total_elapsed / 60:.2f} MINUTES!")
    print("All models, gridded estimates, diagnostic tables, and maps are saved under outputs/")
    print("=" * 80)


if __name__ == "__main__":
    main()
