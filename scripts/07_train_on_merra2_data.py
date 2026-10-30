"""
Script 07: Train and Evaluate Models on Real MERRA-2 Aerosol & Meteorology Dataset.
Performs:
  1. Spatio-temporal collocation with real NASA MERRA-2 and MODIS observations.
  2. Multi-feature physical engineering (stagnation index, ventilation index, black carbon ratio, extinction).
  3. Spatial Group K-Fold Cross-Validation across held-out monitoring stations (preventing spatial leakage).
  4. Benchmarking across Linear Baseline (Ridge), Random Forest, and XGBoost.
  5. Calculation of R², RMSE, MAE, MBE, IOA, and FAC2.
  6. Generation of diagnostic parity plots, station breakdown, and feature importance charts.
  7. Final model serialization to outputs/models/.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.merra2_loader import MERRA2DataLoader
from src.evaluation.model_evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    print("=" * 80)
    print("STEP 07: MODEL TRAINING ON NASA MERRA-2 & MODIS MULTI-SENSOR DATASET")
    print("=" * 80)

    # 1. Collocate ground observations with MERRA-2 and MODIS
    logger.info("Assembling real MERRA-2 + MODIS collocated dataset...")
    loader = MERRA2DataLoader()
    df_matched = loader.assemble_collocated_merra2_dataset(save_output=True)
    matched_dataset_path = Path("data/processed/matched_merra2_training_dataset.csv")

    # 2. Run Spatial Cross-Validation and Model Evaluation
    output_dir = Path("outputs/evaluation")
    models_dir = Path("outputs/models")

    evaluator = ModelEvaluator(
        dataset_path=matched_dataset_path,
        output_dir=output_dir,
        models_dir=models_dir,
        prefix="merra2_",
    )

    print("\nExecuting Spatial Group K-Fold Cross-Validation across held-out stations...")
    results = evaluator.run_spatial_cross_validation(n_splits=5)

    print("\n" + "=" * 80)
    print("MODEL BENCHMARK RESULTS (TRAINED ON REAL MERRA-2 & MODIS):")
    print("=" * 80)
    summary_cols = ["r2", "rmse", "mae", "mbe", "nmb_pct", "ioa", "fac2"]
    available_cols = [c for c in summary_cols if c in results["summary"].columns]
    print(results["summary"][available_cols].to_string())
    print("-" * 80)
    print(f"Top performing model: {results['best_model']}")
    print(f"Artifacts and parity plots saved to: {output_dir}")
    print(f"Trained model binaries saved to: {models_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
