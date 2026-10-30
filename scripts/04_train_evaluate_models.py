"""
Script 04: Train and Evaluate Machine Learning Models (Weeks 3-6).
Executes spatial cross-validation and held-out station validation across:
  - Linear Baseline (Ridge)
  - Random Forest Regressor
  - XGBoost Regressor
Computes R², RMSE, MAE, MBE, IOA, and FAC2. Saves trained models and benchmark reports.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.model_evaluator import ModelEvaluator


def main():
    print("=" * 70)
    print("STEP 04: MODEL TRAINING & SPATIAL CROSS-VALIDATION")
    print("=" * 70)

    evaluator = ModelEvaluator()
    results = evaluator.run_spatial_cross_validation()

    print("\n[SUCCESS] Model training and evaluation completed:")
    print("\nMODEL BENCHMARK COMPARISON (Spatial Group K-Fold):")
    print("-" * 75)
    print(results["summary"][["r2", "rmse", "mae", "mbe", "ioa", "fac2"]].to_string())
    print("-" * 75)
    print(f"Top performing model: {results['best_model']}")
    print("Artifacts saved to: outputs/evaluation/ and outputs/models/")


if __name__ == "__main__":
    main()
