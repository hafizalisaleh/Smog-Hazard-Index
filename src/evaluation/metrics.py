"""
Air Quality Model Performance Metrics.
Implements:
  - R² (Coefficient of Determination)
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - MBE (Mean Bias Error)
  - NMB (%) (Normalized Mean Bias)
  - IOA (Willmott Index of Agreement)
  - FAC2 (Fraction of Predictions within a Factor of Two)
"""

from typing import Dict, Union

import numpy as np
import pandas as pd


def compute_evaluation_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """Compute standard meteorological and atmospheric modeling metrics."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]

    n = len(yt)
    if n == 0:
        return {
            "n_samples": 0,
            "r2": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "mbe": np.nan,
            "nmb_pct": np.nan,
            "ioa": np.nan,
            "fac2": np.nan,
        }

    err = yp - yt
    abs_err = np.abs(err)

    # 1. R²
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    ss_res = np.sum(err ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else np.nan

    # 2. RMSE & MAE
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(abs_err))

    # 3. Mean Bias Error (MBE) & Normalized Mean Bias (NMB %)
    mbe = float(np.mean(err))
    mean_yt = float(np.mean(yt))
    nmb_pct = float((mbe / mean_yt) * 100.0) if mean_yt != 0 else np.nan

    # 4. Willmott Index of Agreement (IOA, 0 to 1)
    denom = np.sum((np.abs(yp - mean_yt) + np.abs(yt - mean_yt)) ** 2)
    ioa = float(1.0 - (ss_res / denom)) if denom > 0 else np.nan

    # 5. FAC2 (Fraction within factor of 2: 0.5 <= yp / yt <= 2.0)
    nz = yt > 0
    if np.sum(nz) > 0:
        ratio = yp[nz] / yt[nz]
        fac2 = float(np.mean((ratio >= 0.5) & (ratio <= 2.0)))
    else:
        fac2 = np.nan

    return {
        "n_samples": int(n),
        "r2": round(r2, 4),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "mbe": round(mbe, 2),
        "nmb_pct": round(nmb_pct, 2),
        "ioa": round(ioa, 4),
        "fac2": round(fac2, 4),
    }


def calculate_station_metrics(
    eval_df: pd.DataFrame,
    actual_col: str = "actual",
    pred_col: str = "pred",
    station_col: str = "station_id",
) -> pd.DataFrame:
    """Calculate evaluation metrics broken down per monitoring station."""
    rows = []
    for st_id, grp in eval_df.groupby(station_col):
        metrics = compute_evaluation_metrics(grp[actual_col], grp[pred_col])
        metrics["station_id"] = st_id
        if "station_name" in grp.columns:
            metrics["station_name"] = grp["station_name"].iloc[0]
        if "district" in grp.columns:
            metrics["district"] = grp["district"].iloc[0]
        rows.append(metrics)

    res = pd.DataFrame(rows)
    # Put identifier columns first
    id_cols = [c for c in ["station_id", "station_name", "district"] if c in res.columns]
    other_cols = [c for c in res.columns if c not in id_cols]
    return res[id_cols + other_cols]
