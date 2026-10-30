"""
Modular Time-Series ARIMA Forecaster for Ground PM2.5 Monitoring Stations.
Refactored from the initial monolithic script to support any station across Punjab,
automated order selection, naive persistence benchmarking, rolling-origin evaluation,
and 168-hour (7-day) probabilistic forecasting with confidence intervals.
"""

import itertools
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


class ARIMAForecaster:
    """Trains, evaluates, and forecasts hourly PM2.5 using ARIMA time-series models."""

    def __init__(
        self,
        station_id: str = "uet",
        data_path: Union[str, Path] = "data/processed/ground_pm25_hourly.csv",
        output_dir: Union[str, Path] = "outputs/arima",
        horizon: int = 168,
        log_transform: bool = True,
        alpha: float = 0.05,
    ):
        self.station_id = station_id.lower()
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir) / self.station_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.horizon = horizon
        self.log_transform = log_transform
        self.alpha = alpha
        self.best_order = None

    def _to_model(self, x: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
        return np.log1p(np.maximum(x, 0.0)) if self.log_transform else x

    def _from_model(self, x: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
        return np.clip(np.expm1(x), 0.0, None) if self.log_transform else np.clip(x, 0.0, None)

    def load_station_series(self) -> pd.Series:
        """Load continuous hourly PM2.5 series for the designated monitoring station."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Hourly ground data file not found at {self.data_path}")
        df = pd.read_csv(self.data_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        st_df = df[df["station_id"].str.lower() == self.station_id].sort_values("timestamp")

        if st_df.empty:
            raise ValueError(f"No records found for station {self.station_id}")

        series = st_df.set_index("timestamp")["pm25"].asfreq("h")
        logger.info(f"Loaded {len(series):,} hourly steps for {self.station_id.upper()} ({series.notna().sum():,} valid)")
        return series

    def split_chronological(
        self, series: pd.Series, train_frac: float = 0.70, val_frac: float = 0.15
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Split series chronologically into train, validation, and untouched test blocks."""
        n = len(series)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))

        train = series.iloc[:n_train]
        val = series.iloc[n_train : n_train + n_val]
        test = series.iloc[n_train + n_val :]
        return train, val, test

    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute RMSE, MAE, MAPE, and R2 on original scale."""
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        m = np.isfinite(yt) & np.isfinite(yp)
        yt, yp = yt[m], yp[m]

        if yt.size == 0:
            return {"n": 0, "rmse": np.nan, "mae": np.nan, "mape_pct": np.nan, "r2": np.nan}

        err = yp - yt
        nz = yt > 0
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)

        return {
            "n": int(yt.size),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "mape_pct": float(np.mean(np.abs(err[nz] / yt[nz])) * 100) if nz.any() else np.nan,
            "r2": float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else np.nan,
        }

    def grid_search(
        self,
        train_obs: pd.Series,
        val_obs: pd.Series,
        p_list: List[int] = [0, 1, 2, 4, 12, 24],
        d_list: List[int] = [0, 1],
        q_list: List[int] = [0, 1, 2, 4],
        maxiter: int = 100,
    ) -> Tuple[Tuple[int, int, int], pd.DataFrame]:
        """Grid search over ARIMA orders scored on validation RMSE."""
        train_y = self._to_model(train_obs)
        val_y = self._to_model(val_obs)

        candidates = list(itertools.product(p_list, d_list, q_list))
        logger.info(f"Testing {len(candidates)} ARIMA candidates on validation set...")

        records = []
        for p, d, q in candidates:
            order = (p, d, q)
            rec = {"order": f"({p},{d},{q})", "p": p, "d": d, "q": q, "status": "ok", "val_rmse": np.nan}
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(train_y, order=order, enforce_stationarity=True, enforce_invertibility=True)
                    res = model.fit(method="statespace", method_kwargs={"maxiter": maxiter})

                # One-step rolling forecast on validation
                res_val = res.extend(val_y)
                val_pred_m = res_val.fittedvalues.loc[val_y.index]
                val_pred = self._from_model(val_pred_m)

                met = self.regression_metrics(val_obs.values, val_pred.values)
                rec.update(
                    {
                        "val_rmse": met["rmse"],
                        "val_mae": met["mae"],
                        "val_r2": met["r2"],
                        "aic": float(res.aic),
                        "bic": float(res.bic),
                    }
                )
            except Exception as e:
                rec["status"] = "error"
                rec["error"] = str(e)[:100]

            records.append(rec)

        res_df = pd.DataFrame(records).dropna(subset=["val_rmse"]).sort_values("val_rmse")
        if res_df.empty:
            # Fallback default order
            self.best_order = (1, 1, 1)
        else:
            best_row = res_df.iloc[0]
            self.best_order = (int(best_row["p"]), int(best_row["d"]), int(best_row["q"]))

        logger.info(f"Selected ARIMA order for {self.station_id.upper()}: {self.best_order}")
        res_df.to_csv(self.output_dir / "grid_search_results.csv", index=False)
        return self.best_order, res_df

    def forecast_future(
        self,
        full_series: pd.Series,
        order: Optional[Tuple[int, int, int]] = None,
        use_diurnal_cycle: bool = True,
    ) -> pd.DataFrame:
        """
        Fit ARIMA and forecast the next 168 hours with 95% CI.
        When use_diurnal_cycle=True, extracts the empirical 24-hour diurnal pattern
        so the multi-day forecast reproduces realistic daily smog oscillations
        instead of decaying into a flat constant line.
        """
        order = order or self.best_order or (2, 1, 1)
        y = self._to_model(full_series)

        fut_idx = pd.date_range(full_series.index[-1] + pd.Timedelta(hours=1), periods=self.horizon, freq="h")

        if use_diurnal_cycle:
            # 1. Extract 24-hour diurnal profile from recent observations (last 60 days)
            recent_window = y.iloc[-60 * 24 :] if len(y) > 60 * 24 else y
            diurnal_means = recent_window.groupby(recent_window.index.hour).mean()
            diurnal_anomaly = diurnal_means - recent_window.mean()

            # 2. Deseasonalize residuals
            residuals = y - y.index.hour.map(diurnal_anomaly)

            logger.info(f"Fitting Diurnal-ARIMA{order} on {len(residuals):,} hours for {self.horizon}-hour forecast...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(residuals, order=order, enforce_stationarity=True, enforce_invertibility=True)
                res = model.fit(method="statespace")

            fc = res.get_forecast(steps=self.horizon)
            ci = fc.conf_int(alpha=self.alpha)

            # 3. Re-inject diurnal wave into point forecast and confidence bounds
            diurnal_fut = fut_idx.hour.map(diurnal_anomaly)
            fc_mean_model = fc.predicted_mean + diurnal_fut
            ci_lower_model = ci.iloc[:, 0] + diurnal_fut
            ci_upper_model = ci.iloc[:, 1] + diurnal_fut

            point_pred = self._from_model(fc_mean_model.to_numpy())
            lower_ci = self._from_model(ci_lower_model.to_numpy())
            upper_ci = self._from_model(ci_upper_model.to_numpy())
        else:
            logger.info(f"Fitting plain ARIMA{order} on {len(y):,} hours...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(y, order=order, enforce_stationarity=True, enforce_invertibility=True)
                res = model.fit(method="statespace")

            fc = res.get_forecast(steps=self.horizon)
            ci = fc.conf_int(alpha=self.alpha)

            point_pred = self._from_model(fc.predicted_mean.to_numpy())
            lower_ci = self._from_model(ci.iloc[:, 0].to_numpy())
            upper_ci = self._from_model(ci.iloc[:, 1].to_numpy())

        out_df = pd.DataFrame(
            {
                "timestamp": fut_idx,
                "forecast_ugm3": np.round(point_pred, 2),
                "lower_95": np.round(lower_ci, 2),
                "upper_95": np.round(upper_ci, 2),
            }
        ).set_index("timestamp")

        out_df.to_csv(self.output_dir / "future_forecast_168h.csv")

        # Plot 168h forecast
        self.plot_forecast(full_series, out_df, order)
        return out_df

    def plot_forecast(self, history: pd.Series, forecast_df: pd.DataFrame, order: Tuple[int, int, int]) -> Path:
        """Plot past 14 days and upcoming 7-day forecast with confidence envelope."""
        recent_hist = history.iloc[-14 * 24 :]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(recent_hist.index, recent_hist.values, color="black", lw=1.2, label="Observed History (14 Days)")
        ax.plot(
            forecast_df.index,
            forecast_df["forecast_ugm3"],
            color="crimson",
            lw=1.8,
            label=f"ARIMA{order} Forecast ({self.horizon}h)",
        )
        ax.fill_between(
            forecast_df.index,
            forecast_df["lower_95"],
            forecast_df["upper_95"],
            color="steelblue",
            alpha=0.25,
            label="95% Confidence Interval",
        )
        ax.axvline(history.index[-1], color="gray", linestyle="--", lw=1)

        ax.set_title(f"{self.station_id.upper()} Monitoring Station: 168-Hour Ahead PM2.5 Forecast", fontsize=12)
        ax.set_ylabel("PM2.5 (ug/m³)", fontsize=10)
        ax.set_xlabel("Time", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
        plt.tight_layout()

        plot_path = self.output_dir / "forecast_168h_plot.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved forecast figure to: {plot_path}")
        return plot_path


if __name__ == "__main__":
    forecaster = ARIMAForecaster(station_id="uet")
    s = forecaster.load_station_series()
    train, val, test = forecaster.split_chronological(s)
    # Quick test grid
    order, _ = forecaster.grid_search(train, val, p_list=[1, 2], d_list=[1], q_list=[1], maxiter=30)
    fc = forecaster.forecast_future(s, order=order)
    print("168-h forecast preview:")
    print(fc.head(5))
