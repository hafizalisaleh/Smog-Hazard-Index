"""
XGBoost Regressor for Ground-Level PM2.5 Estimation.
State-of-the-art gradient boosting regressor handling complex multi-variable interactions.
"""

from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

try:
    from .base_model import BaseAirQualityModel
except (ImportError, ValueError):
    from base_model import BaseAirQualityModel


class XGBoostModel(BaseAirQualityModel):
    """XGBoost Regressor for air quality modeling."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        log_transform: bool = True,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        super().__init__(model_name="XGBoost", log_transform=log_transform)
        self.xgb = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        self.feature_names = list(X.columns)
        y_trans = self._transform_target(y.to_numpy())
        self.xgb.fit(X, y_trans)
        self.model = self.xgb
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_aligned = X[self.feature_names]
        raw_pred = self.xgb.predict(X_aligned)
        return self._inverse_transform_target(raw_pred)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        return pd.DataFrame(
            {"feature": self.feature_names, "importance": self.xgb.feature_importances_}
        ).sort_values("importance", ascending=False)
