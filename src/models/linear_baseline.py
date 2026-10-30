"""
Linear Baseline Regressor for Ground-Level PM2.5 Estimation.
Provides standard reference performance using regularized linear regression (Ridge/OLS).
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .base_model import BaseAirQualityModel
except (ImportError, ValueError):
    from base_model import BaseAirQualityModel


class LinearBaselineModel(BaseAirQualityModel):
    """Linear Baseline using Imputation + Standardization + Ridge Regression."""

    def __init__(self, alpha: float = 25.0, log_transform: bool = True):
        super().__init__(model_name="Linear_Baseline_Ridge", log_transform=log_transform)
        self.alpha = alpha
        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=self.alpha, random_state=42)),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearBaselineModel":
        self.feature_names = list(X.columns)
        y_trans = self._transform_target(y.to_numpy())
        self.pipeline.fit(X, y_trans)
        self.model = self.pipeline
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_aligned = X[self.feature_names]
        raw_pred = self.pipeline.predict(X_aligned)
        # Constrain to physical air quality limits (0 to 1200 ug/m3) in transformed space
        if self.log_transform:
            raw_pred = np.clip(raw_pred, 0.0, np.log1p(1200.0))
        preds = self._inverse_transform_target(raw_pred)
        return np.clip(preds, 0.0, 1200.0)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        reg = self.pipeline.named_steps["regressor"]
        coefs = np.abs(reg.coef_)
        return pd.DataFrame({"feature": self.feature_names, "importance": coefs}).sort_values(
            "importance", ascending=False
        )
