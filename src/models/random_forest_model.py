"""
Random Forest Regressor for Ground-Level PM2.5 Estimation.
Nonlinear ensemble method capturing complex aerosol-meteorological interactions.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

try:
    from .base_model import BaseAirQualityModel
except (ImportError, ValueError):
    from base_model import BaseAirQualityModel


class RandomForestModel(BaseAirQualityModel):
    """Random Forest Regressor with median imputation."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 16,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        log_transform: bool = True,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        super().__init__(model_name="Random_Forest", log_transform=log_transform)
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("regressor", self.rf),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        self.feature_names = list(X.columns)
        y_trans = self._transform_target(y.to_numpy())
        self.pipeline.fit(X, y_trans)
        self.model = self.pipeline.named_steps["regressor"]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_aligned = X[self.feature_names]
        raw_pred = self.pipeline.predict(X_aligned)
        return self._inverse_transform_target(raw_pred)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        return pd.DataFrame(
            {"feature": self.feature_names, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)
