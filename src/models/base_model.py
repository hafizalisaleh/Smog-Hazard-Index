"""
Base Model Abstract Wrapper for PM2.5 Estimation.
Provides consistent interfaces for training, target transformation (log1p),
predicting, feature importance extraction, and persistence.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd


class BaseAirQualityModel(ABC):
    """Abstract base class for all PM2.5 estimation models."""

    def __init__(self, model_name: str, log_transform: bool = True):
        self.model_name = model_name
        self.log_transform = log_transform
        self.model = None
        self.feature_names: List[str] = []

    def _transform_target(self, y: np.ndarray) -> np.ndarray:
        return np.log1p(np.maximum(y, 0.0)) if self.log_transform else y

    def _inverse_transform_target(self, y_pred: np.ndarray) -> np.ndarray:
        return np.clip(np.expm1(y_pred), 0.0, None) if self.log_transform else np.clip(y_pred, 0.0, None)

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseAirQualityModel":
        """Fit model on feature matrix X and ground truth target y."""
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ground-level PM2.5 in original physical units (ug/m3)."""
        if self.model is None:
            raise RuntimeError(f"Model {self.model_name} is not trained yet.")
        # Ensure feature alignment
        X_aligned = X[self.feature_names] if self.feature_names else X
        raw_pred = self.model.predict(X_aligned)
        return self._inverse_transform_target(raw_pred)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Extract feature importances if supported by the estimator."""
        if hasattr(self.model, "feature_importances_"):
            return pd.DataFrame(
                {"feature": self.feature_names, "importance": self.model.feature_importances_}
            ).sort_values("importance", ascending=False)
        return None

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model object to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_name": self.model_name,
                "log_transform": self.log_transform,
                "feature_names": self.feature_names,
                "model": self.model,
            },
            path,
        )

    def load(self, filepath: Union[str, Path]) -> "BaseAirQualityModel":
        """Load trained model object from disk."""
        data = joblib.load(filepath)
        self.model_name = data["model_name"]
        self.log_transform = data["log_transform"]
        self.feature_names = data["feature_names"]
        self.model = data["model"]
        return self
