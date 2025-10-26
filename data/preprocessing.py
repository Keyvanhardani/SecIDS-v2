"""
CAN-Bus Data Preprocessing

Normalization, scaling, and data cleaning for CAN features.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Optional, Dict, Any
import pickle


class CANPreprocessor:
    """
    Preprocessing pipeline for CAN features.

    Handles:
      - Feature normalization/scaling
      - Outlier handling
      - Missing value imputation
      - Feature selection
    """

    def __init__(
        self,
        scaler_type: str = "standard",
        handle_outliers: bool = True,
        outlier_std: float = 3.0,
        exclude_columns: list = None
    ):
        """
        Args:
            scaler_type: "standard", "minmax", or "robust"
            handle_outliers: Clip outliers to N std devs
            outlier_std: Number of std devs for outlier clipping
            exclude_columns: Columns to exclude from scaling (e.g., categorical)
        """
        self.scaler_type = scaler_type
        self.handle_outliers = handle_outliers
        self.outlier_std = outlier_std
        self.exclude_columns = exclude_columns or ['can_id']

        # Initialize scaler
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler: {scaler_type}")

        self.fitted = False
        self.feature_columns = None

    def fit(self, df: pd.DataFrame) -> 'CANPreprocessor':
        """Fit scaler on training data"""
        # Separate features to scale
        scale_cols = [col for col in df.columns if col not in self.exclude_columns]
        self.feature_columns = df.columns.tolist()

        if len(scale_cols) > 0:
            self.scaler.fit(df[scale_cols])

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scaler"""
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        df = df.copy()

        # Separate columns
        scale_cols = [col for col in df.columns if col not in self.exclude_columns]
        exclude_cols = [col for col in df.columns if col in self.exclude_columns]

        # Handle outliers
        if self.handle_outliers and len(scale_cols) > 0:
            df[scale_cols] = self._clip_outliers(df[scale_cols])

        # Scale
        if len(scale_cols) > 0:
            df[scale_cols] = self.scaler.transform(df[scale_cols])

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(df)
        return self.transform(df)

    def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip values beyond N std devs"""
        df = df.copy()

        for col in df.columns:
            mean = df[col].mean()
            std = df[col].std()

            if std > 0:
                lower = mean - self.outlier_std * std
                upper = mean + self.outlier_std * std
                df[col] = df[col].clip(lower, upper)

        return df

    def save(self, path: str):
        """Save preprocessor state"""
        state = {
            'scaler': self.scaler,
            'scaler_type': self.scaler_type,
            'exclude_columns': self.exclude_columns,
            'feature_columns': self.feature_columns,
            'fitted': self.fitted
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> 'CANPreprocessor':
        """Load preprocessor state"""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        preprocessor = cls(scaler_type=state['scaler_type'])
        preprocessor.scaler = state['scaler']
        preprocessor.exclude_columns = state['exclude_columns']
        preprocessor.feature_columns = state['feature_columns']
        preprocessor.fitted = state['fitted']

        return preprocessor


# Test
if __name__ == "__main__":
    # Synthetic data
    np.random.seed(42)
    data = {
        'can_id': np.random.choice([100, 200, 300], 1000),
        'delta_t': np.random.exponential(0.01, 1000),
        'entropy': np.random.uniform(0, 8, 1000),
        'hamming': np.random.randint(0, 64, 1000)
    }

    df = pd.DataFrame(data)
    print(f"Original:\n{df.describe()}\n")

    # Preprocess
    preprocessor = CANPreprocessor(scaler_type="standard", exclude_columns=['can_id'])
    df_scaled = preprocessor.fit_transform(df)

    print(f"Scaled:\n{df_scaled.describe()}\n")
    print(f"CAN IDs preserved: {df_scaled['can_id'].equals(df['can_id'])}")
